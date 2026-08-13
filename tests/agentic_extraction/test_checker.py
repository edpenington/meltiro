"""Tests for the checker client.

The Anthropic call itself is mocked; what is under test is the wrapper
logic: reading the verdict tool call off the response, verdict validation,
cost accounting, the re-ask a reply that records no verdict gets, the parallel
fan-out, and the error-wrapping in run_checker_batch.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from meltiro.checker import (
    CHECKER_TOOL_REPROMPT,
    CheckerConfig,
    check_one_field,
    run_checker_batch,
)
from meltiro.errors import CheckerError
from meltiro.prompt_partials import stage_predicates
from meltiro.rates import Rates
from meltiro.tools import CHECKER_VERDICT_TOOL_NAME
from direktoro import NormalisedResponse, NormalisedUsage, model_info
from direktoro.registry import OPENROUTER_BASE_URL

# The run's structure predicates, which a checker fingerprint takes as an
# argument because it keeps none of its own. A checker-on, reviewer-on
# pipeline with the reviewer's own writes unchecked is the ordinary shape, and
# these tests vary the prompts and the template rather than the structure, so
# they hold it fixed.
PREDICATES = stage_predicates(2, True, False)


# Content blocks, as plain classes rather than MagicMocks. Two reasons: a
# MagicMock answers `getattr(block, "type", None)` with a new mock rather than
# None, so it would pass a type test it should fail; and `name` is a MagicMock
# CONSTRUCTOR argument, so `MagicMock(name="record_verdict")` names the mock
# instead of giving it a `.name` — the exact attribute the tool reader keys on.
class _Text:
    """A minimal Anthropic-shaped text block."""
    type = "text"

    def __init__(self, text):
        self.text = text


class _Thinking:
    """A reasoning block, as a thinking model leads its response with."""
    type = "thinking"

    def __init__(self, thinking=""):
        self.thinking = thinking


class _ToolUse:
    """A tool_use block: what a verdict actually arrives in."""
    type = "tool_use"

    def __init__(self, name, tool_input):
        self.name = name
        self.input = tool_input


def _verdict_block(verdict="ok", rationale="matches the quote",
                   *, name=CHECKER_VERDICT_TOOL_NAME):
    return _ToolUse(name, {"verdict": verdict, "rationale": rationale})


def _stream_returning_content(content, *, input_tokens=100, output_tokens=20,
                              cache_creation=0, cache_read=0,
                              stop_reason="tool_use"):
    """A mock of anthropic's streaming context manager over `content`."""
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    response.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.text_stream = iter([])
    stream_cm.get_final_message = MagicMock(return_value=response)
    return stream_cm


def _stream_returning_verdict(verdict="ok", rationale="matches the quote",
                              **kw):
    return _stream_returning_content(
        [_verdict_block(verdict, rationale)], **kw)


def _client_returning_content(content, **kw):
    client = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_stream_returning_content(content, **kw))
    return client


def _client_returning(verdict="ok", rationale="matches the quote", **kw):
    """A client whose every call answers with the verdict tool."""
    client = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_stream_returning_verdict(verdict, rationale, **kw))
    return client


def _client_returning_streams(*streams):
    """A client handing back a different stream per call, in order."""
    client = MagicMock()
    client.messages.stream = MagicMock(side_effect=list(streams))
    return client


# The rate card these checker calls are priced at, in USD per million tokens.
# It is the operator's number in a real run and a fixture's here; what matters
# to these tests is that a card is present, so the costing path runs at all.
RATES = Rates(input_per_1m=3.0, output_per_1m=15.0,
              cache_read_per_1m=0.3, cache_write_per_1m=3.75)


def _config(rates=RATES):
    return CheckerConfig(
        checker_model="claude-sonnet-4-6",
        max_tokens=1024, sampling={"temperature": 0.0}, concurrency=4, rates=rates,
    )


class TestCheckerWithoutAKey:
    def test_missing_key_env_is_a_checker_error_naming_the_variable(
            self, monkeypatch):
        # No client injected and the variable the checker model's key lives in
        # is unset, so no adapter can be built. The field fails as a
        # CheckerError naming the variable, which is what the orchestrator's
        # pre-spend preflight quotes back to an operator.
        config = _config()
        env = model_info(config.checker_model).api_key_env
        monkeypatch.delenv(env, raising=False)
        with pytest.raises(CheckerError) as excinfo:
            check_one_field(
                system_message_blocks=[{"type": "text", "text": "sys"}],
                user_message_blocks=[{"type": "text", "text": "user"}],
                config=config,
            )
        assert env in str(excinfo.value)


class TestCheckOneField:
    def test_ok_verdict(self):
        client = _client_returning("ok", "matches the quote")
        result = check_one_field(
            system_message_blocks=[{"type": "text", "text": "sys"}],
            user_message_blocks=[{"type": "text", "text": "user"}],
            config=_config(), client=client,
        )
        assert result["verdict"] == "ok"
        assert result["rationale"] == "matches the quote"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20
        assert result["cost_usd"] > 0
        # Answered on the first ask, which is the ordinary case.
        assert result["reprompted"] == 0

    def test_challenge_verdict(self):
        client = _client_returning("challenge", "wrong")
        result = check_one_field(
            system_message_blocks=[],
            user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["verdict"] == "challenge"

    def test_the_verdict_record_is_the_verdict_and_its_rationale(self):
        # Two keys of judgement and no third: whatever a checker model puts
        # beside them is read by nobody, so the schema offers nowhere to put
        # it and the record carries nowhere to keep it.
        client = _client_returning("ok", "r")
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["verdict"] == "ok"
        assert result["rationale"] == "r"
        assert "notes" not in result

    def test_call_sends_the_verdict_tool_and_asks_for_it(self):
        # The verdict tool is on every checker request, and the call asks for
        # it by name on an endpoint that honours a forced choice. Without both
        # there is no contract for the reply to satisfy.
        client = _client_returning()
        check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        kwargs = client.messages.stream.call_args.kwargs
        assert [t["name"] for t in kwargs["tools"]] == [
            CHECKER_VERDICT_TOOL_NAME]
        assert kwargs["tool_choice"] == {
            "type": "tool", "name": CHECKER_VERDICT_TOOL_NAME}

    def test_a_non_forcing_model_is_asked_with_auto(self):
        # The two GLM vision endpoints 404 a forced tool_choice, so their
        # calls go out under "auto" instead — which is why a tool-free reply
        # has to be recoverable at all.
        config = _config(rates=None)
        config.checker_model = "z-ai/glm-4.6v"
        adapter = _StubAdapter(
            _routed_response([_verdict_block()], model="z-ai/glm-4.6v"))
        check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=config, adapter=adapter,
        )
        assert adapter.calls[0]["tool_choice"] == {"type": "auto"}

    def test_a_thinking_block_before_the_tool_call_is_read(self):
        # A thinking model leads its response with reasoning blocks, and on
        # the default display those carry no text at all. The verdict is found
        # by block type, so what precedes it is immaterial.
        client = _client_returning_content(
            [_Thinking(), _Text("Let me look at the quote."),
             _verdict_block("challenge", "the quote gives no denominator")])
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["verdict"] == "challenge"
        assert result["rationale"] == "the quote gives no denominator"

    def test_call_sends_the_checkers_own_configured_sampling(self):
        # The WIRE side of the checker's decoding contract. checker_fp folds in
        # the checker's resolved decoding params, and that promise ("a
        # fingerprint folds in exactly what is sent") only holds while the call
        # site reads the same config value the fingerprint does. A non-default
        # value, so a hardcoded 0.0 anywhere on the path cannot pass.
        client = _client_returning()
        config = _config()
        config.sampling = {"temperature": 0.35}
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[],
            config=config, client=client,
        )
        assert client.messages.stream.call_args.kwargs["temperature"] == 0.35

    def test_call_omits_temperature_for_a_no_temperature_checker_model(self):
        # The quirk applies to the checker's live call too: a checker model that
        # rejects temperature is sent none, whatever the config asked for.
        client = _client_returning()
        config = _config()
        config.checker_model = "claude-opus-4-8"
        config.sampling = {"temperature": 0.35}
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[],
            config=config, client=client,
        )
        assert "temperature" not in client.messages.stream.call_args.kwargs

    def test_invalid_verdict_raises(self):
        # A tool call IS an answer, so a verdict outside the vocabulary is
        # refused rather than re-asked: the checker judged, and the judgement
        # is unusable.
        client = _client_returning("maybe", "unsure")
        with pytest.raises(CheckerError, match="Invalid verdict"):
            check_one_field(
                system_message_blocks=[],
                user_message_blocks=[],
                config=_config(), client=client,
            )
        assert client.messages.stream.call_count == 1

    def test_a_tool_free_reply_is_reasked_and_then_answered(self):
        # A model under "auto" may decline to call the tool. The field is
        # re-asked once and the verdict that arrives is an ordinary one.
        client = _client_returning_streams(
            _stream_returning_content([_Text("The value looks fine to me.")]),
            _stream_returning_verdict("ok", "the quote carries the figure"),
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[{
                "type": "text", "text": "field"}],
            config=_config(), client=client,
        )
        assert result["verdict"] == "ok"
        assert result["reprompted"] == 1
        assert client.messages.stream.call_count == 2

    def test_a_text_only_reply_is_replayed_and_corrected_after_itself(self):
        # Three messages: the field's message as the first ask sent it, the
        # model's own reply verbatim, and the correction as a new user turn.
        # The model is corrected AFTER a reply it can see it gave, which is
        # what makes a different reply likely — it has been shown the same one
        # field and the same one value either way.
        adapter = _SequenceAdapter(
            _direct_response([_Text("The value looks fine to me.")]),
            _direct_response([_verdict_block("ok", "the quote carries it")]),
        )
        result = check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert result["verdict"] == "ok"
        assert len(adapter.calls) == 2
        assert adapter.calls[0]["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "field"}]},
        ]
        assert adapter.calls[1]["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "field"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "The value looks fine to me."}]},
            {"role": "user", "content": [
                {"type": "text", "text": CHECKER_TOOL_REPROMPT}]},
        ]

    def test_a_replayed_reply_keeps_every_text_block_in_order(self):
        # The reply is replayed WHOLE. A model that answered in two blocks is
        # quoted back in two blocks, in the order it wrote them: replaying one
        # of them, or both the other way round, would put a reply in its mouth
        # that it did not give.
        adapter = _SequenceAdapter(
            _direct_response([_Text("First, the value."),
                              _Text("Second, the evidence.")]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert adapter.calls[1]["messages"][1] == {
            "role": "assistant", "content": [
                {"type": "text", "text": "First, the value."},
                {"type": "text", "text": "Second, the evidence."}]}

    @pytest.mark.parametrize("empty", ["", None])
    def test_a_text_block_carrying_no_text_is_not_replayed(self, empty):
        # An empty text block says nothing, and an empty text block is content
        # a provider refuses, so it is dropped on the way into the replay
        # rather than sent back as a turn the model has to be told about.
        adapter = _SequenceAdapter(
            _direct_response([_Text(empty), _Text("the real sentence")]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert adapter.calls[1]["messages"][1] == {
            "role": "assistant", "content": [
                {"type": "text", "text": "the real sentence"}]}

    def test_a_reply_of_nothing_but_empty_text_is_not_replayed_at_all(self):
        # Nothing to quote back, so the correction rides on the field's message
        # as it does for any other reply that cannot be replayed.
        adapter = _SequenceAdapter(
            _direct_response([_Text("")]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert [m["role"] for m in adapter.calls[1]["messages"]] == ["user"]

    def test_the_reasks_first_message_is_byte_identical_to_the_first_asks(
            self):
        # The field the second verdict is about is the field the first ask
        # asked about, to the byte: the blocks the caller rendered are passed
        # through rather than rebuilt, so nothing about the value under review
        # can drift between the two asks. `_SequenceAdapter` copies what it
        # captures, so this compares the two calls rather than one object with
        # itself.
        blocks = [
            {"type": "text", "text": "field"},
            {"type": "text", "text": "the value and its evidence"},
        ]
        adapter = _SequenceAdapter(
            _direct_response([_Text("prose")]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[], user_message_blocks=blocks,
            config=_config(), adapter=adapter,
        )
        first, second = adapter.calls
        assert second["messages"][0] is not first["messages"][0]
        assert (json.dumps(second["messages"][0])
                == json.dumps(first["messages"][0]))

    def test_the_correction_is_the_reasks_last_message_and_nothing_else(self):
        # The correction is a user turn of its own. Appending it to the field's
        # blocks instead would put it before the reply it corrects, and correct
        # a reply the model had not yet given.
        adapter = _SequenceAdapter(
            _direct_response([_Text("prose")]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        messages = adapter.calls[1]["messages"]
        assert messages[-1] == {"role": "user", "content": [
            {"type": "text", "text": CHECKER_TOOL_REPROMPT}]}
        assert CHECKER_TOOL_REPROMPT not in json.dumps(messages[:-1])
        assert CHECKER_TOOL_REPROMPT not in json.dumps(
            adapter.calls[0]["messages"])

    # The four replies that reach the re-ask, each named by what it did. What
    # they share is that the field has no verdict, which is what the correction
    # states; no one mechanism below is true of the other three.
    VERDICTLESS_REPLIES = [
        ("prose", [_Text("The value looks fine to me.")]),
        ("called twice", [_verdict_block("ok", "a"),
                          _verdict_block("ok", "b")]),
        ("called with no arguments", [_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                                               None)]),
        ("called with a list", [_ToolUse(CHECKER_VERDICT_TOOL_NAME, ["ok"])]),
    ]

    @pytest.mark.parametrize(
        "content", [c for _shape, c in VERDICTLESS_REPLIES],
        ids=[shape for shape, _c in VERDICTLESS_REPLIES])
    def test_every_verdictless_reply_is_reasked_with_the_correction(
            self, content):
        adapter = _SequenceAdapter(
            _direct_response(content),
            _direct_response([_verdict_block("ok", "r")]),
        )
        result = check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert result["reprompted"] == 1
        assert CHECKER_TOOL_REPROMPT in json.dumps(
            adapter.calls[1]["messages"])

    @pytest.mark.parametrize(
        "content", [c for _shape, c in VERDICTLESS_REPLIES[1:]],
        ids=[shape for shape, _c in VERDICTLESS_REPLIES[1:]])
    def test_a_reply_holding_a_tool_call_is_corrected_but_not_replayed(
            self, content):
        # Each of these called the verdict tool and still left the field with
        # no verdict. None of them is replayed: a tool_use block quoted back
        # here is a call the correction would leave with no result after it,
        # which is not a conversation any provider accepts. So the re-ask is
        # one user turn — the field once more, correction on the end.
        adapter = _SequenceAdapter(
            _direct_response(content),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert [m["role"] for m in adapter.calls[1]["messages"]] == ["user"]
        assert adapter.calls[1]["messages"][0]["content"] == [
            {"type": "text", "text": "field"},
            {"type": "text", "text": CHECKER_TOOL_REPROMPT},
        ]

    def test_a_second_tool_free_reply_degrades_with_both_asks_spend(self):
        # The corrected reply came back without a verdict too. Both asks were
        # billed, and the failure carries both — the shape the re-ask took is
        # no part of what it cost.
        adapter = _SequenceAdapter(
            _direct_response([_Text("still prose")], input_tokens=40,
                             output_tokens=5),
            _direct_response([_Text("still prose")], input_tokens=60,
                             output_tokens=15),
        )
        with pytest.raises(CheckerError, match="no record_verdict call") as exc:
            check_one_field(
                system_message_blocks=[],
                user_message_blocks=[{"type": "text", "text": "field"}],
                config=_config(), adapter=adapter,
            )
        assert [m["role"] for m in adapter.calls[1]["messages"]] == [
            "user", "assistant", "user"]
        assert exc.value.spent["responses"] == 2
        assert exc.value.spent["input_tokens"] == 100
        assert exc.value.spent["output_tokens"] == 20

    def test_a_reasked_field_reports_both_calls_spend(self):
        # Both asks were billed. Reporting only the one that answered would
        # understate the run.
        client = _client_returning_streams(
            _stream_returning_content([_Text("no tool")], input_tokens=40,
                                      output_tokens=5),
            _stream_returning_verdict(input_tokens=60, output_tokens=15),
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20

    def test_a_persistently_tool_free_checker_raises(self):
        # Declining twice is a misconfigured checker model, not a slow one.
        client = _client_returning_streams(
            _stream_returning_content([_Text("still prose")]),
            _stream_returning_content([_Text("still prose")]),
        )
        with pytest.raises(CheckerError, match="no record_verdict call"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client,
            )
        assert client.messages.stream.call_count == 2

    def test_a_wrong_tool_is_not_taken_as_a_verdict(self):
        # Only the verdict tool answers. Another tool name is a verdict-free
        # reply as far as this field is concerned, and it is re-asked as one.
        adapter = _SequenceAdapter(
            _direct_response(
                [_ToolUse("mark_complete", {"quality_check": {}})]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        result = check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert result["verdict"] == "ok"
        assert result["reprompted"] == 1
        assert [m["role"] for m in adapter.calls[1]["messages"]] == ["user"]
        assert adapter.calls[1]["messages"][0]["content"] == [
            {"type": "text", "text": "field"},
            {"type": "text", "text": CHECKER_TOOL_REPROMPT},
        ]

    def test_prose_beside_a_tool_call_is_corrected_without_being_replayed(
            self):
        # A reply that both narrated and called something is not replayed
        # either, prose and all: quoting back the text while dropping the call
        # it came with would be correcting a reply the model never gave. So the
        # test of the replay is the WHOLE reply, not the part of it that fits.
        adapter = _SequenceAdapter(
            _direct_response([_Text("I'll mark this one complete."),
                              _ToolUse("mark_complete", {})]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert [m["role"] for m in adapter.calls[1]["messages"]] == ["user"]
        assert "I'll mark this one complete." not in json.dumps(
            adapter.calls[1]["messages"])

    def test_a_reply_leading_with_reasoning_is_not_replayed(self):
        # A reasoning block cannot be replayed without the signature it came
        # with, so a reply holding one is corrected without being shown.
        adapter = _SequenceAdapter(
            _direct_response([_Thinking("weighing the quote"),
                              _Text("it looks fine")]),
            _direct_response([_verdict_block("ok", "r")]),
        )
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[{"type": "text", "text": "field"}],
            config=_config(), adapter=adapter,
        )
        assert [m["role"] for m in adapter.calls[1]["messages"]] == ["user"]

    def test_empty_content_is_reasked_then_raises(self):
        # An empty content list is a reply with no verdict in it, handled as
        # any other verdict-free reply — and never as an IndexError that would
        # escape the per-field handling. There is nothing in it to replay, so
        # that ask is the field and the correction as one user turn.
        adapter = _SequenceAdapter(_direct_response([]))
        with pytest.raises(CheckerError, match="no record_verdict call"):
            check_one_field(
                system_message_blocks=[],
                user_message_blocks=[{"type": "text", "text": "field"}],
                config=_config(), adapter=adapter,
            )
        assert len(adapter.calls) == 2
        assert [m["role"] for m in adapter.calls[1]["messages"]] == ["user"]

    def test_truncation_is_named_and_not_reasked(self):
        # A cap that truncated once will truncate again, so the ask is not
        # repeated. The message carries the CHECKER's cap and the pipeline.yaml
        # key that set it, which is the line an operator would edit; naming the
        # model instead would point at the wrong thing entirely.
        client = _client_returning_content(
            [_Thinking()], stop_reason="max_tokens")
        with pytest.raises(CheckerError) as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client,
            )
        assert "1024" in str(exc.value)
        assert "checker_max_tokens" in str(exc.value)
        assert client.messages.stream.call_count == 1

    def test_cache_read_discount_applied(self):
        client = _client_returning(
            input_tokens=100, output_tokens=20,
            cache_creation=0, cache_read=5000,
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        # Cache reads carry their own rate on the card, well below the
        # full-price input rate, so the same prompt size costs less when most
        # of it is served from cache. The counters are kept apart all the way
        # through for exactly this reason.
        cheap = result["cost_usd"]
        # Compare to a non-cached call with the same total prompt size.
        client2 = _client_returning(input_tokens=5100, output_tokens=20)
        result2 = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client2,
        )
        assert cheap < result2["cost_usd"]

    def test_no_rate_card_records_tokens_and_no_cost(self):
        # A direct checker model with no rates states NO cost. Not 0.0, which
        # would read as a free call and would silently pull a run's total down;
        # the token counters are the record, and they came from the provider.
        client = _client_returning(input_tokens=100, output_tokens=20)
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(rates=None), client=client,
        )
        assert result["cost_usd"] is None
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20


def _direct_response(content, *, input_tokens=100, output_tokens=20,
                     stop_reason=None):
    """A canned NormalisedResponse from a DIRECT (un-routed) checker call.

    Priced against the run's rate card rather than from the response, so it
    carries no `reported_cost` at all — which is the ordinary state for a
    direct call and never a missing receipt.
    """
    return NormalisedResponse(
        content=content,
        usage=NormalisedUsage(input_tokens=input_tokens,
                              output_tokens=output_tokens),
        resolved_model="claude-sonnet-4-6", provider="anthropic",
        base_url=None, raw_request={}, raw_response={},
        decoding_params={"max_tokens": 1024}, stop_reason=stop_reason)


def _routed_response(content, *, model="z-ai/glm-5v-turbo",
                     reported_cost=0.0123, generation_id="gen-chk-1",
                     served="Z.AI", input_tokens=5, output_tokens=1):
    """A canned NormalisedResponse from a gateway-routed checker call."""
    return NormalisedResponse(
        content=content,
        usage=NormalisedUsage(input_tokens=input_tokens,
                              output_tokens=output_tokens),
        resolved_model=model, provider="openrouter",
        base_url=OPENROUTER_BASE_URL, raw_request={}, raw_response={},
        decoding_params={"max_tokens": 1024},
        generation_id=generation_id, served_provider=served,
        reported_cost=reported_cost)


class _StubAdapter:
    """A provider adapter returning a canned NormalisedResponse, so a routed
    checker call runs fully offline with no client/SDK."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def create_message(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _RaisingAdapter:
    """An adapter whose call never reaches a provider, so nothing is billed."""

    def create_message(self, **kwargs):
        from direktoro import ProviderError
        raise ProviderError("connection refused")


class _SequenceAdapter:
    """An adapter handing back one canned outcome per call, in order.

    An outcome is a NormalisedResponse to return or an exception to raise, so
    a check that is asked twice, or a batch whose first field fails and whose
    second answers, can be scripted whole. The last outcome repeats, so a
    script shorter than the calls made does not run out.
    """

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def create_message(self, **kwargs):
        outcome = self._outcomes[min(len(self.calls), len(self._outcomes) - 1)]
        # The messages are captured BY VALUE. A re-ask passes the first ask's
        # message object straight through, so two stored references would be
        # one object, and a test comparing the two asks would be comparing it
        # with itself and passing whatever the code did to it in between.
        self.calls.append({**kwargs,
                           "messages": copy.deepcopy(kwargs.get("messages"))})
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class TestCheckOneFieldRoutedCost:
    """A ROUTED (gateway-served) checker model is priced FROM the response
    (OpenRouter usage.cost -> reported_cost). That figure is a charge the
    gateway states rather than one anybody predicts, so it needs no rate card
    and these configs carry none. A missing reported cost is a loud fault,
    NEVER a silent $0."""

    ROUTED = "z-ai/glm-5v-turbo"

    def _config(self):
        return CheckerConfig(checker_model=self.ROUTED,
                             max_tokens=1024, sampling={"temperature": 0.0}, concurrency=4)

    def _response(self, reported_cost, *, generation_id="gen-chk-1",
                  served="Z.AI"):
        return _routed_response(
            [_verdict_block("ok", "r")], model=self.ROUTED,
            reported_cost=reported_cost, generation_id=generation_id,
            served=served)

    def test_reported_cost_is_used_with_no_rate_card(self):
        # No rates on this config, so the only figure available is the
        # response-reported one; a non-None cost here proves that branch is
        # taken. The routing receipts must ride in _provenance for the
        # orchestrator to ledger.
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=self._config(), adapter=_StubAdapter(
                self._response(reported_cost=0.0123)))
        assert result["cost_usd"] == pytest.approx(0.0123)
        assert result["_provenance"]["generation_id"] == "gen-chk-1"
        assert result["_provenance"]["served_provider"] == "Z.AI"

    def test_a_missing_receipt_states_its_coverage_and_keeps_the_verdict(self):
        # The ANSWERING call came back with no charge on it. The verdict was
        # asked for, answered and billed, and it is what the run is buying: an
        # unreadable price is a gap in the accounting, stated as one. The
        # counters are the whole of what was billed either way.
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=self._config(), adapter=_StubAdapter(
                self._response(reported_cost=None)))
        assert result["verdict"] == "ok"
        assert result["cost_usd"] == 0.0
        assert result["cost_incomplete"] is True
        assert result["unreceipted_responses"] == 1
        assert (result["input_tokens"], result["output_tokens"]) == (5, 1)

    def test_a_reasked_success_is_not_poisoned_by_the_first_asks_receipt(self):
        # The first ask came back without the verdict tool AND without a
        # charge on it; the second answered and was priced. The verdict is the
        # second call's, and it stands: raising on the first ask's missing
        # receipt would throw away a verdict that was asked for, answered, and
        # paid for, over the price of the ask that failed. The gap is recorded
        # instead, so the figure states its own coverage.
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=self._config(),
            adapter=_SequenceAdapter(
                _routed_response([_Text("prose")], model=self.ROUTED,
                                 reported_cost=None),
                self._response(reported_cost=0.02)))
        assert result["verdict"] == "ok"
        assert result["reprompted"] == 1
        assert result["cost_usd"] == pytest.approx(0.02)
        assert result["cost_incomplete"] is True
        assert result["unreceipted_responses"] == 1

    def test_a_failure_prices_what_it_can_and_says_what_it_could_not(self):
        # Neither ask recorded a verdict, and neither came back with a charge.
        # Costing this must not raise: the field is being degraded, not
        # reported, and an exception here would leave the fan-out around it
        # (see TestFailuresDegradeOnlyTheirOwnField).
        adapter = _SequenceAdapter(
            _routed_response([_Text("prose")], model=self.ROUTED,
                             reported_cost=None))
        with pytest.raises(CheckerError, match="no record_verdict call") as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=self._config(), adapter=adapter)
        assert exc.value.spent["cost_usd"] == 0.0
        assert exc.value.spent["cost_incomplete"] is True
        assert exc.value.spent["unreceipted_responses"] == 2

    def test_a_failure_still_costs_the_receipts_it_does_have(self):
        # One ask charged, one not. The figure is the charge that was stated,
        # and it is marked as covering less than the whole check rather than
        # being withheld or rounded up to nothing.
        adapter = _SequenceAdapter(
            _routed_response([_Text("prose")], model=self.ROUTED,
                             reported_cost=0.004),
            _routed_response([_Text("prose")], model=self.ROUTED,
                             reported_cost=None))
        with pytest.raises(CheckerError) as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=self._config(), adapter=adapter)
        assert exc.value.spent["cost_usd"] == pytest.approx(0.004)
        assert exc.value.spent["unreceipted_responses"] == 1


class TestRunCheckerBatch:
    def test_parallel_fan_out(self):
        client = _client_returning()
        calls = [
            {"field_path": f"study.field{i}",
             "system_message_blocks": [],
             "user_message_blocks": []}
            for i in range(5)
        ]
        results = run_checker_batch(
            calls=calls, config=_config(), client=client,
        )
        assert set(results.keys()) == {
            f"study.field{i}" for i in range(5)
        }
        for r in results.values():
            assert r["verdict"] == "ok"

    def test_one_error_doesnt_abort_the_batch(self):
        # The first field's checker call answers with a verdict outside the
        # vocabulary; the second answers properly. Concurrency is 1 so the
        # side_effect order is the field order.
        client = _client_returning_streams(
            _stream_returning_verdict("maybe", "unsure"),
            _stream_returning_verdict("ok", "fine"),
        )
        results = run_checker_batch(
            calls=[
                {"field_path": "study.bad", "system_message_blocks": [],
                 "user_message_blocks": []},
                {"field_path": "study.good", "system_message_blocks": [],
                 "user_message_blocks": []},
            ],
            config=CheckerConfig(
                checker_model="claude-sonnet-4-6",
                max_tokens=1024, concurrency=1),
            client=client,
        )
        assert results["study.bad"]["verdict"] == "challenge"  # wrapped error
        assert "error" in results["study.bad"]
        assert results["study.good"]["verdict"] == "ok"

    def test_on_complete_called(self):
        client = _client_returning()
        seen = []
        run_checker_batch(
            calls=[{"field_path": "x", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=_config(), client=client,
            on_complete=lambda fp, r: seen.append((fp, r["verdict"])),
        )
        assert seen == [("x", "ok")]

    def test_empty_content_wrapped_as_field_error(self):
        # An empty-content response for one field must not abort the batch;
        # it is wrapped as a challenged field, like any other CheckerError.
        client = _client_returning_content([])
        results = run_checker_batch(
            calls=[{"field_path": "study.x", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=_config(), client=client,
        )
        assert results["study.x"]["verdict"] == "challenge"
        assert "error" in results["study.x"]

    def test_a_degraded_field_still_reports_what_its_asks_cost(self):
        # Both asks reached the provider and were billed. A field degraded to
        # an error-origin challenge must carry that spend, or the run's total
        # understates itself by however many checks went this way.
        client = _client_returning_content(
            [_Text("prose")], input_tokens=30, output_tokens=4)
        results = run_checker_batch(
            calls=[{"field_path": "study.x", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=_config(), client=client,
        )
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["input_tokens"] == 60
        assert degraded["output_tokens"] == 8
        assert degraded["cost_usd"] > 0

    def test_a_failure_that_never_reached_the_provider_costs_nothing(self):
        client = MagicMock()
        client.messages.stream = MagicMock(
            side_effect=RuntimeError("connection refused"))
        with patch("meltiro.checker._build_checker_adapter") as build:
            build.return_value = _RaisingAdapter()
            results = run_checker_batch(
                calls=[{"field_path": "study.x", "system_message_blocks": [],
                        "user_message_blocks": []}],
                config=_config(), client=client,
            )
        assert results["study.x"]["error_origin"] is True
        assert results["study.x"]["input_tokens"] == 0
        assert results["study.x"]["cost_usd"] == 0.0

    def test_a_failing_on_complete_stops_the_batch(self):
        # The callback is the run's spend ledger and its meta checkpoint, not
        # an audit-side nicety. Swallowing its failure would bank verdicts
        # against a total that had stopped counting what they cost, and the
        # run would finish reporting a number it knows to be wrong.
        client = _client_returning()

        def _ledger(field_path, result):
            raise RuntimeError("meta write failed")

        with pytest.raises(RuntimeError, match="meta write failed"):
            run_checker_batch(
                calls=[{"field_path": "study.x", "system_message_blocks": [],
                        "user_message_blocks": []}],
                config=_config(), client=client, on_complete=_ledger)

    def test_results_ordered_by_field_path(self):
        # Results are returned sorted by field path, not by nondeterministic
        # completion order, so the tool result the model reads and the session
        # event that records it are reproducible.
        client = _client_returning()
        unsorted_paths = ["study.zeta", "record.relationship_2.gauge",
                          "study.alpha", "record.relationship_1.gauge",
                          "study.mid"]
        calls = [
            {"field_path": p, "system_message_blocks": [],
             "user_message_blocks": []}
            for p in unsorted_paths
        ]
        results = run_checker_batch(
            calls=calls, config=_config(), client=client,
        )
        assert list(results.keys()) == sorted(unsorted_paths)


class TestFailuresDegradeOnlyTheirOwnField:
    """A batch is a set of paid siblings. Whatever one field's check does on
    its way to failing, the other fields' verdicts come back.

    Concurrency is 1 throughout, so the scripted adapter's outcomes land in
    field order.
    """

    ROUTED = "z-ai/glm-5v-turbo"

    def _config(self):
        return CheckerConfig(checker_model=self.ROUTED,
                             max_tokens=1024, concurrency=1)

    def _calls(self, *paths):
        return [{"field_path": p, "system_message_blocks": [],
                 "user_message_blocks": []} for p in paths]

    def _prose(self, **kw):
        return _routed_response([_Text("prose")], model=self.ROUTED, **kw)

    def _ok(self, **kw):
        return _routed_response([_verdict_block("ok", "fine")],
                                model=self.ROUTED, **kw)

    def test_a_failure_with_no_receipt_prices_itself_and_spares_the_batch(self):
        # Both of the failed field's asks came back without a charge on them.
        # Costing that must not raise: the exception would leave the pool
        # through `fut.result()` and take the second field's paid verdict with
        # it.
        adapter = _SequenceAdapter(
            self._prose(reported_cost=None), self._prose(reported_cost=None),
            self._ok(reported_cost=0.01))
        results = run_checker_batch(
            calls=self._calls("study.bad", "study.good"),
            config=self._config(), adapter=adapter)
        assert results["study.bad"]["error_origin"] is True
        assert results["study.bad"]["cost_usd"] == 0.0
        assert results["study.bad"]["cost_incomplete"] is True
        assert results["study.bad"]["unreceipted_responses"] == 2
        assert results["study.good"]["verdict"] == "ok"

    def test_an_answering_call_with_no_receipt_keeps_its_verdict(self):
        # A verdict the checker gave, on a call that was made and billed. The
        # only thing missing is the gateway's charge for it, and the whole
        # cost of treating that as fatal falls on the answer: the field would
        # be recorded as unchecked, its check slot spent, and an error-origin
        # challenge banked at $0 for work that was paid for. So the verdict
        # stands, the counters are the real sums, and the figure says what it
        # does not cover.
        adapter = _SequenceAdapter(
            self._ok(reported_cost=None), self._ok(reported_cost=0.01))
        results = run_checker_batch(
            calls=self._calls("study.bad", "study.good"),
            config=self._config(), adapter=adapter)
        verdict = results["study.bad"]
        assert verdict["verdict"] == "ok"
        assert verdict["error_origin"] is False
        assert "error" not in verdict
        assert (verdict["input_tokens"], verdict["output_tokens"]) == (5, 1)
        assert verdict["cost_usd"] == 0.0
        assert verdict["cost_incomplete"] is True
        assert verdict["unreceipted_responses"] == 1
        assert results["study.good"]["verdict"] == "ok"

    def test_an_unexpected_exception_is_named_and_kept_to_its_field(self):
        # Anything at all: a fault in the plumbing this engine does not have a
        # structured error for. The batch still returns, and the exception's
        # type is in the record rather than swallowed, so it is not
        # indistinguishable from an ordinary provider failure.
        adapter = _SequenceAdapter(
            ValueError("adapter fell over"), self._ok(reported_cost=0.01))
        results = run_checker_batch(
            calls=self._calls("study.bad", "study.good"),
            config=self._config(), adapter=adapter)
        assert results["study.bad"]["error_origin"] is True
        assert "ValueError: adapter fell over" == results["study.bad"]["error"]
        assert results["study.bad"]["cost_usd"] == 0.0
        assert results["study.good"]["verdict"] == "ok"


class TestEveryRaiseCarriesItsSpend:
    """A failed check reports what its calls cost. A raise with no spend on it
    ledgers billed work at $0, and the run's total then reads low with nothing
    in the artefact to contradict it."""

    def _calls(self):
        return [{"field_path": "study.x", "system_message_blocks": [],
                 "user_message_blocks": []}]

    def test_an_invalid_verdict_carries_the_call_that_produced_it(self):
        # A verdict outside the vocabulary is an answer that arrived unusable.
        # The call that produced it was billed exactly like one that answered.
        client = _client_returning("maybe", "unsure", input_tokens=100,
                                   output_tokens=20)
        results = run_checker_batch(
            calls=self._calls(), config=_config(), client=client)
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["input_tokens"] == 100
        assert degraded["output_tokens"] == 20
        assert degraded["cost_usd"] > 0

    def test_a_provider_error_carries_the_asks_that_came_before_it(self):
        # The first ask reached the provider, came back without a verdict, and
        # was billed; the re-ask hit a provider error. The field is degraded
        # holding the first ask's spend.
        from direktoro import ProviderError
        adapter = _SequenceAdapter(
            _direct_response([_Text("prose")], input_tokens=40,
                             output_tokens=5),
            ProviderError("gateway refused the re-ask"))
        results = run_checker_batch(
            calls=self._calls(), config=_config(), adapter=adapter)
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["input_tokens"] == 40
        assert degraded["output_tokens"] == 5
        assert degraded["cost_usd"] > 0

    def test_a_fault_with_no_structured_error_is_priced_from_its_calls(
            self, monkeypatch):
        # A fault this module has no vocabulary for, raised AFTER the response
        # came back: the ask was made and billed, whatever went wrong reading
        # it. Only the check itself knows what had completed by then, so it is
        # what prices the failure — the fan-out's backstop holds no responses
        # and would ledger the call at $0.
        import direktoro

        def _boom(response, name):
            raise ValueError("verdict reader fell over")

        monkeypatch.setattr(direktoro, "extract_tool_call", _boom)
        adapter = _SequenceAdapter(
            _direct_response([_Text("prose")], input_tokens=90,
                             output_tokens=12))
        results = run_checker_batch(
            calls=self._calls(), config=_config(), adapter=adapter)
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["error"] == "ValueError: verdict reader fell over"
        assert degraded["input_tokens"] == 90
        assert degraded["output_tokens"] == 12
        assert degraded["cost_usd"] > 0

    def test_a_failure_on_an_unpriced_model_states_no_cost(self):
        # A direct model with no rate card: nothing prices its calls, and the
        # tokens are the whole of the record. A failed check says so exactly
        # as a successful one does — a 0.0 here would be this field asserting
        # its calls were free, and it would let the run state a total that
        # silently leaves them out.
        client = _client_returning("maybe", "unsure", input_tokens=100,
                                   output_tokens=20)
        results = run_checker_batch(
            calls=self._calls(), config=_config(rates=None), client=client)
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["input_tokens"] == 100
        assert degraded["output_tokens"] == 20
        assert degraded["cost_usd"] is None

    def test_a_partial_spend_mapping_still_degrades_its_field(self):
        # A CheckerError from a library caller, carrying half a spend record.
        # The degradation is the fan-out's last line, so reading a counter it
        # does not hold would raise out of the pool and take the batch's other
        # verdicts with it — the exact failure this path exists to prevent.
        class _PartialAdapter:
            def create_message(self, **kwargs):
                raise CheckerError("half a record",
                                   spent={"input_tokens": 7})

        results = run_checker_batch(
            calls=self._calls(), config=_config(),
            adapter=_PartialAdapter())
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["input_tokens"] == 7
        assert degraded["output_tokens"] == 0
        assert degraded["reprompted"] == 0
        assert degraded["cost_usd"] == 0.0

    def test_a_failure_that_reached_no_provider_costs_zero(self):
        # The other side of it, and why the null above is not simply "unknown
        # cost": no call was made, so zero tokens were billed, and zero tokens
        # cost zero under any card or none. Withholding a figure here would
        # withhold the run's total over a call that never happened.
        results = run_checker_batch(
            calls=self._calls(), config=_config(rates=None),
            adapter=_RaisingAdapter())
        degraded = results["study.x"]
        assert degraded["error_origin"] is True
        assert degraded["input_tokens"] == 0
        assert degraded["cost_usd"] == 0.0


class TestTruncationKeepsWhatItPaidFor:
    """The cap is read AFTER the verdict. A tool call is the whole of what the
    checker is asked for, so a reply that made one and then ran out of room
    answered the question."""

    def test_a_complete_verdict_on_a_capped_reply_is_kept(self):
        client = _client_returning_content(
            [_verdict_block("challenge", "the quote gives no denominator")],
            stop_reason="max_tokens")
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client)
        assert result["verdict"] == "challenge"
        assert result["rationale"] == "the quote gives no denominator"

    def test_a_capped_reply_with_an_unusable_verdict_still_names_the_cap(self):
        # The verdict is outside the vocabulary, so nothing was recovered and
        # the cap is the fault worth naming — it is the line an operator would
        # edit. Two faults describe this one reply, and the cap is the one
        # that CAUSED the other: reported as an invalid verdict it would send
        # an operator to look at a checker model that answered as well as a
        # cut-off reply allowed. Not re-asked: a cap that truncated once will
        # truncate again.
        client = _client_returning_content(
            [_verdict_block("maybe", "unsure")], stop_reason="max_tokens")
        with pytest.raises(CheckerError, match="max_tokens cap") as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client)
        assert "Invalid verdict" not in str(exc.value)
        assert client.messages.stream.call_count == 1

    def test_a_capped_reply_with_no_verdict_at_all_names_the_cap(self):
        # The other arm of the same guard: the cap cut the reply off before
        # the tool call, so there is no verdict to keep. Read as an ordinary
        # tool-free reply this would buy a second ask under the same cap and
        # then report a checker that declined to call its tool — two claims
        # about the model, for what is a number in pipeline.yaml.
        client = _client_returning_content(
            [_Text("Looking at the quote,")], stop_reason="max_tokens")
        with pytest.raises(CheckerError, match="max_tokens cap") as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client)
        assert "no record_verdict call" not in str(exc.value)
        assert client.messages.stream.call_count == 1


class TestAReplyWithNothingToReadAVerdictFrom:
    """A verdict is an object of named arguments. Anything else is an absence
    of one, handled as any other tool-free reply and never as an exception
    thrown from the middle of the check."""

    def test_a_tool_input_that_is_not_an_object_is_a_tool_free_reply(self):
        client = _client_returning_streams(
            _stream_returning_content([_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                                                "ok")]),
            _stream_returning_content([_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                                                "ok")]),
        )
        with pytest.raises(CheckerError, match="no record_verdict call") as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client)
        # What arrived is named, so the failure is not reported as a model
        # that declined to call the tool when it called it with a string.
        assert "not an object of arguments" in str(exc.value)
        assert client.messages.stream.call_count == 2

    def test_a_tool_call_with_no_arguments_at_all_is_named(self):
        # Reads back as "no tool call and no reason why", which would leave
        # the failure message trailing a bare `None`.
        client = _client_returning_streams(
            _stream_returning_content([_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                                                None)]),
            _stream_returning_content([_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                                                None)]),
        )
        with pytest.raises(CheckerError, match="no arguments at all"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client)

    def test_it_degrades_the_field_rather_than_the_batch(self):
        client = _client_returning_content(
            [_ToolUse(CHECKER_VERDICT_TOOL_NAME, ["ok"])])
        results = run_checker_batch(
            calls=[{"field_path": "study.x", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=_config(), client=client)
        assert results["study.x"]["error_origin"] is True


class TestCheckerConfig:
    def test_fingerprint_stable(self, synthetic_template, tmp_path,
                                 monkeypatch):
        sys_path = tmp_path / "sys.md"
        sys_path.write_text("you are a checker", encoding="utf-8")
        cfg = CheckerConfig(max_tokens=1024, 
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
        )
        a = cfg.fingerprint(synthetic_template, predicates=PREDICATES,
                            max_checks_per_field=2)
        b = cfg.fingerprint(synthetic_template, predicates=PREDICATES,
                            max_checks_per_field=2)
        assert a == b
        assert a.startswith("checker_fp:")

    def test_fingerprint_changes_with_prompt(
            self, synthetic_template, tmp_path):
        sys_path = tmp_path / "sys.md"

        sys_path.write_text("v1", encoding="utf-8")
        cfg1 = CheckerConfig(max_tokens=1024, 
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
        )
        fp1 = cfg1.fingerprint(synthetic_template, predicates=PREDICATES,
                               max_checks_per_field=2)

        sys_path.write_text("v2", encoding="utf-8")
        cfg2 = CheckerConfig(max_tokens=1024, 
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
        )
        fp2 = cfg2.fingerprint(synthetic_template, predicates=PREDICATES,
                               max_checks_per_field=2)

        assert fp1 != fp2

    def test_fingerprint_hashes_substituted_reference_list(
            self, synthetic_template, tmp_path):
        # The checker LLM sees the reference-substituted system prompt, so
        # editing a reference list that appears in the checker system prompt
        # must move checker_fp even though the raw prompt file is unchanged.
        sys_path = tmp_path / "sys.md"
        sys_path.write_text(
            "You are a checker. Canonical tools:\n{reference:gauge_list}",
            encoding="utf-8")
        cfg = CheckerConfig(max_tokens=1024, 
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
        )
        fp_a = cfg.fingerprint(
            synthetic_template, {"gauge_list": [{"tool_name": "WDS-9"}]},
            predicates=PREDICATES, max_checks_per_field=2)
        fp_b = cfg.fingerprint(
            synthetic_template, {"gauge_list": [{"tool_name": "SRI-7"}]},
            predicates=PREDICATES, max_checks_per_field=2)
        assert fp_a != fp_b

    def test_fingerprint_changes_with_checker_context_fields(
            self, synthetic_template, tmp_path):
        # The template's checker context fields drive every per-record checker
        # call's context label (build_record_context), so reversing them must
        # move checker_fp.
        sys_path = tmp_path / "sys.md"
        sys_path.write_text("you are a checker", encoding="utf-8")
        cfg = CheckerConfig(max_tokens=1024, 
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
        )
        fp_before = cfg.fingerprint(synthetic_template,
                                    predicates=PREDICATES,
                                    max_checks_per_field=2)
        synthetic_template["checker_context_fields"] = list(
            reversed(synthetic_template["checker_context_fields"]))
        fp_after = cfg.fingerprint(synthetic_template,
                                   predicates=PREDICATES,
                                   max_checks_per_field=2)
        assert fp_before != fp_after

    def test_from_env_ignores_checker_decoding_env_vars(self, monkeypatch):
        # The checker's decoding knobs come from the config bundle and NOT the
        # environment: CHECKER_TEMPERATURE and CHECKER_MAX_TOKENS are not read
        # at all, so from_env leaves both unspecified whatever the shell
        # holds. Reading them would let checker_fp differ between two machines
        # running the same config bundle.
        monkeypatch.setenv("CHECKER_TEMPERATURE", "0.9")
        monkeypatch.setenv("CHECKER_MAX_TOKENS", "77")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6")
        assert cfg.sampling is None
        assert cfg.max_tokens is None

    def test_from_env_refuses_a_zero_concurrency_override(self, monkeypatch):
        # 0 workers is not a thread pool and not a way to disable the checker.
        # Under a truthy presence test it would read as "no override given"
        # and quietly hand back the environment's value or the built-in
        # default, running the checker at a parallelism nobody asked for.
        monkeypatch.setenv("CHECKER_CONCURRENCY", "9")
        with pytest.raises(ValueError, match="positive integer"):
            CheckerConfig.from_env(model_override="claude-sonnet-4-6",
                                   concurrency_override=0)

    def test_from_env_refuses_a_negative_concurrency_override(self):
        with pytest.raises(ValueError, match="positive integer"):
            CheckerConfig.from_env(model_override="claude-sonnet-4-6",
                                   concurrency_override=-4)

    def test_an_override_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("CHECKER_CONCURRENCY", "9")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6",
                                     concurrency_override=3)
        assert cfg.concurrency == 3

    def test_the_environment_still_supplies_the_value_with_no_override(
            self, monkeypatch):
        # The env fallback is untouched by the guard above: its EFFECTIVE value
        # is checked one step later by the CLI, with the operator-facing
        # message (see tests/test_cli_loop_config.py).
        monkeypatch.setenv("CHECKER_CONCURRENCY", "0")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6")
        assert cfg.concurrency == 0
