"""Tests for the checker client.

The Anthropic call itself is mocked; what is under test is the wrapper
logic: reading the verdict tool call off the response, verdict validation,
cost accounting, the re-ask a tool-free reply gets, the parallel fan-out, and
the error-wrapping in run_checker_batch.
"""

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
from direktoro import NormalisedResponse, NormalisedUsage
from direktoro.registry import OPENROUTER_BASE_URL

# The run's structure predicates, which a checker fingerprint takes as an
# argument because it keeps none of its own. A checker-on, reviewer-on
# pipeline is the ordinary shape, and these tests vary the prompts and the
# template rather than the structure, so they hold it fixed.
PREDICATES = stage_predicates(2, True)


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


def _verdict_block(verdict="ok", rationale="matches the quote", notes=None,
                   *, name=CHECKER_VERDICT_TOOL_NAME):
    payload = {"verdict": verdict, "rationale": rationale}
    if notes is not None:
        payload["notes"] = notes
    return _ToolUse(name, payload)


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
                              notes=None, **kw):
    return _stream_returning_content(
        [_verdict_block(verdict, rationale, notes)], **kw)


def _client_returning_content(content, **kw):
    client = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_stream_returning_content(content, **kw))
    return client


def _client_returning(verdict="ok", rationale="matches the quote",
                      notes=None, **kw):
    """A client whose every call answers with the verdict tool."""
    client = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_stream_returning_verdict(
            verdict, rationale, notes, **kw))
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
        api_key="sk-test", checker_model="claude-sonnet-4-6",
        max_tokens=1024, sampling={"temperature": 0.0}, concurrency=4, rates=rates,
    )


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

    def test_notes_ride_through_when_given(self):
        client = _client_returning("ok", "r", notes="borderline")
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["notes"] == "borderline"

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

    def test_call_sends_the_configured_checker_temperature(self):
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
        # re-asked once, as its own single-turn call carrying the nudge, and
        # the verdict that arrives is an ordinary one.
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
        # The re-ask is a fresh single-turn call: one user message, carrying
        # the field again plus the nudge, and no assistant turn replayed back.
        second = client.messages.stream.call_args_list[1].kwargs
        assert [m["role"] for m in second["messages"]] == ["user"]
        assert second["messages"][0]["content"][-1]["text"] == \
            CHECKER_TOOL_REPROMPT
        assert second["messages"][0]["content"][0]["text"] == "field"

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
        # Only the verdict tool answers. Another tool name is a tool-free
        # reply as far as the verdict is concerned.
        client = _client_returning_streams(
            _stream_returning_content(
                [_ToolUse("mark_complete", {"quality_check": {}})]),
            _stream_returning_verdict("ok", "r"),
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["verdict"] == "ok"
        assert result["reprompted"] == 1

    def test_empty_content_is_reasked_then_raises(self):
        # An empty content list is a reply with no verdict in it, handled as
        # any other tool-free reply — and never as an IndexError that would
        # escape the per-field handling.
        client = _client_returning_streams(
            _stream_returning_content([]),
            _stream_returning_content([]),
        )
        with pytest.raises(CheckerError, match="no record_verdict call"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client,
            )

    def test_truncation_is_named_and_not_reasked(self):
        # A cap that truncated once will truncate again, so the ask is not
        # repeated; the message names the cap rather than the model.
        client = _client_returning_content(
            [_Thinking()], stop_reason="max_tokens")
        with pytest.raises(CheckerError, match="max_tokens"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client,
            )
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


class TestCheckOneFieldRoutedCost:
    """A ROUTED (gateway-served) checker model is priced FROM the response
    (OpenRouter usage.cost -> reported_cost). That figure is a charge the
    gateway states rather than one anybody predicts, so it needs no rate card
    and these configs carry none. A missing reported cost is a loud fault,
    NEVER a silent $0."""

    ROUTED = "z-ai/glm-5v-turbo"

    def _config(self):
        return CheckerConfig(api_key="x", checker_model=self.ROUTED,
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

    def test_missing_reported_cost_raises_loudly(self):
        with pytest.raises(RuntimeError, match="no reported cost"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=self._config(), adapter=_StubAdapter(
                    self._response(reported_cost=None)))


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
                api_key="x", checker_model="claude-sonnet-4-6",
                concurrency=1),
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


class TestCheckerConfig:
    def test_fingerprint_stable(self, synthetic_template, tmp_path,
                                 monkeypatch):
        sys_path = tmp_path / "sys.md"
        sys_path.write_text("you are a checker", encoding="utf-8")
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="sk-x",
        )
        a = cfg.fingerprint(synthetic_template, predicates=PREDICATES)
        b = cfg.fingerprint(synthetic_template, predicates=PREDICATES)
        assert a == b
        assert a.startswith("checker_fp:")

    def test_fingerprint_changes_with_prompt(
            self, synthetic_template, tmp_path):
        sys_path = tmp_path / "sys.md"
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")

        sys_path.write_text("v1", encoding="utf-8")
        cfg1 = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp1 = cfg1.fingerprint(synthetic_template, predicates=PREDICATES)

        sys_path.write_text("v2", encoding="utf-8")
        cfg2 = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp2 = cfg2.fingerprint(synthetic_template, predicates=PREDICATES)

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
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp_a = cfg.fingerprint(
            synthetic_template, {"gauge_list": [{"tool_name": "WDS-9"}]},
            predicates=PREDICATES)
        fp_b = cfg.fingerprint(
            synthetic_template, {"gauge_list": [{"tool_name": "SRI-7"}]},
            predicates=PREDICATES)
        assert fp_a != fp_b

    def test_fingerprint_changes_with_checker_context_fields(
            self, synthetic_template, tmp_path):
        # The template's checker context fields drive every per-record checker
        # call's context label (build_record_context), so reversing them must
        # move checker_fp.
        sys_path = tmp_path / "sys.md"
        sys_path.write_text("you are a checker", encoding="utf-8")
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp_before = cfg.fingerprint(synthetic_template,
                                    predicates=PREDICATES)
        synthetic_template["checker_context_fields"] = list(
            reversed(synthetic_template["checker_context_fields"]))
        fp_after = cfg.fingerprint(synthetic_template,
                                   predicates=PREDICATES)
        assert fp_before != fp_after

    def test_from_env_ignores_checker_decoding_env_vars(self, monkeypatch):
        # The checker's decoding knobs come from the config bundle and NOT the
        # environment: CHECKER_TEMPERATURE and CHECKER_MAX_TOKENS are not read
        # at all, so from_env returns the dataclass defaults whatever the shell
        # holds. Reading them would let checker_fp differ between two machines
        # running the same config bundle.
        monkeypatch.setenv("CHECKER_TEMPERATURE", "0.9")
        monkeypatch.setenv("CHECKER_MAX_TOKENS", "77")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6")
        assert cfg.sampling is None
        assert cfg.max_tokens == 1024

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
