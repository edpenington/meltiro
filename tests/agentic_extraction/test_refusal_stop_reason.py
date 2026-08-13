"""A refused turn is an answer, not a turn that happened to carry no tool call.

direktoro maps a host content filter and an outright decline onto one canonical
`stop_reason` of `"refusal"`, ahead of `tool_use`: a response that declined has
declined whatever else it contains.

Read as an ordinary tool-free turn, a refusal is re-prompted, refused again,
and dies three paid calls later as `text_only_stall` — a stall guard naming a
cause that was never the cause. These tests pin the other behaviour on both
paths that read a stop reason: the extractor stops on the first refusal with a
status and a message that name it, and the checker degrades the field the same
way.

A stop reason is also the record of how a turn ENDED, and it is kept for every
turn of every stage: a reviewer turn a filter blocked or the output cap cut off
reaches the transcript with a note saying so, exactly as an extractor turn
does.

Offline: responses are constructed and handed to stub adapters; no provider is
reached.
"""

import json

import pytest

from direktoro import NormalisedResponse, NormalisedUsage
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig, check_one_field
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import CheckerError
from meltiro.orchestrator import Orchestrator
from meltiro.tools import CHECKER_VERDICT_TOOL_NAME
from meltiro.transcript import render_transcript

pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-8"

# The shipped template's one REQUIRED quality-check variable: `mark_complete`
# takes the caller's quality check as an argument, so a scripted conclusion
# has to carry one.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, name, tool_input, block_id="tu-1"):
        self.name = name
        self.input = tool_input
        self.id = block_id


def _response(content, *, model, stop_reason):
    return NormalisedResponse(
        content=list(content),
        usage=NormalisedUsage(input_tokens=100, output_tokens=10),
        resolved_model=model, provider="anthropic", base_url=None,
        raw_request={"model": model}, raw_response={},
        decoding_params={"max_tokens": 4096}, stop_reason=stop_reason)


def _orch(config_dir, bundle_dir, out_dir):
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model=CHECKER),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=50, extractor_max_tokens=4096,
    )
    orch.prepare_new_session()
    orch._adapter_for_role = lambda role: object()
    return orch


def _events(orch, name):
    return [e for e in orch.session.read_events() if e.get("event") == name]


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------

class TestTheExtractorStopsOnTheFirstRefusal:

    def test_one_refused_turn_ends_the_run_as_error(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        calls = []

        def _call(adapter, tool_defs):
            calls.append(1)
            return _response([_Text("I can't help with that.")],
                             model=EXTRACTOR, stop_reason="refusal")
        orch._call_extractor = _call

        status = orch.run()
        err = capsys.readouterr().err

        assert status == "error"
        # ONE call, not four: the text-only bound is three re-prompts away and
        # every one of them would be billed.
        assert len(calls) == 1
        assert orch.session.meta["status"] == "error"
        # The message names the refusal rather than a stall.
        message = orch.session.meta["error_message"]
        assert "refused" in message and "refusal" in message
        assert "refused" in err
        assert _events(orch, "text_only_stall") == []

    def test_the_refusal_is_its_own_event_and_rides_the_turn_record(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch._call_extractor = lambda a, t: _response(
            [_Text("no")], model=EXTRACTOR, stop_reason="refusal")

        orch.run()
        capsys.readouterr()

        refused, = _events(orch, "extractor_refused")
        assert refused["stop_reason"] == "refusal"
        # And the turn's own assistant_message carries it, so the artefact
        # records how the turn ended without the wire log.
        assistant, = _events(orch, "assistant_message")
        assert assistant["stop_reason"] == "refusal"
        assert assistant["turn_id"] == refused["turn_id"]

    def test_the_transcript_says_the_refusal_in_words(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The transcript is what an operator reads to find out why a run
        # stopped. A refusal reaching it as its own JSON is the one outcome
        # the document cannot explain.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch._call_extractor = lambda a, t: _response(
            [_Text("I can't help with that.")],
            model=EXTRACTOR, stop_reason="refusal")

        orch.run()
        capsys.readouterr()

        transcript = (orch.session.session_dir / "diagnostics" /
                      "transcript.md").read_text(encoding="utf-8")
        assert "unrecognised event" not in transcript
        # Three sentences run together here, and each one carries a part the
        # others do not. The turn's own stop note has the mechanism, read off
        # its stop reason, so the prose above it is not read as an answer...
        assert ("the endpoint refused it (`stop_reason` `refusal`) — a host "
                "content filter blocked the reply, or the model declined the "
                "request") in transcript
        # ... the refusal event has what it cost the run ...
        assert "there is no extraction to call valid or invalid" in transcript
        # ... and the error the run stopped with has what to do about it.
        assert ("Check the paper text and the rendered prompt for material "
                "the provider's filter blocks") in transcript

    def test_the_mechanism_survives_a_turn_record_without_the_field(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # `meltiro transcript` renders a session from its files, and a session
        # whose `assistant_message` carries no `stop_reason` still has the
        # refusal event's copy of it. Read off the turn record alone, that
        # document would say what the refusal cost the run and never say a
        # filter or a decline is what happened — the one thing the operator
        # needs before deciding whether to change the prompt or the paper.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch._call_extractor = lambda a, t: _response(
            [_Text("I can't help with that.")],
            model=EXTRACTOR, stop_reason="refusal")

        orch.run()
        capsys.readouterr()

        events = orch.session.read_events()
        assert [e.pop("stop_reason") for e in events
                if e["event"] == "assistant_message"] == ["refusal"]
        orch.session.tool_calls_path.write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

        document = render_transcript(orch.session.session_dir)

        # Once: the refusal event's own sentence says what the run lost, and
        # the mechanism belongs to the turn note above it.
        assert document.count(
            "a host content filter blocked the reply, or the model declined "
            "the request") == 1
        assert "there is no extraction to call valid or invalid" in document

    def test_a_refusal_outranks_a_tool_call_in_the_same_turn(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # direktoro's vocabulary puts `refusal` ahead of `tool_use`: a reply
        # that declined has declined whatever else it emitted. Dispatching the
        # call would act on a turn the provider says was blocked.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch._call_extractor = lambda a, t: _response(
            [_ToolUse("view_summary", {})],
            model=EXTRACTOR, stop_reason="refusal")

        assert orch.run() == "error"
        capsys.readouterr()
        assert orch.session.meta.get("tool_call_count", 0) == 0

    def test_an_ordinary_tool_free_turn_is_still_re_prompted(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The pair to the tests above: `end_turn` with no tool call is the
        # state the re-prompt exists for, and it still gets one.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch._call_extractor = lambda a, t: _response(
            [_Text("Let me think about this.")],
            model=EXTRACTOR, stop_reason="end_turn")

        assert orch.run() == "failed_validation"
        capsys.readouterr()
        assert orch.session.meta["failure_reason"] == "text_only_stall"
        assert _events(orch, "extractor_refused") == []
        # Every turn's stop reason is on its own record.
        assert all(e["stop_reason"] == "end_turn"
                   for e in _events(orch, "assistant_message"))


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

class _StubAdapter:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def create_message(self, **kwargs):
        self.calls += 1
        return self._response


class TestTheCheckerDegradesARefusedField:

    def test_a_refused_check_names_the_refusal(self):
        adapter = _StubAdapter(_response(
            [_Text("I can't assess that.")],
            model=CHECKER, stop_reason="refusal"))
        with pytest.raises(CheckerError) as caught:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=CheckerConfig(checker_model=CHECKER, max_tokens=1024),
                adapter=adapter)
        message = str(caught.value)
        assert "refused" in message and "refusal" in message
        # Not re-asked: the same field, value and evidence would be blocked
        # again, and the re-ask would be billed.
        assert adapter.calls == 1
        # The call was billed, so it is priced like any other failure.
        assert caught.value.spent["input_tokens"] == 100

    def test_a_refusal_outranks_a_verdict_in_the_same_reply(self):
        adapter = _StubAdapter(_response(
            [_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                      {"verdict": "ok", "rationale": "fine"})],
            model=CHECKER, stop_reason="refusal"))
        with pytest.raises(CheckerError, match="refused"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=CheckerConfig(checker_model=CHECKER, max_tokens=1024),
                adapter=adapter)

    def test_an_ordinary_verdict_is_unaffected(self):
        adapter = _StubAdapter(_response(
            [_ToolUse(CHECKER_VERDICT_TOOL_NAME,
                      {"verdict": "ok", "rationale": "fine"})],
            model=CHECKER, stop_reason="tool_use"))
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=CheckerConfig(checker_model=CHECKER, max_tokens=1024),
            adapter=adapter)
        assert result["verdict"] == "ok"
        assert result["error_origin"] is False


# ---------------------------------------------------------------------------
# The reviewer
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    """One response per review turn, in order, the last repeating."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create_message(self, **kwargs):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _review_orch(config_dir, bundle_dir, out_dir, responses):
    """A prepared orchestrator whose REVIEW stage runs the real loop against
    `responses`, with the extractor loop stubbed at its own seam."""
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model=CHECKER),
        review_model=REVIEWER,
        max_checks_per_field=0, final_review=True,
        max_tool_calls=50, extractor_max_tokens=4096, review_max_tokens=4096,
    )
    orch.prepare_new_session()

    def _extractor():
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"

    orch._extractor_loop = _extractor
    adapter = _ScriptedAdapter(responses)
    orch._adapter_for_role = lambda role: adapter
    return orch


class TestAReviewerTurnRecordsHowItEnded:

    def test_a_truncated_reviewer_turn_says_so_in_the_transcript(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A reviewer turn cut off at the output cap, then one that concludes.
        # The truncated turn's prose stops mid-sentence, and a reader with no
        # note above it reads that as the reviewer's whole thought.
        orch = _review_orch(
            config_dir, bundle_minimal_dir, tmp_path / "runs",
            [_response([_Text("The primary aim as extracted does not match "
                              "the abstract, and the sample size is")],
                       model=REVIEWER, stop_reason="max_tokens"),
             _response([_ToolUse("mark_complete",
                                 {"summary": "reviewed",
                                  "quality_check": dict(QUALITY_CHECK)})],
                       model=REVIEWER, stop_reason="tool_use")])

        orch.run()
        capsys.readouterr()

        # The turn's own record carries how it ended, the extractor's rule
        # applied to the stage that also has turns.
        assert [e["stop_reason"] for e in _events(orch, "assistant_message")] \
            == ["max_tokens", "tool_use"]
        transcript = (orch.session.session_dir / "diagnostics" /
                      "transcript.md").read_text(encoding="utf-8")
        review = transcript[transcript.index("## 4. The review"):]
        assert "it hit the output cap (`stop_reason` `max_tokens`)" in review
