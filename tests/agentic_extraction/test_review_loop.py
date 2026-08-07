"""The final reviewer's bounded tool loop.

The reviewer's catalogue includes the read-only `view_*` tools, and the stage
loops rather than taking one model call: inspect, act, then `mark_complete`. A
reviewer that opens by inspecting a record ("let me look at record 35 first,
which appears to have a clear internal contradiction") gets that result fed
back and can act on it. A single call with no tool results returned would
swallow the question and discard the insight: a reviewer handed inspection
tools but unable to see what they say is a design contradiction.

These drive the REAL `run()` and the REAL `_review_loop` with faked provider
calls (a fake adapter returning canned NormalisedResponses, following
test_pipeline_structure.py). The extractor loop is stubbed at its own seam and
the reviewer-side checker is off (the shipped default), so nothing touches the
network, but every assertion below is about code that really ran: no
hand-rolled event dicts.

Covered:
  - view -> edit -> mark_complete: all three turns happen, the edit lands
  - a reviewer that only views still finalises review_clean
  - the bound terminates a reviewer that never stops, with an honest status
  - the budget is not model-visible
  - every review turn logs an assistant_message
  - abandon_extraction is honoured (failed_validation + review_surrendered)
  - spend accrues across multiple review turns
  - the text-only re-prompt, and its bound
  - the repeated-identical-failure guard
  - review turns never replay into the EXTRACTOR conversation
  - `mark_complete` is DISPATCHED (it carries the reviewer's own quality check,
    filed under its own role key) and still ends the review
"""

import copy
import json
from types import SimpleNamespace

import pytest

from meltiro import prompt_builder
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.extraction_record import ROLE_EXTRACTOR
from meltiro.orchestrator import Orchestrator
from direktoro import NormalisedResponse, NormalisedUsage
from meltiro.rates import Rates
from meltiro.run_log import load_log
from meltiro.session import result_to_model_text
from meltiro.statuses import VALIDATED_STATUSES
from meltiro.tools import get_tool_definitions

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


# The shipped template's one REQUIRED quality-check variable. `mark_complete`
# takes the caller's quality check as a required argument, so a scripted
# conclusion has to carry one. The reviewer's is never gated on it (it is the
# only exit from a fresh-context loop), but a script that omitted it would be
# modelling a reviewer the tool schema does not permit.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


def _mark_complete(tool_id, summary):
    return _tool_use(tool_id, "mark_complete",
                     {"summary": summary,
                      "quality_check": dict(QUALITY_CHECK)})


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _resp(*blocks, usage=None):
    return NormalisedResponse(
        content=list(blocks),
        usage=usage or NormalisedUsage(),
        resolved_model=REVIEWER, provider="anthropic", base_url=None,
        raw_request={"model": REVIEWER}, raw_response={},
        wire_request={"model": REVIEWER}, decoding_params={"max_tokens": 1024},
    )


class _ScriptedAdapter:
    """Returns a scripted sequence of responses, one per review turn, and
    records what each call was made with.

    The recorded kwargs are DEEP-COPIED: the loop appends to its `messages`
    list in place, so holding the reference would show every call the final
    conversation rather than what that turn actually sent.

    The last response repeats once the script is exhausted, which is what lets
    a test model a reviewer that never stops.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create_message(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


def _orch(config_dir, bundle_dir, out_dir, *, max_checks_per_field=2,
          max_review_tool_calls=30):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(checker_model=CHECKER, api_key="x"),
        review_model=REVIEWER,
        max_checks_per_field=max_checks_per_field,
        max_review_tool_calls=max_review_tool_calls,
        final_review=True,
        # A card for every role, so the reviewer's turns produce a dollar
        # figure at all. What is under test here is that every turn's spend is
        # accumulated and checkpointed, not what any of it costs.
        rates={role: Rates(input_per_1m=15.0, output_per_1m=75.0,
                           cache_read_per_1m=1.5, cache_write_per_1m=18.75)
               for role in ("extractor", "checker", "review")},
        api_key="x",
    )


def _prepared(config_dir, bundle_dir, out_dir, responses, **kwargs):
    """A prepared orchestrator whose review stage will run the REAL loop
    against `responses`, with the extractor loop stubbed. The reviewer-side
    checker is off (the shipped default), so the loop makes no checker call."""
    orch = _orch(config_dir, bundle_dir, out_dir, **kwargs)
    orch.prepare_new_session()

    def _extractor():
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"

    orch._extractor_loop = _extractor
    adapter = _ScriptedAdapter(responses)
    orch._adapter_for_role = lambda role: adapter
    return orch, adapter


def _events(orch, name):
    return [e for e in orch.session.read_events() if e.get("event") == name]


def _a_study_var(orch):
    """A real study-field variable from the shipped template."""
    return next(iter(orch.dispatcher._study_field_specs))


def _a_record_id(orch):
    return orch.extraction_record.records[0]["record_id"]


# ---------------------------------------------------------------------------
# view -> edit -> mark_complete
# ---------------------------------------------------------------------------

def test_reviewer_views_then_edits_then_completes(
        config_dir, bundle_minimal_dir, tmp_path):
    """The whole point: the reviewer inspects, SEES the result, then acts.

    Three turns must all happen, the edit must land on disk, and the run must
    finalise. A single-call review stage would end at turn 1's view.
    """
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_text("Let me look at what was extracted first."),
                  _tool_use("t1", "view_summary", {})),
            None,  # placeholder; replaced below once the var name is known
            _resp(_mark_complete("t3", "fixed aim")),
        ])
    var = _a_study_var(orch)
    adapter._responses[1] = _resp(
        _text("The primary aim is wrong; correcting it."),
        _tool_use("t2", "update_study",
                  {"study": {var: {"value": None, "evidence": None}}}),
    )

    status = orch.run()

    # All three turns really happened, in order. `mark_complete` is dispatched
    # like any other call, because it carries the reviewer's own quality check
    # and that has to be recorded; what it does not do is gate.
    assert len(adapter.calls) == 3
    tool_calls = _events(orch, "review_tool_call")
    assert [e["tool"] for e in tool_calls] == [
        "view_summary", "update_study", "mark_complete"]
    assert tool_calls[-1]["result"]["status"] == "ok"

    # Turn 2 saw turn 1's tool result: the loop fed it back rather than
    # discarding it. The reviewer's second call was made with the view's
    # result already in the conversation.
    second_turn_messages = adapter.calls[1]["messages"]
    tool_results = [b for m in second_turn_messages
                    if m["role"] == "user"
                    for b in m["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "t1"
    assert "view" in tool_results[0]["content"]

    # The edit landed on disk.
    assert var in orch.extraction_record.study
    saved = json.loads(orch.session.extraction_record_path.read_text())
    assert var in saved["study"]

    # So did the reviewer's quality check, under its own role key. The
    # extractor loop was stubbed and recorded none, so the block holds the
    # reviewer's alone: two opinions kept apart, never one with the author
    # lost.
    assert saved["quality_check"] == {"review": QUALITY_CHECK}

    # The edit is booked as applied, so the nothing-landed branch stays quiet.
    # `mark_complete` is not a mutation, so dispatching it changes neither
    # count.
    assert _events(orch, "final_review_edits_none_applied") == []
    assert status == "complete"


# ---------------------------------------------------------------------------
# Read-only reviewer: review_clean, and not misreported as a failed edit
# ---------------------------------------------------------------------------

def test_reviewer_that_only_views_finalises_clean(
        config_dir, bundle_minimal_dir, tmp_path):
    """A reviewer that only READ the extraction output changed nothing, and
    the run finalises complete without its reads being booked as edits.

    This is the read-only-vs-mutation accounting: `view_*` calls are not edits.
    """
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_tool_use("t1", "view_summary", {})),
            _resp(_tool_use("t2", "view_study_fields", {})),
            _resp(_text("Everything checks out."),
                  _mark_complete("t3", "verified")),
        ])

    status = orch.run()

    assert len(adapter.calls) == 3
    assert [e["tool"] for e in _events(orch, "review_tool_call")] == [
        "view_summary", "view_study_fields", "mark_complete"]
    # Read-only: attempted no edit, so not misreported as failed edits either.
    # `mark_complete` is dispatched but is not in MUTATING_TOOLS, so it is not
    # an edit and does not turn a reader into a writer.
    assert _events(orch, "final_review_edits_none_applied") == []
    assert status == "complete"


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------

def test_reviewer_that_never_stops_hits_the_bound_and_says_so(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    # A reviewer that inspects forever and never concludes. The scripted
    # adapter repeats its last response, so only the bound can stop this.
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [_resp(_tool_use("t", "view_summary", {}))],
        max_review_tool_calls=4)

    status = orch.run()

    # Bounded exactly, and the run is terminal with an honest reason.
    assert len(_events(orch, "review_tool_call")) == 4
    assert status == "failed_validation"
    assert status not in VALIDATED_STATUSES
    meta = orch.session.meta
    assert meta["status"] == "failed_validation"
    assert meta["failure_reason"] == "review_cap_hit"
    # Not a pause: the review conversation is fresh-context and is never
    # replayed, so there is nothing for a resume to continue.
    assert "pause_reason" not in meta
    cap_hit = _events(orch, "review_cap_hit")
    assert len(cap_hit) == 1
    assert cap_hit[0]["review_tool_calls"] == 4
    # The reviewer never confirmed, so the extraction is never shipped as
    # complete, and the run log says the validation did not pass.
    entry = load_log(out)[0]
    assert entry["status"] == "failed_validation"
    assert entry["validation_passed"] is False


def test_review_calls_do_not_consume_the_extractor_budget(
        config_dir, bundle_minimal_dir, tmp_path):
    """The reviewer's calls are counted per review conversation, not against
    `meta.tool_call_count`, which is the extractor's count and provenance."""
    out = tmp_path / "runs"
    orch, _ = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_tool_use("t1", "view_summary", {})),
            _resp(_mark_complete("t2", "ok")),
        ])

    assert orch.run() == "complete"
    # The view and the dispatched mark_complete: two review calls.
    assert len(_events(orch, "review_tool_call")) == 2
    # The stubbed extractor dispatched nothing, and neither of the reviewer's
    # calls leaked into the extractor's counter.
    assert orch.session.meta["tool_call_count"] == 0


# ---------------------------------------------------------------------------
# The budget is not model-visible
# ---------------------------------------------------------------------------

def test_review_budget_is_not_model_visible(
        config_dir, bundle_minimal_dir, tmp_path):
    """The cap rides in no fingerprint, so the reviewer must not be able to
    read a cap-derived number: it would be reasoning from a value that leaves
    no provenance trace. It survives as UI-only telemetry only."""
    out = tmp_path / "runs"
    # A distinctive bound, unlikely to occur by chance in a tool result.
    cap = 17
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_tool_use("t1", "view_summary", {})),
            _resp(_mark_complete("t2", "ok")),
        ],
        max_review_tool_calls=cap)

    assert orch.run() == "complete"

    # The telemetry IS captured in the event log, for the transcript.
    call = _events(orch, "review_tool_call")[0]
    assert call["result"]["_tool_call_budget_remaining"] == cap

    # ...but the model-facing serialisation strips it: no budget key, and no
    # cap-derived number, in what the reviewer actually saw.
    model_text = result_to_model_text(call["result"])
    assert "_tool_call_budget_remaining" not in model_text
    assert "budget" not in model_text
    assert str(cap) not in model_text

    # The same, checked against the real tool_result blocks sent on the wire.
    second_turn_messages = adapter.calls[1]["messages"]
    sent = [b["content"] for m in second_turn_messages
            if m["role"] == "user"
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert sent and all(str(cap) not in s for s in sent)
    assert all("budget" not in s for s in sent)

    # Nor does the cap reach the reviewer through its rendered system prompt
    # (config_bundle bans a `{max_review_tool_calls}` placeholder outright).
    assert str(cap) not in adapter.calls[0]["system"][0]["text"]


# ---------------------------------------------------------------------------
# assistant_message per turn
# ---------------------------------------------------------------------------

def test_every_review_turn_logs_an_assistant_message(
        config_dir, bundle_minimal_dir, tmp_path):
    """Every model turn that dispatches tools records its verbatim ordered
    content. For the reviewer this is also what puts its reasoning in the
    transcript at all, rather than only in the raw API log."""
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_text("Checking record 1."),
                  _tool_use("t1", "view_summary", {})),
            _resp(_mark_complete("t2", "ok")),
        ])

    assert orch.run() == "complete"

    messages = _events(orch, "assistant_message")
    review_messages = [e for e in messages if e.get("stage") == "review"]
    assert len(review_messages) == len(adapter.calls) == 2
    # One per turn, each with a distinct turn id.
    assert len({e["turn_id"] for e in review_messages}) == 2
    # Verbatim and ORDERED: the provider's text-then-tool_use order is kept,
    # not forced text-first or dropped.
    assert review_messages[0]["content"] == [
        {"type": "text", "text": "Checking record 1."},
        {"type": "tool_use", "id": "t1", "name": "view_summary", "input": {}},
    ]


def test_review_turns_never_replay_into_the_extractor_conversation(
        config_dir, bundle_minimal_dir, tmp_path):
    """The review conversation is fresh-context and is not `self.messages`.
    Its turn traffic carries `stage: "review"` so a resume (after a crash
    mid-review, the one path that can reach it) rebuilds only the extractor's
    turns and never splices reviewer turns into them."""
    out = tmp_path / "runs"
    orch, _ = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_text("Inspecting."), _tool_use("t1", "view_summary", {})),
            _resp(_mark_complete("t2", "ok")),
        ])
    assert orch.run() == "complete"

    # Review traffic is in the log...
    assert len(_events(orch, "assistant_message")) == 2
    assert len(_events(orch, "review_tool_call")) == 2
    # ...but replay sees none of it. (The extractor loop was stubbed, so it
    # contributed no turns of its own: anything here would be the reviewer's.)
    assert orch.session.replay_messages() == []


# ---------------------------------------------------------------------------
# abandon_extraction is honoured
# ---------------------------------------------------------------------------

def test_reviewer_abandon_finalises_failed_validation(
        config_dir, bundle_minimal_dir, tmp_path):
    """A reviewer's `abandon_extraction` is honoured, mirroring the extractor's
    surrender path: the run finalises `failed_validation`, not `complete`. The
    flag means the same thing from either stage (this model, with tool access
    and the whole paper, judges that no valid extraction can be produced
    honestly from these inputs)."""
    out = tmp_path / "runs"
    reason = ("the paper reports no relationship between a gauge and any "
              "lifecycle outcome; every extracted record is spurious")
    orch, _ = _prepared(
        config_dir, bundle_minimal_dir, out,
        [_resp(_text("This extraction cannot be salvaged."),
               _tool_use("t1", "abandon_extraction", {"reason": reason}))])

    status = orch.run()

    assert status == "failed_validation"
    assert status not in VALIDATED_STATUSES
    meta = orch.session.meta
    assert meta["status"] == "failed_validation"
    # Review-prefixed, so run.json alone says WHICH stage surrendered.
    assert meta["failure_reason"] == "review_surrendered"
    assert meta["failed_validation_reason"] == reason
    abandoned = _events(orch, "review_abandoned")
    assert len(abandoned) == 1
    assert abandoned[0]["reason"] == reason

    # No data is lost: the extraction output is still written for inspection,
    # exactly as on the extractor's surrender path. The status says "do not
    # trust this", not "throw it away".
    assert orch.session.extraction_record_path.exists()

    entry = load_log(out)[0]
    assert entry["status"] == "failed_validation"
    assert entry["validation_passed"] is False
    assert any(reason in e for e in entry["validation_errors"])


def test_reviewer_abandon_wins_over_mark_complete_in_the_same_batch(
        config_dir, bundle_minimal_dir, tmp_path):
    # Mirrors the extractor loop's precedence: a surrender in the batch beats a
    # mark_complete alongside it.
    out = tmp_path / "runs"
    orch, _ = _prepared(
        config_dir, bundle_minimal_dir, out,
        [_resp(_tool_use("t1", "abandon_extraction", {"reason": "unreadable"}),
               _mark_complete("t2", "fine"))])

    assert orch.run() == "failed_validation"
    assert orch.session.meta["failure_reason"] == "review_surrendered"


def test_reviewer_abandon_without_a_reason_does_not_end_the_run(
        config_dir, bundle_minimal_dir, tmp_path):
    """The dispatcher requires a reason, so a reason-less surrender fails
    validation, does not latch, and the loop simply continues: an accidental
    surrender cannot kill a run."""
    out = tmp_path / "runs"
    orch, _ = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_tool_use("t1", "abandon_extraction", {"reason": "  "})),
            _resp(_mark_complete("t2", "ok")),
        ])

    assert orch.run() == "complete"
    assert orch.extraction_record.abandoned_flag is False
    assert _events(orch, "review_abandoned") == []
    # A failed abandon is not a mutation, so nothing books it as an edit.
    assert _events(orch, "final_review_edits_none_applied") == []


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------

def test_spend_accrues_across_multiple_review_turns(
        config_dir, bundle_minimal_dir, tmp_path):
    """Every review turn's usage is accumulated and checkpointed to meta as it
    accrues, so a crash mid-review cannot lose the cost."""
    out = tmp_path / "runs"
    usage = NormalisedUsage(input_tokens=1000, output_tokens=500,
                            cache_read_input_tokens=200,
                            cache_creation_input_tokens=100)
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_tool_use("t1", "view_summary", {}), usage=usage),
            _resp(_tool_use("t2", "view_study_fields", {}), usage=usage),
            _resp(_mark_complete("t3", "ok"), usage=usage),
        ])

    assert orch.run() == "complete"
    assert len(adapter.calls) == 3

    # Three turns' tokens, summed: no turn's usage is dropped or double-counted.
    meta = orch.session.meta
    assert meta["input_tokens"] == 3000
    assert meta["output_tokens"] == 1500
    assert meta["cache_read_tokens"] == 600
    assert meta["cache_creation_tokens"] == 300
    assert meta["cost_usd"] > 0

    # One priced call per turn, and the run log agrees with meta.
    entry = load_log(out)[0]
    assert entry["input_tokens"] == 3000
    assert entry["output_tokens"] == 1500
    assert entry["cost_usd"] == pytest.approx(meta["cost_usd"])


def test_spend_is_checkpointed_to_meta_before_the_run_finalises(
        config_dir, bundle_minimal_dir, tmp_path):
    """A crash mid-review must not lose the cost already incurred: meta is
    written as the spend accrues, not only at finalisation."""
    out = tmp_path / "runs"
    usage = NormalisedUsage(input_tokens=1000, output_tokens=500)
    seen = {}
    orch, _ = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_tool_use("t1", "view_summary", {}), usage=usage),
            _resp(_mark_complete("t2", "ok"), usage=usage),
        ])

    # Snapshot run.json ON DISK partway through the review, exactly as a crash
    # would see it: the dispatcher runs mid-turn, after turn 1 was priced.
    # The spy mirrors `dispatch`'s full signature, `role` included: one
    # dispatcher serves both loops, and the review loop passes the role
    # explicitly.
    real_dispatch = orch.dispatcher.dispatch

    def _spy(name, args, meta=None, role=ROLE_EXTRACTOR):
        if not seen:
            seen.update(json.loads(orch.session.meta_path.read_text()))
        return real_dispatch(name, args, meta=meta, role=role)

    orch.dispatcher.dispatch = _spy
    assert orch.run() == "complete"

    # Turn 1's spend was already durable on disk before turn 2 ran.
    assert seen["input_tokens"] == 1000
    assert seen["cost_usd"] > 0


# ---------------------------------------------------------------------------
# Text-only turns
# ---------------------------------------------------------------------------

def test_text_only_reviewer_is_reprompted_and_can_recover(
        config_dir, bundle_minimal_dir, tmp_path):
    """A reviewer that narrates its conclusion without calling mark_complete is
    a real failure mode. Asking once is far cheaper than discarding the review,
    so it gets the extractor's re-prompt treatment."""
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_text("The extraction looks correct to me.")),
            _resp(_mark_complete("t1", "verified")),
        ])

    assert orch.run() == "complete"
    assert len(adapter.calls) == 2
    reprompts = _events(orch, "review_reprompt")
    assert len(reprompts) == 1
    assert "mark_complete" in reprompts[0]["text"]
    # The re-prompt really reached the model, after its own text turn: the
    # conversation keeps user/assistant alternation.
    roles = [m["role"] for m in adapter.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    # The text-only turn still recorded its verbatim content.
    assert len(_events(orch, "assistant_message")) == 2
    # The text on the WIRE is prompt_builder's constant, not an inline copy in
    # the loop. Engine prose has one home per piece of wording, so a reader of
    # the source can see everything the reviewer is sent by reading that
    # module; an inline copy here would drift from it silently.
    sent = adapter.calls[1]["messages"][2]["content"][0]["text"]
    assert sent == prompt_builder.REVIEW_TOOL_REPROMPT
    assert reprompts[0]["text"] == prompt_builder.REVIEW_TOOL_REPROMPT


def test_forcing_reviewer_reprompt_is_not_an_auto_degrade_retry(
        config_dir, bundle_minimal_dir, tmp_path):
    """A forcing reviewer (the default Claude REVIEWER) that narrates before
    calling mark_complete is still re-prompted (the general guard), but that is
    NOT an auto-degrade retry and is never counted in meta."""
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_text("The extraction looks correct to me.")),
            _resp(_mark_complete("t1", "verified")),
        ])

    assert orch.run() == "complete"
    # The re-prompt fired (proves the site ran)...
    assert len(_events(orch, "review_reprompt")) == 1
    # ...but a forcing model is never counted as an auto-degrade retry.
    assert "auto_degrade_retries" not in orch.session.meta


def test_non_forcing_reviewer_reprompt_records_auto_degrade_retry(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    """When the reviewer's model cannot force a tool (a routed GLM endpoint,
    simulated here so the priced-Claude cost path is untouched), a tool-free
    re-prompt is recorded as one auto-degrade retry for the review stage."""
    out = tmp_path / "runs"
    monkeypatch.setattr(
        "meltiro.orchestrator.supports_forced_tool_choice", lambda m: False)
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [
            _resp(_text("The extraction looks correct to me.")),
            _resp(_mark_complete("t1", "verified")),
        ])

    assert orch.run() == "complete"
    assert len(_events(orch, "review_reprompt")) == 1
    assert orch.session.meta["auto_degrade_retries"] == {"review": 1}


def test_text_only_reviewer_spiral_is_bounded(
        config_dir, bundle_minimal_dir, tmp_path):
    """A reviewer that never calls a tool dispatches nothing, so the tool-call
    bound alone would never advance. The text-only bound is what makes
    termination total."""
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [_resp(_text("I am thinking about it."))])

    status = orch.run()

    assert len(adapter.calls) == orch.max_consecutive_text_only_turns == 3
    assert status == "failed_validation"
    assert orch.session.meta["failure_reason"] == "review_text_only_stall"
    stall = _events(orch, "review_text_only_stall")
    assert len(stall) == 1
    assert stall[0]["consecutive_text_only_turns"] == 3


def test_empty_review_completion_is_an_infrastructure_error(
        config_dir, bundle_minimal_dir, tmp_path):
    # Neither text nor a tool call is an empty completion: an infrastructure
    # failure, not a judgement about the extraction.
    out = tmp_path / "runs"
    orch, _ = _prepared(config_dir, bundle_minimal_dir, out, [_resp()])

    assert orch.run() == "error"
    assert len(_events(orch, "final_review_no_response")) == 1


# ---------------------------------------------------------------------------
# Repeated identical failures
# ---------------------------------------------------------------------------

def test_reviewer_wedged_on_an_identical_failure_stalls(
        config_dir, bundle_minimal_dir, tmp_path):
    """The extractor's repeated-failure guard, shared: a reviewer wedged
    re-submitting the SAME rejected call stops well before the bound, so we
    stop paying for a call that keeps failing identically."""
    out = tmp_path / "runs"
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, out,
        [_resp(_tool_use("t", "update_study",
                         {"study": {"not_a_real_field": {"value": "x",
                                                         "evidence": None}}}))],
        max_review_tool_calls=100)

    status = orch.run()

    # Stopped by the guard at 5, far short of the bound of 100.
    assert len(_events(orch, "review_tool_call")) == \
        orch.max_consecutive_identical_failures == 5
    assert status == "failed_validation"
    assert orch.session.meta["failure_reason"] == "review_stalled"
    stall = _events(orch, "review_repeated_failure_stall")
    assert len(stall) == 1
    assert stall[0]["tool"] == "update_study"
    assert stall[0]["consecutive_identical_failures"] == 5
    # The stall path still records the turn's verbatim assistant content, so
    # the "every tool-calling turn logs an assistant_message" rule stays total.
    assert len(_events(orch, "assistant_message")) == len(adapter.calls)


# ---------------------------------------------------------------------------
# The bound is an operational budget, not config identity
# ---------------------------------------------------------------------------

def test_changing_the_review_bound_moves_no_fingerprint(
        config_dir, bundle_minimal_dir, tmp_path):
    """`max_review_tool_calls` is an operational budget, exactly like the
    extractor's cap: it is not in the prompt, the tools, or the decoding params,
    so two runs that differ only in it ask the reviewer the identical question
    with the identical instrument and must share every fingerprint. Folding it
    into review_fp would report a methodology change where none happened."""
    a = _orch(config_dir, bundle_minimal_dir, tmp_path / "a",
              max_review_tool_calls=10)
    a.prepare_new_session()
    b = _orch(config_dir, bundle_minimal_dir, tmp_path / "b",
              max_review_tool_calls=99)
    b.prepare_new_session()

    assert a.session.meta["review_fp"] == b.session.meta["review_fp"]
    assert a.session.meta["config_fp"] == b.session.meta["config_fp"]
    assert a.session.meta["checker_fp"] == b.session.meta["checker_fp"]
    assert a.session.meta["prompt_hash"] == b.session.meta["prompt_hash"]


def test_meta_records_the_review_bound_per_segment(
        config_dir, bundle_minimal_dir, tmp_path):
    """The provenance home of a budget that rides in no fingerprint: meta.caps,
    per segment, plus a `resumed` event when a segment changes it."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, max_review_tool_calls=12)
    orch.prepare_new_session()
    assert orch.session.meta["caps"]["max_review_tool_calls"] == 12
    session_dir = orch.session.session_dir

    orch2 = _orch(config_dir, bundle_minimal_dir, out, max_review_tool_calls=25)
    orch2.resume_session(session_dir)
    assert orch2.session.meta["caps"]["max_review_tool_calls"] == 25
    resumed = [e for e in orch2.session.read_events()
               if e.get("event") == "resumed"]
    assert len(resumed) == 1
    assert resumed[0]["max_review_tool_calls"] == 25
    assert resumed[0]["previous_max_review_tool_calls"] == 12


# ---------------------------------------------------------------------------
# What the reviewer's provider call actually carried
# ---------------------------------------------------------------------------
#
# Both guarantees below are one keyword each at a single call site in
# `_final_review`, so both are pinned on the WIRE rather than on the source:
# these read what the adapter was handed, which is what the model sees.

def test_the_reviewer_is_sent_its_own_tool_catalogue(
        config_dir, bundle_minimal_dir, tmp_path):
    # `record_initial_check` is the extractor's opening call and is recorded
    # once. Handing it to the reviewer would offer it a door onto the
    # extractor's account of its own run.
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        [_resp(_mark_complete("t1", "checked"))])
    orch.run()

    sent = [t["name"] for t in adapter.calls[0]["tools"]]
    assert "record_initial_check" not in sent
    extractor_names = [
        t["name"] for t in get_tool_definitions(orch.template)]
    assert sent == [n for n in extractor_names
                    if n != "record_initial_check"]
    assert len(sent) == len(extractor_names) - 1

    # And `mark_complete` is described to the reviewer in its own terms, not
    # the extractor's: it concludes a review, it does not end an extraction.
    mc = next(t for t in adapter.calls[0]["tools"]
              if t["name"] == "mark_complete")
    assert "review" in mc["description"].lower()
    assert "quality_check" in mc["input_schema"]["properties"]
    assert mc["input_schema"]["required"] == ["quality_check"]


def test_the_reviewer_is_sent_no_check_block(
        config_dir, bundle_minimal_dir, tmp_path):
    # The extractor's self-assessment is an anchor on the question the
    # reviewer is about to be asked, so it is withheld exactly as the
    # checker's verdicts are. Read off the assembled user message.
    orch, adapter = _prepared(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        [_resp(_mark_complete("t1", "checked"))])
    orch.extraction_record.record_initial_check(
        {"text_readable": True}, role=ROLE_EXTRACTOR)
    orch.extraction_record.record_quality_check(
        {"general_notes": "extractor said this"}, role=ROLE_EXTRACTOR)
    orch.run()

    rendered = "\n".join(
        b.get("text", "") for b in adapter.calls[0]["messages"][0]["content"]
        if isinstance(b, dict))
    # The three absences below are only evidence if the message was read at
    # all: were the user blocks ever to stop being text dicts, `rendered`
    # would collapse to "" and every `not in` would hold for nothing.
    assert orch.bundle.title in rendered
    assert "extractor said this" not in rendered
    assert "initial_check" not in rendered
    assert "quality_check" not in rendered

    # Positive control: both blocks are still recorded, role-keyed, on disk.
    full = orch.extraction_record.to_dict()
    assert full["initial_check"][ROLE_EXTRACTOR]["text_readable"] is True
    assert full["quality_check"][ROLE_EXTRACTOR]["general_notes"] == \
        "extractor said this"
