"""`diagnostics/field_history.json`: the per-field history, and its aggregate.

Two things are on trial here. First, that each event kind is derived from the
right signal in `tool_calls.jsonl`, particularly `overruled`, which is an
inference from a later write rather than anything the run logs explicitly.
Second, that the file is genuinely DERIVED: rebuild it from the event log
alone and you get the same bytes back, so it is a convenience shape and never
a second source of truth.

The unit tests below drive `build_field_history` on event dicts of the shape
the orchestrator appends. The end-to-end tests run the real extractor and the
real review loop (provider calls and the checker fan-out stubbed) so the
shapes are not taken on trust.
"""

import copy
import json
from types import SimpleNamespace

import pytest

from direktoro import NormalisedResponse, NormalisedUsage
from meltiro import orchestrator as orch_mod
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.field_history import build_field_history
from meltiro.orchestrator import Orchestrator


# Every stage's key variable is present for this module: these tests
# reach the orchestrator's pre-spend key preflight, and the provider
# calls behind it are stubbed.
pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-8"

FIELD_VAR = "title"
FIELD_PATH = f"study.{FIELD_VAR}"
QUOTE = "A synthetic study of baseline CRT-HD scores"

# `mark_complete` carries the caller's quality check as a required argument, so
# a scripted call needs the shipped template's one REQUIRED quality-check
# variable. One helper rather than the literal at each call site.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


def _mark_complete_input(summary):
    return {"summary": summary, "quality_check": dict(QUALITY_CHECK)}


# ---------------------------------------------------------------------------
# Event builders, matching what the orchestrator appends
# ---------------------------------------------------------------------------

def _dispatch(event, *, tool="update_study", turn_id=1, diffs=None,
              failed=None, verdicts=None, applied_changes=None, ts="T"):
    result = {
        "status": "ok",
        "applied_changes": applied_changes or {},
        "_field_diffs": diffs or {},
        "failed_fields": failed or {},
    }
    if verdicts:
        result["_checker_verdicts"] = verdicts
    return {"ts": ts, "event": event, "turn_id": turn_id, "tool": tool,
            "tool_use_id": "t1", "args": {}, "result": result}


def _verdict(kind, *, rationale="off the quote", error_origin=False,
             value="v", evidence="<q>q</q>", note=None, cost=0.001):
    return {
        "verdict": kind, "rationale": rationale, "notes": None,
        "value_checked": value, "evidence_checked": evidence,
        "note_checked": note, "error_origin": error_origin,
        "stage": "extractor", "input_tokens": 1, "output_tokens": 1,
        "cache_creation_tokens": 0, "cache_read_tokens": 0,
        "cost_usd": cost,
    }


def _kinds(history, path=FIELD_PATH):
    return [e["kind"] for e in history["fields"][path]["events"]]


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

class TestEventKinds:
    def test_a_write_is_proposed_then_applied(self):
        h = build_field_history([
            _dispatch("tool_call_applied",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}}),
        ])
        assert _kinds(h) == ["proposed", "applied"]
        applied = h["fields"][FIELD_PATH]["events"][1]
        assert applied["before"] is None
        assert applied["after"] == "v"
        assert applied["changed"] is True
        assert applied["stage"] == "extractor"

    def test_a_validation_failure_is_proposed_then_rejected_with_its_codes(
            self):
        h = build_field_history([
            _dispatch("tool_call_failed", failed={FIELD_PATH: [
                {"path": FIELD_PATH, "code": "quote_not_found",
                 "message": "no such quote"},
                {"path": FIELD_PATH, "code": "type_mismatch",
                 "message": "wrong type"},
            ]}),
        ])
        assert _kinds(h) == ["proposed", "rejected"]
        rejected = h["fields"][FIELD_PATH]["events"][1]
        assert [e["code"] for e in rejected["errors"]] == [
            "quote_not_found", "type_mismatch"]
        assert rejected["errors"][0]["message"] == "no such quote"
        assert h["fields"][FIELD_PATH]["final"]["writes"] == 0
        assert h["fields"][FIELD_PATH]["final"]["rejections"] == 1

    def test_a_partial_call_records_both_outcomes(self):
        h = build_field_history([
            _dispatch("tool_call_partial",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      failed={"study.doi": [{"code": "unknown_field",
                                             "message": "nope"}]}),
        ])
        assert _kinds(h) == ["proposed", "applied"]
        assert _kinds(h, "study.doi") == ["proposed", "rejected"]

    def test_a_check_carries_what_the_checker_scored_including_the_note(self):
        # The note is part of what the checker saw, so a field's history has to
        # show it: without it a reader cannot tell what the verdict was against.
        h = build_field_history([
            _dispatch("tool_call_applied",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict(
                          "ok", value="v", evidence="<q>q</q>",
                          note="taken from Table 2")}),
        ])
        assert _kinds(h) == ["proposed", "applied", "checked"]
        checked = h["fields"][FIELD_PATH]["events"][2]
        assert checked["value_checked"] == "v"
        assert checked["evidence_checked"] == "<q>q</q>"
        assert checked["note_checked"] == "taken from Table 2"
        assert checked["error_origin"] is False

    def test_a_challenge_carries_the_checkers_rationale(self):
        h = build_field_history([
            _dispatch("tool_call_applied",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict(
                          "challenge", rationale="the quote says otherwise")}),
        ])
        assert _kinds(h) == ["proposed", "applied", "challenged"]
        assert h["fields"][FIELD_PATH]["events"][2]["rationale"] == (
            "the quote says otherwise")
        assert h["fields"][FIELD_PATH]["final"]["unresolved_challenge"] is True

    def test_an_exhausted_retry_verdict_is_named_apart_from_a_challenge(self):
        # An error-origin verdict is an absence of information, not an
        # objection, so it is neither a challenge nor something to overrule.
        h = build_field_history([
            _dispatch("tool_call_applied",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict("challenge",
                                                     error_origin=True)}),
        ])
        assert _kinds(h) == ["proposed", "applied", "check_error"]
        assert h["aggregate"]["challenges_raised"] == 0
        assert h["aggregate"]["check_errors"] == 1
        assert h["fields"][FIELD_PATH]["final"]["unresolved_challenge"] is False

    def test_a_changed_write_after_a_challenge_is_a_revision(self):
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1,
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict("challenge")}),
            _dispatch("tool_call_applied", turn_id=2,
                      diffs={FIELD_PATH: {"before": "v", "after": "w"}}),
        ])
        assert _kinds(h) == ["proposed", "applied", "challenged",
                             "proposed", "revised_after_challenge"]
        revision = h["fields"][FIELD_PATH]["events"][4]
        assert revision["answers_challenge"] is True
        assert revision["changed"] is True
        assert h["aggregate"]["challenges_revised"] == 1

    def test_an_unchanged_write_after_a_challenge_is_an_overrule(self):
        # The one event kind that is inferred rather than read off a signal.
        # Nothing records "the extractor considered the challenge and stood by
        # its value". The log carries the challenge, and then a later write
        # that applied the same value again, and re-submitting a value that
        # has just been challenged is standing by it.
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1,
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict("challenge")}),
            _dispatch("tool_call_applied", turn_id=2,
                      diffs={FIELD_PATH: {"before": "v", "after": "v"}}),
        ])
        assert _kinds(h) == ["proposed", "applied", "challenged",
                             "proposed", "overruled"]
        overrule = h["fields"][FIELD_PATH]["events"][4]
        assert overrule["changed"] is False
        assert overrule["answers_challenge"] is True
        assert h["aggregate"]["challenges_overruled"] == 1
        assert h["aggregate"]["challenges_revised"] == 0

    def test_an_unchanged_write_with_no_challenge_is_just_applied(self):
        # The overrule inference is only available where there is a challenge
        # to overrule. A model rewriting a value it was never challenged on is
        # doing nothing of interest.
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1,
                      diffs={FIELD_PATH: {"before": None, "after": "v"}}),
            _dispatch("tool_call_applied", turn_id=2,
                      diffs={FIELD_PATH: {"before": "v", "after": "v"}}),
        ])
        assert _kinds(h) == ["proposed", "applied", "proposed", "applied"]
        assert h["aggregate"]["challenges_overruled"] == 0

    def test_a_challenge_nobody_answered_emits_no_write_event(self):
        # Not overruled: nobody acted. The final state is what says the
        # challenge still stands, and it asks the same question
        # run.checker_diagnostics.unresolved_challenges asks.
        h = build_field_history([
            _dispatch("tool_call_applied",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict("challenge")}),
        ])
        assert "overruled" not in _kinds(h)
        assert h["fields"][FIELD_PATH]["final"]["unresolved_challenge"] is True
        assert h["aggregate"]["fields_with_unresolved_challenge"] == 1

    def test_a_reviewer_write_is_named_for_the_reviewer(self):
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1,
                      diffs={FIELD_PATH: {"before": None, "after": "v"}}),
            _dispatch("review_tool_call", turn_id=2,
                      diffs={FIELD_PATH: {"before": "v", "after": "w"}}),
        ])
        assert _kinds(h) == ["proposed", "applied",
                             "proposed", "revised_by_reviewer"]
        review = h["fields"][FIELD_PATH]["events"][3]
        assert review["stage"] == "review"
        assert h["aggregate"]["fields_reviewer_touched"] == 1
        assert h["fields"][FIELD_PATH]["final"]["last_write_stage"] == "review"

    def test_a_reviewer_write_answering_a_challenge_records_both_facts(self):
        # Naming the writer and answering the checker are two different facts,
        # so one event carries both rather than the derivation having to pick.
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1,
                      diffs={FIELD_PATH: {"before": None, "after": "v"}},
                      verdicts={FIELD_PATH: _verdict("challenge")}),
            _dispatch("review_tool_call", turn_id=2,
                      diffs={FIELD_PATH: {"before": "v", "after": "w"}}),
        ])
        review = h["fields"][FIELD_PATH]["events"][4]
        assert review["kind"] == "revised_by_reviewer"
        assert review["answers_challenge"] is True
        assert h["aggregate"]["challenges_revised"] == 1

    def test_removing_a_record_removes_every_field_on_it(self):
        # `remove_record` writes no field diffs, so the removed_record_id in
        # applied_changes is the only signal those fields are gone.
        a, b = "record.r1.gauge", "record.r1.outcome_variable"
        other = "record.r2.gauge"
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1, tool="add_record",
                      diffs={a: {"before": None, "after": "x"},
                             b: {"before": None, "after": "y"},
                             other: {"before": None, "after": "z"}}),
            _dispatch("tool_call_applied", turn_id=2, tool="remove_record",
                      applied_changes={"removed_record_id": "r1"}),
        ])
        assert _kinds(h, a) == ["proposed", "applied", "removed"]
        assert _kinds(h, b) == ["proposed", "applied", "removed"]
        assert _kinds(h, other) == ["proposed", "applied"]
        removed = h["fields"][a]["events"][2]
        assert removed["record_id"] == "r1"
        assert removed["value_before"] == "x"
        assert h["fields"][a]["final"]["present"] is False
        assert h["fields"][a]["final"]["value"] is None
        assert h["fields"][other]["final"]["present"] is True

    def test_non_dispatch_events_are_ignored(self):
        h = build_field_history([
            {"event": "session_started", "ts": "T"},
            {"event": "assistant_message", "turn_id": 1, "content": []},
            {"event": "resumed", "max_tool_calls": 5},
            _dispatch("tool_call_applied",
                      diffs={FIELD_PATH: {"before": None, "after": "v"}}),
            {"event": "terminate", "status": "complete"},
        ])
        assert list(h["fields"]) == [FIELD_PATH]


class TestAggregate:
    def test_the_aggregate_answers_whether_the_stages_earn_their_cost(self):
        other = "study.doi"
        h = build_field_history([
            _dispatch("tool_call_applied", turn_id=1,
                      diffs={FIELD_PATH: {"before": None, "after": "v"},
                             other: {"before": None, "after": "d"}},
                      verdicts={FIELD_PATH: _verdict("challenge", cost=0.002),
                                other: _verdict("ok", cost=0.003)}),
            # The extractor stands by the challenged value.
            _dispatch("tool_call_applied", turn_id=2,
                      diffs={FIELD_PATH: {"before": "v", "after": "v"}}),
            # The reviewer changes the other one.
            _dispatch("review_tool_call", turn_id=3,
                      diffs={other: {"before": "d", "after": "e"}}),
        ])
        assert h["aggregate"] == {
            "fields_written": 2,
            "fields_checked": 2,
            "fields_reviewer_touched": 1,
            "fields_with_unresolved_challenge": 1,
            "challenges_raised": 1,
            "challenges_revised": 0,
            "challenges_overruled": 1,
            "check_errors": 0,
            "checker_cost_usd": 0.005,
        }

    def test_an_empty_log_is_an_empty_history_not_a_failure(self):
        h = build_field_history([])
        assert h["fields"] == {}
        assert h["aggregate"]["fields_written"] == 0
        assert h["aggregate"]["checker_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# End to end: a real extractor and a real checker fan-out seam
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir, *, max_checks_per_field=2,
          final_review=False, review_model=None):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model=CHECKER),
        review_model=review_model,
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        max_tool_calls=50,
        extractor_max_tokens=4096,
        review_max_tokens=4096,
    )


def _write_turn(tool_id, value):
    return SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", id=tool_id, name="update_study",
        input={"study": {FIELD_VAR: {"value": value,
                                     "evidence": f"<q>{QUOTE}</q>"}}})])


def _view_turn(tool_id):
    return SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", id=tool_id, name="view_summary", input={})])


def _drive(orch, turns):
    """Play `turns` in order, latching mark_complete on the last one so the
    loop ends cleanly (the last turn must be read-only: a write clears the
    flag as it lands)."""
    # Open the extractor's initial-check ordering gate directly rather than
    # scripting the `record_initial_check` turn that opens it. That call writes
    # its own per-field diffs, so a real one would add `initial_check.*` fields
    # to every history below and change the aggregate counts these tests read:
    # the gate is not what is on trial, the derived history is.
    orch.extraction_record.initial_check_recorded = True
    orch._adapter_for_role = lambda role: object()
    orch._played = []

    def _call(adapter, tool_defs):
        idx = min(len(orch._played), len(turns) - 1)
        orch._played.append(idx)
        if idx == len(turns) - 1:
            orch.extraction_record.mark_complete()
        return turns[idx]

    orch._call_extractor = _call


def _stub_fanout(monkeypatch, verdicts_per_call):
    """Stub the checker fan-out with a scripted verdict per fan-out call.

    Every field checked in one fan-out gets that fan-out's verdict, which is
    all these tests need: the field under test is the only checkable one they
    write. The last entry repeats once the script is exhausted.
    """
    fanouts_run = [0]

    def _run(*, calls, config, on_complete=None, api_logger=None, **kw):
        idx = min(fanouts_run[0], len(verdicts_per_call) - 1)
        fanouts_run[0] += 1
        verdict, rationale = verdicts_per_call[idx]
        return {c["field_path"]: {
            "verdict": verdict, "rationale": rationale, "notes": None,
            "error_origin": False, "input_tokens": 1, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.001,
        } for c in calls}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _run)
    return fanouts_run


def test_a_real_challenge_then_revision_is_recorded_in_order(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, max_checks_per_field=2)
    orch.prepare_new_session()
    _drive(orch, [_write_turn("t1", "A synthetic study"),
                  _write_turn("t2", QUOTE),
                  _view_turn("t3")])
    _stub_fanout(monkeypatch, [("challenge", "the quote says otherwise"),
                               ("ok", "fine")])

    assert orch.run() == "complete"

    history = json.loads(orch.session.field_history_path.read_text())
    assert _kinds(history) == ["proposed", "applied", "challenged",
                              "proposed", "revised_after_challenge",
                              "checked"]
    final = history["fields"][FIELD_PATH]["final"]
    assert final["value"] == QUOTE
    assert final["checks"] == 2
    assert final["last_verdict"] == "ok"
    assert final["unresolved_challenge"] is False
    assert history["aggregate"]["challenges_raised"] == 1
    assert history["aggregate"]["challenges_revised"] == 1


def test_a_real_overrule_is_inferred_from_the_repeated_write(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # One check per field, so the second write gets no verdict of its own and
    # the only signal available is that the same value was written again.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, max_checks_per_field=1)
    orch.prepare_new_session()
    _drive(orch, [_write_turn("t1", "A synthetic study"),
                  _write_turn("t2", "A synthetic study"),
                  _view_turn("t3")])
    _stub_fanout(monkeypatch, [("challenge", "not shown by that quote")])

    assert orch.run() == "complete"

    history = json.loads(orch.session.field_history_path.read_text())
    assert _kinds(history) == ["proposed", "applied", "challenged",
                              "proposed", "overruled"]
    assert history["aggregate"]["challenges_overruled"] == 1
    # The checker never signed the value off, and the final state says so
    # in the same terms run.checker_diagnostics does.
    assert history["fields"][FIELD_PATH]["final"][
        "unresolved_challenge"] is True
    assert orch.session.meta["checker_diagnostics"][
        "unresolved_challenges"] == [FIELD_PATH]


def test_field_history_regenerates_identically_from_the_event_log_alone(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    """The file is derived, so it is never a second source of truth.

    Nothing but `tool_calls.jsonl` is read here: not the extraction output,
    not run.json, not the session object. The rebuilt document must match the
    written one byte for byte, or the file is carrying something the log
    cannot account for.
    """
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, max_checks_per_field=2)
    orch.prepare_new_session()
    _drive(orch, [_write_turn("t1", "A synthetic study"),
                  _write_turn("t2", QUOTE),
                  _view_turn("t3")])
    _stub_fanout(monkeypatch, [("challenge", "no"), ("ok", "fine")])
    assert orch.run() == "complete"

    log_path = orch.session.session_dir / "diagnostics" / "tool_calls.jsonl"
    events = [json.loads(line)
              for line in log_path.read_text().splitlines() if line.strip()]
    rebuilt = build_field_history(events)

    written = (orch.session.session_dir / "diagnostics" /
               "field_history.json").read_text()
    assert json.dumps(rebuilt, indent=2, ensure_ascii=False) == written


def test_a_paused_session_carries_a_field_history_too(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, max_checks_per_field=1)
    orch.max_tool_calls = 1
    orch.prepare_new_session()
    _drive(orch, [_write_turn("t1", "A synthetic study")])
    _stub_fanout(monkeypatch, [("ok", "fine")])

    assert orch.run() == "in_progress"
    history = json.loads(orch.session.field_history_path.read_text())
    assert _kinds(history) == ["proposed", "applied", "checked"]


# ---------------------------------------------------------------------------
# End to end: a real review loop
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create_message(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


def _review_resp(*blocks):
    return NormalisedResponse(
        content=list(blocks), usage=NormalisedUsage(),
        resolved_model=REVIEWER, provider="anthropic", base_url=None,
        raw_request={"model": REVIEWER}, raw_response={},
        wire_request={"model": REVIEWER}, decoding_params={"max_tokens": 1024},
    )


def test_a_real_reviewer_edit_is_recorded_as_a_reviewer_revision(
        config_dir, bundle_minimal_dir, tmp_path):
    """The reviewer's own tool calls are `review_tool_call` events, so the
    stage a write came from is read off the event name rather than inferred."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, max_checks_per_field=0,
                 final_review=True, review_model=REVIEWER)
    orch.prepare_new_session()

    def _extractor():
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"

    orch._extractor_loop = _extractor
    adapter = _ScriptedAdapter([
        _review_resp(SimpleNamespace(
            type="tool_use", id="r1", name="update_study",
            input={"study": {FIELD_VAR: {"value": QUOTE,
                                         "evidence": f"<q>{QUOTE}</q>"}}})),
        _review_resp(SimpleNamespace(
            type="tool_use", id="r2", name="mark_complete",
            input=_mark_complete_input("checked"))),
    ])
    orch._adapter_for_role = lambda role: adapter

    assert orch.run() == "complete"

    history = json.loads(orch.session.field_history_path.read_text())
    # The reviewer's `mark_complete` is dispatched like any other call, so it
    # too appears in the event log as a `review_tool_call`. It writes no field,
    # so it contributes no history event and the field's trail is the edit
    # alone.
    assert _kinds(history) == ["proposed", "revised_by_reviewer"]
    event = history["fields"][FIELD_PATH]["events"][1]
    assert event["stage"] == "review"
    assert event["after"] == QUOTE
    assert history["aggregate"]["fields_reviewer_touched"] == 1
