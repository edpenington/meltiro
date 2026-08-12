"""The per-field check budget survives a pause and resume.

`max_checks_per_field` is the TOTAL number of checker calls one field may
receive across the whole session, so a resumed segment must not hand a field a
fresh allowance. Nothing stores the counts: they are counted back off the
`_checker_verdicts` recorded on each tool call's result, which is the same
record every other consumer reads. A second copy in meta would be a source of
truth able to drift from the verdicts themselves.

The budget is config identity, not an operational budget, so a resume under a
changed value is refused by the fingerprint drift gate. Only the operational
caps (the tool-call cap, say) may move between segments.

Everything here is offline: a real Session, dispatcher, and extraction record
back the orchestrator; the provider adapter, `_call_extractor`, and the checker
fan-out are stubbed.
"""

from types import SimpleNamespace

import pytest

from meltiro import orchestrator as orch_mod
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import ResumeRefused
from meltiro.orchestrator import Orchestrator


# A study field plus a verbatim quote from the synthetic bundle, so a real
# dispatch validates and applies and the trigger fires on the result.
FIELD_VAR = "title"
FIELD_PATH = f"study.{FIELD_VAR}"
QUOTE = "A synthetic study of baseline CRT-HD scores"


def _orch(config_dir, bundle_dir, out_dir, *, cap, max_checks_per_field=1):
    """An extractor + checker Orchestrator (reviewer off).

    The post-mark_complete cleanup bonus is 0 so a spent tool-call cap pauses
    the run immediately: that pause is what these tests resume from.
    """
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=1024, checker_model="claude-sonnet-4-6",
                                     api_key="x"),
        review_model=None,
        max_checks_per_field=max_checks_per_field, final_review=False,
        max_tool_calls=cap,
        extractor_max_tokens=4096,
        api_key="x",
    )


def _write_field_response(tool_id, value):
    """One `update_study` call writing the field with valid evidence."""
    return SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", id=tool_id, name="update_study",
        input={"study": {FIELD_VAR: {"value": value,
                                     "evidence": f"<q>{QUOTE}</q>"}}})])


def _view_summary_response(tool_id):
    """A read-only `view_summary` call. It writes nothing, so it does not clear
    the mark_complete flag a turn latches just before returning it."""
    return SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", id=tool_id, name="view_summary", input={})])


def _drive_extractor(orch, tool_id, value, *, then_complete=False):
    """Stub the extractor: one write, then (optionally) a read-only turn that
    latches mark_complete so the loop ends instead of running to the cap.

    The two turns cannot be one: `update_study` clears the completion flag as
    it writes, which is exactly the invariant that makes a write after
    mark_complete re-open the extraction."""
    # The extractor's initial-check ordering gate is opened directly rather
    # than by scripting the `record_initial_check` turn that opens it: the
    # tool-call cap here is 1, so that turn would spend the whole budget these
    # tests exist to account for. Session persists the flag in run.json beside
    # the record-id counters, and this runs on the resumed segment too, so the
    # second segment's write is gated exactly as the first's was.
    orch.extraction_record.initial_check_recorded = True
    orch._adapter_for_role = lambda role: object()
    turns = [_write_field_response(tool_id, value)]
    if then_complete:
        turns.append(_view_summary_response(tool_id + "-done"))

    def _call(adapter, tool_defs):
        idx = min(len(orch._driven), len(turns) - 1)
        orch._driven.append(idx)
        if then_complete and idx == len(turns) - 1:
            orch.extraction_record.mark_complete()
        return turns[idx]

    orch._driven = []
    orch._call_extractor = _call


def _stub_fanout(monkeypatch, seen):
    """Stub the fan-out, recording the field paths of every call it is given."""
    def _fake(*, calls, config, on_complete=None, api_logger=None, **kw):
        seen.extend(c["field_path"] for c in calls)
        return {c["field_path"]: {
            "verdict": "challenge", "rationale": "not shown by that quote",
            "notes": None, "error_origin": False, "input_tokens": 1,
            "output_tokens": 1, "cache_creation_tokens": 0,
            "cache_read_tokens": 0, "cost_usd": 0.0,
        } for c in calls}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _fake)


def _pause_after_one_check(config_dir, bundle_dir, out_dir, monkeypatch, seen,
                           *, max_checks_per_field=1):
    """Drive a fresh session through one checked write into a tool-cap pause.
    Returns the session directory."""
    orch = _orch(config_dir, bundle_dir, out_dir, cap=1,
                 max_checks_per_field=max_checks_per_field)
    orch.prepare_new_session()
    _drive_extractor(orch, "s1", "A synthetic study")
    _stub_fanout(monkeypatch, seen)

    assert orch.run() == "in_progress"
    assert orch.session.meta["pause_reason"] == "tool_cap_hit"
    assert orch.session.meta["checker_calls_run"] == 1
    assert seen == [FIELD_PATH]
    return orch.session.session_dir


def test_a_spent_budget_is_not_refreshed_by_a_resume(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    out = tmp_path / "runs"
    seen = []
    session_dir = _pause_after_one_check(
        config_dir, bundle_minimal_dir, out, monkeypatch, seen)

    # Resume with a raised tool-call cap (an operational budget, out of every
    # fingerprint, so this resume is accepted) and rewrite the same field.
    orch2 = _orch(config_dir, bundle_minimal_dir, out, cap=50)
    orch2.resume_session(session_dir)
    _drive_extractor(orch2, "s2", "A synthetic study, revised",
                     then_complete=True)
    assert orch2.run() == "complete"

    # The field had its one allowed check in the first segment, so the rewrite
    # in the second is not checked again.
    assert seen == [FIELD_PATH]
    assert orch2.session.meta["checker_calls_run"] == 1


def test_a_resumed_segment_gets_exactly_the_remaining_allowance(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # With a budget of 2 and one check spent, the resumed segment gets one
    # more: neither a fresh allowance nor none at all.
    out = tmp_path / "runs"
    seen = []
    session_dir = _pause_after_one_check(
        config_dir, bundle_minimal_dir, out, monkeypatch, seen,
        max_checks_per_field=2)

    orch2 = _orch(config_dir, bundle_minimal_dir, out, cap=50,
                  max_checks_per_field=2)
    orch2.resume_session(session_dir)
    _drive_extractor(orch2, "s2", "A synthetic study, revised",
                     then_complete=True)
    assert orch2.run() == "complete"

    assert seen == [FIELD_PATH, FIELD_PATH]
    assert orch2.session.meta["checker_calls_run"] == 2


def test_the_counts_come_back_off_the_event_log_at_resume_time(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The reconstruction is what makes the log the single source of truth, so
    # pin it at the seam: a fresh orchestrator carries no counts, and
    # resume_session fills them in from the recorded verdicts.
    out = tmp_path / "runs"
    seen = []
    session_dir = _pause_after_one_check(
        config_dir, bundle_minimal_dir, out, monkeypatch, seen,
        max_checks_per_field=2)

    orch2 = _orch(config_dir, bundle_minimal_dir, out, cap=50,
                  max_checks_per_field=2)
    assert orch2._check_counts == {}
    orch2.resume_session(session_dir)
    assert orch2._check_counts == {FIELD_PATH: 1}


def test_meta_stores_no_second_copy_of_the_counts(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # Only the aggregate tally is in meta. A per-field copy there could drift
    # from the verdicts a diagnostics reader would believe, so there isn't one.
    out = tmp_path / "runs"
    seen = []
    session_dir = _pause_after_one_check(
        config_dir, bundle_minimal_dir, out, monkeypatch, seen)

    import json
    meta = json.loads(
        (session_dir / "diagnostics" / "run.json").read_text())
    assert meta["checker_calls_run"] == 1
    assert not any(isinstance(v, dict) and FIELD_PATH in v
                   for v in meta.values())


def test_resume_under_a_changed_budget_is_refused(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The budget rides in the fingerprinted structure block, so it is config
    # identity: raising it mid-session would change the instrument, and the
    # drift gate refuses rather than letting it happen quietly.
    out = tmp_path / "runs"
    seen = []
    session_dir = _pause_after_one_check(
        config_dir, bundle_minimal_dir, out, monkeypatch, seen)

    resumed = _orch(config_dir, bundle_minimal_dir, out, cap=50,
                    max_checks_per_field=2)
    with pytest.raises(ResumeRefused):
        resumed.resume_session(session_dir)
