"""The tool-call cap is an operational budget, not config identity.

`max_tool_calls` reaches NO fingerprint. Folding it in refuses the documented
"hit the tool-call cap, resume with a raised cap" recovery at the config-drift
gate, and it reaches config_fp by two routes at once: the `{max_tool_calls}`
number rendered into the extractor prompt, and the decoding-params hash. A
resumed conversation replays the ORIGINAL rendered prompt regardless, so
raising the cap can never change prompt provenance; it only moves the
enforcement bound. The cap is recorded in run.json per segment instead.

These tests pin that contract, all offline (a real Session + dispatcher back
the orchestrator; the provider adapter and `_call_extractor` are stubbed):

  (a) a resume with a RAISED cap passes the drift gate and runs to completion;
  (b) the drift gate is otherwise intact: a changed temperature or template
      still refuses resume;
  (c) changing the cap does not move config_fp;
  (d) meta records the cap and a `resumed` event records a raised cap.
"""

import shutil
from types import SimpleNamespace

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import ResumeRefused
from meltiro.orchestrator import Orchestrator

EXTRACTOR = "claude-opus-4-8"


def _orch(config_dir, bundle_dir, out_dir, *, cap,
          extractor=EXTRACTOR, sampling={"temperature": 0.0}):
    """An extractor-only Orchestrator (checker and reviewer off)."""
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor,
        checker_config=CheckerConfig(max_tokens=1024, checker_model="claude-sonnet-4-6",
                                     api_key="x"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=cap,
        sampling=sampling,
        extractor_max_tokens=4096,
        api_key="x",
    )


def _view_summary_resp(tool_id):
    """A one-tool-call (view_summary) response. view_summary is read-only and
    does not clear the mark_complete flag, so a test can set the flag just
    before returning one to end the loop cleanly without a full extraction."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id,
                                 name="view_summary", input={})],
    )


def _run_to_cap_pause(orch, tool_id):
    """Drive a fresh session to a tool-call-cap pause (status in_progress)."""
    orch.prepare_new_session()
    orch._adapter_for_role = lambda role: object()
    orch._call_extractor = lambda adapter, tool_defs: _view_summary_resp(tool_id)
    assert orch.run() == "in_progress"
    assert orch.session.meta["pause_reason"] == "tool_cap_hit"
    return orch.session.session_dir


# ---------------------------------------------------------------------------
# (a) A raised-cap resume is accepted and completes.
# ---------------------------------------------------------------------------

def test_resume_with_raised_cap_is_accepted_and_completes(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    # Segment 1: cap of 1, one tool-calling turn, then a cap-hit pause.
    session_dir = _run_to_cap_pause(
        _orch(config_dir, bundle_minimal_dir, out, cap=1), "s1")

    # Segment 2 is built with the RAISED cap from the start. The cap is out of
    # the fingerprint, so config_fp is unmoved, the resume is accepted, and the
    # same conversation continues to completion.
    orch2 = _orch(config_dir, bundle_minimal_dir, out, cap=50)
    orch2.resume_session(session_dir)  # must not raise
    orch2._adapter_for_role = lambda role: object()

    def _fake2(adapter, tool_defs):
        orch2.extraction_record.mark_complete()
        return _view_summary_resp("s2")

    orch2._call_extractor = _fake2
    assert orch2.run() == "complete"
    # The whole-run tool count carried across the resume: one call per segment.
    assert orch2.session.meta["tool_call_count"] == 2


# ---------------------------------------------------------------------------
# (b) The drift gate is otherwise intact.
# ---------------------------------------------------------------------------

def test_changed_temperature_still_refuses_resume(
        config_dir, bundle_minimal_dir, tmp_path):
    # A genuine decoding change must still refuse resume. z-ai/glm-5v-turbo accepts
    # temperature (claude-opus-4-8 rejects it, so it would be a no-op there),
    # so changing it moves the resolved decoding dict, the decoding hash, and
    # config_fp. An in_progress session (prepared, not run) is resumable; the
    # drift gate fires before any spend.
    orch1 = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                  extractor="z-ai/glm-5v-turbo", cap=5, sampling={"temperature": 0.0})
    orch1.prepare_new_session()
    session_dir = orch1.session.session_dir

    orch2 = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                  extractor="z-ai/glm-5v-turbo", cap=5, sampling={"temperature": 0.9})
    with pytest.raises(ResumeRefused):
        orch2.resume_session(session_dir)


def test_changed_template_still_refuses_resume(
        config_dir, bundle_minimal_dir, tmp_path):
    # A template edit moves template_hash, hence config_fp, so resume refuses.
    cfg = tmp_path / "config"
    shutil.copytree(config_dir, cfg)

    orch1 = _orch(cfg, bundle_minimal_dir, tmp_path / "runs", cap=5)
    orch1.prepare_new_session()
    session_dir = orch1.session.session_dir

    tmpl = cfg / "extraction_template.yaml"
    tmpl.write_text(
        tmpl.read_text(encoding="utf-8") + "\n# a trailing comment\n",
        encoding="utf-8")

    orch2 = _orch(cfg, bundle_minimal_dir, tmp_path / "runs", cap=5)
    with pytest.raises(ResumeRefused):
        orch2.resume_session(session_dir)


# ---------------------------------------------------------------------------
# (c) The cap does not move config_fp.
# ---------------------------------------------------------------------------

def test_changing_cap_does_not_move_config_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    a = _orch(config_dir, bundle_minimal_dir, tmp_path / "a", cap=100)
    a.prepare_new_session()
    # A different cap: it is the operational budget, not config identity.
    b = _orch(config_dir, bundle_minimal_dir, tmp_path / "b", cap=999)
    b.prepare_new_session()

    assert a.session.meta["config_fp"] == b.session.meta["config_fp"]
    # The cap is absent from the prompt too, so the paper-independent prompt
    # hash is identical: it carries no cap number to differ by.
    assert a.session.meta["prompt_hash"] == b.session.meta["prompt_hash"]


# ---------------------------------------------------------------------------
# (d) meta records the cap; a raised-cap resume records the change.
# ---------------------------------------------------------------------------

def test_meta_records_cap_and_resume_event_records_the_change(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    orch1 = _orch(config_dir, bundle_minimal_dir, out, cap=7)
    orch1.prepare_new_session()
    # Session.create froze the starting bounds in meta.caps (the provenance
    # home of the budget now that it is out of config identity).
    assert orch1.session.meta["caps"]["max_tool_calls"] == 7
    session_dir = orch1.session.session_dir

    orch2 = _orch(config_dir, bundle_minimal_dir, out, cap=42)
    orch2.resume_session(session_dir)
    # The recorded bounds now reflect the segment in force.
    assert orch2.session.meta["caps"]["max_tool_calls"] == 42
    # A `resumed` event records the per-segment bounds, and a raise is visible
    # as new != previous.
    resumed = [e for e in orch2.session.read_events()
               if e.get("event") == "resumed"]
    assert len(resumed) == 1
    assert resumed[0]["max_tool_calls"] == 42
    assert resumed[0]["previous"] == 7


def test_refused_resume_does_not_mutate_meta(
        config_dir, bundle_minimal_dir, tmp_path):
    # The drift gate must refuse BEFORE any meta mutation: a refused resume
    # leaves the paused session byte-identical (no caps update, no `resumed`
    # event), so the audit trail never shows a segment that was not run.
    out = tmp_path / "runs"
    # z-ai/glm-5v-turbo accepts temperature (claude-opus-4-8 drops it via its quirk,
    # which would make the change a fingerprint no-op), so the 0.0 -> 0.9
    # change below genuinely moves config_fp and triggers the refusal.
    orch1 = _orch(config_dir, bundle_minimal_dir, out,
                  extractor="z-ai/glm-5v-turbo", cap=7, sampling={"temperature": 0.0})
    orch1.prepare_new_session()
    session_dir = orch1.session.session_dir
    meta_path = session_dir / "diagnostics" / "run.json"
    events_path = session_dir / "diagnostics" / "tool_calls.jsonl"
    meta_before = meta_path.read_bytes()
    events_before = events_path.read_bytes()

    orch2 = _orch(config_dir, bundle_minimal_dir, out,
                  extractor="z-ai/glm-5v-turbo", cap=42, sampling={"temperature": 0.9})
    with pytest.raises(ResumeRefused):
        orch2.resume_session(session_dir)
    assert meta_path.read_bytes() == meta_before
    assert events_path.read_bytes() == events_before
