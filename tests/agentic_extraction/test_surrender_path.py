"""The deliberate-surrender path.

`abandon_extraction` lets the extractor end a run it cannot complete honestly.
The run finalises as `failed_validation` (terminal, not resumable, exit 0), the
extraction output so far is written, and the stated reason lands in run.json
and the run log. These tests cover the dispatcher tool, the extractor-loop
signal, and a full offline run() end to end with a fake adapter.
"""

from types import SimpleNamespace

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import ResumeRefused
from meltiro.extraction_record import ExtractionRecord
from meltiro.orchestrator import Orchestrator
from meltiro.run_log import load_log
from meltiro.session import Session
from meltiro.statuses import VALIDATED_STATUSES
from meltiro.tools import ToolDispatcher


def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


def _resp(*blocks):
    return SimpleNamespace(content=list(blocks))


# ---------------------------------------------------------------------------
# Dispatcher tool
# ---------------------------------------------------------------------------

def test_abandon_requires_reason(synthetic_template, paper_text, image_labels):
    record = ExtractionRecord()
    disp = ToolDispatcher(record, synthetic_template, paper_text, image_labels)
    result = disp.dispatch("abandon_extraction", {"reason": "  "})
    assert result["status"] == "validation_failed"
    assert any(e["code"] == "missing_field" for e in result["errors"])
    assert record.abandoned_flag is False


def test_abandon_latches_flag_and_reason(
        synthetic_template, paper_text, image_labels):
    record = ExtractionRecord()
    disp = ToolDispatcher(record, synthetic_template, paper_text, image_labels)
    result = disp.dispatch(
        "abandon_extraction", {"reason": "OCR garble, no readable text"})
    assert result["status"] == "ok", result
    assert record.abandoned_flag is True
    assert record.abandon_reason == "OCR garble, no readable text"


# ---------------------------------------------------------------------------
# Extractor loop signal
# ---------------------------------------------------------------------------

def _bare_loop_orch(tmp_path, template, paper_text, image_labels):
    orch = Orchestrator.__new__(Orchestrator)
    orch.template = template
    record = ExtractionRecord()
    orch.extraction_record = record
    orch.dispatcher = ToolDispatcher(record, template, paper_text, image_labels)
    orch.session = Session.create(
        "demo-001",
        config_fp="config_fp:abcabcabcabc",
        checker_fp="checker_fp:def", review_fp="review_fp:xyz",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        extractor_model="opus", checker_model="sonnet", review_model="opus",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        runs_dir=tmp_path,
    )
    orch.messages = [{"role": "user",
                      "content": [{"type": "text", "text": "start"}]}]
    orch.max_tool_calls = 100
    orch._mark_complete_has_fired = False
    orch.max_consecutive_text_only_turns = 3
    orch.max_consecutive_identical_failures = 5
    orch._turn_counter = 0
    # The checker is off: this exercises the loop's surrender signal, and a
    # checker call would be a network call.
    orch.max_checks_per_field = 0
    orch._check_counts = {}
    orch._adapter_for_role = lambda role: object()
    return orch


def test_extractor_loop_returns_abandoned(
        tmp_path, synthetic_template, paper_text, image_labels):
    orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                           image_labels)

    def _fake(adapter, tool_defs):
        return _resp(_tool_use("ab1", "abandon_extraction",
                               {"reason": "no extractable relationships"}))

    orch._call_extractor = _fake
    status = orch._extractor_loop()

    assert status == "extractor_abandoned"
    events = orch.session.read_events()
    abandoned = [e for e in events if e["event"] == "extractor_abandoned"]
    assert len(abandoned) == 1
    assert abandoned[0]["reason"] == "no extractable relationships"


# ---------------------------------------------------------------------------
# Full offline run() end to end
# ---------------------------------------------------------------------------

def _extractor_only_orch(config_dir, bundle_dir, out_dir):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=1024, checker_model="claude-sonnet-4-6",
                                     api_key="x"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        extractor_max_tokens=4096,
        api_key="x",
    )


def test_run_surrenders_to_failed_validation(
        config_dir, bundle_minimal_dir, tmp_path):
    out_dir = tmp_path / "runs"
    orch = _extractor_only_orch(config_dir, bundle_minimal_dir, out_dir)
    orch.prepare_new_session()
    orch._adapter_for_role = lambda role: object()

    reason = "the extracted text is OCR garble; no relationships are legible"

    def _fake(adapter, tool_defs):
        return _resp(_tool_use("ab1", "abandon_extraction",
                               {"reason": reason}))

    orch._call_extractor = _fake
    status = orch.run()

    # Terminal status and the mechanism/reason recorded in meta.
    assert status == "failed_validation"
    assert status not in VALIDATED_STATUSES
    meta = orch.session.meta
    assert meta["status"] == "failed_validation"
    assert meta["failure_reason"] == "surrendered"
    assert meta["failed_validation_reason"] == reason

    # The extraction output so far is written.
    assert orch.session.extraction_record_path.exists()

    # The run log carries the terminal status, a False validation flag, and
    # the reason in the error detail.
    log = load_log(out_dir)
    assert len(log) == 1
    entry = log[0]
    assert entry["status"] == "failed_validation"
    assert entry["validation_passed"] is False
    assert any(reason in e for e in entry["validation_errors"])

    # The audit trail records the surrender and the terminate.
    events = [e["event"] for e in orch.session.read_events()]
    assert "extractor_abandoned" in events
    assert "terminate" in events


def test_failed_validation_session_not_resumable(
        config_dir, bundle_minimal_dir, tmp_path):
    # A surrendered session is terminal: its status is not in_progress, so
    # find_in_progress ignores it AND a direct resume is refused. Both halves
    # are asserted, because they are different gates. `find_in_progress` is a
    # discovery filter that `--auto-resume` consults; `Session.resume` is the
    # gate an operator hits when they pass `--resume SESSION_DIR` by hand, and
    # that path never goes near the filter. A surrender is a model's stated
    # judgement that no honest extraction is possible from these inputs, so
    # continuing the same conversation would be resuming past the finding.
    out_dir = tmp_path / "runs"
    orch = _extractor_only_orch(config_dir, bundle_minimal_dir, out_dir)
    orch.prepare_new_session()
    session_dir = orch.session.session_dir
    orch._adapter_for_role = lambda role: object()
    orch._call_extractor = lambda adapter, tool_defs: _resp(
        _tool_use("ab1", "abandon_extraction", {"reason": "unreadable"}))
    assert orch.run() == "failed_validation"

    # Not surfaced as an in-progress session to auto-resume.
    assert Session.find_in_progress("demo-001", runs_dir=out_dir) is None

    # And refused outright when named directly, naming the status found.
    with pytest.raises(ResumeRefused) as excinfo:
        Session.resume(session_dir)
    assert "failed_validation" in str(excinfo.value)
