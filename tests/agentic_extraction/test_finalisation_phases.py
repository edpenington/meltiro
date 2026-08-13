"""Stopping a run is two phases, and a fault in the second cannot rewrite the
first.

`run()` persists the outcome — the terminal status in run.json, the run-log
entry beside it — inside its try, and renders the derived documents (field
history, transcript) outside it. Everything here is about that seam:

  - a transcript that will not write leaves a complete run complete, with ONE
    run-log entry, and says so on stderr;
  - the same fault over a PAUSED run leaves the session in_progress and
    resumable, which is the outcome the operator paid for;
  - a rendering fault never escapes `run()`.

The run log is a cross-run ledger a consumer sums into a bill, so a second
entry for one session is not a duplicate line but double-counted money. That
is what the entry counts below are checking.

Fully offline: a real Session backs a real Orchestrator, the extractor loop is
stubbed out, and no provider is ever reached.
"""

import json
import sys

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import AgenticExtractionError
from meltiro.orchestrator import Orchestrator
from meltiro.run_log import load_log

pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"


def _orch(config_dir, bundle_dir, out_dir, *, cap=50):
    """An extractor-only Orchestrator (checker and reviewer off)."""
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(
            max_tokens=1024, checker_model="claude-sonnet-4-6"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=cap,
        extractor_max_tokens=4096,
    )


def _break_transcript(orch, monkeypatch, *, message="disk is full"):
    """Make transcript rendering fail the way a full disk would."""
    def _boom():
        raise OSError(message)
    monkeypatch.setattr(orch.session, "write_transcript", _boom)


def _entries_for(out_dir, session_dir):
    """The run-log entries naming this session."""
    return [e for e in load_log(out_dir)
            if e.get("session_dir") == str(session_dir)]


# ---------------------------------------------------------------------------
# A complete run whose transcript will not render
# ---------------------------------------------------------------------------

def test_transcript_failure_leaves_a_complete_run_complete(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")
    _break_transcript(orch, monkeypatch)

    status = orch.run()

    assert status == "complete"
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "complete"
    # Exactly one ledger entry: a second would double-count this run's spend
    # for anyone summing the log.
    entries = _entries_for(out, orch.session.session_dir)
    assert len(entries) == 1
    assert entries[0]["status"] == "complete"
    # And the operator is told, by name, which document is missing and from
    # which session.
    err = capsys.readouterr().err
    assert "transcript.md" in err
    assert str(orch.session.session_dir) in err


def test_transcript_failure_is_recorded_in_the_session_warnings(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")
    _break_transcript(orch, monkeypatch)

    orch.run()

    warnings = json.loads(orch.session.meta_path.read_text()).get("warnings")
    assert warnings, "a missing artefact must leave a durable trace"
    assert any("transcript.md" in w for w in warnings)


# ---------------------------------------------------------------------------
# A PAUSED run whose transcript will not render stays resumable
# ---------------------------------------------------------------------------

def test_transcript_failure_on_a_pause_keeps_the_session_resumable(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop", lambda: "tool_cap_hit")
    _break_transcript(orch, monkeypatch)

    status = orch.run()

    assert status == "in_progress"
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "in_progress"
    assert meta["pause_reason"] == "tool_cap_hit"
    # A pause writes no ledger entry at all; the run is not finished.
    assert _entries_for(out, orch.session.session_dir) == []
    assert "transcript.md" in capsys.readouterr().err

    # The proof that resumability survived: the resume gate admits it.
    orch2 = _orch(config_dir, bundle_minimal_dir, out)
    orch2.resume_session(orch.session.session_dir)
    assert orch2.session.meta["status"] == "in_progress"


# ---------------------------------------------------------------------------
# A persistent rendering fault does not escape run()
# ---------------------------------------------------------------------------

def test_a_persistent_render_fault_does_not_raise_out_of_run(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    # Both documents fail, and so does the attempt to record the failure in
    # meta: the whole of phase (b) is unavailable, as a read-only filesystem
    # would make it.
    def _boom(*args, **kwargs):
        raise OSError("read-only file system")
    monkeypatch.setattr(orch.session, "write_transcript", _boom)
    monkeypatch.setattr(orch.session, "write_field_history", _boom)
    monkeypatch.setattr(orch.session, "add_warning", _boom)

    status = orch.run()

    assert status == "complete"
    assert len(_entries_for(out, orch.session.session_dir)) == 1
    err = capsys.readouterr().err
    # Both documents are named, each on its own line.
    assert "field_history.json" in err
    assert "transcript.md" in err


# ---------------------------------------------------------------------------
# Finalisation is idempotent
# ---------------------------------------------------------------------------

def test_finalise_twice_writes_one_run_log_entry(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    assert orch._finalise("complete") == "complete"
    # A second call is what a fault after the first one produces, via run()'s
    # catch-all. It answers with the status already on disk and writes nothing.
    assert orch._finalise("error") == "complete"

    entries = _entries_for(out, orch.session.session_dir)
    assert len(entries) == 1
    assert entries[0]["status"] == "complete"
    assert json.loads(orch.session.meta_path.read_text())["status"] == \
        "complete"


def test_a_fault_after_finalisation_keeps_the_persisted_status(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    """The exact double-finalisation shape: something fails inside the try
    AFTER the status is persisted. The persisted status stands."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    real_finalise = orch._finalise
    faults = [OSError("the fault that used to finalise a second time")]

    def _finalise_then_fail(status, **kwargs):
        answer = real_finalise(status, **kwargs)
        if faults:
            raise faults.pop()
        return answer

    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")
    monkeypatch.setattr(orch, "_finalise", _finalise_then_fail)

    status = orch.run()

    assert status == "complete"
    assert len(_entries_for(out, orch.session.session_dir)) == 1
    assert json.loads(orch.session.meta_path.read_text())["status"] == \
        "complete"
    capsys.readouterr()


# ---------------------------------------------------------------------------
# The composed error message reaches the run record (group 3)
# ---------------------------------------------------------------------------

def test_a_failed_run_states_what_failed_on_stderr_and_in_the_record(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    message = "ANTHROPIC_API_KEY is not set for the extractor stage"

    def _raise():
        raise AgenticExtractionError(message)
    monkeypatch.setattr(orch, "_extractor_loop", _raise)

    status = orch.run()

    assert status == "error"
    assert message in capsys.readouterr().err
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["error_message"] == message
    entry, = _entries_for(out, orch.session.session_dir)
    assert entry["validation_errors"] != ["error"]
    assert any(message in e for e in entry["validation_errors"])


def test_an_unexpected_crash_states_its_type_in_the_record(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    def _raise():
        raise ZeroDivisionError("division by zero")
    monkeypatch.setattr(orch, "_extractor_loop", _raise)

    status = orch.run()

    assert status == "error"
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["error_message"] == "ZeroDivisionError: division by zero"
    assert "ZeroDivisionError" in capsys.readouterr().err
    entry, = _entries_for(out, orch.session.session_dir)
    assert any("ZeroDivisionError" in e for e in entry["validation_errors"])


# ---------------------------------------------------------------------------
# A run log that cannot be appended to
# ---------------------------------------------------------------------------
#
# The ledger lives under --out, which the session does not: a permission on
# the run root alone, a lock, a full disk are all faults that reach the append
# and nothing else. The run itself is over by then, and its status stands.

def _break_run_log(monkeypatch, *, message="run root is read-only"):
    """Make the run-log append fail the way an unwritable run root would."""
    def _boom(*args, **kwargs):
        raise OSError(message)
    monkeypatch.setattr("meltiro.orchestrator.append_session_entry", _boom)


def test_a_complete_run_stays_complete_when_the_ledger_cannot_be_written(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")
    _break_run_log(monkeypatch)

    status = orch.run()

    # The status the run earned, returned and persisted — not a traceback.
    assert status == "complete"
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "complete"
    # And the session says the ledger is missing this run, in both the shape a
    # reader tests and the shape a reader reads.
    assert meta["run_log_entry_written"] is False
    assert any("run-log entry could not be written" in w
               for w in meta.get("warnings", []))
    assert _entries_for(out, orch.session.session_dir) == []
    err = capsys.readouterr().err
    assert "run-log entry could not be written" in err
    assert "Traceback" not in err


def test_a_writable_ledger_leaves_no_unwritten_flag(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    """The other half of the flag's meaning: absent means it was written."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    assert orch.run() == "complete"
    capsys.readouterr()

    meta = json.loads(orch.session.meta_path.read_text())
    assert "run_log_entry_written" not in meta
    assert len(_entries_for(out, orch.session.session_dir)) == 1


def test_a_failed_run_still_finalises_error_when_the_ledger_cannot_be_written(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    """The error path is where an unguarded append hurt most: `_finalise` runs
    from inside run()'s except handler, so a raise there leaves run() with the
    LEDGER's fault in place of the one that ended the run — as a traceback out
    of a function whose contract is to return a status."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    message = "ANTHROPIC_API_KEY is not set for the extractor stage"

    def _raise():
        raise AgenticExtractionError(message)
    monkeypatch.setattr(orch, "_extractor_loop", _raise)
    _break_run_log(monkeypatch)

    status = orch.run()

    assert status == "error"
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "error"
    # The sentence behind the status survives the ledger fault: it is what the
    # operator needs, and it is not in the run log to be read from.
    assert meta["error_message"] == message
    assert meta["run_log_entry_written"] is False
    err = capsys.readouterr().err
    assert message in err
    assert "run-log entry could not be written" in err


def test_an_unwritable_ledger_and_meta_together_do_not_raise(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    """One fault takes out both the ledger and meta — a read-only filesystem.
    The guard on the report is what keeps `_finalise` to its contract."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    # The filesystem goes read-only at the ledger append: everything the run
    # had already written is on disk, and nothing after it can be.
    gone = []
    real_write_meta = orch.session.write_meta

    def _append(*args, **kwargs):
        gone.append(True)
        raise OSError("run root is read-only")

    def _write_meta(*args, **kwargs):
        if gone:
            raise OSError("read-only file system")
        return real_write_meta(*args, **kwargs)

    def _boom(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("meltiro.orchestrator.append_session_entry", _append)
    monkeypatch.setattr(orch.session, "write_meta", _write_meta)
    monkeypatch.setattr(orch.session, "add_warning", _boom)

    assert orch.run() == "complete"
    assert "run-log entry could not be written" in capsys.readouterr().err


def test_an_unwritable_ledger_and_a_dead_stderr_do_not_raise(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    """The other fault the report has to survive: a stream that raises on
    write.

    A closed pipe is not a read-only disk, so the WARNING line is guarded on
    its own account, and the run keeps the status it earned whether or not
    anyone is left to read about it."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    _break_run_log(monkeypatch)

    class _DeadStream:
        def write(self, text):
            raise OSError("broken pipe")

        def flush(self):
            raise OSError("broken pipe")

    monkeypatch.setattr(sys, "stderr", _DeadStream())

    assert orch._finalise("complete") == "complete"

    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "complete"
    # The two reports that do not go through the stream still landed: one
    # unreachable reader costs the others nothing.
    assert meta["run_log_entry_written"] is False
    assert any("run-log entry could not be written" in w
               for w in meta.get("warnings", []))


def test_a_stale_reason_does_not_outlive_the_stop_that_wrote_it(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    """`_finalise` writes the reason keys before the status, so a fault in
    between leaves a session carrying the reason of a stop it never finished
    making. The finalisation that DOES land is run()'s `error`, which has no
    reason — and a run.json reading `error` beside `failure_reason:
    surrendered` describes two different runs."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    orch.extraction_record.abandon(  # the surrender the first stop was for
        "the paper reports no usable outcome")
    monkeypatch.setattr(orch, "_extractor_loop", lambda: "extractor_abandoned")

    real_finalise = orch.session.finalise
    faults = [OSError("the fault between the reason and the status")]

    def _finalise_once_broken(status):
        if faults:
            raise faults.pop()
        return real_finalise(status)
    monkeypatch.setattr(orch.session, "finalise", _finalise_once_broken)

    assert orch.run() == "error"
    capsys.readouterr()

    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "error"
    assert "failure_reason" not in meta
    assert "failed_validation_reason" not in meta


def test_the_transcript_outcome_table_carries_the_error_message(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    message = "the extractor model refused the request"

    def _raise():
        raise AgenticExtractionError(message)
    monkeypatch.setattr(orch, "_extractor_loop", _raise)
    orch.run()
    capsys.readouterr()

    transcript = (orch.session.session_dir / "diagnostics" /
                  "transcript.md").read_text()
    assert message in transcript
