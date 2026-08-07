"""Tests for the session-level run-log builder."""

import json
import os
import subprocess
from pathlib import Path

from meltiro import run_log
from meltiro.run_entry import (
    append_session_entry, build_entry)
from meltiro.run_log import git_state
from meltiro.session import Session


def _make_session(tmp_path):
    return Session.create(
        "376",
        config_fp="config_fp:aabbccdd",
        checker_fp="checker_fp:11223344",
        review_fp="review_fp:55667788",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        extractor_model="claude-opus-4-7",
        checker_model="claude-sonnet-4-5",
        review_model="claude-opus-4-7",
        tool_set_hash="ts_hash", template_hash="tmpl_hash",
        prompt_hash="prmpt_hash",
        runs_dir=tmp_path,
    )


def test_build_entry_shape(tmp_path):
    s = _make_session(tmp_path)
    s.increment_tool_call_count()
    s.record_checker_calls(1)
    s.finalise("complete")
    e = build_entry(s, input_tokens=12345, output_tokens=678,
                    cost_usd=0.42, validation_passed=True)
    assert e["study_id"] == "376"
    assert e["agentic"] is True
    assert e["tool_call_count"] == 1
    assert e["checker_calls_run"] == 1
    assert e["input_tokens"] == 12345
    assert e["cost_usd"] == 0.42
    assert e["status"] == "complete"
    assert e["model"] == "claude-opus-4-7"
    assert e["checker_model"] == "claude-sonnet-4-5"
    assert e["review_model"] == "claude-opus-4-7"
    assert e["config_fp"] == "config_fp:aabbccdd"
    assert e["checker_fp"] == "checker_fp:11223344"
    assert e["review_fp"] == "review_fp:55667788"
    # The whole-run fingerprint rides into the run-log entry too, so a consumer
    # keying on run_fp never has to open run.json. It is the same value the
    # session derived and stored, i.e. run_fingerprint over the three stage
    # fps and the engine fp.
    from meltiro.fingerprint import run_fingerprint
    assert e["run_fp"] == run_fingerprint(
        "config_fp:aabbccdd", "checker_fp:11223344", "review_fp:55667788",
        "engine_fp:eng")
    assert e["run_fp"] == s.meta["run_fp"]
    assert "session_dir" in e
    assert "result_file" in e
    # Both per-role fields are always written, so a consumer never has to tell
    # "this run priced nothing" apart from "this entry lacks the key".
    assert e["cost_rates"] == {}
    assert e["usage_by_role"] == {}


def test_build_entry_carries_the_per_role_spend(tmp_path):
    # Pricing is per role, so the index carries a role's counters, its cost,
    # and the card that produced it together: a consumer checks a role's
    # arithmetic from this row alone.
    s = _make_session(tmp_path)
    s.finalise("complete")
    card = {"input_per_1m": 5.0, "output_per_1m": 25.0,
            "cache_read_per_1m": 0.5, "cache_write_per_1m": 6.25,
            "source": "table", "as_of": "2026-08-07", "table_version": 1}
    by_role = {"extractor": {
        "input_tokens": 1000, "output_tokens": 400,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 0.015, "cost_rates": card}}
    e = build_entry(s, input_tokens=1000, output_tokens=400, cost_usd=0.015,
                    cost_rates={"extractor": card}, usage_by_role=by_role)
    assert e["cost_rates"] == {"extractor": card}
    assert e["usage_by_role"] == by_role


def test_append_session_entry_writes_to_log(tmp_path):
    s = _make_session(tmp_path)
    s.finalise("complete")
    append_session_entry(s, log_dir=tmp_path, input_tokens=10)

    log_path = tmp_path / "run_log.json"
    assert log_path.exists()
    with open(log_path) as f:
        log = json.load(f)
    assert len(log) == 1
    assert log[0]["study_id"] == "376"
    assert log[0]["agentic"] is True
    # append_run adds timestamp, git_commit, and git_dirty on its own.
    assert "timestamp" in log[0]
    assert "git_commit" in log[0]
    # The code-version anchor: a bool when git is available, None otherwise.
    assert "git_dirty" in log[0]
    assert log[0]["git_dirty"] in (True, False, None)


class TestRecordedPathsAreAbsolute:
    """run_log.json is the cross-run index a downstream consumer parses, and
    nothing in any artefact records the cwd a run was invoked from, so a
    relative `session_dir` / `result_file` is unresolvable. Both are recorded
    absolute whatever the caller passed in: the Session resolves its directory
    at construction."""

    def test_entry_paths_are_absolute_from_a_relative_runs_dir(
            self, tmp_path, monkeypatch):
        # Exactly the CLI's shape: `--out runs` with a relative path, resolved
        # against the invocation cwd.
        monkeypatch.chdir(tmp_path)
        s = Session.create(
            "376",
            config_fp="config_fp:aabbccdd", checker_fp=None, review_fp=None,
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            extractor_model="claude-opus-4-7", checker_model=None,
            review_model=None, tool_set_hash="ts", template_hash="tm",
            prompt_hash="pr",
            runs_dir=Path("runs"),
        )
        s.finalise("complete")
        e = build_entry(s)
        assert Path(e["session_dir"]).is_absolute()
        assert Path(e["result_file"]).is_absolute()
        # Absolute AND correct: the recorded paths point at the real files.
        assert Path(e["session_dir"]).is_dir()
        assert Path(e["result_file"]).exists()

    def test_entry_paths_are_absolute_for_a_resumed_session(
            self, tmp_path, monkeypatch):
        # `--resume` hands Session.resume a path straight off the command line,
        # so resolving only the run root would leave a resumed run's entry
        # relative.
        monkeypatch.chdir(tmp_path)
        created = _make_session(tmp_path / "runs")
        rel = Path(os.path.relpath(created.session_dir, tmp_path))
        assert not rel.is_absolute()

        s = Session.resume(rel)
        s.finalise("complete")
        e = build_entry(s)
        assert Path(e["session_dir"]).is_absolute()
        assert Path(e["result_file"]).is_absolute()
        assert Path(e["session_dir"]) == created.session_dir

    def test_recorded_paths_are_canonical_through_a_symlinked_run_root(
            self, tmp_path):
        # Deliberate choice (resolve() over abspath): the recorded path names
        # the directory the run actually wrote to, so it survives the symlink
        # being repointed and two spellings of one destination record the same
        # path. The documented cost is that the recorded path need NOT be
        # string-prefixed by `--out`, so a consumer must not match on that.
        real = tmp_path / "real_runs"
        real.mkdir()
        link = tmp_path / "linked_runs"
        link.symlink_to(real, target_is_directory=True)

        s = Session.create(
            "376",
            config_fp="config_fp:aabbccdd", checker_fp=None, review_fp=None,
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            extractor_model="claude-opus-4-7", checker_model=None,
            review_model=None, tool_set_hash="ts", template_hash="tm",
            prompt_hash="pr",
            runs_dir=link,
        )
        s.finalise("complete")
        e = build_entry(s)

        recorded = Path(e["session_dir"])
        assert recorded.is_absolute()
        assert str(recorded).startswith(str(real))
        # The path is correct however it is spelled: it names the same
        # directory the caller's own (symlinked) path does.
        assert recorded.samefile(link / "376" / "sessions" / s.meta["session_id"])
        assert Path(e["result_file"]).exists()

    def test_appended_log_entry_keeps_absolute_paths(self, tmp_path,
                                                     monkeypatch):
        # The end-to-end property a consumer depends on: what lands in the
        # file on disk is absolute, and resolves from any cwd.
        monkeypatch.chdir(tmp_path)
        s = Session.create(
            "376",
            config_fp="config_fp:aabbccdd", checker_fp=None, review_fp=None,
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            extractor_model="claude-opus-4-7", checker_model=None,
            review_model=None, tool_set_hash="ts", template_hash="tm",
            prompt_hash="pr",
            runs_dir=Path("runs"),
        )
        s.finalise("complete")
        append_session_entry(s, log_dir=Path("runs"))

        with open(tmp_path / "runs" / "run_log.json") as f:
            log = json.load(f)
        monkeypatch.chdir(Path(os.sep))
        assert Path(log[0]["session_dir"]).is_dir()
        assert Path(log[0]["result_file"]).exists()


def test_git_state_shape():
    # (short_commit, dirty): commit is str-or-None, dirty is bool-or-None.
    commit, dirty = git_state()
    assert commit is None or isinstance(commit, str)
    assert dirty in (True, False, None)


def test_git_state_graceful_when_git_unavailable(monkeypatch):
    # When git is not on PATH, both components degrade to None rather than
    # raising, preserving the existing graceful behaviour.
    def _no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", _no_git)
    assert run_log.git_state() == (None, None)


def test_terminal_status_propagates(tmp_path):
    s = _make_session(tmp_path)
    s.finalise("failed_validation")
    e = build_entry(s, validation_passed=False,
                    validation_errors=["failed_validation: surrendered"])
    assert e["status"] == "failed_validation"
    assert e["validation_passed"] is False
    assert e["validation_errors"] == ["failed_validation: surrendered"]
