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
    # raising, preserving the existing graceful behaviour. The install record
    # is silenced too, so this asserts git's own degradation and not whichever
    # way the test machine happens to have meltiro installed.
    def _no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", _no_git)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: None)
    assert run_log.git_state() == (None, None)


def _repo_with_commit(path):
    """A git repository at `path` holding whatever `path` already contains."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "initial"], cwd=path, check=True)
    return path


def _installed_layout(tmp_path):
    """A consumer repository with meltiro installed into a venv inside it.

    The ordinary layout — virtualenv at the project root — and the one that
    puts the package several levels inside a repository that is not its own.
    """
    consumer = tmp_path / "consumer"
    (consumer / "notes").mkdir(parents=True)
    (consumer / "notes" / "protocol.md").write_text("review\n")
    _repo_with_commit(consumer)
    anchor = (consumer / ".venv" / "lib" / "python3.14" / "site-packages"
              / "meltiro")
    anchor.mkdir(parents=True)
    (anchor / "run_log.py").write_text("# installed copy\n")
    return consumer, anchor


def test_git_state_declines_a_repo_that_does_not_track_the_package(
        tmp_path, monkeypatch):
    # A repository is found above an installed copy whenever the venv sits
    # inside one, and its HEAD describes the consumer's work rather than this
    # code. Untracked here means unattributable, so nothing is claimed.
    consumer, anchor = _installed_layout(tmp_path)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: None)
    # The walk does reach the consumer's repository: it is the attribution
    # that is refused, not the lookup that fails.
    assert run_log._get_git_commit() is not None
    assert run_log._anchor_tracked_in_repo() is False
    assert run_log.git_state() == (None, None)


def test_git_state_reports_the_repo_that_tracks_the_package(
        tmp_path, monkeypatch):
    # A source checkout tracks the package's files, so its HEAD is a true
    # description of them and the tree state beside it is this code's.
    checkout = tmp_path / "meltiro"
    anchor = checkout / "src" / "meltiro"
    anchor.mkdir(parents=True)
    (anchor / "run_log.py").write_text("# source\n")
    _repo_with_commit(checkout)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: None)

    commit, dirty = run_log.git_state()
    assert commit and dirty is False
    (anchor / "run_log.py").write_text("# an uncommitted edit\n")
    assert run_log.git_state() == (commit, True)


def test_the_dirty_flag_describes_the_package_not_the_whole_repo(
        tmp_path, monkeypatch):
    # The flag is read as a statement about the CODE THAT RAN. Asked of the
    # whole repository it is set by an edit to a file the engine never loads,
    # which for a copy vendored into a consumer's tree means their unrelated
    # work marks this package modified — the same false claim this module
    # exists to stop making, arriving through a different door.
    consumer, anchor = _installed_layout(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=consumer, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "vendored"], cwd=consumer, check=True)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: None)

    commit, dirty = run_log.git_state()
    assert commit and dirty is False
    # The consumer's own work, touching nothing of this package's.
    (consumer / "notes" / "protocol.md").write_text("revised\n")
    assert run_log.git_state() == (commit, False)
    # An edit to the package itself is what the flag is for.
    (anchor / "run_log.py").write_text("# patched after install\n")
    assert run_log.git_state() == (commit, True)


def test_a_tracked_install_still_has_its_tree_measured(
        tmp_path, monkeypatch):
    # The install record names a meltiro commit, which is the better answer to
    # "which meltiro is this" than a consumer repo's own HEAD. But a vendored
    # copy that the consumer committed HAS a working tree, so reporting the
    # tree unknown beside that commit would deny a fact git will state on
    # request — and would read as "installed, therefore pristine" over files
    # that have been patched.
    consumer, anchor = _installed_layout(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=consumer, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "vendored"], cwd=consumer, check=True)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: "abc1234")

    assert run_log.git_state() == ("abc1234", False)
    (anchor / "run_log.py").write_text("# patched after install\n")
    assert run_log.git_state() == ("abc1234", True)


def test_git_env_overrides_cannot_move_the_anchor(tmp_path, monkeypatch):
    # `GIT_DIR` without `GIT_WORK_TREE` makes git treat the process cwd as
    # that repository's work tree, so an unrelated repo would answer for this
    # package's files. Git exports it into everything it spawns — hooks,
    # `git bisect run`, `git rebase --exec` — so a run started from inside one
    # would otherwise attribute this code to whatever repository invoked it.
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "somebody-elses-work.md").write_text("not this package\n")
    _repo_with_commit(unrelated)
    anchor = tmp_path / "site-packages" / "meltiro"
    anchor.mkdir(parents=True)
    (anchor / "run_log.py").write_text("# installed copy\n")
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: None)
    monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))

    assert run_log.git_state() == (None, None)


def test_an_undecodable_path_name_degrades_rather_than_raising(
        tmp_path, monkeypatch):
    # `git ls-files` streams tracked path NAMES, so unlike `git status
    # --porcelain` on a clean tree it prints something on every call. A name
    # that is not valid in the process encoding must not raise out of a helper
    # whose contract is that it answers or returns None — least of all from
    # `append_run`, which runs after a completed extraction and would take the
    # run-log entry down with it.
    checkout = tmp_path / "meltiro"
    anchor = checkout / "src" / "meltiro"
    anchor.mkdir(parents=True)
    (anchor / "run_log.py").write_text("# source\n")
    (anchor / "caf\u00e9.py").write_text("# non-ascii name\n")
    _repo_with_commit(checkout)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: None)
    # An ASCII locale, which is what a stripped-down container gives a run.
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("PYTHONUTF8", "0")

    commit, dirty = run_log.git_state()
    assert commit and dirty is False


def test_git_state_prefers_the_installs_own_record(tmp_path, monkeypatch):
    # Where the installer wrote down what it fetched, that is the answer, and
    # the enclosing repository is not consulted at all. No working tree exists
    # to be clean or dirty, so the second component is null rather than a
    # guess.
    consumer, anchor = _installed_layout(tmp_path)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor)
    monkeypatch.setattr(run_log, "_installed_commit", lambda: "abc1234")
    assert run_log.git_state() == ("abc1234", None)


def _fake_distribution(monkeypatch, located, payload, version=None):
    """Stand in for the installed distribution's metadata.

    `located` is what `locate_file` reports the package directory to be,
    `payload` the text of `direct_url.json` (None for an install that has
    none, as a wheel from an index does), and `version` what the dist-info
    declares — defaulting to the running version, since that is the ordinary
    case and the mismatch is what a test has to ask for.
    """
    import importlib.metadata

    from meltiro import __version__

    declared = __version__ if version is None else version

    class _Dist:
        version = declared

        def locate_file(self, name):
            return located

        def read_text(self, name):
            return payload if name == "direct_url.json" else None

    monkeypatch.setattr(
        importlib.metadata.Distribution, "from_name", lambda name: _Dist())


def _direct_url(**vcs_info):
    return json.dumps({"url": "https://github.com/edpenington/meltiro",
                       "vcs_info": vcs_info})


def test_installed_commit_reads_the_pip_vcs_record(tmp_path, monkeypatch):
    # An install from a git URL records the resolved commit, and it is the
    # actual answer to which meltiro this is. Abbreviated to the length the
    # field carries from a checkout.
    anchor = tmp_path / "site-packages" / "meltiro"
    anchor.mkdir(parents=True)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor.resolve())
    _fake_distribution(monkeypatch, anchor, _direct_url(
        vcs="git", requested_revision="v0.4.0",
        commit_id="0123456789abcdef0123456789abcdef01234567"))
    assert run_log._installed_commit() == "0123456"


def test_installed_commit_ignores_metadata_for_another_copy(
        tmp_path, monkeypatch):
    # A distribution in the environment and a source tree ahead of it on
    # sys.path both answer to the name, and only one of them is running. The
    # record counts only when it belongs to the package that is imported.
    anchor = tmp_path / "checkout" / "src" / "meltiro"
    anchor.mkdir(parents=True)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor.resolve())
    _fake_distribution(monkeypatch, tmp_path / "site-packages" / "meltiro",
                       _direct_url(
                           vcs="git",
                           commit_id="0123456789abcdef0123456789abcdef0123"))
    assert run_log._installed_commit() is None


def test_installed_commit_none_without_a_vcs_record(tmp_path, monkeypatch):
    # An editable install, a directory install and a wheel from an index all
    # record no commit; the tree, where there is one, answers instead.
    anchor = tmp_path / "site-packages" / "meltiro"
    anchor.mkdir(parents=True)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor.resolve())
    for payload in (None,
                    json.dumps({"url": "file:///src",
                                "dir_info": {"editable": True}}),
                    "{not json",
                    _direct_url(vcs="hg", commit_id="abcdef1234"),
                    _direct_url(vcs="git", commit_id="")):
        _fake_distribution(monkeypatch, anchor, payload)
        assert run_log._installed_commit() is None


def test_installed_commit_ignores_a_dist_info_for_another_version(
        tmp_path, monkeypatch):
    # `Distribution.from_name` returns the FIRST match and prefers no version,
    # and `locate_file` resolves through the dist-info's parent, so every
    # distribution in one site-packages passes that guard identically. Two
    # `meltiro-*.dist-info` side by side is what `pip install --target` twice
    # leaves behind; without the version check the older one's commit would be
    # recorded beside the running version, a record contradicting itself.
    anchor = tmp_path / "site-packages" / "meltiro"
    anchor.mkdir(parents=True)
    monkeypatch.setattr(run_log, "_CODE_ANCHOR", anchor.resolve())
    _fake_distribution(monkeypatch, anchor, _direct_url(
        vcs="git",
        commit_id="0123456789abcdef0123456789abcdef01234567"),
        version="0.0.1-not-the-running-version")
    assert run_log._installed_commit() is None


def test_installed_commit_none_when_the_package_is_not_installed(monkeypatch):
    # Running from a tree on sys.path with nothing installed: the lookup
    # raises, and the question passes to the repository.
    import importlib.metadata

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(
        importlib.metadata.Distribution, "from_name", _missing)
    assert run_log._installed_commit() is None


def test_terminal_status_propagates(tmp_path):
    s = _make_session(tmp_path)
    s.finalise("failed_validation")
    e = build_entry(s, validation_passed=False,
                    validation_errors=["failed_validation: surrendered"])
    assert e["status"] == "failed_validation"
    assert e["validation_passed"] is False
    assert e["validation_errors"] == ["failed_validation: surrendered"]
