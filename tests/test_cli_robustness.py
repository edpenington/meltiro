"""The `extract` command's remaining rough edges: what it refuses, when it
says a run is starting, and what it says about settings it will not read.

Every test here is about the moment BEFORE the first paid call. A refusal that
arrives late has already printed a banner and a rate report for a run that
never happens; a bound that was coerced rather than checked has already ridden
into `run.json` as a number the operator appears to have chosen; a setting for
a stage that is off reads exactly like one in force.

No network and no API key: the runs here refuse before an adapter exists, and
the ones that get further stub the loop.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from meltiro import cli
from meltiro.orchestrator import Orchestrator


def _run(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    code = excinfo.value.code
    return 0 if code is None else code


def _config_with(tmp_path, config_dir, **overrides):
    """A copy of the shipped config with `overrides` merged into pipeline.yaml.

    A None value DELETES the key, so a test can drop `review_model` as easily
    as it can set one.
    """
    dst = tmp_path / "config"
    shutil.copytree(config_dir, dst)
    path = dst / "pipeline.yaml"
    pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if value is None:
            pipeline.pop(key, None)
        else:
            pipeline[key] = value
    path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    return dst


def _extract(config, paper, out, *extra):
    return _run(["extract", "--config", str(config), "--paper", str(paper),
                 "--out", str(out), *extra])


# ---------------------------------------------------------------------------
# (a) Operational bounds are checked, never coerced
# ---------------------------------------------------------------------------

class TestABoundIsCheckedNotCoerced:
    """`int("50")`, `int(50.9)` and `int(True)` all succeed, and each writes a
    bound nobody chose into `run.json` looking exactly like one the operator
    did choose. Every bound the CLI reads from pipeline.yaml is type-checked
    instead."""

    @pytest.mark.parametrize("key", [
        "max_tool_calls",
        "max_review_tool_calls",
        "max_checks_per_field",
        "checker_concurrency",
    ])
    @pytest.mark.parametrize("value", ["50", 2.5, True])
    def test_a_non_integer_bound_is_refused_naming_the_key(
            self, key, value, tmp_path, config_dir, bundle_minimal_dir,
            capsys):
        config = _config_with(tmp_path, config_dir, **{key: value})
        code = _extract(config, bundle_minimal_dir, tmp_path / "runs",
                        "--dry-run")
        err = capsys.readouterr().err
        assert code == 1
        assert key in err
        assert "must be an integer" in err

    def test_an_integer_bound_still_loads(
            self, tmp_path, config_dir, bundle_minimal_dir):
        config = _config_with(tmp_path, config_dir, max_tool_calls=7)
        assert _extract(config, bundle_minimal_dir, tmp_path / "runs",
                        "--dry-run") == 0

    def test_a_non_integer_env_concurrency_names_the_variable(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # The value came from the shell, so the message must send the operator
        # to the shell rather than to pipeline.yaml. pipeline.yaml's own key
        # wins over the variable, so it is removed for this test.
        monkeypatch.setenv("CHECKER_CONCURRENCY", "lots")
        config = _config_with(tmp_path, config_dir, checker_concurrency=None)
        code = _extract(config, bundle_minimal_dir, tmp_path / "runs",
                        "--dry-run")
        err = capsys.readouterr().err
        assert code == 1
        assert "CHECKER_CONCURRENCY" in err
        assert "environment variable" in err


# ---------------------------------------------------------------------------
# (b) An unusable --out is a refusal, not a crash partway through
# ---------------------------------------------------------------------------

class TestAnUnusableOutIsRefusedFirst:

    def test_out_pointing_at_a_file_is_refused(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        not_a_dir = tmp_path / "runs"
        not_a_dir.write_text("I am a file.\n", encoding="utf-8")
        code = _extract(config_dir, bundle_minimal_dir, not_a_dir)
        captured = capsys.readouterr()
        assert code == 1
        assert "--out" in captured.err
        assert "not a directory" in captured.err
        # Before the banner: nothing announced a run that never started.
        assert "=== Study" not in captured.out

    def test_an_unwritable_out_is_refused(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        parent = tmp_path / "locked"
        parent.mkdir()
        parent.chmod(0o500)
        try:
            code = _extract(config_dir, bundle_minimal_dir, parent / "runs")
            captured = capsys.readouterr()
            assert code == 1
            assert "--out" in captured.err
            assert "=== Study" not in captured.out
        finally:
            parent.chmod(0o700)

    def test_a_usable_out_is_created(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch):
        monkeypatch.setattr(Orchestrator, "run", lambda self: "complete")
        out = tmp_path / "deep" / "runs"
        assert _extract(config_dir, bundle_minimal_dir, out) == 0
        assert out.is_dir()

    def test_a_writable_root_passes_and_keeps_its_probe_to_itself(
            self, tmp_path, monkeypatch):
        """The branch that lets a run start: the root is created, the probe is
        written and removed, and the directory is left exactly as clean as it
        was found.

        The probe's name carries the pid because one run root is shared: a
        batch is several `meltiro extract` processes probing the same
        directory, and a fixed name makes the probe a file two of them own at
        once.
        """
        names = []
        real_write_text = Path.write_text

        def _record(self, *args, **kwargs):
            names.append(self.name)
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _record)
        out = tmp_path / "deep" / "runs"

        assert cli._out_dir_problem(out) is None
        assert out.is_dir()
        assert names == [f".meltiro-write-probe.{os.getpid()}"]
        # Nothing left behind: the probe is removed, not merely overwritten.
        assert list(out.iterdir()) == []

    def test_a_probe_that_vanished_is_not_a_refusal(self, tmp_path,
                                                    monkeypatch):
        """A probe removed underneath this process — by a concurrent run's
        cleanup, or a tmp reaper — is not evidence that the root is
        unwritable, and the write it just completed is evidence that it is."""
        monkeypatch.setattr(Path, "write_text",
                            lambda self, *args, **kwargs: 0)
        assert cli._out_dir_problem(tmp_path) is None

    def test_a_probe_a_killed_run_left_behind_is_swept(self, tmp_path):
        """The probe is named for a process, so once that process is gone
        nothing else will ever come back for it: a run SIGKILLed between the
        write and the unlink leaves a permanent dotfile at the run root, and
        one per killed run. The next probe of the same root removes them."""
        orphan = tmp_path / ".meltiro-write-probe.999999"
        orphan.write_text("", encoding="utf-8")
        kept = tmp_path / "run_log.json"
        kept.write_text("[]", encoding="utf-8")

        assert cli._out_dir_problem(tmp_path) is None

        assert not orphan.exists()
        # The sweep is keyed to the probe's own name: nothing else at the run
        # root is this function's to remove.
        assert kept.exists()

    def test_a_probe_this_process_cannot_remove_stops_only_itself(
            self, tmp_path, monkeypatch):
        """A shared run root collects probes from every user who ran a batch
        there, and one this process may not unlink is one stale file — not a
        reason to leave every other run's behind, and not a reason to refuse a
        root that is writable."""
        for pid in (999998, 999999):
            (tmp_path / f".meltiro-write-probe.{pid}").write_text(
                "", encoding="utf-8")
        blocked = []
        real_unlink = Path.unlink

        def _first_removal_fails(self, *args, **kwargs):
            if not blocked:
                blocked.append(self.name)
                raise PermissionError(f"not permitted: {self.name}")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _first_removal_fails)

        assert cli._out_dir_problem(tmp_path) is None

        # The one that raised is the only one left: the sweep carried on past
        # it, and this process's own probe was written and removed as ever.
        left = sorted(p.name for p in
                      tmp_path.glob(".meltiro-write-probe.*"))
        assert left == blocked


# ---------------------------------------------------------------------------
# (c) Nothing is announced above a refusal
# ---------------------------------------------------------------------------

class TestNothingIsAnnouncedAboveARefusal:
    """The banner and the rate report both say "a run is starting, and it will
    cost money". Printed above a refusal they describe a run that did not
    happen, and send the operator looking for a session nobody created."""

    def test_a_refused_resume_prints_no_pricing(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # A real paused session, then a paper edit, so the refusal is the
        # paper gate rather than an argument mistake.
        paper = tmp_path / "paper"
        shutil.copytree(bundle_minimal_dir, paper)
        out = tmp_path / "runs"
        monkeypatch.setattr(Orchestrator, "run",
                            lambda self: self._pause("tool_cap_hit"))
        assert _extract(config_dir, paper, out) == 0
        session, = (out / "demo-001" / "sessions").iterdir()
        capsys.readouterr()

        text = paper / "text.md"
        text.write_text(text.read_text(encoding="utf-8") + "\nedited\n",
                        encoding="utf-8")
        code = _run(["extract", "--config", str(config_dir),
                     "--paper", str(paper), "--resume", str(session)])
        captured = capsys.readouterr()
        assert code == 2
        assert "Resume refused" in captured.err
        assert "Pricing," not in captured.out

    def test_a_started_run_does_print_both(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # The pair to the test above: when a run genuinely starts, the operator
        # still gets the banner and the report.
        monkeypatch.setattr(Orchestrator, "run", lambda self: "complete")
        assert _extract(config_dir, bundle_minimal_dir, tmp_path / "runs") == 0
        out = capsys.readouterr().out
        assert "=== Study demo-001 ===" in out
        assert "Pricing, extractor" in out


# ---------------------------------------------------------------------------
# (d) One study per invocation
# ---------------------------------------------------------------------------

class TestADuplicatePaperIsSkipped:

    def test_the_same_study_twice_runs_once_and_says_so(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # Two directories, one study id: the same paper, and running it twice
        # would bill it twice and leave two run-log rows nothing tells apart.
        second = tmp_path / "copy"
        shutil.copytree(bundle_minimal_dir, second)
        runs = []
        monkeypatch.setattr(Orchestrator, "run",
                            lambda self: runs.append(self.study_id)
                            or "complete")
        code = _run(["extract", "--config", str(config_dir),
                     "--paper", str(bundle_minimal_dir),
                     "--paper", str(second),
                     "--out", str(tmp_path / "runs")])
        out = capsys.readouterr().out
        assert code == 0
        assert runs == ["demo-001"]
        assert "skipping duplicate --paper" in out
        assert str(second) in out
        # One line, not one per skipped bundle beyond the first.
        assert out.count("skipping duplicate --paper") == 1


# ---------------------------------------------------------------------------
# (e) A bundle that says nothing still briefs every model
# ---------------------------------------------------------------------------

class TestASilentBundleStillRuns:

    def test_empty_prompt_files_load_and_the_run_is_still_briefed(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch):
        # A review may have nothing of its own to say to a role. The engine
        # composes that role's spine regardless, so the run starts and the
        # captured instrument shows the model was told what the tools do.
        config = _config_with(tmp_path, config_dir)
        for name in ("extractor_system", "review_system", "checker_system"):
            (config / "prompts" / f"{name}.md").write_text(
                "", encoding="utf-8")

        monkeypatch.setattr(Orchestrator, "run", lambda self: "complete")
        out_dir = tmp_path / "runs"
        assert _extract(config, bundle_minimal_dir, out_dir) == 0

        session, = (out_dir / "demo-001" / "sessions").iterdir()
        system = (session / "diagnostics" / "instrument"
                  / "system_prompt.txt").read_text(encoding="utf-8")
        assert "record_initial_check" in system


# ---------------------------------------------------------------------------
# (f) --resume takes a session directory
# ---------------------------------------------------------------------------

class TestResumeNamesWhatItWants:

    def test_pointing_resume_at_the_wrong_directory_exits_2(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        code = _run(["extract", "--config", str(config_dir),
                     "--paper", str(bundle_minimal_dir),
                     "--resume", str(tmp_path)])
        err = capsys.readouterr().err
        # Exit 2 like every other guard on this path (wrong study, --out
        # disagreement, a refused resume), not 1.
        assert code == 2
        assert "run.json" in err
        # The fuller message: what --resume actually takes, and where it lives.
        assert "SESSION directory" in err
        assert "extraction_output.json" in err


# ---------------------------------------------------------------------------
# (g) Settings for a stage that is off are named
# ---------------------------------------------------------------------------

class TestSettingsForADisabledStageAreNamed:

    def test_review_keys_under_no_final_review(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        config = _config_with(tmp_path, config_dir, final_review=False)
        assert _extract(config, bundle_minimal_dir, tmp_path / "runs",
                        "--dry-run") == 0
        err = capsys.readouterr().err
        assert "ignored-stage-settings" in err
        assert "review_model" in err
        assert "final reviewer" in err

    def test_checker_keys_under_a_zero_check_budget(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        config = _config_with(tmp_path, config_dir, max_checks_per_field=0)
        assert _extract(config, bundle_minimal_dir, tmp_path / "runs",
                        "--dry-run") == 0
        err = capsys.readouterr().err
        assert "ignored-stage-settings" in err
        assert "checker_model" in err
        assert "checker" in err

    def test_an_all_stages_on_run_says_nothing(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        assert _extract(config_dir, bundle_minimal_dir, tmp_path / "runs",
                        "--dry-run") == 0
        assert "ignored-stage-settings" not in capsys.readouterr().err
