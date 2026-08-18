"""Tests for the meltiro CLI (validate-bundle + extract --dry-run).

No network, no API key required. The dry-run smoke test asserts that the
extract path renders the system prompt without ever constructing an
Anthropic client.
"""

import json
import shutil

import pytest
import yaml

import meltiro
from meltiro import cli
from meltiro.errors import AgenticExtractionError
from meltiro.fingerprint import engine_fingerprint
from meltiro.orchestrator import Orchestrator
from meltiro.run_log import engine_identity


def _run(argv):
    """Invoke main(argv); return the SystemExit code."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    code = excinfo.value.code
    return 0 if code is None else code


def _config_copy_without_key(tmp_path, config_dir, key):
    """Copy the shipped config into tmp_path and drop `key` from
    pipeline.yaml (rewritten via yaml, so all remaining keys stay known)."""
    dst = tmp_path / "config"
    shutil.copytree(config_dir, dst)
    pipeline_path = dst / "pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    pipeline.pop(key, None)
    pipeline_path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    return dst


class TestVersion:
    """`meltiro --version`: the release, and the engine axis it produces.

    The version is folded into `engine_fp` and from there into every run's
    `run_fp`, so an operator holding a run record needs to read it without
    opening Python. It takes no subcommand: the action fires during parsing,
    ahead of the required-subcommand check.
    """

    def test_version_needs_no_subcommand_and_exits_0(self, capsys):
        assert _run(["--version"]) == 0
        out = capsys.readouterr().out
        assert out.startswith(f"meltiro {meltiro.__version__}\n")

    def test_version_prints_the_engine_fingerprint_a_run_records(self, capsys):
        # Not decoration: this is the value `run.json` carries and `run_fp`
        # folds in, so the printed line must be the one a run would record
        # from the same tree, not a lookalike computed some other way.
        _run(["--version"])
        out = capsys.readouterr().out
        expected = engine_fingerprint(*engine_identity())
        assert expected in out
        assert "direktoro" in out

    def test_version_reports_the_working_tree_state(self, capsys):
        # A dirty tree makes the commit an incomplete description of the code
        # that ran, which is why engine_fp folds the flag in. Whichever state
        # this checkout is in, --version must name it rather than stay silent.
        _run(["--version"])
        out = capsys.readouterr().out
        assert any(s in out for s in
                   ("clean tree", "dirty tree", "tree not examined",
                    "no git repository"))


class TestValidateBundle:
    def test_valid_bundle_exit_0(self, bundle_minimal_dir, capsys):
        code = _run(["validate-bundle", str(bundle_minimal_dir)])
        out = capsys.readouterr().out
        assert code == 0
        assert "OK:" in out

    def test_invalid_bundle_exit_1(self, tmp_path, bundle_minimal_dir, capsys):
        bad = tmp_path / "bad"
        shutil.copytree(bundle_minimal_dir, bad)
        (bad / "text.md").unlink()
        code = _run(["validate-bundle", str(bad)])
        out = capsys.readouterr().out
        assert code == 1
        assert "INVALID:" in out
        assert "text.md is missing" in out

    def test_undeclared_crop_exit_1(self, tmp_path, bundle_minimal_dir,
                                    capsys):
        # A crop nobody declared is a hard error through the same problem
        # list, so the exhibit cross-checks reach the CLI's exit code.
        bad = tmp_path / "bad"
        shutil.copytree(bundle_minimal_dir, bad)
        src = bad / "figures" / "table_01.png"
        (bad / "figures" / "table_07.png").write_bytes(src.read_bytes())
        code = _run(["validate-bundle", str(bad)])
        out = capsys.readouterr().out
        assert code == 1
        assert "figures/table_07.png is not declared" in out

    def test_mixed_exit_1(self, tmp_path, bundle_minimal_dir, capsys):
        bad = tmp_path / "bad"
        shutil.copytree(bundle_minimal_dir, bad)
        (bad / "manifest.json").unlink()
        code = _run(["validate-bundle", str(bundle_minimal_dir), str(bad)])
        out = capsys.readouterr().out
        assert code == 1
        assert "OK:" in out
        assert "INVALID:" in out


class TestExtractDryRun:
    def test_dry_run_prints_prompt_and_makes_no_network_call(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # Guarantee zero network: if any code path tries to build an
        # Anthropic client, fail loudly.
        def _boom(self, role):
            raise AssertionError("dry-run must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)

        out_dir = tmp_path / "runs"
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
            "--dry-run",
        ])
        out = capsys.readouterr().out
        assert code == 0
        # The rendered system prompt + tool catalogue are printed IN FULL: a
        # dry run exists to show what would be sent, so a character cut would
        # hide the part an author most needs to read.
        assert "=== SYSTEM MESSAGE ===" in out
        assert "TOOL CATALOGUE" in out
        assert "truncated" not in out
        # A dry run creates NO session: no sessions/ dir, no run.json, no
        # extraction_output.json anywhere under the study output. The dry_run
        # report dir must not read as a session to a status-scanning consumer.
        assert not (out_dir / "demo-001" / "sessions").exists()
        assert list((out_dir / "demo-001").rglob("run.json")) == []
        assert list((out_dir / "demo-001").rglob(
            "extraction_output.json")) == []

    def test_dry_run_writes_untruncated_report_files(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        import json
        out_dir = tmp_path / "runs"
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
            "--dry-run",
        ])
        out = capsys.readouterr().out
        assert code == 0
        report = out_dir / "demo-001" / "dry_run"
        assert report.is_dir()
        # The report is NOT a session: no session-shaped files.
        assert not (report / "run.json").exists()
        assert not (report / "extraction_output.json").exists()
        assert not (report / "tool_calls.jsonl").exists()
        # The full, untruncated rendered system prompt is present in the file
        # AND on stdout. The length assertions below are what make this a real
        # test of untruncated output rather than a test of a short prompt.
        system_text = (report / "extractor_system.md").read_text(
            encoding="utf-8")
        assert len(system_text) > 4000
        assert system_text in out
        # The canonical tool catalogue is valid JSON, in full.
        catalogue = (report / "tool_catalogue.json").read_text(
            encoding="utf-8")
        assert len(catalogue) > 2000
        json.loads(catalogue)
        # All three stage fingerprints plus the whole-run run_fp are
        # recorded, with no status key.
        fps = json.loads(
            (report / "fingerprints.json").read_text(encoding="utf-8"))
        assert fps["config_fp"] and fps["checker_fp"] and fps["review_fp"]
        assert fps["run_fp"]
        assert "status" not in fps
        # The attached exhibits and the rendered checker + review prompts
        # are written.
        assert (report / "attached_exhibits.txt").exists()
        assert (report / "checker_system.md").exists()
        assert (report / "review_system.md").exists()

    def test_dry_run_previews_the_exhibits_as_a_role_is_shown_them(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        # End to end from the manifest: the bundle's declared caption reaches
        # the preview beside the label a role must cite, in the form the user
        # message will carry. The fixture declares one exhibit.
        out_dir = tmp_path / "runs"
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
            "--dry-run",
        ])
        capsys.readouterr()
        assert code == 0
        report = out_dir / "demo-001" / "dry_run"
        expected = ("[table_01] Table 1. Primary and secondary "
                    "associations between baseline CRT-HD total score and "
                    "each outcome")
        assert expected in (report / "attached_exhibits.txt").read_text(
            encoding="utf-8")
        # ... and neither system prompt names the paper's exhibits at all.
        for name in ("extractor_system.md", "review_system.md"):
            assert "table_01" not in (report / name).read_text(
                encoding="utf-8")

    def test_dry_run_rerun_prunes_toggled_off_stage_files(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        # Re-running a dry run into the SAME --out with a stage toggled off
        # must not leave a stale prompt file from the previous config next to
        # a fingerprints.json that says the stage is off. The report dir is
        # swapped whole, so files the current run does not write disappear.
        import json
        out_dir = tmp_path / "runs"
        argv_base = [
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
            "--dry-run",
        ]
        # First run: checker + review on (the shipped config enables both).
        code = _run(argv_base)
        capsys.readouterr()
        assert code == 0
        report = out_dir / "demo-001" / "dry_run"
        assert (report / "checker_system.md").exists()
        assert (report / "review_system.md").exists()

        # Second run into the same out dir with both stages off.
        code = _run(argv_base + ["--max-checks-per-field", "0",
                                 "--no-final-review"])
        capsys.readouterr()
        assert code == 0
        # The stale conditional prompt files are gone (pruned by the swap),
        # not left rotting next to a fingerprints.json that disagrees.
        assert not (report / "checker_system.md").exists()
        assert not (report / "review_system.md").exists()
        # The always-written files remain.
        assert (report / "extractor_system.md").exists()
        assert (report / "tool_catalogue.json").exists()
        assert (report / "attached_exhibits.txt").exists()
        # fingerprints.json now describes the toggled-off config, and matches
        # the surviving directory contents (no checker/review prompts).
        fps = json.loads(
            (report / "fingerprints.json").read_text(encoding="utf-8"))
        assert fps["structure"]["checker"] is False
        assert fps["structure"]["review"] is False
        assert fps["checker_model"] is None
        assert fps["review_model"] is None
        # No temp swap dir is left behind under the study dir.
        assert list((out_dir / "demo-001").glob("dry_run.tmp.*")) == []

    def test_dry_run_without_out_writes_no_files(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # With no --out, a dry run prints only: it must NOT create ./runs or
        # write any report files.
        monkeypatch.chdir(tmp_path)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--dry-run",
        ])
        out = capsys.readouterr().out
        assert code == 0
        assert "=== SYSTEM MESSAGE ===" in out
        assert not (tmp_path / "runs").exists()

    def test_dry_run_with_resume_exit_2(self, tmp_path, config_dir,
                                        bundle_minimal_dir, capsys):
        # Resuming a dry run makes no sense (a dry run creates no session):
        # fail loudly with a usage error.
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--dry-run",
            "--resume", str(tmp_path / "session"),
        ])
        err = capsys.readouterr().err
        assert code == 2
        assert "dry-run" in err and "resume" in err

    def test_dry_run_with_auto_resume_exit_2(self, tmp_path, config_dir,
                                             bundle_minimal_dir, capsys):
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--dry-run",
            "--auto-resume",
        ])
        err = capsys.readouterr().err
        assert code == 2
        assert "dry-run" in err

    def test_bad_bundle_path_exit_1(self, tmp_path, config_dir, capsys):
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(tmp_path / "does-not-exist"),
            "--out", str(tmp_path / "runs"),
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "Invalid paper bundle" in err

    def test_bad_config_path_exit_1(self, tmp_path, bundle_minimal_dir,
                                    capsys):
        code = _run([
            "extract",
            "--config", str(tmp_path / "no-config"),
            "--paper", str(bundle_minimal_dir),
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "Invalid config bundle" in err

    def test_resume_with_two_papers_exit_2(self, tmp_path, config_dir,
                                           bundle_minimal_dir, capsys):
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--paper", str(bundle_minimal_dir),
            "--resume", str(tmp_path / "session"),
        ])
        err = capsys.readouterr().err
        assert code == 2
        assert "exactly one --paper" in err

    def test_unknown_extractor_model_exit_1(self, tmp_path, config_dir,
                                            bundle_minimal_dir, capsys):
        # B1: an unknown model must fail loudly at startup, even on a dry-run.
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--extractor-model", "totally-made-up-model-9000",
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "unknown model" in err
        assert "totally-made-up-model-9000" in err
        # The message lists the known ids so the user can pick one.
        assert "claude-opus-4-7" in err

    def test_missing_checker_model_exit_1(self, tmp_path, config_dir,
                                          bundle_minimal_dir, capsys):
        # B2: a config that omits checker_model (and no --checker-model flag)
        # must fail loudly, exactly like the extractor and review models.
        config = _config_copy_without_key(tmp_path, config_dir,
                                          "checker_model")
        code = _run([
            "extract",
            "--config", str(config),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "checker_model" in err

    def test_missing_checker_model_satisfied_by_flag(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # The --checker-model flag alone satisfies the requirement.
        def _boom(self, role):
            raise AssertionError("dry-run must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)
        config = _config_copy_without_key(tmp_path, config_dir,
                                          "checker_model")
        code = _run([
            "extract",
            "--config", str(config),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--checker-model", "claude-sonnet-4-6",
            "--dry-run",
        ])
        assert code == 0

    def test_study_error_status_exit_1(self, tmp_path, config_dir,
                                       bundle_minimal_dir, monkeypatch):
        # B3: a study that finalises status "error" makes the command exit 1.
        monkeypatch.setattr(Orchestrator, "run", lambda self: "error")

        def _boom(self, role):
            raise AssertionError("must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
        ])
        assert code == 1

    def test_a_provider_account_pause_exits_1(self, tmp_path, config_dir,
                                              bundle_minimal_dir, monkeypatch):
        # The other pause. The session is resumable exactly like a cap pause,
        # but it stopped on something no rerun clears — a spent balance, a
        # revoked key — and every remaining paper in a batch will stop the
        # same way. Reporting that to a script as success would be worse than
        # what this failure did when it was terminal, so it keeps exit 1.
        def _pause(self):
            self.session.meta["pause_reason"] = "provider_account"
            self.session.write_meta()
            return "in_progress"
        monkeypatch.setattr(Orchestrator, "run", _pause)

        def _boom(self, role):
            raise AssertionError("must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
        ])
        assert code == 1

    def test_paused_status_exit_0(self, tmp_path, config_dir,
                                  bundle_minimal_dir, monkeypatch):
        # B3: a paused run produced a session and an extraction output, so it
        # stays exit 0. Only a hard "error" fails the command. The tool-call
        # cap is a budget the operator SET, reached; nothing outside the
        # process needs fixing before the next paper can run.
        monkeypatch.setattr(
            Orchestrator, "run", lambda self: "in_progress")

        def _boom(self, role):
            raise AssertionError("must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
        ])
        assert code == 0

    def test_failed_validation_exit_0_and_terminal(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch):
        # A stall now finalises as failed_validation: exit 0 (a session and a
        # partial extraction output were produced), but the status is terminal,
        # NOT flipped back to in_progress: it is not resumable.
        monkeypatch.setattr(
            Orchestrator, "run",
            lambda self: self._finalise("failed_validation",
                                        failure_reason="stalled"))

        def _boom(self, role):
            raise AssertionError("must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)
        out_dir = tmp_path / "runs"
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
        ])
        assert code == 0
        import json
        sessions = list((out_dir / "demo-001" / "sessions").iterdir())
        assert len(sessions) == 1
        meta = json.loads(
            (sessions[0] / "diagnostics" / "run.json").read_text())
        # Terminal: the CLI does NOT rewrite the status, so it stays
        # failed_validation with the mechanism recorded.
        assert meta["status"] == "failed_validation"
        assert meta["failure_reason"] == "stalled"

    def test_tool_cap_pause_exit_0_and_resumable(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch):
        # The tool-call cap PAUSES: the session stays in_progress (natively,
        # no meta rewrite) with a pause_reason, so --resume can reattach. The
        # command exits 0.
        monkeypatch.setattr(
            Orchestrator, "run", lambda self: self._pause("tool_cap_hit"))

        def _boom(self, role):
            raise AssertionError("must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)
        out_dir = tmp_path / "runs"
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
        ])
        assert code == 0
        import json
        sessions = list((out_dir / "demo-001" / "sessions").iterdir())
        assert len(sessions) == 1
        meta = json.loads(
            (sessions[0] / "diagnostics" / "run.json").read_text())
        assert meta["status"] == "in_progress"
        assert meta["pause_reason"] == "tool_cap_hit"

    def test_startup_guard_error_is_clean_and_exit_1(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # B3: an exception raised from the dry-run report path (for example the
        # study-identity guard, which dry_run_report runs) fails with a clean
        # one-line stderr message and exit 1, not a raw traceback.
        def _raise(self, report_dir=None):
            raise AgenticExtractionError("no study-identity context")
        monkeypatch.setattr(Orchestrator, "dry_run_report", _raise)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "no study-identity context" in err


class TestABrokenBundleRefusesCleanly:
    """A defect in the config bundle reaches the operator as a refusal, never
    as a traceback.

    `load_config_bundle` raises `ConfigBundleError` and nothing else, and the
    CLI's one handler prints it and exits 1. These are the three cases that
    used to escape around that handler: a template-model violation, a missing
    required template key, and malformed YAML.
    """

    def _broken(self, tmp_path, config_dir, edit):
        dst = tmp_path / "config"
        shutil.copytree(config_dir, dst)
        edit(dst)
        return dst

    def _extract(self, config, bundle_minimal_dir, tmp_path):
        return _run([
            "extract",
            "--config", str(config),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--dry-run",
        ])

    def test_an_unknown_template_field_key(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        def _edit(dst):
            tmpl = dst / "extraction_template.yaml"
            tmpl.write_text(
                tmpl.read_text(encoding="utf-8").replace(
                    "      required: true", "      requred: true", 1),
                encoding="utf-8")
        config = self._broken(tmp_path, config_dir, _edit)
        code = self._extract(config, bundle_minimal_dir, tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "Traceback" not in err
        assert "requred" in err

    def test_a_misspelt_variable_key(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        def _edit(dst):
            tmpl = dst / "extraction_template.yaml"
            tmpl.write_text(
                tmpl.read_text(encoding="utf-8").replace(
                    "    - variable:", "    - varible:", 1),
                encoding="utf-8")
        config = self._broken(tmp_path, config_dir, _edit)
        code = self._extract(config, bundle_minimal_dir, tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "Traceback" not in err
        assert "`variable:`" in err

    def test_a_broken_pipeline_yaml(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        def _edit(dst):
            (dst / "pipeline.yaml").write_text(
                "extractor_model: [unclosed\n", encoding="utf-8")
        config = self._broken(tmp_path, config_dir, _edit)
        code = self._extract(config, bundle_minimal_dir, tmp_path)
        err = capsys.readouterr().err
        assert code == 1
        assert "Traceback" not in err
        assert "pipeline.yaml" in err
        # pyyaml's own file/line/column diagnostic, which is the half that
        # points at the edit.
        assert "line" in err and "column" in err


class TestAFailedRunSaysWhatFailed:
    """`error` names a category. The run also composed a sentence, and that
    sentence has to reach the operator and the run record.

    A missing API key is the sharpest case: the pre-spend preflight names the
    environment variable and the stage, and that is the whole fix. Buried in
    the event log it costs a search; on stderr and in run.json it costs
    nothing.
    """

    def _key_env(self, model="claude-opus-4-8"):
        from direktoro import model_info
        return model_info(model).api_key_env

    def test_a_missing_key_prints_the_variable_and_records_it(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        env = self._key_env()
        monkeypatch.delenv(env, raising=False)

        def _boom(self, role):
            raise AssertionError("must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)

        out_dir = tmp_path / "runs"
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
        ])
        captured = capsys.readouterr()
        assert code == 1
        # (a) on stderr, ahead of the run summary on stdout.
        assert env in captured.err
        assert "missing API key(s) before run start" in captured.err

        session, = (out_dir / "demo-001" / "sessions").iterdir()
        meta = json.loads((session / "diagnostics" / "run.json").read_text())
        # (b) in run.json beside the status.
        assert meta["status"] == "error"
        assert env in meta["error_message"]

        # (c) in the run log, in place of the bare status word.
        log = json.loads((out_dir / "run_log.json").read_text())
        entry, = [e for e in log if e["session_dir"] == str(session)]
        assert entry["validation_errors"] != ["error"]
        assert any(env in e for e in entry["validation_errors"])

        # (d) in the transcript's outcome table.
        document = (session / "diagnostics" / "transcript.md").read_text()
        assert env in document

        # And the run summary says it too, rather than printing a bare status.
        assert "what failed:" in captured.out


class TestRetiredModelRejection:
    """A config naming a `retired` model fails loudly at config load, which is
    the new-run acceptance gate, NOT mid-run with a provider 404. The
    retired ids are injected synthetically so the test does not depend on any
    real id being withdrawn. A retired entry routed like a Claude model keeps
    the dry-run path otherwise identical to a live entry.
    """

    def _inject(self, monkeypatch, model_id, *, retired):
        from direktoro import registry
        m = registry.Model(
            registry.PROVIDER_ANTHROPIC, None, registry.ANTHROPIC_KEY_ENV,
            supports_images=True, forced_tool_choice=True, retired=retired)
        monkeypatch.setitem(registry.MODEL_REGISTRY, model_id, m)

    def test_retired_extractor_model_exit_1(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # A retired extractor id is known (so it passes is_known_model) but
        # must be rejected at startup, even on a dry-run, before any spend.
        self._inject(monkeypatch, "synthetic-retired-x", retired=True)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--extractor-model", "synthetic-retired-x",
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "retired model" in err
        assert "synthetic-retired-x" in err
        # Names the offending role and says it cannot be used for new runs.
        assert "extractor_model" in err
        assert "new runs" in err

    def test_retired_checker_model_exit_1(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # The gate covers the checker model too (an enabled stage), so a
        # retired checker is caught at load, not on its first challenge call.
        self._inject(monkeypatch, "synthetic-retired-checker", retired=True)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--checker-model", "synthetic-retired-checker",
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 1
        assert "retired model" in err
        assert "synthetic-retired-checker" in err
        assert "checker_model" in err

    def test_non_retired_synthetic_model_is_unaffected(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys,
            monkeypatch):
        # A known, non-retired model passes the gate exactly as before: the
        # guard trips only on retired=True, so the field/guard do not disturb
        # the normal new-run path.
        self._inject(monkeypatch, "synthetic-live-x", retired=False)
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--extractor-model", "synthetic-live-x",
            "--dry-run",
        ])
        err = capsys.readouterr().err
        assert code == 0
        assert "retired model" not in err


class TestCheckerDecodingWiring:
    """The config bundle, plus CLI flags, is the SINGLE source for the
    checker's decoding params. `checker_decoding` in pipeline.yaml wires
    through to CheckerConfig, and the environment does not feed it at all."""

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True)

    def _orch(self, config_dir, bundle_minimal_dir, tmp_path, loop_cfg):
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        return cli._build_orchestrator(
            config, bundle, tmp_path / "runs", loop_cfg, self._args())

    def test_checker_decoding_from_pipeline_is_wired(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_decoding"] = {"temperature": 0.7}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling == {"temperature": 0.7}

    def test_an_unwritten_checker_block_is_specified_nowhere(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No checker_decoding key: the checker specifies none, so none is
        # sent and the model's own defaults apply. The dataclass carries no
        # default value to stand in for one nobody wrote.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("checker_decoding", None)
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling is None
        assert orch.checker_config.thinking is None

    def test_an_explicit_zero_in_the_block_is_honoured(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # 0.0 is the value a checker is most often given, and a truthy read of
        # the block would drop it and leave the model sampling.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_decoding"] = {"temperature": 0.0}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling == {"temperature": 0.0}

    def test_environment_does_not_feed_checker_decoding(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # CHECKER_TEMPERATURE / CHECKER_MAX_TOKENS are NOT read: a shell that
        # sets them must not change the resolved checker config. An environment
        # leak here moves checker_fp between shells, so the same bundle would
        # fingerprint differently depending on who ran it.
        monkeypatch.setenv("CHECKER_TEMPERATURE", "0.9")
        monkeypatch.setenv("CHECKER_MAX_TOKENS", "77")
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("checker_decoding", None)
        loop_cfg["checker_max_tokens"] = 1024
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling is None
        assert orch.checker_config.max_tokens == 1024


class TestCheckerMaxTokensWiring:
    """`checker_max_tokens` from pipeline.yaml wires through to CheckerConfig,
    and is required whenever the checker stage is on. A zero or negative cap
    is rejected loudly at startup rather than shipped to the provider as 0."""

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True)

    def _orch(self, config_dir, bundle_minimal_dir, tmp_path, loop_cfg):
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        return cli._build_orchestrator(
            config, bundle, tmp_path / "runs", loop_cfg, self._args())

    def test_checker_max_tokens_from_pipeline_is_wired(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_max_tokens"] = 2048
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.max_tokens == 2048

    def test_a_missing_checker_max_tokens_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # No checker_max_tokens key, checker on: there is no cap to run under
        # and none for meltiro to invent, so the run is refused at startup
        # naming the key rather than started under a number nobody chose.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("checker_max_tokens", None)
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "checker_max_tokens" in capsys.readouterr().err

    def test_zero_checker_max_tokens_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A zero token cap is never a valid checker budget, and it fails at
        # startup, before any spend.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_max_tokens"] = 0
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "checker_max_tokens" in err

    def test_negative_checker_max_tokens_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_max_tokens"] = -5
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "checker_max_tokens" in err


class TestPerRoleDecodingWiring:
    """Each role's decoding block is independently tunable and reaches only its
    own stage. Nothing is inherited: a role whose block is absent specifies
    nothing, whatever the other roles say. An explicit 0.0 inside a block is
    honoured rather than swallowed."""

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True)

    def _orch(self, config_dir, bundle_minimal_dir, tmp_path, loop_cfg):
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        return cli._build_orchestrator(
            config, bundle, tmp_path / "runs", loop_cfg, self._args())

    def test_the_extractor_block_is_the_extractors(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["extractor_decoding"] = {"temperature": 0.4}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.sampling == {"temperature": 0.4}

    def test_the_review_block_is_the_reviewers_alone(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["review_decoding"] = {"temperature": 0.6}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_sampling == {"temperature": 0.6}
        # And it does not leak into the other two roles.
        assert orch.sampling == {"temperature": 1.0}
        assert orch.checker_config.sampling == {"temperature": 0.0}

    def test_an_absent_review_block_inherits_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No review_decoding key: the reviewer specifies nothing and its
        # model's own defaults apply. It does NOT take the extractor's block —
        # the two roles do different work, and a control copied silently
        # between them would be recorded as the reviewer's own choice.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("review_decoding", None)
        loop_cfg["extractor_decoding"] = {"temperature": 0.3}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_sampling is None
        assert orch.sampling == {"temperature": 0.3}

    def test_explicit_zero_in_the_review_block_is_honoured(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A truthy read of the block would drop the 0.0 and leave the reviewer
        # specifying nothing while pipeline.yaml says it is greedy.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["extractor_decoding"] = {"temperature": 0.3}
        loop_cfg["review_decoding"] = {"temperature": 0.0}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_sampling == {"temperature": 0.0}

    def test_an_unwritten_temperature_is_specified_nowhere(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No block names it anywhere: nothing is specified, so nothing is sent
        # and each model's own default applies. There is deliberately no engine
        # default value — one would be indistinguishable from an operator's
        # choice, and would be reported as inert against a model that refuses
        # it.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("extractor_decoding", None)
        loop_cfg.pop("review_decoding", None)
        loop_cfg.pop("checker_decoding", None)
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        # None on every role: one spelling of "specified nothing", the same
        # one direktoro's split returns for a block that names no control.
        assert orch.sampling is None
        assert orch.review_sampling is None
        assert orch.checker_config.sampling is None

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 5.0, 100])
    def test_a_value_outside_the_models_band_is_refused_at_startup(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, bad):
        # The checker's model documents a temperature band, and a value outside
        # it is a 400 on a paid call. Refused at startup instead, naming the
        # role and the block to edit.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_decoding"] = {"temperature": bad}
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "checker_decoding" in capsys.readouterr().err

    @pytest.mark.parametrize("ok", [0.0, 0.2, 1.0])
    def test_a_value_inside_the_band_is_carried_through(
            self, config_dir, bundle_minimal_dir, tmp_path, ok):
        # The band is inclusive at both ends. The accepted value is read back
        # off the built orchestrator, so this pins that it was carried through
        # rather than only that nothing raised on the way.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_decoding"] = {"temperature": ok}
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling == {"temperature": ok}


class TestPerRoleDecodingFingerprints:
    """A per-role decoding block moves that role's stage fingerprint and no
    other, but only for a model that actually accepts what it names: the
    fingerprint folds in what is SENT, so a model that refuses a control has a
    fingerprint inert to it."""

    def _args(self, **over):
        from types import SimpleNamespace
        base = dict(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True)
        base.update(over)
        return SimpleNamespace(**base)

    def _fps(self, config_dir, bundle_minimal_dir, tmp_path, loop_cfg,
             **args_over):
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        orch = cli._build_orchestrator(
            config, bundle, tmp_path / "runs", loop_cfg,
            self._args(**args_over))
        return orch._build_fingerprints()

    def test_the_checker_block_moves_only_checker_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # claude-sonnet-4-6 (the shipped checker) accepts temperature, so this
        # is a live knob and must move checker_fp alone.
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg)
        hot_cfg = dict(base_cfg)
        hot_cfg["checker_decoding"] = {"temperature": 0.5}
        hot = self._fps(config_dir, bundle_minimal_dir, tmp_path, hot_cfg)
        assert hot["checker_fp"] != base["checker_fp"]
        assert hot["config_fp"] == base["config_fp"]
        assert hot["review_fp"] == base["review_fp"]

    def test_the_review_block_moves_only_review_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Repointed at a temperature-accepting review model, since the shipped
        # claude-opus-4-8 rejects temperature and would move nothing.
        over = {"review_model": "claude-sonnet-4-6"}
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg,
                         **over)
        hot_cfg = dict(base_cfg)
        hot_cfg["review_decoding"] = {"temperature": 0.5}
        hot = self._fps(config_dir, bundle_minimal_dir, tmp_path, hot_cfg,
                        **over)
        assert hot["review_fp"] != base["review_fp"]
        assert hot["config_fp"] == base["config_fp"]
        assert hot["checker_fp"] == base["checker_fp"]

    def test_the_extractor_block_moves_only_config_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Both the extractor and the reviewer are repointed at a
        # temperature-accepting model, so that review_fp holding proves the
        # reviewer does NOT ride the extractor's block, rather than merely
        # proving the review model ignores it.
        over = {"extractor_model": "claude-sonnet-4-6",
                "review_model": "claude-sonnet-4-6"}
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg,
                         **over)
        hot_cfg = dict(base_cfg)
        hot_cfg["extractor_decoding"] = {"temperature": 0.5}
        hot = self._fps(config_dir, bundle_minimal_dir, tmp_path, hot_cfg,
                        **over)
        assert hot["config_fp"] != base["config_fp"]
        assert hot["review_fp"] == base["review_fp"]
        assert hot["checker_fp"] == base["checker_fp"]

    def test_temperature_is_inert_for_a_no_temperature_model(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The shipped models: claude-opus-4-8 rejects temperature, so neither
        # the extractor's nor the reviewer's value reaches the wire and neither
        # fingerprint moves. This pins the fact that makes the shipped values
        # reasoned defaults rather than live settings.
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg)
        hot_cfg = dict(base_cfg)
        hot_cfg["extractor_decoding"] = {"temperature": 0.5}
        hot_cfg["review_decoding"] = {"temperature": 0.5}
        hot = self._fps(config_dir, bundle_minimal_dir, tmp_path, hot_cfg)
        assert hot["config_fp"] == base["config_fp"]
        assert hot["review_fp"] == base["review_fp"]

    def test_dry_run_report_shows_resolved_decoding_params(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The inertness above is otherwise undiscoverable without a paid run, so
        # the dry run reports what each role actually sends: no temperature for
        # the opus roles, the checker's for the sonnet one.
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        orch = cli._build_orchestrator(
            config, bundle, tmp_path / "runs", config.pipeline, self._args())
        report = orch.dry_run_report()
        dec = report["fingerprints"]["decoding_params"]
        assert "temperature" not in dec["extractor"]
        assert "temperature" not in dec["review"]
        assert dec["checker"]["temperature"] == 0.0

    def test_dry_run_decoding_params_are_per_role(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Every role repointed at a temperature-accepting model and given a
        # DISTINCT temperature, so the report is pinned to reading each role's
        # own value. With the shipped opus roles this is unobservable (nothing
        # is sent either way), which is exactly why it needs its own case.
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        loop_cfg = dict(config.pipeline)
        loop_cfg["extractor_decoding"] = {"temperature": 0.2}
        loop_cfg["review_decoding"] = {"temperature": 0.7}
        loop_cfg["checker_decoding"] = {"temperature": 0.4}
        orch = cli._build_orchestrator(
            config, bundle, tmp_path / "runs", loop_cfg,
            self._args(extractor_model="claude-sonnet-4-6",
                       review_model="claude-sonnet-4-6"))
        dec = orch.dry_run_report()["fingerprints"]["decoding_params"]
        assert dec["extractor"]["temperature"] == 0.2
        assert dec["review"]["temperature"] == 0.7
        assert dec["checker"]["temperature"] == 0.4

    def test_disabled_stages_report_no_decoding_params(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A disabled stage's model is not required, so it must not be resolved
        # through the registry: None rather than a guessed dict.
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        orch = cli._build_orchestrator(
            config, bundle, tmp_path / "runs", dict(config.pipeline),
            self._args(max_checks_per_field=0, final_review=False))
        dec = orch.dry_run_report()["fingerprints"]["decoding_params"]
        assert dec["checker"] is None
        assert dec["review"] is None
        assert dec["extractor"] is not None


def load_config_bundle_pipeline(config_dir):
    from meltiro.config_bundle import load_config_bundle
    return load_config_bundle(config_dir).pipeline


class TestTheRunRecordsWhatWasSpecified:
    """`run.json` carries both halves of the decoding story: the block the
    operator wrote, and the params the wire actually carried.

    A model that refuses a sampling control is sent none of it, silently and
    by design, and the value moves no fingerprint. With only the wire side
    recorded, "wrote a temperature the model dropped" and "wrote nothing at
    all" are the same artefact — and the first is a methodological claim the
    run did not honour.
    """

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True)

    def _orch(self, config_dir, bundle_minimal_dir, tmp_path, loop_cfg):
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        return cli._build_orchestrator(
            config, bundle, tmp_path / "runs", loop_cfg, self._args())

    def test_a_refused_value_and_an_unwritten_one_are_told_apart(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The fixture's extractor model refuses the sampling controls, so
        # these two runs send byte-identical decoding params. Only the record
        # of what was ASKED FOR separates them.
        written = dict(load_config_bundle_pipeline(config_dir))
        written["extractor_decoding"] = {"temperature": 0.3}
        unwritten = dict(load_config_bundle_pipeline(config_dir))
        unwritten.pop("extractor_decoding", None)

        a = self._orch(config_dir, bundle_minimal_dir, tmp_path, written)
        b = self._orch(config_dir, bundle_minimal_dir, tmp_path, unwritten)

        assert a.decoding_specified["extractor"] == {"temperature": 0.3}
        assert "extractor" not in b.decoding_specified
        assert (a._decoding_params_meta()["extractor"]
                == b._decoding_params_meta()["extractor"])
        assert "temperature" not in a._decoding_params_meta()["extractor"]

    def test_the_written_block_reaches_run_json(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch):
        # Written at session creation, so it is in the artefact whatever the
        # run then does — including a run that never gets an answer back from
        # the role whose block it records.
        monkeypatch.setattr(Orchestrator, "run", lambda self: "in_progress")
        out_dir = tmp_path / "runs"
        assert _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(out_dir),
        ]) == 0
        sessions = list((out_dir / "demo-001" / "sessions").iterdir())
        meta = json.loads(
            (sessions[0] / "diagnostics" / "run.json").read_text())
        # Verbatim, per role, exactly as the fixture pipeline writes them.
        assert meta["decoding_specified"] == {
            "extractor": {"temperature": 1.0},
            "review": {"temperature": 0.0},
            "checker": {"temperature": 0.0},
        }


class TestTheTotalSaysWhatItCovers:
    """Three states, three lines. A run states its total, states none, or
    states a floor — and the floor is the one that would otherwise be
    indistinguishable from a whole bill that happened to be small."""

    def _extract(self, tmp_path, config_dir, bundle_minimal_dir):
        return _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
        ])

    def _spending(self, cost, *, unreceipted=0, unpriced=False):
        """A run that banks `cost` and, optionally, a gap in its coverage.

        `unpriced` is the separate fault: a role ran that nothing could price,
        so the run states no figure at all. It combines with `unreceipted`,
        because a run can meet both.
        """
        def _run_with(self):
            self._cost_usd = cost
            self._cost_counted = True
            self._cost_unpriced = unpriced
            if unreceipted:
                self._cost_incomplete = True
                self._unreceipted_calls = unreceipted
            self._checkpoint_usage_to_meta()
            # The checkpoint is in-memory; a real run flushes it at the next
            # meta write, and this stub makes no other.
            self.session.write_meta()
            return "complete"
        return _run_with

    def test_a_missing_receipt_makes_the_printed_total_a_floor(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        monkeypatch.setattr(Orchestrator, "run",
                            self._spending(0.0123, unreceipted=2))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "Total cost: at least $0.0123 (2 call(s) returned no receipt)" \
            in out

    def test_a_fully_receipted_run_prints_the_figure_plainly(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # The qualifier has to mean something where it appears.
        monkeypatch.setattr(Orchestrator, "run", self._spending(0.0123))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "Total cost: $0.0123" in out
        assert "at least" not in out

    def test_an_unpriced_run_still_reports_its_missing_receipts(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # The two faults are independent, and the unpriced one is the louder.
        # Returning on it early drops the quieter one entirely: the operator is
        # told nothing was priced and never told that calls also came back with
        # no charge to price.
        monkeypatch.setattr(Orchestrator, "run",
                            self._spending(0.0, unreceipted=2, unpriced=True))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        flat = " ".join(capsys.readouterr().out.split())
        assert ("Total cost: not priced (tokens recorded; a role ran with "
                "neither a `rates:` card nor a price-table entry) — 2 call(s) "
                "returned no receipt and are missing from any figure") in flat
        # And no floor: there is no figure for one to be a floor over.
        assert "at least" not in flat

    def test_a_zero_sum_with_a_gap_is_not_printed_as_a_floor(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # Every receipt the run got is in the figure and it is still nothing.
        # "at least $0.0000" reads as a bill just above zero; what is true is
        # that nothing receipted was charged, over calls nobody can price.
        monkeypatch.setattr(Orchestrator, "run",
                            self._spending(0.0, unreceipted=2))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert ("Total cost: no receipted charge (2 call(s) returned no "
                "receipt)") in out
        assert "$0.0000" not in out

    def test_the_coverage_is_in_run_json_too(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch):
        # stdout scrolls away; the artefact is what a ledger is built from, so
        # the figure carries its coverage there as well.
        out_dir = tmp_path / "runs"
        monkeypatch.setattr(Orchestrator, "run",
                            self._spending(0.0123, unreceipted=2))
        assert _run([
            "extract", "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir), "--out", str(out_dir),
        ]) == 0
        sessions = list((out_dir / "demo-001" / "sessions").iterdir())
        meta = json.loads(
            (sessions[0] / "diagnostics" / "run.json").read_text())
        assert meta["cost_usd"] == 0.0123
        assert meta["cost_incomplete"] is True
        assert meta["unreceipted_calls"] == 2


class TestADegradedCheckerIsLoud:
    """A checker that answered nothing leaves a run with no challenges in it,
    which on stdout reads exactly like a run the checker was happy with. The
    end-of-run summary has to tell those two apart, or the quieter outcome is
    the worse one."""

    def _extract(self, tmp_path, config_dir, bundle_minimal_dir):
        return _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
        ])

    def _finishing_with(self, diagnostics):
        def _run_with(self):
            self.session.meta["checker_diagnostics"] = diagnostics
            return "complete"
        return _run_with

    def test_a_checker_that_answered_nothing_prints_a_note(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        monkeypatch.setattr(Orchestrator, "run", self._finishing_with({
            "fields_checked": 3, "checks_run": 3, "checks_reprompted": 0,
            "unresolved_challenges": [],
            "checker_errors": ["study.a", "study.b", "study.c"]}))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "NOTE: 3 field(s) ended with no verdict at all" in out
        assert "the checker call" in out
        assert "run.checker_diagnostics" in out

    def test_the_two_checker_counts_are_never_printed_as_a_fraction(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # They count different things. `checker_errors` is one entry per FIELD
        # whose last verdict was an error; `checks_run` counts every check the
        # run made, and a field can be checked more than once. Printed as "N
        # field(s) of the M check(s)" the pair reads as a fraction, and here it
        # would read as 2 of 5 when both fields the checker was asked about
        # failed.
        monkeypatch.setattr(Orchestrator, "run", self._finishing_with({
            "fields_checked": 2, "checks_run": 5, "checks_reprompted": 0,
            "unresolved_challenges": [],
            "checker_errors": ["study.a", "study.b"]}))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "2 field(s) ended with no verdict at all" in out
        assert "This run made 5 check(s) in total." in out
        assert "of the 5 check(s)" not in out

    def test_re_asked_checks_are_counted_out_loud(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # Every re-ask was a second billed call, and a model that needs
        # nudging is marginal for the role even when every nudge worked.
        monkeypatch.setattr(Orchestrator, "run", self._finishing_with({
            "fields_checked": 2, "checks_run": 2, "checks_reprompted": 2,
            "unresolved_challenges": [], "checker_errors": []}))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "NOTE: 2 check(s) needed a re-ask before answering or failing" \
            in out

    def test_the_re_ask_note_claims_only_what_the_number_counts(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # A re-asked check that then failed is in this tally too, so the note
        # cannot say the re-asks were followed by a verdict. Here every check
        # made was re-asked AND every one of them failed: a note claiming a
        # verdict would be false about all of them at once.
        monkeypatch.setattr(Orchestrator, "run", self._finishing_with({
            "fields_checked": 2, "checks_run": 2, "checks_reprompted": 2,
            "unresolved_challenges": [],
            "checker_errors": ["study.a", "study.b"]}))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "before the checker recorded a" not in out
        assert "answering or failing" in out

    def test_a_healthy_checker_says_nothing_extra(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # The counterpart: the notice has to mean something when it appears.
        monkeypatch.setattr(Orchestrator, "run", self._finishing_with({
            "fields_checked": 2, "checks_run": 2, "checks_reprompted": 0,
            "unresolved_challenges": [], "checker_errors": []}))
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "verdict at all" not in out
        assert "re-asked" not in out

    def test_the_label_says_checks_not_calls(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        # `checker_calls_run` counts CHECKS: a check re-asked once made two
        # provider calls and is one of these. Printed as "Checker calls" the
        # figure reads as a call count and understates the wire.
        monkeypatch.setattr(Orchestrator, "run", lambda self: "in_progress")
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        assert "  Checks run: " in capsys.readouterr().out


class TestAutoResumeSaysWhatItPassedOver:
    """`--auto-resume` starting fresh is a decision to spend the study's money
    again. When there was in-progress work it could not use, it says so: silence
    is indistinguishable from there having been nothing there."""

    def _stale_session(self, out_dir, study_id="demo-001"):
        from meltiro.session import Session
        return Session.create(
            study_id, config_fp="config_fp:stale",
            checker_fp="checker_fp:stale", review_fp="review_fp:stale",
            instrument_fp="instrument_fp:x", extractor_call_fp="call_fp:e",
            checker_call_fp="call_fp:c", review_call_fp="call_fp:r",
            engine_fp="engine_fp:e", extractor_model="opus",
            checker_model="sonnet", review_model="opus",
            tool_set_hash="ts", template_hash="th", prompt_hash="ph",
            runs_dir=out_dir)

    def _extract(self, tmp_path, config_dir, bundle_minimal_dir):
        return _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "runs"),
            "--auto-resume",
        ])

    def test_an_unmatched_in_progress_session_is_named(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        monkeypatch.setattr(Orchestrator, "run", lambda self: "in_progress")
        self._stale_session(tmp_path / "runs")
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        out = capsys.readouterr().out
        assert "1 in-progress session(s) for this study were left alone" in out
        assert "extractor fingerprint" in out
        assert "Starting fresh." in out

    def test_nothing_to_resume_says_nothing(
            self, tmp_path, config_dir, bundle_minimal_dir, monkeypatch,
            capsys):
        monkeypatch.setattr(Orchestrator, "run", lambda self: "in_progress")
        assert self._extract(tmp_path, config_dir, bundle_minimal_dir) == 0
        assert "left alone" not in capsys.readouterr().out


class TestResumeWritesToTheSessionsOwnRoot:
    """A resumed run appends its run-log entry beside the session it continues.

    The root is DERIVED from the session path rather than taken from `--out`,
    and an `--out` that disagrees with it is refused rather than obeyed.
    Taking it from `--out` splits a run from its own run-log index, which is
    what makes runs comparable: the pause message prints a resume command
    carrying no `--out`, so an operator following the tool's own recovery
    instructions would append the entry to the ./runs default while the
    session itself sat under the original root.
    """

    def _paused_session(self, tmp_path, study_id):
        """A session directory deep enough to carry a run root, with the
        run.json a resume reads before doing anything else."""
        root = tmp_path / "myruns"
        s = root / study_id / "sessions" / "20260101_000000_abc123"
        (s / "diagnostics").mkdir(parents=True)
        (s / "diagnostics" / "run.json").write_text(
            json.dumps({"study_id": study_id, "status": "in_progress"}))
        return root, s

    def test_an_out_pointing_elsewhere_is_refused(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        # Exercised through cli.main, so deleting the guard fails this test.
        _, session = self._paused_session(tmp_path, "demo-001")
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(tmp_path / "somewhere-else"),
            "--resume", str(session),
        ])
        err = capsys.readouterr().err
        assert code == 2, err
        assert "disagrees with the session" in err
        assert str(tmp_path / "myruns") in err

    def test_an_out_naming_the_sessions_own_root_is_accepted(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        # The guard must refuse only a genuine mismatch. Passing the right root
        # gets past it, and the run then fails for its own unrelated reasons.
        root, session = self._paused_session(tmp_path, "demo-001")
        code = _run([
            "extract",
            "--config", str(config_dir),
            "--paper", str(bundle_minimal_dir),
            "--out", str(root),
            "--resume", str(session),
        ])
        assert "disagrees with the session" not in capsys.readouterr().err
        # Exit 2 is what the guard itself returns (see the refusal test above),
        # so getting past it means a DIFFERENT non-zero code: the stub session
        # cannot actually resume. `code != 0` alone would also hold for a
        # reworded version of the very guard this test claims to have cleared.
        assert code not in (0, 2)


class TestAliasEditMovesEveryStageFingerprint:
    """An alias edit reaches all three stage fingerprints of a real run.

    Aliases are rendered into no prompt, so no prompt hash can carry them, yet
    every stage that writes or judges a reference value is affected: the
    extractor and the reviewer both drive the same `ToolDispatcher` (an alias
    decides which entered values are accepted and what they canonicalise to),
    and the checker is shown the same vocabulary. A run in which two of the
    three moved and one held would report that stage as unchanged while it
    worked under a different vocabulary.
    """

    def _args(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True)

    def _fps(self, config_dir, bundle_minimal_dir, out_dir):
        from meltiro.bundle import load_bundle
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(str(bundle_minimal_dir))
        orch = cli._build_orchestrator(
            config, bundle, out_dir, dict(config.pipeline), self._args())
        return orch._build_fingerprints()

    def test_editing_an_alias_moves_all_three(
            self, config_dir, bundle_minimal_dir, tmp_path):
        import shutil
        import yaml as _yaml

        base_dir = tmp_path / "base"
        edited_dir = tmp_path / "edited"
        shutil.copytree(config_dir, base_dir)
        shutil.copytree(config_dir, edited_dir)

        ref = edited_dir / "reference" / "gauge_list.yaml"
        data = _yaml.safe_load(ref.read_text(encoding="utf-8"))
        entries = next(v for v in data.values() if isinstance(v, list))
        target = next(e for e in entries if isinstance(e, dict))
        target["aliases"] = list(target.get("aliases") or []) + ["a handle"]
        ref.write_text(_yaml.safe_dump(data), encoding="utf-8")

        base = self._fps(base_dir, bundle_minimal_dir, tmp_path / "r1")
        edited = self._fps(edited_dir, bundle_minimal_dir, tmp_path / "r2")

        # Canonical names are untouched, so no rendered prompt differs: the
        # content hash is the only route by which any of these can move.
        for key in ("config_fp", "checker_fp", "review_fp"):
            assert edited[key] != base[key], key
        assert edited["instrument_fp"] != base["instrument_fp"]
        assert edited["run_fp"] != base["run_fp"]
