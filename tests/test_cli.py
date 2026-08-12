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
        def _boom(self):
            raise AssertionError("dry-run must not construct an API client")
        monkeypatch.setattr(Orchestrator, "_anthropic_client", _boom)

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
        # Figure labels and the rendered checker + review prompts are written.
        assert (report / "figure_labels.txt").exists()
        assert (report / "checker_system.md").exists()
        assert (report / "review_system.md").exists()

    def test_dry_run_renders_exhibit_captions_into_both_prompts(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        # End to end from the manifest: the bundle's declared caption reaches
        # the extractor's and the reviewer's rendered system prompt, beside
        # the label they must cite. The fixture declares one exhibit.
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
        expected = ("- `table_01`: Table 1. Primary and secondary "
                    "associations between baseline CRT-HD total score and "
                    "each outcome")
        for name in ("extractor_system.md", "review_system.md"):
            assert expected in (report / name).read_text(encoding="utf-8")

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
        assert (report / "figure_labels.txt").exists()
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
        def _boom(self):
            raise AssertionError("dry-run must not construct an API client")
        monkeypatch.setattr(Orchestrator, "_anthropic_client", _boom)
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

        def _boom(self):
            raise AssertionError("must not construct an API client")
        monkeypatch.setattr(Orchestrator, "_anthropic_client", _boom)
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
        # stays exit 0. Only a hard "error" fails the command.
        monkeypatch.setattr(
            Orchestrator, "run", lambda self: "in_progress")

        def _boom(self):
            raise AssertionError("must not construct an API client")
        monkeypatch.setattr(Orchestrator, "_anthropic_client", _boom)
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

        def _boom(self):
            raise AssertionError("must not construct an API client")
        monkeypatch.setattr(Orchestrator, "_anthropic_client", _boom)
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

        def _boom(self):
            raise AssertionError("must not construct an API client")
        monkeypatch.setattr(Orchestrator, "_anthropic_client", _boom)
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
            retired=retired)
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


class TestCheckerTemperatureWiring:
    """The config bundle, plus CLI flags, is the SINGLE source for the
    checker's decoding params. `checker_temperature` in pipeline.yaml wires
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

    def test_checker_temperature_from_pipeline_is_wired(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_temperature"] = 0.7
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling == {"temperature": 0.7}

    def test_an_unwritten_checker_temperature_is_specified_nowhere(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No checker_temperature key: the checker specifies none, so none is
        # sent and the model's own default applies. The dataclass carries no
        # default value to stand in for one nobody wrote.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("checker_temperature", None)
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling is None

    def test_environment_does_not_feed_checker_decoding(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # CHECKER_TEMPERATURE / CHECKER_MAX_TOKENS are NOT read: a shell that
        # sets them must not change the resolved checker config. An environment
        # leak here moves checker_fp between shells, so the same bundle would
        # fingerprint differently depending on who ran it.
        monkeypatch.setenv("CHECKER_TEMPERATURE", "0.9")
        monkeypatch.setenv("CHECKER_MAX_TOKENS", "77")
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("checker_temperature", None)
        loop_cfg.pop("checker_max_tokens", None)
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling is None
        assert orch.checker_config.max_tokens == 1024


class TestCheckerMaxTokensWiring:
    """`checker_max_tokens` from pipeline.yaml wires through to CheckerConfig
    under an `is not None` presence contract, NOT a truthy one: a truthy check
    treats an explicit 0 as absent and silently substitutes the default. A zero
    or negative cap is instead rejected loudly at startup, rather than falling
    back to the default or being shipped to the provider as 0."""

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

    def test_default_checker_max_tokens_is_1024(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No checker_max_tokens key: the dataclass default (1024) stands.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("checker_max_tokens", None)
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.max_tokens == 1024

    def test_zero_checker_max_tokens_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A zero token cap is never a valid checker budget. A truthy presence
        # check would fall back to 1024 silently; `is not None` lets the 0
        # reach a loud validation that fails at startup, before any spend.
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


class TestPerRoleTemperatureWiring:
    """Each role's temperature is independently tunable and reaches only its own
    stage. `temperature` is the extractor's (and the reviewer's default);
    `review_temperature` decouples the reviewer; the checker never inherits
    either. An explicit 0.0 is honoured under the `is not None` presence
    contract rather than swallowed as absent."""

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

    def test_temperature_is_the_extractors(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["temperature"] = 0.4
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.sampling == {"temperature": 0.4}

    def test_review_temperature_from_pipeline_is_wired(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["review_temperature"] = 0.6
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_sampling == {"temperature": 0.6}
        # And it does not leak into the other two roles.
        assert orch.sampling == {"temperature": loop_cfg["temperature"]}
        assert orch.checker_config.sampling == {"temperature": 0.0}

    def test_absent_review_temperature_inherits_extractor(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No review_temperature key: the reviewer inherits `temperature`, which
        # is what it got unconditionally before the key existed. Adding the key
        # must not silently change an existing config's reviewer.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("review_temperature", None)
        loop_cfg["temperature"] = 0.3
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_sampling == {"temperature": 0.3}

    def test_explicit_zero_review_temperature_is_honoured(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The point of `is not None` over a truthy check: an explicit 0.0 must
        # keep the reviewer greedy while the extractor samples, not be swallowed
        # as absent and inherit the extractor's 0.3.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["temperature"] = 0.3
        loop_cfg["review_temperature"] = 0.0
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_sampling == {"temperature": 0.0}

    def test_an_unwritten_temperature_is_specified_nowhere(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Neither key present: nothing is specified, so nothing is sent and
        # each model's own default applies. There is deliberately no engine
        # default value — one would be indistinguishable from an operator's
        # choice, and would be reported as inert against a model that refuses
        # it.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg.pop("temperature", None)
        loop_cfg.pop("review_temperature", None)
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.sampling == {}
        assert orch.review_sampling == {}

    @pytest.mark.parametrize("key", [
        "temperature", "review_temperature", "checker_temperature"])
    @pytest.mark.parametrize("bad", [-0.1, -1, 2.1, 5.0, 100])
    def test_out_of_range_temperature_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, key, bad):
        # Range-checked at startup for every role. For a model that refuses
        # the control the provider never sees the value to reject it, so
        # startup is the only place it can be caught at all.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg[key] = bad
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert key in capsys.readouterr().err

    @pytest.mark.parametrize("key", [
        "temperature", "review_temperature", "checker_temperature"])
    @pytest.mark.parametrize("ok", [0.0, 0.2, 1.0, 2.0])
    def test_in_range_temperature_accepted(
            self, config_dir, bundle_minimal_dir, tmp_path, key, ok):
        # The band is inclusive at both ends and must not reject a usable
        # value. The accepted value is read back off the built orchestrator, so
        # this pins that it was carried through rather than only that nothing
        # raised on the way.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg[key] = ok
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        resolved = {
            "temperature": lambda o: o.sampling,
            "review_temperature": lambda o: o.review_sampling,
            "checker_temperature": lambda o: o.checker_config.sampling,
        }[key](orch)
        assert resolved == {"temperature": ok}


class TestPerRoleThinkingWiring:
    """The six `<role>_thinking_mode` / `<role>_thinking_effort` keys reach the
    role they name and nothing else, and a bad value fails at startup.

    The per-role plumbing is the thing under test here; what the values then DO
    (reach the wire, move the fingerprints, force a cap floor) is pinned in
    tests/agentic_extraction/test_thinking.py against the real engine.
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

    def test_no_thinking_keys_means_no_spec_on_any_role(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The shipped example names none of the six, and must keep behaving
        # exactly as it did: three roles, no thinking spec anywhere.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.thinking is None
        assert orch.review_thinking is None
        assert orch.checker_config.thinking is None

    def test_each_role_reads_its_own_two_keys(
            self, config_dir, bundle_minimal_dir, tmp_path):
        from direktoro import Thinking
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["extractor_thinking_mode"] = "adaptive"
        loop_cfg["extractor_thinking_effort"] = "max"
        loop_cfg["review_thinking_effort"] = "low"
        loop_cfg["checker_thinking_mode"] = "disabled"
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.thinking == Thinking(mode="adaptive", effort="max")
        assert orch.review_thinking == Thinking(effort="low")
        assert orch.checker_config.thinking == Thinking(mode="disabled")

    def test_the_reviewer_does_not_inherit_the_extractors_spec(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Deliberately unlike `review_temperature`, which inherits when absent.
        # Inheriting would make "no spec" inexpressible for the reviewer, and a
        # thinking mode is not a dial to copy silently between roles that do
        # different work.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["extractor_thinking_effort"] = "max"
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_thinking is None
        assert orch.checker_config.thinking is None

    @pytest.mark.parametrize("key,bad", [
        ("extractor_thinking_mode", "sometimes"),
        ("checker_thinking_mode", "budget"),
        ("review_thinking_effort", "maximum"),
        ("checker_thinking_effort", "HIGH"),
    ])
    def test_a_bad_value_fails_at_startup_naming_its_key(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, key, bad):
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg[key] = bad
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert key in capsys.readouterr().err

    def test_a_thinking_checker_with_the_shipped_cap_fails_at_startup(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The cap hazard, reached through the CLI: the shipped
        # `checker_max_tokens: 1024` cannot hold a think plus a verdict, so
        # asking the checker to think is refused with a one-line startup error
        # rather than a traceback or a truncated run.
        #
        # The checker is repointed at Opus 4.8 to ask that question in
        # isolation. On the shipped `claude-sonnet-4-6` this configuration is
        # refused one step EARLIER, for pairing `checker_temperature` with
        # active thinking (the test below), so it never reaches the cap. Opus
        # 4.8 rejects the temperature parameter outright, so direktoro omits it
        # and the cap is the only thing left to fail on.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        loop_cfg["checker_model"] = "claude-opus-4-8"
        loop_cfg["checker_thinking_mode"] = "adaptive"
        loop_cfg["checker_max_tokens"] = 1024
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "checker_max_tokens" in err
        assert "2048" in err

    def test_thinking_on_the_shipped_checker_is_refused_for_its_temperature(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The shipped bundle's own constraint, at the CLI, on its own values.
        #
        # The bundle declares `checker_model: claude-sonnet-4-6` and
        # `checker_temperature: 0.0`. Both are fine, and the bundle AS SHIPPED
        # is legal because it names no thinking key. But Sonnet 4.6 rejects a
        # request that carries a temperature and turns thinking on, so adding
        # `checker_thinking_mode: adaptive` makes it a 400. That is one line,
        # and a natural line for an author to add. Refused at startup,
        # unbilled, naming the role, the model and BOTH keys, so the operator
        # can see that dropping `checker_temperature` is as valid an answer as
        # giving up the thinking.
        #
        # No cap involved: `checker_max_tokens` is raised well clear of the
        # floor so the temperature conflict is unambiguously what fired.
        loop_cfg = dict(load_config_bundle_pipeline(config_dir))
        assert loop_cfg["checker_model"] == "claude-sonnet-4-6"
        assert loop_cfg["checker_temperature"] is not None
        loop_cfg["checker_thinking_mode"] = "adaptive"
        loop_cfg["checker_max_tokens"] = 8192
        with pytest.raises(SystemExit) as excinfo:
            self._orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "checker role" in err
        assert "claude-sonnet-4-6" in err
        assert "checker_temperature" in err
        assert "checker_thinking_mode" in err

    def test_an_unknown_thinking_key_is_still_rejected_by_the_allowlist(
            self, config_dir, tmp_path):
        # The six keys are on the allowlist; a seventh (a typo) is not, and the
        # config bundle refuses it at load, before any of this runs.
        from meltiro.config_bundle import load_config_bundle
        from meltiro.errors import ConfigBundleError
        import shutil
        dest = tmp_path / "cfg"
        shutil.copytree(config_dir, dest)
        pipeline = dest / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8")
            + "\nextractor_thinking_efort: high\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as exc:
            load_config_bundle(dest)
        assert "extractor_thinking_efort" in str(exc.value)


class TestPerRoleTemperatureFingerprints:
    """A per-role temperature moves that role's stage fingerprint and no other,
    but only for a model that actually accepts temperature: the fingerprint
    folds in what is SENT, so a no_temperature model's fingerprint is inert to
    it."""

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

    def test_checker_temperature_moves_only_checker_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # claude-sonnet-4-6 (the shipped checker) accepts temperature, so this
        # is a live knob and must move checker_fp alone.
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg)
        hot_cfg = dict(base_cfg)
        hot_cfg["checker_temperature"] = 0.5
        hot = self._fps(config_dir, bundle_minimal_dir, tmp_path, hot_cfg)
        assert hot["checker_fp"] != base["checker_fp"]
        assert hot["config_fp"] == base["config_fp"]
        assert hot["review_fp"] == base["review_fp"]

    def test_review_temperature_moves_only_review_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Repointed at a temperature-accepting review model, since the shipped
        # claude-opus-4-8 rejects temperature and would move nothing.
        over = {"review_model": "claude-sonnet-4-6"}
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg,
                         **over)
        hot_cfg = dict(base_cfg)
        hot_cfg["review_temperature"] = 0.5
        hot = self._fps(config_dir, bundle_minimal_dir, tmp_path, hot_cfg,
                        **over)
        assert hot["review_fp"] != base["review_fp"]
        assert hot["config_fp"] == base["config_fp"]
        assert hot["checker_fp"] == base["checker_fp"]

    def test_extractor_temperature_moves_only_config_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Both the extractor and the reviewer are repointed at a
        # temperature-accepting model, so that review_fp holding proves the
        # reviewer does NOT ride the extractor's temperature, rather than
        # merely proving the review model ignores it.
        over = {"extractor_model": "claude-sonnet-4-6",
                "review_model": "claude-sonnet-4-6"}
        base_cfg = dict(load_config_bundle_pipeline(config_dir))
        base_cfg["review_temperature"] = 0.0
        base = self._fps(config_dir, bundle_minimal_dir, tmp_path, base_cfg,
                         **over)
        hot_cfg = dict(base_cfg)
        hot_cfg["temperature"] = 0.5
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
        hot_cfg["temperature"] = 1.0
        hot_cfg["review_temperature"] = 1.0
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
        loop_cfg["temperature"] = 0.2
        loop_cfg["review_temperature"] = 0.7
        loop_cfg["checker_temperature"] = 0.4
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
