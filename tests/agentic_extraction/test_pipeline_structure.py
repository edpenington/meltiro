"""Pipeline-structure toggles: run the extractor alone, extractor + checker,
extractor + reviewer, or all three, as a first-class, provenance-tracked
configuration (the structure axis of a model-by-structure comparison).

The checker stage is off exactly when max_checks_per_field is 0; the reviewer
stage is off when final_review is False. A disabled stage requires no model and
records a null stage fingerprint, and the structure moves config_fp so two runs
that differ only in structure never share a fingerprint (and resume refuses
across a structure change through the existing drift gate).

Every stage that would touch the network is stubbed (a fake extractor loop, a
stubbed checker fan-out, a fake review adapter), so these run offline. The fake
extractor loop still makes a REAL tool dispatch, so the inline checker fires at
the same seam it would live: after the dispatcher applies a field.
"""

import shutil
from types import SimpleNamespace

import pytest
import yaml

import meltiro.cli as cli
import meltiro.orchestrator as orch_mod
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import AgenticExtractionError, ResumeRefused
from meltiro.orchestrator import Orchestrator
from direktoro import NormalisedResponse, NormalisedUsage, model_info


# Every stage's key variable is present for this module: these tests
# reach the orchestrator's pre-spend key preflight, and the provider
# calls behind it are stubbed.
pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-7"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-7"

# The four points of the structure axis: (max_checks_per_field, final_review).
STRUCTURES = {
    "extractor_only": (0, False),
    "plus_checker": (2, False),
    "plus_review": (0, True),
    "all_three": (2, True),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir, *, max_checks_per_field, final_review,
          dry_run=False, checker_model=CHECKER, review_model=REVIEWER):
    """A real Orchestrator over the shipped config + synthetic bundle, wired
    for the requested structure. A disabled stage may pass model=None."""
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(
            max_tokens=4096, checker_model=checker_model),
        review_model=review_model,
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        extractor_max_tokens=4096,
        review_max_tokens=4096, dry_run=dry_run,
    )


class _FakeAdapter:
    """Provider adapter that returns a canned NormalisedResponse, so
    _final_review runs fully offline."""

    def __init__(self, response):
        self._response = response

    def create_message(self, **kwargs):
        return self._response


# The shipped template's one REQUIRED quality-check variable. `mark_complete`
# takes the caller's quality check as a required argument, for the reviewer as
# for the extractor, so a scripted conclusion has to carry one.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


def _review_response():
    """A reviewer turn that calls only mark_complete: no edits, so
    _final_review returns review_clean with no post-review checker pass.

    The call is dispatched (that is how the reviewer's own quality check gets
    recorded) but it is not a mutation, so it still counts as no edit.
    """
    return NormalisedResponse(
        content=[SimpleNamespace(type="tool_use", id="rc",
                                 name="mark_complete",
                                 input={"summary": "looks good",
                                        "quality_check":
                                            dict(QUALITY_CHECK)})],
        usage=NormalisedUsage(),
        resolved_model=REVIEWER, provider="anthropic", base_url=None,
        raw_request={"model": REVIEWER}, raw_response={},
        wire_request={"model": REVIEWER}, decoding_params={"max_tokens": 1024},
    )


# A study field plus a verbatim quote from the synthetic bundle, so a real
# dispatch validates and applies.
_A_FIELD = "title"
_A_QUOTE = "A synthetic study of baseline CRT-HD scores"


def _mark_complete_extractor(orch):
    def _fake():
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"
    return _fake


def _writing_extractor(orch):
    """A fake extractor loop that makes one real, validating tool dispatch and
    runs the inline checker over it, exactly as the live loop does.

    ONE dispatch is the point (the assertions below read `tool_calls[0]`), so
    the initial-check ordering gate is opened directly rather than by scripting
    the `record_initial_check` call that opens it: that would be a second
    dispatch in a helper whose whole job is to make exactly one.
    """
    def _fake():
        orch.extraction_record.initial_check_recorded = True
        res = orch.dispatcher.dispatch("update_study", {"study": {
            _A_FIELD: {"value": "A synthetic study",
                       "evidence": f"<q>{_A_QUOTE}</q>"},
        }})
        assert res["applied_fields"] == [f"study.{_A_FIELD}"], res
        orch._check_applied_fields(res, stage="extractor")
        # The real loop logs the dispatch AFTER the merge, which is what puts
        # the verdicts on the durable record; mirror that order here.
        orch.session.append_event({
            "event": "tool_call_applied", "turn_id": 1,
            "tool": "update_study", "args": {}, "result": res})
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"
    return _fake


def _stub_fanout(monkeypatch):
    """Replace the parallel checker fan-out with a clean verdict per call, so
    the trigger and the merge run for real with no provider behind them."""
    def _fake(*, calls, config, on_complete=None, api_logger=None, **kw):
        return {c["field_path"]: {
            "verdict": "ok", "rationale": "supported", "notes": None,
            "error_origin": False, "input_tokens": 1, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.0,
        } for c in calls}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _fake)


def _stub_offline(orch, *, review_on, monkeypatch):
    """Replace the network-touching leaves so a full orch.run() stays offline:
    the extractor loop writes one field and runs the inline checker over it,
    the fan-out is stubbed clean, and (when the reviewer is on) the review
    adapter is a fake."""
    orch._extractor_loop = _writing_extractor(orch)
    _stub_fanout(monkeypatch)
    if review_on:
        resp = _review_response()
        orch._adapter_for_role = lambda role: _FakeAdapter(resp)


def _write_config(src_config, dst, *, drop_models=(), extra=None):
    """Copy the shipped config bundle to `dst`, then drop the named model keys
    and/or overlay `extra` onto pipeline.yaml. Prompt files and reference lists
    are copied intact (load_config_bundle still requires them)."""
    shutil.copytree(src_config, dst)
    p = dst / "pipeline.yaml"
    cfg = yaml.safe_load(p.read_text())
    for k in drop_models:
        cfg.pop(k, None)
    if extra:
        cfg.update(extra)
    p.write_text(yaml.safe_dump(cfg))
    return dst


# ---------------------------------------------------------------------------
# run.json: structure object, null fingerprints, fingerprint moves
# ---------------------------------------------------------------------------

def test_structure_recorded_with_null_fps_and_distinct_config_fps(
        config_dir, bundle_minimal_dir, tmp_path):
    metas = {}
    for name, (rounds, review) in STRUCTURES.items():
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / name,
                     max_checks_per_field=rounds, final_review=review,
                     checker_model=(CHECKER if rounds > 0 else None),
                     review_model=(REVIEWER if review else None))
        orch.prepare_new_session()
        metas[name] = dict(orch.session.meta)

    # The structure object is recorded verbatim.
    assert metas["extractor_only"]["structure"] == {
        "checker": False, "review": False, "max_checks_per_field": 0,
        "check_reviewer_edits": False}
    assert metas["plus_checker"]["structure"] == {
        "checker": True, "review": False, "max_checks_per_field": 2,
        "check_reviewer_edits": False}
    assert metas["plus_review"]["structure"] == {
        "checker": False, "review": True, "max_checks_per_field": 0,
        "check_reviewer_edits": False}
    assert metas["all_three"]["structure"] == {
        "checker": True, "review": True, "max_checks_per_field": 2,
        "check_reviewer_edits": False}

    # A disabled stage records a null fingerprint; an enabled one does not.
    assert metas["extractor_only"]["checker_fp"] is None
    assert metas["extractor_only"]["review_fp"] is None
    assert metas["plus_checker"]["checker_fp"] is not None
    assert metas["plus_checker"]["review_fp"] is None
    assert metas["plus_review"]["checker_fp"] is None
    assert metas["plus_review"]["review_fp"] is not None
    assert metas["all_three"]["checker_fp"] is not None
    assert metas["all_three"]["review_fp"] is not None

    # The structure moves config_fp: all four are distinct.
    config_fps = {m["config_fp"] for m in metas.values()}
    assert len(config_fps) == 4


# ---------------------------------------------------------------------------
# All four structures fire the correct stages end-to-end (offline run())
# ---------------------------------------------------------------------------

def test_run_fires_correct_stages_per_structure(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    for name, (checks, review) in STRUCTURES.items():
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / name,
                     max_checks_per_field=checks, final_review=review,
                     checker_model=(CHECKER if checks > 0 else None),
                     review_model=(REVIEWER if review else None))
        orch.prepare_new_session()
        _stub_offline(orch, review_on=review, monkeypatch=monkeypatch)

        status = orch.run()
        assert status == "complete", name

        meta = orch.session.meta
        events = [e.get("event") for e in orch.session.read_events()]
        tool_calls = [e for e in orch.session.read_events()
                      if e.get("event", "").startswith("tool_call")]

        # Checker: fires iff on. The extractor dispatched one applying write in
        # every structure, so a zero tally means the checker genuinely did not
        # run rather than that nothing was written.
        assert len(tool_calls) == 1, name
        if checks > 0:
            assert meta["checker_calls_run"] == 1, name
            assert "_checker_verdicts" in tool_calls[0]["result"], name
            assert meta["checker_diagnostics"]["fields_checked"] == 1, name
        else:
            assert meta["checker_calls_run"] == 0, name
            assert "_checker_verdicts" not in tool_calls[0]["result"], name
            assert "checker_diagnostics" not in meta, name

        # Reviewer: fires iff on.
        if review:
            assert "final_review_response" in events, name
        else:
            assert "final_review_response" not in events, name


# ---------------------------------------------------------------------------
# Disabled stages: run.json and run_log null the model + fp (never the
# pipeline default of a stage that never ran)
# ---------------------------------------------------------------------------

def test_disabled_stage_models_null_in_meta_and_run_log(
        config_dir, bundle_minimal_dir, tmp_path):
    from meltiro.run_log import load_log

    out = tmp_path / "runs"
    # The CLI passes pipeline.yaml's checker/review models through even when
    # the stages are disabled; reproduce that by supplying non-None models but
    # turning both stages off.
    orch = _orch(config_dir, bundle_minimal_dir, out,
                 max_checks_per_field=0, final_review=False,
                 checker_model=CHECKER, review_model=REVIEWER)
    orch.prepare_new_session()
    orch._extractor_loop = _mark_complete_extractor(orch)
    assert orch.run() == "complete"

    meta = orch.session.meta
    assert meta["checker_model"] is None
    assert meta["review_model"] is None
    assert meta["checker_fp"] is None
    assert meta["review_fp"] is None
    assert meta["extractor_model"] == EXTRACTOR  # the stage that ran

    entry = load_log(out)[-1]
    # run_log is built from meta, so it aligns: no misleading model or fp.
    assert entry["checker_model"] is None
    assert entry["review_model"] is None
    assert entry["checker_fp"] is None
    assert entry["review_fp"] is None
    assert entry["model"] == EXTRACTOR


def test_enabled_stage_models_recorded_in_meta_and_run_log(
        config_dir, bundle_minimal_dir, tmp_path):
    from meltiro.run_log import load_log

    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out,
                 max_checks_per_field=2, final_review=True,
                 checker_model=CHECKER, review_model=REVIEWER)
    orch.prepare_new_session()
    orch._extractor_loop = _mark_complete_extractor(orch)
    orch._adapter_for_role = lambda role: _FakeAdapter(_review_response())
    assert orch.run() == "complete"

    meta = orch.session.meta
    assert meta["checker_model"] == CHECKER
    assert meta["review_model"] == REVIEWER

    entry = load_log(out)[-1]
    assert entry["checker_model"] == CHECKER
    assert entry["review_model"] == REVIEWER
    assert entry["checker_fp"] is not None
    assert entry["review_fp"] is not None


# ---------------------------------------------------------------------------
# max_checks_per_field: 0 makes the checker stage not exist (deliberate)
# ---------------------------------------------------------------------------

def test_max_checks_per_field_zero_skips_checker_entirely(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=False,
                 checker_model=None, review_model=None)
    orch.prepare_new_session()
    orch._extractor_loop = _writing_extractor(orch)
    fired = []
    monkeypatch.setattr(
        orch_mod, "run_checker_batch",
        lambda **kw: fired.append("batch") or {})

    assert orch.run() == "complete"
    # The fan-out was never reached, and nothing was recorded as if it had been.
    assert fired == []
    assert orch.session.meta["checker_calls_run"] == 0
    assert orch.session.meta["checker_fp"] is None
    assert orch.session.meta["structure"]["checker"] is False
    assert "checker_diagnostics" not in orch.session.meta


# ---------------------------------------------------------------------------
# final_review: false makes the reviewer stage not exist
# ---------------------------------------------------------------------------

def test_final_review_false_skips_reviewer_entirely(
        config_dir, bundle_minimal_dir, tmp_path):
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=False,
                 checker_model=None, review_model=None)
    orch.prepare_new_session()
    orch._extractor_loop = _mark_complete_extractor(orch)
    fired = []
    orch._final_review = lambda: (fired.append("review"), "review_clean")[1]

    assert orch.run() == "complete"
    assert fired == []
    assert orch.session.meta["review_fp"] is None
    assert orch.session.meta["structure"]["review"] is False


# ---------------------------------------------------------------------------
# A reviewer that is ON but unusable fails loudly (no silent skip)
# ---------------------------------------------------------------------------

def test_extractor_without_key_raises_naming_the_env_var(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The variable the extractor model's key lives in is unset, so no adapter
    # can be built. The stage refuses by name, before any call is attempted,
    # so an operator is told which variable to set rather than reading a 401.
    env = model_info(EXTRACTOR).api_key_env
    monkeypatch.delenv(env, raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=False)
    orch.prepare_new_session()
    with pytest.raises(AgenticExtractionError) as excinfo:
        orch._extractor_loop()
    assert env in str(excinfo.value)


def test_review_on_without_key_raises_not_skips(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The reviewer's key variable is unset, so the review adapter resolves to
    # None, which raises and names the variable to set. Returning review_clean
    # instead would report a review that never happened. Called directly, past
    # the preflight, because that guard is what holds when the stage is
    # entered.
    env = model_info(REVIEWER).api_key_env
    monkeypatch.delenv(env, raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=True)
    orch.prepare_new_session()
    with pytest.raises(AgenticExtractionError) as excinfo:
        orch._final_review()
    assert env in str(excinfo.value)


def test_run_review_on_without_key_finalises_error(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The loud failure propagates through run() to an "error" status: a run
    # that asked for a review must not ship an un-reviewed extraction. The
    # reviewer is on a provider of its own here, so the run is short of exactly
    # one key and it is the reviewer's.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=True,
                 review_model="gpt-5.6-sol")
    orch.prepare_new_session()
    orch._extractor_loop = _mark_complete_extractor(orch)

    assert orch.run() == "error"
    err = next(e for e in orch.session.read_events()
               if e.get("event") == "error")
    assert "OPENAI_API_KEY" in err["message"]


# ---------------------------------------------------------------------------
# Pre-spend key preflight: every enabled stage's key is verified up front
# ---------------------------------------------------------------------------

def test_preflight_missing_extractor_key_raises(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    monkeypatch.delenv(model_info(EXTRACTOR).api_key_env, raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=False)
    orch.prepare_new_session()
    with pytest.raises(AgenticExtractionError) as excinfo:
        orch._preflight_keys()
    msg = str(excinfo.value)
    assert model_info(EXTRACTOR).api_key_env in msg
    assert "extractor" in msg


def test_preflight_missing_review_key_raises(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The question is per model, not per run: every other provider's key is
    # present, and a GPT reviewer still needs OPENAI_API_KEY, because that is
    # the variable `build_adapter` will read for that model. Only the review
    # stage is flagged, and the message names the variable to set.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=0, final_review=True,
                 review_model="gpt-5.6-sol")
    orch.prepare_new_session()
    with pytest.raises(AgenticExtractionError) as excinfo:
        orch._preflight_keys()
    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    assert "review" in msg


def test_preflight_missing_checker_key_rejected_before_extractor_spends(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # A missing checker key discovered mid-round, after the extractor has
    # fully spent, degrades EVERY field to a challenge, so the preflight
    # rejects it before the extractor loop is ever entered. Extractor key
    # present; checker (routed GLM) key unset. The checker's GLM routes
    # through OpenRouter, so the missing key it names is OPENROUTER_API_KEY.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=2, final_review=False,
                 checker_model="z-ai/glm-5v-turbo")
    orch.prepare_new_session()

    spent = []
    orch._extractor_loop = lambda: (spent.append(True),
                                    "mark_complete_validated")[1]

    assert orch.run() == "error"
    assert spent == [], "extractor must not spend when a checker key is missing"
    err = next(e for e in orch.session.read_events()
               if e.get("event") == "error")
    assert "OPENROUTER_API_KEY" in err["message"]
    assert "checker" in err["message"]


def test_preflight_skipped_for_dry_run_without_keys(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # Dry-run needs no keys for any stage: it never enters the live loop, so no
    # preflight runs and no session is created. dry_run_report renders the
    # instrument (all three stage fingerprints present) with not one key
    # variable set.
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                 max_checks_per_field=2, final_review=True,
                 dry_run=True)
    report = orch.dry_run_report()
    assert orch.session is None
    fp = report["fingerprints"]
    assert fp["config_fp"] and fp["checker_fp"] and fp["review_fp"]
    # Calling run() on a dry-run orchestrator is a wiring error and fails loud.
    with pytest.raises(AgenticExtractionError):
        orch.run()


# ---------------------------------------------------------------------------
# Resume refuses across a structure change (via the config_fp drift gate)
# ---------------------------------------------------------------------------

def test_resume_refused_when_reviewer_toggled(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    started = _orch(config_dir, bundle_minimal_dir, out,
                    max_checks_per_field=2, final_review=True)
    started.prepare_new_session()  # leaves the session in_progress
    session_dir = started.session.session_dir

    resumed = _orch(config_dir, bundle_minimal_dir, out,
                    max_checks_per_field=2, final_review=False, review_model=None)
    with pytest.raises(ResumeRefused):
        resumed.resume_session(session_dir)


def test_resume_refused_when_checker_toggled(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    started = _orch(config_dir, bundle_minimal_dir, out,
                    max_checks_per_field=2, final_review=True)
    started.prepare_new_session()
    session_dir = started.session.session_dir

    resumed = _orch(config_dir, bundle_minimal_dir, out,
                    max_checks_per_field=0, final_review=True, checker_model=None)
    with pytest.raises(ResumeRefused):
        resumed.resume_session(session_dir)


# ---------------------------------------------------------------------------
# CLI model-requirement matrix + dry-run for all four structures
# ---------------------------------------------------------------------------

def _dry_run_args(cfg, bundle_dir, out_dir, *extra):
    return cli._parse_args([
        "extract", "--config", str(cfg), "--paper", str(bundle_dir),
        "--out", str(out_dir), "--dry-run", *extra])


def test_cli_dry_run_all_structures_without_disabled_stage_models(
        config_dir, bundle_minimal_dir, tmp_path):
    # A disabled stage's model is dropped from pipeline.yaml entirely: the run
    # must still succeed, proving it is genuinely not required.
    matrix = {
        "extractor_only": (dict(max_checks_per_field=0, final_review=False),
                           ("checker_model", "review_model")),
        "plus_checker": (dict(max_checks_per_field=2, final_review=False),
                         ("review_model",)),
        "plus_review": (dict(max_checks_per_field=0, final_review=True),
                        ("checker_model",)),
        "all_three": (dict(max_checks_per_field=2, final_review=True), ()),
    }
    for name, (extra, drop) in matrix.items():
        cfg = _write_config(config_dir, tmp_path / f"cfg_{name}",
                            drop_models=drop, extra=extra)
        args = _dry_run_args(cfg, bundle_minimal_dir, tmp_path / f"out_{name}")
        assert cli._cmd_extract(args) == 0, name


def test_cli_missing_required_review_model_fails(
        config_dir, bundle_minimal_dir, tmp_path, capsys):
    # Reviewer ON but review_model absent: fail loudly before any spend.
    cfg = _write_config(config_dir, tmp_path / "cfg",
                        drop_models=("review_model",),
                        extra=dict(max_checks_per_field=2, final_review=True))
    args = _dry_run_args(cfg, bundle_minimal_dir, tmp_path / "out")
    with pytest.raises(SystemExit) as excinfo:
        cli._cmd_extract(args)
    assert excinfo.value.code == 1
    assert "review_model" in capsys.readouterr().err


def test_cli_missing_required_checker_model_fails(
        config_dir, bundle_minimal_dir, tmp_path, capsys):
    # Checker ON but checker_model absent: fail loudly.
    cfg = _write_config(config_dir, tmp_path / "cfg",
                        drop_models=("checker_model",),
                        extra=dict(max_checks_per_field=2, final_review=False))
    args = _dry_run_args(cfg, bundle_minimal_dir, tmp_path / "out")
    with pytest.raises(SystemExit) as excinfo:
        cli._cmd_extract(args)
    assert excinfo.value.code == 1
    assert "checker_model" in capsys.readouterr().err


def test_cli_disabled_stage_unknown_model_not_validated(
        config_dir, bundle_minimal_dir, tmp_path):
    # A bogus checker_model with the checker OFF must not trip the known-model
    # check: only required (enabled-stage) models are validated.
    cfg = _write_config(config_dir, tmp_path / "cfg",
                        extra=dict(max_checks_per_field=0, final_review=True,
                                   checker_model="totally-not-a-real-model"))
    args = _dry_run_args(cfg, bundle_minimal_dir, tmp_path / "out")
    assert cli._cmd_extract(args) == 0


def test_cli_no_final_review_flag_overrides_pipeline(
        config_dir, bundle_minimal_dir, tmp_path):
    # review_model is dropped; the reviewer defaults on, so without the flag the
    # run fails for a missing review_model. --no-final-review disables the
    # reviewer, and the same config then dry-runs clean with no review model.
    cfg = _write_config(config_dir, tmp_path / "cfg",
                        drop_models=("review_model",),
                        extra=dict(max_checks_per_field=2))
    with pytest.raises(SystemExit):
        cli._cmd_extract(_dry_run_args(cfg, bundle_minimal_dir,
                                       tmp_path / "out_fail"))
    ok = _dry_run_args(cfg, bundle_minimal_dir, tmp_path / "out_ok",
                       "--no-final-review")
    assert cli._cmd_extract(ok) == 0
