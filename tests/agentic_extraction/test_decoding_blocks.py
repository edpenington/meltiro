"""Per-role decoding blocks: what pipeline.yaml states, and what it must state.

Each role names one `<role>_decoding` block, an opaque mapping of decoding
parameter names to values. meltiro reads no key inside it: the block goes whole
to direktoro's `split_decoding_config`, which knows which name is a sampling
control and which is a thinking field, and the pair it returns goes to the
resolver that both the adapters and the stage fingerprints call. So the tests
here pin two things and nothing between them: that a block reaches its own
role's call intact, and that a block the model's endpoint would refuse stops
the run at startup, before a client exists and before any spend.

The other half is the output cap. Each ENABLED role states its own
`<role>_max_tokens`; there is no default, because the number bounds what a call
may spend and what it may answer within, and a cap meltiro invented would sit
in the run record looking exactly like one the operator chose. A role whose
stage is off states nothing and needs nothing.

Both guarantees are the LIBRARY's, not the command line's: the constructor
refuses the same configurations, so a consumer that builds an Orchestrator
directly is covered on the same terms as one that runs `meltiro extract`. The
CLI's own gate is the friendly wrapper around it — one stderr line and exit 1
instead of a traceback.

Offline throughout: no orchestrator here is run and no client is ever
constructed, so every guarantee below is one that holds before either exists.
"""

import shutil
from types import SimpleNamespace

import pytest

from direktoro import Thinking
from meltiro import cli, orchestrator
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig, check_one_field, run_checker_batch
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import (
    AgenticExtractionError, CheckerError, ConfigBundleError)
from meltiro.orchestrator import Orchestrator


def _args(**over):
    base = dict(
        max_tool_calls=None, max_checks_per_field=None, final_review=None,
        extractor_model=None, review_model=None, checker_model=None,
        diagnostics="standard", dry_run=True)
    base.update(over)
    return SimpleNamespace(**base)


def _pipeline(config_dir):
    return dict(load_config_bundle(config_dir).pipeline)


def _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg, **args_over):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(str(bundle_minimal_dir))
    return cli._build_orchestrator(
        config, bundle, tmp_path / "runs", loop_cfg, _args(**args_over))


def _resolution_spy(monkeypatch):
    """Record every model startup resolves, and resolve it for real.

    Both gates at once — the CLI's and the constructor's — so a stage this run
    does not use is shown to be skipped by both. Returns the list, which fills
    as the gates run; a model absent from it was never asked about, which is
    the only way to see a skip rather than a resolution that happened to pass.
    """
    asked = []
    real = cli.resolved_decoding_params

    def _spy(model, **kwargs):
        asked.append(model)
        return real(model, **kwargs)

    monkeypatch.setattr(cli, "resolved_decoding_params", _spy)
    monkeypatch.setattr(orchestrator, "resolved_decoding_params", _spy)
    # The positive control lives with the spy: a consumer asserts both that
    # its skipped model is absent AND that a model it does use is present, so
    # a renamed patch target cannot leave `asked` empty and both assertions
    # vacuously true.
    return asked


class TestABlockReachesItsOwnRoleIntact:
    """Whatever a block names arrives at that role's call, and at no other.

    meltiro does not interpret the names, so this is what makes a parameter
    direktoro gains usable from pipeline.yaml without a change on this side.
    """

    def test_the_thinking_fields_of_a_block_reach_direktoros_own_types(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A block mixes sampling controls and thinking fields freely, and the
        # split is direktoro's: the thinking half arrives as a `Thinking`, the
        # sampling half as the mapping the resolver takes.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["extractor_decoding"] = {
            "thinking_mode": "adaptive", "thinking_effort": "high"}
        loop_cfg["review_decoding"] = {"thinking_effort": "low"}
        loop_cfg["checker_decoding"] = {"thinking_mode": "disabled"}
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.thinking == Thinking(mode="adaptive", effort="high")
        assert orch.review_thinking == Thinking(effort="low")
        assert orch.checker_config.thinking == Thinking(mode="disabled")

    def test_a_block_naming_both_halves_splits_into_both(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_decoding"] = {
            "temperature": 0.2, "thinking_mode": "disabled"}
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.sampling == {"temperature": 0.2}
        assert orch.checker_config.thinking == Thinking(mode="disabled")

    def test_a_role_that_names_nothing_gets_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # No block on any role: no thinking spec anywhere, so nothing is sent
        # and every model's own default behaviour stands. Nothing is taken
        # from another role either.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["extractor_decoding"] = {"thinking_effort": "low"}
        loop_cfg.pop("review_decoding", None)
        loop_cfg.pop("checker_decoding", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_thinking is None
        assert orch.checker_config.thinking is None
        # One spelling of "specified nothing" on every role, so no consumer
        # has to treat an empty mapping and a None as the same state.
        assert orch.review_sampling is None
        assert orch.checker_config.sampling is None

    def test_a_checker_blocks_effort_reaches_the_params_it_is_judged_by(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Not the config object but the resolved params: the block is only
        # really wired if it reaches what the adapter sends and what
        # `checker_fp` folds in, which are the same dict by construction.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_decoding"] = {"thinking_effort": "low"}
        loop_cfg["checker_max_tokens"] = 4096
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)

        assert orch._decoding_params_meta()["checker"] == {
            "max_tokens": 4096, "output_config": {"effort": "low"}}
        # And into the checker's call-identity preimage, which is what makes
        # the effort part of `checker_fp` without a component of its own.
        assert '"effort":"low"' in orch.checker_config.call_identity()

    def test_two_spellings_of_one_temperature_share_a_checker_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # `0` and `0.0` are one intent written two ways, and the resolved value
        # folds into call identity byte-for-byte, so a bundle written either
        # way must fingerprint the same. direktoro normalises the float-valued
        # controls as it splits the block; pinned from this side because it is
        # meltiro's fingerprints that would otherwise split.
        def _fp(value, name):
            loop_cfg = _pipeline(config_dir)
            loop_cfg["checker_decoding"] = {"temperature": value}
            orch = _orch(config_dir, bundle_minimal_dir, tmp_path / name,
                         loop_cfg)
            return orch._compute_checker_fp()

        assert _fp(0, "int") == _fp(0.0, "float")


class TestABlockTheEndpointWouldRefuseStopsTheRun:
    """A block that cannot work fails at startup, on one stderr line, exit 1.

    The alternative is a 400 on a paid call — or, for the reviewer, a 400
    after a whole extraction and checker fan-out have already been billed.
    """

    def test_an_unknown_key_inside_a_block_names_the_block(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A misspelled control would otherwise be dropped in silence: absent
        # from the wire, absent from the fingerprint, and present in the
        # bundle a reader believes describes the run.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["extractor_decoding"] = {"temprature": 0.5}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "extractor_decoding" in err
        assert "temprature" in err

    def test_a_disabled_roles_block_is_checked_too(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The reviewer is off, so nothing in its block would ever be sent —
        # but a typo there is still a typo, and saying nothing about it leaves
        # a key that will do nothing when the stage is turned back on.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["final_review"] = False
        loop_cfg["review_decoding"] = {"temprature": 0.5}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "review_decoding" in capsys.readouterr().err

    def test_a_thinking_value_no_mode_exists_for_names_the_block(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A key the split accepts carrying a value nothing accepts. Caught as
        # the block is split, so the report names the block rather than the
        # role's model, which is not the thing that is wrong.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_decoding"] = {"thinking_mode": "sometimes"}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "checker_decoding" in err
        assert "sometimes" in err

    @pytest.mark.parametrize("role", ["extractor", "review", "checker"])
    def test_an_effort_the_model_does_not_have_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, role):
        # The effort ladder is per model, and this one does not carry this
        # level. Every enabled role is resolved against the registry at
        # startup on the same terms, so the reviewer's block is refused before
        # the extraction and the checker fan-out ahead of it have been billed
        # rather than at the reviewer's own first call.
        loop_cfg = _pipeline(config_dir)
        loop_cfg[f"{role}_model"] = "claude-sonnet-4-6"
        loop_cfg[f"{role}_decoding"] = {"thinking_effort": "xhigh"}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert f"{role} role" in err
        assert "xhigh" in err

    def test_sampling_alongside_active_thinking_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # This model takes a temperature, and takes adaptive thinking, and
        # returns a 400 for a request carrying both. Neither half is dropped:
        # which of the two a run needs is a scientific decision, so the
        # refusal names the pair and stops.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_decoding"] = {
            "temperature": 0.0, "thinking_mode": "adaptive"}
        loop_cfg["checker_max_tokens"] = 4096
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "checker role" in captured.err
        assert "temperature" in captured.err
        assert "adaptive" in captured.err
        # And the refusal is all a refused run says: a rate report for calls
        # that will never be made reads as a run about to start.
        assert "Pricing" not in captured.out

    def test_a_cap_a_thinking_call_cannot_answer_within_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The cap covers the reasoning and the answer together, so a thinking
        # role under a small cap spends the whole budget reasoning and returns
        # nothing usable — while still being billed. The floor the refusal
        # quotes is direktoro's own policy figure, not an endpoint fact, and
        # the message says so; asserting the number is what tells this test
        # apart from every other startup failure.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_decoding"] = {"thinking_mode": "adaptive"}
        loop_cfg["checker_max_tokens"] = 1024
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "1024" in err
        assert "2048" in err


class TestTheFlatKeysAreNotConfig:
    """Decoding lives in the blocks, and nowhere else.

    A key the allowlist does not know is refused at config load, so a bundle
    carrying one of the older flat spellings is told so by name rather than
    loaded with the value quietly doing nothing.
    """

    @pytest.mark.parametrize("key,value", [
        ("extractor_thinking_mode", "adaptive"),
        ("extractor_thinking_effort", "high"),
        ("review_thinking_mode", "adaptive"),
        ("review_thinking_effort", "high"),
        ("checker_thinking_mode", "adaptive"),
        ("checker_thinking_effort", "high"),
        ("temperature", 0.5),
        ("review_temperature", 0.5),
        ("checker_temperature", 0.5),
    ])
    def test_a_flat_decoding_key_is_unknown(self, config_dir, tmp_path, key,
                                            value):
        dest = tmp_path / "cfg"
        shutil.copytree(config_dir, dest)
        pipeline = dest / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8") + f"\n{key}: {value}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as exc:
            load_config_bundle(dest)
        assert key in str(exc.value)


class TestEveryEnabledRoleStatesItsCap:
    """`<role>_max_tokens` is required per enabled role, and only per enabled
    role: a stage that makes no calls needs no budget for them."""

    @pytest.mark.parametrize("role", ["extractor", "review", "checker"])
    def test_a_missing_cap_for_an_enabled_role_names_the_key(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, role):
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop(f"{role}_max_tokens", None)
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert f"{role}_max_tokens" in capsys.readouterr().err

    def test_a_disabled_reviewer_needs_no_cap(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["final_review"] = False
        loop_cfg.pop("review_max_tokens", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.review_max_tokens is None

    def test_a_disabled_checker_needs_no_cap(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["max_checks_per_field"] = 0
        loop_cfg.pop("checker_max_tokens", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.max_tokens is None

    def test_a_disabled_reviewers_call_is_never_resolved(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # The other half of needing no cap: a stage that makes no calls has no
        # call to check, so nothing asks the registry whether it could be made.
        # Resolving it anyway would demand a workable block and a cap for a
        # model this run never reaches.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["final_review"] = False
        loop_cfg["review_model"] = "claude-opus-4-7"   # not any other role's
        loop_cfg.pop("review_max_tokens", None)
        asked = _resolution_spy(monkeypatch)
        _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert loop_cfg["extractor_model"] in asked   # the spy saw the run
        assert "claude-opus-4-7" not in asked

    def test_a_disabled_checkers_call_is_never_resolved(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["max_checks_per_field"] = 0
        loop_cfg.pop("checker_max_tokens", None)
        asked = _resolution_spy(monkeypatch)
        _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert loop_cfg["extractor_model"] in asked   # the spy saw the run
        assert loop_cfg["checker_model"] not in asked

    def test_the_stated_cap_reaches_the_role(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["extractor_max_tokens"] = 16384
        loop_cfg["review_max_tokens"] = 8192
        loop_cfg["checker_max_tokens"] = 2048
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.extractor_max_tokens == 16384
        assert orch.review_max_tokens == 8192
        assert orch.checker_config.max_tokens == 2048

    @pytest.mark.parametrize("bad", [4096.0, "4096", True])
    def test_a_cap_that_is_not_a_plain_integer_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, bad):
        # Coercing would put a number in the run record that the bundle does
        # not say, and a bool is not a budget however it multiplies.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["extractor_max_tokens"] = bad
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "extractor_max_tokens" in capsys.readouterr().err


class TestTheLibraryRefusesOnItsOwn:
    """The pre-spend guarantees hold without the command line.

    A consumer that imports `Orchestrator` never reaches the CLI's gate, and a
    guarantee only the CLI enforces is not the library's. The constructor
    refuses three things per enabled role — no usable cap, a model id the
    registry does not know, and a block the model's endpoint would not accept
    — so the failure lands before a session directory exists or a token is
    billed, whichever way the run was started. One deliberate asymmetry
    remains: a RETIRED id resolves here (a past run's model must keep
    resolving), and refusing retired ids for NEW runs stays the CLI's gate.
    """

    def test_an_unknown_model_id_is_refused_naming_its_role(
            self, config_dir, bundle_minimal_dir, tmp_path):
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        with pytest.raises(AgenticExtractionError,
                           match=r"review role .*no-such-model"):
            Orchestrator(
                config, bundle, tmp_path / "runs",
                extractor_model="claude-opus-4-8",
                review_model="no-such-model",
                extractor_max_tokens=4096, review_max_tokens=4096,
                max_checks_per_field=0, final_review=True,
                api_key="x")

    def _kwargs(self, **over):
        base = dict(
            extractor_model="claude-opus-4-8",
            review_model="claude-opus-4-8",
            max_checks_per_field=0, final_review=False,
            extractor_max_tokens=4096, api_key="x")
        base.update(over)
        return base

    def _build(self, config_dir, bundle_minimal_dir, tmp_path, **over):
        return Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_minimal_dir),
            tmp_path / "runs", **self._kwargs(**over))

    def test_an_extractor_with_no_cap_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path):
        with pytest.raises(AgenticExtractionError,
                           match="extractor_max_tokens"):
            self._build(config_dir, bundle_minimal_dir, tmp_path,
                        extractor_max_tokens=None)

    def test_a_reviewer_that_will_run_needs_its_own_cap(
            self, config_dir, bundle_minimal_dir, tmp_path):
        with pytest.raises(AgenticExtractionError, match="review_max_tokens"):
            self._build(config_dir, bundle_minimal_dir, tmp_path,
                        final_review=True, review_max_tokens=None)

    def test_a_block_the_endpoint_would_refuse_is_refused_here_too(
            self, config_dir, bundle_minimal_dir, tmp_path):
        with pytest.raises(AgenticExtractionError, match="extractor role"):
            self._build(config_dir, bundle_minimal_dir, tmp_path,
                        extractor_model="claude-sonnet-4-6",
                        thinking=Thinking(effort="xhigh"))

    def test_a_starving_cap_is_refused_here_too(
            self, config_dir, bundle_minimal_dir, tmp_path):
        with pytest.raises(AgenticExtractionError, match="2048"):
            self._build(config_dir, bundle_minimal_dir, tmp_path,
                        extractor_max_tokens=1024,
                        thinking=Thinking(mode="adaptive"))

    def test_a_checker_call_with_no_cap_is_refused_before_it_is_made(self):
        # A bare `CheckerConfig` is built and filled in stages, so the
        # checker's own entry points hold the line for direct callers (the
        # Orchestrator demands the cap earlier, at construction):
        # `run_checker_batch` refuses the whole fan-out rather than degrading
        # every field to a false challenge. The stub adapter is load-bearing:
        # were the cap check ever reordered below adapter construction, this
        # test must fail on the assertion, not fall through to a real client.
        config = CheckerConfig(checker_model="claude-sonnet-4-6", api_key="x")
        stub = object()
        with pytest.raises(CheckerError, match="max_tokens"):
            check_one_field(system_message_blocks=[], user_message_blocks=[],
                            config=config, adapter=stub)
        with pytest.raises(CheckerError, match="max_tokens"):
            run_checker_batch(calls=[], config=config, adapter=stub)
