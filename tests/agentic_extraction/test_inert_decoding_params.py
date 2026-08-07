"""A decoding value the operator wrote that no model ever sends.

A model may carry a registry quirk that drops a parameter outright: a
`no_temperature` model is sent no `temperature` at all, whatever `pipeline.yaml`
says. The value is then inert, and inert invisibly, because the stage
fingerprints fold in the RESOLVED params — the exact dict the adapter sends. So
`temperature: 1.0` and `temperature: 0.0` produce the SAME `config_fp` for such
a model: two bundles differing in a visible, methodologically meaningful key
collide on every fingerprint, while a reader of the bundle believes the
extractor sampled at 1.0.

Nothing else surfaces it. `validate-bundle` reports the file legal, which it
is; the dry run prints the resolved params, which show the absence but not
which configured value produced it. So a run says so at startup, on stderr and
in `meta.warnings`.

Never fatal: a bundle names one model per role, and a key live for one role is
legitimately inert for another. What the operator is owed is to be told which
of the values they wrote will not be acted on, per role and per key.
"""

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator


# The fixture pipeline names claude-opus-4-8 for the extractor and the
# reviewer (both carry `no_temperature`) and claude-sonnet-4-6 for the checker
# (which accepts it). Its `temperature: 1.0` and `review_temperature: 0.0` are
# therefore inert and its `checker_temperature: 0.0` is live, which is the
# mixed case the warning has to get right in both directions.
NO_TEMPERATURE_MODEL = "claude-opus-4-8"
TEMPERATURE_MODEL = "claude-sonnet-4-6"


def _orch(config_dir, bundle_minimal_dir, tmp_path, **kwargs):
    loop = load_config_bundle(config_dir).pipeline
    kwargs.setdefault("extractor_model", loop["extractor_model"])
    kwargs.setdefault("review_model", loop["review_model"])
    kwargs.setdefault("temperature", float(loop["temperature"]))
    kwargs.setdefault("review_temperature", float(loop["review_temperature"]))
    kwargs.setdefault("checker_config", CheckerConfig(
        checker_model=loop["checker_model"],
        temperature=float(loop["checker_temperature"]), api_key="x"))
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_minimal_dir),
        tmp_path / "runs", api_key="x", **kwargs)


def _inert(orch):
    return [w for w in orch.session.meta["warnings"]
            if w.startswith("inert-decoding-param")]


class TestTheWarningFires:
    def test_a_dropped_temperature_warns_on_stderr_and_in_meta(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        orch.prepare_new_session()

        warnings = _inert(orch)
        assert any("extractor" in w for w in warnings)
        assert "WARNING: inert-decoding-param" in capsys.readouterr().err

    def test_the_message_names_the_role_the_model_and_the_value(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        orch.prepare_new_session()

        message = next(w for w in _inert(orch) if "extractor" in w)
        assert NO_TEMPERATURE_MODEL in message
        assert "temperature" in message
        # The value the operator wrote, so they can find it in their bundle.
        assert "1.0" in message

    def test_one_warning_per_affected_role(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The extractor and the reviewer both run a no_temperature model, and
        # each configured value is its own statement: silencing the second
        # because the first already fired would leave a key inert unremarked.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        orch.prepare_new_session()

        warnings = _inert(orch)
        assert len(warnings) == 2
        assert any("extractor" in w for w in warnings)
        assert any("review" in w for w in warnings)

    def test_a_live_value_is_not_warned_about(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The checker's model accepts temperature, so its configured value
        # does reach the wire and there is nothing to say. A warning that
        # fired on every configured key would be noise, not disclosure.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        orch.prepare_new_session()
        assert not any("checker" in w for w in _inert(orch))

    def test_a_model_that_accepts_the_value_everywhere_says_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(
            config_dir, bundle_minimal_dir, tmp_path,
            extractor_model=TEMPERATURE_MODEL,
            review_model=TEMPERATURE_MODEL)
        orch.prepare_new_session()
        assert _inert(orch) == []
        assert "inert-decoding-param" not in capsys.readouterr().err

    def test_it_is_not_fatal(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A bundle naming several models legitimately has inert keys for some
        # of them, so the run starts.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        orch.prepare_new_session()
        assert orch.session.meta["status"] == "in_progress"

    def test_a_disabled_stage_is_not_reported(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # With the reviewer off its model is never resolved, so nothing can be
        # said about what it would have sent.
        orch = _orch(
            config_dir, bundle_minimal_dir, tmp_path,
            final_review=False, review_model=None, max_checks_per_field=0)
        orch.prepare_new_session()
        assert not any("review" in w for w in _inert(orch))


class TestTheCollisionItDiscloses:
    def test_two_temperatures_collide_on_every_fingerprint(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The reason the warning exists, asserted directly: for a
        # no_temperature model the configured value reaches no fingerprint, so
        # a reader comparing two bundles on config_fp cannot see the
        # difference, and only the warning tells them one is there.
        hot = _orch(config_dir, bundle_minimal_dir, tmp_path / "hot",
                    temperature=1.0)
        hot.prepare_new_session()
        cold = _orch(config_dir, bundle_minimal_dir, tmp_path / "cold",
                     temperature=0.0)
        cold.prepare_new_session()

        assert hot.session.meta["config_fp"] == cold.session.meta["config_fp"]
        assert _inert(hot) and _inert(cold)


class TestWhereItIsSaidAndWhereItIsKept:
    def test_a_dry_run_says_it_but_banks_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A dry run is when an operator checks a bundle, so it must say it;
        # it banks no result, so it persists nothing, matching the dirty-tree
        # warning beside it.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, dry_run=True)
        orch.prepare_new_session()
        assert "WARNING: inert-decoding-param" in capsys.readouterr().err
        assert _inert(orch) == []

    def test_a_resumed_segment_says_it_too(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A resume may raise the tool-call cap and change nothing else, so the
        # operator seeing that segment's stderr gets the same caveat. The
        # persisted list does not grow: `add_warning` dedups an exact repeat.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        orch.prepare_new_session()
        before = _inert(orch)
        capsys.readouterr()

        resumed = _orch(config_dir, bundle_minimal_dir, tmp_path)
        resumed.resume_session(orch.session.session_dir)
        assert "WARNING: inert-decoding-param" in capsys.readouterr().err
        assert _inert(resumed) == before
