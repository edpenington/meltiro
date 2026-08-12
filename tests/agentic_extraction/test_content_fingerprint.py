"""Content fingerprint on ConfigBundle.

The bundle exposes a content-only identity computed at load time. Two things
matter for the consumer:

  - the component hashes it pins, `template_hash` and `reference_lists_hash`,
    are byte-identical to what the orchestrator folds into its run-time
    fingerprints (no second, drifting recipe), and
  - the instrument fingerprint is model-free, engine-free, and reproducible
    from the directory.

The unification test builds a real orchestrator (dry-run, no network) and
reconstructs its `config_fp` from the bundle's `reference_lists_hash`, proving
that value is exactly the one the run folds in.
"""

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import (
    instrument_fingerprint,
    instrument_structure_hash,
    config_fingerprint,
    structure_hash,
    reference_lists_hash,
    tool_set_hash,
)
from meltiro.checker import DEFAULT_CONTEXT_CHARS
from meltiro.orchestrator import DEFAULT_MAX_CHECKS_PER_FIELD, Orchestrator
from direktoro import (
    call_identity_fields, canonical_json, model_info, resolved_decoding_params)
from meltiro.template import load_template
from meltiro.tools import all_tool_definitions


def _dry_run_orchestrator(config, bundle, out_dir):
    loop = config.pipeline
    checker_config = CheckerConfig.from_env(
        model_override=loop["checker_model"])
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model=loop["extractor_model"],
        checker_config=checker_config,
        review_model=loop["review_model"],
        max_tool_calls=int(loop["max_tool_calls"]),
        max_checks_per_field=int(loop["max_checks_per_field"]),
        sampling={"temperature": float(loop["temperature"])},
        extractor_max_tokens=int(loop["extractor_max_tokens"]),
        review_max_tokens=int(loop["review_max_tokens"]),
        dry_run=True,
    )
    orch.prepare_new_session()
    return orch


class TestComponentsExposed:
    def test_all_four_fields_present(self, config_dir):
        cb = load_config_bundle(config_dir)
        assert cb.template_hash
        assert cb.reference_lists_hash
        assert cb.prompts_hash
        assert cb.instrument_fp.startswith("instrument_fp:")

    def test_template_hash_matches_loaded_template(self, config_dir):
        cb = load_config_bundle(config_dir)
        assert cb.template_hash == load_template(cb.template_path)[
            "template_hash"]

    def test_reference_lists_hash_matches_function(self, config_dir):
        cb = load_config_bundle(config_dir)
        assert cb.reference_lists_hash == reference_lists_hash(
            cb.reference_lists)

    def test_instrument_is_content_composite(self, config_dir):
        """The bundle's instrument_fp is the documented composite, computed
        with the same defaults the CLI applies for an absent pipeline key."""
        cb = load_config_bundle(config_dir)
        template = load_template(cb.template_path)
        # BOTH roles' catalogues, which is what the run hashes too. Building
        # the expectation from the extractor's list alone would let this pass
        # while the printed instrument_fp disagreed with every run.
        tools_hash = tool_set_hash(all_tool_definitions(template))
        pipeline = cb.pipeline or {}
        max_checks = int(pipeline.get("max_checks_per_field",
                                      DEFAULT_MAX_CHECKS_PER_FIELD))
        # No checker means no window, so the width is absent rather than a
        # number; with a checker an absent key means the default width.
        if max_checks == 0:
            context_chars = None
        else:
            context_chars = pipeline.get("checker_context_chars")
            if context_chars is None:
                context_chars = DEFAULT_CONTEXT_CHARS
        expected = instrument_fingerprint(
            prompts_hash=cb.prompts_hash,
            template_hash=cb.template_hash,
            tool_set_hash=tools_hash,
            structure_hash=instrument_structure_hash(
                max_checks,
                final_review=bool(pipeline.get("final_review", True)),
                check_reviewer_edits=bool(
                    pipeline.get("check_reviewer_edits", False)),
            ),
            reference_hash=cb.reference_lists_hash,
            checker_context_chars=context_chars,
            checker_context_fields=template.get("checker_context_fields"),
        )
        assert cb.instrument_fp == expected


class TestMatchesOrchestrator:
    """Both paths produce identical component values."""

    def test_template_hash_matches_recorded_run(
            self, tmp_path, config_dir, bundle_minimal_dir):
        cb = load_config_bundle(config_dir)
        orch = _dry_run_orchestrator(
            cb, load_bundle(bundle_minimal_dir), tmp_path / "runs")
        assert orch.session.meta["template_hash"] == cb.template_hash

    def test_reference_hash_is_the_one_folded_into_config_fp(
            self, tmp_path, config_dir, bundle_minimal_dir):
        # Rebuild the run's config_fp from the bundle's reference_lists_hash.
        # Equality proves the bundle exposes exactly the ref hash the run
        # folds in, not a parallel re-derivation that could drift.
        cb = load_config_bundle(config_dir)
        orch = _dry_run_orchestrator(
            cb, load_bundle(bundle_minimal_dir), tmp_path / "runs")
        meta = orch.session.meta
        loop = cb.pipeline
        ext_dec = resolved_decoding_params(
            loop["extractor_model"], sampling={"temperature": float(loop["temperature"])},
            max_tokens=int(loop["extractor_max_tokens"]))
        # The provider-call identity block is direktoro's: model + provider +
        # base_url + Route + wire-keyed resolved decoding params. The rebuild
        # composes exactly what the orchestrator does: this block plus the
        # bundle's content hashes, and nothing from the engine build (no
        # component of a stage fingerprint comes from meltiro's own source).
        info = model_info(loop["extractor_model"])
        call_identity = canonical_json(call_identity_fields(
            loop["extractor_model"], route=info.route, decoding_params=ext_dec))
        rebuilt = config_fingerprint(
            call_identity, meta["prompt_hash"], cb.template_hash,
            tool_set_hash=meta["tool_set_hash"],
            structure_hash=structure_hash(int(loop["max_checks_per_field"])),
            reference_hash=cb.reference_lists_hash,
        )
        assert rebuilt == meta["config_fp"]


class TestComponentSensitivity:
    def test_template_edit_moves_template_hash_and_family(
            self, tmp_path, config_dir):
        import shutil
        dst = tmp_path / "config"
        shutil.copytree(config_dir, dst)
        before = load_config_bundle(dst)
        tmpl = dst / "extraction_template.yaml"
        tmpl.write_text(
            tmpl.read_text(encoding="utf-8") + "\n# a trailing comment\n",
            encoding="utf-8")
        after = load_config_bundle(dst)
        assert after.template_hash != before.template_hash
        assert after.instrument_fp != \
            before.instrument_fp

    def test_alias_edit_moves_reference_hash_not_prompts_or_template(
            self, tmp_path, config_dir):
        import shutil

        import yaml
        dst = tmp_path / "config"
        shutil.copytree(config_dir, dst)
        before = load_config_bundle(dst)
        ref = dst / "reference" / "gauge_list.yaml"
        data = yaml.safe_load(ref.read_text(encoding="utf-8"))
        entries = next(v for v in data.values() if isinstance(v, list))
        entries[0]["aliases"] = ["a fresh synthetic alias"]
        ref.write_text(yaml.safe_dump(data), encoding="utf-8")
        after = load_config_bundle(dst)
        assert after.reference_lists_hash != before.reference_lists_hash
        # An alias is not rendered into any prompt, and it is not template
        # content, so those two hashes must not move.
        assert after.prompts_hash == before.prompts_hash
        assert after.template_hash == before.template_hash
        # The content family folds in reference content, so it does move.
        assert after.instrument_fp != \
            before.instrument_fp

    def test_prompt_edit_moves_prompts_hash(self, tmp_path, config_dir):
        import shutil
        dst = tmp_path / "config"
        shutil.copytree(config_dir, dst)
        before = load_config_bundle(dst)
        prompt = dst / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\nAn extra line.\n",
            encoding="utf-8")
        after = load_config_bundle(dst)
        assert after.prompts_hash != before.prompts_hash
        # A prompt edit does not touch what makes a value legal.
        assert after.template_hash == before.template_hash
        assert after.reference_lists_hash == before.reference_lists_hash
