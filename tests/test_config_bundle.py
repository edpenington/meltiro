"""The config bundle contract: a review is a directory, and it either loads
whole or fails whole.

`load_config_bundle` reads one directory into a frozen `ConfigBundle`: the
extraction template, the four prompt files, `pipeline.yaml`, and the optional
`reference/` lists. Nothing about a review is compiled into the engine, which
is what makes the bundle the unit of provenance. The tests here pin three
properties of that load.

Everything is validated at LOAD time, not at use time. An unknown pipeline key,
a duplicated mapping key, a prompt citing a partial the bundle does not
provide, a `{reference:NAME}` naming a list that does not exist, even one
nested inside a partial: each fails while the bundle is being read. A bundle
that loads and then fails on turn forty has already spent the run's budget to
discover a typo, so the load is where a config author is told. Where several
problems can be found at once they are collected into one error rather than
reported one per run.

A rejection names the key and lists the ones that exist. A knob this engine
does not have does NOT degrade to being silently ignored, because silently
ignoring it means the author believes a setting is in force that is not:
`quote_fuzz_tolerance` is rejected outright rather than pinned to zero, since
exact-substring quote matching is part of the measurement instrument rather
than a tunable, and `post_mark_complete_cap_bonus` is rejected rather than
read as extra extractor headroom that nothing would grant.

The bundle is the ONLY source for what it owns. Decoding parameters are
per-role bundle keys (`checker_decoding`, `review_decoding`) rather than
environment variables, so a run's configuration is readable from the
directory that produced it. The tool-call cap is the deliberate exception in
the other direction: its placeholders are refused in prompts, because a cap
rendered into a prompt would fold an operational budget into prompt_hash and
so into config_fp, and that breaks the "hit the cap, resume with a raised cap"
recovery.
"""

import dataclasses
import shutil

import pytest

from meltiro.checker_prompts import render_checker_user_template
from meltiro.config_bundle import ConfigBundle, load_config_bundle
from meltiro.errors import ConfigBundleError
from meltiro.prompt_partials import stage_predicates


@pytest.fixture
def good_config(tmp_path, config_dir):
    """A writable copy of the config bundle the suite runs against."""
    dst = tmp_path / "config"
    shutil.copytree(config_dir, dst)
    return dst


class TestHappyPath:
    def test_loads_shipped_config(self, config_dir):
        cb = load_config_bundle(config_dir)
        assert isinstance(cb, ConfigBundle)
        assert cb.template_path.name == "extraction_template.yaml"
        assert cb.extractor_system_path.name == "extractor_system.md"
        assert cb.review_system_path.name == "review_system.md"
        assert cb.checker_system_path.name == "checker_system.md"
        # pipeline.yaml parsed to a mapping.
        assert isinstance(cb.pipeline, dict)
        assert cb.pipeline.get("max_tool_calls") == 100

    def test_loads_reference_lists(self, config_dir):
        cb = load_config_bundle(config_dir)
        # The reference/ directory is present and holds the gauge list,
        # keyed by file stem.
        assert cb.reference_dir is not None
        assert "gauge_list" in cb.reference_lists
        assert len(cb.reference_lists["gauge_list"]) > 0

    def test_frozen(self, config_dir):
        cb = load_config_bundle(config_dir)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cb.root = "elsewhere"


class TestMissingFiles:
    def test_missing_directory(self, tmp_path):
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(tmp_path / "nope")
        assert "does not exist" in str(excinfo.value)

    def test_missing_one_prompt_lists_it(self, good_config):
        (good_config / "prompts" / "checker_system.md").unlink()
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        problems = excinfo.value.problems
        assert any("checker_system.md" in p for p in problems)

    def test_lists_every_missing_file(self, good_config):
        (good_config / "pipeline.yaml").unlink()
        (good_config / "extraction_template.yaml").unlink()
        (good_config / "prompts" / "review_system.md").unlink()
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        problems = excinfo.value.problems
        assert len(problems) == 3
        joined = " ".join(problems)
        assert "pipeline.yaml" in joined
        assert "extraction_template.yaml" in joined
        assert "review_system.md" in joined

    def test_pipeline_not_a_mapping(self, good_config):
        (good_config / "pipeline.yaml").write_text(
            "- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "must parse to a mapping" in str(excinfo.value)


class TestPipelineKeys:
    def test_shipped_config_pipeline_keys_all_known(self, config_dir):
        # Every key in the shipped pipeline.yaml must be on the allowlist,
        # otherwise the shipped config would not load.
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        cb = load_config_bundle(config_dir)
        unknown = set(cb.pipeline) - KNOWN_PIPELINE_KEYS
        assert unknown == set()

    def test_unknown_key_rejected(self, good_config):
        # A typo like `checker_decodin:` would be silently ignored without the
        # allowlist; instead it must fail loudly.
        pipeline = good_config / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8") + "\nchecker_decodin: {}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "unknown key" in msg
        assert "checker_decodin" in msg
        # The error also lists the known keys.
        assert "checker_decoding" in msg

    def test_checker_decoding_is_allowed(self, good_config):
        # A role's decoding block is a config-bundle knob, NOT an environment
        # variable, so it must be on the allowlist rather than rejected as
        # unknown. Declared in the shipped config, so it is asserted there
        # rather than appended: appending would duplicate the key, which
        # strict_load rejects.
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        assert "checker_decoding" in KNOWN_PIPELINE_KEYS
        cb = load_config_bundle(good_config)  # must not raise
        assert cb.pipeline["checker_decoding"] == {"temperature": 0.0}

    def test_review_decoding_is_allowed(self, good_config):
        # The reviewer's decoding is independently tunable: the key must be
        # on the allowlist rather than rejected as unknown. Declared in the
        # shipped config, so assert it there.
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        assert "review_decoding" in KNOWN_PIPELINE_KEYS
        cb = load_config_bundle(good_config)  # must not raise
        assert cb.pipeline["review_decoding"] == {"temperature": 0.0}

    def test_review_max_tokens_is_allowed(self, good_config):
        # Declared in the shipped config and wired into the CLI / review_fp;
        # must not be rejected. Read off the allowlist and the loaded bundle,
        # never off the raw YAML text, where a mention inside a comment would
        # satisfy a substring check while the key stayed unknown.
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        assert "review_max_tokens" in KNOWN_PIPELINE_KEYS
        cb = load_config_bundle(good_config)  # must not raise
        assert cb.pipeline["review_max_tokens"]

    def test_quote_fuzz_tolerance_key_is_now_unknown(self, good_config):
        # There is no such knob: strict exact-substring quote matching is part
        # of the measurement instrument, so the setting is absent rather than
        # pinned to zero. A config still carrying it is rejected as an unknown
        # key, like any other stale key.
        pipeline = good_config / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8") + "\nquote_fuzz_tolerance: 0\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "unknown key" in msg
        assert "quote_fuzz_tolerance" in msg

    def test_post_mark_complete_cap_bonus_is_unknown(self, good_config):
        # There is no post-completion tool-call bonus to grant: the checker
        # runs inside the tool call and its challenges ride back in the same
        # tool result, so `mark_complete` ends the extractor loop and any
        # extra headroom after it could never be spent. `max_tool_calls:` is
        # the whole of the extractor's budget. A config asking for the bonus is
        # rejected rather than left believing the extractor has room it has not.
        pipeline = good_config / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8")
            + "\npost_mark_complete_cap_bonus: 50\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "unknown key" in msg
        assert "post_mark_complete_cap_bonus" in msg
        # The known set rides in the message, so `max_tool_calls` (the budget
        # to raise instead) is right there to read.
        assert "max_tool_calls" in msg

    def test_the_cap_bonus_key_is_off_the_known_allowlist(self):
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        assert "post_mark_complete_cap_bonus" not in KNOWN_PIPELINE_KEYS


class TestDuplicateKeys:
    """A duplicated mapping key fails at the parse, rather than collapsing to
    whichever value comes last. Collapsing silently gives the author a config
    that reads one way and behaves another. The pipeline.yaml and
    reference-list parse sites route through the strict loader; the template
    site is covered in test_yaml_strict.py."""

    def test_duplicate_pipeline_key_rejected_naming_it(self, good_config):
        # The shipped pipeline.yaml already sets max_checks_per_field; a second
        # entry must not silently win. It arrives as a ConfigBundleError, the
        # one type this loader raises, carrying the parser's own diagnostic —
        # the duplicated key and both line numbers — inside it.
        pipeline = good_config / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8") + "\nmax_checks_per_field: 0\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "found duplicate key" in msg
        assert "max_checks_per_field" in msg

    def test_duplicate_reference_key_rejected_naming_it(self, good_config):
        # The reference-list loader wraps a YAMLError in a ConfigBundleError,
        # so the duplicate surfaces there (naming the file and the key).
        (good_config / "reference" / "dup.yaml").write_text(
            "entries:\n  - tool_name: A\nentries:\n  - tool_name: B\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "dup.yaml" in msg
        assert "found duplicate key" in msg
        assert "entries" in msg


class TestReferences:
    def test_missing_reference_for_named_source_rejected(self, good_config):
        # The shipped template names canonical_reference: gauge_list; dropping
        # the reference/ dir must fail the cross-validation loudly.
        import shutil as _sh
        _sh.rmtree(good_config / "reference")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "gauge_list" in str(excinfo.value)

    def test_unknown_canonical_reference_rejected(self, good_config):
        # Point a field at a reference list the bundle does not provide.
        tmpl = good_config / "extraction_template.yaml"
        text = tmpl.read_text(encoding="utf-8")
        text = text.replace("canonical_reference: gauge_list",
                            "canonical_reference: does_not_exist", 1)
        tmpl.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "does_not_exist" in str(excinfo.value)

    def test_malformed_reference_file_rejected(self, good_config):
        # An empty *.yaml in reference/ is not a valid list.
        (good_config / "reference" / "broken.yaml").write_text(
            "", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "broken.yaml" in str(excinfo.value)

    def test_reference_aliases_load_through_bundle(self, good_config):
        # A reference file may carry an optional `aliases:` key per entry.
        # Adding one to the first gauge entry loads fine (curation of the
        # shipped list is the owner's job; the loader just supports the key).
        import yaml as _yaml
        ref = good_config / "reference" / "gauge_list.yaml"
        data = _yaml.safe_load(ref.read_text(encoding="utf-8"))
        entries = next(v for v in data.values() if isinstance(v, list))
        entries[0]["aliases"] = ["a synthetic alias", "another one"]
        ref.write_text(_yaml.safe_dump(data), encoding="utf-8")
        cb = load_config_bundle(good_config)
        from meltiro.reference_lists import entry_aliases
        assert entry_aliases(cb.reference_lists["gauge_list"][0]) == \
            ["a synthetic alias", "another one"]

    def test_duplicate_alias_rejected_at_bundle_load(self, good_config):
        import yaml as _yaml
        ref = good_config / "reference" / "gauge_list.yaml"
        data = _yaml.safe_load(ref.read_text(encoding="utf-8"))
        entries = next(v for v in data.values() if isinstance(v, list))
        entries[0]["aliases"] = ["clash"]
        entries[1]["aliases"] = ["Clash"]
        ref.write_text(_yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "duplicates an alias" in str(excinfo.value)

    def test_separator_bearing_name_loads(self, good_config):
        # List-valued reference fields are real string_list arrays validated
        # element by element, so a canonical name containing a comma is fine:
        # nothing anywhere splits a reference value on a separator, so no
        # loader guard has to reserve one.
        import yaml as _yaml
        ref = good_config / "reference" / "gauge_list.yaml"
        data = _yaml.safe_load(ref.read_text(encoding="utf-8"))
        entries = next(v for v in data.values() if isinstance(v, list))
        entries.append({
            "tool_name":
                "Widget Durability Scale for Brackets, Second Edition"})
        ref.write_text(_yaml.safe_dump(data), encoding="utf-8")
        load_config_bundle(good_config)  # must not raise

    def test_prompt_unknown_reference_placeholder_rejected_at_load(
            self, good_config):
        # A prompt citing {reference:NAME} for a list the bundle does not
        # provide must fail at config-load time, not mid-run at render time.
        prompt = good_config / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\n{reference:not_a_list}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "not_a_list" in msg
        assert "extractor_system.md" in msg


class TestPromptPartials:
    def test_shipped_config_uses_partials_and_loads(self, config_dir):
        # The shipped bundle factors its shared review_context /
        # inclusion_criteria blocks into prompts/partials/ and cites them
        # with {include:...}. It must load cleanly.
        cb = load_config_bundle(config_dir)
        partials = config_dir / "prompts" / "partials"
        assert (partials / "review_context.md").is_file()
        assert (partials / "inclusion_criteria.md").is_file()
        # The extractor prompt cites the partials rather than inlining them.
        ext = cb.extractor_system_path.read_text(encoding="utf-8")
        assert "{include:review_context}" in ext
        assert "{include:inclusion_criteria}" in ext

    def test_unknown_partial_rejected_at_load(self, good_config):
        # A prompt citing {include:NAME} for a partial the bundle does not
        # provide must fail at config-load time, not mid-run at render time.
        prompt = good_config / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\n{include:missing_block}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "missing_block" in msg
        assert "extractor_system.md" in msg

    def test_every_missing_partial_reported(self, good_config):
        # All missing partials across all prompts are collected into one
        # error, so the author fixes them in a single pass.
        ext = good_config / "prompts" / "extractor_system.md"
        rev = good_config / "prompts" / "review_system.md"
        ext.write_text(ext.read_text(encoding="utf-8") + "\n{include:aaa}\n",
                       encoding="utf-8")
        rev.write_text(rev.read_text(encoding="utf-8") + "\n{include:bbb}\n",
                       encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        problems = " ".join(excinfo.value.problems)
        assert "aaa" in problems
        assert "bbb" in problems

    def test_nested_include_rejected_at_load(self, good_config):
        # A partial may not itself cite an {include:...}. Nesting fails loudly
        # at load, naming the offending partial.
        partials = good_config / "prompts" / "partials"
        partials.mkdir(parents=True, exist_ok=True)
        (partials / "nester.md").write_text(
            "wraps {include:review_context}", encoding="utf-8")
        prompt = good_config / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\n{include:nester}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "nester" in msg
        assert "nest" in msg.lower()

    def test_reference_inside_partial_validated_at_load(self, good_config):
        # A {reference:NAME} placeholder that lives INSIDE a partial is still
        # validated at load: the reference check expands includes first.
        partials = good_config / "prompts" / "partials"
        partials.mkdir(parents=True, exist_ok=True)
        (partials / "with_ref.md").write_text(
            "<x>{reference:not_a_list}</x>", encoding="utf-8")
        prompt = good_config / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\n{include:with_ref}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "not_a_list" in str(excinfo.value)


class TestCapPlaceholders:
    """The tool-call cap placeholders are NOT substituted. A prompt citing one
    would couple the operational cap into prompt_hash and so into config_fp,
    breaking the "hit the cap, resume with a raised cap" recovery, so such a
    prompt fails loudly at config-load time rather than rendering."""

    def test_shipped_config_has_no_cap_placeholders(self, config_dir):
        # The shipped bundle must remain clean, otherwise it would not load.
        load_config_bundle(config_dir)  # must not raise

    def test_max_tool_calls_placeholder_rejected(self, good_config):
        prompt = good_config / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\nYou get {max_tool_calls} tool calls.\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "{max_tool_calls}" in msg
        assert "extractor_system.md" in msg

    def test_review_cap_placeholder_rejected(self, good_config):
        prompt = good_config / "prompts" / "review_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\nYou get {max_review_tool_calls} tool calls.\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "{max_review_tool_calls}" in msg
        assert "review_system.md" in msg

    def test_cap_placeholder_in_an_engine_override_rejected(
            self, good_config):
        # An override is prompt text the bundle wrote, and it reaches a model
        # without passing through any prompt file, so the guard reads it
        # directly. Pin that surface specifically: a cap placeholder there
        # would otherwise be the one place it could still be interpolated.
        override = good_config / "prompts" / "partials" / "meltiro"
        override.mkdir(parents=True, exist_ok=True)
        (override / "reviewer.md").write_text(
            "Cap: {max_tool_calls}.\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "{max_tool_calls}" in msg
        assert "reviewer.md" in msg

    def test_cap_placeholder_inside_partial_rejected(self, good_config):
        # A placeholder hiding in an included partial is caught too: the check
        # expands includes before scanning, mirroring the reference check.
        partials = good_config / "prompts" / "partials"
        partials.mkdir(parents=True, exist_ok=True)
        (partials / "with_cap.md").write_text(
            "<x>{max_tool_calls}</x>", encoding="utf-8")
        prompt = good_config / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\n{include:with_cap}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "{max_tool_calls}" in str(excinfo.value)


class TestCheckerUserPlaceholderAllowlist:
    """Substitution into the per-field scaffold is a plain `str.replace` per
    known slot, so an unknown placeholder is not an error at render time: it
    survives into the prompt as literal text and the checker reads
    `{field_pat}` where the field path should be. That silent failure is
    hoisted to config-load time, over the one copy of that scaffold a bundle
    can write: its override of the engine's `checker_user` prompt."""

    def _rewrite(self, config, text):
        path = config / "prompts" / "partials" / "meltiro"
        path.mkdir(parents=True, exist_ok=True)
        (path / "checker_user.md").write_text(text, encoding="utf-8")

    def test_the_shipped_bundle_cites_only_known_slots(self, config_dir):
        load_config_bundle(config_dir)  # must not raise

    def test_a_misspelt_slot_is_rejected(self, good_config):
        self._rewrite(good_config, "`{field_pat}`: {field_description}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        msg = str(excinfo.value)
        assert "{field_pat}" in msg
        assert "checker_user.md" in msg
        assert "literal prompt text" in msg

    def test_every_unknown_slot_is_reported(self, good_config):
        self._rewrite(good_config, "{field_pat} {evidence_blk} {valu}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert len(excinfo.value.problems) == 3

    def test_an_omitted_slot_is_not_an_error(self, good_config):
        # A config is free to leave a slot out; only an unknown one fails.
        self._rewrite(good_config, "`{field_path}`\n\n{value}\n")
        load_config_bundle(good_config)  # must not raise

    def test_the_notes_slot_is_allowed(self, good_config):
        self._rewrite(good_config, "{value}\n{notes_block}\n")
        load_config_bundle(good_config)  # must not raise

    def test_a_reference_citation_is_rendered_in(self, good_config):
        # A colon-carrying placeholder is not a slot: it names a reference
        # list, and the scaffold is substituted from it before any per-field
        # slot is filled, so the checker reads the canonical names rather than
        # the token.
        self._rewrite(good_config, "{value}\n{reference:gauge_list}\n")
        bundle = load_config_bundle(good_config)
        rendered = render_checker_user_template(
            bundle.partials_dir, predicates=stage_predicates(2, True, False),
            reference_lists=bundle.reference_lists)
        assert "{reference:gauge_list}" not in rendered
        assert "Widget Durability Scale 9 (WDS-9)" in rendered

    def test_prose_braces_and_json_examples_are_not_caught(self, good_config):
        # The pattern matches a brace-wrapped lowercase identifier and nothing
        # else, so a JSON example or an uppercase token in the prose is left
        # alone.
        self._rewrite(good_config, (
            '{value}\n\nRespond as {"verdict": "ok", "rationale": "..."}\n'
            "Not a slot: {NAME} {Field_Path} {two words} {} {a-b}\n"))
        load_config_bundle(good_config)  # must not raise

    def test_an_unknown_slot_inside_a_checker_partial_is_caught(
            self, good_config):
        # The checker's own prompt file is the other surface, and includes are
        # expanded before the scan there, as they are for the cap and
        # reference checks.
        partials = good_config / "prompts" / "partials"
        partials.mkdir(parents=True, exist_ok=True)
        (partials / "tail.md").write_text("{feild_path}", encoding="utf-8")
        prompt = good_config / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8") + "\n{include:tail}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "{feild_path}" in str(excinfo.value)

    def test_a_cap_placeholder_still_gets_its_own_message(self, good_config):
        # `{max_tool_calls}` is a lowercase identifier, so both guards would
        # match it. The cap guard runs first, so the author gets the targeted
        # explanation rather than the generic unknown-slot one.
        self._rewrite(good_config, "{value}\nCap: {max_tool_calls}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        assert "operational budget" in str(excinfo.value)

    def test_the_allowlist_matches_what_the_renderer_substitutes(self):
        # The allowlist is a second copy of the renderer's slot list, so pin
        # them against each other: a slot added to one and not the other is
        # either an unfillable placeholder or an unusable one.
        import inspect

        from meltiro import checker_prompts
        from meltiro.config_bundle import (
            _CHECKER_USER_PLACEHOLDERS,
            _PLACEHOLDER_TOKEN,
        )

        source = inspect.getsource(checker_prompts.build_checker_user_message)
        substituted = {
            name for line in source.splitlines()
            if ".replace(" in line
            for name in _PLACEHOLDER_TOKEN.findall(line)
        }
        assert substituted == set(_CHECKER_USER_PLACEHOLDERS)


class TestTheLoaderRaisesOneErrorType:
    """`load_config_bundle` raises `ConfigBundleError` and nothing else.

    That is the whole contract every caller is written against: the CLI
    catches exactly that type, prints its message, and exits 1. A defect the
    loader let through as some other exception reached an operator as a
    traceback — the opposite of a bundle that "fails whole" — so the template
    loader's own errors and pyyaml's are carried inside the contract rather
    than escaping around it.
    """

    def test_an_unknown_template_field_key_is_a_bundle_error(
            self, good_config):
        tmpl = good_config / "extraction_template.yaml"
        text = tmpl.read_text(encoding="utf-8")
        text = text.replace("      required: true",
                            "      requred: true", 1)
        tmpl.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        # The template loader's own message, carried through intact.
        assert "requred" in str(excinfo.value)

    def test_a_misspelt_variable_key_names_the_section_and_the_mapping(
            self, good_config):
        # `varible:` used to reach a bare `f["variable"]` KeyError, which named
        # neither the field nor the file it was in.
        tmpl = good_config / "extraction_template.yaml"
        text = tmpl.read_text(encoding="utf-8")
        text = text.replace("    - variable:", "    - varible:", 1)
        tmpl.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        message = str(excinfo.value)
        assert "`variable:`" in message
        # The keys the mapping DOES declare, which is what points at the typo.
        assert "varible" in message

    def test_a_broken_template_yaml_is_a_bundle_error_with_the_line(
            self, good_config):
        tmpl = good_config / "extraction_template.yaml"
        tmpl.write_text("records:\n  widget:\n   - [unclosed\n",
                        encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        message = str(excinfo.value)
        assert "extraction_template.yaml" in message
        # pyyaml's file/line/column line, which is the useful half.
        assert "line" in message and "column" in message

    def test_a_broken_pipeline_yaml_is_a_bundle_error_with_the_line(
            self, good_config):
        (good_config / "pipeline.yaml").write_text(
            "extractor_model: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(good_config)
        message = str(excinfo.value)
        assert "pipeline.yaml" in message
        assert "line" in message and "column" in message
