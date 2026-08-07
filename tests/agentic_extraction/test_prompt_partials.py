"""Tests for the `{include:NAME}` prompt-partials mechanism.

Covers the substitution primitive (meltiro.prompt_partials), its
integration into the extractor prompt builder (includes expand before
`{reference:...}` substitution), and the provenance guarantee that editing a
partial's content moves the config fingerprint.
"""

import pytest

from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import config_fingerprint
from meltiro.prompt_builder import (
    build_system_message,
    compute_prompt_config_hash,
)
from meltiro.prompt_partials import (
    included_names,
    stage_predicates,
    substitute_include_placeholders,
)


class TestSubstitutionPrimitive:
    def test_included_names_finds_all(self):
        text = "a {include:one} b {include:two} c {include:one}"
        assert included_names(text) == {"one", "two"}

    def test_no_placeholder_needs_no_dir(self, tmp_path):
        # A bundle with no `{include:}` never touches the partials directory,
        # so a missing directory is fine when nothing is cited.
        text = "no includes here"
        out = substitute_include_placeholders(text, tmp_path / "nope")
        assert out == text

    def test_expands_and_strips(self, tmp_path):
        partials = tmp_path / "partials"
        partials.mkdir()
        (partials / "block.md").write_text(
            "\n  <block>\ncontent\n</block>\n  \n", encoding="utf-8")
        out = substitute_include_placeholders(
            "before\n{include:block}\nafter", partials)
        # Surrounding whitespace of the partial is stripped; the placeholder
        # sat on its own line so the block lands cleanly between the two.
        assert out == "before\n<block>\ncontent\n</block>\nafter"

    def test_missing_partial_raises(self, tmp_path):
        partials = tmp_path / "partials"
        partials.mkdir()
        with pytest.raises(ConfigBundleError) as excinfo:
            substitute_include_placeholders("{include:absent}", partials)
        assert "absent" in str(excinfo.value)

    def test_nested_include_raises(self, tmp_path):
        partials = tmp_path / "partials"
        partials.mkdir()
        (partials / "outer.md").write_text(
            "outer wraps {include:inner}", encoding="utf-8")
        (partials / "inner.md").write_text("inner", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            substitute_include_placeholders("{include:outer}", partials)
        msg = str(excinfo.value)
        assert "nested" in msg.lower()
        assert "outer" in msg


class TestBuilderIntegration:
    def _write_prompt(self, tmp_path, prompt_body, partials):
        prompts = tmp_path / "prompts"
        (prompts / "partials").mkdir(parents=True)
        (prompts / "extractor_system.md").write_text(
            prompt_body, encoding="utf-8")
        for name, content in partials.items():
            (prompts / "partials" / f"{name}.md").write_text(
                content, encoding="utf-8")
        return prompts / "extractor_system.md"

    def test_include_expands_before_reference(self, tmp_path,
                                              synthetic_template):
        # A partial may itself carry a `{reference:...}` placeholder; because
        # includes expand first, that reference resolves exactly as it would
        # inline.
        system_path = self._write_prompt(
            tmp_path,
            "<top>\n{include:ctx}\n</top>",
            {"ctx": "<review_context>\n{reference:gauge_list}\n"
                    "</review_context>"},
        )
        txt = build_system_message(
            synthetic_template, image_labels=[],
            system_prompt_path=system_path,
            reference_lists={"gauge_list": [
                {"tool_name": "WDS-9"}, {"tool_name": "CRT-HD"}]},
        )
        assert "WDS-9" in txt and "CRT-HD" in txt
        # No placeholders survive.
        assert "{include:" not in txt
        assert "{reference:" not in txt

    def test_missing_partial_fails_loudly_in_builder(self, tmp_path,
                                                     synthetic_template):
        system_path = self._write_prompt(
            tmp_path, "<top>\n{include:ghost}\n</top>", {})
        with pytest.raises(ConfigBundleError) as excinfo:
            build_system_message(
                synthetic_template, image_labels=[],
                system_prompt_path=system_path, reference_lists={})
        assert "ghost" in str(excinfo.value)


class TestFingerprintMovesOnPartialEdit:
    def _bundle(self, tmp_path):
        prompts = tmp_path / "prompts"
        (prompts / "partials").mkdir(parents=True)
        (prompts / "extractor_system.md").write_text(
            "<top>\n{include:shared}\n</top>", encoding="utf-8")
        partial = prompts / "partials" / "shared.md"
        partial.write_text("<shared>\noriginal wording\n</shared>",
                           encoding="utf-8")
        return prompts / "extractor_system.md", partial

    def test_editing_partial_moves_config_fingerprint(self, tmp_path,
                                                      synthetic_template):
        system_path, partial = self._bundle(tmp_path)

        def fp():
            prompt_hash = compute_prompt_config_hash(
                synthetic_template, system_prompt_path=system_path,
                max_checks_per_field=3, reference_lists={})
            return config_fingerprint(
                "claude-opus-4-7", prompt_hash, "template-hash",
                tool_set_hash="tools")

        before = fp()
        # Change ONLY the partial's content: nothing else about the config
        # moves, yet the rendered prompt (and so the fingerprint) must.
        partial.write_text("<shared>\nrevised wording\n</shared>",
                           encoding="utf-8")
        after = fp()
        assert before != after


class TestCheckerUserTemplatePartials:
    """checker_fp must move when a partial cited by the checker user
    template changes (the render path expands includes per field, so the
    fingerprint must hash the expanded text, not the raw file)."""

    def test_editing_cited_partial_moves_checker_fp(
            self, tmp_path, config_dir, checker_system_path):
        from meltiro.checker import CheckerConfig
        from meltiro.reference_lists import load_reference_lists
        from meltiro.template import load_template

        template = load_template(config_dir / "extraction_template.yaml")
        reference_lists = load_reference_lists(config_dir / "reference")

        prompts = tmp_path / "prompts"
        partials = prompts / "partials"
        partials.mkdir(parents=True)
        user_tmpl = prompts / "checker_user_template.md"
        user_tmpl.write_text(
            "{include:ctx}\nfield: {field_description}\n", encoding="utf-8")
        (partials / "ctx.md").write_text("VERSION-A", encoding="utf-8")

        # A real registry id: checker_fp folds in the model's provider and
        # base_url, so an unregistered id fails model resolution.
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(checker_system_path),
            user_prompt_template_path=str(user_tmpl),
        )
        # The run's structure predicates, which the checker takes as an
        # argument because it keeps no copy of the toggles behind them. Held
        # fixed here: what varies is the partial's content.
        predicates = stage_predicates(2, True)
        assert "VERSION-A" in cfg.user_prompt_template_text(
            predicates=predicates)
        fp_a = cfg.fingerprint(template, reference_lists,
                               predicates=predicates)

        (partials / "ctx.md").write_text("VERSION-B", encoding="utf-8")
        assert "VERSION-B" in cfg.user_prompt_template_text(
            predicates=predicates)
        fp_b = cfg.fingerprint(template, reference_lists,
                               predicates=predicates)
        assert fp_a != fp_b
