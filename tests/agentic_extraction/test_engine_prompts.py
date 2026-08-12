"""The engine's own prompt sections: how they resolve, which axis they move.

The engine's contract — what the extractor's tools do, what the pipeline does
around a call, what the checker is and is not shown — is machine behaviour, not
a methodological choice. It ships as named markdown under
`meltiro/engine_prompts/` and a bundle composes it with
`{include:meltiro:NAME}`, so a review that has no opinion about the engine
never has to describe it correctly for the run to behave as documented.

The boundary this file pins is one of OWNERSHIP, and it is not obvious:

  - engine text rides in `engine_fp`, through the source digest that hashes
    `engine_prompts/*.md` beside the package's modules. Rewording a section is
    an engine release, and every bundle's `prompts_hash`, `config_fp`,
    `checker_fp` and `review_fp` hold across it — which is what lets a
    consumer pin those numbers at all.
  - a bundle's OVERRIDE (`prompts/partials/meltiro/NAME.md`) is text the config
    author wrote, so it rides in the config fingerprints like any other prose
    of theirs, and the engine's copy of that section stops being consulted.

The prompts SENT to a model are unaffected by any of this: every include
expands on the wire, always. Only the render taken for a hash differs.
"""

import shutil

import pytest

from meltiro import prompt_partials
from meltiro.checker import CheckerConfig
from meltiro.checker_prompts import build_checker_system_text
from meltiro.config_bundle import _ROLE_SECTIONS, load_config_bundle
from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import (
    config_fingerprint,
    review_config_fingerprint,
    structure_hash,
)
from meltiro.prompt_builder import (
    build_review_system_message,
    build_system_message,
    compute_prompt_config_hash,
)
from meltiro.prompt_partials import (
    HASH,
    WIRE,
    engine_section_names,
    stage_predicates,
    substitute_include_placeholders,
)
from meltiro.reference_lists import load_reference_lists
from meltiro.run_log import _hash_tree
from meltiro.template import load_template

# A stand-in for direktoro's provider-call identity block, which these tests
# hold fixed: what varies here is prompt text.
PINNED_CALL_IDENTITY = "call-identity-engine-prompts"

PREDICATES = stage_predicates(2, True)


def _copy_bundle(tmp_path, config_dir):
    """A writable copy of the fixture bundle."""
    dest = tmp_path / "config"
    shutil.copytree(config_dir, dest)
    return dest


def _write_override(bundle_dir, name, text):
    """Ship a bundle override for engine section `name`."""
    override_dir = bundle_dir / "prompts" / "partials" / "meltiro"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / f"{name}.md").write_text(text, encoding="utf-8")
    return override_dir / f"{name}.md"


def _uncompose(bundle_dir, role, keep=()):
    """Strip a role's prompt of every engine section it composes but `keep`.

    Written against `_ROLE_SECTIONS` rather than a fixed list, so a section
    added to a role is stripped here too and the coverage warning is tested
    against what actually ships.
    """
    prompt = bundle_dir / "prompts" / f"{role}.md"
    text = prompt.read_text(encoding="utf-8")
    for name in _ROLE_SECTIONS[role]:
        if name in keep:
            continue
        text = text.replace(f"{{include:meltiro:{name}}}",
                            f"This review's own {name}.")
    prompt.write_text(text, encoding="utf-8")
    return prompt


def _engine_text(name):
    return (prompt_partials.ENGINE_PROMPTS_DIR / f"{name}.md").read_text(
        encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# What ships, and how it resolves
# ---------------------------------------------------------------------------

class TestTheSectionsShip:
    def test_every_section_is_a_named_non_empty_file(self):
        names = engine_section_names()
        assert names, "the engine ships no prompt sections at all"
        for name in names:
            assert _engine_text(name), f"engine section {name} is empty"

    def test_every_section_belongs_to_exactly_one_role(self):
        # The coverage warning asks whether a role's prompt composes any of
        # ITS sections, so every shipped section has to be classified, and a
        # name says nothing about that (`recording_notes` is the extractor's).
        # This is the gate: a section added to `engine_prompts/` without a line
        # in `_ROLE_SECTIONS` fails here rather than shipping unclassified, and
        # a classified name that no longer ships fails here too.
        shipped = set(engine_section_names())
        claimed = []
        for sections in _ROLE_SECTIONS.values():
            claimed.extend(sections)
        assert sorted(claimed) == sorted(set(claimed)), (
            "a section is claimed by two roles")
        assert set(claimed) == shipped

    def test_each_role_has_a_section_of_its_own(self):
        # A role with no section of its own would make the coverage warning
        # unsatisfiable: no prompt could ever compose one.
        for role, sections in _ROLE_SECTIONS.items():
            assert sections, f"{role} claims no engine section"

    def test_they_are_declared_as_package_data(self):
        # A checkout finds them beside the modules whatever pyproject says; an
        # INSTALL only carries what package-data declares, and a wheel without
        # them cannot render a prompt that composes one. The suite runs from
        # the source tree, so nothing else here would notice.
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        patterns = config["tool"]["setuptools"]["package-data"]["meltiro"]
        assert "engine_prompts/*.md" in patterns

    def test_no_section_nests_an_include(self):
        # Expansion is one level deep everywhere else; the engine's own files
        # are held to the same rule rather than exempted from it.
        for name in engine_section_names():
            text = _engine_text(name)
            assert "{include:" not in text
            assert "{include_if:" not in text


class TestResolution:
    def test_a_composed_section_reaches_the_model(self, config_dir):
        rendered = build_system_message(
            [], system_prompt_path=config_dir / "prompts"
            / "extractor_system.md",
            reference_lists=load_reference_lists(config_dir / "reference"))
        assert "{include:meltiro:" not in rendered
        assert "record_initial_check" in rendered
        # A sentence that exists only in the engine's copy.
        assert _engine_text("recording_notes").splitlines()[0] in rendered

    def test_a_hash_render_keeps_the_directive_instead(self, config_dir):
        rendered = build_system_message(
            [], system_prompt_path=config_dir / "prompts"
            / "extractor_system.md",
            reference_lists=load_reference_lists(config_dir / "reference"),
            mode=HASH)
        assert "{include:meltiro:extractor_workflow}" in rendered
        assert "record_initial_check" not in rendered
        # Only the ENGINE namespace is held back; the bundle's own partials
        # expand in both renders, because the bundle wrote them.
        assert "{include:review_context}" not in rendered

    def test_a_bundle_partial_of_the_same_name_is_a_different_thing(
            self, tmp_path):
        partials = tmp_path / "partials"
        partials.mkdir()
        (partials / "recording_notes.md").write_text(
            "the bundle's own block", encoding="utf-8")
        out = substitute_include_placeholders(
            "{include:recording_notes}", partials)
        assert out == "the bundle's own block"

    def test_conditional_engine_include_follows_its_predicate(self, tmp_path):
        partials = tmp_path / "partials"
        partials.mkdir()
        text = "before\n{include_if:checker:meltiro:checker_briefing}\nafter"
        on = substitute_include_placeholders(
            text, partials, predicates={"checker": True, "review": True})
        off = substitute_include_placeholders(
            text, partials, predicates={"checker": False, "review": True})
        assert _engine_text("checker_briefing") in on
        assert off == "before\nafter"

    def test_a_conditional_engine_include_hashes_as_its_directive(
            self, tmp_path):
        partials = tmp_path / "partials"
        partials.mkdir()
        text = "{include_if:checker:meltiro:checker_briefing}\n"
        out = substitute_include_placeholders(
            text, partials, predicates={"checker": True, "review": True},
            mode=HASH)
        assert out.strip() == "{include_if:checker:meltiro:checker_briefing}"

    def test_an_unknown_render_mode_is_refused(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            substitute_include_placeholders("nothing", tmp_path, mode="both")
        assert "wire" in str(excinfo.value)


class TestUnknownSection:
    """An unknown `meltiro:NAME` is a load-time error naming what exists.

    The namespace is reserved and its vocabulary closed: treating a typo as a
    new section would compose nothing, warn about nothing, and ship a prompt
    that describes an engine the run does not have.
    """

    def test_unknown_section_fails_at_load(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:extractor_workflows}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "extractor_workflows" in message
        for name in engine_section_names():
            assert name in message

    def test_every_prompt_surface_is_checked(self, tmp_path, config_dir):
        # The citation guard reads all four prompts, not just the extractor's
        # above.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:house_style}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "house_style" in message
        assert "checker_system.md" in message

    def test_an_override_may_not_nest_an_include(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes",
                        "ours, plus {include:review_context}")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "nest" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# The override
# ---------------------------------------------------------------------------

class TestTheOverrideDirectoryIsEnumerated:
    """A file in `partials/meltiro/` overrides a section or fails the load.

    The filename is the whole of the wiring, and a near miss is silent in the
    worst way: the bundle ships text its author believes the model reads, the
    engine's own words go out instead, and every fingerprint agrees with both
    of them. Enumerating the directory is what turns that into an error, and
    it is why the comparison is against a LISTING rather than a probe — a
    case-insensitive filesystem answers `recording_notes.md.is_file()` for a
    file called `Recording_Notes.md`, so a probe would pass this bundle on
    macOS and fail it on Linux.
    """

    def test_a_typo_in_the_stem_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_note", "our note policy")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "recording_note.md" in message
        for name in engine_section_names():
            assert name in message

    def test_the_wrong_case_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "Recording_Notes", "our note policy")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "Recording_Notes.md" in str(excinfo.value)

    def test_a_name_no_section_has_is_refused(self, tmp_path, config_dir):
        # Refused on the filename alone: this bundle cites no such section, so
        # nothing else in the load would ever look at the file.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "house_style", "our own block")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "house_style.md" in str(excinfo.value)

    def test_the_correctly_named_override_still_wins(self, tmp_path,
                                                     config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes", "our note policy")
        bundle = load_config_bundle(bundle_dir)
        rendered = build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists)
        assert "our note policy" in rendered
        assert _engine_text("recording_notes") not in rendered


class TestOverrideWins:
    def test_the_override_is_what_the_model_reads(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes",
                        "<notes>this review's own note policy</notes>")
        bundle = load_config_bundle(bundle_dir)
        rendered = build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists)
        assert "this review's own note policy" in rendered
        assert "**Scope notes.**" not in rendered

    def test_the_override_moves_prompts_hash(self, tmp_path, config_dir):
        before = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "recording_notes", "our note policy")
        after = load_config_bundle(overridden_dir)
        assert before.prompts_hash != after.prompts_hash
        assert before.instrument_fp != after.instrument_fp

    def test_the_override_moves_config_fp(self, tmp_path, config_dir):
        def fp(bundle):
            return config_fingerprint(
                PINNED_CALL_IDENTITY,
                compute_prompt_config_hash(
                    system_prompt_path=bundle.extractor_system_path,
                    max_checks_per_field=2,
                    reference_lists=bundle.reference_lists),
                bundle.template_hash)

        plain = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "recording_evidence",
                        "quote what you cite.\n{image_labels_list}")
        assert fp(plain) != fp(load_config_bundle(overridden_dir))

    def test_overriding_the_checker_section_moves_checker_fp(
            self, tmp_path, config_dir):
        template = load_template(config_dir / "extraction_template.yaml")
        references = load_reference_lists(config_dir / "reference")

        def fp(bundle_dir):
            bundle = load_config_bundle(bundle_dir)
            cfg = CheckerConfig(
                max_tokens=1024,
                checker_model="claude-sonnet-4-6",
                system_prompt_path=str(bundle.checker_system_path),
                user_prompt_template_path=str(
                    bundle.checker_user_template_path),
            )
            return cfg.fingerprint(template, references,
                                   predicates=PREDICATES,
                                   max_checks_per_field=2)

        plain_dir = _copy_bundle(tmp_path, config_dir)
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "checker_briefing",
                        "you see one field, and the paper around its quotes.")
        assert fp(plain_dir) != fp(overridden_dir)

    def test_an_identical_override_still_hashes_differently(
            self, tmp_path, config_dir):
        # Deliberate, and the point of the boundary rather than an artefact of
        # it. The two bundles read identically to a model, and they are not the
        # same instrument: one is pinned to whatever the engine's copy of the
        # section says in the release it runs under, and the other to a copy
        # its author owns and an engine release cannot change. A hash that
        # equated them would report the first bundle as unchanged across a
        # release that reworded the section under it.
        plain_dir = _copy_bundle(tmp_path, config_dir)
        copied_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(copied_dir, "recording_notes",
                        _engine_text("recording_notes"))

        plain = load_config_bundle(plain_dir)
        copied = load_config_bundle(copied_dir)

        # Byte-identical on the wire...
        def wire(bundle):
            return build_system_message(
                [], system_prompt_path=bundle.extractor_system_path,
                reference_lists=bundle.reference_lists)

        assert wire(plain) == wire(copied)
        # ...and different instruments.
        assert plain.prompts_hash != copied.prompts_hash


# ---------------------------------------------------------------------------
# The slots a composed section may need filling
# ---------------------------------------------------------------------------

class TestTheCheckerPromptsSlots:
    """A section renders variables, and the prompt composing it must supply
    them.

    Substitution is a plain `str.replace` per slot, so an unsupplied `{name}`
    is no error at render time: it survives into the prompt and the model
    reads the token where the value should be. The checker's system prompt
    supplies exactly one, the check budget; `recording_evidence` renders the
    image-label list, which no checker call has. Composing it there is a load
    error naming the variable rather than a literal token on the wire.
    """

    def _append(self, bundle_dir, text):
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(prompt.read_text(encoding="utf-8") + text,
                          encoding="utf-8")
        return prompt

    def test_the_check_budget_is_substituted(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        self._append(bundle_dir,
                     "\nEach field is checked at most "
                     "{max_checks_per_field} times.\n")
        bundle = load_config_bundle(bundle_dir)
        rendered = build_checker_system_text(
            system_prompt_path=bundle.checker_system_path,
            reference_lists=bundle.reference_lists,
            predicates=PREDICATES, max_checks_per_field=3)
        assert "checked at most 3 times" in rendered
        assert "{max_checks_per_field}" not in rendered

    def test_a_section_the_prompt_cannot_fill_fails_at_load(
            self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        self._append(bundle_dir, "\n{include:meltiro:recording_evidence}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "image_labels_list" in message
        assert "checker_system.md" in message

    def test_an_override_that_needs_one_fails_the_same_way(
            self, tmp_path, config_dir):
        # The override is the bundle's own text, and it is held to the same
        # rule: a variable this prompt does not supply is refused whoever
        # wrote the words around it.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "checker_briefing",
                        "You judge one field.\n{image_labels_list}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "image_labels_list" in str(excinfo.value)

    def test_the_extractor_supplies_that_list_itself(self, config_dir):
        # The same section in the prompt written for it: no load error, and
        # the slot is filled rather than shipped.
        bundle = load_config_bundle(config_dir)
        rendered = build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists)
        assert "{image_labels_list}" not in rendered
        assert "no figures or tables were cropped" in rendered


# ---------------------------------------------------------------------------
# The fingerprint boundary
# ---------------------------------------------------------------------------

class TestEditingAnEngineSection:
    """An engine edit moves the ENGINE axis and leaves the config axes alone.

    The engine's copy is swapped for a writable one so the edit can be made
    without touching the installed package; `engine_section_names` and the
    resolution both read the directory at call time, so the substitution is
    complete.
    """

    @pytest.fixture
    def engine_dir(self, tmp_path, monkeypatch):
        dest = tmp_path / "engine_prompts"
        shutil.copytree(prompt_partials.ENGINE_PROMPTS_DIR, dest)
        monkeypatch.setattr(prompt_partials, "ENGINE_PROMPTS_DIR", dest)
        return dest

    @pytest.fixture
    def bundle_dir(self, tmp_path, config_dir):
        # The reviewer's prompt composes a section too, so all four config
        # fingerprints below genuinely depend on engine text.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        review = bundle_dir / "prompts" / "review_system.md"
        review.write_text(
            review.read_text(encoding="utf-8")
            + "\n<recording_notes>\n{include:meltiro:recording_notes}\n"
              "</recording_notes>\n",
            encoding="utf-8")
        return bundle_dir

    def _fingerprints(self, bundle_dir, config_dir):
        template = load_template(config_dir / "extraction_template.yaml")
        references = load_reference_lists(config_dir / "reference")
        bundle = load_config_bundle(bundle_dir)
        checker = CheckerConfig(
            max_tokens=1024,
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(bundle.checker_system_path),
            user_prompt_template_path=str(bundle.checker_user_template_path),
        )
        return {
            "prompts_hash": bundle.prompts_hash,
            "instrument_fp": bundle.instrument_fp,
            "config_fp": config_fingerprint(
                PINNED_CALL_IDENTITY,
                compute_prompt_config_hash(
                    system_prompt_path=bundle.extractor_system_path,
                    max_checks_per_field=2,
                    reference_lists=bundle.reference_lists),
                bundle.template_hash),
            "checker_fp": checker.fingerprint(
                template, references, predicates=PREDICATES,
                max_checks_per_field=2),
            "review_fp": review_config_fingerprint(
                PINNED_CALL_IDENTITY,
                build_review_system_message(
                    [], system_prompt_path=bundle.review_system_path,
                    reference_lists=bundle.reference_lists, mode=HASH)),
        }

    def _wire(self, bundle_dir):
        bundle = load_config_bundle(bundle_dir)
        return build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists)

    def test_no_config_fingerprint_moves(self, engine_dir, bundle_dir,
                                         config_dir):
        before = self._fingerprints(bundle_dir, config_dir)
        wire_before = self._wire(bundle_dir)
        for name in ("extractor_workflow", "recording_notes",
                     "checker_briefing"):
            path = engine_dir / f"{name}.md"
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\nA sentence a later release added.\n", encoding="utf-8")
        # The edit really did reach the prompts: without this the equality
        # below would hold just as well for an edit that changed nothing.
        assert self._wire(bundle_dir) != wire_before
        after = self._fingerprints(bundle_dir, config_dir)
        assert before == after

    def test_but_the_model_reads_the_edit(self, engine_dir, bundle_dir):
        bundle = load_config_bundle(bundle_dir)
        (engine_dir / "recording_notes.md").write_text(
            "note whatever you like.", encoding="utf-8")
        rendered = build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists)
        assert "note whatever you like." in rendered

    def test_an_overridden_section_is_untouched_by_the_engine_edit(
            self, engine_dir, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path / "overriding", config_dir)
        _write_override(bundle_dir, "recording_notes", "our note policy")
        before = load_config_bundle(bundle_dir).prompts_hash
        (engine_dir / "recording_notes.md").write_text(
            "the engine's new wording.", encoding="utf-8")
        bundle = load_config_bundle(bundle_dir)
        assert bundle.prompts_hash == before
        rendered = build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists)
        assert "our note policy" in rendered
        assert "the engine's new wording." not in rendered


class TestAValueInterpolatedIntoEngineText:
    """The check budget: read by the extractor, absent from `prompt_hash`.

    `extractor_workflow` states the budget, and the prompt hash is taken over
    the render that leaves that section as its directive — so the NUMBER is
    outside the preimage. That is the boundary doing its job rather than a
    hole in it: the pair below is the whole statement. `prompt_hash` reports
    the text the author wrote, which two budgets share, and `config_fp` still
    separates them, because `structure_hash` beside it carries the toggle. A
    consumer grouping runs by `config_fp` never conflates them, and a bundle's
    `prompts_hash` survives an engine release that reworded the sentence the
    number sits in.
    """

    def test_the_extractor_is_told_the_budget(self, config_dir):
        # Without this the equalities below would hold for a number that
        # reaches nobody.
        bundle = load_config_bundle(config_dir)
        rendered = build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=bundle.reference_lists,
            max_checks_per_field=3)
        assert "at most 3 times" in rendered

    def test_prompt_hash_holds_and_config_fp_moves(self, config_dir):
        bundle = load_config_bundle(config_dir)

        def hashes(max_checks):
            prompt_hash = compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=max_checks,
                reference_lists=bundle.reference_lists)
            return prompt_hash, config_fingerprint(
                PINNED_CALL_IDENTITY, prompt_hash, bundle.template_hash,
                structure_hash=structure_hash(max_checks))

        two_prompts, two_fp = hashes(2)
        three_prompts, three_fp = hashes(3)
        assert two_prompts == three_prompts
        assert two_fp != three_fp

    def test_a_bundle_that_states_it_itself_hashes_it(self, tmp_path,
                                                      config_dir):
        # The other side of the ownership rule: text the author wrote is
        # hashed as written, interpolated values included. Overriding the
        # section brings the number inside the preimage with it.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(
            bundle_dir, "extractor_workflow",
            "Extract the study. Each field is checked at most "
            "{max_checks_per_field} times.")
        bundle = load_config_bundle(bundle_dir)

        def prompt_hash(max_checks):
            return compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=max_checks,
                reference_lists=bundle.reference_lists)

        assert prompt_hash(2) != prompt_hash(3)


class TestEngineSourceDigest:
    """`engine_fp`'s meltiro half has to cover the sections.

    It is the only axis that can: no config fingerprint takes engine prose as a
    preimage, by design. If the digest hashed modules alone, rewording the
    engine's contract would move nothing anywhere, and two runs asking
    materially different questions would report the same engine.
    """

    def test_the_package_digest_covers_the_engine_prompts(self):
        import meltiro

        package_dir = prompt_partials.ENGINE_PROMPTS_DIR.parent
        assert meltiro.__file__ is not None
        modules_only = _hash_tree(package_dir, globs=("*.py",))
        everything = _hash_tree(package_dir)
        assert modules_only != everything, (
            "source_hash covers *.py alone, so an edit to an engine prompt "
            "section would move no fingerprint at all")

    def test_editing_a_section_moves_the_digest(self, tmp_path):
        package = tmp_path / "pkg"
        (package / "engine_prompts").mkdir(parents=True)
        (package / "mod.py").write_text("x = 1\n", encoding="utf-8")
        section = package / "engine_prompts" / "workflow.md"
        section.write_text("call the tools in order.\n", encoding="utf-8")

        before = _hash_tree(package)
        section.write_text("call the tools in any order.\n", encoding="utf-8")
        assert _hash_tree(package) != before

    def test_a_section_added_moves_the_digest(self, tmp_path):
        package = tmp_path / "pkg"
        (package / "engine_prompts").mkdir(parents=True)
        (package / "mod.py").write_text("x = 1\n", encoding="utf-8")
        before = _hash_tree(package)
        (package / "engine_prompts" / "extra.md").write_text(
            "one more section.\n", encoding="utf-8")
        assert _hash_tree(package) != before


# ---------------------------------------------------------------------------
# The load-time coverage warning
# ---------------------------------------------------------------------------

class TestCoverageWarning:
    """A bundle composing no engine section for a role loads, and says so.

    Not an error: a review may describe the engine in its own words, and
    refusing to load would make the sections mandatory. But a prompt that
    describes the engine WRONGLY is obeyed rather than corrected, so silence
    would be the worst of the three outcomes.
    """

    def test_the_fixture_bundle_warns_about_nothing(self, config_dir, capsys):
        load_config_bundle(config_dir)
        assert capsys.readouterr().err == ""

    def test_an_extractor_prompt_composing_nothing_warns(
            self, tmp_path, config_dir, capsys):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _uncompose(bundle_dir, "extractor_system")
        load_config_bundle(bundle_dir)
        err = capsys.readouterr().err
        assert "extractor_system.md" in err
        for name in _ROLE_SECTIONS["extractor_system"]:
            assert name in err
        assert "checker_system.md" not in err

    def test_composing_one_of_its_sections_is_enough(
            self, tmp_path, config_dir, capsys):
        # The question is whether the role's contract is described at all, not
        # whether every section of it is composed: a bundle is free to state
        # the evidence grammar or the recording conventions in its own words
        # while taking the engine's workflow verbatim.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _uncompose(bundle_dir, "extractor_system",
                   keep=("recording_conventions",))
        load_config_bundle(bundle_dir)
        assert capsys.readouterr().err == ""

    def test_a_checker_prompt_composing_nothing_warns(
            self, tmp_path, config_dir, capsys):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8").replace(
                "{include:meltiro:checker_briefing}",
                "Judge the value against the evidence."),
            encoding="utf-8")
        load_config_bundle(bundle_dir)
        err = capsys.readouterr().err
        assert "checker_system.md" in err
        assert "checker_briefing" in err

    def test_an_override_counts_as_composition(self, tmp_path, config_dir,
                                               capsys):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        for name in engine_section_names():
            _write_override(bundle_dir, name,
                            f"this review's own {name}.\n"
                            + ("{image_labels_list}"
                               if name == "recording_evidence" else ""))
        load_config_bundle(bundle_dir)
        assert capsys.readouterr().err == ""

    def test_a_disabled_checker_is_passed_over(self, tmp_path, config_dir,
                                               capsys):
        # The warning is about a model being underbriefed, so a stage that
        # places no call is not a case of it. `max_checks_per_field: 0` is the
        # checker's off switch, and the prompt it would have sent is then
        # nobody's briefing.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8").replace(
                "{include:meltiro:checker_briefing}",
                "Judge the value against the evidence."),
            encoding="utf-8")
        pipeline = bundle_dir / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8").replace(
                "max_checks_per_field: 2", "max_checks_per_field: 0"),
            encoding="utf-8")
        load_config_bundle(bundle_dir)
        assert capsys.readouterr().err == ""

    def test_the_extractor_is_warned_about_with_the_checker_off(
            self, tmp_path, config_dir, capsys):
        # The extractor always runs, so turning the checker off silences one
        # role's warning and not the other's.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _uncompose(bundle_dir, "extractor_system")
        pipeline = bundle_dir / "pipeline.yaml"
        pipeline.write_text(
            pipeline.read_text(encoding="utf-8").replace(
                "max_checks_per_field: 2", "max_checks_per_field: 0"),
            encoding="utf-8")
        load_config_bundle(bundle_dir)
        assert "extractor_system.md" in capsys.readouterr().err

    def test_the_warning_does_not_refuse_the_bundle(self, tmp_path,
                                                    config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _uncompose(bundle_dir, "extractor_system")
        assert load_config_bundle(bundle_dir).prompts_hash


# ---------------------------------------------------------------------------
# The wire render is never affected
# ---------------------------------------------------------------------------

def test_every_role_prompt_expands_fully_on_the_wire(config_dir):
    """Whatever the hash does, a model is sent the whole contract."""
    bundle = load_config_bundle(config_dir)
    references = bundle.reference_lists
    rendered = [
        build_system_message(
            [], system_prompt_path=bundle.extractor_system_path,
            reference_lists=references, mode=WIRE),
        build_review_system_message(
            [], system_prompt_path=bundle.review_system_path,
            reference_lists=references, mode=WIRE),
        build_checker_system_text(
            system_prompt_path=bundle.checker_system_path,
            reference_lists=references, predicates=PREDICATES,
            max_checks_per_field=2, mode=WIRE),
    ]
    for text in rendered:
        assert "{include:" not in text
        assert "{include_if:" not in text
