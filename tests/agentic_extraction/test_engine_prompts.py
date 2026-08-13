"""The engine's prompt spines: what composes, what removes them, what they move.

Each role's system message is the engine's spine for that role followed by the
config bundle's prompt file. The engine chooses the spine, so a review that has
no opinion about the machinery never has to describe it correctly for the run
to behave as documented, and a bundle cannot leave a model underbriefed by
saying nothing.

The two boundaries pinned here:

  - EXCLUSION. A section leaves a model's context in exactly two ways: its
    stage is off for this run, or the bundle overrides it with an empty file.
    There is no third.
  - OWNERSHIP. Engine text rides in `engine_fp`, through the source digest
    that hashes `engine_prompts/*.md` beside the package's modules. Rewording
    a section is an engine release, and every bundle's `prompts_hash`,
    `config_fp`, `checker_fp` and `review_fp` hold across it — which is what
    lets a consumer pin those numbers at all. An override
    (`prompts/partials/meltiro/NAME.md`) is text the config author wrote, so it
    rides in the config fingerprints like any other prose of theirs, empty
    overrides included: leaving a section out is a methodological choice.

The engine's half ends with one transition sentence handing the role over to
the review's own briefing. It is composed rather than shipped as a section, and
it falls on the engine's side of both boundaries above.
"""

import shutil

import pytest

from meltiro import checker_prompts, prompt_builder, prompt_partials
from meltiro.checker import CheckerConfig
from meltiro.checker_prompts import (
    CHECKER_BUNDLE_TRANSITION,
    build_checker_system_text,
    build_checker_user_message,
    render_checker_user_template,
)
from meltiro.config_bundle import (
    _CHECKER_USER_PLACEHOLDERS,
    _PLACEHOLDER_TOKEN,
    load_config_bundle,
)
from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import (
    config_fingerprint,
    review_config_fingerprint,
    structure_hash,
)
from meltiro.prompt_builder import (
    EXTRACTOR_BUNDLE_TRANSITION,
    REVIEW_BUNDLE_TRANSITION,
    build_config_prompt_text,
    build_review_system_message,
    build_system_message,
    compute_prompt_config_hash,
    render_bundle_prompt_text,
)
from meltiro.prompt_partials import (
    ENGINE_SPINES,
    REVIEW_SYSTEM,
    engine_section_names,
    stage_predicates,
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


def _engine_text(name):
    return (prompt_partials.ENGINE_PROMPTS_DIR / f"{name}.md").read_text(
        encoding="utf-8").strip()


def _extractor(bundle, **kwargs):
    return build_system_message(
        [], system_prompt_path=bundle.extractor_system_path,
        reference_lists=bundle.reference_lists, **kwargs)


def _reviewer(bundle, **kwargs):
    return build_review_system_message(
        [], system_prompt_path=bundle.review_system_path,
        reference_lists=bundle.reference_lists, **kwargs)


def _checker(bundle, predicates=PREDICATES, **kwargs):
    return build_checker_system_text(
        system_prompt_path=bundle.checker_system_path,
        reference_lists=bundle.reference_lists, predicates=predicates,
        max_checks_per_field=2, **kwargs)


def _engine_half(bundle, *, max_checks_per_field=2, final_review=True):
    """The extractor's system message with the bundle's own text cut out.

    What is left is the engine's half: the spine and the transition sentence
    after it. A review may write whatever it likes in its own prompt file, so
    a claim about what the ENGINE tells a model has to be made against this
    rather than against the whole message.
    """
    predicates = stage_predicates(max_checks_per_field, final_review)
    appended = render_bundle_prompt_text(
        bundle.extractor_system_path, predicates=predicates,
        reference_lists=bundle.reference_lists,
        max_checks_per_field=max_checks_per_field)
    whole = _extractor(bundle, max_checks_per_field=max_checks_per_field,
                       final_review=final_review)
    assert appended and appended in whole, (
        "the bundle's appended text is not in the message verbatim, so "
        "subtracting it leaves something other than the engine's half")
    return whole.replace(appended, "")


def _config_fingerprints(bundle_dir, config_dir):
    """Every fingerprint a config bundle owns, in one dict.

    The whole config side of the ownership boundary, so a test asking whether
    some piece of engine text reaches any of them compares one value and gets
    an answer about all five.
    """
    template = load_template(config_dir / "extraction_template.yaml")
    references = load_reference_lists(config_dir / "reference")
    bundle = load_config_bundle(bundle_dir)
    checker = CheckerConfig(
        max_tokens=1024,
        checker_model="claude-sonnet-4-6",
        system_prompt_path=str(bundle.checker_system_path),
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
            build_config_prompt_text(
                REVIEW_SYSTEM,
                system_prompt_path=bundle.review_system_path,
                max_checks_per_field=2,
                reference_lists=bundle.reference_lists)),
    }


# ---------------------------------------------------------------------------
# What ships, and where it sits
# ---------------------------------------------------------------------------

class TestTheSectionsShip:
    def test_every_section_is_a_named_non_empty_file(self):
        names = engine_section_names()
        assert names, "the engine ships no prompt sections at all"
        for name in names:
            assert _engine_text(name), f"engine section {name} is empty"

    def test_every_section_sits_in_exactly_one_spine(self):
        # The spine is what puts a section in front of a model, so a file
        # nobody's spine names is inert: it ships, it is overridable, and it
        # reaches nothing. This is the gate — a section added to
        # `engine_prompts/` without a line in `ENGINE_SPINES` fails here rather
        # than shipping unread, and a spine naming a file that no longer ships
        # fails here too.
        placed = [name for spine in ENGINE_SPINES.values()
                  for name, _ in spine]
        assert sorted(placed) == sorted(set(placed)), (
            "a section is placed in two spines")
        assert set(placed) == set(engine_section_names())

    def test_every_spine_has_at_least_one_section(self):
        for role, spine in ENGINE_SPINES.items():
            assert spine, f"{role} composes no engine section"

    def test_only_the_declared_predicates_gate_a_section(self):
        for spine in ENGINE_SPINES.values():
            for _, predicate in spine:
                assert predicate is None or \
                    predicate in prompt_partials.PREDICATE_NAMES

    def test_they_are_declared_as_package_data(self):
        # A checkout finds them beside the modules whatever pyproject says; an
        # INSTALL only carries what package-data declares, and a wheel without
        # them cannot render a prompt at all. The suite runs from the source
        # tree, so nothing else here would notice.
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

    def test_no_checker_section_renders_a_slot_the_checker_has_not(self):
        # The checker is sent no image labels, so `{image_labels_list}` in one
        # of its sections would reach the model as a literal token. A bundle's
        # override is refused at load for this; the engine's own copy has
        # nobody to refuse it, so it is pinned here.
        for name, _ in ENGINE_SPINES[prompt_partials.CHECKER_SYSTEM]:
            assert "{image_labels_list}" not in _engine_text(name)

    def test_the_scaffold_cites_only_slots_the_engine_fills(self, config_dir):
        # Substitution into the per-field scaffold is a plain `str.replace`
        # per known slot, so a misspelt one is not a render-time error: it
        # survives into the message and the checker reads `{field_pat}` where
        # the field path should be. A bundle's OVERRIDE of the section is
        # refused at load for exactly this
        # (`config_bundle._validate_checker_placeholders`); the engine's own
        # copy has nobody to refuse it, so it is pinned here — over the shipped
        # file, and over the scaffold as it composes.
        composed = render_checker_user_template(
            load_config_bundle(config_dir).partials_dir,
            predicates=PREDICATES)
        for text in (_engine_text(prompt_partials.CHECKER_USER), composed):
            unknown = sorted({n for n in _PLACEHOLDER_TOKEN.findall(text)
                              if n not in _CHECKER_USER_PLACEHOLDERS})
            assert not unknown, unknown


# ---------------------------------------------------------------------------
# Composition: the spine reaches the model whatever the bundle says
# ---------------------------------------------------------------------------

class TestEveryRoleIsBriefed:
    def test_the_extractor_reads_its_whole_spine(self, config_dir):
        rendered = _extractor(load_config_bundle(config_dir))
        for name, predicate in ENGINE_SPINES["extractor_system"]:
            assert _engine_text(name).splitlines()[0] in rendered, name

    def test_the_reviewer_reads_its_whole_spine(self, config_dir):
        rendered = _reviewer(load_config_bundle(config_dir))
        for name, _ in ENGINE_SPINES["review_system"]:
            assert _engine_text(name).splitlines()[0] in rendered, name

    def test_the_checker_reads_its_whole_spine(self, config_dir):
        rendered = _checker(load_config_bundle(config_dir))
        for name, _ in ENGINE_SPINES["checker_system"]:
            assert _engine_text(name).splitlines()[0] in rendered, name

    def test_a_bundle_that_says_nothing_still_briefs_every_model(
            self, tmp_path, config_dir):
        # The ruling this design exists for: there is no way to compose a
        # prompt that leaves a model undescribed to. Three empty prompt files
        # are legal config, and all three models still read their contract.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        for name in ("extractor_system", "review_system", "checker_system"):
            (bundle_dir / "prompts" / f"{name}.md").write_text(
                "", encoding="utf-8")
        bundle = load_config_bundle(bundle_dir)
        assert "record_initial_check" in _extractor(bundle)
        assert "`mark_complete`" in _reviewer(bundle)
        assert "`record_verdict`" in _checker(bundle)

    def test_the_bundles_text_is_appended_after_the_spine(self, config_dir):
        bundle = load_config_bundle(config_dir)
        rendered = _extractor(bundle)
        spine_tail = _engine_text("recording_conventions").splitlines()[0]
        bundle_line = "<initial_check>"
        assert rendered.index(spine_tail) < rendered.index(bundle_line)

    def test_the_spine_starts_the_message(self, config_dir):
        rendered = _extractor(load_config_bundle(config_dir))
        assert rendered.startswith(_engine_text("extractor_role"))

    def test_no_placeholder_survives_on_the_wire(self, config_dir):
        bundle = load_config_bundle(config_dir)
        for text in (_extractor(bundle), _reviewer(bundle), _checker(bundle)):
            assert "{include:" not in text
            assert "{include_if:" not in text
            assert "{image_labels_list}" not in text
            assert "{max_checks_per_field}" not in text

    def test_the_extractor_supplies_the_image_label_list(self, config_dir):
        rendered = _extractor(load_config_bundle(config_dir))
        assert "no figures or tables were cropped" in rendered

    def test_the_extractor_is_told_the_budget(self, config_dir):
        rendered = _extractor(load_config_bundle(config_dir),
                              max_checks_per_field=3)
        assert "check budget for this run is 3" in rendered

    def test_the_checker_budget_is_substituted(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\nEach field is checked at most {max_checks_per_field} times.\n",
            encoding="utf-8")
        rendered = build_checker_system_text(
            system_prompt_path=(bundle_dir / "prompts" / "checker_system.md"),
            reference_lists=load_config_bundle(bundle_dir).reference_lists,
            predicates=PREDICATES, max_checks_per_field=3)
        assert "checked at most 3 times" in rendered


# ---------------------------------------------------------------------------
# Conditional sections
# ---------------------------------------------------------------------------

CHALLENGE_LINE = "**Checker feedback.**"
HANDOFF_LINE = "**Final review.**"


class TestAStageThatDoesNotRunIsNotDescribed:
    def test_the_checker_sections_go_with_the_checker(self, config_dir):
        bundle = load_config_bundle(config_dir)
        assert CHALLENGE_LINE in _extractor(bundle, max_checks_per_field=2)
        assert CHALLENGE_LINE not in _extractor(bundle,
                                                max_checks_per_field=0)

    def test_the_handoff_goes_with_the_reviewer(self, config_dir):
        bundle = load_config_bundle(config_dir)
        assert HANDOFF_LINE in _extractor(bundle, final_review=True)
        assert HANDOFF_LINE not in _extractor(bundle, final_review=False)

    def test_the_two_toggles_are_independent(self, config_dir):
        bundle = load_config_bundle(config_dir)
        text = _extractor(bundle, max_checks_per_field=0, final_review=True)
        assert HANDOFF_LINE in text and CHALLENGE_LINE not in text

    def test_a_dropped_section_leaves_no_gap(self, config_dir):
        bundle = load_config_bundle(config_dir)
        for text in (_extractor(bundle, max_checks_per_field=0),
                     _extractor(bundle, final_review=False)):
            assert "\n\n\n" not in text

    def test_the_sections_around_it_still_read_in_order(self, config_dir):
        bundle = load_config_bundle(config_dir)
        text = _extractor(bundle, max_checks_per_field=0)
        assert text.index("**Validation feedback.**") < \
            text.index("**Mark complete") < text.index(HANDOFF_LINE)

    def test_the_toggle_moves_the_fingerprints_through_structure(
            self, config_dir):
        # The prompt component holds: the sections that came and went are the
        # engine's, so no config preimage names them. `structure_hash` beside
        # it carries the toggle, so two runs differing in it never share a
        # `config_fp`.
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
        three_prompts, zero_fp = hashes(0)
        assert two_prompts == three_prompts
        assert two_fp != zero_fp

    def test_a_checker_off_extractor_is_never_told_a_checker_exists(
            self, config_dir):
        # Stronger than the section going: a stage that does not run is not
        # named at all. The conditional section carries every checker fact, so
        # the unconditional ones around it describe the machinery an extractor
        # is always in and nothing else.
        assert "checker" not in _engine_half(
            load_config_bundle(config_dir), max_checks_per_field=0).lower()

    def test_with_the_checker_on_it_is_described_in_full(self, config_dir):
        # The other half of the pair. Without it the assertion above would
        # pass just as well if the word had left the engine's vocabulary
        # altogether, and the extractor would be running blind to a stage that
        # answers it back.
        engine_half = _engine_half(load_config_bundle(config_dir),
                                   max_checks_per_field=2)
        assert "checker" in engine_half.lower()
        # The one fact that has nowhere else to live once the unconditional
        # sections stop naming the checker.
        assert "Scope notes are not among them:" in engine_half

    def test_an_override_of_a_silenced_section_is_not_this_runs_question(
            self, tmp_path, config_dir):
        # A section its stage switched off reaches no model, so the bundle's
        # own words for it are not part of what this run asks, and the preimage
        # says so: the overriding bundle and the plain one are the same
        # instrument with the checker off. Turn the checker on and the same
        # file counts, because now a model reads it.
        plain_dir = _copy_bundle(tmp_path, config_dir)
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "extractor_checker_feedback",
                        "A challenge is advisory: revise, or let it stand.")

        def prompt_hash(bundle_dir, max_checks):
            bundle = load_config_bundle(bundle_dir)
            return compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=max_checks,
                reference_lists=bundle.reference_lists)

        assert prompt_hash(plain_dir, 0) == prompt_hash(overridden_dir, 0)
        assert prompt_hash(plain_dir, 2) != prompt_hash(overridden_dir, 2)


# ---------------------------------------------------------------------------
# Overrides: replace in place, or remove
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
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "house_style", "our own block")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "house_style.md" in str(excinfo.value)

    def test_an_override_may_not_nest_an_include(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes",
                        "ours, plus {include:review_context}")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "nest" in str(excinfo.value).lower()


class TestANonEmptyOverrideReplacesInPlace:
    def test_the_override_is_what_the_model_reads(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes",
                        "<notes>this review's own note policy</notes>")
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert "this review's own note policy" in rendered
        assert "**Scope notes.**" not in rendered

    def test_it_lands_where_the_section_sat(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes", "OUR NOTE POLICY")
        rendered = _extractor(load_config_bundle(bundle_dir))
        before = _engine_text("recording_evidence").splitlines()[0]
        after = _engine_text("recording_conventions").splitlines()[0]
        assert rendered.index(before) < rendered.index("OUR NOTE POLICY") \
            < rendered.index(after)

    def test_it_may_render_the_slots_the_section_did(self, tmp_path,
                                                     config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_evidence",
                        "Quote what you cite.\n{image_labels_list}")
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert "{image_labels_list}" not in rendered
        assert "no figures or tables were cropped" in rendered

    def test_a_slot_the_checker_cannot_fill_is_refused(self, tmp_path,
                                                       config_dir):
        # The override is the bundle's own text, and it is held to the rule
        # the prompt it lands in imposes: the checker is sent no image labels,
        # so the token would reach the model verbatim.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "checker_briefing",
                        "You judge one field.\n{image_labels_list}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "image_labels_list" in str(excinfo.value)


class TestAnEmptyOverrideRemovesTheSection:
    """The only way to keep a section out of a model's context.

    Nothing about how a bundle composes its prompts can drop one by accident,
    so removal is a file a config author writes on purpose — and it moves the
    fingerprints for the same reason any other edit to the question does.
    """

    @pytest.mark.parametrize("text", ["", "   \n\n  \n"])
    def test_empty_and_whitespace_both_remove_it(self, tmp_path, config_dir,
                                                 text):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes", text)
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert _engine_text("recording_notes") not in rendered
        assert "**Scope notes.**" not in rendered

    def test_it_removes_that_section_and_no_other(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes", "")
        rendered = _extractor(load_config_bundle(bundle_dir))
        for name, _ in ENGINE_SPINES["extractor_system"]:
            if name == "recording_notes":
                continue
            assert _engine_text(name).splitlines()[0] in rendered, name

    def test_the_hole_closes(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes", "")
        assert "\n\n\n" not in _extractor(load_config_bundle(bundle_dir))

    def test_removing_every_section_leaves_the_bundles_text_alone(
            self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        for name, _ in ENGINE_SPINES["extractor_system"]:
            _write_override(bundle_dir, name, "")
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert rendered.startswith("<review_context>")
        assert not rendered.startswith("\n")

    def test_it_moves_the_config_fingerprints(self, tmp_path, config_dir):
        plain = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        emptied_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(emptied_dir, "recording_notes", "")
        emptied = load_config_bundle(emptied_dir)
        assert plain.prompts_hash != emptied.prompts_hash
        assert plain.instrument_fp != emptied.instrument_fp

    def test_removal_and_replacement_are_different_bundles(self, tmp_path,
                                                           config_dir):
        # Both leave the engine's words out; they are not the same question,
        # so they must not fingerprint alike.
        removed_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(removed_dir, "recording_notes", "")
        replaced_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(replaced_dir, "recording_notes", "our note policy")
        assert load_config_bundle(removed_dir).prompts_hash != \
            load_config_bundle(replaced_dir).prompts_hash


class TestAnOverrideMovesItsRolesFingerprint:
    def test_prompts_hash_and_instrument_fp(self, tmp_path, config_dir):
        before = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "recording_notes", "our note policy")
        after = load_config_bundle(overridden_dir)
        assert before.prompts_hash != after.prompts_hash
        assert before.instrument_fp != after.instrument_fp

    def test_config_fp(self, tmp_path, config_dir):
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

    def test_review_fp(self, tmp_path, config_dir):
        def fp(bundle):
            return review_config_fingerprint(
                PINNED_CALL_IDENTITY,
                build_config_prompt_text(
                    REVIEW_SYSTEM,
                    system_prompt_path=bundle.review_system_path,
                    max_checks_per_field=2,
                    reference_lists=bundle.reference_lists))

        plain = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "reviewer_workflow",
                        "Read the record and fix what is wrong.")
        assert fp(plain) != fp(load_config_bundle(overridden_dir))

    def test_checker_fp(self, tmp_path, config_dir):
        template = load_template(config_dir / "extraction_template.yaml")
        references = load_reference_lists(config_dir / "reference")

        def fp(bundle_dir):
            bundle = load_config_bundle(bundle_dir)
            cfg = CheckerConfig(
                max_tokens=1024,
                checker_model="claude-sonnet-4-6",
                system_prompt_path=str(bundle.checker_system_path),
            )
            return cfg.fingerprint(template, references,
                                   predicates=PREDICATES,
                                   max_checks_per_field=2)

        plain_dir = _copy_bundle(tmp_path, config_dir)
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "checker_briefing",
                        "you see one field, and the paper around its quotes.")
        assert fp(plain_dir) != fp(overridden_dir)

    def test_an_override_of_another_role_does_not_move_this_one(
            self, tmp_path, config_dir):
        # Per-role ownership: the reviewer's sections are not the extractor's,
        # so a bundle rewriting one leaves the other stage's fingerprint alone
        # and the two stages stay separately comparable.
        def prompt_hash(bundle):
            return compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=2,
                reference_lists=bundle.reference_lists)

        plain = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "reviewer_workflow", "ours.")
        assert prompt_hash(plain) == prompt_hash(
            load_config_bundle(overridden_dir))

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

        assert _extractor(plain) == _extractor(copied)
        assert plain.prompts_hash != copied.prompts_hash


# ---------------------------------------------------------------------------
# The handover from the engine's half to the review's
# ---------------------------------------------------------------------------

class TestTheTransitionSentence:
    """One engine sentence between a role's spine and the bundle's own text.

    A system message is two halves written by two authors, and read straight
    through the seam is invisible: the machinery stops being described and the
    review starts, mid-message, with nothing to mark it. The sentence marks it.

    It is emitted only where it is true. A bundle prompt file that is empty is
    promised no briefing, and a bundle that overrode every section away gets no
    lone engine sentence in front of its own opening line.
    """

    def test_each_role_reads_it_between_the_two_halves(self, config_dir):
        bundle = load_config_bundle(config_dir)
        cases = (
            (_extractor(bundle), EXTRACTOR_BUNDLE_TRANSITION,
             "recording_conventions"),
            (_reviewer(bundle), REVIEW_BUNDLE_TRANSITION,
             "reviewer_workflow"),
            (_checker(bundle), CHECKER_BUNDLE_TRANSITION, "checker_verdict"),
        )
        for rendered, transition, last_section in cases:
            spine_tail = _engine_text(last_section).splitlines()[-1]
            assert rendered.index(spine_tail) < rendered.index(transition) \
                < rendered.index("<review_context>")

    def test_an_empty_bundle_prompt_gets_none(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        for name in ("extractor_system", "review_system", "checker_system"):
            (bundle_dir / "prompts" / f"{name}.md").write_text(
                "", encoding="utf-8")
        bundle = load_config_bundle(bundle_dir)
        assert EXTRACTOR_BUNDLE_TRANSITION not in _extractor(bundle)
        assert REVIEW_BUNDLE_TRANSITION not in _reviewer(bundle)
        assert CHECKER_BUNDLE_TRANSITION not in _checker(bundle)

    def test_a_spine_overridden_away_gets_none_either(self, tmp_path,
                                                      config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        for name, _ in ENGINE_SPINES["extractor_system"]:
            _write_override(bundle_dir, name, "")
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert EXTRACTOR_BUNDLE_TRANSITION not in rendered
        assert rendered.startswith("<review_context>")

    def test_it_moves_no_config_fingerprint(self, tmp_path, config_dir,
                                            monkeypatch):
        # Compose-time framing, so it belongs to the engine on exactly the
        # terms a spine section does: the model reads it, `engine_fp` carries
        # it, and no config preimage names it. Asserted by rewording all three
        # and finding every config fingerprint where it was.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        before = _config_fingerprints(bundle_dir, config_dir)

        def wire():
            bundle = load_config_bundle(bundle_dir)
            return (_extractor(bundle), _reviewer(bundle), _checker(bundle))

        wire_before = wire()
        monkeypatch.setattr(prompt_builder, "EXTRACTOR_BUNDLE_TRANSITION",
                            "A later release hands the extractor over so.")
        monkeypatch.setattr(prompt_builder, "REVIEW_BUNDLE_TRANSITION",
                            "And the reviewer so.")
        monkeypatch.setattr(checker_prompts, "CHECKER_BUNDLE_TRANSITION",
                            "And the checker so.")
        # The rewording really did reach all three messages: without this the
        # equality below would hold for an edit that changed nothing.
        for after, was in zip(wire(), wire_before):
            assert after != was
        assert _config_fingerprints(bundle_dir, config_dir) == before


# ---------------------------------------------------------------------------
# The per-field checker scaffold
# ---------------------------------------------------------------------------

class TestTheCheckerUserScaffold:
    def _message(self, bundle):
        blocks = build_checker_user_message(
            "study.primary_aim",
            {"description": "The stated aim."},
            {"value": "An aim", "evidence": "<q>quoted</q>", "notes": None},
            "Summary: a synthetic paper",
            set(),
            partials_dir=bundle.partials_dir,
            predicates=PREDICATES,
        )
        return "".join(b.get("text", "") for b in blocks)

    def test_it_renders_from_the_engine_section(self, config_dir):
        text = self._message(load_config_bundle(config_dir))
        assert "## Field under review" in text
        assert "study.primary_aim" in text
        assert "An aim" in text
        for slot in ("{field_path}", "{field_description}", "{value}",
                     "{evidence_block}", "{identity_context}",
                     "{notes_block}", "{allowed_values_block}",
                     "{extraction_instruction_block}"):
            assert slot not in text

    def test_an_override_is_honoured(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "checker_user",
                        "FIELD {field_path}\nVALUE {value}")
        text = self._message(load_config_bundle(bundle_dir))
        assert text.startswith("FIELD study.primary_aim")
        assert "## Field under review" not in text

    def test_a_reference_list_the_override_cites_is_rendered_in(
            self, tmp_path, config_dir):
        # The scaffold gets the reference pass the three system spines get, so
        # a review that wants the canonical names beside the value under review
        # writes the citation and reads the list.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "checker_user",
                        "FIELD {field_path}\nVALUE {value}\n"
                        "{reference:gauge_list}")
        bundle = load_config_bundle(bundle_dir)
        blocks = build_checker_user_message(
            "study.primary_aim",
            {"description": "The stated aim."},
            {"value": "An aim", "evidence": "<q>quoted</q>", "notes": None},
            "Summary: a synthetic paper",
            set(),
            partials_dir=bundle.partials_dir,
            predicates=PREDICATES,
            reference_lists=bundle.reference_lists,
        )
        text = "".join(b.get("text", "") for b in blocks)
        assert "{reference:gauge_list}" not in text
        assert "Widget Durability Scale 9 (WDS-9)" in text

    def test_an_override_that_misspells_a_slot_is_refused(self, tmp_path,
                                                          config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "checker_user", "FIELD {field_pat}")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "field_pat" in str(excinfo.value)

    def test_an_override_moves_checker_fp(self, tmp_path, config_dir):
        template = load_template(config_dir / "extraction_template.yaml")
        references = load_reference_lists(config_dir / "reference")

        def fp(bundle_dir):
            bundle = load_config_bundle(bundle_dir)
            return CheckerConfig(
                max_tokens=1024, checker_model="claude-sonnet-4-6",
                system_prompt_path=str(bundle.checker_system_path),
            ).fingerprint(template, references, predicates=PREDICATES,
                          max_checks_per_field=2)

        plain_dir = _copy_bundle(tmp_path, config_dir)
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "checker_user",
                        "FIELD {field_path}\nVALUE {value}")
        assert fp(plain_dir) != fp(overridden_dir)


# ---------------------------------------------------------------------------
# What a bundle may no longer say
# ---------------------------------------------------------------------------

class TestTheBundleComposesNoEngineSection:
    def test_an_engine_citation_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:recording_notes}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "extractor_system.md" in message
        assert "recording_notes" in message
        assert "partials/meltiro/recording_notes.md" in message
        # The file, the directive, and the two ways out. Not the section
        # list: the author is deleting a name here rather than choosing one.
        assert "reviewer_workflow" not in message

    def test_a_conditional_engine_citation_is_refused(self, tmp_path,
                                                      config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include_if:checker:meltiro:checker_briefing}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "checker_briefing" in str(excinfo.value)

    def test_a_citation_of_a_name_no_section_has_is_refused_too(
            self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "review_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:house_style}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "house_style" in message
        assert "review_system.md" in message

    def test_a_checker_user_template_in_the_bundle_is_refused(
            self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        (bundle_dir / "prompts" / "checker_user_template.md").write_text(
            "## Field\n{field_path}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "checker_user_template.md" in message
        assert "partials/meltiro/checker_user.md" in message

    def test_a_bundle_carrying_both_defects_is_told_both_at_once(
            self, tmp_path, config_dir):
        # A bundle can hold both at once: the scaffold as a file of its own,
        # and a prompt citing an engine section. One load names both, so the
        # work is one editing pass rather than one per load.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        (bundle_dir / "prompts" / "checker_user_template.md").write_text(
            "## Field\n{field_path}\n", encoding="utf-8")
        prompt = bundle_dir / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:recording_notes}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "checker_user_template.md" in message
        assert "{include:meltiro:recording_notes}" in message


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
        return _copy_bundle(tmp_path, config_dir)

    def _wire(self, bundle_dir):
        return _extractor(load_config_bundle(bundle_dir))

    def test_no_config_fingerprint_moves(self, engine_dir, bundle_dir,
                                         config_dir):
        before = _config_fingerprints(bundle_dir, config_dir)
        wire_before = self._wire(bundle_dir)
        for name in ("extractor_workflow", "recording_notes",
                     "checker_briefing", "reviewer_workflow"):
            path = engine_dir / f"{name}.md"
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\nA sentence a later release added.\n", encoding="utf-8")
        # The edit really did reach the prompts: without this the equality
        # below would hold just as well for an edit that changed nothing.
        assert self._wire(bundle_dir) != wire_before
        after = _config_fingerprints(bundle_dir, config_dir)
        assert before == after

    def test_but_the_model_reads_the_edit(self, engine_dir, bundle_dir):
        (engine_dir / "recording_notes.md").write_text(
            "note whatever you like.", encoding="utf-8")
        assert "note whatever you like." in self._wire(bundle_dir)

    def test_an_overridden_section_is_untouched_by_the_engine_edit(
            self, engine_dir, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path / "overriding", config_dir)
        _write_override(bundle_dir, "recording_notes", "our note policy")
        before = load_config_bundle(bundle_dir).prompts_hash
        (engine_dir / "recording_notes.md").write_text(
            "the engine's new wording.", encoding="utf-8")
        bundle = load_config_bundle(bundle_dir)
        assert bundle.prompts_hash == before
        rendered = _extractor(bundle)
        assert "our note policy" in rendered
        assert "the engine's new wording." not in rendered


class TestAValueInterpolatedIntoEngineText:
    """The check budget: read by the extractor, absent from `prompt_hash`.

    `extractor_checker_feedback` states the budget, and the prompt hash covers
    the config's half of the prompt — so the NUMBER is outside the preimage.
    That is the boundary doing its job rather than a hole in it: the pair
    below is the whole statement. `prompt_hash` reports the text the author
    wrote, which two budgets share, and `config_fp` still separates them,
    because `structure_hash` beside it carries the toggle.
    """

    def test_a_bundle_that_states_it_itself_hashes_it(self, tmp_path,
                                                      config_dir):
        # The other side of the ownership rule: text the author wrote is
        # hashed as written, interpolated values included.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\nEach field is checked at most {max_checks_per_field} times.\n",
            encoding="utf-8")
        bundle = load_config_bundle(bundle_dir)

        def prompt_hash(max_checks):
            return compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=max_checks,
                reference_lists=bundle.reference_lists)

        assert prompt_hash(2) != prompt_hash(3)

    def test_an_override_is_hashed_as_it_was_written(self, tmp_path,
                                                     config_dir):
        # An override is a SECTION, hashed as the author wrote it rather than
        # as it renders: the budget it states is still a run-structure value,
        # and `structure_hash` beside the prompt hash is where a structure
        # value belongs. The model reads the number either way.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(
            bundle_dir, "extractor_workflow",
            "Extract the study. Each field is checked at most "
            "{max_checks_per_field} times.")
        bundle = load_config_bundle(bundle_dir)
        assert "checked at most 3 times" in _extractor(
            bundle, max_checks_per_field=3)

        def prompt_hash(max_checks):
            return compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=max_checks,
                reference_lists=bundle.reference_lists)

        assert prompt_hash(2) == prompt_hash(3)
        plain = load_config_bundle(_copy_bundle(tmp_path / "plain",
                                                config_dir))
        assert prompt_hash(2) != compute_prompt_config_hash(
            system_prompt_path=plain.extractor_system_path,
            max_checks_per_field=2,
            reference_lists=plain.reference_lists)


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
