"""The engine's own prompts: what composes, what removes them, what they move.

Each role's system message is the engine's prompt for that role followed by the
config bundle's prompt file. The engine chooses its own half, so a review that
has no opinion about the machinery never has to describe it correctly for the
run to behave as documented, and a bundle cannot leave a model underbriefed by
saying nothing.

The three boundaries pinned here:

  - EXCLUSION. An engine prompt leaves a model's context in exactly two ways:
    it is a partial whose stage is off for this run, or the bundle overrides it
    with an empty file. There is no third.
  - SILENCE. A stage that will not run is not described to a model that cannot
    reach it. The checker is named only where the checker runs, and the
    reviewer stage is named to the extractor never — no feedback comes back
    from it, so the extractor has nothing to do with the fact that it exists.
  - OWNERSHIP. Engine text rides in `engine_fp`, through the source digest
    that hashes `engine_prompts/*.md` beside the package's modules. Rewording
    an engine prompt is an engine release, and every bundle's `prompts_hash`,
    `config_fp`, `checker_fp` and `review_fp` hold across it — which is what
    lets a consumer pin those numbers at all. An override
    (`prompts/partials/meltiro/NAME.md`) is text the config author wrote, so it
    rides in the config fingerprints like any other prose of theirs, empty
    overrides included: leaving one out is a methodological choice.

The engine's half ends with one transition sentence handing the role over to
the review's own briefing. It is composed rather than shipped as a file, and it
falls on the engine's side of both hashing boundaries above.
"""

import shutil

import pytest
import yaml

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
    CHECKER_SYSTEM,
    CHECKER_USER,
    ENGINE_ROLE_PROMPTS,
    EXPAND_ALL_BRANCHES,
    EXTRACTOR_SYSTEM,
    REVIEW_SYSTEM,
    compose_engine_prompt,
    composed_engine_names,
    engine_citations,
    engine_prompt_names,
    stage_predicates,
)
from meltiro.reference_lists import load_reference_lists
from meltiro.run_log import _hash_tree
from meltiro.template import load_template

# A stand-in for direktoro's provider-call identity block, which these tests
# hold fixed: what varies here is prompt text.
PINNED_CALL_IDENTITY = "call-identity-engine-prompts"

PREDICATES = stage_predicates(2, True)

# The tokens the engine substitutes, and the one citation a role prompt
# carries. A paragraph holding any of them does not reach a model as written,
# so the whole-file assertions below skip it and the no-token assertions cover
# it instead.
UNRENDERED = ("{image_labels_list}", "{max_checks_per_field}", "{include")


def _copy_bundle(tmp_path, config_dir):
    """A writable copy of the fixture bundle."""
    dest = tmp_path / "config"
    shutil.copytree(config_dir, dest)
    return dest


def _write_override(bundle_dir, name, text):
    """Ship a bundle override for engine prompt `name`."""
    override_dir = bundle_dir / "prompts" / "partials" / "meltiro"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / f"{name}.md").write_text(text, encoding="utf-8")
    return override_dir / f"{name}.md"


def _set_max_checks(bundle_dir, value):
    """Rewrite the bundle's own check budget: the toggle `checker` reads."""
    path = bundle_dir / "pipeline.yaml"
    pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
    pipeline["max_checks_per_field"] = value
    path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")


def _engine_text(name):
    return (prompt_partials.ENGINE_PROMPTS_DIR / f"{name}.md").read_text(
        encoding="utf-8").strip()


def _paragraphs(name):
    """The blocks of engine prompt `name` that reach a model as written."""
    return [p for p in _engine_text(name).split("\n\n")
            if p.strip() and not any(t in p for t in UNRENDERED)]


@pytest.fixture
def engine_dir(tmp_path, monkeypatch):
    """A writable copy of the engine's own prompt directory, swapped in.

    Lets a test edit an engine prompt without touching the installed package.
    `engine_prompt_names` and every resolution read the directory at call
    time, so the substitution is complete.
    """
    dest = tmp_path / "engine_prompts"
    shutil.copytree(prompt_partials.ENGINE_PROMPTS_DIR, dest)
    monkeypatch.setattr(prompt_partials, "ENGINE_PROMPTS_DIR", dest)
    return dest


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

    What is left is the engine's half: its prompt for the extractor and the
    transition sentence after it. A review may write whatever it likes in its
    own prompt file, so a claim about what the ENGINE tells a model has to be
    made against this rather than against the whole message.
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
# What ships, and what reads it
# ---------------------------------------------------------------------------

class TestThePromptsShip:
    def test_every_prompt_is_a_named_non_empty_file(self):
        names = engine_prompt_names()
        assert names, "the engine ships no prompts at all"
        for name in names:
            assert _engine_text(name), f"engine prompt {name} is empty"

    def test_every_file_is_a_role_prompt_or_a_partial_one_role_cites(self):
        # Being read by a role is what puts a file in front of a model, so a
        # file no role prompt names is inert: it ships, it is overridable, and
        # it reaches nothing. This is the gate — a file added to
        # `engine_prompts/` that nothing cites fails here rather than shipping
        # unread, and a citation of a file that no longer ships fails here too.
        placed = []
        for role in ENGINE_ROLE_PROMPTS:
            placed += composed_engine_names(
                role, predicates=EXPAND_ALL_BRANCHES)
        assert sorted(placed) == sorted(set(placed)), (
            "a file is read by two roles")
        assert set(placed) == set(engine_prompt_names())

    def test_every_role_reads_a_file_the_engine_ships(self):
        for role, name in ENGINE_ROLE_PROMPTS.items():
            assert name in engine_prompt_names(), role

    def test_only_the_declared_predicates_gate_a_citation(self):
        for name in ENGINE_ROLE_PROMPTS.values():
            for predicate, _ in engine_citations(_engine_text(name)):
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

    def test_a_role_prompt_cites_the_engine_namespace_only(self):
        # A role prompt composing a BUNDLE partial would invert the ownership
        # boundary the two halves are built on: the review's words would be
        # woven into the engine's contract rather than appended after it.
        for name in ENGINE_ROLE_PROMPTS.values():
            for _, cited in engine_citations(_engine_text(name)):
                assert prompt_partials.is_engine_name(cited), (name, cited)

    def test_a_cited_partial_cites_nothing_further(self):
        # Expansion is one level deep everywhere else; the engine's own files
        # are held to the same rule rather than exempted from it.
        roles = set(ENGINE_ROLE_PROMPTS.values())
        for name in engine_prompt_names():
            if name in roles:
                continue
            assert not engine_citations(_engine_text(name)), name

    def test_the_checker_reads_no_slot_the_checker_has_not(self):
        # The checker is sent no image labels, so `{image_labels_list}` in its
        # prompt would reach the model as a literal token. A bundle's override
        # is refused at load for this; the engine's own copy has nobody to
        # refuse it, so it is pinned here.
        for name in composed_engine_names(
                CHECKER_SYSTEM, predicates=EXPAND_ALL_BRANCHES):
            assert "{image_labels_list}" not in _engine_text(name)

    def test_the_scaffold_cites_only_slots_the_engine_fills(self, config_dir):
        # Substitution into the per-field scaffold is a plain `str.replace`
        # per known slot, so a misspelt one is not a render-time error: it
        # survives into the message and the checker reads `{field_pat}` where
        # the field path should be. A bundle's OVERRIDE of it is refused at
        # load for exactly this
        # (`config_bundle._validate_checker_placeholders`); the engine's own
        # copy has nobody to refuse it, so it is pinned here — over the shipped
        # file, and over the scaffold as it composes.
        composed = render_checker_user_template(
            load_config_bundle(config_dir).partials_dir,
            predicates=PREDICATES)
        for text in (_engine_text(CHECKER_USER), composed):
            unknown = sorted({n for n in _PLACEHOLDER_TOKEN.findall(text)
                              if n not in _CHECKER_USER_PLACEHOLDERS})
            assert not unknown, unknown


# ---------------------------------------------------------------------------
# Composition: the engine's half reaches the model whatever the bundle says
# ---------------------------------------------------------------------------

class TestEveryRoleIsBriefed:
    def test_the_extractor_reads_its_whole_prompt(self, config_dir):
        rendered = _extractor(load_config_bundle(config_dir))
        for block in _paragraphs("extractor"):
            assert block in rendered, block[:60]

    def test_the_reviewer_reads_its_whole_prompt(self, config_dir):
        rendered = _reviewer(load_config_bundle(config_dir))
        for block in _paragraphs("reviewer"):
            assert block in rendered, block[:60]

    def test_the_checker_reads_its_whole_prompt(self, config_dir):
        rendered = _checker(load_config_bundle(config_dir))
        for block in _paragraphs("checker"):
            assert block in rendered, block[:60]

    def test_the_extractor_reads_the_partial_its_prompt_cites(self,
                                                              config_dir):
        rendered = _extractor(load_config_bundle(config_dir))
        for block in _paragraphs("extractor_checker_feedback"):
            assert block in rendered, block[:60]

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

    def test_the_bundles_text_is_appended_after_the_engines(self, config_dir):
        bundle = load_config_bundle(config_dir)
        rendered = _extractor(bundle)
        engine_tail = _engine_text("extractor").splitlines()[-1]
        bundle_line = "<initial_check>"
        assert rendered.index(engine_tail) < rendered.index(bundle_line)

    def test_the_engines_half_starts_the_message(self, config_dir):
        rendered = _extractor(load_config_bundle(config_dir))
        assert rendered.startswith(_paragraphs("extractor")[0])

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
# Conditional composition
# ---------------------------------------------------------------------------

CHALLENGE_LINE = "**Checker feedback.**"


class TestAStageThatDoesNotRunIsNotDescribed:
    def test_the_checker_material_goes_with_the_checker(self, config_dir):
        bundle = load_config_bundle(config_dir)
        assert CHALLENGE_LINE in _extractor(bundle, max_checks_per_field=2)
        assert CHALLENGE_LINE not in _extractor(bundle,
                                                max_checks_per_field=0)

    def test_a_dropped_partial_leaves_no_gap(self, config_dir):
        bundle = load_config_bundle(config_dir)
        for text in (_extractor(bundle, max_checks_per_field=0),
                     _extractor(bundle, max_checks_per_field=2)):
            assert "\n\n\n" not in text

    def test_the_text_around_it_still_reads_in_order(self, config_dir):
        bundle = load_config_bundle(config_dir)
        text = _extractor(bundle, max_checks_per_field=0)
        assert text.index("**Validation feedback.**") \
            < text.index("**Mark complete") \
            < text.index("The extractor works within a finite tool-call")

    def test_the_toggle_moves_the_fingerprints_through_structure(
            self, config_dir):
        # The prompt component holds: the text that came and went is the
        # engine's, so no config preimage names it. `structure_hash` beside it
        # carries the toggle, so two runs differing in it never share a
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
        # Stronger than the passage going: a stage that does not run is not
        # named at all. The cited partial carries every checker fact, so the
        # text around it describes the machinery an extractor is always in and
        # nothing else.
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
        # text stops naming the checker.
        assert "Scope notes are not among them:" in engine_half

    def test_an_override_of_a_silenced_partial_is_not_this_runs_question(
            self, tmp_path, config_dir):
        # A partial its stage switched off reaches no model, so the bundle's
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


class TestTheExtractorIsNeverToldOfTheReviewer:
    """The reviewer stage is unmentionable to the extractor.

    The extractor receives no reviewer feedback: `mark_complete` ends its work
    and there is no work after it, both of which the engine states outright.
    So the reviewer is to the extractor what the checker is to a checker-off
    run — not a stage described in the conditional, a stage absent from the
    briefing. A sentence handing the record on would describe machinery the
    extractor can neither influence nor hear back from, and every word of an
    extractor's context is context it is not spending on the paper.

    "Review" meaning the systematic review itself — this review's criteria, the
    review's specifications — is a different word and stays.
    """

    @pytest.mark.parametrize("max_checks", [0, 2])
    @pytest.mark.parametrize("final_review", [False, True])
    def test_no_combination_names_the_stage(self, config_dir, max_checks,
                                            final_review):
        engine_half = _engine_half(
            load_config_bundle(config_dir),
            max_checks_per_field=max_checks,
            final_review=final_review).lower()
        assert "reviewer" not in engine_half
        assert "final review" not in engine_half

    def test_the_review_toggle_changes_nothing_for_the_extractor(
            self, config_dir):
        # The positive statement behind the absence: there is no conditional
        # left for the toggle to switch, so the two runs read identically.
        bundle = load_config_bundle(config_dir)
        assert _extractor(bundle, final_review=True) == \
            _extractor(bundle, final_review=False)

    def test_the_reviewer_is_still_described_to_itself(self, config_dir):
        # The stage exists; it is the extractor that is not told about it.
        assert "reviewer" in _reviewer(load_config_bundle(config_dir)).lower()


# ---------------------------------------------------------------------------
# Overrides: replace, or remove
# ---------------------------------------------------------------------------

class TestTheOverrideDirectoryIsEnumerated:
    """A file in `partials/meltiro/` overrides an engine prompt or fails the
    load.

    The filename is the whole of the wiring, and a near miss is silent in the
    worst way: the bundle ships text its author believes the model reads, the
    engine's own words go out instead, and every fingerprint agrees with both
    of them. Enumerating the directory is what turns that into an error, and
    it is why the comparison is against a LISTING rather than a probe — a
    case-insensitive filesystem answers `extractor.md.is_file()` for a file
    called `Extractor.md`, so a probe would pass this bundle on macOS and fail
    it on Linux.
    """

    def test_a_typo_in_the_stem_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extracter", "our extraction brief")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "extracter.md" in message
        for name in engine_prompt_names():
            assert name in message

    def test_the_wrong_case_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "Extractor", "our extraction brief")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "Extractor.md" in str(excinfo.value)

    def test_a_name_no_prompt_has_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "house_style", "our own block")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "house_style.md" in str(excinfo.value)

    def test_a_name_the_engine_no_longer_ships_is_refused(self, tmp_path,
                                                          config_dir):
        # The whole reason the directory is enumerated rather than probed. A
        # bundle carrying an override under a name this release does not ship
        # is told so at load, rather than running with the engine's words while
        # its author believes it is running with theirs.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "recording_notes", "our note policy")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "recording_notes.md" in message
        assert "extractor" in message

    def test_an_entry_that_is_not_a_file_is_refused(self, tmp_path,
                                                    config_dir):
        # The name matches a prompt the engine ships, so the spelling check
        # alone would pass it — and an override is read with `is_file()`, so
        # the run would render the engine's own words with a directory sitting
        # in the author's bundle under the name of the file replacing them.
        # That is the silent no-op the enumeration exists to prevent, reached
        # by the one route the name check does not cover.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        (bundle_dir / "prompts" / "partials" / "meltiro"
         / "extractor.md").mkdir(parents=True)
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "extractor.md is not a file" in message
        assert "'extractor'" in message

    def test_an_override_may_not_nest_an_include(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "reviewer",
                        "ours, plus {include:review_context}")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "nest" in str(excinfo.value).lower()

    def test_a_partial_override_may_not_nest_one_with_its_stage_off(
            self, tmp_path, config_dir):
        # The engine-side twin of the bundle-side rule
        # (`test_conditional_includes.py`): a defect a disabled stage hides is
        # a defect that surfaces on the day someone turns the stage back on,
        # in a bundle that has been loading cleanly for months. So the
        # override is read and checked whatever the toggles say — at load, and
        # again on the branch composition does not take.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _set_max_checks(bundle_dir, 0)
        _write_override(bundle_dir, "extractor_checker_feedback",
                        "ours, plus {include:review_context}")

        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "extractor_checker_feedback" in message
        assert "nest" in message.lower()

        with pytest.raises(ConfigBundleError) as excinfo:
            compose_engine_prompt(
                EXTRACTOR_SYSTEM, bundle_dir / "prompts" / "partials",
                predicates=stage_predicates(0, True))
        assert "nest" in str(excinfo.value).lower()

    def test_a_role_override_may_not_cite_the_engines_own_partial(
            self, tmp_path, config_dir):
        # An override is rendered literally, so a citation in one would reach
        # the model as a directive rather than as the passage it names.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(
            bundle_dir, "extractor",
            "Extract the study.\n"
            "{include_if:checker:meltiro:extractor_checker_feedback}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "nest" in str(excinfo.value).lower()


class TestANonEmptyOverrideReplacesTheText:
    def test_a_role_override_replaces_the_whole_engine_half(
            self, tmp_path, config_dir):
        # The consequence of one file per role, stated plainly: a bundle that
        # overrides a role's prompt has taken on the whole of that role's
        # engine briefing, the conditional passage included. Nothing of the
        # engine's is composed around it.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor",
                        "<brief>this review's own extraction brief</brief>")
        rendered = _extractor(load_config_bundle(bundle_dir),
                              max_checks_per_field=2)
        assert rendered.startswith("<brief>")
        assert CHALLENGE_LINE not in rendered
        for block in _paragraphs("extractor"):
            assert block not in rendered, block[:60]

    def test_a_partial_override_lands_where_the_citation_sits(
            self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor_checker_feedback",
                        "OUR CHALLENGE POLICY")
        rendered = _extractor(load_config_bundle(bundle_dir))
        before = "**Validation feedback.**"
        after = "**Mark complete"
        assert rendered.index(before) \
            < rendered.index("OUR CHALLENGE POLICY") \
            < rendered.index(after)

    def test_it_may_render_the_slots_the_text_did(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor",
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
        _write_override(bundle_dir, "checker",
                        "You judge one field.\n{image_labels_list}\n")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "image_labels_list" in str(excinfo.value)


class TestAnEmptyOverrideRemovesIt:
    """The only way to keep engine text out of a model's context.

    Nothing about how a bundle composes its prompts can drop it by accident,
    so removal is a file a config author writes on purpose — and it moves the
    fingerprints for the same reason any other edit to the question does.
    """

    @pytest.mark.parametrize("text", ["", "   \n\n  \n"])
    def test_empty_and_whitespace_both_remove_a_partial(self, tmp_path,
                                                        config_dir, text):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor_checker_feedback", text)
        rendered = _extractor(load_config_bundle(bundle_dir),
                              max_checks_per_field=2)
        assert CHALLENGE_LINE not in rendered
        assert "\n\n\n" not in rendered

    def test_it_removes_that_text_and_no_other(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor_checker_feedback", "")
        rendered = _extractor(load_config_bundle(bundle_dir))
        for block in _paragraphs("extractor"):
            assert block in rendered, block[:60]

    @pytest.mark.parametrize("text", ["", "   \n\n  \n"])
    def test_an_empty_role_override_leaves_the_bundles_text_alone(
            self, tmp_path, config_dir, text):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor", text)
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert rendered.startswith("<review_context>")
        assert not rendered.startswith("\n")
        assert CHALLENGE_LINE not in rendered

    def test_it_moves_the_config_fingerprints(self, tmp_path, config_dir):
        plain = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        emptied_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(emptied_dir, "reviewer", "")
        emptied = load_config_bundle(emptied_dir)
        assert plain.prompts_hash != emptied.prompts_hash
        assert plain.instrument_fp != emptied.instrument_fp

    def test_removal_and_replacement_are_different_bundles(self, tmp_path,
                                                           config_dir):
        # Both leave the engine's words out; they are not the same question,
        # so they must not fingerprint alike.
        removed_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(removed_dir, "reviewer", "")
        replaced_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(replaced_dir, "reviewer", "our review policy")
        assert load_config_bundle(removed_dir).prompts_hash != \
            load_config_bundle(replaced_dir).prompts_hash


class TestAnOverrideMovesItsRolesFingerprint:
    def test_prompts_hash_and_instrument_fp(self, tmp_path, config_dir):
        before = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "extractor", "our extraction brief")
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
        _write_override(overridden_dir, "extractor",
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
        _write_override(overridden_dir, "reviewer",
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
        _write_override(overridden_dir, "checker",
                        "you see one field, and the paper around its quotes.")
        assert fp(plain_dir) != fp(overridden_dir)

    def test_an_override_of_another_role_does_not_move_this_one(
            self, tmp_path, config_dir):
        # Per-role ownership: the reviewer's prompt is not the extractor's, so
        # a bundle rewriting one leaves the other stage's fingerprint alone and
        # the two stages stay separately comparable.
        def prompt_hash(bundle):
            return compute_prompt_config_hash(
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=2,
                reference_lists=bundle.reference_lists)

        plain = load_config_bundle(_copy_bundle(tmp_path, config_dir))
        overridden_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(overridden_dir, "reviewer", "ours.")
        assert prompt_hash(plain) == prompt_hash(
            load_config_bundle(overridden_dir))

    def test_an_identical_override_still_hashes_differently(
            self, tmp_path, config_dir):
        # Deliberate, and the point of the boundary rather than an artefact of
        # it. The two bundles read identically to a model, and they are not the
        # same instrument: one is pinned to whatever the engine's copy says in
        # the release it runs under, and the other to a copy its author owns
        # and an engine release cannot change. A hash that equated them would
        # report the first bundle as unchanged across a release that reworded
        # the text under it.
        plain_dir = _copy_bundle(tmp_path, config_dir)
        copied_dir = _copy_bundle(tmp_path / "second", config_dir)
        _write_override(copied_dir, "extractor_checker_feedback",
                        _engine_text("extractor_checker_feedback"))

        plain = load_config_bundle(plain_dir)
        copied = load_config_bundle(copied_dir)

        assert _extractor(plain) == _extractor(copied)
        assert plain.prompts_hash != copied.prompts_hash


class TestAnOverrideCountsWhenItsTextReachedAModel:
    """The rule the config preimage is built on, in both directions.

    An override is the author's own words and belongs to the config's identity
    for exactly as long as those words were sent. A file this run composed
    nowhere was no part of the question it asked, and a fingerprint that moved
    for one would report two runs putting a single question as two.

    A partial goes silent two ways, and the pair below is the second of them:
    its stage is off (pinned above, under conditional composition), or the
    bundle overrode the ROLE prompt that cites it, which replaces that role's
    whole engine half and takes the citation with it.
    """

    PARTIAL_OVERRIDE = "A challenge is advisory: revise, or let it stand."

    def _pair(self, tmp_path, config_dir, *, role_override):
        """Two bundles differing only in an override of the cited partial.

        `role_override` is written to BOTH when it is given, so the partial's
        override is the whole of the difference between them either way.
        """
        plain = _copy_bundle(tmp_path / "plain", config_dir)
        overriding = _copy_bundle(tmp_path / "overriding", config_dir)
        if role_override is not None:
            for bundle_dir in (plain, overriding):
                _write_override(bundle_dir, "extractor", role_override)
        _write_override(overriding, "extractor_checker_feedback",
                        self.PARTIAL_OVERRIDE)
        return plain, overriding

    def _measure(self, bundle_dir):
        """What the extractor read, and every value that claims to identify it.

        The checker is on throughout (`pipeline.yaml` says
        `max_checks_per_field: 2`), so no toggle is silencing anything here.
        """
        bundle = load_config_bundle(bundle_dir)
        return {
            "engine half": _engine_half(bundle),
            "prompts_hash": bundle.prompts_hash,
            "instrument_fp": bundle.instrument_fp,
            "config preimage": build_config_prompt_text(
                EXTRACTOR_SYSTEM,
                system_prompt_path=bundle.extractor_system_path,
                max_checks_per_field=2,
                reference_lists=bundle.reference_lists),
        }

    def test_a_partial_a_role_override_silenced_is_not_this_runs_question(
            self, tmp_path, config_dir):
        # The role override is the whole engine half, so `extractor.md`'s
        # citation is never read and the partial's own override is a file
        # nothing composes. The two bundles send one model one message, and
        # every number that names that message agrees.
        plain, overriding = self._pair(
            tmp_path, config_dir,
            role_override="<brief>this review's own extraction brief</brief>")
        measured = self._measure(overriding)
        assert self.PARTIAL_OVERRIDE not in measured["engine half"]
        assert measured == self._measure(plain)

    def test_the_same_override_counts_where_the_role_is_the_engines(
            self, tmp_path, config_dir):
        # The other half of the pair, and what makes the first a rule rather
        # than a hole: leave the role prompt to the engine and the citation is
        # read, so the same file reaches the model and moves everything.
        plain, overriding = self._pair(tmp_path, config_dir,
                                       role_override=None)
        before, after = self._measure(plain), self._measure(overriding)
        assert self.PARTIAL_OVERRIDE in after["engine half"]
        for name in before:
            assert before[name] != after[name], name


# ---------------------------------------------------------------------------
# The handover from the engine's half to the review's
# ---------------------------------------------------------------------------

class TestTheTransitionSentence:
    """One engine sentence between a role's own prompt and the bundle's text.

    A system message is two halves written by two authors, and read straight
    through the seam is invisible: the machinery stops being described and the
    review starts, mid-message, with nothing to mark it. The sentence marks it.

    It is emitted only where it is true. A bundle prompt file that is empty is
    promised no briefing, and a bundle that overrode the engine's half away
    gets no lone engine sentence in front of its own opening line.
    """

    def test_each_role_reads_it_between_the_two_halves(self, config_dir):
        bundle = load_config_bundle(config_dir)
        cases = (
            (_extractor(bundle), EXTRACTOR_BUNDLE_TRANSITION, "extractor"),
            (_reviewer(bundle), REVIEW_BUNDLE_TRANSITION, "reviewer"),
            (_checker(bundle), CHECKER_BUNDLE_TRANSITION, "checker"),
        )
        for rendered, transition, name in cases:
            engine_tail = _engine_text(name).splitlines()[-1]
            assert rendered.index(engine_tail) < rendered.index(transition) \
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

    def test_an_engine_half_overridden_away_gets_none_either(self, tmp_path,
                                                             config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(bundle_dir, "extractor", "")
        rendered = _extractor(load_config_bundle(bundle_dir))
        assert EXTRACTOR_BUNDLE_TRANSITION not in rendered
        assert rendered.startswith("<review_context>")

    def test_it_moves_no_config_fingerprint(self, tmp_path, config_dir,
                                            monkeypatch):
        # Compose-time framing, so it belongs to the engine on exactly the
        # terms its own prompts do: the model reads it, `engine_fp` carries
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

    def test_it_renders_from_the_engine_prompt(self, config_dir):
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
        # The scaffold gets the reference pass the three system prompts get, so
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

class TestTheBundleComposesNothingOfTheEngines:
    def test_an_engine_citation_is_refused(self, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:extractor}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "extractor_system.md" in message
        assert "partials/meltiro/extractor.md" in message
        # The file, the directive, and the two ways out. Not the list of what
        # the engine ships: the author is deleting a name here rather than
        # choosing one.
        assert "checker_user" not in message

    def test_a_conditional_engine_citation_is_refused(self, tmp_path,
                                                      config_dir):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        prompt = bundle_dir / "prompts" / "checker_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include_if:checker:meltiro:checker}\n",
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        assert "partials/meltiro/checker.md" in str(excinfo.value)

    def test_a_citation_of_a_name_the_engine_has_not_is_refused_too(
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
        # and a prompt citing an engine prompt. One load names both, so the
        # work is one editing pass rather than one per load.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        (bundle_dir / "prompts" / "checker_user_template.md").write_text(
            "## Field\n{field_path}\n", encoding="utf-8")
        prompt = bundle_dir / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8")
            + "\n{include:meltiro:extractor}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle_dir)
        message = str(excinfo.value)
        assert "checker_user_template.md" in message
        assert "{include:meltiro:extractor}" in message


class TestABrokenDirectiveIsQuotedAsTheFileSpellsIt:
    """An engine prompt is the one file whose directives the engine writes.

    Its errors are engine bugs, and the message is read while looking at the
    file: the fix is to find the placeholder and correct it. A message quoting
    a shortened form — the cited name with its `meltiro:` qualifier dropped —
    sends the reader searching for a string no file contains, and the search
    comes back empty on a file that really is broken.
    """

    DIRECTIVE = "{include_if:chekcer:meltiro:extractor_checker_feedback}"

    @pytest.fixture
    def misspelt(self, engine_dir):
        path = engine_dir / "extractor.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "{include_if:checker:", "{include_if:chekcer:"),
            encoding="utf-8")
        return path

    def test_the_load_quotes_it_whole(self, misspelt, config_dir):
        # The load asks what each role composes, so it evaluates the predicate
        # before any prompt is rendered.
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(config_dir)
        assert self.DIRECTIVE in str(excinfo.value)

    def test_so_does_composing_the_prompt(self, misspelt, config_dir):
        # And the composition evaluates it again to choose its branch, so the
        # two messages have to agree with each other and with the file.
        with pytest.raises(ConfigBundleError) as excinfo:
            compose_engine_prompt(
                EXTRACTOR_SYSTEM, config_dir / "prompts" / "partials",
                predicates=PREDICATES)
        assert self.DIRECTIVE in str(excinfo.value)


# ---------------------------------------------------------------------------
# The fingerprint boundary
# ---------------------------------------------------------------------------

class TestEditingAnEnginePrompt:
    """An engine edit moves the ENGINE axis and leaves the config axes alone.

    The edit is made against `engine_dir`, so it is a real one and the
    installed package is left alone.
    """

    @pytest.fixture
    def bundle_dir(self, tmp_path, config_dir):
        return _copy_bundle(tmp_path, config_dir)

    def _wire(self, bundle_dir):
        return _extractor(load_config_bundle(bundle_dir))

    def test_no_config_fingerprint_moves(self, engine_dir, bundle_dir,
                                         config_dir):
        before = _config_fingerprints(bundle_dir, config_dir)
        wire_before = self._wire(bundle_dir)
        for name in ("extractor", "extractor_checker_feedback", "checker",
                     "reviewer"):
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
        (engine_dir / "extractor_checker_feedback.md").write_text(
            "answer a challenge however you like.", encoding="utf-8")
        assert "answer a challenge however you like." in self._wire(bundle_dir)

    def test_an_overridden_prompt_is_untouched_by_the_engine_edit(
            self, engine_dir, tmp_path, config_dir):
        bundle_dir = _copy_bundle(tmp_path / "overriding", config_dir)
        _write_override(bundle_dir, "extractor", "our extraction brief")
        before = load_config_bundle(bundle_dir).prompts_hash
        (engine_dir / "extractor.md").write_text(
            "the engine's new wording.", encoding="utf-8")
        bundle = load_config_bundle(bundle_dir)
        assert bundle.prompts_hash == before
        rendered = _extractor(bundle)
        assert "our extraction brief" in rendered
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
        # An override is an ENGINE prompt, hashed as the author wrote it rather
        # than as it renders: the budget it states is still a run-structure
        # value, and `structure_hash` beside the prompt hash is where a
        # structure value belongs. The model reads the number either way.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        _write_override(
            bundle_dir, "extractor",
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
    """`engine_fp`'s meltiro half has to cover the engine's prompts.

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
            "would move no fingerprint at all")

    def test_editing_a_prompt_moves_the_digest(self, tmp_path):
        package = tmp_path / "pkg"
        (package / "engine_prompts").mkdir(parents=True)
        (package / "mod.py").write_text("x = 1\n", encoding="utf-8")
        engine_prompt = package / "engine_prompts" / "extractor.md"
        engine_prompt.write_text("call the tools in order.\n",
                                 encoding="utf-8")

        before = _hash_tree(package)
        engine_prompt.write_text("call the tools in any order.\n",
                                 encoding="utf-8")
        assert _hash_tree(package) != before

    def test_a_prompt_added_moves_the_digest(self, tmp_path):
        package = tmp_path / "pkg"
        (package / "engine_prompts").mkdir(parents=True)
        (package / "mod.py").write_text("x = 1\n", encoding="utf-8")
        before = _hash_tree(package)
        (package / "engine_prompts" / "extra.md").write_text(
            "one more.\n", encoding="utf-8")
        assert _hash_tree(package) != before
