"""Conditional prompt partials: `{include_if:PREDICATE:NAME}`.

An unconditional include describes a stage whether or not the stage runs. A
partial telling the extractor that a checker will challenge its fields is
false under `max_checks_per_field: 0`, and briefing a model on a stage that
does not exist is the same class of error as briefing it on the wrong one.
The engine already resolves both structure toggles before any prompt is
rendered; this is what lets a prompt act on them.

What the tests below pin, in order: the block appears when its stage is on
and vanishes when it is off; the omission closes its own gap, because the
rendered prompt is hashed and whitespace is therefore not cosmetic; the
partial must exist whether or not its branch is taken, so a typo cannot hide
behind a disabled stage until someone switches that stage back on; an
unknown predicate is refused rather than silently treated as false; and the
toggle reaches `prompts_hash`, so two configs differing only in a
conditional block are two different instruments.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import ConfigBundleError
from meltiro.extraction_record import ExtractionRecord
from meltiro.instrument import Instrument
from meltiro.prompt_builder import build_system_message
from meltiro.prompt_partials import (
    EXPAND_ALL_BRANCHES,
    included_names,
    stage_predicates,
    substitute_include_placeholders,
)
from meltiro.reference_lists import load_reference_lists
from meltiro.template import load_template

from .conftest import checker_trigger_orch

BLOCK = "A challenge is advisory: revise the value or overrule it."
PARTIAL = f"## When a field is challenged\n\n{BLOCK}\n"

REVIEWER_BLOCK = "A reviewer reads this record after the extractor stops."
REVIEWER_PARTIAL = f"## After the extractor\n\n{REVIEWER_BLOCK}\n"


@pytest.fixture
def bundle(tmp_path, config_dir):
    """A writable copy of the config fixture whose extractor prompt carries a
    checker-conditional block."""
    import shutil
    root = tmp_path / "cfg"
    shutil.copytree(config_dir, root)
    (root / "prompts" / "partials" / "checker_protocol.md").write_text(
        PARTIAL, encoding="utf-8")
    prompt = root / "prompts" / "extractor_system.md"
    prompt.write_text(
        prompt.read_text(encoding="utf-8").rstrip("\n")
        + "\n\n{include_if:checker:checker_protocol}\n\nEnd of brief.\n",
        encoding="utf-8")
    return root


def _render(root, *, max_checks_per_field, final_review=True):
    return build_system_message(
        load_template(root / "extraction_template.yaml"), [],
        system_prompt_path=root / "prompts" / "extractor_system.md",
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        reference_lists=load_reference_lists(root / "reference"))


def _set_pipeline(root, **keys):
    path = root / "pipeline.yaml"
    pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
    pipeline.update(keys)
    path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")


class TestTheBlockFollowsItsStage:
    def test_the_block_is_rendered_when_the_checker_runs(self, bundle):
        assert BLOCK in _render(bundle, max_checks_per_field=2)

    def test_the_block_is_absent_when_the_checker_is_off(self, bundle):
        assert BLOCK not in _render(bundle, max_checks_per_field=0)

    def test_the_rest_of_the_prompt_is_untouched_either_way(self, bundle):
        on = _render(bundle, max_checks_per_field=2)
        off = _render(bundle, max_checks_per_field=0)
        assert on.endswith("End of brief.")
        assert off.endswith("End of brief.")

    def test_an_omitted_block_closes_its_own_gap(self, bundle):
        # The rendered prompt is hashed into prompts_hash, so a blank-line run
        # left behind by an omitted block is a fingerprint difference, not a
        # cosmetic one.
        assert "\n\n\n" not in _render(bundle, max_checks_per_field=0)

    def test_the_review_predicate_is_independent_of_the_checker(self, bundle):
        prompt = bundle / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8").replace(
                "{include_if:checker:", "{include_if:review:"),
            encoding="utf-8")
        # The checker is off and the reviewer is on: a review-keyed block
        # must follow the reviewer, not the checker.
        assert BLOCK in _render(
            bundle, max_checks_per_field=0, final_review=True)
        assert BLOCK not in _render(
            bundle, max_checks_per_field=2, final_review=False)


class TestATypoCannotHideBehindAToggle:
    def test_a_missing_partial_fails_even_on_the_branch_not_taken(
            self, bundle):
        (bundle / "prompts" / "partials" / "checker_protocol.md").unlink()
        _set_pipeline(bundle, max_checks_per_field=0)
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle)
        assert "checker_protocol" in str(excinfo.value)

    def test_the_name_is_reported_at_load_time_whatever_the_toggle(
            self, bundle):
        text = (bundle / "prompts" / "extractor_system.md").read_text(
            encoding="utf-8")
        assert "checker_protocol" in included_names(text)

    def test_an_unknown_predicate_is_refused(self, bundle):
        prompt = bundle / "prompts" / "extractor_system.md"
        prompt.write_text(
            prompt.read_text(encoding="utf-8").replace(
                "{include_if:checker:", "{include_if:reviewr:"),
            encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle)
        assert "reviewr" in str(excinfo.value)

    def test_a_nested_include_inside_a_conditional_is_refused(self, bundle):
        (bundle / "prompts" / "partials" / "checker_protocol.md").write_text(
            "{include:inclusion_criteria}\n", encoding="utf-8")
        with pytest.raises(ConfigBundleError) as excinfo:
            load_config_bundle(bundle)
        assert "nested" in str(excinfo.value).lower()


class TestTheRendererRefusesToGuess:
    def test_a_conditional_with_no_predicates_raises(self, bundle):
        # An engine bug, not a config error: a render path that forgets to
        # pass the toggles would otherwise pick a branch silently, which is
        # the failure this feature exists to prevent.
        text = (bundle / "prompts" / "extractor_system.md").read_text(
            encoding="utf-8")
        with pytest.raises(ConfigBundleError):
            substitute_include_placeholders(
                text, bundle / "prompts" / "partials")

    def test_validators_see_every_branch(self, bundle):
        text = (bundle / "prompts" / "extractor_system.md").read_text(
            encoding="utf-8")
        expanded = substitute_include_placeholders(
            text, bundle / "prompts" / "partials",
            predicates=EXPAND_ALL_BRANCHES)
        assert BLOCK in expanded

    def test_stage_predicates_reads_the_toggles_one_way(self):
        assert stage_predicates(0, True) == {"checker": False, "review": True}
        assert stage_predicates(2, False) == {"checker": True, "review": False}


class TestTheToggleReachesTheFingerprint:
    def test_prompts_hash_moves_with_the_toggle(self, bundle):
        _set_pipeline(bundle, max_checks_per_field=2)
        on = load_config_bundle(bundle)
        _set_pipeline(bundle, max_checks_per_field=0)
        off = load_config_bundle(bundle)
        assert on.prompts_hash != off.prompts_hash

    def test_instrument_fp_moves_with_it(self, bundle):
        _set_pipeline(bundle, max_checks_per_field=2)
        on = load_config_bundle(bundle)
        _set_pipeline(bundle, max_checks_per_field=0)
        off = load_config_bundle(bundle)
        assert on.instrument_fp != off.instrument_fp

    def test_a_bundle_citing_no_conditional_hashes_the_same_every_load(
            self, config_dir):
        # The feature is additive: a bundle that names no conditional include
        # is hashed by one deterministic recipe, with nothing about the
        # toggles reaching a prompt at all.
        first = load_config_bundle(config_dir)
        second = load_config_bundle(config_dir)
        assert first.prompts_hash == second.prompts_hash
        assert first.instrument_fp == second.instrument_fp


class TestTheRunHashesWhatItRendered:
    """A CLI flag overrides `pipeline.yaml`, and the instrument axis follows
    the flag.

    `instrument_fp` names the question a run asked. The prompt component of it
    therefore has to describe the text that RENDERED, which under an override
    is not the text `pipeline.yaml`'s own toggles select. Nothing else covers
    this: the bundle's load-time hash has no run to read a flag from, and the
    orchestrator's other components are all toggle-independent."""

    def _instrument(self, root, *, max_checks_per_field, final_review=True):
        config = load_config_bundle(root)
        return Instrument(
            config, load_template(config.template_path),
            config.reference_lists,
            max_checks_per_field=max_checks_per_field,
            final_review=final_review, check_reviewer_edits=False)

    def _fp(self, root, *, max_checks_per_field, final_review=True):
        return self._instrument(
            root, max_checks_per_field=max_checks_per_field,
            final_review=final_review,
        ).fingerprint(tool_hash="tools", checker_context_chars=None)

    def test_an_override_hashes_the_prompts_the_override_renders(self, bundle):
        # pipeline.yaml says the checker runs; the run says it does not. The
        # checker-conditional block is therefore absent from every prompt the
        # run sends, and the component that claims to cover those prompts has
        # to be the one computed without it.
        _set_pipeline(bundle, max_checks_per_field=2)
        config = load_config_bundle(bundle)
        overridden = Instrument(
            config, load_template(config.template_path), config.reference_lists,
            max_checks_per_field=0, final_review=True,
            check_reviewer_edits=False)
        assert BLOCK not in overridden.render_extractor_system_text([])
        assert overridden.config.prompts_hash_for(overridden.predicates()) != \
            config.prompts_hash

    def test_two_bundles_brought_together_by_an_override_agree(self, bundle):
        # The comparison the axis exists for. Two bundles with byte-identical
        # prompt sources, differing only in pipeline.yaml's
        # max_checks_per_field, both RUN with the checker off. They render
        # identical prompts, so they are the same instrument and must carry the
        # same instrument_fp.
        _set_pipeline(bundle, max_checks_per_field=2)
        checker_on_in_file = self._fp(bundle, max_checks_per_field=0)
        _set_pipeline(bundle, max_checks_per_field=0)
        checker_off_in_file = self._fp(bundle, max_checks_per_field=0)
        assert checker_on_in_file == checker_off_in_file, (
            "two runs rendering byte-identical prompts under identical "
            "effective toggles record different instruments, so a comparison "
            "keyed on instrument_fp splits one arm in two on a difference no "
            "model ever saw.")

    def test_the_run_agrees_with_the_bundle_when_nothing_is_overridden(
            self, bundle):
        # The other direction: with no flag in play the run's value and the
        # bundle's printed value are the same number, which is what makes
        # `meltiro fingerprint` a preview of a run rather than a parallel
        # recipe.
        _set_pipeline(bundle, max_checks_per_field=2)
        config = load_config_bundle(bundle)
        instrument = Instrument(
            config, load_template(config.template_path), config.reference_lists,
            max_checks_per_field=2, final_review=True,
            check_reviewer_edits=False)
        assert instrument.config.prompts_hash_for(instrument.predicates()) == \
            config.prompts_hash

    def test_an_override_that_changes_the_prompts_still_moves_the_axis(
            self, bundle):
        # Recomputing must not flatten the axis: two runs of one bundle that
        # render DIFFERENT prompts are different instruments and still say so.
        _set_pipeline(bundle, max_checks_per_field=2)
        with_checker = self._fp(bundle, max_checks_per_field=2)
        without_checker = self._fp(bundle, max_checks_per_field=0)
        assert with_checker != without_checker


# ---------------------------------------------------------------------------
# One producer of the predicate map, for every path that renders or hashes
# ---------------------------------------------------------------------------

@pytest.fixture
def checker_bundle(tmp_path, config_dir):
    """A writable copy of the config fixture whose checker SYSTEM prompt and
    per-field TEMPLATE both cite a reviewer-conditional block."""
    import shutil
    root = tmp_path / "cfg-checker"
    shutil.copytree(config_dir, root)
    (root / "prompts" / "partials" / "after_the_extractor.md").write_text(
        REVIEWER_PARTIAL, encoding="utf-8")
    for name in ("checker_system.md", "checker_user_template.md"):
        path = root / "prompts" / name
        path.write_text(
            path.read_text(encoding="utf-8").rstrip("\n")
            + "\n\n{include_if:review:after_the_extractor}\n",
            encoding="utf-8")
    return root


def _instrument(root, *, final_review):
    config = load_config_bundle(root)
    return Instrument(
        config, load_template(config.template_path), config.reference_lists,
        max_checks_per_field=2, final_review=final_review,
        check_reviewer_edits=False)


def _checker_config(root):
    # A real registry id: checker_fp folds in the model's provider and
    # base_url, so an unregistered id fails model resolution.
    config = load_config_bundle(root)
    return CheckerConfig(
        checker_model="claude-sonnet-4-6",
        system_prompt_path=str(config.checker_system_path),
        user_prompt_template_path=str(config.checker_user_template_path))


class TestTheCheckerRendersAgainstTheRunsPipeline:
    """The checker sends two prompts and hashes both, and it holds no
    structure toggles of its own.

    Three paths read the pipeline's shape on the checker's behalf: the cached
    system prompt, the per-field template rendered for every call, and
    `checker_fp` over the pair. All three resolve their conditional blocks
    through `Instrument.predicates()`, the one place the toggles live, so
    there is no second value for them to fall out of step with. These pin
    that: a reviewer-keyed block in the checker's own prompts follows the
    run's reviewer, and the fingerprint follows the prompts.

    A stale checker prompt is the failure this shape exists to prevent, and
    it is silent: the checker would still answer, still return verdicts, and
    still record a fingerprint, having been briefed on a pipeline the
    extractor and reviewer were not.
    """

    def test_the_system_prompt_follows_the_reviewer(self, checker_bundle):
        on = _instrument(checker_bundle, final_review=True)
        off = _instrument(checker_bundle, final_review=False)
        assert REVIEWER_BLOCK in on.render_checker_system_text()
        assert REVIEWER_BLOCK not in off.render_checker_system_text()

    def test_the_per_field_template_follows_the_reviewer(self,
                                                         synthetic_template,
                                                         checker_bundle):
        # The message a real check is sent, built by the orchestrator: the
        # third render path, and the one furthest from the fingerprint.
        template_path = checker_bundle / "prompts" / "checker_user_template.md"

        def rendered(final_review):
            record = ExtractionRecord()
            record.study["primary_aim"] = {
                "value": "An aim", "evidence": "<q>quoted</q>", "notes": None}
            orch = checker_trigger_orch(
                synthetic_template, record, final_review=final_review)
            orch.config = SimpleNamespace(
                checker_user_template_path=str(template_path))
            calls, _ = orch._build_checker_calls(["study.primary_aim"])
            return "".join(
                block.get("text", "")
                for block in calls[0]["user_message_blocks"]
                if isinstance(block, dict))

        assert REVIEWER_BLOCK in rendered(True)
        assert REVIEWER_BLOCK not in rendered(False)

    def test_checker_fp_follows_the_reviewer(self, checker_bundle):
        cfg = _checker_config(checker_bundle)
        on = _instrument(checker_bundle, final_review=True)
        off = _instrument(checker_bundle, final_review=False)
        assert on.checker_fingerprint(cfg) != off.checker_fingerprint(cfg)

    def test_the_checker_config_holds_no_structure_toggles(self):
        # Ownership, pinned as a fact about the type rather than a habit of
        # its callers. The toggles are the instrument's; a copy kept here
        # would be a second value that only assignment statements hold in
        # step with the first.
        fields = set(CheckerConfig.__dataclass_fields__)
        assert not fields & {"max_checks_per_field", "final_review"}
