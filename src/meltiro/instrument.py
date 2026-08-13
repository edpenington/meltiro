"""The instrument: the question a run asks, apart from who is asked and about
what.

An instrument is everything the config author wrote plus what it implies. It is
the extraction template, the three role system prompts as they render, the tool
catalogues those prompts describe, the reference lists their values are drawn
from, and the pipeline structure (whether a checker runs, how many times it may
look at one field, whether a reviewer runs, whether the reviewer's own writes
are checked). Two runs that share an instrument asked the same question, and
their answers may be compared as answers to it.

A run is the product of three independent things (the same line
`fingerprint.instrument_fingerprint` draws):

  - the INSTRUMENT: what was asked. Owned here.
  - the CALL: which model was asked, under what sampling controls, with what
    output cap
    and what thinking spec, and whether it could see images at all.
  - the PAPER: which study, whose text and figures fill the user message.

Only the first is a methodological choice about the review; swapping the
extractor model or running a second paper leaves the instrument untouched,
which is what makes a model comparison or a two-paper batch interpretable.
The call and the paper enter here as ARGUMENTS and never as state: an
`Instrument` cannot be asked which model it ran under, because it did not
run under one.

The stage fingerprints are the deliberate exception: a `config_fp` or
`review_fp` blends the instrument with the call on purpose — they exist to
refuse a resume whose inputs moved, and a changed model is a changed input.
Each such method takes the call identity from the caller rather than
resolving one, so the instrument stays incapable of naming a model on its
own.

Nothing here reads the clock, the network, or the working tree. Given a config
bundle, every value this module produces is fixed, which is what lets a dry run
preview a real run's fingerprints exactly.
"""

from direktoro import model_supports_images
from meltiro.checker_prompts import build_checker_system_text
from meltiro.fingerprint import (
    reference_lists_hash,
    structure_hash,
    config_fingerprint as _cfg_fp,
    instrument_fingerprint as _instrument_fp,
    instrument_structure_hash as _instrument_structure,
    review_config_fingerprint as _review_fp,
    tool_set_hash as compute_tool_set_hash)
from meltiro.prompt_builder import (
    build_config_prompt_text, build_review_system_message,
    build_system_message, compute_prompt_config_hash)
from meltiro.prompt_partials import REVIEW_SYSTEM, stage_predicates
from meltiro.tools import all_tool_definitions, canonical_tool_set_json


class Instrument:
    """One review's extraction instrument, rendered and fingerprinted.

    Constructed from a config bundle: its template, its reference lists, and
    the pipeline structure the run resolved (pipeline.yaml's values as any CLI
    override left them, because an override changes the instrument and the
    run's value is the true one).

    `template` and `reference_lists` are passed alongside `config` rather than
    read back off it because they are already loaded and validated by the time
    an instrument exists, and loading a second copy here would be a second
    chance to disagree with the one the dispatcher and the validator use.
    """

    def __init__(self, config, template, reference_lists, *,
                 max_checks_per_field, final_review, check_reviewer_edits):
        # config: meltiro.config_bundle.ConfigBundle, holding the prompt paths
        # and the content fingerprints over the bundle as written.
        self.config = config
        self.template = template
        self.reference_lists = reference_lists
        self.max_checks_per_field = max_checks_per_field
        self.final_review = final_review
        self.check_reviewer_edits = check_reviewer_edits

    # ----------------------------------------------------------------------
    # Pipeline structure
    # ----------------------------------------------------------------------

    @property
    def checker_enabled(self):
        """True when the checker runs this run. It is off exactly when
        max_checks_per_field is 0; there is no separate flag."""
        return self.max_checks_per_field > 0

    def structure(self):
        """The run's pipeline structure, recorded in run.json for provenance.

        Every key here also rides in the fingerprinted structure block (see
        fingerprint.structure_hash), so two runs differing in any of them never
        share a stage fingerprint.
        """
        return {
            "checker": self.checker_enabled,
            "review": self.final_review,
            "max_checks_per_field": self.max_checks_per_field,
            "check_reviewer_edits": self.check_reviewer_edits,
        }

    def predicates(self):
        """The `{include_if:PREDICATE:NAME}` map every prompt renders against.

        The run has exactly one pipeline, so exactly one predicate map, and
        this is where it comes from: every prompt render and every prompt
        hash resolves its conditional blocks through this call. Stages that
        only place a call hold no copy to fall out of step with
        (`CheckerConfig` takes the map as an argument and stores nothing).
        """
        return stage_predicates(self.max_checks_per_field, self.final_review)

    def checker_context_chars(self, checker_config):
        """Characters of surrounding paper text the checker is shown on each
        side of a matched quote, or None when the checker is off.

        None rather than the configured number for an off checker, matching
        `checker_model` and `checker_fp`: with no checker there is no window,
        and recording a width would describe a message nothing sends.
        """
        if not self.checker_enabled:
            return None
        return checker_config.context_chars

    # ----------------------------------------------------------------------
    # Content hashes
    # ----------------------------------------------------------------------

    @property
    def template_hash(self):
        """The extraction template's content hash, as the template itself
        carries it. A component of every stage fingerprint and of
        `instrument_fp`, and recorded beside them so a reader can see which
        component moved."""
        return self.template["template_hash"]

    def tool_catalogue(self):
        """Every role's catalogue as canonical JSON: the human-readable form of
        the same bytes `tool_set_hash` digests.

        The catalogues themselves come off `meltiro.tools` given the template,
        and a caller handing a model its tools mid-run asks for them there. The
        instrument owns what is about the catalogue's IDENTITY: this rendering
        and the hash below.
        """
        return canonical_tool_set_json(self.template)

    def tool_set_hash(self):
        """Hash over every role's tool catalogue.

        One component covering both catalogues, so an edit to either moves the
        extractor's and the reviewer's fingerprints together.
        """
        return compute_tool_set_hash(all_tool_definitions(self.template))

    def reference_hash(self):
        """Content hash of the reference lists.

        Rides beside the rendered prompts because the prompts do not carry all
        of it: aliases are rendered nowhere, yet they decide what a tool call
        may write and how it canonicalises.
        """
        return reference_lists_hash(self.reference_lists)

    # ----------------------------------------------------------------------
    # The extractor's instrument
    # ----------------------------------------------------------------------

    def render_extractor_system_text(self, image_labels, image_captions=None):
        """The extractor's rendered system message, for `image_labels`.

        The label set is the caller's because it is a fact about the run, not
        the instrument: a text-only extractor is given an empty set and the
        prompt renders its none-available state, while an image-capable one
        gets the bundle's real labels and the captions beside them.
        """
        return build_system_message(
            image_labels,
            system_prompt_path=self.config.extractor_system_path,
            max_checks_per_field=self.max_checks_per_field,
            final_review=self.final_review,
            reference_lists=self.reference_lists,
            image_captions=image_captions,
        )

    def extractor_prompt_hash(self):
        """Paper-INDEPENDENT hash of the extractor's system prompt.

        The same prompt re-rendered with an empty image-label list, so two
        papers extracted under one config share a `config_fp`. The prompt the
        model is actually sent keeps the real labels.
        """
        return compute_prompt_config_hash(
            system_prompt_path=self.config.extractor_system_path,
            max_checks_per_field=self.max_checks_per_field,
            final_review=self.final_review,
            reference_lists=self.reference_lists,
        )

    def extractor_fingerprint(self, call_identity, *, prompt_hash, tool_hash,
                              supports_images):
        """Fingerprint the extraction stage: this instrument under one call.

        `supports_images` is the extractor model's declared image capability.
        It belongs to the call, not the instrument, and rides in the structure
        block because a text-only extractor is asked a materially different
        question: no figures, an empty label set, and a dispatcher that refuses
        an `<img>` citation of an image nothing ever sent.
        """
        return _cfg_fp(
            call_identity,
            prompt_hash, self.template_hash,
            tool_set_hash=tool_hash,
            structure_hash=structure_hash(
                self.max_checks_per_field,
                final_review=self.final_review,
                supports_images=supports_images,
                check_reviewer_edits=self.check_reviewer_edits,
            ),
            reference_hash=self.reference_hash(),
        )

    # ----------------------------------------------------------------------
    # The reviewer's instrument
    # ----------------------------------------------------------------------

    def render_review_system_text(self, image_labels, image_captions=None):
        """The reviewer's rendered system message, for `image_labels`.

        The single place it is built, so the copy captured into
        `diagnostics/instrument/` at session creation and the copy sent to the
        reviewer later in the run are the same string by construction rather
        than by two call sites agreeing.
        """
        return build_review_system_message(
            image_labels,
            system_prompt_path=self.config.review_system_path,
            max_checks_per_field=self.max_checks_per_field,
            final_review=self.final_review,
            reference_lists=self.reference_lists,
            image_captions=image_captions,
        )

    def review_config_prompt_text(self):
        """The config-owned identity of the reviewer's system prompt.

        Mirrors `extractor_prompt_hash`: paper-independent (no image labels
        reach it) and engine-free, so an engine release that rewords a
        reviewer section leaves `review_fp` where it was and a bundle's own
        edit moves it.
        """
        return build_config_prompt_text(
            REVIEW_SYSTEM,
            system_prompt_path=self.config.review_system_path,
            max_checks_per_field=self.max_checks_per_field,
            final_review=self.final_review,
            reference_lists=self.reference_lists,
        )

    def review_fingerprint(self, call_identity, *, review_model, tool_hash):
        """Fingerprint the final-review stage, or None when the reviewer is
        off (the review model is then not required, so it is not resolved
        through the registry; a null review_fp is recorded instead).

        The review system prompt component is the CONFIG's half, rendered with
        an EMPTY image-label list (mirroring `extractor_prompt_hash`) so two
        papers under one config share the fingerprint and the engine's own
        sections ride in `engine_fp` rather than here; reference-list
        substitution still applies, so editing a canonical name moves it. The
        reference-list CONTENT hash rides beside it for the part no prompt
        carries: aliases are rendered nowhere, yet they change what the
        reviewer's tool calls may write and how they canonicalise (see
        fingerprint.review_config_fingerprint).
        The tool-call cap rides in no fingerprint (see
        fingerprint.structure_hash). `review_max_tokens` and the reviewer's
        own `review_decoding` block ride in `call_identity`, so tuning either
        moves review_fp and only review_fp; a model that declares a sampling
        control refused is sent none of it, so for that model the block's
        value for it moves nothing — the fingerprint folds in what is sent,
        not what was configured.
        """
        if not self.final_review:
            return None
        review_system_text = self.review_config_prompt_text()
        review_structure = structure_hash(
            self.max_checks_per_field,
            supports_images=model_supports_images(review_model),
            check_reviewer_edits=self.check_reviewer_edits,
        )
        return _review_fp(
            call_identity, review_system_text,
            tool_set_hash=tool_hash, structure_hash=review_structure,
            reference_hash=self.reference_hash(),
        )

    # ----------------------------------------------------------------------
    # The checker's instrument
    # ----------------------------------------------------------------------

    def render_checker_system_text(self):
        """The checker's rendered system message.

        One string for the whole run, shared and cached across every per-field
        call. Built here for the same reason as the reviewer's above: the
        captured copy and the sent copy come off one function.
        """
        return build_checker_system_text(
            predicates=self.predicates(),
            max_checks_per_field=self.max_checks_per_field,
            system_prompt_path=self.config.checker_system_path,
            reference_lists=self.reference_lists,
        )

    def checker_fingerprint(self, checker_config):
        """Fingerprint the checker, or None when it is off.

        The checker is off exactly when max_checks_per_field is 0. With it off
        the checker model is not required (it may be None), so calling
        checker_config.fingerprint (which resolves the checker model through
        the registry) is not an option; a null checker_fp is recorded instead.
        This keeps two runs that differ only in whether the checker is on from
        sharing a checker fingerprint, and lets an extractor-only run proceed
        with no checker model at all.

        The structure predicates go in from here, the one place that holds
        them, so the prompts this hashes render exactly as the ones
        `render_checker_system_text` sends.
        """
        if not self.checker_enabled:
            return None
        return checker_config.fingerprint(
            self.template, self.reference_lists,
            predicates=self.predicates(),
            max_checks_per_field=self.max_checks_per_field)

    # ----------------------------------------------------------------------
    # The instrument's own identity
    # ----------------------------------------------------------------------

    def fingerprint(self, *, tool_hash, checker_context_chars):
        """`instrument_fp`: this instrument's identity, MODEL-FREE — the axis a
        model comparison holds fixed. Not engine-free: `tool_hash` carries the
        engine's own tool descriptions (see
        `fingerprint.instrument_fingerprint`).

        Folds the prompt bundle's content hash, the template, the tool
        catalogues, the pipeline structure, the reference lists, and the two
        checker-question knobs (context width, context fields), and nothing
        else.

        `checker_context_chars` is None for an off checker and passes through
        as None: no checker and a checker shown no surrounding text are two
        different questions, and the component says which on its own (see
        fingerprint.instrument_fingerprint). It matches the null `checker_fp`
        a run records.

        The prompt component is recomputed HERE, against this run's own
        predicates, NOT taken from the bundle's load-time `prompts_hash`. The
        two differ exactly when a CLI flag overrode `pipeline.yaml`, and then
        the bundle's value describes text this run never rendered; folding it
        in would report two runs asking the identical question as different
        instruments. Recomputing cannot collide the other way, because
        `structure_hash` beside it pins the effective toggles outright.
        """
        return _instrument_fp(
            self.config.prompts_hash_for(self.predicates()),
            self.template_hash,
            tool_set_hash=tool_hash,
            structure_hash=_instrument_structure(
                self.max_checks_per_field,
                final_review=self.final_review,
                check_reviewer_edits=self.check_reviewer_edits,
            ),
            reference_hash=self.reference_hash(),
            checker_context_chars=checker_context_chars,
            checker_context_fields=self.template.get("checker_context_fields"),
        )
