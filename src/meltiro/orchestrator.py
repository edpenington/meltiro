"""Main loop driver for the agentic extraction pipeline.

Lifecycle of one run (each phase below is implemented in this module):

  start / resume
       v
  EXTRACTOR TURN  (tools enabled)
   - dispatch each tool_use; the dispatcher is deterministic and offline
   - fan the CHECKER out over the fields that call just applied, in
     parallel, and merge its challenges into the same tool result
   - log applied / failed events, feed the tool results back
   - mark_complete validates -> FINAL REVIEW (or FINALISE)
   - else loop back (capped at max_tool_calls)
       v
  FINAL REVIEW  (fresh context, whole extraction output)
   - its own bounded tool loop: inspect with the view tools, revise
     with the editing tools, results fed back each turn
   - with check_reviewer_edits on, its writes are checked on the same terms
   - mark_complete -> FINALISE
   - surrender / a bound fires -> FINALISE (failed_validation)
       v
  FINALISE: save extraction output + meta + run_log

The checker is NOT a stage. It is a probabilistic extension of the
deterministic validator, running per field inside the tool call: a field that
passes validation, holds a non-null value, and carries evidence goes to one
narrow checker call — re-asked once, as a correction, if it comes back with no
verdict — and any challenge comes back in the same tool result as the
validation errors. A challenge is advisory. The model overrules it by ignoring
it (no reply, no tool call, no counter-argument); revising the field sends the
new value for another check only while that field's `max_checks_per_field`
budget lasts. No challenge blocks mark_complete; a field still
challenged when its budget runs out is recorded in `meta.checker_diagnostics`
while the run finalises `complete`.

The final reviewer is told nothing of what the checker did to the extraction it
is reading. Its message carries the paper, the figures, and the extraction
output, and nothing else; with `check_reviewer_edits` on it sees a challenge on
a field it just wrote, in that write's own tool result, and still nothing more.
That toggle is also what decides whether its engine prompt names a checker: a
reviewer whose writes are not checked is briefed on no such stage.

This module owns the high-level state machine and the two things only a running
study has: which model each role calls, and which paper is in front of it.
Everything the config author wrote — template, rendered system prompts, tool
catalogues, reference lists, pipeline structure and the identity hashes over
them — belongs to `meltiro.instrument.Instrument`, which this module holds and
delegates to. It also delegates to prompt_builder, tools (ToolDispatcher),
checker, session, run_log, and checker_prompts.
"""

import json
import os
import shutil
import sys
from pathlib import Path

from direktoro import is_known_model, model_info
from direktoro import model_supports_images
from direktoro import MissingAPIKey, build_adapter
from direktoro import supports_forced_tool_choice
from meltiro.template import iter_fields

from meltiro.checker import (
    CheckerConfig, _build_checker_adapter, reported_cost_or_raise,
    run_checker_batch)
from meltiro.checker_prompts import (
    build_checker_user_message,
    build_record_context, render_degraded_identity_context,
    render_record_identity_context, render_study_identity_context,
    system_message_blocks as checker_sys_blocks,
)
from meltiro.bundle import normalise_label, read_transcription
from meltiro.diagnostics import DEFAULT_DIAGNOSTICS, validate_diagnostics
from meltiro.errors import AgenticExtractionError, truncation_report
from meltiro.extraction_record import ROLE_REVIEW
from meltiro.fingerprint import (
    bundle_fingerprint,
    call_fingerprint as _call_fp,
    run_fingerprint as _run_fp)
from meltiro.instrument import Instrument
from direktoro import (
    create_message_with_retry, is_known_model, resolved_decoding_params)
# The refusal that is about the CALLER rather than the call: an exhausted
# balance or spend cap, a key that is absent, wrong or revoked, a key not
# entitled to this model or endpoint. It is the one provider failure a run can
# pause on, because it is the one whose fix is outside the process and leaves
# the extraction untouched. Everything else direktoro raises stays a plain
# `ProviderError` and stays terminal — a malformed request is equally
# unfixable by waiting, and a run that paused on one would resume into it for
# ever. See `_pause_on_provider_account`.
from direktoro import ProviderAccountError
from meltiro.rates import cache_write_split
from meltiro.prompt_builder import (
    EMPTY_ASSISTANT_PLACEHOLDER, EXTRACTOR_TOOL_REPROMPT,
    REVIEW_TOOL_REPROMPT,
    build_initial_user_blocks,
    build_review_user_blocks,
    image_label_text,
    message_figure_labels,
    render_message_text,
    system_message_blocks as extractor_sys_blocks,
)
from meltiro.template import load_template
from meltiro.run_entry import append_session_entry
from meltiro.run_log import current_engine_fp, engine_identity, git_state
from meltiro.session import Session, result_to_model_text
from meltiro.statuses import CONSIDERED_STATUSES, VALIDATED_STATUSES
from meltiro.tools import (
    MUTATING_TOOLS, ToolDispatcher, canonical_checker_tool_json,
    get_tool_definitions)
from meltiro.validators import missing_required_fields


DEFAULT_MAX_TOOL_CALLS = 100
# The final reviewer's own tool-call budget (`max_review_tool_calls` in
# pipeline.yaml). Operational, not methodology: rides in no fingerprint,
# recorded per segment in run.json (`caps`). See `_review_loop`.
DEFAULT_MAX_REVIEW_TOOL_CALLS = 30
# Cap on consecutive tool-free extractor turns: each is re-prompted, but a
# model that never calls a tool cannot loop (and bill) forever.
DEFAULT_MAX_CONSECUTIVE_TEXT_ONLY_TURNS = 3
# Sibling guard for failing tool calls: once this many consecutive calls fail
# with an identical (tool name, sorted error codes) signature, the extractor
# loop stops rather than burning the budget re-billing the same dead end.
# Engine policy, deliberately NOT a pipeline.yaml key.
DEFAULT_MAX_CONSECUTIVE_IDENTICAL_FAILURES = 5
# Owned by `meltiro.config_bundle` so reading a bundle need not import the
# provider layer; re-exported here because callers and tests import it from
# this module.
from meltiro.config_bundle import DEFAULT_MAX_CHECKS_PER_FIELD  # noqa: E402


# How a decoding block spells a thinking field, and the `Thinking` attribute
# behind each. direktoro's `split_decoding_config` reads the same four names
# out of a role's block; this is the way back, for reporting what the operator
# wrote in the words they wrote it in.
THINKING_KEY_PREFIX = "thinking_"
THINKING_FIELDS = ("mode", "effort", "budget_tokens", "display")


def _indent_continuation(entry, indent="  "):
    """`entry` with every line after the first indented, empty ones included.

    What makes a multi-line entry one entry in a file of them: the reader's
    rule is that an entry starts at column 0, which holds whatever the entry's
    later lines contain. Indenting an empty line to whitespace rather than
    leaving it empty is the point of `including empty ones` — a blank line
    inside an exhibit's printed footnote would otherwise look like the gap
    between two exhibits.
    """
    first, *rest = entry.split("\n")
    return "\n".join([first] + [f"{indent}{line}" for line in rest])


def _configured_thinking(thinking):
    """A `Thinking` spec as the `{key: value}` a decoding block would have
    written, over the fields it actually sets.

    Empty for None (a role that said nothing about thinking) and for a spec
    whose every field is None. An unset field is omitted rather than nulled,
    on the same terms as an unspecified sampling control: no opinion is not a
    value.
    """
    if thinking is None:
        return {}
    return {f"{THINKING_KEY_PREFIX}{field}": getattr(thinking, field)
            for field in THINKING_FIELDS
            if getattr(thinking, field, None) is not None}


def _required_cap(param, value):
    """`value` as a role's output-token cap, or refuse before it can spend.

    A cap has no default: the number bounds what one call may spend and what
    it may answer within, and one this engine invented would sit in a run
    record looking exactly like one the operator wrote. Type-strict for the
    same reason the caps in a bundle are — a bool is not a budget, and a float
    or a numeric string would coerce into a number nobody chose.

    Demanded per ENABLED role, in the constructor, so every entry point is
    refused on the same terms before a session directory exists or a token is
    billed. A stage that makes no calls needs no budget for them.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AgenticExtractionError(
            f"{param} must be a positive integer, got {value!r}. It is the "
            f"output-token cap for that role's calls and has no default. A "
            f"role whose stage is off states none: the reviewer's is off at "
            f"final_review=False, the checker's at max_checks_per_field=0.")
    return value


class Orchestrator:
    """Run one study end-to-end."""

    # Class defaults so a prompt renderer can read these off any Orchestrator,
    # including one a test builds with `__new__`. A real run never falls back
    # to them: `__init__` assigns all three unconditionally.
    final_review = True
    max_checks_per_field = DEFAULT_MAX_CHECKS_PER_FIELD
    # No cards by default: unpriced roles state no cost (see rates.py).
    rates = {}
    # The run's cached checker adapter (see `_checker_adapter`). A class
    # default for the same reason as the three above — a test-built
    # Orchestrator has run no `__init__` — and safe to share because every
    # write is an assignment onto the INSTANCE.
    _cached_checker_adapter = None
    # The stopping state `_finalise` reads (see `__init__` for what each
    # means). Class defaults for the same reason and with the same safety: an
    # `__init__`-less Orchestrator that reaches finalisation finds "not yet
    # finalised, no error recorded" rather than an AttributeError, and both are
    # only ever assigned onto the instance.
    _terminal_status = None
    _error_message = None

    def __init__(self, config, bundle, out_dir, *,
                 extractor_model,
                 checker_config=None,
                 review_model,
                 max_tool_calls=DEFAULT_MAX_TOOL_CALLS,
                 max_review_tool_calls=DEFAULT_MAX_REVIEW_TOOL_CALLS,
                 max_checks_per_field=DEFAULT_MAX_CHECKS_PER_FIELD,
                 check_reviewer_edits=False,
                 sampling=None,
                 review_sampling=None,
                 thinking=None,
                 review_thinking=None,
                 extractor_max_tokens=None,
                 review_max_tokens=None,
                 final_review=True,
                 decoding_specified=None,
                 rates=None,
                 diagnostics=DEFAULT_DIAGNOSTICS,
                 dry_run=False):
        # config: meltiro.config_bundle.ConfigBundle, the review-specific
        # schema, prompts, and reference lists.
        # bundle: meltiro.bundle.PaperBundle, one paper's inputs.
        # out_dir: run root; session dirs land at
        #   {out_dir}/{study_id}/sessions/... and the run log at
        #   {out_dir}/run_log.json.
        self.config = config
        self.bundle = bundle
        self.study_id = bundle.study_id
        self.out_dir = Path(out_dir)

        # Loaded from the config bundle so nothing is CWD-relative. Reference
        # lists were already validated at bundle load.
        self.template = load_template(config.template_path)
        self.reference_lists = config.reference_lists

        self.extractor_model = extractor_model
        self.checker_config = checker_config or CheckerConfig.from_env()
        # The checker's prompt path comes from the config bundle, so the
        # checker fingerprint reflects the config actually used.
        self.checker_config.system_prompt_path = str(
            config.checker_system_path)
        self.review_model = review_model
        self.max_tool_calls = max_tool_calls
        # The reviewer's own budget, counted over the review conversation only:
        # review tool calls do not touch `meta.tool_call_count` (the
        # extractor's count).
        self.max_review_tool_calls = max_review_tool_calls
        # TOTAL checker calls one field may receive across the session. 0
        # turns the checker off (no separate flag): no checker calls, and the
        # checker model + checker_fp are not required.
        self.max_checks_per_field = max_checks_per_field
        # Whether the checker also gates the REVIEWER's tool calls, on the
        # same per-field terms and out of the same per-field budget. Off by
        # default.
        self.check_reviewer_edits = check_reviewer_edits
        # Structure toggle: False means no final-review call, and the review
        # model + review_fp are not required. Every structure toggle moves
        # config_fp (see structure_hash), so a structure change refuses resume
        # via the drift gate.
        self.final_review = final_review
        # Per-role rate cards, `{role: meltiro.rates.Rates}` over "extractor",
        # "checker", "review". A role absent or mapped to None runs unpriced:
        # tokens recorded, no dollar figure, run total withheld (rationale in
        # rates.py; rates reach no fingerprint). The checker's card comes from
        # here too.
        self.rates = dict(rates or {})
        self.checker_config.rates = self.rates.get("checker")
        # How much of the deterministic record to keep (see
        # meltiro.diagnostics). Operational only — changes which files are
        # written, nothing any model is asked — so it rides in no fingerprint.
        # Validated here so an unknown level fails before any spend.
        self.diagnostics = validate_diagnostics(diagnostics)
        # The sampling controls the operator specified, per role, as a
        # `{name: value}` mapping over `direktoro.SAMPLING_PARAMS`. Each role
        # stands alone: `sampling` is the EXTRACTOR's and `review_sampling` the
        # reviewer's, neither taken from the other, and the checker carries its
        # own on CheckerConfig. Each role's mapping reaches only its own stage
        # fingerprint, via that stage's resolved decoding params.
        #
        # No control has a default VALUE here. An unspecified one is not sent
        # and the model's own default applies — a fact about the provider,
        # which may move under an unchanged config, and one this engine
        # neither pins nor pretends to record. Specifying it is how a run fixes
        # it.
        #
        # "Specified nothing" is None on every role, matching what
        # `direktoro.split_decoding_config` returns for a block that names no
        # sampling control and what `CheckerConfig.sampling` already carried.
        # An empty mapping beside a None would be a second spelling of one
        # state, and the two would have to be kept equivalent everywhere they
        # are read.
        self.sampling = dict(sampling) if sampling else None
        self.review_sampling = (
            dict(review_sampling) if review_sampling else None)
        # Per-role thinking (`direktoro.Thinking`, or None). None says nothing
        # and leaves the model's own default thinking behaviour in force.
        self.thinking = thinking
        self.review_thinking = review_thinking
        # The operator's decoding block per role, verbatim, as the config
        # bundle wrote it: `{role: {key: value}}`, and a role that wrote none
        # is absent. Recorded in run.json beside the params the wire actually
        # carried, which are not the same document — a model that refuses a
        # sampling control is sent none of it, and without this the artefact
        # cannot tell a value the operator wrote and the model dropped from a
        # value the operator never wrote. Carried, never read: nothing on the
        # call path consults it, because what is SENT is resolved from
        # `sampling` / `thinking` / the caps below.
        self.decoding_specified = {
            role: dict(block)
            for role, block in (decoding_specified or {}).items()
            if block}
        # One output cap per ENABLED role (see `_required_cap`), the
        # checker's included: `CheckerConfig` alone is constructible without
        # one (its own entry points re-refuse for direct callers), but an
        # Orchestrator that will run checks demands it here, where the
        # refusal lands before a session directory exists — not after the
        # whole extractor loop has been billed.
        self.extractor_max_tokens = _required_cap(
            "extractor_max_tokens", extractor_max_tokens)
        self.review_max_tokens = (
            _required_cap("review_max_tokens", review_max_tokens)
            if self.final_review else review_max_tokens)
        if self.checker_enabled:
            _required_cap("checker_max_tokens", self.checker_config.max_tokens)
        self.dry_run = dry_run
        # Engine constants, not config (see the constants above).
        self.max_consecutive_text_only_turns = (
            DEFAULT_MAX_CONSECUTIVE_TEXT_ONLY_TURNS)
        self.max_consecutive_identical_failures = (
            DEFAULT_MAX_CONSECUTIVE_IDENTICAL_FAILURES)
        # Session-global, strictly increasing turn counter. Turn ids group a
        # turn's events into one assistant/user message pair on replay, so
        # they must never collide across extractor loops or the feedback turns
        # between them. Seeded from the event log on resume.
        self._turn_counter = 0

        # Paper inputs derived once from the bundle. `figures` is the
        # (label, png_bytes) list the prompt builders consume; image_labels
        # is the lower-cased stem set used for image-citation matching.
        self.paper_text = bundle.text
        # The ARTICLE's crops, in label order. A supplement's are attached in
        # its own section rather than here (see `self.supplements`), because
        # the message has to say which document an exhibit came out of.
        self.figures = [(label, path.read_bytes())
                        for label, path in bundle.figures.items()]
        # The maps below are the WHOLE bundle's, article and supplements
        # together. A label is unique across a bundle by the format's own
        # rule, so one flat map resolves any citation without ambiguity, and
        # every consumer of them — the dispatcher validating `<img>`, the
        # checker attaching the crop it names — asks only which file a label
        # is, never which document it sits in. Grouping is the message's job
        # and it is done there.
        #
        # Normalise exactly as the tool dispatcher does (strip + lower) so
        # image-citation matching agrees on both sides.
        self.image_labels = {label.strip().lower()
                             for label in bundle.all_figures()}
        # Exhibit captions, keyed the same normalised way. Paper input, like
        # the labels: rides in the prompts, and in no fingerprint (prompt_hash
        # and review_fp render an empty label list, so two papers under one
        # config share a fingerprint).
        self.image_captions = {
            label.strip().lower(): caption
            for label, caption in bundle.all_exhibits().items()}
        # The footnote each exhibit prints, where the manifest records one, on
        # the same key. Paper input on the same terms as the captions, and
        # carried separately because only some exhibits have one.
        self.image_notes = {
            label.strip().lower(): note
            for label, note in bundle.all_exhibit_notes().items()}
        # The transcription each exhibit carries, where the bundle supplies
        # one, read once here and keyed the same way. Read at construction
        # like the crops beside them, so one run reads each file once and the
        # message, the recorded prompt and the checker's copy are the same
        # bytes rather than three reads of a directory that could move under
        # them.
        self.image_tables = {
            normalise_label(label): read_transcription(path)
            for label, path in bundle.all_tables().items()
        }
        # Where each crop IS, on that same key: the map a role's citation is
        # resolved through when the crop has to be attached again away from
        # the opening message, which is the checker's whole case. It is built
        # here, from `all_figures()`, for the reason the three above are —
        # a checker handed the article's map alone accepts a supplement's
        # label as a citation, says the crop is attached, and attaches
        # nothing, because the label set and the file map disagreed about
        # which document they covered.
        self.image_figures = {
            label.strip().lower(): path
            for label, path in bundle.all_figures().items()}
        # Each supplement as the message builders take one: its name, the
        # title the paper prints for it, its prose where it printed any, and
        # its own crops read here beside the article's. Built once, in name
        # order, so the sections a message carries and the sections a
        # recorded prompt renders are the same list in the same order.
        self.supplements = [
            {"name": supplement.name,
             "title": supplement.title,
             "text": supplement.text,
             "figures": [(label, path.read_bytes())
                         for label, path in supplement.figures.items()]}
            for supplement in bundle.supplements.values()
        ]

        # Refuse a call the registry says cannot be made, in the constructor,
        # so every entry point (CLI, resume, programmatic) fails before a
        # session directory exists or a token is billed. Runs before any
        # other registry lookup, so an unknown model id fails HERE, naming
        # its role, rather than as a bare registry error from whichever
        # lookup happens to run first.
        self._refuse_unworkable_decoding()

        # Declared per-role image capability, from the registry. Every
        # extractor gets no image parts and no label blocks: its message states
        # that none accompany the study, and its dispatcher validates against
        # an empty label set, so citing an image it never saw fails as an
        # unknown label. Checker and reviewer guard on their own role models.
        self.extractor_supports_images = model_supports_images(
            self.extractor_model)

        # Cost / token accumulators across all calls in the session.
        # `_input_tokens` matches the `usage.input_tokens` a response reports:
        # it counts ONLY tokens charged at full price (cache misses + the
        # bit of the prompt that isn't cacheable). cache_creation_tokens
        # (write, 1.25x) and cache_read_tokens (read, 0.1x) are tracked
        # separately so a consumer can quantify prompt-cache savings.
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0
        self._cost_usd = 0.0
        # Two latches gate whether the running sum may be STATED as this run's
        # cost; neither is recoverable from the sum, and `recorded_cost` needs
        # both. `_cost_unpriced`: some call could not be costed (a direct call
        # with no rate card), so the sum is partial and the run states no
        # total. Gateway-routed calls never set it — their charge comes back
        # on the response, no rates needed. `_cost_counted`: some call WAS
        # costed; without it a run with no card and no calls would state
        # $0.00, which reads as "free" rather than "nothing priced anything".
        self._cost_unpriced = False
        self._cost_counted = False
        # A third state the sum cannot hold: some call WAS priced, and the
        # price covers less than the calls behind it — a gateway-served
        # checker call whose response carried no charge (see
        # `checker._spend`). The sum is then a floor rather than a total, and
        # `_unreceipted_calls` says how many calls it does not cover. Kept
        # apart from `_cost_unpriced` because the two report different facts:
        # nothing could price that call, versus something priced it and the
        # receipt did not arrive.
        self._cost_incomplete = False
        self._unreceipted_calls = 0
        # The same five meters again PER ROLE: `{role: counters}`, built by
        # `_role_usage`. Each role carries its own `unpriced` / `counted`
        # latches on the same terms as the run-wide pair, so an unpriced
        # extractor withholds its figure while a priced checker's stands.
        self._usage_by_role = {}

        # Per-call work products built up at session start (see prepare()).
        self.session = None
        self.extraction_record = None
        self.dispatcher = None
        self.system_text = None
        self.initial_user_blocks = None
        self.messages = []  # conversation
        # Per-field checker call counts: field_path -> int. The budget is per
        # FIELD and spans the whole session; a resumed segment rebuilds this
        # from the event log (see _reconstruct_check_counts), not from meta,
        # so there is no second source of truth to drift.
        self._check_counts = {}
        # One-shot latches so each study-identity warning fires at most once
        # per segment: `_identity_degradation_warned` for the fall-back to
        # title + DOI, `_summary_mismatch_advised` for the stderr advisory
        # when manifest and role:summary diverge. The run.json mismatch
        # warning is evaluated once at finalisation from persisted state, so
        # a resume resetting these does not affect it.
        self._identity_degradation_warned = False
        self._summary_mismatch_advised = False
        # Stopping is two phases (see `run`). This latches the status the
        # session was persisted with, so a second `_finalise` — which a fault
        # raised anywhere after the first one lands in run()'s catch-all
        # produces — answers with what is already on disk instead of writing a
        # second terminal status and a second run-log entry. One session, one
        # ledger entry per terminal transition.
        self._terminal_status = None
        # The checker's adapter, built on first use and kept for the run (see
        # `_checker_adapter`). None until then; None is also what an unset key
        # resolves to, so the attribute is not a presence test.
        self._cached_checker_adapter = None
        # The composed message of the failure that ended the run, when one did.
        # Carried from wherever the error was caught to `_finalise`, which puts
        # it in run.json (`error_message`) and in the run-log entry's
        # `validation_errors`, so the sentence an operator needs is in the run
        # record rather than only in the event log.
        self._error_message = None

    # ----------------------------------------------------------------------
    # Pre-spend feasibility
    # ----------------------------------------------------------------------

    def _role_decoding_inputs(self):
        """`{role: (model, max_tokens, sampling, thinking)}` for every ENABLED
        role, in call order.

        The four arguments `direktoro.resolved_decoding_params` takes, gathered
        in one place because three separate readers ask for them — the
        pre-spend feasibility refusal, the resolved-params record run.json
        carries, and the inert-parameter warning — and each role keeps its
        pieces somewhere different (the checker's on `CheckerConfig`, the
        reviewer's under `review_*`). Three copies of that gathering would let
        one reader ask about a call the others never make.

        A disabled stage is absent rather than nulled: it makes no calls, so
        there is nothing to resolve and nothing to say about what it would
        send.
        """
        roles = {"extractor": (self.extractor_model, self.extractor_max_tokens,
                               self.sampling, self.thinking)}
        if self.checker_enabled:
            roles["checker"] = (self.checker_config.checker_model,
                                self.checker_config.max_tokens,
                                self.checker_config.sampling,
                                self.checker_config.thinking)
        if self.final_review:
            roles["review"] = (self.review_model, self.review_max_tokens,
                               self.review_sampling, self.review_thinking)
        return roles

    def _refuse_unworkable_decoding(self):
        """Refuse, before any spend, a role whose call cannot be made.

        One `direktoro.resolved_decoding_params` per ENABLED role — the same
        call the adapters make and the stage fingerprints fold in, so this
        asks the real question rather than a copy of it: does this model's
        endpoint accept this block at this cap. A value outside the model's
        documented band, a mode or an effort level it does not have, a
        sampling control on a request that also turns thinking on where the
        endpoint takes only one of the two, a cap a thinking call cannot
        answer within — each raises an `AgenticExtractionError` naming the
        role whose block it came from.

        A disabled stage is skipped: it makes no calls, so there is no shape
        to reject and no cap to size. An enabled role whose model the
        registry does not know is refused here too, naming the role — no
        call can be resolved for an unknown id, and every later lookup would
        raise a bare registry error with no role attached. A RETIRED id
        resolves: the registry's provenance rule is that a past run's model
        must keep resolving, and refusing retired ids for NEW runs is the
        caller's gate (the CLI applies it before building an Orchestrator).
        """
        roles = [(role, *inputs)
                 for role, inputs in self._role_decoding_inputs().items()]
        for role, model, max_tokens, sampling, thinking in roles:
            if not model or not is_known_model(model):
                raise AgenticExtractionError(
                    f"{role} role ({role}_model: {model!r}): the registry "
                    f"knows no such model id, so no call can be resolved "
                    f"for it.")
            try:
                resolved_decoding_params(model, max_tokens=max_tokens,
                                         sampling=sampling, thinking=thinking)
            except ValueError as exc:
                raise AgenticExtractionError(
                    f"{role} role ({role}_model: {model!r}): {exc}") from exc

    # ----------------------------------------------------------------------
    # The instrument
    # ----------------------------------------------------------------------

    @property
    def instrument(self):
        """The extraction instrument this run asks (`meltiro.instrument`).

        Derived from the run's own values on each access, so there is no
        second copy of the config, template, reference lists or structure
        toggles to drift. Holds no model and no paper; building one just binds
        references.
        """
        return Instrument(
            self.config, self.template, self.reference_lists,
            max_checks_per_field=self.max_checks_per_field,
            final_review=self.final_review,
            check_reviewer_edits=self.check_reviewer_edits,
        )

    # ----------------------------------------------------------------------
    # Pipeline structure
    # ----------------------------------------------------------------------

    @property
    def checker_enabled(self):
        """True when the checker runs this run: off exactly when
        max_checks_per_field is 0, no separate flag."""
        return self.max_checks_per_field > 0

    def _structure_dict(self):
        """The run's pipeline structure, as run.json records it."""
        return self.instrument.structure()

    def _checker_context_chars(self):
        """Characters of surrounding paper text the checker is shown on each
        side of a matched quote, or None when the checker is off."""
        return self.instrument.checker_context_chars(self.checker_config)

    # ----------------------------------------------------------------------
    # Image capability (per role)
    # ----------------------------------------------------------------------

    def _supplements_for(self):
        """The supplement sections a role is sent: all of them.

        A supplement reaches a role as a document — its prose, its crops and
        its transcriptions together — and every role can read all three,
        because `_startup_capability_guard` refuses a run whose models cannot.
        """
        return self.supplements

    def _warn_if_truncated(self, role, max_tokens, response):
        """Say so, loudly, when a role's response stopped on `max_tokens`.

        Truncation has to be named where it happens. Unnamed, a cut-off
        extractor turn looks like a tool-free turn, gets re-prompted as one,
        and burns to a stall guard with the cause nowhere in the record.

        The message names the cap and the pipeline.yaml key that set it, which
        is the line an operator would edit. Written to both stderr and
        `meta.warnings` (deduplicated by `Session.add_warning`).
        """
        if getattr(response, "stop_reason", None) != "max_tokens":
            return
        message = truncation_report(max_tokens, f"{role}_max_tokens")
        print(f"WARNING: {message}", file=sys.stderr)
        if self.session is not None:
            self.session.add_warning(message)

    def _warn_engine_drift(self, segment_engine_fp):
        """Warn when a resumed segment runs under a different engine than the
        one the session's recorded identity names.

        `meta.engine_fp` is fixed at session creation and `meta.run_fp` is
        derived from it, so both describe the engine that STARTED the run; a
        resume after an upgrade of either engine package, or an edit to their
        source, executes the remainder under different code. Disclosed rather
        than refused, so the cap-hit recovery (pause, raise the cap, resume)
        survives an engine that moved in between. The `resumed` event carries
        this segment's full identity. `Session.add_warning` dedups an exact
        repeat, so resuming repeatedly onto one drifted engine records the
        message once.
        """
        recorded = self.session.meta.get("engine_fp")
        if recorded is None or segment_engine_fp == recorded:
            return
        self._record_warning(
            f"engine-drift: this segment runs under engine_fp "
            f"{segment_engine_fp}, while the session records "
            f"{recorded} from when it started, so meta.run_fp names the "
            f"engine behind only part of this run. The `resumed` events in "
            f"the event log carry each segment's own version, commit and "
            f"engine fingerprint; read those before comparing this run "
            f"against another on run_fp."
        )

    # ----------------------------------------------------------------------
    # Entry points
    # ----------------------------------------------------------------------

    def _build_fingerprints(self):
        """Render the extractor system prompt and compute the run's whole
        identity record, without creating or persisting anything.

        Shared by prepare_new_session (which then builds the Session) and
        dry_run_report (which builds no Session), so a dry run computes
        byte-identical fingerprints to the real run it previews: one recipe,
        no parallel copy to drift. An edit to any one stage's inputs moves
        that stage's fingerprint and only that one.
        """
        instrument = self.instrument
        ext_figures, ext_image_labels = self.figures, self.image_labels
        system_text = instrument.render_extractor_system_text()
        # Paper-INDEPENDENT prompt hash: nothing of the paper reaches a system
        # prompt, so two papers under the same code share a config_fp.
        prompt_hash = instrument.extractor_prompt_hash()
        tool_hash = instrument.tool_set_hash()
        ext_dec = resolved_decoding_params(
            self.extractor_model, sampling=self.sampling,
            max_tokens=self.extractor_max_tokens, thinking=self.thinking)
        ext_call_identity = _call_identity(self.extractor_model, ext_dec)
        config_fp = instrument.extractor_fingerprint(
            ext_call_identity,
            prompt_hash=prompt_hash, tool_hash=tool_hash,
            supports_images=self.extractor_supports_images,
        )
        # A disabled stage records a null fingerprint (and needs no usable
        # model), so runs differing only in structure never share a stage
        # fingerprint.
        checker_fp = self._compute_checker_fp()
        review_fp = self._compute_review_fp(tool_hash)
        # The orthogonal axes beside the stage fingerprints (see fingerprint's
        # module docstring). The instrument axis uses the EFFECTIVE structure
        # values this orchestrator holds, not pipeline.yaml's, so a CLI
        # override is reflected.
        instrument_fp = instrument.fingerprint(
            tool_hash=tool_hash,
            checker_context_chars=self._checker_context_chars(),
        )
        extractor_call_fp = _call_fp(ext_call_identity)
        checker_call_fp = (_call_fp(self.checker_config.call_identity())
                           if self.checker_enabled else None)
        review_call_fp = (_call_fp(self._review_call_identity())
                          if self.final_review else None)
        engine_fp = current_engine_fp()
        # Whole-run identity. A disabled checker/review stage contributes a
        # documented sentinel, so the four ablation shapes stay distinct (see
        # fingerprint.run_fingerprint).
        run_fp = _run_fp(config_fp, checker_fp, review_fp, engine_fp)
        return {
            "ext_figures": ext_figures,
            "ext_image_labels": ext_image_labels,
            "system_text": system_text,
            "prompt_hash": prompt_hash,
            "tool_hash": tool_hash,
            "config_fp": config_fp,
            "checker_fp": checker_fp,
            "review_fp": review_fp,
            "instrument_fp": instrument_fp,
            "extractor_call_fp": extractor_call_fp,
            "checker_call_fp": checker_call_fp,
            "review_call_fp": review_call_fp,
            "engine_fp": engine_fp,
            "run_fp": run_fp,
        }

    def prepare_new_session(self):
        """Build the fingerprints + initial messages + Session, return self."""
        # Pre-flight config check, before any API spend.
        self._startup_capability_guard()
        self._startup_identity_guard()
        # Fingerprints + rendered extractor prompt, via the shared recipe a
        # dry run also uses.
        fps = self._build_fingerprints()
        ext_figures = fps["ext_figures"]
        ext_image_labels = fps["ext_image_labels"]
        self.system_text = fps["system_text"]
        prompt_hash = fps["prompt_hash"]
        tool_hash = fps["tool_hash"]
        config_fp = fps["config_fp"]
        checker_fp = fps["checker_fp"]
        review_fp = fps["review_fp"]

        # Session: captures tool_definitions + rendered prompts + image labels
        # inline, so the transcript reproduces what the model actually saw,
        # independent of later edits to the config bundle or paper text. All
        # THREE stage system prompts are captured — including on a run that
        # never reaches the reviewer. Each is None when its stage is off, and
        # Session.create then writes no file for it.
        tool_definitions = get_tool_definitions(self.template)
        # Built ONCE. The conversation starts from these blocks and the record
        # is their text view, so the message sent and the message recorded are
        # one object's contents — and the crops are base64-encoded once rather
        # than encoded twice and thrown away once.
        self.initial_user_blocks, message_labels = self._extractor_message()
        rendered_user_prompt = render_message_text(
            self.initial_user_blocks, message_labels)
        rendered_review_system = self._render_review_system_text() \
            if self.final_review else None
        rendered_checker_system = self._render_checker_system_text() \
            if self.checker_enabled else None
        # The per-field scaffold beside the system prompt it is asked under, so
        # a finished session holds every piece of text its checks were built
        # from. The specimen round a dry run prints is NOT captured: it is an
        # aid to reading the scaffold, and nothing in this run was asked
        # through it.
        rendered_checker_scaffold = \
            self.instrument.render_checker_user_scaffold() \
            if self.checker_enabled else None
        self.session = Session.create(
            self.study_id,
            config_fp=config_fp, checker_fp=checker_fp, review_fp=review_fp,
            instrument_fp=fps["instrument_fp"],
            extractor_call_fp=fps["extractor_call_fp"],
            checker_call_fp=fps["checker_call_fp"],
            review_call_fp=fps["review_call_fp"],
            engine_fp=fps["engine_fp"],
            extractor_model=self.extractor_model,
            # A disabled stage records a null model, mirroring its null stage
            # fingerprint: the CLI passes pipeline.yaml's checker/review
            # models through even when the stage is off, and run.json must not
            # show a model that never ran.
            checker_model=(self.checker_config.checker_model
                           if self.checker_enabled else None),
            review_model=self.review_model if self.final_review else None,
            checker_context_chars=self._checker_context_chars(),
            tool_set_hash=tool_hash,
            template_hash=self.instrument.template_hash,
            prompt_hash=prompt_hash,
            tool_definitions=tool_definitions,
            system_prompt=self.system_text,
            user_prompt=rendered_user_prompt,
            review_system_prompt=rendered_review_system,
            checker_system_prompt=rendered_checker_system,
            checker_user_scaffold=rendered_checker_scaffold,
            image_labels=self._attached_exhibits_record(message_labels),
            runs_dir=self.out_dir,
            caps={
                "max_tool_calls": self.max_tool_calls,
                "max_review_tool_calls": self.max_review_tool_calls,
                "max_checks_per_field": self.max_checks_per_field,
            },
            structure=self._structure_dict(),
            decoding_specified=self.decoding_specified,
            diagnostics=self.diagnostics,
        )
        # Loud run-start signal: any decoding value the operator configured
        # that this run's models never send.
        self._warn_inert_decoding_params()

        # Extraction record + dispatcher. The dispatcher validates `<img>`
        # citations against the label set of every crop the message attached,
        # article and supplements alike.
        self.extraction_record = self.session.load_extraction_record()
        self.dispatcher = ToolDispatcher(
            self.extraction_record, self.template, self.paper_text,
            ext_image_labels, reference_lists=self.reference_lists,
        )

        # Per-image hashes let the transcript flag re-cropping drift after
        # the run, so they cover every crop the message attached: a
        # supplement's crop with no baseline here is a crop whose re-cropping
        # `supplements_fp` reports in aggregate and nothing attributes to a
        # label.
        if self.bundle.all_figures():
            self.session.capture_image_hashes(
                sorted(self.bundle.all_figures().values()))
        # The paper's own fingerprint, from the same bundle. Unconditional — a
        # paper with no figures is still a paper the run must name — and in no
        # other fingerprint (see fingerprint.bundle_fingerprint).
        self.session.capture_bundle_fingerprint(self.bundle)

        # The conversation, from the blocks built above.
        self.messages = [{"role": "user", "content": self.initial_user_blocks}]
        return self

    def resume_session(self, session_dir):
        """Reattach to an in-progress session and rebuild the conversation."""
        # Pre-flight config check, before any API spend.
        self._startup_capability_guard()
        self._startup_identity_guard()
        # Inputs derived from the bundle in __init__. That they are the
        # ORIGINAL bundle's is enforced below, by the paper axes passed to
        # `Session.resume` — not by config_fp, which folds in no part of the
        # paper. The figures and labels are rebuilt from the same bundle the
        # session started against, so the rebuilt prompt and fingerprint match
        # the original session.
        instrument = self.instrument
        ext_figures, ext_image_labels = self.figures, self.image_labels
        self.system_text = instrument.render_extractor_system_text()
        # Paper-INDEPENDENT prompt hash; see _build_fingerprints.
        prompt_hash = instrument.extractor_prompt_hash()
        tool_hash = instrument.tool_set_hash()
        ext_dec = resolved_decoding_params(
            self.extractor_model, sampling=self.sampling,
            max_tokens=self.extractor_max_tokens, thinking=self.thinking)
        expected_fp = instrument.extractor_fingerprint(
            _call_identity(self.extractor_model, ext_dec),
            prompt_hash=prompt_hash, tool_hash=tool_hash,
            supports_images=self.extractor_supports_images,
        )
        # Resume is refused when ANY stage fingerprint moved, not just the
        # extractor's: a changed checker or review prompt/model/decoding
        # would silently produce results under a different config than the
        # one the session started with. A structure change moves config_fp
        # (via the decoding-params hash above), so toggling a stage on or off
        # also refuses resume through the same drift gate. Turning a stage off
        # makes its expected fingerprint null, matching a session that started
        # with the stage off; the config_fp drift is what refuses the mixed
        # case.
        expected_checker_fp = self._compute_checker_fp()
        expected_review_fp = self._compute_review_fp(tool_hash)

        self.session = Session.resume(
            session_dir, expected_config_fp=expected_fp,
            expected_checker_fp=expected_checker_fp,
            expected_review_fp=expected_review_fp,
            # The PAPER, checked on the same terms and in the same place as the
            # three config axes. It rides in none of them, so nothing above can
            # notice an edited text.md or a re-cropped figure.
            expected_bundle=self.bundle)
        # A resumed session is live again: clear any pause_reason left by a
        # tool-call-cap pause so it never lingers past the resume that acted
        # on it (only the pause writes it; the resume clears it).
        if self.session.meta.pop("pause_reason", None) is not None:
            self.session.write_meta()
        # Record the budget bounds governing THIS segment. The caps are
        # operational budgets, not config identity (they ride in no
        # fingerprint), so a resume may legitimately change them — the
        # documented recovery from a cap-hit pause. meta.caps is updated to
        # the values now in force; the `resumed` event below is the
        # per-segment history (a raised bound shows as new != previous). The
        # review budget is recorded too, though the review loop never resumes
        # mid-conversation: a resume that reaches the review stage runs it
        # fresh under the bound recorded here.
        caps = self.session.meta.setdefault("caps", {})
        previous_cap = caps.get("max_tool_calls")
        previous_review_cap = caps.get("max_review_tool_calls")
        caps["max_tool_calls"] = self.max_tool_calls
        caps["max_review_tool_calls"] = self.max_review_tool_calls
        # The diagnostics level is operational on the same terms as the caps:
        # a resume may change it, run.json reports the CURRENT segment's
        # level, the `resumed` event keeps the history. Raising it does not
        # backfill earlier artefacts, and the instrument is captured only at
        # session creation, so a session started at `minimal` never gains one.
        previous_diagnostics = self.session.meta.get("diagnostics")
        self.session.meta["diagnostics"] = self.diagnostics
        # What each role's decoding block SAYS, refreshed to this segment's,
        # on the same terms as the caps and the level above. It is the record
        # of what was asked for rather than of what went on the wire, and the
        # case it exists for is a control the model refuses: editing one of
        # those moves no fingerprint, so the drift gate admits the resume and
        # a snapshot frozen at creation would attribute this segment's asks to
        # the previous segment's block. run.json reports the CURRENT segment's;
        # the `resumed` event below carries both values whenever they differ,
        # which is the per-segment history this snapshot cannot hold.
        # Absent is not empty. A session recorded before this key existed says
        # nothing about what its segment asked for, and reading that silence as
        # "stated no controls" would have the event below announce a change on
        # the first resume of every such session — from a previous value nobody
        # recorded. Undetermined makes no claim: the event carries the pair
        # only when a PRESENT previous value differs. `Session.create` always
        # writes the key (`{}` for a run that states nothing), so an empty
        # mapping here is a real reading, and a move away from it is reported.
        previous_specified = self.session.meta.get("decoding_specified")
        specified_moved = (previous_specified is not None
                           and self.decoding_specified != previous_specified)
        self.session.meta["decoding_specified"] = self.decoding_specified
        # The RATE CARDS, on the same terms as the caps and the decoding block
        # above. They are commercial rather than methodological, so they reach
        # no fingerprint and the drift gate admits a resume that changed them —
        # which means the numbers a segment's spend was costed at can move
        # mid-run with nothing in the record to say so. `meta.cost_rates` is
        # rewritten at every flush and holds only the CURRENT segment's cards,
        # so the segment where they moved is only readable if the previous
        # values are written down when they do. Absent is not empty, exactly as
        # for `decoding_specified`: a session that recorded no cards says
        # nothing about what priced it, and reading that silence as "no cards"
        # would announce a change on the first resume of every such session.
        previous_rates = self.session.meta.get("cost_rates")
        current_rates = self._cost_rates_record()
        rates_moved = (previous_rates is not None
                       and current_rates != previous_rates)
        # A FRESH reading of the whole engine identity for this segment, from
        # the same helper `Session.create` used: both package versions and
        # source digests, plus the git anchor. The creation-time readings —
        # `meta.meltiro_version`, `meta.direktoro_version`, `meta.git_commit`,
        # `meta.git_dirty`, `meta.engine_fp` — are deliberately left alone:
        # `meta.run_fp` is derived from `engine_fp`, so rewriting either would
        # silently restate the run's identity halfway through. The per-segment
        # facts ride on this event instead. Engine drift is recorded, not
        # refused (see _warn_engine_drift); the stage fingerprints are what a
        # resume IS refused on, because a changed prompt or model changes the
        # question.
        segment_commit, segment_dirty = git_state()
        segment_identity = engine_identity()
        segment_version = segment_identity[0]
        segment_direktoro = segment_identity[2]
        segment_engine_fp = current_engine_fp(segment_identity)
        self.session.append_event({
            "event": "resumed",
            "max_tool_calls": self.max_tool_calls,
            "previous": previous_cap,
            "max_review_tool_calls": self.max_review_tool_calls,
            "previous_max_review_tool_calls": previous_review_cap,
            "diagnostics": self.diagnostics,
            "previous_diagnostics": previous_diagnostics,
            "git_dirty": segment_dirty,
            "meltiro_version": segment_version,
            "direktoro_version": segment_direktoro,
            "git_commit": segment_commit,
            "engine_fp": segment_engine_fp,
            # Only when it actually changed, so the event says something when
            # it carries these at all: an unchanged block is the ordinary
            # case, and recording it every resume would bury the segment where
            # the asks moved.
            **({"decoding_specified": self.decoding_specified,
                "previous_decoding_specified": previous_specified}
               if specified_moved else {}),
            # Likewise: carried only when the cards actually moved, so the
            # event says something when it carries them at all.
            **({"cost_rates": current_rates,
                "previous_cost_rates": previous_rates}
               if rates_moved else {}),
        })
        self.session.write_meta()
        self._warn_engine_drift(segment_engine_fp)
        # Rebuild the per-field check counts from the event log: the budget is
        # per field across the whole session, so a resumed segment must not
        # hand a field a fresh allowance. The recorded verdicts are the only
        # durable tally; meta stores no second copy to drift.
        self._check_counts = self._reconstruct_check_counts()
        # Reseed cost/token accumulators from meta so finalise covers the
        # WHOLE run, not just this segment. The accumulators are checkpointed
        # at the cadence they change (per tool call, checker fan-out, review
        # call, plus a clean _pause), so this recovers spend even from a hard
        # crash. See _reseed_usage_from_meta.
        self._reseed_usage_from_meta()
        self.extraction_record = self.session.load_extraction_record()
        self.dispatcher = ToolDispatcher(
            self.extraction_record, self.template, self.paper_text,
            ext_image_labels, reference_lists=self.reference_lists,
        )
        # Re-announce any image omission at this run start (stderr only; the
        # persisted meta.images_omitted carries over from session creation).
        self._warn_inert_decoding_params()
        self.initial_user_blocks, _ = self._extractor_message()
        replayed = self.session.replay_messages()
        self.messages = [
            {"role": "user", "content": self.initial_user_blocks},
        ] + replayed
        # Continue the turn counter past every id already written, so new
        # turns can never reuse an id an earlier loop emitted (which would
        # merge two turns into one message on the next resume).
        self._turn_counter = self.session.max_turn_id()
        return self

    # ----------------------------------------------------------------------
    # Dry run
    # ----------------------------------------------------------------------

    def dry_run_report(self, report_dir=None):
        """Render every prompt and compute all three stage fingerprints
        WITHOUT creating a session or calling any API. This is the whole of
        `--dry-run`.

        Nothing session-shaped is constructed or persisted: no run.json, no
        status, no extraction_output.json, no run-log entry. The rendered
        instrument is printed in full (untruncated) to stdout. When
        `report_dir` is given, the same content is additionally written as
        plain files under that directory, which deliberately does NOT look
        like a session (a consumer scanning
        `<study>/sessions/*/diagnostics/run.json`
        never sees it, and there is no status anywhere in it).

        The fingerprints come from the same `_build_fingerprints` recipe a
        real run uses, so the preview matches what the run would record.

        With the checker on, the report also carries its per-field half: the
        scaffold every check is rendered from, and one specimen check filled in
        from it for a real field of this template. A checker round is otherwise
        the one thing an operator cannot read without paying for it.

        Both halves of the extractor's opening conversation are in it: the
        system message and the user message the run would send, the second
        rendered text-only through the helper a real run captures. That
        message is where the paper's whole text, every supplement's prose and
        each exhibit's label and transcription actually sit, so a report
        showing the instrument alone would preview the smaller half of what a
        run spends its input tokens on and none of the framing the engine
        builds around the study.

        The paper's own fingerprint axes are printed beside the run's, on the
        keys `run.json` records them under. A dry run is per-paper, so the
        question a bundle's axes answer — whether this crop, transcription or
        supplement moved the input's identity — is answerable here without
        paying for the run that would record them.

        Returns the rendered artefacts for inspection by callers and tests.
        """
        # Pre-flight config check, the same guard a real run runs before spend.
        self._startup_capability_guard()
        self._startup_identity_guard()
        fps = self._build_fingerprints()
        self.system_text = fps["system_text"]

        tool_catalogue = self.instrument.tool_catalogue()
        # Every crop the message attaches, as a manifest: the label an `<img>`
        # citation must name, in the order the message attaches them, each
        # named by the document it came out of, because a label alone does not
        # say which. The count is the one number a reader of this report takes
        # on trust, so it is the message's own figure sequence that is counted.
        #
        # The manifest and nothing more. The caption, the printed footnote and
        # the content as text all appear in the USER MESSAGE section above,
        # verbatim, in the block each belongs to — this block used to render
        # them a second time, which was a second answer to what an exhibit
        # arrives with and, once a transcription rode along, the same table
        # printed twice on one report.
        _, attached = self._extractor_message()
        attached_exhibits = [self._exhibit_manifest_entry(label, document)
                             for label, document in attached]

        # The checker and review system prompts render with no API call, so a
        # dry run shows them too. Each is omitted when its stage is off.
        checker_system = (self._render_checker_system_text()
                          if self.checker_enabled else None)
        # The checker's per-field half. The specimen is None for a template
        # declaring no field a check could ever reach, and the report then
        # carries the scaffold alone.
        checker_scaffold = (self.instrument.render_checker_user_scaffold()
                            if self.checker_enabled else None)
        checker_round = (
            self.instrument.render_checker_round_sample(self.checker_config)
            if self.checker_enabled else None)
        # Rendered through the same helper a real run captures and sends, so
        # the preview matches what the reviewer would be shown. It is
        # paper-independent: which crops the reviewer actually receives is
        # settled where they are attached, in its user message.
        review_system = (self._render_review_system_text()
                         if self.final_review else None)

        # The extractor's user message, rendered by the same helper that
        # captures it into a session, so the preview cannot say one thing and
        # the message another. Text-only: an image block is rendered as the
        # label text it is attached under, which is the most a written report
        # can carry of a crop, and the exhibits block below itemises them.
        user_message = self._render_user_prompt_text()
        # The reviewer's, on the same terms and from the same construction the
        # review turn sends. Its exhibits are guarded on the REVIEWER's model,
        # so this is what that role is actually sent.
        review_user_message = (
            render_message_text(*self._review_message(
                self.REVIEW_OUTPUT_PLACEHOLDER))
            if self.final_review else None)

        fingerprints = {
            "study_id": self.study_id,
            "extractor_model": self.extractor_model,
            "checker_model": (self.checker_config.checker_model
                              if self.checker_enabled else None),
            "review_model": self.review_model if self.final_review else None,
            "checker_context_chars": self._checker_context_chars(),
            "config_fp": fps["config_fp"],
            "checker_fp": fps["checker_fp"],
            "review_fp": fps["review_fp"],
            "instrument_fp": fps["instrument_fp"],
            "extractor_call_fp": fps["extractor_call_fp"],
            "checker_call_fp": fps["checker_call_fp"],
            "review_call_fp": fps["review_call_fp"],
            "engine_fp": fps["engine_fp"],
            "run_fp": fps["run_fp"],
            # The paper, folded into none of the axes above: `run_fp` says
            # what would be asked and `bundle_fp` what it would be asked of.
            # Recomputed from the loaded bundle by the one recipe a run
            # records, in the key order `run.json` uses, so a preview and a
            # record of the same bundle are read the same way.
            **bundle_fingerprint(self.bundle),
            "template_hash": self.instrument.template_hash,
            "prompt_hash": fps["prompt_hash"],
            "tool_set_hash": fps["tool_hash"],
            "reference_lists_hash": self.instrument.reference_hash(),
            "structure": self._structure_dict(),
            "decoding_params": self._decoding_params_meta(),
        }

        self._print_dry_run(fingerprints, tool_catalogue, attached_exhibits,
                            checker_system, checker_scaffold, checker_round,
                            review_system, user_message, review_user_message)
        # Loud run-start signals a real run start also emits. The inert-param
        # warning matters most here: the report above prints the RESOLVED
        # params, and only this says which value the operator wrote was
        # dropped to produce them.
        self._warn_inert_decoding_params()

        if report_dir is not None:
            self._write_dry_run_report(
                Path(report_dir), tool_catalogue, attached_exhibits,
                checker_system, checker_scaffold, checker_round,
                review_system, fingerprints, user_message,
                review_user_message)
        return {
            "system_text": self.system_text,
            "user_message": user_message,
            "review_user_message": review_user_message,
            "tool_catalogue": tool_catalogue,
            "attached_exhibits": attached_exhibits,
            "checker_system": checker_system,
            "checker_user_scaffold": checker_scaffold,
            "checker_round_sample": checker_round,
            "review_system": review_system,
            "fingerprints": fingerprints,
        }

    def _decoding_params_meta(self):
        """The decoding params each enabled stage will actually send, per role.

        The same `resolved_decoding_params` the adapters call and the stage
        fingerprints fold in, so this reports what goes on the wire rather
        than what pipeline.yaml asked for. The two differ whenever a model
        declares a refusal: a model that refuses a sampling control is sent
        none of it, so a specified value is silently inert for that role and
        moves no fingerprint.

        A disabled stage records None rather than a guessed dict: its model is
        not required, so it must not be resolved through the registry.
        """
        return {
            "extractor": resolved_decoding_params(
                self.extractor_model, sampling=self.sampling,
                max_tokens=self.extractor_max_tokens, thinking=self.thinking),
            "checker": (
                resolved_decoding_params(
                    self.checker_config.checker_model,
                    sampling=self.checker_config.sampling,
                    max_tokens=self.checker_config.max_tokens,
                    thinking=self.checker_config.thinking)
                if self.checker_enabled else None),
            "review": (
                resolved_decoding_params(
                    self.review_model, sampling=self.review_sampling,
                    max_tokens=self.review_max_tokens,
                    thinking=self.review_thinking)
                if self.final_review else None),
        }

    def _configured_decoding_values(self):
        """What the operator wrote, per enabled role, as `{role: {key: value}}`.

        The config-time scalars, before the registry has had its say, keyed as
        `pipeline.yaml` spells them: the sampling controls under their own
        names, and the thinking fields under `thinking_mode`,
        `thinking_effort`, `thinking_budget_tokens`, `thinking_display` —
        `direktoro.split_decoding_config`'s vocabulary, which is what the
        operator typed. Both halves of a decoding block go inert the same way
        (a model that has no reasoning surface is sent no thinking parameters
        at all), so both belong in the one place that answers "what was asked
        for".

        Only values actually configured appear, and a disabled stage is omitted
        rather than nulled: its model is never resolved, so nothing can be
        said about what it would send.
        """
        def specified(mapping):
            # A None value is "no opinion", so it never counts as configured.
            return {k: v for k, v in (mapping or {}).items() if v is not None}

        configured = {}
        for role, (_model, _cap, sampling, thinking) in \
                self._role_decoding_inputs().items():
            values = specified(sampling)
            values.update(_configured_thinking(thinking))
            configured[role] = values
        return configured

    def _sends_thinking_field(self, role, field):
        """Whether this role's call carries the effect of one thinking field.

        Answered DIFFERENTIALLY — resolve the role's call with the spec, then
        again with this one field cleared, and compare — rather than by looking
        for a key in the resolved dict. A thinking field has no key of its own
        on any wire: an effort is `output_config.effort` on Anthropic's and
        `reasoning_effort` on OpenRouter's, a mode is `thinking.type` on one
        and folded into `reasoning_effort` on the other, and a display setting
        is a key inside a dict that only exists when a mode is also set. A
        table of wire keys would be a fourth copy of direktoro's own mapping,
        wrong the day it grows a wire; asking what the field CHANGES is the
        same question in a form the registry answers for itself.

        True when the cleared spec cannot be built or resolved at all: a field
        the call cannot be made without is not one the call ignores.
        """
        model, max_tokens, sampling, thinking = \
            self._role_decoding_inputs()[role]
        remaining = {f: getattr(thinking, f) for f in THINKING_FIELDS
                     if f != field}
        try:
            # A spec with nothing left in it is not an empty spec — direktoro
            # refuses that shape — it is the state of having said nothing about
            # thinking, which is None. The comparison has to be against the
            # call this role would make with the field simply absent.
            cleared = (type(thinking)(**remaining)
                       if any(v is not None for v in remaining.values())
                       else None)
            with_field = resolved_decoding_params(
                model, max_tokens=max_tokens, sampling=sampling,
                thinking=thinking)
            without = resolved_decoding_params(
                model, max_tokens=max_tokens, sampling=sampling,
                thinking=cleared)
        except ValueError:
            return True
        return with_field != without

    def _warn_inert_decoding_params(self):
        """Warn for every decoding value the bundle sets that this run's models
        never send.

        A registry quirk can drop a parameter outright (see
        `_decoding_params_meta`); the value is then inert, and invisibly so,
        because the stage fingerprints fold in the RESOLVED params — two
        bundles differing only in that key collide on every fingerprint.

        Both halves of a decoding block are checked. A sampling control is on
        the wire or it is not, so membership in the resolved dict settles it. A
        thinking field has no wire key of its own, so it is settled
        differentially by `_sends_thinking_field` — which matters because the
        inert case is real and common: a model with no reasoning surface, or a
        gateway wire that carries only an effort, is sent nothing for a
        configured `thinking_mode` and the run records no trace of the
        omission.

        A warning, never fatal: a key live for one role may be inert for
        another, and stripping it from the roles it still governs is not owed.
        Stderr on every path including a dry run; PERSISTED only on a paid run,
        since a dry run has no artefact to caveat.
        """
        resolved = self._decoding_params_meta()
        models = {
            "extractor": self.extractor_model,
            "checker": (self.checker_config.checker_model
                        if self.checker_enabled else None),
            "review": self.review_model if self.final_review else None,
        }
        for role, configured in sorted(
                self._configured_decoding_values().items()):
            sent = resolved.get(role) or {}
            for key in sorted(configured):
                if key.startswith(THINKING_KEY_PREFIX):
                    if self._sends_thinking_field(
                            role, key[len(THINKING_KEY_PREFIX):]):
                        continue
                elif key in sent:
                    continue
                message = (
                    f"inert-decoding-param: the {role} model "
                    f"{models[role]!r} is sent no {key!r}, so the configured "
                    f"{key} of {configured[key]!r} does not reach it and moves "
                    f"no fingerprint. Two bundles differing only in this key "
                    f"produce identical fingerprints for this role; read the "
                    f"run's recorded decoding params for what was actually "
                    f"sent."
                )
                print("WARNING: " + message, file=sys.stderr)
                if self.session is not None and not self.dry_run:
                    self.session.add_warning(message)

    def _print_dry_run(self, fingerprints, tool_catalogue, attached_exhibits,
                       checker_system, checker_scaffold, checker_round,
                       review_system, user_message, review_user_message):
        """Print the full, untruncated dry-run report to stdout."""
        print("=== DRY RUN (no session created) ===\n")
        print(f"Study: {self.study_id}\n")
        print("=== SYSTEM MESSAGE ===\n")
        print(self.system_text)
        # The extractor's other half, printed untruncated like the rest: the
        # study delimited, then a section per supplement, then the exhibits
        # itemised below in the order that message attaches them. The pair is
        # kept together because it is one conversation.
        print("\n=== USER MESSAGE (image blocks as their labels) ===\n")
        print(user_message)
        print(f"\n=== ATTACHED EXHIBITS ({len(attached_exhibits)}) ===\n")
        for entry in attached_exhibits:
            # One line per exhibit: this is the manifest of what the message
            # above attaches, and what each arrives with is printed there.
            print(f"  {entry}")
        print("\n=== TOOL CATALOGUE (canonical JSON) ===\n")
        print(tool_catalogue)
        if checker_system is not None:
            print("\n=== CHECKER SYSTEM MESSAGE ===\n")
            print(checker_system)
            # The verdict schema alongside the prompt that briefs it. It is
            # engine-owned rather than part of the bundle, so a dry run is
            # where an operator reads the shape their checker must answer in.
            print("\n=== CHECKER TOOL CATALOGUE (canonical JSON) ===\n")
            print(canonical_checker_tool_json())
            print("\n=== CHECKER USER SCAFFOLD ===\n")
            print(checker_scaffold)
            if checker_round is not None:
                print("\n=== CHECKER ROUND, ONE SPECIMEN FIELD ===\n")
                print(checker_round)
        if review_system is not None:
            print("\n=== REVIEW SYSTEM MESSAGE ===\n")
            print(review_system)
            print("\n=== REVIEW USER MESSAGE (image blocks as their "
                  "labels) ===\n")
            print(review_user_message)
        print("\n=== FINGERPRINTS ===\n")
        print(json.dumps(fingerprints, indent=2, sort_keys=False))

    def _write_dry_run_report(self, report_dir, tool_catalogue,
                              attached_exhibits, checker_system,
                              checker_scaffold, checker_round,
                              review_system, fingerprints, user_message,
                              review_user_message):
        """Write the dry-run report as plain files under `report_dir`.

        Deliberately NOT a session: no run.json, no status, no
        extraction_output.json, no tool_calls.jsonl.

        Written into a fresh sibling temp directory and swapped into place, so
        `report_dir` reflects exactly one run's config: a re-run with a stage
        toggled off leaves no stale prompt file behind, and a crash mid-write
        leaves no partial report. The swap is remove-then-rename (a rename
        cannot clobber a non-empty directory); the gap only ever exposes a
        missing report_dir, never a mixture.
        """
        report_dir = Path(report_dir)
        parent = report_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = parent / f"{report_dir.name}.tmp.{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        try:
            (tmp_dir / "extractor_system.md").write_text(
                self.system_text, encoding="utf-8")
            (tmp_dir / "user_message.md").write_text(
                user_message, encoding="utf-8")
            (tmp_dir / "tool_catalogue.json").write_text(
                tool_catalogue, encoding="utf-8")
            # One line per exhibit, so the file is a list a reader or a
            # script can take a line at a time. What each exhibit arrives
            # with is in `user_message.md`, in the block it belongs to.
            (tmp_dir / "attached_exhibits.txt").write_text(
                "".join(f"{entry}\n" for entry in attached_exhibits),
                encoding="utf-8")
            (tmp_dir / "fingerprints.json").write_text(
                json.dumps(fingerprints, indent=2, sort_keys=False) + "\n",
                encoding="utf-8")
            if checker_system is not None:
                (tmp_dir / "checker_system.md").write_text(
                    checker_system, encoding="utf-8")
                (tmp_dir / "checker_tool_catalogue.json").write_text(
                    canonical_checker_tool_json(), encoding="utf-8")
                (tmp_dir / "checker_user_scaffold.md").write_text(
                    checker_scaffold, encoding="utf-8")
                if checker_round is not None:
                    (tmp_dir / "checker_round_sample.md").write_text(
                        checker_round, encoding="utf-8")
            if review_system is not None:
                (tmp_dir / "review_system.md").write_text(
                    review_system, encoding="utf-8")
                (tmp_dir / "review_user_message.md").write_text(
                    review_user_message, encoding="utf-8")
            # Swap the freshly written temp dir into place. A rename cannot
            # replace a non-empty directory, so remove the old report first,
            # then rename. The gap only ever exposes a missing report_dir.
            if report_dir.exists():
                shutil.rmtree(report_dir)
            os.replace(tmp_dir, report_dir)
        finally:
            # On any failure before the swap, drop the temp dir so a crash
            # never leaves a partial report_dir.tmp.* behind. After a
            # successful swap tmp_dir no longer exists, so this is a no-op.
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        print(f"\nDry-run report written under {report_dir}")

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def run(self):
        """Run the loop. Returns the status the session was persisted with.

        Stopping a run is two phases, and they are separated here because they
        fail differently:

          (a) `_finalise` / `_pause` persist the outcome — the terminal status
              in run.json and, for a terminal stop, the run-log entry beside
              it. This runs inside the try, so a fault on the way to it is
              caught and finalises `error`.

          (b) `_render_artefacts` writes the derived documents (field history,
              transcript). This runs OUTSIDE the try, because both are views
              over files already persisted in (a): a rendering fault must cost
              the run its documents and nothing else. Inside the try it would
              re-enter finalisation on an already-finalised session — a second
              run-log entry double-counting the spend, or a resumable cap-hit
              pause overwritten as a terminal error.

        The returned status is therefore always the one on disk, rendering
        fault or not.

        Two things reach (a) as a PAUSE rather than a finalisation, and both
        leave the session `in_progress` and resumable with no run-log entry:
        the tool-call cap, raised from inside the extractor loop, and a
        provider refusing over the account, caught here (see
        `_pause_on_provider_account`). Every other provider failure, and every
        other exception, finalises `error` through the handlers below.
        """
        if self.dry_run:
            # A dry run never enters the live loop: the CLI routes it through
            # dry_run_report(), which creates no session. Reaching run() with
            # dry_run set means the flow was wired wrong; fail loudly rather
            # than fall through into paid API calls.
            raise AgenticExtractionError(
                "run() must not be called on a dry-run orchestrator; a dry "
                "run goes through dry_run_report(), which creates no session.")

        try:
            status = self._run_to_stop()
        except ProviderAccountError as e:
            # Ordered first: the one provider refusal that PAUSES. It is about
            # who is asking rather than what was asked, so the extraction is
            # untouched and the fix is outside the process — the same shape as
            # the tool-call cap, and resumable on the same terms. Every other
            # provider failure falls through to the handlers below and ends
            # the run.
            status = self._pause_on_provider_account(e)
        except AgenticExtractionError as e:
            self._report_run_error(str(e))
            status = self._finalise("error")
        except Exception as e:
            # Catch-all so an unexpected crash doesn't leave the session
            # stranded as `in_progress` forever. The traceback is captured
            # in the jsonl for postmortem.
            import traceback
            self._report_run_error(f"{type(e).__name__}: {e}",
                                   traceback=traceback.format_exc())
            status = self._finalise("error")
        self._render_artefacts()
        return status

    def _pause_on_provider_account(self, error):
        """Pause the run on a provider refusal about the ACCOUNT, and record
        what the provider said. Returns "in_progress", from `_pause`.

        The second thing that pauses rather than terminates, and it pauses for
        the same reason the tool-call cap does: nothing about the extraction is
        wrong, and the fix is a thing a person does outside this process — top
        up a balance, replace a revoked key, grant a key access to the model.
        Once they have, the same session continues; re-running from scratch
        would repay work that succeeded, which for a run that had already
        finished extracting is the entire expensive half.

        What a resume repays is the stage it stopped in, not the run. An
        extractor pause continues the same conversation from where it
        stopped. A REVIEW pause repays the whole review: the reviewer builds
        its context fresh from the finished extraction every time and holds no
        cross-segment state, so the paper, the figures and every review turn
        already made are sent again, and `max_review_tool_calls` is a fresh
        budget for the new conversation rather than a running total. The
        extraction itself is never re-extracted, which is the half worth
        keeping — but a reviewer that had already edited it leaves those
        edits in place, so "nothing is repaid" would be false and is not
        claimed.

        The refusal is `direktoro.ProviderAccountError` and nothing else. That
        class means the provider refused over WHO IS ASKING rather than what
        was asked, which is exactly the property that makes a pause safe: an
        outside fix changes the answer. The plain `ProviderError` beside it
        covers malformed requests and direktoro's own refusals of a served
        response, and those must stay terminal — a run that paused on a
        malformed request would resume into it for ever, since a resume sends
        the same inputs to the same provider. Waiting is NOT the axis: a 400
        is as unfixable by waiting as an empty balance.

        WHAT THIS DEPENDS ON, AND WHERE THE DEPENDENCE LIVES. The pause means
        whatever the class means, and the class is defined in another package.
        Nothing is caught here to be logged: it is caught to stop a run in a
        way that can be RESUMED, on the promise that a human fixes something
        outside this process and the same call then succeeds. So a release
        that admits a new failure into `ProviderAccountError` adds it to what
        this pauses on, in a version bump, with no line of this file changing.

        There is no defence from here. The class carries no discriminator, by
        a decision this package asked for and would ask for again — telling
        its members apart from outside would mean reading the exception's
        message text, which is the thing the taxonomy exists to spare a
        consumer and which would break the first time a provider reworded a
        sentence. The declared floor in `pyproject.toml` is therefore the
        whole of the defence, which makes raising it a semantic act rather
        than routine maintenance: the question to answer before raising it is
        not whether the new version is compatible but whether every failure it
        newly admits satisfies "would a human fixing something outside the
        process make this same call succeed". A refusal that is about the
        REQUEST fails that test however credential-shaped its status looks,
        and pausing on one would strand a run waiting for an account nobody
        needs to touch.

        THREE legs reach a provider and only two of them pause. The
        extractor's refusal and the reviewer's both raise into `run()`; the
        CHECKER's does not, and deliberately so. `checker.py` turns every
        provider failure into a degraded, error-origin verdict on the one
        field it was asked about, because the fan-out runs a batch and letting
        one field's refusal escape `fut.result()` would discard the verdicts
        its siblings had already been billed for. So an account that empties
        during a checker fan-out leaves those fields unchecked rather than
        pausing, the run continues, and `meta.checker_diagnostics
        .checker_errors` records exactly which fields ended with no verdict —
        which `cli._print_checker_health` prints at every status, `complete`
        included, on its own terms: an absence of checking is not an objection
        to a value, and it is louder than a checker that found one thing to
        say. A field left unchecked that way is not re-checked by a resume,
        because a check fires on a field being written and a correct field is
        not rewritten.

        WHAT THE OPERATOR IS TOLD, and it is two different things on purpose.
        The note names `provider_message` — the provider's own sentence,
        parsed once by direktoro in the place that holds the parsed body — so
        an operator reads "You have no credits remaining. Add credits to
        continue using the API" rather than an SDK envelope wrapped around a
        stringified dict. The fallback to `str(error)` is not decoration:
        `provider_message` is None whenever no provider actually spoke, as for
        a refusal direktoro raises on its own authority, and a note with a
        blank where the reason goes would be worse than an ugly one.

        The EVENT keeps both. `str(error)` is the SDK's whole rendering,
        carrying the status and the `type`/`code` a support conversation or a
        methods record may want, and it can be thinner than the sentence
        beside it, so neither subsumes the other. A record that a reader may
        come to years later should hold what was said and how it arrived; the
        line printed at the moment of the stop should hold only what to do.
        """
        message = (
            "the provider refused this call over the account or the "
            "credential rather than the request (an exhausted balance or "
            "spend cap, or a key that is absent, revoked, or not entitled "
            "to this model). Nothing about the extraction is wrong and the "
            "session is resumable once the account is fixed. The provider "
            f"said: {error.provider_message or error}")
        # The event log is where the provider's words live, not run.json:
        # `_pause` writes only what makes the pause durable and resumable, and
        # the sentence is neither. The transcript renders it from here (see
        # `_describe_run_event`), which is also where a reader finds the
        # `provider_retry` events that preceded it on a leg that retried.
        self.session.append_event({
            "event": "provider_account_refused",
            "message": str(error),
            "provider_message": error.provider_message,
        })
        print(f"  PAUSED: {message}", file=sys.stderr)
        return self._pause("provider_account")

    def _report_run_error(self, message, *, traceback=None):
        """Record the failure that ended the run: an `error` event carrying the
        composed message, the message stashed for `_finalise` to persist, and
        one stderr line.

        The stderr line is what an operator reads first. It is printed here,
        at the point of the catch, so it lands ahead of the run summary the
        CLI prints from the returned status — a status word with no sentence
        behind it sends the reader into the transcript for what the exception
        already said.
        """
        event = {"event": "error", "message": message}
        if traceback is not None:
            event["traceback"] = traceback
        self.session.append_event(event)
        self._error_message = message
        print(f"  ERROR: {message}", file=sys.stderr)

    def _render_artefacts(self):
        """Write the derived documents of a stopped run: the per-field history
        and the readable transcript. Phase (b) of stopping (see `run`).

        Both are rebuilt wholesale from files already on disk — the event log,
        run.json, the extraction output — so neither is a source of truth and
        neither can change the outcome. A failure here is therefore reported
        rather than raised: one stderr line naming the document and the session
        it belongs to, and the same sentence appended to `meta.warnings` when
        meta is still writable, so a session missing a transcript says why.
        Both are attempted, because the fault that stopped one need not stop
        the other.
        """
        for name, render in (
                ("field_history.json", self.session.write_field_history),
                ("transcript.md", self.session.write_transcript)):
            try:
                render()
            except Exception as e:
                message = (
                    f"artefact-not-written: could not write {name} for "
                    f"session {self.session.session_dir}: "
                    f"{type(e).__name__}: {e}. The run's status and its "
                    f"run-log entry are unaffected; re-render with "
                    f"`meltiro transcript`.")
                print(f"WARNING: {message}", file=sys.stderr)
                try:
                    self.session.add_warning(message)
                except Exception:
                    # meta is unwritable too, which the same disk fault
                    # explains. The stderr line above is then the whole
                    # record: raising here would undo the guard.
                    pass

    def _run_to_stop(self):
        """Drive the run to its stop and persist the outcome, returning the
        status. run()'s body, split out so run() can render the derived
        documents outside the try this raises into."""
        # Pre-spend key preflight: every enabled stage must have its
        # provider key before any API call. Raised from inside run()'s try, so
        # a missing key finalises as a clean "error" naming the variable and
        # the stage, rather than surfacing mid-run, where a missing checker
        # key degrades every field to a challenge only after the extractor
        # has fully spent.
        self._preflight_keys()
        # A resumed session that had already reached the reviewer re-enters
        # THERE, not at the extractor. The question is which STAGE the run got
        # to, and `current_phase` is the only thing that records it: the
        # extractor's completion claim looks like the same fact and is not,
        # because the reviewer's job is to edit and every edit it makes clears
        # that claim. Routing on the claim would send the common case — a
        # reviewer that changed something, then met a refusal — back into the
        # extractor, which is precisely the trip this exists to avoid, and
        # would hand the extractor a record the reviewer had altered
        # underneath it, invisible in the extractor's own conversation.
        #
        # Re-entering the extractor on a finished extraction is not merely a
        # wasted turn. It replays the whole conversation, pays for it, applies
        # the EXTRACTOR's completeness gate to an extraction the reviewer has
        # already approved — a reviewer that deliberately removed a record
        # would have the extractor told to put it back — and a model handed
        # its own finished work can revise it. A resumed run has to be the run
        # it would have been uninterrupted.
        #
        # `current_phase` is "extracting" for a fresh session and for any
        # pause the extractor took, so this costs a first run nothing and
        # leaves the tool-call-cap pause exactly as it was.
        if self.session.meta.get("current_phase") != "final_review":
            extractor_status = self._extractor_loop()
            stop = self._finalise_loop_stop(extractor_status)
            if stop is not None:
                return stop
            if extractor_status != "mark_complete_validated":
                # `_finalise_loop_stop` answers None both for the one outcome
                # the run continues past and for any it does not recognise, so
                # None alone does not mean "the extractor finished". An
                # outcome added or renamed later raises here and finalises
                # `error` rather than drifting through to `complete`.
                raise AgenticExtractionError(
                    f"unrecognised extractor loop outcome "
                    f"{extractor_status!r}: it is neither a mapped stop nor "
                    f"the completion signal, so the run cannot continue. Map "
                    f"it in _finalise_loop_stop.")

        # Extractor signalled mark_complete. Challenges are advisory (see
        # module docstring); anything still challenged is recorded in
        # meta.checker_diagnostics at finalisation.

        elif not self.final_review:
            # Resumed into a review stage this run does not have. The drift
            # gate should make it unreachable — `final_review` rides in
            # `structure_hash` and so in `config_fp`, so turning the reviewer
            # off refuses the resume — but the two facts are recorded in
            # different files, and the one outcome that must not follow from
            # them disagreeing is shipping `complete` an extraction that
            # skipped BOTH stages.
            raise AgenticExtractionError(
                "this session stopped in the final-review phase but the "
                "reviewer is off for this run, so neither stage would run "
                "and nothing would examine the extraction. Resume with the "
                "configuration the session was started under.")

        # Final review (optional per run: off when final_review is False).
        # When off the run finalises directly after the extractor.
        if self.final_review:
            review_status = self._final_review()
            if review_status == "final_review_no_response":
                # The reviewer returned neither text nor a tool call. That
                # is an infrastructure failure (empty completion), not a
                # judgement about the extraction: record it and finalise as
                # error so the operator re-runs.
                message = ("the final reviewer returned neither text nor a "
                           "tool call (empty completion); treating as an "
                           "infrastructure error.")
                self.session.append_event({
                    "event": "final_review_no_response",
                    "message": message,
                })
                # Through the same channel an exception takes, so a run that
                # ended this way carries its sentence on stderr and in the run
                # record exactly as one that raised does.
                self._error_message = message
                print(f"  ERROR: {message}", file=sys.stderr)
                return self._finalise("error")
            stop = self._finalise_review_stop(review_status)
            if stop is not None:
                return stop
            if review_status == "error":
                return self._finalise("error")
            if review_status != "review_clean":
                # With the reviewer ON, only its confirmation may finalise
                # `complete`. An outcome not mapped above (added later,
                # renamed, mistyped) stops here and finalises `error`
                # rather than shipping an extraction no reviewer signed
                # off on.
                raise AgenticExtractionError(
                    f"unrecognised final review status {review_status!r}: "
                    f"only a confirmed review ('review_clean') may "
                    f"finalise a run as complete. Map it in "
                    f"_final_review / _finalise_review_stop.")

        return self._finalise("complete")

    # ----------------------------------------------------------------------
    # Phases
    # ----------------------------------------------------------------------

    def _extractor_loop(self):
        """Run extractor turns until mark_complete validates or a bound fires.

        Every applied field also goes through the inline checker before its
        tool result is rendered, so a challenge reaches the model in the same
        payload as the validation errors (see `_check_applied_fields`). A
        challenge is advisory (module docstring): it never fails a call,
        never blocks mark_complete, and changes nothing about control flow.

        Returns one of:
          - "mark_complete_validated": the extractor declared completion and
            the completeness gate passed.
          - "extractor_abandoned": the extractor called abandon_extraction
            (deliberate surrender). Finalises the run as failed_validation.
          - "tool_cap_hit": the tool-call cap was reached. This is a resumable
            PAUSE (the caller leaves the session in_progress); raising the cap
            and resuming continues the same conversation.
          - "text_only_stall": the consecutive-text-only bound fired (a model
            that never calls a tool). Terminal (failed_validation): an
            unchanged resume re-runs the same inputs into the same spiral.
          - "extractor_stalled": the repeated-failure guard fired (a model
            wedged re-submitting the SAME rejected call). Terminal
            (failed_validation) for the same reason.
          - "extractor_refused": a turn came back with stop_reason "refusal"
            (a host content filter, or the model declining). Terminal
            (`error`): the request was refused rather than the extraction
            judged, so there is nothing to re-prompt and nothing to call
            invalid.

        Those six are the whole vocabulary. An unrecoverable per-turn failure
        is not among them: it propagates as an exception, which run()'s
        catch-all records as an `error` event and finalises `error`, so the
        mapping this feeds (`_finalise_loop_stop`) covers exactly what it can
        return.

        Termination is guaranteed under any model behaviour: every dispatched
        tool call counts toward the cap regardless of validation outcome, a
        run of tool-free turns has its own strict bound, and the
        repeated-failure guard stops an identically-failing call well before
        the cap. The event log records which bound fired.

        The consecutive-failure counter is loop-local, so a resumed session
        starts a fresh run; no counter state is carried across resume.
        """
        self.session.set_phase("extracting")
        adapter = self._adapter_for_role("extractor")
        if adapter is None:
            env = model_info(self.extractor_model).api_key_env
            raise AgenticExtractionError(
                f"{env} not set; cannot run extractor.")

        tool_defs = get_tool_definitions(self.template)
        consecutive_text_only = 0
        # Repeated-failure guard state (shared with the review loop; see
        # _IdenticalFailureRun). Any applied/partial call, a different
        # signature, or a different tool resets the run.
        failure_run = _IdenticalFailureRun(
            self.max_consecutive_identical_failures)

        while True:
            if self.session.meta.get("tool_call_count", 0) >= \
                    self.max_tool_calls:
                return "tool_cap_hit"

            turn_id = self._next_turn_id()
            self._current_turn_id = turn_id
            response = self._call_extractor(adapter, tool_defs)
            self._record_extractor_response(response, turn_id)

            if getattr(response, "stop_reason", None) == "refusal":
                # The model declined this call: a host content filter stopped
                # the response, or the model refused it outright. That is an
                # ANSWER, and it outranks whatever else the turn carried
                # (direktoro's canonical vocabulary puts `refusal` ahead of
                # `tool_use` for exactly this reason). Read as an ordinary
                # tool-free turn it would be re-prompted, refused again, and
                # end three paid calls later as `text_only_stall` — a stall
                # guard naming a cause that was never the cause. It stops here,
                # on the first one, with the reason recorded.
                self._append_assistant_event(turn_id, response)
                self.session.append_event({
                    "event": "extractor_refused",
                    "turn_id": turn_id,
                    "stop_reason": "refusal",
                })
                return "extractor_refused"

            tool_uses, assistant_text = self._extract_tool_uses(response)
            if not tool_uses:
                # Model returned no tool call. Record the assistant turn so
                # the transcript keeps user/assistant alternation and the
                # model's reasoning is not dropped, then re-prompt. A run of
                # consecutive tool-free turns is capped so this cannot spin.
                assistant_content = self._assistant_content_from_response(
                    response)
                if not assistant_content:
                    # Neither text nor tool_use blocks (for example a
                    # thinking-only truncation). Without a placeholder this
                    # turn would contribute no assistant message, so the
                    # re-prompt below would land as a second consecutive user
                    # message on the next call. Substitute a placeholder so
                    # alternation holds.
                    assistant_content = [{
                        "type": "text",
                        "text": EMPTY_ASSISTANT_PLACEHOLDER,
                    }]
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })
                # Record the verbatim ordered assistant content so replay
                # rebuilds this turn's assistant message byte-identically.
                self._append_assistant_event(turn_id, response,
                                             content=assistant_content)
                # Log an assistant_text event as the human-transcript prose
                # record for this turn. Use the placeholder when the model
                # returned no text, so a resumed transcript keeps the same
                # alternation the live run had.
                self.session.append_event({
                    "event": "assistant_text",
                    "turn_id": turn_id,
                    "text": assistant_text or EMPTY_ASSISTANT_PLACEHOLDER,
                })
                consecutive_text_only += 1
                if consecutive_text_only >= \
                        self.max_consecutive_text_only_turns:
                    self.session.append_event({
                        "event": "text_only_stall",
                        "turn_id": turn_id,
                        "consecutive_text_only_turns": consecutive_text_only,
                    })
                    # A text spiral is terminal, not a resumable pause: raising
                    # the cap does not help a model that never calls a tool,
                    # and an unchanged resume re-runs the same inputs into the
                    # same spiral.
                    self.session.write_meta()
                    return "text_only_stall"
                reprompt = EXTRACTOR_TOOL_REPROMPT
                # The re-prompt event, appended IMMEDIATELY after the assistant
                # events and before any meta write, so the two sides of this
                # turn go to disk together. Behind a write it left a window in
                # which a kill produced a log whose last turn had an assistant
                # side and no user side — and a replay of that ends on an
                # assistant message, which the next call sends as a prefill of
                # the model's own narration. A prefill ending in whitespace is
                # an API 400, so the resume this exists to protect would fail
                # on its first call. Replay drops such a turn now
                # (`Session.replay_messages`); this is the other half, keeping
                # the window as small as one append.
                self.session.append_event({
                    "event": "extractor_reprompt",
                    "turn_id": turn_id,
                    "text": reprompt,
                })
                # A text-only turn still cost an API call, but drives no tool
                # call, so the per-tool-call meta write that flushes the
                # checkpointed spend on a normal turn never fires here. Flush
                # explicitly so a crash after a text-only turn keeps its spend
                # (the accumulators were mirrored into meta by _accumulate_usage
                # inside _call_extractor above).
                self.session.write_meta()
                # An extractor whose registry entry declares no forced tool
                # choice (so it runs under "auto") and that declined to call a
                # tool is re-prompted with a firm nudge; count that
                # auto-degrade retry in meta. No-op for a forcing model, whose
                # text-only turn is the general guard.
                self._maybe_record_auto_degrade_retry("extractor")
                self._append_user_text(reprompt)
                continue

            # A tool-calling turn resets the tool-free run.
            consecutive_text_only = 0

            # Dispatch each tool_use, build the user tool_result list. Every
            # dispatched call counts toward the cap whether it applied
            # cleanly, applied partially, or failed validation; that is what
            # bounds total exposure under persistent validation failure.
            tool_results = []
            for tu in tool_uses:
                # A multi-call batch can drive tool_call_count past the cap
                # mid-batch, so clamp the hint at zero: the model should never
                # be told it has a negative budget remaining.
                budget_remaining = max(
                    0,
                    self.max_tool_calls
                    - self.session.meta.get("tool_call_count", 0)
                )
                res = self.dispatcher.dispatch(
                    tu["name"], tu["input"],
                    meta={"tool_call_budget_remaining": budget_remaining},
                )
                # The checker runs HERE, after the deterministic dispatch and
                # before anything is recorded or rendered: it merges its
                # verdicts into `res`, so the event below carries the full
                # per-field record and `result_to_model_text` renders the
                # challenges into the same tool_result as the validation
                # errors. Doing it in this order is also what makes a resume
                # free: replay re-serialises the stored result and re-sends
                # byte-identical content without a single checker call.
                self._check_applied_fields(res, stage="extractor")
                self.session.append_event({
                    "event": _tool_call_event_name(res["status"]),
                    "turn_id": turn_id,
                    "tool_use_id": tu["id"],
                    "tool": tu["name"],
                    "args": tu["input"],
                    "result": res,
                })
                self._log_canonicalisations(res, turn_id=turn_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result_to_model_text(res),
                })
                self.session.increment_tool_call_count()

                # Repeated-failure guard. An applied or partial call resets
                # the run; a fully-failed call extends it (identical
                # signature) or starts a new one. At the cap the loop stops
                # immediately, mid-batch, rather than keep paying for a call
                # that keeps failing the same way. Returning skips appending
                # this turn to the live self.messages (harmless: the loop is
                # terminating), but the assistant_message event is still
                # logged below, so the "every tool-calling turn records an
                # assistant_message" invariant stays total.
                stalled = failure_run.record(tu["name"], res)
                if stalled is not None:
                    tool_name, error_codes = stalled
                    self.session.append_event({
                        "event": "repeated_failure_stall",
                        "turn_id": turn_id,
                        "tool": tool_name,
                        "error_codes": list(error_codes),
                        "consecutive_identical_failures": failure_run.count,
                        "error_message": _first_error_message(res),
                    })
                    # Log the assistant content before returning: replay
                    # treats a missing assistant_message as a crash artefact
                    # and refuses. The event records the full batch the model
                    # emitted, including calls never dispatched; this session
                    # finalises terminal and is never resumed.
                    self._append_assistant_event(turn_id, response)
                    return "extractor_stalled"

            # Append the assistant message (containing tool_use + any text)
            # and the user message (tool_results).
            assistant_content = self._assistant_content_from_response(response)
            self.messages.append({
                "role": "assistant",
                "content": assistant_content,
            })
            self.messages.append({
                "role": "user",
                "content": tool_results,
            })
            # Record the verbatim ordered assistant content (the provider's
            # original text/tool_use block order) so replay rebuilds this
            # turn's assistant message byte-identically instead of forcing
            # text before tool_use.
            self._append_assistant_event(turn_id, response,
                                         content=assistant_content)

            # Persist extraction output after every batch of applied tool calls.
            self.session.write_extraction_record(self.extraction_record)

            if assistant_text:
                self.session.append_event({
                    "event": "assistant_text",
                    "turn_id": turn_id,
                    "text": assistant_text,
                })

            # Did the model deliberately surrender? Checked before completion
            # so an abandon_extraction call in the batch wins over a stale
            # mark_complete flag. The run ends as failed_validation.
            if self.extraction_record.abandoned_flag:
                self.session.append_event({
                    "event": "extractor_abandoned",
                    "turn_id": turn_id,
                    "reason": self.extraction_record.abandon_reason,
                })
                return "extractor_abandoned"

            # Did the model declare completion? This ends the extractor's
            # work at once: no challenge holds the loop open.
            if self.extraction_record.mark_complete_flag:
                return "mark_complete_validated"

    def _extraction_record_field_envelope(self, field_path):
        """Look up the envelope for a dotted field path in the current
        extraction output. Returns the envelope dict or None when the field is
        absent (e.g. record was removed).

        Path formats:
          - "study.<var>"        → self.extraction_record.study[var]
          - "record.<id>.<var>"  → self.extraction_record.records[*][var]
            (looked up by record_id)
        """
        if not isinstance(field_path, str):
            return None
        parts = field_path.split(".")
        if len(parts) >= 2 and parts[0] == "study":
            return self.extraction_record.study.get(parts[1])
        if len(parts) >= 3 and parts[0] == "record":
            record_id, var = parts[1], parts[2]
            for record in self.extraction_record.records:
                if record.get("record_id") == record_id:
                    return record.get(var)
        return None

    def _final_review(self):
        """Fresh-context final review pass.

        The reviewer sees the full paper, all figure images, and the assembled
        extraction output for the first time, in a conversation of its own —
        looking for the coherence/omission/missing-record issues the
        one-cell-at-a-time checker cannot catch. It runs a bounded tool loop
        (see `_review_loop`) and is expected to call `mark_complete` when
        satisfied.

        Nothing assembled for the reviewer names a checker (see the module
        docstring). With `check_reviewer_edits` on, the fields it WRITES are
        checked on the same per-field terms as the extractor's; its engine
        prompt then describes that protocol, and the challenge in its own
        tool result is the only thing about a checker it is ever shown. With
        the toggle off it is told of no checker at all.

        Returns "review_clean" | "final_review_no_response" |
        "review_abandoned" | "review_cap_hit" | "review_text_only_stall" |
        "review_stalled" | "error".

        The mapping from `_review_loop`'s outcomes onto those is exhaustive and
        defaults to failure: only "review_mark_complete" yields "review_clean",
        and an outcome the mapping does not know raises
        AgenticExtractionError, which run() finalises as `error`.
        """
        self.session.set_phase("final_review")
        adapter = self._adapter_for_role("review")
        if adapter is None:
            # Stage is ON but no usable key. Fail loudly rather than silently
            # skip: skipping would ship an un-reviewed extraction under a
            # config that asked for a review.
            env = model_info(self.review_model).api_key_env
            raise AgenticExtractionError(
                f"{env} not set; cannot run the final review. Set the key, or "
                f"disable the reviewer with final_review: false "
                f"(or --no-final-review).")

        # The reviewer's own message, from the one construction the preview
        # is projected from.
        review_system_text = self._render_review_system_text()
        # Without the check blocks: the reviewer records its OWN quality
        # check, so the extractor's self-assessment is withheld for the same
        # reason the checker's verdicts are (see build_review_user_blocks).
        review_user_blocks, _ = self._review_message(
            self.extraction_record.to_dict(include_checks=False))
        review_messages = [{"role": "user", "content": review_user_blocks}]
        tool_defs = get_tool_definitions(self.template, role=ROLE_REVIEW)

        outcome, edits_attempted, edits_applied = self._review_loop(
            adapter, tool_defs, review_messages, review_system_text)

        # Persist post-review extraction output for every outcome: a reviewer
        # that edited and then spiralled still changed it. NOT a redundant
        # repeat of the loop's per-turn write — the repeated-failure stall
        # returns from INSIDE a batch, before the loop's write, so for that
        # outcome this is the write that lands the batch's applied edits.
        self.session.write_extraction_record(self.extraction_record)

        if outcome == "error":
            return "error"
        if outcome == "review_no_response":
            return "final_review_no_response"
        if outcome in ("review_abandoned", "review_cap_hit",
                       "review_text_only_stall", "review_stalled"):
            # None of these saw the reviewer confirm, so none may finalise
            # `complete`: run() maps each to failed_validation with its own
            # failure_reason.
            return outcome
        if outcome != "review_mark_complete":
            # Exhaustive, defaulting to failure: only the reviewer's
            # confirmation may become "review_clean". An unknown outcome
            # raises, and run() finalises `error`.
            raise AgenticExtractionError(
                f"unrecognised review loop outcome {outcome!r}: it is not a "
                f"mapped stop and it is not the reviewer's confirmation, so "
                f"it must not be read as a clean review. Map it here.")

        # The reviewer confirmed: the run finalises on the extraction output
        # as it now stands.
        if edits_attempted and not edits_applied:
            # The reviewer TRIED to edit and nothing landed. Named for the
            # outcome, not the cause: not all of these are validation failures
            # (update_study with an empty block answers `status: ok` with an
            # empty `applied_changes`, and lands here too).
            self.session.append_event({
                "event": "final_review_edits_none_applied",
                "attempted": edits_attempted,
            })
        return "review_clean"

    def _review_loop(self, adapter, tool_defs, messages, system_text):
        """Run reviewer turns until it concludes or a bound fires.

        Every tool result is fed back, so a reviewer that inspects with the
        `view_*` tools can act on what it saw; `mark_complete` is the
        intended exit.

        Returns `(outcome, mutations_attempted, mutations_applied)`, where
        outcome is one of:
          - "review_mark_complete": the reviewer called mark_complete.
          - "review_abandoned": the reviewer called abandon_extraction
            (deliberate surrender). Terminal: failed_validation.
          - "review_cap_hit": the review tool-call bound fired. Terminal, NOT
            a pause: the review conversation is fresh-context and never
            replayed (not part of `self.messages`), so a resume would re-run
            the whole review rather than continue it.
          - "review_text_only_stall": the consecutive-text-only bound fired (a
            reviewer that never calls a tool; the tool-call bound alone would
            never advance on such turns).
          - "review_stalled": the repeated-failure guard fired (the reviewer
            wedged re-submitting the SAME rejected call).
          - "review_no_response": a turn carried neither text nor a tool call.
            Infrastructure failure, not a judgement: mapped to `error`.
          - "error": an unrecoverable per-turn provider error.

        The two counters cover MUTATING calls only (tools.MUTATING_TOOLS): a
        read-only `view_*` call is not an edit.

        The bounds mirror the extractor loop's (the repeated-failure state
        machine is shared, _IdenticalFailureRun); what differs is that the
        reviewer's cap terminates rather than pauses, and its calls are
        counted per review conversation, never against `meta.tool_call_count`
        (the extractor's budget and provenance).
        """
        consecutive_text_only = 0
        failure_run = _IdenticalFailureRun(
            self.max_consecutive_identical_failures)
        # Bound counter, loop-local by design: it counts THIS review
        # conversation's dispatched calls. The review is never resumed
        # mid-conversation, so there is no cross-segment state to carry.
        review_tool_calls = 0
        mutations_attempted = 0
        mutations_applied = 0

        while True:
            if review_tool_calls >= self.max_review_tool_calls:
                self.session.append_event({
                    "event": "review_cap_hit",
                    "review_tool_calls": review_tool_calls,
                })
                return ("review_cap_hit", mutations_attempted,
                        mutations_applied)

            turn_id = self._next_turn_id()
            try:
                response = self._call_review(
                    adapter, tool_defs, messages, system_text, turn_id)
            except ProviderAccountError:
                # Ordered ahead of the catch-all deliberately, and it RAISES
                # rather than returning an outcome. Every other per-turn
                # failure here is a fact about this run that the review is
                # entitled to end on; this one is a fact about the account,
                # and flattening it into "error" would discard the one piece
                # of information that makes the session resumable. run()
                # pauses on it (see `_pause_on_provider_account`), which is
                # why this leg needs the exception itself rather than a status
                # word — and why the recording below is a plain event with no
                # return.
                self.session.append_event({
                    "event": "review_provider_account_refused",
                    "turn_id": turn_id,
                })
                raise
            except Exception as e:
                self.session.append_event({
                    "event": "review_error", "message": str(e),
                })
                return ("error", mutations_attempted, mutations_applied)

            tool_uses, assistant_text = self._extract_tool_uses(response)
            self.session.append_event({
                "event": "final_review_response",
                "turn_id": turn_id,
                "tool_use_count": len(tool_uses),
                "tool_names": [t["name"] for t in tool_uses],
                "assistant_text": assistant_text,
            })

            if not tool_uses and not assistant_text:
                # Neither text nor a tool call: an empty completion. That is an
                # infrastructure failure rather than a judgement about the
                # extraction, so it does not go through the text-only re-prompt.
                return ("review_no_response", mutations_attempted,
                        mutations_applied)

            assistant_content = self._assistant_content_from_response(response)

            if not tool_uses:
                # Text with no tool call: a reviewer narrating its conclusion
                # without calling mark_complete. Record the turn and re-prompt;
                # the run of tool-free turns is bounded so this cannot spin.
                messages.append({"role": "assistant",
                                 "content": assistant_content})
                self._append_review_assistant_event(
                    turn_id, assistant_content, response)
                self.session.append_event({
                    "event": "assistant_text",
                    "stage": "review",
                    "turn_id": turn_id,
                    "text": assistant_text,
                })
                # A text-only turn drives no tool call, so nothing else flushes
                # the spend this turn accrued. Flush explicitly so a crash here
                # cannot lose it (the accumulators were mirrored into meta by
                # _accumulate_usage inside _call_review).
                self.session.write_meta()
                consecutive_text_only += 1
                if consecutive_text_only >= \
                        self.max_consecutive_text_only_turns:
                    self.session.append_event({
                        "event": "review_text_only_stall",
                        "turn_id": turn_id,
                        "consecutive_text_only_turns": consecutive_text_only,
                    })
                    return ("review_text_only_stall", mutations_attempted,
                            mutations_applied)
                # Auto-degrade provenance: a non-forcing reviewer under "auto"
                # that narrated without calling mark_complete is nudged once;
                # count it. No-op for a forcing model (the general guard).
                self._maybe_record_auto_degrade_retry("review")
                reprompt = REVIEW_TOOL_REPROMPT
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": reprompt}],
                })
                self.session.append_event({
                    "event": "review_reprompt",
                    "turn_id": turn_id,
                    "text": reprompt,
                })
                continue

            # A tool-calling turn resets the tool-free run.
            consecutive_text_only = 0

            # mark_complete IS dispatched now, because it carries the
            # reviewer's own quality check and that has to be recorded. What
            # it does not do is gate: the dispatcher skips the extractor's
            # completeness checks for this role (the reviewer saw the whole
            # assembled output, and it has no second chance to terminate), so
            # the call always succeeds and its presence in the batch remains
            # the terminate signal. Termination is handled after the batch so
            # any edits emitted alongside it land first.
            mark_complete_present = False
            tool_results = []
            for tu in tool_uses:
                if tu["name"] == "mark_complete":
                    mark_complete_present = True
                is_mutation = tu["name"] in MUTATING_TOOLS
                if is_mutation:
                    mutations_attempted += 1
                # A multi-call batch can drive the count past the bound
                # mid-batch, so clamp at zero: the transcript should never show
                # a negative budget. This rides to the dispatcher as UI-only
                # telemetry and is stripped from the model-facing tool_result by
                # result_to_model_text, so the reviewer cannot read a
                # cap-derived number (the cap is out of every fingerprint).
                budget_remaining = max(
                    0, self.max_review_tool_calls - review_tool_calls)
                res = self.dispatcher.dispatch(
                    tu["name"], tu["input"],
                    meta={"tool_call_budget_remaining": budget_remaining},
                    role=ROLE_REVIEW,
                )
                # The reviewer-side checker, off unless the config asks for it.
                # Same trigger, same per-field budget, same place in the order:
                # after the deterministic dispatch, before the event is recorded
                # and before the tool result is rendered.
                if self.check_reviewer_edits:
                    self._check_applied_fields(res, stage="review")
                # The reviewer's quality check rides on a call that cannot
                # fail, so a wrongly-phrased field is dropped inside an `ok`
                # result. Surface that as a run warning, or the discarded
                # opinion vanishes without a trace.
                for w in (res.get("warnings") or []):
                    if w.get("code") == "quality_check_not_recorded":
                        self.session.add_warning(
                            f"review quality_check field not recorded "
                            f"({w.get('path')}): {w.get('message')}")
                review_tool_calls += 1
                self.session.append_event({
                    "event": "review_tool_call",
                    "turn_id": turn_id,
                    "tool_use_id": tu["id"],
                    "tool": tu["name"],
                    "args": tu["input"],
                    "result": res,
                })
                self._log_canonicalisations(res, turn_id=turn_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result_to_model_text(res),
                })
                # `applied_changes` is the dispatcher's own report of what it
                # wrote, empty when a call validated away to nothing even
                # where the status is `ok`. Reading `status` instead would
                # book an `update_study` with an empty block as an edit.
                if is_mutation and res.get("applied_changes"):
                    mutations_applied += 1

                stalled = failure_run.record(tu["name"], res)
                if stalled is not None:
                    tool_name, error_codes = stalled
                    self.session.append_event({
                        "event": "review_repeated_failure_stall",
                        "turn_id": turn_id,
                        "tool": tool_name,
                        "error_codes": list(error_codes),
                        "consecutive_identical_failures": failure_run.count,
                        "error_message": _first_error_message(res),
                    })
                    # Log the turn's verbatim assistant content before returning
                    # so the "every tool-calling turn records an
                    # assistant_message" invariant stays total, exactly as the
                    # extractor loop does on its stall path.
                    self._append_review_assistant_event(
                        turn_id, assistant_content, response)
                    return ("review_stalled", mutations_attempted,
                            mutations_applied)

            messages.append({"role": "assistant", "content": assistant_content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            self._append_review_assistant_event(
                turn_id, assistant_content, response)

            # Persist after every batch, so a crash cannot lose applied edits.
            self.session.write_extraction_record(self.extraction_record)

            if assistant_text:
                self.session.append_event({
                    "event": "assistant_text",
                    "stage": "review",
                    "turn_id": turn_id,
                    "text": assistant_text,
                })

            # Did the reviewer deliberately surrender? Checked before completion
            # so an abandon_extraction call in the batch wins over a
            # mark_complete alongside it, mirroring the extractor loop.
            if self.extraction_record.abandoned_flag:
                self.session.append_event({
                    "event": "review_abandoned",
                    "turn_id": turn_id,
                    "reason": self.extraction_record.abandon_reason,
                })
                return ("review_abandoned", mutations_attempted,
                        mutations_applied)

            if mark_complete_present:
                return ("review_mark_complete", mutations_attempted,
                        mutations_applied)

    def _append_assistant_event(self, turn_id, response, *, content=None):
        """Record one EXTRACTOR turn's verbatim ordered assistant content, and
        why the turn stopped.

        Every extractor turn logs exactly one of these, so `stop_reason` here
        makes the event log the durable record of how each turn ended — a
        content-filter `refusal`, a `max_tokens` truncation, an ordinary
        `tool_use` or `end_turn`. Without it the artefact carries the model's
        words and not the fact that a filter cut them off, and the wire log
        that does carry it is kept only at `--diagnostics full`.

        `content` is the already-built block list when the caller has one
        (it is also what went into `self.messages`, so the two cannot drift);
        omitted, it is derived from `response`.
        """
        self.session.append_event({
            "event": "assistant_message",
            "turn_id": turn_id,
            "content": (content if content is not None
                        else self._assistant_content_from_response(response)),
            "stop_reason": getattr(response, "stop_reason", None),
        })

    def _append_review_assistant_event(self, turn_id, assistant_content,
                                       response):
        """Record a review turn's verbatim ordered assistant content, and why
        the turn stopped.

        Every model turn that dispatches tools logs an `assistant_message`
        event carrying exactly what the provider returned, in order; a turn
        missing one is a crash artefact. The reviewer's turns are held to the
        same rule, which is also what puts its reasoning in the transcript
        rather than only in the raw API log.

        `stop_reason` rides the event for the reason it rides the extractor's
        (`_append_assistant_event`): a reviewer turn cut off at `max_tokens`
        or blocked by a content filter reads as a finished one without it, and
        every reader of how a turn ended — the transcript's note among them —
        reads this field whichever stage the turn belongs to.

        The `stage` marker is what keeps the review conversation out of the
        EXTRACTOR conversation on replay: `Session.replay_messages` rebuilds
        only the extractor's turns, and the reviewer's are a separate,
        fresh-context conversation that a resume re-runs from scratch rather
        than continues.
        """
        self.session.append_event({
            "event": "assistant_message",
            "stage": "review",
            "turn_id": turn_id,
            "content": assistant_content,
            "stop_reason": getattr(response, "stop_reason", None),
        })
    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _preflight_keys(self):
        """Verify every ENABLED stage has its provider API key before any spend.

        Raises AgenticExtractionError naming each missing env var and its
        stage. This is the single pre-spend gate, and it is what stops a
        missing checker key from surfacing as a per-field challenge after the
        extractor has fully spent (see checker.run_checker_batch). Dry-run
        returns before this, so it needs no keys.
        """
        missing = [
            (stage, model_info(model).api_key_env)
            for stage, model, present in self._stage_key_status()
            if not present
        ]
        if missing:
            detail = "; ".join(
                f"{env} (for the {stage} stage)" for stage, env in missing)
            raise AgenticExtractionError(
                f"missing API key(s) before run start: {detail}. Set the "
                f"variable(s), or disable the stage (max_checks_per_field: 0 "
                f"for the checker, final_review: false for the reviewer)."
            )

    def _stage_key_status(self):
        """(stage, model, key_present) for every enabled stage, in spend order.

        Each stage is checked against its OWN model, because that is what
        decides which key the call needs: a reviewer reached through a
        different endpoint from the extractor's makes the run need two
        variables, and a run is short of a key the moment any one enabled
        stage is.
        """
        out = [("extractor", self.extractor_model,
                self._provider_key_present(self.extractor_model))]
        if self.checker_enabled:
            out.append((
                "checker", self.checker_config.checker_model,
                self._provider_key_present(
                    self.checker_config.checker_model)))
        if self.final_review:
            out.append((
                "review", self.review_model,
                self._provider_key_present(self.review_model)))
        return out

    def _provider_key_present(self, model):
        """Whether the environment holds the key this model's adapter will
        read, without importing an SDK or constructing a client.

        `direktoro.build_adapter` resolves the key variable from the model id
        and reads it from the environment, so this asks the environment the
        same question one step earlier — before any spend rather than at the
        stage's first call."""
        return bool(os.environ.get(model_info(model).api_key_env))

    def _role_model(self, role):
        """Model id configured for a role ("extractor" | "review")."""
        if role == "extractor":
            return self.extractor_model
        if role == "review":
            return self.review_model
        raise ValueError(f"unknown role {role!r}")

    def _adapter_for_role(self, role):
        """The adapter a role's calls go through, or None when its key is unset.

        `direktoro.build_adapter` resolves everything about reaching the model
        from the id alone — endpoint, key variable, and the adapter that speaks
        that wire — so per-role routing needs no extra config: naming a
        different model for a role in pipeline.yaml re-points that role and
        nothing else.

        None means one thing only: the key variable the model needs is unset.
        Both call sites raise on it, naming the variable, so no stage is ever
        silently skipped — a run that asked for a reviewer and got none would
        ship an un-reviewed extraction under a config that ordered one.
        """
        try:
            return build_adapter(self._role_model(role))
        except MissingAPIKey:
            return None

    def _checker_adapter(self):
        """The one adapter every checker call in this run goes through.

        Cached, unlike the extractor's and the reviewer's, which are built once
        per loop by the loop that uses them. The checker has no loop of its own:
        it fans out per tool call, so a fresh adapter per fan-out would build a
        fresh provider client, and with it a fresh connection pool, for every
        tool call the run makes — every batch then opening its own TLS
        connections instead of reusing the ones the last batch left warm.

        Built through the checker module's own `_build_checker_adapter`, which
        is what `run_checker_batch` would have called for itself, so hoisting
        the construction here changes when it happens and nothing about what is
        built. The checker's model lives on `CheckerConfig`, not among the
        per-role models `_adapter_for_role` reads.

        None when the key variable is unset, which `run_checker_batch` turns
        into a per-field degradation. The pre-spend preflight has already
        refused that case for an orchestrated run.
        """
        if self._cached_checker_adapter is None:
            self._cached_checker_adapter = _build_checker_adapter(
                self.checker_config)
        return self._cached_checker_adapter

    def _exhibit_manifest_entry(self, label, document):
        """One line of the preview's exhibit manifest."""
        where = f"({document}) " if document else ""
        return f"{where}[{label}]"

    def _attached_exhibits_record(self, labels):
        """The session's record of the exhibits the extractor's message carried.

        One entry per attachment, in the order the message attaches them,
        carrying the label an `<img>` citation must name, the caption the
        paper prints beside it, the footnote it prints under it, and whether
        the exhibit's content rode with it as text. The two texts are the half
        a label alone cannot supply: `table_02` says which
        crop was cited and nothing about which table it is or what its small
        print qualified, and no other capture holds the manifest's wording once
        the bundle directory has moved on. A crop the manifest gave neither of
        records a null in each, which is a different fact from an empty string
        — and, for the footnote, one the reader of a transcript is told apart
        from a session that recorded no footnotes at all.

        `labels` is the message's own figure sequence — the article's
        followed by each supplement's, in the order they attach — so the
        record covers every crop the message carried rather than the
        article's alone. Empty only for a bundle that supplies no crops at
        all.
        """
        return [{"label": label,
                 "caption": self.image_captions.get(normalise_label(label)),
                 "notes": self.image_notes.get(normalise_label(label)),
                 # Whether the exhibit's content rode with it as text. A flag
                 # rather than the markup: the transcription itself is in the
                 # rendered prompt, and repeating it here would put a table in
                 # meta once per attachment. What the flag is for is that the
                 # rendered prompt is captured only from `--diagnostics
                 # standard` up, while this record is kept at every level, so
                 # without it the leanest run could not say afterwards whether
                 # a cell was read as text or off pixels. `tables_fp` says
                 # WHICH transcriptions the bundle held; this says which of
                 # them the message actually carried.
                 "transcribed": normalise_label(label) in self.image_tables}
                for label, _ in labels]

    def _extractor_message(self):
        """The extractor's opening user message, and the labels it attaches.

        The ONE construction of that message in this class: the conversation
        is started from it, the session captures its text view, and
        `--dry-run` prints that same view. A second construction anywhere
        would be a second answer to "what is the extractor sent", and the two
        would agree until one of them was edited.

        The labels beside it are the effective FIGURE SEQUENCE, which is what
        `build_initial_user_blocks` iterates: the same labels, in the same
        order, spelt the same way, the article's followed by each
        supplement's. The normalised label set beside it is the dispatcher's,
        lower-cased and unordered, so a view derived from that would name the
        labels a message carries in a case and an order the message never
        used. A bundle supplying no crops has an empty sequence, and its
        message states that none accompany the study.
        """
        ext_figures = self.figures
        supplements = self._supplements_for()
        blocks = build_initial_user_blocks(
            self.study_id, self.paper_text, ext_figures, self.image_captions,
            self.image_notes, self.image_tables, supplements,
        )
        return blocks, message_figure_labels(ext_figures, supplements)

    # What stands in the preview where the assembled extraction output goes.
    # A dry run happens before there is one, and the alternative — an empty
    # record, rendered as JSON — would preview a shape the reviewer is never
    # sent, which is worse than saying plainly that this part is not knowable
    # yet. Everything around it in that message IS knowable, and is previewed.
    REVIEW_OUTPUT_PLACEHOLDER = (
        "(not previewable: the assembled extraction output as it stands when "
        "the review begins)")

    def _review_message(self, extraction_record_dict):
        """The reviewer's user message, and the labels it attaches.

        The ONE construction of that message, on the extractor's terms above:
        the review turn sends it, and `--dry-run` prints its text view. Its
        exhibits are the whole bundle's, as every role's are.

        `extraction_record_dict` is the only part of it a dry run cannot
        know, which is why it is the caller's argument rather than read from
        `self` here: the review turn passes the output as it stands when the
        review begins, and the preview passes a line saying so.
        """
        review_figures = self.figures
        supplements = self._supplements_for()
        blocks = build_review_user_blocks(
            self.study_id, self.paper_text, review_figures,
            extraction_record_dict,
            self.image_captions, self.image_notes, self.image_tables,
            supplements,
        )
        return blocks, message_figure_labels(review_figures, supplements)

    def _render_user_prompt_text(self):
        """The text view of the message above, projected FROM it.

        Captured into the session at creation time so the transcript view
        never has to re-read the paper text from disk, and printed by
        `--dry-run` so what a run would send can be read before it is paid
        for. Only the crops' bytes are absent; each is named where it
        attaches.
        """
        blocks, labels = self._extractor_message()
        return render_message_text(blocks, labels)

    def _render_review_system_text(self):
        """The reviewer's rendered system message.

        Paper-independent, like the extractor's: which cropped exhibits the
        reviewer actually receives is decided where they are attached, not in
        this text.
        """
        return self.instrument.render_review_system_text()

    def _render_checker_system_text(self):
        """The checker's rendered system message: one string for the whole run,
        shared across every per-field call."""
        return self.instrument.render_checker_system_text()

    def _compute_checker_fp(self):
        """Fingerprint the checker, or None when it is off."""
        return self.instrument.checker_fingerprint(self.checker_config)

    def _compute_review_fp(self, tool_hash):
        """Fingerprint the final-review stage, or None when it is off.

        The review model reaches the fingerprint as its call identity, resolved
        here so the block folded into `review_fp` is the same one hashed on its
        own into `review_call_fp`; the model id itself goes separately because
        the structure block records whether that model can see images.
        """
        if not self.final_review:
            return None
        return self.instrument.review_fingerprint(
            self._review_call_identity(),
            review_model=self.review_model, tool_hash=tool_hash,
        )

    def _review_call_identity(self):
        """The review model's provider-call identity block.

        The one place the reviewer's block is resolved: it reaches `review_fp`
        through the instrument and is hashed on its own into `review_call_fp`,
        so the two can never describe different calls. Only meaningful when
        the reviewer is on; callers guard on `final_review`.
        """
        review_dec = resolved_decoding_params(
            self.review_model, sampling=self.review_sampling,
            max_tokens=self.review_max_tokens, thinking=self.review_thinking)
        return _call_identity(self.review_model, review_dec)

    def _log_canonicalisations(self, res, turn_id=None):
        """Write one `value_canonicalised` event per reference-alias match in
        a dispatch result.

        The dispatcher records these under `_canonicalisations` (a
        model-invisible key) whenever a canonical_reference value matched an
        alias rather than an exact name; each carries the field path, the
        entered value, and the stored canonical value. Emitting them as
        their own event gives the run log an explicit, greppable audit trail
        of every silent name substitution.
        """
        if not isinstance(res, dict):
            return
        for c in res.get("_canonicalisations") or []:
            event = {
                "event": "value_canonicalised",
                "field_path": c.get("path"),
                "entered": c.get("entered"),
                "stored": c.get("stored"),
            }
            if turn_id is not None:
                event["turn_id"] = turn_id
            self.session.append_event(event)

    def _next_turn_id(self):
        """Return a session-global, strictly increasing turn identifier.

        Replay groups a turn's events into one assistant/user message pair
        by turn id, so ids must never collide across extractor loops or the
        feedback turns between them within a session. A single monotonic
        counter (seeded from the event log on resume) guarantees that.
        """
        self._turn_counter += 1
        return self._turn_counter

    # ----------------------------------------------------------------------
    # Study-identity context for the checker
    # ----------------------------------------------------------------------

    def _startup_capability_guard(self):
        """Fail loudly, before any API spend, when a role's model cannot
        accept images.

        This pipeline extracts from the paper AND its exhibits: a table's
        value is read off a crop, an `<img>` citation names one, and the
        checker verifies a value against the exhibit it was read from. A model
        that cannot receive an image cannot do any of that, so a run
        configured with one is not a degraded run, it is a different task
        wearing this one's fingerprints — and it would answer with the same
        `run_fp` a full run answers with.

        Every ENABLED stage is checked, and the message names the role, the
        model and the key that set it, because that is the line an operator
        edits.
        """
        roles = [("extractor", self.extractor_model, "extractor_model")]
        if self.checker_enabled:
            roles.append(("checker", self.checker_config.checker_model,
                          "checker_model"))
        if self.final_review:
            roles.append(("review", self.review_model, "review_model"))
        blind = [(role, model, key) for role, model, key in roles
                 if not model_supports_images(model)]
        if not blind:
            return
        # Each role's model can come from `pipeline.yaml` or from the flag
        # that overrides it, and this refusal cannot tell which — so it names
        # both, as the two sibling refusals about these same three values do.
        # Naming one sends an operator who used the other to a place the value
        # is not.
        named = "; ".join(
            f"the {role} model {model!r} (pipeline.yaml `{key}`, or "
            f"`--{key.replace('_model', '')}-model`)"
            for role, model, key in blind)
        raise AgenticExtractionError(
            f"Image input is not optional in this pipeline: {named} cannot "
            f"accept images. Every role reads the paper's cropped exhibits — "
            f"a value taken from a table is cited as `<img>label</img>` and "
            f"checked against the crop it names — so a text-only model would "
            f"be asked for evidence it cannot see. Configure a model with "
            f"image input for every enabled stage."
        )

    def _startup_identity_guard(self):
        """Fail loudly, before any API spend, when the checker would have no
        study-identity context at all: the bundle carries no `summary` AND
        the template declares no `role: summary` field. Names both remedies.

        Gated on the checker running at all: with `max_checks_per_field: 0`
        there is no call to starve, and the extractor-only ablation is a
        documented arm that must stay runnable.
        """
        if not self.checker_enabled:
            return
        has_manifest_summary = bool((self.bundle.summary or "").strip())
        has_role_summary = "summary" in self.template.get("role_fields", {})
        if not has_manifest_summary and not has_role_summary:
            raise AgenticExtractionError(
                "No study-identity context available for the checker: the "
                "paper bundle manifest has no `summary`, and the extraction "
                "template declares no `role: summary` field. Fix EITHER: add "
                "a `summary` key to the bundle manifest, OR mark a study-level "
                "string field with `role: summary` in the template."
            )

    def _study_identity_context(self):
        """Resolve the Checker's study-identity context.

        Precedence:
          1. bundle.summary (manifest, highest precedence)
          2. the extracted role:summary field's current value (it is
             verbatim-quote-validated like any field)

        When neither is available (no manifest summary and the declared
        role:summary field is still null/empty when the checker runs) the
        run is NOT killed: it falls back to title + DOI as minimal identity
        context and records a one-time warning.

        Also raises the summary-mismatch stderr advisory once when both are
        present. The advisory is about the value in hand right now; the
        persisted warning is decided at finalisation, against the value the
        run ships (see `_check_shipped_summary_mismatch`).
        """
        manifest_summary = (self.bundle.summary or "").strip()
        extracted_summary = self._extracted_summary_value()

        if manifest_summary and extracted_summary:
            self._advise_summary_mismatch(
                manifest_summary, extracted_summary)

        if manifest_summary:
            return render_study_identity_context(manifest_summary)
        if extracted_summary:
            return render_study_identity_context(extracted_summary)
        return self._degraded_identity_context()

    def _record_identity_context(self, record):
        """The checker's identity context for a record-scoped field.

        A record-field check needs both which study this is and which record:
        the record label alone would leave the checker judging a field with no
        idea which paper it came from. So this joins the study-identity context
        (same as a study field gets) with the record context label, which leads
        with the engine record id and appends any `checker_context_fields`
        values as a hint.
        """
        study_ctx = self._study_identity_context()
        record_ctx = build_record_context(
            record, self.template["checker_context_fields"])
        return render_record_identity_context(study_ctx, record_ctx)

    def _extracted_summary_value(self):
        """Current value of the template's role:summary field, or "" when
        there is no such field or its value is null/empty."""
        role_field = self.template.get("role_fields", {}).get("summary")
        if not role_field:
            return ""
        env = self.extraction_record.study.get(role_field["variable"])
        value = env.get("value") if isinstance(env, dict) else env
        if value is None:
            return ""
        return str(value).strip()

    def _degraded_identity_context(self):
        """Minimal identity context (title + DOI from the bundle manifest),
        used when neither a manifest summary nor a populated role:summary
        field is available.

        The first time this is reached in a run segment it persists an
        `identity-degradation` warning and echoes it on stderr: the checker
        judges every study-level field against this context, and title + DOI
        says which paper but not what it found. Latched once per segment —
        the condition is a property of the run, not of any one field.
        """
        if not self._identity_degradation_warned:
            self._identity_degradation_warned = True
            role_field = self.template.get("role_fields", {}).get("summary")
            field_name = role_field["variable"] if role_field else "(none)"
            self._record_warning(
                f"identity-degradation: study {self.study_id} has no manifest "
                f"summary and the role:summary field {field_name!r} is empty; "
                f"the checker is falling back to title + DOI as minimal "
                f"identity context."
            )
        title = (self.bundle.title or "").strip()
        doi = (self.bundle.doi or "").strip()
        return render_degraded_identity_context(title, doi)

    def _advise_summary_mismatch(self, manifest_summary, extracted_summary):
        """Advise on stderr, at most once per run segment, when the manifest
        summary and the role:summary value the checker is about to see diverge
        under fuzzy comparison (never fail).

        This describes a value the run is still working on: the checker
        routinely sees a half-written field, so the advisory is deliberately
        NOT persisted to meta.warnings — `_check_shipped_summary_mismatch`
        re-runs the comparison at finalisation against the shipped value. A
        session event keeps the trace of what the operator saw, labelled as
        the mid-run observation it is.

        The latch counts ADVISORIES, not comparisons: it is set below the
        match check, so a segment that starts out matching and diverges later
        still gets its one warning.
        """
        if self._summary_mismatch_advised:
            return
        if _summaries_match(manifest_summary, extracted_summary):
            return
        self._summary_mismatch_advised = True
        self.session.append_event({
            "event": "summary_mismatch_advisory",
            "phase": self.session.meta.get("current_phase"),
            "message": (
                f"study {self.study_id} manifest summary and the "
                f"role:summary value in hand diverge under fuzzy comparison. "
                f"Mid-run observation only: the value may still change, and "
                f"the shipped output is re-checked at finalisation."),
        })
        print(
            f"WARNING: summary-mismatch (mid-run): study {self.study_id} "
            f"manifest summary and the current role:summary value diverge "
            f"under fuzzy comparison. If this is not resolved by the end of "
            f"the run it will be recorded in meta.warnings.",
            file=sys.stderr,
        )

    def _check_shipped_required_fields(self, status):
        """Sweep the SHIPPED extraction output for template-declared
        `required: true` fields carrying null, and name every one in
        meta.warnings.

        The extractor cannot finish with one unset — its `mark_complete`
        gates on exactly this sweep. The reviewer's is not gated (it has no
        second chance to terminate), so a reviewer edit landing after the
        extractor's gate can null a required field and the run still
        finalises `complete`. A warning rather than a failed status: the
        reviewer's judgement that a value is unsupportable is worth
        recording, but the output must not ship looking complete while a
        required field is empty, so every such field is named — not counted.
        `validate_extraction_output` cannot catch this: a null required field
        gives its legality sweep nothing to fault.

        Only judged for a CONSIDERED status: an aborted run stopped part-way,
        and unset required fields are the expected shape of a snapshot.
        """
        if status not in CONSIDERED_STATUSES:
            return
        missing = missing_required_fields(
            self.template, self.extraction_record.to_dict())
        if not missing:
            return
        self._record_warning(
            f"required-fields-null: this run finalised {status} with "
            f"{len(missing)} template-required field(s) shipping a null "
            f"value: {', '.join(missing)}. The extraction is complete in the "
            f"sense that the pipeline concluded, not in the sense that every "
            f"field the template requires carries a value; treat those fields "
            f"as unanswered."
        )

    def _check_shipped_summary_mismatch(self, status):
        """Evaluate the summary-mismatch tripwire against the value the run
        SHIPS, and persist a warning to meta.warnings when the bundle
        manifest's summary and the shipped role:summary value diverge under
        fuzzy comparison.

        The manifest summary and the extracted one are independent accounts
        of the same study, so divergence is the signal that a run may be
        describing the wrong paper. The warning names the three hypotheses
        (wrong paper, distrusted search-index abstract, extraction error)
        rather than picking one; nothing here can tell them apart.

        meta.warnings is a statement about the finished artefact, so this is
        the ONLY place the mismatch warning is persisted: a divergence the
        run resolved before finishing leaves no warning here (its trace is
        the `summary_mismatch_advisory` event). Reading only persisted state
        also makes the tripwire resume-proof, and it is not gated on the
        checker: the comparison is a property of the bundle and the output.

        Declines to judge when `status` is not CONSIDERED (an aborted run's
        role:summary is a work-in-progress snapshot), and when either side of
        the comparison is absent or empty.
        """
        if status not in CONSIDERED_STATUSES:
            return
        manifest_summary = (self.bundle.summary or "").strip()
        shipped_summary = self._extracted_summary_value()
        if not manifest_summary or not shipped_summary:
            return
        if _summaries_match(manifest_summary, shipped_summary):
            return
        self._record_warning(
            f"summary-mismatch: study {self.study_id} manifest summary and "
            f"the shipped role:summary value diverge under fuzzy comparison; "
            f"the bundle may point at the wrong paper, the manifest summary "
            f"may be a distrusted search-index abstract, or the extraction "
            f"may have got the field wrong."
        )

    def _record_warning(self, message):
        """Persist a non-fatal warning to run.json and echo to stderr."""
        self.session.add_warning(message)
        print("WARNING: " + message, file=sys.stderr)

    def _retry_logger(self, stage):
        """Return an on_retry callback that records each retried transient
        provider failure as a `provider_retry` session event. Failed attempts
        raise before the api-call audit log runs, so this event log entry is
        their only trace."""
        def on_retry(attempt, delay_seconds, error):
            self.session.append_event({
                "event": "provider_retry",
                "stage": stage,
                "attempt": attempt + 1,
                "delay_seconds": delay_seconds,
                "error": str(error),
            })
        return on_retry

    def _maybe_record_auto_degrade_retry(self, stage):
        """Record one auto-degrade retry for `stage` in meta, as provenance.

        Called at a tool-free re-prompt when the stage's model cannot force a
        named tool_choice (any model whose registry entry declares the forced
        choice unsupported: the loop sends tool_choice "auto" — meltiro never
        forces a tool — and the model MAY decline to call one). The count is
        `meta.auto_degrade_retries[stage]`. It stays absent/zero for every
        FORCING model: their tool-free turns are still re-prompted, but that
        is the general guard, not an auto-degrade retry.

        `stage` is "extractor" | "review". The model is read from session
        meta, and an id unknown to the registry (the offline loop harnesses)
        is skipped rather than raised on. Provenance only: NOT a fingerprint
        input (the tool_choice mode is implied by the model's registry
        entry).
        """
        model = self.session.meta.get(f"{stage}_model")
        if (model is None or not is_known_model(model)
                or supports_forced_tool_choice(model)):
            return
        retries = self.session.meta.setdefault("auto_degrade_retries", {})
        retries[stage] = retries.get(stage, 0) + 1
        self.session.write_meta()

    def _ledger_refused_call(self, exc, model, role, *, call_type,
                             turn_id=None):
        """Bank the spend of a call that was served, billed, and then refused.

        `direktoro.ProviderError.response` carries the `NormalisedResponse` a
        post-billing refusal is about — a routed call whose pin did not hold,
        or that came back with no generation id or no gateway charge. The
        tokens were spent whatever the routing layer thinks of the receipt, so
        the wire entry is written and the tokens go into this role's meters and
        the run's, before the exception carries on aborting the stage. A
        refusal raised INSTEAD of a response carries none and there is nothing
        to bank.

        Ledgering the money is all this does. It does not turn a refusal into
        an answer: the caller re-raises, the stage ends, and the run finalises
        `error` exactly as before.

        The wire entry goes through `_log_api_call_guarded`, and the spend is
        accumulated after it and outside that guard: the tokens are the part
        no other record can be rebuilt from, and `_accumulate_usage` prices
        the call, so it is a step that can raise on its own account.
        """
        response = getattr(exc, "response", None)
        if response is None:
            return
        self._log_api_call_guarded(
            call_type, response.raw_request, response.raw_response,
            provider=response.provider, base_url=response.base_url,
            wire_model=response.resolved_model,
            wire_request=response.wire_request,
            turn_id=turn_id)
        self._accumulate_usage(response, model, role, billed_refusal=True)

    def _log_api_call_guarded(self, call_type, request_kwargs, response,
                              **extra):
        """Write one call's verbatim wire entry, swallowing a fault in the
        WRITE and reporting it. Takes `Session.log_api_call`'s arguments.

        The write is swallowed for the reason `checker._log_ask` swallows it:
        the log is a record of the call, and a fault writing it must not become
        the call's outcome. The entry lands under the session's `diagnostics/`,
        and only at `--diagnostics full`, so a disk that filled or a permission
        on that directory reaches this write and nothing else — while the call
        it describes has already happened and already been paid for.
        Unguarded, that fault would end an ordinary successful turn as `error`,
        or replace the ProviderError a refused call is in the middle of
        raising.

        The entry is COMPOSED outside the guard, so the swallow covers the one
        step whose faults are environmental. An unreadable diagnostics level
        and a shape `make_entry` cannot compose are defects in the run itself,
        they hold for every call it makes, and swallowed they would leave a
        wire log that is empty — which at this level reads as a run that made
        no calls — with nothing anywhere saying why.
        """
        entry = self.session.api_call_entry(
            call_type, request_kwargs, response, **extra)
        try:
            self.session.write_api_call_entry(entry)
        except Exception as e:
            self._report_unwritten_wire_entry(call_type, e)

    def _report_unwritten_wire_entry(self, call_type, exc):
        """Record that one call is missing from the wire log: one stderr line
        and a `meta.warnings` sentence.

        `api_calls.jsonl` is one line per call and carries no count, so a write
        that failed leaves a file that reads as whole and a call that reads as
        never made. The call WAS made and was paid for — the run's totals carry
        its spend — so the loss is stated where the finished artefact states
        the rest of what it cannot show.

        Each report is guarded, and guarded separately, for the reason
        `_report_unwritten_run_log` guards its three: the fault that closed the
        log can have taken meta or stderr with it, a raise here would become
        the outcome of the call this path exists to keep intact, and one report
        being unavailable says nothing about the other. The message names the
        call type and the fault and nothing per-call, so a run whose every
        write fails records one sentence per call type rather than one per call
        (`add_warning` dedups on the exact string).
        """
        message = (
            f"wire-log entry could not be written: {type(exc).__name__}: "
            f"{exc}. The {call_type} call it describes was made and billed, "
            f"and this run's totals carry its spend, but "
            f"{self.session.api_calls_path} holds no verbatim record of it, "
            f"so the wire log is short of the calls this run made.")
        try:
            print(f"WARNING: {message}", file=sys.stderr)
        except Exception:
            pass
        try:
            self.session.add_warning(message)
        except Exception:
            pass

    def _call_extractor(self, adapter, tool_defs):
        """Run one extractor turn via the provider adapter and return the
        normalised response. The adapter omits any sampling control this
        model's endpoint declares it refuses."""
        from direktoro import ProviderError
        try:
            response = create_message_with_retry(
                adapter,
                on_retry=self._retry_logger("extractor"),
                model=self.extractor_model,
                max_tokens=self.extractor_max_tokens,
                system=extractor_sys_blocks(self.system_text),
                tools=tool_defs,
                tool_choice={"type": "auto"},
                messages=self.messages,
                sampling=self.sampling,
                thinking=self.thinking,
            )
        except ProviderError as e:
            self._ledger_refused_call(
                e, self.extractor_model, "extractor", call_type="extractor",
                turn_id=getattr(self, "_current_turn_id", None))
            raise
        self._warn_if_truncated(
            "extractor", self.extractor_max_tokens, response)
        # Verbatim API audit log: canonical request + response, plus the
        # provider/endpoint and the resolved wire model.
        self._log_api_call_guarded(
            "extractor", response.raw_request, response.raw_response,
            provider=response.provider, base_url=response.base_url,
            wire_model=response.resolved_model,
            wire_request=response.wire_request,
            turn_id=getattr(self, "_current_turn_id", None),
        )
        self._accumulate_usage(response, self.extractor_model, "extractor")
        self._record_role_provenance("extractor", response)
        return response

    def _call_review(self, adapter, tool_defs, messages, system_text, turn_id):
        """Run one final-review turn via the provider adapter and return the
        normalised response.

        The twin of `_call_extractor`, for the review stage: same audit log,
        usage accounting, and role provenance, against the review model, the
        review system prompt, and the reviewer's own fresh-context message
        list (not `self.messages`). The adapter applies the model's
        sampling refusals.

        The sampling controls are the REVIEWER's own (`review_sampling`), and
        every turn of the loop goes through here, so the value
        `_compute_review_fp` records is the value each review call actually
        sends.
        """
        from direktoro import ProviderError
        try:
            response = create_message_with_retry(
                adapter,
                on_retry=self._retry_logger("review"),
                model=self.review_model,
                max_tokens=self.review_max_tokens,
                system=extractor_sys_blocks(system_text),
                tools=tool_defs,
                tool_choice={"type": "auto"},
                messages=messages,
                sampling=self.review_sampling,
                thinking=self.review_thinking,
            )
        except ProviderError as e:
            self._ledger_refused_call(
                e, self.review_model, "review", call_type="final_review",
                turn_id=turn_id)
            raise
        self._warn_if_truncated(
            "review", self.review_max_tokens, response)
        # Verbatim API audit log, with provider/endpoint and resolved model.
        self._log_api_call_guarded(
            "final_review", response.raw_request, response.raw_response,
            provider=response.provider, base_url=response.base_url,
            wire_model=response.resolved_model,
            wire_request=response.wire_request,
            turn_id=turn_id)
        # Checkpoints the running spend into meta, so a crash mid-review keeps
        # every dollar the review has cost so far (see _checkpoint_usage_to_meta
        # for where each stage flushes it).
        self._accumulate_usage(response, self.review_model, "review")
        self._record_role_provenance("review", response)
        return response

    def _record_role_provenance(self, role, response):
        """Record a role's resolved model and raw decoding params in run.json.

        `role` is "extractor" | "review" (the checker records its own via the
        audit callback). The resolved model is the provider's reported
        response.model, which may differ from the configured alias; the
        decoding params are what the adapter actually sent after quirks (for
        example a sampling control omitted for a model that refuses it). This makes a
        session directory self-contained for reproduction without re-reading
        pipeline.yaml at the recorded commit. Written once (and only when the
        value changes), so the per-turn extractor loop does not rewrite meta
        every turn.
        """
        meta = self.session.meta
        decoding = meta.setdefault("decoding_params", {})
        resolved_key = f"{role}_model_resolved"
        if (meta.get(resolved_key) == response.resolved_model
                and decoding.get(role) == response.decoding_params):
            return
        meta[resolved_key] = response.resolved_model
        decoding[role] = response.decoding_params
        self.session.write_meta()

    def recorded_cost(self):
        """The dollar figure this run states, or None when it states none.

        One definition, read by everything that reports a cost — the meta
        checkpoint, the run-log entry, the CLI summary — so they can never
        disagree. None means "no dollar figure" and is never rendered as
        `$0.00`: either some call had no rate and no reported charge (the sum
        would be partial), or nothing has priced anything and no card would.
        A run holding a card for any role states a total from the start; its
        zero is a real zero. Rationale in rates.py.

        The figure this returns can still cover less than the run: a call
        whose charge could not be read is counted in tokens and not in
        dollars, which makes the sum a floor. That is a property of the
        figure rather than a reason to withhold it, so it rides beside it —
        `unreceipted_calls()` below, `meta.cost_incomplete`, and the words
        every reader of the total prints.
        """
        if self._cost_unpriced:
            return None
        if self._cost_counted or any(
                card is not None for card in self.rates.values()):
            return round(self._cost_usd, 6)
        return None

    def unreceipted_calls(self):
        """How many billed calls `recorded_cost()` does not cover.

        0 for a run whose every call came back with a price on it, which is
        every run that never met a missing receipt. Non-zero says the total
        is a floor and by how many calls, so a reader states "at least"
        rather than a figure that reads as the whole bill.
        """
        return self._unreceipted_calls if self._cost_incomplete else 0

    def _enabled_roles(self):
        """The roles this run actually calls, in call order.

        The same rule every other per-role record follows: a disabled stage
        makes no calls, so it has no spend to report and appears nowhere.
        """
        roles = ["extractor"]
        if self.checker_enabled:
            roles.append("checker")
        if self.final_review:
            roles.append("review")
        return roles

    def _role_usage(self, role):
        """One role's live meters, created zeroed on first use.

        `cache_write_tokens` is the counter a response reports as
        `cache_creation_input_tokens` and `meltiro.rates` prices as
        `cache_write_per_1m`; it is named for the rate here so a per-role record
        and the card that priced it read in the same vocabulary.

        `incomplete`/`unreceipted` are the run-level coverage pair kept per
        role: the run's total is the sum of these figures, so a role's figure
        that leaves calls out has to say so where it is stated, or a reader
        adding the roles up rebuilds the run's floor with the qualifier gone.
        """
        return self._usage_by_role.setdefault(role, {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
            "unpriced": False,
            "counted": False,
            "incomplete": False,
            "unreceipted": 0,
        })

    def _role_cost(self, role):
        """The dollar figure one role states, or None when it states none.

        `recorded_cost`'s rule, applied to one role's meters and one role's
        card: a call this role could not cost withholds the role's figure, and a
        role that has costed nothing and holds no card that would states none
        rather than a zero.
        """
        acc = self._role_usage(role)
        if acc["unpriced"]:
            return None
        if acc["counted"] or self.rates.get(role) is not None:
            return round(acc["cost_usd"], 6)
        return None

    def _cost_rates_record(self):
        """`{role: card-as-record-or-null}` for every enabled role.

        Written beside every per-role figure, so a reader recomputes a role's
        cost from that role's counters and that role's rates without needing to
        know which model ran where. A role with no card is present and null:
        absence is never an available reading.
        """
        return {role: (self.rates[role].as_record()
                       if self.rates.get(role) is not None else None)
                for role in self._enabled_roles()}

    def _usage_by_role_record(self):
        """Each enabled role's counters, its cost, and the rates behind it.

        The per-role half of the spend record. The run-wide totals are sums over
        these plus nothing else, so a reader who wants to know which role spent
        the money reads here and a reader who wants the bill reads the totals.
        `cost_usd` is null for a role that could not be priced, never 0.0.

        A role's coverage rides with its figure on the run's terms: the
        `cost_incomplete`/`unreceipted_calls` pair appears only on a role some
        call's charge could not be read for, and it is what keeps the sum over
        these blocks a floor rather than a total once the run-wide qualifier is
        out of view.
        """
        record = {}
        for role in self._enabled_roles():
            acc = self._role_usage(role)
            record[role] = {
                "input_tokens": acc["input_tokens"],
                "output_tokens": acc["output_tokens"],
                "cache_read_tokens": acc["cache_read_tokens"],
                "cache_write_tokens": acc["cache_write_tokens"],
                "cost_usd": self._role_cost(role),
                "cost_rates": (self.rates[role].as_record()
                               if self.rates.get(role) is not None else None),
                **({"cost_incomplete": True,
                    "unreceipted_calls": acc["unreceipted"]}
                   if acc["incomplete"] else {}),
            }
        return record

    def _checkpoint_usage_to_meta(self):
        """Mirror the live cost and token accumulators into `self.session.meta`
        (in memory only, no disk write of its own). The next meta write flushes
        them: on the extractor path that is the per-tool-call
        `increment_tool_call_count` write, so the hot path adds no extra write
        syscall; the checker flushes per fan-out via `record_checker_calls`,
        the reviewer per call via its provenance write, and a text-only
        extractor turn (which drives no tool call) flushes explicitly.

        Flushing here rather than at `_pause` and `_finalise` alone is what
        survives a hard crash (SIGKILL, power loss) mid-run: with only the two
        terminal writes, every dollar and token since the last one is lost.
        Cost is rounded to 6 dp to match `_pause`/`_finalise`; the
        full-precision value stays in `self._cost_usd`, so re-accumulation
        never loses precision.

        `cost_rates` rides beside `cost_usd` on every flush, so the record
        never holds a dollar figure without the rates that produced it, even
        if the run dies between two calls.
        """
        meta = self.session.meta
        meta["cost_usd"] = self.recorded_cost()
        meta["cost_rates"] = self._cost_rates_record()
        if self._cost_incomplete:
            # Written only once a receipt has actually gone missing, so an
            # ordinary run's record carries no flag saying nothing was wrong,
            # and the count is what makes the figure beside it readable: a
            # dollar total next to "2 calls it does not cover" states its own
            # coverage. Both survive a resume through `_reseed_usage_from_meta`
            # — the earlier segment's gap is still a gap in the run's total.
            meta["cost_incomplete"] = True
            meta["unreceipted_calls"] = self._unreceipted_calls
        meta["input_tokens"] = self._input_tokens
        meta["output_tokens"] = self._output_tokens
        meta["cache_creation_tokens"] = self._cache_creation_tokens
        meta["cache_read_tokens"] = self._cache_read_tokens
        meta["usage_by_role"] = self._usage_by_role_record()

    def _reseed_usage_from_meta(self):
        """Reseed the cost and token accumulators from the resumed session's
        meta so the finalise-time record covers the WHOLE run, not just the
        post-resume segment. The last checkpoint (per tool call, per checker
        fan-out, per review call, or a clean `_pause`) left the pre-crash
        totals in meta; this reads them back exactly once. A session that
        never checkpointed (meta has no cost_usd) reseeds at zero.

        A checkpointed cost of `None` says the earlier segment ran unpriced,
        and that carries across: the run keeps stating no total even if this
        segment has rates, because adding rates now cannot cost calls already
        made. The per-role meters reseed on the same terms, each from its own
        block; a missing per-role block reseeds those meters at zero.

        This is the ONLY reseed site: post-resume spend adds onto this base
        and no already-counted call is re-issued, so nothing double counts.
        tool_call_count needs no reseed: it is persisted incrementally in
        meta and the cap check reads it from there.
        """
        meta = self.session.meta
        self._cost_unpriced = ("cost_usd" in meta and meta["cost_usd"] is None)
        self._cost_counted = isinstance(meta.get("cost_usd"), (int, float))
        self._cost_usd = float(meta.get("cost_usd") or 0.0)
        # A charge that could not be read in an earlier segment is still
        # missing from the reseeded sum, so the coverage carries across with
        # the money it qualifies.
        self._cost_incomplete = bool(meta.get("cost_incomplete"))
        self._unreceipted_calls = int(meta.get("unreceipted_calls") or 0)
        self._input_tokens = int(meta.get("input_tokens") or 0)
        self._output_tokens = int(meta.get("output_tokens") or 0)
        self._cache_creation_tokens = int(
            meta.get("cache_creation_tokens") or 0)
        self._cache_read_tokens = int(meta.get("cache_read_tokens") or 0)
        for role, block in (meta.get("usage_by_role") or {}).items():
            acc = self._role_usage(role)
            acc["input_tokens"] = int(block.get("input_tokens") or 0)
            acc["output_tokens"] = int(block.get("output_tokens") or 0)
            acc["cache_read_tokens"] = int(block.get("cache_read_tokens") or 0)
            acc["cache_write_tokens"] = int(
                block.get("cache_write_tokens") or 0)
            cost = block.get("cost_usd")
            acc["unpriced"] = ("cost_usd" in block and cost is None)
            acc["counted"] = isinstance(cost, (int, float))
            acc["cost_usd"] = float(cost or 0.0)
            # The earlier segment's unread charge is still unread, and it is
            # still this role's, so the coverage reseeds with the money it
            # qualifies rather than only with the run-wide pair above.
            acc["incomplete"] = bool(block.get("cost_incomplete"))
            acc["unreceipted"] = int(block.get("unreceipted_calls") or 0)

    def _accumulate_usage(self, response, model, role, *,
                          billed_refusal=False):
        """Fold one role's call into the run-wide meters and that role's own.

        `role` says whose card prices the call and whose per-role meters it
        lands in. It is required rather than derived from `model`, because two
        roles may legitimately run the same model and their spend still has to
        be told apart.

        `billed_refusal` marks a response that was served and charged and then
        REFUSED by direktoro's routing layer (see `_ledger_refused_call`). It
        changes exactly one thing: a routed response with no readable charge is
        stated as coverage on the run's figure (`cost_incomplete`, and the call
        counted in `unreceipted_calls`) instead of raising. On the ordinary path
        that raise is right — a routed answer the run is about to USE must not
        be ledgered at $0 — but here the refusal has already ended the stage,
        and raising over the accounting would replace the real failure with a
        report about it.
        """
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        # The cache write is TWO counters when it is priced, because the two
        # TTL tiers bill at different multiples of the base input rate. The
        # unsplit total above is what the run REPORTS (one number, the one the
        # provider reports); the split is what it is PRICED from. See
        # `rates.cache_write_split`.
        cache_write_5m, cache_write_1h = cache_write_split(usage)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cache_creation_tokens += cache_create
        self._cache_read_tokens += cache_read
        acc = self._role_usage(role)
        acc["input_tokens"] += input_tokens
        acc["output_tokens"] += output_tokens
        acc["cache_write_tokens"] += cache_create
        acc["cache_read_tokens"] += cache_read
        # Three costing paths. A ROUTED (gateway-served) model is priced FROM
        # the response (its `reported_cost`), a charge the gateway states
        # rather than one anybody predicts; a missing value is a loud fault,
        # never a silent $0. A DIRECT model is priced against THIS
        # ROLE's rate card, and the card is recorded with the figure so the
        # arithmetic stays checkable. A direct call whose role has no card is
        # not costed at all: its tokens are recorded, the role states no figure
        # and the run states no total.
        card = self.rates.get(role)
        if model_info(model).route is not None:
            if billed_refusal and getattr(response, "reported_cost",
                                          None) is None:
                # A refusal ABOUT the missing charge, or one raised before the
                # charge was read at all. The tokens above are the record; the
                # gap is stated rather than raised over (see the docstring).
                self._cost_incomplete = True
                self._unreceipted_calls += 1
                acc["incomplete"] = True
                acc["unreceipted"] += 1
                cost = 0.0
            else:
                cost = reported_cost_or_raise(model, response)
        elif card is not None:
            cost = card.cost_of_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write_5m,
                cache_write_1h_tokens=cache_write_1h,
            )
        else:
            cost = None
        if cost is None:
            self._cost_unpriced = True
            acc["unpriced"] = True
        else:
            self._cost_usd += cost
            self._cost_counted = True
            acc["cost_usd"] += cost
            acc["counted"] = True
        # Fold this call's transport + routing receipts (gateway generation id,
        # served upstream) into meta, where a consumer's ledger reads them.
        self._record_transport(model, response)
        # Checkpoint the running totals into meta so a hard crash keeps the
        # spend recorded up to the last flush (see _checkpoint_usage_to_meta).
        self._checkpoint_usage_to_meta()

    def _record_transport(self, model, response=None, *, generation_id=None,
                          served_provider=None):
        """Fold one call's transport and routing receipts into run.json.

        A consumer building a per-run ledger row reads two provenance fields
        off each run's meta:

          - `transport`: "direct" (every call went straight to its endpoint),
            "openrouter" (only gateway-served calls), or "mixed" when a run's
            stages used both (e.g. a direct extractor with a routed checker).
          - `generation_ids`: the gateway generation ids of the routed calls,
            in call order. These are external audit receipts a direct call has
            no equivalent for. Empty (but present) for a direct-only run.

        `served_provider` (the pinned upstream that served each routed call) is
        threaded in too, deduped, as a small provenance list. The generation id
        / served provider come off the NormalisedResponse for extractor/review
        calls; the checker fan-out passes them explicitly from its per-field
        provenance. Every run makes at least one call, so both keys always exist
        in meta after the first; the list is only appended to (never re-seeded),
        so a resumed run keeps the receipts it already gathered and adds more.
        """
        if response is not None:
            generation_id = getattr(response, "generation_id", None)
            served_provider = getattr(response, "served_provider", None)
        meta = self.session.meta
        routed = model_info(model).route is not None
        this = "openrouter" if routed else "direct"
        current = meta.get("transport")
        meta["transport"] = (this if current is None
                             else current if current == this else "mixed")
        # Always present (empty for a direct-only run) so the ledger never has to
        # distinguish "no routed calls" from "field missing".
        generation_ids = meta.setdefault("generation_ids", [])
        if routed and generation_id is not None:
            generation_ids.append(generation_id)
        if routed and served_provider is not None:
            served = meta.setdefault("served_providers", [])
            if served_provider not in served:
                served.append(served_provider)

    def _record_extractor_response(self, response, turn_id):
        # Saved per-turn in the jsonl via _extractor_loop's append_event;
        # nothing to do at the response level here.
        pass

    def _extract_tool_uses(self, response):
        """Return (list of {id, name, input}, assistant text)."""
        tool_uses = []
        text_parts = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif btype == "text":
                text_parts.append(block.text)
        return tool_uses, "\n".join(text_parts).strip()

    def _assistant_content_from_response(self, response):
        """Convert the SDK response content into plain dicts for messages."""
        out = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                out.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif btype == "text":
                out.append({"type": "text", "text": block.text})
        return out

    def _append_user_text(self, text):
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}],
        })

    # ----------------------------------------------------------------------
    # The inline checker
    # ----------------------------------------------------------------------

    # The two scopes whose fields carry an evidence slot. `initial_check.*` and
    # `quality_check.*` are bare values with no evidence and no note, so nothing
    # there can meet the trigger; excluding them by prefix says that once,
    # rather than letting each of the three conditions below rediscover it.
    _CHECKABLE_PREFIXES = ("study.", "record.")

    def _check_applied_fields(self, res, *, stage):
        """Check the fields this tool call applied, and merge the verdicts into
        its result dict.

        Called after the dispatcher has run and BEFORE the caller appends its
        session event or renders the tool result. The dispatcher itself stays
        deterministic and offline: it reports which fields applied, and the
        fan-out happens here, out of it.

        Two keys are written onto `res`:

          - `checker_challenges`, model-visible: `{field_path: rationale}` for
            the GENUINELY challenged fields only. An unchallenged field needs
            no words spent on it, and listing the `ok`s would bury the
            challenges the model is meant to weigh; an error-origin verdict is
            excluded because its rationale is an engine message about a failed
            call, and a model asked to answer it would be revising against
            plumbing text.
          - `_checker_verdicts`, underscore-prefixed and therefore stripped
            from the model-facing payload by `result_to_model_text`: the FULL
            per-field verdict set, including the `ok`s, each verdict's
            rationale, the value/evidence/note the checker actually
            scored, the error-origin flag, and the per-call tokens and cost.
            The session event carries this unstripped, and it is the only
            durable record of what the checker did.

        Neither key is written when nothing was checked, so a tool call that
        triggered no check serialises byte-for-byte as it would with the
        checker off. Because the merge happens before the event is appended, a
        resume replays the stored result through the same serialisation and
        re-sends identical bytes without making a single checker call.
        """
        calls, envelopes = self._build_checker_calls(
            res.get("applied_fields") or [])
        if not calls:
            return
        verdicts = self._run_checker_fanout(calls)
        # An error-origin verdict is excluded: its rationale is this engine's
        # report of a failed call, not a reading of the paper, and putting it
        # in front of the extractor would have a model revise a value against
        # plumbing text. A field whose check failed is simply unchecked, and
        # the run records that in `_checker_verdicts` and in
        # `checker_diagnostics.checker_errors`.
        challenges = {
            fp: (v.get("rationale") or "").strip()
            for fp, v in verdicts.items()
            if v.get("verdict") == "challenge" and not v.get("error_origin")
        }
        if challenges:
            res["checker_challenges"] = challenges
        res["_checker_verdicts"] = {
            fp: {
                "verdict": v.get("verdict"),
                "rationale": v.get("rationale", ""),
                "value_checked": envelopes.get(fp, {}).get("value"),
                "evidence_checked": envelopes.get(fp, {}).get("evidence"),
                # The EXTRACTOR's note on the field, as the checker was shown
                # it. The checker records no note of its own.
                "note_checked": envelopes.get(fp, {}).get("notes"),
                "error_origin": bool(v.get("error_origin")),
                # How many times this check had to be re-asked before a
                # verdict arrived; 0 on nearly every check. Kept per verdict
                # rather than only totalled, so the run's summary can be
                # rebuilt from the event log like every other checker figure.
                "reprompted": int(v.get("reprompted") or 0),
                "stage": stage,
                "input_tokens": v.get("input_tokens", 0),
                "output_tokens": v.get("output_tokens", 0),
                "cache_creation_tokens": v.get("cache_creation_tokens", 0),
                "cache_read_tokens": v.get("cache_read_tokens", 0),
                # None when the call was not priced. Absent reads the same way:
                # a verdict with no cost figure states none, rather than
                # defaulting to a zero that would read as a free call.
                "cost_usd": v.get("cost_usd"),
                # Carried only when a call's charge could not be read off its
                # response, and then saying how many calls the figure above
                # does not cover. A verdict whose cost is complete records no
                # flag, so this is present exactly where a reader must not add
                # the number up as if it were whole.
                **({"cost_incomplete": True,
                    "unreceipted_responses": v.get("unreceipted_responses")}
                   if v.get("cost_incomplete") else {}),
            }
            for fp, v in verdicts.items()
        }

    def _build_checker_calls(self, applied_paths):
        """Materialise the per-field checker calls for the fields one tool call
        applied. Returns `(calls, envelopes_by_path)`.

        A field is checked when all four hold:

          - it applied in THIS call (the dispatcher's own `applied_fields`
            report, so a field that failed validation is never checked and a
            partial call checks exactly the subset that landed);
          - it is a study or record field (the checkable scopes above);
          - its value is non-null;
          - its evidence is a non-empty string (pure prose counts: the checker
            judges the value against whatever grounds were offered; only a
            value with no grounds at all is uncheckable).

        and the field has not already had `max_checks_per_field` checks. The
        counter is incremented here, as the call is built, so it counts checks
        MADE rather than verdicts liked; an exhausted-retry call spent the same
        money and used the same slot.

        `calls` is sorted by field path so a batch is dispatched in the same
        order on any two runs of the same inputs.
        """
        if not self.checker_enabled:
            return [], {}
        checkable = []
        for path in applied_paths:
            if not isinstance(path, str):
                continue
            if not path.startswith(self._CHECKABLE_PREFIXES):
                continue
            if self._check_counts.get(path, 0) >= self.max_checks_per_field:
                continue
            envelope = self._extraction_record_field_envelope(path)
            if not isinstance(envelope, dict):
                continue
            if envelope.get("value") is None:
                continue
            evidence = envelope.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                continue
            spec, record = self._field_spec_and_record(path)
            if spec is None:
                continue
            checkable.append((path, envelope, spec, record))

        if not checkable:
            return [], {}

        # The four maps the checker resolves a citation through, all four the
        # WHOLE bundle's and all four keyed the same way. They travel together
        # because they answer one question between them — what this label is,
        # where its crop is, what it prints underneath, and what its content
        # says — and a checker holding three of them agrees with the extractor
        # about which labels exist while disagreeing about which crops it can
        # produce.
        checker_image_labels = self.image_labels
        checker_figures = self.image_figures
        checker_notes = self.image_notes
        checker_tables = self.image_tables

        calls = []
        envelopes = {}
        for path, envelope, spec, record in sorted(checkable,
                                                   key=lambda c: c[0]):
            if record is None:
                identity_context = self._study_identity_context()
            else:
                identity_context = self._record_identity_context(record)
            user_blocks = build_checker_user_message(
                field_path=path,
                field_spec=spec,
                envelope=envelope,
                identity_context=identity_context,
                image_labels=checker_image_labels,
                partials_dir=self.config.partials_dir,
                # The run's one predicate map, from the instrument that owns
                # the structure toggles, so this per-field scaffold and the
                # checker system prompt cached beside it resolve their
                # conditional partials the same way, and checker_fp covers the
                # text that was actually sent.
                predicates=self.instrument.predicates(),
                # The same lists rendered into the three system prompts, so a
                # scaffold citing one shows the checker the names the
                # validator will hold its verdict's field to.
                reference_lists=self.reference_lists,
                figures=checker_figures,
                # The footnote each attached crop prints, where the manifest
                # records one. It is inside the crop already, so it adds no
                # context to the checker's narrow view; it makes the smallest
                # print on the exhibit legible without resolving it off pixels.
                exhibit_notes=checker_notes,
                # The content of each attached crop as text, where the bundle
                # transcribes it. On the footnote's side of the checker's
                # narrow view rather than the caption's: it is the exhibit's
                # own content, which the crop already carries as pixels, and
                # not a description of the exhibit from outside it. What it
                # buys is that a cell can be read rather than resolved off an
                # image, which is the whole of what a checker looking at a
                # table is doing.
                exhibit_tables=checker_tables,
                # The paper text the quotes are windowed into, and how wide
                # the window is. The paper is the run's INPUT and rides in no
                # fingerprint; the width is config identity and rides in
                # checker_fp.
                paper_text=self.paper_text,
                context_chars=self.checker_config.context_chars,
            )
            self._check_counts[path] = self._check_counts.get(path, 0) + 1
            calls.append({
                "field_path": path,
                "user_message_blocks": user_blocks,
                # 1-based ordinal of this check for this field, so the API
                # audit log says whether a call was the first look or the
                # re-check after a revision.
                "check_index": self._check_counts[path],
            })
            envelopes[path] = {
                "value": envelope.get("value"),
                "evidence": envelope.get("evidence"),
                "notes": envelope.get("notes"),
            }
        return calls, envelopes

    def _field_spec_and_record(self, field_path):
        """`(field_spec, record_or_None)` for a dotted field path, or
        `(None, None)` when the template declares no such field.

        A study path yields `(spec, None)`; a record path yields the spec plus
        the record dict, which the identity context needs to build its label.
        """
        parts = field_path.split(".")
        if len(parts) == 2 and parts[0] == "study":
            specs = {f["variable"]: f
                     for f in iter_fields(self.template["study_fields"])}
            return specs.get(parts[1]), None
        if len(parts) == 3 and parts[0] == "record":
            record_id, var = parts[1], parts[2]
            record = next(
                (r for r in self.extraction_record.records
                 if r.get("record_id") == record_id), None)
            if record is None:
                return None, None
            specs = {f["variable"]: f
                     for f in iter_fields(self.template["record_fields"])}
            spec = specs.get(var)
            return (spec, record) if spec is not None else (None, None)
        return None, None

    def _run_checker_fanout(self, calls):
        """Run one batch of per-field checker calls in parallel and return
        `{field_path: verdict}`.

        Concurrency is `checker_config.concurrency`. The first `update_study`
        typically carries every populated study field, so a first batch of
        roughly two dozen concurrent calls is the expected shape.

        An exhausted-retry API failure degrades that one field to a challenge
        tagged `error_origin` (checker.run_checker_batch) rather than aborting
        the batch or the run.
        """
        sys_blocks = checker_sys_blocks(self._render_checker_system_text())
        for c in calls:
            c["system_message_blocks"] = sys_blocks

        # Routing receipts for this batch, collected DURING the fan-out but
        # folded into meta AFTER it in sorted field_path order. The
        # `on_complete` callback fires in thread-COMPLETION order
        # (nondeterministic), so appending generation ids straight to meta here
        # would serialise them in a different order every run and churn any
        # ledger cell built from them. `run_checker_batch` already sorts its
        # verdicts by field_path for reproducibility; the receipts must match.
        receipts = []

        def _on_complete(field_path, result):
            # A checker call that could not be costed (no rate card for the
            # checker, no gateway charge) carries None, which withholds the
            # checker's figure and the run's total; the tokens are still
            # counted.
            acc = self._role_usage("checker")
            cost = result.get("cost_usd")
            if cost is None:
                self._cost_unpriced = True
                acc["unpriced"] = True
            else:
                self._cost_usd += cost
                self._cost_counted = True
                acc["cost_usd"] += cost
                acc["counted"] = True
            # A check whose figure covers fewer calls than it made: a
            # gateway-served response was charged and then refused by the
            # routing layer, so its charge never became readable, and the check
            # reported what it could price (checker._spend). The run's total
            # inherits the gap, because it is the sum that a reader would
            # otherwise take for the whole bill.
            if result.get("cost_incomplete"):
                unreceipted = int(result.get("unreceipted_responses") or 0)
                self._cost_incomplete = True
                self._unreceipted_calls += unreceipted
                # And onto the role that made the call, because the per-role
                # record is what a reader sums when they want one stage's
                # bill.
                acc["incomplete"] = True
                acc["unreceipted"] += unreceipted
            self._input_tokens += result.get("input_tokens", 0)
            self._output_tokens += result.get("output_tokens", 0)
            self._cache_creation_tokens += result.get(
                "cache_creation_tokens", 0)
            self._cache_read_tokens += result.get("cache_read_tokens", 0)
            acc["input_tokens"] += result.get("input_tokens", 0)
            acc["output_tokens"] += result.get("output_tokens", 0)
            acc["cache_write_tokens"] += result.get("cache_creation_tokens", 0)
            acc["cache_read_tokens"] += result.get("cache_read_tokens", 0)
            # Mirror the running checker spend into meta in memory; the tool
            # call boundary flushes it to disk, so a crash loses at most this
            # call's partial checker spend rather than the whole run's.
            self._checkpoint_usage_to_meta()
            # Record the checker's resolved model + raw decoding params in
            # run.json once. `_provenance` is model-invisible
            # telemetry; strip it here so it never reaches the verdict record.
            # on_complete runs on the main thread, so this meta write does not
            # race the fan-out.
            prov = result.pop("_provenance", None)
            if prov:
                # Transport is set here (order-independent: it only records
                # direct vs openrouter). The generation id / served provider
                # are buffered keyed by field_path and folded in AFTER the
                # batch in deterministic order, NOT appended here in completion
                # order.
                self._record_transport(self.checker_config.checker_model)
                receipts.append((
                    field_path, prov.get("generation_id"),
                    prov.get("served_provider")))
                if "checker_model_resolved" not in self.session.meta:
                    self.session.meta["checker_model_resolved"] = (
                        prov.get("resolved_model"))
                    self.session.meta.setdefault(
                        "decoding_params", {})["checker"] = (
                            prov.get("decoding_params"))
                    self.session.write_meta()

        def _api_log(request_kwargs, response, **extra):
            self.session.log_api_call(
                "checker", request_kwargs, response, **extra)

        verdicts = run_checker_batch(
            calls=calls, config=self.checker_config,
            # ONE adapter for the whole run, not one per batch. The adapter
            # owns the provider client and through it the connection pool, so
            # building a fresh one each time discards every kept-alive
            # connection and makes the first call of every fan-out pay a new
            # TLS handshake — on a batch of two dozen parallel calls, two dozen
            # of them.
            adapter=self._checker_adapter(),
            on_complete=_on_complete,
            api_logger=_api_log,
            # The checker's retries recorded the same way the extractor's and
            # the reviewer's are: a failed attempt raises before the wire log
            # runs, so this event is its only trace.
            on_retry=self._retry_logger("checker"),
        )
        # Fold this batch's routing receipts into meta in field_path order (the
        # same reproducible order run_checker_batch returns its verdicts in), so
        # two identical re-runs serialise generation_ids identically regardless
        # of thread completion order.
        for _fp, generation_id, served_provider in sorted(
                receipts, key=lambda r: r[0]):
            self._record_transport(
                self.checker_config.checker_model,
                generation_id=generation_id, served_provider=served_provider)
        self.session.record_checker_calls(len(calls))
        return verdicts

    def _reconstruct_check_counts(self):
        """Per-field check counts for this session, rebuilt from the event log.

        The per-field budget spans the whole session, so a resumed segment
        must not hand a field a fresh allowance. Nothing stores the counts:
        they are counted back off the `_checker_verdicts` on each tool call's
        result; a copy in meta would be a second source of truth able to
        drift. Every call built produces exactly one verdict entry, including
        the exhausted-retry ones, so the reconstructed count equals the
        checker calls the field actually received.
        """
        counts = {}
        for ev in self.session.read_events():
            result = ev.get("result")
            if not isinstance(result, dict):
                continue
            for fp in (result.get("_checker_verdicts") or {}):
                counts[fp] = counts.get(fp, 0) + 1
        return counts

    def _checker_diagnostics(self):
        """The run's checker summary for run.json, rebuilt from the event log.

        A challenge is advisory, so a field the extractor never satisfied does
        not hold the run open. It is still worth naming: this is where a
        consumer (or a human deciding whether to look at a cell) reads which
        fields the checker was last unhappy with, and how much checking the run
        bought.

        `unresolved_challenges` lists the fields whose LAST verdict was a
        genuine challenge. `checker_errors` lists those whose last verdict was
        an exhausted-retry artefact, which is an absence of information rather
        than an objection, and so is reported apart from it.

        `checks_reprompted` counts the checks whose reply had to be re-asked
        before a verdict arrived (see `checker.MAX_TOOL_FREE_REPROMPTS`). It
        is a fact about the CHECKER MODEL, not about any field: a model that
        needs nudging is marginal for the role, and one that needs it often is
        the wrong choice even when every nudge worked. Reported beside
        `checker_errors` and never merged into it — a check that was re-asked
        and then answered produced a real verdict, while an error is the
        absence of one.
        """
        last = {}
        total = 0
        reprompted = 0
        for ev in self.session.read_events():
            result = ev.get("result")
            if not isinstance(result, dict):
                continue
            for fp, v in (result.get("_checker_verdicts") or {}).items():
                last[fp] = v
                total += 1
                if v.get("reprompted"):
                    reprompted += 1
        unresolved = sorted(
            fp for fp, v in last.items()
            if v.get("verdict") == "challenge" and not v.get("error_origin"))
        errored = sorted(
            fp for fp, v in last.items() if v.get("error_origin"))
        return {
            "fields_checked": len(last),
            "checks_run": total,
            "checks_reprompted": reprompted,
            "unresolved_challenges": unresolved,
            "checker_errors": errored,
        }

    def _finalise_loop_stop(self, status):
        """Map a hard-stop extractor loop outcome to its terminal action, or
        return None when `status` is not a stop.

        Returns the run() return value (a status string) for a stop; None for
        `mark_complete_validated`, which the caller continues past, and None
        for any status this mapping does not know. None is therefore not an
        all-clear: run() names `mark_complete_validated` explicitly and raises
        on anything else, so an unmapped outcome cannot continue as if the
        extractor had declared completion.
        """
        if status == "tool_cap_hit":
            # A resumable PAUSE: leave the session in_progress so --resume can
            # raise the cap and continue the same conversation.
            return self._pause("tool_cap_hit")
        if status == "text_only_stall":
            return self._finalise("failed_validation",
                                  failure_reason="text_only_stall")
        if status == "extractor_stalled":
            return self._finalise("failed_validation",
                                  failure_reason="stalled")
        if status == "extractor_abandoned":
            return self._finalise("failed_validation",
                                  failure_reason="surrendered")
        if status == "extractor_refused":
            # `error`, not `failed_validation`: the two failed_validation
            # outcomes are judgements the model made about the extraction (it
            # surrendered, or it wedged). A refusal is a judgement about the
            # REQUEST, made by a filter or by the model declining, and nothing
            # about the paper or the template was assessed at all — so there is
            # no extraction to call invalid.
            self._report_run_error(
                "the extractor model refused the request (stop_reason "
                "'refusal'). Check the paper text and the rendered prompt for "
                "material the provider's filter blocks, or run the stage on "
                "another model.")
            return self._finalise("error")
        return None

    def _finalise_review_stop(self, status):
        """Map a hard-stop REVIEW loop outcome to its terminal action, or
        return None when `status` is not a stop.

        The sibling of `_finalise_loop_stop` for the review stage. All four
        end the run without the reviewer confirming the extraction, so all
        four map to `failed_validation`, and none pauses: the review
        conversation is fresh-context and never replayed, so there is nothing
        for a resume to continue.

        The `failure_reason` values are review-prefixed so run.json alone
        says which stage failed and how: `surrendered` is the extractor's,
        `review_surrendered` the reviewer's. Each maps 1:1 to an event in
        tool_calls.jsonl.

        As with `_finalise_loop_stop`, None means only "not one of these
        four"; run() handles the rest and refuses to finalise `complete` on
        anything but the reviewer's confirmation.
        """
        if status == "review_abandoned":
            return self._finalise("failed_validation",
                                  failure_reason="review_surrendered")
        if status == "review_cap_hit":
            return self._finalise("failed_validation",
                                  failure_reason="review_cap_hit")
        if status == "review_text_only_stall":
            return self._finalise("failed_validation",
                                  failure_reason="review_text_only_stall")
        if status == "review_stalled":
            return self._finalise("failed_validation",
                                  failure_reason="review_stalled")
        return None

    def _pause(self, pause_reason):
        """Leave the session in_progress and resumable, recording why it
        paused. Unlike `_finalise` this sets NO terminal status, appends no
        terminate event, and writes no run-log entry: a paused session is not
        finished, it will be resumed and finalised later.

        The tool-call cap is the one bound that pauses rather than terminates:
        raising the cap and resuming continues the same conversation. The
        pause_reason is cleared on resume (see resume_session). Returns
        "in_progress" so the CLI prints the resume hint and exits 0.

        This is phase (a) of stopping and it writes only what makes the pause
        durable and resumable. The derived documents (field history,
        transcript) are `run`'s `_render_artefacts`, outside the try: rendering
        them here once cost a paused run its resumability, because a rendering
        fault fell into the catch-all and finalised the session `error`.
        """
        self.session.write_extraction_record(self.extraction_record)
        self._checkpoint_usage_to_meta()
        self.session.meta["pause_reason"] = pause_reason
        self.session.append_event({"event": "extractor_paused",
                                   "pause_reason": pause_reason})
        self.session.write_meta()
        return "in_progress"

    def _finalise(self, status, *, failure_reason=None):
        """Finalise the session into a terminal status and append the run-log
        entry. Returns the status the session is persisted with.

        `failure_reason` (surrendered / stalled / text_only_stall from the
        extractor; review_surrendered / review_cap_hit / review_text_only_stall
        / review_stalled from the reviewer) is recorded in run.json for
        `failed_validation`, so the status string stays coarse while the
        mechanism, and the stage it fired in, stay auditable.

        Phase (a) of stopping (see `run`), and the whole of it: the terminal
        status and the run-log entry go out together, with nothing fallible
        between them. The derived documents come after, outside run()'s try.

        Never raises. The status and the sentence behind it are the outcome of
        the run, and they land whatever the cross-session ledger does: a ledger
        that cannot be appended to is reported (`_report_unwritten_run_log`)
        and the run keeps the status it earned. Raising instead would put a
        traceback where a status belongs — and, called from run()'s except
        handler, would carry the ledger's fault out of run() in place of the
        error that actually ended the run.

        Idempotent by the latch below. A session finalises ONCE: the run log is
        a cross-run ledger a consumer sums into a bill, so a second entry for
        one session is not a duplicate line but double-counted money. The
        invariant is therefore: exactly one entry when the log is writable, and
        the session says so when it is not.
        """
        if self._terminal_status is not None:
            # Already persisted, so there is nothing left to decide. This is
            # reached when a fault after the first finalisation unwinds into
            # run()'s catch-all, which finalises `error` — the fault is real
            # and is in the event log, but the outcome on disk is the one that
            # actually happened and it stands.
            return self._terminal_status
        self.session.write_extraction_record(self.extraction_record)
        # Tripwires that make a claim about the finished artefact are
        # evaluated here, against the output just written, so a persisted
        # warning is true of what shipped rather than of some state the run
        # passed through. `status` is passed because a claim about the
        # extraction only means something on a run that reached an answer it
        # considered final; see the method.
        self._check_shipped_summary_mismatch(status)
        self._check_shipped_required_fields(status)
        # Stash cost + token counters onto meta so run.json carries them
        # beside the rest of the run record, where a reader of one session
        # finds them without opening the run log. run_log.json carries the
        # same numbers across sessions; the duplication is deliberate, so
        # neither a single session nor the cross-session log needs the other
        # to be read. (The per-call checkpoints keep meta current mid-run;
        # this is the authoritative final flush.)
        self._checkpoint_usage_to_meta()
        if failure_reason is not None:
            self.session.meta["failure_reason"] = failure_reason
        else:
            # A first `_finalise` that faulted part-way through can already
            # have persisted the reason keys of the stop it was writing, and
            # the finalisation that lands is then run()'s `error` — which has
            # no reason. Clearing them here keeps run.json's reason true of the
            # status beside it, rather than describing a stop this session did
            # not finish making.
            self.session.meta.pop("failure_reason", None)
            self.session.meta.pop("failed_validation_reason", None)
        if self.checker_enabled:
            self.session.meta["checker_diagnostics"] = (
                self._checker_diagnostics())
        # The stated surrender reason (from abandon_extraction) lands in meta
        # and drives the run-log validation_errors detail. Either stage can
        # surrender: the flag and its reason live on the shared extraction
        # record, and mean the same thing from whichever stage latched them
        # (this model, with tool access, judged that no valid extraction can be
        # produced honestly from these inputs).
        surrender_detail = None
        if status == "failed_validation" and failure_reason in (
                "surrendered", "review_surrendered"):
            surrender_detail = self.extraction_record.abandon_reason
            if surrender_detail:
                self.session.meta["failed_validation_reason"] = surrender_detail
        # The sentence behind an `error` status, put where a reader of the run
        # record finds it: in run.json beside the status, and in the run-log
        # entry in place of the bare status word it used to carry. "error" on
        # its own names a category, not what went wrong, and the run log is
        # read by consumers that never open the session.
        if self._error_message:
            self.session.meta["error_message"] = self._error_message
        validated = status in VALIDATED_STATUSES
        if validated:
            validation_errors = []
        elif surrender_detail:
            validation_errors = [f"{status}: {surrender_detail}"]
        elif self._error_message:
            validation_errors = [f"{status}: {self._error_message}"]
        else:
            validation_errors = [status]
        self.session.finalise(status)
        # Latched the moment the status is on disk, before the ledger append
        # below can raise: from here on this session HAS a terminal status, and
        # a re-entry must answer with it rather than write another.
        self._terminal_status = status
        try:
            append_session_entry(
                self.session,
                log_dir=self.out_dir,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cache_creation_tokens=self._cache_creation_tokens,
                cache_read_tokens=self._cache_read_tokens,
                cost_usd=self.recorded_cost(),
                cost_rates=self._cost_rates_record(),
                usage_by_role=self._usage_by_role_record(),
                validation_passed=validated,
                validation_errors=validation_errors,
            )
        except Exception as e:
            self._report_unwritten_run_log(e)
        return status

    def _report_unwritten_run_log(self, exc):
        """Record that this run finished with no line in the cross-session
        ledger: one stderr line, a `meta.warnings` sentence, and the flag
        `run_log_entry_written: false` in run.json.

        The ledger is written under `--out`, which the session is not: a full
        disk, a lock, a permission on the run root alone are all faults that
        reach the append and nothing else. The run itself is over and its own
        record is complete, so the fault is reported rather than raised — but
        it cannot be swallowed either, because a consumer summing run_log.json
        into a bill would otherwise be short this run's spend with nothing
        anywhere saying so. The flag is what makes that legible without reading
        prose: absent means the entry was written, and the ledger is the whole
        story; `false` means this session is missing from it.

        Each of the three reports is guarded, and guarded separately, for the
        reason `_render_artefacts` guards its warning: the fault that closed
        the ledger can have taken meta or stderr with it, a raise here would
        carry that fault out of `_finalise` in place of the status, and one
        report being unavailable says nothing about the other two.
        """
        message = (
            f"run-log entry could not be written: {type(exc).__name__}: "
            f"{exc}. The run's own record is complete — run.json carries the "
            f"status and this session's spend — but {self.out_dir}/"
            f"run_log.json has no line for session "
            f"{self.session.session_dir}, so any total summed from the run "
            f"log alone is short by this run.")
        try:
            print(f"WARNING: {message}", file=sys.stderr)
        except Exception:
            pass
        try:
            self.session.meta["run_log_entry_written"] = False
            self.session.write_meta()
        except Exception:
            pass
        try:
            self.session.add_warning(message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_identity(model, resolved_decoding):
    """The canonical provider-call identity block for a stage's `model`, folded
    into that stage's fingerprint (direktoro owns the block).

    Returns `direktoro.canonical_json(direktoro.call_identity_fields(...))`: the
    model id, provider, base_url, Route (for a gateway-routed model), and the
    RESOLVED decoding params (`resolved_decoding`, exactly what the adapter
    sends) keyed under the wire's own parameter name. One block instead of
    separate provider/base_url/decoding fingerprint inputs, so the wire dialect
    lives in one place (direktoro) and meltiro never double-applies it.

    direktoro is imported lazily here so `import meltiro` never pulls
    direktoro.routing at module scope: a consumer that installs the wheel
    `--no-deps` has no direktoro, and only the run-time fingerprint path
    (never plain `import meltiro`) reaches this helper.
    """
    from direktoro import call_identity_fields, canonical_json
    info = model_info(model)
    return canonical_json(call_identity_fields(
        model, route=info.route, decoding_params=resolved_decoding))


# Fuzzy-match threshold for the summary-mismatch tripwire. At/above this
# difflib ratio (with neither summary containing the other) the two
# summaries are treated as the same paper; below it, divergent.
_SUMMARY_MATCH_RATIO = 0.6


def _summaries_match(a, b):
    """Fuzzy equality for the summary-mismatch tripwire.

    Both strings are normalised with quote_check.normalise_quote_text (the
    same canonicalisation used for verbatim-quote matching: NFKC folding,
    smart-quote and dash folding, whitespace collapse, lowercase). They
    count as matching when either normalised form CONTAINS the other (a
    truncated search-index abstract vs the full paper abstract is a common,
    benign case) OR their difflib SequenceMatcher ratio is at least
    _SUMMARY_MATCH_RATIO. Below that, with neither containing the other, they
    count as divergent. An empty normalised form on either side means
    "cannot compare" and does NOT warn.

    The ratio is sensitive to length: a partially-written abstract can score
    below the threshold against the same paper's manifest summary and then
    pass comfortably once complete. That is why the persisted warning is
    decided at finalisation, on the shipped value, and only advised mid-run
    (see `_check_shipped_summary_mismatch`).
    """
    import difflib

    from meltiro.quote_check import normalise_quote_text
    na = normalise_quote_text(a)
    nb = normalise_quote_text(b)
    if not na or not nb:
        return True
    if na in nb or nb in na:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= _SUMMARY_MATCH_RATIO


def _tool_call_event_name(status):
    """Audit event name for a dispatched tool call, keyed by its status.

    A `partial` dispatch (some fields applied, some rejected) earns its
    own honest label rather than being logged as a plain failure: work
    did land, and the transcript should say so.
    """
    if status == "ok":
        return "tool_call_applied"
    if status == "partial":
        return "tool_call_partial"
    return "tool_call_failed"


def _failure_signature(tool_name, res):
    """Signature of a failed tool call: (tool name, sorted error codes).

    The repeated-failure guard treats two failures as "identical" when this
    matches. Error codes come from the dispatch result's flat `errors` list
    (each entry carries a `code`); they are sorted so batch ordering cannot
    make the same underlying failure look different from turn to turn. The
    field paths and messages are deliberately excluded: a model re-submitting
    the same broken call keeps the same codes, which is the case to catch.
    """
    codes = tuple(sorted(
        str(e.get("code"))
        for e in (res.get("errors") or [])
        if isinstance(e, dict) and e.get("code") is not None
    ))
    return (tool_name, codes)


def _first_error_message(res):
    """First human-readable error message from a dispatch result, or None.

    Recorded once on the stall event so the run log carries the reason the
    call kept failing without repeating it for every identical attempt.
    """
    for e in (res.get("errors") or []):
        if isinstance(e, dict) and e.get("message"):
            return e["message"]
    return None


class _IdenticalFailureRun:
    """The repeated-failure guard's state: a run of consecutive tool calls
    that failed with an identical `_failure_signature`.

    Shared by the extractor and review loops — an applied or partial call
    resets the run; a fully-failed call extends it (identical signature) or
    starts a new one — so the two guards cannot drift apart. Instances are
    loop-local: a resumed session starts a fresh run.
    """

    def __init__(self, limit):
        self.limit = limit
        self.signature = None
        self.count = 0

    def record(self, tool_name, res):
        """Fold one dispatch result into the run.

        Returns the offending `(tool name, sorted error codes)` signature once
        `limit` identical failures have run consecutively (the caller should
        then stop), and None otherwise.
        """
        if res["status"] in ("ok", "partial"):
            self.signature = None
            self.count = 0
            return None
        signature = _failure_signature(tool_name, res)
        if signature == self.signature:
            self.count += 1
        else:
            self.signature = signature
            self.count = 1
        if self.count >= self.limit:
            return signature
        return None


# The canonical tool_result serialisation (underscore-key stripping) lives in
# meltiro.session as result_to_model_text, imported at the top of this module.
# The live loop and replay_messages share that one function so a resumed run
# rebuilds byte-identical tool_result content.
