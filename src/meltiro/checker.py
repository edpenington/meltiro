"""Narrow per-field checker LLM client + ThreadPoolExecutor fan-out.

One call per field a tool call just wrote, in parallel up to a configurable
concurrency. Each call gets a cached system prompt (the checker's engine spine
followed by the config bundle's checker prompt and the rendered reference
lists, with no field catalogue in it) plus a small per-field user message
(field identity, the field's definition and allowed values, identity context,
evidence, value, and the field's note). Returns one of `{ok, challenge}` plus a
one-sentence rationale.

The verdict arrives as a tool call (`record_verdict`, defined in
`meltiro.tools`), read off the response by block type. That is what makes the
shape of an answer the engine's business and its content the config bundle's:
the bundle says what the checker is for and how to weigh evidence, and can be
rewritten freely for another review without any edit to it being able to break
how a verdict is read. It also means a reply that leads with reasoning blocks,
or with a sentence of preamble, is read exactly like one that does not.

The checker is a probabilistic extension of the deterministic validator, not a
pipeline stage: the orchestrator fans out over the fields one tool call
applied, and the challenges come back in that call's tool result alongside the
validation errors. Each check sees only the current value, its evidence read in
the surrounding paper text (see `DEFAULT_CONTEXT_CHARS` below and
`checker_prompts`), and the field note, so a re-check after a revision judges a
genuinely fresh context. A reply that records no verdict is re-asked once, as a
correction: a reply that was nothing but text is replayed back as the model's
own turn with the correction after it, and any other reply is corrected in the
field's message itself, so either way the re-asked call is still looking at one
field and nothing else.

Each call goes through direktoro's adapter as one blocking `create_message`
and comes back complete; how that reaches the wire is the adapter's business,
and nothing here depends on it. This module retries with backoff on 429/5xx,
accounts prompt-cache cost, and reports every failure as a structured
`CheckerError`, which is what lets one bad field degrade rather than abort the
extraction. Each such failure carries the spend its asks had already incurred,
and costing one never raises: a check that cannot be priced is reported as a
figure with its coverage stated, because the alternative is one field's
plumbing discarding a batch of paid sibling verdicts.
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meltiro.errors import CheckerError, truncation_report
from meltiro.rates import cache_write_split
from meltiro.fingerprint import (
    checker_config_fingerprint,
    structure_hash,
    field_catalogue_hash,
    reference_lists_hash,
    tool_set_hash,
)
from meltiro.tools import (
    CHECKER_VERDICT_TOOL_NAME,
    CHECKER_VERDICTS,
    checker_tool_definitions,
)

# `direktoro` is imported LAZILY throughout this module, inside the functions
# that actually reach a provider. `meltiro.config_bundle` imports this module,
# so an eager `from direktoro import ...` here would make merely READING a
# config bundle (parsing YAML, no network, no key, no model) require the
# provider layer to be installed — and a consumer that installs the wheel
# `--no-deps` has no direktoro yet still loads bundles, to validate writes
# against the template they carry.
#
# `from __future__ import annotations` keeps the `Thinking` annotation on
# `CheckerConfig.thinking` a string, so the dataclass does not need the type
# at import time either.
if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from direktoro import Thinking


DEFAULT_CONCURRENCY = 10
# Characters of surrounding paper text shown on EACH side of a matched quote
# in the per-field checker message (pipeline.yaml's `checker_context_chars`).
# Enough to carry a table's header row or the sentence a pronoun points back
# to, small enough that the checker is still reading a neighbourhood rather
# than the paper. 0 turns context off.
DEFAULT_CONTEXT_CHARS = 1000

# The verdicts a checker call may come back with, derived from the schema the
# checker is sent (`meltiro.tools.CHECKER_VERDICTS`) so the vocabulary offered
# and the vocabulary accepted are one value. A tool call is coaxed by its
# schema rather than constrained by it, so the returned verdict is still
# checked against this before it is believed.
VALID_VERDICTS = frozenset(CHECKER_VERDICTS)

# How many times one field's check may be re-asked when the reply records no
# verdict. One: a checker model under a forced tool_choice should not need
# even that, and a model that declines twice is misconfigured — the
# error-origin challenge says so at a cost of one extra call rather than
# several.
MAX_TOOL_FREE_REPROMPTS = 1

# The correction a verdict-free checker reply is re-asked with. Four replies
# reach it: prose with no call in it, the verdict tool called twice, called
# with no arguments, and called with something that is not an object of
# arguments. It therefore states the OUTCOME all four share — this field still
# has no verdict — rather than a mechanism that would be true of only one of
# them, and asks for the shape the other three got wrong: one call, arguments
# in one object. Engine framing, like the extractor's and the reviewer's in
# `meltiro.prompt_builder`: the config bundle says what the checker is for, and
# the engine says how an answer reaches it. It rides in no fingerprint for the
# same reason theirs do not — `engine_fp` identifies it.
CHECKER_TOOL_REPROMPT = (
    f"That reply recorded no verdict for this field. Record it now by calling "
    f"the {CHECKER_VERDICT_TOOL_NAME} tool exactly once, with its arguments "
    f"as a single object."
)


@dataclass
class CheckerConfig:
    """How the checker's provider CALL is made, and nothing about the pipeline.

    Model, key, decoding knobs, the width of the quote context, and the path
    of the prompt file this role sends. What it deliberately does not
    hold is the run's pipeline STRUCTURE (the per-field check budget, whether
    a reviewer runs): that is the instrument's
    (`meltiro.instrument.Instrument`). Both `user_prompt_template_text` and
    `fingerprint` below REQUIRE the predicate map as an argument, and
    `fingerprint` requires the check budget beside it; the only thing that
    produces either is the `Instrument`, so the checker's rendered prompts
    and its `checker_fp` cannot describe a different pipeline from the one
    the extractor and reviewer were briefed on.
    """

    # No hardcoded fallback model: there is deliberately no default checker
    # model. A real run must supply one (the CLI requires checker_model from
    # pipeline.yaml or the --checker-model flag, exactly like the extractor
    # and review models). The None default lets tests and programmatic use
    # construct a config and set the model explicitly.
    checker_model: str = None
    # The checker's output-token cap. No default value, on the same terms as
    # the model above: the number bounds what one check may spend and what it
    # may answer within, so a real run states it (the CLI requires
    # `checker_max_tokens` from pipeline.yaml whenever the checker stage is
    # on). The None default lets a caller construct a config and set it
    # explicitly; `_require_cap` refuses a config that still carries None when
    # a call is about to be made.
    max_tokens: int | None = None
    # The sampling controls the operator specified for the checker, as a
    # `{name: value}` mapping over `direktoro.SAMPLING_PARAMS`, or None for
    # "specified none". There is deliberately no default value for any of them:
    # a number nobody chose is indistinguishable from a number somebody did,
    # and it would be reported as inert against a model that refuses it. An
    # unspecified control is not sent, and the model's own default applies.
    sampling: dict = None
    concurrency: int = DEFAULT_CONCURRENCY
    # Characters of surrounding paper text the checker sees on each side of a
    # matched quote. Config identity, not an operational budget: it changes
    # what the checker is asked, so it folds into `fingerprint()` below. Like
    # the other decoding knobs it comes from the config bundle only, never
    # from the environment.
    context_chars: int = DEFAULT_CONTEXT_CHARS
    # The checker's system prompt comes from the config bundle
    # (`ConfigBundle.checker_system_path`); the orchestrator sets it at
    # construction time. No CWD-relative default; a real run must supply it.
    # It also locates `prompts/partials/`, which is where a bundle's overrides
    # of the checker's engine sections and of the per-field scaffold are read
    # from, so the two prompts this role sends resolve against one directory.
    system_prompt_path: str = None
    # The checker's own thinking / reasoning-effort spec, or None to say
    # nothing and leave the model doing whatever its default is. From the
    # config bundle only (the thinking fields of pipeline.yaml's
    # `checker_decoding` block), so the same tree yields the same checker_fp
    # under any shell; it rides the same `resolved_decoding_params` call as
    # the sampling controls and `max_tokens`. Declared after the fields above
    # so positional construction of them is unaffected.
    thinking: Thinking | None = None
    # The CHECKER's USD rate card (`meltiro.rates.Rates`), or None to run the
    # checker unpriced. The Orchestrator takes it out of the run's per-role
    # mapping at construction, so the numbers pricing the checker are the
    # ones resolved for the checker's own model. Commercial rather than
    # methodological, so deliberately absent from `checker_fp`. Declared
    # after the fields above so positional construction of them is
    # unaffected.
    rates: object = None

    @classmethod
    def from_env(cls, model_override=None, concurrency_override=None):
        # The model comes from the caller alone — the config bundle's
        # `checker_model`, or `--checker-model` — and from no environment
        # variable. A shell-supplied model would change what the checker is
        # asked while the bundle that names the run looked unchanged, and
        # `checker_model` rides in `checker_fp`: two runs of one tagged tree
        # would then record different fingerprints for no recorded reason.
        # Same rule as the decoding knobs below.
        model = model_override
        # The checker's decoding knobs (max_tokens, the sampling controls) and its
        # quote-context width are NOT read from the environment: they come
        # from the config bundle via the CLI, so a tagged bundle fully
        # specifies what the checker is asked and the same tree yields the
        # same checker_fp under any shell. The dataclass defaults apply until
        # the CLI overrides them.
        #
        # `is not None`, not a truthy test, so an explicit 0 is read as the
        # value the caller asked for — and then refused: `concurrency`
        # becomes ThreadPoolExecutor's `max_workers`, whose domain is the
        # positive integers, so 0 is neither a smaller pool nor a way to turn
        # the checker off (`max_checks_per_field: 0` is that). Falling back
        # to a default instead would run the checker at a parallelism nobody
        # asked for and record no trace of the substitution.
        if concurrency_override is not None:
            concurrency = int(concurrency_override)
            if concurrency < 1:
                raise ValueError(
                    f"checker concurrency must be a positive integer, got "
                    f"{concurrency}. It is how many checker calls run in "
                    f"parallel, so 0 is not a valid value and does not "
                    f"disable the checker; use max_checks_per_field=0 for "
                    f"that.")
        else:
            # The environment fallback. Its RANGE is guarded the same way one
            # step later, by the CLI; its TYPE has to be settled here, where
            # the string is read, because `int("ten")` raises a bare ValueError
            # naming neither the variable nor what it is for. The message names
            # the VARIABLE, not a pipeline.yaml key: the value came from the
            # shell, and an operator sent to the bundle would find nothing
            # there to fix.
            raw = os.environ.get("CHECKER_CONCURRENCY")
            if raw is None:
                concurrency = DEFAULT_CONCURRENCY
            else:
                try:
                    concurrency = int(raw)
                except ValueError:
                    raise ValueError(
                        f"the CHECKER_CONCURRENCY environment variable must "
                        f"be a positive integer, got {raw!r}. It is how many "
                        f"checker calls run in parallel. Unset it to take the "
                        f"default of {DEFAULT_CONCURRENCY}, or set "
                        f"checker_concurrency in pipeline.yaml, which wins "
                        f"over it.") from None
        return cls(
            checker_model=model,
            concurrency=concurrency,
        )

    def system_prompt_text(self):
        return Path(self.system_prompt_path).read_text(encoding="utf-8")

    @property
    def partials_dir(self):
        """The bundle's `prompts/partials/`, where its overrides of the
        checker's engine sections live. Derived from the system prompt's path
        rather than stored beside it, so the two cannot name different
        bundles."""
        return Path(self.system_prompt_path).parent / "partials"

    def user_prompt_template_text(self, *, predicates):
        """Return the scaffold one per-field checker message is rendered from:
        the engine section `checker_user`, or this bundle's override of it.
        The same call the render path
        (`checker_prompts.build_checker_user_message`) makes, so the text
        hashed into `checker_fp` is the text a check is sent.

        `predicates` is the run's structure map, from
        `Instrument.predicates()`. Required, and taken from the caller rather
        than reconstructed here, so every prompt in the run resolves against
        one pipeline.
        """
        from meltiro.checker_prompts import render_checker_user_template
        return render_checker_user_template(
            self.partials_dir, predicates=predicates)

    def fingerprint(self, template, reference_lists=None, *, predicates,
                    max_checks_per_field):
        """Fingerprint this checker config.

        Hashes the SUBSTITUTED checker prompt, with `{reference:NAME}`
        placeholders expanded rather than left as tokens, so editing a
        reference list the checker is shown moves the fingerprint (mirrors the
        extractor path in prompt_builder.compute_prompt_config_hash).
        `reference_lists` comes from the config bundle; the orchestrator
        passes it in.

        `predicates` is the run's structure predicate map
        (`Instrument.predicates()`), and it is required for the same reason
        the render paths require it: both prompts hashed here are rendered
        with it, so the pipeline the checker is briefed on and the pipeline
        this fingerprint claims are one value rather than two kept in step.
        `max_checks_per_field` arrives from the same place and is required on
        the same terms: it is the one value substituted into the system
        prompt, so a config that states the budget in its own words hashes
        what the checker is actually sent. The toggles behind either are not
        hashed as values of their own — `structure_hash` already carries
        them, and folding them in twice would double-count a single toggle.
        """
        from direktoro import model_info
        # Lazy import to avoid a module-level import cycle
        # (checker_prompts does not import checker, so this is only a
        # precaution against future coupling).
        from meltiro.checker_prompts import (
            build_checker_config_text, checker_user_config_text)
        # Both components carry the CONFIG's half of their prompt: the
        # bundle's own text and its overrides of the checker's engine
        # sections. Rewording an un-overridden section moves engine_fp and
        # leaves every bundle's checker_fp where it was; an override is the
        # bundle's own text and hashes as such.
        system_text = build_checker_config_text(
            system_prompt_path=self.system_prompt_path,
            reference_lists=reference_lists,
            predicates=predicates,
            max_checks_per_field=max_checks_per_field,
        )
        info = model_info(self.checker_model)
        return checker_config_fingerprint(
            self.call_identity(),
            system_text,
            checker_user_config_text(self.partials_dir,
                                     predicates=predicates),
            # The schema the verdict must fit. It is engine-owned and fixed
            # for a release, so this component moves only when the shape of a
            # verdict itself changes — which is exactly when two runs stop
            # being comparable.
            tool_set_hash=tool_set_hash(checker_tool_definitions()),
            # The checker never checks itself, so its structure component
            # folds in only the image-capability toggle.
            structure_hash=structure_hash(
                0, supports_images=info.supports_images,
            ),
            field_catalogue_hash_str=field_catalogue_hash(template),
            reference_hash=reference_lists_hash(reference_lists),
            # The checker context fields drive the checker's per-record context
            # label (build_record_context -> `{identity_context}`), so an edit
            # or reordering must move checker_fp.
            checker_context_fields=template.get("checker_context_fields"),
            # How much paper text surrounds each quote in the per-field
            # message. Widening or narrowing it changes the question the
            # checker is asked, so it moves checker_fp and only checker_fp.
            checker_context_chars=self.context_chars,
        )

    def call_identity(self):
        """The checker model's provider-call identity block.

        Model + provider + base_url + Route + wire-keyed resolved decoding
        params, built by direktoro, which owns that block. It folds into
        `checker_fp` so the same checker model run on two providers gets two
        fingerprints, a resume that switches the checker's
        provider/endpoint/route is refused, and a sampling control the checker
        model refuses moves nothing.

        Exposed as its own method because the orchestrator also hashes it on
        its own into `checker_call_fp`, the call axis for this role. Both come
        from here, so the axis can never describe a different call from the one
        the stage fingerprint covers.

        Imported lazily so `import meltiro` never pulls direktoro.routing at
        module scope.
        """
        from direktoro import model_info, resolved_decoding_params
        from direktoro import call_identity_fields, canonical_json
        info = model_info(self.checker_model)
        checker_dec = resolved_decoding_params(
            self.checker_model, sampling=self.sampling,
            max_tokens=self.max_tokens, thinking=self.thinking)
        return canonical_json(call_identity_fields(
            self.checker_model, route=info.route,
            decoding_params=checker_dec))


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def _compute_cost(rates, input_tokens, output_tokens, cache_write_5m,
                  cache_write_1h, cache_read_tokens):
    """One field's checks, summed and priced under the CHECKER's rate card.

    The card prices the checker's model and no other, and it is recorded
    alongside every figure derived from it, so a reader can redo this arithmetic
    from the tokens in the same record however far the provider's prices have
    since moved.

    Cache writes arrive already split by TTL (`rates.cache_write_split`),
    because the two tiers bill at different multiples of the base input rate
    and a single figure can only be right for one of them.
    """
    return round(rates.cost_of_call(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_5m,
        cache_write_1h_tokens=cache_write_1h,
    ), 6)


def _spend(config, responses):
    """What one field's check cost, summed over every ask it took.

    Returns the four token counters, `responses` (how many billed calls), and
    `cost_usd`. Summed rather than read off the answering call because each
    ask was billed: a field re-asked once and then answered cost two calls,
    and reporting only the second would understate the run.

    Three costing paths: a ROUTED checker model is priced FROM the responses
    (the charge the gateway reported, as `reported_cost`); a DIRECT model is
    priced against the checker's rate card, which is recorded with the run so
    the figure stays checkable; a DIRECT model with no rate card states no cost
    at all, leaving the token counters as the record. None is deliberate there
    and is never a 0.0, which would read as a free call.

    Costing NEVER raises, on any path. A routed response whose charge cannot be
    read is recorded rather than refused: the returned mapping carries
    `cost_incomplete` and `unreceipted_responses`, `cost_usd` covers the
    receipts there were, and the token counters stay whole. One field's check
    runs in a fan-out beside paid siblings and its verdict is what the run is
    buying, so an unreadable price is a gap in the accounting to be stated —
    never grounds for discarding a verdict that was asked for, answered, and
    billed.

    What produces that gap is the POST-BILLING REFUSAL, and only that: a routed
    response is charged by the gateway and then declined by direktoro's routing
    layer — a pin that did not hold, an absent generation id, an absent
    `usage.cost` — and arrives here on the exception (see `_ask_for_verdict`).
    A routed response that direktoro RETURNS always carries its charge, because
    a chargeless one is exactly what it refuses, so no successfully answered
    check can reach the unreceipted branch below.

    Called on the way to a verdict AND on the way out of a failure that had
    already been billed, so a degraded field is priced the same way a
    successful one is.
    """
    from direktoro import model_info

    if not responses:
        # Nothing was billed, so nothing is priced and the registry is not
        # consulted: a failure before the first call landed — a refused
        # connection, a model this build cannot resolve — must not be turned
        # into a second failure by the accounting that reports it.
        return CheckerError.no_spend()

    def total(attr):
        return sum(getattr(r.usage, attr, 0) or 0 for r in responses)

    input_tokens = total("input_tokens")
    output_tokens = total("output_tokens")
    cache_create = total("cache_creation_input_tokens")
    cache_read = total("cache_read_input_tokens")
    # The cache-write total is what the check REPORTS; the per-TTL split is
    # what it is PRICED from, because the two tiers bill at different multiples
    # of the base input rate. Summed per tier across every ask, on the same
    # terms as the counters above.
    splits = [cache_write_split(r.usage) for r in responses]
    cache_write_5m = sum(five for five, _ in splits)
    cache_write_1h = sum(hour for _, hour in splits)

    unreceipted = 0
    if model_info(config.checker_model).route is not None:
        charged = 0.0
        for r in responses:
            reported = getattr(r, "reported_cost", None)
            if reported is None:
                unreceipted += 1
                continue
            charged += reported
        cost_usd = round(charged, 6)
    elif config.rates is not None:
        cost_usd = _compute_cost(
            config.rates, input_tokens, output_tokens,
            cache_write_5m, cache_write_1h, cache_read,
        )
    else:
        cost_usd = None
    spent = {
        "responses": len(responses),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_create,
        "cache_read_tokens": cache_read,
        "cost_usd": cost_usd,
    }
    if unreceipted:
        # Present only when a receipt was missing, so an ordinary spend record
        # carries no flag saying nothing was wrong. The count is what makes the
        # figure readable: a dollar total beside "2 calls it does not cover"
        # states its own coverage.
        spent["cost_incomplete"] = True
        spent["unreceipted_responses"] = unreceipted
    return spent


def reported_cost_or_raise(model, response):
    """The response-reported USD cost for a ROUTED (gateway-served) call.

    Routed models take their cost FROM the response
    (`NormalisedResponse.reported_cost`). That figure is a charge the gateway
    states, not one anybody predicts, so it needs no rate card and is recorded
    whether or not the run configures one. A routed response that carries no
    reported cost is a plumbing fault (`usage.include` did not reach the
    gateway, or the pin returned an unpriced receipt), and pricing it at zero
    would silently undercount the run. So this raises loudly rather than
    ledgering a $0 routed call. Callers branch on the registry:
    `model_info(model).route is not None` -> this; else -> the calling role's
    rate card.

    The raise belongs to the roles whose call stands alone: the orchestrator's
    extractor and review turns, where the conversation is sequential and an
    exception ends a run that has nothing else in flight. The checker prices
    through `_spend` instead, which states an unreadable charge as coverage on
    the figure, because one field's missing receipt there would discard a
    batch of paid sibling verdicts to report a price for the field it came
    from.
    """
    reported = getattr(response, "reported_cost", None)
    if reported is None:
        raise RuntimeError(
            f"routed model {model!r} returned no reported cost "
            f"(NormalisedResponse.reported_cost is None; did usage.include "
            f"reach the gateway, and did the pin resolve?). Refusing to ledger "
            f"it as $0: a routed call's cost comes from the response."
        )
    return reported


def _cost_coverage(spent):
    """The coverage keys a spend record carries when a charge was unreadable.

    Empty for the ordinary case, so a verdict says how far its cost figure
    reaches only when that is less than the whole of it, and a run whose every
    call came back with a receipt serialises with no flag on it.
    """
    if not spent.get("cost_incomplete"):
        return {}
    return {
        "cost_incomplete": True,
        "unreceipted_responses": spent.get("unreceipted_responses"),
    }


def _degraded_verdict(message, spent):
    """The error-origin challenge one failed field degrades to.

    A challenge in shape and an absence of checking in fact: `error_origin`
    says which, and every disposition downstream reads that flag rather than
    the verdict word. `spent` is what the failure had already been billed
    (`CheckerError.spent`, or `CheckerError.no_spend()` for a fault that
    reached no provider), so a degraded field is priced from the same record a
    successful one is. Every counter is read with a zero default: a
    `CheckerError` raised by a library caller may carry a partial mapping, and
    this is the last line before the fan-out returns — a KeyError here would
    abort the batch it exists to protect.
    """
    responses = spent.get("responses", 0)
    cost_usd = spent.get("cost_usd")
    # 0.0 only when no call was made: zero tokens cost zero under any rate
    # card, so that is a real figure over a real (empty) usage and neither
    # needs rates nor withholds a total from a run that has them. A failure
    # that DID make calls keeps None when nothing could price them, exactly as
    # a successful verdict does, so an unpriced direct model withholds the
    # run's total instead of having this field assert the calls were free.
    if cost_usd is None and not responses:
        cost_usd = 0.0
    return {
        "error": message,
        "verdict": "challenge",
        "rationale": f"(checker error: {message})",
        "notes": None,
        # Tagged here, at the source, so a downstream disposition never treats
        # a failed check as a genuine objection: an error-origin challenge must
        # never contribute to a trusted status.
        "error_origin": True,
        # A degraded field still reports the asks it took to get there, so the
        # calibration signal reads the same way on this path as on a
        # successful one.
        "reprompted": max(0, responses - 1),
        # The spend the failure had already incurred: zero when nothing
        # reached the provider or every attempt errored, and a real figure when
        # the calls succeeded and their answers could not be used — a truncated
        # reply, a verdict outside the vocabulary, or replies that called no
        # tool. Those were billed, and a run that ledgered them at zero would
        # understate itself.
        "input_tokens": spent.get("input_tokens", 0),
        "output_tokens": spent.get("output_tokens", 0),
        "cache_creation_tokens": spent.get("cache_creation_tokens", 0),
        "cache_read_tokens": spent.get("cache_read_tokens", 0),
        "cost_usd": cost_usd,
        # ... and, when a routed charge could not be read, how many of the
        # calls behind that figure it does not cover.
        **_cost_coverage(spent),
    }


def _build_checker_adapter(config, client=None):
    """The adapter the checker's calls go through, or None when its key is unset.

    `direktoro.build_adapter` resolves the endpoint, the key variable and the
    adapter that speaks that wire from the checker model's id alone, so the
    per-field fan-out re-points to another checker model with no
    checker-specific config. `client` is an already-constructed provider SDK
    client to wrap: the parallel fan-out builds one and shares it across every
    call, and tests inject a stub. A None return means the key variable is
    unset, which the caller wraps as a per-field CheckerError.
    """
    from direktoro import MissingAPIKey, build_adapter
    try:
        return build_adapter(config.checker_model, client=client)
    except MissingAPIKey:
        return None


def _require_cap(config):
    """Refuse a check that states no output cap, before the call is made.

    `CheckerConfig.max_tokens` has no default value, on the same terms as the
    model: the number bounds what one check may spend and what it may answer
    within, so a run states it. Guarded on both entry points rather than only
    where the config is built, so a library caller that constructs its own
    `CheckerConfig` is refused here instead of at the endpoint, after the
    extractor turn that produced the field has already been billed.
    """
    if config.max_tokens is None:
        raise CheckerError(
            "the checker has no output cap: CheckerConfig.max_tokens is None "
            "and there is no default for it, because the number bounds what "
            "one check may spend and what it may answer within. Set "
            "checker_max_tokens in pipeline.yaml, or set it on the config "
            "directly; a run with no checker sets max_checks_per_field: 0.")


# ---------------------------------------------------------------------------
# Single-call: check one field
# ---------------------------------------------------------------------------

def check_one_field(*, system_message_blocks, user_message_blocks, config,
                    client=None, adapter=None, api_logger=None,
                    api_log_meta=None, on_retry=None):
    """Call the checker LLM for one field via its provider adapter.

    `system_message_blocks` is the cached system prompt as a list of
    content blocks (already including cache_control). `user_message_blocks`
    is the per-field content blocks. `adapter` is a provider adapter; when
    omitted it is built from `config` (and an optional injected `client`).
    `on_retry(attempt, delay_seconds, error)` is handed to direktoro's retry
    loop and fires once per retried transient failure; a failed attempt raises
    before the wire log runs, so it is the only trace one leaves.

    Returns a dict with keys: verdict, rationale, notes, reprompted,
    input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
    cost_usd. `cost_usd` is None when the call cannot be costed — a direct
    model and no rate card on `config` — so the token counters stand alone
    rather than beside a zero that would read as a free call. Every counter
    covers EVERY call this check made, so a re-asked field reports what it
    actually cost.

    Raises CheckerError on any failure, and ONLY CheckerError: this frame
    owns the accumulator the asks land in, so whatever the ask below raises
    leaves here priced from the calls that were actually billed. That is what
    keeps `run_checker_batch`'s backstop honest — the faults it still catches
    are the ones raised outside this function, where no call was made and
    there is genuinely nothing to price.
    """
    responses = []
    try:
        return _ask_for_verdict(
            responses,
            system_message_blocks=system_message_blocks,
            user_message_blocks=user_message_blocks,
            config=config, client=client, adapter=adapter,
            api_logger=api_logger, api_log_meta=api_log_meta,
            on_retry=on_retry,
        )
    except CheckerError:
        # Already priced where it was raised, from these same responses.
        raise
    except Exception as e:
        # A fault this module has no structured error for — and the asks
        # before it were billed all the same.
        raise CheckerError(f"{type(e).__name__}: {e}",
                           spent=_spend(config, responses)) from e


def _ask_for_verdict(responses, *, system_message_blocks, user_message_blocks,
                     config, client=None, adapter=None, api_logger=None,
                     api_log_meta=None, on_retry=None):
    """Ask until a verdict arrives, appending every billed response to
    `responses`, and return the verdict record `check_one_field` documents.

    `responses` belongs to the caller so that a failure anywhere in here can
    still be priced from the calls that had completed when it happened: a
    check re-asked once and then met with a fault was billed for the first
    ask, and a list built and lost inside this frame would ledger it at zero.
    """
    from direktoro import (RETRY_BACKOFF_SECONDS, ProviderError,
                           ProviderRateLimitError, ProviderRetryableError,
                           create_message_with_retry, extract_tool_call,
                           model_info, tool_choice_named)
    _require_cap(config)
    if adapter is None:
        adapter = _build_checker_adapter(config, client)
    if adapter is None:
        # Unreachable in the orchestrated pipeline: Orchestrator._preflight_keys
        # verifies the checker key before any spend, so key absence is caught
        # up front rather than here. Kept as a defensive guard for a direct
        # caller running under an environment that names no key for this model.
        env = model_info(config.checker_model).api_key_env
        raise CheckerError(f"{env} not set; cannot call checker")

    # Retries go through direktoro's `create_message_with_retry`, the same
    # function the extractor and reviewer call, so all three roles wait on one
    # schedule (`RETRY_BACKOFF_SECONDS`) and one rule about which failures are
    # worth retrying — a timeout is deliberately not, because a timed-out
    # request may already have been served and billed. A schedule kept here as
    # well would drift from that one silently, and a checker retrying on its
    # own terms would be invisible: a failed attempt raises before the wire log
    # runs, so `on_retry` is the only trace it leaves.
    retries = len(RETRY_BACKOFF_SECONDS)

    # The verdict tool, and the strongest tool_choice this model's endpoint
    # honours. `tool_choice_named` returns the wire's "auto" form for an
    # endpoint that refuses a forced choice, so a non-forcing checker model
    # needs no branch here; it is why the re-prompt below exists at all.
    #
    # Tools render ahead of the system prompt in the cached prefix, so adding
    # them lengthens what the first call of a run writes to cache and every
    # call after it reads. The schema is fixed for the run, so that is a
    # one-time cost, not a per-call one.
    tools = checker_tool_definitions()
    tool_choice = tool_choice_named(
        config.checker_model, CHECKER_VERDICT_TOOL_NAME)

    tool_input = None
    last_error = None
    # The first ask puts the field on its own. A second ask CORRECTS the reply
    # that recorded no verdict rather than restating the question
    # (`_reask_messages`): a reply that can be replayed is, with the correction
    # as a new user turn after it, and a reply that cannot is corrected in the
    # field's own message. Either way the ask carries the one field and the one
    # value it carried the first time — the correction is the only thing in it
    # that changed, so it is the only thing that can move the answer. The first
    # message is byte-identical across the two asks, the blocks the caller
    # rendered passed through rather than rebuilt, so the field a second
    # verdict is about is the field the first ask asked about, to the byte.
    original_message = {"role": "user", "content": list(user_message_blocks)}
    messages = [original_message]
    for ask in range(MAX_TOOL_FREE_REPROMPTS + 1):
        if ask:
            # `responses[-1]` is the reply that came back without a verdict:
            # this iteration exists only because it did.
            messages = _reask_messages(original_message, responses[-1])

        try:
            response = create_message_with_retry(
                adapter,
                on_retry=on_retry,
                model=config.checker_model,
                max_tokens=config.max_tokens,
                system=system_message_blocks,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                sampling=config.sampling,
                thinking=config.thinking,
            )
        # Each raise carries the spend of the asks that ALREADY completed
        # (`responses` holds them; the ask that just failed is not among
        # them and was not billed). A field re-asked once and then met
        # with a provider error was billed for the first ask, and a
        # degraded field that ledgered it at zero would understate the run
        # by however many checks went this way.
        except ProviderRateLimitError as e:
            raise CheckerError(
                f"Rate limit after {retries} retries: {e}",
                spent=_spend(config, responses))
        except ProviderRetryableError as e:
            raise CheckerError(
                f"Provider error after {retries} retries: {e}",
                spent=_spend(config, responses))
        except ProviderError as e:
            # A refusal that arrived INSTEAD of a response carries none, and
            # there is nothing to bill. A refusal about a response carries it
            # (`ProviderError.response`): the call was served and charged, and
            # direktoro's routing layer declines the RESULT — a pin that did
            # not hold, a missing generation id, a missing gateway charge. That
            # material is banked here, exactly like a successful ask's, before
            # the field degrades. Ledgering it at $0 would hide a paid call,
            # and `_spend` already knows how to state a response whose charge
            # cannot be read: a routed response with no `reported_cost` becomes
            # coverage on the figure (`cost_incomplete`), which is the shape a
            # refused routed call takes.
            billed = getattr(e, "response", None)
            if billed is not None:
                responses.append(billed)
                _log_ask(api_logger, api_log_meta, billed, ask)
            raise CheckerError(f"Provider API error: {e}",
                               spent=_spend(config, responses))

        responses.append(response)

        # Verbatim API audit log, one entry per call — so a re-asked field
        # leaves both asks in the audit trail rather than only the one that
        # answered.
        _log_ask(api_logger, api_log_meta, response, ask)

        # A refusal is the model's answer to this call, and it outranks
        # whatever else the reply carried (direktoro's canonical vocabulary
        # puts `refusal` ahead of `tool_use`). Read before the verdict rather
        # than after it, which is the opposite of the truncation check below
        # and for the opposite reason: a cap cuts a reply off AFTER the verdict
        # it was billed for, so a complete verdict stands; a filter blocks the
        # reply instead of one. Not re-asked either — the same field, the same
        # value and the same evidence would be blocked again.
        if getattr(response, "stop_reason", None) == "refusal":
            raise CheckerError(
                "the checker model refused this field (stop_reason "
                "'refusal': a host content filter stopped the reply, or the "
                "model declined it). No verdict was given, so the field is "
                "recorded as unchecked rather than as objected to.",
                spent=_spend(config, responses))

        # Read the verdict off the response by BLOCK TYPE, so any leading
        # block a model or endpoint emits ahead of the tool call — reasoning,
        # a sentence of preamble — is skipped rather than mistaken for the
        # answer.
        tool_input, last_error = extract_tool_call(
            response, CHECKER_VERDICT_TOOL_NAME)
        if not isinstance(tool_input, dict):
            # A verdict is an object of named arguments. Anything else is a
            # reply with no verdict in it, handled as any other tool-free
            # reply — and never as an AttributeError on the `.get` below,
            # which would leave this field's plumbing fault to be caught as a
            # surprise rather than reported as an absence of checking. A tool
            # call whose input is absent entirely reads back as `(None, None)`,
            # so the reason is stated here rather than left as a bare "None"
            # in the message the failure carries.
            if tool_input is not None:
                last_error = (
                    f"{CHECKER_VERDICT_TOOL_NAME} was called with "
                    f"{type(tool_input).__name__}, not an object of arguments")
            elif last_error is None:
                last_error = (
                    f"{CHECKER_VERDICT_TOOL_NAME} was called with no "
                    f"arguments at all")
            tool_input = None

        # Truncation, named. Unnamed, a turn that ended before the tool call
        # was emitted arrives here as an ordinary tool-free reply, which
        # points at the model's willingness rather than at the cap. The
        # message names the cap and the key that set it, which is the line an
        # operator would edit. Not re-asked: a cap that truncated once will
        # truncate again.
        #
        # Read AFTER the verdict, so a reply that recorded a complete verdict
        # and then ran into the cap keeps the answer it was billed for: the
        # tool call is the whole of what this role is asked for, and whatever
        # the cap cut off came after it. Only a cut-off reply with no usable
        # verdict degrades.
        if (getattr(response, "stop_reason", None) == "max_tokens"
                and (tool_input is None
                     or tool_input.get("verdict") not in VALID_VERDICTS)):
            raise CheckerError(
                truncation_report(config.max_tokens, "checker_max_tokens"),
                spent=_spend(config, responses))

        if tool_input is not None:
            break

    if tool_input is None:
        # Every ask came back without the verdict tool. A model under a forced
        # tool_choice should never reach here; one under "auto" has now
        # declined twice. Either way no verdict was given, which
        # run_checker_batch records as an error-origin challenge — an absence
        # of checking, never an objection to the value.
        raise CheckerError(
            f"Checker gave no {CHECKER_VERDICT_TOOL_NAME} call in "
            f"{len(responses)} ask(s): {last_error}",
            spent=_spend(config, responses))

    verdict = tool_input.get("verdict")
    rationale = tool_input.get("rationale", "")
    notes = tool_input.get("notes")  # optional free-text field
    # The schema advertises the vocabulary; nothing enforces it on the wire,
    # so a verdict outside it is refused here rather than believed. NOT
    # re-asked: a tool-free reply is an answer that never arrived, while this
    # is an answer that arrived invalid, and re-asking it would be
    # second-guessing a judgement the checker did make.
    if verdict not in VALID_VERDICTS:
        raise CheckerError(
            f"Invalid verdict {verdict!r}; expected one of "
            f"{sorted(VALID_VERDICTS)}",
            # The calls that produced this unusable answer were billed, like
            # the ones behind the two failures above. A raise with no spend on
            # it would ledger them at $0.
            spent=_spend(config, responses),
        )

    # `response` is the ask that answered, and is what the provenance
    # below describes; the counters cover every ask. A missing charge on
    # any of them, this one included, is stated as coverage on the figure
    # rather than raised: the verdict was asked for, answered and billed,
    # and it is what the batch is buying — an unreadable price is a gap in
    # the accounting, not grounds for discarding the answer.
    spent = _spend(config, responses)
    input_tokens = spent["input_tokens"]
    output_tokens = spent["output_tokens"]
    cache_create = spent["cache_creation_tokens"]
    cache_read = spent["cache_read_tokens"]
    cost_usd = spent["cost_usd"]
    return {
        "verdict": verdict,
        "rationale": rationale,
        "notes": notes,
        # A genuine model verdict. The error path in run_checker_batch tags
        # its synthetic challenge with error_origin=True so a disposition can
        # tell a real challenge from a failed check; a genuine verdict is
        # never error-origin.
        "error_origin": False,
        # How many times this field had to be re-asked before the verdict
        # arrived. Nearly always 0. It is calibration signal about the CHECKER
        # MODEL rather than about the field: a model that needs nudging is
        # marginal for the role, and a run's diagnostics report it separately
        # from the failures so the two are never read as one number.
        "reprompted": len(responses) - 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_create,
        "cache_read_tokens": cache_read,
        "cost_usd": cost_usd,
        # Present only when an earlier ask's charge could not be read, so the
        # figure above says how far it reaches.
        **_cost_coverage(spent),
        # Model-invisible provenance for the orchestrator to fold into
        # run.json once per run (the checker's resolved model + the raw
        # decoding params actually sent), plus the routing receipts a routed
        # checker call carries: the gateway generation id (an external audit
        # receipt) and the served upstream. Stripped before the per-field
        # audit file is written; never reaches the verdict event.
        "_provenance": {
            "provider": response.provider,
            "base_url": response.base_url,
            "resolved_model": response.resolved_model,
            "decoding_params": response.decoding_params,
            "generation_id": getattr(response, "generation_id", None),
            "served_provider": getattr(response, "served_provider", None),
        },
    }


def _log_ask(api_logger, api_log_meta, response, ask):
    """Write one ask's verbatim wire entry, if the caller wants one.

    Called for every BILLED response, which is every one the provider served —
    including one direktoro's routing layer then refused, because the wire
    record is of what happened on the wire and a refused response happened
    there. `ask` is the 0-based ask index, so a re-asked field's two calls are
    told apart in the log.

    Audit-side errors are swallowed: the log is a record of the call, and a
    fault writing it must not become the call's outcome.
    """
    if api_logger is None:
        return
    try:
        meta = dict(api_log_meta or {})
        meta.update(provider=response.provider,
                    base_url=response.base_url,
                    wire_model=response.resolved_model,
                    wire_request=response.wire_request,
                    ask=ask)
        api_logger(response.raw_request, response.raw_response, **meta)
    except Exception:
        pass


def _reask_messages(original_message, reply):
    """The messages a re-ask sends: a correction, after the reply it corrects.

    A reply made of TEXT BLOCKS AND NOTHING ELSE is replayed whole, in three
    messages: `original_message`, which is the first ask's user message passed
    through as the object that was sent — the field, its value and its evidence
    exactly where they already were; then that text as an assistant turn; then
    `CHECKER_TOOL_REPROMPT` alone, as a new user turn. So the model is
    corrected after a reply it can see it gave, in its own words, with nothing
    of it left out. The text is replayed in the plain-dict shape meltiro
    replays an assistant turn in everywhere else (the orchestrator's
    conversion, which is what `Session.replay_messages` reads back), because
    the log this ask writes is a JSON record and a provider SDK's block object
    is not. A text block carrying no text is dropped on the way: it says
    nothing, and an empty text block is content a provider refuses.

    Any other reply is replayed not at all, and that ask carries the field once
    more with the correction on the end of it: one user turn, valid on any
    wire. Replaying such a reply could only mean replaying the part of it that
    fits — a tool_use block cannot be replayed here, being a call the next
    message would leave with no result after it, and a reasoning block cannot
    be replayed without the signature it came with — and a reply quoted back
    with pieces missing is a reply the model did not give. So a reply holding a
    tool_use or a reasoning block, and a reply that said nothing at all, are
    corrected without being shown.
    """
    correction = {"type": "text", "text": CHECKER_TOOL_REPROMPT}
    blocks = getattr(reply, "content", None) or []
    said = [{"type": "text", "text": block.text} for block in blocks
            if getattr(block, "type", None) == "text"
            and getattr(block, "text", "")]
    text_only = all(getattr(b, "type", None) == "text" for b in blocks)
    if not said or not text_only:
        return [{"role": "user",
                 "content": [*original_message["content"], correction]}]
    return [
        original_message,
        {"role": "assistant", "content": said},
        {"role": "user", "content": [correction]},
    ]


# ---------------------------------------------------------------------------
# Parallel fan-out
# ---------------------------------------------------------------------------

def run_checker_batch(*, calls, config, client=None, adapter=None,
                      on_complete=None, api_logger=None, on_retry=None):
    """Run `check_one_field` for each call in `calls` in parallel.

    One batch is the set of fields one tool call wrote: the orchestrator fans
    out over them after the dispatcher has applied them, and merges the
    verdicts into that call's tool result.

    `calls` is a list of dicts, each with keys:
      - `field_path`: identifier for the field (used in the result map)
      - `system_message_blocks`: cached system prompt (same object reused
        across all calls)
      - `user_message_blocks`: per-field content blocks

    Returns a dict `{field_path: result_dict_or_error_dict}`. A field whose
    check fails is wrapped (`{"error": ..., "rationale": "(checker error)"}`)
    rather than raised, for EVERY exception its call raises and not only the
    structured ones: one field's fault degrades that field, and the paid
    verdicts of its siblings in the same batch are returned beside it. The
    returned dict is ordered by field path, not by completion order: results
    land in nondeterministic `as_completed` order, and that order flows into
    the tool result the model reads and into the session event, so the results
    are sorted before returning to keep runs reproducible.

    `on_retry(attempt, delay_seconds, error)` is passed to every call in the
    batch and fires on the calling thread's behalf from inside the worker
    thread that retried, once per retried transient failure.

    `on_complete(field_path, result)` is an optional callback invoked
    on the calling thread, inside the `as_completed` loop, as each call
    returns; used by the orchestrator to accumulate spend as results land. It
    still fires in completion order (the accumulation is commutative, so order
    is irrelevant there); only the returned mapping is sorted. An exception it
    raises propagates rather than being caught: it is the run's ledger, and a
    batch that banked verdicts past a failed ledger write would count them
    against a total that no longer covers what they cost. What that does NOT
    do is stop the batch: every call was submitted before the loop began, so
    the pool runs them all out on the way through the `with` block. They are
    billed, and their verdicts are lost with the mapping this never returns —
    which is the price of a ledger that cannot be written, and cheaper than
    one that no longer says what the run spent.
    """
    # Refused for the batch rather than per field: with no cap every call in
    # the fan-out fails the same way, and degrading each field to a challenge
    # would turn one configuration fault into a whole run of false challenges.
    _require_cap(config)
    # Build one adapter (and its underlying client) and share it across the
    # whole fan-out, so a single provider connection serves every field.
    if adapter is None:
        adapter = _build_checker_adapter(config, client)

    results = {}

    def _one(call):
        try:
            api_log_meta = {
                "field_path": call.get("field_path"),
                "check_index": call.get("check_index"),
            }
            res = check_one_field(
                system_message_blocks=call["system_message_blocks"],
                user_message_blocks=call["user_message_blocks"],
                config=config, adapter=adapter,
                api_logger=api_logger,
                api_log_meta=api_log_meta,
                on_retry=on_retry,
            )
        except CheckerError as e:
            # The normal failure path. Genuine transient API errors (rate
            # limit / 5xx exhausted, a malformed response) degrade this one
            # field to a challenge rather than aborting the batch. Key absence
            # is NOT expected here: the orchestrator preflight rejects a
            # missing checker key before any spend, so this catch never masks a
            # misconfiguration.
            res = _degraded_verdict(str(e), e.spent)
        except Exception as e:
            # The backstop, and the last line: anything raised out here that
            # is not a CheckerError degrades this field too, rather than
            # propagating out of `fut.result()` and discarding the verdicts
            # its siblings were already billed for. It is priced at nothing
            # because the faults that reach it are the ones raised AROUND the
            # check — a malformed call dict, a fault building the log meta —
            # which made no provider call: `check_one_field` prices anything
            # that goes wrong once it has one, and re-raises it as the
            # CheckerError above carrying that spend. The exception is named
            # in the message rather than swallowed, so a fault that belongs in
            # the vocabulary above is visible in the run rather than
            # indistinguishable from one that does not.
            res = _degraded_verdict(
                f"{type(e).__name__}: {e}", CheckerError.no_spend())
        return call["field_path"], res

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.concurrency,
    ) as pool:
        futures = [pool.submit(_one, c) for c in calls]
        for fut in concurrent.futures.as_completed(futures):
            field_path, res = fut.result()
            results[field_path] = res
            if on_complete is not None:
                # Raised through, not caught. The callback is the run's spend
                # ledger and its meta checkpoint, so a failure inside it means
                # calls that were billed are not being recorded — the batch
                # continuing would bank verdicts against a total that no longer
                # counts them. It runs on the calling thread, so its exception
                # reaches the orchestrator exactly as any other write to the
                # run's own state would.
                on_complete(field_path, res)

    # Deterministic order by field path (see docstring).
    return {fp: results[fp] for fp in sorted(results)}
