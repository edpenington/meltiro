"""Narrow per-field checker LLM client + ThreadPoolExecutor fan-out.

One call per field a tool call just wrote, in parallel up to a configurable
concurrency. Each call gets a cached system prompt (the config bundle's checker
prompt: role, verdict shape, and the rendered reference lists, with no field
catalogue in it) plus a small per-field user message (field identity, the
field's definition and allowed values, identity context, evidence, value, and
the field's note). Returns one of `{ok, challenge}` plus a one-sentence
rationale.

The checker is a probabilistic extension of the deterministic validator, not a
pipeline stage: the orchestrator fans out over the fields one tool call
applied, and the challenges come back in that call's tool result alongside the
validation errors. Each call is single-turn and sees only the current value,
its evidence read in the surrounding paper text (see `DEFAULT_CONTEXT_CHARS`
below and `checker_prompts`), and the field note, so a re-check after a
revision judges a genuinely fresh context.

Each call goes through direktoro's adapter as one blocking `create_message`
and comes back complete; how that reaches the wire is the adapter's business,
and nothing here depends on it. This module retries with backoff on 429/5xx,
accounts prompt-cache cost, and reports every failure as a structured
`CheckerError`, which is what lets one bad field degrade rather than abort the
extraction.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meltiro.errors import CheckerError
from meltiro.fingerprint import (
    checker_config_fingerprint,
    structure_hash,
    field_catalogue_hash,
    reference_lists_hash,
)
from meltiro.thinking import truncation_message

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


DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CONCURRENCY = 10
# Characters of surrounding paper text shown on EACH side of a matched quote
# in the per-field checker message (pipeline.yaml's `checker_context_chars`).
# Enough to carry a table's header row or the sentence a pronoun points back
# to, small enough that the checker is still reading a neighbourhood rather
# than the paper. 0 turns context off.
DEFAULT_CONTEXT_CHARS = 1000

VALID_VERDICTS = {"ok", "challenge"}


@dataclass
class CheckerConfig:
    """How the checker's provider CALL is made, and nothing about the pipeline.

    Model, key, decoding knobs, the width of the quote context, and the paths
    of the two prompt files this role sends. What it deliberately does not
    hold is the run's pipeline STRUCTURE (the per-field check budget, whether
    a reviewer runs): that is the instrument's
    (`meltiro.instrument.Instrument`). Both `user_prompt_template_text` and
    `fingerprint` below REQUIRE the predicate map as an argument, and the
    only thing that produces one is `Instrument.predicates()`, so the
    checker's rendered prompts and its `checker_fp` cannot describe a
    different pipeline from the one the extractor and reviewer were briefed
    on.
    """

    # No hardcoded fallback model: there is deliberately no default checker
    # model. A real run must supply one (the CLI requires checker_model from
    # pipeline.yaml or the --checker-model flag, exactly like the extractor
    # and review models). The None default lets tests and programmatic use
    # construct a config and set the model explicitly.
    checker_model: str = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    concurrency: int = DEFAULT_CONCURRENCY
    # Characters of surrounding paper text the checker sees on each side of a
    # matched quote. Config identity, not an operational budget: it changes
    # what the checker is asked, so it folds into `fingerprint()` below. Like
    # the other decoding knobs it comes from the config bundle only, never
    # from the environment.
    context_chars: int = DEFAULT_CONTEXT_CHARS
    api_key: str = ""
    # Prompt paths come from the config bundle
    # (`ConfigBundle.checker_system_path` / `.checker_user_template_path`);
    # the orchestrator sets them at construction time. No CWD-relative
    # default; a real run must supply them.
    system_prompt_path: str = None
    user_prompt_template_path: str = None
    # The checker's own thinking / reasoning-effort spec, or None to say
    # nothing and leave the model doing whatever its default is. From the
    # config bundle only (pipeline.yaml's `checker_thinking_mode` /
    # `checker_thinking_effort`), so the same tree yields the same checker_fp
    # under any shell; it rides the same `resolved_decoding_params` call as
    # `temperature` and `max_tokens`. Declared after the fields above so
    # positional construction of them is unaffected. See `meltiro.thinking`:
    # the cap hazard it guards is sharpest on this role, whose whole output
    # is a small JSON verdict under a small cap.
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
        from direktoro import is_known_model, model_info
        # The model comes from the caller alone — the config bundle's
        # `checker_model`, or `--checker-model` — and from no environment
        # variable. A shell-supplied model would change what the checker is
        # asked while the bundle that names the run looked unchanged, and
        # `checker_model` rides in `checker_fp`: two runs of one tagged tree
        # would then record different fingerprints for no recorded reason.
        # Same rule as the decoding knobs below.
        model = model_override
        # The checker's API key comes from the provider its model resolves to
        # (ANTHROPIC_API_KEY for Claude, OPENAI_API_KEY for GPT, OPENROUTER_API_KEY
        # for gateway-routed GLM/Qwen), not a fixed Anthropic var. The CLI
        # validates the model id is known before this runs; an unknown or absent
        # model falls back to ANTHROPIC_API_KEY (the run then fails loudly
        # downstream).
        if model and is_known_model(model):
            api_key = os.environ.get(model_info(model).api_key_env, "")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        # The checker's decoding knobs (max_tokens, temperature) and its
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
            # The environment fallback's EFFECTIVE value is guarded the same
            # way one step later, by the CLI, which is where an operator-facing
            # message belongs (see cli._build_orchestrator).
            concurrency = int(os.environ.get(
                "CHECKER_CONCURRENCY", str(DEFAULT_CONCURRENCY)))
        return cls(
            api_key=api_key,
            checker_model=model,
            concurrency=concurrency,
        )

    def system_prompt_text(self):
        return Path(self.system_prompt_path).read_text(encoding="utf-8")

    def user_prompt_template_text(self, *, predicates):
        """Return the checker user template with `{include:NAME}` partials
        expanded, matching what the render path
        (checker_prompts.build_checker_user_message) actually sends. The
        fingerprint hashes this expanded text, so editing a partial cited by
        the template moves checker_fp; hashing the raw file would let a
        partial edit change what the checker sees without moving the
        fingerprint.

        `predicates` is the run's `{include_if:PREDICATE:NAME}` map, from
        `Instrument.predicates()`. Required, and taken from the caller rather
        than reconstructed here, so a `{include_if:review:...}` block in the
        checker's own template resolves against the one pipeline the whole run
        renders against.
        """
        from meltiro.prompt_partials import substitute_include_placeholders
        raw = Path(self.user_prompt_template_path).read_text(encoding="utf-8")
        return substitute_include_placeholders(
            raw, Path(self.user_prompt_template_path).parent / "partials",
            predicates=predicates)

    def fingerprint(self, template, reference_lists=None, *, predicates):
        """Fingerprint this checker config.

        Hashes the SUBSTITUTED checker system prompt, the text the checker
        LLM actually sees after `{reference:NAME}` placeholders are expanded,
        not the raw file. Editing a reference list that appears in the checker
        system prompt therefore moves the fingerprint (mirrors the extractor
        path in prompt_builder.compute_prompt_config_hash). `reference_lists`
        comes from the config bundle; the orchestrator passes it in.

        `predicates` is the run's structure predicate map
        (`Instrument.predicates()`), and it is required for the same reason
        the render paths require it: both prompts hashed here are rendered
        with it, so the pipeline the checker is briefed on and the pipeline
        this fingerprint claims are one value rather than two kept in step.
        The toggles behind it are not hashed as values of their own —
        `structure_hash` already carries them, and folding them in twice would
        double-count a single toggle.
        """
        from direktoro import model_info
        # Lazy import to avoid a module-level import cycle
        # (checker_prompts does not import checker, so this is only a
        # precaution against future coupling).
        from meltiro.checker_prompts import build_checker_system_text
        system_text = build_checker_system_text(
            template,
            system_prompt_path=self.system_prompt_path,
            reference_lists=reference_lists,
            predicates=predicates,
        )
        info = model_info(self.checker_model)
        return checker_config_fingerprint(
            self.call_identity(),
            system_text,
            self.user_prompt_template_text(predicates=predicates),
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
        provider/endpoint/route is refused, and a temperature the checker model
        rejects moves nothing.

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
            self.checker_model, temperature=self.temperature,
            max_tokens=self.max_tokens, thinking=self.thinking)
        return canonical_json(call_identity_fields(
            self.checker_model, route=info.route,
            decoding_params=checker_dec))


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def _compute_cost(rates, input_tokens, output_tokens, cache_creation_tokens,
                  cache_read_tokens):
    """One checker call's USD cost under the CHECKER's rate card.

    The card prices the checker's model and no other, and it is recorded
    alongside every figure derived from it, so a reader can redo this arithmetic
    from the tokens in the same record however far the provider's prices have
    since moved.
    """
    return round(rates.cost_of_call(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_creation_tokens,
    ), 6)


def reported_cost_or_raise(model, response):
    """The response-reported USD cost for a ROUTED (gateway-served) call.

    Routed models take their cost FROM the response (OpenRouter `usage.cost` ->
    `NormalisedResponse.reported_cost`). That figure is a charge the gateway
    states, not one anybody predicts, so it needs no rate card and is recorded
    whether or not the run configures one. A routed response that carries no
    reported cost is a plumbing fault (`usage.include` did not reach the
    gateway, or the pin returned an unpriced receipt), and pricing it at zero
    would silently undercount the run. So this raises loudly rather than
    ledgering a $0 routed call. Callers branch on the registry:
    `model_info(model).route is not None` -> this; else -> the calling role's
    rate card. Shared by the checker and the orchestrator's extractor/review
    accounting.
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


def _strip_code_fences(raw):
    s = raw.strip()
    if s.startswith("```"):
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _parse_checker_json(raw):
    """Parse the checker's JSON response, tolerating trailing text.

    Sonnet occasionally appends a sentence of commentary after the JSON
    object even though the prompt forbids it. `json.JSONDecoder.raw_decode`
    returns the first complete JSON value and stops, so it is used to
    extract just the object and ignore any trailing prose.

    Raises CheckerError if no parseable JSON object can be found at all.
    """
    s = _strip_code_fences(raw).lstrip()
    if not s:
        raise CheckerError("Checker returned an empty response")

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(s)
    except json.JSONDecodeError as e:
        # Heuristic: maybe there's prose BEFORE the JSON. Find the first
        # `{` and try from there.
        first_brace = s.find("{")
        if first_brace > 0:
            try:
                obj, end = decoder.raw_decode(s[first_brace:])
            except json.JSONDecodeError as e2:
                raise CheckerError(
                    f"Checker returned non-JSON: {e2}: {s[:200]!r}"
                ) from e2
        else:
            raise CheckerError(
                f"Checker returned non-JSON: {e}: {s[:200]!r}"
            ) from e

    if not isinstance(obj, dict):
        raise CheckerError(
            f"Checker JSON was not an object: {type(obj).__name__}"
        )
    return obj


def _build_checker_adapter(config, client=None):
    """Resolve the provider adapter for the checker's model.

    The checker model's registry entry picks the provider, so the per-field
    fan-out runs on Claude, GPT, or GLM without any checker-specific config.
    `client` is an already-constructed provider SDK client (the parallel
    fan-out builds one and shares it across every call; tests inject a stub).
    Returns None when no client is available and the provider's key is unset,
    so the caller can wrap that as a per-field CheckerError.
    """
    from direktoro import PROVIDER_ANTHROPIC, model_info
    info = model_info(config.checker_model)
    if info.provider == PROVIDER_ANTHROPIC:
        if client is None:
            if not config.api_key:
                return None
            import anthropic
            client = anthropic.Anthropic(api_key=config.api_key)
        from direktoro import AnthropicAdapter
        return AnthropicAdapter(client, base_url=info.base_url)
    if client is None:
        if not config.api_key:
            return None
        import openai
        client = openai.OpenAI(
            api_key=config.api_key, base_url=info.base_url)
    from direktoro import OpenAIAdapter
    return OpenAIAdapter(
        client, provider=info.provider, base_url=info.base_url)


# ---------------------------------------------------------------------------
# Single-call: check one field
# ---------------------------------------------------------------------------

def check_one_field(*, system_message_blocks, user_message_blocks, config,
                    client=None, adapter=None, api_logger=None,
                    api_log_meta=None):
    """Call the checker LLM for one field via its provider adapter.

    `system_message_blocks` is the cached system prompt as a list of
    content blocks (already including cache_control). `user_message_blocks`
    is the per-field content blocks. `adapter` is a provider adapter; when
    omitted it is built from `config` (and an optional injected `client`).

    Returns a dict with keys: verdict, rationale, input_tokens,
    output_tokens, cache_creation_tokens, cache_read_tokens, cost_usd.
    `cost_usd` is None when the call cannot be costed — a direct model and no
    rate card on `config` — so the token counters stand alone rather than beside
    a zero that would read as a free call.

    Raises CheckerError on any failure.
    """
    from direktoro import (ProviderError, ProviderRateLimitError,
                           ProviderRetryableError, model_info)
    if adapter is None:
        adapter = _build_checker_adapter(config, client)
    if adapter is None:
        # Unreachable in the orchestrated pipeline: Orchestrator._preflight_keys
        # verifies the checker key before any spend, so key absence is caught
        # up front rather than here. Kept as a defensive guard for direct
        # callers that build a config with no key.
        env = model_info(config.checker_model).api_key_env
        raise CheckerError(f"{env} not set; cannot call checker")

    max_retries = 3
    backoff = [2, 4, 8]

    response = None
    for attempt in range(max_retries + 1):
        try:
            response = adapter.create_message(
                model=config.checker_model,
                max_tokens=config.max_tokens,
                system=system_message_blocks,
                messages=[{"role": "user", "content": user_message_blocks}],
                temperature=config.temperature,
                thinking=config.thinking,
            )
            break
        except ProviderRateLimitError as e:
            if attempt < max_retries:
                time.sleep(backoff[attempt])
                continue
            raise CheckerError(f"Rate limit after {max_retries} retries: {e}")
        except ProviderRetryableError as e:
            if attempt < max_retries:
                time.sleep(backoff[attempt])
                continue
            raise CheckerError(
                f"Provider error after {max_retries} retries: {e}")
        except ProviderError as e:
            raise CheckerError(f"Provider API error: {e}")

    # Verbatim API audit log (one entry per call). The adapter carries the
    # canonical request and the resolved provider/model for provenance.
    if api_logger is not None:
        try:
            meta = dict(api_log_meta or {})
            meta.update(provider=response.provider,
                        base_url=response.base_url,
                        wire_model=response.resolved_model,
                        wire_request=response.wire_request)
            api_logger(response.raw_request, response.raw_response, **meta)
        except Exception:
            # Audit-side errors must not abort the checker call.
            pass

    # Truncation, named. On a thinking model the cap covers the reasoning too
    # (see `meltiro.thinking`), so the whole budget can go on reasoning and
    # the JSON be cut off before it closes; unnamed, that reaches
    # `_parse_checker_json` as "Checker returned non-JSON", which points at
    # the model's formatting rather than at the cap. The startup guard in
    # meltiro.thinking refuses caps that cannot fit a think plus an answer,
    # but adaptive thinking can still spend a legal cap, so truncation stays
    # possible. Raised as a CheckerError so run_checker_batch degrades this
    # one field to an error-origin challenge rather than aborting the
    # extraction.
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise CheckerError(
            truncation_message("checker", config.checker_model,
                               config.max_tokens, thinking=config.thinking))

    # Guard the first content block: an empty content list or a non-text
    # first block (for example a stray tool_use) would otherwise raise
    # IndexError/AttributeError, which is not a CheckerError and so would
    # escape the per-field error handling and kill the whole extraction.
    # Raise CheckerError instead, so run_checker_batch wraps it as a single
    # challenged field.
    content = response.content
    if not content:
        raise CheckerError("Checker response had no content blocks")
    raw_text = getattr(content[0], "text", None)
    if raw_text is None:
        block_type = getattr(
            content[0], "type", type(content[0]).__name__)
        raise CheckerError(
            f"Checker response's first content block has no text "
            f"(block type {block_type!r})")
    parsed = _parse_checker_json(raw_text)
    verdict = parsed.get("verdict")
    rationale = parsed.get("rationale", "")
    notes = parsed.get("notes")  # optional free-text field
    if verdict not in VALID_VERDICTS:
        raise CheckerError(
            f"Invalid verdict {verdict!r}; expected one of {VALID_VERDICTS}"
        )

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    # Three costing paths: a ROUTED checker model is priced FROM the response
    # (OpenRouter usage.cost -> reported_cost; a missing value is a loud fault,
    # never $0); a DIRECT model is priced against the checker's rate card, which
    # is recorded with the run so the figure stays checkable; a DIRECT model
    # with no rate card states no cost at all, leaving the token counters above
    # as the record. None is deliberate here and is never a 0.0, which would
    # read as a free call.
    checker_info = model_info(config.checker_model)
    if checker_info.route is not None:
        cost_usd = round(
            reported_cost_or_raise(config.checker_model, response), 6)
    elif config.rates is not None:
        cost_usd = _compute_cost(
            config.rates, input_tokens, output_tokens,
            cache_create, cache_read,
        )
    else:
        cost_usd = None
    return {
        "verdict": verdict,
        "rationale": rationale,
        "notes": notes,
        # A genuine model verdict. The error path in run_checker_batch tags its
        # synthetic challenge with error_origin=True so a disposition can tell a
        # real challenge from an exhausted-retry one; a genuine verdict is never
        # error-origin.
        "error_origin": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_create,
        "cache_read_tokens": cache_read,
        "cost_usd": cost_usd,
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


# ---------------------------------------------------------------------------
# Parallel fan-out
# ---------------------------------------------------------------------------

def run_checker_batch(*, calls, config, client=None, adapter=None,
                      on_complete=None, api_logger=None):
    """Run `check_one_field` for each call in `calls` in parallel.

    One batch is the set of fields one tool call wrote: the orchestrator fans
    out over them after the dispatcher has applied them, and merges the
    verdicts into that call's tool result.

    `calls` is a list of dicts, each with keys:
      - `field_path`: identifier for the field (used in the result map)
      - `system_message_blocks`: cached system prompt (same object reused
        across all calls)
      - `user_message_blocks`: per-field content blocks

    Returns a dict `{field_path: result_dict_or_error_dict}`. Errors
    are wrapped (`{"error": ..., "rationale": "(checker error)"}`) so
    one bad field doesn't abort the batch. The returned dict is ordered by
    field path, not by completion order: results land in nondeterministic
    `as_completed` order, and that order flows into the tool result the
    model reads and into the session event, so the results are sorted before
    returning to keep runs reproducible.

    `on_complete(field_path, result)` is an optional callback invoked
    on the calling thread, inside the `as_completed` loop, as each call
    returns; used by the orchestrator to accumulate spend as results land. It
    still fires in completion order (the accumulation is commutative, so order
    is irrelevant there); only the returned mapping is sorted.
    """
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
            )
        except CheckerError as e:
            # Genuine transient API errors (rate limit / 5xx exhausted, a
            # malformed response) degrade this one field to a challenge rather
            # than aborting the batch. Key absence is NOT expected here: the
            # orchestrator preflight rejects a missing checker key before any
            # spend, so this catch never masks a misconfiguration.
            res = {
                "error": str(e),
                "verdict": "challenge",
                "rationale": f"(checker error: {e})",
                "notes": None,
                # Tag the error origin here, at the source, so a downstream
                # disposition never treats an exhausted-retry challenge as a
                # genuine one: an error-origin challenge must never contribute
                # to a trusted status.
                "error_origin": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                # Zero, not "not priced": this record carries zero tokens
                # because the attempts that would have spent them all failed,
                # and zero tokens cost zero under any rate card. It is a real
                # figure over a real (empty) usage, so it neither needs rates
                # nor withholds a total from a run that has them.
                "cost_usd": 0.0,
            }
        return call["field_path"], res

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.concurrency,
    ) as pool:
        futures = [pool.submit(_one, c) for c in calls]
        for fut in concurrent.futures.as_completed(futures):
            field_path, res = fut.result()
            results[field_path] = res
            if on_complete is not None:
                try:
                    on_complete(field_path, res)
                except Exception:
                    # Audit-side errors shouldn't abort the batch.
                    pass

    # Deterministic order by field path (see docstring).
    return {fp: results[fp] for fp in sorted(results)}
