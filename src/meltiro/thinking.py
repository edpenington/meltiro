"""Per-role thinking / reasoning effort, and the output-cap guard it needs.

Each of the three LLM roles (extractor, checker, reviewer) can name its own
thinking mode and reasoning effort in `pipeline.yaml`. Both keys are optional
per role, and a role that names neither is left alone entirely: no `thinking`
spec is built, `direktoro` emits no thinking wire keys, and no fingerprint
moves.

    extractor_thinking_mode / extractor_thinking_effort
    checker_thinking_mode   / checker_thinking_effort
    review_thinking_mode    / review_thinking_effort

`mode` is `adaptive` (the model decides when and how much to reason) or
`disabled` (no reasoning). `effort` is one of direktoro's `EFFORT_LEVELS`
(`low`, `medium`, `high`, `xhigh`, `max`) and governs how much the model spends
overall. The two are independent: a role may raise its effort without touching
its mode, or turn thinking off without naming an effort.

Both values reach the wire through `direktoro.resolved_decoding_params`, the
single source of truth for the request AND for the decoding-params block
inside `direktoro.call_identity_fields`. meltiro folds that block into every
stage fingerprint (see `meltiro.fingerprint`), so a mode or effort is call
identity without a fingerprint component of its own: two runs differing only
in reasoning effort get different stage fingerprints and a different
`run_fp`. What does NOT move: `instrument_fp` (model-free by construction)
and `engine_fp` (engine versions and source hashes, nothing to do with
effort).

The cap hazard is the defect this module exists for. Claude Opus 5 and
Sonnet 5 think when the `thinking` parameter is OMITTED, and on every Claude
model `max_tokens` caps thinking AND response text together. A role whose
output cap was sized on a non-thinking model is silently under-provisioned
the moment it is pointed at a thinking one: the model spends the budget
reasoning and the answer is cut off. For the checker, whose whole output is
a small JSON verdict under a 1024-token cap, the verdict never arrives, and
the failure looks like a parse error.

`check_role_thinking` refuses that configuration at startup, before a client
exists and before any spend. It does NOT resize the cap: raising a cap is a
spend decision, not meltiro's to make silently.

A floor is not a size, and the refusal has to say so. The floor is the
arithmetic minimum: the smallest allocation a think has ever cost plus the
smallest answer a role can usefully emit. A cap sitting AT it will truncate
on essentially every call — it is where meltiro stops refusing, not a
working size. It does not scale with reasoning effort: Anthropic publishes
no effort-to-thinking-token figure, so a per-effort floor would be a number
meltiro invented. The refusal instead names the effort in force and the one
sizing figure the vendor publishes (see `_cap_sizing_advice`).

The other half of the promise is at run time: a response that stopped on
`max_tokens` is reported as truncation by name rather than surfacing as a
downstream parse failure, and a role that turned thinking ON explicitly is
not told it has a non-thinking problem.
"""

from meltiro.errors import ThinkingConfigError


# direktoro is imported INSIDE each function, never at module scope:
# `meltiro.config_bundle` reads this module's `PIPELINE_KEYS` and is itself
# imported by `import meltiro`, which must keep working with direktoro absent
# (a `--no-deps` wheel consumer has none). No path reachable from a bare
# import needs the provider layer.


# The thinking modes meltiro exposes in `pipeline.yaml`. direktoro also has
# `budget` (a fixed `budget_tokens` allocation), deliberately NOT exposed:
# it needs a companion `budget_tokens` number, and the Claude 4.7-generation
# and later — the whole working set — reject it with a 400. A bundle that
# asks for it is told so by name rather than handed to direktoro to fail.
ACCEPTED_MODES = ("adaptive", "disabled")

# Response headroom a THINKING role must keep beyond the smallest thinking
# allocation the endpoint admits. meltiro's own number (the answer side of
# the cap): 1024 tokens is roughly what the smallest role emits (the
# checker's one-sentence JSON verdict), so a cap that cannot spare this much
# on top of a minimum think cannot produce a usable answer from a thinking
# model. A floor, not a sizing recommendation: the extractor and reviewer
# are configured around 32 times above it.
MIN_RESPONSE_TOKENS = 1024

# The effort levels at which Anthropic publishes a starting `max_tokens`, and
# the figure it publishes. Source: the Claude API migration guidance for Opus
# 4.7 ("At `xhigh` or `max`, set a large `max_tokens` ... Start at 64K and tune
# from there"), restated for Opus 5 ("At `xhigh`/`max`, set `max_tokens` to at
# least 64K") and, without a number, for Sonnet 5 ("set a large output token
# budget ... up to the 128k cap"). A STARTING POINT to tune, not a floor, not
# a guarantee, and not measured for any role's output; meltiro enforces
# nothing against it and quotes it only in the refusal. Below `xhigh` the
# vendor publishes no figure at all, and meltiro does not invent one.
TOP_EFFORT_LEVELS = ("xhigh", "max")
PUBLISHED_CAP_START_AT_TOP_EFFORT = 64000

# pipeline.yaml key names per role, so a refusal names the key the operator has
# to edit rather than describing it. Keyed by the role names meltiro uses
# internally and in `run.json` (`decoding_params`, `structure`).
_ROLE_KEYS = {
    "extractor": {
        "model": "extractor_model",
        "max_tokens": "extractor_max_tokens",
        "mode": "extractor_thinking_mode",
        "effort": "extractor_thinking_effort",
        # The extractor's temperature is the bare `temperature` key. It is not
        # global despite the name (the checker and reviewer have their own), and
        # a refusal that called it `extractor_temperature` would name a key that
        # does not exist.
        "temperature": "temperature",
    },
    "checker": {
        "model": "checker_model",
        "max_tokens": "checker_max_tokens",
        "mode": "checker_thinking_mode",
        "effort": "checker_thinking_effort",
        "temperature": "checker_temperature",
    },
    "review": {
        "model": "review_model",
        "max_tokens": "review_max_tokens",
        "mode": "review_thinking_mode",
        "effort": "review_thinking_effort",
        "temperature": "review_temperature",
    },
}

# Every pipeline.yaml key this module owns, for the config bundle's known-key
# allowlist. Derived from _ROLE_KEYS so the allowlist can never drift from the
# keys actually read.
PIPELINE_KEYS = frozenset(
    keys[which] for keys in _ROLE_KEYS.values() for which in ("mode", "effort")
)


def build_thinking(role, mode=None, effort=None):
    """The `direktoro.Thinking` spec for one role, or None when it names neither.

    None means no thinking parameter reaches the wire and the model does its
    own default — for Opus 5 and Sonnet 5 that default is to think, which is
    the state the cap guard below worries about.

    Raises `ThinkingConfigError` for a mode or effort meltiro does not
    accept, naming the role's own key. Values are checked against direktoro's
    real vocabularies (`ACCEPTED_MODES` is a subset of its `THINKING_MODES`;
    `effort` against its `EFFORT_LEVELS`) so the two cannot drift.
    """
    from direktoro import EFFORT_LEVELS, Thinking

    keys = _ROLE_KEYS[role]
    if mode is None and effort is None:
        return None
    if mode is not None and mode not in ACCEPTED_MODES:
        raise ThinkingConfigError(
            f"{keys['mode']}: {mode!r} is not a thinking mode meltiro accepts; "
            f"the modes are {list(ACCEPTED_MODES)}. "
            f"('budget' is a direktoro mode meltiro does not expose: it needs "
            f"a companion budget_tokens number, and the Claude 4.7-generation "
            f"and later reject it with a 400. Ask for 'adaptive' with an "
            f"{keys['effort']} instead.) Fix pipeline.yaml.")
    if effort is not None and effort not in EFFORT_LEVELS:
        raise ThinkingConfigError(
            f"{keys['effort']}: {effort!r} is not a known reasoning-effort "
            f"level; the levels are {list(EFFORT_LEVELS)}, ascending. Not every "
            f"model accepts every level (the ladder gained 'xhigh' with Opus "
            f"4.7), and a level the model does not have is refused separately, "
            f"against that model's registry entry. Fix pipeline.yaml.")
    return Thinking(mode=mode, effort=effort)


def will_think(model, thinking):
    """Whether a call to `model` carrying `thinking` will reason before answering.

    The question the cap guard turns on, and it cannot be answered from the
    spec alone: a bundle that says nothing about thinking still gets thinking on
    Opus 5 and Sonnet 5, whose registry entries declare `default_on`. So:

      - an explicit `disabled` mode -> False. Either the endpoint accepts
        `{"type": "disabled"}`, or it has no such mode but does not think unless
        asked, in which case direktoro satisfies the request by emitting
        nothing. The remaining case, a model that thinks by default AND has no
        disabled mode, is refused by direktoro before the call, so it never
        reaches a cap question.
      - an explicit `adaptive` mode -> True.
      - no mode named (no spec at all, or effort only) -> the model's declared
        `default_on`.

    A model whose registry entry declares NO thinking support returns False.
    Not a claim that it does not reason: direktoro refuses to emit a thinking
    shape for it at all (every non-Anthropic entry today takes reasoning
    effort from its own `reasoning_effort` registry quirk), so meltiro has
    neither a spec to inspect nor a documented default, and a cap refusal
    built on that would be guessing.
    """
    from direktoro import thinking_support

    support = thinking_support(model)
    if support is None:
        return False
    if thinking is not None and thinking.mode is not None:
        return thinking.mode != "disabled"
    return support.default_on


def thinking_cap_floor(model):
    """The cap BELOW which meltiro refuses a thinking role, or None.

    A refusal threshold, not a recommended size, and not effort-sensitive.
    The ANSWER side is `MIN_RESPONSE_TOKENS`, meltiro's own number. The THINK
    side is `support.budget_min` — the minimum `budget_tokens` on an endpoint
    with a settable thinking budget, which no model in the working set has
    (`claude-opus-5`, `claude-sonnet-5` and `claude-opus-4-8` reject
    `budget_tokens` with a 400; only the 4.6-generation declares the mode,
    deprecated). It is kept anyway: 1024 is the smallest allocation any
    Claude endpoint has ever called a think, so it is the smallest defensible
    LOWER BOUND. What an adaptive think actually costs varies with effort,
    prompt and model, and Anthropic publishes no figure, so anything above
    the bound would be a guess producing false refusals. Reading it off the
    registry means the floor tracks direktoro if the number moves;
    `tests/agentic_extraction/test_thinking.py` pins both claims against the
    real registry.

    Clearing this floor says almost nothing about whether the cap works;
    `check_role_thinking` states that in the refusal itself.

    None when the model declares no thinking support, matching `will_think`.
    """
    from direktoro import thinking_support

    support = thinking_support(model)
    if support is None:
        return None
    return support.budget_min + MIN_RESPONSE_TOKENS


def _cap_sizing_advice(keys, support, thinking):
    """The effort-aware half of the refusal: what effort applies, and what,
    if anything, the vendor publishes about sizing a cap for it.

    The floor cannot consult effort without inventing a number (see
    `thinking_cap_floor`), so effort is consulted here instead. It changes no
    decision, only what the operator is told — a refusal whose first
    suggested number is the floor itself hands the operator a cap that
    truncates. Three cases: no effort surface -> say so and stop; `xhigh` or
    `max` -> quote `PUBLISHED_CAP_START_AT_TOP_EFFORT`, labelled a starting
    point rather than a floor; anything else -> say plainly no figure is
    published for that level, rather than generalising the xhigh number down.

    The effort in force is the role's own `effort` when it names one, else
    the model's registry `default_effort`: the level that actually applies to
    the call.
    """
    if support.default_effort is None:
        return (f"This model declares no reasoning-effort surface, so there is "
                f"no effort level to size against.")
    if thinking is not None and thinking.effort is not None:
        effort = thinking.effort
        whose = f"named by `{keys['effort']}`"
    else:
        effort = support.default_effort
        whose = (f"this model's default, in force because {keys['effort']} "
                 f"is unset")
    if effort in TOP_EFFORT_LEVELS:
        return (
            f"Reasoning effort here is {effort!r} ({whose}), the level at which "
            f"a think spends most. For {' and '.join(repr(e) for e in TOP_EFFORT_LEVELS)} "
            f"Anthropic publishes {PUBLISHED_CAP_START_AT_TOP_EFFORT} as the "
            f"STARTING max_tokens to tune from — a starting point for a role "
            f"doing real work, not a floor, and not measured for this role's "
            f"output.")
    return (
        f"Reasoning effort here is {effort!r} ({whose}). Anthropic publishes no "
        f"max_tokens figure for that level — the figure it does publish "
        f"({PUBLISHED_CAP_START_AT_TOP_EFFORT}) is for "
        f"{' and '.join(repr(e) for e in TOP_EFFORT_LEVELS)} only and does not "
        f"generalise down — so this cap has to be sized from what the role "
        f"actually emits, plus room for a think, and then tuned.")


def _temperature_exit(role, model, max_tokens, temperature, thinking):
    """The extra sentence for a shape that is illegal ONLY because of its
    temperature, or `""` when the temperature is not what broke it.

    On the 4.6-generation and earlier endpoints `temperature` is incompatible
    with active thinking, so `claude-sonnet-4-6` and `claude-haiku-4-5-*`
    turn a legal-looking pair of config values into a 400. direktoro REFUSES
    the pairing rather than dropping one side: dropping either would change
    the run's sampling or its reasoning behind the caller's back. The refusal
    already names the thinking keys; for this one shape the temperature key
    is an equally valid edit, so the message must name both doors.

    WHICH refusal this is, is not re-derived here: meltiro asks direktoro the
    same question again with no temperature, and if that shape resolves, the
    temperature is what made the first one illegal. The rule stays in the one
    place that owns it, and the retry is a pure function over the registry —
    no client, no network, no spend.
    """
    from direktoro import ThinkingUnsupported, resolved_decoding_params

    if temperature is None:
        return ""
    try:
        resolved_decoding_params(
            model, temperature=None, max_tokens=max_tokens, thinking=thinking)
    except ThinkingUnsupported:
        # Something other than the temperature is wrong (an effort level this
        # model does not have, a mode it does not accept). Naming the
        # temperature key would send the operator to edit a line that is not
        # the problem.
        return ""
    keys = _ROLE_KEYS[role]
    return (
        f" meltiro checked: this thinking shape IS accepted by {model!r} with "
        f"no temperature, so `{keys['temperature']}` is the other end of this "
        f"conflict and either key can be the one you change. Removing "
        f"`{keys['temperature']}` lets the {role} think and hands sampling "
        f"back to the endpoint's own default; keeping it means this role "
        f"cannot think on this model. meltiro will not drop either one for "
        f"you: which of the two a run needs is a scientific decision, and "
        f"silently changing the sampling of a run you are about to publish is "
        f"exactly the quiet wrongness this guard exists to prevent.")


def check_role_thinking(role, model, *, max_tokens, temperature, thinking):
    """Refuse, before any spend, a role whose thinking configuration cannot work.

    Two refusals, in the order a call would hit them:

    1. A shape the model's endpoint would reject. Resolved through the same
       `direktoro.resolved_decoding_params` the adapter and the fingerprints
       call, so the check is the real thing; `ThinkingUnsupported` is
       re-raised as a `ThinkingConfigError` carrying the role. This covers
       the role's TEMPERATURE as well as its thinking keys — some models
       accept a temperature and accept thinking but reject a request carrying
       both — and when the temperature is what made the shape illegal the
       refusal says so and names the key (`_temperature_exit`).

    2. An output cap that cannot fit a think plus an answer. See the module
       docstring for the arithmetic: a thinking role under
       `thinking_cap_floor` truncates by construction, not by bad luck.

    A model the registry does not know is skipped rather than raised on: the CLI
    validates every enabled role's model id before an Orchestrator is built, so
    an unknown id here is an offline harness with a synthetic model, for which
    both questions are undefined.
    """
    from direktoro import (
        ThinkingUnsupported, is_known_model, resolved_decoding_params,
        thinking_support)

    keys = _ROLE_KEYS[role]
    if not model or not is_known_model(model):
        return
    try:
        resolved_decoding_params(
            model, temperature=temperature, max_tokens=max_tokens,
            thinking=thinking)
    except ThinkingUnsupported as exc:
        raise ThinkingConfigError(
            f"{role} role ({keys['model']}: {model!r}) asks for a thinking "
            f"shape this model's endpoint would reject: {exc}"
            f"{_temperature_exit(role, model, max_tokens, temperature, thinking)} "
            f"Fix {keys['mode']} / {keys['effort']} in pipeline.yaml.") from exc

    if not will_think(model, thinking):
        return
    floor = thinking_cap_floor(model)
    if floor is None or max_tokens >= floor:
        return

    support = thinking_support(model)
    if thinking is not None and thinking.mode == "adaptive":
        why = f"`{keys['mode']}: adaptive` asks for it"
    else:
        why = (f"{model!r} thinks when the thinking parameter is omitted "
               f"(direktoro registry: thinking_support({model!r}).default_on "
               f"is True) and this config names no {keys['mode']}")
    raise ThinkingConfigError(
        f"{role} role ({keys['model']}: {model!r}) will think, but "
        f"{keys['max_tokens']} is {max_tokens}. On this model max_tokens caps "
        f"the thinking AND the response together, and {max_tokens} is under "
        f"meltiro's refusal floor of {floor} — the arithmetic minimum of the "
        f"smallest allocation any Claude endpoint has called a think "
        f"({support.budget_min}) plus room to answer ({MIN_RESPONSE_TOKENS}). "
        f"Thinking is on here because {why}. "
        f"{_cap_sizing_advice(keys, support, thinking)} "
        f"READ THIS BEFORE EDITING THE CAP: {floor} is the point at which "
        f"meltiro STOPS REFUSING, not a cap that works. It is a lower bound, "
        f"not a size — a cap at or near it will still truncate on essentially "
        f"every call, because an adaptive think spends what it decides to and "
        f"meltiro has no measured figure for what that is. Clearing the floor "
        f"is not a guarantee that a think plus an answer fits. "
        f"Fix pipeline.yaml one of three ways: size {keys['max_tokens']} for a "
        f"think PLUS the output this role actually writes, then tune it "
        f"(meltiro will not size it for you — that is a spend decision); set "
        f"`{keys['mode']}: disabled` to turn thinking off for this role; or "
        f"point {keys['model']} at a model that does not think by default.")


def truncation_message(role, model, max_tokens, *, thinking):
    """The message for a response that stopped on `max_tokens`.

    The run-time half of the startup guard's promise. A cap clearing
    `thinking_cap_floor` can still be spent entirely on reasoning, so
    truncation stays possible above the floor; what must not survive is
    truncation that is not NAMED — unnamed, the checker's cut-off verdict
    reaches the operator as "Checker returned non-JSON", pointing at the
    model's formatting rather than at the cap.

    `thinking` is the role's OWN spec, keyword-only with no default, on
    purpose: diagnosing from the model's `default_on` alone gets the one
    configuration this module exists for exactly backwards (a role that sets
    `adaptive` on a default-off model does think, and would be handed the
    non-thinking advice for a thinking cause). Every call site has the spec;
    pass `None` explicitly for a role that named neither key.
    """
    from direktoro import is_known_model

    thinks = will_think(model, thinking) if is_known_model(model) else False
    if not thinks:
        why = ""
    elif thinking is not None and thinking.mode == "adaptive":
        why = "this role asked for thinking (`adaptive`)"
    else:
        why = "this model thinks by default unless the role sets a thinking mode"
    hint = (
        f" Thinking is on for this call — {why} — and max_tokens caps thinking "
        f"and response together, so the reasoning may have consumed the budget "
        f"before the answer started."
        if thinks else
        " Raise the role's max_tokens, or shorten what it is asked to produce.")
    return (
        f"{role} response was TRUNCATED: the provider stopped on max_tokens "
        f"({max_tokens}) for model {model!r}, so the output is incomplete."
        + hint)
