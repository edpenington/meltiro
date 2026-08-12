#!/usr/bin/env python3
"""Command-line interface for the agentic extraction pipeline.

Six subcommands:

    meltiro extract --config CONFIG_DIR --paper BUNDLE_DIR [--paper ...]
    meltiro transcript SESSION_DIR --out OUT_FILE
    meltiro validate-bundle BUNDLE_DIR [BUNDLE_DIR ...]
    meltiro validate --config CONFIG_DIR EXTRACTION_OUTPUT [--paper BUNDLE_DIR]
                     [--producer llm|human]
    meltiro fingerprint --config CONFIG_DIR
    meltiro render-template --config CONFIG_DIR --view VIEW --out OUT_FILE

plus `meltiro --version`, which takes no subcommand and prints the release
together with the `engine_fp` this tree would record — the version is folded
into that fingerprint and from there into every run's `run_fp`, so an operator
reading a run record needs it without opening Python (see `_version_text`).

`extract` runs the agentic loop over one or more paper bundles against a
config bundle; sessions land under `--out` (default ./runs) at
`{out}/{study_id}/sessions/...` with the run log at `{out}/run_log.json`.

`transcript` renders a finished or paused session as one readable Markdown
document. Every run already writes the same document to
`{session}/diagnostics/transcript.md` at every stop; this subcommand exists so
a session can be re-rendered later, after the renderer improves, without
paying for the run again. It reads the session only, makes no API call, and
touches no fingerprint.

`validate-bundle` reports every problem with each paper bundle and exits
non-zero if any bundle is invalid.

`render-template` renders the config bundle's extraction template as a
human-readable Markdown document, in the operational or publication view, to
an explicit output file. It is a read-only projection of the template and
touches no fingerprint.

Examples:
    meltiro extract --config path/to/config-bundle --paper papers/demo-001
    meltiro extract --config path/to/config-bundle --paper papers/demo-001 --dry-run
    meltiro extract --config path/to/config-bundle --paper papers/a --paper papers/b
    meltiro transcript runs/demo-001/sessions/20260728_120000_481502_abc123 --out t.md
    meltiro validate-bundle papers/demo-001
    meltiro render-template --config path/to/config-bundle --view operational --out template.md
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

from meltiro.bundle import load_bundle, validate_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.diagnostics import DEFAULT_DIAGNOSTICS, DIAGNOSTICS_LEVELS
from direktoro import (
    is_known_model, known_models, model_info, resolved_decoding_params,
    split_decoding_config)
from meltiro.errors import (
    AgenticExtractionError, BundleError, ConfigBundleError, RatesConfigError,
    ResumeRefused, SessionError)
from meltiro.orchestrator import (
    DEFAULT_MAX_CHECKS_PER_FIELD,
    DEFAULT_MAX_REVIEW_TOOL_CALLS,
    DEFAULT_MAX_TOOL_CALLS,
    Orchestrator,
)
from meltiro.rates import Rates, parse_rates
from meltiro.reference_lists import load_reference_list_labels
from meltiro.render_template import render_template
from meltiro.session import Session
from meltiro.template import load_template
from meltiro.transcript import render_transcript
from meltiro.validators import validate_extraction_output


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _version_text():
    """The `--version` block: the release, and the engine axis it produces.

    `engine_fp` is what a run writes into `run.json` and folds into `run_fp`;
    it hashes both packages' versions and both their source digests, so an
    operator holding a run record compares that one line and knows whether
    this tree made it — which versions alone cannot answer for a working
    tree. The checkout (commit, clean/dirty) is printed under it: that says
    WHERE this copy came from once the fingerprint has said whether it
    matches.

    The first line is the conventional `<prog> <version>` and is what a
    script should read; everything after it is detail for a human.
    """
    from meltiro.fingerprint import engine_fingerprint
    from meltiro.run_log import engine_identity, git_state

    identity = engine_identity()
    version, direktoro = identity[0], identity[2]
    commit, dirty = git_state()
    if commit is None:
        tree = "no git repository"
    elif dirty is None:
        tree = f"commit {commit}, tree not examined"
    else:
        tree = f"commit {commit}, {'dirty' if dirty else 'clean'} tree"
    return (
        f"meltiro {version}\n"
        f"{engine_fingerprint(*identity)}\n"
        f"  {tree}\n"
        f"  direktoro {direktoro if direktoro else '(not installed)'}"
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="meltiro",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Resolved eagerly: argparse's `version` action takes a string, and the
    # cost is one `git` call on a command that does nothing else. It sits on
    # the top-level parser, before the subcommand, so `meltiro --version`
    # answers without naming one — the action exits during parsing, ahead of
    # the required-subcommand check below.
    p.add_argument("--version", action="version", version=_version_text())
    sub = p.add_subparsers(dest="command", required=True)

    ex = sub.add_parser(
        "extract",
        help="Run the agentic extraction loop over paper bundle(s).",
        description="Run the agentic extraction loop over one or more paper "
                    "bundles against a config bundle.",
    )
    ex.add_argument("--config", required=True, metavar="CONFIG_DIR",
                    help="Path to the config bundle directory.")
    ex.add_argument("--paper", required=True, action="append",
                    metavar="BUNDLE_DIR", dest="paper",
                    help="Path to a paper bundle directory. Repeatable; each "
                         "bundle is processed in sequence.")
    ex.add_argument("--out", default=None, metavar="DIR",
                    help="Run root for sessions + run_log.json "
                         "(default: ./runs). On --dry-run no session is "
                         "written; the report files are additionally written "
                         "under {out}/{study}/dry_run/ only when --out is "
                         "given.")
    ex.add_argument("--dry-run", action="store_true",
                    help="Render the full system prompt, tool catalogue, "
                         "figure labels, and stage fingerprints, print them "
                         "untruncated, and exit (no API calls, no session "
                         "created).")
    ex.add_argument("--extractor-model", help="Override extractor model.")
    ex.add_argument("--checker-model", help="Override checker model.")
    ex.add_argument("--review-model", help="Override final-review model.")
    ex.add_argument("--max-tool-calls", type=int,
                    help="Override the per-study tool-call cap.")
    ex.add_argument("--max-checks-per-field", type=int,
                    help="Override the total number of checks one field may "
                         "receive. 1 checks a field once and never re-checks "
                         "it; 0 disables the checker entirely (no checker "
                         "calls, no checker model required).")
    ex.add_argument("--final-review", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Run the fresh-context final reviewer (default: on, "
                         "or pipeline.yaml's final_review). Pass "
                         "--no-final-review to disable it; the review model is "
                         "then not required.")
    ex.add_argument("--diagnostics", choices=list(DIAGNOSTICS_LEVELS),
                    default=DEFAULT_DIAGNOSTICS,
                    help="How much of the run's deterministic record to keep "
                         "under the session's diagnostics/ directory. Each "
                         "level is a strict superset of the one below. "
                         "minimal: run.json, field_history.json, "
                         "transcript.md and tool_calls.jsonl. standard "
                         "(default): plus "
                         "instrument/, the prompts and tool definitions as "
                         "sent. full: plus api_calls.jsonl, the verbatim wire "
                         "log. extraction_output.json and tool_calls.jsonl "
                         "are written at every level, so every level can be "
                         "resumed. Operational, not methodology: it moves no "
                         "fingerprint, and the level is recorded in run.json.")
    ex.add_argument("--resume", metavar="SESSION_DIR",
                    help="Resume a specific in-progress session (needs exactly "
                         "one --paper).")
    ex.add_argument("--auto-resume", action="store_true",
                    help="Auto-resume the most recent in-progress session per "
                         "paper (if config_fp matches); else start fresh.")

    tr = sub.add_parser(
        "transcript",
        help="Render a session as one readable Markdown document.",
        description="Render one extraction session as a single Markdown "
                    "document: the instrument as sent, every turn in order "
                    "with each tool result's per-field outcome and every "
                    "checker verdict beside the field it judged, the "
                    "extraction output, and what happened to each field. A "
                    "run writes the same document to "
                    "diagnostics/transcript.md at every stop; this command "
                    "re-renders a session afterwards, which is what lets an "
                    "already-paid run be read through an improved renderer. "
                    "It reads the session only: no API call, no fingerprint "
                    "touched, nothing in the session changed.",
    )
    tr.add_argument("session", metavar="SESSION_DIR",
                    help="Path to a session directory (the one holding "
                         "extraction_output.json and diagnostics/), at "
                         "{out}/{study_id}/sessions/{timestamp}_{fp}/.")
    tr.add_argument("--out", required=True, metavar="OUT_FILE",
                    help="Path of the Markdown file to write. Required, so "
                         "re-rendering never silently overwrites the copy in "
                         "the session directory.")

    vb = sub.add_parser(
        "validate-bundle",
        help="Validate one or more paper bundles.",
        description="Report every problem with each paper bundle; exit 1 if "
                    "any bundle is invalid.",
    )
    vb.add_argument("bundle", nargs="+", metavar="BUNDLE_DIR",
                    help="Paper bundle directory to validate.")

    va = sub.add_parser(
        "validate",
        help="Re-validate the field values in an extraction output.",
        description="Re-validate every field value in an extraction_output."
                    "json against a config bundle: types, options, reference "
                    "lists, and (for engine-produced output) the template's "
                    "evidence contract. Pass --paper to also check evidence "
                    "quotes verbatim against the paper text and resolve "
                    "<img> labels against its figures. Exit 1 if any field "
                    "value fails. Thin wrapper over "
                    "meltiro.validators.validate_extraction_output.",
    )
    va.add_argument("--config", required=True, metavar="CONFIG_DIR",
                    help="Path to the config bundle directory.")
    va.add_argument("extraction_output", metavar="EXTRACTION_OUTPUT_JSON",
                    help="Path to an extraction_output.json to re-validate.")
    va.add_argument("--paper", metavar="BUNDLE_DIR",
                    help="Optional paper bundle; when given, evidence quotes "
                         "carried by a value are checked verbatim against the "
                         "paper text and <img> labels are resolved against "
                         "its figures. Without it those two verdicts are "
                         "unreachable and are not reported.")
    va.add_argument("--producer", choices=("llm", "human"), default="llm",
                    help="Who produced the values, which decides whether the "
                         "evidence contract is enforced. llm (the default) "
                         "holds every field to the template's own `evidence: "
                         "required` / `evidence: optional` flag, so a value "
                         "asserted with no quote and no figure reference "
                         "fails. That is the contract engine-produced output "
                         "was written under, and the one a third party "
                         "auditing a published extraction needs re-checked. "
                         "human demands no evidence at all and checks values "
                         "alone, for hand-authored comparison data nobody "
                         "promised evidence for.")

    fp = sub.add_parser(
        "fingerprint",
        help="Print a config bundle's content fingerprint as JSON.",
        description="Print the config bundle's content fingerprint (the "
                    "component hashes plus the instrument fingerprint) "
                    "as machine-readable JSON. Thin wrapper over "
                    "load_config_bundle.",
    )
    fp.add_argument("--config", required=True, metavar="CONFIG_DIR",
                    help="Path to the config bundle directory.")

    rt = sub.add_parser(
        "render-template",
        help="Render the extraction template as a Markdown document.",
        description="Render the config bundle's extraction template as a "
                    "human-readable Markdown document. One view per "
                    "invocation, written to an explicit output file. A "
                    "read-only projection of the template: it runs no "
                    "extraction and touches no fingerprint.",
    )
    rt.add_argument("--config", required=True, metavar="CONFIG_DIR",
                    help="Path to the config bundle directory (the template is "
                         "read from its extraction_template.yaml).")
    rt.add_argument("--view", required=True,
                    choices=("operational", "publication"),
                    help="Which view to render: operational (reviewer / "
                         "extractor facing, with instructions and QA blocks) "
                         "or publication (paper facing, descriptions only).")
    rt.add_argument("--out", required=True, metavar="OUT_FILE",
                    help="Path of the Markdown file to write.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def _required_max_tokens(loop_cfg, key):
    """The cap `key` states, else exit 1.

    Required rather than defaulted: the cap bounds what one call may spend and
    what it may answer within, and both are decisions about the run. A number
    meltiro chose would look in the run record exactly like a number the
    operator chose.

    Strict about the type as well as the range, on the same terms as
    `checker_context_chars` below: a bool is not a budget, and a float or a
    quoted string would coerce silently into a cap the bundle does not say and
    ride into the run record as one the operator wrote. Zero and negatives
    are not budgets either — they reach the provider as-is and fail at the
    role's first call, which for the reviewer is after a whole extraction and
    checker fan-out have been billed.
    """
    value = loop_cfg.get(key)
    if value is None:
        print(
            f"pipeline.yaml must set {key}. It is the output-token cap for "
            f"that role's calls, and it has no default: the number bounds "
            f"what the role may spend and what it may answer within, so it is "
            f"the operator's to state. A role whose stage is off needs no cap "
            f"(the checker's stage is off at max_checks_per_field: 0, the "
            f"reviewer's at final_review: false).",
            file=sys.stderr)
        raise SystemExit(1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        print(
            f"{key} must be a positive integer, got {value!r}. It is the "
            f"output-token cap for that role's calls: a zero or negative "
            f"number is not an output budget, and a decimal or a quoted "
            f"number is not the cap the run would record. Fix pipeline.yaml.",
            file=sys.stderr)
        raise SystemExit(1)
    return value


def _resolve_role_rates(enabled_roles, operator_cards, *, today):
    """Each enabled role's rate card, and the startup line that says why.

    Returns `({role: Rates-or-None}, [line, ...])` over `enabled_roles`, a list
    of `(role, model)` in call order.

    Three sources, tried in this order and never across roles — each role
    runs its own model, so another role's numbers would price it wrong:

      - the card the operator gave THIS role under `rates:`;
      - direktoro's dated price table, for a direct model whose published
        rate it carries. The card records the date the vendor's page was
        read and the table version, so the run says which reading priced it;
      - nothing, and the role runs unpriced: its tokens are recorded, it
        states no dollar figure, and the run states no total.

    A ROUTED model is priced from the charge the gateway reports on each
    response; it needs no card, and the table does not price it.

    `today` is passed in rather than read here, so the age this reports is
    a function of its arguments alone. The CLI is where the clock is read.
    """
    from direktoro.prices import PRICES_VERSION, price_age_days, price_for

    cards = {}
    report = []
    for role, model in enabled_roles:
        operator_card = operator_cards.get(role)
        if operator_card is not None:
            cards[role] = operator_card
            report.append(
                f"  Pricing, {role} ({model}): the card written for this role "
                f"under `rates:` in pipeline.yaml.")
            continue
        if model_info(model).route is not None:
            cards[role] = None
            report.append(
                f"  Pricing, {role} ({model}): routed, so every call is priced "
                f"at the charge the gateway reports for it.")
            continue
        entry = price_for(model)
        if entry is None:
            cards[role] = None
            report.append(
                f"  Pricing, {role} ({model}): unpriced. Its tokens are "
                f"recorded, it states no dollar figure, and the run states no "
                f"total. Give the role a card under `rates:` in pipeline.yaml "
                f"to price it.")
            continue
        cards[role] = Rates.from_table(entry, PRICES_VERSION)
        report.append(
            f"  Pricing, {role} ({model}): direktoro price table v"
            f"{PRICES_VERSION}, read {entry.as_of} "
            f"({price_age_days(model, today)} days ago).")
    return cards, report


def _build_orchestrator(config, bundle, out_dir, loop_cfg, args):
    # Numeric overrides use `is not None` so an explicit 0 is honoured; `or`
    # would discard it, which for a cap or a check budget is the difference
    # between "off" and "default".
    #
    # Every fallback below is the orchestrator's own DEFAULT_* constant, not
    # a literal repeated here: the run record writes the resolved number into
    # `caps`, and a CLI that disagreed with the library would put a cap in
    # the artefact that the documented default does not explain.
    if args.max_tool_calls is not None:
        max_tool_calls = args.max_tool_calls
    else:
        max_tool_calls = int(loop_cfg.get(
            "max_tool_calls", DEFAULT_MAX_TOOL_CALLS))
    # Same positive-integer rule as the reviewer's budget below. A cap of
    # zero or less is not a smaller budget: the run pauses at once having
    # done nothing, and `--resume` reaches the same bound again.
    if max_tool_calls < 1:
        print(
            f"max_tool_calls must be a positive integer, got "
            f"{max_tool_calls}. The extractor needs at least one tool call to "
            f"record anything; a cap below 1 pauses the run before it starts. "
            f"Fix pipeline.yaml (or --max-tool-calls).",
            file=sys.stderr)
        raise SystemExit(1)
    if args.max_checks_per_field is not None:
        max_checks_per_field = args.max_checks_per_field
    else:
        max_checks_per_field = int(loop_cfg.get(
            "max_checks_per_field", DEFAULT_MAX_CHECKS_PER_FIELD))
    if max_checks_per_field < 0:
        print(
            f"max_checks_per_field must be zero or a positive integer, got "
            f"{max_checks_per_field}. 0 disables the checker; 1 checks each "
            f"field once; 2 adds one re-check after a revision. Fix "
            f"pipeline.yaml.",
            file=sys.stderr)
        raise SystemExit(1)

    # The reviewer's own tool-call budget. A reviewer allowed zero tool calls
    # could never even call mark_complete, so every run would terminate on
    # the bound with the review stage nominally on: a config error, failed at
    # startup rather than after the extractor has fully spent.
    max_review_tool_calls = int(loop_cfg.get(
        "max_review_tool_calls", DEFAULT_MAX_REVIEW_TOOL_CALLS))
    if max_review_tool_calls < 1:
        print(
            f"max_review_tool_calls must be a positive integer, got "
            f"{max_review_tool_calls}. The reviewer needs at least one tool "
            f"call to reach mark_complete. To run without a reviewer, set "
            f"final_review: false (or pass --no-final-review). Fix "
            f"pipeline.yaml.",
            file=sys.stderr)
        raise SystemExit(1)

    # Pipeline structure for this run. The checker is off when
    # max_checks_per_field is 0 (no separate flag); the reviewer stage is off
    # when final_review is False. A CLI --final-review / --no-final-review
    # override wins over pipeline.yaml (default on when neither sets it).
    # `check_reviewer_edits` extends the checker to the reviewer's own tool
    # calls; it is off by default and has no CLI flag, because it is
    # methodology rather than an operational budget.
    checker_enabled = max_checks_per_field > 0
    if args.final_review is not None:
        final_review = args.final_review
    else:
        final_review = bool(loop_cfg.get("final_review", True))
    check_reviewer_edits = bool(loop_cfg.get("check_reviewer_edits", False))

    # Model-requirement matrix: the extractor model is always required; the
    # checker model iff the checker is on; the review model iff the reviewer is
    # on. A disabled stage's model is deliberately NOT demanded, so an
    # extractor-only run needs no checker or review model at all. There is no
    # hardcoded fallback model for any role.
    extractor_model = args.extractor_model or loop_cfg.get("extractor_model")
    review_model = args.review_model or loop_cfg.get("review_model")
    checker_model = args.checker_model or loop_cfg.get("checker_model")
    # `(role, model)` for every stage that will run, in call order. It is the
    # one list of what this run actually calls: the model guards below check
    # exactly these, and pricing resolves exactly these, so a stage cannot be
    # gated by one and priced by the other.
    enabled_roles = [("extractor", extractor_model)]
    if checker_enabled:
        enabled_roles.append(("checker", checker_model))
    if final_review:
        enabled_roles.append(("review", review_model))
    required_models = [(f"{role}_model", model)
                       for role, model in enabled_roles]
    missing = [k for k, v in required_models if not v]
    if missing:
        flags = ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        print(
            f"pipeline.yaml must set {', '.join(missing)} "
            f"(or pass the matching flag: {flags})",
            file=sys.stderr)
        raise SystemExit(1)

    # Every required model must be in direktoro's model registry, which says
    # where a model is served, how to reach it and what it can accept: an id
    # in no registry entry has no endpoint to call, so fail at startup,
    # before any spend. Only the required (enabled-stage) models are checked;
    # a disabled stage may leave its model unset.
    unknown = [(label, m) for label, m in required_models
               if not is_known_model(m)]
    if unknown:
        listed = "; ".join(f"{label}={m!r}" for label, m in unknown)
        print(
            f"unknown model(s): {listed}. Known models: "
            f"{', '.join(known_models())}. Fix pipeline.yaml or pass a known "
            f"--extractor-model / --checker-model / --review-model.",
            file=sys.stderr)
        raise SystemExit(1)

    # Reject any required model the registry marks `retired`. The check
    # belongs here and NOT inside the shared `model_info` lookup: the
    # registry keeps a retired entry so past runs still resolve against it
    # (provenance calls `model_info`), and only starting a NEW run with a
    # withdrawn id is the error. Failing at config load turns a mid-run 404
    # into a clean startup failure before any spend.
    retired = [(label, m) for label, m in required_models
               if model_info(m).retired]
    if retired:
        listed = "; ".join(f"{label}={m!r}" for label, m in retired)
        print(
            f"retired model(s): {listed}. These ids are retired and cannot be "
            f"used for new runs (the provider has withdrawn them, so a run "
            f"would fail on the first API call). The registry keeps them only "
            f"to price and resolve past runs. Fix pipeline.yaml or pass a "
            f"current --extractor-model / --checker-model / --review-model.",
            file=sys.stderr)
        raise SystemExit(1)

    checker_config = CheckerConfig.from_env(model_override=checker_model)
    # How many checker calls run at once; it becomes
    # `ThreadPoolExecutor(max_workers=...)`, so 0 is not a way to turn the
    # checker off (that is `max_checks_per_field: 0`) and would abort the
    # first checker batch after the extractor had already spent. `is not
    # None` so a configured 0 reaches that guard instead of reading as
    # absent; the EFFECTIVE value is checked, so a CHECKER_CONCURRENCY of 0
    # in the environment fails here too. Purely operational — how fast the
    # same calls are made — so it rides into no fingerprint.
    if loop_cfg.get("checker_concurrency") is not None:
        checker_config.concurrency = int(loop_cfg["checker_concurrency"])
    if checker_config.concurrency < 1:
        print(
            f"checker_concurrency must be a positive integer, got "
            f"{checker_config.concurrency}. It is how many checker calls run "
            f"in parallel, so 0 is not a valid value and does not disable the "
            f"checker; set max_checks_per_field: 0 for that. Fix pipeline.yaml "
            f"(or the CHECKER_CONCURRENCY environment variable).",
            file=sys.stderr)
        raise SystemExit(1)
    # One output cap per ENABLED role, each stated in pipeline.yaml. A cap
    # bounds what a call may spend and what it may answer within, so there is
    # no number meltiro could supply for an operator who did not write one.
    # A disabled stage makes no calls, so it needs none.
    caps = {role: _required_max_tokens(loop_cfg, f"{role}_max_tokens")
            for role, _ in enabled_roles}
    if checker_enabled:
        checker_config.max_tokens = caps["checker"]

    # Characters of surrounding paper text the checker sees on each side of a
    # matched quote. Strict about the type as well as the range: a float or a
    # quoted string would coerce silently and ride into checker_fp as
    # something other than what the bundle wrote. 0 is legal (the checker
    # sees the quote alone). `is not None` so an explicit 0 reaches
    # validation rather than being swallowed as absent.
    if loop_cfg.get("checker_context_chars") is not None:
        context_chars = loop_cfg["checker_context_chars"]
        if (isinstance(context_chars, bool)
                or not isinstance(context_chars, int)
                or context_chars < 0):
            print(
                f"checker_context_chars must be a non-negative integer, got "
                f"{context_chars!r}. It is how many characters of surrounding "
                f"paper text the checker sees on each side of a quote; 0 "
                f"shows the quote alone. Fix pipeline.yaml.",
                file=sys.stderr)
            raise SystemExit(1)
        checker_config.context_chars = context_chars

    # One decoding block per role, each an opaque mapping of decoding
    # parameter names to values. meltiro reads no key inside a block: it hands
    # the whole mapping to direktoro's `split_decoding_config`, which knows
    # which name is a sampling control and which is a thinking field, and
    # returns the pair its resolver takes. A block naming something direktoro
    # does not emit fails here, before a session or a bill, rather than being
    # silently dropped from the wire and from the fingerprint that records it.
    #
    # Every role's block is split, enabled or not, so a typo in a stage this
    # run happens to leave off is still named. A key absent from a block stays
    # absent all the way to the wire: nothing substitutes a value for it, and
    # nothing is taken from another role, so the model's own default applies
    # and the run records that this role specified none. Pin a parameter by
    # writing it in that role's own block.
    blocks = {}
    for role in ("extractor", "review", "checker"):
        key = f"{role}_decoding"
        try:
            blocks[role] = split_decoding_config(loop_cfg.get(key))
        except ValueError as e:
            print(f"{key}: {e} Fix pipeline.yaml.", file=sys.stderr)
            raise SystemExit(1)
    checker_config.sampling, checker_config.thinking = blocks["checker"]

    # Each enabled role's call resolved against the registry, before a client
    # exists and before any spend. It is the same `resolved_decoding_params`
    # the adapters and the stage fingerprints call, so this asks the real
    # question: whether the model's endpoint accepts this block at this cap.
    # A value outside the model's documented band, an effort or a mode it does
    # not have, a cap a thinking call cannot answer within — each fails here
    # rather than on a paid call, and from HERE so `--resume` (whose
    # _build_orchestrator call sits outside its own try) is covered on the
    # same terms as a fresh run. Pure registry arithmetic: no client, no
    # network.
    #
    # Ahead of the pricing lines below, so a refused run prints its refusal
    # and nothing else: a rate report for calls that will never be made reads
    # as a run about to start.
    for role, model in enabled_roles:
        sampling, thinking = blocks[role]
        try:
            resolved_decoding_params(model, max_tokens=caps[role],
                                     sampling=sampling, thinking=thinking)
        except ValueError as e:
            print(
                f"{role} role ({role}_model: {model!r}): {e} Fix "
                f"{role}_decoding / {role}_max_tokens in pipeline.yaml.",
                file=sys.stderr)
            raise SystemExit(1)

    # The rate cards `rates:` gives per role, if any. Optional: a role the block
    # does not name takes its rates from direktoro's price table below. A block
    # that IS written must be complete and usable, and every value is read for
    # presence rather than truth, so a legitimate `0.0` rate is honoured. See
    # `meltiro.rates`.
    try:
        operator_cards = parse_rates(loop_cfg)
    except RatesConfigError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    # One card per enabled role, resolved here and reported on stdout so an
    # operator sees what will price each stage before a token is spent.
    rates, pricing_report = _resolve_role_rates(
        enabled_roles, operator_cards, today=date.today())
    for line in pricing_report:
        print(line)

    extractor_sampling, extractor_thinking = blocks["extractor"]
    review_sampling, review_thinking = blocks["review"]
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor_model,
        checker_config=checker_config,
        review_model=review_model,
        max_tool_calls=max_tool_calls,
        max_review_tool_calls=max_review_tool_calls,
        max_checks_per_field=max_checks_per_field,
        check_reviewer_edits=check_reviewer_edits,
        sampling=extractor_sampling,
        review_sampling=review_sampling,
        thinking=extractor_thinking,
        review_thinking=review_thinking,
        extractor_max_tokens=caps["extractor"],
        # The reviewer's OWN cap, wired into the final-review call and
        # review_fp, so the reviewer is never sized by the extractor's
        # number.
        review_max_tokens=caps.get("review"),
        final_review=final_review,
        # Commercial, not methodological, so it reaches no fingerprint
        # (see meltiro.rates). One card per role, keyed by role name;
        # the Orchestrator hands the checker's to the checker.
        rates=rates,
        # Operational, so it comes from the command line only and never
        # from the config bundle: it changes which diagnostics files a run
        # writes and nothing about what any model is asked, so it must not
        # be able to ride into a fingerprint via pipeline.yaml.
        diagnostics=args.diagnostics,
        dry_run=args.dry_run,
    )


def _print_run_summary(orch, status):
    print(f"  Status: {status}")
    print(f"  Session: {orch.session.session_dir}")
    print(f"  Tool calls dispatched: {orch.session.meta.get('tool_call_count')}")
    print(f"  Checker calls: {orch.session.meta.get('checker_calls_run')}")
    # A run states a total or it states none. "not priced" is printed where the
    # figure would go, never `$0.0000`, which would tell an operator the run was
    # free when what actually happened is that nothing said what it cost.
    cost = orch.recorded_cost()
    print(f"  Total cost: ${cost:.4f}" if cost is not None
          else "  Total cost: not priced (tokens recorded; a role ran with "
               "neither a `rates:` card nor a price-table entry)")
    meta = orch.session.meta
    if status == "in_progress":
        # A tool-call-cap PAUSE. The session is genuinely in_progress and its
        # meta says so, so --resume reattaches and continues the same
        # conversation once the cap is raised.
        print()
        print("  NOTE: the tool-call cap fired; the run paused with the "
              "session")
        print("  left in_progress (pause_reason: "
              f"{meta.get('pause_reason')}). The extraction output may be")
        print("  mid-extraction. Raise the cap and resume the same "
              "conversation:")
        print()
        print("    meltiro extract --config <config> --paper <bundle> \\")
        print(f"      --resume {orch.session.session_dir} \\")
        print("      --max-tool-calls 200")
    elif status == "failed_validation":
        # Terminal and NOT resumable. A config-side fix moves the fingerprint
        # (Session.resume then refuses on drift), and an unchanged resume feeds
        # the same inputs back to the same model, which is expected to reach
        # the same dead end (a sampled model may not, but the run is not worth
        # re-billing on that chance).
        reason = meta.get("failure_reason")
        detail = meta.get("failed_validation_reason")
        print()
        print("  NOTE: the run ended in failed_validation "
              f"(failure_reason: {reason}). This is a")
        print("  terminal outcome and is NOT resumable: fix the config or the")
        print("  inputs, then start a FRESH run (a resume would replay into "
              "the")
        print("  same failure, or be refused on config-fingerprint drift).")
        if detail:
            print(f"    surrender reason: {detail}")
    elif status == "complete":
        # A challenge the extractor never satisfied does not hold the run open
        # (the checker is advisory), but it is worth naming at the end of a run
        # so an operator can look at the cell if they want to.
        unresolved = (meta.get("checker_diagnostics") or {}).get(
            "unresolved_challenges") or []
        if unresolved:
            print()
            print(f"  NOTE: {len(unresolved)} field(s) were still challenged "
                  f"by the checker when their")
            print("  per-field check budget ran out. The checker is advisory, "
                  "so this does")
            print("  not change the status; the fields are listed in "
                  "run.checker_diagnostics.")


def _run_one(config, bundle, out_dir, loop_cfg, args):
    """Run one study. Returns the terminal status string, or "error" when the
    run raises or the study finalises in an error state.

    Startup guards raised from prepare_new_session/resume_session (the
    study-identity guard, an unresolved reference placeholder, a missing API
    key) are AgenticExtractionError subclasses, caught here so they surface
    as a clean one-line stderr message and an "error" status rather than a
    raw traceback.
    """
    print(f"\n=== Study {bundle.study_id} ===")
    try:
        orch = _build_orchestrator(config, bundle, out_dir, loop_cfg, args)

        if args.dry_run:
            # A dry run creates NO session: render the instrument, print it
            # in full, and (only when --out was given) write the report files
            # under {out}/{study}/dry_run/. Nothing to resume or finalise, so
            # a non-error sentinel is returned and the run summary skipped.
            #
            # `args.out is None` is the sole signal for "print only, write
            # nothing"; it must stay keyed on the raw absence of --out, not
            # on the resolved out_dir, which always defaults to ./runs (see
            # _cmd_extract).
            report_dir = (out_dir / bundle.study_id / "dry_run"
                          if args.out is not None else None)
            orch.dry_run_report(report_dir=report_dir)
            return "dry_run"

        if args.auto_resume:
            # Find the most recent in_progress session for this study whose
            # EXTRACTOR config still matches, and try to resume it. The
            # config_fp filter is what makes that reliable: find_in_progress
            # returns only the single newest in-progress session, so without
            # the filter a newer session under a drifted config hides an older
            # in-progress session this config could resume, and that older
            # session's banked API spend is thrown away and re-spent from
            # scratch. config_fp comes from the orchestrator's own fingerprint
            # recipe (the same one prepare_new_session and the dry run use), so
            # learning it needs no throwaway session dir and emits no warning.
            # The full three-fingerprint drift gate in resume_session stays the
            # authority on whether the chosen candidate is actually resumable;
            # this only narrows WHICH candidate is offered to it. Either branch
            # prepares the session exactly once.
            expected_config_fp = orch._build_fingerprints()["config_fp"]
            candidate = Session.find_in_progress(
                bundle.study_id, runs_dir=out_dir,
                expected_config_fp=expected_config_fp)
            resumed = False
            if candidate is not None:
                print(f"  Auto-resuming session: {candidate}")
                try:
                    orch.resume_session(candidate)
                    resumed = True
                except ResumeRefused as e:
                    print(f"  Resume refused, starting fresh: {e}")
            if not resumed:
                orch.prepare_new_session()
        else:
            orch.prepare_new_session()

        status = orch.run()
    except AgenticExtractionError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return "error"
    _print_run_summary(orch, status)
    return status


def _resume_one(config, bundle, out_dir, loop_cfg, args):
    """Resume one study. Returns the terminal status string, or "error" when
    the run raises or finalises in an error state."""
    session_dir = Path(args.resume)
    meta_path = Session.meta_path_for(session_dir)
    if not meta_path.exists():
        print(f"No diagnostics/run.json at {meta_path}", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    # Guard against pointing --resume at a session for a different study.
    if str(meta.get("study_id")) != str(bundle.study_id):
        print(
            f"--resume session is for study {meta.get('study_id')!r} but "
            f"--paper is study {bundle.study_id!r}. Point --paper at the "
            f"matching bundle.",
            file=sys.stderr,
        )
        sys.exit(2)
    # The run root comes from the SESSION, not from --out. A session lives at
    # {root}/{study}/sessions/{timestamp}, so the root is recoverable from the
    # path the operator already supplied, and recovering it is what keeps a
    # resumed segment's run-log entry in the same index as the run it
    # continues. Taking --out instead would let a resume that omits the flag
    # append silently to ./runs while the session sits elsewhere, splitting a
    # run from its own index — and the pause message prints a resume command
    # with no --out, so following the tool's own instructions would cause
    # exactly that.
    session_root = session_dir.resolve().parents[2]
    if args.out is not None and Path(args.out).resolve() != session_root:
        print(
            f"--out {args.out} disagrees with the session's own run root "
            f"{session_root}. A resumed session writes its run-log entry to "
            f"the root it already lives in; drop --out, or point it there.",
            file=sys.stderr,
        )
        sys.exit(2)
    out_dir = session_root

    orch = _build_orchestrator(config, bundle, out_dir, loop_cfg, args)
    try:
        orch.resume_session(session_dir)
        status = orch.run()
    except ResumeRefused as e:
        print(f"Resume refused: {e}", file=sys.stderr)
        sys.exit(2)
    except AgenticExtractionError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return "error"
    _print_run_summary(orch, status)
    return status


def _cmd_extract(args):
    if args.resume and args.auto_resume:
        print("--resume and --auto-resume are mutually exclusive.",
              file=sys.stderr)
        return 2
    if args.dry_run and (args.resume or args.auto_resume):
        print("--dry-run cannot be combined with --resume or --auto-resume: "
              "a dry run creates no session, so there is nothing to resume.",
              file=sys.stderr)
        return 2
    if args.resume and len(args.paper) != 1:
        print("--resume requires exactly one --paper.", file=sys.stderr)
        return 2

    try:
        config = load_config_bundle(args.config)
    except ConfigBundleError as e:
        print(str(e), file=sys.stderr)
        return 1

    # --out defaults to ./runs for a real run. On a dry run its ABSENCE means
    # "print only" (see _run_one), so the default must stay local to out_dir
    # and never migrate onto args.out itself: an argparse-level --out default
    # would silently make bare dry runs start writing report dirs.
    out_dir = Path(args.out) if args.out is not None else Path("./runs")
    loop_cfg = config.pipeline

    # Load + validate every bundle up front so a bad bundle fails loudly
    # before any run starts.
    bundles = []
    for paper_dir in args.paper:
        try:
            bundles.append(load_bundle(paper_dir))
        except BundleError as e:
            print(str(e), file=sys.stderr)
            return 1

    # Exit nonzero (1) only when a study raises or finalises in status
    # "error"; every other terminal status exits 0. A paused run
    # ("in_progress", the tool-call cap) and "failed_validation" (a
    # produced-but-invalid extraction) each still leave a session and an
    # extraction output, so only a hard "error" fails the command. (Usage and
    # resume-refusal errors exit 2, handled in _resume_one / _cmd_extract
    # argument checks.)
    if args.resume:
        status = _resume_one(config, bundles[0], out_dir, loop_cfg, args)
        return 1 if status == "error" else 0

    statuses = [
        _run_one(config, bundle, out_dir, loop_cfg, args)
        for bundle in bundles
    ]
    return 1 if any(s == "error" for s in statuses) else 0


# ---------------------------------------------------------------------------
# transcript (Markdown rendering of one session)
# ---------------------------------------------------------------------------

def _cmd_transcript(args):
    """Render one session directory to a Markdown file.

    Strict inputs, and no partial document: a missing session directory, a
    missing or unparseable run.json, a corrupt event log, a missing extraction
    output, and an unknown recorded diagnostics level are each a loud failure
    with exit 1 and nothing written. A session that legitimately kept less
    (`--diagnostics minimal` captured no instrument) is NOT an error: the
    document says what the level stopped it from showing, in place of the
    section it would have rendered.
    """
    try:
        document = render_transcript(args.session)
    except SessionError as e:
        print(str(e), file=sys.stderr)
        return 1

    out_path = Path(args.out)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    print(f"Wrote transcript to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# validate-bundle
# ---------------------------------------------------------------------------

def _cmd_validate_bundle(args):
    any_invalid = False
    for path in args.bundle:
        problems = validate_bundle(path)
        if problems:
            any_invalid = True
            print(f"INVALID: {path}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"OK: {path}")
    return 1 if any_invalid else 0


# ---------------------------------------------------------------------------
# validate (re-validate stored field values)
# ---------------------------------------------------------------------------

def _cmd_validate(args):
    """Re-validate every field value in an extraction_output.json.

    Thin wrapper over validators.validate_extraction_output: loads the config
    bundle + template, reads the extraction output, optionally loads a paper
    bundle for quote checking, and reports every field failure. Exits 1 when
    any field value fails.

    Two switches decide what is in scope, and the verdict says which of them
    were on rather than claiming the whole file was checked: `--producer`
    (whether the template's evidence contract is enforced at all) and
    `--paper` (whether an evidence quote can be checked against its source).
    """
    try:
        config = load_config_bundle(args.config)
    except ConfigBundleError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        with open(args.extraction_output, "r", encoding="utf-8") as f:
            extraction_output = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read extraction output "
              f"{args.extraction_output}: {e}", file=sys.stderr)
        return 1
    if not isinstance(extraction_output, dict):
        print("extraction output must be a JSON object with study / records / "
              "initial_check / quality_check blocks.", file=sys.stderr)
        return 1

    paper_text = None
    image_labels = None
    if args.paper:
        try:
            bundle = load_bundle(args.paper)
        except BundleError as e:
            print(str(e), file=sys.stderr)
            return 1
        paper_text = bundle.text
        image_labels = set(bundle.figures)

    template = load_template(config.template_path)
    failures, warnings = validate_extraction_output(
        template, extraction_output, config.reference_lists,
        paper_text=paper_text, image_labels=image_labels,
        producer_kind=args.producer,
    )

    for w in warnings:
        print(f"WARNING {w['path']} [{w['code']}]: {w['message']}")
    for fail in failures:
        print(f"FAIL {fail['path']} [{fail['code']}]: {fail['message']}")
    if failures:
        print(f"\n{len(failures)} field value(s) failed re-validation.")
        return 1

    checked = ["field values: types, options, and reference lists"]
    not_checked = []
    if args.producer == "llm":
        checked.append(
            "the evidence contract: every field the template marks "
            "`evidence: required` carries a quote or a figure reference")
    else:
        not_checked.append(
            "the evidence contract: --producer human demands no evidence. "
            "Re-run with --producer llm to hold engine-produced values to "
            "the template's own flag")
    if paper_text is not None:
        checked.append(
            "evidence quotes, verbatim against the paper text, and <img> "
            "labels against the bundle's figures")
    else:
        not_checked.append(
            "whether the evidence quoted is really in the paper: pass "
            "--paper to check quotes and figure labels against the bundle")
    print("OK: no field value failed the checks that ran.")
    for line in checked:
        print(f"  checked: {line}")
    for line in not_checked:
        print(f"  not checked: {line}")
    return 0


# ---------------------------------------------------------------------------
# fingerprint (content fingerprint of a config bundle)
# ---------------------------------------------------------------------------

def _cmd_fingerprint(args):
    """Print a config bundle's content fingerprint as machine-readable JSON.

    Thin wrapper over load_config_bundle: prints the component hashes the
    consumer pins on plus the instrument fingerprint, the model-free,
    engine-free identity of everything the config author wrote. The composite
    config_fp is deliberately not printed (it folds in the extractor model,
    which this command does not know); the note field says so explicitly.
    """
    try:
        config = load_config_bundle(args.config)
    except ConfigBundleError as e:
        print(str(e), file=sys.stderr)
        return 1

    payload = {
        "template_hash": config.template_hash,
        "reference_lists_hash": config.reference_lists_hash,
        "prompts_hash": config.prompts_hash,
        "instrument_fp": config.instrument_fp,
        "note": (
            "config_fp is not printable from a config directory alone: it "
            "folds in the extractor model (and the run's decoding "
            "parameters), which this command does not know. The consumer "
            "pins (template_hash, reference_lists_hash), which alone decide "
            "whether a stored value is still legal. instrument_fp IS "
            "printable here and is what a run records too, so it is the key "
            "to group runs of this config across models and meltiro "
            "versions. A CLI flag overriding pipeline.yaml (for example "
            "--max-checks-per-field) changes the instrument, so a run started "
            "that way records a different instrument_fp from this one."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


# ---------------------------------------------------------------------------
# render-template (Markdown projection of the extraction template)
# ---------------------------------------------------------------------------

def _cmd_render_template(args):
    """Render the config bundle's extraction template to a Markdown file.

    A read-only projection: it loads the template plus the reference-list
    display labels (not the prompts or pipeline config), renders the requested
    view, and writes it to `--out`. Strict inputs: a missing config directory,
    a missing extraction template, a template that fails to parse, or a
    malformed reference-list `label:` all fail loudly with exit 1 and no file
    written. There is no silent fallback.
    """
    config_dir = Path(args.config)
    if not config_dir.exists():
        print(f"config bundle directory does not exist: {config_dir}",
              file=sys.stderr)
        return 1
    if not config_dir.is_dir():
        print(f"config bundle path is not a directory: {config_dir}",
              file=sys.stderr)
        return 1
    template_path = config_dir / "extraction_template.yaml"
    if not template_path.is_file():
        print(f"config bundle is missing extraction_template.yaml: "
              f"{template_path}", file=sys.stderr)
        return 1

    # load_template raises ValueError on a template that violates the model and
    # a yaml.YAMLError (its base is yaml.error.YAMLError) on malformed YAML;
    # OSError covers a read fault. Any of them is a loud, clean load failure.
    try:
        template = load_template(template_path)
    except (ValueError, OSError, yaml.YAMLError) as e:
        print(f"could not load extraction template {template_path}: {e}",
              file=sys.stderr)
        return 1

    # Presentation-only display labels for the reference lists (used to phrase
    # canonical-reference value domains). A missing reference/ dir yields no
    # labels; a malformed `label:` fails loudly with nothing written.
    try:
        labels = load_reference_list_labels(config_dir / "reference")
    except ConfigBundleError as e:
        print(f"could not load reference-list labels: {e}", file=sys.stderr)
        return 1

    document = render_template(template, args.view, labels)

    out_path = Path(args.out)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    print(f"Wrote {args.view} view to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    load_dotenv()
    args = _parse_args(argv)
    if args.command == "validate-bundle":
        return sys.exit(_cmd_validate_bundle(args))
    if args.command == "validate":
        return sys.exit(_cmd_validate(args))
    if args.command == "fingerprint":
        return sys.exit(_cmd_fingerprint(args))
    if args.command == "render-template":
        return sys.exit(_cmd_render_template(args))
    if args.command == "transcript":
        return sys.exit(_cmd_transcript(args))
    if args.command == "extract":
        return sys.exit(_cmd_extract(args))
    # argparse `required=True` on the subparser makes this unreachable, but
    # the fallback is explicit rather than implied.
    print("No command given.", file=sys.stderr)
    return sys.exit(2)


if __name__ == "__main__":
    main()
