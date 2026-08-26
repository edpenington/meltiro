"""Build and append a session-level entry to the run log (`run_log.json`
at the run root).

The agentic pipeline produces one run-log entry per finished session.
Extra fields are added under `agentic: true` so consumers can branch
when relevant.
"""

from meltiro import __version__
from meltiro.run_log import alteksto_version, append_run, direktoro_version


def build_entry(session, *, input_tokens=0, output_tokens=0,
                cache_creation_tokens=0, cache_read_tokens=0,
                cost_usd=None, cost_rates=None, usage_by_role=None,
                validation_passed=True, validation_errors=None):
    """Build the run-log entry for a finished session.

    `session_dir` and `result_file` are recorded as absolute paths: the run
    log is a cross-run index parsed by a consumer that never sees the cwd the
    run was invoked from. The Session resolves its directory at construction
    (see `Session.__init__`), so the paths read here are already absolute
    even under a relative `--out` or `--resume`.

    Args:
        session: meltiro.session.Session.
        input_tokens: regular (cache-miss) input tokens, summed across
            every API call in the session.
        output_tokens: output tokens.
        cache_creation_tokens / cache_read_tokens: prompt-cache write
            and read counters, persisted separately from `cost_usd` so a
            consumer can report cache savings on their own.
        cost_usd: aggregate USD across every call, or None when the run
            states no figure. None is never rendered as zero — an unpriced
            run was not a free one (see `rates.py`) — and the token counters
            above are the durable record of what it spent. A run whose
            `run.json` marks the total as covering fewer calls than it made
            carries that here too (`cost_incomplete`, `unreceipted_calls`),
            because the figure is then a floor.
        cost_rates: `{role: card}`, the rate card each role's calls were
            costed at (`meltiro.rates.Rates.as_record()`, which carries the
            card's own provenance), null for a role that had none. Recorded
            beside the figures so a reader recomputes them from the tokens
            in this same entry rather than against current prices.
        usage_by_role: `{role: {input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, cost_usd, cost_rates}}`,
            the same meters as the run-wide fields above split by role. The
            run-wide totals are sums over these and nothing else. A role's
            `cost_usd` is null on exactly the terms the run's is: unpriced,
            never zero. A role whose figure covers fewer calls than it made
            carries the same pair the run does (`cost_incomplete`,
            `unreceipted_calls`), so summing these blocks rebuilds the run's
            floor as a floor rather than as a total.
        validation_passed: whether the run reached the status whose extraction
            is the canonical, usable answer (`status in VALIDATED_STATUSES`).
            A record of HOW THE RUN ENDED rather than a verdict on the shipped
            bytes: no field VALUE is re-validated at finalisation, so a status
            of `complete` says the loop reached completion, not that every
            value would pass `validators.validate_extraction_output` today.
            One check IS re-run over the shipped output —
            `validators.missing_required_fields`, which is what raises the
            `required-fields-null` warning
            (`Orchestrator._check_shipped_required_fields`) — so read
            `meta.warnings` for what the run stands behind its answer despite.
            For an independent verdict on the values, run
            `validators.validate_extraction_output` over the shipped file.
        validation_errors: list of string error summaries to include in
            the log entry.
    """
    meta = session.meta
    return {
        "study_id": meta["study_id"],
        "result_file": str(session.extraction_record_path),
        "session_dir": str(session.session_dir),
        "prompt_hash": meta.get("prompt_hash"),
        "template_hash": meta.get("template_hash"),
        "model": meta.get("extractor_model"),
        "checker_model": meta.get("checker_model"),
        "review_model": meta.get("review_model"),
        "config_fp": meta.get("config_fp"),
        "checker_fp": meta.get("checker_fp"),
        "review_fp": meta.get("review_fp"),
        # The three orthogonal axes, copied from run.json so the run log can
        # answer the comparison questions (instrument A/B, model swap, engine
        # diff) without opening every session directory. See fingerprint.py.
        "instrument_fp": meta.get("instrument_fp"),
        "extractor_call_fp": meta.get("extractor_call_fp"),
        "checker_call_fp": meta.get("checker_call_fp"),
        "review_call_fp": meta.get("review_call_fp"),
        "engine_fp": meta.get("engine_fp"),
        # The whole-run identity (extractor + checker + reviewer + engine),
        # what a consumer keys a run on rather than config_fp alone; producer
        # strings are built as `llm:<run_fp>`. See fingerprint.run_fingerprint.
        "run_fp": meta.get("run_fp"),
        # The PAPER, folded into none of the fingerprints above: `run_fp`
        # says what was asked, `bundle_fp` what it was asked of. See
        # fingerprint.bundle_fingerprint.
        "bundle_fp": meta.get("bundle_fp"),
        "text_fp": meta.get("text_fp"),
        "figures_fp": meta.get("figures_fp"),
        "manifest_fp": meta.get("manifest_fp"),
        "tables_fp": meta.get("tables_fp"),
        "tool_set_hash": meta.get("tool_set_hash"),
        "status": meta.get("status"),
        "agentic": True,
        "tool_call_count": meta.get("tool_call_count", 0),
        "checker_calls_run": meta.get("checker_calls_run", 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cost_usd": cost_usd,
        # Carried only when a charge could not be read, and then saying how
        # many calls the figure above does not cover. This is a cross-run
        # index, read by a consumer summing many runs into one bill, so a
        # floor that arrived here dressed as a total would be added up as one.
        **({"cost_incomplete": True,
            "unreceipted_calls": meta.get("unreceipted_calls")}
           if meta.get("cost_incomplete") else {}),
        "cost_rates": cost_rates or {},
        "usage_by_role": usage_by_role or {},
        "validation_passed": validation_passed,
        "validation_errors": validation_errors or [],
        # The readable half of the engine identity, next to the git anchor
        # `append_run` adds below it; `engine_fp` above is the value that
        # identifies the code. `direktoro_version` is null only when the
        # package is absent (an install that places no calls at all), and
        # `alteksto_version` names what admitted the paper this row is about,
        # which no fingerprint here carries.
        "meltiro_version": __version__,
        "direktoro_version": direktoro_version(),
        "alteksto_version": alteksto_version(),
    }


def append_session_entry(session, *, log_dir, **kwargs):
    """Append a session entry to the run log (`log_dir/run_log.json`).

    `log_dir` is REQUIRED; it is the run root supplied by the caller.

    Returns the appended entry dict.
    """
    entry = build_entry(session, **kwargs)
    append_run(entry, log_dir=log_dir)
    return entry
