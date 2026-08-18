"""Canonical run-status taxonomy for extraction runs.

One vocabulary shared by session state, the orchestrator's provenance flag,
CLI exit codes, and the run log. Each status answers one question: what
should a consumer do with this session?

* ``in_progress``: resume it (it is live, or paused). A paused session records
  a ``pause_reason`` in run.json and is the only status ``Session.resume``
  accepts. Two things pause. ``tool_cap_hit`` is a budget the operator set,
  reached; raise the cap and resume. ``provider_account`` is the provider
  refusing over WHO IS ASKING rather than what was asked — an exhausted
  balance or spend cap, a key absent, revoked, or not entitled to the model —
  where the extraction is untouched and the fix is outside the process; fix
  the account and resume. Both leave the same resumable session, and the CLI
  distinguishes them only in what it exits: a cap pause is the command doing
  what it was asked, an account pause is not (see ``cli._command_status``).
* ``complete``: trust it. The extraction is the canonical, usable answer.
* ``failed_validation``: the extraction genuinely failed on this paper; do
  not resume, investigate. ``meta.failure_reason`` records the mechanism
  (``surrendered`` / ``stalled`` / ``text_only_stall`` from the extractor;
  ``review_surrendered`` / ``review_cap_hit`` / ``review_stalled`` /
  ``review_text_only_stall`` from the final reviewer).
* ``error``: an infrastructure or config problem; fix and re-run.

There is no adjudication status: the checker's challenges are advisory and
none blocks ``mark_complete``, so a field still challenged when its check
budget runs out is recorded in ``meta.checker_diagnostics`` and the run
finalises ``complete``.

Exit codes are ``meltiro.cli``'s own mapping, documented there and in the
README.
"""

# Every legal value of ``meta.status``. ``in_progress`` is the live/paused
# state; the other three are terminal (a ``Session.finalise`` target).
RUN_STATUSES = frozenset({
    "in_progress",
    "complete",
    "failed_validation",
    "error",
})

# Terminal statuses: a finished run finalises into exactly one of these.
# ``in_progress`` is never a finalise target (a paused run simply stays
# in_progress without finalising).
TERMINAL_STATUSES = RUN_STATUSES - frozenset({"in_progress"})

# The run whose extraction is the canonical, usable answer: the only status
# admitted into accuracy aggregation (the run log's ``validation_passed``
# flag).
VALIDATED_STATUSES = frozenset({"complete"})

# The runs that reached an answer they CONSIDERED final. The other two
# terminal statuses are aborts that still ship whatever the extraction
# record held — a work-in-progress snapshot. A check that faults the
# extraction ITSELF should skip them; one recording what happened applies
# to every terminal status.
CONSIDERED_STATUSES = frozenset({"complete"})
