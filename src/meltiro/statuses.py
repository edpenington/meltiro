"""Canonical run-status taxonomy for extraction runs.

One vocabulary shared by session state, the orchestrator's provenance flag,
CLI exit codes, and the run log. Each status answers one question: what
should a consumer do with this session?

* ``in_progress``: resume it (it is live, or paused on the tool-call cap). A
  paused session records a ``pause_reason`` in run.json and is the only
  status ``Session.resume`` accepts.
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
