"""Per-field history, derived from the session event log.

`diagnostics/field_history.json` answers two questions a flat event log makes
expensive: how did each value come to be, and are the checker and the reviewer
earning their cost. It is DERIVED, wholly and only, from
`diagnostics/tool_calls.jsonl`, so it is a convenience shape and never a
second source of truth. `build_field_history` is a pure function of the parsed
event list: a rebuild from the log alone, at any time, produces the same
bytes.

One entry per field path, each carrying an ordered list of what happened to
that field plus its final state, and one small run-level aggregate.

## Where each event kind comes from

Every dispatched tool call is logged as one event (`tool_call_applied`,
`tool_call_partial`, `tool_call_failed` from the extractor,
`review_tool_call` from the final reviewer), carrying the dispatcher's whole
result. Four keys on that result are all this module reads:

- `_field_diffs`, a `{path: {before, after}}` map of the fields the call
  actually wrote;
- `failed_fields`, a `{path: [error, ...]}` map of the fields it proposed that
  validation rejected;
- `applied_changes.removed_record_id`, present only for a `remove_record`
  call;
- `_checker_verdicts`, the per-field record of what the checker judged,
  merged onto the result before the event was appended.

From those:

`proposed`
    The field appeared in this call, in either map. It records the attempt,
    and is always followed by the attempt's outcome. The proposed VALUE is not
    recorded here: for a write that landed it is the outcome event's `after`,
    and for a rejection the dispatcher records only its error messages about
    the value (the call's verbatim `args` are in `tool_calls.jsonl`).

`rejected`
    The path is in `failed_fields`. Carries every error the dispatcher
    returned for it, each with its `code`.

`applied`
    The path is in `_field_diffs`, and none of the more specific write
    outcomes below apply. Carries `before`, `after`, and whether the value
    changed.

`challenged`, `checked`, `check_error`
    One per entry in `_checker_verdicts`, in the order the log records them.
    `checked` is a clean `ok`; `challenged` is a genuine objection;
    `check_error` is a verdict whose `error_origin` is set, an exhausted-retry
    API failure degraded to a challenge, which is an absence of information
    rather than an objection and so is named apart from it. Each carries what
    the checker actually scored: the value, the evidence, and the field's own
    extractor note (`note_checked`), because the note is part of what the
    checker saw.

`revised_after_challenge`
    A later write to a field carrying an OUTSTANDING challenge that CHANGED
    its value. The writer answered the checker.

`overruled`
    A later write to a field carrying an outstanding challenge that left the
    value unchanged. An inference, not something the run logs explicitly: a
    writer re-submitting a value it has just been challenged on is standing
    by it. A field challenged and never written again is NOT overruled —
    nobody acted, so no event is emitted, and `final.unresolved_challenge`
    says the challenge still stands.

`revised_by_reviewer`
    A write from the review stage (a `review_tool_call` event) that changed
    the value. The stage is read off the event name, not off the result.

`removed`
    A `remove_record` call reports `applied_changes.removed_record_id`, and no
    field diffs: the record and every field on it are gone. One `removed`
    event is emitted for each already-tracked `record.<id>.*` path, carrying
    the value the field held when it went.

A field carrying an outstanding challenge is the only state this module
threads across events, and it is set by `challenged` and cleared by the next
write to that field or by that field's removal. Where two rules could name one
write, they are applied in this order: `overruled` (an unchanged value under
an outstanding challenge) first, then `revised_by_reviewer` (a changed value
from the review stage), then `revised_after_challenge`, then plain `applied`.
A write that answered a challenge also carries `answers_challenge: true`, so
a reviewer edit that happens to answer a challenge records both facts on one
event rather than having to pick a name.
"""

# Event names that carry a dispatched tool call's result, and the stage each
# belongs to. The stage is read off the event NAME: the extractor's dispatches
# and the reviewer's are separate events precisely so a consumer never has to
# infer which loop wrote a field.
DISPATCH_EVENTS = {
    "tool_call_applied": "extractor",
    "tool_call_partial": "extractor",
    "tool_call_failed": "extractor",
    "review_tool_call": "review",
}


def build_field_history(events):
    """Build the field-history document from a session's parsed event list.

    Pure: no IO, no clock, no randomness, and nothing read that is not in
    `events`. That is what makes `field_history.json` regenerable from
    `tool_calls.jsonl` alone, byte for byte.

    Returns `{"aggregate": {...}, "fields": {path: {"events": [...],
    "final": {...}}}}`. Field paths appear in first-touch order, and each
    field's events in log order.
    """
    fields = {}
    state = {}

    def _field(path):
        if path not in fields:
            fields[path] = {"events": [], "final": {}}
            state[path] = {
                "outstanding_challenge": False,
                "present": True,
                "value": None,
                "writes": 0,
                "rejections": 0,
                "checks": 0,
                "reviewer_touched": False,
                "last_write_stage": None,
                "last_verdict": None,
                "last_verdict_error_origin": False,
            }
        return fields[path], state[path]

    checker_cost = 0.0
    # Set by the first verdict that carries no cost figure. The aggregate below
    # then states none either: a sum over the verdicts that happened to be
    # priced would read as what the checker cost, while covering only part of
    # it.
    checker_cost_unpriced = False
    # Set by the first verdict whose own figure covered fewer calls than the
    # check made, and summed over them: a check is re-asked when its first
    # reply records no verdict, so one check can leave more than one charge
    # unread. The aggregate is then a floor over the checking, on the same
    # terms the run's total is a floor over the run.
    checker_cost_incomplete = False
    checker_unreceipted = 0
    challenges_raised = 0
    challenges_revised = 0
    challenges_overruled = 0
    check_errors = 0

    for ev in events:
        stage = DISPATCH_EVENTS.get(ev.get("event"))
        if stage is None:
            continue
        result = ev.get("result")
        if not isinstance(result, dict):
            continue
        base = {
            "ts": ev.get("ts"),
            "turn_id": ev.get("turn_id"),
            "stage": stage,
            "tool": ev.get("tool"),
        }
        diffs = result.get("_field_diffs") or {}
        failed = result.get("failed_fields") or {}
        verdicts = result.get("_checker_verdicts") or {}

        # Writes first, then rejections: a partial call applied its valid
        # subset before reporting the rest, and the ordering within one call
        # is presentational either way (they are one dispatch).
        for path, diff in diffs.items():
            entry, st = _field(path)
            before = diff.get("before")
            after = diff.get("after")
            changed = before != after
            entry["events"].append({"kind": "proposed", **base})
            kind, resolves = _write_kind(st, stage, changed)
            written = {"kind": kind, **base,
                       "before": before, "after": after, "changed": changed}
            if resolves:
                written["answers_challenge"] = True
            entry["events"].append(written)
            st["outstanding_challenge"] = False
            st["present"] = True
            st["value"] = after
            st["writes"] += 1
            st["last_write_stage"] = stage
            if stage == "review":
                st["reviewer_touched"] = True
            if resolves and changed:
                challenges_revised += 1
            if kind == "overruled":
                challenges_overruled += 1

        for path, errors in failed.items():
            entry, st = _field(path)
            entry["events"].append({"kind": "proposed", **base})
            entry["events"].append({
                "kind": "rejected", **base,
                "errors": [
                    {"code": e.get("code"), "message": e.get("message")}
                    for e in (errors or []) if isinstance(e, dict)
                ],
            })
            st["rejections"] += 1

        # A removed record takes every field on it. `remove_record` writes no
        # field diffs, so this is the only signal that those fields are gone.
        removed_id = (result.get("applied_changes") or {}).get(
            "removed_record_id")
        if removed_id:
            prefix = f"record.{removed_id}."
            for path in list(fields):
                if not path.startswith(prefix):
                    continue
                if not state[path]["present"]:
                    continue
                st = state[path]
                fields[path]["events"].append({
                    "kind": "removed", **base,
                    "record_id": removed_id,
                    "value_before": st["value"],
                })
                st["present"] = False
                st["value"] = None
                st["outstanding_challenge"] = False
                if stage == "review":
                    st["reviewer_touched"] = True

        # The checker runs inside the dispatch, after the write, so its
        # verdicts follow the write events for the same call.
        for path, verdict in verdicts.items():
            entry, st = _field(path)
            error_origin = bool(verdict.get("error_origin"))
            judgement = verdict.get("verdict")
            if error_origin:
                kind = "check_error"
            elif judgement == "challenge":
                kind = "challenged"
            else:
                kind = "checked"
            entry["events"].append({
                "kind": kind, **base,
                "verdict": judgement,
                "rationale": verdict.get("rationale"),
                "checker_notes": verdict.get("notes"),
                "value_checked": verdict.get("value_checked"),
                "evidence_checked": verdict.get("evidence_checked"),
                "note_checked": verdict.get("note_checked"),
                "error_origin": error_origin,
                "cost_usd": verdict.get("cost_usd"),
            })
            st["checks"] += 1
            st["last_verdict"] = judgement
            st["last_verdict_error_origin"] = error_origin
            if kind == "challenged":
                st["outstanding_challenge"] = True
                challenges_raised += 1
            elif kind == "check_error":
                check_errors += 1
            else:
                st["outstanding_challenge"] = False
            cost = verdict.get("cost_usd")
            if cost is None:
                checker_cost_unpriced = True
            else:
                checker_cost += float(cost)
            if verdict.get("cost_incomplete"):
                checker_cost_incomplete = True
                checker_unreceipted += int(
                    verdict.get("unreceipted_responses") or 0)

    for path, entry in fields.items():
        st = state[path]
        # `unresolved_challenge` is deliberately the same question
        # `run.checker_diagnostics.unresolved_challenges` asks: was the LAST
        # verdict this field received a genuine challenge? A field revised
        # after a challenge but never re-checked (its per-field budget ran
        # out) still reads true here, which is honest: the checker never
        # signed the new value off. The event list is where a reader sees
        # that somebody did answer.
        entry["final"] = {
            "present": st["present"],
            "value": st["value"],
            "writes": st["writes"],
            "rejections": st["rejections"],
            "last_write_stage": st["last_write_stage"],
            "checks": st["checks"],
            "last_verdict": st["last_verdict"],
            "last_verdict_error_origin": st["last_verdict_error_origin"],
            "unresolved_challenge": (
                st["last_verdict"] == "challenge"
                and not st["last_verdict_error_origin"]),
            "reviewer_touched": st["reviewer_touched"],
        }

    aggregate = {
        "fields_written": sum(
            1 for e in fields.values() if e["final"]["writes"]),
        "fields_checked": sum(
            1 for e in fields.values() if e["final"]["checks"]),
        "fields_reviewer_touched": sum(
            1 for e in fields.values() if e["final"]["reviewer_touched"]),
        "fields_with_unresolved_challenge": sum(
            1 for e in fields.values()
            if e["final"]["unresolved_challenge"]),
        "challenges_raised": challenges_raised,
        "challenges_revised": challenges_revised,
        "challenges_overruled": challenges_overruled,
        "check_errors": check_errors,
        "checker_cost_usd": (
            None if checker_cost_unpriced else round(checker_cost, 6)),
        # Carried only when a charge actually went unread, and then saying how
        # many calls the figure above leaves out. Present exactly where a
        # reader must not take that figure for the whole of what the checker
        # cost; an ordinary run records no flag saying nothing went wrong.
        **({"checker_cost_incomplete": True,
            "checker_unreceipted_responses": checker_unreceipted}
           if checker_cost_incomplete else {}),
    }
    return {"aggregate": aggregate, "fields": fields}


def _write_kind(st, stage, changed):
    """Name one applied write, and say whether it answered a challenge.

    Returns `(kind, answers_challenge)`. The rules are applied in the order
    documented at the top of this module: an unchanged value under an
    outstanding challenge is `overruled` whichever stage wrote it (the writer
    stood by the value), then a changed value from the reviewer is
    `revised_by_reviewer`, then a changed value answering a challenge is
    `revised_after_challenge`, and anything else is a plain `applied`.
    """
    outstanding = st["outstanding_challenge"]
    if outstanding and not changed:
        return "overruled", True
    if stage == "review" and changed:
        return "revised_by_reviewer", outstanding
    if outstanding and changed:
        return "revised_after_challenge", True
    return "applied", False
