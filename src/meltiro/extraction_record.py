"""Mutable extraction output.

The extraction output is the live state that the extractor mutates via tool calls.
It owns:

  - the four canonical blocks (initial_check, study, records,
    quality_check),
  - the auto-assigned record IDs (`<entity>_<n>` in call order, where
    `<entity>` is the record entity the template declares),
  - the `mark_complete` flag (set when the extractor declares completion;
    cleared when the checker challenges a field),

Schema convention:
  - `study` and each entry in `records` are dicts of
    `variable -> {value, evidence, notes}` envelopes plus an implicit
    `record_id` for records.
  - `study` and each record also carry the reserved `notes` key (see
    `NOTES_KEY`): the scope note, a free-text string or null holding the
    extractor's commentary about that whole scope. It is not a field: no
    template may declare a field named `notes`, it is never validated, and
    the checker never sees it.
  - `initial_check` and `quality_check` are keyed by the ROLE that recorded
    them (`{role: {variable: value}}`); each role's block is a flat
    `variable -> value` dict (no envelopes; they describe the extraction
    process, not the paper).

Provenance of the check blocks
------------------------------
Both check blocks are self-assessments, so their author is part of the datum:
both are keyed by role at the top level,

    "initial_check": {"extractor": {...}},
    "quality_check": {"extractor": {...}, "review": {...}}

The extractor's answers are recorded once and are never editable afterwards —
no tool in the reviewer's catalogue accepts a check-block argument — and the
reviewer records its OWN quality check under its own key. Only the extractor
records an initial check (a pre-extraction act with no review-stage
equivalent); the block is still role-keyed so the two shapes stay uniform.

Role keys are a closed engine-owned vocabulary (`ROLES`); template field
variables live one level deeper, so a template may declare a field named
`extractor` without ambiguity.
"""

import copy


# The roles that may record a check block. Engine-owned and closed: these are
# the same role names the run record uses (`<role>_model_resolved`, the
# checker's `stage=`), so one vocabulary spans the run's whole provenance.
ROLE_EXTRACTOR = "extractor"
ROLE_REVIEW = "review"
ROLES = (ROLE_EXTRACTOR, ROLE_REVIEW)


# The reserved scope-note key, inside the `study` block and every record
# object: a free-text string or null. `meltiro.template` rejects a field
# whose variable is `notes`, so the key can never collide with a field.
# Every walk over the study block or a record must skip it rather than
# mistake it for a field envelope.
NOTES_KEY = "notes"

# Sentinel for "the caller supplied no note at all", distinct from an explicit
# `None` (which clears the scope note). Without it a call that never mentioned
# notes would silently wipe one written earlier.
_UNSET = object()


class ExtractionRecord:
    """Mutable extraction state. Lives in memory for one session."""

    def __init__(self):
        # Both check blocks are keyed by the role that recorded them (see the
        # module docstring). A role's key appears only once that role has
        # actually recorded something, so an absent key reads as "this role
        # said nothing" rather than as an empty answer.
        self.initial_check = {}
        # The study block always carries the reserved `notes` key, so the
        # scope note is a stable part of the shape rather than a key that
        # appears once something writes it.
        self.study = {NOTES_KEY: None}
        self.records = []  # list of dicts, each with record_id and notes
        self.quality_check = {}
        # Whether the extractor has recorded its initial check yet — the
        # gate's state, NOT a derived property: a template declaring no
        # initial-check fields leaves `initial_check[extractor]` empty even
        # after a successful call, and deriving the gate from emptiness would
        # deadlock such a bundle forever. Session bookkeeping, persisted in
        # run.json and threaded back on resume; never part of to_dict().
        self.initial_check_recorded = False
        # Next record index per entity. Monotonic and never reused: a
        # removed record's index is never reissued (see from_dict for why).
        # Session bookkeeping, not part of to_dict(): serialised into
        # run.json and threaded back on resume via set_record_id_counters.
        self._record_id_counters = {}
        self.mark_complete_flag = False
        # Deliberate-surrender latch. Set by `abandon` when the extractor
        # declares it cannot produce a valid extraction honestly; the
        # orchestrator finalises such a run as `failed_validation`. Held in
        # memory only (not serialised): a surrender ends the run, so it never
        # needs to survive a resume.
        self.abandoned_flag = False
        self.abandon_reason = None

    # ----------------------------------------------------------------------
    # Mutations
    # ----------------------------------------------------------------------

    def record_initial_check(self, fields, role=ROLE_EXTRACTOR):
        """Record one role's initial check and latch the ordering gate.

        Shallow-merged into that role's block, so a re-call revises rather
        than replaces. The gate flag is set even when `fields` is empty: the
        call happened, which is what the gate asks about. Unlike a field
        write this does NOT clear `mark_complete_flag` — the initial check
        describes the inputs, not the extraction.

        Returns the list of variables applied.
        """
        applied = self._merge_check_block(self.initial_check, fields, role)
        if role == ROLE_EXTRACTOR:
            self.initial_check_recorded = True
        return applied

    def record_quality_check(self, fields, role):
        """Record one role's quality check, shallow-merged into its own block.

        Each role writes under its own key, so "who said this" is structural
        rather than a convention a caller has to maintain.

        Returns the list of variables applied.
        """
        return self._merge_check_block(self.quality_check, fields, role)

    @staticmethod
    def _merge_check_block(block, fields, role):
        """Shallow-merge one role's answers into a role-keyed check block.

        The role's key is minted only when there is something to put under
        it, so an absent key always means "this role recorded nothing" and
        `{"extractor": {}}` never appears.
        """
        if not fields:
            return []
        target = block.setdefault(role, {})
        applied = []
        for k, v in fields.items():
            target[k] = v
            applied.append(k)
        return applied

    def initial_check_for(self, role=ROLE_EXTRACTOR):
        """One role's initial-check block, or {} when it recorded none."""
        return self.initial_check.get(role, {})

    def quality_check_for(self, role):
        """One role's quality-check block, or {} when it recorded none."""
        return self.quality_check.get(role, {})

    def apply_update_study(self, study=None, notes=_UNSET):
        """Apply a dict of updates to the study-level state.

        Study FIELDS only: the check blocks are not writable here by either
        role — each has one path and one author — so a reviewer revising a
        study field can never overwrite the extractor's account of its run in
        the same call.

        Shallow-merged; keys not included are left unchanged, and setting an
        existing key to None explicitly clears it.

        `notes` is the study scope note: omit to leave it untouched, pass a
        string to write, None to clear. Writing a scope note does NOT clear
        the mark_complete flag — the flag forces re-declaration on checkable
        content, and a scope note is never checked.

        Returns a dict describing what changed (used by the dispatcher to
        build the tool_result payload); `notes_written` is present and True
        only when this call wrote the scope note.
        """
        applied = {"study_fields": []}

        if study:
            for k, env in study.items():
                self.study[k] = env
                applied["study_fields"].append(k)

        # Any FIELD change clears the mark_complete flag; extractor must
        # explicitly re-declare. Checked before the scope note is written so
        # a note-only call leaves the flag alone.
        if any(applied.values()):
            self.mark_complete_flag = False

        if notes is not _UNSET:
            self.study[NOTES_KEY] = notes
            applied["notes_written"] = True

        return applied

    def add_record(self, fields, entity, notes=_UNSET):
        """Append a record. Auto-assigns the next `<entity>_<n>` ID.

        `entity` is the record entity noun the template declares (e.g.
        "relationship"); it prefixes the id and keys the per-entity counter,
        which never reuses an index even across a remove, so a challenge
        recorded against one id can never be re-pointed at an unrelated later
        record. Raises ValueError unless `entity` is a non-empty string: a
        silent fallback would mint ambiguous ids.

        Every record is minted with the reserved `notes` key alongside its
        `record_id`; `notes` writes it at creation, else it starts null.

        Returns the new record_id.
        """
        if not isinstance(entity, str) or not entity:
            raise ValueError(
                f"add_record requires a non-empty entity name, got "
                f"{entity!r}."
            )
        n = self._record_id_counters.get(entity, 1)
        record_id = f"{entity}_{n}"
        self._record_id_counters[entity] = n + 1
        record = {"record_id": record_id,
                  NOTES_KEY: None if notes is _UNSET else notes}
        for k, env in fields.items():
            record[k] = env
        self.records.append(record)
        self.mark_complete_flag = False
        return record_id

    def update_record(self, record_id, fields, notes=_UNSET):
        """Revise fields on an existing record.

        `notes` is the record's scope note: omit it to leave the stored note
        untouched, pass a string to write one, or None to clear it. As with
        `apply_update_study`, writing a scope note does not clear the
        mark_complete flag, because the note is never checked.

        Returns the list of FIELD names applied; the scope note is not one of
        them (it is not a field).

        Raises KeyError if the ID is unknown.
        """
        record = self._find_record(record_id)
        if record is None:
            raise KeyError(f"Unknown record_id: {record_id}")
        applied = []
        for k, env in fields.items():
            record[k] = env
            applied.append(k)
        if applied:
            self.mark_complete_flag = False
        if notes is not _UNSET:
            record[NOTES_KEY] = notes
        return applied

    def remove_record(self, record_id):
        """Hard-delete a record. Other records are not renumbered.

        Raises KeyError if the ID is unknown.
        """
        for i, record in enumerate(self.records):
            if record.get("record_id") == record_id:
                del self.records[i]
                self.mark_complete_flag = False
                return
        raise KeyError(f"Unknown record_id: {record_id}")

    def mark_complete(self):
        """Set the mark_complete flag. The validator decides whether the
        extraction output is genuinely ready; this just records the model's claim."""
        self.mark_complete_flag = True

    def abandon(self, reason):
        """Latch a deliberate surrender with the extractor's stated reason.

        The extractor calls this (via the `abandon_extraction` tool) when it
        cannot produce a valid extraction honestly. The orchestrator detects
        the flag and finalises the run as `failed_validation`; the reason is
        recorded in run.json and the run log.
        """
        self.abandoned_flag = True
        self.abandon_reason = reason

    # ----------------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------------

    def _find_record(self, record_id):
        for record in self.records:
            if record.get("record_id") == record_id:
                return record
        return None

    def get_record(self, record_id):
        """Return a deep copy of a record (or None)."""
        record = self._find_record(record_id)
        return copy.deepcopy(record) if record else None

    def record_ids(self):
        return [r.get("record_id") for r in self.records]

    def has_record(self, record_id):
        return self._find_record(record_id) is not None

    # ----------------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------------

    def to_dict(self, include_checks=True):
        """Serialise to the canonical extraction output JSON shape.

        Both check blocks are role-keyed maps (see the module docstring).
        `include_checks=False` is the view handed to the final reviewer,
        which forms an independent second opinion: the blocks stay on disk
        and in the run record, they are simply not shown to the stage whose
        job is to disagree with them.
        """
        out = {
            "study": copy.deepcopy(self.study),
            "records": copy.deepcopy(self.records),
        }
        if include_checks:
            return {
                "initial_check": copy.deepcopy(self.initial_check),
                "study": out["study"],
                "records": out["records"],
                "quality_check": copy.deepcopy(self.quality_check),
            }
        return out

    @classmethod
    def from_dict(cls, d):
        """Reconstruct the four canonical blocks from a serialised extraction
        output dict (resume).

        Deliberately does NOT re-derive the record-id counter from the
        surviving records: max(index)+1 over survivors would reissue a
        removed record's id (add R1, R2, remove R2, resume, add -> R2 again),
        silently re-pointing any challenge held against the removed one. The
        counter is threaded back from run.json via set_record_id_counters by
        the caller (see Session).

        The reserved scope-note key is restored to null where the serialised
        block does not carry it, so the in-memory shape matches a freshly
        built record whatever produced the JSON.
        """
        a = cls()
        a.initial_check = _role_keyed(d.get("initial_check"))
        a.study = copy.deepcopy(d.get("study", {}))
        a.study.setdefault(NOTES_KEY, None)
        a.records = copy.deepcopy(d.get("records", []))
        for record in a.records:
            if isinstance(record, dict):
                record.setdefault(NOTES_KEY, None)
        a.quality_check = _role_keyed(d.get("quality_check"))
        # `initial_check_recorded` is NOT derived here: an extractor block that
        # is present but empty is a real state (a template declaring no
        # initial-check fields), and deriving the gate from it would reopen the
        # gate on resume. The caller threads the flag back from run.json via
        # `set_initial_check_recorded`, exactly as it does the id counters.
        return a

    def record_id_counters(self):
        """The per-entity next-index bookkeeping, as a plain dict copy.

        Session persists this into run.json alongside every extraction-output
        write so the counter survives a pause. It is NOT part of to_dict (the
        consumer-facing extraction output stays exactly the four canonical
        blocks).
        """
        return dict(self._record_id_counters)

    def set_initial_check_recorded(self, recorded):
        """Restore the ordering gate's state (from run.json on resume).

        Without this a resumed extractor would be told to record its initial
        check again, and would either comply (overwriting a real answer with a
        post-hoc one) or wedge against the gate.
        """
        self.initial_check_recorded = bool(recorded)

    def set_record_id_counters(self, counters):
        """Load the per-entity next-index bookkeeping (from run.json on resume).

        Replaces the in-memory counters wholesale. A missing or empty mapping
        leaves every entity starting at 1, which is correct for a fresh
        session (no ids minted yet).
        """
        self._record_id_counters = dict(counters or {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _role_keyed(block):
    """Normalise a serialised check block to the role-keyed shape.

    Keeps only entries whose value is itself a mapping, so a hand-built flat
    block (`{variable: value}`) normalises to `{}` — an honest "no role
    recorded anything" — rather than being re-nested under a fabricated role
    or silently misread by the accessors.
    """
    return {role: copy.deepcopy(fields)
            for role, fields in (block or {}).items()
            if isinstance(fields, dict)}
