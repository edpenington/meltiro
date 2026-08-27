"""Deterministic per-field validators.

These run on every tool call before any change is applied to the extraction output.
The dispatcher decides what to do with the results (per field: a field whose
envelope validates is written even when a sibling field in the same call
fails, and only the failed fields are reported back for revision).

Two kinds of check live here:

  - validate_envelope: type + structure + verbatim-quote check on one
    field's `{value, evidence, notes}` envelope. It also canonicalises a
    categorical value to an option's exact spelling on store when the
    value matches only by case/whitespace. The `notes` slot is checked for
    shape only (a string or null): a note is the extractor's own commentary,
    never a claim, so it is never quote-checked and never counts toward
    satisfying `evidence: required`.
  - validate_gate_rules: cross-field checks on one record, driven by the
    template's declared `gates` (a gated field is only expected under certain
    values of a controlling field). These produce WARNINGS, not errors,
    because the deterministic check can't be sure (the model may have a
    legitimate reason); the checker will catch the substantive cases. The
    engine names no field itself: every gate's field names come from the
    template.

Categorical fields come in two shapes: a hard enum (value must match one
of the `options` exactly) and an `allow_other` field (the options are
typical values, but any non-empty free-text string is accepted). Fields
carrying `canonical_reference` are validated against a reference list;
that logic lives in `_validate_reference_value` and is reached via
`validate_envelope` when a reference index is supplied.

The verbatim-quote check is delegated to meltiro.quote_check.
"""

import re
from dataclasses import dataclass

from meltiro.bundle import normalise_label
from meltiro.extraction_record import NOTES_KEY
from meltiro.quote_check import validate_evidence
from meltiro.reference_lists import resolve_reference_value


# No required-field allowlist lives here. Required-ness is declared in the
# template with a per-field `required: true` flag and enforced generically by
# the dispatcher (meltiro.tools._handle_mark_complete), so this module names
# no field of any particular review. The cross-field gate rules below carry no
# field names either: they apply whatever `gates` the template declares (see
# template._parse_gates), which is what keeps the engine template-agnostic.


# Year range accepted for any field typed `year`. Generous bounds: studies
# cite older work, and plausibility is not this validator's job.
_YEAR_MIN = 1800
_YEAR_MAX = 2100

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalise_option(s):
    """Case- and whitespace-insensitive normal form for enum-option
    matching: strip, lowercase, and collapse internal whitespace runs.

    This is the normalisation used to decide whether a categorical value
    canonicalises to an option's exact spelling. It is deliberately
    narrower than the verbatim-quote normaliser (no ligature/dash folding):
    option lists are short, hand-authored labels, and folding dashes would
    let, say, "N-A" match "N/A" spuriously.
    """
    return re.sub(r"\s+", " ", str(s).strip().lower())


def match_option(value, options):
    """Return the option `value` should canonicalise to, or None.

    An exact match wins. Otherwise, if `value` matches exactly one option
    after case/whitespace normalisation, that option's exact spelling is
    returned (so the stored value is canonicalised). No match, or an
    ambiguous match against several options, returns None.
    """
    if value in options:
        return value
    nv = _normalise_option(value)
    hits = [o for o in options if _normalise_option(o) == nv]
    if len(hits) == 1:
        return hits[0]
    return None


def _integral_year(value):
    """Return `value` as an int year, or None if it is not integral.

    A `year` field is an integer, but the API tool schema and some models
    round-trip it as a float (2019.0). An integral float is accepted and
    coerce it to int (2019.0 -> 2019). A non-integral float (2019.5) or a
    non-numeric value returns None. Booleans are rejected because bool is a
    subclass of int.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


# ---------------------------------------------------------------------------
# Type checks per field_type
# ---------------------------------------------------------------------------

def _check_value_type(value, field_spec, path):
    """Return a list of {path, code, message} errors for the value's type.

    Null is allowed for every type except boolean: a boolean field answers a
    yes/no question and null is neither, so a null boolean is a type error.
    This is enforced here, not only in the tool schema: direktoro's canonical
    tool contract says a boolean field in a tool call's input must not arrive
    null — not every wire catches it, so this validator enforces it, and
    without the check a null boolean would pass runtime validation and be
    stored. Categorical values must match one of the declared options exactly.
    """
    ft = field_spec["field_type"]
    errors = []

    def _err(code, msg):
        errors.append({"path": path, "code": code, "message": msg})

    if value is None:
        # Every type but boolean accepts null (an unextracted field).
        if ft == "boolean":
            _err("type_mismatch", "Expected boolean, got null.")
        return errors

    if ft == "boolean":
        if not isinstance(value, bool):
            _err("type_mismatch", f"Expected boolean, got {type(value).__name__}.")
    elif ft == "integer":
        # bool is a subclass of int; reject explicitly.
        if isinstance(value, bool) or not isinstance(value, int):
            _err("type_mismatch", f"Expected integer, got {type(value).__name__}.")
    elif ft in ("number",):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _err("type_mismatch",
                 f"Expected number, got {type(value).__name__}.")
    elif ft == "year":
        year = _integral_year(value)
        if year is None:
            if isinstance(value, float):
                _err("type_mismatch",
                     f"Expected year (integer), got the non-integral number "
                     f"{value}. Provide a four-digit calendar year.")
            else:
                _err("type_mismatch",
                     f"Expected year (integer), got {type(value).__name__}.")
        elif not (_YEAR_MIN <= year <= _YEAR_MAX):
            _err("year_out_of_range",
                 f"Year {year} outside the accepted range "
                 f"[{_YEAR_MIN}, {_YEAR_MAX}].")
    elif ft == "date":
        if not isinstance(value, str):
            _err("type_mismatch",
                 f"Expected date string YYYY-MM-DD, got {type(value).__name__}.")
        elif not _DATE_PATTERN.match(value):
            _err("date_format",
                 f"Date '{value}' is not in YYYY-MM-DD format.")
    elif ft == "categorical":
        opts = field_spec["options"] or []
        opts_str = " | ".join(opts)
        if field_spec.get("allow_other"):
            # allow_other: the options are typical values; any non-empty
            # string is accepted. A value that matches an option (exactly
            # or after case/whitespace normalisation) is canonicalised to
            # the option's spelling on store (see validate_envelope).
            if not isinstance(value, str):
                _err("type_mismatch",
                     f"Expected a string (typically one of {opts_str}, but "
                     f"free text is allowed when none fits). Got "
                     f"{type(value).__name__}.")
            elif not value.strip():
                _err("empty_value",
                     "Value must be a non-empty string (or null). Provide a "
                     f"value (typically one of {opts_str}) or free text.")
        elif not isinstance(value, str):
            # Hard enum.
            _err("type_mismatch",
                 f"Categorical field must be one of: {opts_str}. "
                 f"Got {type(value).__name__}.")
        elif match_option(value, opts) is None:
            # No exact or case/whitespace match. Keep the case hint so a
            # near-miss still points at the intended option.
            case_hint = ""
            lower = _normalise_option(value)
            for o in opts:
                if _normalise_option(o) == lower:
                    case_hint = f" Did you mean '{o}'?"
                    break
            _err("invalid_option",
                 f"Value '{value}' is not in the allowed list. Pick "
                 f"EXACTLY one of: {opts_str}.{case_hint} If none of these "
                 f"fits, set the value to null; do not invent a new "
                 f"option, and do not paraphrase.")
    elif ft == "string_list":
        if not isinstance(value, list) or not all(
            isinstance(x, str) for x in value
        ):
            _err("type_mismatch",
                 "Expected list of strings.")
    else:
        # string (free text), and any field_type not matched above:
        # any string is fine.
        if not isinstance(value, str):
            _err("type_mismatch",
                 f"Expected string, got {type(value).__name__}.")

    return errors


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------

def _validate_reference_value(value, field_spec, reference_index,
                              path_prefix):
    """Validate a `canonical_reference` field value against a reference index.

    Returns `(errors, canonical_value, canonicalisations)`.

    A `type: string` field carries one reference name; a `type: string_list`
    field carries a real JSON array of names, and every element is validated
    independently. There is no string splitting anywhere: the model submits the
    array, so a canonical name containing a comma or a semicolon is just a
    name, not a boundary this validator has to guess at.

    On success (no errors), `canonical_value` is the value to store: the
    canonical spelling for a string field, or the list of canonical
    spellings (element order preserved) for a string_list field.
    `canonicalisations` is a list of `{path, entered, stored}` dicts, one
    per element that matched via an ALIAS rather than an exact name; the
    dispatcher turns each into a tool_result note and the orchestrator into
    a `value_canonicalised` event. An exact-name match canonicalises
    spelling silently (no event).

    Per-element failure modes: an element that normalises to empty, an
    element matching more than one entry, an element off the list (with the
    closest canonical names ranked by fuzzy similarity), and two elements
    that canonicalise to the SAME entry (rejected as a duplicate, never
    silently deduped). An empty list is valid (nothing to validate);
    required-ness stays the job of the existing gates.
    """
    ref_name = field_spec.get("canonical_reference")
    if reference_index is None:
        return ([{
            "path": f"{path_prefix}.value",
            "code": "reference_unavailable",
            "message": (
                f"Field is validated against reference list {ref_name!r} but "
                f"no reference index was supplied. This is a config/wiring "
                f"error, not a model error."),
        }], value, [])

    is_list = isinstance(value, list)
    tokens = value if is_list else [value]

    errors = []
    stored = []
    canonicalisations = []
    seen = {}  # canonical name -> the element that first resolved to it
    for i, token in enumerate(tokens):
        err_path = (f"{path_prefix}.value[{i}]" if is_list
                    else f"{path_prefix}.value")
        result = resolve_reference_value(token, reference_index)
        status = result["status"]
        if status in ("canonical", "alias"):
            canonical = result["canonical"]
            if canonical in seen:
                errors.append({
                    "path": err_path,
                    "code": "duplicate_reference",
                    "message": (
                        f"'{token}' duplicates an earlier entry: "
                        f"'{seen[canonical]}' already resolves to "
                        f"'{canonical}'. List each reference entry at most "
                        f"once; remove the duplicate."),
                })
                continue
            seen[canonical] = token
            stored.append(canonical)
            if status == "alias":
                canonicalisations.append({
                    "path": path_prefix,
                    "entered": token,
                    "stored": canonical,
                })
        elif status == "ambiguous":
            cands = ", ".join(result["candidates"])
            errors.append({
                "path": err_path,
                "code": "ambiguous_reference",
                "message": (
                    f"'{token}' matches more than one reference-list entry: "
                    f"{cands}. Use the exact canonical name of the one you "
                    f"mean."),
            })
        elif status == "empty":
            errors.append({
                "path": err_path,
                "code": "empty_value",
                "message": (
                    f"Reference names must be non-empty; provide an exact "
                    f"name from the {ref_name} reference list or remove the "
                    f"entry."),
            })
        else:  # no_match
            suggestions = result.get("suggestions", [])
            hint = (" Closest names: " + ", ".join(suggestions) + "."
                    if suggestions else "")
            errors.append({
                "path": err_path,
                "code": "off_list_reference",
                "message": (
                    f"'{token}' is not in the {ref_name} reference list.{hint} "
                    f"Use an exact name from the list in your system prompt; "
                    f"do not invent or paraphrase a name."),
            })

    if errors:
        return errors, value, []
    canonical_value = stored if is_list else stored[0]
    return [], canonical_value, canonicalisations


def _run_value_checks(value, field_spec, *, reference_index, path_prefix,
                      canonicalisations, paper_text, image_labels, evidence,
                      run_evidence, evidence_required, weak_matches=None):
    """Shared value-level validation for one field. Single source of truth.

    Runs the type check, the categorical/reference canonicalisation, and
    (when `run_evidence`) the evidence quote check. Returns
    `(errors, value)`, where `value` is the value to store: canonicalised
    to an option's or reference entry's exact spelling on a match, an
    integral float year coerced to int, otherwise unchanged. Alias
    canonicalisations are appended to `canonicalisations`.

    `weak_matches`, when supplied, collects every quote that passed only on
    the case-folded tier (see `quote_check.validate_evidence`). It is
    optional because the two callers want different things from it: the LLM
    path records it into the run's artefact, and the single-field consumer
    path returns a verdict on one value with nowhere to put a provenance
    note.

    Both callers route through here so validation never forks:
    `validate_envelope` (LLM tool calls, evidence always checked) and
    `validate_value` (the importable single-field entry point, where the
    producer kind decides whether evidence runs at all).
    """
    errors = []

    # Coerce an integral float year to a plain int (2019.0 -> 2019) so the
    # stored value is clean. Non-integral years are left untouched and are
    # flagged as type errors by the check below.
    if field_spec.get("field_type") == "year" and isinstance(value, float):
        coerced = _integral_year(value)
        if coerced is not None:
            value = coerced

    type_errors = _check_value_type(value, field_spec, f"{path_prefix}.value")
    errors.extend(type_errors)

    # Canonicalise a categorical value to an option's exact spelling when it
    # matches only by case/whitespace. Runs only when the value passed its
    # type check and is a non-null string. Applies to both hard enums and
    # allow_other fields (a value that resolves to a listed option is stored
    # as that option; unmatched allow_other free text is left as-is).
    if not type_errors and isinstance(value, str) \
            and field_spec.get("field_type") == "categorical" \
            and field_spec.get("options"):
        matched = match_option(value, field_spec["options"])
        if matched is not None and matched != value:
            value = matched

    # Reference-list fields are a strict closed set: the value (one name on
    # a string field, a real list of names on a string_list field) must
    # resolve to canonical entry names (or unique aliases), element by
    # element. Canonicalise the stored spelling(s) on success; reject
    # off-list, ambiguous, empty, or duplicate values. The type check above
    # guarantees the value shape (str, or list of str) before this runs.
    if not type_errors and value is not None \
            and field_spec.get("canonical_reference"):
        ref_errors, new_value, canon_events = _validate_reference_value(
            value, field_spec, reference_index, path_prefix)
        errors.extend(ref_errors)
        if not ref_errors:
            value = new_value
            canonicalisations.extend(canon_events)

    if run_evidence:
        errors.extend(validate_evidence(
            evidence=evidence, paper_text=paper_text,
            image_labels=image_labels, value=value,
            field_path=path_prefix, evidence_required=evidence_required,
            weak_matches=weak_matches,
        ))

    return errors, value


def _check_notes_shape(notes, path):
    """Shape check for a `notes` slot: a string, or null.

    Returns a list of {path, code, message} errors (empty when legal). This
    is the ONLY check a note ever gets: it is never parsed for `<q>` or
    `<img>` tags, never quote-checked against the paper, and never counts
    toward satisfying `evidence: required`.
    """
    if notes is None or isinstance(notes, str):
        return []
    return [{
        "path": path,
        "code": "type_mismatch",
        "message": (
            f"Expected a string or null for 'notes', got "
            f"{type(notes).__name__}. Notes are free text; put any "
            f"structured content in the field's value."
        ),
    }]


def validate_envelope(envelope, field_spec, paper_text, image_labels,
                      path_prefix, reference_index=None,
                      canonicalisations=None, weak_quote_matches=None):
    """Validate one `{value, evidence, notes}` envelope against the field
    spec, the paper text, and the available image labels.

    `evidence` is a single string carrying any combination of
    <q>...</q> verbatim quotes, <img>label</img> image references, and
    free-text reasoning prose outside the tags. See
    `quote_check.parse_evidence_string` for the parser.

    `notes` is the field note: whatever justifies or explains the value and is
    not a verbatim quote. It is checked for shape only (see
    `_check_notes_shape`). An ABSENT `notes` key is treated as null and is not
    an error: the LLM tool schema requires the slot, but the human-producer
    path never sends one.

    `reference_index` is the resolution index (from
    `reference_lists.build_reference_index`) for the field's
    `canonical_reference` list, or None for non-reference fields. When a
    reference field's value canonicalises via an alias, a
    `{path, entered, stored}` dict is appended to the caller-supplied
    `canonicalisations` list (if given) so the dispatcher can surface the
    note and event.

    `weak_quote_matches` is the same shape of caller-supplied sink for a
    different provenance fact: every evidence quote that PASSED, but only on
    the case-folded tier, arrives in it as `{path, quote, tier}` (see
    `quote_check.validate_evidence`). Nothing here fails on it, and passing
    no sink is exactly as valid as passing one.

    Returns a list of {path, code, message} dicts. Empty = valid. Mutates
    `envelope["value"]` in place when a categorical or reference value
    canonicalises to a different spelling.

    This is the LLM-producer path: evidence is always checked, with the
    template's per-field `evidence: required | optional` flag deciding
    whether a non-null value must carry a quote or image. The value-level
    logic is shared with `validate_value` via `_run_value_checks`.
    """
    if not isinstance(envelope, dict):
        return [{
            "path": path_prefix,
            "code": "not_an_envelope",
            "message": "Expected an object with keys {value, evidence, notes}.",
        }]

    if "value" not in envelope:
        # Without a value the downstream checks cannot run meaningfully.
        return [{
            "path": f"{path_prefix}.value",
            "code": "missing_key",
            "message": "Envelope is missing the 'value' key.",
        }]

    canon_sink = canonicalisations if canonicalisations is not None else []
    notes_errors = _check_notes_shape(
        envelope.get("notes"), f"{path_prefix}.notes")
    evidence_required = (field_spec.get("evidence", "required") == "required")
    errors, new_value = _run_value_checks(
        envelope.get("value"), field_spec,
        reference_index=reference_index, path_prefix=path_prefix,
        canonicalisations=canon_sink, paper_text=paper_text,
        image_labels=image_labels, evidence=envelope.get("evidence"),
        run_evidence=True, evidence_required=evidence_required,
        weak_matches=weak_quote_matches,
    )
    # Write the canonicalised/coerced value back. When nothing changed this
    # is a no-op; on a failed reference/type check the value is unchanged.
    envelope["value"] = new_value
    return notes_errors + errors


# ---------------------------------------------------------------------------
# Importable single-field entry point (the consumer surface)
# ---------------------------------------------------------------------------

_PRODUCER_KINDS = ("human", "llm")

# Evidence verdicts that cannot be reached without the paper bundle. One
# bundle supplies both halves of the source — the text a quote is checked
# verbatim against, and the label set an `<img>` reference resolves in — so
# their absence is a single condition.
#
# These are dropped rather than reported when there is no source: without it
# every quote in the file would be reported as absent from a text that was
# never supplied, which reads as wholesale fabrication and is really a
# missing argument. The PRESENCE half of the evidence contract needs no
# source and always runs for an LLM producer, so an audit without the bundle
# still catches a value asserted with no evidence behind it and only defers
# whether the evidence quoted is real.
_SOURCE_DEPENDENT_EVIDENCE_CODES = frozenset({
    "quote_not_in_text",
    "unknown_image_label",
})


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating one field value.

    - `ok`: True iff there are no errors.
    - `value`: the value to store, canonicalised to an option's or a
      reference entry's exact spelling (and an integral float year coerced
      to int) when it matched, otherwise the value as supplied.
    - `errors`: list of {path, code, message} dicts; blocking.
    - `warnings`: list of {path, code, message} dicts; informational
      (category-gate hints), only populated when `sibling_values` is given.
    - `canonicalisations`: list of {path, entered, stored} dicts, one per
      value that matched a reference list via an ALIAS. A value that matched a
      canonical NAME records nothing here, even when the spelling it was
      entered with differs: the match is made on a normalised form (case,
      spacing and punctuation folded, see `reference_lists._normalise_ref`),
      so re-spelling a name to its canonical form is a silent correction and
      only standing in for a DIFFERENT name is reported.
    """

    ok: bool
    value: object
    errors: list
    warnings: list
    canonicalisations: list


def _reference_index_for_field(field_spec, reference_lists):
    """Build the reference-resolution index for a field's
    `canonical_reference` list, or return None.

    `reference_lists` may be a plain `{name: entries}` mapping or a loaded
    `ConfigBundle` (its `.reference_lists` is used). None is returned for a
    non-reference field, or when the named list is absent; in the latter case
    `_validate_reference_value` REPORTS a `reference_unavailable` error — it
    raises nothing, like every other check in this module, so the field fails
    with a message naming the wiring fault instead of the call failing with an
    exception (see `validate_value`: nothing here raises on data).
    """
    name = field_spec.get("canonical_reference")
    if not name:
        return None
    lists = getattr(reference_lists, "reference_lists", reference_lists)
    if not lists or name not in lists:
        return None
    from meltiro.reference_lists import build_reference_index
    return build_reference_index(lists[name])


def validate_value(field_spec, value, reference_lists=None,
                   producer_kind="human", *, evidence=None, paper_text=None,
                   image_labels=None, sibling_values=None, path="value",
                   reference_index=None, gates=None):
    """Validate one field's value. The importable single-field entry point.

    Args:
        field_spec: the resolved per-field spec dict (field_type, options,
            allow_other, canonical_reference, evidence, variable, ...).
        value: the proposed value (bare, not an envelope).
        reference_lists: the loaded reference lists as a `{name: entries}`
            mapping or a `ConfigBundle`; used to build the resolution index
            for a `canonical_reference` field. Ignored if `reference_index`
            is passed directly.
        producer_kind: "human" or "llm". Value-level checks (type, options,
            reference lists with alias canonicalisation, and gate rules when
            `sibling_values` is given) run for both. Evidence handling
            differs: an LLM producer is held to the template's
            required/optional flag, so a required-evidence field with a value
            and no quote or image reference fails; a human producer is never
            required to supply evidence, and any evidence it does supply is
            quote-checked only when `paper_text` is available (with no paper
            text the quote-checking layer does not run at all). For an LLM
            producer with no paper text, the presence half of the contract
            still runs and the source-dependent verdicts are withheld; see
            `_SOURCE_DEPENDENT_EVIDENCE_CODES`.
        evidence: the field's evidence string (or None).
        paper_text: full paper text for quote checking, or None.
        image_labels: iterable of available figure/table labels, or None.
        sibling_values: the record's other field values (a
            `variable -> value | envelope` map) to run cross-field gate
            warnings against; None skips gate rules (they are inherently
            cross-field).
        path: dotted path used in error/warning messages.
        reference_index: a pre-built resolution index, bypassing
            `reference_lists`.
        gates: the template's parsed gate list (`template["gates"]`). Gate
            warnings run only when both `sibling_values` and `gates` are
            supplied; an absent or empty `gates` list produces none. The
            engine names no gated field itself, so the caller passes the
            review's declared gates from the loaded template.

    Returns a `ValidationResult`. Never raises on invalid data (it reports);
    it raises only on a caller error (an unknown `producer_kind`).
    """
    if producer_kind not in _PRODUCER_KINDS:
        raise ValueError(
            f"unknown producer_kind {producer_kind!r}; expected one of "
            f"{_PRODUCER_KINDS}.")

    if reference_index is None:
        reference_index = _reference_index_for_field(field_spec,
                                                     reference_lists)

    image_set = {normalise_label(lbl) for lbl in (image_labels or [])}
    canonicalisations = []

    # Producer-conditional evidence policy. LLM producers always run the
    # evidence layer (paper text is always present in the pipeline) with the
    # template's per-field required/optional flag. Human producers are never
    # required to supply evidence, and their volunteered evidence is
    # quote-checked only when paper text is available; with none, the
    # quote-checking layer does not run.
    if producer_kind == "llm":
        run_evidence = True
        evidence_required = (
            field_spec.get("evidence", "required") == "required")
    else:
        run_evidence = paper_text is not None
        evidence_required = False

    errors, canonical_value = _run_value_checks(
        value, field_spec, reference_index=reference_index, path_prefix=path,
        canonicalisations=canonicalisations, paper_text=paper_text,
        image_labels=image_set, evidence=evidence, run_evidence=run_evidence,
        evidence_required=evidence_required,
    )

    if run_evidence and paper_text is None:
        errors = [e for e in errors
                  if e["code"] not in _SOURCE_DEPENDENT_EVIDENCE_CODES]

    warnings = []
    if sibling_values is not None:
        gate_warnings = validate_gate_rules(
            sibling_values, gates or [], path_prefix="")
        var = field_spec.get("variable")
        warnings = ([w for w in gate_warnings if w.get("path") == var]
                    if var is not None else gate_warnings)

    return ValidationResult(
        ok=not errors, value=canonical_value, errors=errors,
        warnings=warnings, canonicalisations=canonicalisations,
    )


# ---------------------------------------------------------------------------
# Cross-field rules
# ---------------------------------------------------------------------------

def validate_gate_rules(record_field_values, gates, path_prefix=""):
    """Cross-field gate checks for a single record, driven by the template's
    declared `gates`.

    Each gate names a controlling field (`when_field`), a gated field
    (`field`), and the controlling values (`allowed_values`) under which the
    gated field is expected. When the gated field carries a truthy value but
    the controlling field's value is not one of the allowed values (compared
    case- and whitespace-insensitively), a WARNING is emitted. The
    deterministic check just flags the suspicious combination; the checker
    decides whether the value is genuinely justified.

    `gates` is the template's parsed gate list (`template["gates"]`, see
    template.load_template). An empty list means the template declares no
    cross-field gates and no warning is ever produced. The engine names no
    field itself: every field name here comes from the template.

    Returns a list of {path, code, message} WARNINGS (not errors).

    `path_prefix` (e.g. "record.<new>") is prepended to the gated field so the
    warning path matches the field's canonical path. The dispatcher passes the
    record's prefix, which is what puts a gate warning in front of the model
    beside the field it is about.

    It defaults to "" — a BARE variable name, not a canonical field path — and
    `validate_value` passes "" deliberately: that caller filters the warnings
    down to the one field it is validating by comparing the warning path
    against `field_spec["variable"]`, so a prefixed path would match nothing
    and every gate warning would be dropped. The two callers therefore want
    different paths out of this function, and each asks for the one it can
    use.
    """
    warnings = []
    prefix = f"{path_prefix}." if path_prefix else ""

    for gate in gates or []:
        when_field = gate["when_field"]
        gated_field = gate["field"]
        allowed = gate["allowed_values"]
        controlling = _envelope_value(record_field_values.get(when_field))
        # A controlling value that cannot be read as text (None, missing, or a
        # non-string) can't be gated against, so skip this gate: an unset
        # controlling field is not evidence that the gated field is wrong.
        if not isinstance(controlling, str):
            continue
        gated_value = _envelope_value(record_field_values.get(gated_field))
        if not gated_value:
            continue
        allowed_norms = {_normalise_option(a) for a in allowed}
        if _normalise_option(controlling) in allowed_norms:
            continue
        warnings.append({
            "path": f"{prefix}{gated_field}",
            "code": "category_gate",
            "message": (
                f"{gated_field} is set but {when_field} is "
                f"'{controlling}', not one of: {', '.join(allowed)}."
            ),
        })

    return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _envelope_value(envelope):
    """Pull `.value` from an envelope, or return the value if it's already
    a plain (non-dict) value. None-safe."""
    if envelope is None:
        return None
    if isinstance(envelope, dict):
        return envelope.get("value")
    return envelope


# ---------------------------------------------------------------------------
# Batch sweep (the `meltiro validate` surface)
# ---------------------------------------------------------------------------

def _all_specs(sections):
    """`{variable: field_spec}` for a block.

    The re-validation sweep recognises every declared field. This mirrors the
    dispatcher's `_index_specs` (every field is addressable now that there are
    no hidden pipeline-managed fields).
    """
    from meltiro.template import iter_fields
    return {f["variable"]: f for f in iter_fields(sections)}


def validate_extraction_output(template, extraction_output,
                               reference_lists=None, *, paper_text=None,
                               image_labels=None, producer_kind="human"):
    """Re-validate every field value in an extraction output.

    Walks the study envelopes, the record envelopes, and the bare
    initial_check / quality_check blocks, calling `validate_value` on each.
    Returns `(failures, warnings)`, each a flat list of {path, code, message}
    dicts; `failures` empty means every value is legal.

    `producer_kind` decides how much of the evidence contract is in scope.
    Under `"human"` no evidence is ever demanded, so an output whose every
    required-evidence field carries a null evidence string passes on its
    values alone. Pass `"llm"` for engine-produced output to hold each field
    to the template's own `evidence:` flag; that is the only setting under
    which this function re-verifies the evidence contract. The default is
    `"human"` because the per-save consumer path this also serves is a
    person editing values, for whom the template's flag is not a promise
    anyone made.

    `paper_text` decides whether the source-dependent half runs: with it,
    evidence quotes are checked verbatim and `<img>` labels are resolved
    against the bundle's figures; without it those verdicts are unreachable
    and are not reported (see `_SOURCE_DEPENDENT_EVIDENCE_CODES`).

    Cross-field gate warnings run per record from the template's declared
    `gates` (empty when the template declares none).

    Notes are shape-checked, never adjudicated. The reserved scope-note key
    on the study block and on each record is not a field: it is skipped by the
    per-field sweep and checked only for being a string or null, as is each
    envelope's own `notes` slot. Nothing else about a note is validated.

    This is the batch entry point behind `meltiro validate`; the consumer's
    per-save path calls `validate_value` directly.
    """
    failures = []
    warnings = []
    gates = template.get("gates") or []

    def _sweep(spec, value, path, *, evidence=None, sibling_values=None):
        if spec is None:
            failures.append({
                "path": path, "code": "unknown_field",
                "message": (f"'{path}' is not a known field in the template "
                            f"for this block."),
            })
            return
        result = validate_value(
            spec, value, reference_lists, producer_kind, evidence=evidence,
            paper_text=paper_text, image_labels=image_labels,
            sibling_values=sibling_values, path=path, gates=gates)
        failures.extend(result.errors)
        warnings.extend(result.warnings)

    study_specs = _all_specs(template["study_fields"])
    record_specs = _all_specs(template["record_fields"])
    initial_specs = _all_specs(template["initial_check_fields"])
    quality_specs = _all_specs(template["quality_check_fields"])

    study = extraction_output.get("study") or {}
    failures.extend(_check_notes_shape(study.get(NOTES_KEY), "study.notes"))
    for var, env in study.items():
        if var == NOTES_KEY:
            continue  # the scope note, not a field
        if not _is_envelope(env, f"study.{var}", failures):
            continue
        failures.extend(
            _check_notes_shape(env.get("notes"), f"study.{var}.notes"))
        _sweep(study_specs.get(var), env.get("value"), f"study.{var}",
               evidence=env.get("evidence"))

    for record in extraction_output.get("records") or []:
        rid = record.get("record_id")
        failures.extend(_check_notes_shape(
            record.get(NOTES_KEY), f"record.{rid}.notes"))
        siblings = {k: v for k, v in record.items()
                    if k not in ("record_id", NOTES_KEY)}
        for var, env in record.items():
            if var in ("record_id", NOTES_KEY):
                continue  # the engine-assigned id and the scope note
            path = f"record.{rid}.{var}"
            if not _is_envelope(env, path, failures):
                continue
            failures.extend(
                _check_notes_shape(env.get("notes"), f"{path}.notes"))
            _sweep(record_specs.get(var), env.get("value"), path,
                   evidence=env.get("evidence"), sibling_values=siblings)

    # The two check blocks are keyed by the ROLE that recorded them
    # (`{role: {variable: value}}`), so the sweep descends one level before it
    # reaches a variable. Reading them flat would treat each role name as a
    # field, report the whole block as unknown fields, and — the half that
    # matters — leave the values underneath unvalidated.
    # A non-mapping under a role is reported once rather than iterated, so a
    # hand-edited file degrades to one clear failure instead of a spray.
    for block_name, block, specs in [
        ("initial_check", extraction_output.get("initial_check") or {},
         initial_specs),
        ("quality_check", extraction_output.get("quality_check") or {},
         quality_specs),
    ]:
        for role, answers in (block or {}).items():
            if not isinstance(answers, dict):
                failures.append({
                    "path": f"{block_name}.{role}",
                    "code": "not_a_role_block",
                    "message": (
                        f"Expected an object of variable -> value under "
                        f"{block_name}.{role}: {block_name} is keyed by the "
                        f"role that recorded it."
                    ),
                })
                continue
            for var, value in answers.items():
                _sweep(specs.get(var), value, f"{block_name}.{role}.{var}")

    return failures, warnings


def missing_required_fields(template, extraction_output):
    """Every template-declared `required: true` field shipping a null value.

    Returns a sorted list of dotted paths (`study.<var>`,
    `record.<record_id>.<var>`). Empty means every required field carries a
    value.

    This is PRESENCE, which `validate_extraction_output` deliberately does
    not cover: that function sweeps the values an output STORES and says
    whether each is legal, so a required field that is null, or absent from
    the output altogether, gives it nothing to fault.

    The dispatcher's extractor-side `mark_complete` asks the same question
    at the moment the extractor tries to finish, and refuses until the
    answer is empty. That gate binds the extractor alone: the reviewer's
    `mark_complete` is not completeness-gated (it saw the whole assembled
    output and has no second chance to terminate), so a reviewer edit after
    the extractor finished can null a required field and the run still
    ships. The guarantee is "the extractor cannot leave its own pass
    incomplete", not "no output ships with a required field unset", which is
    why this exists as a function anyone can call over a finished output.

    Records with no `record_id` report under `record.None`, which is what the
    rest of the engine's paths do for an unidentified record; an output in that
    state has a deeper problem than this sweep, and inventing an index here
    would name a record that nothing else names.
    """
    from meltiro.template import required_field_names

    missing = []

    study = extraction_output.get("study") or {}
    for var in required_field_names(template["study_fields"]):
        env = study.get(var)
        value = env.get("value") if isinstance(env, dict) else env
        if value is None:
            missing.append(f"study.{var}")

    record_required = required_field_names(template["record_fields"])
    for record in extraction_output.get("records") or []:
        rid = record.get("record_id")
        for var in record_required:
            env = record.get(var)
            value = env.get("value") if isinstance(env, dict) else env
            if value is None:
                missing.append(f"record.{rid}.{var}")

    return sorted(missing)


def _is_envelope(env, path, failures):
    """True if `env` is a `{value, ...}` envelope; else record a failure."""
    if isinstance(env, dict) and "value" in env:
        return True
    failures.append({
        "path": path, "code": "not_an_envelope",
        "message": "Expected an object with a 'value' key.",
    })
    return False
