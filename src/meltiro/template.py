"""
Parse the extraction template YAML into structured field definitions.

Reads the config bundle's extraction_template.yaml and returns field
schemas used by the prompt builder, JSON schema generator, and validation
logic.

Each YAML field carries an explicit shape:
  - `options: [...]` for enum (categorical) fields, OR
  - `type: <string|integer|number|boolean|year|date|string_list>` for
    open-value fields.

These are mutually exclusive. `null` is allowed for every type except
`boolean` (the runtime validator enforces this). Per-field "required"
semantics are declared here via the `required: true` flag and enforced at
mark_complete by the dispatcher (see meltiro.tools); the engine never names
a specific required field.
"""

import hashlib
import re
from pathlib import Path

from meltiro.extraction_record import NOTES_KEY
from meltiro.yaml_strict import strict_load


# Canonical type names accepted in the YAML. Everything maps cleanly to
# JSON Schema in the tool-definitions module.
_VALID_TYPES = {"string", "integer", "number", "boolean",
                "year", "date", "string_list"}

# Allowed values for the per-field `evidence:` flag. There is no default:
# envelope fields must declare one explicitly (a forgotten flag fails
# loudly at load). "required" means verbatim quote(s) must support every
# non-null value. Mark fields whose value is the extractor's judgement
# (a justification or a holistic assessment) as "optional" so they're not
# forced to fabricate a quote.
_VALID_EVIDENCE = {"required", "optional"}

# A categorical field (one carrying an `options:` list) is a HARD enum by
# default: its value must match one of the options exactly (after
# case/whitespace canonicalisation). Adding `allow_other: true` turns the
# option list into a set of typical values while still accepting any
# free-text string that fits none of them; that free text goes directly in the
# field's own value.

# Allowed values for the per-field `role:` flag. A role marks a field for
# mechanical wiring by the pipeline (never by field name):
#   - `summary`: this field's extracted value is the Checker's study-identity
#     context when the paper bundle carries no manifest `summary`.
# `summary` is the only role: the study id is a pipeline concern sourced from
# the bundle manifest and recorded in the run's output metadata, not a template
# field, so there is no `role: study_id`.
# A role-bearing field must be a plain-string field, sit in its role's scope,
# and at most one field may claim each role; these constraints are enforced in
# load_template (`_resolve_role_fields`). Unknown role values are rejected in
# `_parse_field`.
_VALID_ROLES = {"summary"}

# Which extraction scope each role must sit in. `_resolve_role_fields` places
# study-scoped roles on study fields and rejects a role that lands in the
# wrong scope.
_ROLE_SCOPES = {"summary": "study"}

# Top-level keys the template loader consumes. The repeated-row entity and
# its extraction sections live under the single `records:` mapping (see
# `_parse_records`), not their own top-level blocks. Quality-assessment fields
# are ordinary sections carrying `qa: true` (see `_parse_sections`), so there
# is no separate QA block at either scope.
_KNOWN_TOP_KEYS = {
    "study_extraction",
    "records",
    "llm_initial_check",
    "llm_quality_check",
}

# Optional top-level keys: allowed but not required. `gates:` declares the
# review's cross-field gate rules (see `_parse_gates`); a template that
# omits it simply has no gates. Kept separate from `_KNOWN_TOP_KEYS` so the
# missing-required-key check does not demand it.
_OPTIONAL_TOP_KEYS = {"gates"}

# Subkeys of one gate rule under the optional `gates:` block.
_GATE_KEYS = {"when_field", "field", "allowed_values"}

# Keys one section may carry, in any block. `section` is the title, `label` an
# optional human-facing name, `qa` an optional presentation flag marking the
# section as quality assessment, `extraction_instruction` optional
# section-level guidance, and `fields` the field list. Anything else is a load
# error, so a mistyped key fails loudly instead of being ignored.
_SECTION_KEYS = {"section", "label", "qa", "extraction_instruction", "fields"}

# Keys one field may carry, in any block. `variable` (the stable identifier)
# and `description` are required; every other key is optional and defaulted.
# `type` and `options` are mutually exclusive (see `_resolve_field_type`), and
# `evidence` is required on envelope fields and forbidden on the bare-value
# check blocks, but both rules are shape checks applied after this allowlist.
# Anything outside the set is a load error, so a mistyped key (`requred:`,
# `evidnce:`, `canonial_reference:`) fails loudly instead of being silently
# ignored. Silence is the worst outcome available here: the field would quietly
# take its default (`required: false`, no reference validation), the run would
# complete, and fingerprint.field_catalogue_hash would record the defaulted
# config as though the author had written it.
_FIELD_KEYS = {
    "variable",
    "description",
    "label",
    "extraction_instruction",
    "type",
    "options",
    "allow_other",
    "evidence",
    "role",
    "required",
    "canonical_reference",
    "soft_canonicalisation",
}

# Subkeys of one record-type definition under `records:`.
_RECORD_TYPE_KEYS = {"plural", "description", "extraction_instruction",
                     "checker_context_fields", "extraction"}

# The record entity noun (the `records:` key, i.e. `singular`) is threaded
# into record ids (`<singular>_<n>`) and from there into dotted field paths
# (`record.<id>.<var>`), so it is constrained to a strict slug at load rather
# than left to mis-route mid-run: a `.` would add a path segment, so
# `_extraction_record_field_envelope` (which splits the path on `.`) would
# look up the wrong record_id, find nothing, and silently treat a challenged
# field as absent. The single-underscore part keeps `<singular>_<n>` ids'
# noun/number split unambiguous. `plural` is not constrained: it is only
# rendered into model-facing text, never into an id or a path.
_RECORD_ENTITY_SLUG = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


# ---------------------------------------------------------------------------
# Top-level key validation
# ---------------------------------------------------------------------------

def _validate_top_level_keys(raw):
    """Reject unknown or missing top-level keys, loudly.

    The allowlist is the whole rule: a key outside `_KNOWN_TOP_KEYS` plus the
    optional `gates:` block (see `_OPTIONAL_TOP_KEYS`) is a load error, and the
    message names every key that IS accepted so a wrong guess is one read away
    from the right one. A missing required key fails the same way. Nothing is
    remapped: a template declares the schema this loader reads, or it does not
    load.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"extraction_template.yaml must parse to a mapping, got "
            f"{type(raw).__name__}."
        )
    allowed = _KNOWN_TOP_KEYS | _OPTIONAL_TOP_KEYS
    unknown = sorted(k for k in raw if k not in allowed)
    if unknown:
        raise ValueError(
            f"extraction_template.yaml has unknown top-level key(s) "
            f"{unknown}. Known keys: {sorted(allowed)}."
        )
    missing = sorted(_KNOWN_TOP_KEYS - set(raw))
    if missing:
        raise ValueError(
            f"extraction_template.yaml is missing required top-level key(s) "
            f"{missing}. Required: {sorted(_KNOWN_TOP_KEYS)}."
        )


# ---------------------------------------------------------------------------
# records block
# ---------------------------------------------------------------------------

def _parse_records(raw):
    """Parse and validate the required top-level `records:` mapping.

    `records:` maps a record-type name (the singular entity noun) to its
    definition: an optional `plural:`, a required `description:` (rendered
    into the tool descriptions), an optional `extraction_instruction:`
    (guidance for what counts as
    one record, rendered as the leading clause of the add_record tool
    description), an optional `checker_context_fields:` list of record field
    names, and a required non-empty `extraction:` section list. Any other
    subkey is a load error: quality-assessment sections are ordinary
    `extraction:` sections marked `qa: true`, so there is no second section
    list to declare.

    Returns `(entity, extraction_raw, context_fields)` where `entity` is
    `{singular, plural, description, extraction_instruction}` (the same shape
    the parsed template has always exposed under `record_entity`, plus the
    record-level instruction), `extraction_raw` is the unparsed section list,
    and `context_fields` is the ordered list of record field names the checker
    shows (after the record id) to label a record; it is empty when the key is
    absent. This function validates its shape only (a list of non-empty
    strings); that each name resolves to an existing record-scoped field is a
    whole-template check done in `load_template` via
    `_validate_checker_context_fields`.

    Exactly one record type is supported: zero entries or more than one is a
    loud load error.
    """
    if "records" not in raw or raw["records"] is None:
        raise ValueError(
            "extraction_template.yaml is missing the required top-level "
            "`records:` block. It maps a record-type name (the singular "
            "entity noun) to its definition: an optional `plural:`, a "
            "required `description:`, a required non-empty `extraction:` "
            "section list, and an optional `checker_context_fields:` "
            "field-name list."
        )
    block = raw["records"]
    if not isinstance(block, dict):
        raise ValueError(
            f"`records:` must be a mapping from a record-type name to its "
            f"definition, got {type(block).__name__}."
        )
    # One record type only; see the docstring.
    if len(block) != 1:
        raise ValueError(
            f"`records:` must declare exactly one record type; got "
            f"{len(block)} ({sorted(block)!r}). A template describes one "
            f"repeated entity; a review needing two needs two templates."
        )
    singular, definition = next(iter(block.items()))
    if not isinstance(singular, str) or not singular.strip():
        raise ValueError(
            f"`records:` key (the record-type name) must be a non-empty "
            f"string, got {singular!r}."
        )
    singular = singular.strip()
    if not _RECORD_ENTITY_SLUG.match(singular):
        raise ValueError(
            f"`records:` key (the record-type name) {singular!r} is not a "
            f"valid entity slug. It is minted into record ids "
            f"(`{singular}_1`), dotted field paths, and audit filenames, so "
            f"it must be lowercase ASCII letters, digits, and single internal "
            f"underscores, starting with a letter (e.g. `treatment_effect`). "
            f"Names with a `.`, a leading or trailing `_`, a `__`, an "
            f"uppercase letter, or a leading digit are rejected."
        )
    if singular == "study":
        # A record entity named `study` would collide with the `study.` scope
        # prefix in gate references: every correctly-scoped gate for it would
        # be rejected as a study-scoped controller, with a message telling the
        # author to write exactly the spelling that was rejected.
        raise ValueError(
            "`records:` key (the record-type name) must not be `study`: the "
            "`study.` prefix is reserved for study-scoped references in gate "
            "rules. Choose a different entity name."
        )
    if not isinstance(definition, dict):
        raise ValueError(
            f"`records.{singular}` must be a mapping with keys "
            f"{sorted(_RECORD_TYPE_KEYS)}, got {type(definition).__name__}."
        )
    unknown = sorted(set(definition) - _RECORD_TYPE_KEYS)
    if unknown:
        raise ValueError(
            f"`records.{singular}` has unknown subkey(s) {unknown}. Only "
            f"{sorted(_RECORD_TYPE_KEYS)} are allowed."
        )

    description = definition.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"`records.{singular}.description` is required and must be a "
            f"non-empty string; it is rendered into the tool descriptions."
        )

    plural = definition.get("plural")
    if plural is None:
        plural = singular + "s"
    elif not isinstance(plural, str) or not plural.strip():
        raise ValueError(
            f"`records.{singular}.plural` must be a non-empty string when "
            f"present, got {plural!r}. Omit it to default to the singular "
            f"plus 's'."
        )
    else:
        plural = plural.strip()

    # Optional record-level `extraction_instruction`: the guidance for how to
    # recognise one record versus another (when something counts as a separate
    # <entity>, when NOT to combine). It renders as the leading clause of the
    # add_record tool description (see meltiro.tools), so it rides tool_set_hash.
    record_instruction = definition.get("extraction_instruction")
    if record_instruction is not None:
        if not isinstance(record_instruction, str) or \
                not record_instruction.strip():
            raise ValueError(
                f"`records.{singular}.extraction_instruction` must be a "
                f"non-empty string when present, got {record_instruction!r}. "
                f"Omit it if the record entity needs no extra guidance."
            )
        record_instruction = record_instruction.strip()

    extraction_raw = definition.get("extraction")
    if not isinstance(extraction_raw, list) or not extraction_raw:
        raise ValueError(
            f"`records.{singular}.extraction` is required and must be a "
            f"non-empty list of sections."
        )

    # `checker_context_fields` is the OPTIONAL ordered list of record field
    # names the checker shows (after the engine record id) to label a record
    # (see build_record_context). Absent means an empty list: the checker then
    # labels each record by its id alone. Shape only here (a list of non-empty
    # strings); field-existence is validated once the record fields are
    # parsed (see `_validate_checker_context_fields`).
    context_raw = definition.get("checker_context_fields")
    if context_raw is None:
        context_raw = []
    if not isinstance(context_raw, list):
        raise ValueError(
            f"`records.{singular}.checker_context_fields` must be a list of "
            f"record field names when present, got "
            f"{type(context_raw).__name__}. Omit it to label records by id "
            f"alone."
        )
    context_fields = []
    for name in context_raw:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"`records.{singular}.checker_context_fields` entries must be "
                f"non-empty strings naming record fields, got {name!r}."
            )
        context_fields.append(name.strip())

    entity = {
        "singular": singular,
        "plural": plural,
        "description": description.strip(),
        "extraction_instruction": record_instruction,
    }
    return entity, extraction_raw, context_fields


# ---------------------------------------------------------------------------
# YAML section parser
# ---------------------------------------------------------------------------

def _resolve_field_type(f):
    """Resolve a YAML field into (field_type, options).

    Errors loudly if the field is malformed. Failing at template load beats
    extracting silently under an unknown shape, which is the standing rule for
    every input this module reads.

    A literal "Other" entry inside an options list is a config error: free
    text outside the option set is expressed with `allow_other: true`.
    """
    var = f["variable"]
    has_options = "options" in f and f["options"] is not None
    has_type = "type" in f and f["type"] is not None

    if has_options and has_type:
        raise ValueError(
            f"Field {var!r}: both `type:` and `options:` are set. They are "
            f"mutually exclusive (a field is either an enum or an open type)."
        )
    if has_options:
        options = list(f["options"])
        if not options:
            raise ValueError(f"Field {var!r}: `options:` is empty.")
        if any(str(o).strip().lower() == "other" for o in options):
            raise ValueError(
                f"Field {var!r}: `options:` contains a literal \"Other\" "
                f"entry. Remove it and set `allow_other: true` so the field "
                f"accepts free text when none of the listed options fits."
            )
        return "categorical", options
    if has_type:
        t = str(f["type"]).strip()
        if t not in _VALID_TYPES:
            raise ValueError(
                f"Field {var!r}: unknown `type: {t!r}`. Valid: "
                f"{sorted(_VALID_TYPES)}."
            )
        return t, None
    raise ValueError(
        f"Field {var!r}: neither `type:` nor `options:` is set. Every "
        f"field needs one. See the header in extraction_template.yaml."
    )


def _derive_label(source):
    """Default human-facing label: underscores to spaces, first letter
    capitalised. 'qa_reporting' -> 'Qa reporting'. Only the first character
    is touched, so embedded casing (acronyms in a section title) is left
    alone."""
    spaced = source.replace("_", " ")
    if not spaced:
        return spaced
    return spaced[0].upper() + spaced[1:]


def _resolve_label(raw_label, *, default, what):
    """Resolve a field/section `label:`.

    Returns `raw_label` (stripped) when present, else the derived `default`.
    A present-but-non-string or empty/whitespace-only label fails loudly:
    the label is presentation metadata, but a malformed one is still a
    config error. Labels are presentation-only: never sent to a model and
    excluded from the field-catalogue hash (see
    fingerprint.field_catalogue_hash). config_fp still moves on a label
    edit, as on any byte edit to the file, via the whole-file
    template_hash."""
    if raw_label is None:
        return default
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError(
            f"{what}: `label:` must be a non-empty string when present, got "
            f"{raw_label!r}. Omit the key to fall back to the derived default."
        )
    return raw_label.strip()


def _parse_field(f, *, envelope, section_name):
    """Parse one YAML field dict into the canonical field dict shape.

    Validates `evidence:` and rejects the reserved `notes` variable name. Any
    key outside `_FIELD_KEYS` fails loudly, so a mistyped field key
    (`requred:`, `evidnce:`) is a load error rather than a silently ignored
    line that leaves the field on its default. Errors loudly on any malformed
    input.
    """
    # `variable:` is read before anything else and is what every message below
    # names the field by, so its absence is answered here rather than as a
    # KeyError out of the first read. The section and the keys the mapping DOES
    # declare are both in the message, because that is what locates the field
    # in a template of several hundred lines and what makes a near-miss
    # spelling (`varible:`) visible as the cause.
    if not isinstance(f, dict):
        raise ValueError(
            f"Section {section_name!r}: every entry under `fields:` must be a "
            f"mapping declaring at least `variable:` and one of `type:` / "
            f"`options:`, got {type(f).__name__} ({f!r})."
        )
    if "variable" not in f:
        raise ValueError(
            f"Section {section_name!r}: a field declares no `variable:`. The "
            f"mapping declares {sorted(f)}. `variable:` is the field's stable "
            f"identifier — it names the field in every tool schema, every "
            f"stored value and every error — so a field without one cannot be "
            f"loaded. Add it, or fix its spelling."
        )
    var = f["variable"]
    # `notes` is the reserved scope-note key: it sits inside the study block
    # and inside every record object, alongside `record_id`, holding that
    # scope's free-text commentary. A field of that name would collide with
    # it, so the name is refused in every scope (the two bare-value check
    # blocks included, so the rule reads the same wherever an author looks).
    if var == NOTES_KEY:
        raise ValueError(
            f"Field {var!r}: `{NOTES_KEY}` is a reserved key, not an available "
            f"variable name. Each scope (the study block and every record) "
            f"carries a reserved `{NOTES_KEY}` holding that scope's free-text "
            f"commentary, and every field carries its own `notes` slot inside "
            f"its envelope, so a field named `{NOTES_KEY}` would collide with "
            f"the scope note. Rename the field, or delete it and use the scope "
            f"note."
        )
    # The unknown-key allowlist, applied before any key is read, so a mistyped
    # key is named in the error rather than showing up as a downstream symptom
    # (a misspelt `type:` would otherwise surface as "neither `type:` nor
    # `options:` is set", and a misspelt `required:` as nothing at all).
    unknown = sorted(set(f) - _FIELD_KEYS)
    if unknown:
        raise ValueError(
            f"Field {var!r}: unknown field key(s) {unknown}. Only "
            f"{sorted(_FIELD_KEYS)} are allowed."
        )
    ei_str = f.get("extraction_instruction", "") or ""
    field_type, options = _resolve_field_type(f)

    if envelope:
        evidence = f.get("evidence")
        if evidence is None:
            raise ValueError(
                f"Field {var!r}: `evidence:` is missing. Every envelope "
                f"field must declare `evidence: required` or "
                f"`evidence: optional`; defaults are forbidden so a "
                f"forgotten flag fails loudly at load. Valid: "
                f"{sorted(_VALID_EVIDENCE)}."
            )
        if evidence not in _VALID_EVIDENCE:
            raise ValueError(
                f"Field {var!r}: unknown `evidence: {evidence!r}`. "
                f"Valid: {sorted(_VALID_EVIDENCE)}."
            )
    else:
        if "evidence" in f:
            raise ValueError(
                f"Field {var!r}: `evidence:` set on a non-envelope field "
                f"(initial_check / quality_check blocks use bare values). "
                f"Remove the key."
            )
        evidence = None

    # Field-level role: marks the field for mechanical wiring by the
    # pipeline (see _VALID_ROLES). Per-field checks here: the value must be
    # a known role, and the field must be a plain string (a role carrying
    # an options list or a non-string type is rejected). The study-level
    # and at-most-one-per-role constraints span the whole template and are
    # enforced in load_template via _resolve_role_fields.
    role = f.get("role")
    if role is not None:
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Field {var!r}: unknown `role: {role!r}`. Valid: "
                f"{sorted(_VALID_ROLES)}."
            )
        if field_type != "string":
            kind = "an options list" if field_type == "categorical" \
                else f"type {field_type!r}"
            raise ValueError(
                f"Field {var!r}: `role: {role}` requires a plain string "
                f"field (`type: string`), but this field has {kind}."
            )

    # Optional `canonical_reference: <name>` names a reference list this
    # field's values are validated against. The name must resolve to a
    # `<name>.yaml` file in the config bundle's `reference/` directory;
    # that cross-check happens at config-bundle load (config_bundle.py), not
    # here, because the template parser has no view of the bundle. The key
    # is only valid on envelope fields (the bare-value initial_check /
    # quality_check blocks never run reference validation, so accepting it
    # there would silently not enforce it) and only on `type: string` (one
    # name) or `type: string_list` (a real JSON array of names, validated
    # element by element).
    if not envelope and "canonical_reference" in f:
        raise ValueError(
            f"Field {var!r}: `canonical_reference:` set on a non-envelope "
            f"field (initial_check / quality_check blocks use bare values, "
            f"which never run reference validation, so the key would be "
            f"silently unenforced). Remove the key or move the field to an "
            f"envelope block."
        )
    canonical_reference = f.get("canonical_reference")
    if canonical_reference is not None:
        if not isinstance(canonical_reference, str) or \
                not canonical_reference.strip():
            raise ValueError(
                f"Field {var!r}: `canonical_reference:` must be a non-empty "
                f"string, got {canonical_reference!r}."
            )
        canonical_reference = canonical_reference.strip()
        if field_type not in ("string", "string_list"):
            kind = "an options list" if field_type == "categorical" \
                else f"type {field_type!r}"
            raise ValueError(
                f"Field {var!r}: `canonical_reference:` requires "
                f"`type: string` (one reference name) or `type: string_list` "
                f"(a list of reference names), but this field has {kind}. "
                f"Reference validation only runs on string values, so the "
                f"key would be silently unenforced here."
            )

    # Optional `allow_other: true` turns a categorical field from a hard
    # enum into one that accepts any free-text string outside the option
    # list. It is only valid on a field with `options:`; combining it with
    # `canonical_reference:` is impossible by construction (that key
    # requires a string or string_list type and is rejected above).
    allow_other = f.get("allow_other", False)
    if not isinstance(allow_other, bool):
        raise ValueError(
            f"Field {var!r}: `allow_other:` must be true or false, got "
            f"{allow_other!r}."
        )
    if allow_other and field_type != "categorical":
        raise ValueError(
            f"Field {var!r}: `allow_other: true` is only valid on a "
            f"categorical field (one with an `options:` list)."
        )

    # Optional `soft_canonicalisation: true` is a CONSUMER-FACING declaration,
    # inert in the engine. It signals to UI and analysis consumers that this
    # field's values should be auto-suggested and collapsed against values
    # entered earlier (a working vocabulary that builds up as extraction
    # proceeds), with NO hard validation and NO analytical hard lines. The
    # engine parses, validates, and exposes it, but takes no runtime action on
    # it: no schema change, no checker change, no validation of values. It is
    # therefore deliberately NOT in field_catalogue_hash (it changes nothing
    # the extractor or checker does). It is only valid on a free-value field
    # (`type:`, not `options:`) and is mutually exclusive with
    # `canonical_reference` (hard canonicalisation): a field is either a strict
    # closed set or a soft-collapse open vocabulary, not both.
    soft_canonicalisation = f.get("soft_canonicalisation", False)
    if not isinstance(soft_canonicalisation, bool):
        raise ValueError(
            f"Field {var!r}: `soft_canonicalisation:` must be true or false, "
            f"got {soft_canonicalisation!r}."
        )
    if soft_canonicalisation and field_type == "categorical":
        raise ValueError(
            f"Field {var!r}: `soft_canonicalisation: true` is only valid on a "
            f"free-value field (one with a `type:`), not a categorical field "
            f"(one with an `options:` list). A categorical field already has a "
            f"fixed vocabulary."
        )
    if soft_canonicalisation and canonical_reference is not None:
        raise ValueError(
            f"Field {var!r}: `soft_canonicalisation: true` cannot be combined "
            f"with `canonical_reference:`. `canonical_reference` is hard "
            f"canonicalisation (a strict closed set, validated); "
            f"`soft_canonicalisation` is a consumer-side soft collapse with no "
            f"validation. Choose one."
        )

    # Optional presentation label for human-facing surfaces. Absent means a
    # derived default. Presentation-only, excluded from field_catalogue_hash
    # and the tool schemas; see _resolve_label.
    label = _resolve_label(
        f.get("label"), default=_derive_label(var), what=f"Field {var!r}")

    # Optional `required: true` marks the field as required at mark_complete:
    # an envelope field (study / record) must carry a non-null value; a bare
    # initial_check / quality_check field must be present and non-null. The
    # dispatcher enforces this generically (meltiro.tools) so the engine
    # never names a specific required field. Default False. Unlike `label`,
    # `required` changes validation behaviour, so it DOES enter
    # field_catalogue_hash.
    required_flag = f.get("required", False)
    if not isinstance(required_flag, bool):
        raise ValueError(
            f"Field {var!r}: `required:` must be true or false, got "
            f"{required_flag!r}."
        )

    return {
        "variable": var,
        "label": label,
        "description": f["description"],
        "extraction_instruction": ei_str if ei_str else None,
        "field_type": field_type,
        "options": options,
        "allow_other": allow_other,
        "evidence": evidence,
        "role": role,
        "required": required_flag,
        "canonical_reference": canonical_reference,
        "soft_canonicalisation": soft_canonicalisation,
    }


def _parse_sections(sections, *, envelope):
    """Parse YAML section list into nested section dicts.

    `envelope` is True for the two blocks whose fields use the
    {value, evidence, notes} envelope (study_extraction and the record
    `extraction:` sections under `records:`). For those, every field MUST
    declare `evidence: required` or `evidence: optional`; a missing flag fails
    loudly at load. False for the bare-value blocks (llm_initial_check,
    llm_quality_check); evidence is meaningless there and not required (or
    recorded).

    Any key outside `_SECTION_KEYS` fails loudly, so a mistyped section key
    (`qaa:`, `labell:`) is a load error rather than a silently ignored line.
    Notes are not among them: they are not declared in the template at all.
    Every field carries its own `notes` slot in its envelope, and commentary
    about a whole study or record goes in that scope's reserved `notes` key.

    Returns list of section dicts, each with keys: section, label, qa,
    extraction_instruction, fields. Each field dict carries all twelve of
    variable, label, description, extraction_instruction, field_type, options,
    allow_other, evidence, role, required, canonical_reference,
    soft_canonicalisation — every key always present, defaulted to None or
    False where the template declared nothing, so a reader indexes rather than
    guards. See `load_template` for what each means.
    """
    parsed = []
    seen_variables = set()
    if not isinstance(sections, list):
        raise ValueError(
            f"a field block must be a list of sections, got "
            f"{type(sections).__name__}. Each section is a mapping with a "
            f"`section:` title and a `fields:` list."
        )
    for section in sections:
        # `section:` is the title every message about this section names it
        # by, so its absence is answered here rather than as a KeyError out of
        # the first read. The keys the mapping DOES declare are in the message,
        # which is what makes a misspelt title key visible as the cause.
        if not isinstance(section, dict):
            raise ValueError(
                f"every entry in a field block must be a section mapping with "
                f"a `section:` title and a `fields:` list, got "
                f"{type(section).__name__} ({section!r})."
            )
        if "section" not in section:
            raise ValueError(
                f"a section declares no `section:` title. The mapping "
                f"declares {sorted(section)}, and `section:` is required: it "
                f"is how the section is named in the rendered template and in "
                f"every load error about the fields inside it."
            )
        section_name = section["section"]
        unknown = sorted(set(section) - _SECTION_KEYS)
        if unknown:
            raise ValueError(
                f"Section {section_name!r}: unknown section key(s) {unknown}. "
                f"Only {sorted(_SECTION_KEYS)} are allowed."
            )
        section_label = _resolve_label(
            section.get("label"), default=section_name,
            what=f"Section {section_name!r}")
        # Optional `qa: true` marks the section as quality assessment. It is
        # PRESENTATION ONLY: render_template groups the flagged sections
        # under their own heading in the operational document and leaves them
        # out of the publication document; nothing else reads it. Never sent
        # to a model and not in field_catalogue_hash, so flipping it moves no
        # fingerprint but config_fp, via the whole-file template_hash.
        qa_flag = section.get("qa", False)
        if not isinstance(qa_flag, bool):
            raise ValueError(
                f"Section {section_name!r}: `qa:` must be true or false, got "
                f"{qa_flag!r}."
            )
        fields = []
        for f in section.get("fields") or []:
            field_dict = _parse_field(
                f, envelope=envelope, section_name=section_name)
            if field_dict["variable"] in seen_variables:
                raise ValueError(
                    f"Field {field_dict['variable']!r}: duplicate "
                    f"variable name in section {section_name!r}."
                )
            seen_variables.add(field_dict["variable"])
            fields.append(field_dict)

        parsed.append({
            "section": section_name,
            "label": section_label,
            "qa": qa_flag,
            "extraction_instruction": section.get("extraction_instruction") or None,
            "fields": fields,
        })

    return parsed


def iter_fields(sections):
    """Yield all field dicts from nested sections."""
    for section in sections:
        yield from section["fields"]


def required_field_names(sections):
    """Return the set of variable names in `sections` flagged `required: true`.

    The mark_complete gate uses this to enforce required-ness without the
    engine ever naming a specific field: the template declares which fields
    are required, per block. See the callers in meltiro.tools.
    """
    return {f["variable"] for f in iter_fields(sections) if f.get("required")}


def _validate_checker_context_fields(context_fields, record_fields, singular):
    """Validate `checker_context_fields` against the parsed record fields.

    `context_fields` is the ordered list from
    `records.<name>.checker_context_fields` (already shape-checked in
    `_parse_records`; may be empty). Enforces the whole-template rules a single
    field parse can't see:
      - every name must resolve to an existing record-scoped field (any field
        in the record's `extraction:` sections, quality-assessment ones
        included). The reserved scope-note key is not a field, so naming it
        here fails under that same rule;
      - duplicate names are rejected: a repeated entry would render a doubled
        component in the record label for no signal.

    Returns the list unchanged (order preserved), so the loaded template can
    expose it as `checker_context_fields`.
    """
    seen = set()
    for name in context_fields:
        if name in seen:
            raise ValueError(
                f"`records.{singular}.checker_context_fields` lists {name!r} "
                f"more than once. Each entry must appear exactly once."
            )
        seen.add(name)
    by_name = {}
    for f in iter_fields(record_fields):
        by_name.setdefault(f["variable"], f)
    for name in context_fields:
        if name not in by_name:
            raise ValueError(
                f"`records.{singular}.checker_context_fields` names {name!r}, "
                f"which is not a record-scoped field. Every entry must be an "
                f"existing field in `records.{singular}.extraction`."
            )
    return list(context_fields)


def _resolve_gate_ref(value, key, where, entity):
    """Resolve a scoped gate reference `<entity>.<variable>` to its bare
    variable name.

    A gate's `when_field` and `field` must be written as
    `<entity>.<variable>`, where `<entity>` is the record type declared as
    the `records:` block key, so a bare name has no silent record-scope
    binding to fall back on when a study field shares it. Record entity
    names are slug-validated and carry no dot, so the split is on the FIRST
    `.` and is unambiguous.

    The gate check runs inside a single record and sees only that record's
    sibling field values (see meltiro.validators.validate_gate_rules), so a
    `study.`-scoped controller is rejected explicitly rather than as a generic
    unknown name: study-level fields are simply not in scope. A bare name, a
    prefix that is not the declared entity, and a malformed path (no dot,
    empty segment, extra dots, internal whitespace) all fail loudly with their
    own message.

    Returns the bare variable name. The parsed gate stores that bare name, so
    `_validate_gates` and meltiro.validators.validate_gate_rules keep reading
    bare record-field names unchanged: the scoping is a load-time-resolution
    concern only.
    """
    text = value.strip()
    if "." not in text:
        raise ValueError(
            f"{where}: `{key}` is {value!r}, a bare field name. A gate "
            f"reference must be a scoped path `<entity>.<variable>` naming a "
            f"field on this template's record entity `{entity}`: write "
            f"`{entity}.{text}`."
        )
    prefix, _, variable = text.partition(".")
    malformed = (
        not prefix or not variable or "." in variable
        or any(c.isspace() for c in prefix)
        or any(c.isspace() for c in variable)
    )
    if malformed:
        raise ValueError(
            f"{where}: `{key}` is {value!r}, which is not a well-formed "
            f"scoped path. Use exactly `<entity>.<variable>` with a single "
            f"`.` and no empty or whitespace segments, e.g. "
            f"`{entity}.<variable>`."
        )
    if prefix == "study":
        raise ValueError(
            f"{where}: `{key}` is {value!r}, a `study.`-scoped path. "
            f"Study-scoped gate controllers are not supported: the gate check "
            f"runs inside a single record and sees only that record's sibling "
            f"field values, never study-level fields. Name a field on the "
            f"record entity `{entity}` (`{entity}.{variable}`) instead."
        )
    if prefix != entity:
        raise ValueError(
            f"{where}: `{key}` is {value!r}, but its entity prefix `{prefix}` "
            f"is not this template's record entity `{entity}`. Gate fields are "
            f"record-scoped; write `{entity}.{variable}`."
        )
    return variable


def _parse_gates(raw, entity):
    """Parse and validate the optional top-level `gates:` block.

    `gates:` is an optional list of cross-field gate rules. Each rule names a
    controlling record field (`when_field`), a gated record field (`field`),
    and the controlling values (`allowed_values`) under which the gated field
    is expected. At validation time (see
    meltiro.validators.validate_gate_rules) a record whose gated field carries
    a value while the controlling field's value is not among `allowed_values`
    earns a WARNING, never an error: the deterministic check flags a suspicious
    combination and the checker decides whether it is genuinely justified.

    `when_field` and `field` are scoped paths of the form `<entity>.<variable>`
    (see `_resolve_gate_ref`); `entity` is the record type declared as the
    `records:` block key. The prefix is validated here and stripped, so the
    parsed gate stores bare variable names and everything downstream stays
    unchanged.

    This is where a review declares its cross-field conditionals; the engine
    names no field itself. Shape only here: this validates each rule is
    well-formed and rejects a malformed one loudly (strict inputs, no silent
    drop). That `when_field` and `field` resolve to existing record-scoped
    fields is a whole-template check done in `load_template` via
    `_validate_gates`, once the record fields are parsed.

    Returns the parsed list of `{when_field, field, allowed_values}` dicts
    (bare variable names), empty when the block is absent or null.
    """
    if "gates" not in raw or raw["gates"] is None:
        return []
    block = raw["gates"]
    if not isinstance(block, list):
        raise ValueError(
            f"`gates:` must be a list of cross-field gate rules when present, "
            f"got {type(block).__name__}."
        )
    gates = []
    seen = set()
    for i, rule in enumerate(block):
        where = f"`gates[{i}]`"
        if not isinstance(rule, dict):
            raise ValueError(
                f"{where} must be a mapping with keys {sorted(_GATE_KEYS)}, "
                f"got {type(rule).__name__}."
            )
        unknown = sorted(set(rule) - _GATE_KEYS)
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {unknown}. Only "
                f"{sorted(_GATE_KEYS)} are allowed."
            )
        missing = sorted(_GATE_KEYS - set(rule))
        if missing:
            raise ValueError(
                f"{where} is missing required key(s) {missing}. Every gate "
                f"rule needs {sorted(_GATE_KEYS)}."
            )
        when_ref = rule["when_field"]
        gated_ref = rule["field"]
        for key, val in (("when_field", when_ref), ("field", gated_ref)):
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"{where}: `{key}` must be a non-empty string naming a "
                    f"scoped record field path `<entity>.<variable>`, got "
                    f"{val!r}."
                )
        when_field = _resolve_gate_ref(when_ref, "when_field", where, entity)
        gated_field = _resolve_gate_ref(gated_ref, "field", where, entity)
        if when_field == gated_field:
            raise ValueError(
                f"{where}: `when_field` and `field` both resolve to "
                f"{when_field!r}; a gate must relate two different fields."
            )
        allowed_raw = rule["allowed_values"]
        if not isinstance(allowed_raw, list) or not allowed_raw:
            raise ValueError(
                f"{where}: `allowed_values` must be a non-empty list of "
                f"controlling values, got {allowed_raw!r}."
            )
        allowed = []
        for v in allowed_raw:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"{where}: `allowed_values` entries must be non-empty "
                    f"strings, got {v!r}."
                )
            allowed.append(v.strip())
        pair = (when_field, gated_field)
        if pair in seen:
            raise ValueError(
                f"{where}: duplicate gate for `when_field` {when_field!r} and "
                f"`field` {gated_field!r}. Declare each field pair at most once."
            )
        seen.add(pair)
        gates.append({
            "when_field": when_field,
            "field": gated_field,
            "allowed_values": allowed,
        })
    return gates


def _validate_gates(gates, record_fields):
    """Validate parsed gate rules against the parsed record fields.

    Each gate's `when_field` and `field` must resolve to an existing
    record-scoped field (any field in the record's `extraction:` sections,
    quality-assessment ones included).
    Gates run on one record's sibling field values (see
    meltiro.validators.validate_gate_rules), so a name that is not a record
    field is a config error. Returns the list unchanged.
    """
    by_name = {}
    for f in iter_fields(record_fields):
        by_name.setdefault(f["variable"], f)
    for gate in gates:
        for role_key in ("when_field", "field"):
            name = gate[role_key]
            spec = by_name.get(name)
            if spec is None:
                raise ValueError(
                    f"`gates` names {name!r} (as `{role_key}`), which is not a "
                    f"record-scoped field. A gate relates two record fields."
                )
    return gates


def _resolve_role_fields(template):
    """Collect and validate the template's role-bearing fields.

    Enforces the whole-template constraints a single `_parse_field` call
    can't see:
      - a role must sit in its declared scope (see `_ROLE_SCOPES`):
        the study-scoped role (`summary`) only on `study_extraction`
        fields; a role on a check block, or in the wrong scope, fails
        loudly;
      - at most one field may claim each role.

    Returns `{role_name: field_dict}`. Per-field shape (known role, plain
    string type) was already enforced in `_parse_field`.
    """
    # The two blocks a role may sit in, mapped to the scope name they carry.
    scoped_blocks = {"study_fields": "study", "record_fields": "record"}
    # Blocks that may never carry a role. Roles wire study-level or
    # record-level extraction fields, not the process check blocks. A
    # quality-assessment section (`qa: true`) is an ordinary extraction
    # section, so a role there is judged by scope like any other: the flag is
    # presentation only and no validation consults it.
    role_free_blocks = [
        "initial_check_fields", "quality_check_fields",
    ]
    for block_key in role_free_blocks:
        for f in iter_fields(template[block_key]):
            if f.get("role"):
                raise ValueError(
                    f"Field {f['variable']!r} declares `role: {f['role']}` "
                    f"in block {block_key!r}, but roles are only allowed on "
                    f"study-level (study_extraction) or record-level "
                    f"(records) extraction fields."
                )

    role_fields = {}
    for block_key, scope in scoped_blocks.items():
        for f in iter_fields(template[block_key]):
            role = f.get("role")
            if not role:
                continue
            required_scope = _ROLE_SCOPES[role]
            if required_scope != scope:
                where = ("study-level (study_extraction)"
                         if required_scope == "study"
                         else "record-level (records)")
                raise ValueError(
                    f"Field {f['variable']!r} declares `role: {role}` on a "
                    f"{scope}-level field, but `role: {role}` is only allowed "
                    f"on {where} fields."
                )
            if role in role_fields:
                raise ValueError(
                    f"Duplicate `role: {role}`: fields "
                    f"{role_fields[role]['variable']!r} and {f['variable']!r} "
                    f"both declare it. At most one field may claim each role."
                )
            role_fields[role] = f
    return role_fields


def _compute_file_hash(path):
    """SHA-256 hash of the template file for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_template(template_path):
    """Load the extraction template and return structured field definitions.

    `template_path` is REQUIRED; the template is review-specific and comes
    from a config bundle (`ConfigBundle.template_path`). There is no
    CWD-relative default.

    Returns dict with keys:
      - record_entity: {singular, plural, description, extraction_instruction}
        naming the repeated-row entity (from the single entry under the
        required top-level `records:` mapping; the mapping key is the singular
        name). `extraction_instruction` is the optional record-level guidance
        (None when absent), rendered as the leading clause of add_record.
      - checker_context_fields: ordered list of record field names (from the
        record type's optional `checker_context_fields:` list; empty when
        absent) the checker shows after the record id to label a record. Every
        name is an existing record-scoped field; see
        meltiro.checker_prompts.build_record_context.
      - study_fields: list of section dicts from study_extraction,
        quality-assessment sections (`qa: true`) included
      - record_fields: list of section dicts from the record type's
        `extraction:` sections (under `records:`), quality-assessment
        sections (`qa: true`) included
      - initial_check_fields: list of section dicts from llm_initial_check
      - quality_check_fields: list of section dicts from llm_quality_check
      - role_fields: {role_name: field_dict} for every field declaring a
        `role:` (the study-scoped `summary` role). Empty when none are
        declared.
      - gates: list of `{when_field, field, allowed_values}` cross-field gate
        rules from the optional top-level `gates:` block (empty when absent).
        Each names a controlling record field, a gated record field, and the
        controlling values under which the gated field is expected; the
        validator turns a mismatch into a warning
        (meltiro.validators.validate_gate_rules).
      - template_hash: SHA-256 of the YAML file
      - template_path: resolved path to the template

    Each section dict has: section (title), label (human-facing name;
    defaults to the title), qa (bool; True marks the section as quality
    assessment, a presentation-only flag read by
    meltiro.render_template and nothing else), extraction_instruction,
    fields (list).

    Each field dict has all twelve of the keys below, ALWAYS PRESENT: a key
    the template did not declare is defaulted (to None, or to False for the
    two flags) rather than left out, so a reader indexes the shape instead of
    guarding every access. `variable` and `description` are the two the
    template must supply; a field declaring no `variable` is a load error
    naming its section (see `_parse_field`).

      - variable: stable identifier
      - label: human-facing name for presentation surfaces (workbook-style
        UIs, reports, tables). Defaults to the variable with underscores
        replaced by spaces and the first letter capitalised. Presentation
        only: it never enters any fingerprint.
      - description: published-table-ready description
      - extraction_instruction: optional edge-case guidance
      - field_type: "string" | "categorical" | "integer" | "number" |
        "boolean" | "year" | "date" | "string_list"
      - options: list (categorical) or None
      - allow_other: bool (categorical only; True accepts free text
        outside the option list, False is a hard enum)
      - evidence: "required" | "optional" (envelope blocks only;
        None for non-envelope)
      - role: "summary" | None; mechanical-wiring marker (see `role_fields`)
      - required: bool; when True the field must be non-null (envelope
        blocks) or present-and-non-null (bare check blocks) before
        mark_complete is accepted. Enforced generically by the dispatcher.
      - canonical_reference: the name of a reference list under the config
        bundle's `reference/`, or None. When set, the value must resolve to an
        exact entry name in that list (or to one of its aliases, which is
        stored canonicalised); `load_config_bundle` refuses a template naming
        a list the bundle does not provide.
      - soft_canonicalisation: bool; a CONSUMER-FACING declaration, inert in
        the engine. When True it tells UI and analysis consumers to
        auto-suggest and collapse this field's values against values entered
        earlier (a soft, growing vocabulary), with no hard validation. Only
        on free-value fields; mutually exclusive with canonical_reference.
        The engine takes no runtime action on it, it is never sent to a
        model, and it is excluded from the field-catalogue hash; config_fp
        still moves on any byte edit to the file via the whole-file
        template_hash.

    Notes are not declared in the template at all. Every field carries its own
    `notes` slot inside its envelope, and each scope (the study block, every
    record) carries a reserved `notes` key holding that scope's commentary, so
    `notes` is refused as a field variable name. Neither kind of note is
    validated or checked.

    Reference lists (the gauge list for the worked config) live in the config
    bundle's `reference/` directory; load them via
    meltiro.reference_lists.load_reference_lists().
    """
    path = Path(template_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = strict_load(f)

    _validate_top_level_keys(raw)

    # The repeated-row entity and its extraction sections both come from the
    # single entry under the top-level `records:` mapping; the loaded
    # template exposes them under the entity-neutral `record_entity` /
    # `record_fields` keys the rest of the engine reads.
    #
    # There are two envelope field blocks, not four: quality-assessment
    # sections are ordinary sections marked `qa: true` for the renderer, so
    # a QA field is extracted, validated, and checked exactly like any other
    # field of its scope.
    entity, record_extraction_raw, context_field_names = _parse_records(raw)
    template = {
        "record_entity": entity,
        "study_fields": _parse_sections(
            raw["study_extraction"], envelope=True),
        "record_fields": _parse_sections(
            record_extraction_raw, envelope=True),
        "initial_check_fields": _parse_sections(
            raw["llm_initial_check"], envelope=False),
        "quality_check_fields": _parse_sections(
            raw["llm_quality_check"], envelope=False),
        "template_hash": _compute_file_hash(path),
        "template_path": str(path.resolve()),
    }

    # The optional ordered list of record field names the checker shows
    # after the record id to label a record (see build_record_context),
    # validated against the now-parsed record fields, so the engine reads
    # the order from the template rather than a review-specific constant.
    template["checker_context_fields"] = _validate_checker_context_fields(
        context_field_names, template["record_fields"], entity["singular"])

    # Role-bearing fields (the `summary` role): validated across the whole
    # template (in its declared scope, at most one per role) and exposed
    # under `role_fields` so the pipeline can wire them mechanically, never by
    # field name.
    template["role_fields"] = _resolve_role_fields(template)

    # Cross-field gate rules (optional): the review declares which record
    # fields are only expected under which controlling-field values, and the
    # validator applies them generically (validate_gate_rules). The engine
    # names no gated field itself; every name comes from the template.
    template["gates"] = _validate_gates(
        _parse_gates(raw, entity["singular"]), template["record_fields"])

    return template
