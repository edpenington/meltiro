"""Tool definitions in direktoro's canonical tool shape, plus the dispatcher.

Ten tools the extractor can call, nine of which the final reviewer also gets.
Each tool's input_schema is a JSON Schema the API uses to coax the model into
well-formed args; richer validation (verbatim quotes, gate rules,
reference-list canonicalisation, type matches against the template) happens in
the dispatcher and lands in the `tool_result` payload.

The checker has a catalogue of its own, one tool wide: `record_verdict`, at the
bottom of this module. It is not dispatched here — `meltiro.checker` reads the
call off the response itself — and it is hashed apart from the two catalogues
above, for the reasons given where it is defined.

The two check blocks have one path each and one author each
-----------------------------------------------------------
`initial_check` and `quality_check` are self-assessments about how the
extraction went. Each is reachable through exactly one tool, and the engine
enforces when:

  - `record_initial_check` is the extractor's FIRST call. Until it lands, the
    dispatcher refuses every mutating call from the extractor: a report on
    the inputs must be written before they are extracted from, and the engine
    enforces the ordering rather than leaving it to the prompt.
  - `quality_check` is a REQUIRED argument of `mark_complete`, so there is no
    window in which a run is complete and unassessed.

The reviewer's catalogue carries neither `record_initial_check` nor any
check-block argument on `update_study`, so it cannot touch what the extractor
recorded. It records its own quality check through its own `mark_complete`,
under its own role key. See `meltiro.extraction_record` for the output shape.

Two address spaces, deliberately different
------------------------------------------
A check-block field has two paths, and they do not match. The dispatcher's
model-facing paths (`applied_fields`, `failed_fields`, `field_diffs`) address
the TOOL CALL, whose check-block properties are flat: `initial_check.<var>`,
`quality_check.<var>`. `validators.validate_extraction_output` addresses the
FILE, whose blocks are role-keyed: `initial_check.<role>.<var>`. Telling a
model to fix `quality_check.extractor.general_notes` would invite it to send
`{"extractor": {...}}`, which no tool here accepts. Each path names the thing
its reader can actually edit.

Validation policy is per field for the three field-writing tools
(`update_study`, `add_record`, `update_record`): each field is validated on
its own, the fields that pass are written, and the result reports `ok` when
every field applied, `partial` when some applied and some failed, and
`validation_failed` when none applied. Only the failed fields need to be
resubmitted.

A STRUCTURAL problem with the call itself addresses no field validly, so it
stays all-or-nothing and returns `validation_failed`. All three refuse a field
map that is not an object. `add_record` and `update_record` additionally
refuse an EMPTY or missing one, and `update_record` an unknown `record_id`:
each of those calls exists only to write fields, so one carrying none has
nothing to do. `update_study` is the exception and takes an empty map as `ok`,
because its optional `notes` argument is a legitimate thing to send on its own
— a note-only call writes the study scope note and no field, and refusing it
for having no fields would refuse the call it was written to make.

Warnings (e.g. category-gate violations) are informational only; the call
still applies.

The dispatcher does not own the budget counter; the orchestrator passes the
remaining budget in via the `meta` dict and the dispatcher echoes it back
under the `_tool_call_budget_remaining` key. The leading underscore marks it
UI-only: `result_to_model_text` (meltiro.session, shared by the live loop
and replay) strips every underscore-prefixed key from the model-facing
content, so the budget reaches the event log but never the model — the cap
rides in no fingerprint, and a cap-derived number in model-visible content
would let two runs sharing a config_fp show the model different inputs.
"""

import difflib
import json
import re

from meltiro.extraction_record import (
    NOTES_KEY,
    ROLE_EXTRACTOR,
    ROLE_REVIEW,
    ROLES,
)
from meltiro.template import iter_fields, required_field_names

from meltiro.validators import (
    _check_value_type,
    validate_envelope,
    validate_gate_rules,
)


# The tools that can change EXTRACTED CONTENT — the study fields and the
# records. Not "writes to the file": `record_initial_check` and
# `mark_complete` also change `extraction_output.json`, but their check
# blocks are the model's account of its own run, not anything read out of
# the paper. Callers asking "did the extraction move" must test membership
# here, NOT "is this tool something other than mark_complete": the read-only
# `view_*` tools answer `status: ok`, so that test books a read as a write.
# Enumerating the MUTATING tools makes a tool added later default to
# non-mutating, the safe direction: it can under-trigger a post-hoc check,
# never fabricate an edit. A test pins this set against `dispatch`'s
# catalogue, so a new tool cannot join without being classified.
MUTATING_TOOLS = frozenset({
    "update_study", "add_record", "update_record", "remove_record",
})

# Tools the extractor may not call before `record_initial_check` has landed.
# Every mutation, plus `mark_complete`: a template that declares no REQUIRED
# initial-check field would otherwise let a run complete having never made the
# call at all, because mark_complete's own gate only checks required fields.
_INITIAL_CHECK_GATED_TOOLS = MUTATING_TOOLS | {"mark_complete"}

# Handlers whose behaviour depends on which role is calling. Kept as an
# explicit set rather than given to every handler, so a handler that has no
# business knowing the role cannot quietly start branching on it.
_ROLE_AWARE_HANDLERS = frozenset({
    "record_initial_check", "mark_complete",
    "view_summary", "view_study_fields",
})

# Which tool owns each check block, for the message a model gets when it
# addresses one to a tool that does not take it. One sentence each, naming the
# tool.
_CHECK_BLOCK_HOME = {
    "initial_check": (
        "Record it with `record_initial_check`, which is your first call of "
        "the run."
    ),
    "quality_check": (
        "Pass it as the `quality_check` argument of `mark_complete`, which "
        "requires it."
    ),
}

# The top-level arguments each tool accepts, mirroring the `properties` of
# its `input_schema`. Anything else is REFUSED rather than ignored: a field
# map addressed to a name the tool does not read would otherwise be answered
# `ok` having stored nothing, with no way for the model to see or correct it.
#
# `record_initial_check` is deliberately absent: its top-level arguments ARE
# its fields (the schema is flat), so an unrecognised name there is a field
# name, answered through the per-field `unknown_field` path with a spelling
# hint. A test pins this map against the tool definitions.
_TOOL_ARGUMENTS = {
    "update_study": frozenset({"study", NOTES_KEY}),
    "add_record": frozenset({"fields", NOTES_KEY}),
    "update_record": frozenset({"record_id", "fields", NOTES_KEY}),
    "remove_record": frozenset({"record_id", "reason"}),
    "mark_complete": frozenset({"quality_check"}),
    "abandon_extraction": frozenset({"reason"}),
    "view_summary": frozenset(),
    "view_study_fields": frozenset(),
    "view_record": frozenset({"record_id"}),
}

# Where each tool's field map belongs, appended to the refusal so a model that
# sent one under the wrong name can retry correctly rather than guess. The
# argument differs per tool (`update_study` takes `study`, the record tools
# take `fields`), which is exactly the confusion this sentence answers. Tools
# that carry no field map have no entry and get the argument list alone.
_FIELD_MAP_HOME = {
    "update_study": (
        "Study-level fields go in the `study` argument, as a map of variable "
        "name -> {value, evidence, notes}."
    ),
    "add_record": (
        "Record fields go in the `fields` argument, as a map of variable "
        "name -> {value, evidence, notes}."
    ),
    "update_record": (
        "Record fields go in the `fields` argument, as a map of variable "
        "name -> {value, evidence, notes}."
    ),
    "mark_complete": (
        "Your quality-check answers go in the `quality_check` argument, as a "
        "map of variable name -> value."
    ),
}


def _entity(template):
    """The template's record-entity metadata {singular, plural, description}.

    Tool descriptions and LLM-facing messages are rendered from these so the
    engine stays generic while the model reads review-natural wording (for
    the worked config, "relationship" / "relationships").
    """
    return template["record_entity"]


def _record_id_pattern(ent):
    """JSON-schema pattern an auto-assigned record_id matches for this entity.

    Record ids are `<entity>_<n>` (e.g. `relationship_1`), so the pattern is
    anchored on the template's singular entity noun. The entity name is
    regex-escaped so a noun with a special character can never widen or break
    the pattern. This is model-facing guidance in the tool schema (the API
    coaxes well-formed args from it); the dispatcher's `has_record` check is
    the authoritative guard.
    """
    return f"^{re.escape(ent['singular'])}_[0-9]+$"


def _record_id_examples(ent):
    """Two illustrative record ids for this entity, e.g. `relationship_1,
    relationship_2`, for tool descriptions."""
    s = ent["singular"]
    return f"{s}_1, {s}_2"


# Every declared template field is an ordinary extractable field: it appears in
# the LLM tool schema and is validated per its scope (the checker fans out over
# every study and record field, quality-assessment ones included).
# The one value the pipeline owns, the study id, is not a template field at
# all: the engine reads it from the bundle manifest and records it in the run's
# output metadata (run.json `study_id`), never as a per-record extracted field.


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
#
# Each tool's input_schema is built from the extraction template. Every
# field is enumerated as a named property with its description,
# extraction_instruction, and (for categorical fields) `enum` constraint.
# `additionalProperties: false` at the field-map level rejects unknown
# variables at the API layer; the dispatcher's runtime checks of unknown
# variables become defence-in-depth.
#
# The one exception is `update_record`, whose field properties are slimmed
# to bare {value, evidence, notes} envelopes with no description and no enum
# list.
# `add_record` carries the authoritative record field reference and rides in
# the same request, so re-emitting the whole catalogue on update_record costs
# ~3.3k tokens (measured over the rendered catalogue) for signal the model
# already has. Server-side validation is per-field and does not read the
# schema, so the slim schema validates identically.
#
# The record id is the sole record-level field the model never sets: it is
# auto-assigned by `add_record`, so it never reaches the schema and an attempt
# to set it lands in the generic `unknown_field` branch.


# The field-note slot inside every envelope. No `description`: the framing
# is stated ONCE per block, in the parent object's description, rather than
# paid for on every field of every request.
_NOTES_SUBSCHEMA = {"type": ["string", "null"]}

# The once-per-block framing for the field-note slot.
_FIELD_NOTES_FRAMING = (
    "To record information justifying or explaining a field's value that is "
    "not a verbatim quote, put it in that field's `notes`. A note is never "
    "validated and never substitutes for required evidence; leave it null "
    "when there is nothing to add."
)


def _scope_notes_schema(what):
    """The reserved scope-note argument on a block-writing tool.

    `what` names the scope in model-facing wording ("study", "relationship").
    A scope note is the extractor's holistic commentary about that whole
    scope, as opposed to a field note, which explains one value. It is never
    validated and is not shown to the checker (the checker judges one field at
    a time, and feeding it the extractor's whole-record reasoning is exactly
    the correlation its narrow context exists to prevent).
    """
    return {
        "type": ["string", "null"],
        "description": (
            f"Optional free-text note about this {what} as a whole: "
            f"observations, caveats, and reasoning that inform several fields "
            f"rather than justifying one. Notes explaining a single field's "
            f"value belong in that field's own `notes` slot. Omit to leave any "
            f"existing note unchanged; pass null to clear it."
        ),
    }


def _field_value_subschema(field):
    """JSON Schema for a single field's `value`. Null allowed for every
    type except boolean (matches the runtime validator)."""
    ft = field["field_type"]
    if ft == "categorical" and field.get("options"):
        if field.get("allow_other"):
            # allow_other: the option list is a set of typical values shown
            # in the description; the machine surface is free text. The model
            # may return any string, and the checker is briefed to flag a
            # free-text value where a listed option would fit.
            return {"type": ["string", "null"]}
        # Hard enum: the API rejects anything outside the list (plus null).
        return {"enum": list(field["options"]) + [None]}
    if ft == "boolean":
        return {"type": "boolean"}
    if ft == "integer":
        return {"type": ["integer", "null"]}
    if ft == "number":
        return {"type": ["number", "null"]}
    if ft == "year":
        # Year is an integer, so advertise it as such. A ["number","null"]
        # schema would let a model-emitted 2019.0 pass the API layer then
        # fail the runtime integer validator as type_mismatch, an avoidable
        # failure that feeds the retry loop. The validator coerces integral
        # floats (2019.0 -> 2019) so the two agree.
        return {"type": ["integer", "null"]}
    if ft == "string_list":
        return {
            "anyOf": [
                {"type": "null"},
                {"type": "array", "items": {"type": "string"}},
            ],
        }
    # string, date: string-shaped
    return {"type": ["string", "null"]}


def _slim_value_subschema(field):
    """`value` subschema for a slimmed (update_record) envelope.

    Same type coaxing as `_field_value_subschema` with one change: a
    categorical field's `enum` option list is dropped and the value
    collapses to a plain nullable string. The full enumerated catalogue
    (options, descriptions, typical-values tails) rides in `add_record`,
    present in the same request, so re-emitting the enum here is redundant.
    Non-categorical types (integer, boolean, year, string_list, ...) keep
    their type so the model is still coaxed toward the right JSON shape.
    """
    if field["field_type"] == "categorical":
        return {"type": ["string", "null"]}
    return _field_value_subschema(field)


def _field_description(field):
    """Compose the property description from the field's published
    `description` plus any `extraction_instruction`. allow_other option
    lists, reference-list notes, and evidence-optional notes append. Parts
    are separated by " · "; the separator alone signals "additional note",
    with no explicit bracket labels.

    Section-level guidance is *not* prepended here; it lives at the
    parent-object description level (assembled by `_section_guidance`)
    so it appears once per section rather than blanketing every field.
    """
    parts = []
    parts.append((field.get("description") or "").strip())
    instr = (field.get("extraction_instruction") or "").strip()
    if instr:
        parts.append(instr)
    if (field.get("field_type") == "categorical"
            and field.get("options")
            and field.get("allow_other")):
        # allow_other: the machine surface is free text (no `enum`), so the
        # options ride in the description with an "Other (specify)"
        # affordance. Hard enums omit this tail: their options are the schema
        # `enum`. "Other (specify)" is presentation only — never added to the
        # option set or schema; the free text goes directly in the value.
        listed_str = "; ".join(str(o) for o in field["options"])
        parts.append(f"Options: {listed_str}; Other (specify)")
    if field.get("canonical_reference"):
        # Reference fields are a strict closed set: names must come from
        # the reference list rendered into the system prompt; off-list
        # values are rejected by the validator. A string_list field carries
        # a real JSON array of names, each validated independently.
        if field.get("field_type") == "string_list":
            parts.append(
                f"Provide a JSON array; each element must be an exact name "
                f"from the {field['canonical_reference']} reference list "
                f"shown in your system prompt. Off-list or duplicate "
                f"entries are rejected."
            )
        else:
            parts.append(
                f"Value must be an exact name from the "
                f"{field['canonical_reference']} reference list shown in "
                f"your system prompt; off-list values are rejected."
            )
    if field.get("evidence") == "optional":
        parts.append("Evidence optional")
    else:
        parts.append("Evidence required")
    return " · ".join(p for p in parts if p)


def _envelope_property_schema(field):
    """Schema for ONE envelope-wrapped field: {value, evidence, notes}.

    The evidence string carries all three concerns (verbatim quote, image
    reference, interpretive prose) in one field, using inline tags. Format:

        <q>verbatim paper text</q>: copied character-for-character from
        the paper; verbatim-checked.
        <img>label</img>: figure/table reference
        by cropped-image filename stem (e.g. <img>table_03</img>).
        Anything outside tags is free-text reasoning / synthesis.

    Multiple <q> and <img> blocks may appear in any order, intermixed
    with prose. Required-evidence fields need at least one <q> or <img>
    when value is non-null; optional-evidence fields may carry pure
    prose, empty string, or null.

    `notes` is the field note: whatever justifies or explains this field's
    value and is NOT a verbatim quote. It is never validated, never
    quote-checked, and never counts toward satisfying `evidence: required`;
    the checker does read it. It is `required` in the JSON-Schema sense and
    nullable, exactly as `evidence` is: strict structured-output modes reject
    a property that is absent from `required`, so a nullable-and-required
    slot is what makes "no note" expressible everywhere.
    """
    return {
        "type": "object",
        "description": _field_description(field),
        "properties": {
            "value": _field_value_subschema(field),
            # No `description` on `evidence`; the full format spec
            # (verbatim <q> tags, <img> labels, prose, required-vs-optional
            # rules) lives in the evidence passage of the engine's
            # extractor prompt. Repeating it on every envelope field
            # cost ~9k tokens.
            "evidence": {"type": ["string", "null"]},
            "notes": _NOTES_SUBSCHEMA,
        },
        "required": ["value", "evidence", "notes"],
        "additionalProperties": False,
    }


def _slim_envelope_property_schema(field):
    """Slimmed {value, evidence, notes} envelope for update_record.

    The envelope shape is identical to `_envelope_property_schema` (a
    `value` slot, an `evidence` slot, a `notes` slot, all three required, no
    additional keys), but the per-field `description` and the categorical
    `enum` list are dropped. update_record revises fields on a record that
    `add_record` already described in the same request, so the field reference
    does not need to be re-emitted here. Server-side validation is per-field
    and does not read the schema, so an update validates exactly as an add
    would.
    """
    return {
        "type": "object",
        "properties": {
            "value": _slim_value_subschema(field),
            "evidence": {"type": ["string", "null"]},
            "notes": _NOTES_SUBSCHEMA,
        },
        "required": ["value", "evidence", "notes"],
        "additionalProperties": False,
    }


def _bare_value_property_schema(field):
    """Schema for ONE non-enveloped field (initial_check / quality_check
    fields are plain {variable: value} maps, no envelope)."""
    schema = dict(_field_value_subschema(field))
    schema["description"] = _field_description(field)
    return schema


def _section_guidance(sections):
    """Render once-per-section guidance for a block of sections.

    Returns a string of the form:

        Section guidance:
        - <Section A>: <extraction_instruction>
        - <Section B>: <extraction_instruction>

    Only sections with a non-empty `extraction_instruction` are
    included. Returns "" if no section in the block carries one.
    """
    bullets = []
    for section in sections:
        instr = (section.get("extraction_instruction") or "").strip()
        if not instr:
            continue
        name = section.get("section") or ""
        head = f"{name}: " if name else ""
        bullets.append(f"- {head}{instr}")
    if not bullets:
        return ""
    return "Section guidance:\n" + "\n".join(bullets)


def _properties_from_sections(sections, *, envelope, exclude_vars=(),
                              slim=False):
    """Walk a template block (study_fields, record_fields,
    initial_check_fields, ...) and emit a {variable: property_schema}
    map suitable for use as a JSON Schema `properties` block.

    Every declared field is offered to the model: no field is pipeline-managed
    or hidden from the schema, and an allow_other field's free text goes
    directly in its own value. The `exclude_vars` argument is the only
    exclusion.

    `slim=True` (envelope blocks only) emits bare {value, evidence, notes}
    envelopes with no per-field description and no enum list; used by
    update_record, whose field reference lives authoritatively in
    add_record. Ignored when `envelope` is False.
    """
    props = {}
    for section in sections:
        for f in section["fields"]:
            var = f["variable"]
            if var in exclude_vars:
                continue
            if envelope:
                if slim:
                    props[var] = _slim_envelope_property_schema(f)
                else:
                    props[var] = _envelope_property_schema(f)
            else:
                props[var] = _bare_value_property_schema(f)
    return props


def _block_desc(base, sections):
    """A block description with the sections' own guidance appended."""
    guidance = _section_guidance(sections)
    return f"{base}\n\n{guidance}" if guidance else base


def _record_initial_check_tool(template):
    """The extractor's mandatory opening call.

    Properties are FLAT (one per declared initial-check variable) rather than
    nested under a block key: the tool name already says which block this is,
    and there is no sibling argument for them to be distinguished from.

    Template-declared `required: true` fields become JSON-Schema `required`,
    so the API coaxes a complete answer on the first attempt rather than
    leaving the dispatcher to reject an incomplete one.
    """
    props = _properties_from_sections(
        template["initial_check_fields"], envelope=False)
    required = sorted(required_field_names(template["initial_check_fields"]))
    schema = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return {
        "name": "record_initial_check",
        "description": _block_desc(
            "Record your pre-extraction check of the supplied inputs. This "
            "must be your FIRST tool call: until it lands, every call that "
            "would change the extraction output is refused. Report what you "
            "find in the material you were handed before you extract "
            "anything from it, so the report describes the inputs rather "
            "than being reconstructed afterwards. It is recorded once, "
            "attributed to you, and is not editable afterwards.",
            template["initial_check_fields"],
        ),
        "input_schema": schema,
    }


def _update_study_tool(template):
    study_sections = template["study_fields"]
    study_props = _properties_from_sections(study_sections, envelope=True)

    return {
        "name": "update_study",
        "description": (
            "Set or update study-level extraction fields. Most "
            "extraction fields carry an evidence envelope: "
            "{value, evidence: str | null, notes: str | null} where the "
            "evidence string mixes "
            "<q>verbatim quote</q> blocks, <img>label</img> references, and "
            "brief interpretive prose. Fields not included in this call are "
            "left unchanged. Validation is per-field: if one field fails, "
            "the others still apply; see `applied_fields` and "
            "`failed_fields` on the result. Only the failed fields need to "
            "be resubmitted. All allowed variables, their descriptions, and "
            "(where applicable) the allowed value sets are enumerated in the "
            "properties below. This tool does NOT carry the initial or "
            "quality check: those are `record_initial_check` and "
            "`mark_complete` respectively."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "study": {
                    "type": "object",
                    "description": _block_desc(
                        "Study-level fields. Each property is an evidence "
                        "envelope of the form {value, evidence, notes}; see "
                        "the individual property descriptions for type and "
                        "evidence requirements. " + _FIELD_NOTES_FRAMING,
                        study_sections,
                    ),
                    "properties": study_props,
                    "additionalProperties": False,
                },
                "notes": _scope_notes_schema("study"),
            },
            # No `required` list — the one field-writing tool without one. A
            # call carrying only `notes` is a complete act (a scope note with
            # no field change), and requiring `study` would call it
            # malformed. The dispatcher's argument guard, not this list,
            # refuses a field map sent under any other name.
            "additionalProperties": False,
        },
    }


def _add_record_tool(template):
    ent = _entity(template)
    record_sections = template["record_fields"]
    field_props = _properties_from_sections(record_sections, envelope=True)
    fields_desc = (
        f"{ent['singular'].capitalize()}-level fields as evidence envelopes "
        "of the form {value, evidence, notes}; see the individual property "
        "descriptions for type and evidence requirements. "
        + _FIELD_NOTES_FRAMING
    )
    guidance = _section_guidance(record_sections)
    if guidance:
        fields_desc = f"{fields_desc}\n\n{guidance}"
    # Required record fields are template-declared (`required: true`), not a
    # hardcoded engine allowlist. Empty is possible (a config need not
    # require any); drop the "must include" clause then.
    required_names = sorted(required_field_names(template["record_fields"]))
    minimum = (f" Must include {', '.join(required_names)} at minimum."
               if required_names else "")
    # The optional record-level `extraction_instruction` leads the description:
    # it is where "what counts as one <entity>, do not combine them" guidance
    # lives, and it rides in tool_set_hash.
    instr = (ent.get("extraction_instruction") or "").strip()
    lead = (f"What counts as one {ent['singular']}: {instr} " if instr else "")
    return {
        "name": "add_record",
        "description": (
            f"{lead}Append a new {ent['singular']} record "
            f"({ent['description']}). record_id is assigned automatically "
            f"({_record_id_examples(ent)}, ... in call order); do NOT include "
            f"it.{minimum} Validation is per-field: if one field fails, the "
            "others still apply; see `applied_fields` and `failed_fields` on "
            f"the result. The {ent['singular']} is created with the fields "
            "that passed and its record_id is returned, so only the failed "
            "fields need to be resubmitted, with update_record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "description": fields_desc,
                    "properties": field_props,
                    "additionalProperties": False,
                },
                "notes": _scope_notes_schema(ent["singular"]),
            },
            "required": ["fields"],
            "additionalProperties": False,
        },
    }


def _update_record_tool(template):
    ent = _entity(template)
    record_sections = template["record_fields"]
    # Slim envelopes: no per-field description, no enum list, and no
    # section guidance. add_record carries the authoritative record field
    # reference (and the section guidance) and rides in the same request;
    # re-emitting all of it here costs ~3.3k tokens of pure duplication.
    field_props = _properties_from_sections(
        record_sections, envelope=True, slim=True)
    fields_desc = (
        f"{ent['singular'].capitalize()} fields to revise. Each is an "
        "evidence envelope {value, evidence, notes}; only include fields you "
        f"want to change. See add_record for the full {ent['singular']} field "
        "reference (meanings, allowed values, evidence requirements). "
        + _FIELD_NOTES_FRAMING
    )
    return {
        "name": "update_record",
        "description": (
            f"Revise one or more fields on an existing {ent['singular']}. "
            "record_id must reference an existing entry "
            f"({_record_id_examples(ent)}, ...). Fields not included are "
            "unchanged. Validation is per-field: if one field fails, the "
            "others still apply; see `applied_fields` and `failed_fields` on "
            "the result. Only the failed fields need to be resubmitted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string",
                              "pattern": _record_id_pattern(ent)},
                "fields": {
                    "type": "object",
                    "description": fields_desc,
                    "properties": field_props,
                    "additionalProperties": False,
                },
                "notes": _scope_notes_schema(ent["singular"]),
            },
            "required": ["record_id", "fields"],
            "additionalProperties": False,
        },
    }


def _remove_record_tool(template):
    ent = _entity(template)
    return {
        "name": "remove_record",
        "description": (
            f"Remove a {ent['singular']} that was added in error. Provide a "
            "rationale; this is recorded in the session audit log but not in "
            f"the final extraction output. Other {ent['plural']} are NOT "
            "renumbered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string",
                              "pattern": _record_id_pattern(ent)},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["record_id", "reason"],
            "additionalProperties": False,
        },
    }


def _mark_complete_tool(template, role=ROLE_EXTRACTOR):
    """Declare completion AND record this role's quality check, in one call.

    The quality check is a required argument rather than something to have
    sent earlier, so "the run is complete" and "here is how it went" cannot
    come apart. Both roles get the same fields; each role's answers are
    stored under its own key.

    The two roles differ in what a bad answer costs. For the EXTRACTOR the
    quality check is part of the completeness gate: a missing or invalid field
    fails the call and the extractor re-calls with it fixed, exactly as it
    already does for a missing required study field. For the REVIEWER
    `mark_complete` stays unconditional — it is the only exit from a
    fresh-context loop with no replay, and making termination contingent on
    validation would let a reviewer that cannot phrase its quality check spin
    against the bound instead of finishing. Its invalid fields are dropped
    with a warning and the review ends.
    """
    qc_props = _properties_from_sections(
        template["quality_check_fields"], envelope=False)
    qc_required = sorted(
        required_field_names(template["quality_check_fields"]))
    qc_schema = {
        "type": "object",
        "description": _block_desc(
            "Your own assessment of how this extraction went, recorded "
            "under your role and never overwritten by another. Required: "
            "completing the extraction and reporting on it are one act.",
            template["quality_check_fields"],
        ),
        "properties": qc_props,
        "additionalProperties": False,
    }
    if qc_required:
        qc_schema["required"] = qc_required

    if role == ROLE_REVIEW:
        base = (
            "Conclude the review. Call it once you are satisfied the "
            "extraction output is correct and complete, having made any "
            "revisions you judged necessary. Calling it ends the review."
        )
    else:
        base = (
            "Declare the extraction finished and record your quality check. "
            "Only call it once you are confident the extraction output is "
            "complete and every field is justified by the evidence recorded "
            "for it. Calling it ends your turn at the extraction output."
        )

    return {
        "name": "mark_complete",
        "description": base,
        "input_schema": {
            "type": "object",
            "properties": {
                "quality_check": qc_schema,
            },
            "required": ["quality_check"],
            "additionalProperties": False,
        },
    }


_ABANDON_EXTRACTION_TOOL = {
    "name": "abandon_extraction",
    "description": (
        "Last resort. Declare that a valid extraction cannot be produced "
        "honestly from these inputs (for example the paper text is "
        "unreadable, or it reports none of the records this review requires) "
        "and end the run with status failed_validation. The extraction output "
        "so far is kept for inspection. Do NOT use this to escape a field "
        "that is merely hard: for that, keep working, or let your best "
        "supported answer stand. Only call this when no "
        "amount of further work could yield a valid extraction."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "minLength": 1,
                # Names no file. A tool description is hashed verbatim into
                # `tool_set_hash`, so every filename that appears in one is a
                # future fingerprint move waiting on a layout change. Saying
                # where the reason goes, rather than which file it lands in,
                # tells the model everything it needs and cannot go stale.
                "description": (
                    "Required. A concrete explanation of why no valid "
                    "extraction is possible from these inputs. Recorded in "
                    "the session diagnostics and the run log for the operator."
                ),
            },
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}


def _view_summary_tool(template):
    ent = _entity(template)
    return {
        "name": "view_summary",
        "description": (
            "Return a compact snapshot of the current extraction output so "
            "you can answer 'what have I added so far' without re-reading the "
            "trace. Reports filled/total counts for study fields and each "
            f"metadata block, plus a list of {ent['plural']} with their "
            "record id, a short context label, filled/total counts, and that "
            f"{ent['singular']}'s note. The study's own note is included too. "
            "Cheap to call; "
            "use whenever you want to reorient. For full envelopes call "
            "`view_study_fields` or `view_record`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }


_VIEW_STUDY_FIELDS_TOOL = {
    "name": "view_study_fields",
    "description": (
        "Return the full current contents of the study-level blocks: "
        "`study` envelopes (each with its value, evidence, and field note) "
        "and the study's own `notes`, plus `initial_check` and "
        "`quality_check` bare-value maps. Use when you need to inspect what "
        "is already in the record; typically after `view_summary` flags "
        "something worth checking."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _view_record_tool(template):
    ent = _entity(template)
    return {
        "name": "view_record",
        "description": (
            f"Return the full envelope for one already-added "
            f"{ent['singular']}, including its own note and every field's "
            f"note. Use when `view_summary` lists a {ent['singular']} you "
            "want to inspect before deciding whether to update or remove it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string",
                              "pattern": _record_id_pattern(ent)},
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
    }


def get_tool_definitions(template, role=ROLE_EXTRACTOR):
    """Build the list of tool definitions for one role.

    Schemas are derived from `template` so editing the extraction
    template propagates to tool_set_hash, and the model sees enum
    constraints + descriptions at the point of use. Order is stable so
    tool_set_hash is deterministic given a template.

    The reviewer's catalogue is the extractor's minus `record_initial_check`:
    the initial check is a report on the inputs made before extraction
    begins, and the review stage has no honest moment for one. Everything
    else is shared, `mark_complete` included — the reviewer records its own
    quality check through it (see `_mark_complete_tool` for how the roles
    differ in what a bad answer costs).
    """
    tools = []
    if role == ROLE_EXTRACTOR:
        tools.append(_record_initial_check_tool(template))
    tools.extend([
        _update_study_tool(template),
        _add_record_tool(template),
        _update_record_tool(template),
        _remove_record_tool(template),
        _mark_complete_tool(template, role=role),
        _ABANDON_EXTRACTION_TOOL,
        _view_summary_tool(template),
        _VIEW_STUDY_FIELDS_TOOL,
        _view_record_tool(template),
    ])
    return tools


def all_tool_definitions(template):
    """Every role's catalogue, keyed by role.

    The hashed and reported unit. The two catalogues genuinely differ — the
    reviewer has no `record_initial_check`, and `mark_complete` is described
    to each role in its own terms — so hashing one alone would leave
    model-facing schema text out of every fingerprint. One combined value
    keeps a single tool-set component while covering all of it: an edit to
    either catalogue moves `config_fp` and `review_fp` together, which is the
    honest reading when both stages read from one template.
    """
    return {role: get_tool_definitions(template, role=role) for role in ROLES}


def canonical_tool_set_json(template):
    """Canonical JSON serialisation of the tool catalogues, for tool_set_hash."""
    return json.dumps(
        all_tool_definitions(template),
        sort_keys=True, separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# The checker's verdict tool
# ---------------------------------------------------------------------------

# The two verdicts, in the order the schema offers them. A tuple rather than a
# set so the rendered enum is deterministic: it is hashed into `checker_fp`,
# and a set's iteration order would move that fingerprint between runs of one
# unchanged tree. `meltiro.checker` derives its accepted-verdict set from this,
# so the vocabulary the schema advertises and the one the reader enforces
# cannot drift apart.
CHECKER_VERDICTS = ("ok", "challenge")

CHECKER_VERDICT_TOOL_NAME = "record_verdict"

# The checker's whole catalogue: one tool, carrying one verdict on one field.
# Two properties and no third: a verdict and the one short sentence behind it
# are the whole of what a check is bought for, and every one of them is read by
# the extractor or the reviewer that received it.
#
# Deliberately NOT part of `ROLES` / `all_tool_definitions` above. Those
# catalogues are derived from the extraction template and hash into `config_fp`
# and `review_fp`; this schema is fixed, holds no template content, and belongs
# to `checker_fp` alone. Keeping it separate means an edit here moves the
# checker's fingerprint and nothing else, which is the honest reading: it
# changes what the checker may answer, and touches neither the extractor's
# tools nor the reviewer's.
_CHECKER_VERDICT_TOOL = {
    "name": CHECKER_VERDICT_TOOL_NAME,
    "description": (
        "Record your verdict on the field you were asked about. This is how a "
        "verdict is given: a reply that calls no tool records nothing."
    ),
    "input_schema": {
        "type": "object",
        # `rationale` is declared before `verdict` because a model filling this
        # schema generates its properties in declaration order, and the
        # checker is asked to work through the evidence and let the verdict
        # follow from it. Declaring `verdict` first would invite a verdict
        # chosen up front and justified afterwards.
        "properties": {
            "rationale": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Required. One short sentence working through whether the "
                    "evidence supports the value. Write it before choosing "
                    "the verdict and let the verdict follow from it. It is "
                    "read by the extractor, so it has to carry its point "
                    "succinctly."
                ),
            },
            "verdict": {
                "type": "string",
                "enum": list(CHECKER_VERDICTS),
                "description": (
                    "Required. `ok` when the evidence supports the value, "
                    "either directly or by a derivation checkable from the "
                    "quote. `challenge` when the evidence genuinely does not "
                    "support it — the value contradicts the evidence, asserts "
                    "a number or category the quote does not justify, or the "
                    "quote is unrelated to the claim. Give `challenge` only "
                    "if you expect the extractor to change the value or the "
                    "evidence; if working through the rationale leaves you "
                    "content with the value, the verdict is `ok`."
                ),
            },
        },
        "required": ["rationale", "verdict"],
        "additionalProperties": False,
    },
}


def checker_tool_definitions():
    """The checker's tool catalogue: its verdict, and nothing else.

    A list of one, returned as a list because that is what the provider
    adapters take and what a second checker tool would extend.
    """
    return [_CHECKER_VERDICT_TOOL]


def canonical_checker_tool_json():
    """Canonical JSON of the checker's tool catalogue, for `checker_fp`.

    The twin of `canonical_tool_set_json` for the checker's own schema, kept
    apart from it for the reason given on `_CHECKER_VERDICT_TOOL`: the two
    hash into different fingerprints.
    """
    return json.dumps(
        checker_tool_definitions(),
        sort_keys=True, separators=(",", ":"),
    )


def _misplaced_scope_note_hint(var, tool_name):
    """Targeted hint for a model that put the reserved `notes` key inside a
    field map. It is not a field there; it is a sibling argument of the field
    map on the same tool call, so name the correction rather than let the
    generic unknown-field message send the model hunting for a typo."""
    if var != NOTES_KEY:
        return ""
    return (
        f" Note: `{NOTES_KEY}` is not a field. Pass the scope note as the "
        f"top-level `{NOTES_KEY}` argument of {tool_name}, alongside the "
        f"field map, or put a per-field note in that field's own `notes` "
        f"slot inside its envelope."
    )


def _suggest_closest_field(unknown, candidates, n=3, cutoff=0.6):
    """Use difflib to suggest the closest valid field names.

    Returns a short string like " Did you mean: 'sample_size', 'sample_n'?"
    suitable for appending to an unknown_field error message. Empty
    string when nothing close enough matches the cutoff.
    """
    if not unknown:
        return ""
    matches = difflib.get_close_matches(
        unknown, list(candidates), n=n, cutoff=cutoff)
    if not matches:
        return ""
    quoted = ", ".join(f"'{m}'" for m in matches)
    return f" Did you mean: {quoted}?"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Validate + apply one tool_use block at a time.

    Stateful: holds a reference to the ExtractionRecord being mutated, the
    template (for field specs), the paper text (for verbatim checks),
    and the image label set (for image-sourced evidence bypass).

    Call `dispatch(tool_name, tool_input, meta=...)` for each `tool_use`
    block emitted by the model.
    """

    def __init__(self, extraction_record, template, paper_text, image_labels,
                 reference_lists=None):
        self.extraction_record = extraction_record
        self.template = template
        self.paper_text = paper_text
        self.image_labels = set(
            lbl.strip().lower() for lbl in image_labels
        )
        # Per-list resolution indexes for canonical_reference fields, built
        # once and reused for every dispatch. Reviewer edits on resume go
        # through the same dispatcher, so they hit the same validation path.
        from meltiro.reference_lists import build_reference_index
        self._reference_indexes = {
            name: build_reference_index(entries)
            for name, entries in (reference_lists or {}).items()
        }
        # The template-declared record-entity nouns (singular/plural) are
        # substituted into LLM-facing runtime messages so the model reads
        # review-natural wording (for the worked config, "relationship" /
        # "relationships"). Tool names (add_record, update_record, ...)
        # stay generic and literal and are never rewritten.
        self._entity = _entity(template)
        # One flat field list per scope, quality-assessment sections included
        # (they are ordinary sections marked `qa: true`). The dispatcher's spec
        # indexes must cover exactly what the tool input_schemas advertise,
        # else the model gets told a field is valid and then sees it rejected
        # at runtime as unknown_field.
        self._study_field_specs = self._index_specs(template["study_fields"])
        self._record_field_specs = self._index_specs(template["record_fields"])

    @staticmethod
    def _index_specs(sections):
        """Build a {variable: field_spec} lookup for every declared field.

        Every template field is addressable by the model, so the index carries
        them all; an unknown variable the model submits is handled by the
        `unknown_field` path with a useful hint.
        """
        return {field["variable"]: field for field in iter_fields(sections)}

    def _reference_index_for(self, spec):
        """The reference-resolution index for a field's canonical_reference
        list, or None for a non-reference field (or a missing list, which is
        caught earlier at config-bundle load)."""
        name = spec.get("canonical_reference")
        if not name:
            return None
        return self._reference_indexes.get(name)

    def _study_field_count_total(self):
        return len(self._study_field_specs)

    def _read_scope_notes(self, args, warnings, what):
        """Read the reserved `notes` argument off a tool call.

        Returns `(supplied, value)`. `supplied` is False when the call did not
        mention notes at all, which leaves any stored note untouched; a
        supplied None clears it.

        A non-string, non-null note is refused with a WARNING rather than an
        error, and the call proceeds without it: writing a scope note is never
        a validation failure and never moves a call's ok / partial /
        validation_failed status, which is decided by the fields alone. The
        warning is the loud part: the model is told the note was not recorded
        and why.
        """
        if NOTES_KEY not in args:
            return False, None
        notes = args[NOTES_KEY]
        if notes is not None and not isinstance(notes, str):
            warnings.append({
                "path": NOTES_KEY,
                "code": "notes_not_recorded",
                "message": (
                    f"The {what} note was not recorded: `notes` must be a "
                    f"string (or null to clear it), got "
                    f"{type(notes).__name__}. The rest of this call was "
                    f"unaffected; resubmit the note as a string."
                ),
            })
            return False, None
        return True, notes

    def _build_extraction_record_summary(self):
        study_total = self._study_field_count_total()
        non_null_study = 0
        for var, env in self.extraction_record.study.items():
            # Skip anything the specs don't know about: the reserved scope-note
            # key (never declarable as a field) and any other unknown stored
            # key. Counts only non-null declared study fields.
            if var not in self._study_field_specs:
                continue
            if isinstance(env, dict) and env.get("value") is not None:
                non_null_study += 1
        return {
            "study_fields_filled": non_null_study,
            "study_fields_total": study_total,
            "records_count": len(self.extraction_record.records),
        }

    def _result(self, status, errors=None, warnings=None,
                applied_changes=None, applied_fields=None,
                failed_fields=None, warnings_by_field=None,
                field_diffs=None, meta=None, canonicalisations=None,
                weak_quote_matches=None):
        # `errors` flattens both call-level and per-field errors, so a
        # consumer that wants the whole list (the transcript renderer, for
        # one) reads one key. `failed_fields` groups the same errors by field
        # path for the extractor's revision logic.
        flat_errors = list(errors or [])
        for _path, errs in (failed_fields or {}).items():
            flat_errors.extend(errs)
        out = {
            "status": status,
            "applied_changes": applied_changes or {},
            # `_field_diffs` is a flat `{path: {before, after}}` dict the
            # transcript renderer uses to draw the per-field before/after
            # table. The leading underscore marks it as UI-only telemetry:
            # the orchestrator strips any underscore-prefixed key from the
            # tool_result content it returns to the LLM so this never
            # pollutes conversation history. The unstripped dict still
            # lands in the session event for transcript rendering.
            "_field_diffs": field_diffs or {},
            "errors": flat_errors,
            "warnings": warnings or [],
            "applied_fields": applied_fields or [],
            "failed_fields": failed_fields or {},
            "warnings_by_field": warnings_by_field or {},
            "extraction_output_summary": self._build_extraction_record_summary(),
        }
        # Reference-list canonicalisations (alias matches).
        # `canonicalisation_notes` is a short model-visible line per
        # canonicalised field; `_canonicalisations`
        # carries the structured records (field path, entered, stored) for
        # the orchestrator to write as `value_canonicalised` events, and is
        # stripped from the model-facing tool_result (leading underscore).
        # The key is spelled out rather than left as a bare `notes` so it
        # cannot be read as the extractor's own field or scope note.
        canon = canonicalisations or []
        if canon:
            out["canonicalisation_notes"] = [
                f"{c['path']}: recorded as '{c['stored']}' "
                f"(entered '{c['entered']}')"
                for c in canon
            ]
        out["_canonicalisations"] = canon
        # Evidence quotes that PASSED only once both sides were lowercased
        # (`quote_check` tier `case_folded`). The other forgiving tiers cover
        # differences a PDF converter introduces; case is a difference the
        # model chose, so a case-folded match is a weaker claim in kind — and
        # a passing quote produces no error to carry the tier. Recorded per
        # applied field, reaching tool_calls.jsonl and the transcript.
        # Underscore-prefixed (stripped from the model-facing tool_result):
        # the fold succeeded, so there is nothing for the model to fix.
        # Provenance for whoever reads the run, not feedback for the run.
        out["_weak_quote_matches"] = weak_quote_matches or []
        # The remaining tool-call budget is UI-only telemetry. The leading
        # underscore makes `result_to_model_text` strip it from the model-facing
        # tool_result: the cap is out of every fingerprint, so the model must
        # not read a cap-derived number. The unstripped value stays in the
        # session event for transcript rendering.
        if meta and "tool_call_budget_remaining" in meta:
            out["_tool_call_budget_remaining"] = meta[
                "tool_call_budget_remaining"]
        return out

    # ----------------------------------------------------------------------
    # Public dispatch
    # ----------------------------------------------------------------------

    def dispatch(self, tool_name, tool_input, meta=None,
                 role=ROLE_EXTRACTOR):
        """Dispatch one tool call on behalf of `role`.

        Returns the tool_result payload (a dict). On `validation_failed`,
        no changes are applied to the extraction output.

        `role` is explicit rather than dispatcher state because one dispatcher
        instance serves both loops: the extractor's and, on the same
        ExtractionRecord, the reviewer's. It decides three things — whether the
        initial-check ordering gate applies, which role's key a quality check
        is filed under, and whether the `view_*` tools reveal the check blocks
        at all.
        """
        handlers = {
            "record_initial_check": self._handle_record_initial_check,
            "update_study": self._handle_update_study,
            "add_record": self._handle_add_record,
            "update_record": self._handle_update_record,
            "remove_record": self._handle_remove_record,
            "mark_complete": self._handle_mark_complete,
            "abandon_extraction": self._handle_abandon_extraction,
            "view_summary": self._handle_view_summary,
            "view_study_fields": self._handle_view_study_fields,
            "view_record": self._handle_view_record,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "",
                    "code": "unknown_tool",
                    "message": f"Unknown tool: {tool_name!r}. Use one of: "
                               f"{sorted(handlers)}.",
                }],
                meta=meta,
            )
        if not isinstance(tool_input, dict):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "",
                    "code": "malformed_input",
                    "message": "Tool input must be a JSON object.",
                }],
                meta=meta,
            )

        # `record_initial_check` is not in the reviewer's catalogue. A model
        # can still name a tool it was not given, so refuse it here rather
        # than let the reviewer file a retrospective initial check over the
        # extractor's.
        if tool_name == "record_initial_check" and role == ROLE_REVIEW:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "",
                    "code": "tool_not_available_to_role",
                    "message": (
                        "`record_initial_check` is the extractor's "
                        "pre-extraction report on the inputs and is recorded "
                        "once. It is not part of the review. Record your own "
                        "assessment of the extraction in the "
                        "`quality_check` argument of `mark_complete`."
                    ),
                }],
                meta=meta,
            )

        # The ordering gate. Until the extractor has recorded its initial
        # check, nothing it does may change the extraction output. This is the
        # whole of the enforcement: the prompt asks, and this refuses.
        # `abandon_extraction` and the read-only `view_*` tools are ungated on
        # purpose — an honest surrender and a look at an empty record must
        # both stay available to a model that has got itself stuck.
        if (role == ROLE_EXTRACTOR
                and tool_name in _INITIAL_CHECK_GATED_TOOLS
                and not self.extraction_record.initial_check_recorded):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "",
                    "code": "initial_check_required",
                    "message": (
                        f"`{tool_name}` was not applied: call "
                        "`record_initial_check` first. It is a report on the "
                        "inputs you were handed, made before you extract "
                        "from them, so it has to come first to mean "
                        "anything. Nothing else has changed — make that call, "
                        "then resubmit this one unaltered."
                    ),
                }],
                meta=meta,
            )

        # An argument the tool does not read is answered, never dropped. It is
        # checked here rather than in each handler, so no handler carries its
        # own argument list and none of them can disagree about what its tool
        # takes.
        unconditional = self._is_unconditional_exit(tool_name, role)
        unknown_args = self._unknown_argument_problems(
            tool_name, tool_input, refused=not unconditional)
        if unknown_args and not unconditional:
            return self._result("validation_failed", errors=unknown_args,
                                meta=meta)

        if tool_name in _ROLE_AWARE_HANDLERS:
            result = handler(tool_input, meta=meta, role=role)
        else:
            result = handler(tool_input, meta=meta)

        # The reviewer's `mark_complete` is the one call that cannot be made
        # contingent on validation (see `_mark_complete_tool`): it is the only
        # exit from a fresh-context loop with no replay, so refusing it over an
        # argument name would let a review spin against the tool-call bound
        # instead of finishing. The refusal degrades to a warning — the
        # argument is still named and the loss still announced, and the review
        # still ends — which is how that call already treats a quality check
        # it cannot record.
        if unknown_args:
            result["warnings"] = list(result.get("warnings") or []) + [
                {**problem, "code": "argument_not_read"}
                for problem in unknown_args
            ]
        return result

    @staticmethod
    def _is_unconditional_exit(tool_name, role):
        """True for the one call whose success is not the dispatcher's to
        withhold: the reviewer's `mark_complete`."""
        return tool_name == "mark_complete" and role == ROLE_REVIEW

    def _unknown_argument_problems(self, tool_name, args, *, refused):
        """Describe every top-level argument its tool does not take.

        Returns a list of {path, code, message} dicts, empty when the call is
        clean. `refused` says which of the two outcomes the caller is about to
        apply, so the message states what actually happened to the rest of the
        call rather than assuming.

        Refusing is the normal outcome and is the point of the check: a
        misnamed field map is a payload the model believes it recorded, and
        applying the rest of the call around it would report success over a
        loss no one can see. Nothing is applied, so the model can resubmit the
        same content under the right name.

        Each entry names the offending argument, lists what the tool does
        take, and (for a tool that carries one) says where its field map
        belongs. A check block addressed to a tool that does not take it keeps
        its own `block_moved` code and the sentence naming the tool that owns
        it: the same finding, with a more specific answer available.
        """
        allowed = _TOOL_ARGUMENTS.get(tool_name)
        if allowed is None:
            return []
        unknown = sorted(k for k in args if k not in allowed)
        if not unknown:
            return []

        if refused:
            outcome = "Nothing in this call was applied"
            retry = (" Resubmit the call with its contents under those names."
                     if allowed else " Resubmit it with no arguments.")
        else:
            outcome = "It was not read; the rest of this call was applied"
            retry = ""
        takes = (", ".join(f"`{a}`" for a in sorted(allowed))
                 if allowed else "no arguments")
        field_map_home = _FIELD_MAP_HOME.get(tool_name)

        problems = []
        for name in unknown:
            if name in _CHECK_BLOCK_HOME:
                problems.append({
                    "path": name,
                    "code": "block_moved",
                    "message": (
                        f"`{name}` is not an argument of {tool_name}. "
                        f"{_CHECK_BLOCK_HOME[name]} {outcome}."
                    ),
                })
                continue
            home = f" {field_map_home}" if field_map_home else ""
            problems.append({
                "path": name,
                "code": "unknown_argument",
                "message": (
                    f"`{name}` is not an argument of {tool_name}. {outcome}. "
                    f"{tool_name} takes {takes}.{home}{retry}"
                ),
            })
        return problems

    # ----------------------------------------------------------------------
    # Individual tool handlers
    # ----------------------------------------------------------------------

    def _handle_record_initial_check(self, args, meta, role=ROLE_EXTRACTOR):
        """Record the extractor's pre-extraction check and open the gate.

        Per-field validation, like the field-writing tools: the valid
        variables land and only the failed ones need resubmitting. The
        ordering gate opens as soon as ANY well-formed call lands, including
        one that applied nothing because every field failed — the gate asks
        whether the extractor has looked at its inputs and answered, not
        whether it answered well. A structurally unusable call is the
        exception: refused whole, gate stays shut.

        Re-calling revises, deliberately: an initial check the extractor can
        correct is worth more than one frozen at first guess, and every
        version is in the event log.
        """
        applied_fields = []
        failed_fields = {}
        valid = {}

        # `args` is the block itself (flat properties), not a wrapper.
        for var, value in args.items():
            errs = self._validate_metadata_field(
                var, value, "initial_check",
                self.template["initial_check_fields"])
            path = f"initial_check.{var}"
            if errs:
                failed_fields[path] = errs
            else:
                applied_fields.append(path)
                valid[var] = value

        field_diffs = {}
        before = self.extraction_record.initial_check_for(role)
        for var in valid:
            field_diffs[f"initial_check.{var}"] = {
                "before": before.get(var),
                "after": valid[var],
            }

        applied = self.extraction_record.record_initial_check(valid, role=role)

        status = "validation_failed" if (failed_fields and not applied_fields) \
            else ("partial" if failed_fields else "ok")
        return self._result(
            status,
            applied_changes={"initial_check": applied, "recorded_by": role},
            applied_fields=applied_fields,
            failed_fields=failed_fields,
            field_diffs=field_diffs,
            meta=meta,
        )

    def _handle_update_study(self, args, meta):
        """Apply per-field updates. Valid fields land in the extraction
        record even if siblings fail validation; the extractor only
        needs to retry the failed ones, not the whole call.

        The optional top-level `notes` argument writes the study scope note.
        It is applied outside the per-field machinery: it is not a field, it
        is never validated, and it never moves this call's status. It still
        goes down with a STRUCTURALLY malformed call (a block that is not an
        object), which is rejected wholesale before any of it is read; the
        note does not cause that failure, it is carried away by it.

        `study` and `notes` are the only arguments this tool reads, and
        `dispatch` has already refused the call if it carried any other, so a
        field map addressed to a different name never reaches this handler to
        be dropped.
        """
        study = args.get("study") or {}

        # Defensive: models occasionally pass strings or lists where a
        # dict is expected, especially when responding to checker
        # challenges. Structural problems remain whole-call failures.
        if not isinstance(study, dict):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "study",
                    "code": "type_mismatch",
                    "message": (
                        f"Expected an object for 'study', got "
                        f"{type(study).__name__}. Pass a map of "
                        f"variable -> envelope."
                    ),
                }],
                meta=meta,
            )

        applied_fields = []
        failed_fields = {}
        warnings = []
        notes_supplied, notes = self._read_scope_notes(args, warnings, "study")

        # Study fields: type/quote-check each. An unknown variable falls into
        # the `unknown_field` branch below.
        valid_study = {}
        canonicalisations = []
        weak_quote_matches = []
        for var, env in study.items():
            path = f"study.{var}"
            spec = self._study_field_specs.get(var)
            if spec is None:
                hint = _suggest_closest_field(
                    var, self._study_field_specs.keys())
                notes_hint = _misplaced_scope_note_hint(var, "update_study")
                failed_fields[path] = [{
                    "path": path,
                    "code": "unknown_field",
                    "message": (
                        f"'{var}' is not a known study field.{hint}"
                        f"{notes_hint} "
                        "Use the exact variable name from the field "
                        "catalogue in the system prompt; field names are "
                        "case-sensitive."
                    ),
                }]
                continue
            field_canon = []
            field_weak = []
            errs = validate_envelope(
                env, spec, self.paper_text, self.image_labels,
                path_prefix=path,
                reference_index=self._reference_index_for(spec),
                canonicalisations=field_canon,
                weak_quote_matches=field_weak,
            )
            if errs:
                failed_fields[path] = errs
            else:
                applied_fields.append(path)
                valid_study[var] = env
                # Only surface canonicalisations, and the quote tiers beside
                # them, for fields that applied: a rejected field stores
                # nothing, so a note about how its quote matched would describe
                # evidence that is not in the output.
                canonicalisations.extend(field_canon)
                weak_quote_matches.extend(field_weak)

        # Snapshot the per-field "before" state for the diff table.
        # Envelopes -> value scalar; bare blocks -> the value as-is.
        def _before_envelope(var):
            env = self.extraction_record.study.get(var)
            return env.get("value") if isinstance(env, dict) else env

        field_diffs = {}
        for var in valid_study:
            new_env = valid_study[var]
            new_val = (new_env.get("value")
                       if isinstance(new_env, dict) else new_env)
            field_diffs[f"study.{var}"] = {
                "before": _before_envelope(var),
                "after": new_val,
            }

        # Apply the valid subset, if anything survived. The scope note rides
        # along; it is applied even when no field did, so a note-only call
        # still lands.
        applied_changes = {}
        if valid_study or notes_supplied:
            note_arg = ({"notes": notes} if notes_supplied else {})
            applied_changes = self.extraction_record.apply_update_study(
                study=valid_study or None,
                **note_arg,
            )

        # Status is decided by the FIELDS alone: a scope note can neither
        # fail a call nor rescue one.
        if failed_fields and not applied_fields:
            status = "validation_failed"
        elif failed_fields:
            status = "partial"
        else:
            status = "ok"
        return self._result(
            status,
            applied_changes=applied_changes,
            applied_fields=applied_fields,
            failed_fields=failed_fields,
            field_diffs=field_diffs,
            warnings=warnings,
            meta=meta,
            canonicalisations=canonicalisations,
            weak_quote_matches=weak_quote_matches,
        )

    def _handle_add_record(self, args, meta):
        """Append a record, applying per-field updates. Valid fields land on
        the new record even if siblings fail validation, and the new record_id
        rides back with the failures so the extractor follows up with
        update_record rather than resending the whole record.

        Required-field completeness is not gated here, so a record may be
        created incomplete and filled in over several calls. The EXTRACTOR's
        `mark_complete` is where that debt is collected: it checks every
        record's template-declared required fields and refuses until each
        carries a value. That gate belongs to the extractor alone. The
        reviewer's `mark_complete` runs no completeness check (see
        `_mark_complete_tool`, which explains why its exit cannot be made
        contingent), so a record the reviewer adds after the extractor has
        finished is held to per-field validation and to nothing else. Read the
        guarantee as "the extractor cannot leave its own pass incomplete",
        not as "no output can ship with a required field unset".

        The optional `notes` argument writes the new record's scope note. A
        call whose every field failed mints no record, so there is nothing for
        a note to attach to and it is dropped with them.
        """
        fields = args.get("fields") or {}
        if not isinstance(fields, dict):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "fields",
                    "code": "type_mismatch",
                    "message": (
                        "add_record 'fields' must be an object "
                        f"(variable -> envelope), got "
                        f"{type(fields).__name__}."
                    ),
                }],
                meta=meta,
            )
        if not fields:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "fields",
                    "code": "missing_fields",
                    "message": "add_record requires a non-empty 'fields' "
                               "object.",
                }],
                meta=meta,
            )
        (valid_fields, failed_fields, warnings, canonicalisations,
         weak_quote_matches) = \
            self._validate_record_fields(fields, path_prefix="record.<new>")
        notes_supplied, notes = self._read_scope_notes(
            args, warnings, self._entity["singular"])

        # Every field failed: nothing was validly addressed, so no record is
        # minted, no record-id index is consumed, and the failure paths keep
        # the placeholder prefix (there is no record to point them at).
        if not valid_fields:
            return self._result("validation_failed",
                                failed_fields=failed_fields,
                                warnings=warnings, meta=meta)

        note_arg = ({"notes": notes} if notes_supplied else {})
        record_id = self.extraction_record.add_record(
            valid_fields, self._entity["singular"], **note_arg)
        # The record id is assigned only now, so rewrite the placeholder
        # prefix on the failure paths and the canonicalisation records.
        failed_fields = self._repoint_to_new_record(record_id, failed_fields)
        for c in canonicalisations + weak_quote_matches:
            # Both carry a field path recorded before the id existed, so both
            # need the placeholder rewritten to the record actually minted.
            # A path left naming `record.<new>` points at nothing a reader of
            # the finished output can find.
            c["path"] = c["path"].replace(
                "record.<new>", f"record.{record_id}", 1)
        applied = {
            "record_id": record_id,
            "record_fields": sorted(valid_fields.keys()),
        }
        if notes_supplied:
            applied["notes_written"] = True
        applied_fields = []
        field_diffs = {}
        for var, env in valid_fields.items():
            new_val = env.get("value") if isinstance(env, dict) else env
            path = f"record.{record_id}.{var}"
            applied_fields.append(path)
            field_diffs[path] = {"before": None, "after": new_val}
        return self._result("partial" if failed_fields else "ok",
                            applied_changes=applied,
                            applied_fields=applied_fields,
                            failed_fields=failed_fields,
                            field_diffs=field_diffs,
                            warnings=warnings, meta=meta,
                            canonicalisations=canonicalisations,
                            weak_quote_matches=weak_quote_matches)

    @staticmethod
    def _repoint_to_new_record(record_id, failed_fields):
        """Rewrite the `record.<new>` placeholder prefix to the real record
        path, on both the grouping keys and each error's own `path`.

        `add_record` validates before an id exists, so a partial add's failure
        paths name the placeholder. Repointing them is what lets the extractor
        address the failures with `update_record` on the id this call returned.
        """
        real = f"record.{record_id}"
        out = {}
        for path, errs in failed_fields.items():
            for err in errs:
                err["path"] = err["path"].replace("record.<new>", real, 1)
            out[path.replace("record.<new>", real, 1)] = errs
        return out

    def _handle_update_record(self, args, meta):
        """Apply per-field revisions to one record. Valid fields land even if
        siblings fail validation; the extractor only needs to retry the failed
        ones, not the whole record.

        A structural problem (no record_id, an unknown record_id, a `fields`
        map that is not an object or is empty) addresses no field validly, so
        it stays a whole-call failure, and the optional `notes` argument does
        not rescue it: a scope note never moves a call's status. Notes ride
        with a call that revises at least one field.
        """
        record_id = args.get("record_id")
        fields = args.get("fields") or {}

        if not isinstance(record_id, str) or not record_id:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "record_id",
                    "code": "missing_field",
                    "message": "record_id is required.",
                }],
                meta=meta,
            )
        if not self.extraction_record.has_record(record_id):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "record_id",
                    "code": "unknown_record",
                    "message": (
                        f"No {self._entity['singular']} with id {record_id}. "
                        f"Current ids: {self.extraction_record.record_ids()}."
                    ),
                }],
                meta=meta,
            )
        if not isinstance(fields, dict):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "fields",
                    "code": "type_mismatch",
                    "message": (
                        "update_record 'fields' must be an object "
                        f"(variable -> envelope), got "
                        f"{type(fields).__name__}."
                    ),
                }],
                meta=meta,
            )
        if not fields:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "fields",
                    "code": "missing_fields",
                    "message": "update_record requires a non-empty "
                               "'fields' object.",
                }],
                meta=meta,
            )

        # `existing` is a deep copy, so the post-update view the gate rules
        # read is built from it without touching the stored record.
        existing = self.extraction_record.get_record(record_id) or {}
        (valid_fields, failed_fields, warnings, canonicalisations,
         weak_quote_matches) = \
            self._validate_record_fields(
                fields, path_prefix=f"record.{record_id}", existing=existing)
        notes_supplied, notes = self._read_scope_notes(
            args, warnings, self._entity["singular"])

        if not valid_fields:
            return self._result("validation_failed",
                                failed_fields=failed_fields,
                                warnings=warnings, meta=meta)

        # Snapshot before-values BEFORE applying so the diff reads true.
        before_envs = {}
        for var in valid_fields:
            prior = existing.get(var)
            before_envs[var] = (
                prior.get("value") if isinstance(prior, dict) else prior)

        note_arg = ({"notes": notes} if notes_supplied else {})
        applied_keys = self.extraction_record.update_record(
            record_id, valid_fields, **note_arg)
        applied = {
            "record_id": record_id,
            "record_fields": sorted(applied_keys),
        }
        if notes_supplied:
            applied["notes_written"] = True
        applied_fields = []
        field_diffs = {}
        for var, env in valid_fields.items():
            new_val = env.get("value") if isinstance(env, dict) else env
            path = f"record.{record_id}.{var}"
            applied_fields.append(path)
            field_diffs[path] = {
                "before": before_envs.get(var),
                "after": new_val,
            }
        return self._result("partial" if failed_fields else "ok",
                            applied_changes=applied,
                            applied_fields=applied_fields,
                            failed_fields=failed_fields,
                            field_diffs=field_diffs,
                            warnings=warnings, meta=meta,
                            canonicalisations=canonicalisations,
                            weak_quote_matches=weak_quote_matches)

    def _handle_remove_record(self, args, meta):
        record_id = args.get("record_id")
        reason = args.get("reason")
        if not isinstance(record_id, str) or not record_id:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "record_id",
                    "code": "missing_field",
                    "message": "record_id is required.",
                }],
                meta=meta,
            )
        if not isinstance(reason, str) or not reason.strip():
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "reason",
                    "code": "missing_field",
                    "message": "reason is required and must be a non-empty "
                               "string.",
                }],
                meta=meta,
            )
        if not self.extraction_record.has_record(record_id):
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "record_id",
                    "code": "unknown_record",
                    "message": (
                        f"No {self._entity['singular']} with id {record_id}."
                    ),
                }],
                meta=meta,
            )
        self.extraction_record.remove_record(record_id)
        return self._result("ok", applied_changes={
            "removed_record_id": record_id,
        }, meta=meta)

    def _handle_mark_complete(self, args, meta, role=ROLE_EXTRACTOR):
        """Record this role's quality check, and (for the extractor) gate
        completion on the extraction being complete enough to ship.

        The quality check arrives HERE rather than having been written
        earlier, so a run cannot be declared finished and unassessed. It is
        filed under `role`, beside — never over — any other role's.

        Which fields are required is template-declared (`required: true`), so
        this gate names no specific field: the engine stays generic. Bare
        check-block fields must be present and non-null; envelope fields
        (study / record) must carry a non-null value.

        The reviewer's call is not gated (see `_mark_complete_tool`): it saw
        the whole assembled output, its confirmation is not subject to the
        extractor's completeness gate, and it has no second chance to
        terminate. So for the reviewer this records what validates, warns
        about what does not, and always succeeds.
        """
        errors = []
        warnings = []

        # The quality check supplied on THIS call. Validated per field against
        # the template, exactly as the initial check is.
        qc_arg = args.get("quality_check")
        valid_qc = {}
        if qc_arg is None:
            # Absent entirely. For the extractor this fails even when the
            # template marks no quality-check field `required`: complete-and-
            # report-in-one-call is an engine property, not a config one (the
            # initial-check gate is unconditional for the same reason).
            if role == ROLE_EXTRACTOR:
                errors.append({
                    "path": "quality_check",
                    "code": "quality_check_required",
                    "message": (
                        "mark_complete requires a `quality_check` object. "
                        "Completing the extraction and reporting on how it "
                        "went are one call, so there is no way to declare "
                        "the run finished without saying how it went."
                    ),
                })
            qc_arg = {}
        if not isinstance(qc_arg, dict):
            errors.append({
                "path": "quality_check",
                "code": "type_mismatch",
                "message": (
                    f"Expected an object for 'quality_check', got "
                    f"{type(qc_arg).__name__}. Pass a map of "
                    f"variable -> value."
                ),
            })
        else:
            for var, value in qc_arg.items():
                errs = self._validate_metadata_field(
                    var, value, "quality_check",
                    self.template["quality_check_fields"])
                if errs:
                    errors.extend(errs)
                else:
                    valid_qc[var] = value

        # Required quality-check fields must be present and non-null. Read off
        # this call merged over anything this role recorded EARLIER — which,
        # for the extractor, can only be a previous SUCCESSFUL mark_complete
        # whose flag a later edit then cleared, since a rejected call records
        # nothing. So the merge is what lets a re-declaration after one more
        # field edit stand on the quality check already given, rather than a
        # way to accumulate a required answer across failed attempts.
        merged_qc = dict(self.extraction_record.quality_check_for(role))
        merged_qc.update(valid_qc)
        for var in sorted(
                required_field_names(self.template["quality_check_fields"])):
            if merged_qc.get(var) is None:
                errors.append({
                    "path": f"quality_check.{var}",
                    "code": "metadata_required",
                    "message": (
                        f"quality_check.{var} must be supplied to "
                        "mark_complete. Completing the extraction and "
                        "reporting on how it went are one call."
                    ),
                })

        # The extractor's initial check must exist. The ordering gate in
        # `dispatch` already refuses mark_complete before it, so this covers
        # the case where the call landed but a required field failed.
        if role == ROLE_EXTRACTOR:
            initial = self.extraction_record.initial_check_for(role)
            for var in sorted(required_field_names(
                    self.template["initial_check_fields"])):
                if initial.get(var) is None:
                    errors.append({
                        "path": f"initial_check.{var}",
                        "code": "metadata_required",
                        "message": (
                            f"initial_check.{var} must be set before "
                            "mark_complete. Resubmit it with "
                            "`record_initial_check`."
                        ),
                    })

        # Everything below is the extractor's completeness gate. The reviewer
        # records its quality check and concludes; it is not re-asked whether
        # the extraction it just reviewed is complete.
        if role == ROLE_REVIEW:
            if errors:
                # Not a failure: the review still ends. Say what was dropped.
                warnings.extend({**e, "code": "quality_check_not_recorded"}
                                for e in errors)
            applied = self.extraction_record.record_quality_check(
                valid_qc, role=role)
            return self._result(
                "ok",
                applied_changes={"quality_check": applied,
                                 "recorded_by": role},
                warnings=warnings,
                meta=meta,
            )

        # Study-level envelope fields flagged required must be non-null.
        study_required = required_field_names(self.template["study_fields"])
        for var in sorted(study_required):
            env = self.extraction_record.study.get(var)
            val = env.get("value") if isinstance(env, dict) else env
            if val is None:
                errors.append({
                    "path": f"study.{var}",
                    "code": "study_field_required",
                    "message": (
                        f"study.{var} must have a non-null value before "
                        "mark_complete."
                    ),
                })

        # At least one record, each carrying its required fields non-null.
        record_required = required_field_names(self.template["record_fields"])
        if not self.extraction_record.records:
            errors.append({
                "path": "records",
                "code": "no_records",
                "message": (
                    f"At least one {self._entity['singular']} must be added "
                    "before mark_complete. Use add_record."
                ),
            })
        else:
            for record in self.extraction_record.records:
                rid = record.get("record_id")
                for req in sorted(record_required):
                    env = record.get(req)
                    val = env.get("value") if isinstance(env, dict) else env
                    if val is None:
                        errors.append({
                            "path": f"record.{rid}.{req}",
                            "code": "record_field_required",
                            "message": (
                                f"{req} must have a non-null value on every "
                                f"{self._entity['singular']} before "
                                "mark_complete."
                            ),
                        })

        if errors:
            # Nothing is recorded on a failed call, the quality check
            # included: a rejected mark_complete leaves the extraction exactly
            # as it was, and the extractor re-calls with its quality check
            # again once the named problems are fixed.
            return self._result("validation_failed", errors=errors, meta=meta)

        before = self.extraction_record.quality_check_for(role)
        field_diffs = {f"quality_check.{var}": {"before": before.get(var),
                                                "after": valid_qc[var]}
                       for var in valid_qc}
        applied_qc = self.extraction_record.record_quality_check(
            valid_qc, role=role)
        self.extraction_record.mark_complete()
        return self._result(
            "ok",
            applied_changes={"mark_complete": True,
                             "quality_check": applied_qc,
                             "recorded_by": role},
            # Diffs so the quality check keeps a field history. It arrives
            # through this tool rather than through `update_study`, which
            # emits diffs of its own, so without these it would be the one
            # part of the record with no per-field trace.
            applied_fields=[f"quality_check.{var}" for var in valid_qc],
            field_diffs=field_diffs,
            meta=meta,
        )

    def _handle_abandon_extraction(self, args, meta):
        """Deliberate surrender: latch the abandon flag with a required reason.

        This runs NO extraction-completeness gate (that is the point: it is
        the escape when the mark_complete gate genuinely cannot be met). It
        only checks that a non-empty reason was supplied. The orchestrator
        detects the latched flag and finalises the run as `failed_validation`,
        recording the reason in run.json and the run log.
        """
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "reason",
                    "code": "missing_field",
                    "message": (
                        "abandon_extraction requires a non-empty 'reason' "
                        "explaining why no valid extraction is possible."
                    ),
                }],
                meta=meta,
            )
        self.extraction_record.abandon(reason.strip())
        return self._result(
            "ok",
            applied_changes={"abandon_extraction": True,
                             "reason": reason.strip()},
            meta=meta,
        )

    # ----------------------------------------------------------------------
    # Read-only view tools
    # ----------------------------------------------------------------------
    #
    # Designed for one job: let the extractor answer "what have I added so
    # far" without re-reading its trace. Compact by design; `view_summary`
    # gives counts + a per-record id and context label; full envelopes
    # are reached via `view_study_fields` / `view_record` only when
    # the extractor genuinely needs them. These tool calls count against
    # the same `max_tool_calls` budget as the mutating tools so unbounded
    # view-spam can't run away.

    def _bare_block_counts(self, block_dict, sections):
        """Filled/total for a bare-value block (initial_check, quality_check)."""
        total = 0
        filled = 0
        for f in iter_fields(sections):
            total += 1
            if block_dict.get(f["variable"]) is not None:
                filled += 1
        return {"filled": filled, "total": total}

    def _record_filled_total(self, record):
        """Filled/total for one record's envelopes.

        Uses the dispatcher's pre-indexed record spec map. Filled means the
        envelope's `value` is non-null.
        """
        total = len(self._record_field_specs)
        filled = 0
        for var in self._record_field_specs:
            env = record.get(var)
            if isinstance(env, dict) and env.get("value") is not None:
                filled += 1
        return filled, total

    def _view_response(self, view, meta=None):
        """Standard response shape for view tools.

        Does not pretend to apply changes; `applied_fields`/`failed_fields`
        are absent. `extraction_output_summary` is still included so the
        per-tool-call summary stays stable across tool types.
        """
        out = {
            "status": "ok",
            "view": view,
            "extraction_output_summary": self._build_extraction_record_summary(),
        }
        # UI-only telemetry, stripped from the model-facing tool_result by the
        # leading underscore (see `_result` for the rationale).
        if meta and "tool_call_budget_remaining" in meta:
            out["_tool_call_budget_remaining"] = meta[
                "tool_call_budget_remaining"]
        return out

    def _handle_view_summary(self, args, meta, role=ROLE_EXTRACTOR):
        from meltiro.checker_prompts import (
            build_record_context,
        )
        # study
        study_total = self._study_field_count_total()
        study_filled = 0
        for var, env in self.extraction_record.study.items():
            # Fields only: the reserved scope-note key is not one, and is
            # reported separately below.
            if var not in self._study_field_specs:
                continue
            if isinstance(env, dict) and env.get("value") is not None:
                study_filled += 1
        # records
        records = []
        for record in self.extraction_record.records:
            filled, total = self._record_filled_total(record)
            records.append({
                "record_id": record.get("record_id"),
                "context": build_record_context(
                    record, self.template["checker_context_fields"]),
                "filled": filled,
                "total": total,
                # The record's scope note, so the extractor can read back its
                # own commentary without a second call.
                NOTES_KEY: record.get(NOTES_KEY),
            })
        view = {
            "study_fields": {"filled": study_filled, "total": study_total},
            "study_notes": self.extraction_record.study.get(NOTES_KEY),
            "records": records,
        }
        # The check blocks are the caller's OWN answers or nothing. The
        # reviewer is shown neither its filled/total nor its content, for the
        # same reason the assembled output it is given omits them: its second
        # opinion has to be its own. See `_check_block_counts_for`.
        view.update(self._check_block_counts_for(role))
        return self._view_response(view, meta=meta)

    def _check_block_counts_for(self, role):
        """Filled/total for the check blocks this role may see, or nothing."""
        if role != ROLE_EXTRACTOR:
            return {}
        return {
            "initial_check": self._bare_block_counts(
                self.extraction_record.initial_check_for(role),
                self.template["initial_check_fields"]),
            "quality_check": self._bare_block_counts(
                self.extraction_record.quality_check_for(role),
                self.template["quality_check_fields"]),
        }

    def _handle_view_study_fields(self, args, meta, role=ROLE_EXTRACTOR):
        # The study block carries the reserved `notes` key inline, so the
        # scope note is returned with the field envelopes (each of which
        # carries its own field note) without any extra assembly.
        view = {"study": dict(self.extraction_record.study)}
        # Only the extractor sees check blocks here, and only its own. If the
        # reviewer could read them through a view tool, stripping them from
        # the assembled output it is handed would buy nothing: the anchor
        # would just cost it one extra tool call.
        if role == ROLE_EXTRACTOR:
            view["initial_check"] = dict(
                self.extraction_record.initial_check_for(role))
            view["quality_check"] = dict(
                self.extraction_record.quality_check_for(role))
        return self._view_response(view, meta=meta)

    def _handle_view_record(self, args, meta):
        record_id = args.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            return self._result(
                "validation_failed",
                errors=[{
                    "path": "record_id",
                    "code": "missing_record_id",
                    "message": (
                        "view_record requires a string record_id (e.g. "
                        f"'{self._entity['singular']}_1')."
                    ),
                }],
                meta=meta,
            )
        for record in self.extraction_record.records:
            if record.get("record_id") == record_id:
                # `dict(record)` carries the record's reserved `notes` key
                # alongside its field envelopes, so the scope note and every
                # field note come back with it.
                return self._view_response({
                    "record": dict(record),
                }, meta=meta)
        known = [r.get("record_id") for r in self.extraction_record.records]
        return self._result(
            "validation_failed",
            errors=[{
                "path": "record_id",
                "code": "unknown_record",
                "message": (
                    f"No {self._entity['singular']} with id {record_id!r}. "
                    f"Current {self._entity['plural']}: {known or '(none)'}."
                ),
            }],
            meta=meta,
        )

    # ----------------------------------------------------------------------
    # Per-block validators
    # ----------------------------------------------------------------------

    def _validate_metadata_field(self, var, value, block_name, sections):
        """Validate ONE field in an initial_check / quality_check block.

        Returns a list of {path, code, message} dicts (empty list on
        success). Used by the partial-validation handler to validate
        each variable independently.
        """
        specs = {f["variable"]: f for f in iter_fields(sections)}
        spec = specs.get(var)
        if spec is None:
            hint = _suggest_closest_field(var, specs.keys())
            return [{
                "path": f"{block_name}.{var}",
                "code": "unknown_field",
                "message": (
                    f"'{var}' is not a known {block_name} field.{hint} "
                    "Use the exact variable name from the field "
                    "catalogue; field names are case-sensitive."
                ),
            }]
        return _check_value_type(value, spec, f"{block_name}.{var}")

    def _validate_record_fields(self, fields, path_prefix, existing=None):
        """Validate a record fields dict field by field (add and update).

        Returns (valid_fields, failed_fields, warnings, canonicalisations,
        weak_quote_matches):

          - `valid_fields` is {variable: envelope} for the fields that passed,
            in submission order, ready to apply;
          - `failed_fields` groups the rest by field path, exactly as
            `update_study` reports them;
          - `warnings` are the template-declared cross-field gate warnings;
          - `canonicalisations` are the reference-list alias matches, for the
            fields that passed only, mirroring `update_study`: a rejected
            field's canonicalisation is dropped with the field;
          - `weak_quote_matches` are the evidence quotes that passed only on
            the case-folded tier, on the same passed-fields-only terms.

        Gates run against the POST-UPDATE view: `existing` (the record's
        stored fields, None for a new record) merged with the valid subset,
        which is the state the record is actually left in. A rejected value
        never reaches that view, so it can never raise a warning.

        The auto-assigned record-id field is never a declared field, so a
        hallucinated attempt to set it lands in the `unknown_field` branch
        below.
        """
        valid_fields = {}
        failed_fields = {}
        canonicalisations = []
        weak_quote_matches = []

        for var, env in fields.items():
            path = f"{path_prefix}.{var}"
            spec = self._record_field_specs.get(var)
            if spec is None:
                hint = _suggest_closest_field(
                    var, self._record_field_specs.keys())
                # Many extractor mistakes here are study-level fields
                # accidentally placed on a record; flag that
                # specifically since the model can recover by moving
                # the field rather than searching for a synonym.
                study_hint = ""
                if var in self._study_field_specs:
                    study_hint = (
                        f" Note: '{var}' is a STUDY-level field; pass "
                        f"it via `update_study.study`, not on a "
                        f"{self._entity['singular']}."
                    )
                notes_hint = _misplaced_scope_note_hint(
                    var, "add_record / update_record")
                failed_fields[path] = [{
                    "path": path,
                    "code": "unknown_field",
                    "message": (
                        f"'{var}' is not a known {self._entity['singular']} "
                        f"field.{hint}{study_hint}{notes_hint} Use the exact "
                        "variable name from the field catalogue; field names "
                        "are case-sensitive."
                    ),
                }]
                continue
            field_canon = []
            field_weak = []
            errs = validate_envelope(
                env, spec, self.paper_text, self.image_labels,
                path_prefix=path,
                reference_index=self._reference_index_for(spec),
                canonicalisations=field_canon,
                weak_quote_matches=field_weak,
            )
            if errs:
                failed_fields[path] = errs
            else:
                valid_fields[var] = env
                canonicalisations.extend(field_canon)
                weak_quote_matches.extend(field_weak)

        # Cross-field checks: template-declared gates (warnings only). The two
        # reserved keys are dropped: neither the record id nor the scope note
        # is a field, so neither can gate or be gated.
        check_view = {k: v for k, v in (existing or {}).items()
                      if k not in ("record_id", NOTES_KEY)}
        check_view.update(valid_fields)
        warnings = validate_gate_rules(
            check_view, self.template.get("gates") or [], path_prefix)

        return (valid_fields, failed_fields, warnings, canonicalisations,
                weak_quote_matches)
