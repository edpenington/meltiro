"""Render a parsed extraction template as a human-readable Markdown document.

Two views, one template model (`meltiro.template.load_template`). Both are
documents for a human, not a transparent dump of the template YAML:

  operational
      The human extractor / reviewer facing view. Every field is a table row
      carrying its human-facing label (with the machine variable kept as a
      small secondary, so an extraction team can map guidance to data columns),
      its value domain (a readable type label, the allowed options, or the
      controlled-vocabulary requirement), its required flag, its evidence
      policy, its description, and its extraction instruction. Section-level and
      record-level extraction instructions are surfaced, the record entity is
      named, and the initial-check and quality-check blocks are included. This
      is everything a human needs to extract each field the same way the model
      is asked to. Sections the template marks `qa: true` are grouped under
      their own quality-appraisal heading, per scope, after the extraction
      sections.

  publication
      The academic reader facing view. Study-level and record-level fields as a
      flat Section / Field / Description / Values table each. Descriptions only:
      no extraction instructions, no `qa: true` sections, no check blocks, and
      no machine variable (a paper's reader does not need it).

Nothing here is authored per-field prose: every heading is structural, every
short label names the field's own value domain (`_TYPES`), and every
remaining line is a template value (a description, an extraction
instruction, an option, a field or entity name). The LLM plumbing a human
never follows (a field's mechanical `role`, its consumer-only
`soft_canonicalisation` flag) is deliberately not rendered.

Nothing is review-specific in this module: the record entity noun and every
field name, description, option, reference-list name, and reference-list
display label come from the loaded config, so the same code renders any
template.

The render is idempotent: the same template model produces byte-identical
output every time. There is no timestamp, no provenance header, and no
nondeterministic ordering (the template's declared section and field order is
preserved).
"""


# Human-facing labels for the field types, so the value cell reads for a person
# ("Text", "Year", "Text (multiple)") rather than exposing the machine type
# name ("string", "year", "string_list"). Categorical fields never reach this
# map:
# they render their own option list.
_TYPES = {
    "string": "Text",
    "year": "Year",
    "integer": "Number",
    "number": "Number",
    "boolean": "Yes/No",
    "string_list": "Text (multiple)",
    "date": "Date",
}

# The operational field table. Full-word headers; one shape for every block.
_OP_HEADER = ["Field", "Type / values", "Required", "Evidence",
              "Description", "Extraction instruction"]

# The publication field table.
_PUB_HEADER = ["Section", "Field", "Description", "Values"]


# ---------------------------------------------------------------------------
# Cell + table helpers
# ---------------------------------------------------------------------------

def _inline(text):
    """Collapse `text` to a single line (whitespace runs to one space)."""
    return " ".join(str(text or "").split())


def _cell(text):
    """A single-line table cell: whitespace collapsed, pipes escaped."""
    return _inline(text).replace("|", "\\|")


def _mcell(text):
    """A multi-line table cell: each line collapsed, blank lines dropped,
    joined with `<br>`, pipes escaped. Keeps a bulleted extraction instruction
    legible inside one cell."""
    lines = [" ".join(ln.split()) for ln in str(text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return "<br>".join(lines).replace("|", "\\|")


def _table(rows, header):
    """A GitHub-flavoured Markdown table whose cells pass through `_cell`."""
    ncol = len(header)
    out = ["| " + " | ".join(_cell(c) for c in header) + " |",
           "|" + "|".join(":---" for _ in range(ncol)) + "|"]
    out.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def _raw_table(rows, header):
    """A Markdown table whose cells are already formatted (pipes escaped,
    newlines rendered as `<br>`). Unlike `_table` it does not re-escape, so the
    operational cells can carry `<br>`, backticks, and `**`."""
    ncol = len(header)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(":---" for _ in range(ncol)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Value domain
# ---------------------------------------------------------------------------

def _values(field, labels):
    """The readable value domain of a field, for a human reader.

    In priority order:
      - a `canonical_reference` field renders the controlled-vocabulary
        requirement in words. When the referenced list declares a display
        `label` (see `reference_lists.load_reference_list_labels`), the label
        is the proper-noun phrase: "Names from the <label>" for a list field,
        "Name from the <label>" for a single-value one (the article "the"
        stays here in the template). With no label it falls back to the raw
        list id (the file stem) in backticks: "List of names from the `X`
        reference list" / "Name from the `X` reference list", where `X` is the
        list the value must come from. `labels` is `{list_name: label}` for
        the lists that declare one; a list absent from it takes the fallback;
      - a categorical field renders its allowed options, with `Other (specify)`
        appended when `allow_other` is set;
      - any other field renders its human type label from `_TYPES` ("Text",
        "Year", "Text (multiple)", ...), not the machine `field_type`.
    """
    ref = field["canonical_reference"]
    if ref:
        label = (labels or {}).get(ref)
        if label:
            if field["field_type"] == "string_list":
                return f"Names from the {label}"
            return f"Name from the {label}"
        if field["field_type"] == "string_list":
            return f"List of names from the `{ref}` reference list"
        return f"Name from the `{ref}` reference list"
    if field["field_type"] == "categorical":
        out = "; ".join(str(o) for o in field["options"])
        if field["allow_other"]:
            out += "; Other (specify)"
        return out
    return _TYPES.get(field["field_type"], field["field_type"])


def _op_values(field, labels):
    """The operational Type / values cell: the readable value domain collapsed
    to a single line with pipes escaped, for the raw table (which does not
    re-escape its cells). A stray newline in a reference-list `label:` or a
    categorical option collapses to one well-formed cell instead of spilling
    mid-row; on well-formed value domains this is a no-op."""
    return _cell(_values(field, labels))


# ---------------------------------------------------------------------------
# Operational view
# ---------------------------------------------------------------------------

def _op_field_cell(field):
    """The Field cell: the human-facing label in bold over the machine
    variable in code font. The label leads (this is a human document); the
    variable is the secondary an extraction team maps guidance to data
    columns by. It goes through `_cell` because nothing validates its surface
    form at load, so a pipe or newline in it cannot break the row."""
    return f"**{_mcell(field['label'])}**<br>`{_cell(field['variable'])}`"


def _op_field_row(field, labels):
    """One operational table row for `field`.

    Required renders Yes / No. Evidence renders the policy (`required` /
    `optional`) for envelope fields and is blank for the bare-value check
    blocks that carry no evidence policy. An absent extraction instruction
    leaves its cell blank.
    """
    return [
        _op_field_cell(field),
        _op_values(field, labels),
        "Yes" if field["required"] else "No",
        field["evidence"] or "",
        _mcell(field["description"]),
        _mcell(field["extraction_instruction"]),
    ]


def _split_qa(sections):
    """Split a scope's section list into `(ordinary, quality_assessment)`.

    A section marked `qa: true` is a quality-assessment section. The flag is
    presentation only: it groups quality-appraisal items under their own
    heading and keeps them out of the publication view; the fields themselves
    are ordinary fields, extracted, validated and checked like any other.
    Declaration order is preserved within each half.
    """
    return ([s for s in sections if not s["qa"]],
            [s for s in sections if s["qa"]])


def _op_section_blocks(blocks, labels):
    """Render a list of section dicts as `### <label>` subsections, each with
    its section-level extraction instruction (when present) and its field
    table."""
    out = []
    for block in blocks:
        out.append(f"### {block['label']}")
        out.append("")
        instr = block["extraction_instruction"]
        if instr and str(instr).strip():
            out.append(f"_Section extraction instruction._ {_mcell(instr)}")
            out.append("")
        fields = block["fields"]
        if fields:
            rows = [_op_field_row(f, labels) for f in fields]
            out.append(_raw_table(rows, _OP_HEADER))
            out.append("")
    return "\n".join(out).rstrip()


def render_operational(template, labels):
    """Render the operational Markdown document from a parsed template."""
    entity = template["record_entity"]
    study_sections, study_qa = _split_qa(template["study_fields"])
    record_sections, record_qa = _split_qa(template["record_fields"])
    parts = [
        "# Extraction template (operational)",
        "",
        "## Study-level extraction",
        "",
        _op_section_blocks(study_sections, labels),
        "",
        "## Record-level extraction",
        "",
        f"_Entity._ `{entity['singular']}` "
        f"(plural: {_inline(entity['plural'])})",
        "",
        f"_Description._ {_inline(entity['description'])}",
    ]
    rec_instr = entity["extraction_instruction"]
    if rec_instr and str(rec_instr).strip():
        parts += ["",
                  f"_Record extraction instruction._ {_mcell(rec_instr)}"]
    parts += ["", _op_section_blocks(record_sections, labels)]

    parts += ["", "## Study-level quality appraisal", "",
              _op_section_blocks(study_qa, labels)]
    if record_qa:
        parts += ["", "## Record-level quality appraisal", "",
                  _op_section_blocks(record_qa, labels)]

    parts += ["", "## Initial check", "",
              _op_section_blocks(template["initial_check_fields"], labels)]
    parts += ["", "## Quality check", "",
              _op_section_blocks(template["quality_check_fields"], labels)]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Publication view
# ---------------------------------------------------------------------------

def _pub_field_table(blocks, labels):
    rows = [[block["label"], f["label"], f["description"], _values(f, labels)]
            for block in blocks for f in block["fields"]]
    return _table(rows, header=_PUB_HEADER)


def render_publication(template, labels):
    """Render the publication Markdown document from a parsed template.

    Quality-assessment sections (`qa: true`) are left out at both scopes: this
    view is the academic reader's field list, and quality appraisal is reported
    separately from the extracted data.
    """
    entity = template["record_entity"]
    study_sections, _ = _split_qa(template["study_fields"])
    record_sections, _ = _split_qa(template["record_fields"])
    parts = [
        "# Extraction template (publication)",
        "",
        "## Study-level fields",
        "",
        _pub_field_table(study_sections, labels),
        "",
        "## Record-level fields",
        "",
        _inline(entity["description"]),
        "",
        _pub_field_table(record_sections, labels),
    ]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_VIEWS = {
    "operational": render_operational,
    "publication": render_publication,
}


def render_template(template, view, labels=None):
    """Render `template` (a `load_template` result) in `view`.

    `view` is `"operational"` or `"publication"`. An unknown view raises
    `ValueError` (the CLI restricts the choice, so this is a defensive guard
    for direct callers). Returns the Markdown document as a string.

    `labels` is the optional `{reference_list_name: display_label}` map from
    `reference_lists.load_reference_list_labels`, used to phrase a
    canonical-reference field's value domain with the list's human display
    name. It is presentation-only and defaults to none (every canonical
    reference then falls back to the raw list id in backticks).
    """
    try:
        renderer = _VIEWS[view]
    except KeyError:
        raise ValueError(
            f"unknown view {view!r}; expected one of {sorted(_VIEWS)}.")
    return renderer(template, labels)
