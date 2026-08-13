"""Per-field checker prompt assembly.

One LLM call per field a tool call just wrote. The system message is shared
(and cached); the user message is small and specific to one field: the
field's definition and allowed values, an identity label, and the current
value, evidence and field note, each quote shown again inside the paper text
around it. A field re-checked after a revision gets a genuinely fresh
context: nothing from the earlier check is carried in.

`build_checker_system_text(...)` returns the cacheable system prompt: the
checker's engine spine (what the checker is, what it is shown, what a verdict
means) followed by the bundle's checker prompt file, with `{include:...}`
partials expanded, `{reference:...}` lists rendered in, and the run's
per-field check budget substituted. No field catalogue: every field-specific
detail reaches the checker through the user message.
`build_checker_user_message(...)` returns that message as a list of content
blocks: the rendered text, preceded for an image-sourced field by a caption
block and the cropped PNG, which IS the evidence. The cache_control
wrapper is `prompt_builder.system_message_blocks`, re-exported here so a
checker call site need not reach across to the extractor's module.

Each quote is followed by the paper text SURROUNDING it, so the checker
rules on the evidence read in context rather than on the quote alone (a cell
reading `88 (7.0)` is a count or a percentage depending on a column header
the cell does not carry). Window width comes from the config bundle
(`checker_context_chars`, default 1000 characters each side, 0 for none);
placement comes from `quote_context`. The context is the PAPER's text: the
prose the extractor writes into its evidence string stays withheld, and the
field's `notes` remain the one sanctioned channel for the extractor's own
reasoning.

A record context label leads with the engine-assigned record id and, when
the template declares `checker_context_fields`, appends those fields' values
joined by ` | ` as a human-readable hint. The engine holds no
review-specific field names, and the record id is the only identifier
(content never identifies a record).

The user message's scaffold is the engine section `checker_user`, and the
engine writes the wording that fills its slots as well. Every slot is on an
allowlist (`config_bundle._CHECKER_USER_PLACEHOLDERS`), checked at load
against a bundle's override of that section, so an override that misspells
one fails there instead of shipping the literal token to the model. Slot
wording is FRAMING — engine text, not config — so it rides in no
fingerprint; the run's recorded `engine_fp` identifies it.

`{reference:NAME}` resolves in the scaffold on the same terms as in the three
system spines: the composed text is substituted once, before any per-field
slot is filled, so an override that cites a list gets the same rendered block
the extractor and the reviewer read. What the scaffold contributes to
`checker_fp` is the override as the author wrote it
(`checker_user_config_text`), placeholder and all; the list's own content
rides in `reference_hash` beside it, as it does for every other override.

The system prompt has an allowlist of its own
(`config_bundle._CHECKER_SYSTEM_PLACEHOLDERS`), holding the one slot this
function substitutes: `{max_checks_per_field}`. The checker is sent no image
labels, so `{image_labels_list}` in a bundle's checker prompt or in an
override of a checker section is a load error naming that variable rather
than a literal token in front of the model.
"""

import base64
import json
from pathlib import Path

from meltiro.prompt_builder import (  # noqa: F401
    bundle_root_for_prompt, system_message_blocks)
from meltiro.reference_lists import substitute_reference_placeholders
from meltiro.prompt_partials import (
    CHECKER_SYSTEM,
    CHECKER_USER,
    compose_engine_spine,
    config_prompt_preimage,
    engine_override_pairs,
    join_blocks,
    substitute_include_placeholders,
)
from meltiro.quote_context import (
    QUOTE_CLOSE_MARKER,
    QUOTE_OPEN_MARKER,
    quote_context_windows,
    render_window,
)


# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

def _load(path):
    return Path(path).read_text(encoding="utf-8").strip()


def _partials_dir(system_prompt_path):
    return Path(system_prompt_path).parent / "partials"


def _bundle_root_for_partials(partials_dir):
    """The config bundle directory holding `prompts/partials/`.

    The mirror of `prompt_builder.bundle_root_for_prompt` for the one render
    path that is handed the partials directory rather than a prompt file. Used
    only to locate a `ConfigBundleError`, so a caller rendering from elsewhere
    still gets a message naming what went wrong.
    """
    return Path(partials_dir).parent.parent


def _fill_slots(text, system_prompt_path, reference_lists,
                max_checks_per_field):
    text = substitute_reference_placeholders(
        text, reference_lists,
        path=bundle_root_for_prompt(system_prompt_path))
    return text.replace("{max_checks_per_field}", str(max_checks_per_field))


def render_checker_bundle_text(*, system_prompt_path, max_checks_per_field,
                               reference_lists=None, predicates=None):
    """Render the config bundle's own checker prompt file.

    The half a review writes, appended after the checker's engine spine on the
    wire and hashed on its own into `checker_fp` (see
    `build_checker_config_text`).
    """
    text = substitute_include_placeholders(
        _load(system_prompt_path), _partials_dir(system_prompt_path),
        predicates=predicates)
    return _fill_slots(text, system_prompt_path, reference_lists,
                       max_checks_per_field)


def build_checker_system_text(*, system_prompt_path, max_checks_per_field,
                              reference_lists=None, predicates=None):
    """Render the checker's system prompt text.

    The checker's engine spine first, then the config bundle's checker prompt
    file appended after it. `system_prompt_path` is REQUIRED; it comes from
    the config bundle (`ConfigBundle.checker_system_path`), and it also
    locates the `partials/` directory both halves resolve against.

    The system prompt is generic across all per-field calls; every
    field-specific detail lives in the per-field user message built by
    `build_checker_user_message`.

    Every `{reference:NAME}` placeholder is substituted with the rendered
    reference list (same blocks the extractor and reviewer see) so the
    checker has the canonical names in context.

    `{max_checks_per_field}` is substituted with the run's per-field check
    budget, the one slot this prompt has (`load_config_bundle` refuses any
    other, so nothing else can reach the checker as a literal token). It is
    REQUIRED and has no default: the value is the run's structure, held by
    `Instrument`, and a budget nobody chose would describe a run that never
    happened — the same reason `predicates` must be passed in rather than
    guessed.
    """
    spine = compose_engine_spine(
        CHECKER_SYSTEM, _partials_dir(system_prompt_path),
        predicates=predicates)
    return join_blocks(
        _fill_slots(spine, system_prompt_path, reference_lists,
                    max_checks_per_field),
        render_checker_bundle_text(
            system_prompt_path=system_prompt_path,
            max_checks_per_field=max_checks_per_field,
            reference_lists=reference_lists, predicates=predicates),
    )


def build_checker_config_text(*, system_prompt_path, max_checks_per_field,
                              reference_lists=None, predicates=None):
    """The config-owned identity of the checker's system prompt.

    The bundle's appended text as it renders, plus the bundle's overrides of
    the checker sections this run composes — and nothing the engine wrote (see
    `prompt_partials.config_prompt_preimage`). Folded into `checker_fp`, so
    rewording the engine's briefing moves `engine_fp` and leaves every
    bundle's `checker_fp` where it was.
    """
    return config_prompt_preimage(
        render_checker_bundle_text(
            system_prompt_path=system_prompt_path,
            max_checks_per_field=max_checks_per_field,
            reference_lists=reference_lists, predicates=predicates),
        engine_override_pairs(CHECKER_SYSTEM,
                              _partials_dir(system_prompt_path),
                              predicates=predicates),
    )


def render_checker_user_template(partials_dir, *, predicates,
                                 reference_lists=None):
    """The scaffold one per-field checker message is rendered from.

    The engine section `checker_user`, or the bundle's override of it. A spine
    of one: there is no bundle file to append, because a message whose whole
    content is engine-supplied slots has nothing for a review to add around
    them, and a review that wants different wording overrides the section.

    Every `{reference:NAME}` placeholder in the composed text is substituted
    with the named list's rendered block, once, before the caller fills the
    per-field slots — the same pass the three system spines get, so a name a
    scaffold cites resolves rather than reaching the checker as a token.
    """
    return substitute_reference_placeholders(
        compose_engine_spine(CHECKER_USER, partials_dir,
                             predicates=predicates),
        reference_lists,
        path=_bundle_root_for_partials(partials_dir))


def checker_user_config_text(partials_dir, *, predicates):
    """The config-owned identity of the per-field scaffold.

    The bundle's override of `checker_user` when it ships one, and nothing at
    all when it does not. Folded into `checker_fp` beside the system prompt's
    preimage, so a review that rewords the scaffold says so in the
    fingerprint and one that takes the engine's wording holds steady across a
    release that rewords it.
    """
    return config_prompt_preimage(
        "", engine_override_pairs(CHECKER_USER, partials_dir,
                                  predicates=predicates))


# ---------------------------------------------------------------------------
# Per-field user message
# ---------------------------------------------------------------------------

_NO_IDENTITY_CONTEXT = "(no identity context available)"

_NO_RECORD_ID = "(unidentified record)"

# Engine wording for the checker's identity context. The orchestrator
# resolves the per-paper VALUES (which summary, title, DOI, record label);
# the labels and the join are engine wording the checker reads, composed
# through the render functions below so it has a single home.
_SUMMARY_LABEL = "Summary: "

_TITLE_LABEL = "Title: "

_DOI_LABEL = "DOI: "

_IDENTITY_JOIN = "\n"


def render_study_identity_context(summary):
    """The checker's study-identity line for a resolved summary value.

    `summary` is the per-paper value the orchestrator picked (manifest summary
    or the extracted role:summary field); only the `Summary: ` label is engine
    wording.
    """
    return _SUMMARY_LABEL + summary


def render_degraded_identity_context(title, doi):
    """Minimal study-identity context (title + DOI) when no summary is
    available. `title` and `doi` are per-paper values, already stripped by the
    caller; empty components are skipped, and with neither present the
    no-identity fallback is returned.
    """
    parts = []
    if title:
        parts.append(_TITLE_LABEL + title)
    if doi:
        parts.append(_DOI_LABEL + doi)
    return _IDENTITY_JOIN.join(parts) if parts else _NO_IDENTITY_CONTEXT


def render_record_identity_context(study_ctx, record_ctx):
    """Join the study-identity context with the record-context label for a
    record-scoped check. Only the join is engine wording; both operands are
    already-rendered context strings.
    """
    return study_ctx + _IDENTITY_JOIN + record_ctx


def _envelope_value(env):
    if env is None:
        return None
    if isinstance(env, dict):
        return env.get("value")
    return env


def build_record_context(record, context_vars):
    """One-line record label for the checker's `{identity_context}` slot.

    The label ALWAYS leads with the engine-assigned record id: content is never
    used to identify a record. `context_vars` is the ordered list of record
    field names to read, from the loaded template's optional
    `checker_context_fields` (may be empty). When one or more of those fields
    are populated their values are appended after the record id, joined by
    ` | `, as a human-readable hint (for example `relationship_3 — Widget
    Durability Scale 9 (WDS-9) | Total score | Unplanned removal rate`);
    missing/empty components are skipped. With no context fields (an empty
    list, or none populated) the bare record id is returned.
    """
    record_id = record.get("record_id") or _NO_RECORD_ID
    parts = []
    for var in context_vars:
        val = _envelope_value(record.get(var))
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        parts.append(s)
    if not parts:
        return record_id
    return f"{record_id} — {' | '.join(parts)}"


def _render_extraction_instruction_block(field_spec):
    """The `{extraction_instruction_block}` slot: engine framing around the
    template's per-field instruction. Empty when the field declares none."""
    extraction_instr = field_spec.get("extraction_instruction") or ""
    if not extraction_instr:
        return ""
    return "\n_Extraction instruction_: " + extraction_instr.strip()


def _render_allowed_values_block(field_spec, value):
    """The `{allowed_values_block}` slot: how the checker is briefed on what
    the field may hold.

    The wording is engine framing; the option names and reference-list name are
    config and ride in `field_catalogue_hash`.
    """
    if field_spec.get("options"):
        opts_str = " | ".join(field_spec["options"])
        if field_spec.get("allow_other"):
            # allow_other: free text is allowed, so present the options with an
            # explicit trailing "Other (specify)" affordance in both branches.
            # How to brief depends on the stored value: when it is one of the
            # options, brief it as a hard choice; when it is free text outside
            # the list, ask the checker whether a listed option would fit
            # better.
            opts_with_other = opts_str + " | Other (specify)"
            if isinstance(value, str) and value in field_spec["options"]:
                return (
                    "_Allowed values_ (open list; the stored value is one of "
                    "the listed options, so treat it as a hard choice): "
                    + opts_with_other
                )
            return (
                "_Typical values_ (open list; the stored value is free "
                "text outside this list, which is permitted, but consider "
                "whether one of these canonical options is actually more "
                "appropriate): " + opts_with_other
            )
        return "_Allowed values_: " + opts_str

    if field_spec.get("canonical_reference"):
        # A reference field is a strict closed set: names come from the
        # reference list in the checker's system prompt. Spelling is
        # validator-guaranteed by the time the checker sees the value, so
        # the checker judges selection and evidence, not orthography. A
        # string_list field stores a real list of names.
        ref = field_spec["canonical_reference"]
        if field_spec.get("field_type") == "string_list":
            return (
                f"_Type_: list of names from the {ref} reference list shown "
                "in your system prompt. Spelling is validator-guaranteed; "
                "judge whether the selection of names is justified by the "
                "evidence."
            )
        return (
            f"_Allowed values_: an exact name from the {ref} reference "
            "list shown in your system prompt (off-list values are "
            "rejected)."
        )

    if field_spec.get("field_type"):
        return f"_Type_: {field_spec['field_type']}"
    return ""


def _render_notes_block(notes):
    """The `{notes_block}` slot: the extractor's note on THIS field.

    The note is the extractor's written-down reasoning for the value, and the
    checker is shown it: withholding it would leave the checker judging a
    value whose stated grounds it cannot see. The scope notes (the study's,
    each record's) are NOT rendered here or anywhere else in the checker's
    context: `orchestrator._build_checker_calls` builds one call per FIELD and
    a scope note belongs to no field, so nothing ever puts one in front of a
    checker.

    Returns "" when the field carries no note (whitespace counts as none), so
    the slot disappears rather than leaving an empty heading.
    """
    text = (notes or "").strip() if isinstance(notes, str) else ""
    if not text:
        return ""
    return (
        "\n## Extractor's note on this field\n\n"
        "The extractor recorded this as its own explanation of the value. It "
        "is commentary, not evidence: nothing in it has been verified against "
        "the paper.\n\n"
        f"{text}"
    )


_CONTEXT_LEAD_IN = (
    "The paper's own text around the quote, so the quote can be read in "
    "context rather than alone. The quoted span is wrapped in "
    f"{QUOTE_OPEN_MARKER} and {QUOTE_CLOSE_MARKER}; everything else is "
    "surrounding paper text. It is the paper's words, never the extractor's, "
    "and it is not itself the evidence offered: use it to settle what the "
    "evidence means (what a table column heads, what a pronoun refers to, "
    "what unit a number carries), then judge the value against the evidence "
    "read in that context."
)

_NO_CONTEXT_RESOLVED = (
    "no surrounding context could be resolved: this quote could not be "
    "located in the paper text, so it is shown above on its own and there is "
    "nothing further to read it against."
)


def _context_entry_label(quote_index, quote_count, passage_index,
                         passage_count):
    """The label line introducing one rendered window, or "" when the message
    carries a single window and there is nothing to disambiguate."""
    parts = []
    if quote_count > 1:
        parts.append(f"Quote {quote_index + 1}")
    if passage_count > 1:
        passage = f"passage {passage_index + 1} of {passage_count}"
        parts.append(passage if parts else passage.capitalize())
    if not parts:
        return ""
    return ", ".join(parts) + ":"


def _render_context_block(quotes, paper_text, context_chars):
    """The `_The quote in context_` section appended to the evidence block.

    One entry per quote: its windows, each rendered with the quoted span
    marked, or an explicit statement that no context could be resolved for
    it. Returns "" when the caller asked for no context (`context_chars` of
    0) or has no paper text to window into.
    """
    if context_chars <= 0 or not paper_text or not quotes:
        return ""

    entries = []
    rendered_any = False
    for i, quote in enumerate(quotes):
        windows = quote_context_windows(quote, paper_text, context_chars)
        if not windows:
            label = _context_entry_label(i, len(quotes), 0, 1)
            entries.append(
                f"{label} {_NO_CONTEXT_RESOLVED}".strip()
                if label else _NO_CONTEXT_RESOLVED.capitalize()
            )
            continue
        rendered_any = True
        for j, window in enumerate(windows):
            label = _context_entry_label(i, len(quotes), j, len(windows))
            body = render_window(paper_text, window)
            entries.append(f"{label}\n\n{body}" if label else body)

    heading = ("_The quotes in context_" if len(quotes) > 1
               else "_The quote in context_")
    body = "\n\n".join(entries)
    # The lead-in describes markers and surrounding text. With nothing
    # resolved there is neither, so the section is the statement alone rather
    # than a promise the message does not keep.
    if not rendered_any:
        return f"{heading}. {body}"
    return f"{heading}. {_CONTEXT_LEAD_IN}\n\n{body}"


def _render_quotes_section(quotes, lead="Verbatim quotes from the paper:"):
    """The quotes as the evidence the extractor offered, unchanged by the
    context machinery: one bare quoted string, or a numbered list."""
    if len(quotes) == 1:
        return f'"{quotes[0]}"'
    return lead + "\n" + "\n".join(
        f'  {i+1}. "{q}"' for i, q in enumerate(quotes)
    )


def _render_evidence_block(evidence, image_labels, paper_text=None,
                           context_chars=0):
    """Return (text_for_user_message, list_of_image_labels_to_attach).

    Parses the unified evidence string into quotes + images + prose, and
    withholds the prose from the checker: the evidence slot shows the
    mechanical evidence (verbatim quotes + image references) only. The
    extractor's argument belongs in the field's `notes`, rendered separately
    (`_render_notes_block`); prose smuggled inside the evidence string stays
    withheld. The image labels are returned so the caller can attach the
    cropped PNGs as content blocks.

    With `paper_text` and a positive `context_chars`, each quote is followed
    by the paper text surrounding it (see `_render_context_block`) — paper
    content, not extractor argument. An `<img>` citation has no text
    position, so image evidence is unaffected: the cropped PNG is attached
    and is itself the evidence. An evidence string carrying no tags at all
    cites no image, so the whole content is treated as a single quote.
    """
    if not evidence:
        return ("(no evidence provided)", [])

    s = str(evidence)
    # Tagged string: parse out quotes + images, leave the prose unrendered.
    if "<q>" in s or "<img>" in s:
        from meltiro.quote_check import (
            parse_evidence_string,
        )
        quotes, images, _prose = parse_evidence_string(s)
        quotes = [q.strip() for q in quotes]
        sections = []
        if quotes:
            sections.append(_render_quotes_section(quotes))
            sections.append(
                _render_context_block(quotes, paper_text, context_chars))
        valid_imgs = []
        if images:
            valid_imgs = [
                img.strip() for img in images
                if img.strip().lower() in image_labels
            ]
            if len(valid_imgs) == 1:
                sections.append(
                    f"Image reference: `{valid_imgs[0]}`; the cropped "
                    "image is attached below; treat it AS the evidence."
                )
            elif valid_imgs:
                sections.append(
                    "Image references (cropped images attached below): "
                    + ", ".join(f"`{img}`" for img in valid_imgs)
                )
        return (_join_sections(sections) or "(no evidence provided)",
                [img.lower() for img in valid_imgs])

    # Untagged, prose-only evidence is a live shape, not a fault: a model may
    # answer with bare prose. Treat the whole string as a single quote.
    return (_join_sections([
        _render_quotes_section([s]),
        _render_context_block([s], paper_text, context_chars),
    ]), [])


def _join_sections(sections):
    """Join the evidence block's sections, dropping the empty ones (an absent
    context block must leave no blank gap behind it)."""
    return "\n\n".join(s for s in sections if s)


def _image_caption(label):
    """The caption block that precedes an attached image, so the checker can
    correlate the PNG with the image reference in the text body."""
    return f"[{label}]"


def _attach_image_block(label, figures):
    """Read the cropped PNG for an image-cited field from the paper
    bundle's `figures` map (label -> Path) and return an image content block.
    Lookup is case-insensitive so it tolerates the checker's lower-cased
    label. Returns None if the label isn't in the map or the file is missing
    (caller decides how to surface that)."""
    if not figures:
        return None
    lookup = {str(k).lower(): v for k, v in figures.items()}
    png_path = lookup.get(str(label).lower())
    if png_path is None:
        return None
    png_path = Path(png_path)
    if not png_path.exists():
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png_path.read_bytes()).decode("ascii"),
        },
    }


def build_checker_user_message(
        field_path,
        field_spec,
        envelope,
        identity_context,
        image_labels,
        *,
        partials_dir,
        figures=None,
        paper_text=None,
        context_chars=0,
        predicates=None,
        reference_lists=None,
):
    """Build the per-field user message.

    Args:
        field_path: e.g. "study.primary_aim" or
            "record.relationship_3.effect_size".
        field_spec: template field dict (variable, description,
            allowed_values, options, extraction_instruction).
        envelope: the {value, evidence, notes} envelope under review. The
            field's own note is rendered into `{notes_block}`; the scope notes
            (study, record) are never shown to the checker.
        identity_context: study-identity summary string (study) or the study
            context plus the record context label (record).
        image_labels: set of figure stems (lower-case) used to detect
            image-sourced evidence.
        partials_dir: REQUIRED; the config bundle's `prompts/partials/`
            (`ConfigBundle.partials_dir`). The scaffold comes from the engine
            section `checker_user`, and this is where a bundle's override of
            it is read from.
        figures: the paper bundle's figures map (label -> png Path). Used
            to attach the cropped PNG for image-cited evidence.
        paper_text: the paper's full text, used to window the surrounding
            context around each quote. None means no context is rendered.
        context_chars: characters of surrounding paper text shown on each
            side of a matched quote (`checker_context_chars` in
            pipeline.yaml). 0, the default here, renders no context at all.
        reference_lists: the config bundle's loaded reference lists, rendered
            into any `{reference:NAME}` the scaffold cites.

    Returns a list of content blocks suitable for the user message of the
    checker call: the rendered text as one block, preceded for an
    image-sourced field by a caption block and the cropped PNG per image
    its evidence cites. No cache_control on any of them: what the checker's
    calls share is the system prompt, and that is where the marker goes
    (`prompt_builder.system_message_blocks`); this message is one field's.
    """
    if envelope is None:
        envelope = {}
    value = envelope.get("value")
    evidence = envelope.get("evidence")
    notes = envelope.get("notes")

    tmpl = render_checker_user_template(
        partials_dir, predicates=predicates, reference_lists=reference_lists)

    extraction_instr_block = _render_extraction_instruction_block(field_spec)
    allowed_block = _render_allowed_values_block(field_spec, value)

    evidence_text, image_attach_labels = _render_evidence_block(
        evidence, image_labels, paper_text=paper_text,
        context_chars=context_chars)

    notes_block = _render_notes_block(notes)

    rendered = (
        tmpl
        .replace("{field_path}", field_path)
        .replace("{field_description}",
                 (field_spec.get("description") or "").strip())
        .replace("{extraction_instruction_block}", extraction_instr_block)
        .replace("{allowed_values_block}", allowed_block)
        .replace("{identity_context}",
                 (identity_context or _NO_IDENTITY_CONTEXT).strip())
        .replace("{evidence_block}", evidence_text)
        .replace("{value}", _format_value(value))
        .replace("{notes_block}", notes_block)
    )

    blocks = [{"type": "text", "text": rendered}]

    # Attach every cropped PNG referenced by an <img>...</img> tag in the
    # evidence string. Images go BEFORE the text in the user message, so the
    # checker reads the question after the image it is about; a tiny "[label]"
    # caption block precedes each image so the checker can correlate them with
    # the references in the text body.
    if image_attach_labels and figures:
        image_blocks = []
        for label in image_attach_labels:
            img_block = _attach_image_block(label, figures)
            if img_block is not None:
                image_blocks.extend([
                    {"type": "text", "text": _image_caption(label)},
                    img_block,
                ])
        blocks = image_blocks + blocks

    return blocks


def _format_value(value):
    if value is None:
        return "null"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        # Render lists as JSON so a string_list value (e.g. a list of
        # reference names) reads unambiguously in the checker message.
        return json.dumps(value, ensure_ascii=False)
    return str(value)
