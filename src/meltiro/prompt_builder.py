"""Build the extractor's and reviewer's system + user messages.

The system message is the LARGE cacheable block: review context, workflow
description, evidence convention, image-label list, rendered reference lists.
The field catalogue is NOT in it; it lives in the tool `input_schema`s, which
are built from the extraction template (see `meltiro.tools`).
The initial user message is also cached: paper text + every cropped
image. Both get `cache_control: ephemeral` markers so turns 2..N pay the
0.1x cache-read rate on the bulk of the prompt.

After the first turn the orchestrator only appends `tool_use` /
`tool_result` blocks; those are not cached and grow per turn.

This module owns every piece of engine-authored FRAMING for the extractor
and reviewer roles: the text *meltiro* itself writes around the config
bundle's prompts, as opposed to the prompt files (config) and the paper
(input). That includes the user-block headers and each role's tool re-prompt
(separate texts naming separate tools). Each piece of engine wording lives
here, next to the builders that emit it, so it has a single home.

Nothing here tells the reviewer that a checker exists. The reviewer is shown
the paper, the figures, and the assembled extraction output, and nothing
else: a checker challenge is a per-field note delivered in a tool result,
never a preamble to the review.

Framing is engine text, so it rides in no fingerprint; the run's recorded
`engine_fp` identifies it (see `fingerprint`'s module docstring).
"""

import base64
import hashlib
import json
from pathlib import Path

from meltiro.reference_lists import substitute_reference_placeholders
from meltiro.prompt_partials import (
    stage_predicates,
    substitute_include_placeholders,
)


def _load_text(path):
    return Path(path).read_text(encoding="utf-8").strip()


def bundle_root_for_prompt(system_prompt_path):
    """The config bundle directory holding `system_prompt_path`.

    Every prompt a bundle owns sits at `<root>/prompts/<name>.md`
    (`config_bundle.load_config_bundle` requires exactly that layout), so the
    root is two levels up. It is used only to locate a `ConfigBundleError`, so
    a caller rendering a prompt from somewhere else still gets a message
    naming what went wrong; only the directory it points at would be off.
    """
    return Path(system_prompt_path).parent.parent


# ---------------------------------------------------------------------------
# Engine-authored framing (extractor + reviewer roles)
# ---------------------------------------------------------------------------

def _extractor_header(study_id):
    """The extractor's opening framing line.

    One definition, used by both `build_initial_user_blocks` (what is sent)
    and `render_user_prompt_text` (what is recorded), so the transcript
    record cannot drift from the message.
    """
    return (
        f"Extract study {study_id}. The full text follows; "
        "every cropped table and figure for this paper is attached below."
    )


EXTRACTOR_TOOL_REPROMPT = (
    "You must call a tool to make progress. Use "
    "update_study / add_record to extract, or "
    "mark_complete when you believe the extraction output "
    "is done."
)

# The reviewer's own tool re-prompt, sent when a review turn returns text
# and no tool call. Names the reviewer's tools rather than the extractor's
# (the read-only view tools are the reviewer's to use), so it is a second
# piece of framing, not a reuse of the first.
REVIEW_TOOL_REPROMPT = (
    "You must call a tool to make progress. Use the view tools "
    "to inspect the extraction output, update_study / "
    "update_record / add_record / remove_record to revise it, "
    "or mark_complete when you are satisfied with it."
)

# Stand-in text for an assistant turn that carried neither text nor a tool
# call (for example a thinking-only truncation). The API rejects an empty
# assistant message, and omitting the turn entirely would land the re-prompt
# as a second consecutive user message; a placeholder keeps user/assistant
# alternation intact both live and on replay. It goes into the conversation
# as an assistant turn, so the model re-reads it on every subsequent request:
# framing like any other.
EMPTY_ASSISTANT_PLACEHOLDER = "(the model returned no text or tool call.)"


def _render_image_labels(image_labels, captions=None):
    """Render the image-label list a role is shown in its system prompt.

    Each line carries the label and, when the paper bundle declared one, the
    exhibit's caption: a bare `table_01` says nothing about what the image
    holds, so a model reading it has to guess which crop to cite. The label
    stays first and stays code-formatted on its own, because it is what an
    `<img>label</img>` citation must contain; the caption follows it as
    description, after a colon.

    `captions` is a label -> caption map (`PaperBundle.exhibits`). A label with
    no entry renders as the bare label, so a caller with no caption map (a
    fingerprint render, a test) gets the label list and nothing else.
    """
    if not image_labels:
        return "(no figures or tables were cropped for this study)"
    captions = captions or {}
    lines = []
    for lbl in sorted(image_labels):
        caption = captions.get(lbl)
        lines.append(f"- `{lbl}`: {caption}" if caption else f"- `{lbl}`")
    return "\n".join(lines)


def build_system_message(image_labels, *,
                         system_prompt_path,
                         max_checks_per_field=2,
                         final_review=True,
                         reference_lists=None,
                         image_captions=None):
    """Build the extractor's system message text.

    `system_prompt_path` is REQUIRED; it comes from the config bundle
    (`ConfigBundle.extractor_system_path`).

    Every `{reference:NAME}` placeholder is substituted with the config
    bundle's rendered reference list; an unresolvable one fails loudly.

    `image_captions` is the paper bundle's label -> caption map, rendered
    alongside each label in `{image_labels_list}` so the extractor can tell
    which crop to cite without guessing.

    The tool-call cap has no placeholder: `{max_tool_calls}` is not
    substituted, and `load_config_bundle` rejects any prompt that cites it —
    the cap is an operational budget kept out of `prompt_hash` and
    `config_fp` (see `config_bundle`).

    Returns the rendered string. The orchestrator wraps it in a
    cache_control text block before sending.
    """
    prompt_template = _load_text(system_prompt_path)
    rendered = prompt_template
    # Expand `{include:NAME}` partials BEFORE reference substitution, so a
    # partial may itself carry `{reference:...}` placeholders.
    rendered = substitute_include_placeholders(
        rendered, Path(system_prompt_path).parent / "partials",
        predicates=stage_predicates(max_checks_per_field, final_review))
    rendered = substitute_reference_placeholders(
        rendered, reference_lists,
        path=bundle_root_for_prompt(system_prompt_path))
    rendered = rendered.replace(
        "{image_labels_list}",
        _render_image_labels(image_labels, image_captions))
    rendered = rendered.replace("{max_checks_per_field}",
                                str(max_checks_per_field))
    return rendered


def compute_prompt_config_hash(*, system_prompt_path,
                                max_checks_per_field, reference_lists,
                                final_review=True):
    """Paper-independent hash of the system prompt CONFIG.

    Renders the prompt with an empty `image_labels` list so the hash
    reflects the prompt template + reference lists + per-field check budget
    only, NOT the per-paper figure/table label set (nor, therefore, the
    per-paper exhibit captions rendered beside those labels). Two extractions
    of different papers under the same code share this hash.

    Because the rendered prompt inlines the reference lists' contents, this
    hash also captures reference-list CONTENT: editing a reference list
    moves the prompt_hash, and hence config_fp, with it. The tool-call cap is
    NOT an input: it has no placeholder in the prompt (see
    `build_system_message`), so raising it never moves this hash.

    Used as the `prompt_hash` input to `config_fp`, so a consumer grouping
    runs by that fingerprint groups by config, not by paper.
    """
    text = build_system_message(
        image_labels=[],
        system_prompt_path=system_prompt_path,
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        reference_lists=reference_lists,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def system_message_blocks(text):
    """Wrap the rendered system text in the Anthropic cache_control shape."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def render_user_prompt_text(study_id, paper_text, image_labels):
    """Render the text-only view of the initial user message.

    Mirrors `build_initial_user_blocks` minus the base64 image bytes,
    suitable for capturing inline into the session as the canonical
    "what was the user prompt" record. The exact text strings match
    those `build_initial_user_blocks` emits as text content blocks.
    """
    header = _extractor_header(study_id)
    parts = [
        header,
        "--- PAPER TEXT ---\n" + (paper_text or "")
        + "\n--- END PAPER TEXT ---",
    ]
    for label in image_labels:
        parts.append(f"[{label}]")
        parts.append(f"(image: {label}.png)")
    return "\n\n".join(parts)


def build_initial_user_blocks(study_id, paper_text, figures):
    """Build the initial user message content blocks.

    Args:
        study_id: study identifier (used in the header).
        paper_text: the paper's full text (from PaperBundle.text).
        figures: list of (label, png_bytes) tuples.

    Returns a list of content blocks. The LAST block carries
    cache_control: ephemeral so the whole user message caches.
    """
    blocks = []
    blocks.append({"type": "text", "text": _extractor_header(study_id)})

    blocks.append({
        "type": "text",
        "text": "--- PAPER TEXT ---\n" + paper_text + "\n--- END PAPER TEXT ---",
    })

    for label, png_bytes in figures:
        # Label first so the model can refer to it by name in `source`.
        blocks.append({"type": "text", "text": f"[{label}]"})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            },
        })

    # Attach cache_control to the last block; caches the whole user
    # prefix up to and including it.
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_review_system_message(image_labels, *,
                                 system_prompt_path,
                                 max_checks_per_field=2,
                                 final_review=True,
                                 reference_lists=None,
                                 image_captions=None):
    """Build the FINAL REVIEW system message text.

    `system_prompt_path` is REQUIRED; it comes from the config bundle
    (`ConfigBundle.review_system_path`).

    The review prompt is separate from the extractor's: it frames the model
    as the reviewer of an already-completed extraction. The same image labels
    are rendered in, with the exhibit captions beside them exactly as the
    extractor saw them; the field catalogue reaches it through the tool
    `input_schema`s, as it does the extractor. `{reference:NAME}`
    placeholders are substituted; the tool-call cap placeholders are rejected
    at config-load time (see `build_system_message`).
    """
    prompt_template = _load_text(system_prompt_path)
    rendered = prompt_template
    # Expand `{include:NAME}` partials BEFORE reference substitution, so a
    # partial may itself carry `{reference:...}` placeholders.
    rendered = substitute_include_placeholders(
        rendered, Path(system_prompt_path).parent / "partials",
        predicates=stage_predicates(max_checks_per_field, final_review))
    rendered = substitute_reference_placeholders(
        rendered, reference_lists,
        path=bundle_root_for_prompt(system_prompt_path))
    rendered = rendered.replace(
        "{image_labels_list}",
        _render_image_labels(image_labels, image_captions))
    rendered = rendered.replace("{max_checks_per_field}",
                                str(max_checks_per_field))
    return rendered


def build_review_user_blocks(study_id, paper_text, figures,
                             extraction_record_dict):
    """Build the user content blocks for the final-review pass.

    The reviewer sees the paper text + all cropped images + the
    assembled extraction output as a JSON block, framed as "review and confirm
    or revise". Last block carries cache_control.

    Nothing here reveals that a checker exists: no challenge, no rationale,
    no count of contested cells. The reviewer forms its own view — the
    independent second opinion the stage is for — and telling it which cells
    a narrower model doubted would anchor that view on the checker's. The
    same argument excludes the extractor's own check blocks, stripped by the
    caller (`ExtractionRecord.to_dict(include_checks=False)`): the reviewer
    records its OWN quality check when it concludes. The blocks are on disk
    and in the run record either way; they are withheld from the one stage
    whose value depends on not having seen them.
    """
    blocks = []
    header = (
        f"You are reviewing the assembled extraction output for "
        f"study {study_id}. Read the paper, examine the "
        f"figures, and decide whether the extraction output is correct and "
        f"complete. You have as many turns as you need: the view tools let "
        f"you inspect the extraction output and return their results to you, "
        f"and the editing tools let you revise it. Call `mark_complete` when "
        f"satisfied. Do NOT make stylistic changes; only revise "
        f"where the existing extraction is wrong."
    )
    blocks.append({"type": "text", "text": header})

    blocks.append({
        "type": "text",
        "text": ("--- PAPER TEXT ---\n" + paper_text
                 + "\n--- END PAPER TEXT ---"),
    })

    for label, png_bytes in figures:
        blocks.append({"type": "text", "text": f"[{label}]"})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            },
        })

    extraction_record_text = json.dumps(
        extraction_record_dict, indent=2, ensure_ascii=False)
    blocks.append({
        "type": "text",
        "text": (
            "--- ASSEMBLED EXTRACTION OUTPUT (to review) ---\n"
            + extraction_record_text
            + "\n--- END EXTRACTION OUTPUT ---\n\n"
            "If the extraction output is correct, call `mark_complete` with a one-"
            "sentence summary. If it needs revisions, call `update_study` "
            "/ `update_record` / `add_record` / "
            "`remove_record` first, then `mark_complete`. The review ends only "
            "when you call `mark_complete`, so call it once you are satisfied."
        ),
    })

    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks
