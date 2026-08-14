"""Build the extractor's and reviewer's system + user messages.

The system message is the LARGE cacheable block: the role's engine prompt
(`meltiro.prompt_partials`), which closes the tag naming the role it briefs,
and then the config bundle's own prompt file for that role, with the rendered
reference lists substituted into both halves. Nothing about the paper is in
it, so one config yields one system message however many studies it extracts.
The field catalogue is NOT in it either; it lives in the tool `input_schema`s,
which are built from the extraction template (see `meltiro.tools`).
The initial user message is also cached: paper text + every cropped image,
each image under its label and the paper's caption for it, which is where a
role learns what an `<img>` citation may name — and, where the role is sent
none, the statement that none accompany the study, so exhibit presence is a
fact of the message on both sides of it. Both get
`cache_control: ephemeral` markers so turns 2..N pay the 0.1x cache-read rate
on the bulk of the prompt.

After the first turn the orchestrator only appends `tool_use` /
`tool_result` blocks; those are not cached and grow per turn.

This module owns every piece of engine-authored FRAMING for the extractor
and reviewer roles: the text *meltiro* itself writes around the config
bundle's prompts, as opposed to the prompt files (config) and the paper
(input). That includes the user-block headers and each role's tool re-prompt
(separate texts naming separate tools). Each piece of engine wording lives
here, next to the builders that emit it, so it has a single home. The handover
from the engine's half of a system message to the review's is not among them:
the engine's half closes its own `<extractor>` or `<reviewer>` tag, so the seam
is marked by the prompt file rather than by wording composed around it.

Nothing the reviewer is HANDED names a checker. It is shown the paper, the
figures, and the assembled extraction output, and nothing else: a challenge is
a per-field note delivered in the tool result of a write the reviewer itself
made, never a preamble to the review and never a report on the extractor's
run. What the reviewer's engine prompt says about that protocol is the
engine's own text (`meltiro.engine_prompts`), and it follows the toggle that
decides whether such a note can arrive at all: the checker running AND
`check_reviewer_edits` on.

Framing is engine text, so it rides in no fingerprint; the run's recorded
`engine_fp` identifies it (see `fingerprint`'s module docstring).
"""

import base64
import hashlib
import json
from pathlib import Path

from meltiro.reference_lists import substitute_reference_placeholders
from meltiro.prompt_partials import (
    EXTRACTOR_SYSTEM,
    REVIEW_SYSTEM,
    compose_engine_prompt,
    config_prompt_preimage,
    engine_override_pairs,
    join_blocks,
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


# The nudge a tool-free extractor turn is answered with. It names
# `record_initial_check` FIRST, because that is the first move the workflow
# demands and the one the dispatcher gates everything else behind: a model
# re-prompted toward `update_study` before the initial check has landed is
# being pointed at a call that will be refused, and the turn after it is
# another tool-free turn against the same bound.
EXTRACTOR_TOOL_REPROMPT = (
    "You must call a tool to make progress. If you have not yet called "
    "record_initial_check, call it now: the dispatcher refuses every "
    "extraction call until it lands. Otherwise use update_study / add_record "
    "to extract, or mark_complete when you believe the extraction output is "
    "done."
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


# The text block a role reads in place of the attachments when its figure
# sequence is empty. Presence of an exhibit is a fact about the MESSAGE, not
# about the machinery, so the message is where it is stated: a bundle whose
# manifest declares `exhibits: []` and a text-only role that is sent no image
# parts both arrive here, and each reads the same statement rather than a
# system prompt promising an attachment that never follows.
NO_EXHIBITS_NOTICE = "(no cropped figures or tables accompany this study)"


def image_label_text(label, captions=None):
    """The text block that introduces one attached exhibit.

    The label first and alone in brackets, because it is what an
    `<img>label</img>` citation must contain; the paper's own caption after
    it, so a model reading `[table_01]` can tell which crop it is looking at
    without guessing. `captions` is a label -> caption map
    (`PaperBundle.exhibits`), looked up on the same normalised key the
    dispatcher matches a citation on; a label the map has no entry for renders
    as the bare label, which is what a caller with no caption map at all gets.

    One definition, used by the message builders and by the text-only render
    the session captures, so the recorded prompt cannot drift from the
    message.
    """
    caption = (captions or {}).get(str(label).strip().lower())
    return f"[{label}] {caption}" if caption else f"[{label}]"


def _partials_dir(system_prompt_path):
    return Path(system_prompt_path).parent / "partials"


def _fill_slots(text, system_prompt_path, *, reference_lists,
                max_checks_per_field):
    """Substitute everything the engine puts into a rendered prompt.

    Applied to the engine's half and to the bundle's appended text by the
    same call, so a slot means the same thing whichever half wrote it.

    Two substitutions and no third: the bundle's reference lists, and the
    run's per-field check budget. A system message therefore says nothing
    about a particular paper, which is what makes it identical across every
    study extracted under one config.
    """
    text = substitute_reference_placeholders(
        text, reference_lists,
        path=bundle_root_for_prompt(system_prompt_path))
    return text.replace("{max_checks_per_field}", str(max_checks_per_field))


def render_bundle_prompt_text(system_prompt_path, *, predicates,
                              reference_lists=None, max_checks_per_field=2):
    """Render the config bundle's own prompt file for a role.

    The half a review writes: its prompt file with `{include:NAME}` partials
    expanded, `{reference:NAME}` lists inlined, and the engine's slots filled.
    It is appended after the engine's half on the wire, and it is the
    `prompt` component of that role's config preimage, so the text a model
    reads and the text a fingerprint covers come off one function.
    """
    text = substitute_include_placeholders(
        _load_text(system_prompt_path), _partials_dir(system_prompt_path),
        predicates=predicates)
    return _fill_slots(
        text, system_prompt_path, reference_lists=reference_lists,
        max_checks_per_field=max_checks_per_field)


def _render_engine_half(role, system_prompt_path, *, predicates,
                        reference_lists, max_checks_per_field):
    text = compose_engine_prompt(
        role, _partials_dir(system_prompt_path), predicates=predicates)
    return _fill_slots(
        text, system_prompt_path, reference_lists=reference_lists,
        max_checks_per_field=max_checks_per_field)


def build_system_message(*,
                         system_prompt_path,
                         max_checks_per_field=2,
                         final_review=True,
                         check_reviewer_edits=False,
                         reference_lists=None):
    """Build the extractor's system message text.

    The extractor's engine prompt first, then the config bundle's prompt file
    appended after it. `system_prompt_path` is REQUIRED; it comes from the
    config bundle (`ConfigBundle.extractor_system_path`), and it also locates
    the `partials/` directory both halves resolve against.

    Every `{reference:NAME}` placeholder is substituted with the config
    bundle's rendered reference list; an unresolvable one fails loudly.

    Nothing about the paper reaches here. The cropped exhibits the extractor
    may cite are labelled where they arrive, in the user message
    (`build_initial_user_blocks`), so this text is one string for the whole
    config rather than one per study.

    The tool-call cap has no placeholder: `{max_tool_calls}` is not
    substituted, and `load_config_bundle` rejects any prompt that cites it —
    the cap is an operational budget kept out of `prompt_hash` and
    `config_fp` (see `config_bundle`).

    Returns the rendered string. The orchestrator wraps it in a
    cache_control text block before sending.
    """
    predicates = stage_predicates(max_checks_per_field, final_review,
                                  check_reviewer_edits)
    slots = dict(reference_lists=reference_lists,
                 max_checks_per_field=max_checks_per_field)
    return join_blocks(
        _render_engine_half(EXTRACTOR_SYSTEM, system_prompt_path,
                            predicates=predicates, **slots),
        render_bundle_prompt_text(system_prompt_path, predicates=predicates,
                                  **slots),
    )


def build_config_prompt_text(role, *, system_prompt_path,
                             max_checks_per_field, reference_lists,
                             final_review=True,
                             check_reviewer_edits=False):
    """The config-owned identity of one role's prompt, as a canonical string.

    Two components (see `prompt_partials.config_prompt_preimage`): the
    bundle's appended text as it renders, and the bundle's overrides of the
    engine prompts this run composes for `role`.

    Nothing of the paper reaches it, because nothing of the paper reaches a
    system message: two extractions of different papers under one config
    share the value.
    Reference-list CONTENT does reach it, inlined into the bundle's own text:
    editing a list moves it. Values the engine substitutes into text of the
    author's own reach it too — a bundle that writes `{max_checks_per_field}`
    into its prompt hashes the number.

    Engine text reaches it never. Whatever the engine's half says, and
    whatever the engine substitutes into it, is outside the preimage: the
    check budget stated in `extractor_checker_feedback` moves this by exactly
    nothing. That is the boundary, not a gap in it. The budget reaches a run's
    identity on the structure axis instead — `structure_hash`, folded into
    `config_fp` beside this value and into `instrument_fp` — so two runs
    differing only in the budget carry different fingerprints while the prompt
    component they share correctly reports the same authored text.
    """
    predicates = stage_predicates(max_checks_per_field, final_review,
                                  check_reviewer_edits)
    return config_prompt_preimage(
        render_bundle_prompt_text(
            system_prompt_path, predicates=predicates,
            reference_lists=reference_lists,
            max_checks_per_field=max_checks_per_field),
        engine_override_pairs(role, _partials_dir(system_prompt_path),
                              predicates=predicates),
    )


def compute_prompt_config_hash(*, system_prompt_path,
                                max_checks_per_field, reference_lists,
                                final_review=True,
                                check_reviewer_edits=False):
    """Paper-independent hash of the extractor prompt's CONFIG.

    SHA-256 of `build_config_prompt_text` for the extractor role: what the
    config author wrote for this role and nothing else. The tool-call cap
    reaches it never — it has no placeholder in any prompt (see
    `build_system_message`), so raising it and resuming is not refused as
    config drift.

    Used as the `prompt_hash` input to `config_fp`, so a consumer grouping
    runs by that fingerprint groups by config, not by paper.
    """
    text = build_config_prompt_text(
        EXTRACTOR_SYSTEM,
        system_prompt_path=system_prompt_path,
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        check_reviewer_edits=check_reviewer_edits,
        reference_lists=reference_lists,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def system_message_blocks(text):
    """Wrap the rendered system text in the canonical cache_control shape."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def render_user_prompt_text(study_id, paper_text, image_labels,
                            image_captions=None):
    """Render the text-only view of the initial user message.

    Mirrors `build_initial_user_blocks` minus the base64 image bytes,
    suitable for capturing inline into the session as the canonical
    "what was the user prompt" record. The exact text strings match
    those `build_initial_user_blocks` emits as text content blocks, captions
    and the no-exhibits notice included.

    `image_labels` is the labels of the figure sequence that message carries,
    in the order it carries them. A caller passing a re-sorted or re-cased set
    would record a prompt naming the same exhibits in a different form from the
    one that was sent, which is the one thing this function exists to rule out.
    """
    header = _extractor_header(study_id)
    parts = [
        header,
        "--- PAPER TEXT ---\n" + (paper_text or "")
        + "\n--- END PAPER TEXT ---",
    ]
    labels = list(image_labels)
    if not labels:
        parts.append(NO_EXHIBITS_NOTICE)
    for label in labels:
        parts.append(image_label_text(label, image_captions))
        parts.append(f"(image: {label}.png)")
    return "\n\n".join(parts)


def build_initial_user_blocks(study_id, paper_text, figures,
                              image_captions=None):
    """Build the initial user message content blocks.

    Args:
        study_id: study identifier (used in the header).
        paper_text: the paper's full text (from PaperBundle.text).
        figures: list of (label, png_bytes) tuples.
        image_captions: label -> caption map (`PaperBundle.exhibits`), so each
            attachment arrives under the caption the paper prints beside it.

    Returns a list of content blocks. The LAST block carries
    cache_control: ephemeral so the whole user message caches.

    An empty `figures` gets `NO_EXHIBITS_NOTICE` in their place, so a role
    reads what accompanies this study either way rather than inferring the
    absence from a message that simply stops.
    """
    blocks = []
    blocks.append({"type": "text", "text": _extractor_header(study_id)})

    blocks.append({
        "type": "text",
        "text": "--- PAPER TEXT ---\n" + paper_text + "\n--- END PAPER TEXT ---",
    })

    if not figures:
        blocks.append({"type": "text", "text": NO_EXHIBITS_NOTICE})

    for label, png_bytes in figures:
        # Label first so the model can refer to it by name in `source`, and
        # the caption beside it so it knows which exhibit it is looking at.
        blocks.append({"type": "text",
                       "text": image_label_text(label, image_captions)})
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


def build_review_system_message(*,
                                 system_prompt_path,
                                 max_checks_per_field=2,
                                 final_review=True,
                                 check_reviewer_edits=False,
                                 reference_lists=None):
    """Build the FINAL REVIEW system message text.

    The reviewer's engine prompt first, then the config bundle's review
    prompt file appended after it. `system_prompt_path` is REQUIRED; it comes
    from the config bundle (`ConfigBundle.review_system_path`).

    The reviewer's engine prompt is its own: it frames the model as the reader
    of an already-completed extraction, where the extractor's frames the
    writer of one. Its cropped exhibits arrive labelled in the user message
    exactly as the extractor's did (`build_review_user_blocks`); the field
    catalogue reaches it through the tool `input_schema`s, as it does the
    extractor. `{reference:NAME}` placeholders are substituted; the tool-call
    cap placeholders are rejected at config-load time (see
    `build_system_message`).
    """
    predicates = stage_predicates(max_checks_per_field, final_review,
                                  check_reviewer_edits)
    slots = dict(reference_lists=reference_lists,
                 max_checks_per_field=max_checks_per_field)
    return join_blocks(
        _render_engine_half(REVIEW_SYSTEM, system_prompt_path,
                            predicates=predicates, **slots),
        render_bundle_prompt_text(system_prompt_path, predicates=predicates,
                                  **slots),
    )


def build_review_user_blocks(study_id, paper_text, figures,
                             extraction_record_dict, image_captions=None):
    """Build the user content blocks for the final-review pass.

    The reviewer sees the paper text + all cropped images, each under its
    label and the paper's caption for it, + the assembled extraction output as
    a JSON block, framed as "review and confirm or revise". With no images to
    attach it reads `NO_EXHIBITS_NOTICE` where they would have been, on the
    same terms as the extractor. Last block carries cache_control.

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

    if not figures:
        blocks.append({"type": "text", "text": NO_EXHIBITS_NOTICE})

    for label, png_bytes in figures:
        blocks.append({"type": "text",
                       "text": image_label_text(label, image_captions)})
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
            "If the extraction output is correct, call `mark_complete`. "
            "If it needs revisions, call `update_study` "
            "/ `update_record` / `add_record` / "
            "`remove_record` first, then `mark_complete`. The review ends only "
            "when you call a terminating tool, so call `mark_complete` once "
            "you are satisfied."
        ),
    })

    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks
