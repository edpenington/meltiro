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
role learns what an `<img>` citation may name, with the exhibit's printed
footnote and its transcription where the bundle carries them — and, where the
study supplies none, the statement that none accompany it, so exhibit presence
is a fact of the message either way. Both get
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

from meltiro import framing
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

def _extractor_header(study_id, supplements=None):
    """The extractor's opening framing line.

    One definition, used wherever the extractor's opening message is built,
    so the transcript record cannot drift from the message.
    """
    return (
        f"Extract study {study_id}. The full text follows, and every cropped "
        "table and figure this study supplies is attached below."
        + (" Supplementary material follows the paper, each supplement in a "
           "section of its own carrying its own prose and exhibits."
           if supplements else "")
        + " Where something is not attached, the message says so in its "
          "place."
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


# The text block a role reads in place of the attachments when the message
# carries none at all. Presence of an exhibit is a fact about the MESSAGE, not
# about the machinery, so the message is where it is stated: a role reads what
# accompanies this study rather than inferring it from a message that simply
# stops after the paper text. Every role can read a crop — a run whose model
# cannot is refused at startup — so this now says one thing only: the STUDY
# supplies none.
NO_EXHIBITS_NOTICE = "(no cropped figures or tables accompany this study)"

# And where the article prints none but a supplement does. The message
# attaches crops, so `NO_EXHIBITS_NOTICE` would be false; the article having
# no exhibits of its own is still a fact a role would otherwise infer from a
# gap, and a review whose initial check asks whether the figures were included
# is asking about exactly this. A paper that ships its data tables as
# supplementary material is the ordinary case here, not an edge.
NO_ARTICLE_EXHIBITS_NOTICE = (
    "(the article prints no cropped figures or tables of its own; the "
    "supplementary material's follow in their own sections)")


# What introduces an exhibit's printed footnote where the manifest records
# one. A word rather than a tag, because the line sits in a message a model
# reads rather than in a structure it parses, and it names what the words are:
# the paper's own footnote to that exhibit, not an instruction from the
# engine.
EXHIBIT_FOOTNOTE_PREFIX = "Footnote:"

# What introduces an exhibit's transcription where the bundle carries one. A
# word rather than a tag, on the footnote's terms and for its reason: the line
# sits in a message a model reads rather than in a structure it parses, and it
# names what follows — the exhibit's own content as text, not an instruction
# and not a second exhibit.
EXHIBIT_TRANSCRIPTION_PREFIX = "Content as text:"


# What opens and closes a supplement's section of the message. A supplement is
# a separate document from the article and often not reviewed to its standard,
# so which document a value was read from is part of the claim; the message is
# where that is kept, because an `<img>` label carries no such mark. The
# section is delimited on both sides for the reason the paper text is: a role
# has to be able to tell where the article stops without inferring it from a
# change of subject.
def supplement_open(name, title):
    """The line that opens one supplement's section."""
    return f"--- SUPPLEMENT {name}: {title} ---"


def supplement_close(name):
    """The line that closes it, naming the same supplement."""
    return f"--- END SUPPLEMENT {name} ---"


# What a supplement's own prose is wrapped in, inside its section. Distinct
# from the article's `PAPER TEXT` markers by name, so a role reading a quote
# back can tell which of the two it came out of — and so that "the paper
# text" means one thing in this message.
#
# Every delimiter this module writes is defined in `meltiro.framing`, which
# the loader imports to refuse a bundle whose own text spells one. Two places
# holding the vocabulary would be two answers to what a boundary is.
SUPPLEMENT_TEXT_OPEN = framing.SUPPLEMENT_TEXT_OPEN
SUPPLEMENT_TEXT_CLOSE = framing.SUPPLEMENT_TEXT_CLOSE

# What stands in a supplement's section where its prose would be. A
# supplement that is a run of data tables prints none, and the format lets it
# say so by carrying no text.md; stating it keeps the section's shape the same
# either way, on the terms `NO_EXHIBITS_NOTICE` is stated on.
#
# Two of them, because "its exhibits follow" has to be true. A supplement with
# no prose AND no exhibits is a valid bundle — the declaration may assert an
# empty exhibit list — and its section would otherwise promise attachments and
# then close.
NO_SUPPLEMENT_TEXT_NOTICE = (
    "(this supplement prints no prose; its exhibits follow)")
NO_SUPPLEMENT_CONTENT_NOTICE = (
    "(this supplement prints no prose and supplies no exhibits)")
# And the fourth state: prose, and no exhibits. A supplement may be a protocol
# or a statistical appendix and print nothing to crop, so the section has to
# say that rather than end after the prose — the message states what
# accompanies a document on both sides, or a role reads a gap and guesses.
NO_SUPPLEMENT_EXHIBITS_NOTICE = (
    "(this supplement supplies no cropped figures or tables)")


def supplement_blocks(supplement, captions=None, notes=None, tables=None):
    """The content blocks for one supplement's section of a message.

    `supplement` is a mapping carrying `name`, `title`, `text` (or None) and
    `figures`, a list of `(label, png_bytes)` in the order they attach. Its
    exhibits are introduced by exactly the helper the article's are, and
    looked up in the same flat maps, because a label means one exhibit
    across the whole bundle: what the section keeps apart is the DOCUMENT,
    not the citation.
    """
    blocks = [{"type": "text",
               "text": supplement_open(supplement["name"],
                                       supplement["title"])}]
    text = supplement.get("text")
    if text:
        blocks.append({
            "type": "text",
            "text": (SUPPLEMENT_TEXT_OPEN + "\n" + text + "\n"
                     + SUPPLEMENT_TEXT_CLOSE),
        })
    elif supplement.get("figures"):
        blocks.append({"type": "text", "text": NO_SUPPLEMENT_TEXT_NOTICE})
    else:
        blocks.append({"type": "text",
                       "text": NO_SUPPLEMENT_CONTENT_NOTICE})
    if text and not supplement.get("figures"):
        blocks.append({"type": "text",
                       "text": NO_SUPPLEMENT_EXHIBITS_NOTICE})

    for label, png_bytes in supplement.get("figures") or []:
        blocks.append({"type": "text",
                       "text": image_label_text(label, captions, notes,
                                                tables)})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            },
        })

    blocks.append({"type": "text",
                   "text": supplement_close(supplement["name"])})
    return blocks


def image_label_text(label, captions=None, notes=None, tables=None):
    """The text block that introduces one attached exhibit.

    The label first and alone in brackets, because it is what an
    `<img>label</img>` citation must contain; the paper's own caption after
    it, so a model reading `[table_01]` can tell which crop it is looking at
    without guessing; on its own line the footnote the paper prints under
    the exhibit, where the manifest records one; and after that the exhibit's
    transcription, where the bundle carries one. The crop carries that
    footnote as pixels and `text.md` does not carry it at all, so supplying it
    as text is what lets a model read a table's small print without resolving
    it off the image — and it is why the engine prompts say a fact taken from
    it is cited as `<img>label</img>` rather than quoted.

    The transcription is the same bargain one step further: `text.md` carries
    no table content either, only a sentinel where the exhibit sits, so
    without it every cell of every table has to be read off pixels. It is
    emitted verbatim, the markup included. A pipe table cannot express a
    header that spans columns or a stub that spans rows, which is what the
    format chose HTML to keep, so flattening it here would drop the structure
    at the one point where a reader is deciding which column a number sits
    under.

    It arrives after the crop's other text and before the image itself, so a
    role reads the exhibit's own words before looking at it. What a citation
    MEANS is untouched: the crop remains what the exhibit is, and a fact
    taken from either is `<img>label</img>`.

    `captions`, `notes` and `tables` are label -> text maps
    (`PaperBundle.exhibits`, `PaperBundle.exhibit_notes`, and the markup read
    off `PaperBundle.tables`), looked up on the same normalised key the
    dispatcher matches a citation on. A label no map has an entry for renders
    as the bare label, which is what a caller with no maps at all gets.

    One definition, used by the message builders and by the text-only render
    the session captures, so the recorded prompt cannot drift from the
    message.
    """
    key = str(label).strip().lower()
    caption = (captions or {}).get(key)
    line = f"[{label}] {caption}" if caption else f"[{label}]"
    footnote = (notes or {}).get(key)
    if footnote:
        line = f"{line}\n{EXHIBIT_FOOTNOTE_PREFIX} {footnote}"
    transcription = (tables or {}).get(key)
    if not transcription:
        return line
    return f"{line}\n{EXHIBIT_TRANSCRIPTION_PREFIX}\n{transcription}"


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


def message_figure_labels(figures, supplements=None):
    """What a message attaches, in the order it attaches them: `(label,
    document)` pairs, `document` being None for the article's crops and the
    supplement's name for a supplement's.

    The article's figure sequence followed by each supplement's, which is the
    order `build_initial_user_blocks` and `build_review_user_blocks` emit
    their image blocks in. ONE definition, because two consumers depend on it
    and a second answer would be a second story about one message: a text view
    pairs labels to image blocks by position, and the dry run's exhibit
    manifest names each crop by the document it came out of. Neither derives
    the sequence itself.
    """
    attached = [(label, None) for label, _ in figures]
    for supplement in supplements or []:
        attached += [(label, supplement["name"])
                     for label, _ in supplement.get("figures") or []]
    return attached


def render_message_text(blocks, image_labels):
    """The text-only view of a message, PROJECTED from that message.

    Every text block's text verbatim, in the message's own order, with
    `(image: LABEL.png)` written where each attachment's bytes are. Nothing
    is re-rendered: the strings here are the strings in `blocks`, so a view
    of a message cannot say one thing while the message says another. That is
    the whole reason this takes blocks rather than the material they were
    built from.

    `image_labels` is `message_figure_labels`' sequence for those blocks, and
    is consumed one entry per image block. An `image` block carries the crop's
    bytes and no label, so the label has to come from the sequence that built
    it; a caller passing a re-sorted, re-cased or short sequence is naming
    attachments the message never carried, and the checks below refuse it
    rather than writing a wrong label. A bare label is accepted too, for a
    caller that has only the names.
    """
    labels = iter([entry[0] if isinstance(entry, tuple) else entry
                   for entry in image_labels])
    parts = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append(block["text"])
        elif kind == "image":
            label = next(labels, None)
            if label is None:
                raise ValueError(
                    "the message attaches more images than the figure "
                    "sequence has labels, so a text view of it would name "
                    "the wrong crop")
            parts.append(framing.image_placeholder(label))
        else:
            raise ValueError(f"unrenderable content block: {kind!r}")
    left = list(labels)
    if left:
        raise ValueError(
            f"the figure sequence carries labels the message does not "
            f"attach: {left}")
    return "\n\n".join(parts)


def _no_figures_blocks(figures, supplements):
    """What stands where the article's attachments would be, if anything.

    Three cases, and the message states which one it is rather than leaving a
    role to read a gap: the study supplies no crops at all, the article
    supplies none while a supplement does, or the article supplies some and
    nothing needs saying.
    """
    if figures:
        return []
    if any(supplement.get("figures") for supplement in supplements or []):
        return [{"type": "text", "text": NO_ARTICLE_EXHIBITS_NOTICE}]
    return [{"type": "text", "text": NO_EXHIBITS_NOTICE}]


def build_initial_user_blocks(study_id, paper_text, figures,
                              image_captions=None, image_notes=None,
                              image_tables=None, supplements=None):
    """Build the initial user message content blocks.

    Args:
        study_id: study identifier (used in the header).
        paper_text: the paper's full text (from PaperBundle.text).
        figures: list of (label, png_bytes) tuples.
        image_captions: label -> caption map (`PaperBundle.exhibits`), so each
            attachment arrives under the caption the paper prints beside it.
        image_notes: label -> footnote map (`PaperBundle.exhibit_notes`), so
            an exhibit's printed footnote arrives as text beside the crop
            that prints it.
        image_tables: label -> transcription markup, for the exhibits whose
            content the bundle carries as text. Emitted verbatim beside the
            crop; a label absent from it means the crop is the content.
        supplements: the supplementary material, in the order it is carried;
            each a mapping of `name`, `title`, `text` (or None) and
            `figures`. Each gets a delimited section of its own after the
            article, because a value read from a supplement is a claim about
            that document rather than about the paper.

    Returns a list of content blocks. The LAST block carries
    cache_control: ephemeral so the whole user message caches.

    An empty `figures` gets `NO_EXHIBITS_NOTICE` in their place, so a role
    reads what accompanies this study either way rather than inferring the
    absence from a message that simply stops.
    """
    blocks = []
    blocks.append({"type": "text",
                   "text": _extractor_header(study_id, supplements)})

    blocks.append({
        "type": "text",
        "text": (framing.PAPER_TEXT_OPEN + "\n" + paper_text
                 + "\n" + framing.PAPER_TEXT_CLOSE),
    })

    blocks.extend(_no_figures_blocks(figures, supplements))

    for label, png_bytes in figures:
        # Label first so the model can refer to it by name in `source`, and
        # the caption beside it so it knows which exhibit it is looking at.
        blocks.append({"type": "text",
                       "text": image_label_text(label, image_captions,
                                                image_notes, image_tables)})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            },
        })

    # Every supplement, each in its own delimited section after the article.
    # They sit inside the cached prefix, so a bundle carrying supplementary
    # material pays for it once and reads it at the cache rate on every turn
    # after the first.
    for supplement in supplements or []:
        blocks.extend(supplement_blocks(
            supplement, image_captions, image_notes, image_tables))

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
                             extraction_record_dict, image_captions=None,
                             image_notes=None, image_tables=None,
                             supplements=None):
    """Build the user content blocks for the final-review pass.

    The reviewer sees the paper text + all cropped images, each under its
    label, the paper's caption for it, its printed footnote where the
    manifest records one and its transcription where the bundle carries one,
    + the assembled extraction output as
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
        f"complete."
        + (" Supplementary material follows the paper, each supplement in a "
           "section of its own carrying its own prose and exhibits."
           if supplements else "")
        + f" You have as many turns as you need: the view tools let "
        f"you inspect the extraction output and return their results to you, "
        f"and the editing tools let you revise it. Call `mark_complete` when "
        f"satisfied. Do NOT make stylistic changes; only revise "
        f"where the existing extraction is wrong."
    )
    blocks.append({"type": "text", "text": header})

    blocks.append({
        "type": "text",
        "text": (framing.PAPER_TEXT_OPEN + "\n" + paper_text
                 + "\n" + framing.PAPER_TEXT_CLOSE),
    })

    blocks.extend(_no_figures_blocks(figures, supplements))

    for label, png_bytes in figures:
        blocks.append({"type": "text",
                       "text": image_label_text(label, image_captions,
                                                image_notes, image_tables)})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            },
        })

    # Before the extraction output, so the reviewer has read every document
    # the extractor was given by the time it is shown what to review.
    for supplement in supplements or []:
        blocks.extend(supplement_blocks(
            supplement, image_captions, image_notes, image_tables))

    extraction_record_text = json.dumps(
        extraction_record_dict, indent=2, ensure_ascii=False)
    blocks.append({
        "type": "text",
        "text": (
            framing.REVIEW_OUTPUT_OPEN + "\n"
            + extraction_record_text
            + "\n" + framing.REVIEW_OUTPUT_CLOSE + "\n\n"
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
