"""The engine's prompt spines, and the `{include:NAME}` partials a bundle owns.

Each role's system message is built from two halves. The ENGINE SPINE comes
first: an ordered run of named sections, shipped as one markdown file each
under `meltiro/engine_prompts/` and assembled here in code. The config
bundle's own prompt file for that role is APPENDED after it. Neither half can
be lost by accident — the spine is chosen by the engine and the bundle's file
is required — so a model is briefed on the machinery whatever the bundle says
about the review.

Between the two halves the engine writes one transition sentence, so a model
reading straight through knows where its briefing on the machinery ends and the
review's own briefing begins (`join_role_message`; each builder holds its
role's wording).

The spines are declared in `ENGINE_SPINES`, one entry per role, each section
paired with the pipeline stage it depends on or `None` for a section that is
always composed:

  - `extractor_system`, `checker_system`, `review_system` -- the three system
    messages. The bundle's prompt file of the same name supplies the appended
    half.
  - `checker_user` -- the scaffold the per-field checker message is rendered
    from. The engine owns it whole; there is no bundle file to append.

Two stage predicates gate a conditional section, matching the two structure
toggles the engine resolves before any prompt is rendered:

  - `checker` -- true when `max_checks_per_field > 0`
  - `review`  -- true when `final_review` is on

A section whose stage is off is left out entirely, and the sections around it
close up: a run with no checker must not brief its extractor on challenges
that cannot arrive.

OVERRIDING A SECTION. A bundle ships `prompts/partials/meltiro/NAME.md` to
supply its own words for engine section `NAME`. Non-empty text REPLACES that
section where it sits in the spine. Text that is empty (or whitespace only)
REMOVES the section, and that is the only way a section is excluded from a
model's context. The name must be a section the engine ships; that directory
is ENUMERATED at load (`config_bundle._validate_engine_overrides`), so a file
whose name is not exactly a shipped section's is refused rather than sitting
there overriding nothing.

HASHING follows ownership. An override is hand-authored, so it belongs to the
config's identity and rides in the config fingerprints, empty overrides
included: excluding a section is a methodological choice and has to move them.
An un-overridden section is engine text and reaches no config preimage
(`config_prompt_preimage`); it moves `engine_fp` instead, through the source
digest that hashes `engine_prompts/*.md` beside the package's modules (see
`run_log.source_hash`). Which sections compose at all is decided by the engine
and the run's structure toggles, and those toggles ride in `structure_hash`.

THE BUNDLE'S OWN PARTIALS. A bundle prompt may cite a reusable block of its
own with `{include:NAME}`, replaced at render time by the content of
`prompts/partials/NAME.md` in the same bundle, so a block several prompts
share lives in one file and the copies cannot drift apart.
`{include_if:PREDICATE:NAME}` makes such a block conditional on a stage, on
the same terms as a spine section.

Rules:

  - Includes expand BEFORE `{reference:NAME}` substitution
    (`reference_lists.substitute_reference_placeholders`), so a partial may
    itself contain `{reference:...}` placeholders and they resolve exactly
    as they would inline. Conditional includes resolve in the same pass.
  - A partial may NOT itself contain an `{include:...}` or
    `{include_if:...}` placeholder, and neither may an override. Nesting is
    rejected loudly rather than resolved recursively; the shared blocks this
    exists for are flat, and one level keeps the expansion trivial to reason
    about.
  - A conditional's partial must EXIST whether or not its predicate is on,
    so a typo cannot hide behind a disabled stage and surface only when
    someone turns that stage back on.
  - An omitted conditional takes its whole line with it, plus one following
    blank line, so a disabled block leaves no ragged gap in the rendered
    prompt. A rendered prompt is hashed into `prompts_hash`, so whitespace
    is not cosmetic here.
  - The `meltiro:` namespace names engine sections and nothing else. A bundle
    prompt citing `{include:meltiro:NAME}` is refused: the engine composes its
    own sections, and a prompt that also cited one would compose it twice.

An included partial's content is inserted with surrounding whitespace
stripped, so the placeholder is expected to sit on its own line as a
block-level include. The `partials/` directory is optional: a bundle whose
prompts cite no include placeholder and overrides no section needs no
`partials/` directory at all.

The config bundle validates every placeholder at load time
(`config_bundle._prompt_partial_problems`) so a missing partial or a nested
include fails before any API spend. The render-time function here raises the
same loud error, so a broken include never ships silently in a prompt.
"""

import re
from pathlib import Path

from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import canonical_json


# The reserved namespace, and the directory its sections are read from. The
# directory is where a section's TEXT comes from; `ENGINE_SPINES` below is
# where its position comes from, and every shipped file appears in exactly one
# spine (tests/agentic_extraction/test_engine_prompts.py pins that).
ENGINE_NAMESPACE = "meltiro"
ENGINE_PROMPTS_DIR = Path(__file__).resolve().parent / "engine_prompts"

# The roles a spine is composed for. The first three name the bundle prompt
# file whose text is appended after the spine; `checker_user` has no bundle
# file, because the per-field scaffold is the engine's whole to write.
EXTRACTOR_SYSTEM = "extractor_system"
CHECKER_SYSTEM = "checker_system"
REVIEW_SYSTEM = "review_system"
CHECKER_USER = "checker_user"

# The predicates a conditional section or include may name. Deliberately
# closed: an unknown predicate is a typo or a stage that does not exist, and
# silently treating it as false would hide a block rather than report the
# mistake.
PREDICATE_NAMES = ("checker", "review")

# Each role's spine: the sections the engine composes, in order, each with the
# stage it depends on or None for one that is always composed. Order is the
# prompt's order, so a section moves by moving its line here.
ENGINE_SPINES = {
    EXTRACTOR_SYSTEM: (
        ("extractor_role", None),
        ("extractor_workflow", None),
        ("extractor_checker_feedback", "checker"),
        ("extractor_completion", None),
        ("extractor_review_handoff", "review"),
        ("extractor_tool_budget", None),
        ("recording_evidence", None),
        ("recording_notes", None),
        ("recording_conventions", None),
    ),
    CHECKER_SYSTEM: (
        ("checker_role", None),
        ("checker_briefing", None),
        ("checker_verdict", None),
    ),
    REVIEW_SYSTEM: (
        ("reviewer_role", None),
        ("reviewer_record", None),
        ("reviewer_workflow", None),
    ),
    CHECKER_USER: (
        ("checker_user", None),
    ),
}

# A file-stem-shaped token, mirroring the reference-placeholder grammar in
# reference_lists.py, optionally qualified by the engine namespace. The
# namespace is matched so a prompt citing one is REPORTED rather than left as
# a literal token in front of a model.
_NAME = r"[A-Za-z0-9_.\-]+"
_QUALIFIED_NAME = rf"(?:{ENGINE_NAMESPACE}:)?{_NAME}"

# Matches `{include:NAME}`.
_INCLUDE_PLACEHOLDER = re.compile(rf"\{{include:({_QUALIFIED_NAME})\}}")

# Matches `{include_if:PREDICATE:NAME}`.
_INCLUDE_IF_PLACEHOLDER = re.compile(
    rf"\{{include_if:({_NAME}):({_QUALIFIED_NAME})\}}")

# The same placeholder when it stands alone on its line, with the line's
# newline and one following blank line, so omitting it closes the gap.
_INCLUDE_IF_LINE = re.compile(
    rf"(?m)^[ \t]*\{{include_if:({_NAME}):({_QUALIFIED_NAME})\}}[ \t]*"
    r"\n(?:[ \t]*\n)?")

# Sentinel for the load-time validators, which must see EVERY branch: a
# reference or a banned placeholder hiding inside a disabled block is still a
# defect in the bundle, and a config that only validates on the toggles it
# happens to ship today would pass its own check and fail when a stage is
# switched on. Not for render paths -- see `substitute_include_placeholders`.
EXPAND_ALL_BRANCHES = "expand-all-branches"

# How two composed blocks are separated, whether they are two spine sections
# or the spine and the bundle's appended text. One blank line, applied in one
# place, so every rendered prompt has the same shape and a hash taken over one
# cannot disagree with the message sent from another.
BLOCK_SEPARATOR = "\n\n"


def stage_predicates(max_checks_per_field, final_review):
    """The predicate map for a run's structure toggles.

    One place decides what `checker` and `review` mean, so a render path
    cannot disagree with the hash path about whether a stage counts as on.
    """
    return {
        "checker": int(max_checks_per_field or 0) > 0,
        "review": bool(final_review),
    }


def engine_section_names():
    """Every engine section the package ships, sorted.

    Read off `engine_prompts/` every call, so the shipped files are the whole
    answer and no second list can fall out of step with them. What a section
    is FOR — which spine it sits in, at what position, behind which stage —
    is `ENGINE_SPINES`.
    """
    return tuple(sorted(p.stem for p in ENGINE_PROMPTS_DIR.glob("*.md")))


def spine_sections(role):
    """The `(name, predicate)` pairs composing `role`'s spine, in order."""
    try:
        return ENGINE_SPINES[role]
    except KeyError:
        raise ValueError(
            f"unknown prompt role {role!r}; the engine composes a spine for "
            f"{list(ENGINE_SPINES)}. This is an engine bug: a role is added by "
            f"adding its spine.") from None


def is_engine_name(name):
    """True for a name qualified with the reserved `meltiro:` namespace."""
    return str(name).startswith(f"{ENGINE_NAMESPACE}:")


def _unqualified(name):
    return str(name).split(":", 1)[1] if is_engine_name(name) else str(name)


def _all_cited_names(text):
    names = {m.group(1) for m in _INCLUDE_PLACEHOLDER.finditer(text)}
    names |= {m.group(2) for m in _INCLUDE_IF_PLACEHOLDER.finditer(text)}
    return names


def included_names(text):
    """Return every BUNDLE partial name cited by `text`, conditional ones
    included.

    Used at config-load time to check every prompt's includes against the
    `partials/` directory. Conditional names are returned whatever their
    predicate would evaluate to, so a partial cited only by a disabled
    branch is still required to exist. A `meltiro:`-qualified citation is
    excluded here and returned by `engine_cited_names` for its own refusal.
    """
    return {n for n in _all_cited_names(text) if not is_engine_name(n)}


def engine_cited_names(text):
    """Return every `meltiro:`-qualified name cited by `text`, unqualified.

    Every one of them is a defect: the engine composes its own sections, so a
    prompt has none to cite. Collected rather than raised on the first, so the
    load error names them all.
    """
    return {_unqualified(n) for n in _all_cited_names(text)
            if is_engine_name(n)}


def engine_citation_message(name, where=None):
    """The message a `{include:meltiro:NAME}` citation is refused with.

    One wording for the load-time check and the render-time one, so a bundle
    is told the same thing wherever the citation is found.
    """
    prefix = f"prompt {where} cites" if where else "prompt cites"
    return (
        f"{prefix} {{include:{ENGINE_NAMESPACE}:{name}}}. The engine composes "
        f"its own sections into each role's prompt, and this file supplies the "
        f"text appended after them, so delete the placeholder. To supply your "
        f"own wording for that section instead, ship "
        f"prompts/partials/{ENGINE_NAMESPACE}/{name}.md; an empty file leaves "
        f"the section out altogether."
    )


def _reject_nesting(name, content, where):
    if _INCLUDE_PLACEHOLDER.search(content) or \
            _INCLUDE_IF_PLACEHOLDER.search(content):
        raise ConfigBundleError(
            [f"{where} '{name}.md' contains a nested include placeholder; "
             f"nesting is not supported. Inline the nested content or "
             f"flatten the partials."]
        )


def engine_override_dir(partials_dir):
    """Where a bundle puts its overrides of the engine's sections."""
    return Path(partials_dir) / ENGINE_NAMESPACE


def engine_override_path(name, partials_dir):
    """Where a bundle puts its own text for engine section `name`."""
    return engine_override_dir(partials_dir) / f"{name}.md"


def engine_override_entries(partials_dir):
    """Every entry name shipped in `prompts/partials/meltiro/`, sorted.

    A LISTING, not a probe: a case-insensitive filesystem answers
    `Path("recording_notes.md").is_file()` for a file named
    `Recording_Notes.md` and a case-sensitive one does not, so a check built on
    probes reaches different verdicts on macOS and Linux for the same bundle.
    Each listed name is the name the filesystem actually holds, so the
    load-time enumeration in `config_bundle` compares like with like
    everywhere. Every entry is returned, directories and non-markdown included:
    an override resolves under exactly one name, and everything else in that
    directory resolves under none.
    """
    override_dir = engine_override_dir(partials_dir)
    if not override_dir.is_dir():
        return ()
    return tuple(sorted(p.name for p in override_dir.iterdir()))


def read_engine_section(name, partials_dir):
    """Return `(text, overridden)` for an engine section.

    The bundle's `prompts/partials/meltiro/NAME.md` wins when it exists, and
    the flag says which happened so a hashing caller can tell hand-authored
    text from engine text. An override that is empty or whitespace-only
    returns the empty string, which is how a section is left out of a model's
    context.

    Whichever file supplies the text is held to the no-nesting rule. The
    engine's own copy is checked by the same call as an override: a shipped
    section carrying `{include:...}` would put the literal directive in front
    of a model, which is the failure the rule exists to prevent whoever wrote
    the text.
    """
    known = engine_section_names()
    if name not in known:
        raise ConfigBundleError(
            [f"unknown engine section '{name}'; the engine's sections are "
             f"{list(known)}."]
        )
    override = engine_override_path(name, partials_dir)
    overridden = override.is_file()
    source = override if overridden else ENGINE_PROMPTS_DIR / f"{name}.md"
    content = source.read_text(encoding="utf-8")
    _reject_nesting(
        name, content,
        "engine-section override" if overridden else "engine section")
    return content.strip(), overridden


def compose_engine_spine(role, partials_dir, *, predicates):
    """Render `role`'s engine spine: its sections joined by a blank line.

    Overrides are resolved per section, and a section whose stage is off or
    whose override is empty contributes nothing at all — no heading, no gap.
    A section is dropped for one of two reasons, and they are different
    reasons: its stage is off for this run, or the bundle overrode it with an
    empty file. Only the second is the config author's choice, so only the
    second reaches a config fingerprint (see `config_prompt_preimage`).
    Slots (`{image_labels_list}`, `{max_checks_per_field}`) are left for the
    caller to fill, so the spine and the bundle's appended text are filled by
    the same substitution.
    """
    parts = []
    for name, predicate in spine_sections(role):
        if predicate is not None and not _predicate_value(
                predicate, name, predicates):
            continue
        text, _ = read_engine_section(name, partials_dir)
        if text:
            parts.append(text)
    return BLOCK_SEPARATOR.join(parts)


def engine_override_pairs(role, partials_dir, *, predicates):
    """The bundle's overrides of `role`'s spine, as sorted `[name, text]`
    pairs.

    The config-owned half of an engine spine, and the whole of what a spine
    contributes to a config fingerprint. Empty overrides are included and
    carry their empty string: leaving a section out is a decision about what
    the model is asked, so it has to move the fingerprints exactly as
    rewriting the section would.

    A section whose stage is off is skipped whether or not the bundle
    overrides it: it reaches no model this run, and the toggle that silenced
    it already rides in `structure_hash`.
    """
    pairs = []
    for name, predicate in spine_sections(role):
        if predicate is not None and not _predicate_value(
                predicate, name, predicates):
            continue
        override = engine_override_path(name, partials_dir)
        if override.is_file():
            pairs.append([name, override.read_text(encoding="utf-8").strip()])
    return sorted(pairs)


def all_engine_override_pairs(partials_dir, *, predicates):
    """Every role's override pairs, merged and sorted.

    For the bundle-wide `prompts_hash`, which covers the whole prompt surface
    rather than one role's. A section belongs to exactly one spine, so the
    merge cannot produce two entries for one name.
    """
    merged = {}
    for role in ENGINE_SPINES:
        for name, text in engine_override_pairs(
                role, partials_dir, predicates=predicates):
            merged[name] = text
    return sorted([name, text] for name, text in merged.items())


def config_prompt_preimage(prompt_text, override_pairs):
    """The config-owned identity of one role's prompt, as a canonical string.

    Two components, and only these two: the bundle's appended text as it
    renders, and the bundle's overrides of the sections this run composed. An
    un-overridden section contributes nothing, which is what keeps every
    bundle's config fingerprints steady across an engine release that rewords
    one.

    Hashed by `prompt_builder.compute_prompt_config_hash` and folded verbatim
    into `checker_fp` and `review_fp`, so all three stages state ownership the
    same way.
    """
    return canonical_json({
        "prompt": prompt_text,
        "engine_overrides": override_pairs,
    })


def join_blocks(*blocks):
    """Join rendered blocks with one blank line, dropping the empty ones.

    An empty spine (every section overridden away) or an empty bundle prompt
    file leaves no leading or trailing gap behind it: a rendered prompt is
    hashed, so whitespace is content.
    """
    return BLOCK_SEPARATOR.join(b for b in (b.strip() for b in blocks) if b)


def join_role_message(spine, transition, bundle_text):
    """Join one role's system message: spine, transition sentence, bundle text.

    The transition is the engine's signpost from its own half of the message to
    the review's, and each role has its own (the builders hold the wording,
    beside the rest of their framing). It is emitted only when there is text on
    BOTH sides of it: a bundle whose prompt file is empty is promised no
    briefing that never arrives, and a bundle that overrode every section away
    reads its own opening line first rather than a lone engine sentence.

    Compose-time framing, so it is engine text like the user-block headers: it
    reaches the wire and `engine_fp`, and no config preimage is built through
    here (see `config_prompt_preimage`, which takes the bundle's text on its
    own).
    """
    blocks = [spine, bundle_text]
    if spine.strip() and bundle_text.strip():
        blocks.insert(1, transition)
    return join_blocks(*blocks)


def _read_partial(name, partials_dir, placeholder):
    """The text a bundle placeholder expands to."""
    if is_engine_name(name):
        raise ConfigBundleError(
            [engine_citation_message(_unqualified(name))])

    partial_path = Path(partials_dir) / f"{name}.md"
    if not partial_path.is_file():
        raise ConfigBundleError(
            [f"prompt cites unknown partial '{name}' via "
             f"{placeholder}; expected a file at "
             f"{partial_path}. Add the partial or fix the placeholder."]
        )
    content = partial_path.read_text(encoding="utf-8")
    _reject_nesting(name, content, "partial")
    return content.strip()


def _predicate_value(predicate, name, predicates):
    if predicate not in PREDICATE_NAMES:
        raise ConfigBundleError(
            [f"prompt cites unknown include predicate '{predicate}' via "
             f"{{include_if:{predicate}:{name}}}; known predicates: "
             f"{list(PREDICATE_NAMES)}."]
        )
    if predicates is EXPAND_ALL_BRANCHES:
        return True
    if predicates is None:
        raise ConfigBundleError(
            [f"prompt cites {{include_if:{predicate}:{name}}} but the "
             f"renderer was given no structure toggles to evaluate it "
             f"against. This is an engine bug, not a config error: a render "
             f"path must pass `predicates`, because choosing a branch "
             f"silently is the failure conditional includes exist to "
             f"prevent."]
        )
    return bool(predicates[predicate])


def substitute_include_placeholders(text, partials_dir, *, predicates=None):
    """Replace every include placeholder in `text` with a bundle partial.

    `{include:NAME}` is substituted with the stripped content of
    `partials_dir / "NAME.md"`. `{include_if:PREDICATE:NAME}` is substituted
    with the same content when `predicates[PREDICATE]` is true, and removed
    when it is false -- taking its line and one following blank line with it
    when it stands alone, so the omission leaves no ragged gap.

    A placeholder naming a partial that does not exist raises
    `ConfigBundleError`, whether or not its branch is taken, so a typo cannot
    hide behind a disabled stage. So does a partial that itself contains an
    include placeholder: nesting is not supported. So does a
    `meltiro:`-qualified name: the engine composes its own sections, and this
    text is the half appended after them.

    `predicates` is a mapping from `PREDICATE_NAMES` to bool, normally built
    by `stage_predicates`. Pass `EXPAND_ALL_BRANCHES` from a validator that
    must inspect every branch. Omitting it is legal only for text citing no
    conditional include; text that does cite one raises, rather than quietly
    picking a branch.

    `partials_dir` is only read when `text` actually cites a partial, so a
    bundle with no include placeholders never needs the directory to exist.
    """
    partials_dir = Path(partials_dir)

    def _repl_conditional_line(match):
        predicate, name = match.group(1), match.group(2)
        content = _read_partial(
            name, partials_dir, f"{{include_if:{predicate}:{name}}}")
        if _predicate_value(predicate, name, predicates):
            return f"{content}\n\n"
        return ""

    def _repl_conditional_inline(match):
        predicate, name = match.group(1), match.group(2)
        content = _read_partial(
            name, partials_dir, f"{{include_if:{predicate}:{name}}}")
        if _predicate_value(predicate, name, predicates):
            return content
        return ""

    def _repl_include(match):
        name = match.group(1)
        return _read_partial(name, partials_dir, f"{{include:{name}}}")

    # Line-standing conditionals first, so an omitted block closes its own
    # gap; whatever is left is inline and substitutes in place.
    text = _INCLUDE_IF_LINE.sub(_repl_conditional_line, text)
    text = _INCLUDE_IF_PLACEHOLDER.sub(_repl_conditional_inline, text)
    return _INCLUDE_PLACEHOLDER.sub(_repl_include, text)
