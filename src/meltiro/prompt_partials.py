"""The engine's own prompts, and the `{include:NAME}` partials a bundle owns.

Each role's system message is built from two halves. The ENGINE PROMPT comes
first: one markdown file per role under `meltiro/engine_prompts/`, rendered
here. The config bundle's own prompt file for that role is APPENDED after it.
Neither half can be lost by accident — the engine chooses its own prompt and
the bundle's file is required — so a model is briefed on the machinery
whatever the bundle says about the review.

Between the two halves the engine writes one transition sentence, so a model
reading straight through knows where its briefing on the machinery ends and the
review's own briefing begins (`join_role_message`; each builder holds its
role's wording).

`ENGINE_ROLE_PROMPTS` names the file each role renders:

  - `extractor_system`, `checker_system`, `review_system` -- the three system
    messages, from `extractor.md`, `checker.md` and `reviewer.md`. The
    bundle's prompt file of the same role supplies the appended half.
  - `checker_user` -- the scaffold the per-field checker message is rendered
    from, `checker_user.md`. The engine owns it whole; there is no bundle
    file to append.

A role prompt may compose an engine PARTIAL by citing
`{include:meltiro:NAME}` or `{include_if:PREDICATE:meltiro:NAME}`, which is
how a passage that belongs to one stage only reaches a model exactly when
that stage runs. `extractor.md` cites `extractor_checker_feedback` that way.
Citation is one level deep and engine-only: a cited partial may cite nothing
further, and a role prompt may cite no bundle partial, which would invert the
ownership boundary the two halves are built on.

Two stage predicates gate a conditional citation, matching the two structure
toggles the engine resolves before any prompt is rendered:

  - `checker` -- true when `max_checks_per_field > 0`
  - `review`  -- true when `final_review` is on

A partial whose stage is off is left out entirely, and the text around it
closes up: a run with no checker must not brief its extractor on challenges
that cannot arrive.

OVERRIDING. A bundle ships `prompts/partials/meltiro/NAME.md` to supply its
own words for engine prompt `NAME`. Non-empty text REPLACES that file's text,
rendered literally — an override carries no citations of its own, so
overriding a ROLE prompt replaces the whole of that role's engine half,
conditional partial included. Text that is empty (or whitespace only) REMOVES
it, and that is the only way an engine prompt is excluded from a model's
context. The name must be one the engine ships; that directory is ENUMERATED
at load (`config_bundle._validate_engine_overrides`), so a file whose name is
not exactly a shipped one is refused rather than sitting there overriding
nothing.

HASHING follows ownership. An override is hand-authored, so it belongs to the
config's identity and rides in the config fingerprints, empty overrides
included: excluding an engine prompt is a methodological choice and has to
move them. It rides in them when its text reached a model this run, and an
override that reached none moves nothing: a partial whose stage is off, or one
whose role prompt the bundle overrode whole, is a file this run never composed.
An un-overridden engine prompt reaches no config preimage
(`config_prompt_preimage`); it moves `engine_fp` instead, through the source
digest that hashes `engine_prompts/*.md` beside the package's modules (see
`run_log.source_hash`). Which files compose at all is decided by the engine
and the run's structure toggles, and those toggles ride in `structure_hash`.

THE BUNDLE'S OWN PARTIALS. A bundle prompt may cite a reusable block of its
own with `{include:NAME}`, replaced at render time by the content of
`prompts/partials/NAME.md` in the same bundle, so a block several prompts
share lives in one file and the copies cannot drift apart.
`{include_if:PREDICATE:NAME}` makes such a block conditional on a stage, on
the same terms as an engine partial.

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
  - The `meltiro:` namespace names the engine's own prompts and nothing else.
    A bundle prompt citing `{include:meltiro:NAME}` is refused: the engine
    composes its own, and a prompt that also cited one would compose it twice.

An included partial's content is inserted with surrounding whitespace
stripped, so the placeholder is expected to sit on its own line as a
block-level include. The `partials/` directory is optional: a bundle whose
prompts cite no include placeholder and overrides nothing needs no
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


# The reserved namespace, and the directory the engine's prompts are read
# from. Every shipped file is either a role's prompt or a partial exactly one
# role prompt cites (tests/agentic_extraction/test_engine_prompts.py pins
# that), so nothing in there can ship unread.
ENGINE_NAMESPACE = "meltiro"
ENGINE_PROMPTS_DIR = Path(__file__).resolve().parent / "engine_prompts"

# The roles an engine prompt is rendered for. The first three name the bundle
# prompt file whose text is appended after it; `checker_user` has no bundle
# file, because the per-field scaffold is the engine's whole to write.
EXTRACTOR_SYSTEM = "extractor_system"
CHECKER_SYSTEM = "checker_system"
REVIEW_SYSTEM = "review_system"
CHECKER_USER = "checker_user"

# The predicates a conditional citation or include may name. Deliberately
# closed: an unknown predicate is a typo or a stage that does not exist, and
# silently treating it as false would hide a block rather than report the
# mistake.
PREDICATE_NAMES = ("checker", "review")

# The one file each role's engine half is rendered from. Its own text decides
# what else composes and in what order, so a passage moves by moving it in
# that file rather than by editing a table here.
ENGINE_ROLE_PROMPTS = {
    EXTRACTOR_SYSTEM: "extractor",
    CHECKER_SYSTEM: "checker",
    REVIEW_SYSTEM: "reviewer",
    CHECKER_USER: "checker_user",
}

ROLE_PROMPT_NAMES = frozenset(ENGINE_ROLE_PROMPTS.values())

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

# How two composed blocks are separated, whether they are an engine partial
# and the text around it or the engine's half and the bundle's appended one.
# One blank line, applied in one place, so every rendered prompt has the same
# shape and a hash taken over one cannot disagree with the message sent from
# another.
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


def engine_prompt_names():
    """Every prompt the engine ships, sorted.

    Read off `engine_prompts/` every call, so the shipped files are the whole
    answer and no second list can fall out of step with them. Which of them a
    role renders is `ENGINE_ROLE_PROMPTS` and the citations inside the file it
    names.
    """
    return tuple(sorted(p.stem for p in ENGINE_PROMPTS_DIR.glob("*.md")))


def role_prompt_name(role):
    """The engine prompt `role` renders."""
    try:
        return ENGINE_ROLE_PROMPTS[role]
    except KeyError:
        raise ValueError(
            f"unknown prompt role {role!r}; the engine renders a prompt for "
            f"{list(ENGINE_ROLE_PROMPTS)}. This is an engine bug: a role is "
            f"added by giving it a file.") from None


def _shipped_text(name):
    """The engine's own copy of a prompt, whatever a bundle says about it."""
    return (ENGINE_PROMPTS_DIR / f"{name}.md").read_text(
        encoding="utf-8").strip()


def engine_citations(text):
    """The `(predicate, name)` pairs `text` cites, in the order it cites them.

    `predicate` is `None` for an unconditional `{include:...}`. Names are
    returned as written, namespace and all, so a caller can tell an engine
    citation from a bundle one it must refuse. Found with the same two
    patterns the substitution below expands, so what a caller enumerates and
    what a render replaces cannot disagree.
    """
    found = [(m.start(), None, m.group(1))
             for m in _INCLUDE_PLACEHOLDER.finditer(text)]
    found += [(m.start(), m.group(1), m.group(2))
              for m in _INCLUDE_IF_PLACEHOLDER.finditer(text)]
    return [(predicate, name)
            for _, predicate, name in sorted(found, key=lambda c: c[0])]


def composed_engine_names(role, *, predicates):
    """Every engine prompt name `role` composes, in the order it reads them.

    The role's own file, then each partial it cites whose stage is on. Read
    off the SHIPPED files, so this is the engine's own answer about the shape
    of a role's briefing — what a caller asking what the engine composes for a
    role needs, whatever any bundle then says about the text
    (`config_bundle._validate_checker_placeholders` picks each override's slot
    allowlist by it, and the engine's own tests read it to show no shipped file
    is unread).

    Which of these names a bundle's override actually reaches is the narrower
    question, and `engine_override_pairs` is where it is asked: a bundle that
    overrides the ROLE prompt supplies that whole half itself, and the partial
    cited by the file it replaced composes nowhere.
    """
    name = role_prompt_name(role)
    names = [name]
    for predicate, cited in engine_citations(_shipped_text(name)):
        if predicate is None or _predicate_value(predicate, cited, predicates):
            names.append(_unqualified(cited))
    return names


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

    Every one of them is a defect in a BUNDLE prompt: the engine composes its
    own, so a bundle prompt has none to cite. Collected rather than raised on
    the first, so the load error names them all.
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
        f"its own prompts for each role, and this file supplies the text "
        f"appended after them, so delete the placeholder. To supply your own "
        f"wording instead, ship "
        f"prompts/partials/{ENGINE_NAMESPACE}/{name}.md; "
        f"an empty file leaves that text out altogether."
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
    """Where a bundle puts its overrides of the engine's prompts."""
    return Path(partials_dir) / ENGINE_NAMESPACE


def engine_override_path(name, partials_dir):
    """Where a bundle puts its own text for engine prompt `name`."""
    return engine_override_dir(partials_dir) / f"{name}.md"


def engine_override_entries(partials_dir):
    """Every entry name shipped in `prompts/partials/meltiro/`, sorted.

    A LISTING, not a probe: a case-insensitive filesystem answers
    `Path("extractor.md").is_file()` for a file named `Extractor.md` and a
    case-sensitive one does not, so a check built on probes reaches different
    verdicts on macOS and Linux for the same bundle.
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


def read_engine_prompt(name, partials_dir):
    """Return `(text, overridden)` for one of the engine's prompts.

    The bundle's `prompts/partials/meltiro/NAME.md` wins when it exists, and
    the flag says which happened so a hashing caller can tell hand-authored
    text from engine text. An override that is empty or whitespace-only
    returns the empty string, which is how an engine prompt is left out of a
    model's context.

    An override carries no citations, whichever file it replaces: it is
    rendered literally, so an override of a role prompt supplies that role's
    whole engine half. A shipped PARTIAL carries none either — citation is one
    level deep, and a directive inside an expansion would reach a model as
    literal text. A shipped ROLE prompt is the one file that may cite, and
    `compose_engine_prompt` is what expands it.
    """
    known = engine_prompt_names()
    if name not in known:
        raise ConfigBundleError(
            [f"unknown engine prompt '{name}'; the engine's prompts are "
             f"{list(known)}."]
        )
    override = engine_override_path(name, partials_dir)
    overridden = override.is_file()
    source = override if overridden else ENGINE_PROMPTS_DIR / f"{name}.md"
    content = source.read_text(encoding="utf-8")
    if overridden:
        _reject_nesting(name, content, "engine-prompt override")
    elif name not in ROLE_PROMPT_NAMES:
        _reject_nesting(name, content, "engine partial")
    return content.strip(), overridden


def compose_engine_prompt(role, partials_dir, *, predicates):
    """Render `role`'s engine half: its prompt file, citations expanded.

    A cited partial whose stage is off contributes nothing at all — no
    heading, no gap — and neither does one the bundle overrode with an empty
    file. Those are different reasons: the stage is off for this run, or the
    config author decided the words should not be sent. Only the second is a
    choice about what this run asks, so only the second reaches a config
    fingerprint (see `config_prompt_preimage`).

    An overridden role prompt renders literally, conditional citation and all:
    the bundle supplied the whole of that role's engine half, so there is
    nothing of the engine's left to compose into it — and nothing for an
    override of the partial it silenced to change, here or in the fingerprints
    (`engine_override_pairs`).

    The `{max_checks_per_field}` slot is left for the caller to fill, so this
    half and the bundle's appended text are filled by the same substitution.
    """
    name = role_prompt_name(role)
    text, overridden = read_engine_prompt(name, partials_dir)
    if overridden:
        return text
    return _expand_engine_citations(
        text, name, partials_dir, predicates=predicates)


def _composing_engine_names(role, partials_dir, *, predicates):
    """The engine prompt names whose text reaches a model for `role` this run.

    `composed_engine_names` is the same question of the shipped files, and the
    two agree wherever the bundle leaves the role prompt alone. Where it does
    not, an override of a ROLE prompt is that role's whole engine half,
    rendered literally: `compose_engine_prompt` returns it without reading the
    citation the shipped file carries, so the partial that citation names
    composes nowhere and the set is the role's own name alone.
    """
    name = role_prompt_name(role)
    if engine_override_path(name, partials_dir).is_file():
        return [name]
    return composed_engine_names(role, predicates=predicates)


def engine_override_pairs(role, partials_dir, *, predicates):
    """The bundle's overrides of `role`'s engine half, as sorted `[name,
    text]` pairs.

    The config-owned part of that half, and the whole of what it contributes
    to a config fingerprint. Empty overrides are included and carry their
    empty string: leaving an engine prompt out is a decision about what the
    model is asked, so it has to move the fingerprints exactly as rewriting it
    would.

    An override counts exactly when its text reached a model this run, so an
    override of a partial that composed nowhere is skipped. There are two ways
    a partial composes nowhere and both are here: its stage is off, and the
    toggle that silenced it already rides in `structure_hash`; or the bundle
    overrode the role prompt that cites it, and that override is the whole of
    what the role read. A file no model was shown moved no word of what this
    run asked, and a fingerprint that moved for it would report two runs
    putting one question as two.
    """
    pairs = []
    for name in _composing_engine_names(role, partials_dir,
                                        predicates=predicates):
        override = engine_override_path(name, partials_dir)
        if override.is_file():
            pairs.append([name, override.read_text(encoding="utf-8").strip()])
    return sorted(pairs)


def all_engine_override_pairs(partials_dir, *, predicates):
    """Every role's override pairs, merged and sorted.

    For the bundle-wide `prompts_hash`, which covers the whole prompt surface
    rather than one role's. An engine prompt belongs to exactly one role, so
    the merge cannot produce two entries for one name.
    """
    merged = {}
    for role in ENGINE_ROLE_PROMPTS:
        for name, text in engine_override_pairs(
                role, partials_dir, predicates=predicates):
            merged[name] = text
    return sorted([name, text] for name, text in merged.items())


def config_prompt_preimage(prompt_text, override_pairs):
    """The config-owned identity of one role's prompt, as a canonical string.

    Two components, and only these two: the bundle's appended text as it
    renders, and the bundle's overrides of the engine prompts this run
    composed. An un-overridden engine prompt contributes nothing, which is
    what keeps every bundle's config fingerprints steady across an engine
    release that rewords one.

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

    An engine half overridden away, or an empty bundle prompt file, leaves no
    leading or trailing gap behind it: a rendered prompt is hashed, so
    whitespace is content.
    """
    return BLOCK_SEPARATOR.join(b for b in (b.strip() for b in blocks) if b)


def join_role_message(engine_text, transition, bundle_text):
    """Join one role's system message: engine half, transition, bundle half.

    The transition is the engine's signpost from its own half of the message to
    the review's, and each role has its own (the builders hold the wording,
    beside the rest of their framing). It is emitted only when there is text on
    BOTH sides of it: a bundle whose prompt file is empty is promised no
    briefing that never arrives, and a bundle that overrode the engine's half
    away reads its own opening line first rather than a lone engine sentence.

    Compose-time framing, so it is engine text like the user-block headers: it
    reaches the wire and `engine_fp`, and no config preimage is built through
    here (see `config_prompt_preimage`, which takes the bundle's text on its
    own).
    """
    blocks = [engine_text, bundle_text]
    if engine_text.strip() and bundle_text.strip():
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
    """Whether `predicate` is on, or a loud error naming the directive.

    `name` is the cited name AS THE FILE SPELLS IT, `meltiro:` qualifier and
    all, because the messages below quote the whole directive back and an
    author fixing a typo searches their file for what they wrote. A caller
    that has already unqualified the name would send them looking for a
    placeholder no file contains.
    """
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


def _expand_engine_citations(text, where, partials_dir, *, predicates):
    """Expand a role prompt's `{include:meltiro:NAME}` citations.

    The engine-side counterpart of `substitute_include_placeholders`, sharing
    its patterns and so its whitespace behaviour: a conditional standing alone
    on its line takes the line and one following blank line with it when its
    stage is off, and a partial overridden away does the same, so what a
    reader sees is a document with a paragraph in it or a document without
    one, never a gap where a paragraph was.

    A cited partial is resolved through `read_engine_prompt`, so the bundle's
    override of it lands here exactly as an override of a role prompt lands in
    `compose_engine_prompt`.
    """
    def resolve(name, placeholder):
        if not is_engine_name(name):
            raise ConfigBundleError(
                [f"engine prompt '{where}.md' cites {placeholder}, which "
                 f"names a partial of the config bundle's. An engine prompt "
                 f"composes the engine's own text only. This is an engine "
                 f"bug: the bundle's text is appended after this file, never "
                 f"woven into it."]
            )
        return read_engine_prompt(_unqualified(name), partials_dir)[0]

    def _conditional(match, *, standalone):
        predicate, name = match.group(1), match.group(2)
        content = resolve(name, f"{{include_if:{predicate}:{name}}}")
        if not _predicate_value(predicate, name, predicates):
            return ""
        if standalone:
            return f"{content}\n\n" if content else ""
        return content

    text = _INCLUDE_IF_LINE.sub(
        lambda m: _conditional(m, standalone=True), text)
    text = _INCLUDE_IF_PLACEHOLDER.sub(
        lambda m: _conditional(m, standalone=False), text)
    return _INCLUDE_PLACEHOLDER.sub(
        lambda m: resolve(m.group(1), f"{{include:{m.group(1)}}}"), text)


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
    `meltiro:`-qualified name: the engine composes its own prompts, and this
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
