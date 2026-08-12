"""Expand `{include:NAME}` partials in a config bundle's prompt files.

A prompt file may cite a reusable block with an `{include:NAME}`
placeholder. At render time the placeholder is replaced by the content of
`prompts/partials/NAME.md` within the same config bundle. This lets a block
that several prompts share live in one file instead of being copy-pasted, so
the copies cannot drift apart.

`meltiro:` is a RESERVED include namespace holding the engine's own contract:
`{include:meltiro:NAME}` resolves to `meltiro/engine_prompts/NAME.md`, one
file per section, listed by `engine_section_names()`. Those sections describe
what the ENGINE does — the extractor's tool workflow, the evidence grammar the
validator enforces, the checker's isolation — facts a config author would
otherwise have to restate correctly for the pipeline to behave as described. A
bundle composes them rather than re-describing them, and a name outside the
list is refused rather than treated as a bundle partial.

A bundle OVERRIDES an engine section by shipping
`prompts/partials/meltiro/NAME.md`: the override wins wherever that section is
cited, so any review can replace any engine text with its own. The name must
still be a known engine section; an override cannot invent one. That directory
is ENUMERATED at load (`config_bundle._validate_engine_overrides`), so a file
whose name is not exactly a shipped section's is refused rather than sitting
there overriding nothing.

The two resolutions differ for HASHING. Everything hand-authored belongs to
the config's identity and everything engine-authored to the engine's, so a
render taken for a hash (`mode=HASH`) leaves an un-overridden engine section
as its directive token, unexpanded, while an overridden one contributes the
override's text like any other config prose. A render taken for the wire
(`mode=WIRE`, the default) always expands everything: the model is sent the
whole contract either way. Engine text moves `engine_fp` instead, through the
source digest that hashes `engine_prompts/*.md` beside the package's modules
(see `run_log.source_hash`).

A prompt may also make a block CONDITIONAL on a pipeline stage, with
`{include_if:PREDICATE:NAME}`. The block is included when that stage is
enabled for the run and omitted entirely when it is not: a partial telling
the extractor that a checker will challenge its fields is false under
`max_checks_per_field: 0`, and a prompt must not brief a model on a stage
that does not run.

Two predicates, matching the two structure toggles the engine resolves
before any prompt is rendered:

  - `checker` -- true when `max_checks_per_field > 0`
  - `review`  -- true when `final_review` is on

Rules:

  - Includes expand BEFORE `{reference:NAME}` substitution
    (`reference_lists.substitute_reference_placeholders`), so a partial may
    itself contain `{reference:...}` placeholders and they resolve exactly
    as they would inline. Conditional includes resolve in the same pass.
  - A partial may NOT itself contain an `{include:...}` or
    `{include_if:...}` placeholder. Nesting is rejected loudly rather than
    resolved recursively; the shared blocks this exists for are flat, and
    one level keeps the expansion trivial to reason about.
  - A conditional's partial must EXIST whether or not its predicate is on,
    so a typo cannot hide behind a disabled stage and surface only when
    someone turns that stage back on.
  - An omitted conditional takes its whole line with it, plus one following
    blank line, so a disabled block leaves no ragged gap in the rendered
    prompt. A rendered prompt is hashed into `prompts_hash`, so whitespace
    is not cosmetic here.

An included partial's content is inserted with surrounding whitespace
stripped, so the placeholder is expected to sit on its own line as a
block-level include. The `partials/` directory is optional: a bundle whose
prompts cite no include placeholder needs no `partials/` directory at all.

The config bundle validates every placeholder at load time
(`config_bundle._validate_prompt_partials`) so a missing partial, an unknown
engine section, or a nested include fails before any API spend. The
render-time function here raises the same loud error, so a broken include
never ships silently in a prompt.
"""

import re
from pathlib import Path

from meltiro.errors import ConfigBundleError


# The reserved namespace, and the directory its sections are read from. The
# directory is the single source of what exists: a section is added by adding
# its file, and nothing lists the names a second time.
ENGINE_NAMESPACE = "meltiro"
ENGINE_PROMPTS_DIR = Path(__file__).resolve().parent / "engine_prompts"

# A file-stem-shaped token, mirroring the reference-placeholder grammar in
# reference_lists.py, optionally qualified by the reserved namespace. One
# capture group either way, so a name carries its own namespace and the
# resolution below is the only place that splits it.
_NAME = r"[A-Za-z0-9_.\-]+"
_QUALIFIED_NAME = rf"(?:{ENGINE_NAMESPACE}:)?{_NAME}"

# Matches `{include:NAME}` and `{include:meltiro:NAME}`.
_INCLUDE_PLACEHOLDER = re.compile(rf"\{{include:({_QUALIFIED_NAME})\}}")

# Matches `{include_if:PREDICATE:NAME}`, the name qualified or not.
_INCLUDE_IF_PLACEHOLDER = re.compile(
    rf"\{{include_if:({_NAME}):({_QUALIFIED_NAME})\}}")

# The same placeholder when it stands alone on its line, with the line's
# newline and one following blank line, so omitting it closes the gap.
_INCLUDE_IF_LINE = re.compile(
    rf"(?m)^[ \t]*\{{include_if:({_NAME}):({_QUALIFIED_NAME})\}}[ \t]*"
    r"\n(?:[ \t]*\n)?")

# The predicates a conditional include may name. Deliberately closed: an
# unknown predicate is a typo or a stage that does not exist, and silently
# treating it as false would hide a block rather than report the mistake.
PREDICATE_NAMES = ("checker", "review")

# The two renders a prompt is put through, and the ONE thing that differs
# between them: whether an un-overridden engine section expands.
#
#   WIRE -- what a model is sent. Everything expands.
#   HASH -- what a config fingerprint is taken over. An un-overridden
#           `{include:meltiro:NAME}` contributes its directive token instead of
#           its text, so engine prose rides in `engine_fp` and a bundle's
#           fingerprints hold across an engine release that reworded it.
#
# One argument rather than two rendering paths: the text a run sends and the
# text it fingerprints come off the same function, so they cannot drift in any
# respect but this one.
WIRE = "wire"
HASH = "hash"
_MODES = (WIRE, HASH)

# Sentinel for the load-time validators, which must see EVERY branch: a
# reference or a banned placeholder hiding inside a disabled block is still a
# defect in the bundle, and a config that only validates on the toggles it
# happens to ship today would pass its own check and fail when a stage is
# switched on. Not for render paths -- see `substitute_include_placeholders`.
EXPAND_ALL_BRANCHES = "expand-all-branches"


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
    """The engine sections a prompt may cite as `{include:meltiro:NAME}`.

    Read off `engine_prompts/` every call, so the shipped files are the whole
    answer and no second list can fall out of step with them.
    """
    return tuple(sorted(p.stem for p in ENGINE_PROMPTS_DIR.glob("*.md")))


def is_engine_name(name):
    """True for a name in the reserved `meltiro:` namespace."""
    return str(name).startswith(f"{ENGINE_NAMESPACE}:")


def engine_section(name):
    """Strip the reserved namespace off `name`, or None when it has none."""
    if not is_engine_name(name):
        return None
    return str(name).split(":", 1)[1]


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
    branch is still required to exist. Engine sections are excluded — they
    resolve from the package, not from `partials/`, and are returned by
    `engine_included_names` for their own check.
    """
    return {n for n in _all_cited_names(text) if not is_engine_name(n)}


def engine_included_names(text):
    """Return every ENGINE section cited by `text`, unqualified, conditional
    ones included.

    The `meltiro:` prefix is stripped, so the values compare directly against
    `engine_section_names()`.
    """
    return {engine_section(n) for n in _all_cited_names(text)
            if is_engine_name(n)}


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


def read_engine_section(name, partials_dir, placeholder):
    """Return `(text, overridden)` for an engine section.

    The bundle's `prompts/partials/meltiro/NAME.md` wins when it exists: a
    review may replace any engine text with its own, and the flag says which
    happened so a hashing caller can tell hand-authored text from engine text.
    The name must be a known section either way — an override cannot invent
    one, because a typo that silently became a new section would compose
    nothing and report nothing.

    Whichever file supplies the text is held to the no-nesting rule. The
    engine's own copy is checked by the same call as an override: a shipped
    section carrying `{include:...}` would put the literal directive in front
    of a model, which is the failure the rule exists to prevent whoever wrote
    the text.
    """
    known = engine_section_names()
    if name not in known:
        raise ConfigBundleError(
            [f"prompt cites unknown engine section '{name}' via "
             f"{placeholder}; the "
             f"{ENGINE_NAMESPACE}: namespace is reserved for the engine's own "
             f"sections, which are {list(known)}. Fix the name, or drop the "
             f"{ENGINE_NAMESPACE}: prefix to cite a partial of your own."]
        )
    override = engine_override_path(name, partials_dir)
    overridden = override.is_file()
    source = override if overridden else ENGINE_PROMPTS_DIR / f"{name}.md"
    content = source.read_text(encoding="utf-8")
    _reject_nesting(
        name, content,
        "engine-section override" if overridden else "engine section")
    return content.strip(), overridden


def _read_partial(name, partials_dir, placeholder, mode=WIRE):
    """The text a placeholder expands to, engine namespace included.

    `mode` is `WIRE` or `HASH`; it decides only what an UN-OVERRIDDEN engine
    section contributes — its text on the wire, its directive `placeholder` in
    a hash. Everything else is config, so it expands the same either way.
    """
    section = engine_section(name)
    if section is not None:
        content, overridden = read_engine_section(
            section, partials_dir, placeholder)
        if overridden or mode == WIRE:
            return content
        return placeholder

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


def substitute_include_placeholders(text, partials_dir, *, predicates=None,
                                    mode=WIRE):
    """Replace every include placeholder in `text`.

    `{include:NAME}` is substituted with the stripped content of
    `partials_dir / "NAME.md"`, and `{include:meltiro:NAME}` with the engine
    section of that name (or the bundle's override of it, when
    `partials/meltiro/NAME.md` exists). `{include_if:PREDICATE:NAME}` is
    substituted with the same content when `predicates[PREDICATE]` is true,
    and removed when it is false -- taking its line and one following blank
    line with it when it stands alone, so the omission leaves no ragged gap.
    The namespace composes with predicates exactly as a plain name does.

    A placeholder naming a partial that does not exist raises
    `ConfigBundleError`, whether or not its branch is taken, so a typo cannot
    hide behind a disabled stage. So does an unknown engine section, and so
    does a partial that itself contains an include placeholder: nesting is not
    supported.

    `predicates` is a mapping from `PREDICATE_NAMES` to bool, normally built
    by `stage_predicates`. Pass `EXPAND_ALL_BRANCHES` from a validator that
    must inspect every branch. Omitting it is legal only for text citing no
    conditional include; text that does cite one raises, rather than quietly
    picking a branch.

    `mode` is `WIRE` (the default: every include expands, which is what a
    model is sent) or `HASH` (an un-overridden engine section stays as its own
    directive token, so engine prose reaches no config fingerprint). A
    disabled branch is omitted in both, so the toggles a run resolved still
    move a hash taken under `HASH`.

    `partials_dir` is only read when `text` actually cites a partial, so a
    bundle with no include placeholders never needs the directory to exist.
    """
    if mode not in _MODES:
        raise ValueError(
            f"unknown render mode {mode!r}; expected one of {list(_MODES)}. "
            f"This is an engine bug: a render path picks {WIRE} or {HASH}, "
            f"and the choice decides whether engine text reaches a config "
            f"fingerprint.")
    partials_dir = Path(partials_dir)

    def _repl_conditional_line(match):
        predicate, name = match.group(1), match.group(2)
        content = _read_partial(
            name, partials_dir, f"{{include_if:{predicate}:{name}}}", mode)
        if _predicate_value(predicate, name, predicates):
            return f"{content}\n\n"
        return ""

    def _repl_conditional_inline(match):
        predicate, name = match.group(1), match.group(2)
        content = _read_partial(
            name, partials_dir, f"{{include_if:{predicate}:{name}}}", mode)
        if _predicate_value(predicate, name, predicates):
            return content
        return ""

    def _repl_include(match):
        name = match.group(1)
        return _read_partial(
            name, partials_dir, f"{{include:{name}}}", mode)

    # Line-standing conditionals first, so an omitted block closes its own
    # gap; whatever is left is inline and substitutes in place.
    text = _INCLUDE_IF_LINE.sub(_repl_conditional_line, text)
    text = _INCLUDE_IF_PLACEHOLDER.sub(_repl_conditional_inline, text)
    return _INCLUDE_PLACEHOLDER.sub(_repl_include, text)
