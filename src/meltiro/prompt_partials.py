"""Expand `{include:NAME}` partials in a config bundle's prompt files.

A prompt file may cite a reusable block with an `{include:NAME}`
placeholder. At render time the placeholder is replaced by the content of
`prompts/partials/NAME.md` within the same config bundle. This lets a block
that several prompts share live in one file instead of being copy-pasted, so
the copies cannot drift apart.

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
(`config_bundle._validate_prompt_partials`) so a missing partial or a
nested include fails before any API spend. The render-time function here
raises the same loud error, so a broken include never ships silently in a
prompt.
"""

import re
from pathlib import Path

from meltiro.errors import ConfigBundleError


# Matches `{include:NAME}` where NAME is a file-stem-shaped token, mirroring
# the reference-placeholder grammar in reference_lists.py.
_INCLUDE_PLACEHOLDER = re.compile(r"\{include:([A-Za-z0-9_.\-]+)\}")

# Matches `{include_if:PREDICATE:NAME}`.
_INCLUDE_IF_PLACEHOLDER = re.compile(
    r"\{include_if:([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-]+)\}")

# The same placeholder when it stands alone on its line, with the line's
# newline and one following blank line, so omitting it closes the gap.
_INCLUDE_IF_LINE = re.compile(
    r"(?m)^[ \t]*\{include_if:([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-]+)\}[ \t]*"
    r"\n(?:[ \t]*\n)?")

# The predicates a conditional include may name. Deliberately closed: an
# unknown predicate is a typo or a stage that does not exist, and silently
# treating it as false would hide a block rather than report the mistake.
PREDICATE_NAMES = ("checker", "review")

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


def included_names(text):
    """Return every partial name cited by `text`, conditional ones included.

    Used at config-load time to check every prompt's includes against the
    `partials/` directory. Conditional names are returned whatever their
    predicate would evaluate to, so a partial cited only by a disabled
    branch is still required to exist.
    """
    names = {m.group(1) for m in _INCLUDE_PLACEHOLDER.finditer(text)}
    names |= {m.group(2) for m in _INCLUDE_IF_PLACEHOLDER.finditer(text)}
    return names


def _read_partial(name, partials_dir, placeholder):
    partial_path = Path(partials_dir) / f"{name}.md"
    if not partial_path.is_file():
        raise ConfigBundleError(
            [f"prompt cites unknown partial '{name}' via "
             f"{placeholder}; expected a file at "
             f"{partial_path}. Add the partial or fix the placeholder."]
        )
    content = partial_path.read_text(encoding="utf-8")
    if _INCLUDE_PLACEHOLDER.search(content) or \
            _INCLUDE_IF_PLACEHOLDER.search(content):
        raise ConfigBundleError(
            [f"partial '{name}.md' contains a nested include placeholder; "
             f"nesting is not supported. Inline the nested content or "
             f"flatten the partials."]
        )
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
    """Replace every include placeholder in `text`.

    `{include:NAME}` is substituted with the stripped content of
    `partials_dir / "NAME.md"`. `{include_if:PREDICATE:NAME}` is substituted
    with the same content when `predicates[PREDICATE]` is true, and removed
    when it is false -- taking its line and one following blank line with it
    when it stands alone, so the omission leaves no ragged gap.

    A placeholder naming a partial that does not exist raises
    `ConfigBundleError`, whether or not its branch is taken, so a typo cannot
    hide behind a disabled stage. So does a partial that itself contains an
    include placeholder: nesting is not supported.

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
