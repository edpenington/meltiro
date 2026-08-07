"""Load and render reference lists from a config bundle's `reference/` dir.

A reference list is a YAML file naming canonical entities the review cares
about (the synthetic fixture's accepted durability gauges, say). Each file's
stem is the list's name. A template field names a list via
`canonical_reference:`; a prompt cites one via a `{reference:NAME}`
placeholder substituted at render time.

The engine is entity-agnostic but not shape-agnostic. An entry is one of
exactly two forms:

  - a bare non-empty string, which is its own canonical name; or
  - a mapping carrying `tool_name` (the canonical name) plus the optional
    `aliases:` and `search_terms:` keys (`_ENTRY_KEYS`).

Both are enforced at load (`_validate_entries`), before any model call:
`entry_canonical_name` reads one key, and the bare-string fallback would
stringify a misspelt mapping into a Python repr reaching the prompt, the
stored value, and `reference_lists_hash` — a clean-looking run over a
vocabulary nobody wrote. A misspelt `aliases:` would silently disable that
entry's canonicalisation.

Two canonical names in one list may not share a normalised form
(`_validate_canonical_names`): matching is by normalised form, so a
colliding pair makes every lookup of either name ambiguous and leaves any
field referencing the list impossible to fill.

A file's top level is either a YAML list of entries, or a mapping with
exactly one list-valued key (the entries) plus the optional scalar `label:`
(`_LIST_FILE_KEYS`) — a display name used only by the template render
(`load_reference_list_labels`); it reaches no model and moves no
fingerprint. Any other top-level key is a load error.

Two loaders here — entries and labels — share one rule set: both route each
file through `_validate_list_file`, so neither can accept a file the other
would reject.
"""

import re
from pathlib import Path

import yaml

from meltiro.errors import ConfigBundleError
from meltiro.yaml_strict import strict_load


# Matches `{reference:NAME}` where NAME is a file-stem-shaped token.
_REFERENCE_PLACEHOLDER = re.compile(r"\{reference:([A-Za-z0-9_.\-]+)\}")

# The canonical-name key of a mapping entry. `entry_canonical_name` reads this
# and nothing else, so an entry that lacks it has no canonical name at all.
_ENTRY_NAME_KEY = "tool_name"

# Keys one reference-list entry may carry. Anything else is a load error —
# the module docstring says why silence is the worst outcome here.
# `search_terms` is read by no code in this package (it is for upstream
# screening; `reference_lists_hash` excludes it deliberately) but is a
# documented part of the entry shape, so it is allowed and ignored.
_ENTRY_KEYS = {_ENTRY_NAME_KEY, "search_terms", "aliases"}

# Top-level keys a reference-list FILE may carry alongside its single
# list-valued entries key (whose name the review chooses, so it cannot be
# allowlisted). `label` is the presentation-only display name; see
# `load_reference_list_labels`. Any other top-level key is a load error, so
# a misspelt `labell:` fails instead of being silently dropped.
_LIST_FILE_KEYS = {"label"}


def referenced_names(text):
    """Return the set of reference-list names cited by `{reference:NAME}`
    placeholders in `text`.

    Used at config-load time to check every prompt's placeholders against
    the loaded reference lists, so an unresolvable placeholder fails loudly
    before any run rather than mid-run at render time.
    """
    return {m.group(1) for m in _REFERENCE_PLACEHOLDER.finditer(text)}


# ---------------------------------------------------------------------------
# Entry accessors + canonicalisation index
# ---------------------------------------------------------------------------
#
# A `canonical_reference` field's value must resolve to a canonical entry
# name or an alias of one. The same accessor feeds the prompt render and the
# validator, keeping the prompt the model reads and the value the validator
# accepts in lock-step.


def entry_canonical_name(entry):
    """Canonical name of a reference entry.

    Mirrors `render_reference_block`: a mapping renders by its `tool_name`, a
    bare string as itself — the exact spelling a matched value canonicalises
    to on store.

    The `str(entry)` fallback serves the bare-string form only; a mapping
    lacking `tool_name` is rejected at load (`_validate_entries`), because a
    Python repr must never become a canonical name. The check lives at load
    so this accessor stays pure and total for hand-built entry lists.
    """
    if isinstance(entry, dict) and _ENTRY_NAME_KEY in entry:
        return str(entry[_ENTRY_NAME_KEY])
    return str(entry)


def entry_aliases(entry):
    """Optional alias list for a reference entry; [] when none declared.

    `aliases` is the canonicalisation key, distinct from `search_terms`
    (upstream screening only; never canonicalises). A non-list value is
    caught by `_validate_aliases`, a misspelt key by `_validate_entries` —
    both at load. This accessor stays defensive so the index builder never
    crashes on a hand-built entry.
    """
    if not isinstance(entry, dict):
        return []
    raw = entry.get("aliases")
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    return [raw]


def _normalise_ref(text):
    """Normal form for reference-list matching.

    A thin wrapper over the verbatim-quote normaliser
    (`quote_check.normalise_quote_text`): NFKC folding, smart-quote and dash
    folding, whitespace collapse, and lowercasing. Reusing it means
    reference matching tolerates the same surface variation quote-checking
    does, and there is one normalisation of record.
    """
    from meltiro.quote_check import normalise_quote_text
    return normalise_quote_text(text)


def build_reference_index(entries):
    """Build a resolution index for `resolve_reference_value`.

    Returns `{"names": {norm: {canonical, ...}}, "aliases": {norm:
    {canonical, ...}}}`, where `norm` is the normalised form of a canonical
    name or alias and the value set holds the exact canonical spelling(s)
    it maps to. Pure: it does no validation (that is `_validate_aliases`,
    run at load), so a hand-built entries list can exercise the ambiguous
    branches directly.
    """
    names = {}
    aliases = {}
    for e in entries or []:
        name = entry_canonical_name(e)
        names.setdefault(_normalise_ref(name), set()).add(name)
    for e in entries or []:
        name = entry_canonical_name(e)
        nn = _normalise_ref(name)
        for alias in entry_aliases(e):
            na = _normalise_ref(alias)
            if not na or na == nn:
                continue
            aliases.setdefault(na, set()).add(name)
    return {"names": names, "aliases": aliases}


def resolve_reference_value(value, index):
    """Resolve one entered value against a reference index.

    Returns one of:
      - {"status": "canonical", "canonical": name}: matched a canonical
        entry name (names take priority over aliases).
      - {"status": "alias", "canonical": name}: matched exactly one alias.
      - {"status": "ambiguous", "candidates": [name, ...]}: matched several
        entries.
      - {"status": "no_match", "suggestions": [name, ...]}: matched nothing;
        the closest canonical names ranked by fuzzy similarity.
      - {"status": "empty"}: the value normalised to empty.
    """
    nv = _normalise_ref(value)
    if not nv:
        return {"status": "empty"}
    name_hits = index["names"].get(nv)
    if name_hits:
        if len(name_hits) == 1:
            return {"status": "canonical", "canonical": next(iter(name_hits))}
        return {"status": "ambiguous", "candidates": sorted(name_hits)}
    alias_hits = index["aliases"].get(nv)
    if alias_hits:
        if len(alias_hits) == 1:
            return {"status": "alias", "canonical": next(iter(alias_hits))}
        return {"status": "ambiguous", "candidates": sorted(alias_hits)}
    return {"status": "no_match",
            "suggestions": rank_reference_suggestions(nv, index)}


def rank_reference_suggestions(normalised_value, index, n=5):
    """Top `n` canonical names closest to `normalised_value`.

    Scores against normalised canonical names AND aliases (an alias can be
    the nearest surface form), but always returns canonical names.
    difflib.SequenceMatcher ratio over the normalised strings; ties broken
    alphabetically for determinism.
    """
    import difflib

    scored = {}

    def _consider(norm_key, canon_set):
        for canon in canon_set:
            r = difflib.SequenceMatcher(None, normalised_value, norm_key).ratio()
            if r > scored.get(canon, -1.0):
                scored[canon] = r

    for norm_key, canon_set in index["names"].items():
        _consider(norm_key, canon_set)
    for norm_key, canon_set in index["aliases"].items():
        _consider(norm_key, canon_set)

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [canon for canon, _ in ranked[:n]]


def load_reference_lists(reference_dir):
    """Load every `*.yaml` in `reference_dir`. Return `{name: entries}`.

    `name` is the file stem. A file's top level is either a YAML list of
    entries, or a mapping with exactly one list-valued key (the entries)
    whose only permitted sibling is the scalar `label:` (see
    `load_reference_list_labels`). Non-YAML, empty, or entry-less files are
    rejected loudly, as are a malformed `label:`, an unknown top-level key
    (`_LIST_FILE_KEYS`), and any entry outside `_ENTRY_KEYS`
    (`_validate_entries`) — all at load, before any model call.

    The return shape is deliberately `{name: entries}` and nothing else:
    every fingerprint (`reference_lists_hash`, and through it `config_fp` /
    `checker_fp`) depends on it, so the presentation-only label rides a
    separate loader and never enters this map.
    """
    reference_dir = Path(reference_dir)
    out = {}
    for path in sorted(reference_dir.glob("*.yaml")):
        out[path.stem] = _load_one(path)
    return out


def load_reference_list_labels(reference_dir):
    """Load every `*.yaml` in `reference_dir`. Return `{name: label}` for the
    lists that declare a non-empty scalar top-level `label:`.

    `name` is the file stem, matching `load_reference_lists`. The label names
    how the list reads in human-facing renders: with `label: Gauge Reference
    List` a canonical-reference field's value domain reads "Name from the
    Gauge Reference List" instead of the raw list id. A label-less list is
    simply absent, and the render falls back to the file stem.

    Consumed only by the template render (`render_template`); the label never
    reaches a model and moves no fingerprint, so it must not ride in the
    `{name: entries}` map the pipeline hashes. The rule set IS shared: both
    loaders route each file through `_validate_list_file`, so a render that
    only wants a display name still refuses a file whose entries would never
    load.
    """
    reference_dir = Path(reference_dir)
    labels = {}
    if not reference_dir.is_dir():
        return labels
    for path in sorted(reference_dir.glob("*.yaml")):
        _entries, label = _validate_list_file(_read_raw(path), path)
        if label is not None:
            labels[path.stem] = label
    return labels


def _read_raw(path):
    """Strict-parse one reference-list file to its raw YAML value.

    Shared by the entries loader and the labels loader so both read a file the
    same way and report an unreadable file identically.
    """
    try:
        return strict_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as e:
        raise ConfigBundleError(
            [f"reference list {path.name} could not be read as YAML: {e}"],
            path=path.parent,
        )


def _label_from_raw(raw, path):
    """The optional top-level display `label:` of a reference list, or None.

    Presentation-only: never rendered into a prompt and inert to
    `reference_lists_hash`. Only a top-level mapping can carry one; a
    top-level YAML list has no place for a sibling key. A `label:` that is
    present but not a non-empty string is a loud config error; the value
    itself is returned verbatim.
    """
    if not isinstance(raw, dict) or "label" not in raw:
        return None
    label = raw["label"]
    if not isinstance(label, str) or not label.strip():
        raise ConfigBundleError(
            [f"reference list {path.name} has a `label:` that is not a "
             f"non-empty string ({label!r})."],
            path=path.parent,
        )
    return label


def _load_one(path):
    entries, _label = _validate_list_file(_read_raw(path), path)
    return entries


def _validate_list_file(raw, path):
    """Apply every reference-list rule to one file's parsed YAML.

    Returns `(entries, label)`. The single rule set both loaders run. Order
    matters: the shape and key allowlist runs before the name and alias
    rules, which report by canonical name and so need every entry known to
    have one.
    """
    label = _label_from_raw(raw, path)
    entries = _entries_from_raw(raw, path)
    if not entries:
        raise ConfigBundleError(
            [f"reference list {path.name} has no entries"], path=path.parent)
    _validate_entries(entries, path)
    _validate_canonical_names(entries, path)
    _validate_aliases(entries, path)
    return entries, label


def _entry_where(entry, position, path):
    """How one entry is named in a load error: by canonical name when it has
    one, else by 1-based position — a mistyped `tool_name:` is exactly the
    case with no name to quote.
    """
    if isinstance(entry, dict):
        name = entry.get(_ENTRY_NAME_KEY)
        if isinstance(name, str) and name.strip():
            return f"reference list {path.name}: entry {name!r}"
    return f"reference list {path.name}: entry #{position}"


def _validate_entries(entries, path):
    """Validate the shape and keys of every entry, at load.

    Two forms are legal (see the module docstring): a bare non-empty string,
    or a mapping carrying `tool_name` and only keys from `_ENTRY_KEYS`.
    Everything else — numbers, `None` (a blank list item), nested lists — is
    refused for one reason: `entry_canonical_name` would fall through to
    `str(entry)`, and a Python repr must never become a canonical name. Every
    offending entry is collected so one bad list reports them all.
    """
    problems = []
    for position, entry in enumerate(entries, start=1):
        where = _entry_where(entry, position, path)
        if isinstance(entry, str):
            if not entry.strip():
                problems.append(
                    f"{where} is an empty string. A bare-string entry is its "
                    f"own canonical name, so it must be non-empty.")
            continue
        if not isinstance(entry, dict):
            problems.append(
                f"{where} must be a mapping with keys "
                f"{sorted(_ENTRY_KEYS)}, or a bare non-empty string naming "
                f"the entry; got {type(entry).__name__} ({entry!r}). Quote a "
                f"scalar to use it as a bare-string entry.")
            continue
        unknown = sorted(set(entry) - _ENTRY_KEYS)
        if unknown:
            problems.append(
                f"{where} has unknown entry key(s) {unknown}. Only "
                f"{sorted(_ENTRY_KEYS)} are allowed.")
        if _ENTRY_NAME_KEY not in entry:
            problems.append(
                f"{where} has no `{_ENTRY_NAME_KEY}:`. Every mapping entry "
                f"needs one: it is the canonical name rendered into the "
                f"prompt, stored as the extracted value, and hashed into the "
                f"config fingerprint.")
        elif (not isinstance(entry[_ENTRY_NAME_KEY], str)
                or not entry[_ENTRY_NAME_KEY].strip()):
            problems.append(
                f"{where} has a `{_ENTRY_NAME_KEY}:` that is not a non-empty "
                f"string ({entry[_ENTRY_NAME_KEY]!r}).")
    if problems:
        raise ConfigBundleError(problems, path=path.parent)


def _validate_canonical_names(entries, path):
    """Reject two canonical names that share a normalised form.

    Matching is by normalised form (`_normalise_ref`), so `"WDS-9"` and
    `"wds-9"` are one index key; `resolve_reference_value` then answers
    `ambiguous` for EVERY value entered for either name — including each name
    typed exactly as written — and a field referencing the list can never be
    filled. The index builder is deliberately pure, so the collision is
    caught here at load, where renaming one entry is the whole fix. The
    message names both entries and the colliding form, because the two
    spellings are by construction hard to tell apart by eye. Every colliding
    pair is collected.
    """
    problems = []
    first_by_norm = {}
    for e in entries:
        name = entry_canonical_name(e)
        norm = _normalise_ref(name)
        if norm in first_by_norm:
            problems.append(
                f"reference list {path.name}: canonical names "
                f"{first_by_norm[norm]!r} and {name!r} both normalise to "
                f"{norm!r}. Reference matching is by normalised form, so every "
                f"value entered for either name would resolve to both and be "
                f"rejected as ambiguous, leaving any field that references "
                f"this list impossible to fill. Rename one of the two entries.")
            continue
        first_by_norm[norm] = name
    if problems:
        raise ConfigBundleError(problems, path=path.parent)


def _validate_aliases(entries, path):
    """Validate the optional `aliases:` key on every entry.

    Rules:
      - each alias must be a non-empty string;
      - an alias duplicating another entry's canonical name or another
        entry's alias (after normalisation) is a config error;
      - an alias equal to its own entry's canonical name is redundant but
        harmless, and is allowed (it simply has no effect).

    Every offending alias is collected so one bad list reports them all.
    """
    problems = []
    # All canonical names first, so an alias can be checked against a name
    # that appears later in the file.
    norm_to_name = {}
    for e in entries:
        name = entry_canonical_name(e)
        norm_to_name.setdefault(_normalise_ref(name), name)

    seen_alias = {}  # normalised alias -> owning canonical name
    for e in entries:
        if not isinstance(e, dict) or e.get("aliases") is None:
            continue
        name = entry_canonical_name(e)
        norm_name = _normalise_ref(name)
        raw = e["aliases"]
        if not isinstance(raw, list):
            problems.append(
                f"reference list {path.name}: entry {name!r} has an "
                f"`aliases:` value that is not a list.")
            continue
        for alias in raw:
            if not isinstance(alias, str) or not alias.strip():
                problems.append(
                    f"reference list {path.name}: entry {name!r} has an "
                    f"alias that is not a non-empty string ({alias!r}).")
                continue
            na = _normalise_ref(alias)
            if not na:
                problems.append(
                    f"reference list {path.name}: entry {name!r} has an "
                    f"alias {alias!r} that normalises to empty.")
                continue
            if na == norm_name:
                # Redundant with the entry's own name; allowed, no effect.
                continue
            if na in norm_to_name:
                problems.append(
                    f"reference list {path.name}: alias {alias!r} on entry "
                    f"{name!r} duplicates the canonical name of entry "
                    f"{norm_to_name[na]!r}.")
                continue
            if na in seen_alias and seen_alias[na] != name:
                problems.append(
                    f"reference list {path.name}: alias {alias!r} on entry "
                    f"{name!r} duplicates an alias of entry "
                    f"{seen_alias[na]!r}.")
                continue
            seen_alias[na] = name
    if problems:
        raise ConfigBundleError(problems, path=path.parent)


def _entries_from_raw(raw, path):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        entries_keys = [k for k, v in raw.items() if isinstance(v, list)]
        if len(entries_keys) != 1:
            raise ConfigBundleError(
                [f"reference list {path.name} must be a YAML list, or a "
                 f"mapping with exactly one list-valued key; got a mapping "
                 f"with {len(entries_keys)} list-valued key(s)"],
                path=path.parent,
            )
        # Top-level allowlist: the entries key plus `label:`, nothing else,
        # so a misspelt `labell:` fails instead of being silently dropped.
        unknown = sorted(set(raw) - set(entries_keys) - _LIST_FILE_KEYS)
        if unknown:
            raise ConfigBundleError(
                [f"reference list {path.name} has unknown top-level key(s) "
                 f"{unknown}. Only the entries key ({entries_keys[0]!r}) and "
                 f"{sorted(_LIST_FILE_KEYS)} are allowed."],
                path=path.parent,
            )
        return raw[entries_keys[0]]
    raise ConfigBundleError(
        [f"reference list {path.name} must be a non-empty YAML list (or a "
         f"mapping with one list-valued key), got {type(raw).__name__}"],
        path=path.parent,
    )


def render_reference_block(entries):
    """Render a reference list as the text block substituted into prompts.

    Each entry renders by its canonical name (`entry_canonical_name`); the
    surrounding prompt supplies the human framing, so the block is just the
    bulleted names. Aliases are deliberately NOT rendered: they are a
    matching aid, not part of what the model is shown.
    """
    return "\n".join(f"- {entry_canonical_name(e)}" for e in (entries or []))


def substitute_reference_placeholders(text, reference_lists, *, path=None):
    """Replace every `{reference:NAME}` placeholder in `text` with the named
    list's rendered block.

    An unresolvable placeholder raises `ConfigBundleError` rather than
    shipping a broken prompt. A list no prompt cites is fine — simply not
    substituted. `path` is the config bundle directory, carried into the
    error so the message says which bundle is short a list; optional because
    some callers hold no bundle path, and the message then names only the
    list and placeholder.
    """
    reference_lists = reference_lists or {}

    def _repl(match):
        name = match.group(1)
        if name not in reference_lists:
            raise ConfigBundleError(
                [f"prompt cites unknown reference list '{name}' via "
                 f"{{reference:{name}}}; available: {sorted(reference_lists)}. "
                 f"Add a reference/{name}.yaml to the config bundle or fix "
                 f"the placeholder."],
                path=path,
            )
        return render_reference_block(reference_lists[name])

    return _REFERENCE_PLACEHOLDER.sub(_repl, text)
