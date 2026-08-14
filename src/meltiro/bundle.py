"""Paper bundle: the user-supplied input contract for one paper.

A paper bundle is a self-contained directory the user assembles for a
single paper. Papers are copyrighted, so they are NEVER bundled with the
code; the user supplies them at run time. The layout is deliberately
minimal and human-authorable:

    my-paper/
      manifest.json    (required)  metadata + identity
      text.md          (required)  the paper's extracted full text (UTF-8)
      figures/         cropped tables/figures as *.png, one per manifest
                       exhibit (absent when the manifest declares none)

`manifest.json` keys:
  - schema_version  (required, integer, must equal 2)
  - id              (required, non-empty str, ^[A-Za-z0-9._-]+$)
  - title           (required, non-empty str)
  - exhibits        (required, list of {label, caption}, each optionally
                     carrying notes; may be empty)
  - doi             (optional, str)
  - summary         (optional, str; if present must be non-empty)
Any unknown key, or any wrong type, is an error.

`summary` is the CHECKER's identity context for study-level fields (the
checker never reads the paper). For a published paper paste a trusted
abstract; for grey literature paste an executive summary or a couple of
hand-written sentences. When absent, the pipeline falls back to the
extracted `role: summary` field (see template.py); when neither exists it
DEGRADES rather than failing — the checker is given the manifest's title and
DOI as minimal identity context, which says which paper this is but not what
it found, and the run records an `identity-degradation` warning saying so
(`Orchestrator._degraded_identity_context`).

Nothing here is consulted at all on a run with no checker
(`max_checks_per_field: 0`): the identity context exists to be sent to the
checker, so a bundle with no summary and no populated summary field runs
clean under that configuration.

Each `figures/*.png` file's stem is its label; this matches how the
extractor cites image evidence as `<img>label</img>` (the filename stem
already IS the citation token). Non-`.png` files in `figures/` are an
error: an unexpected asset fails loudly rather than being ignored.

`exhibits` is the manifest's declaration of what the bundle supplies as
cropped images: one `{"label": ..., "caption": ...}` object per table or
figure, the label being the `figures/<label>.png` stem and the caption
the exhibit's caption as the paper prints it. It is REQUIRED, and it may
be empty (`[]`) for a paper that genuinely contains no tables and no
figures. Requiring it is the point: the author either enumerates the
exhibits or explicitly asserts there are none, so a bundle that quietly
ships no crops for a paper full of tables is not expressible.

An exhibit whose printed footnote the bundle transcribes carries `notes`
beside its caption. The crop takes in the footnote lines as printed, so
they are already in the image; `notes` carries the same words as text, so
a role reads the definitions and units a table states under itself without
reading pixels. The key is OPTIONAL — an exhibit with no footnote omits it
— and a present one must be a non-empty string, on the same terms as
`summary`. Exhibit footnotes do not appear in `text.md`, so a footnote's
words are not quotable as `<q>` evidence; the exhibit's own `<img>` label
is what cites them.

Two cross-checks bind the declaration to the directory, both hard errors:
every declared label must have a `figures/<label>.png`, and every
`figures/*.png` must be declared. The first catches a manifest promising
an image that is not there; the second catches a stray or misnamed crop,
which would otherwise become a label the extractor can cite but no human
has vouched for.

What this cannot check is whether the paper contains an exhibit nobody
cropped, or whether a crop is any good: a crop missing its header row
passes every check here. The first is asked of the model that reads the
paper (the extraction template's initial-check block), the second stays a
human or agent job.

Two entry points:
  - `validate_bundle(path)` returns EVERY problem as a list of strings
    (empty list == valid). Nothing is raised.
  - `load_bundle(path)` returns a frozen `PaperBundle`, or raises
    `BundleError` carrying the full problem list.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from meltiro.errors import BundleError


SCHEMA_VERSION = 2
# `\Z`, not `$`: in Python `$` also matches immediately before a trailing
# newline, so `^[A-Za-z0-9._-]+$` accepts "1702\n" and `re.match` does not
# change that. Both values this guards are broken by a newline they let
# through -- the id becomes a filesystem path component, and a label becomes
# both a `figures/*.png` stem and the token the extractor cites as
# `<img>label</img>`, which cannot round-trip with a newline in it.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+\Z")
# The id is used verbatim as a filesystem path component (the session dir
# lands at {out}/{study_id}/...), so an id that is all punctuation is a
# path-traversal hazard: "." and ".." resolve to the current and parent
# directory. Requiring at least one letter or digit rejects ".", "..", and
# any other alphanumeric-free id while still matching _ID_PATTERN.
_ID_ALNUM = re.compile(r"[A-Za-z0-9]")
# An exhibit label names a file under `figures/`, so it obeys the same
# filename-safe character rule the id does: letters, digits, dot, underscore,
# dash. That also keeps a separator out of a label, so `figures/<label>.png`
# can only ever resolve inside the bundle.
_LABEL_PATTERN = _ID_PATTERN

# Manifest field contract: name -> (required?, python_type, allow_empty?)
# `str` covers the JSON string type; bool is intentionally NOT a valid
# int (see _is_int) so schema_version: true is rejected. `exhibits` is a
# structured list with its own validator (see _validate_exhibits), so it
# carries the "exhibits" type marker and no emptiness flag: an empty list is
# a legitimate assertion that the paper has no tables and no figures.
_MANIFEST_FIELDS = {
    "schema_version": (True, "int", None),
    "id": (True, "str", False),
    "title": (True, "str", False),
    "exhibits": (True, "exhibits", None),
    "doi": (False, "str", True),
    # `summary` is optional, but if present it must be a non-empty string
    # (allow_empty=False): an empty summary is a mistake, not a signal.
    "summary": (False, "str", False),
}

# The key set of one `exhibits` entry: two required, one optional, and no
# other key accepted, on the same terms as the manifest's own key contract.
# `notes` carries the exhibit's printed footnote, which most exhibits do not
# have, so it is the one key an entry may leave out.
_EXHIBIT_REQUIRED_KEYS = ("label", "caption")
_EXHIBIT_OPTIONAL_KEYS = ("notes",)
_EXHIBIT_KEYS = _EXHIBIT_REQUIRED_KEYS + _EXHIBIT_OPTIONAL_KEYS


@dataclass(frozen=True)
class PaperBundle:
    """A validated, loaded paper bundle. Immutable."""

    root: Path
    study_id: str
    title: str
    doi: str | None
    summary: str | None
    text: str
    figures: dict[str, Path]  # label -> png path, sorted by label
    # label -> caption, sorted by label. Validation guarantees these are the
    # same labels `figures` carries, so the two maps stay in lockstep.
    exhibits: dict[str, str]
    # label -> printed footnote text, sorted by label, and holding an entry
    # only for an exhibit whose manifest entry declared `notes`. A subset of
    # `exhibits`' labels rather than a parallel map with nulls in it, so
    # "this exhibit has no footnote" is the absence of a key.
    exhibit_notes: dict[str, str]


def _is_int(value):
    """True for a genuine JSON integer. Rejects bool (a Python int
    subclass) so `schema_version: true` doesn't sneak through."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_bundle(path):
    """Return a list of ALL problems with the bundle at `path`.

    Empty list means the bundle is valid. This never raises for a
    malformed bundle; it collects and returns every issue so the caller
    (or the CLI) can report them all at once.
    """
    problems = []
    root = Path(path)

    if not root.exists():
        return [f"bundle directory does not exist: {root}"]
    if not root.is_dir():
        return [f"bundle path is not a directory: {root}"]

    manifest_problems, declared_labels = _validate_manifest(root)
    figure_problems, present_labels = _validate_figures(root)
    problems.extend(manifest_problems)
    problems.extend(_validate_text(root))
    problems.extend(figure_problems)
    # The two cross-checks bind the declaration to the directory, and run only
    # when both sides are themselves well formed: a malformed `exhibits` block
    # or an unusable `figures/` has already been reported, and cross-checking
    # against it would bury that report under derived noise.
    if declared_labels is not None and present_labels is not None:
        problems.extend(_cross_check_exhibits(declared_labels, present_labels))
    return problems


def _validate_manifest(root):
    """Return `(problems, declared_labels)` for `manifest.json`.

    `declared_labels` is the list of `exhibits` labels when the block is
    structurally sound, and None when it is missing or malformed (so the
    caller skips the cross-checks against `figures/`).
    """
    problems = []
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json is missing"], None
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [f"manifest.json could not be read as UTF-8: {e}"], None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return [f"manifest.json is not valid JSON: {e}"], None
    if not isinstance(data, dict):
        return [f"manifest.json must be a JSON object, got "
                f"{type(data).__name__}"], None

    # Unknown keys.
    for key in data:
        if key not in _MANIFEST_FIELDS:
            problems.append(f"manifest.json has unknown key: {key!r}")

    # Required present + type/emptiness checks.
    declared_labels = None
    for name, (required, ptype, allow_empty) in _MANIFEST_FIELDS.items():
        if name not in data:
            if required:
                problems.append(f"manifest.json is missing required key: "
                                f"{name!r}")
            continue
        value = data[name]
        if ptype == "exhibits":
            exhibit_problems, labels = _validate_exhibits(value)
            problems.extend(exhibit_problems)
            if not exhibit_problems:
                declared_labels = labels
        elif ptype == "int":
            if not _is_int(value):
                problems.append(f"manifest.json key {name!r} must be an "
                                f"integer, got {type(value).__name__}")
                continue
            if name == "schema_version" and value != SCHEMA_VERSION:
                problems.append(f"manifest.json schema_version must be "
                                f"{SCHEMA_VERSION}, got {value}")
        elif ptype == "str":
            if not isinstance(value, str):
                problems.append(f"manifest.json key {name!r} must be a "
                                f"string, got {type(value).__name__}")
                continue
            if allow_empty is False and not value.strip():
                problems.append(f"manifest.json key {name!r} must be a "
                                f"non-empty string")
            if name == "id" and value.strip():
                if not _ID_PATTERN.match(value):
                    problems.append(
                        f"manifest.json id {value!r} must match "
                        f"^[A-Za-z0-9._-]+$ (letters, digits, dot, "
                        f"underscore, dash only)")
                elif not _ID_ALNUM.search(value):
                    problems.append(
                        f"manifest.json id {value!r} must contain at least "
                        f"one letter or digit; ids like '.' or '..' are "
                        f"rejected because the id is used directly as a "
                        f"filesystem path component")
    return problems, declared_labels


def _validate_exhibits(value):
    """Validate the manifest's `exhibits` value.

    Returns `(problems, labels)`: the declared labels in declaration order,
    and every problem with the block's shape. An empty list is valid and
    yields `([], [])`, the author's explicit assertion that the paper
    contains no tables and no figures.
    """
    if not isinstance(value, list):
        return ([f"manifest.json key 'exhibits' must be a list of "
                 f"{{label, caption}} objects, got "
                 f"{type(value).__name__}"], [])
    problems = []
    labels = []
    seen = set()
    for index, entry in enumerate(value):
        where = f"manifest.json exhibits[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} must be an object with 'label' and "
                            f"'caption', got {type(entry).__name__}")
            continue
        for key in sorted(entry):
            if key not in _EXHIBIT_KEYS:
                problems.append(f"{where} has unknown key: {key!r} (an "
                                f"exhibit carries 'label' and 'caption', and "
                                f"optionally 'notes')")
        for key in _EXHIBIT_REQUIRED_KEYS:
            if key not in entry:
                problems.append(f"{where} is missing required key: {key!r}")
                continue
            if not isinstance(entry[key], str):
                problems.append(f"{where} key {key!r} must be a string, got "
                                f"{type(entry[key]).__name__}")
            elif not entry[key].strip():
                problems.append(f"{where} key {key!r} must be a non-empty "
                                f"string")
        # Optional, but a present one is held to exactly the rule the required
        # strings are held to. An exhibit with no printed footnote omits the
        # key; an empty string is a mistake, not a signal.
        for key in _EXHIBIT_OPTIONAL_KEYS:
            if key not in entry:
                continue
            if not isinstance(entry[key], str):
                problems.append(f"{where} key {key!r} must be a string when "
                                f"present, got {type(entry[key]).__name__}")
            elif not entry[key].strip():
                problems.append(f"{where} key {key!r} must be a non-empty "
                                f"string when present; an exhibit with no "
                                f"printed footnote omits the key")
        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        if not _LABEL_PATTERN.match(label):
            problems.append(
                f"{where} label {label!r} must match ^[A-Za-z0-9._-]+$ "
                f"(letters, digits, dot, underscore, dash only): it is the "
                f"stem of a figures/*.png file and the token the extractor "
                f"cites as <img>label</img>")
            continue
        if label in seen:
            problems.append(f"{where} label {label!r} is declared more than "
                            f"once; exhibit labels must be unique within a "
                            f"bundle")
            continue
        seen.add(label)
        labels.append(label)
    return problems, labels


def _cross_check_exhibits(declared_labels, present_labels):
    """Bind the manifest's declaration to the `figures/` directory. Both
    directions are hard errors; the module docstring says why.
    """
    problems = []
    declared = set(declared_labels)
    present = set(present_labels)
    for label in sorted(declared - present):
        problems.append(
            f"manifest.json declares exhibit {label!r} but there is no "
            f"figures/{label}.png")
    for label in sorted(present - declared):
        problems.append(
            f"figures/{label}.png is not declared in manifest.json "
            f"'exhibits'; every supplied image must be declared with its "
            f"caption")
    return problems


def _validate_text(root):
    text_path = root / "text.md"
    if not text_path.exists():
        return ["text.md is missing"]
    try:
        text = text_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [f"text.md could not be read as UTF-8: {e}"]
    if not text.strip():
        return ["text.md is empty"]
    return []


def _validate_figures(root):
    """Return `(problems, present_labels)` for the `figures/` directory.

    `present_labels` is the stem of every `*.png` in it, and None when the
    directory itself is unusable (so the caller skips the cross-checks). A
    missing `figures/` is not an error here: it is the no-images case, and
    the manifest's `exhibits` is what says whether that is correct.
    """
    figures_dir = root / "figures"
    if not figures_dir.exists():
        return [], []  # figures/ is optional; an empty label set
    if not figures_dir.is_dir():
        return [f"figures exists but is not a directory: {figures_dir}"], None
    problems = []
    labels = []
    for child in sorted(figures_dir.iterdir()):
        if child.name.startswith("."):
            continue  # hidden OS metadata (.DS_Store etc.) is not an asset
        if child.is_dir():
            problems.append(f"figures/ contains a subdirectory (only .png "
                            f"files allowed): {child.name}")
        elif child.suffix.lower() != ".png":
            problems.append(f"figures/ contains a non-png file (only .png "
                            f"files allowed): {child.name}")
        else:
            labels.append(child.stem)
    return problems, labels


def load_bundle(path):
    """Load and return a `PaperBundle`. Raises `BundleError` on any problem.

    Runs the full `validate_bundle` first, so the raised error lists
    every problem, not just the first one hit.
    """
    root = Path(path)
    problems = validate_bundle(root)
    if problems:
        raise BundleError(problems, path=root)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    text = (root / "text.md").read_text(encoding="utf-8")

    figures = {}
    figures_dir = root / "figures"
    if figures_dir.is_dir():
        for child in figures_dir.iterdir():
            # Same enumeration the validator ran, so what was validated and
            # what is loaded cannot diverge (a case-varying suffix included).
            if child.name.startswith(".") or child.is_dir():
                continue
            if child.suffix.lower() == ".png":
                figures[child.stem] = child
    # Re-sort by label for a stable, deterministic order.
    figures = {k: figures[k] for k in sorted(figures)}
    # Validation has already established that these are exactly the labels
    # `figures` carries, so sorting both by label keeps them in lockstep.
    declared = sorted(manifest["exhibits"], key=lambda e: e["label"])
    exhibits = {e["label"]: e["caption"] for e in declared}
    # Only the exhibits that declared a footnote, so a label missing from this
    # map is an exhibit the paper prints no notes under.
    exhibit_notes = {e["label"]: e["notes"] for e in declared if "notes" in e}

    return PaperBundle(
        root=root,
        study_id=manifest["id"],
        title=manifest["title"],
        doi=manifest.get("doi"),
        summary=manifest.get("summary"),
        text=text,
        figures=figures,
        exhibits=exhibits,
        exhibit_notes=exhibit_notes,
    )
