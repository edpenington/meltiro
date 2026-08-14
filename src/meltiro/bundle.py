"""Paper bundle: the input one run is given, and what is done with it here.

A paper bundle is a self-contained directory holding one paper: its full
text as markdown, the cropped tables and figures, and a manifest naming
them. Papers are copyrighted, so they are NEVER shipped with the code; a
run is handed one at the moment it starts.

    my-paper/
      manifest.json    (required)  identity and the exhibit declaration
      text.md          (required)  the paper's full text as markdown, UTF-8
      figures/         (optional)  cropped tables and figures, *.png only

The format belongs to *alteksto* (github.com/edpenington/alteksto), which
specifies it in `docs/bundle.md`, produces bundles to it, and enforces it.
`alteksto.bundle.validate_bundle` is the verdict `load_bundle` refuses
behind, so what this package accepts and what that specification describes
are one set by construction rather than two implementations that agree
today. Anyone assembling a bundle, from a PDF or by hand, builds to it.

What this module adds is what a valid bundle MEANS to a run:

  - `id` names the session directory (`{out}/{study_id}/...`), which is why
    the format restricts it to filename-safe characters with at least one
    alphanumeric among them.
  - `text.md` is the whole text the models are shown, and the authority
    every `<q>` quote in the evidence is checked against, verbatim and
    markdown syntax included (`quote_check.py`).
  - each `figures/<label>.png` is attached to the message under its label,
    and that label is the token a model cites as `<img>label</img>`: the
    filename stem already IS the citation.
  - an exhibit's `caption` introduces its crop in the message, so a model
    reading `[table_01]` can tell which exhibit it is looking at, and its
    `notes` (the footnote the paper prints under the exhibit, which the
    crop carries as pixels and `text.md` does not carry at all) follows the
    caption as text, so small print does not have to be read off the image.
  - `summary` is the CHECKER's identity context for study-level fields (the
    checker never reads the paper). For a published paper that is the
    abstract; for grey literature an executive summary or a couple of
    hand-written sentences. When absent, the pipeline falls back to the
    extracted `role: summary` field (see template.py); when neither exists
    it DEGRADES rather than failing — the checker is given the title and
    DOI as minimal identity context, which says which paper this is but not
    what it found, and the run records an `identity-degradation` warning
    saying so (`Orchestrator._degraded_identity_context`).

Nothing in `summary` is consulted at all on a run with no checker
(`max_checks_per_field: 0`): the identity context exists to be sent to the
checker, so a bundle with no summary and no populated summary field runs
clean under that configuration.

No check anywhere reads the pixels. A crop that clips its header row, or
catches the wrong table, is a valid bundle. Whether the paper holds an
exhibit nobody cropped is asked of the model that reads the paper (the
extraction template's initial-check block); whether a crop is any good
stays a human or agent job, on the producing side.

One entry point: `load_bundle(path)` returns a frozen `PaperBundle`, or
raises `BundleError` carrying every problem the format reported, not just
the first.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from alteksto import bundle as paper_bundle_format

from meltiro.errors import BundleError


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
    # label -> the exhibit's printed footnote, for the exhibits that print
    # one. A separate map rather than a richer `exhibits` value: an exhibit
    # without a footnote is simply absent here, so a caller reads "no
    # footnote" as a missing key rather than as a None it has to test for.
    exhibit_notes: dict[str, str]


def load_bundle(path):
    """Load and return a `PaperBundle`. Raises `BundleError` on any problem.

    The format's own validator runs first and in full, so the raised error
    lists every problem with the directory rather than the first one hit,
    and the loading below can read what validation has already established.
    """
    root = Path(path)
    problems = paper_bundle_format.validate_bundle(root)
    if problems:
        raise BundleError(problems, path=root)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    text = (root / "text.md").read_text(encoding="utf-8")

    figures = {}
    figures_dir = root / "figures"
    if figures_dir.is_dir():
        for child in figures_dir.iterdir():
            # The same enumeration the validator ran, so what was validated
            # and what is loaded cannot diverge (a case-varying suffix
            # included).
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
