"""Paper bundle: the input one run is given, and what is done with it here.

A paper bundle is a self-contained directory holding one paper: its full
text as markdown, the cropped tables and figures, and a manifest naming
them. Papers are copyrighted, so they are NEVER shipped with the code; a
run is handed one at the moment it starts.

    my-paper/
      manifest.json    identity and the exhibit declaration
      text.md          the paper's full text as markdown
      figures/         cropped tables and figures

The format belongs to *alteksto* (github.com/edpenington/alteksto), which
specifies it in `docs/bundle.md`, produces bundles to it, and enforces it —
which of those three entries is required, what may sit in each, and what
the manifest holds are all read there. `alteksto.bundle.validate_bundle` is
the verdict `load_bundle` refuses behind, so what this package accepts is
that specification by construction. Anyone assembling a bundle, from a PDF
or by hand, builds to it.

What this module adds is what a valid bundle MEANS to a run:

  - `id` names the session directory (`{out}/{study_id}/...`), which is what
    the format's constraint on it is for: it is a path component here.
  - `text.md` is the whole text the models are shown, and the authority
    every `<q>` quote in the evidence is checked against, verbatim and
    markdown syntax included (`quote_check.py`).
  - each `figures/<label>.png` is attached to the message under its label,
    and that label is the token a model cites as `<img>label</img>`: the
    filename stem already IS the citation.
  - each `tables/<label>.html` is the content of that exhibit as text, and it
    rides in the message beside the crop under the same label. It is carried
    verbatim: the markup a bundle passes validation with is the markup a
    model reads, so the bytes shown are the bytes the producing route checked
    against the page. Rendering it to a pipe table would flatten exactly what
    the format chose HTML to keep, which is that a printed header can span
    columns and a stub can span rows. It does not change what a citation is —
    a fact taken from an exhibit is still cited `<img>label</img>`, because
    the crop is still what the exhibit IS and the transcription is a reading
    of it.
  - an exhibit's `caption` introduces its crop in the message, so a model
    reading `[table_01]` can tell which exhibit it is looking at, and its
    `notes` (the footnote the paper prints under the exhibit, which the crop
    normally carries as pixels and `text.md` does not carry at all) follows
    the caption as text, so small print does not have to be read off the
    image.
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
from dataclasses import dataclass, field
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
    # The empty default carries that reading one step further: a bundle whose
    # exhibits print nothing constructs exactly as it always did.
    exhibit_notes: dict[str, str] = field(default_factory=dict)
    # label -> the path of that exhibit's table transcription, for the
    # exhibits that carry one. Absence is the format's only signal here and a
    # strong one, so it is carried the way the footnote is: a missing key
    # means the crop is the content, which is what every exhibit meant before
    # the directory existed. A path rather than the markup, so this stays a
    # description of the directory and the one read of each file happens
    # where the message is built.
    tables: dict[str, Path] = field(default_factory=dict)


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

    # Which files under `figures/` are exhibits, and what each is called, is
    # the format's answer and not a reading of the directory taken here: a
    # second reading could disagree with the one validation ran and put a
    # label in front of a model that no check ever saw.
    figures = paper_bundle_format.figure_files(root)
    # And which are transcriptions, on the same terms and for the same
    # reason. Unlike `figures/`, a label absent here is not a defect: a
    # bundle may transcribe all, some or none of its exhibits.
    tables = paper_bundle_format.table_files(root)
    # Validation has already established that these are exactly the labels
    # the manifest declares, so sorting both by label keeps them in lockstep.
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
        tables=tables,
    )
