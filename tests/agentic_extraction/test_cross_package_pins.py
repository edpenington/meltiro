"""Two invariants nothing else would notice breaking.

The first is a SEAM: meltiro hashes a figure's bytes to record what the paper
supplied, and direktoro hashes the same bytes to stub them out of the wire log.
The transcript renderer compares the two digests to report a figure re-cropped
after the run, and that comparison is only meaningful while both sides hash the
same bytes the same way. Neither package's tests can see the other's recipe, so
the equality is asserted here, where both are importable — this repo owns the
seam because it is the one that reads across it.

The second is an ORDERING. A paper bundle's figures are sorted by label at load
and hashed as sorted `(label, sha256)` pairs, and both sorts are load-bearing:
`figures_fp` is a published fingerprint, so a filesystem that enumerated
directory entries in a different order would give one paper two fingerprints.
`sorted()` is exactly the kind of call a refactor drops as redundant, because
on a small tmp_path it usually IS — so these tests feed OUT-OF-ORDER input,
which is the only shape that tells the two apart.
"""

import hashlib
import json
import shutil

import pytest

from direktoro import redact_messages
from meltiro.bundle import load_bundle
from meltiro.fingerprint import bundle_fingerprint, figure_hashes


# A real PNG, small enough to inline. Any bytes would do for the digest
# comparison; a plausible file keeps the fixture honest about what is hashed.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d76360606000000004000180fd7f2c0000000049454e"
    "44ae426082")


# ---------------------------------------------------------------------------
# One recipe, two packages
# ---------------------------------------------------------------------------

class TestTheFigureDigestSeamHolds:
    """meltiro's `figure_hashes` and direktoro's `image_ref` redaction must
    produce the SAME digest for the same bytes.

    `Session.capture_image_hashes` records the first at session start; the wire
    log carries the second on every call that sent the image. The transcript
    renderer flags drift by comparing them, so two recipes that diverged would
    make every figure look re-cropped — or, worse, agree by accident today and
    stop agreeing on an upgrade nobody connected to this."""

    def _direktoro_digest(self, raw):
        """The sha256 direktoro puts in an `image_ref`, through its own path.

        Built by handing it a message carrying the image exactly as an adapter
        would, and reading the stub back out: asserting against a
        reimplementation of the recipe would pin this test's arithmetic rather
        than direktoro's.
        """
        import base64

        redacted = redact_messages([{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(raw).decode("ascii"),
                },
            }],
        }])
        stub, = redacted[0]["content"]
        assert stub["type"] == "image_ref"
        return stub

    def test_the_two_digests_of_one_figure_agree(self, tmp_path):
        path = tmp_path / "table_01.png"
        path.write_bytes(_PNG)

        mine = figure_hashes([path])["table_01"]
        theirs = self._direktoro_digest(_PNG)

        assert mine["sha256"] == theirs["sha256"]
        assert mine["byte_length"] == theirs["byte_length"]

    def test_they_agree_on_a_different_figure_too(self, tmp_path):
        # A single fixture could agree by both being wrong in the same way (a
        # constant, a truncation). A second, different payload rules that out.
        other = _PNG + b"trailing bytes that change the digest"
        path = tmp_path / "figure_09.png"
        path.write_bytes(other)

        assert figure_hashes([path])["figure_09"]["sha256"] == \
            self._direktoro_digest(other)["sha256"]

    def test_the_recipe_is_a_plain_sha256_of_the_file_bytes(self, tmp_path):
        # And what BOTH of them are: the digest of the bytes on disk, with no
        # framing, prefix or re-encoding in between. That is what makes the
        # equality above reproducible by a third party holding the PNG.
        path = tmp_path / "table_02.png"
        path.write_bytes(_PNG)
        assert figure_hashes([path])["table_02"]["sha256"] == \
            hashlib.sha256(_PNG).hexdigest()


# ---------------------------------------------------------------------------
# Figures are sorted by label, at both sites
# ---------------------------------------------------------------------------

@pytest.fixture
def out_of_order_bundle(tmp_path, bundle_minimal_dir):
    """A paper bundle whose figure labels sort into a different order from the
    one the manifest lists them in.

    The labels are chosen so alphabetical order (`fig_a`, `fig_b`, `fig_c`)
    reverses the manifest's, which is the only arrangement that can tell a
    sorted walk from an unsorted one.
    """
    dst = tmp_path / "paper"
    shutil.copytree(bundle_minimal_dir, dst)
    figures = dst / "figures"
    figures.mkdir(exist_ok=True)
    for existing in figures.glob("*.png"):
        existing.unlink()

    labels = ["fig_c", "fig_b", "fig_a"]
    for i, label in enumerate(labels):
        # Distinct bytes per figure, so a swapped pair changes the digest
        # payload rather than producing the same pairs in a different order.
        (figures / f"{label}.png").write_bytes(_PNG + bytes([i]))

    manifest_path = dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exhibits"] = [
        {"label": label, "caption": f"Caption for {label}."}
        for label in labels
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dst, labels


class TestFiguresAreOrderedByLabel:

    def test_the_loaded_bundle_orders_figures_by_label(
            self, out_of_order_bundle):
        # `bundle.py`'s sort. Dicts preserve insertion order, so this order IS
        # what every consumer iterating `bundle.figures` sees — the prompt
        # builder's image blocks among them.
        path, labels = out_of_order_bundle
        bundle = load_bundle(path)
        assert list(bundle.figures) == sorted(labels)
        assert list(bundle.figures) != labels, (
            "the fixture must present the labels out of order, or this test "
            "passes with the sort removed")

    def test_the_captions_stay_in_lockstep_with_the_figures(
            self, out_of_order_bundle):
        # Both are sorted by label so a caption cannot be attached to the
        # wrong image.
        path, labels = out_of_order_bundle
        bundle = load_bundle(path)
        assert list(bundle.exhibits) == list(bundle.figures)
        for label, caption in bundle.exhibits.items():
            assert label in caption

    def test_figures_fp_does_not_depend_on_enumeration_order(
            self, out_of_order_bundle):
        # `fingerprint.py`'s sort, which is the SECOND one: it sorts the
        # (label, sha256) pairs itself rather than trusting the order it was
        # handed. Feeding it a deliberately shuffled mapping is what separates
        # the two — with this sort removed, the fingerprint would move.
        path, labels = out_of_order_bundle
        bundle = load_bundle(path)
        as_loaded = bundle_fingerprint(bundle)["figures_fp"]

        shuffled = type(bundle)(
            **{**{f.name: getattr(bundle, f.name)
                  for f in bundle.__dataclass_fields__.values()},
               "figures": {label: bundle.figures[label]
                           for label in reversed(list(bundle.figures))}})
        assert list(shuffled.figures) != list(bundle.figures)
        assert bundle_fingerprint(shuffled)["figures_fp"] == as_loaded

    def test_the_fingerprint_still_moves_when_a_figure_moves(
            self, out_of_order_bundle):
        # The pair to the test above: order-independence must not be
        # insensitivity. Re-cropping one figure moves `figures_fp`.
        path, labels = out_of_order_bundle
        before = bundle_fingerprint(load_bundle(path))["figures_fp"]
        (path / "figures" / "fig_b.png").write_bytes(_PNG + b"recropped")
        assert bundle_fingerprint(load_bundle(path))["figures_fp"] != before
