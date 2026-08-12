"""The paper's own fingerprint: which input a run was given.

Every other fingerprint in this repo describes the QUESTION — the prompts, the
template, the models, the engine. `bundle_fp` describes what the question was
asked of, and it is folded into none of them. That separation is the point:
one config over a hundred papers records one `instrument_fp` and a hundred
`bundle_fp` values, so a consumer can group by either without the two
interfering.

Three parts sit under it, one per thing a bundle is made of, so a moved
`bundle_fp` says WHICH half of the paper changed rather than only that
something did: `text_fp` over `text.md`'s bytes, `figures_fp` over the cropped
images, `manifest_fp` over the manifest's content.
"""

import json
import shutil

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import bundle_fingerprint
from meltiro.orchestrator import Orchestrator
from meltiro.run_entry import build_entry


PARTS = ("text_fp", "figures_fp", "manifest_fp")


def _fp(bundle_dir):
    return bundle_fingerprint(load_bundle(bundle_dir))


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


def _edit_manifest(bundle_dir, **changes):
    path = bundle_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_dir


def _orch(config_dir, bundle_dir, out_dir):
    """A prepared dry-run orchestrator over one paper bundle."""
    config = load_config_bundle(config_dir)
    loop = config.pipeline
    orch = Orchestrator(
        config, load_bundle(bundle_dir), out_dir,
        extractor_model=loop["extractor_model"],
        checker_config=CheckerConfig(max_tokens=1024, 
            checker_model=loop["checker_model"], api_key="x"),
        review_model=loop["review_model"],
        extractor_max_tokens=4096,
        review_max_tokens=4096,
        api_key="x",
        dry_run=True,
    )
    orch.prepare_new_session()
    return orch


# ---------------------------------------------------------------------------
# Shape and determinism
# ---------------------------------------------------------------------------

class TestShape:
    def test_it_returns_the_four_values_self_prefixed(self, bundle_minimal_dir):
        fp = _fp(bundle_minimal_dir)
        assert set(fp) == {"text_fp", "figures_fp", "manifest_fp", "bundle_fp"}
        for name, value in fp.items():
            prefix, _, digest = value.partition(":")
            assert prefix == name
            # Full, untruncated SHA-256 hex, matching every other fingerprint
            # in the module.
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")

    def test_it_is_deterministic(self, bundle_minimal_dir):
        # Recomputed from the same bytes, it is the same number: a reader
        # holding a published `bundle_fp` and a copy of the paper can check
        # that they have the paper the run read.
        assert _fp(bundle_minimal_dir) == _fp(bundle_minimal_dir)

    def test_a_copied_bundle_fingerprints_identically(
            self, bundle_minimal_dir, tmp_path):
        # Content, not location: nothing about where the bundle sits reaches
        # the digest, so a paper archived elsewhere still verifies.
        assert _fp(_copy(bundle_minimal_dir, tmp_path / "elsewhere")) == \
            _fp(bundle_minimal_dir)

    def test_two_different_papers_differ(
            self, bundle_minimal_dir, bundle_tables_dir):
        assert _fp(bundle_minimal_dir)["bundle_fp"] != \
            _fp(bundle_tables_dir)["bundle_fp"]


# ---------------------------------------------------------------------------
# Each part moves only when its own input moves
# ---------------------------------------------------------------------------

class TestEachPartMovesAlone:
    """The reason there are three parts and not one.

    A `bundle_fp` that moved would otherwise leave a reader to diff the whole
    paper to find out what happened. Each part answers for exactly one half of
    the bundle, so exactly one of them moves and names the change."""

    def _assert_only(self, before, after, moved):
        assert after["bundle_fp"] != before["bundle_fp"], (
            "the paper changed and bundle_fp did not, so a run's recorded "
            "input no longer identifies the input it was given.")
        for part in PARTS:
            if part == moved:
                assert after[part] != before[part], part
            else:
                assert after[part] == before[part], part

    def test_editing_the_text_moves_only_text_fp(
            self, bundle_minimal_dir, tmp_path):
        paper = _copy(bundle_minimal_dir, tmp_path / "edited")
        before = _fp(paper)
        text = paper / "text.md"
        text.write_text(
            text.read_text(encoding="utf-8") + "x", encoding="utf-8")
        self._assert_only(before, _fp(paper), "text_fp")

    def test_swapping_a_figure_moves_only_figures_fp(
            self, bundle_minimal_dir, tmp_path):
        # A re-crop: same label, same manifest, different pixels. This is the
        # change a text-level hash cannot see at all, and it changes what the
        # model was shown.
        paper = _copy(bundle_minimal_dir, tmp_path / "recropped")
        before = _fp(paper)
        crop = sorted((paper / "figures").glob("*.png"))[0]
        crop.write_bytes(crop.read_bytes() + b"\x00")
        self._assert_only(before, _fp(paper), "figures_fp")

    def test_editing_the_manifest_moves_only_manifest_fp(
            self, bundle_minimal_dir, tmp_path):
        # The manifest's `summary` is the checker's identity context for
        # study-level fields, so an edit here changes what the run was given
        # even though not a byte of text or image moved.
        paper = _copy(bundle_minimal_dir, tmp_path / "resummarised")
        before = _fp(paper)
        _edit_manifest(paper, summary="A different summary entirely.")
        self._assert_only(before, _fp(paper), "manifest_fp")

    def test_renaming_a_figure_moves_figures_fp_and_manifest_fp(
            self, bundle_minimal_dir, tmp_path):
        # A label is both a figure's identity and a manifest declaration, and
        # it is the token the extractor cites, so a rename is a real change to
        # both halves and both parts say so.
        paper = _copy(bundle_minimal_dir, tmp_path / "relabelled")
        before = _fp(paper)
        crop = sorted((paper / "figures").glob("*.png"))[0]
        crop.rename(crop.with_name("renamed_01.png"))
        _edit_manifest(paper, exhibits=[
            {"label": "renamed_01", "caption": "A table."}])
        after = _fp(paper)
        assert after["figures_fp"] != before["figures_fp"]
        assert after["manifest_fp"] != before["manifest_fp"]
        assert after["text_fp"] == before["text_fp"]
        assert after["bundle_fp"] != before["bundle_fp"]

    def test_reformatting_the_manifest_moves_nothing(
            self, bundle_minimal_dir, tmp_path):
        # `manifest_fp` is taken over the manifest's canonical JSON, so its
        # CONTENT is what is hashed. Reindenting a file or writing its keys in
        # another order changes nothing about the run and must move nothing.
        paper = _copy(bundle_minimal_dir, tmp_path / "reformatted")
        before = _fp(paper)
        path = paper / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(dict(reversed(list(manifest.items()))), indent=8),
            encoding="utf-8")
        assert _fp(paper) == before


# ---------------------------------------------------------------------------
# No figures is a fact, not an absence
# ---------------------------------------------------------------------------

class TestNoFigures:
    """A paper with no crops folds in the module's absence sentinel.

    A digest over an empty payload would be a fixed constant standing equally
    for "this paper supplies no figures" and "nobody hashed any", and those are
    different claims. The sentinel makes the first one a hashed fact."""

    def test_a_bundle_with_no_figures_still_fingerprints(
            self, bundle_unicode_dir):
        fp = _fp(bundle_unicode_dir)
        assert load_bundle(bundle_unicode_dir).figures == {}
        assert len(fp["figures_fp"].partition(":")[2]) == 64

    def test_no_figures_differs_from_one_figure(
            self, bundle_minimal_dir, tmp_path):
        # Same text, same manifest apart from the exhibit declaration the
        # figure set requires: dropping the only crop is a different paper and
        # figures_fp has to say so.
        paper = _copy(bundle_minimal_dir, tmp_path / "stripped")
        with_figure = _fp(paper)
        for crop in (paper / "figures").glob("*.png"):
            crop.unlink()
        (paper / "figures").rmdir()
        _edit_manifest(paper, exhibits=[])
        without = _fp(paper)
        assert without["figures_fp"] != with_figure["figures_fp"]
        assert without["bundle_fp"] != with_figure["bundle_fp"]

    def test_every_figureless_paper_does_not_share_a_bundle_fp(
            self, bundle_unicode_dir, tmp_path):
        # The sentinel is shared by every figureless bundle, which is correct:
        # they agree about their figures. They must still differ overall, on
        # the two parts that describe them.
        other = _copy(bundle_unicode_dir, tmp_path / "other")
        text = other / "text.md"
        text.write_text(
            text.read_text(encoding="utf-8") + "\n\nMore.\n", encoding="utf-8")
        assert _fp(other)["figures_fp"] == _fp(bundle_unicode_dir)["figures_fp"]
        assert _fp(other)["bundle_fp"] != _fp(bundle_unicode_dir)["bundle_fp"]


# ---------------------------------------------------------------------------
# Recorded with the run
# ---------------------------------------------------------------------------

class TestRecordedWithTheRun:
    def test_run_json_carries_all_four(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        expected = _fp(bundle_minimal_dir)
        assert {k: orch.session.meta[k] for k in expected} == expected
        # Persisted, not just in-memory: a consumer reads the file.
        with open(orch.session.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert {k: meta[k] for k in expected} == expected

    def test_a_figureless_paper_records_them_too(
            self, config_dir, bundle_unicode_dir, tmp_path):
        # The capture is unconditional, unlike the per-image hashes beside it:
        # a paper with no crops is still a paper the run must name.
        orch = _orch(config_dir, bundle_unicode_dir, tmp_path / "runs")
        assert orch.session.meta["bundle_fp"] == \
            _fp(bundle_unicode_dir)["bundle_fp"]

    def test_the_run_log_row_carries_all_four(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The run log is the cross-run index a consumer sweeps, so the paper's
        # identity has to be a column there and not only a file in a session
        # directory.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        entry = build_entry(orch.session)
        expected = _fp(bundle_minimal_dir)
        assert {k: entry[k] for k in expected} == expected

    def test_the_per_image_hashes_agree_with_figures_fp(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # One recipe behind both records: the per-image hashes the transcript
        # reads for re-crop drift are the same digests figures_fp is built
        # from, so the two can never disagree about the same file.
        from meltiro.fingerprint import figure_hashes

        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        bundle = load_bundle(bundle_minimal_dir)
        assert orch.session.meta["image_hashes"] == \
            figure_hashes(bundle.figures.values())


# ---------------------------------------------------------------------------
# Folded into nothing
# ---------------------------------------------------------------------------

class TestTheBundleIsFoldedIntoNoOtherFingerprint:
    """The property this whole design rests on.

    `bundle_fp` is recorded BESIDE the run's fingerprints, never inside one. If
    it ever leaked into `config_fp`, `instrument_fp` or `run_fp`, every one of
    those would move per paper: a config could no longer be compared across
    papers, `instrument_fp` would stop grouping runs of one instrument, and a
    published `run_fp` would name a single extraction rather than a
    configuration."""

    @pytest.fixture
    def two_papers(self, bundle_minimal_dir, tmp_path):
        """The same bundle twice, the second with different text, a different
        figure set and a different manifest — every part of a paper changed at
        once, so nothing is left for a leak to hide behind."""
        first = _copy(bundle_minimal_dir, tmp_path / "first")
        second = _copy(bundle_minimal_dir, tmp_path / "second")
        text = second / "text.md"
        text.write_text(
            text.read_text(encoding="utf-8") + "\n\nAnother section.\n",
            encoding="utf-8")
        crop = sorted((second / "figures").glob("*.png"))[0]
        crop.write_bytes(crop.read_bytes() + b"\x00")
        _edit_manifest(second, title="A different title", summary="Different.")
        return first, second

    def test_the_run_axes_hold_across_papers(
            self, config_dir, two_papers, tmp_path):
        first, second = two_papers
        a = _orch(config_dir, first, tmp_path / "a").session.meta
        b = _orch(config_dir, second, tmp_path / "b").session.meta
        for axis in ("config_fp", "checker_fp", "review_fp", "instrument_fp",
                     "extractor_call_fp", "checker_call_fp", "review_call_fp",
                     "engine_fp", "run_fp"):
            assert a[axis] == b[axis], (
                f"{axis} moved with the paper. The paper is the run's INPUT, "
                f"so a fingerprint that moves with it stops being a statement "
                f"about the question and can no longer group runs of one "
                f"config across the papers it was run on.")

    def test_the_bundle_fingerprint_does_move(
            self, config_dir, two_papers, tmp_path):
        # The other half of the same claim: the axes holding is only useful
        # because something recorded beside them says which paper it was.
        first, second = two_papers
        a = _orch(config_dir, first, tmp_path / "a").session.meta
        b = _orch(config_dir, second, tmp_path / "b").session.meta
        for part in PARTS + ("bundle_fp",):
            assert a[part] != b[part], part
