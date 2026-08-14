"""What a paper bundle means here: the refusal, and the loading.

The FORMAT is *alteksto*'s. It specifies what a bundle is, and
`alteksto.bundle.validate_bundle` decides whether a directory is one; the
rules themselves — every manifest key, the id's character class, the
cross-checks against `figures/` — are tested there, against that
specification. Restating them here would build the second implementation
this package exists not to have.

What is tested here is the seam and the loader: that `load_bundle` refuses
exactly what the format refuses and says everything the format said, and
that a bundle it accepts arrives as a `PaperBundle` whose maps carry what
the run puts in front of a model.
"""

import dataclasses
import json
import shutil

import pytest
from alteksto.bundle import validate_bundle

from meltiro.bundle import PaperBundle, load_bundle
from meltiro.errors import BundleError


@pytest.fixture
def good_bundle(tmp_path, bundle_minimal_dir):
    """A writable copy of the synthetic fixture bundle."""
    dst = tmp_path / "bundle"
    shutil.copytree(bundle_minimal_dir, dst)
    return dst


def _write_manifest(bundle_dir, manifest):
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")


def _base_manifest():
    """A valid manifest for `good_bundle`, whose figures/ holds table_01.png.

    `exhibits` is required and cross-checked against `figures/`, so the base
    manifest declares the one image the fixture carries; a test that changes
    the figure set changes this too. No `doi`, `summary` or exhibit `notes`:
    the optional keys are added by the tests that are about them.
    """
    return {
        "schema_version": 2,
        "id": "demo-001",
        "title": "A title",
        "exhibits": [
            {"label": "table_01", "caption": "Table 1. Some results"},
        ],
    }


class TestTheVerdictIsTheFormats:
    """`load_bundle` adds no rule of its own and drops none of the report."""

    def test_the_fixture_bundles_are_bundles(self, bundle_minimal_dir,
                                             bundle_tables_dir,
                                             bundle_unicode_dir):
        # Asked of the format directly. Every other test in the suite runs
        # over these three, so a fixture that drifted out of the specification
        # would fail here rather than somewhere downstream.
        for fixture in (bundle_minimal_dir, bundle_tables_dir,
                        bundle_unicode_dir):
            assert validate_bundle(fixture) == []

    def test_a_refusal_carries_the_whole_report(self, good_bundle):
        # Two faults, so a loader reporting only the first would be visible.
        (good_bundle / "text.md").unlink()
        m = _base_manifest()
        m["bogus"] = 1
        _write_manifest(good_bundle, m)
        with pytest.raises(BundleError) as excinfo:
            load_bundle(good_bundle)
        # Not "at least two problems": the same list, in the same order, so
        # nothing is summarised, reordered or quietly dropped on the way out.
        assert excinfo.value.problems == validate_bundle(good_bundle)
        assert len(excinfo.value.problems) >= 2

    def test_the_schema_version_is_the_formats_to_declare(self, good_bundle):
        # The version this package accepts is not written down here. A bundle
        # built to the superseded version is refused, and the words are the
        # format's.
        m = _base_manifest()
        m["schema_version"] = 1
        _write_manifest(good_bundle, m)
        with pytest.raises(BundleError) as excinfo:
            load_bundle(good_bundle)
        assert any("schema_version" in p for p in excinfo.value.problems)

    def test_a_missing_directory_is_refused_before_anything_is_read(
            self, tmp_path):
        with pytest.raises(BundleError):
            load_bundle(tmp_path / "nope")


class TestLoadBundle:
    def test_happy_path(self, bundle_minimal_dir):
        b = load_bundle(bundle_minimal_dir)
        assert isinstance(b, PaperBundle)
        assert b.study_id == "demo-001"
        assert b.title.startswith("A synthetic study")
        assert b.doi == "10.0000/demo.0001"
        assert b.summary and "CRT-HD" in b.summary
        assert "CRT-HD" in b.text
        assert set(b.figures) == {"table_01"}
        assert b.figures["table_01"].name == "table_01.png"
        assert set(b.exhibits) == {"table_01"}
        assert b.exhibits["table_01"].startswith("Table 1.")
        assert b.exhibit_notes["table_01"].startswith("CI, confidence")

    def test_frozen(self, bundle_minimal_dir):
        b = load_bundle(bundle_minimal_dir)
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.study_id = "other"

    def test_optional_fields_default_none(self, good_bundle):
        _write_manifest(good_bundle, _base_manifest())  # no doi/summary
        b = load_bundle(good_bundle)
        assert b.doi is None
        assert b.summary is None

    def test_an_exhibit_with_no_footnote_is_absent_from_the_notes(
            self, good_bundle):
        # Absence, not an empty string: an exhibit the paper printed no
        # footnote under has no key, so nothing renders under its label.
        _write_manifest(good_bundle, _base_manifest())
        b = load_bundle(good_bundle)
        assert b.exhibit_notes == {}
        assert set(b.exhibits) == {"table_01"}

    def test_the_three_exhibit_maps_stay_in_lockstep(self, good_bundle):
        # A second crop, declared with a footnote of its own, out of label
        # order in the manifest: the loader sorts, so a bundle's maps enumerate
        # the same way whatever order the manifest was written in.
        src = good_bundle / "figures" / "table_01.png"
        (good_bundle / "figures" / "figure_02.png").write_bytes(
            src.read_bytes())
        m = _base_manifest()
        m["exhibits"].insert(0, {"label": "figure_02",
                                 "caption": "Figure 2. Study flow",
                                 "notes": "Units withdrawn before the first "
                                          "round are not shown."})
        _write_manifest(good_bundle, m)
        b = load_bundle(good_bundle)
        assert list(b.exhibits) == ["figure_02", "table_01"]
        assert list(b.figures) == ["figure_02", "table_01"]
        assert set(b.exhibit_notes) == {"figure_02"}
        assert b.exhibits["figure_02"] == "Figure 2. Study flow"
        assert b.exhibit_notes["figure_02"].startswith("Units withdrawn")

    def test_no_figures_dir(self, good_bundle):
        # No figures/ at all, and an `exhibits` list that says so.
        shutil.rmtree(good_bundle / "figures")
        m = _base_manifest()
        m["exhibits"] = []
        _write_manifest(good_bundle, m)
        b = load_bundle(good_bundle)
        assert b.figures == {}
        assert b.exhibits == {}
        assert b.exhibit_notes == {}

    def test_hidden_os_files_in_figures_are_not_loaded_as_crops(
            self, good_bundle):
        # macOS drops .DS_Store into any browsed directory. The format skips
        # it, so the loader must skip it too: enumerating one file more than
        # was validated would put an undeclared label in front of a model.
        (good_bundle / "figures" / ".DS_Store").write_bytes(b"\x00\x01")
        b = load_bundle(good_bundle)
        assert set(b.figures) == {"table_01"}
