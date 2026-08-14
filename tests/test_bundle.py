"""Tests for the paper bundle contract (meltiro.bundle)."""

import dataclasses
import json
import shutil

import pytest

from meltiro.bundle import PaperBundle, load_bundle, validate_bundle
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
    the figure set changes this too.
    """
    return {
        "schema_version": 2,
        "id": "demo-001",
        "title": "A title",
        "exhibits": [
            {"label": "table_01", "caption": "Table 1. Some results"},
        ],
    }


class TestValidateHappyPath:
    def test_fixture_bundle_is_valid(self, bundle_minimal_dir):
        assert validate_bundle(bundle_minimal_dir) == []

    def test_good_copy_is_valid(self, good_bundle):
        assert validate_bundle(good_bundle) == []


class TestValidateFailureModes:
    def test_missing_directory(self, tmp_path):
        problems = validate_bundle(tmp_path / "nope")
        assert len(problems) == 1
        assert "does not exist" in problems[0]

    def test_hidden_os_files_in_figures_are_ignored(self, good_bundle):
        # macOS drops .DS_Store into any browsed directory; hidden
        # metadata files must not fail validation or become figures.
        (good_bundle / "figures" / ".DS_Store").write_bytes(b"\x00\x01")
        assert validate_bundle(good_bundle) == []
        assert ".DS_Store" not in {p.name for p in
                                   load_bundle(good_bundle).figures.values()}

    def test_missing_manifest(self, good_bundle):
        (good_bundle / "manifest.json").unlink()
        problems = validate_bundle(good_bundle)
        assert any("manifest.json is missing" in p for p in problems)

    def test_manifest_not_json(self, good_bundle):
        (good_bundle / "manifest.json").write_text("{not json",
                                                   encoding="utf-8")
        problems = validate_bundle(good_bundle)
        assert any("not valid JSON" in p for p in problems)

    def test_unknown_key(self, good_bundle):
        m = _base_manifest()
        m["extra"] = "nope"
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("unknown key" in p and "extra" in p for p in problems)

    def test_missing_title(self, good_bundle):
        m = _base_manifest()
        del m["title"]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("missing required key" in p and "title" in p
                   for p in problems)

    def test_empty_title(self, good_bundle):
        m = _base_manifest()
        m["title"] = "   "
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("non-empty" in p and "title" in p for p in problems)

    def test_missing_id(self, good_bundle):
        m = _base_manifest()
        del m["id"]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("missing required key" in p and "id" in p
                   for p in problems)

    def test_bad_id_pattern(self, good_bundle):
        m = _base_manifest()
        m["id"] = "demo 001/../x"  # spaces + slashes not allowed
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("^[A-Za-z0-9._-]+$" in p for p in problems)

    def test_id_dotdot_rejected(self, good_bundle):
        # ".." matches the char-class pattern but is a path-traversal hazard:
        # the id is used verbatim as a session-dir path component, so ".."
        # would place the session one level above --out.
        m = _base_manifest()
        m["id"] = ".."
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("at least one letter or digit" in p for p in problems)

    def test_id_dot_rejected(self, good_bundle):
        m = _base_manifest()
        m["id"] = "."
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("at least one letter or digit" in p for p in problems)

    def test_id_all_punctuation_rejected(self, good_bundle):
        # No traversal, but still no alphanumeric character: rejected.
        m = _base_manifest()
        m["id"] = "._-"
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("at least one letter or digit" in p for p in problems)

    def test_id_with_dots_but_alnum_is_valid(self, good_bundle):
        # Dots are still allowed as long as the id has an alphanumeric char.
        m = _base_manifest()
        m["id"] = "demo.001"
        _write_manifest(good_bundle, m)
        assert validate_bundle(good_bundle) == []

    def test_wrong_schema_version_value(self, good_bundle):
        # Forward-only: version 1, the shape before the exhibit `notes` key,
        # is not read as a valid bundle either.
        m = _base_manifest()
        m["schema_version"] = 1
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("schema_version must be 2" in p for p in problems)

    def test_wrong_schema_version_type(self, good_bundle):
        m = _base_manifest()
        m["schema_version"] = "2"  # string, not int
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("must be an integer" in p for p in problems)

    def test_schema_version_bool_rejected(self, good_bundle):
        m = _base_manifest()
        m["schema_version"] = True  # bool is not a valid int here
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("must be an integer" in p for p in problems)

    def test_wrong_doi_type(self, good_bundle):
        m = _base_manifest()
        m["doi"] = 12345
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("doi" in p and "must be a string" in p for p in problems)

    def test_summary_is_optional(self, good_bundle):
        # A manifest with no summary key is valid (summary is optional).
        _write_manifest(good_bundle, _base_manifest())
        assert validate_bundle(good_bundle) == []

    def test_wrong_summary_type(self, good_bundle):
        m = _base_manifest()
        m["summary"] = ["not", "a", "string"]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("summary" in p and "must be a string" in p
                   for p in problems)

    def test_empty_summary_rejected(self, good_bundle):
        # summary is optional but, when present, must be non-empty.
        m = _base_manifest()
        m["summary"] = "   "
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("summary" in p and "non-empty" in p for p in problems)

    def test_abstract_key_now_unknown(self, good_bundle):
        # The manifest schema carries `summary`, not `abstract`; the latter is
        # rejected as unknown (forward-only, no back-compat).
        m = _base_manifest()
        m["abstract"] = "some text"
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("unknown key" in p and "abstract" in p for p in problems)

    def test_missing_text(self, good_bundle):
        (good_bundle / "text.md").unlink()
        problems = validate_bundle(good_bundle)
        assert any("text.md is missing" in p for p in problems)

    def test_empty_text(self, good_bundle):
        (good_bundle / "text.md").write_text("   \n\n", encoding="utf-8")
        problems = validate_bundle(good_bundle)
        assert any("text.md is empty" in p for p in problems)

    def test_non_png_in_figures(self, good_bundle):
        (good_bundle / "figures" / "notes.txt").write_text("x",
                                                           encoding="utf-8")
        problems = validate_bundle(good_bundle)
        assert any("non-png" in p and "notes.txt" in p for p in problems)

    def test_subdirectory_in_figures(self, good_bundle):
        (good_bundle / "figures" / "nested").mkdir()
        problems = validate_bundle(good_bundle)
        assert any("subdirectory" in p for p in problems)

    def test_collects_multiple_problems(self, good_bundle):
        # Break several things at once; validate returns ALL of them.
        m = _base_manifest()
        del m["title"]
        m["bogus"] = 1
        _write_manifest(good_bundle, m)
        (good_bundle / "text.md").write_text("", encoding="utf-8")
        problems = validate_bundle(good_bundle)
        assert len(problems) >= 3


class TestExhibits:
    """`exhibits` is a required declaration of what the bundle supplies as
    cropped images, cross-checked against `figures/` in both directions."""

    def _png(self, bundle_dir, label):
        """Add another PNG to figures/, copied from the fixture's own."""
        src = bundle_dir / "figures" / "table_01.png"
        (bundle_dir / "figures" / f"{label}.png").write_bytes(src.read_bytes())

    def test_declared_and_present_is_valid(self, good_bundle):
        self._png(good_bundle, "figure_02")
        m = _base_manifest()
        m["exhibits"].append(
            {"label": "figure_02", "caption": "Figure 2. Study flow"})
        _write_manifest(good_bundle, m)
        assert validate_bundle(good_bundle) == []

    def test_missing_key_is_an_error(self, good_bundle):
        # Forward-only: silence is what the key exists to prevent, so a
        # manifest without it fails rather than defaulting to empty.
        m = _base_manifest()
        del m["exhibits"]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("missing required key" in p and "exhibits" in p
                   for p in problems)

    def test_declared_label_with_no_png(self, good_bundle):
        m = _base_manifest()
        m["exhibits"].append(
            {"label": "table_02", "caption": "Table 2. Not cropped"})
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("declares exhibit 'table_02'" in p
                   and "no figures/table_02.png" in p for p in problems)

    def test_png_nobody_declared(self, good_bundle):
        # A stray or misnamed crop: a citable label no human vouched for.
        self._png(good_bundle, "table_99")
        problems = validate_bundle(good_bundle)
        assert any("figures/table_99.png is not declared" in p
                   for p in problems)

    def test_empty_list_accepted_for_an_exhibit_free_paper(self, good_bundle):
        shutil.rmtree(good_bundle / "figures")
        m = _base_manifest()
        m["exhibits"] = []
        _write_manifest(good_bundle, m)
        assert validate_bundle(good_bundle) == []

    def test_empty_list_with_an_undeclared_png_still_fails(self, good_bundle):
        m = _base_manifest()
        m["exhibits"] = []
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("figures/table_01.png is not declared" in p
                   for p in problems)

    def test_duplicate_labels(self, good_bundle):
        m = _base_manifest()
        m["exhibits"].append(
            {"label": "table_01", "caption": "Table 1 again"})
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("declared more than once" in p and "table_01" in p
                   for p in problems)

    def test_not_a_list(self, good_bundle):
        m = _base_manifest()
        m["exhibits"] = {"table_01": "Table 1"}
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("'exhibits' must be a list" in p for p in problems)

    def test_entry_not_an_object(self, good_bundle):
        m = _base_manifest()
        m["exhibits"] = ["table_01"]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] must be an object" in p for p in problems)

    def test_entry_missing_caption(self, good_bundle):
        m = _base_manifest()
        m["exhibits"] = [{"label": "table_01"}]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] is missing required key" in p
                   and "caption" in p for p in problems)

    def test_entry_missing_label(self, good_bundle):
        m = _base_manifest()
        m["exhibits"] = [{"caption": "Table 1. Some results"}]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] is missing required key" in p
                   and "label" in p for p in problems)

    def test_entry_extra_key(self, good_bundle):
        # `notes` widened the key set by exactly one; everything else an
        # author might reach for is still refused.
        m = _base_manifest()
        m["exhibits"][0]["page"] = 4
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] has unknown key" in p and "page" in p
                   for p in problems)

    def test_entry_empty_caption(self, good_bundle):
        m = _base_manifest()
        m["exhibits"][0]["caption"] = "   "
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] key 'caption' must be a non-empty string" in p
                   for p in problems)

    def test_entry_empty_label(self, good_bundle):
        m = _base_manifest()
        m["exhibits"][0]["label"] = ""
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] key 'label' must be a non-empty string" in p
                   for p in problems)

    def test_entry_wrong_types(self, good_bundle):
        m = _base_manifest()
        m["exhibits"] = [{"label": 1, "caption": ["a"]}]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("key 'label' must be a string" in p for p in problems)
        assert any("key 'caption' must be a string" in p for p in problems)

    def test_label_character_rules(self, good_bundle):
        # A label is a figures/*.png stem and the token an <img> citation
        # carries, so it obeys the same character rule an id does.
        m = _base_manifest()
        m["exhibits"][0]["label"] = "Table 1/../x"
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("^[A-Za-z0-9._-]+$" in p and "Table 1/../x" in p
                   for p in problems)

    def test_malformed_block_suppresses_the_cross_check(self, good_bundle):
        # The shape problem is the thing to fix; cross-checking a broken
        # declaration against figures/ would bury it under derived noise.
        m = _base_manifest()
        m["exhibits"] = [{"label": "table_01"}]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert not any("not declared" in p for p in problems)

    def test_notes_is_optional(self, good_bundle):
        # Most exhibits print no footnote, so the key is absent from most
        # entries, and its absence is a valid bundle rather than a gap.
        _write_manifest(good_bundle, _base_manifest())
        assert validate_bundle(good_bundle) == []
        assert load_bundle(good_bundle).exhibit_notes == {}

    def test_notes_declared_is_valid(self, good_bundle):
        m = _base_manifest()
        m["exhibits"][0]["notes"] = "CRT-HD, Composite Rig Test (Heavy Duty)."
        _write_manifest(good_bundle, m)
        assert validate_bundle(good_bundle) == []

    def test_wrong_notes_type(self, good_bundle):
        m = _base_manifest()
        m["exhibits"][0]["notes"] = ["a", "footnote"]
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] key 'notes' must be a string when present"
                   in p for p in problems)

    def test_empty_notes_rejected(self, good_bundle):
        # Optional but, when present, non-empty: an exhibit with no printed
        # footnote omits the key rather than declaring an empty one.
        m = _base_manifest()
        m["exhibits"][0]["notes"] = "   "
        _write_manifest(good_bundle, m)
        problems = validate_bundle(good_bundle)
        assert any("exhibits[0] key 'notes' must be a non-empty string when "
                   "present" in p for p in problems)

    def test_load_exposes_notes_only_for_the_exhibits_that_declared_them(
            self, good_bundle):
        # A label missing from the map is an exhibit the paper prints no
        # footnote under, which is why the map is a subset rather than a
        # parallel one carrying nulls.
        self._png(good_bundle, "figure_02")
        m = _base_manifest()
        m["exhibits"][0]["notes"] = "Adjusted for unit age and duty class."
        m["exhibits"].append(
            {"label": "figure_02", "caption": "Figure 2. Study flow"})
        _write_manifest(good_bundle, m)
        b = load_bundle(good_bundle)
        assert b.exhibit_notes == {
            "table_01": "Adjusted for unit age and duty class."}
        assert set(b.exhibit_notes) <= set(b.exhibits)

    def test_load_exposes_captions_keyed_by_label(self, good_bundle):
        self._png(good_bundle, "figure_02")
        m = _base_manifest()
        m["exhibits"].append(
            {"label": "figure_02", "caption": "Figure 2. Study flow"})
        _write_manifest(good_bundle, m)
        b = load_bundle(good_bundle)
        assert b.exhibits == {
            "figure_02": "Figure 2. Study flow",
            "table_01": "Table 1. Some results",
        }
        # Sorted by label, and the same labels the figures map carries.
        assert list(b.exhibits) == sorted(b.exhibits)
        assert set(b.exhibits) == set(b.figures)


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
        # The fixture's one exhibit prints a footnote, and its text is
        # carried beside the caption rather than left in the crop's pixels.
        assert b.exhibit_notes["table_01"].startswith("CRT-HD, Composite Rig")

    def test_frozen(self, bundle_minimal_dir):
        b = load_bundle(bundle_minimal_dir)
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.study_id = "other"

    def test_optional_fields_default_none(self, good_bundle):
        _write_manifest(good_bundle, _base_manifest())  # no doi/summary
        b = load_bundle(good_bundle)
        assert b.doi is None
        assert b.summary is None

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

    def test_load_raises_bundle_error_with_all_problems(self, good_bundle):
        (good_bundle / "text.md").unlink()
        m = _base_manifest()
        m["bogus"] = 1
        _write_manifest(good_bundle, m)
        with pytest.raises(BundleError) as excinfo:
            load_bundle(good_bundle)
        # The error carries every problem, not just the first.
        assert len(excinfo.value.problems) >= 2
