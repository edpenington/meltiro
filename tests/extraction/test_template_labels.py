"""Template field/section `label:` parsing.

A field or section may carry an optional `label:` prose name for
human-facing surfaces (workbook-style UIs, reports, tables). When absent,
the label is derived from the variable name (underscores to spaces, first
letter capitalised). A present-but-malformed label fails loudly. Labels are
presentation-only and never enter any fingerprint.
"""

import copy

import pytest

from meltiro.fingerprint import field_catalogue_hash, tool_set_hash
from meltiro.template import (
    _derive_label,
    _parse_field,
    _parse_sections,
    load_template,
)
from meltiro.tools import get_tool_definitions


def _field(label=None, *, variable="qa_reporting"):
    f = {"variable": variable, "description": "d", "type": "string"}
    if label is not None:
        f["label"] = label
    # envelope=False keeps the focus on label handling (no evidence flag).
    return _parse_field(f, envelope=False, section_name="test")


class TestDeriveLabel:
    def test_underscores_to_spaces_first_letter_capitalised(self):
        assert _derive_label("qa_reporting") == "Qa reporting"

    def test_single_token(self):
        assert _derive_label("authors") == "Authors"

    def test_embedded_casing_preserved(self):
        # Only the first character is touched; the rest is left alone.
        assert _derive_label("study_ID") == "Study ID"


class TestFieldLabel:
    def test_explicit_label_used(self):
        assert _field("QA reporting")["label"] == "QA reporting"

    def test_absent_label_derives_default(self):
        assert _field()["label"] == "Qa reporting"

    def test_label_stripped(self):
        assert _field("  QA reporting  ")["label"] == "QA reporting"

    def test_empty_label_rejected(self):
        with pytest.raises(ValueError, match="`label:` must be a non-empty"):
            _field("")

    def test_whitespace_label_rejected(self):
        with pytest.raises(ValueError, match="`label:` must be a non-empty"):
            _field("   ")

    def test_non_string_label_rejected(self):
        with pytest.raises(ValueError, match="`label:` must be a non-empty"):
            _field(3)


class TestSectionLabel:
    def test_absent_section_label_defaults_to_title(self):
        sections = _parse_sections(
            [{"section": "Study Design",
              "fields": [{"variable": "v", "description": "d",
                          "type": "string"}]}],
            envelope=False)
        assert sections[0]["label"] == "Study Design"

    def test_explicit_section_label_used(self):
        sections = _parse_sections(
            [{"section": "Study Design", "label": "Design details",
              "fields": [{"variable": "v", "description": "d",
                          "type": "string"}]}],
            envelope=False)
        assert sections[0]["label"] == "Design details"

    def test_empty_section_label_rejected(self):
        with pytest.raises(ValueError, match="`label:` must be a non-empty"):
            _parse_sections(
                [{"section": "Study Design", "label": "  ",
                  "fields": [{"variable": "v", "description": "d",
                              "type": "string"}]}],
                envelope=False)

class TestShippedTemplateLabels:
    def test_every_field_and_section_has_a_label(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        for block_key in ("study_fields", "record_fields",
                          "initial_check_fields", "quality_check_fields"):
            for section in t[block_key]:
                assert section["label"], section["section"]
                for f in section["fields"]:
                    assert f["label"], f["variable"]


class TestLabelIsFingerprintNeutral:
    """A label edit must not move the checker's field-catalogue hash nor the
    extractor's tool-set hash; labels are presentation-only."""

    def test_label_edit_does_not_move_field_catalogue_hash(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        before = field_catalogue_hash(t)
        edited = copy.deepcopy(t)
        edited["study_fields"][0]["fields"][0]["label"] = "Totally different"
        edited["study_fields"][0]["label"] = "Renamed section"
        assert field_catalogue_hash(edited) == before

    def test_label_edit_does_not_move_tool_set_hash(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        before = tool_set_hash(get_tool_definitions(t))
        edited = copy.deepcopy(t)
        edited["study_fields"][0]["fields"][0]["label"] = "Totally different"
        edited["study_fields"][0]["label"] = "Renamed section"
        assert tool_set_hash(get_tool_definitions(edited)) == before
