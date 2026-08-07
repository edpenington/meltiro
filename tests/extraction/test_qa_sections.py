"""Quality-assessment sections: the `qa: true` section flag.

The separate QA blocks (`study_qa:` and `records.<entity>.qa:`) are retired.
A quality-assessment section is an ordinary section of `study_extraction:` or
the record's `extraction:` list, marked `qa: true`. The flag is presentation
only: `render_template` groups the flagged sections under their own heading in
the operational document and leaves them out of the publication document, and
nothing else reads it. Its fields are ordinary envelope fields of their scope,
so they reach the tool schemas, the validator's spec index, the required-field
gate, and the checker's per-field fan-out like any other field.

The retirement errors themselves are covered in test_records_block.py.
"""

import pytest

from meltiro.fingerprint import field_catalogue_hash
from meltiro.render_template import render_template
from meltiro.template import _parse_sections, iter_fields, load_template
from meltiro.tools import get_tool_definitions
from meltiro.validators import validate_extraction_output


def _section(**overrides):
    section = {
        "section": "Risk of Bias",
        "fields": [{"variable": "qa_overall", "description": "d",
                    "type": "string", "evidence": "optional"}],
    }
    section.update(overrides)
    return section


# ---------------------------------------------------------------------------
# Parsing the flag
# ---------------------------------------------------------------------------

class TestQaFlagParsing:
    def test_absent_flag_defaults_to_false(self):
        parsed = _parse_sections([_section()], envelope=True)
        assert parsed[0]["qa"] is False

    @pytest.mark.parametrize("flag", [True, False])
    def test_flag_parses_as_declared(self, flag):
        parsed = _parse_sections([_section(qa=flag)], envelope=True)
        assert parsed[0]["qa"] is flag

    @pytest.mark.parametrize("bad", ["true", 1, [], None])
    def test_non_boolean_flag_rejected(self, bad):
        with pytest.raises(ValueError, match="`qa:` must be true or false"):
            _parse_sections([_section(qa=bad)], envelope=True)

    def test_mistyped_section_key_rejected(self):
        # The section key allowlist means a typo fails loudly rather than being
        # silently ignored, which would leave the author's QA section rendered
        # as an ordinary extraction section.
        with pytest.raises(ValueError, match="unknown section key"):
            _parse_sections([_section(qaa=True)], envelope=True)

    def test_flag_valid_on_bare_value_blocks_too(self):
        # The flag is a section key, not an envelope concern, so the parser
        # accepts it in any block. Nothing in the engine treats a flagged check
        # section specially; the renderer never groups those blocks.
        parsed = _parse_sections(
            [{"section": "Initial Check", "qa": True,
              "fields": [{"variable": "text_readable", "description": "d",
                          "type": "boolean"}]}],
            envelope=False)
        assert parsed[0]["qa"] is True


# ---------------------------------------------------------------------------
# The shipped configs: QA fields are ordinary fields of their scope
# ---------------------------------------------------------------------------

class TestShippedTemplate:
    def _qa_variables(self, sections):
        return [f["variable"] for s in sections if s["qa"]
                for f in s["fields"]]

    def test_qa_sections_live_in_the_scope_field_list(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        assert self._qa_variables(t["study_fields"]) == ["qa_reporting"]
        assert "rqa_sample_adequate" in self._qa_variables(t["record_fields"])

    def test_qa_fields_are_envelope_fields(self, config_dir):
        # They declare an evidence policy like any other envelope field, which
        # is what makes them checkable.
        t = load_template(config_dir / "extraction_template.yaml")
        checked = 0
        for block in ("study_fields", "record_fields"):
            for section in t[block]:
                if not section["qa"]:
                    continue
                for f in section["fields"]:
                    assert f["evidence"] in ("required", "optional")
                    checked += 1
        # Every assertion above sits behind the `qa` filter, so the sweep is
        # only meaningful if it reached some fields.
        assert checked > 0, "no qa fields found to check"

    def test_qa_fields_are_in_the_tool_schemas(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        tools = {tool["name"]: tool for tool in get_tool_definitions(t)}
        study_props = tools["update_study"][
            "input_schema"]["properties"]["study"]["properties"]
        assert "qa_reporting" in study_props
        record_props = tools["add_record"][
            "input_schema"]["properties"]["fields"]["properties"]
        assert "rqa_sample_adequate" in record_props

    def test_qa_fields_are_known_to_the_validator(self, config_dir):
        # A QA value in the extraction output validates against its spec
        # instead of being reported as an unknown field.
        t = load_template(config_dir / "extraction_template.yaml")
        output = {
            "study": {"qa_reporting": {"value": "Compliant",
                                       "evidence": None, "source": None}},
            "records": [{
                "record_id": "relationship_1",
                "rqa_sample_adequate": {"value": "Adequate",
                                        "evidence": None, "source": None},
            }],
        }
        failures, _ = validate_extraction_output(t, output)
        assert [f for f in failures if f["code"] == "unknown_field"] == []

    def test_qa_field_value_outside_its_options_is_rejected(self, config_dir):
        # The QA field is validated on its own spec, so a bad enum value fails
        # exactly as it would for any other categorical field.
        t = load_template(config_dir / "extraction_template.yaml")
        output = {"study": {"qa_reporting": {
            "value": "Not a listed option", "evidence": None, "source": None}}}
        failures, _ = validate_extraction_output(t, output)
        assert [f["path"] for f in failures] == ["study.qa_reporting.value"]


# ---------------------------------------------------------------------------
# The flag is presentation only: it moves no fingerprint
# ---------------------------------------------------------------------------

class TestFlagIsFingerprintNeutral:
    def _template(self, **section_overrides):
        field = {"variable": "qa_overall", "description": "d",
                 "field_type": "string", "evidence": "optional"}
        section = {"section": "Risk of Bias", "label": "Risk of Bias",
                   "qa": False, "fields": [field]}
        section.update(section_overrides)
        return {
            "study_fields": [section],
            "record_fields": [],
            "initial_check_fields": [],
            "quality_check_fields": [],
        }

    def test_qa_flag_does_not_move_the_field_catalogue_hash(self):
        # Only field attributes are hashed, so a section-level presentation key
        # is out of the catalogue by construction.
        assert field_catalogue_hash(self._template(qa=True)) == \
            field_catalogue_hash(self._template(qa=False))

    def test_section_label_does_not_move_the_field_catalogue_hash(self):
        assert field_catalogue_hash(self._template(label="Quality")) == \
            field_catalogue_hash(self._template(label="Risk of Bias"))

    def test_flipping_the_flag_in_the_shipped_yaml_keeps_the_hash(
            self, config_dir, tmp_path):
        # End to end: dropping `qa: true` from a shipped section changes where
        # the renderer puts it and nothing else, so the checker fingerprint
        # component stays put.
        text = (config_dir / "extraction_template.yaml").read_text(
            encoding="utf-8")
        assert "  qa: true\n" in text
        flat = tmp_path / "extraction_template.yaml"
        flat.write_text(text.replace("  qa: true\n", "", 1), encoding="utf-8")
        before = field_catalogue_hash(
            load_template(config_dir / "extraction_template.yaml"))
        assert field_catalogue_hash(load_template(flat)) == before


# ---------------------------------------------------------------------------
# What the flag does do: grouping in the rendered documents
# ---------------------------------------------------------------------------

class TestRendering:
    def test_operational_groups_qa_sections_under_their_own_heading(
            self, config_dir):
        out = render_template(
            load_template(config_dir / "extraction_template.yaml"),
            "operational")
        study_extraction = out.index("## Study-level extraction")
        study_qa = out.index("## Study-level quality appraisal")
        record_qa = out.index("## Record-level quality appraisal")
        # The flagged sections render after the extraction sections of both
        # scopes, under their own headings.
        assert study_extraction < study_qa < record_qa
        assert out.index("### Reporting") > study_qa
        assert out.index("### Sample and Selection") > record_qa

    def test_publication_omits_qa_sections(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        out = render_template(t, "publication")
        qa_variables = [f["variable"]
                        for block in ("study_fields", "record_fields")
                        for s in t[block] if s["qa"]
                        for f in iter_fields([s])]
        assert qa_variables
        for section in (s for block in ("study_fields", "record_fields")
                        for s in t[block] if s["qa"]):
            assert section["label"] not in out
