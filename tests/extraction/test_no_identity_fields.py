"""There is no identity / pipeline-managed field concept.

`identity: true` and the `role: study_id` stamping machinery do not exist:
EVERY declared template field is an ordinary extractable field (in the tool
schema, checker-checked, validated normally). The one value the pipeline owns,
the study id, is not a template field at all: the engine reads it from the
bundle manifest and records it in the run's output metadata (run.json
`study_id`).

These tests pin:
  - `identity:` fails loudly at load, as a key the field allowlist does not
    define;
  - `role: study_id` and `role: doi` are not roles this engine wires;
  - the shipped bibliographic fields (title, authors, doi, ...) are ordinary
    fields: present in the extractor tool schema and in the checker fan-out;
  - the shipped template declares no per-record study_id field.
"""

import json

import pytest

from meltiro.template import _parse_field, load_template
from meltiro.tools import get_tool_definitions


def _field(extra):
    f = {"variable": "v", "description": "d", "type": "string",
         "evidence": "required"}
    f.update(extra)
    return f


class TestLoadRejections:
    def test_identity_flag_rejected_as_an_undefined_field_key(self):
        # There is no pipeline-managed field category to flag, so `identity:`
        # is a key like any other the schema does not define: refused by the
        # field allowlist, named in the message, never silently ignored.
        with pytest.raises(ValueError) as exc:
            _parse_field(_field({"identity": True}),
                         envelope=True, section_name="Identity")
        assert "unknown field key(s)" in str(exc.value)
        assert "'identity'" in str(exc.value)

    def test_role_study_id_is_not_a_role(self):
        with pytest.raises(ValueError,
                           match=r"unknown `role: 'study_id'`"):
            _parse_field(_field({"role": "study_id"}),
                         envelope=True, section_name="Identity")

    def test_role_doi_is_not_a_role(self):
        with pytest.raises(ValueError, match=r"unknown `role: 'doi'`"):
            _parse_field(_field({"role": "doi"}),
                         envelope=True, section_name="Identity")

    def test_shipped_template_has_no_study_id_field(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        study_vars = {f["variable"]
                      for s in t["study_fields"] for f in s["fields"]}
        record_vars = {f["variable"]
                       for s in t["record_fields"] for f in s["fields"]}
        # The study id is not a template field at all (study or record).
        assert "study_id" not in study_vars
        assert "study_id" not in record_vars
        # The engine-assigned record id is not a template field either.
        assert "relationship_id" not in record_vars


class TestBibliographicFieldsAreExtractable:
    def test_bibliographic_fields_in_extractor_schema(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        tools = get_tool_definitions(t)
        study_props = next(
            x for x in tools if x["name"] == "update_study"
        )["input_schema"]["properties"]["study"]["properties"]
        for var in ("title", "authors", "year", "journal", "doi",
                    "study_label"):
            assert var in study_props, var

    def test_no_pipeline_managed_field_code_in_tool_blob(self, config_dir):
        # There is no `pipeline_managed_field` path, so nothing renders it.
        t = load_template(config_dir / "extraction_template.yaml")
        blob = json.dumps(get_tool_definitions(t))
        assert "pipeline_managed" not in blob

    def test_bibliographic_fields_are_checked(self, monkeypatch, config_dir):
        # A populated bibliographic field is a genuinely-extracted value the
        # checker fans out over; it is NOT skipped as metadata.
        from meltiro.extraction_record import ExtractionRecord

        from agentic_extraction.conftest import checker_trigger_orch

        monkeypatch.setattr(
            "meltiro.orchestrator.build_checker_user_message",
            lambda **kw: [{"type": "text", "text": kw["field_path"]}])
        t = load_template(config_dir / "extraction_template.yaml")
        record = ExtractionRecord()
        record.study["title"] = {"value": "A Study",
                                 "evidence": "<q>A Study</q>", "notes": None}
        orch = checker_trigger_orch(t, record)
        calls, _ = orch._build_checker_calls(["study.title"])
        assert [c["field_path"] for c in calls] == ["study.title"]
