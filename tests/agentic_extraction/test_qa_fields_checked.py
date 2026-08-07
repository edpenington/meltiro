"""Quality-assessment fields reach the checker.

A quality-assessment section is an ordinary section of its scope's field list,
marked `qa: true`. Its fields are therefore in the checker's field-spec index
like any other, and the trigger admits them: a QA field is an ordinary envelope
field that can carry evidence, so there is no reason it should go unchecked.
Giving QA its own top-level block instead would put its fields outside the
index, where they would fall through the skip branch silently, unexempted by
anything a reader could find.

The one genuine exemption holds: a variable the template does not declare at
all has no spec to brief the checker with.

These load the shipped worked example rather than a hand-built template dict:
the point is that the load path and the trigger agree.
"""

from meltiro.extraction_record import ExtractionRecord
from meltiro.template import load_template

from .conftest import checker_trigger_orch


def _env(value, evidence="<q>q</q>"):
    return {"value": value, "evidence": evidence, "notes": None}


def _checked(monkeypatch, template, study, records):
    """The field paths the trigger admits when every stored field applied."""
    monkeypatch.setattr(
        "meltiro.orchestrator.build_checker_user_message",
        lambda **kw: [{"type": "text", "text": "stub:" + kw["field_path"]}],
    )
    record = ExtractionRecord()
    record.study.update(study)
    record.records.extend(records)
    applied = [f"study.{var}" for var in study]
    for rec in records:
        applied.extend(f"record.{rec['record_id']}.{var}" for var in rec
                       if var != "record_id")
    orch = checker_trigger_orch(template, record)
    calls, _ = orch._build_checker_calls(applied)
    return [c["field_path"] for c in calls]


def test_study_qa_field_carrying_evidence_is_checked(monkeypatch, config_dir):
    template = load_template(config_dir / "extraction_template.yaml")
    paths = _checked(
        monkeypatch, template,
        study={
            "study_label": _env("Smith 2019"),
            "qa_reporting": _env("Compliant", "<q>STROBE checklist</q>"),
            "study_notes": _env("commentary"),
        },
        records=[],
    )
    assert "study.qa_reporting" in paths
    assert "study.study_label" in paths
    # An undeclared variable has no spec to brief the checker with.
    assert "study.study_notes" not in paths


def test_record_qa_field_carrying_evidence_is_checked(monkeypatch, config_dir):
    template = load_template(config_dir / "extraction_template.yaml")
    paths = _checked(
        monkeypatch, template,
        study={},
        records=[{
            "record_id": "relationship_1",
            "effect_size": _env("0.42"),
            "rqa_sample_adequate": _env("Adequate", "<q>N=2,450</q>"),
            "relationship_notes": _env("commentary"),
        }],
    )
    assert "record.relationship_1.rqa_sample_adequate" in paths
    assert "record.relationship_1.effect_size" in paths
    assert "record.relationship_1.relationship_notes" not in paths


def test_every_qa_field_of_both_scopes_is_checkable(monkeypatch, config_dir):
    # Not just the two named above: every field of every `qa: true` section is
    # checkable, so a section added later is covered without a new test.
    template = load_template(config_dir / "extraction_template.yaml")
    study_qa = [f["variable"] for s in template["study_fields"] if s["qa"]
                for f in s["fields"]]
    record_qa = [f["variable"] for s in template["record_fields"] if s["qa"]
                 for f in s["fields"]]
    assert study_qa and record_qa
    paths = _checked(
        monkeypatch, template,
        study={var: _env("Compliant") for var in study_qa},
        records=[dict({var: _env("Adequate") for var in record_qa},
                      record_id="relationship_1")],
    )
    assert set(paths) == (
        {f"study.{var}" for var in study_qa}
        | {f"record.relationship_1.{var}" for var in record_qa}
    )
