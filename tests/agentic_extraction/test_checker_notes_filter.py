"""What the checker sees of the two kinds of note.

The reserved scope-note key (`notes` on the study block, and on each record
beside `record_id`) must never reach a checker call. It is not a field, and it
is deliberately withheld: a record's note is the extractor's holistic reasoning
about the whole record, and feeding it into every per-field call would
correlate the two stages, which is the thing the checker's narrow context
exists to prevent.

A FIELD's own note is the opposite case: it rides inside the envelope the
trigger hands to the checker, and the checker reads it (see
tests/agentic_extraction/test_checker_prompts.py for the rendering).

The dispatcher never reports a scope note in `applied_fields`, so the trigger
is normally never offered one. These tests offer it one anyway, because the
guarantee has to hold on the trigger's own terms rather than on a caller
remembering to filter first.
"""

from meltiro.extraction_record import ExtractionRecord

from .conftest import checker_trigger_orch


def _env(value, evidence="<q>q</q>", notes=None):
    return {"value": value, "evidence": evidence, "notes": notes}


def _stub_user_message(monkeypatch):
    monkeypatch.setattr(
        "meltiro.orchestrator.build_checker_user_message",
        lambda **kw: [{"type": "text", "text": "stub:" + kw["field_path"]}],
    )


def test_the_study_scope_note_is_not_a_checkable_field(monkeypatch,
                                                       synthetic_template):
    _stub_user_message(monkeypatch)
    record = ExtractionRecord()
    record.study["primary_aim"] = _env("Aim A")
    record.study["notes"] = "holistic commentary about the whole study"
    orch = checker_trigger_orch(synthetic_template, record)

    calls, _ = orch._build_checker_calls(["study.primary_aim", "study.notes"])
    assert [c["field_path"] for c in calls] == ["study.primary_aim"]


def test_a_record_scope_note_is_not_a_checkable_field(monkeypatch,
                                                      synthetic_template):
    _stub_user_message(monkeypatch)
    record = ExtractionRecord()
    rid = record.add_record({"gauge": _env("WDS-9")}, "relationship",
                            notes="why this record exists at all")
    orch = checker_trigger_orch(synthetic_template, record)

    calls, _ = orch._build_checker_calls(
        [f"record.{rid}.gauge", f"record.{rid}.notes"])
    assert [c["field_path"] for c in calls] == [f"record.{rid}.gauge"]


def test_a_null_scope_note_is_skipped_without_a_shape_error(monkeypatch,
                                                            synthetic_template):
    # The reserved key is present from the start, holding null. A bare string
    # (or None) is not an envelope, so the trigger must skip it rather than
    # stumble reading a value off it.
    _stub_user_message(monkeypatch)
    record = ExtractionRecord()
    record.study["notes"] = None
    record.study["primary_aim"] = _env("Aim A")
    orch = checker_trigger_orch(synthetic_template, record)

    calls, _ = orch._build_checker_calls(["study.notes", "study.primary_aim"])
    assert [c["field_path"] for c in calls] == ["study.primary_aim"]


def test_a_fields_own_note_travels_with_it_to_the_checker(monkeypatch,
                                                          synthetic_template):
    # The field note is NOT filtered: it is part of the envelope handed to the
    # checker, which is how the checker gets to read it, and it is recorded as
    # part of what the verdict was passed on.
    _stub_user_message(monkeypatch)
    record = ExtractionRecord()
    record.study["notes"] = "scope note"
    record.study["primary_aim"] = _env("Aim A", notes="field note")
    orch = checker_trigger_orch(synthetic_template, record)

    calls, envelopes = orch._build_checker_calls(["study.primary_aim"])
    assert [c["field_path"] for c in calls] == ["study.primary_aim"]
    assert envelopes["study.primary_aim"]["notes"] == "field note"
