"""Scope notes: the reserved `notes` key on the study block and each record.

A scope note is the extractor's commentary about a whole scope, as opposed to a
field note, which explains one value. It is a RESERVED key, not a field: no
template may declare a field named `notes`, so nothing can collide with it, and
every walk over the study block or a record has to skip it rather than mistake
it for an envelope.

Writing one is never a validation failure and never moves a call's
ok / partial / validation_failed status, which is decided by the fields alone.
The checker never sees a scope note; that is pinned in
test_checker_notes_filter.py.
"""

import pytest

from meltiro.extraction_record import ExtractionRecord
from meltiro.tools import ToolDispatcher, get_tool_definitions
from meltiro.validators import validate_extraction_output

from .conftest import open_initial_check_gate


Q_GAUGE = "<q>WDS-9 was administered</q>"
Q_ABSENT = "<q>no sentence in the paper says this</q>"


@pytest.fixture
def dispatcher(synthetic_template, paper_text, image_labels):
    # The initial-check ordering gate is opened here, once: every mutating
    # call is refused until `record_initial_check` has landed, and that rule
    # is not what this file is about (see test_tools.py for it).
    a = open_initial_check_gate(ExtractionRecord())
    return a, ToolDispatcher(a, synthetic_template, paper_text, image_labels)


def _env(value, evidence=Q_GAUGE):
    return {"value": value, "evidence": evidence, "notes": None}


class TestTheReservedKeyIsAlwaysThere:
    def test_a_fresh_study_block_carries_a_null_note(self):
        assert ExtractionRecord().study == {"notes": None}

    def test_every_record_is_minted_with_a_null_note(self):
        a = ExtractionRecord()
        a.add_record({"gauge": _env("WDS-9")}, "relationship")
        assert a.records[0]["notes"] is None

    def test_from_dict_restores_a_missing_reserved_key(self):
        # The in-memory shape matches a freshly built record whatever produced
        # the JSON, so nothing downstream has to test for the key's presence.
        a = ExtractionRecord.from_dict({
            "study": {"primary_aim": _env("x")},
            "records": [{"record_id": "relationship_1"}],
        })
        assert a.study["notes"] is None
        assert a.records[0]["notes"] is None

    def test_notes_round_trip_through_serialisation(self):
        a = ExtractionRecord()
        a.apply_update_study(study={"primary_aim": _env("x")},
                             notes="study-wide commentary")
        a.add_record({"gauge": _env("WDS-9")}, "relationship",
                     notes="record commentary")
        out = a.to_dict()
        assert out["study"]["notes"] == "study-wide commentary"
        assert out["records"][0]["notes"] == "record commentary"
        restored = ExtractionRecord.from_dict(out)
        assert restored.study["notes"] == "study-wide commentary"
        assert restored.records[0]["notes"] == "record commentary"


class TestTheToolArguments:
    @pytest.mark.parametrize("tool_name", [
        "update_study", "add_record", "update_record"])
    def test_each_block_writing_tool_takes_a_scope_note(
            self, synthetic_template, tool_name):
        tool = next(t for t in get_tool_definitions(synthetic_template)
                    if t["name"] == tool_name)
        schema = tool["input_schema"]["properties"]["notes"]
        assert schema["type"] == ["string", "null"]
        assert "description" in schema
        # Optional: omitting it leaves any stored note untouched.
        assert "notes" not in tool["input_schema"].get("required", [])

    def test_update_study_writes_the_study_note(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("update_study", {"notes": "one batch, two papers"})
        assert res["status"] == "ok"
        assert record.study["notes"] == "one batch, two papers"
        assert res["applied_changes"]["notes_written"] is True

    def test_add_record_writes_the_record_note(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("add_record", {
            "fields": {"gauge": _env("WDS-9")},
            "notes": "this row is the paper's headline analysis",
        })
        assert res["status"] == "ok", res
        assert record.records[0]["notes"] == \
            "this row is the paper's headline analysis"
        assert res["applied_changes"]["notes_written"] is True

    def test_update_record_writes_the_record_note(self, dispatcher):
        record, d = dispatcher
        d.dispatch("add_record", {"fields": {"gauge": _env("WDS-9")}})
        res = d.dispatch("update_record", {
            "record_id": "relationship_1",
            "fields": {"gauge": _env("WDS-9")},
            "notes": "revised after re-reading table 2",
        })
        assert res["status"] == "ok", res
        assert record.records[0]["notes"] == "revised after re-reading table 2"

    def test_an_omitted_note_leaves_an_earlier_one_alone(self, dispatcher):
        record, d = dispatcher
        d.dispatch("update_study", {"notes": "keep me"})
        d.dispatch("update_study", {"study": {"primary_aim": _env("x")}})
        assert record.study["notes"] == "keep me"

    def test_an_explicit_null_clears_the_note(self, dispatcher):
        record, d = dispatcher
        d.dispatch("update_study", {"notes": "provisional"})
        d.dispatch("update_study", {"notes": None})
        assert record.study["notes"] is None


class TestAScopeNoteNeverMovesTheStatus:
    def test_a_note_does_not_rescue_a_failed_call(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("update_study", {
            "study": {"not_a_field": _env("x")},
            "notes": "a note",
        })
        # Status is decided by the fields: every field failed.
        assert res["status"] == "validation_failed"
        # The note still landed: it addresses no field, so nothing rejected it.
        assert record.study["notes"] == "a note"

    def test_a_note_does_not_spoil_a_clean_call(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("update_study", {
            "study": {"primary_aim": _env("Assess WDS-9")},
            "notes": "a note",
        })
        assert res["status"] == "ok"
        assert res["errors"] == []

    def test_a_note_on_a_partial_call_leaves_it_partial(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("add_record", {
            "fields": {"gauge": _env("WDS-9"),
                       "outcome_variable":
                           _env("Unplanned removal", Q_ABSENT)},
            "notes": "a note",
        })
        assert res["status"] == "partial"
        assert record.records[0]["notes"] == "a note"

    def test_a_wholly_failed_add_record_mints_no_record_to_note(
            self, dispatcher):
        # No record, so nothing for the note to attach to; it goes with them.
        record, d = dispatcher
        res = d.dispatch("add_record", {
            "fields": {"gauge": _env("WDS-9", Q_ABSENT)},
            "notes": "a note",
        })
        assert res["status"] == "validation_failed"
        assert record.records == []

    def test_a_malformed_note_warns_rather_than_failing(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("update_study", {
            "study": {"primary_aim": _env("Assess WDS-9")},
            "notes": {"text": "not a string"},
        })
        assert res["status"] == "ok"
        assert res["errors"] == []
        # Loud, but not fatal: the model is told the note was not recorded.
        assert [w["code"] for w in res["warnings"]] == ["notes_not_recorded"]
        assert record.study["notes"] is None

    def test_a_scope_note_does_not_clear_mark_complete(self, dispatcher):
        # The flag forces a re-declaration when CHECKABLE content moves. A
        # scope note is never checked, so writing one must not cost a round.
        record, d = dispatcher
        record.mark_complete()
        d.dispatch("update_study", {"notes": "an afterthought"})
        assert record.mark_complete_flag is True
        # A field write still clears it.
        d.dispatch("update_study", {"study": {"primary_aim": _env("x")}})
        assert record.mark_complete_flag is False


class TestNothingMistakesTheKeyForAField:
    def test_the_sweep_skips_the_reserved_key(self, synthetic_template):
        failures, _warnings = validate_extraction_output(
            synthetic_template,
            {"study": {"notes": "commentary", "primary_aim": _env("x")},
             "records": [{"record_id": "relationship_1",
                          "notes": "commentary",
                          "gauge": _env("WDS-9")}]},
        )
        assert failures == []

    def test_the_sweep_shape_checks_the_reserved_key(self, synthetic_template):
        failures, _warnings = validate_extraction_output(
            synthetic_template,
            {"study": {"notes": 17},
             "records": [{"record_id": "relationship_1", "notes": ["a"]}]},
        )
        assert {(f["path"], f["code"]) for f in failures} == {
            ("study.notes", "type_mismatch"),
            ("record.relationship_1.notes", "type_mismatch"),
        }

    def test_a_note_never_gates_or_is_gated(self, synthetic_template,
                                            paper_text, image_labels):
        # Gate rules read a record's sibling FIELD values. The reserved key is
        # dropped from that view, so a note can neither control nor trip a gate.
        record = open_initial_check_gate(ExtractionRecord())
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        d.dispatch("add_record", {
            "fields": {"outcome_category": _env("Failure state")},
            "notes": "Service life",  # would trip the index gate if read
        })
        res = d.dispatch("update_record", {
            "record_id": "relationship_1",
            "fields": {"index_tariff": _env("DI-4")},
        })
        assert [w["message"] for w in res["warnings"]] == [
            "index_tariff is set but outcome_category is 'Failure state', "
            "not one of: Service life."]

    def test_a_note_inside_a_field_map_is_refused_with_a_hint(
            self, dispatcher):
        # A model that puts the scope note where the fields go gets told where
        # it belongs rather than sent hunting for a typo.
        _record, d = dispatcher
        res = d.dispatch("update_study", {"study": {"notes": _env("x")}})
        assert res["status"] == "validation_failed"
        message = res["failed_fields"]["study.notes"][0]["message"]
        assert "is not a field" in message
        assert "top-level `notes` argument" in message


class TestTheViewToolsSurfaceNotes:
    def _view(self, dispatcher, tool, args=None):
        return dispatcher.dispatch(tool, args or {})["view"]

    def test_view_summary_reports_both_scopes(self, dispatcher):
        record, d = dispatcher
        d.dispatch("update_study", {"notes": "study commentary"})
        d.dispatch("add_record", {"fields": {"gauge": _env("WDS-9")},
                                  "notes": "record commentary"})
        view = self._view(d, "view_summary")
        assert view["study_notes"] == "study commentary"
        assert view["records"][0]["notes"] == "record commentary"

    def test_view_study_fields_returns_the_study_note(self, dispatcher):
        record, d = dispatcher
        d.dispatch("update_study", {"notes": "study commentary"})
        view = self._view(d, "view_study_fields")
        assert view["study"]["notes"] == "study commentary"

    def test_view_record_returns_the_record_note_and_field_notes(
            self, dispatcher):
        record, d = dispatcher
        d.dispatch("add_record", {
            "fields": {"gauge": {"value": "WDS-9", "evidence": Q_GAUGE,
                                 "notes": "field commentary"}},
            "notes": "record commentary",
        })
        view = self._view(d, "view_record", {"record_id": "relationship_1"})
        assert view["record"]["notes"] == "record commentary"
        assert view["record"]["gauge"]["notes"] == "field commentary"
