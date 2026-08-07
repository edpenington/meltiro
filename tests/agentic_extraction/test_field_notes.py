"""Field notes: the `notes` slot in every envelope.

The envelope is `{value, evidence, notes}`. The note is where the extractor
records whatever justifies or explains that field's value and is not a verbatim
quote. It sits outside validation: it is never quote-checked, it never counts
toward satisfying `evidence: required`, and there is no such thing as a wrong
note. Its shape is checked (a string, or null) and nothing else.

The checker DOES read it; see test_checker_prompts.py for the rendering.
"""

import pytest

from meltiro.extraction_record import ExtractionRecord
from meltiro.tools import ToolDispatcher, get_tool_definitions
from meltiro.validators import validate_envelope

from .conftest import open_initial_check_gate


Q_GAUGE = "<q>WDS-9 was administered</q>"


@pytest.fixture
def dispatcher(synthetic_template, paper_text, image_labels):
    # The initial-check ordering gate is opened here, once: every mutating
    # call is refused until `record_initial_check` has landed, and that rule
    # is not what this file is about (see test_tools.py for it).
    a = open_initial_check_gate(ExtractionRecord())
    return a, ToolDispatcher(a, synthetic_template, paper_text, image_labels)


def _spec(evidence="optional", field_type="free_text"):
    return {"variable": "v", "field_type": field_type, "options": None,
            "evidence": evidence}


class TestEnvelopeSchema:
    def _envelope_props(self, template, tool_name, block):
        tool = next(t for t in get_tool_definitions(template)
                    if t["name"] == tool_name)
        return tool["input_schema"]["properties"][block]["properties"]

    @pytest.mark.parametrize("tool_name,block", [
        ("update_study", "study"),
        ("add_record", "fields"),
        ("update_record", "fields"),
    ])
    def test_every_envelope_carries_a_required_nullable_notes_slot(
            self, synthetic_template, tool_name, block):
        # Required AND nullable, exactly as `evidence` is: strict
        # structured-output modes reject a property that is absent from
        # `required`, so this is what makes "no note" expressible.
        props = self._envelope_props(synthetic_template, tool_name, block)
        assert props
        for p in props.values():
            assert p["properties"]["notes"] == {"type": ["string", "null"]}
            assert "notes" in p["required"]
            assert p["additionalProperties"] is False

    def test_the_bare_value_blocks_get_no_notes_slot(self, synthetic_template):
        # initial_check and quality_check are bare-value process blocks whose
        # fields are already free text; a note slot there would be noise. Each
        # block has one tool of its own: the initial check is the flat argument
        # list of `record_initial_check`, and the quality check is the
        # `quality_check` argument of `mark_complete`.
        tools = {t["name"]: t
                 for t in get_tool_definitions(synthetic_template)}
        initial = tools["record_initial_check"]["input_schema"]["properties"]
        quality = tools["mark_complete"]["input_schema"]["properties"][
            "quality_check"]["properties"]
        for props in (initial, quality):
            assert props
            for p in props.values():
                assert "properties" not in p  # bare value, not an envelope

    def test_the_notes_framing_is_stated_once_per_block(
            self, synthetic_template):
        # Not per field: a per-field description is paid for once per field per
        # request, which is why `evidence` carries none either.
        tool = next(t for t in get_tool_definitions(synthetic_template)
                    if t["name"] == "add_record")
        block = tool["input_schema"]["properties"]["fields"]
        assert "is not a verbatim quote" in block["description"]
        for p in block["properties"].values():
            assert "description" not in p["properties"]["notes"]


class TestNotesAreShapeCheckedOnly:
    def test_a_string_note_is_accepted(self):
        errors = validate_envelope(
            {"value": "x", "evidence": None, "notes": "read off table 2"},
            _spec(), paper_text="", image_labels=set(), path_prefix="study.v")
        assert errors == []

    def test_a_null_note_is_accepted(self):
        errors = validate_envelope(
            {"value": "x", "evidence": None, "notes": None},
            _spec(), paper_text="", image_labels=set(), path_prefix="study.v")
        assert errors == []

    def test_an_absent_notes_key_is_not_an_error(self):
        # The human-producer path never sends one; an absent key reads as null.
        errors = validate_envelope(
            {"value": "x", "evidence": None},
            _spec(), paper_text="", image_labels=set(), path_prefix="study.v")
        assert errors == []

    @pytest.mark.parametrize("bad", [123, ["a"], {"text": "a"}, True])
    def test_a_non_string_note_is_a_type_error(self, bad):
        errors = validate_envelope(
            {"value": "x", "evidence": None, "notes": bad},
            _spec(), paper_text="", image_labels=set(), path_prefix="study.v")
        assert [e["code"] for e in errors] == ["type_mismatch"]
        assert errors[0]["path"] == "study.v.notes"

    def test_a_note_is_never_quote_checked(self):
        # Tags inside a note are not evidence and are not looked at: this note
        # quotes text that appears nowhere in the paper and still passes.
        errors = validate_envelope(
            {"value": "x", "evidence": None,
             "notes": "<q>no sentence in the paper says this</q>"},
            _spec(), paper_text="the paper says something else",
            image_labels=set(), path_prefix="study.v")
        assert errors == []

    def test_a_note_does_not_satisfy_required_evidence(self):
        # A field whose evidence is required still needs a quote or an image.
        # A note is commentary, so it cannot stand in for one.
        errors = validate_envelope(
            {"value": "x", "evidence": None,
             "notes": "I am quite sure about this"},
            _spec(evidence="required"), paper_text="anything",
            image_labels=set(), path_prefix="study.v")
        assert errors
        assert all(e["path"] != "study.v.notes" for e in errors)

    def test_a_note_survives_a_failed_sibling_check(self):
        # The note's own error, if any, is reported alongside the value's
        # rather than masking it.
        errors = validate_envelope(
            {"value": "x", "evidence": None, "notes": 5},
            _spec(evidence="required"), paper_text="anything",
            image_labels=set(), path_prefix="study.v")
        codes = {(e["path"], e["code"]) for e in errors}
        assert ("study.v.notes", "type_mismatch") in codes
        assert len(codes) > 1


class TestNotesPersistThroughTheDispatcher:
    def test_a_study_field_note_is_stored_wholesale(self, dispatcher):
        record, d = dispatcher
        res = d.dispatch("update_study", {"study": {
            "primary_aim": {"value": "Assess WDS-9", "evidence": Q_GAUGE,
                            "notes": "stated only in the abstract"},
        }})
        assert res["status"] == "ok", res
        assert record.study["primary_aim"]["notes"] == \
            "stated only in the abstract"

    def test_a_record_field_note_survives_add_then_update(self, dispatcher):
        record, d = dispatcher
        d.dispatch("add_record", {"fields": {
            "gauge": {"value": "WDS-9", "evidence": Q_GAUGE,
                      "notes": "the paper never names the version"},
        }})
        assert record.records[0]["gauge"]["notes"] == \
            "the paper never names the version"
        d.dispatch("update_record", {
            "record_id": "relationship_1",
            "fields": {"gauge": {"value": "WDS-9", "evidence": Q_GAUGE,
                                 "notes": "confirmed against table 1"}},
        })
        assert record.records[0]["gauge"]["notes"] == \
            "confirmed against table 1"

    def test_the_note_round_trips_through_serialisation(self, dispatcher):
        record, d = dispatcher
        d.dispatch("update_study", {"study": {
            "primary_aim": {"value": "Assess WDS-9", "evidence": Q_GAUGE,
                            "notes": "a note"},
        }})
        restored = ExtractionRecord.from_dict(record.to_dict())
        assert restored.study["primary_aim"]["notes"] == "a note"
