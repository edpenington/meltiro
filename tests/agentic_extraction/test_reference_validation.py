"""canonical_reference validation: validator path, dispatcher path, events.

Reference fields are a strict closed set validated against a reference list.
A value that matches a canonical name (exactly or by case/whitespace) is
stored as that name; a value that matches a unique alias is stored as the
canonical name with a note + a value_canonicalised record; anything else is
rejected (ambiguous, or off-list with ranked suggestions). A
`type: string_list` reference field carries a real JSON array; every element
is validated independently, duplicates after canonicalisation are rejected,
and the stored value is a list of canonical spellings.
"""

from meltiro.extraction_record import ExtractionRecord
from meltiro.reference_lists import build_reference_index
from meltiro.tools import ToolDispatcher, _field_description, \
    _field_value_subschema
from meltiro.validators import validate_envelope

from .conftest import open_initial_check_gate


REFLIST = [
    {"tool_name": "WDS-9",
     "aliases": ["WDS9", "Widget Durability Scale 9"]},
    {"tool_name": "SRI-7", "aliases": ["SRI7"]},
    {"tool_name": "Composite Rig Test (Heavy Duty)"},
]
AMBIGUOUS_REFLIST = [
    {"tool_name": "Alpha Tool", "aliases": ["shared handle"]},
    {"tool_name": "Beta Tool", "aliases": ["shared handle"]},
]

PAPER = "The WDS-9 and SRI-7 were both administered to the batch."


def _rf(variable, **kw):
    base = {
        "variable": variable, "field_type": "string", "options": None,
        "allow_other": False, "description": "", "extraction_instruction": None,
        "canonical_reference": None, "evidence": "optional",
    }
    base.update(kw)
    return base


def _reference_spec(**kw):
    return _rf("gauge", canonical_reference="gauge_list", **kw)


def _list_reference_spec(**kw):
    return _rf("gauges_collected", field_type="string_list",
               canonical_reference="gauge_list", **kw)


def _template():
    return {
        "record_entity": {"singular": "relationship",
                          "plural": "relationships",
                          "description": "a relationship"},
        "study_fields": [{"section": "S", "extraction_instruction": None,
                          "fields": [_list_reference_spec()]}],
        "record_fields": [{"section": "R", "extraction_instruction": None,
                           "fields": [
                               _reference_spec(),
                               _rf("outcome_variable")]}],
        "initial_check_fields": [],
        "quality_check_fields": [],
        "template_hash": "h",
        "template_path": "/tmp/t.yaml",
    }


def _dispatcher(reference_lists={"gauge_list": REFLIST}):
    # The initial-check ordering gate is opened here, once: every mutating
    # call is refused until `record_initial_check` has landed, and that rule
    # is not what this file is about (see test_tools.py for it). This
    # template declares no initial-check fields at all, which is precisely
    # the case where the gate is a flag rather than a derived emptiness.
    record = open_initial_check_gate(ExtractionRecord())
    return ToolDispatcher(record, _template(), PAPER, set(),
                          reference_lists=reference_lists)


# ---------------------------------------------------------------------------
# Validator level: single-value string fields
# ---------------------------------------------------------------------------

class TestValidatorReference:
    def _index(self):
        return build_reference_index(REFLIST)

    def test_canonical_name_accepted_and_spelled(self):
        env = {"value": "wds-9", "evidence": None}
        errs = validate_envelope(
            env, _reference_spec(), PAPER, set(), path_prefix="record.relationship_1.gauge",
            reference_index=self._index())
        assert errs == []
        assert env["value"] == "WDS-9"

    def test_alias_accepted_canonicalised_and_recorded(self):
        env = {"value": "WDS9", "evidence": None}
        sink = []
        errs = validate_envelope(
            env, _reference_spec(), PAPER, set(), path_prefix="record.relationship_1.gauge",
            reference_index=self._index(), canonicalisations=sink)
        assert errs == []
        assert env["value"] == "WDS-9"
        assert sink == [{"path": "record.relationship_1.gauge", "entered": "WDS9",
                         "stored": "WDS-9"}]

    def test_exact_canonical_name_records_no_event(self):
        env = {"value": "WDS-9", "evidence": None}
        sink = []
        validate_envelope(
            env, _reference_spec(), PAPER, set(), path_prefix="record.relationship_1.gauge",
            reference_index=self._index(), canonicalisations=sink)
        assert sink == []

    def test_off_list_rejected_with_suggestions(self):
        env = {"value": "WDS", "evidence": None}
        errs = validate_envelope(
            env, _reference_spec(), PAPER, set(), path_prefix="record.relationship_1.gauge",
            reference_index=self._index())
        codes = [e["code"] for e in errs]
        assert "off_list_reference" in codes
        msg = next(e["message"] for e in errs
                   if e["code"] == "off_list_reference")
        assert "Closest names" in msg
        assert "WDS-9" in msg

    def test_ambiguous_rejected_with_candidates(self):
        env = {"value": "shared handle", "evidence": None}
        errs = validate_envelope(
            env, _reference_spec(), PAPER, set(), path_prefix="record.relationship_1.gauge",
            reference_index=build_reference_index(AMBIGUOUS_REFLIST))
        msg = next(e["message"] for e in errs
                   if e["code"] == "ambiguous_reference")
        assert "Alpha Tool" in msg and "Beta Tool" in msg

    def test_missing_index_is_wiring_error(self):
        env = {"value": "WDS-9", "evidence": None}
        errs = validate_envelope(
            env, _reference_spec(), PAPER, set(), path_prefix="record.relationship_1.gauge",
            reference_index=None)
        assert any(e["code"] == "reference_unavailable" for e in errs)


# ---------------------------------------------------------------------------
# Validator level: string_list fields (per-element)
# ---------------------------------------------------------------------------

class TestValidatorListReference:
    def _index(self):
        return build_reference_index(REFLIST)

    def _validate(self, value, sink=None, index=None):
        env = {"value": value, "evidence": None}
        errs = validate_envelope(
            env, _list_reference_spec(), PAPER, set(),
            path_prefix="study.gauges_collected",
            reference_index=index if index is not None else self._index(),
            canonicalisations=sink)
        return errs, env

    def test_each_element_validated_and_stored_as_list(self):
        sink = []
        errs, env = self._validate(["wds9", "SRI-7"], sink=sink)
        assert errs == []
        # Real list of canonical spellings, order preserved.
        assert env["value"] == ["WDS-9", "SRI-7"]
        # Only the alias element gets a canonicalisation record.
        assert sink == [{"path": "study.gauges_collected", "entered": "wds9",
                         "stored": "WDS-9"}]

    def test_multiple_alias_elements_record_multiple_events(self):
        sink = []
        errs, env = self._validate(["wds9", "SRI7"], sink=sink)
        assert errs == []
        assert env["value"] == ["WDS-9", "SRI-7"]
        assert sink == [
            {"path": "study.gauges_collected", "entered": "wds9",
             "stored": "WDS-9"},
            {"path": "study.gauges_collected", "entered": "SRI7",
             "stored": "SRI-7"},
        ]

    def test_off_list_element_rejected_naming_element(self):
        errs, env = self._validate(["WDS-9", "Nonsense Tool"])
        err = next(e for e in errs if e["code"] == "off_list_reference")
        assert "Nonsense Tool" in err["message"]
        assert "Closest names" in err["message"]
        # The error path names the offending element's position.
        assert err["path"] == "study.gauges_collected.value[1]"
        # All-or-nothing: the value is not mutated when any element fails.
        assert env["value"] == ["WDS-9", "Nonsense Tool"]

    def test_ambiguous_element_rejected(self):
        errs, _ = self._validate(
            ["shared handle"],
            index=build_reference_index(AMBIGUOUS_REFLIST))
        msg = next(e["message"] for e in errs
                   if e["code"] == "ambiguous_reference")
        assert "Alpha Tool" in msg and "Beta Tool" in msg

    def test_duplicate_after_canonicalisation_rejected(self):
        # "wds9" (alias) and "WDS-9" (canonical) resolve to the same entry:
        # a validation failure, never a silent dedupe.
        errs, env = self._validate(["wds9", "WDS-9"])
        err = next(e for e in errs if e["code"] == "duplicate_reference")
        assert "WDS-9" in err["message"]
        assert err["path"] == "study.gauges_collected.value[1]"
        assert env["value"] == ["wds9", "WDS-9"]

    def test_empty_element_rejected(self):
        errs, _ = self._validate(["WDS-9", "   "])
        err = next(e for e in errs if e["code"] == "empty_value")
        assert err["path"] == "study.gauges_collected.value[1]"

    def test_empty_list_accepted(self):
        errs, env = self._validate([])
        assert errs == []
        assert env["value"] == []

    def test_null_accepted(self):
        errs, env = self._validate(None)
        assert errs == []
        assert env["value"] is None

    def test_non_list_value_is_type_error(self):
        # A string_list field takes a list. A separator-joined string is a
        # plain type mismatch, NOT a multivalue spelling the validator unpacks.
        errs, _ = self._validate("WDS-9; SRI-7")
        assert any(e["code"] == "type_mismatch" for e in errs)


# ---------------------------------------------------------------------------
# Dispatcher level (extractor + reviewer share this path)
# ---------------------------------------------------------------------------

class TestDispatcherReference:
    def test_add_record_canonical_name(self):
        d = _dispatcher()
        res = d.dispatch("add_record",
                         {"fields": {"gauge": {"value": "WDS-9",
                                               "evidence": None}}})
        assert res["status"] == "ok", res
        assert d.extraction_record.records[0]["gauge"]["value"] == "WDS-9"
        assert res["_canonicalisations"] == []

    def test_add_record_alias_canonicalisation_notes_and_event_payload(self):
        d = _dispatcher()
        res = d.dispatch("add_record",
                         {"fields": {"gauge": {"value": "wds9",
                                               "evidence": None}}})
        assert res["status"] == "ok", res
        assert d.extraction_record.records[0]["gauge"]["value"] == "WDS-9"
        assert res["_canonicalisations"] == [
            {"path": "record.relationship_1.gauge", "entered": "wds9",
             "stored": "WDS-9"}]
        assert any("recorded as 'WDS-9'" in n
                   for n in res["canonicalisation_notes"])

    def test_add_record_off_list_rejected(self):
        d = _dispatcher()
        res = d.dispatch("add_record",
                         {"fields": {"gauge": {"value": "Made Up Tool",
                                               "evidence": None}}})
        assert res["status"] == "validation_failed"
        assert any(e["code"] == "off_list_reference" for e in res["errors"])
        assert d.extraction_record.records == []

    def test_add_record_ambiguous_rejected(self):
        d = _dispatcher({"gauge_list": AMBIGUOUS_REFLIST})
        res = d.dispatch("add_record",
                         {"fields": {"gauge": {"value": "shared handle",
                                               "evidence": None}}})
        assert res["status"] == "validation_failed"
        assert any(e["code"] == "ambiguous_reference" for e in res["errors"])

    def test_update_study_list_field(self):
        d = _dispatcher()
        res = d.dispatch("update_study", {"study": {
            "gauges_collected": {"value": ["SRI7", "WDS-9"],
                                 "evidence": None}}})
        assert res["status"] == "ok", res
        assert d.extraction_record.study["gauges_collected"]["value"] == \
            ["SRI-7", "WDS-9"]
        assert res["_canonicalisations"] == [
            {"path": "study.gauges_collected", "entered": "SRI7",
             "stored": "SRI-7"}]
        assert any("recorded as 'SRI-7'" in n
                   for n in res["canonicalisation_notes"])

    def test_reviewer_update_study_list_path_validates_per_element(self):
        # update_study is also the tool the final reviewer dispatches for
        # study-level edits; the same handler validates each element, so an
        # off-list element in the reviewer's list edit is rejected.
        d = _dispatcher()
        d.dispatch("update_study", {"study": {
            "gauges_collected": {"value": ["WDS-9"], "evidence": None}}})
        res = d.dispatch("update_study", {"study": {
            "gauges_collected": {"value": ["WDS-9", "Invented Scale"],
                                 "evidence": None}}})
        assert res["status"] == "validation_failed"
        assert "study.gauges_collected" in res["failed_fields"]
        codes = [e["code"]
                 for e in res["failed_fields"]["study.gauges_collected"]]
        assert "off_list_reference" in codes
        # The pre-edit value stands.
        assert d.extraction_record.study["gauges_collected"]["value"] == \
            ["WDS-9"]

    def test_update_study_duplicate_elements_rejected(self):
        d = _dispatcher()
        res = d.dispatch("update_study", {"study": {
            "gauges_collected": {"value": ["wds9", "WDS-9"],
                                 "evidence": None}}})
        assert res["status"] == "validation_failed"
        codes = [e["code"]
                 for e in res["failed_fields"]["study.gauges_collected"]]
        assert "duplicate_reference" in codes
        assert "gauges_collected" not in d.extraction_record.study

    def test_reviewer_update_record_path_validates_reference(self):
        # update_record is the tool the final reviewer dispatches for record
        # edits; it goes through the same validation, so an off-list edit is
        # rejected.
        d = _dispatcher()
        d.dispatch("add_record",
                   {"fields": {"gauge": {"value": "WDS-9", "evidence": None}}})
        res = d.dispatch("update_record", {
            "record_id": "relationship_1",
            "fields": {"gauge": {"value": "Invented Scale", "evidence": None}}})
        assert res["status"] == "validation_failed"
        assert any(e["code"] == "off_list_reference" for e in res["errors"])
        # The pre-edit value stands.
        assert d.extraction_record.records[0]["gauge"]["value"] == "WDS-9"

    def test_missing_reference_list_is_wiring_error(self):
        d = _dispatcher(reference_lists=None)
        res = d.dispatch("add_record",
                         {"fields": {"gauge": {"value": "WDS-9",
                                               "evidence": None}}})
        assert res["status"] == "validation_failed"
        assert any(e["code"] == "reference_unavailable"
                   for e in res["errors"])


# ---------------------------------------------------------------------------
# Tool schema and checker briefing for string_list reference fields
# ---------------------------------------------------------------------------

class TestToolSchemaReference:
    def test_string_list_reference_schema_is_array(self):
        sch = _field_value_subschema(_list_reference_spec())
        assert sch == {"anyOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "string"}},
        ]}

    def test_string_list_reference_description_mentions_list(self):
        desc = _field_description(_list_reference_spec())
        assert "gauge_list reference list" in desc
        assert "JSON array" in desc

    def test_single_value_reference_description_unchanged(self):
        desc = _field_description(_reference_spec())
        assert "exact name from the gauge_list reference list" in desc

    def test_shipped_gauges_collected_schema(self, config_dir):
        # End-to-end through the shipped config: array type, reference note
        # in the description, no enum constraint.
        from meltiro.template import load_template
        from meltiro.tools import get_tool_definitions
        t = load_template(config_dir / "extraction_template.yaml")
        update_study = next(tool for tool in get_tool_definitions(t)
                            if tool["name"] == "update_study")
        prop = update_study["input_schema"]["properties"]["study"][
            "properties"]["gauges_collected"]
        assert prop["properties"]["value"] == {"anyOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "string"}},
        ]}
        assert "gauge_list reference list" in prop["description"]


class TestCheckerBriefingReference:
    def _text(self, spec, value, checker_user_template_path):
        from meltiro.checker_prompts import build_checker_user_message
        blocks = build_checker_user_message(
            field_path="study.gauges_collected",
            field_spec=spec,
            envelope={"value": value, "evidence": "<q>WDS-9 and SRI-7</q>"},
            identity_context="Summary: ...",
            image_labels=set(),
            user_prompt_path=checker_user_template_path,
        )
        return "\n".join(b.get("text", "") for b in blocks)

    def test_string_list_reference_briefing(self, checker_user_template_path):
        text = self._text(_list_reference_spec(), ["WDS-9", "SRI-7"],
                          checker_user_template_path)
        assert "list of names from the gauge_list reference list" in text
        assert "validator-guaranteed" in text
        # The stored list is rendered as JSON.
        assert '["WDS-9", "SRI-7"]' in text

    def test_single_value_reference_briefing(self, checker_user_template_path):
        text = self._text(_reference_spec(), "WDS-9",
                          checker_user_template_path)
        assert "exact name from the gauge_list reference list" in text
        assert '"WDS-9"' in text
