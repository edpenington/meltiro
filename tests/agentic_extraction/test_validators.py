"""Tests for the deterministic per-field validators."""

import pytest

from meltiro.validators import (
    validate_envelope,
    validate_gate_rules,
)


PAPER = "The WDS-9 was used to assess fatigue severity in 348 units."


def _spec(variable, field_type, options=None, allow_other=False):
    return {
        "variable": variable,
        "field_type": field_type,
        "options": options,
        "allow_other": allow_other,
        "description": "",
        "extraction_instruction": None,
    }


class TestValueType:
    def test_categorical_pass(self):
        errs = validate_envelope(
            {"value": "Yes", "evidence": "<q>The WDS-9 was used</q>"},
            _spec("foo", "categorical", options=["Yes", "No"]),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        assert errs == []

    def test_categorical_rejects_unknown(self):
        errs = validate_envelope(
            {"value": "Maybe", "evidence": "<q>The WDS-9</q>"},
            _spec("foo", "categorical", options=["Yes", "No"]),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        codes = [e["code"] for e in errs]
        assert "invalid_option" in codes

    def test_integer_pass(self):
        errs = validate_envelope(
            {"value": 348, "evidence": "<q>348 units</q>"},
            _spec("sample_size", "integer"),
            paper_text=PAPER, image_labels=set(),
            path_prefix="study.sample_size",
        )
        assert errs == []

    def test_integer_rejects_bool(self):
        # Python: True is technically int, but we reject for clarity.
        errs = validate_envelope(
            {"value": True, "evidence": "<q>348 units</q>"},
            _spec("sample_size", "integer"),
            paper_text=PAPER, image_labels=set(),
            path_prefix="study.sample_size",
        )
        assert any(e["code"] == "type_mismatch" for e in errs)

    def test_year_out_of_range(self):
        errs = validate_envelope(
            {"value": 3024, "evidence": "<q>348 units</q>"},
            _spec("year", "year"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.year",
        )
        codes = [e["code"] for e in errs]
        assert "year_out_of_range" in codes

    def test_year_integral_float_accepted_and_coerced(self):
        # A model may round-trip a year as 2019.0; accept it and coerce the
        # stored value to the int 2019 so it agrees with the integer schema.
        env = {"value": 2019.0, "evidence": "<q>348 units</q>"}
        errs = validate_envelope(
            env, _spec("pub_year", "year"),
            paper_text=PAPER, image_labels=set(),
            path_prefix="study.pub_year",
        )
        assert errs == []
        assert env["value"] == 2019
        assert isinstance(env["value"], int)

    def test_year_non_integral_float_rejected(self):
        env = {"value": 2019.5, "evidence": "<q>348 units</q>"}
        errs = validate_envelope(
            env, _spec("pub_year", "year"),
            paper_text=PAPER, image_labels=set(),
            path_prefix="study.pub_year",
        )
        assert any(e["code"] == "type_mismatch" for e in errs)
        # The invalid value is left untouched, not silently coerced.
        assert env["value"] == 2019.5

    def test_year_plain_int_unchanged(self):
        env = {"value": 2019, "evidence": "<q>348 units</q>"}
        errs = validate_envelope(
            env, _spec("pub_year", "year"),
            paper_text=PAPER, image_labels=set(),
            path_prefix="study.pub_year",
        )
        assert errs == []
        assert env["value"] == 2019

    def test_year_bool_rejected(self):
        # bool is a subclass of int but is not a valid year.
        env = {"value": True, "evidence": "<q>348 units</q>"}
        errs = validate_envelope(
            env, _spec("pub_year", "year"),
            paper_text=PAPER, image_labels=set(),
            path_prefix="study.pub_year",
        )
        assert any(e["code"] == "type_mismatch" for e in errs)

    def test_date_format(self):
        errs = validate_envelope(
            {"value": "Jan 2020", "evidence": "<q>The WDS-9</q>"},
            _spec("d", "date"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.d",
        )
        assert any(e["code"] == "date_format" for e in errs)

    def test_date_pass(self):
        errs = validate_envelope(
            {"value": "2020-01-15", "evidence": "<q>The WDS-9</q>"},
            _spec("d", "date"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.d",
        )
        assert errs == []

    def test_null_value_skips_type_check(self):
        errs = validate_envelope(
            {"value": None, "evidence": None},
            _spec("foo", "categorical", options=["Yes", "No"]),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        assert errs == []

    def test_string_list_pass(self):
        errs = validate_envelope(
            {"value": ["a", "b"], "evidence": "<q>The WDS-9</q>"},
            _spec("foo", "string_list"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        assert errs == []

    def test_string_list_rejects_non_list(self):
        errs = validate_envelope(
            {"value": "not-a-list", "evidence": "<q>The WDS-9</q>"},
            _spec("foo", "string_list"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        assert any(e["code"] == "type_mismatch" for e in errs)


class TestEnvelopeStructure:
    def test_missing_value_key(self):
        errs = validate_envelope(
            {"evidence": "<q>x</q>"},
            _spec("foo", "string"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        assert any(e["code"] == "missing_key" for e in errs)

    def test_not_an_envelope(self):
        errs = validate_envelope(
            "just a string",
            _spec("foo", "free_text"),
            paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
        )
        assert any(e["code"] == "not_an_envelope" for e in errs)


# The gate rules the worked config declares, as the loaded template exposes
# them (template["gates"]). The engine itself names none of these fields; they
# come from the template, so the tests supply them explicitly.
GAUGE_GATES = [
    {"when_field": "outcome_category", "field": "index_tariff",
     "allowed_values": ["Service life"]},
    {"when_field": "outcome_category", "field": "cost_source",
     "allowed_values": ["Cost or resource use"]},
    {"when_field": "outcome_category", "field": "failure_state_definition",
     "allowed_values": ["Failure state"]},
]


class TestGateRules:
    def test_index_tariff_only_for_service_life(self):
        rel = {
            "outcome_category": {"value": "Cost or resource use", "evidence": ["x"],
                                 "source": "y"},
            "index_tariff": {"value": "DI-4", "evidence": ["x"],
                             "source": "y"},
        }
        warnings = validate_gate_rules(rel, GAUGE_GATES)
        assert any(w["path"] == "index_tariff" for w in warnings)

    def test_index_tariff_for_service_life_passes(self):
        rel = {
            "outcome_category": {"value": "Service life", "evidence": ["x"],
                                 "source": "y"},
            "index_tariff": {"value": "DI-4", "evidence": ["x"],
                             "source": "y"},
        }
        warnings = validate_gate_rules(rel, GAUGE_GATES)
        assert all(w["path"] != "index_tariff" for w in warnings)

    def test_cost_source_for_cost_or_resource_use_passes(self):
        rel = {
            "outcome_category": {"value": "Cost or resource use",
                                 "evidence": ["x"], "source": "y"},
            "cost_source": {"value": "Fleet Service Costs",
                            "evidence": ["x"], "source": "y"},
        }
        warnings = validate_gate_rules(rel, GAUGE_GATES)
        assert all(w["path"] != "cost_source" for w in warnings)

    def test_failure_state_definition_only_for_failure_state(self):
        rel = {
            "outcome_category": {"value": "Service life",
                                 "evidence": ["x"], "source": "y"},
            "failure_state_definition": {"value": "Unplanned removal",
                                         "evidence": ["x"], "source": "y"},
        }
        warnings = validate_gate_rules(rel, GAUGE_GATES)
        assert any(w["path"] == "failure_state_definition" for w in warnings)

    def test_no_category_no_warnings(self):
        rel = {
            "outcome_category": {"value": None, "evidence": None,
                                 "source": None},
            "index_tariff": {"value": "DI-4", "evidence": ["x"],
                             "source": "y"},
        }
        warnings = validate_gate_rules(rel, GAUGE_GATES)
        assert warnings == []

    def test_no_gates_declared_no_warnings(self):
        # A template that declares no gates (an empty gate list) produces no
        # warnings, even for a combination the worked config would flag. The
        # engine holds no gate rules of its own.
        rel = {
            "outcome_category": {"value": "Cost or resource use",
                                 "evidence": ["x"], "source": "y"},
            "index_tariff": {"value": "DI-4", "evidence": ["x"],
                             "source": "y"},
        }
        assert validate_gate_rules(rel, []) == []

    def test_gate_message_names_only_template_fields(self):
        # The warning text is built from the gate's own field names and
        # allowed values, nothing hardcoded.
        rel = {
            "band": {"value": "low"},
            "premium_feature": {"value": "on"},
        }
        gates = [{"when_field": "band", "field": "premium_feature",
                  "allowed_values": ["high", "premium"]}]
        warnings = validate_gate_rules(rel, gates)
        assert len(warnings) == 1
        w = warnings[0]
        assert w["path"] == "premium_feature"
        assert w["code"] == "category_gate"
        assert "premium_feature" in w["message"]
        assert "band" in w["message"]
        assert "high, premium" in w["message"]
