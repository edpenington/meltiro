"""Tests for the importable single-field entry point `validate_value`.

This is the importable consumer surface: a caller validates ONE field the
moment a human reviewer saves it, with a producer kind, without an evidence
envelope or paper text. The LLM pipeline's `validate_envelope` must route
through the same underlying logic, so a parity test pins that the two agree for
the LLM path.
"""

import pytest

from meltiro.extraction_record import ROLE_EXTRACTOR
from meltiro.validators import (
    ValidationResult,
    validate_envelope,
    validate_extraction_output,
    validate_value,
)


def _field(variable, field_type="free_text", options=None, allow_other=False,
           canonical_reference=None, evidence="required"):
    return {
        "variable": variable,
        "field_type": field_type,
        "options": options,
        "allow_other": allow_other,
        "canonical_reference": canonical_reference,
        "evidence": evidence,
        "description": "",
        "extraction_instruction": None,
    }


_GAUGE_LISTS = {
    "gauge_list": [
        {"tool_name": "WDS-9", "aliases": ["WDS9"]},
        {"tool_name": "SRI-7"},
    ]
}


class TestProducerKind:
    def test_unknown_producer_kind_raises(self):
        with pytest.raises(ValueError) as excinfo:
            validate_value(_field("x"), "v", producer_kind="robot")
        assert "producer_kind" in str(excinfo.value)

    def test_returns_validation_result(self):
        r = validate_value(_field("x"), "hello")
        assert isinstance(r, ValidationResult)
        assert r.ok is True
        assert r.errors == []


class TestHumanEvidencePolicy:
    def test_evidence_never_demanded_of_human(self):
        # An evidence-required field, human producer, non-null value, no
        # evidence supplied: must NOT raise evidence_required.
        r = validate_value(_field("aim", evidence="required"),
                           "the study aim", producer_kind="human")
        assert r.ok is True
        assert r.errors == []

    def test_no_paper_text_skips_quote_check(self):
        # A human volunteers an evidence quote but supplies no paper text:
        # the quote-checking layer does not run, so a quote that appears in
        # no text is accepted rather than flagged quote_not_in_text.
        r = validate_value(
            _field("aim"), "the study aim", producer_kind="human",
            evidence="<q>a quote that is in no paper</q>", paper_text=None)
        assert r.ok is True
        assert r.errors == []

    def test_volunteered_evidence_quote_checked_when_paper_available(self):
        # With paper text available, a human's volunteered quote IS checked.
        paper = "The study enrolled 348 units."
        good = validate_value(
            _field("aim"), "enrolment", producer_kind="human",
            evidence="<q>enrolled 348 units</q>", paper_text=paper)
        assert good.ok is True

        bad = validate_value(
            _field("aim"), "enrolment", producer_kind="human",
            evidence="<q>enrolled 999 units</q>", paper_text=paper)
        assert bad.ok is False
        assert any(e["code"] == "quote_not_in_text" for e in bad.errors)


class TestLlmEvidencePolicy:
    """The half of the contract that needs no paper, and the half that does.

    An LLM producer is held to the template's `evidence:` flag. Whether a
    value carries evidence at all is decided by the field; whether the
    evidence quoted is real is decided by the paper. The first question is
    always answerable, so it is always asked.
    """

    def test_evidence_demanded_of_llm_without_paper_text(self):
        r = validate_value(_field("aim", evidence="required"),
                           "the study aim", producer_kind="llm")
        assert r.ok is False
        assert [e["code"] for e in r.errors] == ["evidence_required"]

    def test_optional_evidence_field_is_not_demanded(self):
        r = validate_value(_field("aim", evidence="optional"),
                           "the study aim", producer_kind="llm")
        assert r.ok is True

    def test_a_null_value_needs_no_evidence(self):
        r = validate_value(_field("aim", evidence="required"), None,
                           producer_kind="llm")
        assert r.ok is True

    def test_no_verbatim_verdict_without_the_paper(self):
        # Nothing to check the quote against: the presence half is satisfied
        # and no verdict is reached on whether the quote is real. Reporting
        # one would fail every quote in a file for the absence of the paper.
        r = validate_value(
            _field("aim", evidence="required"), "the study aim",
            producer_kind="llm",
            evidence="<q>a quote that is in no paper</q>", paper_text=None)
        assert r.ok is True
        assert r.errors == []

    def test_no_label_verdict_without_the_paper(self):
        r = validate_value(
            _field("aim", evidence="required"), "the study aim",
            producer_kind="llm", evidence="<img>table_01</img>",
            paper_text=None)
        assert r.ok is True

    def test_the_verbatim_verdict_returns_with_the_paper(self):
        r = validate_value(
            _field("aim", evidence="required"), "the study aim",
            producer_kind="llm",
            evidence="<q>a quote that is in no paper</q>",
            paper_text="The study enrolled 348 units.")
        assert r.ok is False
        assert any(e["code"] == "quote_not_in_text" for e in r.errors)

    def test_a_malformed_tag_is_still_structural_without_the_paper(self):
        # Tag structure is not a question about the paper, so withholding the
        # source-dependent verdicts must not take this one with them.
        r = validate_value(
            _field("aim", evidence="required"), "the study aim",
            producer_kind="llm", evidence="<q>unclosed quote",
            paper_text=None)
        assert r.ok is False


class TestValueLevelChecksRunForHuman:
    def test_type_mismatch_reported(self):
        r = validate_value(_field("n", field_type="integer"), "not a number",
                           producer_kind="human")
        assert r.ok is False
        assert any(e["code"] == "type_mismatch" for e in r.errors)

    def test_categorical_canonicalised(self):
        spec = _field("cat", field_type="categorical",
                      options=["Service life", "Cost or resource use"])
        r = validate_value(spec, "service life", producer_kind="human")
        assert r.ok is True
        assert r.value == "Service life"

    def test_hard_enum_off_list_rejected(self):
        spec = _field("cat", field_type="categorical",
                      options=["A", "B"])
        r = validate_value(spec, "C", producer_kind="human")
        assert r.ok is False
        assert any(e["code"] == "invalid_option" for e in r.errors)

    def test_allow_other_free_text_accepted(self):
        spec = _field("pub", field_type="categorical",
                      options=["Academic paper"], allow_other=True)
        r = validate_value(spec, "A working paper", producer_kind="human")
        assert r.ok is True
        assert r.value == "A working paper"

    def test_reference_exact_match_canonicalises_spelling(self):
        spec = _field("gauge", canonical_reference="gauge_list",
                      field_type="string")
        r = validate_value(spec, "wds-9", _GAUGE_LISTS, "human")
        assert r.ok is True
        assert r.value == "WDS-9"
        # Exact (case-only) match canonicalises silently, no alias event.
        assert r.canonicalisations == []

    def test_reference_alias_match_records_canonicalisation(self):
        spec = _field("gauge", canonical_reference="gauge_list",
                      field_type="string")
        r = validate_value(spec, "WDS9", _GAUGE_LISTS, "human")
        assert r.ok is True
        assert r.value == "WDS-9"
        assert r.canonicalisations == [
            {"path": "value", "entered": "WDS9", "stored": "WDS-9"}]

    def test_reference_off_list_rejected(self):
        spec = _field("gauge", canonical_reference="gauge_list",
                      field_type="string")
        r = validate_value(spec, "MADE-UP", _GAUGE_LISTS, "human")
        assert r.ok is False
        assert any(e["code"] == "off_list_reference" for e in r.errors)

    def test_reference_list_missing_is_loud(self):
        # A reference field but no reference lists supplied: loud config
        # error, matching the LLM path.
        spec = _field("gauge", canonical_reference="gauge_list",
                      field_type="string")
        r = validate_value(spec, "WDS-9", reference_lists=None,
                           producer_kind="human")
        assert r.ok is False
        assert any(e["code"] == "reference_unavailable" for e in r.errors)


# The worked config's gate for index_tariff, as the loaded template exposes
# it. The engine names no gated field itself, so the caller supplies the gates.
_INDEX_GATE = [
    {"when_field": "outcome_category", "field": "index_tariff",
     "allowed_values": ["Service life"]},
]


class TestGateRules:
    def test_gate_warning_for_this_field(self):
        # index_tariff set while outcome_category is not Service life:
        # a category-gate WARNING (not an error) for index_tariff.
        siblings = {"outcome_category": "Cost or resource use",
                    "index_tariff": "DI-4"}
        r = validate_value(
            _field("index_tariff"), "DI-4", producer_kind="human",
            sibling_values=siblings, gates=_INDEX_GATE)
        assert r.ok is True  # warnings are not failures
        assert any(w["code"] == "category_gate" for w in r.warnings)

    def test_no_sibling_values_no_gate_rules(self):
        r = validate_value(_field("index_tariff"), "DI-4",
                           producer_kind="human", gates=_INDEX_GATE)
        assert r.warnings == []

    def test_no_gates_no_warnings_even_with_siblings(self):
        # Gate warnings need the template's gates too: sibling values alone,
        # with no gates supplied, produce none.
        siblings = {"outcome_category": "Cost or resource use",
                    "index_tariff": "DI-4"}
        r = validate_value(
            _field("index_tariff"), "DI-4", producer_kind="human",
            sibling_values=siblings)
        assert r.warnings == []


class TestLLMParityWithEnvelope:
    """The LLM path must route through the same logic: validate_value with
    producer_kind='llm' agrees with validate_envelope for the same inputs."""

    def _both(self, spec, value, evidence, paper_text, reference_lists=None):
        from meltiro.reference_lists import build_reference_index
        ref_index = None
        if spec.get("canonical_reference") and reference_lists:
            ref_index = build_reference_index(
                reference_lists[spec["canonical_reference"]])
        env = {"value": value, "evidence": evidence}
        env_errors = validate_envelope(
            env, spec, paper_text, set(), path_prefix="study.f",
            reference_index=ref_index)
        vv = validate_value(
            spec, value, reference_lists, "llm", evidence=evidence,
            paper_text=paper_text, path="study.f", reference_index=ref_index)
        return env_errors, vv, env

    def test_passing_case_agrees(self):
        paper = "The study enrolled 348 units."
        env_errors, vv, env = self._both(
            _field("aim"), "enrolment",
            "<q>enrolled 348 units</q>", paper)
        assert env_errors == vv.errors == []
        assert env["value"] == vv.value

    def test_evidence_required_failure_agrees(self):
        # LLM path: an evidence-required field with a non-null value and no
        # quote must fail identically on both surfaces.
        env_errors, vv, _ = self._both(
            _field("aim", evidence="required"), "some value", "", "paper")
        assert env_errors == vv.errors
        assert any(e["code"] == "evidence_required" for e in vv.errors)

    def test_reference_canonicalisation_agrees(self):
        env_errors, vv, env = self._both(
            _field("gauge", canonical_reference="gauge_list",
                   field_type="string", evidence="optional"),
            "WDS9", "", "paper", reference_lists=_GAUGE_LISTS)
        assert env_errors == vv.errors == []
        assert env["value"] == vv.value == "WDS-9"


class TestNullByType:
    """The runtime validator's documented contract: null is allowed for every
    type except boolean. A boolean answers a yes/no question and null is
    neither, so a null boolean is a type error (the OpenAI-compatible tool path
    runs strict:false and would otherwise store it)."""

    def test_null_boolean_rejected(self):
        r = validate_value(_field("flag", field_type="boolean"), None,
                           producer_kind="human")
        assert r.ok is False
        assert any(e["code"] == "type_mismatch" for e in r.errors)

    def test_null_boolean_error_entry_shape(self):
        # The error entry matches the existing {path, code, message} shape and
        # reuses the existing type_mismatch code. The path is the field path
        # with the runtime validator's `.value` suffix, like every other type
        # error.
        r = validate_value(_field("flag", field_type="boolean"), None,
                           producer_kind="human", path="initial_check.flag")
        [err] = [e for e in r.errors if e["code"] == "type_mismatch"]
        assert set(err) == {"path", "code", "message"}
        assert err["path"] == "initial_check.flag.value"
        assert "boolean" in err["message"].lower()
        assert "null" in err["message"].lower()

    def test_null_allowed_for_scalar_and_list_types(self):
        for ft in ("free_text", "string", "integer", "number", "year",
                   "date", "string_list"):
            r = validate_value(_field("f", field_type=ft), None,
                               producer_kind="human")
            assert r.ok is True, f"null should be allowed for {ft}"
            assert r.errors == []

    def test_null_allowed_for_categorical(self):
        r = validate_value(
            _field("cat", field_type="categorical", options=["A", "B"]),
            None, producer_kind="human")
        assert r.ok is True
        assert r.errors == []


class TestBatchNullBoolean:
    """`meltiro validate` (validate_extraction_output) must surface a stored
    null boolean from an older run as a loud failure, not crash on it."""

    def test_stored_null_boolean_reported_not_crashed(self, synthetic_template):
        # text_readable / figure_tables_included are boolean initial_check
        # fields. A null in one of them (storable via the strict:false
        # OpenAI-compatible tool path) must be reported.
        output = {
            "study": {},
            "records": [],
            # Role-keyed, as a real extraction output is: the batch validator
            # descends through the role before it reaches a variable.
            "initial_check": {ROLE_EXTRACTOR: {
                "text_readable": None,
                "figure_tables_included": True,
                "expected_relationships": 3,
            }},
            "quality_check": {},
        }
        failures, warnings = validate_extraction_output(
            synthetic_template, output)
        # Reported (loud), not a crash. The runtime validator suffixes the
        # field path with `.value` on a type error.
        null_bool = [f for f in failures
                     if f["path"].startswith("initial_check.extractor.text_readable")]
        assert null_bool, "null boolean should be reported as a failure"
        assert null_bool[0]["code"] == "type_mismatch"
        # The valid boolean and integer beside it do not spuriously fail.
        assert not any(
            f["path"].startswith(
                "initial_check.extractor.figure_tables_included")
            or f["path"].startswith(
                "initial_check.extractor.expected_relationships")
            for f in failures)
