"""Partial validation: valid fields apply even when siblings fail.

The dispatcher applies what it can and reports back per field which
fields succeeded and which need attention, rather than rejecting a
whole `update_study` call over one bad field and making the extractor
resubmit a 30-field payload to fix it.

The three field-writing tools share that policy: `update_study`,
`add_record`, and `update_record` all validate field by field. A structural
problem with the call itself (a field map that is not an object or is empty,
a missing or unknown record id) addresses no field validly and stays a
whole-call failure.
"""

import pytest

from meltiro.orchestrator import _IdenticalFailureRun
from meltiro.tools import ToolDispatcher
from meltiro.extraction_record import ExtractionRecord

from .conftest import open_initial_check_gate


# Quotes lifted verbatim from the `paper_text` fixture, so an envelope
# carrying one passes the verbatim check.
Q_GAUGE = "<q>WDS-9 was administered</q>"
Q_REMOVAL = "<q>unplanned removal was 1.34</q>"
Q_INDEX = "<q>DI-4 index was used</q>"
# Not in the paper: reaches `quote_not_in_text`.
Q_ABSENT = "<q>no sentence in the paper says this</q>"


def _env(value, evidence):
    return {"value": value, "evidence": evidence}


@pytest.fixture
def dispatcher(synthetic_template, paper_text, image_labels):
    # The initial-check ordering gate is opened here, once: every mutating
    # call is refused until `record_initial_check` has landed, and that rule
    # is not what this file is about (see test_tools.py for it). These tests
    # are about which FIELDS of a call apply when its siblings fail.
    a = open_initial_check_gate(ExtractionRecord())
    return a, ToolDispatcher(a, synthetic_template, paper_text, image_labels)


class TestPartialUpdateStudy:
    def test_all_valid_returns_ok(self, dispatcher):
        extraction_record, d = dispatcher
        # First field that exists in the synthetic template:
        spec = next(iter(d._study_field_specs))
        result = d.dispatch("update_study", {
            "study": {spec: _env(None, None)},
        })
        assert result["status"] == "ok"
        assert result["applied_fields"] == [f"study.{spec}"]
        assert result["failed_fields"] == {}

    def test_one_bad_one_good_returns_partial(self, dispatcher):
        extraction_record, d = dispatcher
        # The first study field is valid; an unknown variable is invalid.
        spec = next(iter(d._study_field_specs))
        result = d.dispatch("update_study", {
            "study": {
                spec: _env(None, None),
                "this_field_does_not_exist": _env("x", "<q>x</q>"),
            },
        })
        assert result["status"] == "partial"
        assert f"study.{spec}" in result["applied_fields"]
        assert "study.this_field_does_not_exist" in result["failed_fields"]
        # The valid field made it into the extraction output.
        assert spec in extraction_record.study

    def test_all_bad_returns_validation_failed(self, dispatcher):
        extraction_record, d = dispatcher
        result = d.dispatch("update_study", {
            "study": {
                "not_real_1": _env("a", "<q>a</q>"),
                "not_real_2": _env("b", "<q>b</q>"),
            },
        })
        assert result["status"] == "validation_failed"
        assert result["applied_fields"] == []
        assert len(result["failed_fields"]) == 2
        # Extraction output unchanged: no field written, and the reserved
        # scope-note key still holds its initial null.
        assert extraction_record.study == {"notes": None}

    def test_block_level_type_error_kills_whole_call(self, dispatcher):
        extraction_record, d = dispatcher
        # `study` as a string instead of dict; structural problem.
        result = d.dispatch("update_study", {"study": "not an object"})
        assert result["status"] == "validation_failed"
        assert any(
            e["code"] == "type_mismatch" for e in result["errors"]
        )
        # Nothing applied; the reserved scope-note key is untouched.
        assert extraction_record.study == {"notes": None}

    def test_errors_flattened_into_one_list(self, dispatcher):
        # `result["errors"]` carries every error (call-level and
        # per-field), so a consumer that scans the flat list needs no
        # grouping; `failed_fields` is the grouped view of the same set.
        _, d = dispatcher
        result = d.dispatch("update_study", {
            "study": {"not_real": _env("a", "<q>a</q>")},
        })
        codes = [e["code"] for e in result["errors"]]
        assert "unknown_field" in codes


def _seed_record(d, outcome_category="Failure state"):
    """Add one valid record and return its id."""
    result = d.dispatch("add_record", {"fields": {
        "gauge": _env("WDS-9", Q_GAUGE),
        "outcome_variable": _env("Unplanned removal", Q_REMOVAL),
        "outcome_category": _env(outcome_category, Q_REMOVAL),
    }})
    assert result["status"] == "ok", result
    return result["applied_changes"]["record_id"]


class TestPartialUpdateRecord:
    def test_one_bad_one_good_returns_partial(self, dispatcher):
        record, d = dispatcher
        rid = _seed_record(d)
        result = d.dispatch("update_record", {
            "record_id": rid,
            "fields": {
                "effect_size": _env("1.34", Q_REMOVAL),
                # Hard enum: not one of the template's options.
                "outcome_category": _env("Invented category", Q_REMOVAL),
            },
        })
        assert result["status"] == "partial"
        assert result["applied_fields"] == [f"record.{rid}.effect_size"]
        assert list(result["failed_fields"]) == [
            f"record.{rid}.outcome_category"]
        assert result["applied_changes"] == {
            "record_id": rid, "record_fields": ["effect_size"]}
        # Diffs cover the applied field only.
        assert result["_field_diffs"] == {
            f"record.{rid}.effect_size": {"before": None, "after": "1.34"},
        }
        # The record carries the applied field and keeps its prior value on
        # the rejected one.
        stored = record.records[0]
        assert stored["effect_size"]["value"] == "1.34"
        assert stored["outcome_category"]["value"] == "Failure state"

    def test_all_bad_returns_validation_failed(self, dispatcher):
        record, d = dispatcher
        rid = _seed_record(d)
        result = d.dispatch("update_record", {
            "record_id": rid,
            "fields": {
                "effect_size": _env("1.34", Q_ABSENT),
                "not_a_record_field": _env("x", Q_REMOVAL),
            },
        })
        assert result["status"] == "validation_failed"
        assert result["applied_fields"] == []
        assert result["applied_changes"] == {}
        assert set(result["failed_fields"]) == {
            f"record.{rid}.effect_size",
            f"record.{rid}.not_a_record_field",
        }
        # Nothing landed.
        assert "effect_size" not in record.records[0]
        assert "not_a_record_field" not in record.records[0]

    def test_gate_warning_reads_the_merged_post_update_view(self, dispatcher):
        # The gate rule pairs index_tariff with outcome_category, which this
        # call does not touch: the warning can only come from the merged view
        # (the record's stored fields plus the fields that just applied).
        _, d = dispatcher
        rid = _seed_record(d, outcome_category="Failure state")
        result = d.dispatch("update_record", {
            "record_id": rid,
            "fields": {
                "index_tariff": _env("DI-4", Q_INDEX),
                "outcome_variable": _env("Unplanned removal", Q_ABSENT),
            },
        })
        assert result["status"] == "partial"
        assert [w["path"] for w in result["warnings"]] == [
            f"record.{rid}.index_tariff"]
        assert result["warnings"][0]["code"] == "category_gate"

    def test_rejected_field_raises_no_gate_warning(self, dispatcher):
        # The mirror of the canonicalisation rule on update_study: a value
        # that did not apply never reaches the view the gates read, so it
        # cannot raise a warning about itself.
        _, d = dispatcher
        rid = _seed_record(d, outcome_category="Failure state")
        result = d.dispatch("update_record", {
            "record_id": rid,
            "fields": {
                "index_tariff": _env("DI-4", Q_ABSENT),
                "effect_size": _env("1.34", Q_REMOVAL),
            },
        })
        assert result["status"] == "partial"
        assert result["warnings"] == []


class TestStructuralUpdateRecordFailures:
    """Call-level structural problems address no field validly, so they stay
    all-or-nothing: `validation_failed`, no partial semantics."""

    def test_missing_record_id(self, dispatcher):
        _, d = dispatcher
        result = d.dispatch("update_record", {
            "fields": {"effect_size": _env("1.34", Q_REMOVAL)}})
        assert result["status"] == "validation_failed"
        assert result["failed_fields"] == {}
        assert [e["code"] for e in result["errors"]] == ["missing_field"]

    def test_non_string_record_id(self, dispatcher):
        _, d = dispatcher
        result = d.dispatch("update_record", {
            "record_id": 7,
            "fields": {"effect_size": _env("1.34", Q_REMOVAL)}})
        assert result["status"] == "validation_failed"
        assert result["failed_fields"] == {}

    def test_unknown_record_id(self, dispatcher):
        record, d = dispatcher
        _seed_record(d)
        result = d.dispatch("update_record", {
            "record_id": "relationship_99",
            "fields": {"effect_size": _env("1.34", Q_REMOVAL)}})
        assert result["status"] == "validation_failed"
        assert [e["code"] for e in result["errors"]] == ["unknown_record"]
        assert result["failed_fields"] == {}
        assert "effect_size" not in record.records[0]

    def test_fields_not_an_object(self, dispatcher):
        _, d = dispatcher
        rid = _seed_record(d)
        result = d.dispatch("update_record",
                            {"record_id": rid, "fields": ["effect_size"]})
        assert result["status"] == "validation_failed"
        assert [e["code"] for e in result["errors"]] == ["type_mismatch"]
        assert result["failed_fields"] == {}

    def test_empty_fields_object(self, dispatcher):
        _, d = dispatcher
        rid = _seed_record(d)
        result = d.dispatch("update_record", {"record_id": rid, "fields": {}})
        assert result["status"] == "validation_failed"
        assert [e["code"] for e in result["errors"]] == ["missing_fields"]
        assert result["failed_fields"] == {}


class TestPartialAddRecord:
    def test_record_is_created_with_the_valid_subset(self, dispatcher):
        record, d = dispatcher
        result = d.dispatch("add_record", {"fields": {
            "gauge": _env("WDS-9", Q_GAUGE),
            "outcome_variable": _env("Unplanned removal", Q_REMOVAL),
            "outcome_category": _env("Invented category", Q_REMOVAL),
            "effect_size": _env("1.34", Q_ABSENT),
        }})
        assert result["status"] == "partial"
        rid = result["applied_changes"]["record_id"]
        assert rid == "relationship_1"
        assert result["applied_changes"]["record_fields"] == [
            "gauge", "outcome_variable"]
        assert result["applied_fields"] == [
            f"record.{rid}.gauge", f"record.{rid}.outcome_variable"]
        # The failures are reported against the id the call just returned, so
        # the extractor can resubmit them with update_record.
        assert set(result["failed_fields"]) == {
            f"record.{rid}.outcome_category", f"record.{rid}.effect_size"}
        assert all(e["path"].startswith(f"record.{rid}.")
                   for errs in result["failed_fields"].values()
                   for e in errs)
        assert result["_field_diffs"] == {
            f"record.{rid}.gauge": {"before": None, "after": "WDS-9"},
            f"record.{rid}.outcome_variable": {
                "before": None, "after": "Unplanned removal"},
        }
        # The record exists, holding the valid subset and nothing else. The
        # two reserved keys (the engine-assigned id, the scope note) are
        # minted with every record and are not fields.
        assert len(record.records) == 1
        assert set(record.records[0]) == {
            "record_id", "notes", "gauge", "outcome_variable"}
        assert record.records[0]["notes"] is None

    def test_all_bad_creates_no_record(self, dispatcher):
        record, d = dispatcher
        result = d.dispatch("add_record", {"fields": {
            "gauge": _env("WDS-9", Q_ABSENT),
            "not_a_record_field": _env("x", Q_GAUGE),
        }})
        assert result["status"] == "validation_failed"
        assert result["applied_fields"] == []
        assert result["applied_changes"] == {}
        assert record.records == []
        # No id was minted, so the failure paths keep the placeholder prefix.
        assert set(result["failed_fields"]) == {
            "record.<new>.gauge", "record.<new>.not_a_record_field"}

    def test_a_failed_add_consumes_no_record_id(self, dispatcher):
        # The id counter is monotonic and must not advance on a call that
        # created nothing: the next successful add is still relationship_1.
        _, d = dispatcher
        d.dispatch("add_record", {"fields": {
            "gauge": _env("WDS-9", Q_ABSENT)}})
        result = d.dispatch("add_record", {"fields": {
            "gauge": _env("WDS-9", Q_GAUGE)}})
        assert result["status"] == "ok"
        assert result["applied_changes"]["record_id"] == "relationship_1"


class TestStructuralAddRecordFailures:
    def test_fields_not_an_object(self, dispatcher):
        record, d = dispatcher
        result = d.dispatch("add_record", {"fields": "gauge"})
        assert result["status"] == "validation_failed"
        assert [e["code"] for e in result["errors"]] == ["type_mismatch"]
        assert result["failed_fields"] == {}
        assert record.records == []

    def test_empty_fields_object(self, dispatcher):
        record, d = dispatcher
        result = d.dispatch("add_record", {"fields": {}})
        assert result["status"] == "validation_failed"
        assert [e["code"] for e in result["errors"]] == ["missing_fields"]
        assert result["failed_fields"] == {}
        assert record.records == []


class TestRepeatedFailureGuardCountsPartialAsProgress:
    """The guard stops a loop that keeps re-submitting an identically failing
    call. A partial record call applied something, so it is progress and
    resets the run, exactly as a partial `update_study` does."""

    def test_partial_add_record_resets_the_run(self, dispatcher):
        _, d = dispatcher
        run = _IdenticalFailureRun(limit=2)
        failed = d.dispatch("add_record", {"fields": {}})
        assert failed["status"] == "validation_failed"
        assert run.record("add_record", failed) is None
        assert run.count == 1

        partial = d.dispatch("add_record", {"fields": {
            "gauge": _env("WDS-9", Q_GAUGE),
            "effect_size": _env("1.34", Q_ABSENT),
        }})
        assert partial["status"] == "partial"
        assert run.record("add_record", partial) is None
        assert run.count == 0

    def test_partial_update_record_resets_the_run(self, dispatcher):
        _, d = dispatcher
        rid = _seed_record(d)
        run = _IdenticalFailureRun(limit=2)
        failed = d.dispatch("update_record", {"record_id": rid, "fields": {}})
        assert failed["status"] == "validation_failed"
        assert run.record("update_record", failed) is None
        assert run.count == 1

        partial = d.dispatch("update_record", {"record_id": rid, "fields": {
            "effect_size": _env("1.34", Q_REMOVAL),
            "statistical_method": _env("Logistic regression", Q_ABSENT),
        }})
        assert partial["status"] == "partial"
        assert run.record("update_record", partial) is None
        assert run.count == 0
