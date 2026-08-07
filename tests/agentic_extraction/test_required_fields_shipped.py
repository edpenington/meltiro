"""A required field can reach the shipped output null, and the run says so.

The extractor cannot finish with a template-declared `required: true` field
unset: its `mark_complete` gates on exactly that and refuses until every one
carries a value. The REVIEWER's `mark_complete` is not gated, deliberately —
it saw the whole assembled output and has no second chance to terminate, so
making its exit contingent on a completeness check would let a disagreement
about completeness strand a finished run.

The consequence is a state the pipeline can genuinely reach: a reviewer edit
lands after the extractor's gate, nulls a required field, and the run finalises
`complete`. Nothing downstream catches it on its own. `validation_passed` in
the run log is `status == "complete"` and never a re-validation, and
`validate_extraction_output` sweeps the values an output STORES for legality,
which gives a null nothing to fault. So the artefact has to disclose it, and
these pin that it does: `meta.warnings` carries an entry naming every required
field shipping null, on the runs where the claim means something.
"""

import pytest

from meltiro.bundle import load_bundle
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.run_log import load_log
from meltiro.template import load_template
from meltiro.validators import (
    missing_required_fields, validate_extraction_output)


REQUIRED_RECORD_FIELD = "gauge"
Q_GAUGE = "<q>WDS-9 was administered</q>"


def _orch(config_dir, bundle_minimal_dir, tmp_path):
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_minimal_dir),
        tmp_path / "runs",
        extractor_model="claude-opus-4-8",
        review_model=None,
        max_checks_per_field=0, final_review=False,
        api_key="x")
    orch.prepare_new_session()
    return orch


def _add_record(orch, *, gauge_value):
    """One record carrying the fixture's required record fields."""
    orch.extraction_record.add_record({
        REQUIRED_RECORD_FIELD: {
            "value": gauge_value, "evidence": Q_GAUGE, "notes": None},
        "outcome_variable": {
            "value": "unplanned removal", "evidence": Q_GAUGE, "notes": None},
        "outcome_category": {
            "value": "Safety", "evidence": Q_GAUGE, "notes": None},
    }, orch.template["record_entity"]["singular"])


def _warnings(orch):
    return [w for w in orch.session.meta["warnings"]
            if w.startswith("required-fields-null")]


class TestTheSweepItself:
    """`missing_required_fields` answers PRESENCE, which the value sweep
    beside it does not."""

    def test_a_null_required_field_is_reported(self, config_dir):
        template = load_template(
            load_config_bundle(config_dir).template_path)
        output = {"study": {}, "records": [
            {"record_id": "relationship_1",
             REQUIRED_RECORD_FIELD: {"value": None, "evidence": None}},
        ]}
        assert f"record.relationship_1.{REQUIRED_RECORD_FIELD}" in \
            missing_required_fields(template, output)

    def test_an_absent_required_field_is_reported_too(self, config_dir):
        # Absent and null are the same answer to "is it here": a field the
        # output never carried is no more answered than one set to null.
        template = load_template(
            load_config_bundle(config_dir).template_path)
        missing = missing_required_fields(
            template, {"study": {}, "records": [{"record_id": "r1"}]})
        assert f"record.r1.{REQUIRED_RECORD_FIELD}" in missing

    def test_every_missing_field_is_named_not_counted(self, config_dir):
        # A reader deciding whether to use the extraction needs to know WHICH
        # fields are unanswered, so the sweep returns paths.
        template = load_template(
            load_config_bundle(config_dir).template_path)
        missing = missing_required_fields(
            template, {"study": {}, "records": [{"record_id": "r1"}]})
        assert len(missing) > 1
        assert missing == sorted(missing)
        assert all(p.startswith(("study.", "record.")) for p in missing)

    def test_the_value_sweep_does_not_catch_it(self, config_dir):
        # The reason this function exists rather than being folded into
        # validate_extraction_output: a null required field stores nothing
        # illegal, so the value sweep has nothing to fault and reports clean.
        cb = load_config_bundle(config_dir)
        template = load_template(cb.template_path)
        output = {"study": {}, "records": [
            {"record_id": "relationship_1",
             REQUIRED_RECORD_FIELD: {"value": None, "evidence": None,
                                     "notes": None}},
        ]}
        failures, _warns = validate_extraction_output(
            template, output, cb.reference_lists)
        assert failures == []
        assert missing_required_fields(template, output) != []


class TestTheShippedOutputIsSwept:
    def test_a_complete_run_shipping_a_null_required_field_warns(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        _add_record(orch, gauge_value="WDS-9")
        # What a reviewer edit does: the value goes, the record stays.
        orch.extraction_record.records[0][REQUIRED_RECORD_FIELD]["value"] = None
        orch._finalise("complete")

        warnings = _warnings(orch)
        assert len(warnings) == 1
        assert REQUIRED_RECORD_FIELD in warnings[0]
        assert "WARNING: required-fields-null" in capsys.readouterr().err

    def test_the_warning_names_every_field_that_shipped_null(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        _add_record(orch, gauge_value="WDS-9")
        record = orch.extraction_record.records[0]
        record[REQUIRED_RECORD_FIELD]["value"] = None
        record["outcome_variable"]["value"] = None
        orch._finalise("complete")

        warning = _warnings(orch)[0]
        assert REQUIRED_RECORD_FIELD in warning
        assert "outcome_variable" in warning

    def test_a_complete_run_with_every_required_field_set_says_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        _add_record(orch, gauge_value="WDS-9")
        # The study block's required fields, filled the way a finished run
        # leaves them.
        for var in ("abstract",):
            orch.extraction_record.study.setdefault(
                var, {"value": "x", "evidence": None, "notes": None})
        orch._finalise("complete")
        assert _warnings(orch) == []

    @pytest.mark.parametrize("status", ["failed_validation", "error"])
    def test_an_aborted_run_is_not_faulted_for_being_incomplete(
            self, config_dir, bundle_minimal_dir, tmp_path, status):
        # An aborted run stopped part-way, so unset required fields are the
        # expected shape of a work-in-progress snapshot. Naming them would
        # restate the status, and the claim the warning makes — that a run
        # which considered itself finished is not — would be false of it.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        _add_record(orch, gauge_value="WDS-9")
        orch.extraction_record.records[0][REQUIRED_RECORD_FIELD]["value"] = None
        orch._finalise(status, failure_reason="surrendered"
                       if status == "failed_validation" else None)
        assert _warnings(orch) == []

    def test_validation_passed_still_reports_how_the_run_ended(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The run-log flag is `status == "complete"` and stays that: it records
        # that the pipeline concluded and stands behind the result. What it is
        # NOT is a re-validation, which is why the warning beside it has to
        # carry the caveat.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path)
        _add_record(orch, gauge_value="WDS-9")
        orch.extraction_record.records[0][REQUIRED_RECORD_FIELD]["value"] = None
        orch._finalise("complete")

        entry = load_log(tmp_path / "runs")[0]
        assert entry["validation_passed"] is True
        assert entry["status"] == "complete"
        # And the caveat is reachable from the same artefact.
        assert any(w.startswith("required-fields-null")
                   for w in orch.session.meta["warnings"])
