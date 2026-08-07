"""Tests for the closest-match suggestion in unknown_field errors."""

from meltiro.extraction_record import ExtractionRecord
from meltiro.tools import (
    ToolDispatcher, _suggest_closest_field,
)

from .conftest import open_initial_check_gate


Q_GAUGE = "<q>WDS-9 was administered</q>"
Q_REMOVAL = "<q>unplanned removal</q>"


def _dispatcher(template, paper_text, image_labels):
    """A record with the ordering gate already open, plus its dispatcher.

    Every mutating call is refused until `record_initial_check` has landed;
    that rule is not what this file is about (see test_tools.py for it), so
    the gate is opened here once rather than in each test below.
    """
    record = open_initial_check_gate(ExtractionRecord())
    return record, ToolDispatcher(record, template, paper_text, image_labels)


def _env(value, evidence):
    """A well-formed field envelope: evidence is a STRING, and `notes` is
    required. The dispatcher tests below depend on the good fields in a mixed
    call actually applying, so a malformed envelope here would mask the
    unknown-field case each test is named for."""
    return {"value": value, "evidence": evidence, "notes": None}


def _good_record_fields():
    """The three required record fields, all valid, for a mixed add_record
    call whose only failure is the unknown field the test adds."""
    return {
        "gauge": _env("WDS-9", Q_GAUGE),
        "outcome_variable": _env("Unplanned removal", Q_REMOVAL),
        "outcome_category": _env("Failure state", Q_REMOVAL),
    }


class TestSuggestClosest:
    def test_close_match_typo(self):
        s = _suggest_closest_field(
            "primry_aim", ["primary_aim", "secondary_aims", "sample_size"])
        assert "primary_aim" in s

    def test_no_close_match_returns_empty(self):
        s = _suggest_closest_field("xyz", ["primary_aim", "sample_size"])
        assert s == ""

    def test_empty_input(self):
        assert _suggest_closest_field("", ["a", "b"]) == ""

    def test_multiple_suggestions(self):
        s = _suggest_closest_field(
            "size", ["sample_size", "subgroup_n", "study_size"],
            n=2, cutoff=0.4,
        )
        # At least one close match should appear (cutoff is generous).
        assert "sample_size" in s or "study_size" in s


class TestDispatcherHints:
    def test_study_typo_suggests_correct_name(
            self, synthetic_template, paper_text, image_labels):
        # The only field in this call is the typo, so nothing applies and the
        # call fails outright. The hint still names the field that was meant.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "study": {"primry_aim": _env("X", Q_GAUGE)},
        })
        assert result["status"] == "validation_failed"
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_field")
        assert "primary_aim" in msg

    def test_record_typo_suggests(
            self, synthetic_template, paper_text, image_labels):
        # A MIXED call: three valid record fields plus one typo. The valid
        # fields apply and the record is minted, the typo alone fails, and the
        # status is `partial` rather than a whole-call failure.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        fields = _good_record_fields()
        fields["outcom_categry"] = _env("Cost", Q_REMOVAL)
        result = d.dispatch("add_record", {"fields": fields})

        assert result["status"] == "partial"
        assert len(a.records) == 1
        assert a.records[0]["outcome_category"]["value"] == "Failure state"
        assert sorted(result["applied_fields"]) == [
            "record.relationship_1.gauge",
            "record.relationship_1.outcome_category",
            "record.relationship_1.outcome_variable",
        ]
        assert list(result["failed_fields"]) == [
            "record.relationship_1.outcom_categry"]

        unknown_msgs = [e["message"] for e in result["errors"]
                        if e["code"] == "unknown_field"]
        assert any("outcome_category" in m for m in unknown_msgs)

    def test_study_field_on_record_specific_hint(
            self, synthetic_template, paper_text, image_labels):
        # Also a MIXED call. primary_aim is a STUDY field; putting it on a
        # record should produce a specific "this is a study-level field" hint,
        # not just a fuzzy match, while the three valid record fields apply.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        fields = _good_record_fields()
        fields["primary_aim"] = _env("Assess X", Q_GAUGE)
        result = d.dispatch("add_record", {"fields": fields})

        assert result["status"] == "partial"
        assert len(a.records) == 1
        assert "primary_aim" not in a.records[0]
        assert list(result["failed_fields"]) == [
            "record.relationship_1.primary_aim"]

        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_field"
                   and "primary_aim" in e["path"])
        assert "STUDY-level field" in msg
        assert "update_study.study" in msg


class TestDefaults:
    def test_extractor_and_review_models_have_no_default(self):
        # There is deliberately no hardcoded fallback extractor / review
        # model: the CLI must supply both from the config bundle, and a
        # direct construction must pass them explicitly. Removing the
        # defaults is what makes an unset model fail loudly.
        import inspect

        from meltiro.orchestrator import Orchestrator
        sig = inspect.signature(Orchestrator.__init__)
        assert sig.parameters["extractor_model"].default is \
            inspect.Parameter.empty
        assert sig.parameters["review_model"].default is \
            inspect.Parameter.empty

    def test_no_default_model_constants(self):
        import meltiro.orchestrator as orch_mod
        assert not hasattr(orch_mod, "DEFAULT_EXTRACTOR_MODEL")
        assert not hasattr(orch_mod, "DEFAULT_REVIEW_MODEL")

    def test_checker_has_no_default_model(self):
        # There is deliberately no hardcoded checker default: a config that
        # omits checker_model (and passes no --checker-model) must fail
        # loudly, exactly like the extractor and review models. from_env with
        # no override resolves to None rather than silently defaulting — and
        # no environment variable can supply one either, so the model the
        # checker runs on is always the one the bundle or the flag names.
        import os
        from unittest import mock
        from meltiro.checker import CheckerConfig
        with mock.patch.dict(os.environ, {"CHECKER_MODEL": "claude-opus-4-8"},
                             clear=True):
            cfg = CheckerConfig.from_env()
        assert cfg.checker_model is None

    def test_default_tool_cap_is_at_least_100(self):
        from meltiro.orchestrator import (
            DEFAULT_MAX_TOOL_CALLS)
        assert DEFAULT_MAX_TOOL_CALLS >= 100
