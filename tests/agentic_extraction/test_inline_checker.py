"""The checker as an inline extension of the validator.

The checker is not a stage. It runs per field inside the tool call: after the
deterministic dispatcher has applied a call, the orchestrator fans out over the
fields that landed, and the challenges come back in the same tool result as the
validation errors. These tests pin the four things that makes true:

  - the TRIGGER (which fields go to a checker call at all),
  - the per-field BUDGET (`max_checks_per_field`, and its reconstruction on
    resume from the event log rather than from meta),
  - where the verdicts GO (challenges to the model, the full set to the event),
  - that a challenge is ADVISORY (it fails nothing and blocks nothing).

The fan-out itself is stubbed at `meltiro.orchestrator.run_checker_batch`, so
nothing here touches a provider.
"""

import json
from types import SimpleNamespace

import pytest

from meltiro.extraction_record import ExtractionRecord
from meltiro.orchestrator import Orchestrator
from meltiro.session import Session, result_to_model_text
from meltiro.tools import ToolDispatcher

from .conftest import checker_trigger_orch


def _env(value, evidence="<q>The WDS-9 was administered</q>", notes=None):
    return {"value": value, "evidence": evidence, "notes": notes}


# The synthetic template's REQUIRED check-block answers, in the shape the two
# tools that carry them want: `record_initial_check` takes the initial check
# flat, and `mark_complete` takes the quality check as an argument.
INITIAL_CHECK = {"text_readable": True, "figure_tables_included": True,
                 "expected_relationships": 1}
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


def _stub_user_message(monkeypatch):
    monkeypatch.setattr(
        "meltiro.orchestrator.build_checker_user_message",
        lambda **kw: [{"type": "text", "text": "stub:" + kw["field_path"]}],
    )


def _verdict(verdict="ok", rationale="fine", **extra):
    base = {
        "verdict": verdict, "rationale": rationale, "notes": None,
        "error_origin": False, "input_tokens": 10, "output_tokens": 3,
        "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.001,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# The trigger
# ---------------------------------------------------------------------------

class TestTrigger:
    """A field is checked when it applied in THIS call, holds a non-null
    value, and carries a non-empty evidence string, and only until its budget
    is spent."""

    def _calls(self, monkeypatch, template, study, applied, **kw):
        _stub_user_message(monkeypatch)
        record = ExtractionRecord()
        record.study.update(study)
        orch = checker_trigger_orch(template, record, **kw)
        calls, envelopes = orch._build_checker_calls(applied)
        return orch, calls, envelopes

    def test_an_applied_field_with_a_value_and_evidence_is_checked(
            self, monkeypatch, synthetic_template):
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"primary_aim": _env("Aim A")}, ["study.primary_aim"])
        assert [c["field_path"] for c in calls] == ["study.primary_aim"]

    def test_a_field_that_failed_validation_is_never_checked(
            self, monkeypatch, synthetic_template):
        # The dispatcher's `applied_fields` is the trigger's only input for
        # "did this land", so a field that failed validation is absent from it
        # and no call is built, even though a stale value sits in the record.
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"primary_aim": _env("Aim A")}, [])
        assert calls == []

    def test_a_null_value_is_not_checked(self, monkeypatch,
                                         synthetic_template):
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"primary_aim": _env(None)}, ["study.primary_aim"])
        assert calls == []

    @pytest.mark.parametrize("evidence", [None, "", "   ", [], ["<q>x</q>"]])
    def test_evidence_that_is_not_a_non_empty_string_is_not_checked(
            self, monkeypatch, synthetic_template, evidence):
        # A value offered with no grounds at all is not something the checker
        # can judge. A list is not a string, so it fails the same guard rather
        # than being coerced into one.
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"primary_aim": _env("Aim A", evidence=evidence)},
            ["study.primary_aim"])
        assert calls == []

    def test_pure_prose_evidence_is_checked(self, monkeypatch,
                                            synthetic_template):
        # No `<q>` required. "The grounds are prose, not a quote" is a
        # judgement the checker is entitled to make, so the field goes to it.
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"primary_aim": _env("Aim A", evidence="read off Table 2")},
            ["study.primary_aim"])
        assert [c["field_path"] for c in calls] == ["study.primary_aim"]

    def test_check_block_fields_are_never_checked(self, monkeypatch,
                                                  synthetic_template):
        # initial_check / quality_check are bare values with no evidence slot,
        # so nothing there can meet the trigger.
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template, {},
            ["initial_check.text_readable",
             "quality_check.deviation_from_expectations"])
        assert calls == []

    def test_a_record_field_is_checked_with_its_record_context(
            self, monkeypatch, synthetic_template):
        _stub_user_message(monkeypatch)
        record = ExtractionRecord()
        rid = record.add_record({"gauge": _env("WDS-9")}, "relationship")
        orch = checker_trigger_orch(synthetic_template, record)
        calls, _ = orch._build_checker_calls([f"record.{rid}.gauge"])
        assert [c["field_path"] for c in calls] == [f"record.{rid}.gauge"]

    def test_an_undeclared_variable_is_not_checked(self, monkeypatch,
                                                   synthetic_template):
        # A key the template does not declare has no field spec to brief the
        # checker with, so it is skipped rather than checked blind.
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"not_a_field": _env("x")}, ["study.not_a_field"])
        assert calls == []

    def test_calls_are_ordered_by_field_path(self, monkeypatch,
                                             synthetic_template):
        _, calls, _ = self._calls(
            monkeypatch, synthetic_template,
            {"sample_size": _env(348), "primary_aim": _env("Aim A"),
             "qa_reporting": _env("Compliant")},
            ["study.sample_size", "study.qa_reporting", "study.primary_aim"])
        assert [c["field_path"] for c in calls] == [
            "study.primary_aim", "study.qa_reporting", "study.sample_size"]

    def test_the_scored_envelope_is_returned_alongside_the_calls(
            self, monkeypatch, synthetic_template):
        _, calls, envelopes = self._calls(
            monkeypatch, synthetic_template,
            {"primary_aim": _env("Aim A", notes="read off Table 2")},
            ["study.primary_aim"])
        assert envelopes["study.primary_aim"] == {
            "value": "Aim A",
            "evidence": "<q>The WDS-9 was administered</q>",
            "notes": "read off Table 2",
        }


# ---------------------------------------------------------------------------
# The per-field budget
# ---------------------------------------------------------------------------

class TestPerFieldBudget:
    def _orch(self, monkeypatch, template, **kw):
        _stub_user_message(monkeypatch)
        record = ExtractionRecord()
        record.study["primary_aim"] = _env("Aim A")
        return checker_trigger_orch(template, record, **kw)

    def test_zero_disables_the_checker_entirely(self, monkeypatch,
                                                synthetic_template):
        orch = self._orch(monkeypatch, synthetic_template,
                          max_checks_per_field=0)
        assert orch.checker_enabled is False
        assert orch._build_checker_calls(["study.primary_aim"]) == ([], {})

    def test_one_checks_once_and_never_again(self, monkeypatch,
                                             synthetic_template):
        orch = self._orch(monkeypatch, synthetic_template,
                          max_checks_per_field=1)
        first, _ = orch._build_checker_calls(["study.primary_aim"])
        second, _ = orch._build_checker_calls(["study.primary_aim"])
        assert [c["field_path"] for c in first] == ["study.primary_aim"]
        assert second == []

    def test_two_allows_one_recheck_after_a_revision(self, monkeypatch,
                                                     synthetic_template):
        orch = self._orch(monkeypatch, synthetic_template,
                          max_checks_per_field=2)
        for expected in (["study.primary_aim"], ["study.primary_aim"], []):
            calls, _ = orch._build_checker_calls(["study.primary_aim"])
            assert [c["field_path"] for c in calls] == expected

    def test_the_budget_is_per_field_not_per_run(self, monkeypatch,
                                                 synthetic_template):
        orch = self._orch(monkeypatch, synthetic_template,
                          max_checks_per_field=1)
        orch.extraction_record.study["sample_size"] = _env(348)
        orch._build_checker_calls(["study.primary_aim"])
        calls, _ = orch._build_checker_calls(
            ["study.primary_aim", "study.sample_size"])
        # primary_aim is spent; its sibling still has its own allowance.
        assert [c["field_path"] for c in calls] == ["study.sample_size"]

    def test_each_call_carries_its_one_based_check_index(
            self, monkeypatch, synthetic_template):
        orch = self._orch(monkeypatch, synthetic_template)
        first, _ = orch._build_checker_calls(["study.primary_aim"])
        second, _ = orch._build_checker_calls(["study.primary_aim"])
        assert first[0]["check_index"] == 1
        assert second[0]["check_index"] == 2

    def test_counts_are_seeded_from_a_prior_segment(self, monkeypatch,
                                                    synthetic_template):
        # What a resume hands the trigger: a field that already spent its
        # budget in an earlier segment gets no fresh allowance.
        orch = self._orch(monkeypatch, synthetic_template,
                          max_checks_per_field=2,
                          check_counts={"study.primary_aim": 2})
        assert orch._build_checker_calls(["study.primary_aim"]) == ([], {})


# ---------------------------------------------------------------------------
# Where the verdicts go
# ---------------------------------------------------------------------------

def _live_orch(tmp_path, template, paper_text, image_labels, *,
               max_checks_per_field=2, check_reviewer_edits=False):
    """An Orchestrator with a real Session, dispatcher, and extraction record,
    enough for `_check_applied_fields` to run end to end offline."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.template = template
    record = ExtractionRecord()
    orch.extraction_record = record
    orch.dispatcher = ToolDispatcher(record, template, paper_text, image_labels)
    orch.session = Session.create(
        "demo-001", config_fp="config_fp:aaa", checker_fp="checker_fp:bbb",
        review_fp="review_fp:ccc", extractor_model="opus",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        checker_model="sonnet", review_model="opus",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        runs_dir=tmp_path,
    )
    orch.image_labels = image_labels
    orch.paper_text = paper_text
    orch.bundle = SimpleNamespace(figures={})
    orch.config = SimpleNamespace(checker_user_template_path="/unused",
                                  checker_system_path="/unused")
    orch.checker_config = SimpleNamespace(checker_model="claude-sonnet-4-6",
                                          concurrency=4, context_chars=0)
    orch.reference_lists = {}
    orch.max_checks_per_field = max_checks_per_field
    orch.check_reviewer_edits = check_reviewer_edits
    orch._check_counts = {}
    orch._cost_usd = 0.0
    # The accumulators a real __init__ sets, including the pair that decides
    # whether the run may STATE its running sum as a cost. The stubbed verdicts
    # below all carry one, so nothing goes uncosted and the run states a total.
    orch._cost_unpriced = False
    orch._cost_counted = False
    # The coverage pair beside them: a checker call whose gateway charge did
    # not arrive makes the run's sum a floor, and these say so and by how many
    # calls. The stubbed verdicts below are all fully receipted, so a run that
    # ends with either of these moved has been told something by a verdict.
    orch._cost_incomplete = False
    orch._unreceipted_calls = 0
    orch._input_tokens = 0
    orch._output_tokens = 0
    orch._cache_creation_tokens = 0
    orch._cache_read_tokens = 0
    # The per-role meters the checker fan-out folds its spend into, and the
    # per-role cards that price it. Nothing here prices the checker, so its
    # figures come from the stubbed verdicts alone.
    orch._usage_by_role = {}
    orch.rates = {}
    orch._study_identity_context = lambda: "Summary: ctx"
    # The extractor's opening call, and the only thing that opens the ordering
    # gate: until it lands the dispatcher refuses every write below. Made here
    # so each test starts where a real run's first write does. It is not run
    # through `_check_applied_fields`, so it triggers no checker call of its
    # own (a bare check-block value has no evidence to score).
    orch.dispatcher.dispatch("record_initial_check", dict(INITIAL_CHECK))
    return orch


def _stub_fanout(monkeypatch, verdicts_by_path):
    """Stub the whole fan-out, and the two prompt builders it would call.

    The system prompt is rendered by the instrument and the per-field user
    message by the orchestrator, so each is stubbed where its call site binds
    the builder.
    """
    monkeypatch.setattr(
        "meltiro.instrument.build_checker_system_text",
        lambda *a, **kw: "checker system")
    monkeypatch.setattr(
        "meltiro.orchestrator.build_checker_user_message",
        lambda **kw: [{"type": "text", "text": "stub:" + kw["field_path"]}])
    seen = {}

    def _fake(*, calls, config, on_complete=None, api_logger=None, **kw):
        seen["calls"] = calls
        return {c["field_path"]: verdicts_by_path[c["field_path"]]
                for c in calls}

    monkeypatch.setattr("meltiro.orchestrator.run_checker_batch", _fake)
    return seen


class TestVerdictRouting:
    def _applied(self, tmp_path, template, paper_text, image_labels,
                 monkeypatch, verdicts):
        orch = _live_orch(tmp_path, template, paper_text, image_labels)
        _stub_fanout(monkeypatch, verdicts)
        res = orch.dispatcher.dispatch("update_study", {"study": {
            "primary_aim": {
                "value": "Aim A",
                "evidence": "<q>The WDS-9 was administered</q>"},
        }})
        orch._check_applied_fields(res, stage="extractor")
        return orch, res

    def test_a_challenge_reaches_the_model_in_the_same_tool_result(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        _, res = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict("challenge", "the quote is about "
                                           "administration, not the aim")})
        assert res["checker_challenges"] == {
            "study.primary_aim":
                "the quote is about administration, not the aim"}
        sent = json.loads(result_to_model_text(res))
        assert sent["checker_challenges"] == res["checker_challenges"]

    def test_an_ok_verdict_is_recorded_but_not_shown(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        _, res = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict("ok", "supported")})
        assert "checker_challenges" not in res
        assert "checker_challenges" not in json.loads(result_to_model_text(res))
        assert res["_checker_verdicts"]["study.primary_aim"]["verdict"] == "ok"

    def test_the_full_verdict_set_is_stripped_from_the_model_payload(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        _, res = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict("challenge", "no")})
        assert "_checker_verdicts" not in json.loads(result_to_model_text(res))

    def test_the_verdict_record_carries_everything_diagnostics_needs(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        _, res = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict(
                "challenge", "no", notes="a longer aside", cost_usd=0.004,
                input_tokens=120, output_tokens=17)})
        entry = res["_checker_verdicts"]["study.primary_aim"]
        assert entry == {
            "verdict": "challenge",
            "rationale": "no",
            "notes": "a longer aside",
            "value_checked": "Aim A",
            "evidence_checked": "<q>The WDS-9 was administered</q>",
            "note_checked": None,
            "error_origin": False,
            "reprompted": 0,
            "stage": "extractor",
            "input_tokens": 120,
            "output_tokens": 17,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "cost_usd": 0.004,
        }

    def test_an_error_origin_challenge_is_recorded_as_such(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # A failed check degrades to a challenge rather than aborting, and the
        # record says it was a failed call rather than an objection.
        _, res = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict(
                "challenge", "(checker error: rate limit)", error_origin=True,
                cost_usd=0.0)})
        assert res["_checker_verdicts"][
            "study.primary_aim"]["error_origin"] is True
        # And the flag is not decoration: it is what keeps the failure out of
        # the challenges. Asserting only the flag would pass on an engine that
        # recorded it and put the field to the extractor anyway.
        assert "checker_challenges" not in res

    def test_an_error_origin_challenge_never_reaches_the_model(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # Its rationale is this engine's report of a failed call. Shown to the
        # extractor it would be an instruction to revise a value against
        # plumbing text — a paid turn spent answering "(checker error: rate
        # limit)" as though it were a reading of the paper.
        orch, res = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict(
                "challenge", "(checker error: rate limit)", error_origin=True,
                cost_usd=0.0)})
        assert "checker_challenges" not in res
        sent = result_to_model_text(res)
        assert "checker error" not in sent
        assert "checker_challenges" not in sent
        # And the failure cost the field the one slot its own check used, and
        # no more: nothing about degrading it re-checks anything.
        assert orch._check_counts == {"study.primary_aim": 1}

    def test_a_genuine_challenge_beside_a_failed_check_still_reaches_the_model(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # The exclusion is of the failed check, not of the batch it failed in.
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        _stub_fanout(monkeypatch, {
            "study.primary_aim": _verdict(
                "challenge", "the quote is about administration"),
            "study.sample_size": _verdict(
                "challenge", "(checker error: rate limit)",
                error_origin=True),
        })
        res = orch.dispatcher.dispatch("update_study", {"study": {
            "primary_aim": {
                "value": "Aim A",
                "evidence": "<q>The WDS-9 was administered</q>"},
            "sample_size": {
                "value": 100,
                "evidence": "<q>The WDS-9 was administered</q>"},
        }})
        orch._check_applied_fields(res, stage="extractor")
        assert list(res["checker_challenges"]) == ["study.primary_aim"]
        # Both are in the durable record; only one was put to the model.
        assert set(res["_checker_verdicts"]) == {
            "study.primary_aim", "study.sample_size"}

    def test_a_call_that_triggers_nothing_writes_neither_key(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # Byte-identical to a run with the checker off, which is what lets a
        # resume replay the stored result without a single checker call.
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        _stub_fanout(monkeypatch, {})
        res = orch.dispatcher.dispatch("view_summary", {})
        orch._check_applied_fields(res, stage="extractor")
        assert "checker_challenges" not in res
        assert "_checker_verdicts" not in res

    def test_the_call_tally_reaches_meta(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        orch, _ = self._applied(
            tmp_path, synthetic_template, paper_text, image_labels, monkeypatch,
            {"study.primary_aim": _verdict()})
        assert orch.session.meta["checker_calls_run"] == 1

    def test_the_tally_counts_every_call_in_a_fan_out(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # A tool call that triggers ONE check cannot tell a tally of the batch
        # size from a tally of the batch, and a first `update_study` carrying
        # every populated study field is the expected shape, so the figure a
        # run publishes is almost always a sum of fan-outs wider than one.
        # `checker_calls_run` is printed in run.json, run_log.json, the CLI
        # summary and the transcript, and it is the only record of how much
        # checking a run actually bought, so an under-count is a spend figure
        # that reads low with nothing to contradict it.
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        _stub_fanout(monkeypatch, {
            "study.primary_aim": _verdict(),
            "study.sample_size": _verdict(),
            "study.publication_type": _verdict(),
        })
        res = orch.dispatcher.dispatch("update_study", {"study": {
            "primary_aim": {
                "value": "Aim A",
                "evidence": "<q>The WDS-9 was administered</q>"},
            "sample_size": {
                "value": 100,
                "evidence": "<q>The WDS-9 was administered</q>"},
            "publication_type": {
                "value": "Academic paper",
                "evidence": "<q>The WDS-9 was administered</q>"},
        }})
        assert len(res["applied_fields"]) == 3
        orch._check_applied_fields(res, stage="extractor")
        # Three fields checked in one batch: the tally is the number of CALLS,
        # not the number of batches.
        assert orch.session.meta["checker_calls_run"] == 3

    def test_the_tally_sums_across_batches(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # Two tool calls, two fan-outs of different widths. The published
        # figure is the total, so a per-batch count of one would report 2 for
        # the 3 calls this run makes.
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        _stub_fanout(monkeypatch, {
            "study.primary_aim": _verdict(),
            "study.sample_size": _verdict(),
        })
        first = orch.dispatcher.dispatch("update_study", {"study": {
            "primary_aim": {
                "value": "Aim A",
                "evidence": "<q>The WDS-9 was administered</q>"},
            "sample_size": {
                "value": 100,
                "evidence": "<q>The WDS-9 was administered</q>"},
        }})
        orch._check_applied_fields(first, stage="extractor")

        _stub_fanout(monkeypatch, {"study.publication_type": _verdict()})
        second = orch.dispatcher.dispatch("update_study", {"study": {
            "publication_type": {
                "value": "Academic paper",
                "evidence": "<q>The WDS-9 was administered</q>"},
        }})
        orch._check_applied_fields(second, stage="extractor")

        assert orch.session.meta["checker_calls_run"] == 3

    def test_checker_spend_lands_in_the_run_accumulators(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # The fan-out stub calls no on_complete, so drive the accumulation
        # through the real `_run_checker_fanout` callback instead.
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        monkeypatch.setattr(
            "meltiro.instrument.build_checker_system_text",
            lambda *a, **kw: "checker system")
        monkeypatch.setattr(
            "meltiro.orchestrator.build_checker_user_message",
            lambda **kw: [{"type": "text", "text": "stub"}])

        def _fake(*, calls, config, on_complete=None, api_logger=None, **kw):
            out = {}
            for c in calls:
                result = _verdict(cost_usd=0.002, input_tokens=50,
                                  output_tokens=7)
                on_complete(c["field_path"], result)
                out[c["field_path"]] = result
            return out

        monkeypatch.setattr("meltiro.orchestrator.run_checker_batch", _fake)
        orch._run_checker_fanout([
            {"field_path": "study.primary_aim", "user_message_blocks": []}])
        assert orch._cost_usd == pytest.approx(0.002)
        assert orch._input_tokens == 50
        assert orch._output_tokens == 7
        # And into the checker's own meters, so the run can say which role
        # spent it.
        checker = orch._usage_by_role["checker"]
        assert checker["cost_usd"] == pytest.approx(0.002)
        assert checker["input_tokens"] == 50
        assert checker["output_tokens"] == 7


# ---------------------------------------------------------------------------
# A challenge is advisory
# ---------------------------------------------------------------------------

class TestChallengeIsAdvisory:
    def test_a_challenge_does_not_fail_the_tool_call(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        _stub_fanout(monkeypatch, {
            "study.primary_aim": _verdict("challenge", "unsupported")})
        res = orch.dispatcher.dispatch("update_study", {"study": {
            "primary_aim": {
                "value": "Aim A",
                "evidence": "<q>The WDS-9 was administered</q>"},
        }})
        orch._check_applied_fields(res, stage="extractor")
        assert res["status"] == "ok"
        assert res["errors"] == []
        assert res["applied_fields"] == ["study.primary_aim"]

    def test_a_challenged_field_still_reaches_mark_complete(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # No checker verdict is consulted by the completeness gate, so a field
        # the checker never accepted does not hold the run open.
        orch = _live_orch(tmp_path, synthetic_template, paper_text,
                          image_labels)
        _stub_fanout(monkeypatch, {
            "study.primary_aim": _verdict("challenge", "unsupported")})
        # The initial check landed when the dispatcher was built (it has to:
        # nothing else opens the gate), and the quality check rides on
        # `mark_complete` itself, so this call carries study fields alone.
        write = orch.dispatcher.dispatch("update_study", {
            "study": {"primary_aim": {
                "value": "Aim A",
                "evidence": "<q>The WDS-9 was administered</q>"}},
        })
        orch._check_applied_fields(write, stage="extractor")
        assert "study.primary_aim" in write["checker_challenges"]
        orch.dispatcher.dispatch("add_record", {"fields": {
            "gauge": {"value": "WDS-9",
                      "evidence": "<q>The WDS-9 was administered</q>"},
            "outcome_variable": {
                "value": "unplanned removal",
                "evidence": "<q>odds ratio for unplanned removal</q>"},
            "outcome_category": {
                "value": "Cost or resource use",
                "evidence": "<q>Total Fleet Service Costs</q>"},
        }})
        done = orch.dispatcher.dispatch(
            "mark_complete", {"quality_check": dict(QUALITY_CHECK)})
        assert done["status"] == "ok"
        assert orch.extraction_record.mark_complete_flag is True


# ---------------------------------------------------------------------------
# Resume: the budget is reconstructed from the log
# ---------------------------------------------------------------------------

class TestBudgetSurvivesResume:
    def _session_with_verdicts(self, tmp_path, events):
        session = Session.create(
            "demo-001", config_fp="config_fp:aaa", checker_fp="checker_fp:bbb",
            review_fp="review_fp:ccc", extractor_model="opus",
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            checker_model="sonnet", review_model="opus",
            tool_set_hash="ts", template_hash="th", prompt_hash="ph",
            runs_dir=tmp_path,
        )
        for ev in events:
            session.append_event(ev)
        orch = Orchestrator.__new__(Orchestrator)
        orch.session = session
        return orch

    def test_counts_come_back_off_the_recorded_verdicts(self, tmp_path):
        orch = self._session_with_verdicts(tmp_path, [
            {"event": "tool_call_applied", "tool": "update_study", "result": {
                "_checker_verdicts": {
                    "study.primary_aim": {"verdict": "challenge"},
                    "study.sample_size": {"verdict": "ok"}}}},
            {"event": "tool_call_applied", "tool": "update_study", "result": {
                "_checker_verdicts": {
                    "study.primary_aim": {"verdict": "ok"}}}},
        ])
        assert orch._reconstruct_check_counts() == {
            "study.primary_aim": 2, "study.sample_size": 1}

    def test_reviewer_side_checks_count_against_the_same_budget(self, tmp_path):
        # A `review_tool_call` event carries a result in the same shape, so a
        # field checked on the reviewer's write has spent the same allowance.
        orch = self._session_with_verdicts(tmp_path, [
            {"event": "tool_call_applied", "tool": "update_study", "result": {
                "_checker_verdicts": {"study.primary_aim": {"verdict": "ok"}}}},
            {"event": "review_tool_call", "tool": "update_study", "result": {
                "_checker_verdicts": {"study.primary_aim": {"verdict": "ok"}}}},
        ])
        assert orch._reconstruct_check_counts() == {"study.primary_aim": 2}

    def test_an_error_origin_check_still_spends_its_slot(self, tmp_path):
        # The call was made and the money was spent, so it counts. Anything
        # else would let a flaky provider buy a field unlimited checks.
        orch = self._session_with_verdicts(tmp_path, [
            {"event": "tool_call_applied", "tool": "update_study", "result": {
                "_checker_verdicts": {"study.primary_aim": {
                    "verdict": "challenge", "error_origin": True}}}},
        ])
        assert orch._reconstruct_check_counts() == {"study.primary_aim": 1}

    def test_events_without_a_result_are_ignored(self, tmp_path):
        orch = self._session_with_verdicts(tmp_path, [
            {"event": "session_started"},
            {"event": "assistant_text", "text": "thinking"},
            {"event": "tool_call_applied", "tool": "view_summary",
             "result": {"status": "ok"}},
        ])
        assert orch._reconstruct_check_counts() == {}

    def test_the_count_is_not_stored_in_meta(self, tmp_path):
        # The event log is the single source of truth: a second copy in meta
        # could drift from the verdicts a diagnostics reader would believe.
        orch = self._session_with_verdicts(tmp_path, [
            {"event": "tool_call_applied", "tool": "update_study", "result": {
                "_checker_verdicts": {"study.primary_aim": {"verdict": "ok"}}}},
        ])
        # The count is reconstructable from the log, so it exists as a value.
        assert orch._reconstruct_check_counts() == {"study.primary_aim": 1}
        # And no key in meta holds it. Compared against the reconstructed
        # value rather than by guessing at key names, so a second copy filed
        # under any name at all is caught.
        assert 1 not in [v for k, v in orch.session.meta.items()
                         if "check" in k]
        assert not any("check" in k and "count" in k
                       for k in orch.session.meta)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestCheckerDiagnostics:
    def _orch(self, tmp_path, events):
        session = Session.create(
            "demo-001", config_fp="config_fp:aaa", checker_fp="checker_fp:bbb",
            review_fp="review_fp:ccc", extractor_model="opus",
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            checker_model="sonnet", review_model="opus",
            tool_set_hash="ts", template_hash="th", prompt_hash="ph",
            runs_dir=tmp_path,
        )
        for ev in events:
            session.append_event(ev)
        orch = Orchestrator.__new__(Orchestrator)
        orch.session = session
        return orch

    def _tool_call(self, verdicts):
        return {"event": "tool_call_applied", "tool": "update_study",
                "result": {"_checker_verdicts": verdicts}}

    def test_a_field_challenged_to_the_last_is_named(self, tmp_path):
        orch = self._orch(tmp_path, [
            self._tool_call({"study.primary_aim": {
                "verdict": "challenge", "error_origin": False}}),
        ])
        diag = orch._checker_diagnostics()
        assert diag["unresolved_challenges"] == ["study.primary_aim"]
        assert diag["fields_checked"] == 1
        assert diag["checks_run"] == 1

    def test_a_challenge_a_later_check_cleared_is_not_named(self, tmp_path):
        orch = self._orch(tmp_path, [
            self._tool_call({"study.primary_aim": {
                "verdict": "challenge", "error_origin": False}}),
            self._tool_call({"study.primary_aim": {
                "verdict": "ok", "error_origin": False}}),
        ])
        diag = orch._checker_diagnostics()
        assert diag["unresolved_challenges"] == []
        assert diag["checks_run"] == 2
        assert diag["fields_checked"] == 1

    def test_an_error_origin_verdict_is_reported_apart(self, tmp_path):
        # An exhausted retry is an absence of information, not an objection,
        # so it is not listed as a challenge the extractor left standing.
        orch = self._orch(tmp_path, [
            self._tool_call({"study.primary_aim": {
                "verdict": "challenge", "error_origin": True}}),
        ])
        diag = orch._checker_diagnostics()
        assert diag["unresolved_challenges"] == []
        assert diag["checker_errors"] == ["study.primary_aim"]
