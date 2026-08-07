"""The tier a quote matched on reaches the artefact.

`quote_check` runs four tiers. The top three forgive differences a PDF
converter introduces — ligatures, smart quotes, hyphenation across a line
break, stripped citations — so a quote reaching them is still the paper's own
text, rendered differently. The fourth folds case, and that difference is the
model's own: `quote_check` gives it a tier of its own precisely so nothing
downstream reads a folded match as text the paper writes that way.

The tier lives inside a `QuoteMatch` the validator discards, and a PASSING
quote produces no error to carry it, so without a record of its own nothing
downstream ever learns it. These pin that record: the field path, the quote and
the tier ride on the tool result, land in `tool_calls.jsonl`, and are rendered
under the call in the transcript.

They also pin what does NOT happen: the model is not told. The fold already
succeeded, so there is nothing to fix, and reporting it would invite the model
to re-quote text that is already accepted.
"""

import json

from meltiro.quote_check import validate_evidence
from meltiro.session import result_to_model_text
from meltiro.tools import ToolDispatcher
from meltiro.extraction_record import ExtractionRecord

from .conftest import open_initial_check_gate


# In the paper as written, and in the paper only once case is discarded.
Q_DIRECT = "<q>The WDS-9 was administered</q>"
Q_FOLDED = "<q>the wds-9 was administered</q>"


def _dispatcher(synthetic_template, paper_text, image_labels):
    record = open_initial_check_gate(ExtractionRecord())
    return ToolDispatcher(record, synthetic_template, paper_text, image_labels)


def _study_call(dispatcher, evidence):
    return dispatcher.dispatch("update_study", {"study": {
        "primary_aim": {"value": "Aim A", "evidence": evidence,
                        "notes": None},
    }})


class TestTheCheckerReportsTheTier:
    def test_a_folded_match_is_collected(self, paper_text):
        sink = []
        errors = validate_evidence(
            evidence=Q_FOLDED, paper_text=paper_text, image_labels=set(),
            value="Aim A", field_path="study.primary_aim", weak_matches=sink)
        # It PASSES. The record is provenance, not a rejection.
        assert errors == []
        assert len(sink) == 1
        assert sink[0]["tier"] == "case_folded"
        assert sink[0]["path"] == "study.primary_aim.evidence[<q>0]"
        assert "wds-9" in sink[0]["quote"]

    def test_a_direct_match_is_not_collected(self, paper_text):
        # Only the fold is recorded. A note on every passing quote would be
        # noise, and the three tiers above the fold are the paper's own text.
        sink = []
        validate_evidence(
            evidence=Q_DIRECT, paper_text=paper_text, image_labels=set(),
            value="Aim A", field_path="study.primary_aim", weak_matches=sink)
        assert sink == []

    def test_a_failed_quote_is_not_collected(self, paper_text):
        # A quote that matched no tier is an error, and errors have their own
        # channel. Recording it here too would double-report it and imply a
        # match that never happened.
        sink = []
        errors = validate_evidence(
            evidence="<q>no sentence in the paper says this</q>",
            paper_text=paper_text, image_labels=set(), value="Aim A",
            field_path="study.primary_aim", weak_matches=sink)
        assert errors and errors[0]["code"] == "quote_not_in_text"
        assert sink == []

    def test_no_sink_is_as_valid_as_a_sink(self, paper_text):
        # The accumulator is the caller's, so a caller that wants no record is
        # unaffected — the single-field consumer path passes none.
        assert validate_evidence(
            evidence=Q_FOLDED, paper_text=paper_text, image_labels=set(),
            value="Aim A", field_path="study.primary_aim") == []


class TestItReachesTheToolResult:
    def test_a_study_field_records_it(
            self, synthetic_template, paper_text, image_labels):
        d = _dispatcher(synthetic_template, paper_text, image_labels)
        res = _study_call(d, Q_FOLDED)
        assert res["status"] == "ok"
        assert [r["path"] for r in res["_weak_quote_matches"]] == \
            ["study.primary_aim.evidence[<q>0]"]
        assert res["_weak_quote_matches"][0]["tier"] == "case_folded"

    def test_a_direct_quote_records_nothing(
            self, synthetic_template, paper_text, image_labels):
        d = _dispatcher(synthetic_template, paper_text, image_labels)
        res = _study_call(d, Q_DIRECT)
        assert res["_weak_quote_matches"] == []

    def test_a_record_field_records_it_under_the_minted_id(
            self, synthetic_template, paper_text, image_labels):
        # `add_record` validates before an id exists, so the path is recorded
        # against the `record.<new>` placeholder and has to be repointed at
        # the record actually minted. A path naming the placeholder points at
        # nothing a reader of the finished output can find.
        d = _dispatcher(synthetic_template, paper_text, image_labels)
        res = d.dispatch("add_record", {"fields": {
            "gauge": {"value": "WDS-9", "evidence": Q_FOLDED, "notes": None},
            "outcome_variable": {"value": "unplanned removal",
                                 "evidence": Q_DIRECT, "notes": None},
            "outcome_category": {"value": "Failure state",
                                 "evidence": Q_DIRECT, "notes": None},
        }})
        assert res["status"] == "ok"
        paths = [r["path"] for r in res["_weak_quote_matches"]]
        assert paths == ["record.relationship_1.gauge.evidence[<q>0]"]
        assert not any("<new>" in p for p in paths)

    def test_a_rejected_field_contributes_nothing(
            self, synthetic_template, paper_text, image_labels):
        # A rejected field stores nothing, so a note about how its quote
        # matched would describe evidence that is not in the output. The
        # folded quote here rides on a field whose VALUE fails its type check.
        d = _dispatcher(synthetic_template, paper_text, image_labels)
        res = d.dispatch("update_study", {"study": {
            "sample_size": {"value": "not an integer", "evidence": Q_FOLDED,
                            "notes": None},
        }})
        assert res["status"] == "validation_failed"
        assert res["_weak_quote_matches"] == []


class TestTheModelIsNotTold:
    def test_the_record_is_stripped_from_the_model_payload(
            self, synthetic_template, paper_text, image_labels):
        # The fold already succeeded, so there is nothing for the model to
        # fix, and telling it would invite a re-quote of accepted text. This
        # is provenance for whoever reads the run, not feedback for the run.
        d = _dispatcher(synthetic_template, paper_text, image_labels)
        res = _study_call(d, Q_FOLDED)
        assert res["_weak_quote_matches"]
        assert "_weak_quote_matches" not in json.loads(
            result_to_model_text(res))

    def test_the_field_still_applies(
            self, synthetic_template, paper_text, image_labels):
        # Recording the tier changes no verdict: a folded match passes, and
        # the value is stored exactly as a directly-matched one would be.
        d = _dispatcher(synthetic_template, paper_text, image_labels)
        res = _study_call(d, Q_FOLDED)
        assert res["applied_fields"] == ["study.primary_aim"]
        assert d.extraction_record.study["primary_aim"]["value"] == "Aim A"


class TestAReaderSeesIt:
    """Rendered into transcript.md, under the call that wrote the field.

    Rendered end to end rather than by calling the renderer directly: a
    section that draws correctly but is never reached from `_render_tool_call`
    reaches no reader at all, which is the failure this whole item is about.
    """

    def _document(self, tmp_path, weak_matches):
        from .test_transcript import _hand_built_session
        from meltiro.transcript import render_transcript

        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "tool_call_applied", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_study",
             "args": {"study": {"primary_aim": {"value": "Aim A"}}},
             "result": {
                 "status": "ok",
                 "applied_changes": {"study_fields": ["primary_aim"]},
                 "_field_diffs": {"study.primary_aim": {
                     "before": None, "after": "Aim A"}},
                 "errors": [], "warnings": [], "failed_fields": {},
                 "_weak_quote_matches": weak_matches,
             }},
            {"ts": "T1", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "text", "text": ""}]},
        ])
        return render_transcript(session_dir)

    def test_the_transcript_names_the_quote_and_the_tier(self, tmp_path):
        document = self._document(tmp_path, [{
            "path": "study.primary_aim.evidence[<q>0]",
            "quote": "the wds-9 was administered",
            "tier": "case_folded",
        }])
        assert "**Matched on case only.**" in document
        assert "study.primary_aim.evidence[<q>0]" in document
        assert "the wds-9 was administered" in document
        assert "case_folded" in document

    def test_the_transcript_says_nothing_when_every_quote_matched(
            self, tmp_path):
        # The common case. A section printed empty on every call would train a
        # reader to skip it on the one call where it matters.
        assert "Matched on case only" not in self._document(tmp_path, [])
