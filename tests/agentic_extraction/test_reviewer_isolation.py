"""Nothing assembled for the reviewer names a checker, nor what the
extractor thought.

The final review is the fresh-context second opinion: it reads the paper and
the assembled extraction output and forms its own view. Telling it which cells
a narrower model doubted would anchor that view on the checker's, which is the
one thing an independent second opinion must not be. So nothing the engine
assembles for the reviewer names a checker, quotes a challenge, or counts
contested fields, and there is no parameter through which one could.

The same rule covers a second anchor. The two check blocks are the
EXTRACTOR's self-assessment of its own run, and the reviewer records its own
alongside, so showing it the extractor's would anchor the second opinion just
as the checker's verdicts would. It is withheld from the assembled message AND
from the two view tools, because a block the reviewer could fetch in one extra
tool call is not withheld at all.

The single exception is deliberate and narrow: with `check_reviewer_edits` on,
a field the REVIEWER itself writes is checked on the same per-field terms the
extractor's fields are, and the challenge arrives inside that tool result. That
is a note on its own proposed value, not a briefing about the run, and it is
off by default. Both halves are pinned here.

What these test is everything the engine ASSEMBLES for the reviewer: the
message, the tool catalogue, and the two view tools. None of them names a
checker under any configuration, `check_reviewer_edits` on included.

The reviewer's SYSTEM PROMPT is the one surface that varies, and it varies on
that toggle: with the reviewer's edits checked it describes the exception in
general terms, and with them unchecked — the default — it names no checker at
all, because none can reach it. That is pinned in
tests/agentic_extraction/test_engine_prompts.py, over the whole toggle grid.
What must never happen either way is a run's own challenges, rationales or
contested-cell counts reaching the reviewer, and no assembled surface carries
them.
"""

import inspect
import json
from types import SimpleNamespace

import pytest

import meltiro.instrument as instrument_mod
import meltiro.orchestrator as orch_mod
from meltiro.extraction_record import (
    ExtractionRecord, ROLE_EXTRACTOR, ROLE_REVIEW)
from meltiro.orchestrator import Orchestrator
from meltiro.prompt_builder import build_review_user_blocks
from meltiro.session import Session, result_to_model_text
from meltiro.tools import ToolDispatcher


CHALLENGE_RATIONALE = "the quote names the instrument, not the sample size"

# Every word that would betray the checker's existence to the reviewer. Matched
# case-insensitively against the whole assembled message.
FORBIDDEN = ("checker", "challenge", "rationale", "contested", "unresolved",
             "verdict", "second opinion")

# The extractor's check-block answers, carrying values distinctive enough to
# find by substring if either block ever reached the reviewer. The initial
# check's variables are two booleans and an integer, so the integer is the one
# that can be given a fingerprint.
EXTRACTOR_INITIAL_CHECK = {
    "text_readable": True,
    "figure_tables_included": True,
    "expected_relationships": 4242,
}
EXTRACTOR_QUALITY_CHECK = {
    "deviation_from_expectations":
        "EXTRACTORS-OWN-ACCOUNT: fewer than expected, and I am unsure why",
}
REVIEWER_QUALITY_CHECK = {
    "deviation_from_expectations": "one relationship, as I read it",
}


def _env(value, evidence="<q>The WDS-9 was administered</q>"):
    return {"value": value, "evidence": evidence, "notes": None}


def _orch(tmp_path, template, paper_text, image_labels, *,
          check_reviewer_edits=False):
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
    orch.study_id = "demo-001"
    orch.paper_text = paper_text
    orch.figures = []
    orch.image_labels = image_labels
    orch.image_notes = {}
    orch.image_tables = {}
    orch.image_figures = {}
    orch.bundle = SimpleNamespace(figures={}, tables={})
    orch.config = SimpleNamespace(partials_dir="/unused",
                                  checker_system_path="/unused")
    orch.checker_config = SimpleNamespace(checker_model="claude-sonnet-4-6",
                                          concurrency=4, context_chars=0)
    orch.reference_lists = {}
    orch.max_checks_per_field = 2
    orch.check_reviewer_edits = check_reviewer_edits
    orch._check_counts = {}
    orch._cost_usd = 0.0
    orch._input_tokens = 0
    orch._output_tokens = 0
    orch._cache_creation_tokens = 0
    orch._cache_read_tokens = 0
    orch._study_identity_context = lambda: "Summary: ctx"
    return orch


def _stub_fanout(monkeypatch, *, verdict="challenge",
                 rationale=CHALLENGE_RATIONALE):
    # The checker's system prompt is rendered by the instrument and its
    # per-field user message by the orchestrator, so each builder is stubbed
    # where its call site binds it.
    monkeypatch.setattr(instrument_mod, "build_checker_system_text",
                        lambda *a, **kw: "checker system")
    monkeypatch.setattr(
        orch_mod, "build_checker_user_message",
        lambda **kw: [{"type": "text", "text": "stub:" + kw["field_path"]}])
    seen = []

    def _fake(*, calls, config, on_complete=None, api_logger=None, **kw):
        seen.extend(c["field_path"] for c in calls)
        return {c["field_path"]: {
            "verdict": verdict, "rationale": rationale, "notes": None,
            "error_origin": False, "input_tokens": 1, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.0,
        } for c in calls}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _fake)
    return seen


def _challenged_run(tmp_path, template, paper_text, image_labels, monkeypatch,
                    **kw):
    """An orchestrator whose extraction output holds a field the checker
    challenged, with the challenge on the durable record."""
    orch = _orch(tmp_path, template, paper_text, image_labels, **kw)
    _stub_fanout(monkeypatch)
    # The extractor's opening call: nothing else opens the ordering gate, and
    # until it lands every write below is refused.
    orch.dispatcher.dispatch(
        "record_initial_check", dict(EXTRACTOR_INITIAL_CHECK))
    res = orch.dispatcher.dispatch("update_study", {"study": {
        "primary_aim": _env("Aim A"),
    }})
    orch._check_applied_fields(res, stage="extractor")
    orch.session.append_event({
        "event": "tool_call_applied", "turn_id": 1, "tool": "update_study",
        "args": {}, "result": res})
    assert res["checker_challenges"] == {"study.primary_aim":
                                         CHALLENGE_RATIONALE}
    return orch


# ---------------------------------------------------------------------------
# Nothing the engine assembles for the reviewer reveals a checker
# ---------------------------------------------------------------------------

def test_the_reviewer_message_after_a_challenged_run_says_nothing_of_it(
        tmp_path, synthetic_template, paper_text, image_labels, monkeypatch):
    orch = _challenged_run(tmp_path, synthetic_template, paper_text,
                           image_labels, monkeypatch)

    # Assembled exactly as `_final_review` assembles it, check blocks and all:
    # `include_checks=False` is part of what the reviewer is shown, so a test
    # that passed the full dict would be testing a message the engine never
    # builds.
    blocks = build_review_user_blocks(
        orch.study_id, orch.paper_text, orch.figures,
        orch.extraction_record.to_dict(include_checks=False))
    rendered = "\n".join(b["text"] for b in blocks if b["type"] == "text")

    # The challenged value IS in the message (the reviewer reviews it) ...
    assert "Aim A" in rendered
    # ... but nothing about the challenge is.
    assert CHALLENGE_RATIONALE not in rendered
    lowered = rendered.lower()
    for word in FORBIDDEN:
        assert word not in lowered, word


def test_the_tool_catalogue_names_no_checker(synthetic_template):
    # The message is not the only thing the reviewer reads. `_final_review`
    # builds its tools from the same template the extractor's come from, so a
    # tool DESCRIPTION is a channel into the reviewer exactly like the prompt
    # is, and it is one an assembled-message test cannot see. Wording such as
    # "triggers a checker pass that may require revisions before the session
    # can finalise" on `mark_complete`, or "do not use this to escape a checker
    # challenge you disagree with" on `abandon_extraction`, announces to the
    # reviewer the one thing it must not know. A description that drifts from
    # the code leaks whatever it still names, true of the engine or not.
    #
    # Checked over BOTH role catalogues, which genuinely differ (the reviewer
    # has no `record_initial_check`, and `mark_complete` is described to each
    # role in its own words), so checking only the extractor's would leave the
    # reviewer-specific wording — the wording the reviewer actually reads —
    # unexamined.
    from meltiro.tools import all_tool_definitions

    # `rationale` is dropped from the list here, unlike in the assembled
    # message: `remove_record` legitimately asks for a rationale for the
    # removal, which is ordinary English about the model's own argument rather
    # than a reference to a checker's. Every other word stays, because none of
    # them has an innocent reading inside a tool description.
    forbidden = tuple(w for w in FORBIDDEN if w != "rationale")
    catalogues = all_tool_definitions(synthetic_template)
    assert set(catalogues) == {ROLE_EXTRACTOR, ROLE_REVIEW}
    for role, tools in catalogues.items():
        text = json.dumps(tools).lower()
        for word in forbidden:
            assert word not in text, (role, word)


def test_build_review_user_blocks_has_no_challenge_parameter():
    # A parameter is a door. Deleting the block but keeping the parameter would
    # leave the next caller free to reopen it, so the signature is pinned too.
    params = inspect.signature(build_review_user_blocks).parameters
    assert list(params) == [
        "study_id", "paper_text", "figures", "extraction_record_dict",
        "image_captions", "image_notes", "image_tables",
            "supplements"]


def test_final_review_passes_the_reviewer_nothing_but_the_output():
    # The engine side of the same guarantee: the reviewer's message is built
    # from the study, the paper, the figures, and the extraction output, and
    # nothing else reaches it.
    #
    # It is pinned across the two methods that share the construction, because
    # that is where the guarantee now lives: `_review_message` assembles the
    # message — the one construction, so the preview a dry run prints and the
    # message the review turn sends cannot differ — and `_final_review`
    # decides what output goes into it.
    build = inspect.getsource(Orchestrator._review_message)
    assert "unresolved" not in build.lower()
    assert "build_review_user_blocks(" in build
    src = inspect.getsource(Orchestrator._final_review)
    assert "unresolved" not in src.lower()
    assert "_review_message(" in src
    # And it hands over the output WITHOUT the check blocks. The withholding is
    # a keyword at this one call site, so it is pinned at this one call site.
    assert "to_dict(include_checks=False)" in src


# ---------------------------------------------------------------------------
# Nor is the reviewer shown the extractor's own account of its run
# ---------------------------------------------------------------------------

class TestTheCheckBlocksAreWithheld:
    """The extractor's self-assessment is an anchor too, so it is withheld.

    Both check blocks describe how the extraction went rather than what the
    paper says, and the reviewer records its OWN. Showing it the extractor's
    would give its second opinion the same head start the checker's verdicts
    would. Withholding has to hold on every surface the reviewer can reach:
    the assembled message it is handed, and the two view tools it can call.
    """

    def _recorded(self, tmp_path, template, paper_text, image_labels):
        """An orchestrator whose extractor recorded both of its check
        blocks, each carrying a value distinctive enough to find by
        substring."""
        orch = _orch(tmp_path, template, paper_text, image_labels)
        orch.dispatcher.dispatch(
            "record_initial_check", dict(EXTRACTOR_INITIAL_CHECK))
        orch.extraction_record.record_quality_check(
            dict(EXTRACTOR_QUALITY_CHECK), role=ROLE_EXTRACTOR)
        orch.dispatcher.dispatch("update_study", {"study": {
            "primary_aim": _env("Aim A"),
        }})
        return orch

    def test_both_blocks_are_recorded_and_role_keyed(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # The control for everything below: the blocks really are there, under
        # the extractor's own key, so a later "not shown" assertion cannot
        # pass merely because nothing was ever recorded.
        orch = self._recorded(tmp_path, synthetic_template, paper_text,
                              image_labels)
        full = orch.extraction_record.to_dict()
        assert full["initial_check"] == {
            ROLE_EXTRACTOR: EXTRACTOR_INITIAL_CHECK}
        assert full["quality_check"] == {
            ROLE_EXTRACTOR: EXTRACTOR_QUALITY_CHECK}

    def test_the_assembled_message_carries_neither_block(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        orch = self._recorded(tmp_path, synthetic_template, paper_text,
                              image_labels)
        shown = orch.extraction_record.to_dict(include_checks=False)
        # Dropped entirely rather than emptied: an empty block would read as
        # "the extractor recorded nothing", which is a different claim.
        assert set(shown) == {"study", "records"}

        blocks = build_review_user_blocks(
            orch.study_id, orch.paper_text, orch.figures, shown)
        rendered = "\n".join(b["text"] for b in blocks if b["type"] == "text")
        # The extraction IS in the message (the reviewer reviews it) ...
        assert "Aim A" in rendered
        # ... and neither self-assessment is, by value or by block name.
        assert "4242" not in rendered
        assert EXTRACTOR_QUALITY_CHECK[
            "deviation_from_expectations"] not in rendered
        assert "initial_check" not in rendered
        assert "quality_check" not in rendered

    def test_view_study_fields_shows_the_blocks_only_to_the_extractor(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # Stripping the blocks from the assembled message buys nothing if the
        # reviewer can fetch them with one tool call, so the view tools strip
        # them too. The extractor's own view is the positive control.
        orch = self._recorded(tmp_path, synthetic_template, paper_text,
                              image_labels)
        mine = orch.dispatcher.dispatch(
            "view_study_fields", {}, role=ROLE_EXTRACTOR)["view"]
        assert mine["initial_check"] == EXTRACTOR_INITIAL_CHECK
        assert mine["quality_check"] == EXTRACTOR_QUALITY_CHECK

        theirs = orch.dispatcher.dispatch(
            "view_study_fields", {}, role=ROLE_REVIEW)["view"]
        assert set(theirs) == {"study"}
        assert "Aim A" in json.dumps(theirs)  # it still sees the extraction
        assert "4242" not in json.dumps(theirs)

    def test_view_summary_reports_the_block_counts_only_to_the_extractor(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # Even the filled/total counts are withheld: "the extractor answered
        # all three" is itself a reading of how the run went.
        orch = self._recorded(tmp_path, synthetic_template, paper_text,
                              image_labels)
        mine = orch.dispatcher.dispatch(
            "view_summary", {}, role=ROLE_EXTRACTOR)["view"]
        assert mine["initial_check"] == {"filled": 3, "total": 3}
        assert mine["quality_check"] == {"filled": 1, "total": 1}

        theirs = orch.dispatcher.dispatch(
            "view_summary", {}, role=ROLE_REVIEW)["view"]
        assert "initial_check" not in theirs
        assert "quality_check" not in theirs
        # The reviewer still gets the study/record counts it came for.
        assert theirs["study_fields"]["filled"] == 1

    def test_the_reviewer_cannot_write_over_the_extractors_initial_check(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # `record_initial_check` is not in the reviewer's catalogue, and a
        # model can always name a tool it was not given, so the dispatcher
        # refuses it by role rather than trusting the catalogue to be the
        # whole guard.
        orch = self._recorded(tmp_path, synthetic_template, paper_text,
                              image_labels)
        res = orch.dispatcher.dispatch(
            "record_initial_check", {"text_readable": False},
            role=ROLE_REVIEW)
        assert res["status"] == "validation_failed"
        assert [e["code"] for e in res["errors"]] == [
            "tool_not_available_to_role"]
        assert orch.extraction_record.initial_check_for(ROLE_EXTRACTOR) == \
            EXTRACTOR_INITIAL_CHECK
        assert orch.extraction_record.initial_check_for(ROLE_REVIEW) == {}

    def test_the_reviewers_quality_check_lands_beside_the_extractors(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # The reviewer records its own through its own `mark_complete`, under
        # its own key. Two opinions kept apart beat one with the author lost.
        orch = self._recorded(tmp_path, synthetic_template, paper_text,
                              image_labels)
        res = orch.dispatcher.dispatch("mark_complete", {
            "summary": "reviewed",
            "quality_check": dict(REVIEWER_QUALITY_CHECK),
        }, role=ROLE_REVIEW)
        assert res["status"] == "ok"
        assert res["applied_changes"]["recorded_by"] == ROLE_REVIEW
        assert orch.extraction_record.quality_check == {
            ROLE_EXTRACTOR: EXTRACTOR_QUALITY_CHECK,
            ROLE_REVIEW: REVIEWER_QUALITY_CHECK,
        }


# ---------------------------------------------------------------------------
# The reviewer-side toggle
# ---------------------------------------------------------------------------

class TestCheckReviewerEdits:
    def _reviewer_write(self, orch):
        # Dispatched as the REVIEWER, which is what the review loop does: the
        # initial-check ordering gate is the extractor's, so a reviewer edit is
        # ungated whatever the extractor did or did not record.
        res = orch.dispatcher.dispatch("update_study", {"study": {
            "sample_size": {"value": 348,
                            "evidence": "<q>348 units</q>"},
        }}, role=ROLE_REVIEW)
        assert res["applied_fields"] == ["study.sample_size"]
        if orch.check_reviewer_edits:
            orch._check_applied_fields(res, stage="review")
        return res

    def test_off_by_default_the_reviewers_writes_are_not_checked(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        orch = _orch(tmp_path, synthetic_template, paper_text, image_labels)
        assert orch.check_reviewer_edits is False
        seen = _stub_fanout(monkeypatch)
        res = self._reviewer_write(orch)
        assert seen == []
        assert "checker_challenges" not in res
        assert "_checker_verdicts" not in res

    def test_on_the_reviewer_sees_a_challenge_on_its_own_field(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        orch = _orch(tmp_path, synthetic_template, paper_text, image_labels,
                     check_reviewer_edits=True)
        seen = _stub_fanout(monkeypatch)
        res = self._reviewer_write(orch)

        assert seen == ["study.sample_size"]
        # The challenge is on the field the reviewer just proposed, and only
        # that field: it is a note on its own value, not a briefing.
        assert list(res["checker_challenges"]) == ["study.sample_size"]
        sent = json.loads(result_to_model_text(res))
        assert list(sent["checker_challenges"]) == ["study.sample_size"]
        # And the verdict record says which stage the check served.
        assert res["_checker_verdicts"][
            "study.sample_size"]["stage"] == "review"

    def test_it_spends_the_same_per_field_budget(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # One budget per field for the whole session, whichever stage wrote it.
        orch = _orch(tmp_path, synthetic_template, paper_text, image_labels,
                     check_reviewer_edits=True)
        orch.max_checks_per_field = 1
        seen = _stub_fanout(monkeypatch)
        # The extractor writes and spends the field's only check. Its opening
        # `record_initial_check` is what lets that write land at all.
        orch.dispatcher.dispatch(
            "record_initial_check", dict(EXTRACTOR_INITIAL_CHECK))
        extractor_res = orch.dispatcher.dispatch("update_study", {"study": {
            "sample_size": {"value": 300,
                            "evidence": "<q>348 units</q>"},
        }})
        orch._check_applied_fields(extractor_res, stage="extractor")
        assert seen == ["study.sample_size"]
        # The reviewer then rewrites it; the budget is spent, so no re-check.
        res = self._reviewer_write(orch)
        assert seen == ["study.sample_size"]
        assert "_checker_verdicts" not in res


@pytest.mark.parametrize("flag", [True, False])
def test_the_toggle_moves_the_structure_fingerprint(flag):
    from meltiro.fingerprint import structure_hash
    on = structure_hash(2, check_reviewer_edits=True)
    off = structure_hash(2, check_reviewer_edits=False)
    assert on != off
    assert structure_hash(2, check_reviewer_edits=flag) in (on, off)
