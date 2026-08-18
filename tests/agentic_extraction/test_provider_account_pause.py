"""A provider refusing over the ACCOUNT pauses the run; anything else ends it.

`direktoro.ProviderAccountError` means the provider refused over who is asking
rather than what was asked — an exhausted balance or spend cap, a key absent,
revoked, or not entitled to the model. Nothing about the extraction is wrong
and the fix is outside the process, which is the same shape as a tool-call cap
and resumable on the same terms.

The axis is NOT "can waiting fix it". A malformed request is exactly as
unfixable by waiting as an empty balance, and a run that paused on one would
resume into it for ever. So the negative case below carries as much weight as
the positives: a plain `ProviderError` must still finalise `error`.

Two legs reach a provider and can raise it. The extractor's propagates
naturally; the reviewer's does not, because `_review_loop` wraps its call in a
bare `except Exception` and would otherwise flatten the refusal into an
`"error"` outcome before anything could classify it.

And the pause is only worth having if the resume is cheap, which is what the
re-entry tests are about: a session that already holds the extractor's
completion claim must resume into the REVIEWER, not pay a live extractor turn
to be told again what run.json already records.

Fully offline: a real Session, dispatcher and extraction record back the
orchestrator; every provider call is stubbed.
"""

import json

import pytest
from direktoro import ProviderAccountError, ProviderError

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.run_log import load_log

pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
REVIEWER = "claude-opus-4-8"

# The shape direktoro actually hands over for a spent account: the OpenAI SDK
# renders `Error code: {status} - {body}`, so the provider's instruction
# arrives inside a stringified body dict rather than as a clean sentence. The
# fixture mirrors it so nothing here is asserted against a tidier message than
# the one that reaches an operator.
SPENT = (
    "Error code: 429 - {'message': 'You have no credits remaining. Add "
    "credits to continue using the API at "
    "https://platform.openai.com/settings/organization/billing/.', "
    "'type': 'insufficient_quota', 'param': None, "
    "'code': 'credit_balance_exhausted'}")


def _orch(config_dir, bundle_dir, out_dir, *, final_review=False):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(
            max_tokens=1024, checker_model="claude-sonnet-4-6"),
        review_model=REVIEWER if final_review else None,
        max_checks_per_field=0, final_review=final_review,
        max_tool_calls=50,
        extractor_max_tokens=4096,
        review_max_tokens=4096,
    )


def _meta(session_dir):
    return json.loads((session_dir / "diagnostics" / "run.json").read_text())


def _events(session_dir):
    path = session_dir / "diagnostics" / "tool_calls.jsonl"
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The extractor leg
# ---------------------------------------------------------------------------

def test_an_account_refusal_pauses_instead_of_finalising(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    def _refused():
        raise ProviderAccountError(SPENT)
    monkeypatch.setattr(orch, "_extractor_loop", _refused)

    status = orch.run()

    assert status == "in_progress"
    meta = _meta(orch.session.session_dir)
    assert meta["status"] == "in_progress"
    assert meta["pause_reason"] == "provider_account"
    # A pause is not a finish: no terminal status, no terminate event, and
    # above all no run-log entry. The log is a ledger a consumer sums into a
    # bill, and a paused run has not produced the thing an entry claims.
    assert meta["current_phase"] != "done"
    names = [e.get("event") for e in _events(orch.session.session_dir)]
    assert "terminate" not in names
    assert not [e for e in load_log(out)
                if e.get("session_dir") == str(orch.session.session_dir)]
    # The provider's own words are recorded verbatim, in the event log rather
    # than run.json: the pause record carries what makes it resumable, the
    # event log carries what happened.
    refusals = [e for e in _events(orch.session.session_dir)
                if e.get("event") == "provider_account_refused"]
    assert len(refusals) == 1
    assert refusals[0]["message"] == SPENT
    # And an operator reads it as the run stops, not only in the transcript.
    assert SPENT in capsys.readouterr().err


def test_an_ordinary_provider_error_still_ends_the_run(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The load-bearing negative. A malformed request cannot be fixed by
    # waiting either, so "is it retryable" is the wrong axis and the pause
    # must key on the account class alone. A run that paused here would resume
    # into the same refusal for ever, because a resume sends the same inputs.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    def _refused():
        raise ProviderError("Error code: 400 - malformed request")
    monkeypatch.setattr(orch, "_extractor_loop", _refused)

    assert orch.run() == "error"
    meta = _meta(orch.session.session_dir)
    assert meta["status"] == "error"
    assert "pause_reason" not in meta
    assert len([e for e in load_log(out)
                if e.get("session_dir") == str(orch.session.session_dir)]) == 1


# ---------------------------------------------------------------------------
# The review leg, which flattens every other failure into a status word
# ---------------------------------------------------------------------------

def test_the_review_leg_pauses_rather_than_flattening_the_refusal(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, capsys):
    # `_review_loop` catches Exception around its provider call and returns
    # ("error", ...), which run() finalises. An account refusal has to reach
    # run() as itself, so the loop lets this one class through.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    def _refused(*a, **kw):
        raise ProviderAccountError(SPENT)
    monkeypatch.setattr(orch, "_call_review", _refused)

    assert orch.run() == "in_progress"
    meta = _meta(orch.session.session_dir)
    assert meta["status"] == "in_progress"
    assert meta["pause_reason"] == "provider_account"
    names = [e.get("event") for e in _events(orch.session.session_dir)]
    assert "review_provider_account_refused" in names
    assert "provider_account_refused" in names
    # The reviewer's generic failure note is NOT what this took: that one ends
    # the run, and using it here would have lost the refusal's class.
    assert "review_error" not in names
    assert not [e for e in load_log(out)
                if e.get("session_dir") == str(orch.session.session_dir)]


def test_an_ordinary_review_failure_still_ends_the_run(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The same negative on the review leg: everything the reviewer's catch-all
    # already handled keeps being handled by it.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    def _refused(*a, **kw):
        raise ProviderError("Error code: 400 - malformed request")
    monkeypatch.setattr(orch, "_call_review", _refused)

    assert orch.run() == "error"
    names = [e.get("event") for e in _events(orch.session.session_dir)]
    assert "review_error" in names
    assert "provider_account_refused" not in names


# ---------------------------------------------------------------------------
# What the pause is worth: re-entry
# ---------------------------------------------------------------------------

def _pause_in_review(orch, monkeypatch, *, edit=False):
    """Drive `orch` to a real account pause inside the final-review stage.

    The extractor is stubbed at the loop boundary; `_final_review` itself runs,
    so the phase is set the way a live run sets it, and the refusal is raised
    from inside the review loop where a live one would arrive.
    """
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    def _review(*a, **kw):
        if edit:
            # What a reviewer is FOR. It clears the extractor's completion
            # claim as it lands, which is why the claim cannot be the signal
            # a resume routes on.
            orch.extraction_record.apply_update_study(
                study={"title": {"value": "corrected by the reviewer",
                                 "evidence": None}})
            orch.session.write_extraction_record(orch.extraction_record)
        raise ProviderAccountError(SPENT)
    monkeypatch.setattr(orch, "_review_loop", _review)
    return orch.run()


@pytest.mark.parametrize("edit", [False, True])
def test_a_review_pause_resumes_into_review_whatever_the_reviewer_touched(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch, edit):
    # The defect this routing exists to avoid. A reviewer that edited before
    # the refusal has cleared the extractor's completion claim, so routing on
    # that claim sent the resume back into the extractor — replaying the whole
    # conversation, paying a live turn, and handing the extractor a record the
    # reviewer had altered underneath it. The phase is what records which
    # stage the run reached, and it is not moved by an edit.
    out = tmp_path / "runs"
    first = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    first.prepare_new_session()
    session_dir = first.session.session_dir

    assert _pause_in_review(first, monkeypatch, edit=edit) == "in_progress"
    meta = _meta(session_dir)
    assert meta["pause_reason"] == "provider_account"
    assert meta["current_phase"] == "final_review"

    second = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    second.resume_session(session_dir)
    called = []
    monkeypatch.setattr(second, "_extractor_loop",
                        lambda: called.append("extractor") or "unreachable")
    monkeypatch.setattr(second, "_final_review", lambda: "review_clean")

    assert second.run() == "complete"
    assert called == []
    assert "pause_reason" not in _meta(session_dir)


def test_a_spent_tool_call_cap_cannot_downgrade_the_account_pause(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The cap is tested at the TOP of the extractor loop, so a run can finish
    # `mark_complete_validated` with the cap exactly spent. Re-entering that
    # loop on resume returns `tool_cap_hit` at once: the resume would call no
    # provider, overwrite `pause_reason`, walk the phase backwards, and — via
    # `cli._command_status` — turn the account pause's exit 1 into exit 0,
    # telling a batch script the run progressed while the account is still
    # spent. Routing on the phase never re-enters the loop, so the cap is
    # never consulted and the signal survives.
    out = tmp_path / "runs"
    first = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    first.prepare_new_session()
    session_dir = first.session.session_dir
    first.session.meta["tool_call_count"] = first.max_tool_calls
    first.session.write_meta()

    assert _pause_in_review(first, monkeypatch) == "in_progress"

    second = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    second.resume_session(session_dir)
    monkeypatch.setattr(second, "_final_review", lambda: "review_clean")
    assert second.run() == "complete"
    meta = _meta(session_dir)
    assert meta["status"] == "complete"
    assert meta.get("pause_reason") is None


def test_the_extractors_gate_is_not_applied_to_a_reviewed_extraction(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The reviewer's `mark_complete` is ungated by design; the extractor's is
    # not. A reviewer that deliberately removed a record leaves an extraction
    # the extractor's completeness gate would reject, so a resume that
    # re-entered the extractor would demand back what the reviewer removed —
    # making a paused-and-resumed run something other than the run it would
    # have been uninterrupted.
    out = tmp_path / "runs"
    first = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    first.prepare_new_session()
    session_dir = first.session.session_dir
    assert _pause_in_review(first, monkeypatch, edit=True) == "in_progress"

    second = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    second.resume_session(session_dir)
    reviewed = []
    monkeypatch.setattr(second, "_extractor_loop",
                        lambda: pytest.fail("the extractor was re-entered"))
    monkeypatch.setattr(
        second, "_final_review",
        lambda: reviewed.append("review") or "review_clean")
    assert second.run() == "complete"
    assert reviewed == ["review"]


def test_a_resume_into_a_run_without_the_reviewer_refuses(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # Phase and configuration are recorded in different files, and the one
    # outcome that must not follow from them disagreeing is shipping
    # `complete` an extraction that skipped both stages. The drift gate should
    # make it unreachable; this is what happens if it ever is not.
    out = tmp_path / "runs"
    first = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    first.prepare_new_session()
    session_dir = first.session.session_dir
    assert _pause_in_review(first, monkeypatch) == "in_progress"

    second = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    second.resume_session(session_dir)
    # Reach past the drift gate to the state it exists to prevent.
    second.final_review = False
    assert second.run() == "error"
    assert _meta(session_dir)["status"] == "error"


def test_the_transcript_reads_the_pause_without_promising_a_resume_point(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # A reviewer-leg pause resumes into a FRESH review conversation, not the
    # one it stopped in, so the note the cap pause writes ("resumed into the
    # same conversation") would be false for exactly the case this was built
    # for.
    from meltiro.transcript import render_transcript

    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    orch.prepare_new_session()
    monkeypatch.setattr(orch, "_extractor_loop",
                        lambda: "mark_complete_validated")

    def _refused(*a, **kw):
        raise ProviderAccountError(SPENT)
    monkeypatch.setattr(orch, "_call_review", _refused)
    orch.run()

    document = render_transcript(orch.session.session_dir)
    assert "| Paused because | `provider_account` |" in document
    assert "resumed into the same conversation" not in document
    assert "into a fresh review conversation" in document
    # The provider's own words reach the reader whole, dict wrapper and all.
    assert "credit_balance_exhausted" in document


def test_a_session_written_before_the_field_existed_resumes_unchanged(
        config_dir, bundle_minimal_dir, tmp_path):
    # Absent reads as no claim, which is what such a session was already
    # doing: it re-enters the extractor exactly as it did before.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    meta_path = orch.session.session_dir / "diagnostics" / "run.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("mark_complete_flag", None)
    meta_path.write_text(json.dumps(meta))

    reloaded = _orch(config_dir, bundle_minimal_dir, out)
    reloaded.resume_session(orch.session.session_dir)
    assert reloaded.extraction_record.mark_complete_flag is False
