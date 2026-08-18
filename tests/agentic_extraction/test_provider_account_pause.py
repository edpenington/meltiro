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

def test_the_completion_claim_survives_the_pause(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The claim is session bookkeeping run.json holds, because the
    # consumer-facing extraction output does not carry it.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    orch.prepare_new_session()

    def _complete():
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"
    monkeypatch.setattr(orch, "_extractor_loop", _complete)

    def _refused(*a, **kw):
        raise ProviderAccountError(SPENT)
    monkeypatch.setattr(orch, "_call_review", _refused)

    orch.run()
    assert _meta(orch.session.session_dir)["mark_complete_flag"] is True


def test_a_resume_holding_the_claim_never_calls_the_extractor(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The whole point. Re-entering the extractor would replay the
    # conversation and pay a live turn for an answer run.json already holds —
    # and a model handed its own finished work back can revise it, so the turn
    # is not merely wasted.
    out = tmp_path / "runs"
    first = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    first.prepare_new_session()
    session_dir = first.session.session_dir

    def _complete():
        first.extraction_record.mark_complete()
        return "mark_complete_validated"
    monkeypatch.setattr(first, "_extractor_loop", _complete)

    def _refused(*a, **kw):
        raise ProviderAccountError(SPENT)
    monkeypatch.setattr(first, "_call_review", _refused)
    assert first.run() == "in_progress"

    # The account is fixed; the same session continues.
    second = _orch(config_dir, bundle_minimal_dir, out, final_review=True)
    second.resume_session(session_dir)
    assert second.extraction_record.mark_complete_flag is True

    called = []
    monkeypatch.setattr(second, "_extractor_loop",
                        lambda: called.append("extractor") or "unreachable")
    monkeypatch.setattr(second, "_final_review", lambda: "review_clean")

    assert second.run() == "complete"
    assert called == []
    # And the pause it resumed from is gone, so nothing later reads a stale
    # reason off a live session.
    assert "pause_reason" not in _meta(session_dir)


def test_an_edit_after_the_claim_sends_the_resume_back_to_the_extractor(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The claim is only ever true of the record exactly as it stands: every
    # field write clears it as it lands. That is what makes persisting it safe
    # — a resumed session cannot skip the extractor on a stale claim, because
    # the thing that would make it stale is the thing that clears it.
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    orch.extraction_record.mark_complete()
    orch.session.write_extraction_record(orch.extraction_record)
    assert _meta(orch.session.session_dir)["mark_complete_flag"] is True

    orch.extraction_record.apply_update_study(
        study={"title": {"value": "revised after the claim",
                         "evidence": None}})
    orch.session.write_extraction_record(orch.extraction_record)
    assert _meta(orch.session.session_dir)["mark_complete_flag"] is False


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
