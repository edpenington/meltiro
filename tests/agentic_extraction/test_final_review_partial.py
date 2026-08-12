"""Final-review edit loop: partial reviewer edits count as applied.

A reviewer `update_study` that returns `partial` (some fields applied, some
rejected) already wrote its valid subset to disk. The accounting must treat
that as applied, rather than logging it as if nothing had happened: a reviewer
that revised the extraction and one that changed nothing are different facts,
and only the second gets the `final_review_edits_none_applied` event.

These drive `_final_review` offline: a real Session, ToolDispatcher, and
ExtractionRecord back the orchestrator, but the client is faked and the prompt
builders and usage/logging are stubbed, so nothing touches the network. The
reviewer-side checker is off (the shipped default), so no checker call is
reachable from here either.
"""

from types import SimpleNamespace

import meltiro.instrument as instrument_mod
import meltiro.orchestrator as orch_mod
from meltiro.extraction_record import ExtractionRecord
from meltiro.orchestrator import Orchestrator
from meltiro.session import Session
from meltiro.tools import ToolDispatcher


def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


# The synthetic template's one REQUIRED quality-check variable. `mark_complete`
# carries the caller's quality check as a required argument, so a scripted
# conclusion has to supply one; the reviewer's is filed under its own role.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


def _mark_complete(tool_id, summary):
    return _tool_use(tool_id, "mark_complete",
                     {"summary": summary,
                      "quality_check": dict(QUALITY_CHECK)})


def _resp(*blocks):
    return SimpleNamespace(content=list(blocks),
                           usage=SimpleNamespace(input_tokens=0,
                                                 output_tokens=0))


class _FakeStream:
    def __init__(self, response):
        self._response = response
        self.text_stream = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class _FakeClient:
    """Fake Anthropic client that records the kwargs of every call.

    `calls` keeps each request the orchestrator actually made, so a test can
    assert on the WIRE request and not merely on the resolved attribute or the
    fingerprint. Without it the review call site is untestable: per-role
    temperature makes `_call_review` and `_compute_review_fp` two independent
    reads of `self.review_sampling` that can drift apart. A fingerprint
    recording one temperature while the wire sends another is precisely what
    `providers.resolved_decoding_params` promises cannot happen ("a fingerprint
    folds in exactly what is sent"), so the wire side needs pinning too.

    A single `response` is replayed for every turn. A LIST is played in order
    with its last entry sticking, which is what lets a test drive a genuine
    multi-turn review: the reviewer loops, so "the call" means EVERY call, and
    a per-turn read of the wrong attribute would mis-send on every one.
    """

    def __init__(self, response):
        self.calls = []
        self.messages = SimpleNamespace(stream=self._stream)
        self._responses = (list(response) if isinstance(response, list)
                           else [response])

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return _FakeStream(self._responses[idx])


def _review_orch(tmp_path, template, paper_text, image_labels, response,
                 monkeypatch):
    """A real Orchestrator with __init__ bypassed, wired so `_final_review`
    runs offline against a canned review `response`."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.template = template
    record = ExtractionRecord()
    orch.extraction_record = record
    orch.dispatcher = ToolDispatcher(record, template, paper_text, image_labels)
    orch.session = Session.create(
        "demo-001",
        config_fp="config_fp:abcabcabcabc",
        checker_fp="checker_fp:def",
        review_fp="review_fp:xyz",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        extractor_model="opus", checker_model="sonnet",
        review_model="opus",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        runs_dir=tmp_path,
    )
    orch.study_id = "demo-001"
    orch.paper_text = paper_text
    orch.figures = []
    orch.image_labels = image_labels
    # No bundle here, so no declared exhibit captions: the label list renders
    # exactly as it did before captions existed.
    orch.image_captions = {}
    orch.reference_lists = {}
    orch.config = SimpleNamespace(review_system_path=None)
    orch.max_tool_calls = 100
    orch.max_review_tool_calls = 30
    orch.max_checks_per_field = 2
    # The shipped default: the reviewer's own writes are not checked, so no
    # checker call is reachable from this loop.
    orch.check_reviewer_edits = False
    orch._check_counts = {}
    orch.review_model = "claude-opus-4-7"
    orch.review_max_tokens = 1024
    orch.sampling = {"temperature": 0.0}
    # Review-loop bounds + turn counter (the reviewer now runs a bounded loop).
    orch.max_consecutive_text_only_turns = 3
    orch.max_consecutive_identical_failures = 5
    orch._turn_counter = 0
    # The reviewer reads its own temperature, not the extractor's. __init__ is
    # bypassed here, so the inherit-when-None resolution never runs and the
    # attribute has to be set explicitly.
    orch.review_sampling = {"temperature": 0.0}
    # Likewise the reviewer's own thinking spec, for the same reason. None is
    # the "say nothing" state every bundle that names no thinking key gets, and
    # it is what these accounting tests want: no thinking parameter reaches the
    # wire, so the review call is byte-identical to what it was before the seam.
    orch.review_thinking = None

    # One client instance, held on the orchestrator, so a test can read back the
    # kwargs of the review call it made (`orch._fake_client.calls`).
    orch._fake_client = _FakeClient(response)
    orch._anthropic_client = lambda: orch._fake_client
    # Keep the test to the accounting under study: stub the prompt builders,
    # usage accounting, verbatim API log, and the per-field checker fan-out.
    # The reviewer's system prompt is rendered by the instrument and its user
    # blocks by the orchestrator, so each builder is stubbed where its call
    # site binds it.
    monkeypatch.setattr(instrument_mod, "build_review_system_message",
                        lambda *a, **k: "SYS")
    monkeypatch.setattr(orch_mod, "build_review_user_blocks",
                        lambda *a, **k: [{"type": "text", "text": "U"}])
    orch._accumulate_usage = lambda *a, **k: None
    orch.session.log_api_call = lambda *a, **k: None
    return orch


def _events(orch):
    return [e["event"] for e in orch.session.read_events()]


def _inspect_then_conclude():
    """A two-turn review: the reviewer inspects, sees the result, then ends.

    The point of the loop, and the point of driving these through more than one
    turn: `_call_review` is reached once per turn, so a wire assertion made on
    one call pins one turn. Every turn must carry the reviewer's decoding
    params, not just the first.
    """
    return [
        _resp(_tool_use("r1", "view_summary", {})),
        _resp(_mark_complete("r2", "reviewed")),
    ]


class TestReviewCallUsesReviewTemperature:
    """EVERY turn of the reviewer's loop must send the REVIEWER's temperature.

    `_call_review` and `_compute_review_fp` read `self.review_sampling`
    independently. Pinning only the fingerprint (test_cli.py) leaves the call
    site free to drift to the extractor's `self.sampling`, which would make
    review_fp record one temperature while the wire sends another: the one
    thing `providers.resolved_decoding_params` promises cannot happen. This
    asserts the wire request itself.

    The reviewer issues a call per turn through a single helper, so one wrong
    read mis-sends on EVERY turn of every review rather than once. These
    therefore drive a multi-turn review and assert the temperature of each
    call, not of `calls[0]`: an assertion on the first call alone would pass
    against a loop that read the right attribute once and the wrong one
    thereafter.
    """

    def test_every_review_turn_sends_the_reviewers_temperature_not_the_extractors(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        orch = _review_orch(tmp_path, synthetic_template, paper_text,
                            image_labels, _inspect_then_conclude(), monkeypatch)
        # A temperature-accepting review model, so the parameter reaches the
        # wire at all (the shipped claude-opus-4-8 rejects it, which would make
        # this assertion vacuous), and two DISTINCT values so the assertion can
        # only pass by reading the reviewer's.
        orch.review_model = "claude-sonnet-4-6"
        orch.sampling = {"temperature": 0.9}
        orch.review_sampling = {"temperature": 0.1}

        assert orch._final_review() == "review_clean"

        # Non-vacuous in both directions: the review really did loop (more than
        # one call), and every one of its calls carried the reviewer's value.
        assert len(orch._fake_client.calls) == 2
        assert [c["temperature"] for c in orch._fake_client.calls] == [0.1, 0.1]

    def test_every_review_turn_omits_temperature_for_no_temperature_model(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # The refusal applies on the live call, not just in the fingerprint:
        # a model that refuses temperature is sent none at all (Anthropic
        # returns 400 for Opus 4.7+ if it is present), whatever the config
        # asked for.
        # It has to hold per turn for the same reason the value does.
        orch = _review_orch(tmp_path, synthetic_template, paper_text,
                            image_labels, _inspect_then_conclude(), monkeypatch)
        orch.review_model = "claude-opus-4-8"
        orch.review_sampling = {"temperature": 0.7}

        assert orch._final_review() == "review_clean"

        assert len(orch._fake_client.calls) == 2
        assert all("temperature" not in c for c in orch._fake_client.calls)


class TestPartialReviewEditCountsAsApplied:
    def test_partial_edit_is_not_booked_as_no_change(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        good_var = next(iter(
            ToolDispatcher(ExtractionRecord(), synthetic_template, paper_text,
                           image_labels)._study_field_specs))
        # One valid field (set to null) plus one unknown field: the dispatcher
        # applies the valid subset and returns `partial`. `mark_complete` rides
        # in the same batch so the review loop concludes on this turn.
        response = _resp(
            _tool_use("r1", "update_study", {"study": {
                good_var: {"value": None, "evidence": None},
                "not_a_real_field": {"value": "x", "evidence": ["<q>x</q>"]},
            }}),
            _mark_complete("mc", "revised one field"),
        )
        orch = _review_orch(tmp_path, synthetic_template, paper_text,
                            image_labels, response, monkeypatch)

        status = orch._final_review()

        # The dispatch was a genuine partial. `mark_complete` is dispatched
        # too now (it is how the reviewer's own quality check gets recorded),
        # so the batch produces two calls; only the first is the edit under
        # study.
        review_calls = [e for e in orch.session.read_events()
                        if e["event"] == "review_tool_call"]
        assert [e["tool"] for e in review_calls] == ["update_study",
                                                     "mark_complete"]
        assert review_calls[0]["result"]["status"] == "partial"
        # The reviewer's `mark_complete` is never gated: it always answers ok,
        # and its quality check is filed under the reviewer's own role key,
        # beside the extractor's rather than over it.
        assert review_calls[1]["result"]["status"] == "ok"
        assert orch.extraction_record.quality_check == {
            "review": QUALITY_CHECK}
        # The valid subset really is on disk.
        assert good_var in orch.extraction_record.study
        # The partial counted as applied, so the nothing-landed branch did NOT
        # fire.
        events = _events(orch)
        assert "final_review_edits_none_applied" not in events
        assert status == "review_clean"

    def test_an_edit_that_landed_nothing_is_recorded_as_such(
            self, tmp_path, synthetic_template, paper_text, image_labels,
            monkeypatch):
        # Every field unknown: validation_failed, nothing written. The reviewer
        # tried and nothing landed, which is the one case the event names, so
        # the partial fix must not swallow it.
        response = _resp(
            _tool_use("r1", "update_study", {"study": {
                "not_real_1": {"value": "a", "evidence": ["<q>a</q>"]},
                "not_real_2": {"value": "b", "evidence": ["<q>b</q>"]},
            }}),
            _mark_complete("mc", "no changes needed"),
        )
        orch = _review_orch(tmp_path, synthetic_template, paper_text,
                            image_labels, response, monkeypatch)

        status = orch._final_review()

        review_calls = [e for e in orch.session.read_events()
                        if e["event"] == "review_tool_call"]
        assert review_calls[0]["tool"] == "update_study"
        assert review_calls[0]["result"]["status"] == "validation_failed"
        # No field written; the reserved scope-note key holds its initial null.
        assert orch.extraction_record.study == {"notes": None}
        events = _events(orch)
        assert "final_review_edits_none_applied" in events
        assert status == "review_clean"
