"""Whole-run cost/token accounting across a pause-and-resume cycle.

A tool-call-cap pause writes NO run-log row (by design: the run is not
finished), so the single row written at finalise time must cover the whole
run. The accumulators are zeroed in Orchestrator.__init__, so resume_session
reseeds them from the paused session's run.json (which _pause persisted);
without the reseed, the finalise-time meta and run-log totals would
silently cover the post-resume segment ALONE.

tool_call_count needs no reseed: it is persisted incrementally in meta and
the cap check reads it from there, so the whole-run count carries across
resume on its own (which is also why resuming under the same cap re-pauses
immediately; the documented recovery is to raise the cap).
"""

import json
from types import SimpleNamespace

import pytest

from direktoro import NormalisedUsage
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.rates import Rates
from meltiro.run_log import load_log


# Every stage's key variable is present for this module: these tests
# reach the orchestrator's pre-spend key preflight, and the provider
# calls behind it are stubbed.
pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"

# The extractor's rate card, which is the only role these runs call. Costs here
# are this card's arithmetic over the usage each fake response reports, so the
# expectations below are derived the same way the run derives them and no price
# list is restated in this file.
CARD = Rates(input_per_1m=15.0, output_per_1m=75.0,
             cache_read_per_1m=1.5, cache_write_per_1m=18.75)
RATES = {"extractor": CARD}


def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


def _resp_with_usage(tool_id, input_tokens, output_tokens):
    """A one-tool-call (view_summary) response carrying real usage."""
    return SimpleNamespace(
        content=[_tool_use(tool_id, "view_summary", {})],
        usage=NormalisedUsage(input_tokens=input_tokens,
                              output_tokens=output_tokens),
    )


def _orch(config_dir, bundle_dir, out_dir, *, cap):
    """Extractor-only Orchestrator (checker and reviewer off)."""
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(
            max_tokens=1024, checker_model="claude-sonnet-4-6"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=cap,
        rates=RATES,
        extractor_max_tokens=4096,
    )


def test_resume_totals_cover_both_segments(
        config_dir, bundle_minimal_dir, tmp_path):
    out_dir = tmp_path / "runs"
    seg1_cost = CARD.cost_of_call(input_tokens=1000, output_tokens=100)
    seg2_cost = CARD.cost_of_call(input_tokens=2000, output_tokens=200)

    # ------------------------------------------------------------------
    # Segment 1: cap of 1, one tool-calling turn with known usage, pause.
    # ------------------------------------------------------------------
    orch1 = _orch(config_dir, bundle_minimal_dir, out_dir, cap=1)
    orch1.prepare_new_session()
    orch1._adapter_for_role = lambda role: object()

    def _fake1(adapter, tool_defs):
        resp = _resp_with_usage("s1", input_tokens=1000, output_tokens=100)
        # Mirror the real _call_extractor: accumulate this turn's usage.
        orch1._accumulate_usage(resp, EXTRACTOR, "extractor")
        return resp

    orch1._call_extractor = _fake1
    assert orch1.run() == "in_progress"

    session_dir = orch1.session.session_dir
    meta1 = json.loads(
        (session_dir / "diagnostics" / "run.json").read_text())
    assert meta1["status"] == "in_progress"
    assert meta1["pause_reason"] == "tool_cap_hit"
    # _pause persisted segment 1's totals to meta.
    assert meta1["input_tokens"] == 1000
    assert meta1["output_tokens"] == 100
    assert meta1["cost_usd"] == round(seg1_cost, 6)
    assert meta1["tool_call_count"] == 1
    # A pause writes no run-log row: the run is not finished.
    assert not (out_dir / "run_log.json").exists()

    # ------------------------------------------------------------------
    # Segment 2: resume (same cap, so the fingerprint gate accepts), then
    # raise the cap and complete with more known usage.
    # ------------------------------------------------------------------
    orch2 = _orch(config_dir, bundle_minimal_dir, out_dir, cap=1)
    orch2.resume_session(session_dir)

    # The reseed: accumulators carry segment 1's totals, not zero.
    assert orch2._input_tokens == 1000
    assert orch2._output_tokens == 100
    assert orch2._cost_usd == round(seg1_cost, 6)
    # And the pause marker is cleared by the resume.
    assert "pause_reason" not in orch2.session.meta

    # Raise the cap after resuming so the loop has headroom. The cap is out of
    # the fingerprint, so building segment 2 with the raised cap directly
    # would also pass the gate (see test_cap_out_of_fingerprint); raising it
    # here keeps this test focused on the accounting, not the cap.
    orch2.max_tool_calls = 50
    orch2._adapter_for_role = lambda role: object()

    def _fake2(adapter, tool_defs):
        # view_summary does not clear the mark_complete flag, so setting it
        # before the turn lets the loop end cleanly without a full valid
        # extraction output (same trick as the loop-termination tests).
        orch2.extraction_record.mark_complete()
        resp = _resp_with_usage("s2", input_tokens=2000, output_tokens=200)
        orch2._accumulate_usage(resp, EXTRACTOR, "extractor")
        return resp

    orch2._call_extractor = _fake2
    assert orch2.run() == "complete"

    # Finalise-time meta covers the WHOLE run: segment 1 + segment 2.
    meta2 = orch2.session.meta
    assert meta2["status"] == "complete"
    assert meta2["input_tokens"] == 3000
    assert meta2["output_tokens"] == 300
    assert meta2["cost_usd"] == round(seg1_cost + seg2_cost, 6)
    # tool_call_count persisted across the resume: one call per segment.
    assert meta2["tool_call_count"] == 2

    # Exactly one run-log row, carrying the whole-run totals.
    log = load_log(out_dir)
    assert len(log) == 1
    entry = log[0]
    assert entry["status"] == "complete"
    assert entry["validation_passed"] is True
    assert entry["input_tokens"] == 3000
    assert entry["output_tokens"] == 300
    assert entry["cost_usd"] == round(seg1_cost + seg2_cost, 6)
    assert entry["tool_call_count"] == 2
    # And the rates that produced that figure ride with it, in both records, so
    # neither states a cost a reader cannot recompute.
    assert meta2["cost_rates"] == {"extractor": CARD.as_record()}
    assert entry["cost_rates"] == {"extractor": CARD.as_record()}
    # The extractor is the only role that ran, so its per-role block carries
    # the whole run's spend and reseeded across the pause with it.
    assert meta2["usage_by_role"]["extractor"] == {
        "input_tokens": 3000, "output_tokens": 300,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": round(seg1_cost + seg2_cost, 6),
        "cost_rates": CARD.as_record(),
    }
    assert entry["usage_by_role"] == meta2["usage_by_role"]
