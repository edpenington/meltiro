"""Cost and token aggregates survive a hard crash and never double count.

The accumulators are checkpointed into meta at the cadence they change:
`_accumulate_usage` mirrors them in memory, and the per-tool-call meta write
flushes them for free. Carrying cost_usd / input_tokens / output_tokens to
run.json ONLY via `_pause` and `_finalise` would lose every dollar and token
spent since the last of those, so a SIGKILL or a power loss mid-extractor-loop
would leave the headline numbers undercounting real spend even though the raw
calls survived in api_calls.jsonl.

Resume reseeds them from meta through a single site, so a resumed run continues
the whole-run total without recounting the pre-crash spend.

These tests are fully offline: a real Session backs a real Orchestrator, and
spend is injected by calling `_accumulate_usage` with a synthetic usage object
(the provider adapter and the API are never touched).
"""

from types import SimpleNamespace

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.rates import Rates

EXTRACTOR = "claude-opus-4-8"

# A rate card, so these runs produce a dollar figure to carry across a crash at
# all. Costing is exercised elsewhere; here the card only has to be present.
# These orchestrators run the extractor alone, so the extractor's is the only
# card that prices anything.
RATES = {"extractor": Rates(input_per_1m=15.0, output_per_1m=75.0,
                            cache_read_per_1m=1.5, cache_write_per_1m=18.75)}


def _orch(config_dir, bundle_dir, out_dir, *, cap=50, rates=RATES):
    """An extractor-only Orchestrator (checker and reviewer off), mirroring the
    cap-bonus tests' offline harness."""
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model="claude-sonnet-4-6",
                                     api_key="x"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=cap,
        sampling={"temperature": 0.0},
        rates=rates,
        extractor_max_tokens=4096,
        api_key="x",
    )


def _usage(*, inp, out, cache_create=0, cache_read=0):
    """A synthetic normalised response carrying only the usage the accumulator
    reads. Shaped like the SDK response: `.usage` with the four token fields."""
    return SimpleNamespace(usage=SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    ))


# ---------------------------------------------------------------------------
# A hard crash mid-loop keeps every dollar and token spent up to the last
# tool-call meta write.
# ---------------------------------------------------------------------------

def test_hard_crash_mid_loop_preserves_accumulated_spend(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()
    session_dir = orch.session.session_dir

    # Spend accrues mid-extractor-loop: _call_extractor calls _accumulate_usage,
    # which mirrors the running totals into meta in memory.
    orch._accumulate_usage(
        _usage(inp=1000, out=400, cache_create=200, cache_read=50),
        EXTRACTOR, "extractor")
    expected_cost = orch._cost_usd
    assert expected_cost > 0  # the rate card must have priced this call
    # A tool call flushes meta to disk (the piggybacked write). The test STOPS
    # here: no _pause, no _finalise, exactly as a SIGKILL would.
    orch.session.increment_tool_call_count()

    # Reload the session as --resume does. A fresh Orchestrator starts its
    # accumulators at zero, then reseeds them from the persisted meta.
    orch2 = _orch(config_dir, bundle_minimal_dir, out)
    orch2.resume_session(session_dir)

    assert orch2._input_tokens == 1000
    assert orch2._output_tokens == 400
    assert orch2._cache_creation_tokens == 200
    assert orch2._cache_read_tokens == 50
    assert orch2._cost_usd == pytest.approx(expected_cost, abs=1e-6)


# ---------------------------------------------------------------------------
# Across a pause and a resume the whole-run totals are counted exactly once.
# ---------------------------------------------------------------------------

def test_pause_and_resume_do_not_double_count_spend(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    orch1 = _orch(config_dir, bundle_minimal_dir, out)
    orch1.prepare_new_session()
    session_dir = orch1.session.session_dir

    # Segment 1 spend, then a clean pause persists it.
    orch1._accumulate_usage(_usage(inp=1000, out=400), EXTRACTOR,
                            "extractor")
    seg1_cost = orch1._cost_usd
    orch1._pause("tool_cap_hit")

    # Resume reseeds segment-1 spend exactly once (not zero, not doubled).
    orch2 = _orch(config_dir, bundle_minimal_dir, out)
    orch2.resume_session(session_dir)
    assert orch2._input_tokens == 1000
    assert orch2._output_tokens == 400
    assert orch2._cost_usd == pytest.approx(seg1_cost, abs=1e-6)

    # Segment 2 spend adds on top; finalise records the whole-run total.
    orch2._accumulate_usage(_usage(inp=300, out=100), EXTRACTOR,
                            "extractor")
    whole_run_cost = orch2._cost_usd
    orch2._finalise("complete")

    meta = orch2.session.meta
    # The crisp no-double-count proof: 1000 + 300 and 400 + 100, never 2000 or
    # a lost segment.
    assert meta["input_tokens"] == 1300
    assert meta["output_tokens"] == 500
    assert whole_run_cost > seg1_cost
    assert meta["cost_usd"] == round(whole_run_cost, 6)


# ---------------------------------------------------------------------------
# The checkpoint/reseed helpers round-trip, including the float cost deltas the
# checker's per-field audit accumulates outside the tool-call path.
# ---------------------------------------------------------------------------

def test_checkpoint_and_reseed_usage_round_trip(
        config_dir, bundle_minimal_dir, tmp_path):
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    orch.prepare_new_session()

    # Extractor-style token accumulation plus a checker-style fractional cost
    # delta (the audit callback adds result["cost_usd"] directly).
    orch._input_tokens = 1200
    orch._output_tokens = 640
    orch._cache_creation_tokens = 80
    orch._cache_read_tokens = 300
    orch._cost_usd = 0.123456789
    orch._checkpoint_usage_to_meta()

    meta = orch.session.meta
    assert meta["input_tokens"] == 1200
    assert meta["output_tokens"] == 640
    assert meta["cache_creation_tokens"] == 80
    assert meta["cache_read_tokens"] == 300
    assert meta["cost_usd"] == round(0.123456789, 6)

    # A fresh Orchestrator (accumulators at zero) reseeds from that meta.
    orch2 = _orch(config_dir, bundle_minimal_dir, out)
    orch2.session = orch.session
    orch2._reseed_usage_from_meta()
    assert orch2._input_tokens == 1200
    assert orch2._output_tokens == 640
    assert orch2._cache_creation_tokens == 80
    assert orch2._cache_read_tokens == 300
    assert orch2._cost_usd == round(0.123456789, 6)
