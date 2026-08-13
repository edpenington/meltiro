"""A call the gateway served and charged for, and the routing layer then
refused, is money that was spent.

direktoro raises `ProviderError` (and its `ProviderRouteMismatch` subclass) on a
routed response whose provenance does not hold up: the served upstream is not
the pinned one, the generation id is missing, the gateway charge is missing.
Those refusals are ABOUT a response, and they carry it — usage and cost as far
as they got — on `exc.response`, because the provider billed for it either way.

Ledgered at $0 the call is invisible: a run reports fewer tokens than it bought
and a total that quietly excludes a call. These tests pin the other behaviour —
the tokens counted, the missing charge stated as coverage, the field degraded
naming the refusal, and the paid siblings in the same batch untouched.

Fully offline: the refusals are constructed and handed to stub adapters; no
provider is reached.
"""

import pytest

import meltiro.orchestrator as orch_mod
from direktoro import (
    NormalisedResponse, NormalisedUsage, ProviderError, ProviderRouteMismatch)
from direktoro.registry import OPENROUTER_BASE_URL
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig, check_one_field, run_checker_batch
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import CheckerError
from meltiro.orchestrator import Orchestrator
from meltiro.tools import CHECKER_VERDICT_TOOL_NAME, get_tool_definitions

ROUTED = "z-ai/glm-5v-turbo"


class _ToolUse:
    type = "tool_use"

    def __init__(self, name, tool_input):
        self.name = name
        self.input = tool_input


def _verdict_block(verdict="ok", rationale="matches the quote"):
    return _ToolUse(CHECKER_VERDICT_TOOL_NAME,
                    {"verdict": verdict, "rationale": rationale})


def _billed(content=(), *, input_tokens=120, output_tokens=8,
            reported_cost=None, generation_id=None, served=None,
            model=ROUTED):
    """The response a post-billing refusal carries: real usage, and only the
    routing fields that were established before the refusal fired."""
    return NormalisedResponse(
        content=list(content),
        usage=NormalisedUsage(input_tokens=input_tokens,
                              output_tokens=output_tokens),
        resolved_model=model, provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        raw_request={"model": model}, raw_response={"model": model},
        wire_request={"model": model, "messages": []},
        decoding_params={"max_tokens": 1024},
        generation_id=generation_id, served_provider=served,
        reported_cost=reported_cost)


def _pin_mismatch(response):
    """A pin that did not hold, carrying the billed response it refuses."""
    exc = ProviderRouteMismatch(
        "routed response was served by 'Novita', not the pinned ('Z.AI',)")
    exc.response = response
    return exc


class _RefusingAdapter:
    """An adapter whose call is served, billed, and then refused."""

    def __init__(self, exc):
        self._exc = exc

    def create_message(self, **kwargs):
        raise self._exc


class _PerFieldAdapter:
    """One outcome per field path, so a mid-batch refusal can be aimed at a
    single field while its siblings answer normally."""

    def __init__(self, outcomes, default):
        self._outcomes = outcomes
        self._default = default

    def create_message(self, **kwargs):
        marker = str(kwargs.get("messages"))
        for key, outcome in self._outcomes.items():
            if key in marker:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        if isinstance(self._default, Exception):
            raise self._default
        return self._default


def _checker_config():
    return CheckerConfig(checker_model=ROUTED, max_tokens=1024, concurrency=4)


# ---------------------------------------------------------------------------
# One check
# ---------------------------------------------------------------------------

class TestOneRefusedCheck:

    def test_a_pin_mismatch_is_priced_from_the_response_it_refuses(self):
        exc = _pin_mismatch(_billed([_verdict_block()]))
        with pytest.raises(CheckerError) as caught:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_checker_config(), adapter=_RefusingAdapter(exc))
        spent = caught.value.spent
        # The call happened: its tokens are the run's whether or not the
        # receipt held up.
        assert spent["responses"] == 1
        assert spent["input_tokens"] == 120
        assert spent["output_tokens"] == 8
        # The charge was never read, so the figure states its own coverage
        # rather than claiming to be the whole of it.
        assert spent["cost_incomplete"] is True
        assert spent["unreceipted_responses"] == 1
        # And the refusal is named, not swallowed into a generic failure.
        assert "served by 'Novita'" in str(caught.value)

    def test_a_refusal_that_reached_no_response_bills_nothing(self):
        # The other kind of ProviderError: raised INSTEAD of a response. There
        # is nothing to bank, and the guard must not invent a call.
        with pytest.raises(CheckerError) as caught:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_checker_config(),
                adapter=_RefusingAdapter(ProviderError("connection refused")))
        spent = caught.value.spent
        assert spent["responses"] == 0
        assert spent["input_tokens"] == 0
        assert "cost_incomplete" not in spent

    def test_a_refused_call_whose_charge_did_arrive_is_costed(self):
        # A pin mismatch caught after the gateway's charge was read. The money
        # is known, so it is counted rather than left as coverage.
        exc = _pin_mismatch(_billed(reported_cost=0.0042,
                                    generation_id="gen-1", served="Novita"))
        with pytest.raises(CheckerError) as caught:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_checker_config(), adapter=_RefusingAdapter(exc))
        spent = caught.value.spent
        assert spent["cost_usd"] == pytest.approx(0.0042)
        assert "cost_incomplete" not in spent

    def test_the_refused_call_reaches_the_wire_log(self):
        # The wire log records what happened on the wire, and a served-then-
        # refused response happened there.
        entries = []
        exc = _pin_mismatch(_billed([_verdict_block()]))
        with pytest.raises(CheckerError):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_checker_config(), adapter=_RefusingAdapter(exc),
                api_logger=lambda req, resp, **extra: entries.append(extra),
                api_log_meta={"field_path": "study.primary_aim",
                              "check_index": 1})
        assert len(entries) == 1
        assert entries[0]["field_path"] == "study.primary_aim"
        assert entries[0]["ask"] == 0


# ---------------------------------------------------------------------------
# One refusal inside a batch
# ---------------------------------------------------------------------------

class TestARefusalMidBatch:

    def _calls(self):
        return [
            {"field_path": "study.primary_aim",
             "system_message_blocks": [],
             "user_message_blocks": [{"type": "text",
                                      "text": "FIELD-AIM"}]},
            {"field_path": "study.sample_size",
             "system_message_blocks": [],
             "user_message_blocks": [{"type": "text",
                                      "text": "FIELD-SIZE"}]},
        ]

    def test_one_field_degrades_and_its_siblings_do_not(self):
        adapter = _PerFieldAdapter(
            {"FIELD-AIM": _pin_mismatch(_billed([_verdict_block()]))},
            _billed([_verdict_block()], input_tokens=30, output_tokens=4,
                    reported_cost=0.001, generation_id="gen-ok",
                    served="Z.AI"))
        results = run_checker_batch(
            calls=self._calls(), config=_checker_config(), adapter=adapter)

        refused = results["study.primary_aim"]
        sibling = results["study.sample_size"]

        # The refused field degrades to an error-origin challenge naming the
        # refusal: an absence of checking, never an objection to the value.
        assert refused["error_origin"] is True
        assert refused["verdict"] == "challenge"
        assert "served by 'Novita'" in refused["error"]
        # ... and it is priced from the call that was billed.
        assert refused["input_tokens"] == 120
        assert refused["output_tokens"] == 8
        assert refused["cost_incomplete"] is True
        assert refused["unreceipted_responses"] == 1

        # The sibling was asked, answered, and paid for. None of that moves.
        assert sibling["error_origin"] is False
        assert sibling["verdict"] == "ok"
        assert sibling["cost_usd"] == pytest.approx(0.001)
        assert "cost_incomplete" not in sibling


# ---------------------------------------------------------------------------
# The run total states the coverage
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir, *, extractor_model=ROUTED):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor_model,
        # 4096, not the unit tests' 1024: this model reasons by default and
        # direktoro refuses a reasoning call capped below its own floor.
        checker_config=CheckerConfig(max_tokens=4096, checker_model=ROUTED),
        review_model="claude-opus-4-7",
        extractor_max_tokens=4096, review_max_tokens=4096,
    )
    orch.prepare_new_session()
    return orch


class TestTheRunStatesItsCoverage:

    def test_a_refused_checker_call_makes_the_total_a_floor(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")

        def _fake_batch(*, calls, config, on_complete=None, api_logger=None,
                        **kw):
            degraded = {
                "error": "Provider API error: pin did not hold",
                "verdict": "challenge", "rationale": "", "notes": None,
                "error_origin": True, "reprompted": 0,
                "input_tokens": 120, "output_tokens": 8,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "cost_usd": 0.0,
                "cost_incomplete": True, "unreceipted_responses": 1,
            }
            if on_complete is not None:
                on_complete("study.primary_aim", degraded)
            return {"study.primary_aim": degraded}

        monkeypatch.setattr(orch_mod, "run_checker_batch", _fake_batch)
        orch._run_checker_fanout(
            [{"field_path": "study.primary_aim", "user_message_blocks": []}])

        # The tokens are in the run's meters...
        assert orch._input_tokens == 120
        assert orch._output_tokens == 8
        # ... and the run says how far its figure reaches.
        assert orch.unreceipted_calls() == 1
        assert orch.session.meta["cost_incomplete"] is True
        role = orch._usage_by_role_record()["checker"]
        assert role["cost_incomplete"] is True
        assert role["unreceipted_calls"] == 1

    def test_a_refused_extractor_call_is_ledgered_before_the_stage_ends(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The extractor's refusals abort the stage, as they always have. What
        # changes is that the money reaches the ledger on the way out.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        exc = _pin_mismatch(_billed(input_tokens=900, output_tokens=40))

        with pytest.raises(ProviderRouteMismatch):
            orch._call_extractor(_RefusingAdapter(exc),
                                 get_tool_definitions(orch.template))

        assert orch._input_tokens == 900
        assert orch._output_tokens == 40
        assert orch.unreceipted_calls() == 1
        extractor = orch._usage_by_role_record()["extractor"]
        assert extractor["input_tokens"] == 900
        assert extractor["cost_incomplete"] is True

    def test_a_refused_extractor_call_whose_charge_arrived_is_costed(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        exc = _pin_mismatch(_billed(input_tokens=900, output_tokens=40,
                                    reported_cost=0.031,
                                    generation_id="gen-9", served="Novita"))

        with pytest.raises(ProviderRouteMismatch):
            orch._call_extractor(_RefusingAdapter(exc),
                                 get_tool_definitions(orch.template))

        assert orch.recorded_cost() == pytest.approx(0.031)
        assert orch.unreceipted_calls() == 0

    def test_a_refusal_with_no_response_leaves_the_meters_alone(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")

        with pytest.raises(ProviderError):
            orch._call_extractor(
                _RefusingAdapter(ProviderError("connection refused")),
                get_tool_definitions(orch.template))

        assert orch._input_tokens == 0
        assert orch.unreceipted_calls() == 0

    def test_a_failing_wire_log_costs_the_run_neither_spend_nor_its_error(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        """The audit trail is a record of the call; it is not the call's
        outcome, and it is not the money. A `--diagnostics full` write that
        fails must leave both where they were — same rule as
        `checker._log_ask`."""
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        exc = _pin_mismatch(_billed(input_tokens=900, output_tokens=40))

        def _boom(*args, **kwargs):
            raise OSError("no space left on device")
        monkeypatch.setattr(orch.session, "write_api_call_entry", _boom)

        # The refusal is what comes out, not the disk fault.
        with pytest.raises(ProviderRouteMismatch):
            orch._call_extractor(_RefusingAdapter(exc),
                                 get_tool_definitions(orch.template))

        # And the spend is banked: the fault was swallowed where it happened,
        # so the accounting that follows it ran as it always does.
        assert orch._input_tokens == 900
        assert orch._output_tokens == 40
        assert orch._usage_by_role_record()["extractor"]["input_tokens"] == 900
