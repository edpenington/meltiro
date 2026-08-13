"""Seams where meltiro and direktoro have to agree, and one place they do not.

Each test here is about a rule this repo shares with the package under it: the
model list a refusal offers, the retry schedule three roles wait on, the
counters a cache write is priced by, the vocabulary a decoding block is checked
against. Where the two packages diverge on purpose — the JSON escaping every
published fingerprint is taken over — that is pinned too, so the divergence
stays a decision rather than becoming a surprise.

Offline throughout.
"""

import inspect

import pytest

import direktoro
import meltiro.checker as checker_mod
import meltiro.fingerprint as fingerprint_mod
from direktoro import (
    NormalisedUsage, ProviderRateLimitError, RETRY_BACKOFF_SECONDS, Thinking,
    known_models, model_info)
from meltiro import cli
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig, check_one_field
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import CheckerError, RatesConfigError
from meltiro.orchestrator import Orchestrator
from meltiro.rates import Rates, cache_write_split, parse_rates

pytestmark = pytest.mark.usefixtures("stage_keys")


# ---------------------------------------------------------------------------
# (a) The model list a refusal offers is the STARTABLE one
# ---------------------------------------------------------------------------

class TestTheUnknownModelMessageOffersStartableIds:

    def test_the_suggestion_list_excludes_retired_ids(self, capsys):
        # Offering a retired id would have the operator paste it in and be
        # refused two gates later, by the retirement check.
        from types import SimpleNamespace
        retired = [m for m in known_models() if model_info(m).retired]
        if not retired:
            pytest.skip("the registry currently retires no model")
        loop_cfg = {"extractor_model": "no-such-model",
                    "extractor_max_tokens": 4096,
                    "max_checks_per_field": 0, "final_review": False}
        args = SimpleNamespace(
            max_tool_calls=None, max_checks_per_field=None, final_review=None,
            extractor_model=None, review_model=None, checker_model=None,
            diagnostics="standard", dry_run=True, out=None)
        with pytest.raises(SystemExit):
            cli._build_orchestrator(None, None, "/tmp/unused", loop_cfg, args)
        err = capsys.readouterr().err
        assert "unknown model(s)" in err
        for model in retired:
            assert model not in err

    def test_the_retired_check_asks_the_registrys_own_predicate(self):
        # The registry owns what "retired" means. Reading the flag off the
        # record here would be a second definition, free to drift from the one
        # `known_models(include_retired=False)` is built from.
        source = inspect.getsource(cli._build_orchestrator)
        assert "is_retired(m)" in source
        assert "known_models(include_retired=False)" in source


# ---------------------------------------------------------------------------
# (b) One divergence, deliberately kept
# ---------------------------------------------------------------------------

class TestTheCanonicalJsonDivergenceIsDeliberate:
    """meltiro's `canonical_json` escapes non-ASCII and direktoro's does not.

    Aligning them would move `prompts_hash`, `tool_set_hash`, `instrument_fp`
    and all three stage fingerprints for every bundle carrying a non-ASCII
    byte, and every already-published value would stop verifying. So the
    difference is pinned here and explained at the definition, rather than
    removed."""

    def test_the_two_serialisations_differ_on_non_ascii(self):
        payload = {"quote": "temperature rose 5°C"}
        assert direktoro.canonical_json(payload) != \
            fingerprint_mod.canonical_json(payload)
        # meltiro escapes...
        assert "\\u00b0" in fingerprint_mod.canonical_json(payload)
        # ... direktoro emits the character.
        assert "°" in direktoro.canonical_json(payload)

    def test_they_agree_on_everything_else(self):
        # Sorted keys and tight separators, so the divergence is exactly one
        # thing and not a general drift.
        payload = {"b": 1, "a": [2, 3]}
        assert direktoro.canonical_json(payload) == \
            fingerprint_mod.canonical_json(payload)

    def test_the_definition_says_so(self):
        doc = fingerprint_mod.canonical_json.__doc__
        assert "ensure_ascii" in doc
        assert "direktoro.canonical_json" in doc


# ---------------------------------------------------------------------------
# (c) One retry schedule for all three roles
# ---------------------------------------------------------------------------

class _AlwaysRateLimited:
    def __init__(self):
        self.calls = 0

    def create_message(self, **kwargs):
        self.calls += 1
        raise ProviderRateLimitError("429")


class TestTheCheckerRetriesOnTheSharedSchedule:

    def test_it_waits_the_shared_number_of_times(self, monkeypatch):
        # Not a schedule of its own: the checker exhausts direktoro's, so all
        # three roles back off the same way and a change to the schedule moves
        # all three together.
        monkeypatch.setattr("time.sleep", lambda _s: None)
        adapter = _AlwaysRateLimited()
        with pytest.raises(CheckerError, match="Rate limit"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=CheckerConfig(checker_model="claude-sonnet-4-6",
                                     max_tokens=1024),
                adapter=adapter)
        assert adapter.calls == len(RETRY_BACKOFF_SECONDS) + 1

    def test_each_retry_reaches_the_on_retry_callback(self, monkeypatch):
        # A failed attempt raises before the wire log runs, so this callback is
        # the only trace one leaves — which is what lets the orchestrator write
        # the same `provider_retry` event the extractor writes.
        monkeypatch.setattr("time.sleep", lambda _s: None)
        seen = []
        with pytest.raises(CheckerError):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=CheckerConfig(checker_model="claude-sonnet-4-6",
                                     max_tokens=1024),
                adapter=_AlwaysRateLimited(),
                on_retry=lambda attempt, delay, error: seen.append(
                    (attempt, delay)))
        assert [delay for _a, delay in seen] == list(RETRY_BACKOFF_SECONDS)

    def test_the_checker_keeps_no_schedule_of_its_own(self):
        source = inspect.getsource(checker_mod._ask_for_verdict)
        assert "create_message_with_retry" in source
        assert "time.sleep" not in source


# ---------------------------------------------------------------------------
# (d) One checker adapter per run
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=4096,
                                     checker_model="claude-sonnet-4-6"),
        review_model=None,
        max_checks_per_field=2, final_review=False,
        extractor_max_tokens=4096,
    )


class TestOneCheckerAdapterPerRun:

    def test_the_adapter_is_built_once_and_reused(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # A fresh adapter per fan-out is a fresh provider client, and with it a
        # fresh connection pool, for every tool call the run makes.
        built = []
        import meltiro.orchestrator as orch_mod

        def _build(config, client=None):
            built.append(config.checker_model)
            return object()
        monkeypatch.setattr(orch_mod, "_build_checker_adapter", _build)

        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        first = orch._checker_adapter()
        assert orch._checker_adapter() is first
        assert orch._checker_adapter() is first
        assert built == ["claude-sonnet-4-6"]

    def test_the_fan_out_is_handed_that_adapter(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        import meltiro.orchestrator as orch_mod

        seen = {}

        def _fake_batch(*, calls, config, adapter=None, **kw):
            seen["adapter"] = adapter
            return {}
        monkeypatch.setattr(orch_mod, "run_checker_batch", _fake_batch)

        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch.prepare_new_session()
        sentinel = object()
        orch._cached_checker_adapter = sentinel
        orch._run_checker_fanout([])
        assert seen["adapter"] is sentinel


# ---------------------------------------------------------------------------
# (e) The inert warning covers the thinking half of a decoding block
# ---------------------------------------------------------------------------

class TestAnInertThinkingFieldIsNamed:
    """A thinking field has no wire key of its own, so `key in resolved` cannot
    settle it. The inert case is real: a gateway wire that carries only an
    effort is sent nothing for a configured mode, and the run would record no
    trace of the omission."""

    def _routed_orch(self, config_dir, bundle_dir, out_dir, thinking):
        return Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model="z-ai/glm-5v-turbo",
            checker_config=CheckerConfig(max_tokens=4096,
                                         checker_model="claude-sonnet-4-6"),
            review_model=None,
            max_checks_per_field=0, final_review=False,
            thinking=thinking, extractor_max_tokens=8192,
        )

    def test_a_mode_that_reaches_nothing_is_warned_about(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # `Thinking(mode="adaptive")` resolves to nothing at all on this
        # model's wire, so the configured value moves no fingerprint.
        orch = self._routed_orch(config_dir, bundle_minimal_dir,
                                 tmp_path / "runs", Thinking(mode="adaptive"))
        orch._warn_inert_decoding_params()
        err = capsys.readouterr().err
        assert "inert-decoding-param" in err
        assert "thinking_mode" in err

    def test_an_effort_that_does_reach_the_wire_is_not_warned_about(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The pair: the same model DOES carry an effort, so nothing is inert.
        orch = self._routed_orch(config_dir, bundle_minimal_dir,
                                 tmp_path / "runs", Thinking(effort="low"))
        orch._warn_inert_decoding_params()
        assert "thinking_effort" not in capsys.readouterr().err

    def test_the_configured_values_carry_the_thinking_keys(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = self._routed_orch(
            config_dir, bundle_minimal_dir, tmp_path / "runs",
            Thinking(mode="adaptive", effort="low"))
        configured = orch._configured_decoding_values()["extractor"]
        assert configured["thinking_mode"] == "adaptive"
        assert configured["thinking_effort"] == "low"


# ---------------------------------------------------------------------------
# (f) A resume that re-prices the run says so
# ---------------------------------------------------------------------------

_CARD = Rates(input_per_1m=3.0, output_per_1m=15.0,
              cache_read_per_1m=0.3, cache_write_per_1m=3.75)
_DEARER = Rates(input_per_1m=9.0, output_per_1m=45.0,
                cache_read_per_1m=0.9, cache_write_per_1m=11.25)


def _priced_orch(config_dir, bundle_dir, out_dir, card):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=1024,
                                     checker_model="claude-sonnet-4-6"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        rates={"extractor": card}, extractor_max_tokens=4096,
    )


class TestAResumeRecordsRatesThatMoved:
    """Rate cards reach no fingerprint, so the drift gate admits a resume that
    changed them — and `meta.cost_rates` holds only the CURRENT segment's. The
    segment where the numbers moved is readable only if the previous values are
    written down when they do."""

    def _resumed_events(self, orch):
        return [e for e in orch.session.read_events()
                if e.get("event") == "resumed"]

    def test_changed_cards_ride_the_resumed_event(
            self, config_dir, bundle_minimal_dir, tmp_path):
        out = tmp_path / "runs"
        first = _priced_orch(config_dir, bundle_minimal_dir, out, _CARD)
        first.prepare_new_session()
        first._checkpoint_usage_to_meta()
        first.session.write_meta()

        second = _priced_orch(config_dir, bundle_minimal_dir, out, _DEARER)
        second.resume_session(first.session.session_dir)

        event, = self._resumed_events(second)
        assert event["previous_cost_rates"]["extractor"]["input_per_1m"] == 3.0
        assert event["cost_rates"]["extractor"]["input_per_1m"] == 9.0

    def test_unchanged_cards_say_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The ordinary case. Recording it every resume would bury the segment
        # where the numbers actually moved.
        out = tmp_path / "runs"
        first = _priced_orch(config_dir, bundle_minimal_dir, out, _CARD)
        first.prepare_new_session()
        first._checkpoint_usage_to_meta()
        first.session.write_meta()

        second = _priced_orch(config_dir, bundle_minimal_dir, out, _CARD)
        second.resume_session(first.session.session_dir)

        event, = self._resumed_events(second)
        assert "cost_rates" not in event
        assert "previous_cost_rates" not in event


# ---------------------------------------------------------------------------
# (g) Cache writes are two counters because they are two prices
# ---------------------------------------------------------------------------

class TestCacheWritesArePricedByTier:

    def test_a_reported_split_is_used_as_reported(self):
        usage = NormalisedUsage(
            cache_creation_input_tokens=1000,
            cache_creation_5m_input_tokens=600,
            cache_creation_1h_input_tokens=400)
        assert cache_write_split(usage) == (600, 400)

    def test_an_unsplit_total_is_the_five_minute_tier(self):
        # This engine writes only the plain `ephemeral` marker, which IS the
        # 5-minute tier, so an unattributed remainder is a reading rather than
        # a guess — and it is priced rather than dropped.
        usage = NormalisedUsage(cache_creation_input_tokens=1000)
        assert cache_write_split(usage) == (1000, 0)

    def test_a_partial_split_attributes_the_remainder(self):
        usage = NormalisedUsage(
            cache_creation_input_tokens=1000,
            cache_creation_1h_input_tokens=250)
        assert cache_write_split(usage) == (750, 250)

    def test_the_two_tiers_are_priced_at_their_own_rates(self):
        card = Rates(input_per_1m=0.0, output_per_1m=0.0,
                     cache_read_per_1m=0.0, cache_write_per_1m=10.0,
                     cache_write_1h_per_1m=16.0)
        assert card.cost_of_call(cache_write_tokens=1_000_000) == \
            pytest.approx(10.0)
        assert card.cost_of_call(cache_write_1h_tokens=1_000_000) == \
            pytest.approx(16.0)
        # Folded together they would price at one rate; kept apart they do not.
        assert card.cost_of_call(cache_write_tokens=1_000_000,
                                 cache_write_1h_tokens=1_000_000) == \
            pytest.approx(26.0)

    def test_one_hour_writes_with_no_one_hour_rate_raise(self):
        # Exactly what direktoro's own costing does for a counter it was handed
        # tokens but no rate for: those tokens were billed, so pricing them at
        # zero would understate the total silently.
        card = Rates(input_per_1m=1.0, output_per_1m=1.0,
                     cache_read_per_1m=1.0, cache_write_per_1m=1.0)
        assert card.cache_write_1h_per_1m is None
        with pytest.raises(ValueError, match="cache_write_1h"):
            card.cost_of_call(cache_write_1h_tokens=10)

    def test_a_card_without_the_optional_rate_still_prices_ordinary_traffic(
            self):
        card = Rates(input_per_1m=1.0, output_per_1m=1.0,
                     cache_read_per_1m=1.0, cache_write_per_1m=1.0)
        assert card.cost_of_call(input_tokens=1_000_000,
                                 cache_write_tokens=1_000_000) == \
            pytest.approx(2.0)

    def test_parse_rates_accepts_the_optional_key(self):
        block = {"input_per_1m": 3.0, "output_per_1m": 15.0,
                 "cache_read_per_1m": 0.3, "cache_write_per_1m": 3.75,
                 "cache_write_1h_per_1m": 6.0}
        card = parse_rates({"rates": {"extractor": block}})["extractor"]
        assert card.cache_write_1h_per_1m == 6.0
        assert card.as_record()["cache_write_1h_per_1m"] == 6.0

    def test_parse_rates_does_not_require_it(self):
        block = {"input_per_1m": 3.0, "output_per_1m": 15.0,
                 "cache_read_per_1m": 0.3, "cache_write_per_1m": 3.75}
        card = parse_rates({"rates": {"extractor": block}})["extractor"]
        assert card.cache_write_1h_per_1m is None

    def test_the_optional_rate_is_held_to_the_same_value_rules(self):
        block = {"input_per_1m": 3.0, "output_per_1m": 15.0,
                 "cache_read_per_1m": 0.3, "cache_write_per_1m": 3.75,
                 "cache_write_1h_per_1m": -1.0}
        with pytest.raises(RatesConfigError, match="cache_write_1h_per_1m"):
            parse_rates({"rates": {"extractor": block}})

    def test_a_table_card_states_no_one_hour_rate(self):
        # direktoro's table publishes the 5-minute write rate and no other, so
        # a table card must not invent one from it.
        from direktoro.prices import PRICES_VERSION, price_for
        entry = price_for("claude-opus-4-8")
        card = Rates.from_table(entry, PRICES_VERSION)
        assert card.cache_write_per_1m > 0
        assert card.cache_write_1h_per_1m is None


class TestTheRunPricesByTier:

    def test_a_one_hour_write_is_priced_at_its_own_rate(
            self, config_dir, bundle_minimal_dir, tmp_path):
        from types import SimpleNamespace
        card = Rates(input_per_1m=0.0, output_per_1m=0.0,
                     cache_read_per_1m=0.0, cache_write_per_1m=10.0,
                     cache_write_1h_per_1m=16.0)
        orch = _priced_orch(config_dir, bundle_minimal_dir,
                            tmp_path / "runs", card)
        orch.prepare_new_session()
        orch._accumulate_usage(
            SimpleNamespace(usage=NormalisedUsage(
                cache_creation_input_tokens=1_000_000,
                cache_creation_1h_input_tokens=1_000_000)),
            "claude-opus-4-8", "extractor")
        # 16.0, the 1-hour rate — not 10.0, which folding the tiers together
        # would have charged.
        assert orch.recorded_cost() == pytest.approx(16.0)
        # The REPORTED counter stays the unsplit total the provider gave.
        assert orch._cache_creation_tokens == 1_000_000
