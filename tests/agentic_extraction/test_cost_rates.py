"""What a run RECORDS about what it cost, per role, priced and unpriced.

A run puts three roles to work and each runs its own model, so pricing is per
role and so is every figure in the record. A role's card comes from the operator
or from direktoro's dated table; a role with neither runs unpriced. That is the
contract these tests pin:

  - a cost never appears without the rates that produced it, in `run.json`, in
    the run log, and on screen, and the card says where it came from. The record
    is self-describing, so a reader recomputes a role's figure from that role's
    token counters and that role's card in the same record, and either agrees
    with it or knows exactly why not;
  - the operator's card wins for the role it names, and names no other: a
    bundle can price one role by hand and leave the rest on the table default;
  - a role with no card and no table entry records its token counters — which
    come from the provider and cannot rot — and NO dollar figure at all. Never a
    zero, which would read as free. Its silence withholds the run's total too,
    because a sum over the priced roles would wear a total's clothes;
  - a gateway-reported charge is a fact about what was billed rather than a
    prediction, so it is recorded whether or not a card exists, and its absence
    stays a loud fault.

Everything here is offline: synthetic usage, stubbed adapters, no API key.
"""

import json
from types import SimpleNamespace

import pytest

from direktoro import NormalisedResponse, NormalisedUsage
from direktoro.prices import PRICES_VERSION, price_for
from direktoro.registry import OPENROUTER_BASE_URL
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import RatesConfigError
from meltiro.field_history import build_field_history
from meltiro.orchestrator import Orchestrator
from meltiro.rates import RATE_KEYS, ROLE_KEYS, Rates, parse_rates
from meltiro.run_log import load_log

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
ROUTED_EXTRACTOR = "z-ai/glm-5v-turbo"

CARD = Rates(input_per_1m=15.0, output_per_1m=75.0,
             cache_read_per_1m=1.5, cache_write_per_1m=18.75)
# The extractor is the only role the harness below runs, so this is a fully
# priced run and `{}` is a fully unpriced one.
RATES = {"extractor": CARD}

# A four-key block as an operator writes one under a role.
_BLOCK = {"input_per_1m": 3.0, "output_per_1m": 15.0,
          "cache_read_per_1m": 0.3, "cache_write_per_1m": 3.75}


def _orch(config_dir, bundle_dir, out_dir, *, rates, extractor=EXTRACTOR):
    """An extractor-only Orchestrator (checker and reviewer off)."""
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=extractor,
        checker_config=CheckerConfig(checker_model=CHECKER, api_key="x"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        rates=rates,
        api_key="x",
    )


def _usage(*, inp=1000, out=400, cache_create=200, cache_read=50):
    """A synthetic response carrying only the usage the accumulator reads."""
    return SimpleNamespace(usage=SimpleNamespace(
        input_tokens=inp, output_tokens=out,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    ))


def _routed_response(reported_cost):
    return NormalisedResponse(
        content=[], usage=NormalisedUsage(input_tokens=1000, output_tokens=400),
        resolved_model=ROUTED_EXTRACTOR, provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        raw_request={}, raw_response={}, wire_request={"messages": []},
        decoding_params={"max_tokens": 32768},
        generation_id="gen-1", served_provider="Z.AI",
        reported_cost=reported_cost)


def _finished(config_dir, bundle_dir, out_dir, *, rates, usage_response,
              extractor=EXTRACTOR):
    """Run one study to `complete` over a single synthetic turn, and return
    `(orchestrator, meta, run-log entry)`.

    The turn dispatches `view_summary`, a read that clears no completion flag,
    so setting the flag first lets the loop end cleanly without a full valid
    extraction. The usage the accumulator sees comes from `usage_response`,
    which is what these tests are actually about.
    """
    orch = _orch(config_dir, bundle_dir, out_dir, rates=rates,
                 extractor=extractor)
    orch.prepare_new_session()
    orch._adapter_for_role = lambda role: object()

    def _turn(adapter, tool_defs):
        orch.extraction_record.mark_complete()
        orch._accumulate_usage(usage_response, extractor, "extractor")
        return SimpleNamespace(content=[
            SimpleNamespace(type="tool_use", id="t1", name="view_summary",
                            input={})])

    orch._call_extractor = _turn
    assert orch.run() == "complete"
    return orch, orch.session.meta, load_log(out_dir)[0]


# ---------------------------------------------------------------------------
# Parsing: `rates:` maps a role to its card
# ---------------------------------------------------------------------------

class TestParsingTheBlock:
    def test_an_absent_block_names_no_role(self):
        # Not a fault and not a card: every role then falls through to the
        # price table, which the resolver does and this function does not.
        assert parse_rates({}) == {}

    def test_each_named_role_gets_its_own_card(self):
        cards = parse_rates({"rates": {
            "extractor": dict(_BLOCK),
            "review": dict(_BLOCK, input_per_1m=9.0),
        }})
        assert set(cards) == {"extractor", "review"}
        assert cards["extractor"].input_per_1m == 3.0
        assert cards["review"].input_per_1m == 9.0

    def test_a_parsed_card_is_marked_as_the_operator_s_own(self):
        # Provenance travels with the numbers. An operator card has no reading
        # date and no table behind it, and says so with nulls rather than by
        # leaving the keys out.
        card = parse_rates({"rates": {"checker": dict(_BLOCK)}})["checker"]
        assert card.source == "operator"
        assert card.as_of is None
        assert card.table_version is None
        assert card.as_record() == dict(
            _BLOCK, source="operator", as_of=None, table_version=None)

    def test_zero_is_a_rate_and_not_an_absence(self):
        card = parse_rates({"rates": {
            "extractor": dict(_BLOCK, cache_read_per_1m=0.0)}})["extractor"]
        assert card.cache_read_per_1m == 0.0
        assert card.cost_of_call(cache_read_tokens=1_000_000) == 0.0

    def test_a_card_at_the_top_level_is_refused_with_the_shape_it_needs(self):
        # Four rates under `rates:` name no role, so they price no model. The
        # refusal has to say what the block takes, or the operator's next guess
        # is as good as their last.
        with pytest.raises(RatesConfigError) as excinfo:
            parse_rates({"rates": dict(_BLOCK)})
        message = str(excinfo.value)
        for name in ROLE_KEYS + RATE_KEYS:
            assert name in message

    def test_an_unknown_role_is_refused_naming_the_three(self):
        # A card filed under a role that does not exist prices nothing, and
        # leaves the role it was meant for silently on a table default.
        with pytest.raises(RatesConfigError) as excinfo:
            parse_rates({"rates": {"reviewer": dict(_BLOCK)}})
        message = str(excinfo.value)
        assert "'reviewer'" in message
        assert "extractor, checker, review" in message

    def test_a_bare_or_empty_block_is_refused(self):
        # Both are a block the operator meant to write and did not. Reading
        # either as "take the table defaults" would discard the intent silently.
        for value in (None, {}, [1, 2]):
            with pytest.raises(RatesConfigError):
                parse_rates({"rates": value})

    @pytest.mark.parametrize("key", sorted(_BLOCK))
    def test_an_incomplete_card_is_refused_and_named_with_its_role(self, key):
        with pytest.raises(RatesConfigError) as excinfo:
            parse_rates({"rates": {
                "checker": {k: v for k, v in _BLOCK.items() if k != key}}})
        message = str(excinfo.value)
        assert key in message
        assert "checker" in message

    @pytest.mark.parametrize("bad", [-1.0, "3.0", True, [3.0]])
    def test_an_unusable_value_is_refused(self, bad):
        with pytest.raises(RatesConfigError) as excinfo:
            parse_rates({"rates": {
                "review": dict(_BLOCK, output_per_1m=bad)}})
        assert "output_per_1m" in str(excinfo.value)

    def test_every_fault_in_every_role_is_reported_at_once(self):
        # One run of the parser, one list of everything wrong: an operator
        # fixing a three-role block should not have to re-run to find fault two.
        with pytest.raises(RatesConfigError) as excinfo:
            parse_rates({"rates": {
                "extractor": dict(_BLOCK, input_per_1m=-1.0),
                "checker": {k: v for k, v in _BLOCK.items()
                            if k != "output_per_1m"},
                "review": dict(_BLOCK, nonsense=1.0),
            }})
        message = str(excinfo.value)
        assert "extractor: input_per_1m" in message
        assert "checker: missing rate(s) output_per_1m" in message
        assert "review: unknown key(s) 'nonsense'" in message

    def test_a_role_block_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(RatesConfigError) as excinfo:
            parse_rates({"rates": {"extractor": 3.0}})
        assert "extractor: must be a mapping" in str(excinfo.value)


class TestACardFromThePriceTable:
    def test_it_carries_the_table_s_rates_dates_and_version(self):
        entry = price_for(EXTRACTOR)
        card = Rates.from_table(entry, PRICES_VERSION)
        assert card.source == "table"
        assert card.as_of == entry.as_of
        assert card.table_version == PRICES_VERSION
        # The four rates are the entry's own, translated back from the counter
        # names direktoro keys them by.
        assert card.input_per_1m == entry.input_per_1m
        assert card.output_per_1m == entry.output_per_1m
        assert card.cache_read_per_1m == entry.cache_read_per_1m
        assert card.cache_write_per_1m == entry.cache_write_per_1m

    def test_its_record_states_where_it_came_from(self):
        record = Rates.from_table(price_for(EXTRACTOR),
                                  PRICES_VERSION).as_record()
        assert record["source"] == "table"
        assert record["as_of"]
        assert record["table_version"] == PRICES_VERSION

    def test_it_prices_a_call_like_any_other_card(self):
        card = Rates.from_table(price_for(EXTRACTOR), PRICES_VERSION)
        assert card.cost_of_call(input_tokens=1_000_000) == pytest.approx(
            card.input_per_1m)


# ---------------------------------------------------------------------------
# A priced run states its cost AND the rates behind it
# ---------------------------------------------------------------------------

class TestPricedRun:
    def test_meta_and_run_log_carry_the_cost_and_its_rates(
            self, config_dir, bundle_minimal_dir, tmp_path):
        out = tmp_path / "runs"
        orch, meta, entry = _finished(
            config_dir, bundle_minimal_dir, out, rates=RATES,
            usage_response=_usage())

        expected = round(CARD.cost_of_call(
            input_tokens=1000, output_tokens=400,
            cache_write_tokens=200, cache_read_tokens=50), 6)
        assert expected > 0
        for record in (meta, entry):
            assert record["cost_usd"] == expected
            # The figure never travels without the numbers that produced it,
            # and they are filed under the role they priced.
            assert record["cost_rates"] == {"extractor": CARD.as_record()}
            assert record["usage_by_role"]["extractor"]["cost_usd"] == expected

    def test_the_recorded_cost_is_recomputable_from_the_record_alone(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The point of recording the card: a reader with nothing but this run's
        # JSON reproduces its cost, whatever the provider charges by then. Per
        # role, because that is the granularity the rates are stated at.
        out = tmp_path / "runs"
        _, meta, _ = _finished(config_dir, bundle_minimal_dir, out,
                               rates=RATES, usage_response=_usage())
        block = meta["usage_by_role"]["extractor"]
        card = block["cost_rates"]
        recomputed = (
            block["input_tokens"] / 1e6 * card["input_per_1m"]
            + block["output_tokens"] / 1e6 * card["output_per_1m"]
            + block["cache_read_tokens"] / 1e6 * card["cache_read_per_1m"]
            + block["cache_write_tokens"] / 1e6 * card["cache_write_per_1m"])
        assert block["cost_usd"] == pytest.approx(recomputed, abs=1e-6)
        assert meta["cost_usd"] == pytest.approx(recomputed, abs=1e-6)

    def test_the_card_is_on_disk_not_only_in_memory(
            self, config_dir, bundle_minimal_dir, tmp_path):
        out = tmp_path / "runs"
        orch, _, _ = _finished(config_dir, bundle_minimal_dir, out,
                               rates=RATES, usage_response=_usage())
        on_disk = json.loads(
            (orch.session.session_dir / "diagnostics" / "run.json").read_text())
        assert on_disk["cost_rates"] == {"extractor": CARD.as_record()}
        assert on_disk["cost_usd"] > 0

    def test_only_the_enabled_roles_appear(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The checker and the reviewer are off, so they make no calls, have no
        # spend, and are recorded nowhere. A role listed with zeros would read
        # as one that ran and cost nothing.
        out = tmp_path / "runs"
        _, meta, entry = _finished(config_dir, bundle_minimal_dir, out,
                                   rates=RATES, usage_response=_usage())
        for record in (meta, entry):
            assert set(record["usage_by_role"]) == {"extractor"}
            assert set(record["cost_rates"]) == {"extractor"}


# ---------------------------------------------------------------------------
# An unpriced role states tokens and nothing else
# ---------------------------------------------------------------------------

class TestUnpricedRun:
    def test_no_dollar_figure_anywhere_but_the_tokens_survive(
            self, config_dir, bundle_minimal_dir, tmp_path):
        out = tmp_path / "runs"
        orch, meta, entry = _finished(
            config_dir, bundle_minimal_dir, out, rates={},
            usage_response=_usage())

        for record in (meta, entry):
            # None, and emphatically not 0.0: this run was not free, it was
            # unpriced, and a zero would report the wrong one of those.
            assert record["cost_usd"] is None
            assert record["cost_rates"] == {"extractor": None}
            block = record["usage_by_role"]["extractor"]
            assert block["cost_usd"] is None
            assert block["cost_rates"] is None
            # The counters are the durable record. They come from the provider,
            # so nothing about them can go stale.
            assert record["input_tokens"] == 1000
            assert record["output_tokens"] == 400
            assert record["cache_creation_tokens"] == 200
            assert record["cache_read_tokens"] == 50
            assert block["input_tokens"] == 1000
            assert block["cache_write_tokens"] == 200
        assert orch.recorded_cost() is None

    def test_the_key_is_present_and_null_rather_than_absent(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A consumer must not have to tell "this run states no cost" apart
        # from "this record does not carry the key at all". Every key is always
        # written, so absence is never an available reading.
        out = tmp_path / "runs"
        orch, _, entry = _finished(config_dir, bundle_minimal_dir, out,
                                   rates={}, usage_response=_usage())
        on_disk = json.loads(
            (orch.session.session_dir / "diagnostics" / "run.json").read_text())
        for record in (on_disk, entry):
            assert "cost_usd" in record and record["cost_usd"] is None
            block = record["usage_by_role"]["extractor"]
            assert "cost_usd" in block and block["cost_usd"] is None
            assert "cost_rates" in block and block["cost_rates"] is None


# ---------------------------------------------------------------------------
# One role priced, another not
# ---------------------------------------------------------------------------

class TestMixedPricingAcrossRoles:
    def _mixed(self, config_dir, bundle_dir, out_dir, rates):
        """A checker-enabled run: one extractor turn plus one checker verdict.

        The checker's spend arrives through the fan-out callback rather than
        `_accumulate_usage`, so this exercises both paths into the per-role
        meters in one run.
        """
        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model=EXTRACTOR,
            checker_config=CheckerConfig(checker_model=CHECKER, api_key="x"),
            review_model=None,
            max_checks_per_field=2, final_review=False,
            rates=rates, api_key="x",
        )
        orch.prepare_new_session()
        orch._accumulate_usage(_usage(), EXTRACTOR, "extractor")
        return orch

    def _fold_checker(self, orch, cost):
        acc = orch._role_usage("checker")
        if cost is None:
            orch._cost_unpriced = True
            acc["unpriced"] = True
        else:
            orch._cost_usd += cost
            orch._cost_counted = True
            acc["cost_usd"] += cost
            acc["counted"] = True
        acc["input_tokens"] += 100
        orch._input_tokens += 100
        orch._checkpoint_usage_to_meta()

    def test_operator_and_table_cards_coexist_in_one_run(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The bundle prices the extractor by hand and leaves the checker to the
        # table. Both figures stand, each beside the card and the provenance
        # that produced it.
        table_card = Rates.from_table(price_for(CHECKER), PRICES_VERSION)
        orch = self._mixed(config_dir, bundle_minimal_dir, tmp_path / "runs",
                           {"extractor": CARD, "checker": table_card})
        self._fold_checker(orch, 0.002)
        by_role = orch.session.meta["usage_by_role"]
        assert by_role["extractor"]["cost_rates"]["source"] == "operator"
        assert by_role["checker"]["cost_rates"]["source"] == "table"
        assert by_role["checker"]["cost_rates"]["as_of"] == table_card.as_of
        assert by_role["extractor"]["cost_usd"] > 0
        assert by_role["checker"]["cost_usd"] == pytest.approx(0.002)
        assert orch.recorded_cost() == pytest.approx(
            by_role["extractor"]["cost_usd"] + 0.002)

    def test_one_unpriced_role_withholds_the_total_and_keeps_the_others(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The extractor has no card, so it states no figure and the run states
        # no total. The checker's own figure is unaffected: it was priced, and
        # withholding it would hide a fact the record actually holds.
        orch = self._mixed(config_dir, bundle_minimal_dir, tmp_path / "runs",
                           {"checker": CARD})
        self._fold_checker(orch, 0.002)
        meta = orch.session.meta
        assert meta["cost_usd"] is None
        assert meta["usage_by_role"]["extractor"]["cost_usd"] is None
        assert meta["usage_by_role"]["checker"]["cost_usd"] == \
            pytest.approx(0.002)
        assert meta["cost_rates"] == {"extractor": None,
                                      "checker": CARD.as_record()}


# ---------------------------------------------------------------------------
# A gateway-reported charge needs no rate card
# ---------------------------------------------------------------------------

class TestReportedChargeStandsAlone:
    def test_a_routed_run_records_its_charge_with_no_card(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The charge is a fact the gateway states, not a figure anybody
        # computed, so it is recorded whether or not rates exist and it carries
        # no card behind it. That is the one cost in these artefacts with no
        # rates beside it, and it is the one that needs none.
        from meltiro.run_entry import build_entry
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     rates={}, extractor=ROUTED_EXTRACTOR)
        orch.prepare_new_session()
        orch._accumulate_usage(_routed_response(0.0123), ROUTED_EXTRACTOR,
                               "extractor")

        entry = build_entry(orch.session, cost_usd=orch.recorded_cost(),
                            cost_rates=orch._cost_rates_record(),
                            usage_by_role=orch._usage_by_role_record())
        for record in (orch.session.meta, entry):
            assert record["cost_usd"] == pytest.approx(0.0123)
            assert record["cost_rates"] == {"extractor": None}
            block = record["usage_by_role"]["extractor"]
            assert block["cost_usd"] == pytest.approx(0.0123)
            assert block["cost_rates"] is None

    def test_a_missing_charge_is_still_a_loud_fault(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Dropping the run's total to None would be a quieter failure than the
        # one this refusal exists to prevent, so an unpriced run does not soften
        # it: a routed call with no reported cost still raises.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     rates={}, extractor=ROUTED_EXTRACTOR)
        orch.prepare_new_session()
        with pytest.raises(RuntimeError, match="no reported cost"):
            orch._accumulate_usage(_routed_response(None), ROUTED_EXTRACTOR,
                                   "extractor")


# ---------------------------------------------------------------------------
# Mixed and resumed runs never state a partial total
# ---------------------------------------------------------------------------

class TestAPartialTotalIsNeverStated:
    def test_one_uncosted_call_withholds_the_whole_total(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A routed call's charge is real and a direct call with no card cannot
        # be costed. Summing only the first would produce a number that reads as
        # what the run cost while covering a fraction of it.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     rates={})
        orch.prepare_new_session()
        orch._accumulate_usage(_routed_response(0.0123), ROUTED_EXTRACTOR,
                               "extractor")
        assert orch.recorded_cost() == pytest.approx(0.0123)
        orch._accumulate_usage(_usage(), EXTRACTOR, "extractor")
        assert orch.recorded_cost() is None

    def test_adding_rates_at_resume_cannot_price_the_earlier_segment(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Rates typed after the fact do not reach calls already made, so a run
        # that spent part of itself unpriced keeps stating no total. The card
        # recorded is this segment's, which is what checks the figures this
        # segment does produce.
        out = tmp_path / "runs"
        first = _orch(config_dir, bundle_minimal_dir, out, rates={})
        first.prepare_new_session()
        session_dir = first.session.session_dir
        first._accumulate_usage(_usage(), EXTRACTOR, "extractor")
        first._pause("tool_cap_hit")
        assert first.session.meta["cost_usd"] is None

        second = _orch(config_dir, bundle_minimal_dir, out, rates=RATES)
        second.resume_session(session_dir)
        second._accumulate_usage(_usage(), EXTRACTOR, "extractor")
        assert second.recorded_cost() is None
        second._checkpoint_usage_to_meta()
        meta = second.session.meta
        assert meta["cost_usd"] is None
        assert meta["cost_rates"] == {"extractor": CARD.as_record()}
        # The role that ran unpriced before the pause keeps stating no figure.
        assert meta["usage_by_role"]["extractor"]["cost_usd"] is None

    def test_usage_by_role_round_trips_through_checkpoint_and_reseed(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The per-role meters are reseeded from meta on resume, so the finished
        # record covers the whole run per role and not only the last segment.
        out = tmp_path / "runs"
        first = _orch(config_dir, bundle_minimal_dir, out, rates=RATES)
        first.prepare_new_session()
        session_dir = first.session.session_dir
        first._accumulate_usage(_usage(inp=1000, out=400, cache_create=200,
                                       cache_read=50), EXTRACTOR, "extractor")
        first._pause("tool_cap_hit")
        seg1 = first.session.meta["usage_by_role"]["extractor"]["cost_usd"]

        second = _orch(config_dir, bundle_minimal_dir, out, rates=RATES)
        second.resume_session(session_dir)
        assert second._usage_by_role["extractor"]["input_tokens"] == 1000
        assert second._usage_by_role["extractor"]["cache_write_tokens"] == 200
        second._accumulate_usage(_usage(inp=500, out=100, cache_create=0,
                                        cache_read=0), EXTRACTOR, "extractor")
        second._checkpoint_usage_to_meta()
        block = second.session.meta["usage_by_role"]["extractor"]
        assert block["input_tokens"] == 1500
        assert block["output_tokens"] == 500
        assert block["cost_usd"] > seg1
        # And the per-role sum is still the run-wide one, so neither view can
        # drift from the other.
        assert block["cost_usd"] == second.session.meta["cost_usd"]


# ---------------------------------------------------------------------------
# The derived views say "not priced" rather than "$0"
# ---------------------------------------------------------------------------

class TestDerivedViews:
    def test_the_cli_summary_says_not_priced(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        from meltiro import cli
        out = tmp_path / "runs"
        orch, _, _ = _finished(config_dir, bundle_minimal_dir, out,
                               rates={}, usage_response=_usage())
        cli._print_run_summary(orch, "complete")
        printed = capsys.readouterr().out
        assert "not priced" in printed
        assert "$0.0000" not in printed

    def test_the_cli_summary_prints_a_priced_total(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        from meltiro import cli
        out = tmp_path / "runs"
        orch, meta, _ = _finished(config_dir, bundle_minimal_dir, out,
                                  rates=RATES, usage_response=_usage())
        cli._print_run_summary(orch, "complete")
        assert f"${meta['cost_usd']:.4f}" in capsys.readouterr().out

    def test_the_transcript_states_the_card_each_role_was_priced_at(
            self, config_dir, bundle_minimal_dir, tmp_path):
        out = tmp_path / "runs"
        orch, _, _ = _finished(config_dir, bundle_minimal_dir, out,
                               rates=RATES, usage_response=_usage())
        text = (orch.session.session_dir / "diagnostics"
                / "transcript.md").read_text()
        assert "USD per million tokens" in text
        # The role, its model, the input rate, and where the card came from,
        # all on one row.
        assert "| extractor | `claude-opus-4-8` | 15.0 |" in text
        assert "| operator |" in text

    def test_the_transcript_names_the_table_reading_behind_a_table_card(
            self, config_dir, bundle_minimal_dir, tmp_path):
        card = Rates.from_table(price_for(EXTRACTOR), PRICES_VERSION)
        out = tmp_path / "runs"
        orch, _, _ = _finished(config_dir, bundle_minimal_dir, out,
                               rates={"extractor": card},
                               usage_response=_usage())
        text = (orch.session.session_dir / "diagnostics"
                / "transcript.md").read_text()
        assert f"| table | {card.as_of} |" in text

    def test_the_transcript_says_a_routed_role_is_priced_per_call(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A routed role is fully priced with no card, so the document must not
        # file it beside the roles nothing priced.
        out = tmp_path / "runs"
        orch = _orch(config_dir, bundle_minimal_dir, out, rates={},
                     extractor=ROUTED_EXTRACTOR)
        orch.prepare_new_session()
        orch._accumulate_usage(_routed_response(0.0123), ROUTED_EXTRACTOR,
                               "extractor")
        orch._finalise("complete")
        text = (orch.session.session_dir / "diagnostics"
                / "transcript.md").read_text()
        assert (f"| extractor | `{ROUTED_EXTRACTOR}` | *(per call)* | "
                f"*(per call)* | *(per call)* | *(per call)* | "
                f"gateway charge |") in text
        assert "runs a routed model" in text
        # Its own rows state a figure, so nothing about it reads as unpriced.
        assert "| extractor | $0.012300 |" in text

    def test_the_transcript_tells_a_role_that_never_ran_from_an_unpriced_one(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A run that pauses before the reviewer's turn leaves that role with no
        # calls and no figure. Rendering it as unpriced would blame the pricing
        # for a silence the pause caused.
        out = tmp_path / "runs"
        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_minimal_dir),
            out, extractor_model=EXTRACTOR,
            checker_config=CheckerConfig(checker_model=CHECKER, api_key="x"),
            review_model=ROUTED_EXTRACTOR,
            max_checks_per_field=0, final_review=True,
            rates={"extractor": CARD, "review": None}, api_key="x")
        orch.prepare_new_session()
        orch._accumulate_usage(_usage(), EXTRACTOR, "extractor")
        orch._pause("tool_cap_hit")
        text = (orch.session.session_dir / "diagnostics"
                / "transcript.md").read_text()
        assert "| review | *(no calls)* | 0 | 0 | 0 | 0 |" in text
        assert "*(no calls)* | *(no calls)*" in text

    def test_the_transcript_says_not_priced_and_shows_no_zero_dollars(
            self, config_dir, bundle_minimal_dir, tmp_path):
        out = tmp_path / "runs"
        orch, _, _ = _finished(config_dir, bundle_minimal_dir, out,
                               rates={}, usage_response=_usage())
        text = (orch.session.session_dir / "diagnostics"
                / "transcript.md").read_text()
        spend = text.split("### Spend", 1)[1].split("###", 1)[0]
        assert "| Cost | *(not priced)* |" in spend
        assert "$" not in spend
        assert "neither an operator card nor a price-table entry" in spend


# ---------------------------------------------------------------------------
# The per-field history aggregate follows the same rule
# ---------------------------------------------------------------------------

def _verdict_event(path, cost):
    """One dispatch event carrying a single checker verdict at `cost`."""
    return {
        "ts": "T", "event": "tool_call_applied", "turn_id": 1,
        "tool": "update_study",
        "result": {
            "status": "ok",
            "applied_changes": {},
            "failed_fields": {},
            "_field_diffs": {path: {"before": None, "after": "v"}},
            "_checker_verdicts": {
                path: {"verdict": "ok", "rationale": "r", "cost_usd": cost},
            },
        },
    }


class TestCheckerCostAggregate:
    def test_priced_verdicts_sum(self):
        history = build_field_history(
            [_verdict_event("study.a", 0.002), _verdict_event("study.b", 0.003)])
        assert history["aggregate"]["checker_cost_usd"] == 0.005

    def test_one_unpriced_verdict_withholds_the_aggregate(self):
        # Summing the priced ones alone would answer "what did the checker
        # cost?" with a number covering only part of it.
        history = build_field_history(
            [_verdict_event("study.a", 0.002), _verdict_event("study.b", None)])
        assert history["aggregate"]["checker_cost_usd"] is None

    def test_a_run_with_no_checker_still_reports_zero(self):
        # Nothing was checked, so nothing was spent checking. That zero is a
        # real measurement rather than a withheld one.
        history = build_field_history([])
        assert history["aggregate"]["checker_cost_usd"] == 0.0
