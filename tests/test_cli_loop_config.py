"""Tests for how the CLI reads pipeline.yaml into the run's config objects.

Every key an operator writes must either take effect or be refused out loud.
A key read under a TRUTHY presence check does neither for a falsy value: it is
discarded and the built-in default silently stands, so pipeline.yaml says one
thing and the run does another while the provenance record describes only what
ran. These tests pin the presence contract (`is not None`) and the domain
guard for the loop settings whose valid domain excludes their falsy value.

No network, no API key: every orchestrator here is built with dry_run=True
and no client is ever constructed.
"""

from types import SimpleNamespace

import pytest

from meltiro import cli
from meltiro.checker import DEFAULT_CONCURRENCY
from meltiro.orchestrator import (
    DEFAULT_EXTRACTOR_MAX_TOKENS,
    DEFAULT_MAX_CHECKS_PER_FIELD,
    DEFAULT_MAX_REVIEW_TOOL_CALLS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_REVIEW_MAX_TOKENS,
)


def _args(**over):
    base = dict(
        max_tool_calls=None, max_checks_per_field=None, final_review=None,
        extractor_model=None, review_model=None, checker_model=None,
        diagnostics="standard", dry_run=True)
    base.update(over)
    return SimpleNamespace(**base)


def _pipeline(config_dir):
    from meltiro.config_bundle import load_config_bundle
    return dict(load_config_bundle(config_dir).pipeline)


def _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg, **args_over):
    from meltiro.bundle import load_bundle
    from meltiro.config_bundle import load_config_bundle
    config = load_config_bundle(config_dir)
    bundle = load_bundle(str(bundle_minimal_dir))
    return cli._build_orchestrator(
        config, bundle, tmp_path / "runs", loop_cfg, _args(**args_over))


class TestAbsentKeysFallBackToTheDeclaredDefaults:
    """A key a bundle omits falls back to the orchestrator's own DEFAULT_*
    constant, not to a literal the CLI keeps separately.

    Two copies of one default drift, and this one drifts invisibly: the run
    record writes the RESOLVED cap into `caps`, so a CLI disagreeing with the
    library would put a number in the artefact that the documented default
    does not explain, on exactly the runs whose author never wrote the key.
    """

    @pytest.mark.parametrize("key,attr,expected", [
        ("max_tool_calls", "max_tool_calls", DEFAULT_MAX_TOOL_CALLS),
        ("max_checks_per_field", "max_checks_per_field",
         DEFAULT_MAX_CHECKS_PER_FIELD),
        ("max_review_tool_calls", "max_review_tool_calls",
         DEFAULT_MAX_REVIEW_TOOL_CALLS),
        ("extractor_max_tokens", "extractor_max_tokens",
         DEFAULT_EXTRACTOR_MAX_TOKENS),
        ("review_max_tokens", "review_max_tokens", DEFAULT_REVIEW_MAX_TOKENS),
    ])
    def test_an_absent_key_takes_the_orchestrator_constant(
            self, config_dir, bundle_minimal_dir, tmp_path, key, attr,
            expected):
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop(key, None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert getattr(orch, attr) == expected

    def test_the_resolved_default_cap_reaches_the_run_record(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # `caps` in run.json is what a reader of a finished session sees, so
        # the default has to arrive there as the documented number.
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("max_tool_calls", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        orch.prepare_new_session()
        assert orch.session.meta["caps"]["max_tool_calls"] == \
            DEFAULT_MAX_TOOL_CALLS


class TestCheckerConcurrencyWiring:
    """`checker_concurrency` is how many checker calls run in parallel: it
    becomes ThreadPoolExecutor's `max_workers`, whose domain is the positive
    integers. It is read under an `is not None` presence contract, so a
    configured 0 reaches validation and is refused at startup instead of being
    swallowed as absent and replaced by the default."""

    def test_checker_concurrency_from_pipeline_is_wired(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_concurrency"] = 3
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.concurrency == 3

    def test_default_checker_concurrency_when_key_absent(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("checker_concurrency", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.concurrency == DEFAULT_CONCURRENCY

    def test_zero_checker_concurrency_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A thread pool of zero workers cannot exist (ThreadPoolExecutor
        # raises), so 0 is not a way to disable the checker; it is a config
        # error, and it fails here before any spend rather than at the first
        # checker batch.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_concurrency"] = 0
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "checker_concurrency" in capsys.readouterr().err

    def test_negative_checker_concurrency_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_concurrency"] = -4
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "checker_concurrency" in capsys.readouterr().err

    def test_zero_checker_concurrency_from_environment_rejected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys,
            monkeypatch):
        # The EFFECTIVE value is guarded, not just the bundle's: with the key
        # absent the value comes from CheckerConfig.from_env's
        # CHECKER_CONCURRENCY fallback, and an unusable value there must fail
        # at startup on the same terms.
        monkeypatch.setenv("CHECKER_CONCURRENCY", "0")
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("checker_concurrency", None)
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "checker_concurrency" in capsys.readouterr().err

    def test_checker_concurrency_moves_no_fingerprint(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Operational, not methodology: parallelism changes how fast the same
        # calls are made and nothing about what any model is asked, so two runs
        # differing only in it are the same instrument.
        base_cfg = _pipeline(config_dir)
        base_cfg["checker_concurrency"] = 1
        base = _orch(config_dir, bundle_minimal_dir, tmp_path,
                     base_cfg)._build_fingerprints()
        wide_cfg = dict(base_cfg)
        wide_cfg["checker_concurrency"] = 8
        wide = _orch(config_dir, bundle_minimal_dir, tmp_path,
                     wide_cfg)._build_fingerprints()
        assert wide == base


class TestOutputCapsAreUsableBudgets:
    """Every role's output cap is optional and defaults when absent, so a
    number that IS written is deliberate. Zero and negatives are not budgets:
    they reach the provider as `max_tokens=0` and fail at that role's first
    call, which for the reviewer is after a whole extraction has been billed.
    All three roles are guarded on the same terms."""

    @pytest.mark.parametrize("key", ["extractor_max_tokens",
                                     "review_max_tokens",
                                     "checker_max_tokens"])
    @pytest.mark.parametrize("bad", [0, -1])
    def test_unusable_output_cap_rejected(
            self, key, bad, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg[key] = bad
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert key in capsys.readouterr().err

    @pytest.mark.parametrize("key,attr", [
        ("extractor_max_tokens", "extractor_max_tokens"),
        ("review_max_tokens", "review_max_tokens"),
    ])
    def test_usable_output_cap_is_wired(
            self, key, attr, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg[key] = 4096
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert getattr(orch, attr) == 4096


# A complete rate card, in the shape an operator writes under one role of
# `rates:`.
_CARD = {"input_per_1m": 3.0, "output_per_1m": 15.0,
         "cache_read_per_1m": 0.3, "cache_write_per_1m": 3.75}
# The same card as it is recorded, with the provenance an operator card carries:
# written by hand, so no reading date and no table behind it.
_CARD_RECORD = dict(_CARD, source="operator", as_of=None, table_version=None)


class TestRatesWiring:
    """`rates:` gives a USD-per-million-token card to a NAMED ROLE, because
    each role runs its own model and a card is a statement about one model's
    prices. It is optional per role: a role the block leaves out takes its
    rates from direktoro's dated price table, and one whose model the table
    does not price runs recording tokens and no dollar figure at all.

    A card that IS written must be complete and usable, and every fault in it
    is refused here, at startup, before any spend — the alternative is an
    operator believing a role was priced when it was not, or priced against
    something other than what they wrote. Each rate is read for PRESENCE rather
    than truth, so a legitimate `0.0` (a provider with no prompt-cache tier)
    is honoured instead of being swallowed as absent."""

    def test_absent_rates_take_every_role_from_the_price_table(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The fixture names no `rates:`, and all three of its models are direct
        # models the table prices, so all three roles are priced without the
        # operator writing a number.
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("rates", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert set(orch.rates) == {"extractor", "checker", "review"}
        for card in orch.rates.values():
            assert card.source == "table"
            # The provenance that makes the figure traceable: the day the
            # vendor's page was read, and which table data it was read from.
            assert card.as_of and card.table_version is not None
        assert orch.checker_config.rates is orch.rates["checker"]

    def test_a_role_the_table_cannot_price_runs_unpriced(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # A direct model with no table entry and no card. The role records its
        # tokens and states no figure; what must NOT appear is a 0.0, which
        # would read as a role that cost nothing.
        from direktoro import prices
        monkeypatch.setattr(prices, "price_for", lambda model: None)
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("rates", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.rates == {"extractor": None, "checker": None,
                              "review": None}
        assert orch.checker_config.rates is None
        assert orch.recorded_cost() is None

    def test_an_operator_card_wins_over_the_table_for_that_role_alone(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Naming one role does not silence the other two: the extractor prices
        # at what the operator wrote, and the checker and reviewer still price
        # at what the table publishes for their own models.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {"extractor": dict(_CARD)}
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.rates["extractor"].as_record() == _CARD_RECORD
        assert orch.rates["checker"].source == "table"
        assert orch.rates["review"].source == "table"

    def test_each_role_is_wired_to_its_own_card(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Three cards, three roles, no crossing over: the checker prices at the
        # checker's numbers and at nothing else's.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {
            "extractor": dict(_CARD),
            "checker": dict(_CARD, input_per_1m=1.0),
            "review": dict(_CARD, input_per_1m=9.0),
        }
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.rates["extractor"].input_per_1m == 3.0
        assert orch.rates["checker"].input_per_1m == 1.0
        assert orch.rates["review"].input_per_1m == 9.0
        assert orch.checker_config.rates is orch.rates["checker"]

    def test_a_disabled_role_is_not_priced(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A stage that is off makes no calls, so it has no model to price and
        # appears in no pricing record.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["max_checks_per_field"] = 0
        loop_cfg["final_review"] = False
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert set(orch.rates) == {"extractor"}
        assert orch.checker_config.rates is None

    def test_zero_rates_are_honoured_not_swallowed(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A provider with no prompt-cache tier charges zero for cache traffic,
        # and 0.0 is that fact rather than a missing value. Read under a truthy
        # check it would vanish, and the run would then refuse to cost a call
        # with cache tokens in it.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {"extractor": dict(_CARD, cache_read_per_1m=0.0,
                                               cache_write_per_1m=0.0)}
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        card = orch.rates["extractor"]
        assert card.cache_read_per_1m == 0.0
        assert card.cache_write_per_1m == 0.0
        # And the zero is applied, not treated as "no rate for this counter"
        # (which direktoro refuses for a non-zero counter).
        assert card.cost_of_call(cache_read_tokens=1_000_000) == 0.0

    @pytest.mark.parametrize("key", sorted(_CARD))
    def test_an_incomplete_card_is_refused(
            self, key, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # All four together, or none. A partial card prices a role correctly or
        # not depending on which counters the traffic happened to use, so the
        # completeness of the recorded figure would be an accident.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {
            "checker": {k: v for k, v in _CARD.items() if k != key}}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert key in err
        # Named with its role, so a bundle pricing three roles says which one.
        assert "checker" in err

    def test_an_unknown_rate_key_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The same trap the top-level pipeline allowlist closes: a misspelt key
        # inside a card would be silently ignored, and the run would then
        # refuse to cost a call using that counter — mid-run, after spending.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {
            "extractor": dict(_CARD, cache_write_per_million=3.75)}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "cache_write_per_million" in capsys.readouterr().err

    def test_an_unknown_role_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A card filed under a role that does not exist prices nothing, and
        # would leave the role it was meant for silently on the table default.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {"reviewer": dict(_CARD)}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "reviewer" in err
        assert "extractor, checker, review" in err

    def test_a_card_written_at_the_top_level_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # Four rates directly under `rates:` name no role, so they price no
        # model. The refusal states the shape the block takes.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = dict(_CARD)
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "extractor, checker, review" in err
        assert "input_per_1m" in err

    @pytest.mark.parametrize("bad", [-1.0, "3.0", None, True, [3.0]])
    def test_an_unusable_rate_value_is_refused(
            self, bad, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A negative rate reduces a total, a string coerces to something the
        # operator did not write, and a bool is an integer wearing a disguise.
        # None means the key is not really there at all.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {"review": dict(_CARD, input_per_1m=bad)}
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "input_per_1m" in capsys.readouterr().err

    def test_a_bare_rates_key_is_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # `rates:` with nothing under it parses to None. Treating that as
        # "take the table defaults" would silently ignore a block the operator
        # clearly meant to write, which is the failure the whole known-key
        # allowlist exists to prevent.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = None
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "rates" in capsys.readouterr().err

    def test_rates_is_on_the_pipeline_key_allowlist(self):
        # The allowlist is what makes a bundle's keys meaningful: an absent
        # entry here turns a correct `rates:` block into a load error.
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        assert "rates" in KNOWN_PIPELINE_KEYS


class TestStartupSaysHowEachRoleIsPriced:
    """Every enabled role gets one line at startup naming what will price it.

    The operator is about to spend money under an arrangement they may not have
    written: a table default, a gateway's per-call charge, or nothing at all.
    Each of those is a different thing to know before the first call, and only
    the first is visible in `pipeline.yaml`."""

    def test_a_table_priced_role_names_the_date_and_its_age(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("rates", None)
        _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        out = capsys.readouterr().out
        assert "Pricing, extractor (claude-opus-4-8): direktoro price table" \
            in out
        assert "days ago" in out

    def test_an_operator_priced_role_says_where_the_card_came_from(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["rates"] = {"checker": dict(_CARD)}
        _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert ("Pricing, checker (claude-sonnet-4-6): the card written for "
                "this role under `rates:` in pipeline.yaml."
                ) in capsys.readouterr().out

    def test_a_routed_role_is_priced_per_call_not_called_unpriced(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # A routed model's cost is the charge the gateway reports, so the role
        # is fully priced with no card and is not looked up in the table.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["review_model"] = "z-ai/glm-5v-turbo"
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        out = capsys.readouterr().out
        assert ("Pricing, review (z-ai/glm-5v-turbo): routed, so every call "
                "is priced at the charge the gateway reports for it.") in out
        assert "unpriced" not in out
        assert orch.rates["review"] is None

    def test_an_unpriced_role_says_the_run_states_no_total(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys,
            monkeypatch):
        from direktoro import prices
        monkeypatch.setattr(prices, "price_for", lambda model: None)
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("rates", None)
        _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        out = capsys.readouterr().out
        assert "Pricing, extractor (claude-opus-4-8): unpriced." in out
        assert "the run states no total" in out


class TestToolCallCapsAreUsableBudgets:
    """A cap of zero or less is not a smaller budget. The role is bound before
    its first call, so the run stops having recorded nothing, and a resume
    reaches the same bound. Both roles' caps are held to the same rule."""

    @pytest.mark.parametrize("key", ["max_tool_calls", "max_review_tool_calls"])
    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_cap_below_one_is_refused(
            self, key, bad, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg[key] = bad
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert key in capsys.readouterr().err

    def test_the_cli_flag_is_held_to_the_same_rule(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # --max-tool-calls overrides pipeline.yaml, so the guard has to sit on
        # the resolved value rather than on the bundle's.
        loop_cfg = _pipeline(config_dir)
        with pytest.raises(SystemExit):
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg,
                  max_tool_calls=0)
        assert "max_tool_calls" in capsys.readouterr().err
