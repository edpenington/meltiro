"""Tests for the checker client.

The Anthropic call itself is mocked; what is under test is the wrapper
logic: JSON parsing, verdict validation, cost accounting, the parallel
fan-out, and the error-wrapping in run_checker_batch.
"""

from unittest.mock import MagicMock, patch

import pytest

from meltiro.checker import (
    CheckerConfig,
    check_one_field,
    run_checker_batch,
    _strip_code_fences,
)
from meltiro.errors import CheckerError
from meltiro.prompt_partials import stage_predicates
from meltiro.rates import Rates
from direktoro import NormalisedResponse, NormalisedUsage
from direktoro.registry import OPENROUTER_BASE_URL

# The run's structure predicates, which a checker fingerprint takes as an
# argument because it keeps none of its own. A checker-on, reviewer-on
# pipeline is the ordinary shape, and these tests vary the prompts and the
# template rather than the structure, so they hold it fixed.
PREDICATES = stage_predicates(2, True)


def _stream_returning(content_text, *, input_tokens=100,
                      output_tokens=20, cache_creation=0, cache_read=0):
    """Build a mock that mimics anthropic's streaming context manager."""
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.text_stream = iter([])
    stream_cm.get_final_message = MagicMock(return_value=response)
    return stream_cm


def _client_returning(content_text, **kw):
    client = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_stream_returning(content_text, **kw))
    return client


def _stream_returning_content(content, *, input_tokens=100, output_tokens=20):
    """Like _stream_returning but with a caller-supplied `content` list, so a
    test can hand back an empty list or a first block that has no `.text`."""
    response = MagicMock()
    response.content = content
    response.usage = MagicMock(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.text_stream = iter([])
    stream_cm.get_final_message = MagicMock(return_value=response)
    return stream_cm


def _client_returning_content(content):
    client = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_stream_returning_content(content))
    return client


# The rate card these checker calls are priced at, in USD per million tokens.
# It is the operator's number in a real run and a fixture's here; what matters
# to these tests is that a card is present, so the costing path runs at all.
RATES = Rates(input_per_1m=3.0, output_per_1m=15.0,
              cache_read_per_1m=0.3, cache_write_per_1m=3.75)


def _config(rates=RATES):
    return CheckerConfig(
        api_key="sk-test", checker_model="claude-sonnet-4-6",
        max_tokens=1024, temperature=0.0, concurrency=4, rates=rates,
    )


class TestStripCodeFences:
    def test_no_fence(self):
        assert _strip_code_fences('{"a":1}') == '{"a":1}'

    def test_json_fence(self):
        assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_plain_fence(self):
        assert _strip_code_fences('```\n{"a":1}\n```') == '{"a":1}'


class TestCheckOneField:
    def test_ok_verdict(self):
        client = _client_returning(
            '{"verdict": "ok", "rationale": "matches the quote"}'
        )
        result = check_one_field(
            system_message_blocks=[{"type": "text", "text": "sys"}],
            user_message_blocks=[{"type": "text", "text": "user"}],
            config=_config(), client=client,
        )
        assert result["verdict"] == "ok"
        assert result["rationale"] == "matches the quote"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20
        assert result["cost_usd"] > 0

    def test_challenge_verdict(self):
        client = _client_returning(
            '{"verdict": "challenge", "rationale": "wrong"}'
        )
        result = check_one_field(
            system_message_blocks=[],
            user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["verdict"] == "challenge"

    def test_call_sends_the_configured_checker_temperature(self):
        # The WIRE side of the checker's decoding contract. checker_fp folds in
        # the checker's resolved decoding params, and that promise ("a
        # fingerprint folds in exactly what is sent") only holds while the call
        # site reads the same config value the fingerprint does. A non-default
        # value, so a hardcoded 0.0 anywhere on the path cannot pass.
        client = _client_returning('{"verdict": "ok", "rationale": "r"}')
        config = _config()
        config.temperature = 0.35
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[],
            config=config, client=client,
        )
        assert client.messages.stream.call_args.kwargs["temperature"] == 0.35

    def test_call_omits_temperature_for_a_no_temperature_checker_model(self):
        # The quirk applies to the checker's live call too: a checker model that
        # rejects temperature is sent none, whatever the config asked for.
        client = _client_returning('{"verdict": "ok", "rationale": "r"}')
        config = _config()
        config.checker_model = "claude-opus-4-8"
        config.temperature = 0.35
        check_one_field(
            system_message_blocks=[],
            user_message_blocks=[],
            config=config, client=client,
        )
        assert "temperature" not in client.messages.stream.call_args.kwargs

    def test_invalid_verdict_raises(self):
        client = _client_returning(
            '{"verdict": "maybe", "rationale": "unsure"}'
        )
        with pytest.raises(CheckerError, match="Invalid verdict"):
            check_one_field(
                system_message_blocks=[],
                user_message_blocks=[],
                config=_config(), client=client,
            )

    def test_non_json_raises(self):
        client = _client_returning("not json at all")
        with pytest.raises(CheckerError, match="non-JSON"):
            check_one_field(
                system_message_blocks=[],
                user_message_blocks=[],
                config=_config(), client=client,
            )

    def test_empty_content_raises_checker_error(self):
        # An empty content list must become a CheckerError, not an
        # unguarded IndexError that would escape the per-field handling.
        client = _client_returning_content([])
        with pytest.raises(CheckerError, match="no content blocks"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client,
            )

    def test_non_text_first_block_raises_checker_error(self):
        # A first block with no `.text` (for example a stray tool_use) must
        # become a CheckerError, not an unguarded AttributeError.
        import types
        block = types.SimpleNamespace(type="tool_use")
        client = _client_returning_content([block])
        with pytest.raises(CheckerError, match="no text"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=_config(), client=client,
            )

    def test_strips_code_fences(self):
        client = _client_returning(
            '```json\n{"verdict": "ok", "rationale": "..."}\n```'
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        assert result["verdict"] == "ok"

    def test_cache_read_discount_applied(self):
        client = _client_returning(
            '{"verdict": "ok", "rationale": "..."}',
            input_tokens=100, output_tokens=20,
            cache_creation=0, cache_read=5000,
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client,
        )
        # Cache reads carry their own rate on the card, well below the
        # full-price input rate, so the same prompt size costs less when most
        # of it is served from cache. The counters are kept apart all the way
        # through for exactly this reason.
        cheap = result["cost_usd"]
        # Compare to a non-cached call with the same total prompt size.
        client2 = _client_returning(
            '{"verdict": "ok", "rationale": "..."}',
            input_tokens=5100, output_tokens=20,
        )
        result2 = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(), client=client2,
        )
        assert cheap < result2["cost_usd"]

    def test_no_rate_card_records_tokens_and_no_cost(self):
        # A direct checker model with no rates states NO cost. Not 0.0, which
        # would read as a free call and would silently pull a run's total down;
        # the token counters are the record, and they came from the provider.
        client = _client_returning(
            '{"verdict": "ok", "rationale": "..."}',
            input_tokens=100, output_tokens=20,
        )
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=_config(rates=None), client=client,
        )
        assert result["cost_usd"] is None
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20


class _Text:
    """A minimal Anthropic-shaped text block for a stubbed NormalisedResponse."""
    type = "text"

    def __init__(self, text):
        self.text = text


class _StubAdapter:
    """A provider adapter returning a canned NormalisedResponse, so a routed
    checker call runs fully offline with no client/SDK."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def create_message(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class TestCheckOneFieldRoutedCost:
    """A ROUTED (gateway-served) checker model is priced FROM the response
    (OpenRouter usage.cost -> reported_cost). That figure is a charge the
    gateway states rather than one anybody predicts, so it needs no rate card
    and these configs carry none. A missing reported cost is a loud fault,
    NEVER a silent $0."""

    ROUTED = "z-ai/glm-5v-turbo"

    def _config(self):
        return CheckerConfig(api_key="x", checker_model=self.ROUTED,
                             max_tokens=1024, temperature=0.0, concurrency=4)

    def _response(self, reported_cost, *, generation_id="gen-chk-1",
                  served="Z.AI"):
        return NormalisedResponse(
            content=[_Text('{"verdict": "ok", "rationale": "r"}')],
            usage=NormalisedUsage(input_tokens=5, output_tokens=1),
            resolved_model=self.ROUTED, provider="openrouter",
            base_url=OPENROUTER_BASE_URL, raw_request={}, raw_response={},
            decoding_params={"max_tokens": 1024},
            generation_id=generation_id, served_provider=served,
            reported_cost=reported_cost)

    def test_reported_cost_is_used_with_no_rate_card(self):
        # No rates on this config, so the only figure available is the
        # response-reported one; a non-None cost here proves that branch is
        # taken. The routing receipts must ride in _provenance for the
        # orchestrator to ledger.
        result = check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=self._config(), adapter=_StubAdapter(
                self._response(reported_cost=0.0123)))
        assert result["cost_usd"] == pytest.approx(0.0123)
        assert result["_provenance"]["generation_id"] == "gen-chk-1"
        assert result["_provenance"]["served_provider"] == "Z.AI"

    def test_missing_reported_cost_raises_loudly(self):
        with pytest.raises(RuntimeError, match="no reported cost"):
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=self._config(), adapter=_StubAdapter(
                    self._response(reported_cost=None)))


class TestRunCheckerBatch:
    def test_parallel_fan_out(self):
        client = _client_returning(
            '{"verdict": "ok", "rationale": "..."}'
        )
        calls = [
            {"field_path": f"study.field{i}",
             "system_message_blocks": [],
             "user_message_blocks": []}
            for i in range(5)
        ]
        results = run_checker_batch(
            calls=calls, config=_config(), client=client,
        )
        assert set(results.keys()) == {
            f"study.field{i}" for i in range(5)
        }
        for r in results.values():
            assert r["verdict"] == "ok"

    def test_one_error_doesnt_abort_the_batch(self):
        # First call returns invalid JSON, second returns ok.
        client = MagicMock()
        client.messages.stream = MagicMock(side_effect=[
            _stream_returning("not json"),
            _stream_returning('{"verdict": "ok", "rationale": "fine"}'),
        ])
        results = run_checker_batch(
            calls=[
                {"field_path": "study.bad", "system_message_blocks": [],
                 "user_message_blocks": []},
                {"field_path": "study.good", "system_message_blocks": [],
                 "user_message_blocks": []},
            ],
            config=CheckerConfig(
                api_key="x", checker_model="claude-sonnet-4-6",
                concurrency=1),
            client=client,
        )
        assert results["study.bad"]["verdict"] == "challenge"  # wrapped error
        assert "error" in results["study.bad"]
        assert results["study.good"]["verdict"] == "ok"

    def test_on_complete_called(self):
        client = _client_returning(
            '{"verdict": "ok", "rationale": "..."}'
        )
        seen = []
        run_checker_batch(
            calls=[{"field_path": "x", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=_config(), client=client,
            on_complete=lambda fp, r: seen.append((fp, r["verdict"])),
        )
        assert seen == [("x", "ok")]

    def test_empty_content_wrapped_as_field_error(self):
        # An empty-content response for one field must not abort the batch;
        # it is wrapped as a challenged field, like any other CheckerError.
        client = _client_returning_content([])
        results = run_checker_batch(
            calls=[{"field_path": "study.x", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=_config(), client=client,
        )
        assert results["study.x"]["verdict"] == "challenge"
        assert "error" in results["study.x"]

    def test_results_ordered_by_field_path(self):
        # Results are returned sorted by field path, not by nondeterministic
        # completion order, so the tool result the model reads and the session
        # event that records it are reproducible.
        client = _client_returning('{"verdict": "ok", "rationale": "..."}')
        unsorted_paths = ["study.zeta", "record.relationship_2.gauge",
                          "study.alpha", "record.relationship_1.gauge",
                          "study.mid"]
        calls = [
            {"field_path": p, "system_message_blocks": [],
             "user_message_blocks": []}
            for p in unsorted_paths
        ]
        results = run_checker_batch(
            calls=calls, config=_config(), client=client,
        )
        assert list(results.keys()) == sorted(unsorted_paths)


class TestCheckerConfig:
    def test_fingerprint_stable(self, synthetic_template, tmp_path,
                                 monkeypatch):
        sys_path = tmp_path / "sys.md"
        sys_path.write_text("you are a checker", encoding="utf-8")
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="sk-x",
        )
        a = cfg.fingerprint(synthetic_template, predicates=PREDICATES)
        b = cfg.fingerprint(synthetic_template, predicates=PREDICATES)
        assert a == b
        assert a.startswith("checker_fp:")

    def test_fingerprint_changes_with_prompt(
            self, synthetic_template, tmp_path):
        sys_path = tmp_path / "sys.md"
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")

        sys_path.write_text("v1", encoding="utf-8")
        cfg1 = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp1 = cfg1.fingerprint(synthetic_template, predicates=PREDICATES)

        sys_path.write_text("v2", encoding="utf-8")
        cfg2 = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp2 = cfg2.fingerprint(synthetic_template, predicates=PREDICATES)

        assert fp1 != fp2

    def test_fingerprint_hashes_substituted_reference_list(
            self, synthetic_template, tmp_path):
        # The checker LLM sees the reference-substituted system prompt, so
        # editing a reference list that appears in the checker system prompt
        # must move checker_fp even though the raw prompt file is unchanged.
        sys_path = tmp_path / "sys.md"
        sys_path.write_text(
            "You are a checker. Canonical tools:\n{reference:gauge_list}",
            encoding="utf-8")
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp_a = cfg.fingerprint(
            synthetic_template, {"gauge_list": [{"tool_name": "WDS-9"}]},
            predicates=PREDICATES)
        fp_b = cfg.fingerprint(
            synthetic_template, {"gauge_list": [{"tool_name": "SRI-7"}]},
            predicates=PREDICATES)
        assert fp_a != fp_b

    def test_fingerprint_changes_with_checker_context_fields(
            self, synthetic_template, tmp_path):
        # The template's checker context fields drive every per-record checker
        # call's context label (build_record_context), so reversing them must
        # move checker_fp.
        sys_path = tmp_path / "sys.md"
        sys_path.write_text("you are a checker", encoding="utf-8")
        user_path = tmp_path / "user.md"
        user_path.write_text("template", encoding="utf-8")
        cfg = CheckerConfig(
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(sys_path),
            user_prompt_template_path=str(user_path),
            api_key="x",
        )
        fp_before = cfg.fingerprint(synthetic_template,
                                    predicates=PREDICATES)
        synthetic_template["checker_context_fields"] = list(
            reversed(synthetic_template["checker_context_fields"]))
        fp_after = cfg.fingerprint(synthetic_template,
                                   predicates=PREDICATES)
        assert fp_before != fp_after

    def test_from_env_ignores_checker_decoding_env_vars(self, monkeypatch):
        # The checker's decoding knobs come from the config bundle and NOT the
        # environment: CHECKER_TEMPERATURE and CHECKER_MAX_TOKENS are not read
        # at all, so from_env returns the dataclass defaults whatever the shell
        # holds. Reading them would let checker_fp differ between two machines
        # running the same config bundle.
        monkeypatch.setenv("CHECKER_TEMPERATURE", "0.9")
        monkeypatch.setenv("CHECKER_MAX_TOKENS", "77")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024

    def test_from_env_refuses_a_zero_concurrency_override(self, monkeypatch):
        # 0 workers is not a thread pool and not a way to disable the checker.
        # Under a truthy presence test it would read as "no override given"
        # and quietly hand back the environment's value or the built-in
        # default, running the checker at a parallelism nobody asked for.
        monkeypatch.setenv("CHECKER_CONCURRENCY", "9")
        with pytest.raises(ValueError, match="positive integer"):
            CheckerConfig.from_env(model_override="claude-sonnet-4-6",
                                   concurrency_override=0)

    def test_from_env_refuses_a_negative_concurrency_override(self):
        with pytest.raises(ValueError, match="positive integer"):
            CheckerConfig.from_env(model_override="claude-sonnet-4-6",
                                   concurrency_override=-4)

    def test_an_override_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("CHECKER_CONCURRENCY", "9")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6",
                                     concurrency_override=3)
        assert cfg.concurrency == 3

    def test_the_environment_still_supplies_the_value_with_no_override(
            self, monkeypatch):
        # The env fallback is untouched by the guard above: its EFFECTIVE value
        # is checked one step later by the CLI, with the operator-facing
        # message (see tests/test_cli_loop_config.py).
        monkeypatch.setenv("CHECKER_CONCURRENCY", "0")
        cfg = CheckerConfig.from_env(model_override="claude-sonnet-4-6")
        assert cfg.concurrency == 0
