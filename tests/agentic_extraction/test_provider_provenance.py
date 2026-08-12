"""Provenance for the provider abstraction, via the centralised call-identity
block.

Covers:
  - the provider-call identity block (model + provider + base_url + Route +
    wire-keyed resolved decoding params, owned by direktoro) folds into the
    three stage fingerprints, so the same model on two providers (or two
    distinct routed models) get distinct fingerprints;
  - at the orchestrator level, naming a GPT extractor model moves config_fp
    (and only config_fp) versus a Claude extractor, offline via
    prepare_new_session (no API call);
  - run.json records, per role, the API-resolved model string and the raw
    decoding params actually sent (temperature omitted for models that reject
    it).
"""

import pytest

import meltiro.orchestrator as orch_mod
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig, _build_checker_adapter
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import (
    checker_config_fingerprint,
    config_fingerprint,
    review_config_fingerprint,
)
from meltiro.orchestrator import Orchestrator
from direktoro import call_identity_fields, canonical_json
from direktoro.providers import (
    AnthropicAdapter, NormalisedResponse, NormalisedUsage, OpenAIAdapter,
    resolved_decoding_params)
from direktoro.registry import OPENAI_BASE_URL, OPENROUTER_BASE_URL, model_info
from meltiro.tools import get_tool_definitions


def _identity(model, decoding_params=None):
    """The canonical provider-call identity block for `model`, exactly as
    the orchestrator/checker fold it into each stage fingerprint: model id +
    provider + base_url + Route (for a routed model) + wire-keyed resolved
    decoding params, serialised byte-stably."""
    return canonical_json(call_identity_fields(
        model, route=model_info(model).route, decoding_params=decoding_params))


# ---------------------------------------------------------------------------
# The call-identity block folds into every stage fingerprint
# ---------------------------------------------------------------------------

class TestCallIdentityComponent:
    """The provider-call identity block folds into each stage fingerprint, so
    model / provider / base_url / route / decoding all enter through one opaque
    component. direktoro owns the block; meltiro ONLY composes it, and holds no
    provider= or base_url= input of its own to drift out of step with it."""

    def test_config_fp_moves_with_provider(self):
        # claude (anthropic) vs gpt (openai): different provider in the block.
        a = config_fingerprint(_identity("claude-opus-4-8"), "p", "t")
        b = config_fingerprint(_identity("gpt-5.6-sol"), "p", "t")
        assert a != b

    def test_config_fp_moves_with_base_url_and_route(self):
        # gpt (direct, api.openai.com) vs a routed GLM (the OpenRouter gateway,
        # carrying a Route): the block's base_url and route both differ.
        a = config_fingerprint(_identity("gpt-5.6-sol"), "p", "t")
        b = config_fingerprint(_identity("z-ai/glm-5v-turbo"), "p", "t")
        assert a != b

    def test_config_fp_stable_for_same_identity(self):
        a = config_fingerprint(_identity("gpt-5.6-sol"), "p", "t")
        b = config_fingerprint(_identity("gpt-5.6-sol"), "p", "t")
        assert a == b

    def test_checker_fp_moves_with_provider(self):
        a = checker_config_fingerprint(_identity("claude-opus-4-8"), "s", "u")
        b = checker_config_fingerprint(_identity("gpt-5.6-sol"), "s", "u")
        assert a != b

    def test_review_fp_moves_with_provider(self):
        a = review_config_fingerprint(_identity("claude-opus-4-8"), "s")
        b = review_config_fingerprint(_identity("z-ai/glm-5v-turbo"), "s")
        assert a != b

    def test_routed_models_get_distinct_fps(self):
        # The routed GLM/Qwen slugs share the gateway, provider, and base_url,
        # but differ in slug (and, for qwen, upstream), so their blocks differ:
        # the routed registry never aliases one routed model onto another.
        glm5v = config_fingerprint(_identity("z-ai/glm-5v-turbo"), "p", "t")
        glm46 = config_fingerprint(_identity("z-ai/glm-4.6v"), "p", "t")
        qwen = config_fingerprint(
            _identity("qwen/qwen3-vl-235b-a22b-instruct"), "p", "t")
        assert len({glm5v, glm46, qwen}) == 3

    def test_resolved_decoding_folds_into_the_block(self):
        # The block carries the RESOLVED decoding params, so two runs of the
        # same model sending different params get distinct config_fps.
        cool = config_fingerprint(
            _identity("z-ai/glm-5v-turbo", resolved_decoding_params(
                "z-ai/glm-5v-turbo", sampling={"temperature": 0.0}, max_tokens=100)),
            "p", "t")
        warm = config_fingerprint(
            _identity("z-ai/glm-5v-turbo", resolved_decoding_params(
                "z-ai/glm-5v-turbo", sampling={"temperature": 0.9}, max_tokens=100)),
            "p", "t")
        assert cool != warm


# ---------------------------------------------------------------------------
# Orchestrator: a cross-provider extractor moves config_fp only (offline)
# ---------------------------------------------------------------------------

def _prepared(config_dir, bundle_dir, out_dir, extractor_model):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor_model,
        checker_config=CheckerConfig(
            checker_model="claude-sonnet-4-6", api_key="x"),
        review_model="claude-opus-4-7",
        api_key="x",
    )
    orch.prepare_new_session()
    return orch


def test_cross_provider_extractor_moves_only_config_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    # Same prompts, template, tools, and decoding; only the extractor's
    # provider differs (Claude vs GPT). config_fp must move; checker_fp and
    # review_fp (both still Claude) must not. No API call is made.
    a = _prepared(config_dir, bundle_minimal_dir, tmp_path / "a",
                  "claude-opus-4-7")
    b = _prepared(config_dir, bundle_minimal_dir, tmp_path / "b",
                  "gpt-5.6-sol")

    assert a.session.meta["config_fp"] != b.session.meta["config_fp"]
    assert a.session.meta["checker_fp"] == b.session.meta["checker_fp"]
    assert a.session.meta["review_fp"] == b.session.meta["review_fp"]


def test_routed_extractors_get_distinct_config_fps(
        config_dir, bundle_minimal_dir, tmp_path):
    # The routed GLM/Qwen extractors (all vision-capable, all served via the
    # OpenRouter gateway but distinct slugs/upstreams) must each carry a
    # distinct config_fp, offline via prepare_new_session. This proves the
    # routed registry entries do not alias onto one another's fingerprint.
    glm5v = _prepared(config_dir, bundle_minimal_dir, tmp_path / "glm5v",
                      "z-ai/glm-5v-turbo")
    glm46 = _prepared(config_dir, bundle_minimal_dir, tmp_path / "glm46",
                      "z-ai/glm-4.6v")
    qwen = _prepared(config_dir, bundle_minimal_dir, tmp_path / "qwen",
                     "qwen/qwen3-vl-235b-a22b-instruct")
    fps = {glm5v.session.meta["config_fp"],
           glm46.session.meta["config_fp"],
           qwen.session.meta["config_fp"]}
    assert len(fps) == 3


# ---------------------------------------------------------------------------
# run.json records the resolved model + raw decoding params per role
# ---------------------------------------------------------------------------

class _FakeAdapter:
    """A provider adapter that returns a canned NormalisedResponse, so
    _final_review runs fully offline.

    `calls` records the kwargs of every request, so a test can assert what a
    role actually SENT rather than only what its fingerprint recorded. The two
    are separate reads of the same config value, and the repo's central promise
    (`providers.resolved_decoding_params`: "a fingerprint folds in exactly what
    is sent") only holds while they agree.
    """

    def __init__(self, response):
        self._response = response
        self.calls = []

    def create_message(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_review_meta_records_resolved_model_and_decoding_params(
        config_dir, bundle_minimal_dir, tmp_path):
    orch = _prepared(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     "claude-opus-4-7")

    # Reviewer returns no tool calls and no text: _final_review records
    # provenance then returns final_review_no_response. The resolved model
    # differs from the configured alias (as a real dated snapshot would), and
    # the decoding params omit temperature (Opus rejects it).
    response = NormalisedResponse(
        content=[],
        usage=NormalisedUsage(),
        resolved_model="claude-opus-4-7-20260601",
        provider="anthropic",
        base_url=None,
        raw_request={"model": "claude-opus-4-7"},
        raw_response={},
        decoding_params={"max_tokens": orch.review_max_tokens},
    )
    orch._adapter_for_role = lambda role: _FakeAdapter(response)

    status = orch._final_review()

    assert status == "final_review_no_response"
    meta = orch.session.meta
    assert meta["review_model"] == "claude-opus-4-7"          # configured
    assert meta["review_model_resolved"] == "claude-opus-4-7-20260601"
    assert meta["decoding_params"]["review"] == {
        "max_tokens": orch.review_max_tokens}
    # No temperature key: the reviewer model rejects it, so it was not sent.
    assert "temperature" not in meta["decoding_params"]["review"]


# ---------------------------------------------------------------------------
# Role wiring composes across three providers (offline, fake adapters)
# ---------------------------------------------------------------------------

def _prepared_full(config_dir, bundle_dir, out_dir, *,
                   extractor_model, checker_model, review_model):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor_model,
        checker_config=CheckerConfig(checker_model=checker_model, api_key="x"),
        review_model=review_model,
        api_key="x",
    )
    orch.prepare_new_session()
    return orch


def test_each_role_selects_its_provider_adapter(
        config_dir, bundle_minimal_dir, tmp_path):
    # Three roles, three providers: GPT extractor (OpenAI), Claude checker
    # (Anthropic), routed GLM reviewer (OpenRouter gateway). Each role resolves
    # the right adapter and endpoint from the registry, offline. The routed
    # reviewer resolves to an OpenAIAdapter pointed at the gateway; its Route
    # is not threaded through meltiro's adapter construction, because no
    # shipped config names a routed model. What is pinned here is the
    # role -> adapter -> endpoint wiring, which holds either way.
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="gpt-5.6-sol",
        checker_model="claude-sonnet-4-6",
        review_model="z-ai/glm-5v-turbo")

    # The three stage fingerprints are all distinct (three providers).
    m = orch.session.meta
    assert len({m["config_fp"], m["checker_fp"], m["review_fp"]}) == 3

    # Stub the underlying client constructors so no network/SDK is needed.
    orch._anthropic_client = lambda: "ANTHROPIC_CLIENT"
    orch._openai_client = lambda info: ("OPENAI_CLIENT", info.base_url)

    ext = orch._adapter_for_role("extractor")
    assert isinstance(ext, OpenAIAdapter)
    assert ext.provider == "openai"
    assert ext.base_url == OPENAI_BASE_URL

    rev = orch._adapter_for_role("review")
    assert isinstance(rev, OpenAIAdapter)
    assert rev.provider == "openrouter"
    assert rev.base_url == OPENROUTER_BASE_URL

    # The checker resolves its own adapter from its model (Anthropic here).
    checker_adapter = _build_checker_adapter(
        orch.checker_config, client="STUB")
    assert isinstance(checker_adapter, AnthropicAdapter)


def test_extractor_provenance_recorded_for_gpt(
        config_dir, bundle_minimal_dir, tmp_path):
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="gpt-5.6-sol",
        checker_model="claude-sonnet-4-6",
        review_model="claude-opus-4-7")

    response = NormalisedResponse(
        content=[],
        usage=NormalisedUsage(input_tokens=100, output_tokens=10),
        resolved_model="gpt-5.6-sol-2026-07-09",
        provider="openai",
        base_url=OPENAI_BASE_URL,
        raw_request={"model": "gpt-5.6-sol"},
        raw_response={"model": "gpt-5.6-sol-2026-07-09"},
        wire_request={"model": "gpt-5.6-sol", "input": []},
        decoding_params={"max_output_tokens": 32768,
                         "reasoning": {"effort": "medium"}})
    orch._call_extractor(_FakeAdapter(response),
                         get_tool_definitions(orch.template))

    meta = orch.session.meta
    assert meta["extractor_model"] == "gpt-5.6-sol"
    assert meta["extractor_model_resolved"] == "gpt-5.6-sol-2026-07-09"
    assert meta["decoding_params"]["extractor"] == {
        "max_output_tokens": 32768, "reasoning": {"effort": "medium"}}


# ---------------------------------------------------------------------------
# The WIRE side of the per-role temperature contract. The fingerprint side is
# pinned in test_cli.py; these pin what each role actually sends, so the two
# cannot drift apart. Every role reads its own temperature at two independent
# sites (its call and its fingerprint), and only their agreement makes
# `resolved_decoding_params`' promise true.
# ---------------------------------------------------------------------------

def test_extractor_call_sends_the_extractors_temperature(
        config_dir, bundle_minimal_dir, tmp_path):
    # A temperature-accepting extractor (the shipped Opus rejects it, which
    # would make the assertion vacuous), with the reviewer's temperature set to
    # a DIFFERENT value, so this can only pass by reading the extractor's own.
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_minimal_dir)
    orch = Orchestrator(
        config, bundle, tmp_path / "runs",
        extractor_model="claude-sonnet-4-6",
        checker_config=CheckerConfig(
            checker_model="claude-sonnet-4-6", api_key="x"),
        review_model="claude-opus-4-7",
        sampling={"temperature": 0.3}, review_sampling={"temperature": 0.8}, api_key="x")
    orch.prepare_new_session()
    adapter = _FakeAdapter(NormalisedResponse(
        content=[], usage=NormalisedUsage(input_tokens=10, output_tokens=1),
        resolved_model="claude-sonnet-4-6", provider="anthropic",
        raw_request={"model": "claude-sonnet-4-6"},
        raw_response={"model": "claude-sonnet-4-6"},
        decoding_params={"max_tokens": 32768, "temperature": 0.3}))

    orch._call_extractor(adapter, get_tool_definitions(orch.template))

    assert adapter.calls[0]["sampling"] == {"temperature": 0.3}


def test_extractor_call_omits_temperature_for_no_temperature_model(
        config_dir, bundle_minimal_dir, tmp_path):
    # The adapter applies the quirk, so the orchestrator passes the configured
    # value down regardless; this pins that the value reaching the adapter is
    # the extractor's, and that the resolver then drops it for Opus.
    orch = _prepared(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     "claude-opus-4-7")
    assert resolved_decoding_params(
        "claude-opus-4-7", sampling=orch.sampling,
        max_tokens=orch.extractor_max_tokens) == {"max_tokens": 32768}


# ---------------------------------------------------------------------------
# Fingerprint folds in the RESOLVED decoding params: a fingerprint hashes what
# is actually sent. Both directions, at the orchestrator level.
# ---------------------------------------------------------------------------

def _orch_for_fp(config_dir, bundle_dir, out_dir, *, extractor_model,
                 checker_model="claude-sonnet-4-6",
                 review_model="claude-opus-4-7", sampling={"temperature": 0.0}):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor_model,
        checker_config=CheckerConfig(checker_model=checker_model, api_key="x",
                                     sampling=sampling),
        review_model=review_model,
        sampling=sampling,
        api_key="x")
    orch.prepare_new_session()
    return orch


def test_temperature_change_does_not_move_no_temperature_config_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    # No spurious split: the extractor (Opus) rejects temperature, so two runs
    # differing only in the config temperature send identical decoding params
    # and must share config_fp.
    a = _orch_for_fp(config_dir, bundle_minimal_dir, tmp_path / "a",
                     extractor_model="claude-opus-4-7", sampling={"temperature": 0.0})
    b = _orch_for_fp(config_dir, bundle_minimal_dir, tmp_path / "b",
                     extractor_model="claude-opus-4-7", sampling={"temperature": 0.9})
    assert a.session.meta["config_fp"] == b.session.meta["config_fp"]


def test_temperature_change_moves_accepting_extractor_config_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    # Sonnet accepts temperature, so it is actually sent: changing it must move
    # config_fp (the sent params differ).
    a = _orch_for_fp(config_dir, bundle_minimal_dir, tmp_path / "a",
                     extractor_model="claude-sonnet-4-6", sampling={"temperature": 0.0})
    b = _orch_for_fp(config_dir, bundle_minimal_dir, tmp_path / "b",
                     extractor_model="claude-sonnet-4-6", sampling={"temperature": 0.9})
    assert a.session.meta["config_fp"] != b.session.meta["config_fp"]


def test_registry_quirk_edit_moves_all_three_stage_fingerprints(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # All three stages on Sonnet at temperature 0.0 (sent). Declaring the
    # control refused drops it from every stage's sent params, so editing that
    # one registry declaration moves config_fp, checker_fp, and review_fp
    # together. A registry edit therefore cannot change the wire without moving
    # a fingerprint.
    before = _orch_for_fp(config_dir, bundle_minimal_dir, tmp_path / "before",
                          extractor_model="claude-sonnet-4-6",
                          checker_model="claude-sonnet-4-6",
                          review_model="claude-sonnet-4-6", sampling={"temperature": 0.0})
    b_config = before.session.meta["config_fp"]
    b_checker = before.session.meta["checker_fp"]
    b_review = before.session.meta["review_fp"]

    import dataclasses
    from direktoro.registry import MODEL_REGISTRY
    monkeypatch.setitem(
        MODEL_REGISTRY, "claude-sonnet-4-6",
        dataclasses.replace(model_info("claude-sonnet-4-6"),
                            rejects_sampling=frozenset({"temperature"})))

    after = _orch_for_fp(config_dir, bundle_minimal_dir, tmp_path / "after",
                         extractor_model="claude-sonnet-4-6",
                         checker_model="claude-sonnet-4-6",
                         review_model="claude-sonnet-4-6", sampling={"temperature": 0.0})
    assert after.session.meta["config_fp"] != b_config
    assert after.session.meta["checker_fp"] != b_checker
    assert after.session.meta["review_fp"] != b_review


def test_checker_provenance_recorded_for_routed_glm(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="claude-opus-4-7",
        checker_model="z-ai/glm-5v-turbo",
        review_model="claude-opus-4-7")

    # One checker call, and a fan-out that reports one clean verdict carrying
    # the routed-GLM provenance the orchestrator folds into run.json.
    _calls = [{"field_path": "study.x", "user_message_blocks": []}]

    def _fake_batch(*, calls, config, on_complete=None, api_logger=None,
                    **kw):
        result = {
            "verdict": "ok", "rationale": "", "notes": None,
            "input_tokens": 5, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.0,
            "_provenance": {
                "provider": "openrouter", "base_url": OPENROUTER_BASE_URL,
                "resolved_model": "z-ai/glm-5v-turbo",
                "decoding_params": {"max_tokens": 1024,
                                    "temperature": 0.0}},
        }
        if on_complete is not None:
            on_complete("study.x", result)
        return {"study.x": {"verdict": "ok", "rationale": "", "notes": None}}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _fake_batch)

    orch._run_checker_fanout(_calls)

    meta = orch.session.meta
    assert meta["checker_model"] == "z-ai/glm-5v-turbo"
    assert meta["checker_model_resolved"] == "z-ai/glm-5v-turbo"
    assert meta["decoding_params"]["checker"] == {
        "max_tokens": 1024, "temperature": 0.0}


# ---------------------------------------------------------------------------
# Reported-cost path + transport / generation-id provenance
# ---------------------------------------------------------------------------

def _routed_extractor_response(reported_cost, *, generation_id="gen-abc",
                               served="Z.AI"):
    """A routed-GLM extractor response carrying the OpenRouter routing receipts
    (generation id, served upstream) and a response-reported cost."""
    return NormalisedResponse(
        content=[], usage=NormalisedUsage(input_tokens=100, output_tokens=10),
        resolved_model="z-ai/glm-5v-turbo", provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        raw_request={"model": "z-ai/glm-5v-turbo"},
        raw_response={"model": "z-ai/glm-5v-turbo"},
        wire_request={"model": "z-ai/glm-5v-turbo", "messages": []},
        decoding_params={"max_tokens": 32768},
        generation_id=generation_id, served_provider=served,
        reported_cost=reported_cost)


def test_routed_extractor_costs_from_reported_and_records_receipts(
        config_dir, bundle_minimal_dir, tmp_path):
    # A routed extractor: cost comes FROM the response. This run configures no
    # rate card at all, so the only way a figure can appear is the reported-cost
    # branch, which proves it is the one taken. The transport + generation id
    # land in meta too, so what a run spent and where it was served are both
    # readable from run.json alone.
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="z-ai/glm-5v-turbo",
        checker_model="claude-sonnet-4-6",
        review_model="claude-opus-4-7")

    orch._call_extractor(
        _FakeAdapter(_routed_extractor_response(reported_cost=0.0123)),
        get_tool_definitions(orch.template))

    meta = orch.session.meta
    assert orch._cost_usd == pytest.approx(0.0123)     # from the response
    assert meta["transport"] == "openrouter"
    assert meta["generation_ids"] == ["gen-abc"]
    assert meta["served_providers"] == ["Z.AI"]


def test_routed_extractor_missing_reported_cost_raises_loudly(
        config_dir, bundle_minimal_dir, tmp_path):
    # A routed response with no reported cost is a plumbing fault; recording it
    # as $0 is forbidden, so accumulation raises loudly.
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="z-ai/glm-5v-turbo",
        checker_model="claude-sonnet-4-6",
        review_model="claude-opus-4-7")

    with pytest.raises(RuntimeError, match="no reported cost"):
        orch._call_extractor(
            _FakeAdapter(_routed_extractor_response(reported_cost=None)),
            get_tool_definitions(orch.template))


def test_direct_only_run_records_direct_transport_and_empty_generation_ids(
        config_dir, bundle_minimal_dir, tmp_path):
    # A direct (Anthropic/OpenAI) extractor: transport "direct", generation_ids
    # present-but-empty (so a consumer never has to distinguish absent from
    # none), and no served_providers key.
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="gpt-5.6-sol",
        checker_model="claude-sonnet-4-6",
        review_model="claude-opus-4-7")

    response = NormalisedResponse(
        content=[], usage=NormalisedUsage(input_tokens=100, output_tokens=10),
        resolved_model="gpt-5.6-sol", provider="openai",
        base_url=OPENAI_BASE_URL, raw_request={"model": "gpt-5.6-sol"},
        raw_response={"model": "gpt-5.6-sol"},
        wire_request={"model": "gpt-5.6-sol", "input": []},
        decoding_params={"max_output_tokens": 32768})
    orch._call_extractor(_FakeAdapter(response),
                         get_tool_definitions(orch.template))

    meta = orch.session.meta
    assert meta["transport"] == "direct"
    assert meta["generation_ids"] == []
    assert "served_providers" not in meta


def test_mixed_transport_direct_extractor_routed_checker(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # A direct extractor with a routed checker: the run's transport collapses to
    # "mixed", and the routed checker call's generation id is captured.
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="claude-opus-4-7",
        checker_model="z-ai/glm-5v-turbo",
        review_model="claude-opus-4-7")

    # Direct extractor call first -> transport "direct".
    orch._call_extractor(
        _FakeAdapter(NormalisedResponse(
            content=[], usage=NormalisedUsage(input_tokens=50, output_tokens=5),
            resolved_model="claude-opus-4-7", provider="anthropic",
            base_url=None, raw_request={"model": "claude-opus-4-7"},
            raw_response={}, decoding_params={"max_tokens": 32768})),
        get_tool_definitions(orch.template))
    assert orch.session.meta["transport"] == "direct"

    # A routed checker batch carrying a generation id -> transport "mixed".
    _calls = [{"field_path": "study.x", "user_message_blocks": []}]

    def _fake_batch(*, calls, config, on_complete=None, api_logger=None,
                    **kw):
        result = {
            "verdict": "ok", "rationale": "", "notes": None,
            "input_tokens": 5, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.0042,
            "_provenance": {
                "provider": "openrouter", "base_url": OPENROUTER_BASE_URL,
                "resolved_model": "z-ai/glm-5v-turbo",
                "decoding_params": {"max_tokens": 1024, "temperature": 0.0},
                "generation_id": "gen-chk-9", "served_provider": "Z.AI"},
        }
        if on_complete is not None:
            on_complete("study.x", result)
        return {"study.x": {"verdict": "ok", "rationale": "", "notes": None}}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _fake_batch)
    orch._run_checker_fanout(_calls)

    meta = orch.session.meta
    assert meta["transport"] == "mixed"
    assert meta["generation_ids"] == ["gen-chk-9"]
    assert meta["served_providers"] == ["Z.AI"]


def test_checker_generation_ids_are_field_ordered_not_completion_ordered(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    # The checker fan-out's on_complete callback fires in thread-COMPLETION order,
    # but the generation-id receipts must land in meta in field_path order (the
    # same reproducible order run_checker_batch returns verdicts in), so two
    # identical re-runs serialise them identically and comparing their run.json
    # shows no difference that thread scheduling invented. Here field
    # "study.zzz" COMPLETES BEFORE "study.aaa"; the stored order must still be
    # [aaa, zzz].
    orch = _prepared_full(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        extractor_model="claude-opus-4-7",
        checker_model="z-ai/glm-5v-turbo",
        review_model="claude-opus-4-7")

    _calls = [{"field_path": "study.aaa", "user_message_blocks": []},
              {"field_path": "study.zzz", "user_message_blocks": []}]

    def _result(gid):
        return {
            "verdict": "ok", "rationale": "", "notes": None,
            "input_tokens": 5, "output_tokens": 1,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.001,
            "_provenance": {
                "provider": "openrouter", "base_url": OPENROUTER_BASE_URL,
                "resolved_model": "z-ai/glm-5v-turbo",
                "decoding_params": {"max_tokens": 1024, "temperature": 0.0},
                "generation_id": gid, "served_provider": "Z.AI"},
        }

    def _fake_batch(*, calls, config, on_complete=None, api_logger=None,
                    **kw):
        # Completion order is REVERSED vs field order: zzz first, then aaa.
        if on_complete is not None:
            on_complete("study.zzz", _result("gen-zzz"))
            on_complete("study.aaa", _result("gen-aaa"))
        return {"study.aaa": {"verdict": "ok"}, "study.zzz": {"verdict": "ok"}}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _fake_batch)
    orch._run_checker_fanout(_calls)

    # Field order (aaa before zzz), NOT completion order (zzz before aaa).
    assert orch.session.meta["generation_ids"] == ["gen-aaa", "gen-zzz"]
