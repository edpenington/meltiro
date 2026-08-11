"""Per-role thinking / reasoning effort, and the output-cap guard.

Three things are pinned here, in the order they matter:

  1. **Additivity.** A bundle that names no thinking key gets no thinking
     parameter on the wire and EVERY fingerprint unmoved. That is what makes
     the six thinking keys additive: an existing `pipeline.yaml` that has
     never heard of them keeps its identity and its results.

  2. **Effort reaches call identity.** `direktoro` routes thinking through
     `resolved_decoding_params`, the single source of truth for both the wire
     request and the decoding-params block inside `call_identity_fields`.
     meltiro folds that block into every stage
     fingerprint, so effort becomes call identity with no new fingerprint
     component: two runs differing ONLY in a role's reasoning effort get a
     different `config_fp` / `checker_fp` / `review_fp`, a different per-role
     `call_fp`, and a different `run_fp` — while `instrument_fp` (model-free)
     and `engine_fp` (version + commit) hold. Both halves are asserted: a
     fingerprint axis that moved when it should not would be as wrong as one
     that failed to move.

  3. **The cap hazard.** Claude Opus 5 and Sonnet 5 think when the `thinking`
     parameter is OMITTED, and `max_tokens` caps thinking AND response text
     together. A checker at `checker_max_tokens: 1024` pointed at Sonnet 5 can
     therefore spend its whole budget reasoning and be cut off before the JSON
     verdict is closed. That configuration is refused at construction, before a
     provider client exists; and a truncation that happens above the floor
     anyway is named as truncation rather than surfacing as a parse error.

Everything here is offline: no API key, no client, no network, no spend. The
capability facts (which model thinks by default, which effort levels exist,
what the minimum thinking allocation is) are read from direktoro's REAL
registry rather than hand-copied, so the two repos cannot drift apart silently.
"""

import pytest

from direktoro import (
    EFFORT_LEVELS, THINKING_MODES, Thinking, ThinkingSupport,
    resolved_decoding_params, thinking_support)

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import KNOWN_PIPELINE_KEYS, load_config_bundle
from meltiro.errors import CheckerError, ThinkingConfigError
from meltiro.orchestrator import Orchestrator
from meltiro import thinking as thinking_mod


# The settled working set: Opus 5 extractor and reviewer, Sonnet 5 checker.
# Both think when the parameter is omitted, which is the whole reason this
# module exists. Named here rather than inline so a repoint is one edit.
OPUS_5 = "claude-opus-5"
SONNET_5 = "claude-sonnet-5"
# The generation before it: adaptive thinking exists but is OFF unless asked.
OPUS_4_8 = "claude-opus-4-8"
SONNET_4_6 = "claude-sonnet-4-6"


def _orch(config, bundle, out_dir, **over):
    """A prepared dry-run orchestrator, with pipeline.yaml overridable.

    Mirrors the helper in test_fingerprint_axes.py, plus the three thinking
    specs. `prepare_new_session` is what computes and records the fingerprints,
    so the returned orchestrator's `session.meta` carries them.
    """
    loop = dict(config.pipeline)
    thinking = over.pop("thinking", None)
    review_thinking = over.pop("review_thinking", None)
    checker_thinking = over.pop("checker_thinking", None)
    checker_max_tokens = over.pop("checker_max_tokens", None)
    loop.update(over)
    checker_config = CheckerConfig.from_env(
        model_override=loop["checker_model"])
    checker_config.temperature = float(loop.get("checker_temperature", 0.0))
    checker_config.thinking = checker_thinking
    if checker_max_tokens is not None:
        checker_config.max_tokens = checker_max_tokens
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model=loop["extractor_model"],
        checker_config=checker_config,
        review_model=loop["review_model"],
        max_tool_calls=int(loop["max_tool_calls"]),
        max_checks_per_field=int(loop["max_checks_per_field"]),
        final_review=bool(loop.get("final_review", True)),
        temperature=float(loop["temperature"]),
        thinking=thinking,
        review_thinking=review_thinking,
        extractor_max_tokens=int(loop["extractor_max_tokens"]),
        review_max_tokens=int(loop["review_max_tokens"]),
        dry_run=True,
    )
    orch.prepare_new_session()
    return orch


@pytest.fixture
def meta(tmp_path, config_dir, bundle_minimal_dir):
    def _build(name="runs", **over):
        return _orch(load_config_bundle(config_dir),
                     load_bundle(bundle_minimal_dir),
                     tmp_path / name, **over).session.meta
    return _build


_UNSET = object()


@pytest.fixture
def build(tmp_path, config_dir, bundle_minimal_dir):
    """Construct an Orchestrator only (no session), for the startup guards."""
    counter = {"n": 0}

    def _build(**over):
        counter["n"] += 1
        config = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        loop = dict(config.pipeline)
        thinking = over.pop("thinking", None)
        review_thinking = over.pop("review_thinking", None)
        checker_thinking = over.pop("checker_thinking", None)
        checker_max_tokens = over.pop("checker_max_tokens", None)
        # Sentinel rather than None, because None is a MEANINGFUL value here:
        # it is how a role says "send no temperature at all", which is the one
        # way a temperature-accepting model can be asked to think.
        checker_temperature = over.pop("checker_temperature", _UNSET)
        loop.update(over)
        checker_config = CheckerConfig.from_env(
            model_override=loop["checker_model"])
        checker_config.thinking = checker_thinking
        if checker_max_tokens is not None:
            checker_config.max_tokens = checker_max_tokens
        if checker_temperature is not _UNSET:
            checker_config.temperature = checker_temperature
        return Orchestrator(
            config, bundle, tmp_path / f"runs{counter['n']}",
            extractor_model=loop["extractor_model"],
            checker_config=checker_config,
            review_model=loop["review_model"],
            max_tool_calls=int(loop["max_tool_calls"]),
            max_checks_per_field=int(loop["max_checks_per_field"]),
            final_review=bool(loop.get("final_review", True)),
            temperature=float(loop["temperature"]),
            thinking=thinking,
            review_thinking=review_thinking,
            extractor_max_tokens=int(loop["extractor_max_tokens"]),
            review_max_tokens=int(loop["review_max_tokens"]),
            dry_run=True,
        )
    return _build


# ---------------------------------------------------------------------------
# 1. Additivity
# ---------------------------------------------------------------------------

class TestSayingNothingChangesNothing:
    def test_no_spec_resolves_identically_to_the_pre_seam_call(self):
        # The seam's whole claim to being additive: passing `thinking=None` and
        # not passing it at all produce the same dict, so every consumer that
        # names no thinking key keeps its wire request and its fingerprints.
        for model in (OPUS_4_8, SONNET_4_6, OPUS_5, SONNET_5):
            before = resolved_decoding_params(
                model, temperature=0.0, max_tokens=1024)
            after = resolved_decoding_params(
                model, temperature=0.0, max_tokens=1024, thinking=None)
            assert before == after, model
            assert "thinking" not in after
            assert "output_config" not in after

    def test_shipped_example_sends_no_thinking_parameter(self, tmp_path,
                                                         config_dir,
                                                         bundle_minimal_dir):
        # The shipped worked example names no thinking key, so none of its three
        # roles puts a thinking parameter on the wire. This is the assertion
        # that would fail if a default ever leaked into the engine.
        orch = _orch(load_config_bundle(config_dir),
                     load_bundle(bundle_minimal_dir), tmp_path / "runs")
        meta = orch._decoding_params_meta()
        # All three roles are reported, so the loop below cannot pass by the
        # report having gone empty.
        assert set(meta) == {"extractor", "checker", "review"}
        for role, dec in meta.items():
            assert "thinking" not in dec, role
            assert "output_config" not in dec, role

    def test_the_six_keys_are_on_the_pipeline_allowlist(self):
        # A pipeline.yaml key the allowlist does not know is rejected at load,
        # so the keys and the allowlist have to be one fact. They are: the
        # allowlist splices in `meltiro.thinking.PIPELINE_KEYS`.
        assert thinking_mod.PIPELINE_KEYS <= KNOWN_PIPELINE_KEYS
        assert thinking_mod.PIPELINE_KEYS == {
            "extractor_thinking_mode", "extractor_thinking_effort",
            "checker_thinking_mode", "checker_thinking_effort",
            "review_thinking_mode", "review_thinking_effort",
        }


# ---------------------------------------------------------------------------
# 2. Effort reaches call identity, and only where it should
# ---------------------------------------------------------------------------

class TestEffortReachesTheWire:
    def test_effort_becomes_an_anthropic_wire_key(self):
        dec = resolved_decoding_params(
            OPUS_5, temperature=0.0, max_tokens=32768,
            thinking=Thinking(effort="max"))
        assert dec["output_config"] == {"effort": "max"}

    def test_adaptive_mode_becomes_an_anthropic_wire_key(self):
        dec = resolved_decoding_params(
            OPUS_5, temperature=0.0, max_tokens=32768,
            thinking=Thinking(mode="adaptive"))
        assert dec["thinking"] == {"type": "adaptive"}


class TestEffortReachesTheFingerprints:
    """Two runs differing ONLY in reasoning effort must fingerprint differently.

    The claim direktoro makes on meltiro's behalf, checked against meltiro's
    real fingerprint recipe rather than assumed from the wire dict.
    """

    def test_extractor_effort_moves_config_fp_call_fp_and_run_fp(self, meta):
        a = meta("a", extractor_model=OPUS_5, review_model=OPUS_5)
        b = meta("b", extractor_model=OPUS_5, review_model=OPUS_5,
                 thinking=Thinking(effort="max"))
        assert a["config_fp"] != b["config_fp"]
        assert a["extractor_call_fp"] != b["extractor_call_fp"]
        assert a["run_fp"] != b["run_fp"]

    def test_extractor_effort_moves_no_other_role_and_no_other_axis(
            self, meta):
        a = meta("a", extractor_model=OPUS_5, review_model=OPUS_5)
        b = meta("b", extractor_model=OPUS_5, review_model=OPUS_5,
                 thinking=Thinking(effort="max"))
        # The other two roles are untouched, which is what makes a per-role
        # spec worth having.
        assert a["checker_fp"] == b["checker_fp"]
        assert a["checker_call_fp"] == b["checker_call_fp"]
        assert a["review_fp"] == b["review_fp"]
        assert a["review_call_fp"] == b["review_call_fp"]
        # The instrument is model-free by construction, and thinking is a
        # decoding parameter, so it must NOT move: "same instrument, different
        # reasoning effort" has to stay a single-axis diff.
        assert a["instrument_fp"] == b["instrument_fp"]
        # And the engine is the meltiro version and commit, which reasoning
        # effort has nothing to do with. (The task brief expected effort to
        # land in engine_fp; it does not, and must not.)
        assert a["engine_fp"] == b["engine_fp"]

    def test_two_different_efforts_are_two_different_fingerprints(self, meta):
        # Not merely "spec vs no spec": the LEVEL is what is being compared, so
        # a grid sweeping low/high/max gets three distinct run identities.
        low = meta("low", extractor_model=OPUS_5, review_model=OPUS_5,
                   thinking=Thinking(effort="low"))
        high = meta("high", extractor_model=OPUS_5, review_model=OPUS_5,
                    thinking=Thinking(effort="high"))
        mx = meta("max", extractor_model=OPUS_5, review_model=OPUS_5,
                  thinking=Thinking(effort="max"))
        fps = {low["run_fp"], high["run_fp"], mx["run_fp"]}
        assert len(fps) == 3

    def test_checker_effort_moves_only_the_checker(self, meta):
        a = meta("a", checker_model=SONNET_5, checker_max_tokens=4096)
        b = meta("b", checker_model=SONNET_5, checker_max_tokens=4096,
                 checker_thinking=Thinking(effort="low"))
        assert a["checker_fp"] != b["checker_fp"]
        assert a["checker_call_fp"] != b["checker_call_fp"]
        assert a["run_fp"] != b["run_fp"]
        assert a["config_fp"] == b["config_fp"]
        assert a["review_fp"] == b["review_fp"]
        assert a["instrument_fp"] == b["instrument_fp"]

    def test_review_effort_moves_only_the_reviewer(self, meta):
        a = meta("a", review_model=OPUS_5)
        b = meta("b", review_model=OPUS_5,
                 review_thinking=Thinking(effort="xhigh"))
        assert a["review_fp"] != b["review_fp"]
        assert a["review_call_fp"] != b["review_call_fp"]
        assert a["run_fp"] != b["run_fp"]
        assert a["config_fp"] == b["config_fp"]
        assert a["checker_fp"] == b["checker_fp"]
        assert a["instrument_fp"] == b["instrument_fp"]

    def test_a_mode_moves_the_fingerprint_too(self, meta):
        # Turning thinking OFF is as much a change to the question as raising
        # the effort, and it must not collapse onto the spec-free run.
        a = meta("a", extractor_model=OPUS_5, review_model=OPUS_5)
        b = meta("b", extractor_model=OPUS_5, review_model=OPUS_5,
                 thinking=Thinking(mode="disabled"))
        assert a["config_fp"] != b["config_fp"]
        assert a["run_fp"] != b["run_fp"]

    def test_recorded_decoding_params_report_the_effort(self, tmp_path,
                                                        config_dir,
                                                        bundle_minimal_dir):
        # The dry-run decoding-params report is what an operator reads before
        # paying, so it has to show the effort that will be sent.
        orch = _orch(load_config_bundle(config_dir),
                     load_bundle(bundle_minimal_dir), tmp_path / "runs",
                     extractor_model=OPUS_5, review_model=OPUS_5,
                     thinking=Thinking(mode="adaptive", effort="high"))
        dec = orch._decoding_params_meta()
        assert dec["extractor"]["output_config"] == {"effort": "high"}
        assert dec["extractor"]["thinking"] == {"type": "adaptive"}
        # And not on a role that did not ask for it.
        assert "output_config" not in dec["checker"]


# ---------------------------------------------------------------------------
# 3. The cap hazard
# ---------------------------------------------------------------------------

class TestCapGuardRefusesBeforeSpend:
    """A thinking role given a cap that cannot fit a think is refused.

    `max_tokens` caps thinking AND response together, so this is arithmetic,
    not bad luck. Every refusal here happens in `Orchestrator.__init__`, before
    a session directory, a provider client, or a token of spend.
    """

    def test_sonnet_5_checker_at_1024_is_refused(self, build):
        # THE case. A `checker_max_tokens: 1024` sized for a checker that does
        # not think, pointed at Sonnet 5, which thinks when the parameter is
        # omitted. The config names no thinking mode, so the verdict would be
        # truncated with nothing in the run record saying why.
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_5, checker_max_tokens=1024)
        msg = str(exc.value)
        assert "checker" in msg
        assert "checker_max_tokens" in msg
        assert "2048" in msg  # the threshold that fired, reported not prescribed
        # The message must say WHY thinking is on, since nothing in the bundle
        # asked for it.
        assert "default_on" in msg

    def test_the_refusal_names_all_three_ways_out(self, build):
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_5, checker_max_tokens=1024)
        msg = str(exc.value)
        assert "checker_thinking_mode" in msg   # turn thinking off
        assert "checker_model" in msg           # or repoint the role
        assert "spend decision" in msg          # and why it is not done for you

    # -- The advice must not walk the operator into a truncating cap ----------

    def test_the_refusal_never_prescribes_a_cap_that_would_truncate(
            self, build):
        # THE pin. A refusal whose first suggestion is "raise
        # `checker_max_tokens` to at least 2048" hands the operator a cap that
        # clears validation and then truncates on essentially every checker
        # call, burning the run's checker budget to produce nothing: the guard
        # would walk them into the failure it exists to prevent.
        #
        # The floor may be REPORTED, since it is the threshold that fired, but
        # it must never appear as the target of a raise, and the message must
        # say in terms that clearing it is not sufficient. Checked structurally
        # as well as by phrase, so a reworded prescription is caught too.
        import re
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_5, checker_max_tokens=1024)
        msg = str(exc.value)
        floor = thinking_mod.thinking_cap_floor(SONNET_5)
        for prescription in (f"at least {floor}", f"to {floor}",
                             "raise checker_max_tokens"):
            assert prescription not in msg, prescription
        assert not re.search(
            rf"(raise|increase|set|use|at least)[^.]{{0,60}}\b{floor}\b", msg)
        # ...and the distinction is stated outright, not left to be inferred.
        assert "STOPS REFUSING" in msg
        assert "not a cap that works" in msg
        assert "not a guarantee" in msg

    def test_the_refusal_names_the_reasoning_effort_actually_in_force(
            self, build):
        # The floor cannot consult effort without inventing a number (see
        # `test_the_floor_does_not_scale_with_reasoning_effort`), so the MESSAGE
        # consults it instead. With no effort named, the level in force is the
        # model's registry default — not "none", which is what an operator
        # reading the bundle would otherwise assume.
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_5, checker_max_tokens=1024)
        msg = str(exc.value)
        assert repr(thinking_support(SONNET_5).default_effort) in msg
        assert "checker_thinking_effort is unset" in msg

    def test_top_effort_is_given_the_one_figure_the_vendor_publishes(
            self, build):
        # At `xhigh` / `max` Anthropic publishes a starting `max_tokens` to tune
        # from. That is a real, sourced, effort-conditional anchor, and it is
        # the only one — so it is quoted here, labelled as a starting point.
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_5, checker_max_tokens=1024,
                  checker_thinking=Thinking(mode="adaptive", effort="max"))
        msg = str(exc.value)
        assert "'max'" in msg
        assert str(thinking_mod.PUBLISHED_CAP_START_AT_TOP_EFFORT) in msg
        assert "STARTING" in msg
        # Quoted as guidance, never smuggled in as a threshold meltiro enforces.
        assert "not a floor" in msg

    def test_a_lower_effort_is_not_handed_the_top_effort_figure(self, build):
        # The published figure is for `xhigh` / `max` only. Repeating it at
        # `low` as though it generalised would be the same class of error as
        # offering the floor itself as a target: an authoritative-sounding
        # number that does not apply.
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_5, checker_max_tokens=1024,
                  checker_thinking=Thinking(mode="adaptive", effort="low"))
        msg = str(exc.value)
        assert "'low'" in msg
        assert "publishes no max_tokens figure for that level" in msg
        assert "does not generalise down" in msg

    def test_disabling_thinking_makes_the_small_cap_legal_again(self, build):
        # The operator's choice, taken explicitly: a Sonnet 5 checker with
        # thinking off is a checker that fits in 1024 tokens.
        orch = build(checker_model=SONNET_5, checker_max_tokens=1024,
                     checker_thinking=Thinking(mode="disabled"))
        assert orch.checker_config.thinking.mode == "disabled"

    def test_raising_the_cap_makes_the_thinking_checker_legal(self, build):
        orch = build(checker_model=SONNET_5, checker_max_tokens=2048)
        assert orch.checker_config.max_tokens == 2048

    def test_a_non_thinking_checker_at_1024_is_untouched(self, build):
        # Sonnet 4.6 does not think unless asked, so a 1024-token checker
        # pointed at it is sound. The guard must NOT fire here: a false refusal
        # would break every bundle sized for a non-thinking checker.
        orch = build(checker_model=SONNET_4_6, checker_max_tokens=1024)
        assert orch.checker_config.max_tokens == 1024

    def test_asking_a_non_thinking_model_to_think_still_needs_the_cap(
            self, build):
        # A model that does not think by default can be ASKED to, and then the
        # same cap arithmetic applies.
        #
        # Opus 4.8 rather than Sonnet 4.6, though both are
        # non-thinking-by-default. Sonnet 4.6 ACCEPTS a temperature, and a
        # request carrying a temperature AND active thinking is a 400 on that
        # generation, so it is refused for its temperature (see
        # TestTemperatureAndThinkingCannotBothApply) and never reaches the cap
        # question this test exists to ask. Opus 4.8 carries the
        # `no_temperature` quirk, so the checker's temperature is omitted from
        # the request and the cap is the ONLY thing left to fail on.
        with pytest.raises(ThinkingConfigError, match="2048"):
            build(checker_model=OPUS_4_8, checker_max_tokens=1024,
                  checker_thinking=Thinking(mode="adaptive"))

    def test_a_disabled_checker_is_not_cap_checked(self, build):
        # A stage that makes no calls has no cap to size, on the same terms as
        # its model not being required and its fingerprint being null.
        orch = build(checker_model=SONNET_5, checker_max_tokens=1024,
                     max_checks_per_field=0)
        assert orch.checker_enabled is False
        # The SAME cap with the stage switched on is refused, so it is the
        # disabling that silences the guard and not the numbers.
        with pytest.raises(ThinkingConfigError, match="checker_max_tokens"):
            build(checker_model=SONNET_5, checker_max_tokens=1024,
                  max_checks_per_field=1)

    def test_a_disabled_reviewer_is_not_cap_checked(self, build):
        orch = build(review_model=SONNET_5, review_max_tokens=100,
                     final_review=False)
        assert orch.final_review is False
        with pytest.raises(ThinkingConfigError, match="review_max_tokens"):
            build(review_model=SONNET_5, review_max_tokens=100,
                  final_review=True)

    def test_the_extractor_and_reviewer_are_guarded_too(self, build):
        with pytest.raises(ThinkingConfigError, match="extractor_max_tokens"):
            build(extractor_model=OPUS_5, extractor_max_tokens=1500)
        with pytest.raises(ThinkingConfigError, match="review_max_tokens"):
            build(review_model=OPUS_5, review_max_tokens=1500)

    def test_the_working_set_at_shipped_caps_is_accepted(self, build):
        # Opus 5 extractor and reviewer at 32768, Sonnet 5 checker at 4096: the
        # owner's working set, sized for thinking. It must construct cleanly,
        # otherwise the guard is refusing the configuration it exists to enable.
        orch = build(extractor_model=OPUS_5, review_model=OPUS_5,
                     checker_model=SONNET_5, checker_max_tokens=4096)
        assert orch.extractor_model == OPUS_5


class TestShapesTheEndpointWouldRejectAreRefusedEarly:
    """direktoro refuses a 400-producing shape before a client exists; meltiro
    re-raises it with the role attached, because "which role" is the first
    thing an operator needs and the traceback does not say."""

    def test_an_effort_level_the_model_lacks_is_refused(self, build):
        # `xhigh` arrived with Opus 4.7; Sonnet 4.6 does not have it.
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_4_6, checker_max_tokens=8192,
                  checker_thinking=Thinking(effort="xhigh"))
        assert "checker" in str(exc.value)
        assert "xhigh" in str(exc.value)

    def test_disabled_thinking_above_opus_5s_effort_ceiling_is_refused(
            self, build):
        # Opus 5 accepts `{"type": "disabled"}` only at effort `high` or below;
        # pairing it with `max` is a 400. Refused here, unbilled.
        with pytest.raises(ThinkingConfigError) as exc:
            build(extractor_model=OPUS_5,
                  thinking=Thinking(mode="disabled", effort="max"))
        assert "extractor" in str(exc.value)

    def test_a_thinking_spec_on_a_model_with_no_declared_surface_is_refused(
            self, build):
        # Every non-Anthropic entry leaves `thinking` undeclared: direktoro
        # refuses to guess a shape for it rather than emitting Anthropic wire
        # keys onto an OpenAI-family endpoint. Discovered from the registry,
        # not named, so this keeps testing the real rule as entries come and go.
        undeclared = _undeclared_models()
        assert undeclared, "registry has no undeclared-thinking entry to test"
        with pytest.raises(ThinkingConfigError):
            build(extractor_model=undeclared[0],
                  thinking=Thinking(effort="high"))

    def test_no_spec_on_such_a_model_is_still_fine(self, build):
        # The refusal is about EMITTING a shape, not about the model: a run
        # that names no thinking key still works on every entry.
        undeclared = _undeclared_models()
        orch = build(extractor_model=undeclared[0])
        assert orch.extractor_model == undeclared[0]


def _undeclared_models():
    """Live, non-retired registry entries that declare no thinking surface."""
    from direktoro import known_models, model_info
    return [m for m in known_models()
            if not model_info(m).retired and thinking_support(m) is None]


def _temperature_accepting_thinkers():
    """Live entries that accept a temperature AND can be asked to think adaptively.

    The models on which meltiro can build the illegal pair. Discovered from
    direktoro's real registry rather than named, so an entry added or a quirk
    changed there is covered here with no edit — the same pattern as
    `_undeclared_models`.

    Two filters, and both matter:

      - accepts a temperature: no `no_temperature` quirk, and
        `supports_sampling_params`. A model with the quirk never has the
        conflict, because direktoro omits the temperature for every request it
        sends.
      - supports `adaptive`: `ACCEPTED_MODES` is meltiro's whole thinking
        vocabulary, so a model whose only on-mode is `budget` cannot be asked
        to think from a pipeline.yaml at all. `claude-haiku-4-5-*` is exactly
        that, and it is why the conflict is reachable on one model today rather
        than two — see
        `test_haiku_is_affected_at_the_api_but_unreachable_from_a_bundle`.
    """
    from direktoro import known_models, model_info
    out = []
    for m in known_models():
        info = model_info(m)
        if info.retired:
            continue
        quirks = info.quirks or {}
        if quirks.get("no_temperature") or not info.supports_sampling_params:
            continue
        support = thinking_support(m)
        if support is not None and "adaptive" in support.modes:
            out.append(m)
    return out


class TestTemperatureAndThinkingCannotBothApply:
    """A temperature and active thinking are a 400 on the 4.6-generation
    endpoints, and meltiro refuses that pair at startup rather than on the
    first paid call.

    This is not a rule meltiro owns. Anthropic documents that on the older
    models `temperature` and `top_k` are incompatible with thinking; direktoro
    encodes it, and REFUSES the pair rather than silently dropping either side,
    because dropping the temperature would change a scientific run's sampling
    behind the caller's back and dropping the thinking would change its
    reasoning. meltiro's job is to surface that refusal with the role and the
    two keys attached, before a client exists.

    It matters here because an ordinary bundle is one line away from it: a
    checker on `claude-sonnet-4-6` (which accepts a temperature) carrying
    `checker_temperature: 0.0` produces the pair the moment
    `checker_thinking_mode: adaptive` is added. That is a natural thing for an
    author to reach for, so the refusal has to name both keys and the role,
    rather than leaving the author to work out which of two innocuous lines
    is the problem.
    """

    def test_the_registry_still_has_a_model_this_can_happen_on(self):
        # If this ever goes empty the conflict has been designed out of the
        # working set and the rest of this class is vacuous — which is worth
        # being told about rather than silently passing.
        assert _temperature_accepting_thinkers()

    def test_the_shipped_checker_model_is_one_of_them(self, config_dir):
        # Read off the real bundle, not asserted against a literal, so a
        # repoint of `checker_model` moves this test with it.
        pipeline = load_config_bundle(config_dir).pipeline
        assert pipeline["checker_model"] in _temperature_accepting_thinkers()
        # And it does carry a temperature, which is the other half.
        assert pipeline["checker_temperature"] is not None

    def test_a_temperature_plus_adaptive_thinking_is_refused(self, build):
        for model in _temperature_accepting_thinkers():
            with pytest.raises(ThinkingConfigError) as exc:
                build(checker_model=model, checker_max_tokens=8192,
                      checker_temperature=0.0,
                      checker_thinking=Thinking(mode="adaptive"))
            msg = str(exc.value)
            # The role, so the operator knows which of three to look at.
            assert "checker role" in msg
            assert model in msg
            # Both keys, because either one is a legitimate thing to change and
            # a message naming only the thinking keys points at one of two
            # doors.
            assert "checker_thinking_mode" in msg
            assert "checker_temperature" in msg

    def test_dropping_the_temperature_is_a_real_exit(self, build):
        # The refusal claims the shape is accepted without a temperature. That
        # claim is checked, not just asserted in prose: an exit the operator is
        # sent to that did not actually work would be worse than no advice.
        for model in _temperature_accepting_thinkers():
            orch = build(checker_model=model, checker_max_tokens=8192,
                         checker_temperature=None,
                         checker_thinking=Thinking(mode="adaptive"))
            assert orch.checker_config.thinking == Thinking(mode="adaptive")
            assert orch.checker_config.temperature is None

    def test_disabling_thinking_is_the_other_exit(self, build):
        # The symmetric choice: keep sampling control, do not think.
        for model in _temperature_accepting_thinkers():
            orch = build(checker_model=model, checker_max_tokens=1024,
                         checker_temperature=0.0,
                         checker_thinking=Thinking(mode="disabled"))
            assert orch.checker_config.temperature == 0.0

    def test_saying_nothing_about_thinking_keeps_the_temperature_legal(
            self, build):
        # The state every existing bundle is in, including the shipped one.
        # A false refusal here would break every adopter of that bundle.
        for model in _temperature_accepting_thinkers():
            orch = build(checker_model=model, checker_max_tokens=1024,
                         checker_temperature=0.0, checker_thinking=None)
            assert orch.checker_config.thinking is None

    def test_a_no_temperature_model_has_no_conflict(self, build):
        # Opus 4.8 rejects the temperature parameter outright, so direktoro
        # omits it and there is no pair to refuse. This is why the working
        # set's extractor and reviewer are unaffected.
        orch = build(checker_model=OPUS_4_8, checker_max_tokens=8192,
                     checker_temperature=0.0,
                     checker_thinking=Thinking(mode="adaptive"))
        assert orch.checker_config.thinking == Thinking(mode="adaptive")

    def test_a_refusal_that_is_not_about_temperature_does_not_name_the_key(
            self, build):
        # The advice is earned per refusal, by asking direktoro whether the
        # same shape resolves without a temperature — not attached to every
        # `ThinkingUnsupported`. An effort level the model lacks is still wrong
        # with no temperature at all, so sending the operator to edit
        # `checker_temperature` would waste the one edit they make.
        with pytest.raises(ThinkingConfigError) as exc:
            build(checker_model=SONNET_4_6, checker_max_tokens=8192,
                  checker_temperature=0.0,
                  checker_thinking=Thinking(mode="adaptive", effort="xhigh"))
        msg = str(exc.value)
        assert "xhigh" in msg
        assert "checker_temperature" not in msg

    def test_haiku_is_affected_at_the_api_but_unreachable_from_a_bundle(self):
        # The second model the API restriction covers. It accepts a temperature
        # and it can think, so the pair IS illegal on it — but its only on-mode
        # is `budget`, which meltiro deliberately does not expose (see
        # `thinking.ACCEPTED_MODES`), so no pipeline.yaml can construct the
        # pair on it. Pinned so that exposing `budget` later cannot quietly
        # open a path this class does not cover.
        from direktoro import known_models, model_info
        haikus = [m for m in known_models()
                  if "haiku-4-5" in m and not model_info(m).retired]
        assert haikus, "registry has no live haiku-4-5 entry"
        for model in haikus:
            quirks = model_info(model).quirks or {}
            assert not quirks.get("no_temperature")
            support = thinking_support(model)
            assert support is not None and "budget" in support.modes
            assert "adaptive" not in support.modes
            assert not set(support.modes) & set(thinking_mod.ACCEPTED_MODES)


class TestBuildThinking:
    def test_naming_neither_key_yields_no_spec(self):
        assert thinking_mod.build_thinking("checker", None, None) is None

    def test_effort_alone_is_a_spec(self):
        spec = thinking_mod.build_thinking("checker", None, "low")
        assert spec == Thinking(effort="low")

    def test_mode_alone_is_a_spec(self):
        spec = thinking_mod.build_thinking("review", "disabled", None)
        assert spec == Thinking(mode="disabled")

    def test_an_unknown_mode_names_the_role_s_own_key(self):
        with pytest.raises(ThinkingConfigError) as exc:
            thinking_mod.build_thinking("extractor", "sometimes", None)
        assert "extractor_thinking_mode" in str(exc.value)

    def test_budget_mode_is_refused_by_name(self):
        # direktoro has a `budget` mode; meltiro deliberately does not expose it
        # (it needs a companion budget_tokens number, and the 4.7-generation
        # rejects it outright). The refusal has to say so rather than reading as
        # a typo.
        with pytest.raises(ThinkingConfigError) as exc:
            thinking_mod.build_thinking("checker", "budget", None)
        assert "budget" in str(exc.value)
        assert "adaptive" in str(exc.value)

    def test_an_unknown_effort_names_the_role_s_own_key(self):
        with pytest.raises(ThinkingConfigError) as exc:
            thinking_mod.build_thinking("review", None, "maximum")
        assert "review_thinking_effort" in str(exc.value)


class TestWillThink:
    def test_the_working_set_thinks_when_nothing_is_said(self):
        assert thinking_mod.will_think(OPUS_5, None) is True
        assert thinking_mod.will_think(SONNET_5, None) is True

    def test_the_previous_generation_does_not(self):
        assert thinking_mod.will_think(OPUS_4_8, None) is False
        assert thinking_mod.will_think(SONNET_4_6, None) is False

    def test_an_explicit_disabled_mode_wins(self):
        assert thinking_mod.will_think(
            OPUS_5, Thinking(mode="disabled")) is False

    def test_an_explicit_adaptive_mode_wins(self):
        assert thinking_mod.will_think(
            SONNET_4_6, Thinking(mode="adaptive")) is True

    def test_effort_alone_does_not_decide_the_mode(self):
        # effort and mode are independent: naming an effort on a model that
        # does not think by default does not turn thinking on.
        assert thinking_mod.will_think(
            OPUS_4_8, Thinking(effort="max")) is False
        assert thinking_mod.will_think(
            OPUS_5, Thinking(effort="low")) is True


# ---------------------------------------------------------------------------
# Cross-repo contract: the capability facts come from direktoro's real code
# ---------------------------------------------------------------------------

class TestDirektoroContract:
    """Imports direktoro's REAL registry, not a hand-copied constant.

    If direktoro renames a mode, drops an effort level, changes the minimum
    thinking allocation, or flips a model's `default_on`, these go red rather
    than meltiro silently enforcing a floor or accepting a mode that no longer
    means what it meant.
    """

    def test_meltiros_modes_are_a_subset_of_direktoros(self):
        assert set(thinking_mod.ACCEPTED_MODES) <= set(THINKING_MODES)

    def test_budget_is_the_only_direktoro_mode_meltiro_withholds(self):
        withheld = set(THINKING_MODES) - set(thinking_mod.ACCEPTED_MODES)
        assert withheld == {"budget"}

    def test_the_floor_is_direktoros_minimum_plus_meltiros_headroom(self):
        for model in (OPUS_5, SONNET_5, OPUS_4_8, SONNET_4_6):
            support = thinking_support(model)
            assert thinking_mod.thinking_cap_floor(model) == (
                support.budget_min + thinking_mod.MIN_RESPONSE_TOKENS), model

    def test_the_floors_think_side_rests_on_a_parameter_the_models_reject(
            self):
        # Stated in code rather than left implied. The think half of the floor
        # is `budget_min` — the minimum for a `budget_tokens` allocation — and
        # the models this seam exists for do not accept that parameter at all:
        # direktoro's registry says so by omitting `budget` from their modes,
        # and the endpoints answer a `budget_tokens` with a 400. So on the
        # working set the floor's arithmetic rests on a quantity the API no
        # longer exposes.
        #
        # It is kept deliberately, as the smallest allocation any Claude
        # endpoint has ever called a think and therefore the smallest defensible
        # LOWER BOUND — not as a live minimum, and not as a measurement of what
        # an adaptive think costs. `thinking_cap_floor.__doc__` says so, and
        # this is the test that goes red if direktoro ever changes the premise.
        for model in (OPUS_5, SONNET_5, OPUS_4_8):
            assert "budget" not in thinking_support(model).modes, model
        # The 4.6 generation is the one that still declares budget mode
        # (deprecated), which is why the number is in the registry at all.
        assert "budget" in thinking_support(SONNET_4_6).modes

    def test_the_floor_does_not_scale_with_reasoning_effort(self):
        # Deliberate, and the reason the refusal carries the sizing advice
        # instead. An adaptive think spends more at higher effort, but Anthropic
        # publishes no effort-to-thinking-token figure — adaptive thinking has
        # no budget to tune, by design — so a per-effort floor would be a number
        # meltiro invented, and an invented floor produces false refusals on
        # configurations nobody has shown to be broken.
        #
        # The cost of that choice is pinned here honestly: at `max` effort a cap
        # of exactly the floor is ACCEPTED, and will very probably truncate. The
        # refusal message is what carries that warning. If a measured
        # per-effort floor ever exists — it needs paid calls to obtain — this is
        # the test that should change.
        assert thinking_mod.thinking_cap_floor(OPUS_5) == 2048
        assert EFFORT_LEVELS, "no effort levels to sweep"
        for level in EFFORT_LEVELS:
            thinking_mod.check_role_thinking(
                "checker", OPUS_5, max_tokens=2048, temperature=None,
                thinking=Thinking(mode="adaptive", effort=level))
            # One token below the floor is refused at that same effort, so the
            # calls above are an accepted cap rather than a guard that has
            # stopped looking.
            with pytest.raises(ThinkingConfigError):
                thinking_mod.check_role_thinking(
                    "checker", OPUS_5, max_tokens=2047, temperature=None,
                    thinking=Thinking(mode="adaptive", effort=level))

    def test_the_working_set_really_does_think_by_default(self):
        # The premise of the whole guard, read off direktoro's registry. If this
        # ever goes false the guard becomes a false refusal and must be revisited.
        assert thinking_support(OPUS_5).default_on is True
        assert thinking_support(SONNET_5).default_on is True

    def test_a_model_with_no_declared_surface_has_no_floor(self):
        # `will_think` and `thinking_cap_floor` agree on the undeclared case:
        # no claim either way, and no refusal invented from the absence. Asked
        # of a live registry entry that declares no thinking surface, not of a
        # bare ThinkingSupport(), because the branch under test is the one
        # `thinking_support` returning None takes.
        undeclared = _undeclared_models()
        assert undeclared, "registry has no undeclared-thinking entry to test"
        for model in undeclared:
            assert thinking_mod.thinking_cap_floor(model) is None, model
            assert thinking_mod.will_think(model, None) is False, model
        # A declared model DOES get a floor, so `None` above is the absence
        # speaking and not the function having stopped computing one.
        assert thinking_mod.thinking_cap_floor(OPUS_5) == (
            ThinkingSupport().budget_min + thinking_mod.MIN_RESPONSE_TOKENS)

    def test_every_effort_level_meltiro_accepts_is_a_direktoro_level(self):
        assert EFFORT_LEVELS, "no effort levels to sweep"
        for level in EFFORT_LEVELS:
            assert thinking_mod.build_thinking(
                "extractor", None, level).effort == level
        # And a level direktoro does not publish is refused, so the sweep
        # above is an allowlist rather than a pass-through.
        with pytest.raises(ThinkingConfigError):
            thinking_mod.build_thinking("extractor", None, "colossal")


# ---------------------------------------------------------------------------
# Truncation is named, never silent
# ---------------------------------------------------------------------------

class _StopReasonAdapter:
    """A checker adapter that returns one response with a chosen stop_reason."""

    def __init__(self, stop_reason, verdict="ok", rationale="x"):
        self._stop_reason = stop_reason
        self._verdict = verdict
        self._rationale = rationale

    def create_message(self, **kwargs):
        from types import SimpleNamespace
        from meltiro.tools import CHECKER_VERDICT_TOOL_NAME
        block = SimpleNamespace(
            type="tool_use", name=CHECKER_VERDICT_TOOL_NAME,
            input={"verdict": self._verdict, "rationale": self._rationale})
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0)
        return SimpleNamespace(
            content=[block], usage=usage, stop_reason=self._stop_reason,
            resolved_model=SONNET_5, provider="anthropic", base_url=None,
            raw_request={}, raw_response={}, wire_request=None,
            decoding_params={}, reported_cost=None)


class TestTruncationIsNamed:
    def _check(self, adapter, max_tokens=1024):
        from meltiro.checker import check_one_field
        config = CheckerConfig(checker_model=SONNET_5, api_key="x",
                               max_tokens=max_tokens)
        return check_one_field(
            system_message_blocks=[], user_message_blocks=[],
            config=config, adapter=adapter)

    def test_a_max_tokens_stop_is_reported_as_truncation(self):
        with pytest.raises(CheckerError) as exc:
            self._check(_StopReasonAdapter("max_tokens"))
        msg = str(exc.value)
        assert "TRUNCATED" in msg
        assert "max_tokens" in msg
        assert "1024" in msg

    def test_the_truncation_message_explains_the_thinking_cause(self):
        # A checker on a model that thinks by default gets the reason it was
        # truncated, not just the fact.
        msg = thinking_mod.truncation_message("checker", SONNET_5, 1024,
                                              thinking=None)
        assert "thinks by default" in msg
        assert "caps thinking and response together" in msg

    def test_a_non_thinking_model_gets_the_plain_advice(self):
        msg = thinking_mod.truncation_message("extractor", OPUS_4_8, 4096,
                                              thinking=None)
        assert "thinks by default" not in msg
        assert "Raise the role's max_tokens" in msg

    # -- The diagnosis must read the ROLE's spec, not the model default -------

    def test_a_role_that_turned_thinking_on_is_not_told_it_has_no_think(self):
        # `truncation_message` takes the role's spec, NOT `will_think(model,
        # None)`. Reading the model's `default_on` alone misdiagnoses the one
        # configuration this seam adds: a role that explicitly sets `adaptive`
        # on a model whose default is off DOES think, and would be handed the
        # non-thinking advice ("raise max_tokens, or shorten what it is asked
        # to produce") for a truncation that thinking caused.
        msg = thinking_mod.truncation_message(
            "extractor", OPUS_4_8, 32768, thinking=Thinking(mode="adaptive"))
        assert "Thinking is on for this call" in msg
        assert "this role asked for thinking" in msg
        assert "caps thinking and response together" in msg
        assert "Raise the role's max_tokens" not in msg

    def test_a_role_that_turned_thinking_off_gets_the_plain_advice(self):
        # The mirror image, and the reason the spec cannot simply be assumed
        # from the model either: `disabled` on a model that thinks by default
        # means the truncation is NOT a thinking problem, and hinting that it
        # might be would send the operator after the wrong cause.
        msg = thinking_mod.truncation_message(
            "checker", SONNET_5, 1024, thinking=Thinking(mode="disabled"))
        assert "Thinking is on for this call" not in msg
        assert "thinks by default" not in msg
        assert "Raise the role's max_tokens" in msg

    def test_effort_alone_does_not_change_the_diagnosis(self):
        # Consistent with `will_think`: naming an effort is not turning thinking
        # on. An Opus 4.8 role with an effort but no mode still does not think,
        # so it still gets the plain advice.
        msg = thinking_mod.truncation_message(
            "extractor", OPUS_4_8, 32768, thinking=Thinking(effort="max"))
        assert "Thinking is on for this call" not in msg
        assert "Raise the role's max_tokens" in msg

    def test_the_spec_is_keyword_only_with_no_default(self):
        # What actually protects the call sites. With no default, a caller that
        # forgets the role's spec gets a TypeError at the call, not a silently
        # wrong diagnosis printed into a run's warnings, which is the failure
        # mode this whole test class exists to prevent. A default would reopen
        # that failure for EVERY call site at once.
        import inspect
        param = inspect.signature(
            thinking_mod.truncation_message).parameters["thinking"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty

    def test_the_checker_seam_hands_over_the_roles_own_spec(self):
        # End-to-end through `check_one_field`, so the wiring is pinned and not
        # just the helper: an Opus 4.8 checker that asked for thinking must be
        # told thinking is why it truncated.
        from meltiro.checker import check_one_field
        config = CheckerConfig(checker_model=OPUS_4_8, api_key="x",
                               max_tokens=4096,
                               thinking=Thinking(mode="adaptive"))
        with pytest.raises(CheckerError) as exc:
            check_one_field(
                system_message_blocks=[], user_message_blocks=[],
                config=config, adapter=_StopReasonAdapter("max_tokens"))
        assert "Thinking is on for this call" in str(exc.value)

    def test_the_orchestrator_seam_hands_over_the_roles_own_spec(
            self, build, capsys):
        # The other half of the wiring. `_warn_if_truncated` is the extractor
        # and reviewer path, and it warns to stderr rather than raising, so a
        # wrong diagnosis here would be quieter still.
        from types import SimpleNamespace
        orch = build(extractor_model=OPUS_4_8,
                     thinking=Thinking(mode="adaptive"))
        orch._warn_if_truncated(
            "extractor", orch.extractor_model, orch.extractor_max_tokens,
            SimpleNamespace(stop_reason="max_tokens"), thinking=orch.thinking)
        assert "Thinking is on for this call" in capsys.readouterr().err

    def test_a_normal_stop_is_parsed_as_before(self):
        # The guard must not fire on an ordinary completion: this is the path
        # every real verdict takes.
        res = self._check(_StopReasonAdapter("end_turn"))
        assert res["verdict"] == "ok"

    def test_a_truncated_field_degrades_rather_than_aborting_the_run(self):
        # run_checker_batch wraps a CheckerError as an error-origin challenge on
        # that one field, so a truncated verdict costs one field's judgement
        # rather than the whole extraction.
        from meltiro.checker import run_checker_batch
        config = CheckerConfig(checker_model=SONNET_5, api_key="x",
                               max_tokens=1024, concurrency=1)
        out = run_checker_batch(
            calls=[{"field_path": "study.f1", "system_message_blocks": [],
                    "user_message_blocks": []}],
            config=config, adapter=_StopReasonAdapter("max_tokens"))
        res = out["study.f1"]
        assert res["verdict"] == "challenge"
        assert res["error_origin"] is True
        assert "TRUNCATED" in res["error"]
