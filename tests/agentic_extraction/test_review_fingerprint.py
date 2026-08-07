"""Orchestrator-level provenance tests for the final-review stage.

These build a real Orchestrator against the shipped config bundle + the
synthetic paper bundle and call prepare_new_session (no API), then inspect
the recorded fingerprints. One test drives _final_review offline with a
stubbed client to prove the review call honours review_max_tokens.

Covers:
  - review_fp and review_model are recorded in run.json
  - editing the review system prompt moves review_fp and only review_fp
  - changing review_model moves review_fp and only review_fp
  - changing review_max_tokens moves review_fp and only review_fp
  - the review LLM call uses review_max_tokens, not extractor_max_tokens
  - changing max_tool_calls does NOT move config_fp: the tool-call cap is an
    operational budget, out of every fingerprint (cap-out-of-fingerprint)
"""

import shutil
from types import SimpleNamespace

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator


def _prepared_orch(config_dir, bundle_dir, out_dir, *,
                   review_model="claude-opus-4-7",
                   review_max_tokens=4096,
                   extractor_max_tokens=32768,
                   max_tool_calls=100):
    """Construct an Orchestrator and run prepare_new_session (no network)."""
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    orch = Orchestrator(
        config, bundle, out_dir,
        extractor_model="claude-opus-4-7",
        checker_config=CheckerConfig(
            checker_model="claude-sonnet-4-6", api_key="x"),
        review_model=review_model,
        review_max_tokens=review_max_tokens,
        extractor_max_tokens=extractor_max_tokens,
        max_tool_calls=max_tool_calls,
        api_key="x",
    )
    orch.prepare_new_session()
    return orch


def _fps(orch):
    m = orch.session.meta
    return m["config_fp"], m["checker_fp"], m["review_fp"]


def test_review_fp_and_model_recorded_in_meta(
        config_dir, bundle_minimal_dir, tmp_path):
    orch = _prepared_orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
    m = orch.session.meta
    assert m["review_fp"].startswith("review_fp:")
    assert m["review_model"] == "claude-opus-4-7"
    # The three stage fingerprints are distinct namespaces.
    assert m["config_fp"].startswith("config_fp:")
    assert m["checker_fp"].startswith("checker_fp:")


def test_editing_review_prompt_moves_only_review_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    cfg = tmp_path / "config"
    shutil.copytree(config_dir, cfg)

    orch_a = _prepared_orch(cfg, bundle_minimal_dir, tmp_path / "a")
    config_a, checker_a, review_a = _fps(orch_a)

    review_prompt = cfg / "prompts" / "review_system.md"
    review_prompt.write_text(
        review_prompt.read_text(encoding="utf-8")
        + "\n\nExtra reviewer guidance appended for the test.\n",
        encoding="utf-8")

    orch_b = _prepared_orch(cfg, bundle_minimal_dir, tmp_path / "b")
    config_b, checker_b, review_b = _fps(orch_b)

    assert review_a != review_b          # the review prompt changed
    assert config_a == config_b          # extractor stage untouched
    assert checker_a == checker_b        # checker stage untouched


def test_changing_review_model_moves_only_review_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    orch_a = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "a",
        review_model="claude-opus-4-7")
    orch_b = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "b",
        review_model="claude-sonnet-4-6")

    config_a, checker_a, review_a = _fps(orch_a)
    config_b, checker_b, review_b = _fps(orch_b)

    assert review_a != review_b
    assert config_a == config_b
    assert checker_a == checker_b


def test_changing_review_max_tokens_moves_only_review_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    orch_a = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "a",
        review_max_tokens=4096)
    orch_b = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "b",
        review_max_tokens=8192)

    config_a, checker_a, review_a = _fps(orch_a)
    config_b, checker_b, review_b = _fps(orch_b)

    assert review_a != review_b
    assert config_a == config_b
    assert checker_a == checker_b


def test_changing_the_tool_call_cap_does_not_move_config_fp(
        config_dir, bundle_minimal_dir, tmp_path):
    # The tool-call cap is an operational bound recorded in meta, not config
    # identity. It rides in no fingerprint, so changing it must NOT move
    # config_fp (nor review_fp): a resume under a different cap is accepted,
    # and the two runs share a session namespace. This is the review-stage twin
    # of the extractor's cap-out-of-fingerprint stability test.
    orch_a = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "a", max_tool_calls=100)
    orch_b = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "b", max_tool_calls=999)

    assert orch_a.session.meta["config_fp"] == \
        orch_b.session.meta["config_fp"]
    assert orch_a.session.meta["review_fp"] == \
        orch_b.session.meta["review_fp"]


# ---------------------------------------------------------------------------
# The review call honours review_max_tokens, driven offline.
# ---------------------------------------------------------------------------

class _CaptureStream:
    def __init__(self, resp):
        self._resp = resp
        self.text_stream = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._resp


class _CaptureMessages:
    def __init__(self, sink, resp):
        self._sink = sink
        self._resp = resp

    def stream(self, **kwargs):
        self._sink["kwargs"] = kwargs
        return _CaptureStream(self._resp)


class _CaptureClient:
    def __init__(self, sink, resp):
        self.messages = _CaptureMessages(sink, resp)


def test_review_call_uses_configured_review_max_tokens(
        config_dir, bundle_minimal_dir, tmp_path):
    orch = _prepared_orch(
        config_dir, bundle_minimal_dir, tmp_path / "runs",
        review_max_tokens=4096, extractor_max_tokens=32768)

    sink = {}
    # Empty content -> _final_review returns final_review_no_response after
    # building (and dispatching) the review call, which is all we need to
    # inspect the request's max_tokens.
    resp = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    orch._anthropic_client = lambda: _CaptureClient(sink, resp)

    status = orch._final_review()

    assert status == "final_review_no_response"
    assert sink["kwargs"]["max_tokens"] == 4096
    assert sink["kwargs"]["max_tokens"] != orch.extractor_max_tokens
