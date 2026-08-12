"""The audit-log entry meltiro composes (meltiro.api_logger).

Which content-block shapes carry inline image bytes, and how each is stubbed,
belongs to `direktoro.wire_log` and is tested there. What this module covers is
meltiro's own share: the entry envelope, the rule that a wire request is stored
only when it differs from the canonical request already in the entry, and the
lock that keeps the parallel checker fan-out's lines whole. No network, no API
key.
"""

import base64
import json
import threading

from direktoro import NormalisedResponse, NormalisedUsage
from meltiro.api_logger import ApiCallWriter, make_entry
from meltiro.session import Session

# An opaque OpenAI-compatible base_url, used only as a provenance string.
# api_logger never validates the value against the registry (it just stores
# it), so a literal keeps this test hermetic and independent of the registry's
# routing decisions.
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


PNG = b"\x89PNG\r\n\x1a\nfake-bytes"
PNG_B64 = base64.b64encode(PNG).decode("ascii")


# ---------------------------------------------------------------------------
# Entry composition
# ---------------------------------------------------------------------------

class TestEntryComposition:
    def test_entry_carries_the_canonical_request_and_provenance(self):
        entry = make_entry(
            "extractor",
            {"model": "m", "max_tokens": 1024, "temperature": 0.3,
             "system": "SYS", "tools": [{"name": "t"}],
             "tool_choice": {"type": "auto"},
             "messages": [{"role": "user", "content": "hi"}]},
            {"id": "msg_1", "content": []},
            provider="openai_compat", base_url=ZAI_BASE_URL,
            wire_model="glm-5.2", turn_id=7)
        assert entry["event"] == "api_call"
        assert entry["call_type"] == "extractor"
        assert entry["ts"]
        assert entry["request"] == {
            "model": "m", "max_tokens": 1024, "temperature": 0.3,
            "system": "SYS", "tools": [{"name": "t"}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}]}
        # Call-site identifiers ride through verbatim.
        assert entry["provider"] == "openai_compat"
        assert entry["base_url"] == ZAI_BASE_URL
        assert entry["wire_model"] == "glm-5.2"
        assert entry["turn_id"] == 7

    def test_every_request_key_is_recorded(self):
        # The record is the canonical request key for key. Which decoding keys
        # a call carries is the adapter's answer and differs by model, so a
        # request the entry does not enumerate is still a request the entry
        # must carry whole.
        entry = make_entry(
            "extractor",
            {"model": "claude-opus-4-8", "max_tokens": 4096,
             "temperature": 0.3, "top_p": 0.95, "top_k": 40,
             "system": "SYS", "messages": [],
             "thinking": {"type": "enabled", "budget_tokens": 2048},
             "output_config": {"effort": "high"}},
            {"id": "msg_1", "content": []})
        assert entry["request"]["top_p"] == 0.95
        assert entry["request"]["top_k"] == 40
        assert entry["request"]["thinking"] == {
            "type": "enabled", "budget_tokens": 2048}
        assert entry["request"]["output_config"] == {"effort": "high"}

    def test_request_images_are_stubbed(self):
        # meltiro delegates the stubbing, but the entry it composes must
        # actually carry the redacted messages rather than the raw ones.
        entry = make_entry(
            "extractor",
            {"model": "m", "messages": [
                {"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": PNG_B64}}]}]},
            {"id": "msg_1", "content": []})
        stub = entry["request"]["messages"][0]["content"][0]
        assert stub["type"] == "image_ref"
        assert stub["byte_length"] == len(PNG)

    def test_response_dict_is_stored_verbatim(self):
        entry = make_entry("checker", {"model": "m", "messages": []},
                           {"model": "m", "status": "completed"})
        assert entry["response"] == {"model": "m", "status": "completed"}

    def test_response_object_is_flattened(self):
        # Session.log_api_call is public and takes whatever a caller hands it;
        # an SDK object still lands in the log as a dict rather than as a repr.
        class _Resp:
            def model_dump(self):
                return {"id": "msg_1", "content": []}

        entry = make_entry("extractor", {"model": "m", "messages": []},
                           _Resp())
        assert entry["response"] == {"id": "msg_1", "content": []}


# ---------------------------------------------------------------------------
# The wire request is stored only when it differs
# ---------------------------------------------------------------------------

class TestWireRequestStoredOnlyWhenItDiffers:
    def test_a_differing_wire_request_is_stored_redacted(self):
        wire = {"model": "glm-5.2", "input": [
            {"role": "user", "content": [
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{PNG_B64}"}]}]}
        entry = make_entry(
            "extractor",
            {"model": "glm-5.2", "messages": []},
            {"model": "glm-5.2", "status": "completed"},
            provider="openai_compat", base_url=ZAI_BASE_URL,
            wire_model="glm-5.2", wire_request=wire)
        assert entry["wire_request"]["input"][0]["content"][0]["type"] \
            == "image_ref"

    def test_a_differing_wire_lands_beside_the_canonical_request(self):
        # Both records, each whole: the canonical request keeps its own keys
        # while the wire request states what the endpoint was actually sent.
        entry = make_entry(
            "extractor",
            {"model": "glm-5.2", "messages": [], "temperature": 0.3,
             "top_p": 0.95},
            {"model": "glm-5.2", "status": "completed"},
            wire_request={"model": "glm-5.2", "input": [],
                          "reasoning": {"effort": "high"}})
        assert entry["request"]["top_p"] == 0.95
        assert entry["wire_request"]["reasoning"] == {"effort": "high"}

    def test_the_canonical_request_is_not_stored_twice(self):
        # Where the canonical shape IS the wire shape, the adapter hands both
        # names the same object; storing it under both keys would double every
        # entry's prompt. Suppressing the second copy costs the entry nothing,
        # because the first one is the whole request.
        request = {"model": "claude-sonnet-4-6", "max_tokens": 4096,
                   "top_p": 0.95, "top_k": 40,
                   "thinking": {"type": "enabled", "budget_tokens": 2048},
                   "output_config": {"effort": "high"},
                   "messages": [
                       {"role": "user",
                        "content": [{"type": "text", "text": "hi"}]}]}
        entry = make_entry(
            "extractor", request, {"id": "msg_1", "content": []},
            wire_request=request)
        assert "wire_request" not in entry
        assert entry["request"]["top_p"] == 0.95
        assert entry["request"]["top_k"] == 40
        assert entry["request"]["thinking"] == {
            "type": "enabled", "budget_tokens": 2048}
        assert entry["request"]["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# The same rule, through a real Session and back off disk
# ---------------------------------------------------------------------------

def _full_session(runs_dir):
    """A session at `--diagnostics full`, the level that keeps the wire log."""
    return Session.create(
        "376",
        config_fp="config_fp:abc123def456", checker_fp="checker_fp:def",
        review_fp="review_fp:xyz", instrument_fp="instrument_fp:inst",
        extractor_call_fp="call_fp:ext", checker_call_fp="call_fp:chk",
        review_call_fp="call_fp:rev", engine_fp="engine_fp:eng",
        extractor_model="claude-opus-4-8", checker_model="claude-sonnet-4-6",
        review_model="claude-opus-4-8",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        runs_dir=runs_dir, diagnostics="full")


class TestAliasedWireThroughTheSession:
    def test_the_decoding_params_land_once_in_api_calls_jsonl(self, tmp_path):
        # The Anthropic adapter records ONE dict under both `raw_request` and
        # `wire_request` (the canonical format is that wire), so this is the
        # path a real Claude run takes through the suppression above. Every
        # decoding parameter the run asked for has to be in the file, and in it
        # once.
        request = {
            "model": "claude-opus-4-8", "max_tokens": 4096,
            "temperature": 0.3, "top_p": 0.95, "top_k": 40,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "output_config": {"effort": "high"},
            "system": "SYS", "tools": [{"name": "update_study"}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": "hi"}]}],
        }
        response = NormalisedResponse(
            content=[],
            usage=NormalisedUsage(input_tokens=812, output_tokens=96),
            resolved_model="claude-opus-4-8", provider="anthropic",
            base_url=None, raw_request=request,
            raw_response={"id": "msg_1", "content": []},
            wire_request=request,
            decoding_params={"temperature": 0.3, "top_p": 0.95})

        session = _full_session(tmp_path)
        session.log_api_call(
            "extractor", response.raw_request, response.raw_response,
            provider=response.provider, base_url=response.base_url,
            wire_model=response.resolved_model,
            wire_request=response.wire_request, turn_id=1)

        lines = session.api_calls_path.read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "wire_request" not in entry
        assert entry["request"] == request
        # Once, on the raw line: the suppression is what makes the entry
        # single-copy, and the record above is what makes it complete.
        for key in ("top_p", "top_k", "thinking", "output_config"):
            assert lines[0].count(f'"{key}"') == 1, key


# ---------------------------------------------------------------------------
# The writer's lock
# ---------------------------------------------------------------------------

class TestApiCallWriter:
    def test_concurrent_writes_land_as_whole_lines(self, tmp_path):
        # Checker entries run to tens of KiB, past the size POSIX append keeps
        # atomic, so the lock is what stops the parallel fan-out from
        # interleaving two entries into one unparseable line.
        path = tmp_path / "api_calls.jsonl"
        writer = ApiCallWriter(path)
        big = "x" * 40000
        entries = [{"i": i, "pad": big} for i in range(24)]

        threads = [threading.Thread(target=writer.write, args=(e,))
                   for e in entries]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 24
        assert sorted(json.loads(line)["i"] for line in lines) == list(range(24))
