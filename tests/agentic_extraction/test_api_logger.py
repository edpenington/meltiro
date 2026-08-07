"""Audit-log redaction for inbound image content blocks (meltiro.api_logger).

The provider adapters themselves, and their wire-translation tests, belong to
the shared `direktoro` package. What this module covers is meltiro-specific:
`api_logger` captures every LLM call verbatim to `api_calls.jsonl`, redacting
ONLY the base64 image bytes in the request. That redaction is coupled to the
two provider wire shapes
direktoro emits (the Responses `input_image` and the Chat Completions nested
`image_url.url`), so the tests reference those shapes directly with
hand-authored fixtures. No network, no API key.
"""

import base64

from meltiro.api_logger import (
    _redact_content_block, make_entry, redact_wire_request)

# An opaque OpenAI-compatible base_url, used only as a provenance string in the
# make_entry redaction test. api_logger never validates the value against the
# registry (it just stores it), so a literal keeps this test hermetic and
# independent of the registry's routing decisions.
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


PNG = b"\x89PNG\r\n\x1a\nfake-bytes"
PNG_B64 = base64.b64encode(PNG).decode("ascii")


# ---------------------------------------------------------------------------
# Audit-log redaction covers the OpenAI input_image shape
# ---------------------------------------------------------------------------

class TestInputImageRedaction:
    def test_redact_input_image_block(self):
        block = {"type": "input_image",
                 "image_url": f"data:image/png;base64,{PNG_B64}"}
        out = _redact_content_block(block)
        assert out["type"] == "image_ref"
        assert out["media_type"] == "image/png"
        assert out["byte_length"] == len(PNG)
        assert "sha256" in out and out["sha256"]

    def test_plain_image_url_passes_through(self):
        block = {"type": "input_image",
                 "image_url": "https://example.com/x.png"}
        assert _redact_content_block(block) == block

    def test_redact_chat_image_url_block(self):
        # Chat Completions nests the data URL under image_url.url.
        block = {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}
        out = _redact_content_block(block)
        assert out["type"] == "image_ref"
        assert out["media_type"] == "image/png"
        assert out["byte_length"] == len(PNG)
        assert "sha256" in out and out["sha256"]

    def test_redact_wire_request_chat_messages(self):
        wire = {"model": "glm-5.2", "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [
                {"type": "text", "text": "see figure"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "ok"},
        ]}
        out = redact_wire_request(wire)
        redacted_img = out["messages"][1]["content"][1]
        assert redacted_img["type"] == "image_ref"
        assert redacted_img["byte_length"] == len(PNG)
        # String-content messages (system, tool) are untouched.
        assert out["messages"][0] == {"role": "system", "content": "SYS"}
        assert out["messages"][2] == {"role": "tool", "tool_call_id": "c",
                                      "content": "ok"}

    def test_redact_wire_request_input(self):
        wire = {"model": "glm-5.2", "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "see figure"},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{PNG_B64}"}]},
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
        ]}
        out = redact_wire_request(wire)
        redacted_img = out["input"][0]["content"][1]
        assert redacted_img["type"] == "image_ref"
        assert redacted_img["byte_length"] == len(PNG)
        # The function_call_output item is untouched.
        assert out["input"][1] == {"type": "function_call_output",
                                   "call_id": "c", "output": "ok"}

    def test_make_entry_stores_redacted_wire_and_provider(self):
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
        assert entry["provider"] == "openai_compat"
        assert entry["base_url"] == ZAI_BASE_URL
        assert entry["wire_model"] == "glm-5.2"
        assert entry["wire_request"]["input"][0]["content"][0]["type"] \
            == "image_ref"
        # A dict response (OpenAI model_dump) is stored verbatim.
        assert entry["response"] == {"model": "glm-5.2", "status": "completed"}
