"""Verbatim capture of every provider API call made by the extractor,
checker, and final reviewer. Each call's full request kwargs in the
canonical Anthropic shape (model, system, tools, messages, params) and full
response (id, model, stop_reason, usage, content) are written to
`api_calls.jsonl` in the session directory at the time of the call; a
non-Anthropic provider's differing wire request is stored beside it,
redacted the same way (`redact_wire_request`).

The ONLY redaction is for inbound image content blocks: the model's
input messages contain base64-encoded PNG bytes (large, and identical
to the paper bundle's cropped figures). To avoid duplicating image bytes
per call, those blocks are replaced with an
`image_ref` carrying `media_type`, `sha256`, and `byte_length`. The
hashes match the per-session `image_hashes` captured at session start,
so the transcript renderer can detect drift if a figure is re-cropped
after the run.

Response content blocks never contain images (the model emits text and
tool_use only), so no response redaction is required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Image redaction (request side only)
# ---------------------------------------------------------------------------

def _image_ref(media_type: Any, data: str) -> dict:
    """Build an `image_ref` stub from base64 `data`, or a decode-error stub."""
    try:
        raw = base64.b64decode(data) if data else b""
    except Exception:
        # Defensive: log what is in hand rather than crash the audit log.
        return {
            "type": "image_ref",
            "media_type": media_type,
            "sha256": None,
            "byte_length": None,
            "_decode_error": True,
        }
    return {
        "type": "image_ref",
        "media_type": media_type,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
    }


def _redact_content_block(block: Any) -> Any:
    """Replace an inbound image content block with an `image_ref` stub
    carrying the bytes' sha256 and length. Handles both provider shapes:

      - Anthropic: `{"type": "image", "source": {"type": "base64",
        "media_type": ..., "data": <base64>}}`;
      - OpenAI Responses: `{"type": "input_image", "image_url":
        "data:<media>;base64,<base64>"}`.

      - OpenAI Chat Completions: `{"type": "image_url", "image_url":
        {"url": "data:<media>;base64,<base64>"}}`.

    Other blocks (and non-base64 image references) pass through unchanged.
    """
    if not isinstance(block, dict):
        return block
    if block.get("type") == "image":
        src = block.get("source") or {}
        if src.get("type") != "base64":
            return block
        return _image_ref(src.get("media_type"), src.get("data") or "")
    if block.get("type") == "input_image":
        url = block.get("image_url")
        if isinstance(url, str) and url.startswith("data:") \
                and ";base64," in url:
            header, b64 = url.split(";base64,", 1)
            media_type = header[len("data:"):] or None
            return _image_ref(media_type, b64)
        # A plain (non-data) image URL is a small reference; keep it.
        return block
    if block.get("type") == "image_url":
        url = block.get("image_url")
        # Chat Completions nests the URL: {"image_url": {"url": ...}}.
        inner = url.get("url") if isinstance(url, dict) else None
        if isinstance(inner, str) and inner.startswith("data:") \
                and ";base64," in inner:
            header, b64 = inner.split(";base64,", 1)
            media_type = header[len("data:"):] or None
            return _image_ref(media_type, b64)
        return block
    return block


def _redact_content_list(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    return [_redact_content_block(b) for b in content]


def redact_messages(messages: Any) -> Any:
    """Walk a `messages` list and substitute inbound image blocks with
    `image_ref` stubs. Other blocks (text, tool_use, tool_result) pass
    through verbatim; their content is needed for full audit fidelity.
    """
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, list):
            out.append({**m, "content": _redact_content_list(content)})
        else:
            out.append(m)
    return out


def redact_system(system: Any) -> Any:
    """`system` may be a string OR a list of content blocks (Anthropic
    supports both). Images are not expected in the system block, but
    redact defensively in case future templates attach reference images.
    """
    if isinstance(system, list):
        return _redact_content_list(system)
    return system


def redact_wire_request(wire: Any) -> Any:
    """Redact image bytes from a provider wire request before logging it.

    The OpenAI Responses request carries its conversation in a top-level
    `input` list; the Chat Completions request carries it in a `messages` list.
    Either way, message items hold content arrays that may include image data
    URLs (`input_image` for Responses, `image_url` parts for Chat
    Completions); walk those and stub the base64 bytes. Non-message items
    (function_call / function_call_output, `role:"tool"` results) and
    string-valued content pass through unchanged.
    """
    if not isinstance(wire, dict):
        return wire
    for key in ("input", "messages"):
        items = wire.get(key)
        if not isinstance(items, list):
            continue
        redacted = []
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("content"),
                                                     list):
                redacted.append(
                    {**item, "content": _redact_content_list(item["content"])})
            else:
                redacted.append(item)
        return {**wire, key: redacted}
    return wire


# ---------------------------------------------------------------------------
# Response serialisation
# ---------------------------------------------------------------------------

def serialise_response(response: Any) -> dict | None:
    """Convert an Anthropic SDK `Message` object to a plain dict.

    Prefers pydantic v2's `.model_dump()`. Falls back to attribute
    access. Never raises; returns `{"_serialisation_error": str}` on
    any failure so the api log stays append-only.
    """
    if response is None:
        return None
    # OpenAI responses are already normalised to a plain dict (model_dump) by
    # the adapter; store it verbatim.
    if isinstance(response, dict):
        return response
    try:
        if hasattr(response, "model_dump"):
            d = response.model_dump()
        elif hasattr(response, "dict"):
            d = response.dict()
        else:
            d = {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "stop_sequence": getattr(response, "stop_sequence", None),
                "role": getattr(response, "role", None),
                "type": getattr(response, "type", None),
                "usage": (response.usage.__dict__
                          if hasattr(response, "usage") and response.usage else None),
                "content": [
                    (b if isinstance(b, dict)
                     else (b.model_dump() if hasattr(b, "model_dump")
                           else getattr(b, "__dict__", str(b))))
                    for b in (response.content or [])
                ],
            }
        return d
    except Exception as exc:
        return {"_serialisation_error": repr(exc)}


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------

def make_entry(call_type: str, request_kwargs: dict, response: Any,
               **extra: Any) -> dict:
    """Compose one API-call log entry. Pure: no IO. The Session writes
    the result to api_calls.jsonl under a lock.

    `call_type` ∈ {"extractor", "checker", "final_review"}. `extra`
    carries call-site identifiers: `turn_id` for extractor, `round`
    and `field_path` for checker, provider/base_url/wire_model for provenance,
    and an optional `wire_request` (the provider's actual wire request, stored
    redacted for non-Anthropic providers whose wire request differs from the
    canonical one).
    """
    wire_request = extra.pop("wire_request", None)
    request = {
        "model": request_kwargs.get("model"),
        "max_tokens": request_kwargs.get("max_tokens"),
        "temperature": request_kwargs.get("temperature"),
        "system": redact_system(request_kwargs.get("system")),
        "tools": request_kwargs.get("tools"),
        "tool_choice": request_kwargs.get("tool_choice"),
        "messages": redact_messages(request_kwargs.get("messages")),
    }
    entry = {
        "ts": _utc_now_iso(),
        "event": "api_call",
        "call_type": call_type,
        "request": request,
        "response": serialise_response(response),
        **extra,
    }
    if wire_request is not None:
        entry["wire_request"] = redact_wire_request(wire_request)
    return entry


# ---------------------------------------------------------------------------
# Thread-safe writer
# ---------------------------------------------------------------------------

class ApiCallWriter:
    """Wraps the session's api_calls.jsonl with a lock for the parallel
    checker fan-out. POSIX append is line-atomic only for writes
    smaller than PIPE_BUF (4 KiB); checker entries can be ~20 KiB, so a
    lock is held per write.
    """

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
