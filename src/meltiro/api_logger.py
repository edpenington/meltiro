"""Verbatim capture of every API call made by the extractor, checker, and
final reviewer. The `request` record is the canonical request as the adapter
built it — every key it carries, with only inline image bytes stubbed — and
the `response` record is the full response (id, model, stop_reason, usage,
content). Both are written to `api_calls.jsonl` in the session directory at
the time of the call; a wire request that differs from the canonical one is
stored beside it, redacted the same way (`redact_wire_request`).

The ONLY redaction is for inbound image content blocks: the model's
input messages contain base64-encoded PNG bytes (large, and identical
to the paper bundle's cropped figures). To avoid duplicating image bytes
per call, those blocks are replaced with an
`image_ref` carrying `media_type`, `sha256`, and `byte_length`. The
hashes match the per-session `image_hashes` captured at session start,
so the transcript renderer can detect drift if a figure is re-cropped
after the run. Which block shapes carry image bytes on which wire is
direktoro's knowledge, so the redactors are imported from there; what this
module owns is the entry envelope, the file, and the lock around it.

Response content blocks never contain images (the model emits text and
tool_use only), so no response redaction is required.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from direktoro import (
    redact_messages, redact_system, redact_wire_request, response_to_dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    and an optional `wire_request` — the request that actually went out.

    That key is written ONLY when the wire request is not the canonical one
    already stored under `request`. Every adapter response carries a
    `wire_request`, and where the canonical shape IS the wire shape the two are
    the same object, so an identity test tells them apart for nothing and keeps
    a log entry from carrying the same megabytes of prompt twice. `!=` would
    answer the same question at the cost of a deep compare over every image
    block on every call.

    `response` is a plain dict on every adapter path; `response_to_dict` is
    kept for a caller that hands `Session.log_api_call` an SDK object directly.
    """
    wire_request = extra.pop("wire_request", None)
    # The canonical request key for key, with `system` and `messages` passed
    # through the redactors. Which keys a call carries is the adapter's answer
    # and varies by model — a thinking spec, an output config, whichever
    # sampling controls that model accepts — so anything short of every key
    # omits exactly what nobody thought to enumerate. It is also the only copy
    # wherever the canonical request IS the wire request, which is suppressed
    # below.
    request = dict(request_kwargs)
    if "system" in request:
        request["system"] = redact_system(request["system"])
    if "messages" in request:
        request["messages"] = redact_messages(request["messages"])
    entry = {
        "ts": _utc_now_iso(),
        "event": "api_call",
        "call_type": call_type,
        "request": request,
        "response": response_to_dict(response),
        **extra,
    }
    if wire_request is not None and wire_request is not request_kwargs:
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
