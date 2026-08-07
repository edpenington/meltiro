"""Identity warnings do not duplicate across resumes.

The identity degradation (the checker falling back to title + DOI) is gated by
a one-shot in-memory latch that fires at most once per run. The latch lives
only in memory, so it resets on resume, and a run resumed N times re-hits the
degraded path N times. `add_warning` therefore deduplicates the persisted list:
an exact-match warning is recorded ONCE, however many segments re-hit that
path. Appending unconditionally would grow meta.warnings with N identical
copies of one fact, and a reader counting warnings would read that as N
problems.
"""

from types import SimpleNamespace

from meltiro.orchestrator import Orchestrator
from meltiro.session import Session


def _create(tmp_path):
    return Session.create(
        "376",
        config_fp="config_fp:abc123def456",
        checker_fp="checker_fp:def",
        review_fp="review_fp:xyz",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        extractor_model="opus", checker_model="sonnet", review_model="opus",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        runs_dir=tmp_path,
    )


def _degraded_stub(session):
    """A bare Orchestrator carrying just what `_degraded_identity_context`
    reads. A fresh stub models a resume: the one-shot latch starts False, as it
    does in __init__ on every fresh process."""
    stub = Orchestrator.__new__(Orchestrator)
    stub.session = session
    stub.study_id = "376"
    stub.template = {"role_fields": {"summary": {"variable": "abstract"}}}
    stub.bundle = SimpleNamespace(title="A synthetic study", doi="10.0/x")
    stub._identity_degradation_warned = False
    return stub


# ---------------------------------------------------------------------------
# Session-level: add_warning skips an exact duplicate but still records a
# genuinely different warning.
# ---------------------------------------------------------------------------

def test_add_warning_skips_exact_duplicate(tmp_path):
    s = _create(tmp_path)
    msg = "identity-degradation: study 376 has no manifest summary ..."

    s.add_warning(msg)
    s.add_warning(msg)  # a resume re-hits the same degraded path
    s.add_warning(msg)  # and another resume
    assert s.meta["warnings"] == [msg]

    # A genuinely different warning is still recorded.
    other = "summary-mismatch: study 376 ..."
    s.add_warning(other)
    assert s.meta["warnings"] == [msg, other]

    # The dedup is on the persisted list, not just the in-memory dict: a
    # reload from disk (as resume does) sees each warning once.
    reloaded = Session.resume(s.session_dir)
    assert reloaded.meta["warnings"] == [msg, other]


# ---------------------------------------------------------------------------
# Resume twice: the identity-degradation warning appears exactly once even
# though the latch resets each segment.
# ---------------------------------------------------------------------------

def test_identity_degradation_warning_survives_two_resumes_once(tmp_path):
    session = _create(tmp_path)

    # Initial run plus two resumes. Each is a fresh stub (latch reset to False),
    # and each re-hits the degraded identity path and warns.
    for _ in range(3):
        _degraded_stub(session)._degraded_identity_context()

    persisted = [w for w in session.meta["warnings"]
                 if w.startswith("identity-degradation")]
    assert len(persisted) == 1
