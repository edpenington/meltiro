"""Run-outcome mapping is exhaustive and defaults to FAILURE.

The safety property under test is the one the reviewer stage exists for: with
`final_review: true`, a run must never finalise `complete` unless the reviewer
actually confirmed the extraction. The prose in `_final_review` and
`_finalise_review_stop` says so; these tests are what make it structural.

The failure this guards against is quiet. A loop outcome reaches `complete` by
not matching any of the failure branches, so an outcome nobody mapped -- one
renamed, one mistyped, one added later -- reads as a confirmation. The result
is an UNREVIEWED extraction finalised as a successful, citable run, with an
artefact that looks clean and an event log that says nothing. The same shape
sits on the extractor side, where `_finalise_loop_stop` answers None both for
the one continue-past outcome (`mark_complete_validated`) and for anything it
does not recognise, so None alone cannot mean the extractor finished.

So there are three things to pin, and the first is the one that matters:

  1. an outcome string the mapping does NOT know must not finalise `complete`
     (it finalises `error`), on both the review and the extractor path;
  2. every REAL outcome maps to the status it is documented to map to, so a
     future rename fails loudly here instead of silently downgrading to the
     default;
  3. the outcome SET itself is pinned against the source, so ADDING an outcome
     to a loop without mapping it is caught at the point it is added, not by
     whichever run first hits it in production, and a mapping branch no loop
     can reach is caught the same way.

These drive the REAL `run()`, `_final_review` and mapping helpers against the
real config bundle, with the two model loops stubbed at their own seams (the
same seam test_review_loop.py stubs the extractor at), so nothing touches the
network and the outcome under test is exactly the one the loop returned.
"""

import ast
import inspect

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.statuses import VALIDATED_STATUSES


# Every stage's key variable is present for this module: these tests
# reach the orchestrator's pre-spend key preflight, and the provider
# calls behind it are stubbed.
pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# The outcome vocabulary, as documented by the two loops
# ---------------------------------------------------------------------------

# `_review_loop`'s complete outcome set, and the run status each one must
# produce. Sourced from its own docstring and verified against every `return`
# in the loop; test_review_loop_outcome_set_is_pinned keeps the two in step.
REVIEW_OUTCOMES = {
    # The ONLY confirmation. The only one that may finalise `complete`.
    "review_mark_complete": ("complete", None),
    "review_abandoned": ("failed_validation", "review_surrendered"),
    "review_cap_hit": ("failed_validation", "review_cap_hit"),
    "review_text_only_stall": ("failed_validation", "review_text_only_stall"),
    "review_stalled": ("failed_validation", "review_stalled"),
    # Infrastructure failures, not judgements about the extraction.
    "review_no_response": ("error", None),
    "error": ("error", None),
}

# `_extractor_loop`'s outcome set, likewise. Unlike the review loop it reports
# no infrastructure failure of its own: a per-turn provider failure propagates
# as an exception and run()'s catch-all finalises `error`, so there is no
# `return "error"` to map. The mapping and the loop therefore cover the same
# five strings exactly, which is what the pinning test below asserts against
# the source.
EXTRACTOR_OUTCOMES = {
    "mark_complete_validated": ("complete", None),
    # The one bound that PAUSES rather than terminates.
    "tool_cap_hit": ("in_progress", None),
    "text_only_stall": ("failed_validation", "text_only_stall"),
    "extractor_stalled": ("failed_validation", "stalled"),
    "extractor_abandoned": ("failed_validation", "surrendered"),
}

# Outcome strings no mapping knows. Each models a real way the vocabulary
# drifts: a rename, a near-miss typo, a newly added outcome, and a stage
# prefix crossed over.
UNKNOWN_REVIEW_OUTCOMES = [
    "review_complete",          # renamed review_mark_complete
    "review_mark_completed",    # typo
    "review_budget_exhausted",  # a new outcome nobody mapped
    "mark_complete_validated",  # the EXTRACTOR's confirmation, crossed over
]

UNKNOWN_EXTRACTOR_OUTCOMES = [
    "mark_complete_confirmed",   # renamed mark_complete_validated
    "mark_complete_validate",    # typo
    "completeness_gate_passed",  # a new outcome nobody mapped
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir, *, final_review=True):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model=CHECKER),
        review_model=REVIEWER,
        max_checks_per_field=2,
        final_review=final_review,
        extractor_max_tokens=4096,
        review_max_tokens=4096,
    )


def _prepared(config_dir, bundle_dir, out_dir, *, review_outcome=None,
              extractor_outcome="mark_complete_validated", final_review=True):
    """An orchestrator whose two model loops return a chosen outcome.

    Both loops are replaced at the seam their own tests use, so `run()`,
    `_final_review` and both mapping helpers run for real against the outcome
    string under test and nothing touches the network. `_adapter_for_role` is
    stubbed non-None because `_final_review` refuses (correctly) to skip a
    review it has no adapter for; the stub is never called, because
    `_review_loop` is.
    """
    orch = _orch(config_dir, bundle_dir, out_dir, final_review=final_review)
    orch.prepare_new_session()

    def _extractor():
        if extractor_outcome == "mark_complete_validated":
            # Match the real loop: the completion flag is what the outcome
            # reports, and finalisation reads the record it left behind.
            orch.extraction_record.mark_complete()
        return extractor_outcome

    orch._extractor_loop = _extractor
    orch._adapter_for_role = lambda role: object()
    if review_outcome is not None:
        orch._review_loop = lambda *a, **k: (review_outcome, 0, 0)
    return orch


def _error_events(orch):
    return [e for e in orch.session.read_events() if e.get("event") == "error"]


def _returned_outcomes(func, *, tuple_first=False):
    """Every literal string `func` can return, read out of its source.

    `tuple_first` reads the first element of a returned tuple, which is the
    shape `_review_loop` returns (outcome, attempted, applied). Reading the
    source is the point: a docstring can go stale, and the whole risk here is
    an outcome that exists in the code and in no mapping.
    """
    tree = ast.parse(inspect.getsource(func).lstrip())
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if tuple_first and isinstance(value, ast.Tuple) and value.elts:
            value = value.elts[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.add(value.value)
    return out


# ---------------------------------------------------------------------------
# 1. The invariant: an unknown outcome never finalises `complete`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", UNKNOWN_REVIEW_OUTCOMES)
def test_an_unknown_review_outcome_never_finalises_complete(
        outcome, config_dir, bundle_minimal_dir, tmp_path):
    """The single most important property in the pipeline.

    With the reviewer on, an outcome the mapping does not recognise means
    nobody confirmed the extraction, so the run must finalise `error`. Anything
    else ships an unreviewed extraction as a citable result with nothing in the
    artefact to show for it.
    """
    out = tmp_path / "runs"
    orch = _prepared(config_dir, bundle_minimal_dir, out,
                     review_outcome=outcome)

    status = orch.run()

    assert status == "error"
    assert orch.session.meta["status"] == "error"
    # The reason is IN the artefact, not merely absent from it: an operator
    # reading the session must find the outcome string that stopped the run.
    errors = _error_events(orch)
    assert errors, "the refusal must be recorded in the event log"
    assert outcome in errors[-1]["message"]


@pytest.mark.parametrize("outcome", UNKNOWN_EXTRACTOR_OUTCOMES)
def test_an_unknown_extractor_outcome_never_finalises_complete(
        outcome, config_dir, bundle_minimal_dir, tmp_path):
    """The same hole on the extractor side.

    `_finalise_loop_stop` answers None for `mark_complete_validated` BY
    DESIGN, so None cannot mean "not a stop, therefore fine": it also answers
    None for everything it does not recognise. run() names the one outcome
    that may continue.

    The reviewer is off here, so nothing downstream could rescue the run: this
    is the extractor mapping alone deciding.
    """
    out = tmp_path / "runs"
    orch = _prepared(config_dir, bundle_minimal_dir, out,
                     extractor_outcome=outcome, final_review=False)

    status = orch.run()

    assert status == "error"
    assert orch.session.meta["status"] == "error"
    errors = _error_events(orch)
    assert errors
    assert outcome in errors[-1]["message"]


def test_an_unknown_extractor_outcome_is_refused_before_the_review_runs(
        config_dir, bundle_minimal_dir, tmp_path):
    """An unmapped extractor outcome must not buy itself a review.

    Running the reviewer on an extraction the extractor never signed off on
    would let a clean review paper over the earlier hole and finalise
    `complete`. The refusal happens first, so the reviewer is never reached.
    """
    out = tmp_path / "runs"
    reviewed = []
    orch = _prepared(config_dir, bundle_minimal_dir, out,
                     extractor_outcome="mark_complete_confirmed",
                     final_review=True)

    def _review_loop(*a, **k):
        reviewed.append(1)
        return ("review_mark_complete", 0, 0)

    orch._review_loop = _review_loop

    assert orch.run() == "error"
    assert reviewed == [], "the reviewer must not run after an unmapped stop"


@pytest.mark.parametrize("status", ["review_ok", "reviewed", "complete"])
def test_an_unknown_review_STATUS_never_finalises_complete(
        status, config_dir, bundle_minimal_dir, tmp_path):
    """The second guard, at the seam where `complete` is actually decided.

    There are two vocabularies here, not one: `_review_loop` returns outcomes,
    `_final_review` maps them onto run statuses, and run() reads the second.
    The tests above drive the first. This one drives the second directly, so
    the guard in run() is held to the same standard as the one in
    `_final_review` and neither can drift into being decoration.
    """
    out = tmp_path / "runs"
    orch = _prepared(config_dir, bundle_minimal_dir, out,
                     review_outcome="review_mark_complete")
    orch._final_review = lambda: status

    assert orch.run() == "error"
    assert orch.session.meta["status"] == "error"
    assert status in _error_events(orch)[-1]["message"]


# ---------------------------------------------------------------------------
# 2. Every real outcome maps to its documented status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "outcome,expected_status,expected_reason",
    [(k, v[0], v[1]) for k, v in sorted(REVIEW_OUTCOMES.items())])
def test_every_review_outcome_maps_to_its_documented_status(
        outcome, expected_status, expected_reason, config_dir,
        bundle_minimal_dir, tmp_path):
    """Renaming an outcome must fail HERE, loudly.

    Without this, a rename on one side of the mapping downgrades silently: the
    failure outcomes would stop matching and (before the exhaustiveness guard)
    fall through to success. With the guard they now fall through to `error`,
    which is safe but still wrong -- a surrendered review is
    `failed_validation`, not an infrastructure error -- so each real outcome
    is pinned to the exact status and `failure_reason` it is documented to
    produce.
    """
    out = tmp_path / "runs"
    orch = _prepared(config_dir, bundle_minimal_dir, out,
                     review_outcome=outcome)

    status = orch.run()

    assert status == expected_status
    assert orch.session.meta["status"] == expected_status
    assert orch.session.meta.get("failure_reason") == expected_reason
    # Only the reviewer's confirmation is a validated run.
    assert (status in VALIDATED_STATUSES) is (
        outcome == "review_mark_complete")


@pytest.mark.parametrize(
    "outcome,expected_status,expected_reason",
    [(k, v[0], v[1]) for k, v in sorted(EXTRACTOR_OUTCOMES.items())])
def test_every_extractor_outcome_maps_to_its_documented_status(
        outcome, expected_status, expected_reason, config_dir,
        bundle_minimal_dir, tmp_path):
    """The same pinning for the extractor's vocabulary, with the reviewer off
    so the extractor mapping is what decides. `tool_cap_hit` is the one
    non-terminal outcome: it PAUSES, leaving the session resumable."""
    out = tmp_path / "runs"
    orch = _prepared(config_dir, bundle_minimal_dir, out,
                     extractor_outcome=outcome, final_review=False)

    status = orch.run()

    assert status == expected_status
    assert orch.session.meta.get("failure_reason") == expected_reason
    if outcome == "tool_cap_hit":
        # A pause is not a finalisation: the session stays in_progress and
        # records why, so --resume can raise the cap and continue.
        assert orch.session.meta["status"] == "in_progress"
        assert orch.session.meta["pause_reason"] == "tool_cap_hit"
    else:
        assert orch.session.meta["status"] == expected_status


def test_only_the_reviewers_confirmation_can_finalise_complete(
        config_dir, bundle_minimal_dir, tmp_path):
    """Stated as one assertion over the whole vocabulary, real and invented:
    of every outcome string tested here, exactly `review_mark_complete`
    finalises `complete` when the reviewer is on."""
    out = tmp_path / "runs"
    completed = set()
    for outcome in list(REVIEW_OUTCOMES) + UNKNOWN_REVIEW_OUTCOMES:
        orch = _prepared(config_dir, bundle_minimal_dir, out / outcome,
                         review_outcome=outcome)
        if orch.run() == "complete":
            completed.add(outcome)

    assert completed == {"review_mark_complete"}


# ---------------------------------------------------------------------------
# 3. The outcome set itself, pinned against the source
# ---------------------------------------------------------------------------

def test_review_loop_outcome_set_is_pinned():
    """Adding an outcome to `_review_loop` must break this test.

    The exhaustiveness guard turns an unmapped outcome into `error` at
    RUNTIME, which is the safe answer but a bad way to find out. This finds
    out at the point the outcome is added: the literal strings `_review_loop`
    returns must be exactly the set the mapping knows about, so a new one
    arrives with a decision about which status it produces.
    """
    assert _returned_outcomes(Orchestrator._review_loop,
                              tuple_first=True) == set(REVIEW_OUTCOMES)


def test_extractor_loop_outcome_set_is_pinned():
    """The same, for the extractor's vocabulary."""
    assert _returned_outcomes(Orchestrator._extractor_loop) == set(
        EXTRACTOR_OUTCOMES)


def test_the_extractor_mapping_is_no_wider_than_the_loop():
    """`_finalise_loop_stop` maps exactly what `_extractor_loop` can return.

    A mapping wider than the loop is safe -- it cannot admit an unreviewed run
    -- but it is a branch no run can reach, so it is never exercised and never
    checked against the behaviour it claims. Reading both out of the source
    keeps them in step: the mapped strings are the loop's outcome set minus
    `mark_complete_validated`, the one outcome the caller continues past rather
    than stopping on.
    """
    mapped = {node.comparators[0].value
              for node in ast.walk(
                  ast.parse(inspect.getsource(
                      Orchestrator._finalise_loop_stop).lstrip()))
              if isinstance(node, ast.Compare)
              and isinstance(node.comparators[0], ast.Constant)}
    assert mapped == set(EXTRACTOR_OUTCOMES) - {"mark_complete_validated"}
