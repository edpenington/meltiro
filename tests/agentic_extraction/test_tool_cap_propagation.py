"""A hard stop from the extractor loop must propagate up to the final terminal
action, before the final-review branch.

An extractor that over-extracts and exhausts its tool calls leaves an
inconsistent extraction output behind. Dropping from there into the final
review finalises `complete` when the run should have paused, so the loop stop
is routed through `_finalise_loop_stop`, which `run` must consult (and act on)
before the review runs at all.
"""

import inspect

from meltiro.orchestrator import Orchestrator


def test_run_routes_the_loop_stop_before_the_review():
    """Inspect Orchestrator.run source: the extractor outcome is routed through
    `_finalise_loop_stop` (which handles the cap-hit pause, surrender, and
    stall), and that routing happens BEFORE the final-review branch, so a run
    left mid-extraction is never reviewed."""
    src = inspect.getsource(Orchestrator.run)
    assert "self._finalise_loop_stop(extractor_status)" in src
    stop_idx = src.find("self._finalise_loop_stop(extractor_status)")
    review_idx = src.find("if self.final_review:")
    assert stop_idx > 0
    assert review_idx > 0
    assert stop_idx < review_idx


def test_finalise_loop_stop_maps_every_hard_stop():
    """The stop-router maps each hard-stop loop outcome, and returns None for a
    non-stop outcome so the caller continues past it."""
    src = inspect.getsource(Orchestrator._finalise_loop_stop)
    # tool_cap_hit pauses (in_progress); the others finalise terminally.
    assert '"tool_cap_hit"' in src and "self._pause(" in src
    assert '"text_only_stall"' in src
    assert '"extractor_stalled"' in src
    assert '"extractor_abandoned"' in src
    assert "failed_validation" in src
    # A non-stop outcome returns None (the caller falls through).
    assert "return None" in src
