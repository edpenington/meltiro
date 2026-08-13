"""The wire log is a record of the call, not the call's outcome.

`--diagnostics full` keeps a verbatim entry per provider call under the
session's `diagnostics/`. That write is the one step of a turn that can fail
for a reason having nothing to do with the turn: the disk filled, the
directory lost its permission. The call itself has already happened and
already been paid for, so a fault there is swallowed at every site that writes
one — `Orchestrator._log_api_call_guarded` for the three the orchestrator
makes, `checker._log_ask` for the checker's.

Unguarded, the common path is what breaks: an ordinary successful turn ends a
run as `error` over a document nobody needs to resume it. What the guard must
NOT swallow is the accounting, which is why the refused-call ledger runs its
entry first and its accumulation after, outside the guard: the tokens are the
part no other record can be rebuilt from.

The guard is around the WRITE alone, and the tests here hold it to both edges.
Composing the entry is not the write: an unreadable diagnostics level or a
shape `make_entry` cannot compose is a defect in the run, it holds for every
call the run makes, and swallowed it leaves an empty wire log for good. And a
write that does fail is reported — the file is one line per call and carries
no count, so nothing else in the session would say a call is missing from it.

Offline: a real Session backs a real Orchestrator, the provider calls are stub
adapters returning constructed responses, and no network is reached.
"""

import json
from types import SimpleNamespace

import pytest

from direktoro import (
    NormalisedResponse, NormalisedUsage, ProviderRouteMismatch)
from direktoro.registry import OPENROUTER_BASE_URL
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import SessionError
from meltiro.orchestrator import Orchestrator
from meltiro.tools import get_tool_definitions

pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-8"
ROUTED = "z-ai/glm-5v-turbo"

# The shipped template's one REQUIRED quality-check variable.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


def _response(*blocks, model=EXTRACTOR, input_tokens=800, output_tokens=60):
    return NormalisedResponse(
        content=list(blocks),
        usage=NormalisedUsage(input_tokens=input_tokens,
                              output_tokens=output_tokens),
        resolved_model=model, provider="anthropic", base_url=None,
        raw_request={"model": model}, raw_response={},
        wire_request={"model": model}, decoding_params={"max_tokens": 4096},
        stop_reason="tool_use")


def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


class _CompletingAdapter:
    """Returns one scripted response per call, and marks the extraction
    complete on the way.

    `view_summary` is a read that clears no completion flag, so setting it
    here is what lets a single synthetic turn end a run cleanly without a full
    valid extraction — the same shape `test_cost_rates` runs to `complete`.
    """

    def __init__(self, orch, response):
        self._orch = orch
        self._response = response

    def create_message(self, **kwargs):
        self._orch.extraction_record.mark_complete()
        return self._response


class _RefusingAdapter:
    """An adapter whose call is served, billed, and then refused."""

    def __init__(self, exc):
        self._exc = exc

    def create_message(self, **kwargs):
        raise self._exc


def _orch(config_dir, bundle_dir, out_dir, *, review_model=None,
          extractor_model=EXTRACTOR):
    """An orchestrator keeping the wire log: the level the guard is about."""
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=extractor_model,
        checker_config=CheckerConfig(max_tokens=1024, checker_model=CHECKER),
        review_model=review_model,
        max_checks_per_field=0, final_review=review_model is not None,
        max_tool_calls=50, extractor_max_tokens=4096, review_max_tokens=4096,
        diagnostics="full",
    )
    orch.prepare_new_session()
    return orch


def _break_wire_log(orch, monkeypatch):
    """Make the wire-log write fail the way a full disk would.

    At the write seam, because that is where a disk fault lands: the entry is
    composed first and outside the guard, and `Session.log_api_call` — both
    halves at once — is not on the guarded path.
    """
    def _boom(*args, **kwargs):
        raise OSError("no space left on device")
    monkeypatch.setattr(orch.session, "write_api_call_entry", _boom)


# ---------------------------------------------------------------------------
# The ordinary successful turn
# ---------------------------------------------------------------------------

def test_a_failing_wire_log_leaves_an_extractor_run_complete(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    """The common path, and the one an unguarded write costs most: a turn that
    was served, answered, and billed, whose only fault is the document written
    about it."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out)
    adapter = _CompletingAdapter(
        orch, _response(_tool_use("t1", "view_summary", {})))
    orch._adapter_for_role = lambda role: adapter
    _break_wire_log(orch, monkeypatch)

    assert orch.run() == "complete"

    # The turn happened and is paid for, so its tokens are the run's.
    assert orch._input_tokens == 800
    assert orch._output_tokens == 60
    meta = json.loads(orch.session.meta_path.read_text())
    assert meta["status"] == "complete"
    assert meta["input_tokens"] == 800
    # And the finished artefact says which record it is missing. Swallowed
    # silently, the empty wire log reads as a run that made no calls.
    warning, = [w for w in meta["warnings"]
                if "wire-log entry could not be written" in w]
    assert "extractor" in warning
    assert "no space left on device" in warning


def test_a_failing_wire_log_leaves_a_review_run_complete(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    """The reviewer's calls are written by the same guard: a fault on a review
    turn's entry is not a verdict on the extraction."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, review_model=REVIEWER)

    def _extractor():
        orch.extraction_record.mark_complete()
        return "mark_complete_validated"

    orch._extractor_loop = _extractor
    adapter = _CompletingAdapter(orch, _response(
        _tool_use("t1", "mark_complete",
                  {"summary": "reviewed",
                   "quality_check": dict(QUALITY_CHECK)}),
        model=REVIEWER))
    orch._adapter_for_role = lambda role: adapter
    _break_wire_log(orch, monkeypatch)

    assert orch.run() == "complete"
    assert orch._usage_by_role_record()["review"]["input_tokens"] == 800


# ---------------------------------------------------------------------------
# The refused call, whose entry and whose accounting are not the same step
# ---------------------------------------------------------------------------

def test_a_refused_calls_entry_is_written_before_its_spend_is_priced(
        config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    """`_accumulate_usage` prices the call, and pricing is fallible: an
    unknown model id, a routed receipt that will not read. The wire entry is
    the record of a call that reached the provider, so it is written first and
    survives that."""
    out = tmp_path / "runs"
    orch = _orch(config_dir, bundle_minimal_dir, out, extractor_model=ROUTED)
    exc = ProviderRouteMismatch(
        "routed response was served by 'Novita', not the pinned ('Z.AI',)")
    exc.response = NormalisedResponse(
        content=[], usage=NormalisedUsage(input_tokens=900, output_tokens=40),
        resolved_model=ROUTED, provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        raw_request={"model": ROUTED}, raw_response={"model": ROUTED},
        wire_request={"model": ROUTED, "messages": []},
        decoding_params={"max_tokens": 1024})

    def _boom(*args, **kwargs):
        raise ValueError(f"unknown model: {ROUTED}")
    monkeypatch.setattr(orch, "_accumulate_usage", _boom)

    with pytest.raises(ValueError):
        orch._call_extractor(_RefusingAdapter(exc),
                             get_tool_definitions(orch.template))

    entry, = [json.loads(line) for line
              in orch.session.api_calls_path.read_text().splitlines()]
    assert entry["call_type"] == "extractor"
    assert entry["wire_model"] == ROUTED


# ---------------------------------------------------------------------------
# What the guard is not: composing the entry
# ---------------------------------------------------------------------------

def test_an_unreadable_diagnostics_level_is_not_swallowed(
        config_dir, bundle_minimal_dir, tmp_path):
    """`validate_diagnostics` exists so an unknown level can never reach a run
    and quietly behave like the default. Swallowed here it would do the
    opposite: every call of the run unlogged, and the level that asked for the
    wire log never questioned."""
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
    orch.session.meta["diagnostics"] = "verbose"

    with pytest.raises(SessionError, match="Unknown diagnostics level"):
        orch._log_api_call_guarded(
            "extractor", {"model": EXTRACTOR, "messages": []}, {})


def test_a_shape_the_entry_cannot_be_composed_from_is_not_swallowed(
        config_dir, bundle_minimal_dir, tmp_path):
    """A call handed a request that is not a request is a bug in the caller,
    and it holds for every call that caller makes. Swallowed, it leaves an
    `api_calls.jsonl` that was never created — at `--diagnostics full` the
    reading of a run that made no calls — with nothing anywhere saying why."""
    orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")

    with pytest.raises(TypeError):
        orch._log_api_call_guarded("extractor", None, {})

    assert not orch.session.api_calls_path.exists()
