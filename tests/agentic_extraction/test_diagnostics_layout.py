"""The session layout, and the `--diagnostics` levels that populate it.

A session directory separates the two things a run produces: the extraction
output, alone at the top, and every deterministic record of how it was
produced, under `diagnostics/`. `--diagnostics` chooses how much of that
record is kept, and the levels are strict supersets of one another.

The load-bearing claim these tests exist to hold is that NOTHING a level omits
can be needed to resume a session or to regenerate a derived artefact. So the
pause-and-resume cycle below is parametrised across all three levels: the
minimum really is sufficient to resume.

Everything here is offline: a real Session, dispatcher, and extraction record
back the orchestrator; the provider adapter and `_call_extractor` are stubbed.
"""

import json
from types import SimpleNamespace

import pytest

from direktoro import NormalisedUsage
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.diagnostics import DIAGNOSTICS_LEVELS
from meltiro.errors import SessionError
from meltiro.orchestrator import Orchestrator
from meltiro.session import Session

EXTRACTOR = "claude-opus-4-8"

# Every file the instrument capture can write. All three stage system
# prompts are captured, not just the extractor's: the transcript's instrument
# section prints each once, and at `standard` (the default) the capture is the
# only place any of them survives.
INSTRUMENT_FILES = {"system_prompt.txt", "user_prompt.txt",
                    "review_system_prompt.txt", "checker_system_prompt.txt",
                    "tool_definitions.json", "image_labels.json"}


def _create(tmp_path, **over):
    kwargs = dict(
        config_fp="config_fp:abc123def456",
        checker_fp="checker_fp:def", review_fp="review_fp:xyz",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        extractor_model="opus", checker_model="sonnet", review_model="opus",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        tool_definitions=[{"name": "update_study"}],
        system_prompt="SYSTEM", user_prompt="USER",
        review_system_prompt="REVIEW", checker_system_prompt="CHECKER",
        image_labels=["fig_01"],
        runs_dir=tmp_path,
    )
    kwargs.update(over)
    return Session.create("376", **kwargs)


class TestLayout:
    def test_extraction_output_is_alone_at_the_top_level(self, tmp_path):
        # The whole point of the split: a reader opening a session sees the
        # result, and one directory holding everything about how it was made.
        s = _create(tmp_path)
        assert sorted(p.name for p in s.session_dir.iterdir()) == [
            "diagnostics", "extraction_output.json"]

    def test_diagnostics_directory_holds_the_rest(self, tmp_path):
        s = _create(tmp_path)
        assert sorted(p.name for p in s.diagnostics_dir.iterdir()) == [
            "instrument", "run.json", "tool_calls.jsonl"]
        assert s.meta_path == s.session_dir / "diagnostics" / "run.json"
        assert s.tool_calls_path.parent == s.diagnostics_dir
        assert s.api_calls_path.parent == s.diagnostics_dir
        assert s.field_history_path.parent == s.diagnostics_dir

    def test_instrument_files_land_under_instrument(self, tmp_path):
        s = _create(tmp_path)
        assert {p.name for p in s.instrument_dir.iterdir()} == INSTRUMENT_FILES
        assert (s.instrument_dir / "system_prompt.txt").read_text() == "SYSTEM"
        assert (s.instrument_dir / "user_prompt.txt").read_text() == "USER"
        assert (s.instrument_dir /
                "review_system_prompt.txt").read_text() == "REVIEW"
        assert (s.instrument_dir /
                "checker_system_prompt.txt").read_text() == "CHECKER"
        assert json.loads(
            (s.instrument_dir / "image_labels.json").read_text()) == ["fig_01"]

    def test_meta_path_for_names_one_place_only(self, tmp_path):
        # Forward-only: the helper resume and auto-resume use looks in exactly
        # one place, so nothing can silently pick up a pre-diagnostics session.
        s = _create(tmp_path)
        assert Session.meta_path_for(s.session_dir) == s.meta_path


class TestLevels:
    @pytest.mark.parametrize("level", DIAGNOSTICS_LEVELS)
    def test_every_level_writes_the_minimum(self, tmp_path, level):
        # extraction_output.json, run.json and tool_calls.jsonl exist at every
        # level. field_history.json is derived at a stop, so it is asserted in
        # the run tests below rather than here.
        s = _create(tmp_path / level, diagnostics=level)
        assert s.extraction_record_path.exists()
        assert s.meta_path.exists()
        assert s.tool_calls_path.exists()

    def test_minimal_omits_the_instrument(self, tmp_path):
        s = _create(tmp_path, diagnostics="minimal")
        assert not s.instrument_dir.exists()
        assert sorted(p.name for p in s.diagnostics_dir.iterdir()) == [
            "run.json", "tool_calls.jsonl"]

    @pytest.mark.parametrize("level", ("standard", "full"))
    def test_standard_and_full_capture_the_instrument(self, tmp_path, level):
        s = _create(tmp_path / level, diagnostics=level)
        assert {p.name for p in s.instrument_dir.iterdir()} == INSTRUMENT_FILES

    @pytest.mark.parametrize("level", ("minimal", "standard"))
    def test_only_full_keeps_the_wire_log(self, tmp_path, level):
        below = _create(tmp_path / level, diagnostics=level)
        below.log_api_call("extractor", {"model": "m", "messages": []},
                           SimpleNamespace(id="msg_1", content=[]))
        # Absent, not empty: an empty file would read as a run that made no
        # calls, which is a different and false claim.
        assert not below.api_calls_path.exists()

    def test_full_keeps_the_wire_log(self, tmp_path):
        s = _create(tmp_path, diagnostics="full")
        s.log_api_call("extractor", {"model": "m", "messages": []},
                       SimpleNamespace(id="msg_1", content=[]))
        lines = s.api_calls_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["call_type"] == "extractor"

    @pytest.mark.parametrize("level", DIAGNOSTICS_LEVELS)
    def test_the_level_is_recorded_in_run_json(self, tmp_path, level):
        # A reader of a finished session must be able to tell "this file was
        # never written" from "this file was lost".
        s = _create(tmp_path / level, diagnostics=level)
        assert s.meta["diagnostics"] == level
        assert json.loads(s.meta_path.read_text())["diagnostics"] == level
        assert s.diagnostics == level

    def test_an_unknown_level_fails_loudly(self, tmp_path):
        with pytest.raises(SessionError, match="Unknown diagnostics level"):
            _create(tmp_path, diagnostics="verbose")

    def test_a_session_recording_no_level_cannot_be_resumed(self, tmp_path):
        # Forward-only: a session directory whose run.json does not say what it
        # kept is not one this version wrote. Guessing a level would silently
        # decide which artefacts the rest of the run produces.
        s = _create(tmp_path)
        meta = json.loads(s.meta_path.read_text())
        del meta["diagnostics"]
        s.meta_path.write_text(json.dumps(meta))
        with pytest.raises(SessionError, match="Unknown diagnostics level"):
            Session.resume(s.session_dir)


# ---------------------------------------------------------------------------
# A real pause-and-resume cycle at every level
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir, *, cap, diagnostics):
    """Extractor-only Orchestrator (checker and reviewer off)."""
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model="claude-sonnet-4-6",
                                     api_key="x"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=cap, diagnostics=diagnostics,
        extractor_max_tokens=4096,
        api_key="x",
    )


def _view_summary_turn(tool_id):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id,
                                 name="view_summary", input={})],
        usage=NormalisedUsage(input_tokens=10, output_tokens=5),
    )


@pytest.mark.parametrize("level", DIAGNOSTICS_LEVELS)
def test_a_session_resumes_and_completes_at_every_level(
        config_dir, bundle_minimal_dir, tmp_path, level):
    """The minimum really is sufficient to resume.

    Replay rebuilds the conversation from `tool_calls.jsonl`, which every
    level keeps, so a `minimal` session pauses on the tool-call cap, resumes
    into the same conversation, and finalises exactly as a `full` one does.
    """
    out_dir = tmp_path / "runs"
    orch1 = _orch(config_dir, bundle_minimal_dir, out_dir, cap=1,
                  diagnostics=level)
    orch1.prepare_new_session()
    orch1._adapter_for_role = lambda role: object()

    def _fake1(adapter, tool_defs):
        resp = _view_summary_turn("s1")
        orch1._accumulate_usage(resp, EXTRACTOR, "extractor")
        return resp

    orch1._call_extractor = _fake1
    assert orch1.run() == "in_progress"

    session_dir = orch1.session.session_dir
    # A pause writes the derived field history too, so a paused session on
    # disk is complete at every level.
    assert (session_dir / "diagnostics" / "field_history.json").exists()

    orch2 = _orch(config_dir, bundle_minimal_dir, out_dir, cap=1,
                  diagnostics=level)
    orch2.resume_session(session_dir)
    # The conversation was rebuilt from the event log alone: the paused turn
    # is back in the messages the next call would send.
    assert any(m["role"] == "assistant" for m in orch2.messages)
    orch2.max_tool_calls = 50
    orch2._adapter_for_role = lambda role: object()

    def _fake2(adapter, tool_defs):
        orch2.extraction_record.mark_complete()
        resp = _view_summary_turn("s2")
        orch2._accumulate_usage(resp, EXTRACTOR, "extractor")
        return resp

    orch2._call_extractor = _fake2
    assert orch2.run() == "complete"
    assert orch2.session.meta["tool_call_count"] == 2

    diagnostics_dir = session_dir / "diagnostics"
    present = {p.name for p in diagnostics_dir.iterdir()}
    assert {"run.json", "field_history.json", "tool_calls.jsonl"} <= present
    assert ("instrument" in present) is (level in ("standard", "full"))
    # Nothing stubs the adapter's logging here, so the wire log is written by
    # no call in this test; what matters is that the level that omits it still
    # reached `complete`.
    assert "api_calls.jsonl" not in present


def test_a_resume_records_the_level_it_ran_under(
        config_dir, bundle_minimal_dir, tmp_path):
    """The level is operational, on exactly the terms `caps` are: a resume may
    change it, run.json reports the CURRENT segment, and the `resumed` event
    is the per-segment history."""
    out_dir = tmp_path / "runs"
    orch1 = _orch(config_dir, bundle_minimal_dir, out_dir, cap=1,
                  diagnostics="minimal")
    orch1.prepare_new_session()
    orch1._adapter_for_role = lambda role: object()
    orch1._call_extractor = lambda adapter, tool_defs: _view_summary_turn("s1")
    assert orch1.run() == "in_progress"
    session_dir = orch1.session.session_dir

    orch2 = _orch(config_dir, bundle_minimal_dir, out_dir, cap=1,
                  diagnostics="full")
    orch2.resume_session(session_dir)
    assert orch2.session.meta["diagnostics"] == "full"
    assert json.loads(
        orch2.session.meta_path.read_text())["diagnostics"] == "full"

    resumed = [e for e in orch2.session.read_events()
               if e.get("event") == "resumed"]
    assert len(resumed) == 1
    assert resumed[0]["diagnostics"] == "full"
    assert resumed[0]["previous_diagnostics"] == "minimal"
    # Raising the level does not backfill the instrument: it is captured once,
    # at session creation, and re-rendering it later would be a reconstruction
    # dressed as a capture.
    assert not orch2.session.instrument_dir.exists()


def test_the_level_moves_no_fingerprint(
        config_dir, bundle_minimal_dir, tmp_path):
    """The level changes which diagnostics files are written and nothing about
    what any model is asked, so it must ride in no fingerprint. If it ever
    did, two runs of one config at different levels would stop comparing."""
    fps = [
        _orch(config_dir, bundle_minimal_dir, tmp_path / level, cap=5,
              diagnostics=level)._build_fingerprints()
        for level in DIAGNOSTICS_LEVELS
    ]
    keys = ("config_fp", "checker_fp", "review_fp", "run_fp", "prompt_hash",
            "tool_hash")
    for key in keys:
        assert len({f[key] for f in fps}) == 1, key
