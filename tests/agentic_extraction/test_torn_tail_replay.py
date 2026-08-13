"""A replayed conversation ends on a user message, whatever the kill took.

A text-only extractor turn writes two sides: the model's words, then the
re-prompt sent back. Killed between them, the log holds a turn with an
assistant side and no user side — and a replay of that ends on an assistant
message, which the provider reads as a PREFILL. The model is then asked to
continue its own narration rather than answer, and a prefill ending in
whitespace is refused outright with a 400, so the resume the pause exists to
protect fails on its first call.

Two halves, both pinned here: the window is one append wide (the re-prompt
event goes out immediately after the assistant events, before any meta write),
and a replay over a log torn inside it drops the dangling turn and re-sends it
cleanly.

Offline: a real Session over a hand-written event log, plus a real Orchestrator
whose extractor turn is stubbed.
"""

import inspect
import json

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator

pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"


def _orch(config_dir, bundle_dir, out_dir, *, cap=50):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024,
                                     checker_model="claude-sonnet-4-6"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=cap, extractor_max_tokens=4096,
    )


def _events(session):
    return [json.loads(line) for line in
            session.tool_calls_path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The window is one append wide
# ---------------------------------------------------------------------------

class TestTheReprompotEventGoesOutWithItsTurn:

    def test_it_is_appended_before_any_meta_write(self):
        # Read off the source: the ORDER is the property, and it is one a
        # behavioural test cannot see (both events are present either way).
        source = inspect.getsource(Orchestrator._extractor_loop)
        # The CONTINUING branch: everything after the stall path returns. The
        # stall's own meta write is on a path that never re-prompts, so it says
        # nothing about this ordering.
        continuing = source.split('return "text_only_stall"', 1)[1]
        reprompt_at = continuing.find('"event": "extractor_reprompt"')
        write_at = continuing.find("self.session.write_meta()")
        assert reprompt_at > 0 and write_at > 0
        assert reprompt_at < write_at, (
            "the re-prompt event must be appended before the meta write: "
            "every write between the assistant side and the user side widens "
            "the window in which a kill leaves an unsendable conversation")

    def test_a_text_only_turn_logs_both_sides_in_order(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        from types import SimpleNamespace
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        orch.prepare_new_session()
        orch._adapter_for_role = lambda role: object()
        orch._call_extractor = lambda a, t: SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Thinking...")],
            stop_reason="end_turn")
        orch.run()
        capsys.readouterr()

        names = [e["event"] for e in _events(orch.session)
                 if e.get("turn_id") == 1]
        assert names[:3] == ["assistant_message", "assistant_text",
                             "extractor_reprompt"]


# ---------------------------------------------------------------------------
# A log torn inside that window still replays
# ---------------------------------------------------------------------------

class TestATornTailReplaysCleanly:

    def _paused_after_two_text_turns(self, config_dir, bundle_dir, out):
        """A session paused with two completed text-only turns behind it."""
        from types import SimpleNamespace
        orch = _orch(config_dir, bundle_dir, out, cap=1)
        orch.prepare_new_session()
        orch._adapter_for_role = lambda role: object()
        orch._call_extractor = lambda a, t: SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Still reading. ")],
            stop_reason="end_turn")
        # The text-only bound fires at 3 turns, so this run stalls; what it
        # leaves behind is the event log these tests tear.
        orch.run()
        return orch.session

    def _truncate_after(self, session, event_name):
        """Drop every line after the FIRST `event_name`, as a kill would."""
        lines = session.tool_calls_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if json.loads(line).get("event") == event_name:
                kept = lines[:i + 1]
                break
        else:
            raise AssertionError(f"no {event_name} event in the log")
        session.tool_calls_path.write_text("\n".join(kept) + "\n")

    def test_a_log_torn_after_assistant_text_ends_on_a_user_message(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        out = tmp_path / "runs"
        session = self._paused_after_two_text_turns(
            config_dir, bundle_minimal_dir, out)
        capsys.readouterr()
        # Tear the log at exactly the window this is about: the assistant side
        # of a turn is on disk and its re-prompt never got there.
        self._truncate_after(session, "assistant_text")

        # Replay through a fresh Session over the torn log, as a resume does.
        from meltiro.session import Session
        replayed = Session(session.session_dir,
                           dict(session.meta)).replay_messages()

        # The dangling turn is dropped whole, so there is nothing left to
        # replay and the extractor re-sends its first turn cleanly.
        assert replayed == []

    def test_a_later_tear_keeps_the_completed_turns_before_it(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        from meltiro.session import Session
        out = tmp_path / "runs"
        session = self._paused_after_two_text_turns(
            config_dir, bundle_minimal_dir, out)
        capsys.readouterr()

        # Keep turn 1 whole and tear turn 2 in the window.
        lines = session.tool_calls_path.read_text().splitlines()
        kept = []
        seen_reprompt = 0
        for line in lines:
            event = json.loads(line)
            kept.append(line)
            if event.get("event") == "extractor_reprompt":
                seen_reprompt += 1
            if seen_reprompt == 1 and event.get("event") == "assistant_text" \
                    and event.get("turn_id") == 2:
                break
        session.tool_calls_path.write_text("\n".join(kept) + "\n")

        replayed = Session(session.session_dir,
                           dict(session.meta)).replay_messages()

        # Turn 1 survives whole; turn 2's dangling half is gone.
        assert [m["role"] for m in replayed] == ["assistant", "user"]
        assert replayed[-1]["content"][0]["type"] == "text"

    def test_an_untorn_log_replays_every_turn(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # The pair: the drop is keyed on a MISSING user side, so a complete
        # log loses nothing.
        from meltiro.session import Session
        out = tmp_path / "runs"
        session = self._paused_after_two_text_turns(
            config_dir, bundle_minimal_dir, out)
        capsys.readouterr()
        replayed = Session(session.session_dir,
                           dict(session.meta)).replay_messages()
        roles = [m["role"] for m in replayed]
        assert roles == ["assistant", "user"] * (len(roles) // 2)
        assert roles[-1] == "user"
