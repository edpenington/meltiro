"""Session is the durable record of one extraction run: create, persist,
resume, replay.

A session owns a directory under `runs/` and everything in it. Three contracts
are pinned here.

Persistence. Whatever the loop holds in memory has to survive the process
exiting. run.json carries the study id, the stage fingerprints, the status, the
accounting accumulators, the record-id counters and the code-version anchor;
the append-only event log carries the turns. Anything a resumed run needs and
run.json does not hold is state that a crash silently loses, so the tests here
assert on the bytes ON DISK rather than on the live object.

Resume refusal. A session is resumable ONLY when the config that produced it
still matches: `expected_config_fp`, `expected_checker_fp` and
`expected_review_fp` are each checked, and a mismatch in any one raises
`ResumeRefused` naming the stage that drifted. Continuing a conversation under
edited config would attribute the second half of a run to a configuration that
never produced the first half, so refusal is the safe direction. Auto-resume
therefore searches WITH the expected fingerprint: the newest in-progress
session is not necessarily the resumable one.

Replay. `replay_messages` reconstructs the provider conversation from the
event log. The assistant turn is replayed from its recorded `assistant_message`
event verbatim, blocks and ordering intact, rather than rebuilt from the tool
calls: a reconstruction has to guess at ordering, and the guess is a divergence
between what the model was told it said and what it actually said.
"""

import json

import pytest

from meltiro.extraction_record import ExtractionRecord
from meltiro.errors import ResumeRefused, SessionError
from meltiro.session import Session
from meltiro.statuses import TERMINAL_STATUSES


def _create(tmp_path, study_id="376", config_fp="config_fp:abc123def456"):
    return Session.create(
        study_id,
        config_fp=config_fp,
        checker_fp="checker_fp:def",
        review_fp="review_fp:xyz",
        instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
        checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
        engine_fp="engine_fp:eng",
        extractor_model="opus", checker_model="sonnet",
        review_model="opus",
        tool_set_hash="ts", template_hash="th", prompt_hash="ph",
        runs_dir=tmp_path,
    )


class TestCreate:
    def test_creates_directory_and_files(self, tmp_path):
        s = _create(tmp_path)
        assert s.session_dir.exists()
        assert s.extraction_record_path.exists()
        assert s.meta_path.exists()
        assert s.tool_calls_path.exists()

    def test_meta_initial_state(self, tmp_path):
        s = _create(tmp_path)
        assert s.meta["status"] == "in_progress"
        assert s.meta["current_phase"] == "extracting"
        assert s.meta["study_id"] == "376"
        assert s.meta["config_fp"] == "config_fp:abc123def456"
        assert s.meta["session_id"].endswith("abc123")  # short_fp suffix

    def test_meta_records_run_fp_derived_from_the_three_stage_fps(
            self, tmp_path):
        # run.json carries a top-level run_fp identifying the whole
        # run-producing configuration. It is derived from the three stage
        # fingerprints the session is created with, so it can never disagree
        # with the config_fp/checker_fp/review_fp recorded alongside it. Pinned
        # against run_fingerprint on those same values (the three stage fps
        # plus the engine fp), and checked on
        # disk (not just in memory) since that is what a downstream consumer
        # parses to build `llm:<run_fp>`.
        from meltiro.fingerprint import run_fingerprint

        s = _create(tmp_path)
        expected = run_fingerprint(
            s.meta["config_fp"], s.meta["checker_fp"], s.meta["review_fp"],
            s.meta["engine_fp"])
        assert s.meta["run_fp"] == expected
        assert s.meta["run_fp"].startswith("run_fp:")
        with open(s.meta_path, encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["run_fp"] == expected

    def test_meta_run_fp_well_defined_for_an_ablation_with_stages_off(
            self, tmp_path):
        # An extractor-only run (checker and reviewer off, both stage fps null)
        # still gets a well-defined run_fp: the disabled stages fold in as the
        # documented sentinel rather than crashing or collapsing to config_fp.
        from meltiro.fingerprint import run_fingerprint

        s = Session.create(
            "111", config_fp="config_fp:aaa", checker_fp=None, review_fp=None,
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            extractor_model="opus", checker_model=None, review_model=None,
            tool_set_hash="ts", template_hash="th", prompt_hash="ph",
            runs_dir=tmp_path)
        assert s.meta["run_fp"] == run_fingerprint(
            "config_fp:aaa", None, None, "engine_fp:eng")
        # Distinct from the all-stages-on run above, whose extractor config
        # differs; the point here is only that the ablation is well-defined.
        assert s.meta["run_fp"].startswith("run_fp:")

    def test_meta_records_code_version_anchor(self, tmp_path):
        # run.json carries the code-version anchor (short commit + a dirty-tree
        # flag), written to disk at session start so a run's output is tied to
        # the exact code that produced it.
        s = _create(tmp_path)
        assert "git_commit" in s.meta
        assert "git_dirty" in s.meta
        # A tri-state flag: True/False when the tree could be read, None when
        # there is no git checkout to read (an installed copy).
        assert s.meta["git_dirty"] is None or isinstance(
            s.meta["git_dirty"], bool)
        assert s.meta["git_commit"] is None or s.meta["git_commit"].strip()
        # Persisted, not just in-memory, and the persisted copy carries the
        # same values rather than merely the same keys.
        with open(s.meta_path, encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["git_commit"] == s.meta["git_commit"]
        assert on_disk["git_dirty"] == s.meta["git_dirty"]
        assert "git_dirty" in on_disk

    def test_meta_anchor_graceful_when_git_unavailable(
            self, tmp_path, monkeypatch):
        # When git is unavailable, both anchor fields degrade to None rather
        # than raising, matching the run-log's graceful behaviour.
        import subprocess
        from meltiro import run_log

        def _no_git(*a, **k):
            raise FileNotFoundError("git")
        monkeypatch.setattr(subprocess, "run", _no_git)
        s = Session.create(
            "999", config_fp="config_fp:aaa", checker_fp=None, review_fp=None,
            instrument_fp="instrument_fp:inst", extractor_call_fp="call_fp:ext",
            checker_call_fp="call_fp:chk", review_call_fp="call_fp:rev",
            engine_fp="engine_fp:eng",
            extractor_model="opus", checker_model=None, review_model=None,
            tool_set_hash="ts", template_hash="th", prompt_hash="ph",
            runs_dir=tmp_path)
        assert s.meta["git_commit"] is None
        assert s.meta["git_dirty"] is None
        assert run_log.git_state() == (None, None)

    def test_initial_extraction_record_is_empty(self, tmp_path):
        s = _create(tmp_path)
        a = s.load_extraction_record()
        # No fields yet; the study block's reserved scope-note key is present
        # from the start, holding null, so the shape is stable.
        assert a.study == {"notes": None}
        assert a.records == []


class TestPersistence:
    def test_extraction_record_round_trip(self, tmp_path):
        s = _create(tmp_path)
        a = ExtractionRecord()
        a.apply_update_study(
            study={"primary_aim": {"value": "X", "evidence": ["q"],
                                   "source": "Abstract"}},
        )
        s.write_extraction_record(a)
        b = s.load_extraction_record()
        assert b.study["primary_aim"]["value"] == "X"

    def test_append_event_and_read(self, tmp_path):
        s = _create(tmp_path)
        s.append_event({"event": "x", "data": 1})
        s.append_event({"event": "y", "data": 2})
        events = s.read_events()
        # The first event was session_started written by create(); next two
        # are ours.
        assert events[-2]["event"] == "x"
        assert events[-1]["event"] == "y"

    def test_increment_counters(self, tmp_path):
        s = _create(tmp_path)
        s.increment_tool_call_count()
        s.increment_tool_call_count()
        s.record_checker_calls(3)
        assert s.meta["tool_call_count"] == 2
        assert s.meta["checker_calls_run"] == 3

    def test_finalise_writes_terminate_event(self, tmp_path):
        s = _create(tmp_path)
        s.finalise("complete")
        events = s.read_events()
        assert any(e.get("event") == "terminate" and
                   e.get("status") == "complete"
                   for e in events)
        # Status updated on disk.
        with open(s.meta_path, "r") as f:
            m = json.load(f)
        assert m["status"] == "complete"


class TestResume:
    def test_resume_in_progress(self, tmp_path):
        s = _create(tmp_path)
        a = ExtractionRecord()
        a.apply_update_study(study={"x": {"value": "y", "evidence": ["q"],
                                          "source": "Abstract"}})
        s.write_extraction_record(a)
        s.append_event({"event": "x"})

        r = Session.resume(s.session_dir,
                           expected_config_fp="config_fp:abc123def456")
        assert r.meta["status"] == "in_progress"
        assert r.load_extraction_record().study["x"]["value"] == "y"

    @pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
    def test_resume_refuses_every_terminal_status(self, tmp_path, status):
        # Parametrised over the taxonomy rather than over a hand-written list,
        # so a fourth terminal status added to `statuses.py` arrives here
        # already gated. Each of the three means something different and none
        # is resumable: `complete` has its answer, and `failed_validation` and
        # `error` stopped for a reason that continuing would paper over.
        s = _create(tmp_path)
        s.finalise(status)
        with pytest.raises(ResumeRefused) as excinfo:
            Session.resume(s.session_dir)
        # The message names the status found, so an operator reading it knows
        # which of the three they hit without opening run.json.
        assert status in str(excinfo.value)

    def test_only_in_progress_is_admitted(self, tmp_path):
        # The complement of the parametrised refusal above: the gate is an
        # allowlist of one, not a denylist that a new status could slip past.
        s = _create(tmp_path)
        assert s.meta["status"] == "in_progress"
        assert Session.resume(s.session_dir).meta["status"] == "in_progress"

    def test_resume_refuses_config_fp_mismatch(self, tmp_path):
        s = _create(tmp_path)
        with pytest.raises(ResumeRefused):
            Session.resume(s.session_dir, expected_config_fp="different_fp")

    def test_resume_refuses_checker_fp_mismatch(self, tmp_path):
        # A changed checker config blocks resume, not just a changed extractor
        # config: EVERY stage fingerprint is gated, not only config_fp.
        s = _create(tmp_path)
        with pytest.raises(ResumeRefused) as excinfo:
            Session.resume(
                s.session_dir,
                expected_config_fp="config_fp:abc123def456",
                expected_checker_fp="checker_fp:different")
        assert "Checker fingerprint drift" in str(excinfo.value)

    def test_resume_refuses_review_fp_mismatch(self, tmp_path):
        # A changed review config blocks resume on the same rule.
        s = _create(tmp_path)
        with pytest.raises(ResumeRefused) as excinfo:
            Session.resume(
                s.session_dir,
                expected_config_fp="config_fp:abc123def456",
                expected_review_fp="review_fp:different")
        assert "Review fingerprint drift" in str(excinfo.value)

    def test_resume_allows_when_all_fps_match(self, tmp_path):
        # All three stage fingerprints match the ones stored at create time.
        s = _create(tmp_path)
        r = Session.resume(
            s.session_dir,
            expected_config_fp="config_fp:abc123def456",
            expected_checker_fp="checker_fp:def",
            expected_review_fp="review_fp:xyz")
        assert r.meta["status"] == "in_progress"

    def test_resume_missing_meta_raises(self, tmp_path):
        with pytest.raises(ResumeRefused):
            Session.resume(tmp_path / "nope")


class TestRecordIdCounterAcrossResume:
    """A removed record's id is never reissued across a resume.

    The record-id counter is persisted in run.json and threaded back on
    resume, so the number ONLY moves forward. Re-deriving it from the surviving
    records instead, as max(surviving index)+1, is the tempting alternative and
    it is wrong across a removal: add R1, add R2, remove R2, pause, resume,
    add reissues 2, and the reissued record silently inherits every challenge
    held against the record that was removed.

    A gap in the final output is the accepted cost. Ids running R1, R2, R4 are
    correct, because a record id identifies a record rather than counting them.
    """

    @staticmethod
    def _env(value):
        return {"value": value, "evidence": None}

    def test_counter_starts_empty_in_meta(self, tmp_path):
        s = _create(tmp_path)
        assert s.meta["record_id_counters"] == {}

    def test_write_extraction_record_persists_counter_to_meta(self, tmp_path):
        s = _create(tmp_path)
        rec = s.load_extraction_record()
        rec.add_record({"gauge": self._env("A")}, "relationship")
        s.write_extraction_record(rec)
        # The counter is in run.json on disk, never in the extraction output.
        with open(s.meta_path) as f:
            meta = json.load(f)
        with open(s.extraction_record_path) as f:
            output = json.load(f)
        assert meta["record_id_counters"] == {"relationship": 2}
        assert "record_id_counters" not in output
        assert set(output) == {
            "initial_check", "study", "records", "quality_check"}

    def test_removed_id_not_reissued_after_resume(self, tmp_path):
        s = _create(tmp_path)
        rec = s.load_extraction_record()
        r1 = rec.add_record({"gauge": self._env("A")}, "relationship")
        r2 = rec.add_record({"gauge": self._env("B")}, "relationship")
        assert (r1, r2) == ("relationship_1", "relationship_2")
        rec.remove_record("relationship_2")
        # Pause: persist the state (and the counter) as the live loop does.
        s.write_extraction_record(rec)

        # Resume: a fresh Session object reattaches to the same directory.
        r = Session.resume(
            s.session_dir, expected_config_fp="config_fp:abc123def456")
        resumed = r.load_extraction_record()
        assert resumed.record_ids() == ["relationship_1"]
        # The next add must take a fresh number, never refill the removed 2.
        new_id = resumed.add_record({"gauge": self._env("C")}, "relationship")
        assert new_id == "relationship_3"
        assert "relationship_2" not in resumed.record_ids()


class TestFindInProgress:
    def test_finds_most_recent(self, tmp_path):
        s1 = _create(tmp_path)
        # Make a finished session; should be ignored.
        s2 = _create(tmp_path)
        s2.finalise("complete")
        # Make another in-progress one (this is the newest).
        s3 = _create(tmp_path)

        found = Session.find_in_progress(
            "376", runs_dir=tmp_path,
            expected_config_fp="config_fp:abc123def456",
        )
        assert found == s3.session_dir

    def test_returns_none_when_no_match(self, tmp_path):
        s = _create(tmp_path)
        s.finalise("complete")
        assert Session.find_in_progress("376", runs_dir=tmp_path) is None

    def test_config_fp_filter(self, tmp_path):
        s = _create(tmp_path, config_fp="config_fp:aaaaaa111111")
        found = Session.find_in_progress(
            "376", runs_dir=tmp_path,
            expected_config_fp="config_fp:bbbbbb222222",
        )
        assert found is None

    def test_older_matching_session_beats_a_newer_drifted_one(self, tmp_path):
        # The NEWEST in-progress session has a drifted config while an OLDER
        # one still matches. A search with no expected_config_fp sees only the
        # newest session, whose drift then makes resume refuse and the run
        # start fresh, re-spending the budget the older matching session had
        # banked. Auto-resume therefore supplies the fingerprint, and the
        # search returns the older MATCHING session.
        older = _create(tmp_path, config_fp="config_fp:matches")
        newer = _create(tmp_path, config_fp="config_fp:drifted")
        assert newer.session_dir != older.session_dir
        found = Session.find_in_progress(
            "376", runs_dir=tmp_path,
            expected_config_fp="config_fp:matches",
        )
        assert found == older.session_dir
        # Without the filter the search surfaces the newer, drifted session:
        # the fingerprint is what selects, NOT recency alone.
        assert Session.find_in_progress(
            "376", runs_dir=tmp_path) == newer.session_dir


class TestReplayMessages:
    def test_replays_tool_calls_in_order(self, tmp_path):
        s = _create(tmp_path)
        turn1_args = {"study": {"x": {"value": "y", "evidence": ["q"],
                                      "source": "Abstract"}}}
        s.append_event({
            "event": "tool_call_applied",
            "turn_id": 1,
            "tool_use_id": "tu_1",
            "tool": "update_study",
            "args": turn1_args,
            "result": {"status": "ok"},
        })
        s.append_event({
            "event": "assistant_message", "turn_id": 1,
            "content": [{"type": "tool_use", "id": "tu_1",
                         "name": "update_study", "input": turn1_args}],
        })
        s.append_event({
            "event": "tool_call_applied",
            "turn_id": 2,
            "tool_use_id": "tu_2",
            "tool": "mark_complete",
            "args": {},
            "result": {"status": "ok"},
        })
        s.append_event({
            "event": "assistant_message", "turn_id": 2,
            "content": [{"type": "tool_use", "id": "tu_2",
                         "name": "mark_complete", "input": {}}],
        })
        msgs = s.replay_messages()
        # turn 1: assistant tool_use + user tool_result
        # turn 2: same
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"][0]["type"] == "tool_use"
        assert msgs[0]["content"][0]["name"] == "update_study"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"][0]["type"] == "tool_result"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"][0]["name"] == "mark_complete"

    def test_assistant_message_content_is_verbatim(self, tmp_path):
        # Replay uses the turn's assistant_message event as the authoritative,
        # byte-identical assistant content: same blocks, same order.
        s = _create(tmp_path)
        content = [
            {"type": "text", "text": "I'll start by populating the basics."},
            {"type": "tool_use", "id": "tu_1", "name": "update_study",
             "input": {"study": {}}},
        ]
        s.append_event({
            "event": "assistant_message", "turn_id": 1, "content": content,
        })
        s.append_event({
            "event": "tool_call_applied", "turn_id": 1,
            "tool_use_id": "tu_1", "tool": "update_study",
            "args": {"study": {}}, "result": {"status": "ok"},
        })
        msgs = s.replay_messages()
        a = msgs[0]
        assert a["role"] == "assistant"
        assert a["content"] == content
        assert [b["type"] for b in a["content"]] == ["text", "tool_use"]

    def test_assistant_message_preserves_tool_use_before_text(self, tmp_path):
        # A response that put its tool_use BEFORE its text replays in that
        # exact order. Rebuilding the turn from its tool calls would have to
        # pick an order, and picking text-first rewrites what the model said.
        s = _create(tmp_path)
        content = [
            {"type": "tool_use", "id": "tu_1", "name": "view_summary",
             "input": {}},
            {"type": "text", "text": "Now let me explain."},
        ]
        s.append_event({
            "event": "assistant_message", "turn_id": 1, "content": content,
        })
        s.append_event({
            "event": "tool_call_applied", "turn_id": 1,
            "tool_use_id": "tu_1", "tool": "view_summary",
            "args": {}, "result": {"status": "ok"},
        })
        a = s.replay_messages()[0]
        assert [b["type"] for b in a["content"]] == ["tool_use", "text"]

    def test_tool_turn_without_assistant_message_refuses(self, tmp_path):
        # Every tool-calling turn logs an assistant_message (the terminal
        # stall path included), so a turn with tool_call events and no
        # assistant_message can only be a crash artefact: a hard kill between
        # the turn's appends, or a torn assistant_message line dropped by the
        # torn-tail repair. The model's text for that turn is unknowable, so
        # replay refuses loudly rather than silently rebuilding a divergent
        # conversation.
        s = _create(tmp_path)
        s.append_event({
            "event": "tool_call_failed", "turn_id": 1,
            "tool_use_id": "tu_1", "tool": "mark_complete",
            "args": {}, "result": {"status": "validation_failed"},
        })
        with pytest.raises(SessionError) as excinfo:
            s.replay_messages()
        assert "crash artefact" in str(excinfo.value)
        assert "assistant_message" in str(excinfo.value)

    def test_text_turn_without_assistant_message_refuses(self, tmp_path):
        # assistant_message is always written before assistant_text, so a
        # truncated log cannot leave the text without the message. A log in
        # that state was edited or written by something else, and there is no
        # faithful way to recover the original block order, so replay refuses
        # loudly rather than guessing.
        s = _create(tmp_path)
        s.append_event({
            "event": "assistant_text", "turn_id": 1, "text": "reasoning",
        })
        s.append_event({
            "event": "tool_call_applied", "turn_id": 1,
            "tool_use_id": "tu_1", "tool": "update_study",
            "args": {"study": {}}, "result": {"status": "ok"},
        })
        with pytest.raises(SessionError) as excinfo:
            s.replay_messages()
        assert "not written by meltiro" in str(excinfo.value)

    def test_replay_strips_underscore_keys_from_tool_result(self, tmp_path):
        # The event log stores the FULL result dict (transcript record), but
        # replay must feed the model exactly what the live loop sent:
        # underscore-prefixed telemetry (_field_diffs, _canonicalisations)
        # stripped out of the tool_result content.
        s = _create(tmp_path)
        result = {
            "status": "ok",
            "applied_fields": ["primary_aim"],
            "_field_diffs": {"primary_aim": {"before": None, "after": "X"}},
            "_canonicalisations": [],
        }
        s.append_event({
            "event": "assistant_message", "turn_id": 1,
            "content": [{"type": "tool_use", "id": "tu_1",
                         "name": "update_study", "input": {"study": {}}}],
        })
        s.append_event({
            "event": "tool_call_applied", "turn_id": 1,
            "tool_use_id": "tu_1", "tool": "update_study",
            "args": {"study": {}}, "result": result,
        })
        user = s.replay_messages()[1]
        assert user["role"] == "user"
        content = json.loads(user["content"][0]["content"])
        assert content == {"status": "ok", "applied_fields": ["primary_aim"]}
        assert "_field_diffs" not in content
        assert "_canonicalisations" not in content
        # The event log itself keeps the full dict, telemetry included.
        raw = [e for e in s.read_events()
               if e.get("event") == "tool_call_applied"][0]
        assert "_field_diffs" in raw["result"]


class TestTornFinalLine:
    """A hard kill mid-append can truncate the last jsonl line. Resume must
    recover that single torn tail, but treat corruption elsewhere as fatal.

    Safe because the extraction record is persisted AFTER the events of its
    batch, so the persisted record can never lead the event log: dropping the
    unparseable tail cannot leave the record ahead of the replayed
    conversation.
    """

    @staticmethod
    def _append_raw(s, text):
        with open(s.tool_calls_path, "a", encoding="utf-8") as f:
            f.write(text)

    def test_resume_drops_torn_final_line(self, tmp_path):
        s = _create(tmp_path)
        s.append_event({"event": "tool_call_applied", "turn_id": 1,
                        "tool_use_id": "tu_1", "tool": "view_summary",
                        "args": {}, "result": {"status": "ok"}})
        # Simulate a power loss mid-append: a partial JSON line, no newline.
        self._append_raw(s, '{"event": "tool_call_appli')
        # read_events would choke on the torn line before repair...
        with pytest.raises(SessionError):
            s.read_events()
        # ...but resume repairs it and succeeds.
        r = Session.resume(s.session_dir,
                           expected_config_fp="config_fp:abc123def456")
        events = r.read_events()
        assert any(e.get("tool_use_id") == "tu_1" for e in events)
        # A torn_line_dropped event and a meta warning record what happened.
        dropped = [e for e in events if e.get("event") == "torn_line_dropped"]
        assert len(dropped) == 1
        assert dropped[0]["raw"] == '{"event": "tool_call_appli'
        assert dropped[0]["byte_length"] == len('{"event": "tool_call_appli')
        assert any("torn final line" in w for w in r.meta["warnings"])

    def test_repaired_log_accepts_further_appends(self, tmp_path):
        # After the torn tail is physically removed, the resumed run can keep
        # appending without turning the torn bytes into a mid-file malformed
        # line on the next read.
        s = _create(tmp_path)
        s.append_event({"event": "x", "turn_id": 1})
        self._append_raw(s, '{"event": "torn')
        r = Session.resume(s.session_dir,
                           expected_config_fp="config_fp:abc123def456")
        r.append_event({"event": "y", "turn_id": 2})
        events = [e["event"] for e in r.read_events()
                  if e.get("event") in ("x", "y")]
        assert events == ["x", "y"]

    def test_malformed_middle_line_stays_fatal(self, tmp_path):
        # A malformed line with valid content AFTER it is not a torn tail; it
        # is corruption, and read_events (used by resume) fails loudly, naming
        # the file and line.
        s = _create(tmp_path)
        self._append_raw(s, 'not json at all\n')
        s.append_event({"event": "after", "turn_id": 1})
        r = Session.resume(s.session_dir,
                           expected_config_fp="config_fp:abc123def456")
        with pytest.raises(SessionError) as excinfo:
            r.read_events()
        msg = str(excinfo.value)
        assert "Malformed JSON" in msg
        assert str(s.tool_calls_path) in msg

    def test_repair_preserves_valid_lines_byte_for_byte(self, tmp_path):
        s = _create(tmp_path)
        s.append_event({"event": "a", "turn_id": 1})
        s.append_event({"event": "b", "turn_id": 2})
        good = s.tool_calls_path.read_text(encoding="utf-8")
        self._append_raw(s, '{"event": "torn')
        Session.resume(s.session_dir,
                       expected_config_fp="config_fp:abc123def456")
        after = s.tool_calls_path.read_text(encoding="utf-8")
        # The good prefix is byte-identical; only the torn_line_dropped audit
        # event was appended after it.
        assert after.startswith(good)
        assert '"event": "torn_line_dropped"' in after[len(good):]

    def test_final_line_missing_only_newline_is_normalised(self, tmp_path):
        # A torn write that kept the whole final event but lost its trailing
        # newline loses no data. Resume restores the newline (no
        # torn_line_dropped, no warning) so the next append does not
        # concatenate onto the unterminated line and corrupt it.
        s = _create(tmp_path)
        s.append_event({"event": "a", "turn_id": 1})
        # Append a COMPLETE event but without its terminating newline.
        self._append_raw(s, json.dumps({"event": "b", "turn_id": 2}))
        assert not s.tool_calls_path.read_text(encoding="utf-8").endswith("\n")
        r = Session.resume(s.session_dir,
                           expected_config_fp="config_fp:abc123def456")
        assert s.tool_calls_path.read_text(encoding="utf-8").endswith("\n")
        r.append_event({"event": "c", "turn_id": 3})
        events = [e["event"] for e in r.read_events()
                  if e.get("event") in ("a", "b", "c")]
        assert events == ["a", "b", "c"]
        assert not any(e.get("event") == "torn_line_dropped"
                       for e in r.read_events())
        assert r.meta.get("warnings", []) == []

    def test_clean_log_is_untouched(self, tmp_path):
        # A well-formed log (last line parses) is not rewritten and gains no
        # spurious torn_line_dropped event on resume.
        s = _create(tmp_path)
        s.append_event({"event": "a", "turn_id": 1})
        before = s.tool_calls_path.read_text(encoding="utf-8")
        r = Session.resume(s.session_dir,
                           expected_config_fp="config_fp:abc123def456")
        assert s.tool_calls_path.read_text(encoding="utf-8") == before
        assert not any(e.get("event") == "torn_line_dropped"
                       for e in r.read_events())


class TestStatusValidation:
    def test_unknown_status_raises(self, tmp_path):
        s = _create(tmp_path)
        with pytest.raises(SessionError):
            s.finalise("not_a_real_status")

    def test_unknown_phase_raises(self, tmp_path):
        s = _create(tmp_path)
        with pytest.raises(SessionError):
            s.set_phase("nonsense")
