"""Extractor-loop termination, honest tool-call labels, dropped-text repair,
and collision-free replay turn ids.

These drive `_extractor_loop` offline: a real Session and ToolDispatcher back
the orchestrator, but the provider adapter is stubbed and `_call_extractor`
returns canned responses, so nothing touches the network.
"""

import dataclasses
from types import SimpleNamespace

from meltiro.bundle import load_bundle
from meltiro.config_bundle import load_config_bundle
from meltiro.extraction_record import ExtractionRecord
from meltiro.orchestrator import Orchestrator
from meltiro.session import Session
from meltiro.tools import ToolDispatcher


# ---------------------------------------------------------------------------
# Fake API response blocks
# ---------------------------------------------------------------------------

def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _resp(*blocks):
    return SimpleNamespace(content=list(blocks))


def _bare_loop_orch(tmp_path, template, paper_text, image_labels, *,
                    max_tool_calls=100, max_text_only=3,
                    max_identical_failures=5):
    """A real Orchestrator with __init__ bypassed, wired with a real Session
    and dispatcher but a stubbed client, so `_extractor_loop` runs offline."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.template = template
    record = ExtractionRecord()
    # The extractor's initial-check ordering gate is open from the start here.
    # These tests are about the loop's control flow (termination, labels,
    # replay), and every scripted call would otherwise be refused with the same
    # `initial_check_required` error, which would make each one fail for a
    # reason it is not about. The gate itself is exercised in test_tools.py.
    record.initial_check_recorded = True
    orch.extraction_record = record
    orch.dispatcher = ToolDispatcher(record, template, paper_text, image_labels)
    orch.session = Session.create(
        "demo-001",
        config_fp="config_fp:abcabcabcabc",
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
    orch.messages = [{"role": "user",
                      "content": [{"type": "text", "text": "start"}]}]
    orch.max_tool_calls = max_tool_calls
    orch._mark_complete_has_fired = False
    orch.max_consecutive_text_only_turns = max_text_only
    orch.max_consecutive_identical_failures = max_identical_failures
    orch._turn_counter = 0
    # The inline checker is off for these tests: they exercise the loop's
    # control flow, and a checker call would be a network call.
    orch.max_checks_per_field = 0
    orch._check_counts = {}
    # The extractor loop resolves a provider adapter per role; stub it
    # non-None so the guard passes (each test replaces _call_extractor).
    orch._adapter_for_role = lambda role: object()
    return orch


def _no_consecutive_user(messages):
    roles = [m["role"] for m in messages]
    return all(not (a == "user" and b == "user")
               for a, b in zip(roles, roles[1:]))


# ---------------------------------------------------------------------------
# Termination and honest labels
# ---------------------------------------------------------------------------

class TestExtractorTermination:
    def test_persistent_validation_failure_hits_cap(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # Every turn submits a call that fails validation. Because every
        # dispatched call counts toward the cap, the loop terminates with
        # tool_cap_hit after exactly `max_tool_calls` API turns.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=4)
        calls = []
        bad = {"study": {"not_a_real_field":
                         {"value": "x", "evidence": ["<q>x</q>"]}}}

        def _fake(client, tool_defs):
            calls.append(1)
            return _resp(_tool_use(f"tu{len(calls)}", "update_study", bad))

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        assert status == "tool_cap_hit"
        assert len(calls) == 4  # bounded, not infinite
        assert orch.session.meta["tool_call_count"] == 4
        events = orch.session.read_events()
        tool_events = [e for e in events if e["event"].startswith("tool_call_")]
        # One logged event per dispatched call, so the two assertions below
        # are read off a log that recorded the run rather than an empty one.
        assert len(tool_events) == 4
        assert all(e["event"] == "tool_call_failed" for e in tool_events)
        assert not any(e["event"] == "tool_call_applied" for e in events)

    def test_partial_dispatch_is_labelled_and_counts(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # A partial dispatch (one field applied, one rejected) gets its own
        # honest event label and still consumes one unit of the cap.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=1)
        good_var = next(iter(orch.dispatcher._study_field_specs))
        partial = {"study": {
            good_var: {"value": None, "evidence": None},
            "not_a_real_field": {"value": "x", "evidence": ["<q>x</q>"]},
        }}

        def _fake(client, tool_defs):
            return _resp(_tool_use("tuP", "update_study", partial))

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        assert status == "tool_cap_hit"
        events = orch.session.read_events()
        assert any(e["event"] == "tool_call_partial" for e in events)
        assert not any(e["event"] == "tool_call_failed" for e in events)
        # Partial counted toward the cap (this is what bounds exposure).
        assert orch.session.meta["tool_call_count"] == 1

    def test_repeated_text_only_terminates_and_keeps_text(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # A model that only ever returns prose is bounded by the strict
        # consecutive-text-only limit, and its text is preserved in both the
        # conversation and the event log.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_text_only=3)
        calls = []

        def _fake(client, tool_defs):
            calls.append(1)
            return _resp(_text(f"still thinking {len(calls)}"))

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        # A text spiral is terminal (failed_validation), not a resumable pause.
        assert status == "text_only_stall"
        assert len(calls) == 3  # strict text-only bound fired
        events = orch.session.read_events()
        assert any(e["event"] == "text_only_stall" for e in events)
        # The assistant's text is logged, not dropped.
        assert sum(1 for e in events if e["event"] == "assistant_text") == 3
        assistant_text_msgs = [
            m for m in orch.messages
            if m["role"] == "assistant" and m["content"]
            and m["content"][0]["type"] == "text"]
        assert len(assistant_text_msgs) == 3
        # Alternation preserved, no two consecutive user messages.
        assert _no_consecutive_user(orch.messages)


# ---------------------------------------------------------------------------
# Auto-degrade provenance: the loop sends tool_choice "auto" and re-prompts a
# tool-free turn; for a model that CANNOT force a tool (a routed GLM vision
# endpoint), each such re-prompt is recorded in meta as an auto-degrade retry.
# A forcing model's text-only turn is the general guard and is never counted.
# The extractor loop stubs _call_extractor, so a routed model id here never
# reaches the (routed-unpriced) cost path.
# ---------------------------------------------------------------------------

class TestAutoDegradeProvenance:
    def test_glm_tool_free_turn_records_retry_then_recovers(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_text_only=5)
        # A non-forcing model: it runs under "auto" and MAY decline the tool.
        orch.session.meta["extractor_model"] = "z-ai/glm-4.6v"
        script = [
            _resp(_text("Let me reason first.")),               # tool-free turn
            (_resp(_tool_use("t2", "view_summary", {})), True),  # then completes
        ]

        def _fake(client, tool_defs):
            item = script.pop(0)
            if isinstance(item, tuple):
                resp, _complete = item
                orch.extraction_record.mark_complete()
                return resp
            return item

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "mark_complete_validated"
        # The single tool-free turn was nudged and counted for the stage.
        assert orch.session.meta["auto_degrade_retries"] == {"extractor": 1}

    def test_glm_all_text_counts_each_reprompt_then_stalls(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_text_only=3)
        orch.session.meta["extractor_model"] = "z-ai/glm-4.6v"

        def _fake(client, tool_defs):
            return _resp(_text("still thinking"))

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "text_only_stall"
        # Turns 1 and 2 re-prompt (counted); turn 3 hits the stall and does not
        # re-prompt, so the count is one below the bound.
        assert orch.session.meta["auto_degrade_retries"] == {"extractor": 2}

    def test_forcing_model_tool_free_turn_is_not_counted(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_text_only=5)
        # A forcing model: a text-only turn is the general agentic guard, still
        # re-prompted, but never an auto-degrade retry.
        orch.session.meta["extractor_model"] = "claude-opus-4-8"
        script = [
            _resp(_text("Let me reason first.")),
            (_resp(_tool_use("t2", "view_summary", {})), True),
        ]

        def _fake(client, tool_defs):
            item = script.pop(0)
            if isinstance(item, tuple):
                resp, _complete = item
                orch.extraction_record.mark_complete()
                return resp
            return item

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "mark_complete_validated"
        # The re-prompt DID fire (proves the site ran)...
        events = orch.session.read_events()
        assert any(e["event"] == "extractor_reprompt" for e in events)
        # ...but a forcing model is never counted as an auto-degrade retry.
        assert "auto_degrade_retries" not in orch.session.meta


# ---------------------------------------------------------------------------
# Repeated-failure guard: sibling to the text-only stall, for failing calls
# ---------------------------------------------------------------------------

class TestRepeatedFailureStall:
    def test_five_identical_failures_stall(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # The loop this guard exists for: mark_complete rejected with the same
        # error every turn (here, because the extraction output is empty). At
        # the fifth identical consecutive failure the loop stops with its own
        # terminal status, well before the tool-call cap.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=100,
                               max_identical_failures=5)
        calls = []

        def _fake(client, tool_defs):
            calls.append(1)
            return _resp(_tool_use(f"mc{len(calls)}", "mark_complete", {}))

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        assert status == "extractor_stalled"
        assert len(calls) == 5  # stopped at the threshold, not the cap
        assert orch.session.meta["tool_call_count"] == 5
        events = orch.session.read_events()
        stall = [e for e in events if e["event"] == "repeated_failure_stall"]
        assert len(stall) == 1
        (ev,) = stall
        assert ev["tool"] == "mark_complete"
        assert ev["consecutive_identical_failures"] == 5
        assert ev["error_codes"]  # non-empty, recorded for the audit trail
        assert ev["error_message"]  # the repeated reason, captured once
        # Every attempt was a genuine failure (nothing applied).
        assert all(e["event"] == "tool_call_failed" for e in events
                   if e["event"].startswith("tool_call_"))
        # The stall path still logs the turn's verbatim assistant_message
        # before returning: the invariant that every tool-calling turn
        # records one is total, so replay can treat a missing one as a
        # crash artefact and refuse.
        stall_turn = ev["turn_id"]
        am = [e for e in events if e["event"] == "assistant_message"
              and e["turn_id"] == stall_turn]
        assert len(am) == 1
        assert [b["type"] for b in am[0]["content"]] == ["tool_use"]
        assert am[0]["content"][0]["name"] == "mark_complete"

    def test_four_failures_then_success_resets(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # Four identical failures do not stall, and an applied/ok call resets
        # the run: two runs of four (eight failures total, but never five
        # consecutive) reach normal completion, proving the reset.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=100,
                               max_identical_failures=5)
        # (response, marks_complete_before_returning)
        script = (
            [(_resp(_tool_use(f"f{i}", "mark_complete", {})), False)
             for i in range(4)]
            + [(_resp(_tool_use("reset", "view_summary", {})), False)]
            + [(_resp(_tool_use(f"g{i}", "mark_complete", {})), False)
               for i in range(4)]
            + [(_resp(_tool_use("done", "view_summary", {})), True)]
        )

        def _fake(client, tool_defs):
            resp, complete = script.pop(0)
            if complete:
                orch.extraction_record.mark_complete()
            return resp

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        assert status == "mark_complete_validated"
        events = orch.session.read_events()
        assert not any(e["event"] == "repeated_failure_stall" for e in events)
        assert orch.session.meta["tool_call_count"] == 10

    def test_alternating_signatures_never_stall(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # Two different failure signatures in strict alternation never build a
        # run of identical failures, so the loop is bounded by the tool-call
        # cap (tool_cap_hit), never by the repeated-failure guard.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=12,
                               max_identical_failures=5)
        bad_study = {"study": {"not_a_real_field":
                               {"value": "x", "evidence": ["<q>x</q>"]}}}
        calls = []

        def _fake(client, tool_defs):
            calls.append(1)
            if len(calls) % 2 == 1:
                return _resp(_tool_use(f"a{len(calls)}", "update_study",
                                       bad_study))
            return _resp(_tool_use(f"b{len(calls)}", "mark_complete", {}))

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        assert status == "tool_cap_hit"
        assert len(calls) == 12
        events = orch.session.read_events()
        assert not any(e["event"] == "repeated_failure_stall" for e in events)

    def test_stall_session_replays(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # After a stall, the recorded failed-call events replay into a balanced
        # conversation (every tool_use paired with its tool_result), so a
        # resumed session reattaches cleanly. The stall event itself is not
        # conversation traffic and is a no-op on replay.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=100,
                               max_identical_failures=5)
        calls = []

        def _fake(client, tool_defs):
            calls.append(1)
            return _resp(_tool_use(f"mc{len(calls)}", "mark_complete", {}))

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "extractor_stalled"

        msgs = orch.session.replay_messages()
        assert _no_consecutive_user(msgs)
        assistant = [m for m in msgs if m["role"] == "assistant"]
        user = [m for m in msgs if m["role"] == "user"]
        # Each of the five failed calls replays as an assistant tool_use turn
        # paired with a user tool_result turn: balanced, no orphan turns.
        assert len(assistant) == 5
        assert len(user) == 5
        for m in assistant:
            assert sum(1 for b in m["content"]
                       if b["type"] == "tool_use") == 1


# ---------------------------------------------------------------------------
# Replay turn ids are globally unique across loops
# ---------------------------------------------------------------------------

class TestReplayTurnIds:
    def test_turn_ids_unique_across_loops_no_replay_merge(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # The collision a resume replay risks: two extractor loops with a
        # tool-free turn between them, where a counter rebased on
        # tool_call_count would give the first loop's completing turn and the
        # second loop's turn the same id. The counter is session-global
        # instead, so every turn id is distinct and replay keeps the loops'
        # assistant messages separate.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=100)
        good_var = next(iter(orch.dispatcher._study_field_specs))
        bad = {"study": {"nope": {"value": "x", "evidence": ["<q>x</q>"]}}}
        # view_summary is read-only: ok, and it does NOT clear the
        # mark_complete flag, so setting the flag just before it lets the
        # loop end on a clean turn without a full valid extraction output.
        script = [
            (_resp(_tool_use("l1t1", "update_study", bad)), False),
            (_resp(_tool_use("l1t2", "view_summary", {})), True),
            (_resp(_tool_use("l2t1", "view_summary", {})), True),
        ]

        def _fake(client, tool_defs):
            resp, complete = script.pop(0)
            if complete:
                orch.extraction_record.mark_complete()
            return resp

        orch._call_extractor = _fake

        assert orch._extractor_loop() == "mark_complete_validated"
        # A non-tool turn between two loop entries. A real run enters the
        # extractor loop once, so this drives the id allocator directly: what
        # is under test is that ids never collide across entries, whatever
        # sits between them.
        # A text-only turn, logged the way the loop logs one: the verbatim
        # assistant message plus the re-prompt sent back under the same id.
        free_turn = orch._next_turn_id()
        orch.session.append_event({
            "event": "assistant_message", "turn_id": free_turn,
            "content": [{"type": "text", "text": "Let me reconsider field X."}],
        })
        orch.session.append_event({
            "event": "extractor_reprompt", "turn_id": free_turn,
            "text": "please call a tool",
        })
        # The dispatcher clears this itself on any field write; set it
        # directly here because this turn writes no field.
        orch.extraction_record.mark_complete_flag = False
        assert orch._extractor_loop() == "mark_complete_validated"

        # Turn ids never collide across loops: the three tool turns and the
        # one tool-free turn between them get four distinct ids. (A turn emits
        # several events sharing its id, for example a tool_call plus its
        # assistant_message, so distinctness is over the id SET, not the flat
        # event list.)
        tids = [e["turn_id"] for e in orch.session.read_events()
                if "turn_id" in e]
        assert set(tids) == {1, 2, 3, 4}

        # Replay keeps the three tool turns as three distinct assistant
        # messages, each carrying exactly one tool_use; a rebased counter
        # would merge the two view_summary turns into one. The tool-free turn
        # replays as a fourth assistant message carrying no tool_use.
        msgs = orch.session.replay_messages()
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 4
        tool_msgs = [m for m in assistant_msgs
                     if any(b["type"] == "tool_use" for b in m["content"])]
        assert len(tool_msgs) == 3
        for m in tool_msgs:
            assert sum(1 for b in m["content"]
                       if b["type"] == "tool_use") == 1

    def test_text_only_turn_replays_with_paired_reprompt(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # A text-only turn logs both the assistant text and its re-prompt,
        # so replay reconstructs a clean assistant/user pair (no lone
        # assistant turn that would break alternation on resume).
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_text_only=5)
        script = [
            _resp(_text("Let me reason first.")),          # text-only turn
            (_resp(_tool_use("t2", "view_summary", {})), True),  # completes
        ]

        def _fake(client, tool_defs):
            item = script.pop(0)
            if isinstance(item, tuple):
                resp, _complete = item
                orch.extraction_record.mark_complete()
                return resp
            return item

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "mark_complete_validated"

        msgs = orch.session.replay_messages()
        roles = [m["role"] for m in msgs]
        assert _no_consecutive_user(msgs)
        assert all(not (a == "assistant" and b == "assistant")
                   for a, b in zip(roles, roles[1:]))
        # The reasoning text survived into the replayed assistant turn.
        assert any(m["role"] == "assistant"
                   and any(b.get("type") == "text"
                           and "reason" in b.get("text", "")
                           for b in m["content"])
                   for m in msgs)


# ---------------------------------------------------------------------------
# Loop and replay edges
# ---------------------------------------------------------------------------

def _thinking(text):
    # A block the orchestrator neither renders into messages nor treats as
    # text/tool_use, standing in for a thinking-only truncation.
    return SimpleNamespace(type="thinking", thinking=text)


class TestEmptyResponseAlternation:
    def test_empty_response_keeps_alternation_live_and_on_replay(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # A response with neither text nor tool_use blocks (a thinking-only
        # truncation) still contributes a placeholder assistant turn, which
        # is what preserves alternation live and on replay. Appending the
        # re-prompt with no preceding assistant message would put two user
        # messages back to back.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_text_only=3)
        calls = []

        def _fake(client, tool_defs):
            calls.append(1)
            return _resp(_thinking(f"just thinking {len(calls)}"))

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        # Empty (thinking-only) turns count as text-only, so the strict bound
        # fires and the run stalls terminally (failed_validation).
        assert status == "text_only_stall"
        assert len(calls) == 3
        # Every empty turn still contributed an assistant message (a text
        # placeholder), so no two consecutive user messages appear.
        assert _no_consecutive_user(orch.messages)
        assistant_msgs = [m for m in orch.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3
        assert all(m["content"] and m["content"][0]["type"] == "text"
                   for m in assistant_msgs)
        # An assistant_text event is logged per empty turn so replay can
        # rebuild the same alternation.
        assert sum(1 for e in orch.session.read_events()
                   if e["event"] == "assistant_text") == 3
        replayed = orch.session.replay_messages()
        assert _no_consecutive_user(replayed)


class TestBudgetHintClamp:
    def test_multi_call_batch_never_reports_negative_budget(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # A single response carrying more tool calls than the remaining cap
        # drives tool_call_count past the cap mid-batch. The budget hint sent
        # to the model must clamp at zero, NEVER go negative.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels, max_tool_calls=1)

        def _fake(client, tool_defs):
            return _resp(
                _tool_use("a", "view_summary", {}),
                _tool_use("b", "view_summary", {}),
                _tool_use("c", "view_summary", {}),
            )

        orch._call_extractor = _fake
        status = orch._extractor_loop()

        assert status == "tool_cap_hit"
        # The budget rides in the event log under the underscore-prefixed
        # telemetry key: it is stripped from the model-facing tool_result but
        # retained here for the transcript.
        budgets = [e["result"].get("_tool_call_budget_remaining")
                   for e in orch.session.read_events()
                   if e["event"].startswith("tool_call_")]
        assert len(budgets) == 3
        assert all(b is not None and b >= 0 for b in budgets)
        # The overshooting calls clamp to exactly zero rather than negative.
        assert budgets == [1, 0, 0]
        # Live path: no cap-derived number rides in the tool_results the live
        # conversation carries (each is the stripped `result_to_model_text`
        # payload). Resume is a separate surface: replayed tool_results must
        # strip the same telemetry keys, which `replay_messages` owns and its
        # own tests cover.
        tool_result_texts = [
            block["content"]
            for m in orch.messages if m["role"] == "user"
            for block in m["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert tool_result_texts
        assert all("budget_remaining" not in t for t in tool_result_texts)


class TestReplayBlockOrdering:
    def test_replay_orders_text_before_tool_use(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # A live assistant response emits text before its tool_use. The
        # orchestrator logs the tool-call event before the assistant_text
        # event, so replaying in event order would invert it; replay restores
        # text-before-tool_use.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels)

        script = [_resp(_text("Reasoning, then acting."),
                        _tool_use("t1", "view_summary", {}))]

        def _fake(client, tool_defs):
            orch.extraction_record.mark_complete()
            return script.pop(0)

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "mark_complete_validated"

        msgs = orch.session.replay_messages()
        assistant = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant) == 1
        block_types = [b["type"] for b in assistant[0]["content"]]
        assert block_types == ["text", "tool_use"]

    def test_replay_preserves_tool_use_before_text(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # The inverse interleaving: tool_use BEFORE trailing text. Replay
        # reads the recorded assistant_message content, which preserves the
        # provider's order, rather than forcing text first.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels)

        script = [_resp(_tool_use("t1", "view_summary", {}),
                        _text("Acting, then explaining."))]

        def _fake(client, tool_defs):
            orch.extraction_record.mark_complete()
            return script.pop(0)

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "mark_complete_validated"

        assistant = [m for m in orch.session.replay_messages()
                     if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert [b["type"] for b in assistant[0]["content"]] == \
            ["tool_use", "text"]


class TestReplayByteIdentity:
    def test_live_sent_content_equals_replayed(
            self, tmp_path, synthetic_template, paper_text, image_labels):
        # The core session-replay contract: what the model was sent live is
        # exactly what a resumed run replays. Both halves of it at once
        # (tool_result telemetry stripping, assistant block order), across
        # two turns with opposite text/tool_use orderings.
        orch = _bare_loop_orch(tmp_path, synthetic_template, paper_text,
                               image_labels)
        good_var = next(iter(orch.dispatcher._study_field_specs))
        update = {"study": {good_var: {
            "value": "WDS-9 study",
            "evidence": ["<q>The WDS-9 was administered to "
                         "348 units</q>"]}}}
        # Turn 1: text BEFORE tool_use, a real update (its dispatch result
        # carries underscore-prefixed _field_diffs telemetry).
        # Turn 2: tool_use then trailing text, and completes the run.
        script = [
            _resp(_text("First I will record the aim."),
                  _tool_use("t1", "update_study", update)),
            _resp(_tool_use("t2", "view_summary", {}),
                  _text("Looks complete.")),
        ]

        def _fake(client, tool_defs):
            resp = script.pop(0)
            if not script:  # the last scripted turn completes the run
                orch.extraction_record.mark_complete()
            return resp

        orch._call_extractor = _fake
        assert orch._extractor_loop() == "mark_complete_validated"

        # Drop the seeded initial user message; replay excludes it (resume
        # prepends the initial user blocks separately).
        live = orch.messages[1:]
        replayed = orch.session.replay_messages()
        assert replayed == live

        # The guarantees are real, not vacuous:
        # - opposite assistant block orders both preserved.
        assert [b["type"] for b in live[0]["content"]] == ["text", "tool_use"]
        assert [b["type"] for b in replayed[2]["content"]] == \
            ["tool_use", "text"]
        # - the update's tool_result carries no underscore telemetry on the
        #   wire, but the event log retains it (the transcript record). Every
        #   update result carries _field_diffs regardless of validation
        #   outcome, so this holds without pinning the test to a dispatch
        #   status.
        assert "_field_diffs" not in replayed[1]["content"][0]["content"]
        update_event = [e for e in orch.session.read_events()
                        if e.get("tool") == "update_study"][0]
        assert "_field_diffs" in update_event["result"]


# ---------------------------------------------------------------------------
# Image-label normalisation matches the dispatcher (strip + lower)
# ---------------------------------------------------------------------------

class TestImageLabelNormalisation:
    def test_labels_are_stripped_and_lowercased(
            self, tmp_path, config_dir, bundle_minimal_dir):
        config = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        png = tmp_path / "fig.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        # Messy labels: leading/trailing space and mixed case.
        bundle = dataclasses.replace(
            bundle, figures={"  Figure_01 ": png, "TABLE_02": png})
        orch = Orchestrator(
            config, bundle, tmp_path / "runs",
            extractor_model="claude-opus-4-7",
            review_model="claude-opus-4-7",
            api_key="")
        assert orch.image_labels == {"figure_01", "table_02"}
