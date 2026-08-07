"""Tests for the mutable ExtractionRecord state container."""

import pytest

from meltiro.extraction_record import (
    ROLE_EXTRACTOR,
    ROLE_REVIEW,
    ExtractionRecord,
)


# The record entity noun the config fixture declares (records: block key). It
# prefixes every auto-assigned record id: `relationship_1`, `relationship_2`, ...
ENTITY = "relationship"


def _env(value, evidence=None, source=None):
    return {"value": value, "evidence": evidence, "source": source}


def _add(rec, fields, entity=ENTITY):
    return rec.add_record(fields, entity)


class TestStudyUpdates:
    def test_each_block_is_written_through_its_own_door(self):
        # The two check blocks have one path each and one author each.
        # `apply_update_study` carries study fields and the scope note and
        # NOTHING else: reaching a check block through it as well would make
        # a role's own account of its run last-writer-wins.
        a = ExtractionRecord()
        a.record_initial_check(
            {"text_readable": True, "expected_relationships": 3})
        a.apply_update_study(
            study={"primary_aim": _env("To assess X", ["q"], "Abstract")},
        )
        a.record_quality_check({"deviation_from_expectations": "none"},
                               role=ROLE_EXTRACTOR)
        assert a.initial_check_for(ROLE_EXTRACTOR)["text_readable"] is True
        assert a.study["primary_aim"]["value"] == "To assess X"
        assert a.quality_check_for(ROLE_EXTRACTOR) == {
            "deviation_from_expectations": "none"}

    def test_apply_update_study_cannot_reach_a_check_block(self):
        # The signature is the enforcement: there is no argument to pass one
        # through, so a caller cannot overwrite another role's self-assessment
        # while revising a study field.
        import inspect
        params = inspect.signature(
            ExtractionRecord.apply_update_study).parameters
        assert set(params) == {"self", "study", "notes"}

    def test_update_clears_mark_complete(self):
        a = ExtractionRecord()
        a.mark_complete_flag = True
        a.apply_update_study(study={"x": _env("y")})
        assert a.mark_complete_flag is False

    def test_partial_update_preserves_other_fields(self):
        a = ExtractionRecord()
        a.apply_update_study(study={
            "primary_aim": _env("Aim 1"),
            "sample_size": _env(100),
        })
        a.apply_update_study(study={"primary_aim": _env("Aim 2")})
        # primary_aim overwritten, sample_size unchanged.
        assert a.study["primary_aim"]["value"] == "Aim 2"
        assert a.study["sample_size"]["value"] == 100


class TestRelationships:
    def test_add_relationship_assigns_sequential_ids(self):
        a = ExtractionRecord()
        r1 = _add(a, {"gauge": _env("WDS-9", ["q"], "Methods")})
        r2 = _add(a, {"gauge": _env("CRT-HD", ["q"], "Methods")})
        assert r1 == "relationship_1"
        assert r2 == "relationship_2"

    def test_update_relationship_modifies_existing(self):
        a = ExtractionRecord()
        _add(a, {"gauge": _env("WDS-9", ["q"], "Methods")})
        a.update_record("relationship_1", {"outcome_variable": _env(
            "DI-4", ["q"], "Methods")})
        rel = a.get_record("relationship_1")
        assert rel["gauge"]["value"] == "WDS-9"
        assert rel["outcome_variable"]["value"] == "DI-4"

    def test_update_relationship_unknown_id_raises(self):
        a = ExtractionRecord()
        with pytest.raises(KeyError):
            a.update_record("relationship_99", {"gauge": _env("WDS-9")})

    def test_remove_relationship_does_not_renumber(self):
        a = ExtractionRecord()
        _add(a, {"gauge": _env("A")})  # relationship_1
        _add(a, {"gauge": _env("B")})  # relationship_2
        _add(a, {"gauge": _env("C")})  # relationship_3
        a.remove_record("relationship_2")
        assert a.record_ids() == ["relationship_1", "relationship_3"]
        # Next add takes a fresh number, never the retired gap.
        r4 = _add(a, {"gauge": _env("D")})
        assert r4 == "relationship_4"

    def test_remove_unknown_id_raises(self):
        a = ExtractionRecord()
        with pytest.raises(KeyError):
            a.remove_record("relationship_1")


class TestEntityPrefix:
    """Record ids are `<entity>_<n>`, prefixed by the record entity the
    template declares. The engine stays generic: the prefix is supplied per
    call, never hardcoded, so a config whose record entity is something else
    gets its own prefix."""

    def test_prefix_is_the_declared_entity(self):
        a = ExtractionRecord()
        assert _add(a, {"x": _env("1")}, entity="outcome") == "outcome_1"
        assert _add(a, {"x": _env("2")}, entity="outcome") == "outcome_2"

    def test_counter_is_keyed_per_entity(self):
        # The counter is keyed by entity, so distinct entities count
        # independently rather than sharing one sequence.
        a = ExtractionRecord()
        assert _add(a, {"x": _env("1")}, entity="relationship") \
            == "relationship_1"
        assert _add(a, {"x": _env("2")}, entity="outcome") == "outcome_1"
        assert _add(a, {"x": _env("3")}, entity="relationship") \
            == "relationship_2"
        assert a.record_id_counters() == {"relationship": 3, "outcome": 2}

    def test_add_record_requires_an_entity(self):
        a = ExtractionRecord()
        for bad in (None, "", 3):
            with pytest.raises(ValueError):
                a.add_record({"x": _env("1")}, bad)


class TestRecordIdNeverRenumbered:
    """Contract guarantee: record ids are never renumbered by later edits or
    deletions within a run, and a removed number is never reissued.

    A consumer keys its cross-producer pairing maps on `record_id`, so this
    regression pins the guarantee that a refactor could otherwise break
    downstream and silently: ids stay `<entity>_1..<entity>_n` in creation
    order, a hard-deleted id is retired and never reused, and no surviving
    record is renumbered.
    """

    def test_add_returns_ids_in_strict_creation_order(self):
        a = ExtractionRecord()
        ids = [_add(a, {"gauge": _env(x)}) for x in "ABCD"]
        assert ids == ["relationship_1", "relationship_2",
                       "relationship_3", "relationship_4"]

    def test_mid_sequence_delete_and_readd_never_renumbers(self):
        a = ExtractionRecord()
        # Create four records, tagging each so the survivors can be shown to
        # keep both their id and their content.
        for letter in "ABCD":
            _add(a, {"gauge": _env(letter)})  # relationship_1..relationship_4
        # Hard-delete one from the middle.
        a.remove_record("relationship_2")
        # Add another: it must take the next fresh number, not refill the gap.
        assert _add(a, {"gauge": _env("E")}) == "relationship_5"
        # Delete another mid-sequence, then add again.
        a.remove_record("relationship_4")
        assert _add(a, {"gauge": _env("F")}) == "relationship_6"

        # Survivors, in creation order, with the retired ids absent.
        assert a.record_ids() == ["relationship_1", "relationship_3",
                                  "relationship_5", "relationship_6"]

        # No surviving record was renumbered or had its content shuffled:
        # each id still maps to exactly the value it was created with.
        by_id = {r["record_id"]: r["gauge"]["value"] for r in a.records}
        assert by_id == {
            "relationship_1": "A", "relationship_3": "C",
            "relationship_5": "E", "relationship_6": "F",
        }

        # The retired ids never reappear, however much churn follows.
        assert "relationship_2" not in a.record_ids()
        assert "relationship_4" not in a.record_ids()
        assert _add(a, {"gauge": _env("G")}) == "relationship_7"

    def test_deleting_every_record_still_never_reuses_ids(self):
        a = ExtractionRecord()
        _add(a, {"gauge": _env("A")})  # relationship_1
        _add(a, {"gauge": _env("B")})  # relationship_2
        a.remove_record("relationship_1")
        a.remove_record("relationship_2")
        assert a.record_ids() == []
        # Even with nothing left, the counter does not rewind to 1.
        assert _add(a, {"gauge": _env("C")}) == "relationship_3"


class TestSerialise:
    def test_round_trip(self):
        a = ExtractionRecord()
        a.record_initial_check({"text_readable": True})
        a.apply_update_study(
            study={"primary_aim": _env("X", ["q"], "Abstract")},
        )
        a.record_quality_check({"deviation_from_expectations": "ok"},
                               role=ROLE_EXTRACTOR)
        a.record_quality_check({"deviation_from_expectations": "one gap"},
                               role=ROLE_REVIEW)
        _add(a, {"gauge": _env("WDS-9", ["q"], "Methods")})
        _add(a, {"gauge": _env("CRT-HD", ["q"], "Methods")})

        d = a.to_dict()
        # to_dict is exactly the four consumer-facing blocks; the record-id
        # counter is session bookkeeping and never leaks into it.
        assert set(d) == {"initial_check", "study", "records", "quality_check"}
        assert "record_id_counters" not in d
        assert "_next_record_index" not in d
        # Both check blocks are keyed by the role that recorded them, so every
        # self-assessment in the file carries its author beside it and two
        # opinions are kept apart rather than collapsed into one.
        assert d["initial_check"] == {"extractor": {"text_readable": True}}
        assert d["quality_check"] == {
            "extractor": {"deviation_from_expectations": "ok"},
            "review": {"deviation_from_expectations": "one gap"},
        }

        b = ExtractionRecord.from_dict(d)
        assert b.to_dict() == d
        # from_dict alone does NOT carry the counter (that is run.json's job);
        # threading it back is what continues the sequence past the survivors.
        b.set_record_id_counters(a.record_id_counters())
        r3 = _add(b, {"gauge": _env("X")})
        assert r3 == "relationship_3"

    def test_from_dict_does_not_reissue_removed_ids(self):
        # from_dict must NOT re-derive the counter as max(surviving index)+1:
        # that reissues a removed id on resume. The counter comes from the
        # persisted bookkeeping (run.json), threaded in via
        # set_record_id_counters.
        d = {
            "study": {},
            "initial_check": {},
            "quality_check": {},
            # relationship_2 was removed before the pause; only 1 and 3 survive.
            "records": [
                {"record_id": "relationship_1", "gauge": _env("A")},
                {"record_id": "relationship_3", "gauge": _env("C")},
            ],
        }
        a = ExtractionRecord.from_dict(d)
        # The persisted counter (three ids ever minted) is restored from meta.
        a.set_record_id_counters({"relationship": 4})
        rnew = _add(a, {"gauge": _env("D")})
        assert rnew == "relationship_4"
        # relationship_2 (removed) is never reissued.
        assert "relationship_2" not in a.record_ids()

    def test_counter_round_trips_through_bookkeeping(self):
        a = ExtractionRecord()
        _add(a, {"gauge": _env("A")})
        _add(a, {"gauge": _env("B")})
        # record_id_counters() is a copy: mutating it does not touch the record.
        counters = a.record_id_counters()
        assert counters == {"relationship": 3}
        counters["relationship"] = 99
        assert a.record_id_counters() == {"relationship": 3}


class TestTheCheckBlocks:
    """Both check blocks are self-assessments, so their author is part of the
    datum: each is keyed by the ROLE that recorded it, written through that
    role's own method, and never reachable by another role."""

    def test_each_role_writes_only_its_own_block(self):
        a = ExtractionRecord()
        a.record_quality_check({"deviation_from_expectations": "none"},
                               role=ROLE_EXTRACTOR)
        a.record_quality_check({"deviation_from_expectations": "two gaps"},
                               role=ROLE_REVIEW)
        assert a.quality_check == {
            "extractor": {"deviation_from_expectations": "none"},
            "review": {"deviation_from_expectations": "two gaps"},
        }

    def test_a_role_that_said_nothing_has_no_key(self):
        # An absent key reads as "this role said nothing" rather than as an
        # empty answer.
        a = ExtractionRecord()
        assert a.quality_check_for(ROLE_REVIEW) == {}
        assert ROLE_REVIEW not in a.quality_check

    def test_a_re_call_revises_rather_than_replacing(self):
        a = ExtractionRecord()
        a.record_initial_check({"text_readable": True,
                                "expected_relationships": 2})
        applied = a.record_initial_check({"expected_relationships": 5})
        assert applied == ["expected_relationships"]
        assert a.initial_check_for(ROLE_EXTRACTOR) == {
            "text_readable": True, "expected_relationships": 5}

    def test_the_gate_is_state_rather_than_derived_emptiness(self):
        # A template declaring no initial-check fields leaves the block empty
        # even after a successful call, so deriving the gate from emptiness
        # would deadlock such a bundle forever.
        a = ExtractionRecord()
        assert a.initial_check_recorded is False
        a.record_initial_check({})
        assert a.initial_check_recorded is True
        assert a.initial_check_for(ROLE_EXTRACTOR) == {}

    def test_only_the_extractor_latches_the_gate(self):
        # The gate is a fact about the extractor's run; another role recording
        # into the block does not open it.
        a = ExtractionRecord()
        a.record_initial_check({"text_readable": True}, role=ROLE_REVIEW)
        assert a.initial_check_recorded is False

    def test_the_gate_is_bookkeeping_threaded_back_on_resume(self):
        # Not part of the consumer-facing output: it rides in run.json beside
        # the record-id counters. Without the thread-back a resumed extractor
        # would be told to record its initial check again, and would either
        # comply with a post-hoc answer or wedge against the gate.
        a = ExtractionRecord()
        a.record_initial_check({"text_readable": True})
        d = a.to_dict()
        assert "initial_check_recorded" not in d
        b = ExtractionRecord.from_dict(d)
        assert b.initial_check_recorded is False
        b.set_initial_check_recorded(True)
        assert b.initial_check_recorded is True

    def test_a_flat_block_degrades_to_an_honest_empty(self):
        # Every block this engine writes is keyed by the role that wrote it,
        # so a flat `{variable: value}` block can only be hand-built, and it
        # names no author. It normalises to "no role recorded anything"
        # rather than being re-nested under a fabricated role, so such a file
        # cannot attribute an answer to an author who never gave it.
        a = ExtractionRecord.from_dict({
            "study": {}, "records": [],
            "initial_check": {"text_readable": True},
            "quality_check": {"deviation_from_expectations": "ok"},
        })
        assert a.initial_check == {}
        assert a.quality_check == {}

    def test_the_reviewers_view_drops_both_blocks(self):
        # `include_checks=False` is the view handed to the final reviewer: it
        # forms an independent second opinion, and the extractor's account of
        # how its run went is exactly the kind of anchor the review stage is
        # not shown. The blocks are still on disk and still in the run record.
        a = ExtractionRecord()
        a.record_initial_check({"text_readable": True})
        a.record_quality_check({"deviation_from_expectations": "none"},
                               role=ROLE_EXTRACTOR)
        _add(a, {"gauge": _env("WDS-9", ["q"], "Methods")})
        assert set(a.to_dict(include_checks=False)) == {"study", "records"}
        assert a.to_dict(include_checks=False)["records"] == \
            a.to_dict()["records"]
