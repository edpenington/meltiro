"""Tool-dispatcher robustness against malformed arguments.

Models occasionally pass strings or lists where the schema specifies an
object, especially when responding to checker-challenge feedback. The
dispatcher must surface these as validation errors, not crash.

The same applies to the NAME a model addresses its payload to. A tool that
reads one argument and ignores the rest answers `ok` to a call it dropped
whole, which is the one failure neither the model nor the operator can see:
the model believes it recorded a block, the result agrees, and nothing is
stored. So every argument outside a tool's declared list is answered, and a
check block sent to a tool that does not take it is answered by name, with the
tool that owns it.
"""

import pytest

from meltiro.extraction_record import (
    ROLE_EXTRACTOR,
    ROLE_REVIEW,
    ExtractionRecord,
)
from meltiro.tools import (
    _TOOL_ARGUMENTS,
    ToolDispatcher,
    all_tool_definitions,
)

from .conftest import (
    INITIAL_CHECK_FIELDS,
    QUALITY_CHECK_FIELDS,
    open_initial_check_gate,
)


def _dispatcher(template, paper_text, image_labels):
    """A record with the ordering gate already open, plus its dispatcher.

    Every mutating call is refused until `record_initial_check` has landed.
    That rule is not what this file is about (see test_tools.py for it), and
    leaving the gate shut would answer every call below with the ordering
    refusal instead of the malformed-argument error each test is named for.
    """
    record = open_initial_check_gate(ExtractionRecord())
    return record, ToolDispatcher(record, template, paper_text, image_labels)


def test_update_study_with_string_study_arg(
        synthetic_template, paper_text, image_labels):
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    result = d.dispatch("update_study", {
        "study": "not an object, where a map of envelopes belongs",
    })
    assert result["status"] == "validation_failed"
    assert any(e["code"] == "type_mismatch" and e["path"] == "study"
               for e in result["errors"])


def test_update_study_with_a_check_block_says_where_it_went(
        synthetic_template, paper_text, image_labels):
    # `initial_check` is NOT an argument here, well-formed or not. The whole
    # call is refused (silently dropping a block the model believed it had
    # recorded is the quiet loss the split exists to end) and the message
    # names the tool that owns the block.
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    for block, home in (("initial_check", "record_initial_check"),
                        ("quality_check", "mark_complete")):
        result = d.dispatch("update_study", {block: ["text_readable", True]})
        assert result["status"] == "validation_failed"
        err = next(e for e in result["errors"] if e["path"] == block)
        assert err["code"] == "block_moved"
        assert home in err["message"]
    assert a.initial_check == {} and a.quality_check == {}


def test_mark_complete_with_list_for_quality_check(
        synthetic_template, paper_text, image_labels):
    # The quality check's own door takes the same treatment: a list where an
    # object belongs is a validation error naming the block, not a crash.
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
    result = d.dispatch("mark_complete", {
        "quality_check": ["deviation_from_expectations", "none"],
    })
    assert result["status"] == "validation_failed"
    assert any(e["code"] == "type_mismatch" and e["path"] == "quality_check"
               for e in result["errors"])
    assert a.mark_complete_flag is False


def test_record_initial_check_with_a_non_object_input(
        synthetic_template, paper_text, image_labels):
    # The dispatcher's own input guard: a tool input that is not a JSON
    # object at all is refused before any handler reads it, and the ordering
    # gate stays shut, because nothing was recorded.
    record = ExtractionRecord()
    d = ToolDispatcher(record, synthetic_template, paper_text, image_labels)
    result = d.dispatch("record_initial_check", ["text_readable"])
    assert result["status"] == "validation_failed"
    assert any(e["code"] == "malformed_input" for e in result["errors"])
    assert record.initial_check_recorded is False


def test_add_record_with_string_fields(
        synthetic_template, paper_text, image_labels):
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    result = d.dispatch("add_record", {"fields": "nope"})
    assert result["status"] == "validation_failed"
    assert any(e["code"] == "type_mismatch" for e in result["errors"])


def test_update_record_with_list_fields(
        synthetic_template, paper_text, image_labels):
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    # Seed a record so update_record gets past the known-record check.
    d.dispatch("add_record", {"fields": {
        "gauge": {"value": "WDS-9", "evidence": "<q>WDS-9 was administered</q>"},
        "outcome_variable": {"value": "X",
                             "evidence": "<q>unplanned removal</q>"},
        "outcome_category": {"value": "Failure state",
                             "evidence": "<q>unplanned removal</q>"},
    }})
    result = d.dispatch("update_record", {
        "record_id": "relationship_1",
        "fields": ["this", "is", "a", "list"],
    })
    assert result["status"] == "validation_failed"
    assert any(e["code"] == "type_mismatch" for e in result["errors"])


# ---------------------------------------------------------------------------
# An argument the tool does not take
# ---------------------------------------------------------------------------
#
# The failure this closes is not a crash and not a rejection: it is a call
# that answered `ok`, reported no error, and stored nothing, because the
# payload arrived under a name the handler never read. `fields` is what the
# sibling record tools take from the same catalogue, so it is the name a model
# reaches for, and a model given `ok` has no reason to look again.

# One valid study field, so a refused call is refused despite carrying content
# that would otherwise have applied.
_GOOD_STUDY = {"primary_aim": {"value": "Aim A",
                               "evidence": "<q>The WDS-9 was administered</q>",
                               "notes": None}}

# Minimal well-formed arguments per tool, to which each test adds one name the
# tool does not take. Without these the call could fail for a second reason
# and the test would pass on the wrong error.
_WELL_FORMED = {
    "update_study": {"study": dict(_GOOD_STUDY)},
    "add_record": {"fields": {
        "gauge": {"value": "WDS-9",
                  "evidence": "<q>WDS-9 was administered</q>"},
        "outcome_variable": {"value": "unplanned removal",
                             "evidence": "<q>unplanned removal</q>"},
        "outcome_category": {"value": "Failure state",
                             "evidence": "<q>unplanned removal</q>"},
    }},
    "update_record": {
        "record_id": "relationship_1",
        "fields": {"effect_size": {
            "value": "1.34",
            "evidence": "<q>odds ratio for unplanned removal</q>"}},
    },
    "remove_record": {"record_id": "relationship_1", "reason": "added twice"},
    "mark_complete": {"quality_check": dict(QUALITY_CHECK_FIELDS)},
    "abandon_extraction": {"reason": "the text is unreadable"},
    "view_summary": {},
    "view_study_fields": {},
    "view_record": {"record_id": "relationship_1"},
}


def test_update_study_refuses_a_field_map_sent_under_another_name(
        synthetic_template, paper_text, image_labels):
    # The reproduced defect: `fields` is what add_record and update_record
    # take, so it is the name a model borrows. Answering `ok` to it loses the
    # whole payload with no one the wiser.
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    for wrong_name in ("fields", "studdy", "study_fields"):
        result = d.dispatch("update_study", {wrong_name: dict(_GOOD_STUDY)})
        assert result["status"] == "validation_failed", wrong_name
        err = next(e for e in result["errors"] if e["path"] == wrong_name)
        assert err["code"] == "unknown_argument"
        # Names the offending argument AND where the field map belongs, so
        # the model can resubmit rather than guess.
        assert wrong_name in err["message"]
        assert "`study`" in err["message"]
        assert "primary_aim" not in a.study, \
            "a refused call must store nothing"


@pytest.mark.parametrize("tool_name", sorted(_TOOL_ARGUMENTS))
def test_every_tool_refuses_an_argument_it_does_not_take(
        tool_name, synthetic_template, paper_text, image_labels):
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    args = dict(_WELL_FORMED[tool_name])
    args["not_an_argument"] = {"anything": "at all"}
    result = d.dispatch(tool_name, args)
    assert result["status"] == "validation_failed"
    err = next(e for e in result["errors"] if e["path"] == "not_an_argument")
    assert err["code"] == "unknown_argument"
    assert tool_name in err["message"]
    # Nothing moved: no study field, no record, no completion, no surrender.
    assert "primary_aim" not in a.study
    assert a.records == []
    assert a.mark_complete_flag is False
    assert a.abandoned_flag is False


@pytest.mark.parametrize("tool_name", sorted(_TOOL_ARGUMENTS))
def test_a_well_formed_call_is_not_refused_by_the_argument_guard(
        tool_name, synthetic_template, paper_text, image_labels):
    # The other half of the guard: it must not refuse a legal call. Whether
    # the call then succeeds on its contents is a separate question (an empty
    # record store has no relationship_1 to update), so this only pins that
    # no argument-level error is raised.
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    result = d.dispatch(tool_name, dict(_WELL_FORMED[tool_name]))
    assert not any(e["code"] in ("unknown_argument", "block_moved")
                   for e in result.get("errors", []))


def test_the_argument_map_covers_every_tool_the_schemas_declare(
        synthetic_template):
    # The map and the tool definitions are two statements of the same fact.
    # A tool that gains an argument without gaining an entry here would have
    # that argument refused at runtime while the schema advertises it.
    for role, definitions in all_tool_definitions(synthetic_template).items():
        for definition in definitions:
            name = definition["name"]
            declared = set(definition["input_schema"].get("properties", {}))
            if name == "record_initial_check":
                # Deliberately absent: its top-level arguments ARE its fields,
                # so an unrecognised one is a field name and is answered as
                # such (see the test below).
                assert name not in _TOOL_ARGUMENTS
                continue
            assert set(_TOOL_ARGUMENTS[name]) == declared, f"{role}/{name}"


def test_record_initial_check_answers_an_unknown_argument_as_a_field(
        synthetic_template, paper_text, image_labels):
    # Its schema is flat, so there is no argument/field distinction to make:
    # a name it does not know is a misspelled field, and it gets the per-field
    # answer with a spelling hint rather than an argument-level refusal.
    record = ExtractionRecord()
    d = ToolDispatcher(record, synthetic_template, paper_text, image_labels)
    result = d.dispatch("record_initial_check", {"text_readble": True})
    assert result["status"] == "validation_failed"
    codes = [e["code"] for e in result["errors"]]
    assert codes == ["unknown_field"]
    assert "text_readable" in result["errors"][0]["message"]


def test_a_check_block_names_its_owner_whichever_tool_it_reaches(
        synthetic_template, paper_text, image_labels):
    # The check blocks have one tool each. Addressed to any other, the answer
    # names that tool rather than leaving the model to rediscover it.
    a, d = _dispatcher(synthetic_template, paper_text, image_labels)
    for tool_name, block, home in (
            ("add_record", "initial_check", "record_initial_check"),
            ("mark_complete", "initial_check", "record_initial_check"),
            ("update_record", "quality_check", "mark_complete"),
    ):
        args = dict(_WELL_FORMED[tool_name])
        args[block] = dict(INITIAL_CHECK_FIELDS)
        result = d.dispatch(tool_name, args)
        assert result["status"] == "validation_failed"
        err = next(e for e in result["errors"] if e["path"] == block)
        assert err["code"] == "block_moved"
        assert home in err["message"]
    assert a.initial_check == {} and a.quality_check == {}


class TestTheReviewersExitStaysUnconditional:
    """`mark_complete` is the reviewer's only way out of a fresh-context loop
    with no replay, so nothing the dispatcher can decide may withhold it. An
    argument it cannot read is announced as a warning and the review ends,
    which is how that call already treats a quality check it cannot record."""

    def test_the_reviewer_is_warned_and_still_concludes(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch(
            "mark_complete",
            {"quality_check": dict(QUALITY_CHECK_FIELDS),
             "summary": "reviewed"},
            role=ROLE_REVIEW,
        )
        assert result["status"] == "ok"
        warned = next(w for w in result["warnings"]
                      if w["path"] == "summary")
        assert warned["code"] == "argument_not_read"
        # The warning says what happened, and what happened is that the rest
        # of the call applied.
        assert "not read" in warned["message"]
        assert a.quality_check_for(ROLE_REVIEW) == QUALITY_CHECK_FIELDS

    def test_the_extractors_own_completion_is_still_refused(
            self, synthetic_template, paper_text, image_labels):
        # The extractor's `mark_complete` IS gated, and it has its whole loop
        # left to correct the call, so the same argument is a refusal there.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch(
            "mark_complete",
            {"quality_check": dict(QUALITY_CHECK_FIELDS),
             "summary": "done"},
            role=ROLE_EXTRACTOR,
        )
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "unknown_argument" for e in result["errors"])
        assert a.mark_complete_flag is False
