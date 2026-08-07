"""The tool surface the model is given, and the dispatcher that enforces it.

Two things sit behind this module, and they are two halves of one contract.

`get_tool_definitions` derives the tool catalogue from the extraction template,
so the schemas a model is offered are a function of the config bundle and NOT
of anything hardcoded in the engine. A template's declared fields become the
properties of `update_study` / `add_record`, its `required: true` variables
become JSON-Schema `required`, and `additionalProperties` is false throughout:
the API is asked to refuse a malformed call before it is ever made, because a
call the schema rejects costs nothing while a call the dispatcher rejects costs
a turn.

`ToolDispatcher` is the authority on whether a call is legal, and it never
trusts the schema to have done the work. The same envelope shape is revalidated
on arrival, an undeclared field comes back as `unknown_field` rather than being
stored, and a rejected call returns a structured `validation_failed` result the
model can act on rather than raising. Order is enforced too:
`record_initial_check` gates the extraction, so a model cannot report on the
inputs after it has already read them.

The dispatcher's results are prose the model reads, so their WORDING is pinned
here as deliberately as their status codes. A result that names the wrong
entity, or that reports success on a call that stored nothing, teaches the
model something false for the rest of the run.
"""

import json

from meltiro.extraction_record import (
    ROLE_EXTRACTOR,
    ROLE_REVIEW,
    ExtractionRecord,
)
from meltiro.tools import (
    MUTATING_TOOLS,
    ToolDispatcher,
    _field_value_subschema,
    canonical_tool_set_json,
    get_tool_definitions,
)

from .conftest import (
    INITIAL_CHECK_FIELDS,
    QUALITY_CHECK_FIELDS,
    open_initial_check_gate,
)


def _dispatcher(template, paper_text, image_labels):
    """A record whose ordering gate is already open, plus its dispatcher.

    Every mutating call (and `mark_complete`) is refused until
    `record_initial_check` has landed. That rule is pinned by
    TestTheInitialCheckGate below; the tests using this helper are about what
    the tools do once the gate is open, so they latch it directly rather than
    opening every test with an unrelated tool call.
    """
    record = open_initial_check_gate(ExtractionRecord())
    return record, ToolDispatcher(record, template, paper_text, image_labels)


def _env(value, quotes=None, source=None):
    """Build a new-shape envelope `{value, evidence: str | None}`.

    `quotes` is a list of strings, folded into `<q>...</q>` blocks so a
    test can name the quotes without writing the tags. `source` is accepted
    and ignored: the unified envelope has no separate source field, and
    figure/table references live in `<img>label</img>` tags inside the
    evidence string. Use `_env_with_image(...)` for an image reference.
    """
    if not quotes:
        evidence = None
    else:
        evidence = "".join(f"<q>{q}</q>" for q in quotes)
    return {"value": value, "evidence": evidence}


def _env_with_image(value, image_label):
    """Envelope referencing a cropped figure/table via <img>label</img>."""
    return {"value": value, "evidence": f"<img>{image_label}</img>"}


class TestToolDefinitions:
    def test_tool_set(self, synthetic_template):
        tools = get_tool_definitions(synthetic_template)
        names = [t["name"] for t in tools]
        assert names == [
            "record_initial_check",
            "update_study", "add_record", "update_record",
            "remove_record", "mark_complete", "abandon_extraction",
            "view_summary", "view_study_fields", "view_record",
        ]

    def test_the_reviewer_catalogue_is_the_extractor_minus_the_first_call(
            self, synthetic_template):
        # The initial check reports on the inputs BEFORE extraction begins;
        # there is no moment in the review at which the equivalent could
        # honestly be performed, so the reviewer is never offered the tool and
        # cannot file a retrospective one over the extractor's. Everything
        # else is shared, `mark_complete` included (that is how the reviewer
        # records its own quality check).
        extractor = [t["name"] for t
                     in get_tool_definitions(synthetic_template,
                                             role=ROLE_EXTRACTOR)]
        reviewer = [t["name"] for t
                    in get_tool_definitions(synthetic_template,
                                            role=ROLE_REVIEW)]
        assert reviewer == [n for n in extractor
                            if n != "record_initial_check"]

    def test_input_schemas_are_objects(self, synthetic_template):
        for tool in get_tool_definitions(synthetic_template):
            assert tool["input_schema"]["type"] == "object"

    def test_mutating_tools_classifies_the_whole_catalogue(
            self, synthetic_template):
        # MUTATING_TOOLS drives whether a reviewer's call counts as an edit (see
        # orchestrator._final_review), so a tool that joins the catalogue without
        # being classified would be silently treated as read-only. Spelling the
        # non-mutating half out here forces the decision at the point a tool is
        # added: this fails until the new name is put in one list or the other.
        names = {t["name"] for t in get_tool_definitions(synthetic_template)}
        non_mutating = {
            # Read-only.
            "view_summary", "view_study_fields", "view_record",
            # Control flags: they latch state the orchestrator reads, but they
            # change no field of the extraction output.
            "mark_complete", "abandon_extraction",
            # Writes a check block, not a field. MUTATING_TOOLS answers "did
            # this stage revise the extracted content", and the check blocks
            # are self-assessment rather than content: nothing the checker
            # re-reads moves. It is also absent from the reviewer's catalogue
            # (and refused for that role), so the one consumer of this set —
            # the final review's did-it-edit accounting — can never see it.
            "record_initial_check",
        }
        assert MUTATING_TOOLS | non_mutating == names
        assert not (MUTATING_TOOLS & non_mutating)

    def test_applied_changes_alone_does_not_identify_a_mutating_tool(
            self, synthetic_template, paper_text, image_labels):
        # Why MUTATING_TOOLS has to exist rather than reading applied_changes.
        # It is tempting to treat "reports applied_changes" as the mutation
        # oracle, and on the view tools' SUCCESS path it happens to hold. It does
        # not hold in general, in both directions:
        #
        #   - abandon_extraction changes no field, yet answers `status: ok` with
        #     a NON-EMPTY applied_changes describing the flag it latched. Keying
        #     on applied_changes would book it as an applied edit. (mark_complete
        #     reports the same way once its completeness gate passes.)
        #   - view_record's validation-failure path returns through the same
        #     `_result` helper as the mutating tools, so a read DOES report
        #     applied_changes there, empty.
        #
        # So the key's presence tracks which helper built the payload, not
        # whether the extraction output moved.
        record, dispatcher = _dispatcher(
            synthetic_template, paper_text, image_labels)

        res = dispatcher.dispatch("abandon_extraction", {"reason": "no data"})
        assert res["status"] == "ok"
        assert res["applied_changes"] == {"abandon_extraction": True,
                                          "reason": "no data"}
        # Nothing edited: no field written, no record added. The study
        # block carries only its reserved scope-note key.
        assert record.study == {"notes": None} and record.records == []

        res = dispatcher.dispatch("view_record", {"record_id": "no_such_id"})
        assert res["status"] == "validation_failed"
        assert res["applied_changes"] == {}                  # a read, reporting

    def test_read_only_tools_report_no_applied_changes_when_they_succeed(
            self, synthetic_template, paper_text, image_labels):
        # The narrow true claim, which is what makes applied_changes a usable
        # "did it land" signal ONCE MUTATING_TOOLS has established that the call
        # was an edit at all: a view tool that succeeds reports nothing applied.
        _record, dispatcher = _dispatcher(
            synthetic_template, paper_text, image_labels)
        for name in ("view_summary", "view_study_fields"):
            res = dispatcher.dispatch(name, {})
            assert res["status"] == "ok"
            assert "applied_changes" not in res
        # And a mutating tool always reports, so "did it land" is answerable for
        # every member of MUTATING_TOOLS.
        assert "applied_changes" in dispatcher.dispatch("update_study",
                                                        {"study": {}})

    def test_canonical_json_is_stable(self, synthetic_template):
        a = canonical_tool_set_json(synthetic_template)
        b = canonical_tool_set_json(synthetic_template)
        assert a == b
        # And parseable.
        json.loads(a)

    def _fields_props(self, synthetic_template, tool_name):
        tool = next(t for t in get_tool_definitions(synthetic_template)
                    if t["name"] == tool_name)
        return tool["input_schema"]["properties"]["fields"]["properties"]

    def test_update_record_schema_is_slimmed(self, synthetic_template):
        # The measured duplication: add_record and update_record both carried
        # the full record field catalogue. add_record keeps it; update_record
        # is slimmed to bare {value, evidence} envelopes. The model reads the
        # field reference from add_record in the same request.
        add_props = self._fields_props(synthetic_template, "add_record")
        upd_props = self._fields_props(synthetic_template, "update_record")

        # Both advertise the same field names (same envelopes, just slimmed).
        assert set(add_props) == set(upd_props)

        # add_record still carries a description on every field property;
        # update_record carries none.
        assert all("description" in p for p in add_props.values())
        assert all("description" not in p for p in upd_props.values())

        # The categorical field's enum option list is present on add_record
        # and dropped on update_record (collapsed to a nullable string).
        assert add_props["outcome_category"]["properties"]["value"] == {
            "enum": ["Cost or resource use", "Service life",
                     "Failure state", None]
        }
        assert upd_props["outcome_category"]["properties"]["value"] == {
            "type": ["string", "null"]
        }

        # The envelope shape is preserved: value + evidence + notes, all
        # three required, no additional keys.
        for p in upd_props.values():
            assert set(p["properties"]) == {"value", "evidence", "notes"}
            assert p["required"] == ["value", "evidence", "notes"]
            assert p["additionalProperties"] is False
            assert p["properties"]["evidence"] == {"type": ["string", "null"]}
            assert p["properties"]["notes"] == {"type": ["string", "null"]}

    def test_update_record_fields_desc_points_at_add_record(
            self, synthetic_template):
        # The slimmed schema drops per-field guidance, so the block
        # description must send the model to add_record for the reference.
        tool = next(t for t in get_tool_definitions(synthetic_template)
                    if t["name"] == "update_record")
        desc = tool["input_schema"]["properties"]["fields"]["description"]
        assert "add_record" in desc


class TestTheInitialCheckGate:
    """`record_initial_check` is the extractor's first call, and the engine
    enforces it.

    The block gets a tool of its own rather than riding as an optional
    argument of `update_study`. Carried as an argument, the ordering is only
    what the prompt asks for, and a model that starts extracting produces a
    pre-extraction report written after the fact, or none at all. An ordering
    requirement the engine enforces is worth more than one a prompt describes.
    The tests elsewhere in this file open the gate by hand (see `_dispatcher`)
    because their subject is what happens after it.
    """

    def _tool(self, template):
        return next(t for t in get_tool_definitions(template)
                    if t["name"] == "record_initial_check")

    def test_the_properties_are_flat_one_per_declared_variable(
            self, synthetic_template):
        # FLAT, not nested under a block key: the tool name already says which
        # block this is, and there is no sibling argument to distinguish them
        # from.
        schema = self._tool(synthetic_template)["input_schema"]
        assert set(schema["properties"]) == {
            "text_readable", "figure_tables_included",
            "expected_relationships"}
        assert schema["additionalProperties"] is False
        # Template-declared `required: true` becomes JSON-Schema `required`,
        # so the API coaxes a complete answer on the first attempt.
        assert schema["required"] == [
            "expected_relationships", "figure_tables_included",
            "text_readable"]

    def test_a_gated_tool_is_refused_until_the_check_lands(
            self, synthetic_template, paper_text, image_labels):
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        blocked = d.dispatch("update_study", {"study": {
            "primary_aim": _env(
                "Assess WDS-9", ["WDS-9 was administered"], "Methods")}})
        assert blocked["status"] == "validation_failed"
        assert [e["code"] for e in blocked["errors"]] == [
            "initial_check_required"]
        # Nothing was applied, so the same call can simply be resubmitted.
        assert record.study == {"notes": None}

        d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        again = d.dispatch("update_study", {"study": {
            "primary_aim": _env(
                "Assess WDS-9", ["WDS-9 was administered"], "Methods")}})
        assert again["status"] == "ok", again

    def test_every_mutation_and_mark_complete_is_gated(
            self, synthetic_template, paper_text, image_labels):
        # The gated set is MUTATING_TOOLS plus mark_complete: a template
        # declaring no REQUIRED initial-check field would otherwise let a run
        # complete having never made the call at all.
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        gated = {
            "update_study": {"study": {}},
            "add_record": {"fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"])}},
            "update_record": {"record_id": "relationship_1", "fields": {}},
            "remove_record": {"record_id": "relationship_1", "reason": "x"},
            "mark_complete": {"quality_check": QUALITY_CHECK_FIELDS},
        }
        assert set(gated) == MUTATING_TOOLS | {"mark_complete"}
        for name, args in gated.items():
            res = d.dispatch(name, args)
            assert [e["code"] for e in res["errors"]] == [
                "initial_check_required"], (name, res)

    def test_looking_and_surrendering_stay_open(
            self, synthetic_template, paper_text, image_labels):
        # An honest surrender and a look at an empty record must both stay
        # available to a model that has got itself stuck before the gate.
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        for name in ("view_summary", "view_study_fields"):
            assert d.dispatch(name, {})["status"] == "ok"
        surrender = d.dispatch("abandon_extraction", {"reason": "no text"})
        assert surrender["status"] == "ok", surrender
        assert record.abandoned_flag is True

    def test_the_block_is_filed_under_the_recording_role(
            self, synthetic_template, paper_text, image_labels):
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        res = d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        assert res["status"] == "ok", res
        assert record.initial_check == {ROLE_EXTRACTOR: INITIAL_CHECK_FIELDS}
        assert res["applied_changes"]["recorded_by"] == ROLE_EXTRACTOR
        # Per-field validation, as with the field-writing tools: a re-call
        # revises, and one bad variable does not cost the good ones.
        revised = d.dispatch("record_initial_check", {
            "expected_relationships": 4, "text_readable": "yes please"})
        assert revised["status"] == "partial"
        assert record.initial_check[ROLE_EXTRACTOR]["expected_relationships"] \
            == 4
        assert record.initial_check[ROLE_EXTRACTOR]["text_readable"] is True

    def test_a_wholly_invalid_check_still_opens_the_gate(
            self, synthetic_template, paper_text, image_labels):
        # The gate asks whether the extractor has looked at its inputs and
        # answered, not whether it answered well: holding the whole run
        # hostage to one malformed property would be a worse failure than a
        # thin initial check.
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        res = d.dispatch("record_initial_check", {"text_readable": "yes"})
        assert res["status"] == "validation_failed"
        assert record.initial_check_recorded is True
        assert d.dispatch("update_study", {"study": {}})["status"] == "ok"

    def test_the_reviewer_cannot_file_one(
            self, synthetic_template, paper_text, image_labels):
        # The tool is not in the reviewer's catalogue, but a model can name a
        # tool it was not given: refuse it rather than let a retrospective
        # report overwrite the extractor's account of the inputs.
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        res = d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS,
                         role=ROLE_REVIEW)
        assert res["status"] == "validation_failed"
        assert [e["code"] for e in res["errors"]] == [
            "tool_not_available_to_role"]
        assert record.initial_check == {}
        assert record.initial_check_recorded is False


class TestUpdateStudy:
    def test_opening_moves_initial_check_then_study(
            self, synthetic_template, paper_text, image_labels):
        # The extraction's opening moves, in the order the engine enforces:
        # the initial check through its own tool, then the study fields. Two
        # separate calls, because a single `update_study` carrying both lets a
        # model start extracting and report on the inputs afterwards.
        record = ExtractionRecord()
        d = ToolDispatcher(record, synthetic_template, paper_text,
                           image_labels)
        opening = d.dispatch("record_initial_check", {
            "text_readable": True,
            "figure_tables_included": True,
            "expected_relationships": 2,
        })
        assert opening["status"] == "ok", opening
        result = d.dispatch("update_study", {
            "study": {
                "primary_aim": _env(
                    "Assess WDS-9 in brackets under load",
                    ["WDS-9 was administered to 348 units under load"],
                    "Methods, paragraph 1",
                ),
                "sample_size": _env(
                    348,
                    ["348 units under load"],
                    "Methods, paragraph 1",
                ),
            },
        })
        assert result["status"] == "ok", result
        # The check block is filed under its author's role, so a reader of the
        # output sees who said it without consulting documentation.
        assert record.initial_check[ROLE_EXTRACTOR]["text_readable"] is True
        assert record.study["primary_aim"]["value"] == \
            "Assess WDS-9 in brackets under load"
        assert record.study["sample_size"]["value"] == 348

    def test_a_check_block_sent_here_is_refused_whole(
            self, synthetic_template, paper_text, image_labels):
        # A model working from habit may still send a check block to
        # update_study. Silently dropping a block it believed it had recorded
        # is exactly the quiet loss the split exists to end, so the whole call
        # is refused and the message names the tool that now owns the block.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "initial_check": {"text_readable": True},
            "study": {"primary_aim": _env(
                "Assess WDS-9", ["WDS-9 was administered"], "Methods")},
        })
        assert result["status"] == "validation_failed"
        assert [e["code"] for e in result["errors"]] == ["block_moved"]
        assert "record_initial_check" in result["errors"][0]["message"]
        # Refused WHOLE: the study field that rode alongside did not land.
        assert a.study == {"notes": None}

    def test_quote_not_in_text_rejects(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "study": {
                "primary_aim": _env(
                    "Assess WDS-9",
                    ["this quote is not in the paper"],
                    "Methods",
                ),
            },
        })
        assert result["status"] == "validation_failed"
        codes = [e["code"] for e in result["errors"]]
        assert "quote_not_in_text" in codes
        # All-or-nothing: nothing was applied, so only the reserved
        # scope-note key is present.
        assert a.study == {"notes": None}

    def test_unknown_field_rejected(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "study": {
                "nonsense_field": _env("x", ["WDS-9"], "Methods"),
            },
        })
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "unknown_field" for e in result["errors"])

    def test_undeclared_reserved_name_rejected(
            self, synthetic_template, paper_text, image_labels):
        # study_id is simply not a declared field in this template. There is no
        # identity / pipeline-managed concept (see
        # tests/extraction/test_no_identity_fields.py) and bibliographic fields
        # are ordinary schema fields, so a name that reads as reserved gets no
        # special handling: it dispatches through the standard `unknown_field`
        # path, exactly like the generic nonsense_field above it.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "study": {
                "study_id": _env("376", ["WDS-9"], "Methods"),
            },
        })
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "unknown_field"
                   for e in result["errors"])

    def test_allow_other_accepts_free_text(
            self, synthetic_template, paper_text, image_labels):
        # publication_type is an allow_other categorical; a value outside
        # the option list is accepted as free text (no companion field).
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "study": {
                "publication_type": _env(
                    "Charity report", ["WDS-9"], "Methods"),
            },
        })
        assert result["status"] == "ok", result
        assert a.study["publication_type"]["value"] == "Charity report"

    def test_allow_other_canonicalises_listed_value(
            self, synthetic_template, paper_text, image_labels):
        # A case variant of a listed option is stored as the option's exact
        # spelling, even on an allow_other field.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_study", {
            "study": {
                "publication_type": _env(
                    "academic paper", ["WDS-9"], "Methods"),
            },
        })
        assert result["status"] == "ok", result
        assert a.study["publication_type"]["value"] == "Academic paper"


class TestAddRelationship:
    def test_add_assigns_id(self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "Unplanned removal", ["unplanned removal was 1.34"],
                    "Results"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        assert result["status"] == "ok", result
        assert result["applied_changes"]["record_id"] == "relationship_1"

    def test_image_sourced_field_ok(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env_with_image("OR", "table_02"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        assert result["status"] == "ok", result

    def test_gate_violation_is_warning(
            self, synthetic_template, paper_text, image_labels):
        # index_tariff set on a Failure state outcome: warning, not error.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "DI-4",
                    ["DI-4 index was used"],
                    "Results"),
                "outcome_category": _env(
                    "Failure state",
                    ["unplanned removal"],
                    "Results"),
                "index_tariff": _env(
                    "DI-4", ["DI-4 index"], "Results"),
            },
        })
        # Warning, not error -> applies.
        assert result["status"] == "ok"
        assert any(w["code"] == "category_gate"
                   for w in result["warnings"])
        assert a.records[0]["index_tariff"]["value"] == "DI-4"

    def test_includes_record_id_rejected(
            self, synthetic_template, paper_text, image_labels):
        # record_id is engine-assigned, never a declared field, so a
        # hallucinated attempt to set it lands in the unknown_field path.
        # Validation is per-field, so the sibling field still applies and the
        # fabricated id is simply not stored.
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("add_record", {
            "fields": {
                "record_id": _env("relationship_99", ["WDS-9"], "Methods"),
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
            },
        })
        assert result["status"] == "partial"
        codes = [e["code"] for e in result["errors"]]
        assert "unknown_field" in codes
        assert a.records[0]["record_id"] == "relationship_1"
        assert a.records[0]["gauge"]["value"] == "WDS-9"


class TestUpdateRelationship:
    def test_update_existing(self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "Unplanned removal", ["unplanned removal was 1.34"],
                    "Results"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        result = d.dispatch("update_record", {
            "record_id": "relationship_1",
            "fields": {
                "effect_size": _env(
                    "1.34", ["odds ratio for unplanned removal was 1.34"],
                    "Results"),
            },
        })
        assert result["status"] == "ok", result
        assert a.records[0]["effect_size"]["value"] == "1.34"

    def test_unknown_id_rejected(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_record", {
            "record_id": "relationship_99",
            "fields": {"gauge": _env("WDS-9", ["WDS-9"], "Methods")},
        })
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "unknown_record"
                   for e in result["errors"])


class TestRemoveRelationship:
    def test_removes(self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "X", ["unplanned removal"], "Results"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        result = d.dispatch("remove_record", {
            "record_id": "relationship_1",
            "reason": "Out of scope on re-read.",
        })
        assert result["status"] == "ok"
        assert a.records == []
        assert result["applied_changes"]["removed_record_id"] == "relationship_1"

    def test_missing_reason(self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "X", ["unplanned removal"], "Results"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        result = d.dispatch("remove_record", {"record_id": "relationship_1"})
        assert result["status"] == "validation_failed"


class TestViewSummary:
    def test_records_carry_build_record_context(
            self, synthetic_template, paper_text, image_labels):
        # view_summary surfaces a per-record `context` string built by
        # build_record_context from the template's checker_context_fields, so
        # the model sees which record each count belongs to. This pins the key
        # name and the derivation at the handler layer; renaming the key or
        # dropping the context would otherwise fail no test.
        from meltiro.checker_prompts import build_record_context
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "Unplanned removal", ["unplanned removal"], "Results"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        # A bare record (no populated context fields) exercises the fallback
        # to the plain record id at this same layer.
        a.records.append({"record_id": "relationship_2"})
        result = d.dispatch("view_summary", {})
        assert result["status"] == "ok"
        records = result["view"]["records"]
        assert [r["record_id"] for r in records] == [
            "relationship_1", "relationship_2"]
        # The emitted context matches build_record_context called with the
        # template's checker_context_fields, descriptors and all.
        assert records[0]["context"] == build_record_context(
            a.records[0], synthetic_template["checker_context_fields"])
        assert records[0]["context"] == (
            "relationship_1 — WDS-9 | Unplanned removal | Failure state")
        # Fallback: the descriptor-free record collapses to the bare id.
        assert records[1]["context"] == "relationship_2"


class TestMarkComplete:
    """The completeness gate, and the quality check that now rides on it.

    `mark_complete` REQUIRES a `quality_check` argument: declaring the
    extraction finished and saying how it went are one act, so there is no
    window in which a run is complete and unassessed. Every call below that
    expects to pass therefore carries `_complete_args()`.
    """

    def _complete_args(self, **overrides):
        """mark_complete arguments answering every REQUIRED quality-check
        variable the synthetic template declares."""
        return {"quality_check": dict(QUALITY_CHECK_FIELDS, **overrides)}

    def _fresh(self, template, paper_text, image_labels):
        """A record and dispatcher with nothing recorded and the gate SHUT.

        Unlike the module-level `_dispatcher`, this class does not latch the
        gate by hand: what mark_complete demands of the initial check is part
        of its subject, so every seeding path below opens the gate through the
        real `record_initial_check` call.
        """
        record = ExtractionRecord()
        return record, ToolDispatcher(record, template, paper_text,
                                      image_labels)

    def _seed_complete(self, dispatcher):
        dispatcher.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        dispatcher.dispatch("update_study", {
            "study": {
                "primary_aim": _env(
                    "Assess WDS-9", ["WDS-9 was administered"], "Methods"),
            },
        })
        dispatcher.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "Unplanned removal",
                    ["unplanned removal"], "Results"),
                "outcome_category": _env(
                    "Failure state",
                    ["unplanned removal"], "Results"),
            },
        })

    def test_complete_passes(self, synthetic_template, paper_text, image_labels):
        a, d = self._fresh(synthetic_template, paper_text, image_labels)
        self._seed_complete(d)
        result = d.dispatch("mark_complete", self._complete_args())
        assert result["status"] == "ok", result
        assert a.mark_complete_flag is True
        # The quality check landed in the same call, under the calling role.
        assert a.quality_check[ROLE_EXTRACTOR] == QUALITY_CHECK_FIELDS

    def test_no_relationships_rejected(
            self, synthetic_template, paper_text, image_labels):
        a, d = self._fresh(synthetic_template, paper_text, image_labels)
        d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        result = d.dispatch("mark_complete", self._complete_args())
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "no_records"
                   for e in result["errors"])

    def test_missing_metadata_rejected(
            self, synthetic_template, paper_text, image_labels):
        # The initial check LANDED (the ordering gate is open) but answered
        # none of the template's required variables, so mark_complete still
        # refuses: the gate asks whether the extractor looked at its inputs,
        # the completeness check asks what it found.
        a, d = self._fresh(synthetic_template, paper_text, image_labels)
        opening = d.dispatch("record_initial_check", {})
        assert opening["status"] == "ok", opening
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env(
                    "X", ["unplanned removal"], "Results"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "Results"),
            },
        })
        result = d.dispatch("mark_complete", self._complete_args())
        assert result["status"] == "validation_failed"
        codes = [e["code"] for e in result["errors"]]
        assert "metadata_required" in codes
        assert a.mark_complete_flag is False

    def test_a_missing_quality_check_is_rejected(
            self, synthetic_template, paper_text, image_labels):
        # The other half of the same gate, on the block that moved here.
        # An ABSENT quality_check fails on its own, before any per-variable
        # rule: "completing the extraction and reporting on how it went are
        # one call" has to hold for a template that marks no quality-check
        # field `required` too, or the guarantee would belong to the config
        # rather than to the engine. This template does declare one, so both
        # errors come back — the unconditional one first.
        a, d = self._fresh(synthetic_template, paper_text, image_labels)
        self._seed_complete(d)
        result = d.dispatch("mark_complete", {})
        assert result["status"] == "validation_failed"
        assert [(e["path"], e["code"]) for e in result["errors"]] == [
            ("quality_check", "quality_check_required"),
            ("quality_check.deviation_from_expectations",
             "metadata_required")]
        # Nothing was recorded by the failed call, the quality check included.
        assert a.mark_complete_flag is False
        assert a.quality_check == {}
        # And the extractor re-calls with it supplied.
        assert d.dispatch(
            "mark_complete", self._complete_args())["status"] == "ok"

    def test_an_unknown_quality_check_variable_is_rejected(
            self, synthetic_template, paper_text, image_labels):
        # The quality check is validated per field against the template, like
        # any other block: an invented variable fails the call rather than
        # being stored as a stray key.
        a, d = self._fresh(synthetic_template, paper_text, image_labels)
        self._seed_complete(d)
        result = d.dispatch("mark_complete", self._complete_args(
            not_a_quality_check_field="x"))
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "unknown_field" for e in result["errors"])
        assert a.mark_complete_flag is False
        assert a.quality_check == {}

    def test_required_gate_is_template_driven(
            self, synthetic_template, paper_text, image_labels):
        # Clearing the template's `required` flags relaxes the gate: the
        # engine names no field itself, so an otherwise-empty extraction
        # passes mark_complete once nothing is declared required. Proves the
        # allowlist moved out of engine code.
        import copy
        relaxed = copy.deepcopy(synthetic_template)
        for block in ("initial_check_fields", "quality_check_fields",
                      "record_fields", "study_fields"):
            for section in relaxed[block]:
                for f in section["fields"]:
                    f["required"] = False
        a, d = self._fresh(relaxed, paper_text, image_labels)
        # The ordering gate is not a required-field rule: it asks only that
        # the call was made, so an empty initial check opens it even here.
        d.dispatch("record_initial_check", {})
        # A single bare record with no fields set at all; nothing is required.
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
            },
        })
        result = d.dispatch("mark_complete", {"quality_check": {}})
        assert result["status"] == "ok", result
        assert a.mark_complete_flag is True

    def test_study_required_flag_enforced(
            self, synthetic_template, paper_text, image_labels):
        # A study-level field flagged required must be non-null at
        # mark_complete: the generic envelope-required check covers study
        # fields, not just records.
        import copy
        tmpl = copy.deepcopy(synthetic_template)
        tmpl["study_fields"][0]["fields"][0]["required"] = True  # primary_aim
        a, d = self._fresh(tmpl, paper_text, image_labels)
        d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
                "outcome_variable": _env("X", ["unplanned removal"], "R"),
                "outcome_category": _env(
                    "Failure state", ["unplanned removal"], "R"),
            },
        })
        # primary_aim (study, required) is still null.
        result = d.dispatch("mark_complete", self._complete_args())
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "study_field_required"
                   for e in result["errors"])


class TestFieldValueSubschema:
    """The API value subschema must agree with the runtime validator.

    A `year` field is an integer; advertising it as a bare `number` let a
    model-emitted 2019.0 pass the API layer then fail the runtime integer
    check, an avoidable failure that fed the retry loop.
    """

    def test_year_advertised_as_integer(self):
        assert _field_value_subschema({"field_type": "year"}) == {
            "type": ["integer", "null"]}

    def test_number_still_advertised_as_number(self):
        assert _field_value_subschema({"field_type": "number"}) == {
            "type": ["number", "null"]}


class TestEntityWordingInResults:
    """Runtime tool_result messages substitute the template-declared record
    entity noun (here 'relationship' / 'relationships') wherever they name
    the entity. Tool names (add_record, update_record, ...) stay generic.
    """

    def test_update_record_unknown_id(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("update_record", {
            "record_id": "relationship_99",
            "fields": {"gauge": _env("WDS-9", ["WDS-9"], "Methods")},
        })
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_record")
        assert "relationship" in msg
        assert "No record with id" not in msg

    def test_remove_record_unknown_id(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("remove_record", {
            "record_id": "relationship_99", "reason": "added by mistake",
        })
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_record")
        assert msg == "No relationship with id relationship_99."

    def test_view_record_unknown_id(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("view_record", {"record_id": "relationship_99"})
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_record")
        assert "No relationship with id" in msg
        assert "Current relationships:" in msg
        assert "No record with id" not in msg
        assert "Current records" not in msg

    def test_mark_complete_no_records(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        # Satisfy both check-block gates so the only failure is "no records".
        d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        result = d.dispatch("mark_complete",
                            {"quality_check": QUALITY_CHECK_FIELDS})
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "no_records")
        assert "At least one relationship must be added" in msg
        # Tool name stays generic and literal.
        assert "add_record" in msg

    def test_mark_complete_record_field_required(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        d.dispatch("record_initial_check", INITIAL_CHECK_FIELDS)
        # Record with only gauge set; the other required fields stay null,
        # so mark_complete flags them.
        d.dispatch("add_record", {
            "fields": {
                "gauge": _env("WDS-9", ["WDS-9 was administered"], "Methods"),
            },
        })
        result = d.dispatch("mark_complete",
                            {"quality_check": QUALITY_CHECK_FIELDS})
        msgs = [e["message"] for e in result["errors"]
                if e["code"] == "record_field_required"]
        assert msgs
        assert all("on every relationship" in m for m in msgs)

    def test_add_record_unknown_field(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("add_record", {
            "fields": {"not_a_field": _env("x", ["WDS-9"], "Methods")},
        })
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_field")
        assert "not a known relationship field" in msg
        assert "not a known record field" not in msg

    def test_study_field_on_record_hint(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        # primary_aim is a STUDY-level field in the synthetic template.
        result = d.dispatch("add_record", {
            "fields": {"primary_aim": _env("x", ["WDS-9"], "Methods")},
        })
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_field")
        assert "not on a relationship" in msg
        # Tool name in the hint stays generic.
        assert "update_study.study" in msg

    def test_wording_tracks_template_entity(
            self, synthetic_template, paper_text, image_labels):
        # Substitution is template-driven, not hardcoded to "relationship":
        # a template declaring a different entity yields that noun.
        template = dict(synthetic_template)
        template["record_entity"] = {
            "singular": "finding",
            "plural": "findings",
            "description": "a reported finding",
        }
        a, d = _dispatcher(template, paper_text, image_labels)
        result = d.dispatch("update_record", {
            "record_id": "finding_1",
            "fields": {"gauge": _env("WDS-9", ["WDS-9"], "Methods")},
        })
        msg = next(e["message"] for e in result["errors"]
                   if e["code"] == "unknown_record")
        assert "No finding with id finding_1" in msg


class TestUnknownTool:
    def test_unknown_tool(self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch("nope", {})
        assert result["status"] == "validation_failed"
        assert any(e["code"] == "unknown_tool" for e in result["errors"])

    def test_meta_passed_through(
            self, synthetic_template, paper_text, image_labels):
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch(
            "mark_complete", {"quality_check": QUALITY_CHECK_FIELDS},
            meta={"tool_call_budget_remaining": 17},
        )
        # No relationships -> validation_failed, but meta still propagates.
        # The echoed key is underscore-prefixed so `result_to_model_text` treats
        # it as UI-only telemetry and strips it from the model-facing payload.
        assert result.get("_tool_call_budget_remaining") == 17
        assert "tool_call_budget_remaining" not in result

    def test_budget_stripped_from_model_facing_text(
            self, synthetic_template, paper_text, image_labels):
        # The remaining tool-call budget is telemetry only: it rides in the
        # dispatch result (and thus the event log) but must not reach the
        # model, since the cap is excluded from every fingerprint.
        from meltiro.session import result_to_model_text
        a, d = _dispatcher(synthetic_template, paper_text, image_labels)
        result = d.dispatch(
            "view_summary", {},
            meta={"tool_call_budget_remaining": 5},
        )
        assert result.get("_tool_call_budget_remaining") == 5
        model_text = result_to_model_text(result)
        parsed = json.loads(model_text)
        assert "_tool_call_budget_remaining" not in parsed
        assert "tool_call_budget_remaining" not in parsed


class TestMutatingToolClassification:
    """`MUTATING_TOOLS` is the single answer to "did this stage actually edit
    anything". The final review runs its post-review checker pass only on a
    real mutation, so a read-only `view_*` call must never be accounted as an
    edit (it would fire a checker pass over an unchanged snapshot).
    """

    # The structural half (every catalogue tool lands on one side of the line)
    # is pinned once, by TestToolDefinitions.
    # test_mutating_tools_classifies_the_whole_catalogue; this class carries the
    # behavioural half that the set actually describes what the tools do.

    def test_view_tools_really_do_not_mutate(
            self, synthetic_template, paper_text, image_labels):
        # The behavioural half: the read-only tools leave the extraction output
        # byte-identical, which is why they are not edits.
        record, d = _dispatcher(synthetic_template, paper_text, image_labels)
        added = d.dispatch("add_record", {"fields": {
            "gauge": _env("WDS-9", ["The WDS-9 was administered"]),
            "outcome_variable": _env("unplanned removal",
                                     ["odds ratio for unplanned removal"]),
            "outcome_category": _env("Failure state",
                                     ["odds ratio for unplanned removal"]),
        }})
        assert added["status"] == "ok", added
        before = json.dumps(record.to_dict(), sort_keys=True)

        for name, args in (("view_summary", {}),
                           ("view_study_fields", {}),
                           ("view_record", {"record_id": "relationship_1"})):
            result = d.dispatch(name, args)
            assert result["status"] == "ok", (name, result)
            assert name not in MUTATING_TOOLS

        assert json.dumps(record.to_dict(), sort_keys=True) == before
