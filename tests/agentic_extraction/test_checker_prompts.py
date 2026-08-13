"""Tests for the per-field checker prompt assembly."""

from meltiro.checker_prompts import (
    build_checker_system_text,
    build_checker_user_message,
    build_record_context,
    system_message_blocks,
)


def _spec_from_template(template, block_key, variable):
    from meltiro.template import iter_fields
    for f in iter_fields(template[block_key]):
        if f["variable"] == variable:
            return f
    raise KeyError(variable)


class TestSystemText:
    def test_describes_verdict_vocabulary(self, synthetic_template,
                                          checker_system_path):
        # Field-specific text lives in the per-field user message, NOT the
        # system prompt. The system prompt is generic: it describes the
        # checker's role, the inputs it receives, and the verdict
        # vocabulary.
        txt = build_checker_system_text(
            system_prompt_path=checker_system_path,
            max_checks_per_field=2,
            reference_lists={"gauge_list": []})
        assert "verdict" in txt.lower()
        assert "ok" in txt and "challenge" in txt
        assert "evidence" in txt.lower()

    def test_no_field_catalogue_bloat(self, synthetic_template,
                                      checker_system_path):
        # A field catalogue in the system prompt would bloat cached tokens
        # with information the checker never needs (sibling fields, fields
        # outside the one under review). Its absence is asserted
        # explicitly, so putting it back has to be a deliberate act.
        txt = build_checker_system_text(
            system_prompt_path=checker_system_path,
            max_checks_per_field=2,
            reference_lists={"gauge_list": []})
        assert "primary_aim" not in txt
        assert "outcome_category" not in txt

    def test_cache_control_wrapper(self, synthetic_template,
                                   checker_system_path):
        txt = build_checker_system_text(
            system_prompt_path=checker_system_path,
            max_checks_per_field=2,
            reference_lists={"gauge_list": []})
        blocks = system_message_blocks(txt)
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}


# A representative context-field order, in the shape a config declares under
# records.<entity>.checker_context_fields. The engine reads the order from the
# template (build_record_context takes it as an argument) and NEVER from a
# constant, so this list is the tests' own.
_CONTEXT_VARS = [
    "gauge", "gauge_score_format", "outcome_variable", "outcome_category",
    "statistical_method", "effect_type", "subgroup",
]


class TestRecordContext:
    def test_leads_with_record_id(self):
        rel = {"record_id": "relationship_3", "gauge": {"value": "WDS-9"}}
        ctx = build_record_context(rel, _CONTEXT_VARS)
        # The engine record id is the identifier, and it leads the label.
        assert ctx.startswith("relationship_3")
        assert "relationship_3 —" in ctx
        assert "WDS-9" in ctx

    def test_full_context(self):
        rel = {
            "record_id": "relationship_1",
            "gauge": {"value": "WDS-9"},
            "gauge_score_format": {"value": "Total score"},
            "outcome_variable": {"value": "Unplanned removal"},
            "outcome_category": {"value": "Failure state"},
            "statistical_method": {"value": "Logistic regression"},
            "effect_type": {"value": "OR"},
            "subgroup": {"value": "Load-bearing brackets only"},
        }
        ctx = build_record_context(rel, _CONTEXT_VARS)
        assert "WDS-9" in ctx
        assert "Unplanned removal" in ctx
        assert "Load-bearing brackets only" in ctx
        # Order: gauge first, subgroup last.
        assert ctx.index("WDS-9") < ctx.index("Unplanned removal")
        assert (ctx.index("Unplanned removal")
                < ctx.index("Load-bearing brackets only"))

    def test_order_follows_the_supplied_list(self):
        # The order is the template's, not any fixed order: reverse the list
        # and the context reads back-to-front.
        rel = {
            "record_id": "relationship_1",
            "gauge": {"value": "WDS-9"},
            "subgroup": {"value": "Load-bearing brackets only"},
        }
        ctx = build_record_context(rel, list(reversed(_CONTEXT_VARS)))
        assert ctx.index("Load-bearing brackets only") < ctx.index("WDS-9")

    def test_fields_outside_the_context_list_are_ignored(self):
        rel = {
            "record_id": "relationship_1",
            "gauge": {"value": "WDS-9"},
            "effect_size": {"value": "1.34"},  # not a context field
        }
        ctx = build_record_context(rel, _CONTEXT_VARS)
        assert "WDS-9" in ctx
        assert "1.34" not in ctx

    def test_missing_components_skipped(self):
        rel = {
            "record_id": "relationship_1",
            "gauge": {"value": "WDS-9"},
            "outcome_variable": {"value": "DI-4"},
            # other context fields missing entirely
        }
        ctx = build_record_context(rel, _CONTEXT_VARS)
        assert "WDS-9" in ctx
        assert "DI-4" in ctx
        # No empty pipes between dropped fields.
        assert "| |" not in ctx

    def test_all_null_falls_back_to_bare_record_id(self):
        rel = {"record_id": "relationship_7"}
        ctx = build_record_context(rel, _CONTEXT_VARS)
        # No populated context fields: the bare record id, no separator.
        assert ctx == "relationship_7"

    def test_empty_context_list_is_bare_record_id(self):
        rel = {"record_id": "relationship_2", "gauge": {"value": "WDS-9"}}
        ctx = build_record_context(rel, [])
        assert ctx == "relationship_2"


class TestUserMessage:
    def test_study_field_text_sourced(self, synthetic_template,
                                      checker_partials_dir):
        spec = _spec_from_template(
            synthetic_template, "study_fields", "primary_aim")
        envelope = {
            "value": "Assess WDS-9 in brackets under load",
            "evidence": "<q>WDS-9 was administered to 348 units</q>",
        }
        blocks = build_checker_user_message(
            field_path="study.primary_aim",
            field_spec=spec,
            envelope=envelope,
            identity_context=(
                "Abstract: A study of WDS-9 in brackets under load."),
            image_labels=set(),
            partials_dir=checker_partials_dir,
        )
        assert len(blocks) == 1
        text = blocks[0]["text"]
        assert "primary_aim" in text
        assert "A study of WDS-9 in brackets under load." in text
        assert "WDS-9 was administered" in text
        assert "Assess WDS-9 in brackets under load" in text

    def test_relationship_field_with_record_context(self, synthetic_template,
                                                     checker_partials_dir):
        spec = _spec_from_template(
            synthetic_template, "record_fields", "effect_size")
        envelope = {
            "value": "1.34",
            "evidence": "<q>odds ratio for unplanned removal was 1.34</q>",
        }
        blocks = build_checker_user_message(
            field_path="record.relationship_1.effect_size",
            field_spec=spec,
            envelope=envelope,
            identity_context="WDS-9 | Unplanned removal | Failure state",
            image_labels=set(),
            partials_dir=checker_partials_dir,
        )
        text = blocks[0]["text"]
        assert "WDS-9 | Unplanned removal | Failure state" in text
        assert "1.34" in text

    def test_multi_quote_evidence(self, synthetic_template,
                                  checker_partials_dir):
        spec = _spec_from_template(
            synthetic_template, "study_fields", "primary_aim")
        envelope = {
            "value": "X",
            "evidence": (
                "<q>first quote</q> <q>second quote</q> <q>third quote</q>"
            ),
        }
        blocks = build_checker_user_message(
            field_path="study.primary_aim",
            field_spec=spec,
            envelope=envelope,
            identity_context="ctx",
            image_labels=set(),
            partials_dir=checker_partials_dir,
        )
        text = blocks[0]["text"]
        assert "first quote" in text
        assert "second quote" in text
        assert "third quote" in text

    def test_image_sourced_attaches_image(self, synthetic_template, tmp_path,
                                          checker_partials_dir):
        # Set up a fake figure file and pass it via the bundle figures map.
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        table_png = tmp_path / "table_02.png"
        table_png.write_bytes(png_bytes)

        spec = _spec_from_template(
            synthetic_template, "record_fields", "effect_size")
        envelope = {
            "value": "1.34",
            "evidence": "<img>table_02</img>",
        }
        blocks = build_checker_user_message(
            field_path="record.relationship_1.effect_size",
            field_spec=spec,
            envelope=envelope,
            identity_context="WDS-9 | Unplanned removal",
            image_labels={"table_02"},
            partials_dir=checker_partials_dir,
            figures={"table_02": table_png},
        )
        # Two prefixed blocks (label + image) before the text.
        assert any(b.get("type") == "image" for b in blocks)
        # Text block describes the image situation.
        text = next(b["text"] for b in blocks if b.get("type") == "text"
                    and "Field under review" in b["text"])
        assert "table_02" in text
        assert "treat it AS the evidence" in text

    def test_a_recheck_sees_a_genuinely_fresh_context(
            self, synthetic_template, checker_partials_dir):
        # A field checked a second time after a revision is checked from
        # scratch: nothing about the earlier check reaches the new message,
        # so the checker judges the new value on its own merits rather than
        # re-arguing with itself. The message is a function of the envelope
        # alone, and there is no slot for anything else.
        spec = _spec_from_template(
            synthetic_template, "study_fields", "primary_aim")
        blocks = build_checker_user_message(
            field_path="study.primary_aim",
            field_spec=spec,
            envelope={"value": "X", "evidence": "<q>quote</q>", "notes": None},
            identity_context="ctx",
            image_labels=set(),
            partials_dir=checker_partials_dir,
        )
        text = blocks[0]["text"]
        for absent in ("Prior", "prior", "round", "challenge", "previous",
                       "Re-evaluate", "rebuttal"):
            assert absent not in text


class TestFieldNoteBlock:
    """The `{notes_block}` slot: the checker DOES see the field's own note.

    A field note is the written-down reasoning a human preparing an extraction
    for checking would hand over with the value, so withholding it would leave
    the checker judging a value whose stated grounds it cannot see. The scope
    notes (study, record) are a different thing and never reach the checker;
    see test_checker_notes_filter.py.
    """

    def _text(self, template, path, envelope, **kwargs):
        blocks = build_checker_user_message(
            field_path="study.primary_aim",
            field_spec=_spec_from_template(template, "study_fields",
                                           "primary_aim"),
            envelope=envelope,
            identity_context="ctx",
            image_labels=set(),
            partials_dir=path,
            **kwargs,
        )
        return blocks[0]["text"]

    def test_a_note_is_rendered_under_its_own_heading(
            self, synthetic_template, checker_partials_dir):
        text = self._text(synthetic_template, checker_partials_dir, {
            "value": "X",
            "evidence": "<q>quote</q>",
            "notes": "read off the third column of table 2",
        })
        assert "Extractor's note on this field" in text
        assert "read off the third column of table 2" in text
        # Framed as commentary, so the checker does not read it as evidence.
        assert "not evidence" in text

    def test_no_note_leaves_no_trace(self, synthetic_template,
                                     checker_partials_dir):
        text = self._text(synthetic_template, checker_partials_dir, {
            "value": "X", "evidence": "<q>quote</q>", "notes": None,
        })
        assert "Extractor's note" not in text
        assert "{notes_block}" not in text

    def test_an_absent_notes_key_leaves_no_trace(
            self, synthetic_template, checker_partials_dir):
        text = self._text(synthetic_template, checker_partials_dir, {
            "value": "X", "evidence": "<q>quote</q>",
        })
        assert "Extractor's note" not in text
        assert "{notes_block}" not in text

    def test_a_whitespace_only_note_is_no_note(
            self, synthetic_template, checker_partials_dir):
        text = self._text(synthetic_template, checker_partials_dir, {
            "value": "X", "evidence": "<q>quote</q>", "notes": "   \n  ",
        })
        assert "Extractor's note" not in text

    def test_the_empty_branch_leaves_the_message_ending_at_the_value(
            self, synthetic_template, checker_partials_dir):
        # The shipped template puts the slot on its own line and it is the last
        # line, so a field with no note produces a message that simply ends at
        # the value, with no empty heading trailing it.
        text = self._text(
            synthetic_template, checker_partials_dir,
            {"value": "X", "evidence": "<q>quote</q>", "notes": None})
        assert text.endswith('## Value claimed by the extractor\n\n"X"\n')

    def test_evidence_prose_is_still_withheld(
            self, synthetic_template, checker_partials_dir):
        # The note is the sanctioned channel for reasoning. Prose smuggled
        # inside the evidence string stays withheld, so the two rules do not
        # contradict: reasoning reaches the checker via the note or not at all.
        text = self._text(synthetic_template, checker_partials_dir, {
            "value": "X",
            "evidence": "<q>quote</q> this prose is my own argument",
            "notes": None,
        })
        assert "this prose is my own argument" not in text
