"""allow_other behaviour: tool schema, validator, checker briefing.

A categorical field without `allow_other` is a hard enum: the tool schema
carries an `enum` constraint and the validator rejects out-of-list values
(canonicalising case/whitespace variants). An `allow_other` field advertises
a plain string, accepts any non-empty free text, and canonicalises a value
that matches a listed option. The checker is briefed differently depending
on whether the stored value is a listed option or free text.
"""

from meltiro.tools import _field_value_subschema, _field_description
from meltiro.validators import validate_envelope, _check_value_type
from meltiro.checker_prompts import (
    build_checker_user_message,
    _render_allowed_values_block,
)


def _hard(options):
    return {
        "variable": "x",
        "field_type": "categorical",
        "options": options,
        "allow_other": False,
        "evidence": "required",
        "description": "A categorical field.",
        "extraction_instruction": None,
    }


def _open(options):
    spec = _hard(options)
    spec["allow_other"] = True
    return spec


PAPER = "The study used a novel bonded-collar variant in a bracket batch."


class TestOtherSpecifyWording:
    """Pin the literal "Other (specify)" affordance at each source function, so
    a wording edit is caught where it is written."""

    def test_extractor_description_tail_lists_options_then_other_specify(self):
        desc = _field_description(_open(["Cost", "Service life"]))
        assert "Options: Cost; Service life; Other (specify)" in desc

    def test_hard_enum_description_has_no_other_specify(self):
        # A hard enum carries its options as the schema `enum`, not a tail, and
        # offers no "Other" escape hatch.
        assert "Other (specify)" not in _field_description(_hard(["A", "B"]))

    def test_checker_allowed_values_block_free_text_branch(self):
        # Value is free text outside the list: the open-list brief, ending in
        # the explicit affordance.
        block = _render_allowed_values_block(
            _open(["Cost", "Service life"]), "bonded-collar variant")
        assert block.endswith("Cost | Service life | Other (specify)")

    def test_checker_allowed_values_block_listed_value_branch(self):
        # Value is a listed option: the hard-choice brief, ending in the
        # explicit affordance.
        block = _render_allowed_values_block(
            _open(["Cost", "Service life"]), "Cost")
        assert block.endswith("Cost | Service life | Other (specify)")

    def test_checker_hard_enum_block_has_no_other_specify(self):
        block = _render_allowed_values_block(_hard(["A", "B"]), "A")
        assert "Other (specify)" not in block


class TestToolSchema:
    def test_hard_enum_emits_strict_enum_with_null(self):
        assert _field_value_subschema(_hard(["A", "B"])) == \
            {"enum": ["A", "B", None]}

    def test_allow_other_emits_string_or_null(self):
        assert _field_value_subschema(_open(["A", "B"])) == \
            {"type": ["string", "null"]}

    def test_allow_other_description_lists_options_with_other_specify(self):
        desc = _field_description(_open(["Academic paper", "Government report"]))
        assert "Options:" in desc
        assert "Academic paper" in desc
        assert "Government report" in desc
        # The explicit escape-hatch affordance, presentation only.
        assert "Other (specify)" in desc
        # Open-list semantics are stated once in the extractor system prompt,
        # not per field: no explanatory sentence rides in the description.
        assert "free text" not in desc.lower()
        assert "if none" not in desc.lower()

    def test_hard_enum_description_omits_option_list(self):
        # Hard enum options live in the schema `enum`, not the description, and
        # a hard enum carries no "Other (specify)" affordance.
        desc = _field_description(_hard(["Academic paper", "Government report"]))
        assert "free text" not in desc.lower()
        assert "Other (specify)" not in desc
        assert "Academic paper" not in desc


class TestValidator:
    def test_hard_enum_rejects_out_of_list(self):
        errs = _check_value_type("C", _hard(["A", "B"]), "x")
        assert any(e["code"] == "invalid_option" for e in errs)

    def test_hard_enum_accepts_listed_value(self):
        assert _check_value_type("A", _hard(["A", "B"]), "x") == []

    def test_allow_other_accepts_free_text(self):
        assert _check_value_type("bonded-collar fitting",
                                 _open(["A", "B"]), "x") == []

    def test_allow_other_rejects_empty_string(self):
        errs = _check_value_type("   ", _open(["A", "B"]), "x")
        assert any(e["code"] == "empty_value" for e in errs)

    def test_allow_other_rejects_non_string(self):
        errs = _check_value_type(123, _open(["A", "B"]), "x")
        assert any(e["code"] == "type_mismatch" for e in errs)

    def test_allow_other_canonicalises_listed_value_on_store(self):
        env = {"value": "academic paper",
               "evidence": "<q>bonded-collar variant</q>"}
        errs = validate_envelope(
            env, _open(["Academic paper", "Government report"]),
            paper_text=PAPER, image_labels=set(), path_prefix="study.x")
        assert errs == []
        assert env["value"] == "Academic paper"

    def test_allow_other_leaves_free_text_untouched_on_store(self):
        env = {"value": "bonded-collar variant",
               "evidence": "<q>bonded-collar variant</q>"}
        errs = validate_envelope(
            env, _open(["Academic paper", "Government report"]),
            paper_text=PAPER, image_labels=set(), path_prefix="study.x")
        assert errs == []
        assert env["value"] == "bonded-collar variant"


class TestCheckerBriefing:
    def _text(self, spec, value, checker_partials_dir):
        blocks = build_checker_user_message(
            field_path="study.x",
            field_spec=spec,
            envelope={"value": value,
                      "evidence": "<q>bonded-collar variant</q>"},
            identity_context="Summary: ...",
            image_labels=set(),
            partials_dir=checker_partials_dir,
        )
        return "\n".join(b.get("text", "") for b in blocks)

    def test_allow_other_free_text_asks_whether_option_fits(
            self, checker_partials_dir):
        text = self._text(_open(["Cost", "Service life"]),
                          "bonded-collar variant", checker_partials_dir)
        assert "Typical values" in text
        assert "more appropriate" in text
        assert "Cost" in text
        # The escape-hatch affordance is presented in the free-text branch too.
        assert "Other (specify)" in text

    def test_allow_other_listed_value_briefs_hard_choice(
            self, checker_partials_dir):
        text = self._text(_open(["Cost", "Service life"]),
                          "Cost", checker_partials_dir)
        assert "hard choice" in text
        assert "Allowed values" in text
        # ... and in the listed-value branch.
        assert "Other (specify)" in text

    def test_hard_enum_briefs_allowed_values(
            self, checker_partials_dir):
        text = self._text(_hard(["Bench test", "Field trial"]), "Field trial",
                          checker_partials_dir)
        assert "Allowed values" in text
        assert "Bench test" in text
