"""Notes are not a kind of field, at either level.

A note is never an invalid note, and never a note unsupported by evidence, so
notes sit outside validation and outside the checker. They exist at two levels,
neither declared in the template:

  - the FIELD note, a `notes` slot in every envelope alongside `value` and
    `evidence`;
  - the SCOPE note, a reserved `notes` key on the study block and on each
    record, beside `record_id`.

This file pins the template-load half of that: a field-level `notes:` key, the
reserved variable name, the section-level `notes:` block, and the key's
absence from the field-catalogue hash. The runtime halves live in
tests/agentic_extraction/test_scope_notes.py (storage and the tool arguments),
test_field_notes.py (the envelope), and test_checker_notes_filter.py (what the
checker sees).
"""

import pytest

from meltiro.fingerprint import field_catalogue_hash
from meltiro.template import _parse_field, _parse_sections


def _field(**extra):
    f = {"variable": "study_context", "description": "Some context",
         "type": "string", "evidence": "optional"}
    f.update(extra)
    return f


class TestNotesIsNotAFieldKey:
    """There is no flag that turns a field into a notes field, so `notes:` on a
    field is a key the schema does not define and the field allowlist refuses
    it. Whatever value it carries: nothing reads it, so accepting it would
    declare an ordinary field where a notes field was meant."""

    @pytest.mark.parametrize("value", [True, False])
    def test_a_notes_key_is_a_load_error_whatever_its_value(self, value):
        with pytest.raises(ValueError) as excinfo:
            _parse_field(_field(notes=value), envelope=True,
                         section_name="Notes")
        message = str(excinfo.value)
        assert "unknown field key(s)" in message
        assert "'notes'" in message

    def test_a_notes_key_is_rejected_on_a_bare_value_field_too(self):
        # The bare-value check blocks carry no notes at all, so the rule reads
        # the same wherever an author looks.
        with pytest.raises(ValueError, match="unknown field key"):
            _parse_field({"variable": "extraction_issues",
                          "description": "d", "type": "string",
                          "notes": True},
                         envelope=False, section_name="Process")

    def test_notes_help_key_rejected(self):
        # Per-field guidance is `extraction_instruction`, which renders into
        # the tool schema; a second guidance key would reach no model.
        with pytest.raises(ValueError) as excinfo:
            _parse_field(_field(notes_help="what to write"),
                         envelope=True, section_name="Notes")
        assert "unknown field key(s)" in str(excinfo.value)
        assert "'notes_help'" in str(excinfo.value)

    def test_an_ordinary_field_still_parses(self):
        # The allowlist rejects the KEY, not free-text fields in general.
        spec = _parse_field(_field(), envelope=True, section_name="Context")
        assert spec["evidence"] == "optional"
        assert "notes" not in spec


class TestReservedVariableName:
    @pytest.mark.parametrize("envelope", [True, False])
    def test_a_field_named_notes_is_rejected_in_every_scope(self, envelope):
        raw = {"variable": "notes", "description": "d", "type": "string"}
        if envelope:
            raw["evidence"] = "optional"
        with pytest.raises(ValueError, match="reserved key"):
            _parse_field(raw, envelope=envelope, section_name="S")

    def test_the_message_explains_the_collision(self):
        with pytest.raises(ValueError) as excinfo:
            _parse_field({"variable": "notes", "description": "d",
                          "type": "string", "evidence": "optional"},
                         envelope=True, section_name="S")
        assert "would collide with the scope note" in str(excinfo.value)

    def test_a_name_merely_containing_notes_is_fine(self):
        # Only the exact reserved key collides. `general_notes` is an ordinary
        # variable name and stays available.
        spec = _parse_field(
            {"variable": "general_notes", "description": "d",
             "type": "string", "evidence": "optional"},
            envelope=True, section_name="S")
        assert spec["variable"] == "general_notes"


class TestSectionLevelNotesRejected:
    def test_section_level_notes_block_rejected(self):
        with pytest.raises(ValueError,
                           match=r"unknown section key\(s\) \['notes'\]"):
            _parse_sections(
                [{"section": "Study Notes", "notes": {"help": "h"}}],
                envelope=True)

    def test_the_message_does_not_point_at_a_notes_field(self):
        # `notes: true` on a field is itself a load error, so the message must
        # NOT name it as the replacement.
        with pytest.raises(ValueError) as excinfo:
            _parse_sections(
                [{"section": "Study Notes", "notes": {"help": "h"}}],
                envelope=True)
        assert "notes: true" not in str(excinfo.value)


class TestNotesOutOfTheFieldCatalogueHash:
    def _tmpl(self, **field_extra):
        field = {"variable": "study_context", "description": "context",
                 "field_type": "string", "evidence": "optional"}
        field.update(field_extra)
        return {
            "study_fields": [{"section": "S", "fields": [field]}],
            "record_fields": [],
            "initial_check_fields": [],
            "quality_check_fields": [],
        }

    def test_a_stray_notes_key_does_not_move_the_catalogue_hash(self):
        # The flag does not exist, so it is not hashed. A parsed field dict
        # cannot carry it (the loader rejects it), and the hash must NOT depend
        # on one turning up from some other source.
        assert field_catalogue_hash(self._tmpl()) == \
            field_catalogue_hash(self._tmpl(notes=True))

    def test_evidence_still_moves_the_catalogue_hash(self):
        # The control: an attribute that really does change what the checker
        # demands still moves the hash.
        assert field_catalogue_hash(self._tmpl()) != \
            field_catalogue_hash(self._tmpl(evidence="required"))
