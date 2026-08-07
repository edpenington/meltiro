"""Template rules for canonical_reference.

canonical_reference is valid only on envelope fields (the bare-value
initial_check / quality_check blocks never run reference validation, so
accepting it there would leave it silently unenforced) and only on
`type: string` (one name) or `type: string_list` (a real JSON array of
names). A list of reference names IS a `type: string_list` field, so there is
no separate multi-value flag to set: a field key spelling one is refused by
the field allowlist wherever it appears.
"""

import pytest

from meltiro.template import _parse_field


def test_canonical_reference_on_non_envelope_rejected():
    with pytest.raises(ValueError, match="canonical_reference.*non-envelope"):
        _parse_field({"variable": "v", "description": "d", "type": "string",
                      "canonical_reference": "gauge_list"},
                     envelope=False, section_name="test")


def test_multivalue_key_rejected_on_envelope_field():
    with pytest.raises(ValueError, match="unknown field key"):
        _parse_field({"variable": "v", "description": "d", "type": "string",
                      "canonical_reference": "gauge_list", "multivalue": True,
                      "evidence": "required"},
                     envelope=True, section_name="test")


def test_multivalue_key_rejected_on_non_envelope_field():
    with pytest.raises(ValueError, match="unknown field key"):
        _parse_field({"variable": "v", "description": "d", "type": "string",
                      "multivalue": True},
                     envelope=False, section_name="test")


def test_multivalue_false_also_rejected():
    # Presence of the key is the error, regardless of value: nothing reads it,
    # so a field declaring it would be validated as a single name while the
    # template claims otherwise.
    with pytest.raises(ValueError, match="unknown field key"):
        _parse_field({"variable": "v", "description": "d", "type": "string",
                      "multivalue": False, "evidence": "required"},
                     envelope=True, section_name="test")


def test_envelope_string_field_accepts_canonical_reference():
    f = _parse_field({"variable": "v", "description": "d", "type": "string",
                      "canonical_reference": "gauge_list",
                      "evidence": "required"},
                     envelope=True, section_name="test")
    assert f["canonical_reference"] == "gauge_list"
    assert f["field_type"] == "string"
    assert "multivalue" not in f


def test_envelope_string_list_field_accepts_canonical_reference():
    f = _parse_field({"variable": "v", "description": "d",
                      "type": "string_list",
                      "canonical_reference": "gauge_list",
                      "evidence": "required"},
                     envelope=True, section_name="test")
    assert f["canonical_reference"] == "gauge_list"
    assert f["field_type"] == "string_list"


def test_canonical_reference_on_integer_field_rejected():
    # Reference validation only runs on string values, so any other type
    # would leave the key silently unenforced.
    with pytest.raises(ValueError, match="canonical_reference.*string"):
        _parse_field({"variable": "v", "description": "d", "type": "integer",
                      "canonical_reference": "gauge_list",
                      "evidence": "required"},
                     envelope=True, section_name="test")


def test_canonical_reference_on_categorical_field_rejected():
    with pytest.raises(ValueError, match="canonical_reference.*options"):
        _parse_field({"variable": "v", "description": "d",
                      "options": ["A", "B"],
                      "canonical_reference": "gauge_list",
                      "evidence": "required"},
                     envelope=True, section_name="test")
