"""Template-load semantics for `allow_other`.

A categorical field (one carrying an `options:` list) is a hard enum by
default. `allow_other: true` turns it into an open list that also accepts
free text. That flag is the whole of the soft-enum machinery: a literal
"Other" option, a second mode key, and misuse of `allow_other` all fail
loudly at load.
"""

import pytest

from meltiro.template import _parse_field


def _parse(field):
    # envelope=False (initial_check/quality_check style) avoids the
    # evidence-required check, keeping these tests focused on the flag logic.
    return _parse_field(field, envelope=False, section_name="test")


def test_categorical_defaults_to_hard_enum():
    f = _parse({"variable": "v", "description": "d", "options": ["A", "B"]})
    assert f["field_type"] == "categorical"
    assert f["allow_other"] is False


def test_allow_other_true_accepted():
    f = _parse({"variable": "v", "description": "d", "options": ["A", "B"],
                "allow_other": True})
    assert f["allow_other"] is True


def test_literal_other_option_rejected():
    with pytest.raises(ValueError, match="allow_other"):
        _parse({"variable": "v", "description": "d",
                "options": ["A", "B", "Other"]})


def test_literal_other_option_rejected_case_insensitive():
    with pytest.raises(ValueError, match="allow_other"):
        _parse({"variable": "v", "description": "d",
                "options": ["A", " other "]})


def test_a_separate_enum_mode_key_is_rejected():
    # `allow_other` is the only switch. A second one would have to agree with
    # it, and a field allowlist that let it through would leave the disagreement
    # silent.
    with pytest.raises(ValueError, match="unknown field key"):
        _parse({"variable": "v", "description": "d", "options": ["A", "B"],
                "enum_mode": "soft"})


def test_allow_other_on_non_categorical_rejected():
    with pytest.raises(ValueError, match="allow_other"):
        _parse({"variable": "v", "description": "d", "type": "string",
                "allow_other": True})


def test_allow_other_with_canonical_reference_rejected():
    # The combination is impossible by construction: canonical_reference
    # requires a string or string_list type, so it can never sit on the
    # categorical (options) field that allow_other requires. The type rule
    # rejects the pairing.
    with pytest.raises(ValueError, match="canonical_reference.*options"):
        _parse_field({"variable": "v", "description": "d",
                      "options": ["A", "B"],
                      "canonical_reference": "gauge_list", "allow_other": True,
                      "evidence": "required"},
                     envelope=True, section_name="test")


def test_allow_other_non_bool_rejected():
    with pytest.raises(ValueError, match="allow_other"):
        _parse({"variable": "v", "description": "d", "options": ["A", "B"],
                "allow_other": "yes"})
