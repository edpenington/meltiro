"""Tests for the strengthened categorical validator messages."""

from meltiro.validators import validate_envelope


def _spec(variable, options, allow_other=False):
    return {
        "variable": variable,
        "field_type": "categorical",
        "options": options,
        "allow_other": allow_other,
        "description": "",
        "extraction_instruction": None,
    }


PAPER = "Some study text"


def test_invalid_option_message_lists_all_options():
    errs = validate_envelope(
        {"value": "Maybe", "evidence": ["Some study text"],
         "source": "Methods"},
        _spec("foo", ["Yes", "No", "Unclear"]),
        paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
    )
    msg = next(e["message"] for e in errs if e["code"] == "invalid_option")
    assert "Yes" in msg
    assert "No" in msg
    assert "Unclear" in msg
    assert "Maybe" in msg
    # And it says what to do instead, not ONLY what failed.
    assert "null" in msg.lower()
    assert "do not" in msg.lower() or "do NOT" in msg


def test_case_mismatch_canonicalises_to_option_spelling():
    # The model wrote "yes" (lower-case) for a "Yes"|"No" categorical. A
    # pure case difference is NOT an error: it canonicalises to the option's
    # exact spelling on store.
    env = {"value": "yes", "evidence": ["Some study text"],
           "source": "Methods"}
    errs = validate_envelope(
        env, _spec("foo", ["Yes", "No"]),
        paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
    )
    assert not any(e["code"] == "invalid_option" for e in errs)
    assert env["value"] == "Yes"


def test_whitespace_mismatch_canonicalises_to_option_spelling():
    # Internal-whitespace and surrounding-whitespace differences also
    # canonicalise rather than fail.
    env = {"value": " cost or   resource use ",
           "evidence": ["Some study text"], "source": "Methods"}
    errs = validate_envelope(
        env, _spec("cat", ["Cost or resource use", "Service life"]),
        paper_text=PAPER, image_labels=set(), path_prefix="record.relationship_1.cat",
    )
    assert not any(e["code"] == "invalid_option" for e in errs)
    assert env["value"] == "Cost or resource use"


def test_no_case_hint_when_genuinely_invalid():
    errs = validate_envelope(
        {"value": "Maybe", "evidence": ["Some study text"],
         "source": "Methods"},
        _spec("foo", ["Yes", "No"]),
        paper_text=PAPER, image_labels=set(), path_prefix="study.foo",
    )
    msg = next(e["message"] for e in errs if e["code"] == "invalid_option")
    assert "Did you mean" not in msg
