"""Tests for the resilient checker-JSON parser.

Sonnet occasionally appends prose after the JSON object even when asked
not to. The parser must extract the first valid JSON object regardless.
"""

import pytest

from meltiro.checker import _parse_checker_json
from meltiro.errors import CheckerError


def test_plain_json():
    obj = _parse_checker_json('{"verdict": "ok", "rationale": "fine"}')
    assert obj == {"verdict": "ok", "rationale": "fine"}


def test_with_trailing_text():
    # The commonest shape: a sentence appended after the JSON object.
    raw = ('{"verdict": "challenge", "rationale": "wrong"}\n'
           "I would have been more confident if the evidence "
           "mentioned the exact threshold.")
    obj = _parse_checker_json(raw)
    assert obj["verdict"] == "challenge"
    assert obj["rationale"] == "wrong"


def test_with_leading_text():
    raw = ('Here is my verdict:\n'
           '{"verdict": "ok", "rationale": "matches"}')
    obj = _parse_checker_json(raw)
    assert obj["verdict"] == "ok"


def test_with_code_fences():
    raw = '```json\n{"verdict": "ok", "rationale": "fine"}\n```'
    obj = _parse_checker_json(raw)
    assert obj["verdict"] == "ok"


def test_with_notes_field():
    raw = ('{"verdict": "ok", "rationale": "evidence supports", '
           '"notes": "borderline; reviewer should double-check"}')
    obj = _parse_checker_json(raw)
    assert obj["notes"] == "borderline; reviewer should double-check"


def test_empty_response_raises():
    with pytest.raises(CheckerError, match="empty"):
        _parse_checker_json("")


def test_no_json_at_all_raises():
    with pytest.raises(CheckerError, match="non-JSON"):
        _parse_checker_json("this is just prose with no braces")


def test_top_level_not_object_raises():
    with pytest.raises(CheckerError, match="not an object"):
        _parse_checker_json('["array", "instead", "of", "object"]')
