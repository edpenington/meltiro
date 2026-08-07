"""`validate_extraction_output` over the role-keyed check blocks.

Both check blocks are keyed by the role that recorded them
(`{role: {variable: value}}`), so the batch validator has to descend one level
before it reaches a variable.

Reading them flat would look EVERY role name up as a field: a perfectly good
extraction output would come back with `initial_check.extractor` and
`quality_check.review` reported as `unknown_field`, and the values underneath,
the half worth validating, would never be checked at all.

These tests go through the batch path, because that is the path a consumer
goes through. `meltiro validate` is the one door this API has, and it is the
door a `--no-deps` consumer uses for production validation, so a spurious
failure there is a downstream outage rather than a cosmetic wrong path.
"""

import pytest

from meltiro.extraction_record import ROLE_EXTRACTOR, ROLE_REVIEW
from meltiro.template import load_template
from meltiro.validators import validate_extraction_output


@pytest.fixture
def template(config_dir):
    return load_template(config_dir / "extraction_template.yaml")


def _output(**blocks):
    base = {"initial_check": {}, "study": {}, "records": [],
            "quality_check": {}}
    base.update(blocks)
    return base


class TestTheRoleKeyIsNotAField:
    def test_a_well_formed_role_keyed_output_validates_clean(self, template):
        # The shape a finished run writes.
        out = _output(
            initial_check={ROLE_EXTRACTOR: {"text_readable": True}},
            quality_check={ROLE_EXTRACTOR: {"general_notes": "went fine"},
                           ROLE_REVIEW: {"general_notes": "confirmed"}},
        )
        failures, _ = validate_extraction_output(template, out)
        assert failures == []

    def test_both_roles_quality_checks_are_validated_not_just_the_first(
            self, template):
        # Each role's answers are swept on their own, so a bad value under one
        # role is reported under that role and does not mask the other.
        out = _output(quality_check={
            ROLE_EXTRACTOR: {"general_notes": "fine"},
            ROLE_REVIEW: {"not_a_variable": "x"},
        })
        failures, _ = validate_extraction_output(template, out)
        assert [(f["path"], f["code"]) for f in failures] == [
            ("quality_check.review.not_a_variable", "unknown_field")]

    def test_a_value_under_a_role_is_type_checked(self, template):
        # The VALUES under a role are validated, not merely the keys.
        out = _output(
            initial_check={ROLE_EXTRACTOR: {"text_readable": "yes please"}})
        failures, _ = validate_extraction_output(template, out)
        assert [f["path"] for f in failures] == [
            "initial_check.extractor.text_readable.value"]
        assert failures[0]["code"] == "type_mismatch"

    def test_a_non_mapping_under_a_role_is_one_clear_failure(self, template):
        # A hand-edited or externally-produced file should degrade to one
        # readable failure rather than a spray of unknown-field noise.
        out = _output(initial_check={ROLE_EXTRACTOR: "text_readable=true"})
        failures, _ = validate_extraction_output(template, out)
        assert [(f["path"], f["code"]) for f in failures] == [
            ("initial_check.extractor", "not_a_role_block")]
        assert "keyed by the role" in failures[0]["message"]

    def test_empty_blocks_validate_clean(self, template):
        # A role that recorded nothing mints no key, so the blocks can be
        # empty maps and that is not an error.
        failures, _ = validate_extraction_output(template, _output())
        assert failures == []
