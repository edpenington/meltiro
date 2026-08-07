"""A mistyped field key fails at load instead of silently defaulting.

Every other structural level of the template already carries an unknown-key
allowlist: top level (`_KNOWN_TOP_KEYS`), record type (`_RECORD_TYPE_KEYS`),
section (`_SECTION_KEYS`), gate rule (`_GATE_KEYS`). The field level is the one
authors type into most, and a key mistyped there is the worst failure this
parser can have: `requred: true` would leave the field optional, `evidnce:` and
`canonial_reference:` would drop their enforcement, the extraction would
complete, and `fingerprint.field_catalogue_hash` would record the defaulted
config as though the author had written it. `_FIELD_KEYS` closes that.

These tests pin:
  - a mistyped key raises, naming the field, the key, and the legal set;
  - the mistyped key is named at the point of the typo, not as a downstream
    symptom ("neither `type:` nor `options:` is set" for a misspelt `type:`);
  - every key in `_FIELD_KEYS` is one a real field can actually carry;
  - a plausible key this schema does not define is refused by that same
    allowlist, with no second list of specially-handled names beside it;
  - the shipped config's fields use legal keys only.
"""

import pytest
import yaml

from meltiro.template import (
    _FIELD_KEYS,
    _parse_field,
    load_template,
)


# Field keys an author might plausibly reach for that this schema does not
# define: a soft-enum flag, a list-valued-field flag, a pipeline-managed id
# marker, a whole-field notes marker, and per-field notes guidance. Every one
# of them would change nothing if it were quietly ignored, which is exactly
# the silent default `_FIELD_KEYS` exists to refuse. Each is spelled with a
# value an author would write, so the rejection is proved on the shape a real
# template would carry rather than on a bare sentinel.
UNDEFINED_FIELD_KEYS = {
    "enum_mode": "soft",
    "multivalue": True,
    "identity": True,
    "notes": True,
    "notes_help": "what to write",
}


def _field(**extra):
    """A minimal valid envelope field, plus whatever the test adds."""
    f = {"variable": "gauge", "description": "The durability gauge used.",
         "type": "string", "evidence": "required"}
    f.update(extra)
    return f


def _parse(f, *, envelope=True):
    return _parse_field(f, envelope=envelope, section_name="Measurement")


class TestMistypedKeyRejected:
    """The motivating defect: a misspelling of any legal key is a load error,
    and the message is enough to fix the template without reading source."""

    def test_misspelt_required_names_field_key_and_legal_set(self):
        with pytest.raises(ValueError) as exc:
            _parse(_field(requred=True))
        message = str(exc.value)
        assert "'gauge'" in message          # which field
        assert "'requred'" in message        # which key
        assert "unknown field key(s)" in message
        assert "'required'" in message       # the legal set, for the fix

    @pytest.mark.parametrize("mistyped", [
        "requred", "evidnce", "canonial_reference", "optoins", "roles",
        "allow_others", "labell", "extraction_instructions",
        "soft_canonicalization",   # the American spelling is not the key
    ])
    def test_each_misspelling_is_a_load_error(self, mistyped):
        with pytest.raises(ValueError) as exc:
            _parse(_field(**{mistyped: True}))
        message = str(exc.value)
        assert f"'{mistyped}'" in message
        assert "unknown field key(s)" in message

    def test_misspelt_type_is_named_not_reported_as_a_missing_shape(self):
        # Without the allowlist running first, a misspelt `type:` surfaces
        # downstream as "neither `type:` nor `options:` is set", sending the
        # author to a line they did write. The typo itself must be named.
        f = {"variable": "gauge", "description": "d", "typ": "string",
             "evidence": "required"}
        with pytest.raises(ValueError) as exc:
            _parse(f)
        message = str(exc.value)
        assert "'typ'" in message
        assert "neither" not in message

    def test_bare_value_check_block_fields_are_checked_too(self):
        # The llm_initial_check / llm_quality_check blocks parse through the
        # same function with envelope=False; the allowlist is not
        # envelope-only.
        f = {"variable": "text_readable", "description": "d",
             "type": "boolean", "requred": True}
        with pytest.raises(ValueError, match="unknown field key"):
            _parse(f, envelope=False)

    def test_several_unknown_keys_are_all_named_sorted(self):
        with pytest.raises(ValueError) as exc:
            _parse(_field(zeta=1, alpha=2))
        message = str(exc.value)
        assert "['alpha', 'zeta']" in message

    def test_a_correctly_spelled_field_still_parses(self):
        # The guard against an allowlist that rejects everything.
        assert _parse(_field(required=True))["required"] is True


class TestNoSilentDefault:
    """The consequence the allowlist exists to prevent: a mistyped key leaving
    the field on the default the author did not choose."""

    def test_required_true_only_reachable_through_the_real_key(self):
        assert _parse(_field(required=True))["required"] is True
        # The misspelling cannot produce a parsed field at all now, so it can
        # never reach field_catalogue_hash as `required: false`.
        with pytest.raises(ValueError):
            _parse(_field(requred=True))

    def test_canonical_reference_misspelling_cannot_drop_validation(self):
        # A misspelt reference key would leave the field unvalidated against
        # the reference list while the config claims otherwise.
        with pytest.raises(ValueError):
            _parse(_field(canonical_referene="gauge_list"))


class TestEveryLegalKeyParses:
    """Each key in `_FIELD_KEYS` is one a real field can carry. Some are
    mutually exclusive (`type`/`options`, `allow_other`/`canonical_reference`,
    `canonical_reference`/`soft_canonicalisation`), so the coverage is split
    across three fields whose key union is the whole allowlist. Adding a key to
    `_FIELD_KEYS` without a field here that parses with it fails this test."""

    CATEGORICAL = {
        "variable": "outcome_category",
        "description": "The class of outcome reported.",
        "label": "Outcome category",
        "extraction_instruction": "Pick the closest class.",
        "options": ["Failure state", "Service life"],
        "allow_other": True,
        "evidence": "required",
        "required": True,
    }
    REFERENCE = {
        "variable": "gauge",
        "description": "The durability gauge used.",
        "type": "string",
        "canonical_reference": "gauge_list",
        "role": "summary",
        "evidence": "required",
    }
    SOFT = {
        "variable": "statistical_method",
        "description": "The method reported.",
        "type": "string",
        "soft_canonicalisation": True,
        "evidence": "optional",
    }

    def test_the_three_fields_cover_the_whole_allowlist(self):
        covered = set(self.CATEGORICAL) | set(self.REFERENCE) | set(self.SOFT)
        assert covered == _FIELD_KEYS

    def test_categorical_field_with_every_key_it_can_carry_parses(self):
        parsed = _parse(dict(self.CATEGORICAL))
        assert parsed["field_type"] == "categorical"
        assert parsed["allow_other"] is True
        assert parsed["required"] is True
        assert parsed["label"] == "Outcome category"
        assert parsed["extraction_instruction"] == "Pick the closest class."

    def test_reference_field_with_every_key_it_can_carry_parses(self):
        parsed = _parse(dict(self.REFERENCE))
        assert parsed["canonical_reference"] == "gauge_list"
        assert parsed["role"] == "summary"
        assert parsed["evidence"] == "required"

    def test_soft_canonicalisation_field_parses(self):
        parsed = _parse(dict(self.SOFT))
        assert parsed["soft_canonicalisation"] is True


class TestUndefinedKeysAreRejected:
    """A key this schema does not define is a load error, on the same terms as
    a typo. The allowlist is the whole rule: nothing is exempted from it, so
    there is no second list of names that could drift out of step and leave a
    key exempt from the allowlist and matched by nothing."""

    def test_undefined_keys_are_not_in_the_legal_set(self):
        assert not (set(UNDEFINED_FIELD_KEYS) & _FIELD_KEYS)

    @pytest.mark.parametrize("key,value",
                             sorted(UNDEFINED_FIELD_KEYS.items()))
    def test_each_undefined_key_is_a_load_error(self, key, value):
        with pytest.raises(ValueError) as exc:
            _parse(_field(**{key: value}))
        message = str(exc.value)
        assert "unknown field key(s)" in message
        assert f"'{key}'" in message      # named at the point of the typo
        assert "'variable'" in message    # the legal set, for the fix

    @pytest.mark.parametrize("key,value",
                             sorted(UNDEFINED_FIELD_KEYS.items()))
    def test_each_undefined_key_is_a_load_error_on_a_bare_value_field(
            self, key, value):
        # The bare-value check blocks parse through the same function with
        # envelope=False, so the rule reads the same wherever an author looks.
        f = {"variable": "text_readable", "description": "d",
             "type": "boolean", key: value}
        with pytest.raises(ValueError, match="unknown field key"):
            _parse(f, envelope=False)


class TestThroughLoadTemplate:
    """The allowlist reached through the real entry point, on the shipped
    config."""

    def _shipped(self, config_dir):
        return config_dir / "extraction_template.yaml"

    def _field_keys_in(self, raw):
        keys = set()
        blocks = [raw.get("study_extraction"), raw.get("llm_initial_check"),
                  raw.get("llm_quality_check")]
        for definition in (raw.get("records") or {}).values():
            blocks.append(definition.get("extraction"))
        for sections in blocks:
            for section in sections or []:
                for f in section.get("fields") or []:
                    keys |= set(f)
        return keys

    def test_shipped_config_uses_legal_field_keys_only(self, config_dir):
        raw = yaml.safe_load(
            self._shipped(config_dir).read_text(encoding="utf-8"))
        assert self._field_keys_in(raw) <= _FIELD_KEYS

    def test_shipped_config_still_loads(self, config_dir):
        template = load_template(self._shipped(config_dir))
        assert template["study_fields"]
        assert template["record_fields"]

    def test_mistyped_key_on_a_shipped_field_fails_at_load(
            self, config_dir, tmp_path):
        raw = yaml.safe_load(
            self._shipped(config_dir).read_text(encoding="utf-8"))
        target = raw["study_extraction"][0]["fields"][0]
        target["requred"] = True
        bad = tmp_path / "extraction_template.yaml"
        bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            load_template(bad)
        message = str(exc.value)
        assert f"{target['variable']!r}" in message
        assert "'requred'" in message

    def test_mistyped_key_on_a_record_field_fails_at_load(
            self, config_dir, tmp_path):
        raw = yaml.safe_load(
            self._shipped(config_dir).read_text(encoding="utf-8"))
        rectype = next(iter(raw["records"]))
        target = raw["records"][rectype]["extraction"][0]["fields"][0]
        target["evidnce"] = "required"
        bad = tmp_path / "extraction_template.yaml"
        bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown field key"):
            load_template(bad)
