"""`soft_canonicalisation: true` field flag.

A consumer-facing declaration that a free-value field's values should be
auto-suggested and collapsed against earlier entries (a soft, growing
vocabulary), with no hard validation. The engine is inert on it: it parses,
validates, and exposes the flag, but takes no runtime action, and it is not in
`field_catalogue_hash` (it changes nothing the extractor or checker does).
"""

import pytest

from meltiro.fingerprint import field_catalogue_hash
from meltiro.template import _parse_field, load_template


def _field(**extra):
    f = {"variable": "effect_type", "description": "d", "type": "string",
         "evidence": "required"}
    f.update(extra)
    return f


class TestParsing:
    def test_default_false(self):
        spec = _parse_field(_field(), envelope=True, section_name="s")
        assert spec["soft_canonicalisation"] is False

    def test_true_on_free_value_field_parses(self):
        spec = _parse_field(_field(soft_canonicalisation=True),
                            envelope=True, section_name="s")
        assert spec["soft_canonicalisation"] is True

    def test_non_bool_rejected(self):
        with pytest.raises(ValueError,
                           match="soft_canonicalisation:` must be true"):
            _parse_field(_field(soft_canonicalisation="yes"),
                         envelope=True, section_name="s")

    def test_rejected_on_categorical_field(self):
        f = {"variable": "x", "description": "d",
             "options": ["A", "B"], "evidence": "optional",
             "soft_canonicalisation": True}
        with pytest.raises(ValueError, match="only valid on a free-value"):
            _parse_field(f, envelope=True, section_name="s")

    def test_mutually_exclusive_with_canonical_reference(self):
        with pytest.raises(ValueError,
                           match="cannot be combined with `canonical_reference"):
            _parse_field(
                _field(soft_canonicalisation=True,
                       canonical_reference="gauge_list"),
                envelope=True, section_name="s")


class TestEngineInert:
    def test_not_in_field_catalogue_hash(self):
        # The flag changes nothing the extractor or checker does, so it is
        # deliberately excluded from the checker's field-catalogue hash.
        def tmpl(flag):
            field = {"variable": "effect_type", "description": "d",
                     "field_type": "string", "evidence": "required",
                     "soft_canonicalisation": flag}
            return {
                "study_fields": [{"section": "S", "fields": [field]}],
                "record_fields": [],
                "initial_check_fields": [],
                "quality_check_fields": [],
            }
        assert field_catalogue_hash(tmpl(True)) == \
            field_catalogue_hash(tmpl(False))

    def test_shipped_template_exposes_the_flag(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        by_var = {f["variable"]: f
                  for s in t["record_fields"] for f in s["fields"]}
        assert by_var["effect_type"]["soft_canonicalisation"] is True
        # A neighbouring free-value field defaults to False.
        assert by_var["effect_size"]["soft_canonicalisation"] is False
