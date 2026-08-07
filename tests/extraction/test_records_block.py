"""records: block parsing and top-level key validation.

The template declares its repeated-row entity and that entity's extraction
sections under a single top-level `records:` mapping:

    records:
      relationship:
        plural: relationships
        description: ...
        extraction: [...]

The mapping key is the singular entity name. `plural` is optional
(defaults to singular + "s"); `description` is required; `extraction` is
required and non-empty. Exactly one record type is supported.
Quality-assessment sections live in `extraction:` marked `qa: true`, so a
separate `qa:` list is a subkey the schema does not define and fails loudly,
as does any top-level key outside the allowlist -- including the plausible
wrong guesses for where record content goes (`record_entity`,
`record_extraction`, `record_qa`, `relationship_extraction`,
`relationship_qa`, `study_qa`).
"""

import pytest
import yaml

from meltiro.template import (
    _parse_records,
    _validate_top_level_keys,
    load_template,
)
from meltiro.tools import get_tool_definitions

# Sentinel: "delete the checker_context_fields key" versus "set it to
# something".
_MISSING = object()

# Top-level blocks an author might reach for that this schema does not define.
# Each is a plausible place to look for record content or for a separate
# quality-assessment block; all of it lives under `records:` and under the
# per-section `qa: true` flag instead. A silently-ignored one would drop whole
# sections from the tool schema, the checker fan-out and the render.
UNDEFINED_TOP_LEVEL_KEYS = [
    "record_entity",
    "record_extraction",
    "record_qa",
    "relationship_extraction",
    "relationship_qa",
    "study_qa",
]


def _good_records():
    return {
        "records": {
            "relationship": {
                "plural": "relationships",
                "description": "a reported relationship between a durability "
                               "gauge score and a lifecycle outcome",
                "checker_context_fields": ["gauge", "outcome_variable"],
                "extraction": [
                    {"section": "Identity", "fields": []},
                    {"section": "Sample", "qa": True, "fields": []},
                ],
            }
        }
    }


class TestParseRecords:
    def test_valid_block_parses(self):
        entity, extraction, context_fields = _parse_records(_good_records())
        assert entity == {
            "singular": "relationship",
            "plural": "relationships",
            "description": "a reported relationship between a durability "
                           "gauge score and a lifecycle outcome",
            "extraction_instruction": None,
        }
        assert extraction == [
            {"section": "Identity", "fields": []},
            {"section": "Sample", "qa": True, "fields": []},
        ]
        assert context_fields == ["gauge", "outcome_variable"]

    def test_mapping_key_is_the_singular_name(self):
        raw = _good_records()
        raw["records"] = {"outcome": raw["records"]["relationship"]}
        entity, _, _ = _parse_records(raw)
        assert entity["singular"] == "outcome"

    @pytest.mark.parametrize("name", ["treatment_effect", "outcome", "effect2"])
    def test_slug_entity_names_accepted(self, name):
        # A normal single-underscore noun (and plain/trailing-digit variants)
        # is a valid entity slug: it mints `<name>_<n>` ids that split cleanly
        # on `.` in a dotted field path.
        raw = _good_records()
        raw["records"] = {name: raw["records"]["relationship"]}
        entity, _, _ = _parse_records(raw)
        assert entity["singular"] == name

    @pytest.mark.parametrize(
        "name",
        [
            "a.b",          # a `.` adds a path segment: silent mis-routing
            "x__y",         # only single internal underscores are allowed
            "2effect",      # leading digit
            "Relationship",  # uppercase
            "effect_",      # trailing underscore
            "_effect",      # leading underscore
        ],
    )
    def test_non_slug_entity_names_rejected_at_load(self, name):
        # The entity noun is minted into ids and dotted paths, so a
        # name the downstream consumers cannot handle is refused at load with a
        # message naming the offender and the allowed charset.
        raw = _good_records()
        raw["records"] = {name: raw["records"]["relationship"]}
        with pytest.raises(ValueError, match="valid entity slug"):
            _parse_records(raw)

    def test_entity_named_study_rejected_at_load(self):
        # `study` is a valid slug but collides with the `study.` scope prefix
        # in gate references: every correctly-scoped gate for such an entity
        # would be rejected with a message prescribing the exact spelling that
        # was just refused. Reserved at load instead.
        raw = _good_records()
        raw["records"] = {"study": raw["records"]["relationship"]}
        with pytest.raises(ValueError, match="must not be `study`"):
            _parse_records(raw)

    def test_plural_defaults_to_singular_plus_s(self):
        raw = _good_records()
        del raw["records"]["relationship"]["plural"]
        entity, _, _ = _parse_records(raw)
        assert entity["plural"] == "relationships"

    def test_a_separate_qa_subkey_is_rejected(self):
        # Quality-assessment sections are ordinary `extraction:` sections
        # marked `qa: true`, so a second section list under `qa:` would be
        # read by nothing: every section it held would vanish from the tool
        # schema, the checker fan-out and the render, with nothing said. The
        # subkey allowlist names the author's own key path and the legal set.
        raw = _good_records()
        raw["records"]["relationship"]["qa"] = [
            {"section": "Sample", "fields": []}]
        with pytest.raises(ValueError) as exc:
            _parse_records(raw)
        message = str(exc.value)
        assert "`records.relationship` has unknown subkey(s) ['qa']" in message
        assert "'extraction'" in message

    def test_a_qa_subkey_is_rejected_whatever_it_holds(self):
        # Presence of the key is the error: an empty or malformed one is not a
        # smaller mistake, it is the same one.
        raw = _good_records()
        raw["records"]["relationship"]["qa"] = None
        with pytest.raises(ValueError, match="unknown subkey"):
            _parse_records(raw)

    def test_context_fields_stripped_and_ordered(self):
        raw = _good_records()
        raw["records"]["relationship"]["checker_context_fields"] = [
            "  gauge  ", "outcome_variable"]
        _, _, context_fields = _parse_records(raw)
        assert context_fields == ["gauge", "outcome_variable"]

    def test_context_fields_optional_absent_is_empty(self):
        # Optional: absent means an empty list (records labelled by id alone).
        raw = _good_records()
        del raw["records"]["relationship"]["checker_context_fields"]
        _, _, context_fields = _parse_records(raw)
        assert context_fields == []

    def test_context_fields_empty_list_allowed(self):
        raw = _good_records()
        raw["records"]["relationship"]["checker_context_fields"] = []
        _, _, context_fields = _parse_records(raw)
        assert context_fields == []

    def test_context_fields_must_be_list(self):
        raw = _good_records()
        raw["records"]["relationship"]["checker_context_fields"] = "gauge"
        with pytest.raises(ValueError,
                           match="checker_context_fields` must be a list"):
            _parse_records(raw)

    @pytest.mark.parametrize("bad", ["", "   ", 3, None])
    def test_context_fields_entries_non_empty_strings(self, bad):
        raw = _good_records()
        raw["records"]["relationship"]["checker_context_fields"] = [
            "gauge", bad]
        with pytest.raises(ValueError,
                           match="checker_context_fields` entries"):
            _parse_records(raw)

    def test_missing_records_rejected(self):
        with pytest.raises(ValueError, match="missing the required top-level"):
            _parse_records({})

    def test_none_records_rejected(self):
        with pytest.raises(ValueError, match="missing the required top-level"):
            _parse_records({"records": None})

    def test_not_a_mapping_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _parse_records({"records": ["relationship"]})

    def test_zero_record_types_rejected(self):
        with pytest.raises(ValueError, match="exactly one record type"):
            _parse_records({"records": {}})

    def test_multiple_record_types_rejected(self):
        raw = _good_records()
        raw["records"]["outcome"] = raw["records"]["relationship"]
        with pytest.raises(ValueError, match="exactly one record type"):
            _parse_records(raw)

    def test_unknown_subkey_rejected(self):
        raw = _good_records()
        raw["records"]["relationship"]["colour"] = "blue"
        with pytest.raises(ValueError, match="unknown subkey"):
            _parse_records(raw)

    def test_description_required(self):
        raw = _good_records()
        del raw["records"]["relationship"]["description"]
        with pytest.raises(ValueError, match="description` is required"):
            _parse_records(raw)

    @pytest.mark.parametrize("bad", ["", "   ", 3, None])
    def test_description_non_empty_string(self, bad):
        raw = _good_records()
        raw["records"]["relationship"]["description"] = bad
        with pytest.raises(ValueError, match="description"):
            _parse_records(raw)

    def test_extraction_required_non_empty(self):
        raw = _good_records()
        raw["records"]["relationship"]["extraction"] = []
        with pytest.raises(ValueError, match="extraction` is required"):
            _parse_records(raw)

    def test_extraction_must_be_list(self):
        raw = _good_records()
        raw["records"]["relationship"]["extraction"] = "nope"
        with pytest.raises(ValueError, match="extraction` is required"):
            _parse_records(raw)

    def test_plural_non_empty_string_when_present(self):
        raw = _good_records()
        raw["records"]["relationship"]["plural"] = "  "
        with pytest.raises(ValueError, match="plural` must be a non-empty"):
            _parse_records(raw)

    def test_record_extraction_instruction_optional_default_none(self):
        entity, _, _ = _parse_records(_good_records())
        assert entity["extraction_instruction"] is None

    def test_record_extraction_instruction_parsed_and_stripped(self):
        raw = _good_records()
        raw["records"]["relationship"]["extraction_instruction"] = (
            "  One entry per distinct relationship; do not combine them.  ")
        entity, _, _ = _parse_records(raw)
        assert entity["extraction_instruction"] == \
            "One entry per distinct relationship; do not combine them."

    @pytest.mark.parametrize("bad", ["", "   ", 3, []])
    def test_record_extraction_instruction_must_be_non_empty_string(self, bad):
        raw = _good_records()
        raw["records"]["relationship"]["extraction_instruction"] = bad
        with pytest.raises(ValueError,
                           match="extraction_instruction` must be a"):
            _parse_records(raw)


class TestTopLevelKeyValidation:
    def _valid_keys(self):
        return {
            "study_extraction": [],
            "records": {},
            "llm_initial_check": [],
            "llm_quality_check": [],
        }

    def test_valid_key_set_accepted(self):
        # No raise (records emptiness is _parse_records' job, not this check).
        _validate_top_level_keys(self._valid_keys())

    @pytest.mark.parametrize("wrong_guess", UNDEFINED_TOP_LEVEL_KEYS)
    def test_a_block_the_schema_does_not_define_is_rejected(self, wrong_guess):
        raw = self._valid_keys()
        raw[wrong_guess] = []
        with pytest.raises(ValueError) as exc:
            _validate_top_level_keys(raw)
        message = str(exc.value)
        assert "unknown top-level key" in message
        assert wrong_guess in message
        # The known set rides in the message, so the author reads the right
        # place for the content off the error rather than the source.
        assert "'records'" in message
        assert "'study_extraction'" in message

    def test_unknown_top_level_key_rejected(self):
        raw = self._valid_keys()
        raw["bogus_key"] = 1
        with pytest.raises(ValueError, match="unknown top-level key"):
            _validate_top_level_keys(raw)

    def test_missing_required_key_rejected(self):
        raw = self._valid_keys()
        del raw["study_extraction"]
        with pytest.raises(ValueError, match="missing required top-level"):
            _validate_top_level_keys(raw)


class TestLoadTemplate:
    def _path(self, config_dir):
        return config_dir / "extraction_template.yaml"

    def test_shipped_template_exposes_record_entity(self, config_dir):
        t = load_template(self._path(config_dir))
        assert t["record_entity"]["singular"] == "relationship"
        assert t["record_entity"]["plural"] == "relationships"
        assert t["record_entity"]["description"].startswith("a reported")
        assert "record_fields" in t
        # One field block per scope, not two: a QA section is an ordinary
        # `record_fields` / `study_fields` section marked `qa: true`, NOT a
        # block of its own. Pinning the whole set of `_fields` keys keeps a
        # reintroduced per-scope QA block from passing unnoticed.
        assert {k for k in t if k.endswith("_fields")} == {
            "study_fields", "record_fields", "initial_check_fields",
            "quality_check_fields", "checker_context_fields", "role_fields",
        }
        assert any(s["qa"] for s in t["record_fields"])
        assert any(s["qa"] for s in t["study_fields"])

    def test_record_extraction_instruction_leads_add_record_description(
            self, config_dir):
        # The record-level extraction_instruction renders as the leading clause
        # of the add_record tool description (so it rides tool_set_hash).
        t = load_template(self._path(config_dir))
        add = next(x for x in get_tool_definitions(t)
                   if x["name"] == "add_record")
        desc = add["description"]
        assert desc.startswith("What counts as one relationship: ")
        assert "Do NOT consolidate" in desc
        # The generic "Append a new ..." clause still follows it.
        assert "Append a new relationship record" in desc

    @pytest.mark.parametrize("wrong_guess", UNDEFINED_TOP_LEVEL_KEYS)
    def test_an_undefined_top_level_key_fails_through_load_template(
            self, config_dir, tmp_path, wrong_guess):
        # The same refusal reached through the real entry point, on the shipped
        # template: a block the schema does not define never loads, so its
        # sections can never be silently absent from a run.
        text = self._path(config_dir).read_text(encoding="utf-8")
        text = f"{wrong_guess}: []\n" + text
        bad = tmp_path / "extraction_template.yaml"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            load_template(bad)
        message = str(exc.value)
        assert "unknown top-level key" in message
        assert wrong_guess in message

    def test_unknown_top_level_key_fails(self, config_dir, tmp_path):
        text = self._path(config_dir).read_text(encoding="utf-8")
        text = "surprise_block: []\n" + text
        bad = tmp_path / "extraction_template.yaml"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="unknown top-level key"):
            load_template(bad)


class TestCheckerContextFieldsValidation:
    """Whole-template validation of `records.<name>.checker_context_fields`,
    done in load_template once the record fields are parsed."""

    def _write_with_context_fields(self, config_dir, tmp_path, context_fields):
        raw = yaml.safe_load(
            (config_dir / "extraction_template.yaml").read_text("utf-8"))
        rectype = next(iter(raw["records"]))
        if context_fields is _MISSING:
            raw["records"][rectype].pop("checker_context_fields", None)
        else:
            raw["records"][rectype]["checker_context_fields"] = context_fields
        out = tmp_path / "extraction_template.yaml"
        out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return out

    def test_shipped_template_exposes_context_fields(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        assert t["checker_context_fields"] == [
            "gauge", "gauge_score_format", "outcome_variable",
            "outcome_category", "statistical_method",
        ]

    def test_valid_context_fields_load(self, config_dir, tmp_path):
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["outcome_variable", "gauge"])
        t = load_template(path)
        assert t["checker_context_fields"] == ["outcome_variable", "gauge"]

    def test_absent_context_fields_default_empty(self, config_dir, tmp_path):
        # Optional: a template that omits the key labels records by id alone.
        path = self._write_with_context_fields(config_dir, tmp_path, _MISSING)
        t = load_template(path)
        assert t["checker_context_fields"] == []

    def test_context_field_can_name_a_qa_field(self, config_dir, tmp_path):
        # rqa_pre_specified is a record QA field, an ordinary record field, so
        # it is a valid (if unusual) context entry.
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "rqa_pre_specified"])
        t = load_template(path)
        assert t["checker_context_fields"] == ["gauge", "rqa_pre_specified"]

    def test_unknown_field_name_fails(self, config_dir, tmp_path):
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "not_a_real_field"])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)

    def test_study_scoped_field_name_fails(self, config_dir, tmp_path):
        # widget_class is a study field, not a record field: out of scope.
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "widget_class"])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)

    def test_engine_id_name_rejected(self, config_dir, tmp_path):
        # relationship_id is the engine-assigned record id, not a template
        # field, so naming it is naming a non-existent field.
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "relationship_id"])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)

    def test_scope_note_key_rejected(self, config_dir, tmp_path):
        # `notes` is the record's reserved scope-note key, not a field, so
        # naming it here is naming a non-existent field. (It could not be
        # declared as one either: the loader refuses the variable name.)
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "notes"])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)

    def test_duplicate_name_rejected(self, config_dir, tmp_path):
        # A repeated entry would render a doubled label component; strict
        # inputs reject it loudly rather than degrading the checker label.
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "outcome_variable", "gauge"])
        with pytest.raises(ValueError, match="more than once"):
            load_template(path)

    def test_duplicate_after_strip_rejected(self, config_dir, tmp_path):
        # Whitespace normalisation must not let a duplicate slip through.
        path = self._write_with_context_fields(
            config_dir, tmp_path, ["gauge", "gauge "])
        with pytest.raises(ValueError, match="more than once"):
            load_template(path)
