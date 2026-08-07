"""Optional `gates:` block parsing and whole-template validation.

The cross-field gate rules live in the template under an optional top-level
`gates:` block, not in the engine (`meltiro.validators.validate_gate_rules`).
Each rule names a controlling record field (`when_field`), a gated record field
(`field`), and the controlling values (`allowed_values`) under which the gated
field is expected. The engine reads whatever the template declares and names NO
field itself.

`when_field` and `field` are scoped paths of the form `<entity>.<variable>`,
where `<entity>` is the record type declared as the `records:` block key. The
prefix is validated and stripped at load; the parsed gate stores bare variable
names so `validate_gate_rules` reads unchanged.

    gates:
      - when_field: relationship.outcome_category
        field: relationship.index_tariff
        allowed_values: [Service life]

The block is optional: a template without it exposes an empty gate list and
produces no warnings. A malformed block fails loudly at load (strict inputs).
"""

import pytest
import yaml

from meltiro.template import (
    _parse_gates,
    _validate_top_level_keys,
    load_template,
)

# The record entity declared by the synthetic config fixture; gate references
# are scoped under it (`<entity>.<variable>`).
_ENTITY = "relationship"


def _one_gate(**overrides):
    gate = {
        "when_field": "relationship.outcome_category",
        "field": "relationship.index_tariff",
        "allowed_values": ["Service life"],
    }
    gate.update(overrides)
    return {"gates": [gate]}


class TestParseGates:
    def test_absent_block_is_empty_list(self):
        assert _parse_gates({}, _ENTITY) == []

    def test_null_block_is_empty_list(self):
        assert _parse_gates({"gates": None}, _ENTITY) == []

    def test_valid_block_parses_and_strips_prefix(self):
        # Outer whitespace is stripped and the `<entity>.` prefix is resolved
        # away: the parsed gate stores bare variable names downstream.
        raw = {"gates": [{
            "when_field": "  relationship.outcome_category  ",
            "field": " relationship.index_tariff ",
            "allowed_values": [" Service life "],
        }]}
        assert _parse_gates(raw, _ENTITY) == [{
            "when_field": "outcome_category",
            "field": "index_tariff",
            "allowed_values": ["Service life"],
        }]

    def test_block_must_be_a_list(self):
        with pytest.raises(ValueError, match="`gates:` must be a list"):
            _parse_gates({"gates": {"when_field": "a"}}, _ENTITY)

    def test_rule_must_be_a_mapping(self):
        with pytest.raises(ValueError, match=r"gates\[0\]` must be a mapping"):
            _parse_gates({"gates": ["not a mapping"]}, _ENTITY)

    def test_unknown_subkey_rejected(self):
        with pytest.raises(ValueError, match="unknown key"):
            _parse_gates(_one_gate(surprise=1), _ENTITY)

    @pytest.mark.parametrize("drop", ["when_field", "field", "allowed_values"])
    def test_missing_required_subkey_rejected(self, drop):
        raw = _one_gate()
        del raw["gates"][0][drop]
        with pytest.raises(ValueError, match="missing required key"):
            _parse_gates(raw, _ENTITY)

    @pytest.mark.parametrize("bad", ["", "   ", 3, None])
    def test_field_names_must_be_non_empty_strings(self, bad):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _parse_gates(_one_gate(field=bad), _ENTITY)

    def test_when_field_and_field_must_differ(self):
        with pytest.raises(ValueError, match="two different fields"):
            _parse_gates(
                _one_gate(field="relationship.outcome_category"), _ENTITY)

    @pytest.mark.parametrize("bad", [[], "Service life", None])
    def test_allowed_values_must_be_non_empty_list(self, bad):
        with pytest.raises(ValueError, match="non-empty list"):
            _parse_gates(_one_gate(allowed_values=bad), _ENTITY)

    def test_allowed_values_entries_must_be_non_empty_strings(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_gates(
                _one_gate(allowed_values=["Service life", ""]), _ENTITY)

    def test_duplicate_field_pair_rejected(self):
        raw = {"gates": [
            {"when_field": "relationship.outcome_category",
             "field": "relationship.index_tariff",
             "allowed_values": ["Service life"]},
            {"when_field": "relationship.outcome_category",
             "field": "relationship.index_tariff",
             "allowed_values": ["Failure state"]},
        ]}
        with pytest.raises(ValueError, match="duplicate gate"):
            _parse_gates(raw, _ENTITY)


class TestScopedReferences:
    """`when_field` / `field` must be `<entity>.<variable>`.

    The prefix binds each gate reference to the declared record entity, so a
    study field that shares a name with a record field cannot bind silently to
    record scope. Each malformed spelling earns its own message.
    """

    def test_bare_name_rejected_with_corrected_spelling(self):
        # No dot: the message shows the required form scoped to this entity.
        with pytest.raises(
                ValueError,
                match=r"bare field name.*relationship\.outcome_category"):
            _parse_gates(_one_gate(when_field="outcome_category"), _ENTITY)

    def test_study_scoped_controller_rejected_explicitly(self):
        # A `study.`-scoped controller is rejected on semantics: the gate check
        # sees only sibling record values, never study-level fields.
        with pytest.raises(
                ValueError,
                match=r"study.*not supported.*single record"):
            _parse_gates(
                _one_gate(when_field="study.design"), _ENTITY)

    def test_wrong_entity_prefix_names_declared_entity(self):
        with pytest.raises(
                ValueError,
                match=r"entity prefix `outcome`.*record entity `relationship`"):
            _parse_gates(
                _one_gate(when_field="outcome.outcome_category"), _ENTITY)

    def test_malformed_multi_dot_path_rejected(self):
        with pytest.raises(ValueError, match="not a well-formed scoped path"):
            _parse_gates(
                _one_gate(field="relationship.index.tariff"), _ENTITY)

    def test_empty_variable_segment_rejected(self):
        with pytest.raises(ValueError, match="not a well-formed scoped path"):
            _parse_gates(_one_gate(field="relationship."), _ENTITY)


class TestGatesTopLevelKey:
    def _valid_keys(self):
        return {
            "study_extraction": [],
            "records": {},
            "llm_initial_check": [],
            "llm_quality_check": [],
        }

    def test_gates_is_an_accepted_optional_key(self):
        raw = self._valid_keys()
        raw["gates"] = []
        _validate_top_level_keys(raw)  # no raise

    def test_gates_is_not_required(self):
        # The four required keys with no gates block: still valid.
        _validate_top_level_keys(self._valid_keys())  # no raise


class TestGatesWholeTemplate:
    """Load-time validation against the parsed record fields."""

    def _write(self, config_dir, tmp_path, gates):
        raw = yaml.safe_load(
            (config_dir / "extraction_template.yaml").read_text("utf-8"))
        if gates is None:
            raw.pop("gates", None)
        else:
            raw["gates"] = gates
        out = tmp_path / "extraction_template.yaml"
        out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return out

    def test_shipped_template_exposes_gates(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        assert {"when_field": "outcome_category", "field": "index_tariff",
                "allowed_values": ["Service life"]} in t["gates"]
        assert len(t["gates"]) == 3

    def test_template_without_gates_has_empty_list(self, config_dir, tmp_path):
        path = self._write(config_dir, tmp_path, None)
        t = load_template(path)
        assert t["gates"] == []

    def test_gate_naming_unknown_field_fails(self, config_dir, tmp_path):
        path = self._write(config_dir, tmp_path, [{
            "when_field": "relationship.outcome_category",
            "field": "relationship.no_such_field",
            "allowed_values": ["Service life"]}])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)

    def test_gate_naming_study_field_fails(self, config_dir, tmp_path):
        # design is a study-level field: correctly prefixed with the
        # record entity, it still fails at the whole-template record-scope
        # check (the field simply is not record-scoped).
        path = self._write(config_dir, tmp_path, [{
            "when_field": "relationship.design",
            "field": "relationship.index_tariff",
            "allowed_values": ["Service life"]}])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)

    def test_gate_naming_engine_id_fails(self, config_dir, tmp_path):
        # The record id (relationship_id) is engine-assigned, not a template
        # field, so a gate that names it is naming a non-existent record field.
        path = self._write(config_dir, tmp_path, [{
            "when_field": "relationship.relationship_id",
            "field": "relationship.index_tariff",
            "allowed_values": ["Service life"]}])
        with pytest.raises(ValueError, match="not a record-scoped field"):
            load_template(path)
