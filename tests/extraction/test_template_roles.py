"""Template field `role:` parsing and enforcement.

`summary` is the ONLY role. Plausible-sounding alternatives an author might
reach for — `doi`, `study_id` — are rejected as unknown; the study id is a
pipeline concern recorded in the run's output metadata, not a template field.
The loader enforces:
  - the role value is known (unknown roles rejected),
  - a role-bearing field is a plain string (no options list, no non-string
    type),
  - a role sits in its scope: the `summary` role on study_extraction,
  - at most one field claims each role.

Valid role fields are exposed as `template["role_fields"]`.
"""

import pytest

from meltiro.template import _parse_field, load_template


def _field(role=None, *, field_type="string", options=None):
    f = {"variable": "v", "description": "d"}
    if options is not None:
        f["options"] = options
    else:
        f["type"] = field_type
    if role is not None:
        f["role"] = role
    # envelope=False keeps the focus on role handling (no evidence flag).
    return _parse_field(f, envelope=False, section_name="test")


class TestPerFieldRoleParsing:
    def test_valid_role_parses(self):
        assert _field("summary")["role"] == "summary"

    def test_absent_role_is_none(self):
        assert _field()["role"] is None

    def test_unknown_role_rejected(self):
        with pytest.raises(ValueError, match="unknown `role"):
            _field("banana")

    def test_doi_is_not_a_role(self):
        # `role: doi` names nothing the engine reads, so it fails loudly with
        # a message naming the retirement.
        with pytest.raises(ValueError, match=r"unknown `role: 'doi'`"):
            _field("doi")

    def test_study_id_is_not_a_role(self):
        # There is no `role: study_id` and no record-stamping: the study id is
        # a pipeline concern (run.json), not a template field.
        with pytest.raises(ValueError,
                           match=r"unknown `role: 'study_id'`"):
            _field("study_id")

    def test_role_with_options_rejected(self):
        with pytest.raises(ValueError, match="plain string"):
            _field("summary", options=["A", "B"])

    def test_role_with_non_string_type_rejected(self):
        with pytest.raises(ValueError, match="plain string"):
            _field("summary", field_type="year")


class TestWholeTemplateRoleEnforcement:
    def _shipped_text(self, config_dir):
        return (config_dir / "extraction_template.yaml").read_text(
            encoding="utf-8")

    def test_shipped_template_exposes_role_fields(self, config_dir):
        t = load_template(config_dir / "extraction_template.yaml")
        assert set(t["role_fields"]) == {"summary"}
        assert t["role_fields"]["summary"]["variable"] == "abstract"

    def test_two_summary_roles_rejected(self, config_dir, tmp_path):
        # The shipped template already marks `abstract` role: summary; also
        # marking `title` role: summary is a second claim on the role.
        text = self._shipped_text(config_dir).replace(
            "  - variable: title\n"
            "    label: Title\n"
            "    description: Full paper title\n"
            "    type: string\n",
            "  - variable: title\n"
            "    label: Title\n"
            "    description: Full paper title\n"
            "    type: string\n"
            "    role: summary\n",
            1,
        )
        bad = tmp_path / "extraction_template.yaml"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate `role: summary"):
            load_template(bad)

    def test_role_on_relationship_field_rejected(self, config_dir, tmp_path):
        # Put the study-scoped `role: summary` on a relationship-level (record)
        # field: a role must sit in its declared scope, so a study-scoped role
        # on a record field is rejected. Record fields sit under
        # `records: relationship: extraction:`, so they are indented deeper.
        block = ("      - variable: gauge\n"
                 "        label: Gauge\n"
                 "        required: true\n"
                 "        description: Which durability gauge is being"
                 " assessed in this relationship\n"
                 "        type: string\n")
        text = self._shipped_text(config_dir)
        assert block in text
        text = text.replace(block, block + "        role: summary\n", 1)
        bad = tmp_path / "extraction_template.yaml"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="study-level"):
            load_template(bad)
