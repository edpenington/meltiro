"""Tests for `meltiro render-template` and the underlying renderer.

Byte-exact golden files (under tests/expected_renders/) pin the rendered
output for both views of the config fixture. They are the only whole-document
check on the renderer: the unit tests below cover individual cells, and
nothing else asserts heading order, section grouping, the record-entity
block, or the publication view's filtering as a single artefact.

Golden coverage is single-template. No SECOND whole template — one with a
different section shape, options on record fields, `allow_other`, and no
reference lists — is rendered end to end. Those features are covered
individually below and by `TestCellSanitisation`; adding a second config
fixture would extend end-to-end cover to them.

Regenerate the goldens deliberately with the CLI (which loads the
reference-list display labels the render uses):

    for view in operational publication; do \
      python -m meltiro.cli render-template \
        --config tests/fixtures/config_synthetic --view "$view" \
        --out "tests/expected_renders/config_synthetic.$view.md"; \
    done

No network, no API key.
"""

import shutil
from pathlib import Path

import pytest

from meltiro import cli
from meltiro.reference_lists import load_reference_list_labels
from meltiro.render_template import (
    _OP_HEADER,
    _op_field_cell,
    _op_field_row,
    _op_values,
    _raw_table,
    render_template,
)
from meltiro.template import load_template

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = Path(__file__).resolve().parent / "expected_renders"
CONFIG_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "config_synthetic"


def _run(argv):
    """Invoke main(argv); return the SystemExit code."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    code = excinfo.value.code
    return 0 if code is None else code


def _template(config_dir):
    return load_template(config_dir / "extraction_template.yaml")


def _labels(config_dir):
    return load_reference_list_labels(config_dir / "reference")


def _field(**overrides):
    """A parsed-field dict with every key the renderer reads, defaulted to a
    plain string field. Tests override only the keys they exercise."""
    base = {
        "variable": "v",
        "label": "V",
        "description": "d",
        "extraction_instruction": None,
        "field_type": "string",
        "options": None,
        "allow_other": False,
        "evidence": "optional",
        "role": None,
        "required": False,
        "canonical_reference": None,
        "soft_canonicalisation": False,
    }
    base.update(overrides)
    return base


def _pipe_lines_all_closed(md):
    """True when the render contains table rows AND every physical line that
    opens a GFM table row (starts with `|`) also closes it (ends with `|`).

    A raw newline injected mid-cell splits one logical row into a line that
    opens with `|` but never closes, so this catches the corruption regardless
    of the offending content. A render carrying NO rows at all is False rather
    than vacuously True: every caller here renders a template that must
    produce a table, so "no rows" is a failure of the render and not a
    well-formed nothing.
    """
    rows = [line for line in md.splitlines() if line.startswith("|")]
    return bool(rows) and all(line.rstrip().endswith("|") for line in rows)


# ---------------------------------------------------------------------------
# Golden files: the render is a byte-exact projection of the template
# ---------------------------------------------------------------------------

class TestGoldenRenders:
    @pytest.mark.parametrize("config_dir, view, expected_name", [
        (CONFIG_FIXTURE, "operational", "config_synthetic.operational.md"),
        (CONFIG_FIXTURE, "publication", "config_synthetic.publication.md"),
    ])
    def test_render_matches_golden(self, config_dir, view, expected_name):
        expected = (EXPECTED_DIR / expected_name).read_text(encoding="utf-8")
        actual = render_template(_template(config_dir), view,
                                 _labels(config_dir))
        assert actual == expected

    def test_cli_render_matches_golden(self, tmp_path):
        out = tmp_path / "template.md"
        code = _run(["render-template", "--config", str(CONFIG_FIXTURE),
                     "--view", "operational", "--out", str(out)])
        assert code == 0
        expected = (EXPECTED_DIR / "config_synthetic.operational.md").read_text(
            encoding="utf-8")
        assert out.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# Idempotency: the same bundle renders byte-identically every time
# ---------------------------------------------------------------------------

class TestIdempotency:
    @pytest.mark.parametrize("view", ["operational", "publication"])
    def test_repeat_render_is_byte_identical(self, view):
        template = _template(CONFIG_FIXTURE)
        labels = _labels(CONFIG_FIXTURE)
        first = render_template(template, view, labels)
        second = render_template(template, view, labels)
        # Reloading the template and labels must not change the output either.
        third = render_template(_template(CONFIG_FIXTURE), view,
                                _labels(CONFIG_FIXTURE))
        assert first == second == third

    def test_cli_two_writes_are_byte_identical(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        assert _run(["render-template", "--config", str(CONFIG_FIXTURE),
                     "--view", "publication", "--out", str(a)]) == 0
        assert _run(["render-template", "--config", str(CONFIG_FIXTURE),
                     "--view", "publication", "--out", str(b)]) == 0
        assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# Content properties (traceability + the settled marker vocabulary)
# ---------------------------------------------------------------------------

class TestContent:
    def test_no_provenance_header_or_timestamp(self):
        out = render_template(_template(CONFIG_FIXTURE), "operational",
                              _labels(CONFIG_FIXTURE))
        # No HTML-comment provenance block, no volatile provenance keys.
        assert "<!--" not in out
        assert "generated_at" not in out
        assert "source_commit" not in out

    def test_allow_other_renders_other_specify(self):
        out = render_template(_template(CONFIG_FIXTURE), "operational",
                              _labels(CONFIG_FIXTURE))
        assert "Other (specify)" in out
        # The retired free-text wording must never appear.
        assert "or free text" not in out.lower()

    def test_type_values_read_as_human_labels(self):
        # The value cell shows the human type label, never the machine
        # field_type, so the document reads for a person.
        out = render_template(_template(CONFIG_FIXTURE), "operational",
                              _labels(CONFIG_FIXTURE))
        assert "| Text |" in out
        assert "| Text (multiple) |" in out
        assert "| Year |" in out
        assert "| Yes/No |" in out
        # None of the machine type names surface as a value cell.
        for machine in ("| string |", "| string_list |", "| year |",
                        "| boolean |", "| integer |", "| number |",
                        "| date |"):
            assert machine not in out

    def test_controlled_vocab_reads_with_label(self):
        # A canonical_reference field states the controlled-vocabulary
        # requirement in words. The gauge_list list declares `label: Gauge
        # Reference List`, so the value domain reads with the human display
        # name, not the raw list id in backticks.
        for view in ("operational", "publication"):
            out = render_template(_template(CONFIG_FIXTURE), view,
                                  _labels(CONFIG_FIXTURE))
            assert "Names from the Gauge Reference List" in out  # string_list
            assert "Name from the Gauge Reference List" in out   # single
            # With a label present, the raw-stem fallback must not appear, and
            # the bare LLM-plumbing marker must never appear.
            assert "`gauge_list` reference list" not in out
            assert "canonical_reference `gauge_list`" not in out

    def test_controlled_vocab_falls_back_to_stem_without_label(self):
        # With no label loaded, the value domain falls back to the raw list id
        # (the file stem) in backticks, keeping the same single-vs-list
        # wording. This is what a reference list declaring no label renders as.
        for view in ("operational", "publication"):
            out = render_template(_template(CONFIG_FIXTURE), view, labels={})
            assert "List of names from the `gauge_list` reference list" in out
            assert "Name from the `gauge_list` reference list" in out
            # The label-branch value cells must not appear when no label is
            # supplied. (Descriptions use lowercase "names from the Gauge
            # Reference List"; these capitalised value-cell forms are unique to
            # the label branch, so their absence is a clean fallback signal.)
            assert "Name from the Gauge Reference List" not in out
            assert "Names from the Gauge Reference List" not in out

    def test_no_notes_field_concept_in_the_render(self):
        # Notes are not a kind of field: every field carries a notes slot in
        # its envelope and every scope a reserved notes key, none of it
        # template-declared. The rendered template documents FIELDS, so no
        # notes-field vocabulary may appear in it.
        for view in ("operational", "publication"):
            out = render_template(_template(CONFIG_FIXTURE), view,
                                  _labels(CONFIG_FIXTURE))
            assert "Free-text notes" not in out
            assert "; notes" not in out

    def test_no_soft_canonicalisation_marker(self):
        # soft_canonicalisation is an engine-inert consumer declaration, not
        # extraction guidance a human follows: it must not surface at all.
        for view in ("operational", "publication"):
            out = render_template(_template(CONFIG_FIXTURE), view,
                                  _labels(CONFIG_FIXTURE))
            assert "soft_canonicalisation" not in out

    def test_publication_omits_instructions_and_qa(self):
        out = render_template(_template(CONFIG_FIXTURE), "publication",
                              _labels(CONFIG_FIXTURE))
        # Publication is descriptions only: no field/section/record
        # extraction instructions and no QA or check blocks.
        assert "Extraction instruction" not in out
        assert "Section extraction instruction" not in out
        assert "quality appraisal" not in out.lower()
        assert "Initial check" not in out
        assert "Quality check" not in out
        # A field-level extraction instruction string must not leak in.
        assert "Use exact names from the Gauge Reference List." not in out

    def test_operational_includes_qa_and_check_blocks(self):
        out = render_template(_template(CONFIG_FIXTURE), "operational",
                              _labels(CONFIG_FIXTURE))
        assert "## Study-level quality appraisal" in out
        assert "## Record-level quality appraisal" in out
        assert "## Initial check" in out
        assert "## Quality check" in out

    def test_no_role_marker_rendered(self):
        # `role: summary` is a mechanical-wiring marker, not part of the
        # human-facing render. It must not surface as a field marker. (The
        # word "summary" itself does appear, verbatim, inside the abstract
        # field's extraction instruction; only the marker syntax is banned.)
        out = render_template(_template(CONFIG_FIXTURE), "operational",
                              _labels(CONFIG_FIXTURE))
        assert "_role:" not in out
        assert "role: summary" not in out

    def test_variable_kept_in_operational_dropped_in_publication(self):
        # Operational leads with the human label and keeps the machine variable
        # as a small secondary (an extraction team maps guidance to data columns
        # by it). Publication, for an academic reader, drops the variable.
        op = render_template(_template(CONFIG_FIXTURE), "operational",
                             _labels(CONFIG_FIXTURE))
        pub = render_template(_template(CONFIG_FIXTURE), "publication",
                              _labels(CONFIG_FIXTURE))
        assert "**Study label**<br>`study_label`" in op
        assert "`study_label`" not in pub
        assert "Study label" in pub

    def test_record_entity_named_from_config(self):
        # The entity noun and its plural come from the config, not the code.
        # One fixture is enough to show it: the entity is renamed on the loaded
        # template and the render must follow. Both the fixture's noun and the
        # renamed one are asserted, so a hardcoded noun fails either way.
        template = _template(CONFIG_FIXTURE)
        before = render_template(template, "operational", _labels(CONFIG_FIXTURE))
        assert "`relationship`" in before
        assert "plural: relationships" in before

        renamed = dict(template)
        renamed["record_entity"] = dict(
            template["record_entity"],
            singular="prevalence_estimate", plural="prevalence estimates")
        after = render_template(renamed, "operational", _labels(CONFIG_FIXTURE))
        assert "`prevalence_estimate`" in after
        assert "plural: prevalence estimates" in after
        assert "`relationship`" not in after


# ---------------------------------------------------------------------------
# Operational cell sanitisation: a load-accepted but awkward value (a
# block-scalar reference-list label, a newline inside an option, a pipe or
# newline in a variable) must not corrupt the raw operational table, and the
# operational and publication views must not diverge on it. The render is
# robust on its own: no upstream validation is added.
# ---------------------------------------------------------------------------

class TestCellSanitisation:
    def test_clean_render_rows_all_closed(self):
        # Positive control: on every well-formed config in the tree, every
        # table row is closed, so the well-formedness invariant is meaningful
        # (not trivially true). Discovered rather than named, so adding a
        # config fixture neither skips it silently nor leaves this pointing at
        # a directory that is gone.
        fixtures = Path(__file__).resolve().parent / "fixtures"
        configs = sorted(p.parent for p in fixtures.glob("*/pipeline.yaml"))
        assert configs, "no config fixture found under tests/fixtures/"
        for config in configs:
            op = render_template(_template(config), "operational",
                                 _labels(config))
            assert _pipe_lines_all_closed(op), config.name

    def test_block_scalar_reference_label_renders_one_closed_row(self):
        # A YAML block-scalar / trailing-newline reference-list label is
        # load-accepted (a non-empty string), so the render must tolerate it.
        # This is exactly what load_reference_list_labels passes through for
        # `label: >\n  Gauge Reference List`.
        labels = {"gauge_list": "Gauge Reference List\n"}
        op = render_template(_template(CONFIG_FIXTURE), "operational", labels)
        pub = render_template(_template(CONFIG_FIXTURE), "publication", labels)
        # No row spills a raw newline mid-cell in either view.
        assert _pipe_lines_all_closed(op)
        assert _pipe_lines_all_closed(pub)
        # The value domain reads as the collapsed phrase, with no trailing
        # newline surviving into the cell (that raw form would break the row).
        assert "Names from the Gauge Reference List" in op
        assert "Name from the Gauge Reference List" in op
        assert "Names from the Gauge Reference List\n" not in op
        assert "Name from the Gauge Reference List\n" not in op
        # Operational and publication agree on the same collapsed text.
        assert "Names from the Gauge Reference List" in pub
        assert "Name from the Gauge Reference List" in pub

    def test_op_values_collapses_reference_label_newline(self):
        # Unit-level: the operational value cell is the collapsed form, which is
        # exactly what the publication path (_cell) would emit, so the two views
        # cannot diverge on it.
        labels = {"gauge_list": "Gauge Reference List\n"}
        listf = _field(field_type="string_list", canonical_reference="gauge_list")
        onef = _field(field_type="string", canonical_reference="gauge_list")
        assert _op_values(listf, labels) == "Names from the Gauge Reference List"
        assert _op_values(onef, labels) == "Name from the Gauge Reference List"

    def test_categorical_option_with_newline_renders_safely(self):
        # An option carrying an embedded newline collapses to a single space and
        # the row stays well-formed (header + separator + one data line).
        field = _field(field_type="categorical",
                       options=["Simple", "Multi\nline"], allow_other=True)
        value = _op_values(field, {})
        assert "\n" not in value
        assert "Multi line" in value
        assert "Other (specify)" in value
        table = _raw_table([_op_field_row(field, {})], _OP_HEADER)
        assert _pipe_lines_all_closed(table)
        assert len(table.splitlines()) == 3

    def test_variable_with_pipe_renders_escaped(self):
        # A pipe in the machine variable is escaped inside the backticks, so the
        # column count is preserved and the row stays well-formed.
        field = _field(variable="a|b")
        cell = _op_field_cell(field)
        assert "`a\\|b`" in cell
        table = _raw_table([_op_field_row(field, {})], _OP_HEADER)
        assert _pipe_lines_all_closed(table)
        assert len(table.splitlines()) == 3

    def test_variable_with_newline_renders_collapsed(self):
        # A newline in the machine variable collapses to a single space, so it
        # cannot spill the row across two physical lines.
        field = _field(variable="line1\nline2")
        cell = _op_field_cell(field)
        assert "\n" not in cell
        assert "`line1 line2`" in cell
        table = _raw_table([_op_field_row(field, {})], _OP_HEADER)
        assert _pipe_lines_all_closed(table)
        assert len(table.splitlines()) == 3


# ---------------------------------------------------------------------------
# Strict inputs: fail loudly, write nothing
# ---------------------------------------------------------------------------

class TestStrictInputs:
    def test_missing_config_dir_exits_1(self, tmp_path, capsys):
        out = tmp_path / "template.md"
        code = _run(["render-template", "--config",
                     str(tmp_path / "nope"), "--view", "operational",
                     "--out", str(out)])
        assert code == 1
        assert "does not exist" in capsys.readouterr().err
        assert not out.exists()

    def test_missing_template_exits_1(self, tmp_path, capsys):
        config = tmp_path / "config"
        config.mkdir()
        out = tmp_path / "template.md"
        code = _run(["render-template", "--config", str(config),
                     "--view", "operational", "--out", str(out)])
        assert code == 1
        assert "missing extraction_template.yaml" in capsys.readouterr().err
        assert not out.exists()

    def test_malformed_template_exits_1(self, tmp_path, capsys):
        config = tmp_path / "config"
        config.mkdir()
        # Parses as YAML but violates the template model (unknown top key),
        # so load_template raises ValueError.
        (config / "extraction_template.yaml").write_text(
            "not_a_real_key: true\n", encoding="utf-8")
        out = tmp_path / "template.md"
        code = _run(["render-template", "--config", str(config),
                     "--view", "operational", "--out", str(out)])
        assert code == 1
        assert "could not load extraction template" in capsys.readouterr().err
        assert not out.exists()

    def test_unparseable_yaml_exits_1(self, tmp_path, capsys):
        config = tmp_path / "config"
        config.mkdir()
        # Broken YAML syntax: strict_load raises a yaml.YAMLError.
        (config / "extraction_template.yaml").write_text(
            "study_extraction: [unterminated\n", encoding="utf-8")
        out = tmp_path / "template.md"
        code = _run(["render-template", "--config", str(config),
                     "--view", "operational", "--out", str(out)])
        assert code == 1
        assert "could not load extraction template" in capsys.readouterr().err
        assert not out.exists()

    def test_malformed_reference_label_exits_1(self, tmp_path, capsys):
        # A valid template plus a reference/ file whose `label:` is not a
        # non-empty string: load_reference_list_labels raises ConfigBundleError,
        # so the CLI fails loudly (exit 1, its own message) with nothing
        # written. End-to-end coverage of the label-load failure path the
        # command's docstring documents, not just the loader in isolation.
        config = tmp_path / "config"
        config.mkdir()
        shutil.copy(CONFIG_FIXTURE / "extraction_template.yaml",
                    config / "extraction_template.yaml")
        reference = config / "reference"
        reference.mkdir()
        (reference / "bad_list.yaml").write_text(
            "label: 123\nentries:\n  - alpha\n", encoding="utf-8")
        out = tmp_path / "template.md"
        code = _run(["render-template", "--config", str(config),
                     "--view", "operational", "--out", str(out)])
        assert code == 1
        assert "could not load reference-list labels" in capsys.readouterr().err
        assert not out.exists()


# ---------------------------------------------------------------------------
# Renderer dispatch guard
# ---------------------------------------------------------------------------

def test_unknown_view_raises():
    with pytest.raises(ValueError, match="unknown view"):
        render_template(_template(CONFIG_FIXTURE), "nonsense")
