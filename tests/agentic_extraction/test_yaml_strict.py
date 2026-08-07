"""Tests for the strict config YAML loader (meltiro.yaml_strict).

PyYAML's `safe_load` silently keeps the last of a duplicated mapping key. The
strict loader rejects the duplicate at parse time so a config file that repeats
a key fails loudly instead of running with whichever value happened to come
last. These tests pin the loader itself; the parse-site tests (that
pipeline.yaml, the template, and reference lists all route through it) live with
each site's own tests (test_config_bundle.py) and here for the template.
"""

import textwrap

import pytest
import yaml

from meltiro.template import load_template
from meltiro.yaml_strict import StrictLoader, strict_load


class TestDuplicateKeys:
    def test_duplicate_key_rejected_naming_the_key(self):
        with pytest.raises(yaml.YAMLError) as excinfo:
            strict_load("max_checks_per_field: 1\nmax_checks_per_field: 2\n")
        msg = str(excinfo.value)
        assert "found duplicate key" in msg
        assert "max_checks_per_field" in msg

    def test_nested_duplicate_rejected(self):
        text = textwrap.dedent(
            """
            outer:
              inner: 1
              inner: 2
            """
        )
        with pytest.raises(yaml.YAMLError) as excinfo:
            strict_load(text)
        assert "inner" in str(excinfo.value)

    def test_duplicate_inside_list_entry_rejected(self):
        # A list of mappings where one entry repeats a key (the shape a
        # reference-list entry takes) is caught too.
        text = textwrap.dedent(
            """
            - name: A
              name: B
            """
        )
        with pytest.raises(yaml.YAMLError) as excinfo:
            strict_load(text)
        assert "name" in str(excinfo.value)


class TestValidDocuments:
    def test_plain_mapping_parses(self):
        assert strict_load("a: 1\nb: 2\n") == {"a": 1, "b": 2}

    def test_merge_key_is_not_a_duplicate(self):
        # `<<` merges an alias into a mapping. The duplicate scan runs BEFORE
        # flatten_mapping and skips merge entries, scanning only the keys the
        # source mapping wrote explicitly; the base loader then resolves the
        # merge, so an override of a merged key wins (standard YAML merge
        # semantics). Do not reorder to flatten-first: PyYAML prepends merged
        # pairs during flatten, which makes a legal override look like a
        # duplicate and rejects exactly this document.
        text = textwrap.dedent(
            """
            defaults: &d
              a: 1
              b: 2
            override:
              <<: *d
              b: 3
            """
        )
        doc = strict_load(text)
        assert doc["override"] == {"a": 1, "b": 3}

    def test_repeated_key_across_sibling_mappings_is_fine(self):
        # The same key name in two different mappings is not a duplicate.
        doc = strict_load("one:\n  k: 1\ntwo:\n  k: 2\n")
        assert doc == {"one": {"k": 1}, "two": {"k": 2}}


class TestLoaderProperties:
    def test_is_a_safeloader_subclass(self):
        assert issubclass(StrictLoader, yaml.SafeLoader)

    def test_rejects_unsafe_python_tags(self):
        # Subclassing SafeLoader preserves its safety: arbitrary Python object
        # tags are still rejected.
        with pytest.raises(yaml.YAMLError):
            strict_load("!!python/object/apply:os.system ['echo hi']")

    def test_unhashable_key_falls_back_to_pyyaml_diagnostic(self):
        # An unhashable mapping key cannot be membership-tested, so the loader
        # hands the node back to PyYAML and its own "unhashable key" diagnostic
        # surfaces rather than a spurious duplicate error.
        with pytest.raises(yaml.YAMLError) as excinfo:
            strict_load("? [a, b]\n: c\n")
        assert "unhashable" in str(excinfo.value)


class TestTemplateParseSite:
    def test_duplicate_top_level_template_key_rejected(self, tmp_path):
        # load_template parses through strict_load before any template-shape
        # validation, so a duplicated top-level key fails at the parse.
        template = tmp_path / "extraction_template.yaml"
        template.write_text(
            "records:\n  a: 1\nrecords:\n  b: 2\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError) as excinfo:
            load_template(template)
        assert "records" in str(excinfo.value)
