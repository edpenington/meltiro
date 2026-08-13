"""Reference-list resolution index, value resolution, and alias load rules."""

import pytest

from meltiro.errors import ConfigBundleError
from meltiro.reference_lists import (
    build_reference_index,
    entry_aliases,
    entry_canonical_name,
    load_reference_list_labels,
    load_reference_lists,
    rank_reference_suggestions,
    render_reference_block,
    resolve_reference_value,
    substitute_reference_placeholders,
)


ENTRIES = [
    {"tool_name": "WDS-9",
     "aliases": ["Widget Durability Scale 9", "WDS9"]},
    {"tool_name": "SRI-7", "aliases": ["SRI7"]},
    {"tool_name": "Composite Rig Test (Heavy Duty)"},
]


class TestEntryAccessors:
    def test_canonical_name_uses_tool_name(self):
        assert entry_canonical_name(ENTRIES[0]) == "WDS-9"

    def test_canonical_name_stringifies_plain_entry(self):
        assert entry_canonical_name("Bare Tool") == "Bare Tool"

    def test_aliases_default_empty(self):
        assert entry_aliases(ENTRIES[2]) == []

    def test_aliases_returned(self):
        assert entry_aliases(ENTRIES[1]) == ["SRI7"]

    def test_render_block_shows_only_names(self):
        block = render_reference_block(ENTRIES)
        assert "- WDS-9" in block
        assert "- SRI-7" in block
        # Aliases are a matching aid, never rendered.
        assert "WDS9" not in block
        assert "Widget Durability Scale 9" not in block


class TestResolveReferenceValue:
    def setup_method(self):
        self.index = build_reference_index(ENTRIES)

    def test_exact_canonical_name(self):
        r = resolve_reference_value("WDS-9", self.index)
        assert r == {"status": "canonical", "canonical": "WDS-9"}

    def test_canonical_name_case_and_space_insensitive(self):
        r = resolve_reference_value("  wds-9 ", self.index)
        assert r["status"] == "canonical"
        assert r["canonical"] == "WDS-9"

    def test_unique_alias(self):
        r = resolve_reference_value("widget durability scale 9",
                                    self.index)
        assert r == {"status": "alias", "canonical": "WDS-9"}

    def test_no_match_returns_ranked_suggestions(self):
        r = resolve_reference_value("WDS", self.index)
        assert r["status"] == "no_match"
        # WDS-9 is by far the closest.
        assert r["suggestions"][0] == "WDS-9"
        assert len(r["suggestions"]) <= 5

    def test_ambiguous_alias_lists_candidates(self):
        # build_reference_index does no validation, so a shared alias
        # produces a genuine ambiguous resolution.
        ambiguous = build_reference_index([
            {"tool_name": "Alpha Tool", "aliases": ["shared handle"]},
            {"tool_name": "Beta Tool", "aliases": ["shared handle"]},
        ])
        r = resolve_reference_value("shared handle", ambiguous)
        assert r["status"] == "ambiguous"
        assert r["candidates"] == ["Alpha Tool", "Beta Tool"]

    def test_canonical_name_takes_priority_over_alias(self):
        # "WDS-9" is entry A's name and (contrived) entry B's alias; the
        # name match wins.
        index = build_reference_index([
            {"tool_name": "WDS-9"},
            {"tool_name": "Other Tool", "aliases": ["WDS-9"]},
        ])
        r = resolve_reference_value("WDS-9", index)
        assert r == {"status": "canonical", "canonical": "WDS-9"}

    def test_empty_value(self):
        assert resolve_reference_value("   ", self.index)["status"] == "empty"


class TestRankSuggestions:
    def test_returns_canonical_names_from_alias_match(self):
        index = build_reference_index(ENTRIES)
        # Query close to an alias, not a canonical name.
        ranked = rank_reference_suggestions("sri7x", index)
        assert ranked[0] == "SRI-7"


class TestAliasLoadValidation:
    def _write(self, tmp_path, body):
        (tmp_path / "reflist.yaml").write_text(body, encoding="utf-8")
        return load_reference_lists(tmp_path)

    def test_valid_aliases_load(self, tmp_path):
        lists = self._write(tmp_path, (
            "list:\n"
            "  - tool_name: WDS-9\n"
            "    aliases: [WDS9, widget durability scale 9]\n"
            "  - tool_name: SRI-7\n"
            "    aliases: [SRI7]\n"
        ))
        assert entry_aliases(lists["reflist"][0]) == \
            ["WDS9", "widget durability scale 9"]

    def test_alias_equal_to_own_name_allowed(self, tmp_path):
        # Redundant but harmless.
        self._write(tmp_path, (
            "list:\n"
            "  - tool_name: WDS-9\n"
            "    aliases: [WDS-9, WDS9]\n"
        ))

    def test_empty_alias_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "    aliases: ['']\n"
            ))
        assert "non-empty string" in str(e.value)

    def test_alias_duplicating_another_alias_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "    aliases: [shared]\n"
                "  - tool_name: SRI-7\n"
                "    aliases: [Shared]\n"
            ))
        assert "duplicates an alias" in str(e.value)

    def test_alias_duplicating_another_name_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "    aliases: [sri-7]\n"
                "  - tool_name: SRI-7\n"
            ))
        assert "duplicates the canonical name" in str(e.value)


class TestCollidingCanonicalNames:
    """Two canonical names sharing a normalised form make the list unusable,
    so the collision is a load error.

    Matching is by normalised form, so both entries land under one index key
    and `resolve_reference_value` answers `ambiguous` for every value entered
    for either name -- each name typed exactly as written included. A field
    referencing that list can then never be filled: no spelling resolves. The
    fault is in the file, so it is caught where the file is read.
    """

    def _write(self, tmp_path, body):
        (tmp_path / "reflist.yaml").write_text(body, encoding="utf-8")
        return load_reference_lists(tmp_path)

    def test_names_differing_only_in_case_are_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "  - tool_name: wds-9\n"
            ))
        message = str(e.value)
        # Both offending entries are named: the two spellings are by
        # construction hard to tell apart by eye.
        assert "'WDS-9'" in message
        assert "'wds-9'" in message
        assert "normalise" in message

    def test_names_differing_only_in_dash_style_are_rejected(self, tmp_path):
        # The normaliser folds dash variants, so an en dash typed where a
        # hyphen was meant is the same name written twice. This is the case
        # least visible in a diff and most likely to survive review.
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "  - tool_name: WDS\N{EN DASH}9\n"
            ))
        assert "normalise" in str(e.value)

    def test_bare_string_entries_collide_too(self, tmp_path):
        # The rule is about canonical names, not about which entry form
        # declares them.
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - Composite Rig Test\n"
                "  - composite  rig  test\n"
            ))
        assert "normalise" in str(e.value)

    def test_every_colliding_pair_is_reported(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            self._write(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "  - tool_name: wds-9\n"
                "  - tool_name: SRI-7\n"
                "  - tool_name: sri-7\n"
            ))
        assert len(e.value.problems) == 2

    def test_distinct_names_still_load(self, tmp_path):
        # The guard against a check that rejects everything.
        lists = self._write(tmp_path, (
            "list:\n"
            "  - tool_name: WDS-9\n"
            "  - tool_name: SRI-7\n"
        ))
        assert len(lists["reflist"]) == 2

    def test_the_collision_is_what_makes_a_field_unfillable(self):
        # Why the load-time rejection is worth having, stated as the behaviour
        # it prevents: with the collision in place, the canonical name typed
        # exactly as written resolves to `ambiguous`, so no value is accepted.
        index = build_reference_index([{"tool_name": "WDS-9"},
                                       {"tool_name": "wds-9"}])
        for spelling in ("WDS-9", "wds-9"):
            assert resolve_reference_value(spelling, index)["status"] == \
                "ambiguous"


class TestReferenceListLabels:
    """The optional list-level display `label:`, consumed only by the render.

    The label is a scalar sibling of the single entries key. It is inert to
    the entries loader's `{name: entries}` contract and to every fingerprint;
    a separate loader reads it for human-facing rendering.
    """

    def _write(self, tmp_path, body, name="gauge_list.yaml"):
        (tmp_path / name).write_text(body, encoding="utf-8")

    def test_label_read_by_stem(self, tmp_path):
        self._write(tmp_path, (
            "label: Gauge Reference List\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ))
        assert load_reference_list_labels(tmp_path) == \
            {"gauge_list": "Gauge Reference List"}

    def test_label_absent_list_is_omitted(self, tmp_path):
        # A label-less list simply does not appear in the labels map, so its
        # caller falls back to the raw stem.
        self._write(tmp_path, (
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ))
        assert load_reference_list_labels(tmp_path) == {}

    def test_missing_reference_dir_yields_no_labels(self, tmp_path):
        assert load_reference_list_labels(tmp_path / "nope") == {}

    def test_label_does_not_enter_entries(self, tmp_path):
        # The entries loader keeps its `{name: entries}` shape: the scalar
        # label is not mistaken for (or added to) the entries.
        self._write(tmp_path, (
            "label: Gauge Reference List\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
            "  - tool_name: SRI-7\n"
        ))
        lists = load_reference_lists(tmp_path)
        assert [entry_canonical_name(e) for e in lists["gauge_list"]] == \
            ["WDS-9", "SRI-7"]

    def test_non_string_label_rejected_by_labels_loader(self, tmp_path):
        self._write(tmp_path, (
            "label: 123\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ))
        with pytest.raises(ConfigBundleError) as e:
            load_reference_list_labels(tmp_path)
        assert "not a non-empty string" in str(e.value)

    def test_empty_label_rejected(self, tmp_path):
        self._write(tmp_path, (
            "label: '   '\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ))
        with pytest.raises(ConfigBundleError) as e:
            load_reference_list_labels(tmp_path)
        assert "not a non-empty string" in str(e.value)

    def test_malformed_label_also_rejected_by_entries_loader(self, tmp_path):
        # Strict inputs: a malformed label fails wherever reference lists load,
        # not only in the render path, so a bad label can never reach a run.
        self._write(tmp_path, (
            "label: 123\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ))
        with pytest.raises(ConfigBundleError) as e:
            load_reference_lists(tmp_path)
        assert "not a non-empty string" in str(e.value)


class TestBothLoadersEnforceOneRuleSet:
    """A file is legal or illegal on the same terms whichever loader reads it.

    The labels loader exists for the template render, which wants a display
    name and nothing else. That is not licence to accept a file the pipeline
    would refuse: the label describes a vocabulary, and rendering "Name from
    the Gauge Reference List" off a list whose entries would never load is a
    published claim about something unusable. Every rule below is asserted
    through BOTH doors, so the two can never drift into disagreeing.
    """

    FAULTS = {
        "unknown entry key": (
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
            "    alises: [WDS9]\n"
        ),
        "missing tool_name": (
            "gauge_reference_list:\n"
            "  - toolname: WDS-9\n"
        ),
        "non-mapping entry": (
            "gauge_reference_list:\n"
            "  - 42\n"
        ),
        "no entries": (
            "gauge_reference_list: []\n"
        ),
        "colliding canonical names": (
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
            "  - tool_name: wds-9\n"
        ),
        "duplicated alias": (
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
            "    aliases: [shared]\n"
            "  - tool_name: SRI-7\n"
            "    aliases: [Shared]\n"
        ),
        "unknown top-level key": (
            "labell: Gauge Reference List\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ),
    }

    def _write(self, tmp_path, body, *, labelled=True):
        text = "label: Gauge Reference List\n" + body if labelled else body
        (tmp_path / "gauge_list.yaml").write_text(text, encoding="utf-8")

    @pytest.mark.parametrize("fault", sorted(FAULTS))
    def test_the_entries_loader_rejects_it(self, tmp_path, fault):
        self._write(tmp_path, self.FAULTS[fault])
        with pytest.raises(ConfigBundleError):
            load_reference_lists(tmp_path)

    @pytest.mark.parametrize("fault", sorted(FAULTS))
    def test_the_labels_loader_rejects_it_too(self, tmp_path, fault):
        # The label is present and perfectly well-formed in every case here,
        # so nothing but the entries rule can be doing the rejecting.
        self._write(tmp_path, self.FAULTS[fault])
        with pytest.raises(ConfigBundleError):
            load_reference_list_labels(tmp_path)

    @pytest.mark.parametrize("fault", sorted(FAULTS))
    def test_the_labels_loader_rejects_it_without_a_label_too(
            self, tmp_path, fault):
        # A label-less list contributes nothing to the returned map, but it is
        # still read, so it is still held to the rules.
        self._write(tmp_path, self.FAULTS[fault], labelled=False)
        with pytest.raises(ConfigBundleError):
            load_reference_list_labels(tmp_path)

    def test_a_sound_list_loads_through_both(self, tmp_path):
        # The guard against loaders that reject everything.
        self._write(tmp_path, (
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
            "    aliases: [WDS9]\n"
        ))
        assert load_reference_lists(tmp_path)["gauge_list"]
        assert load_reference_list_labels(tmp_path) == \
            {"gauge_list": "Gauge Reference List"}


class TestUnresolvablePlaceholder:
    """A `{reference:NAME}` naming a list the bundle does not provide fails,
    and the failure says which bundle is short a list.

    The fix is a file in one directory's `reference/`, so an operator running
    several bundles has to be told which one to open. Every other
    `ConfigBundleError` this module raises carries that locator; this one is
    raised from a render rather than a read, which is exactly when the operator
    is least likely to have the directory in mind.
    """

    TEXT = "Choose from:\n{reference:gauge_list}\n"

    def test_the_error_carries_the_bundle_location(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            substitute_reference_placeholders(self.TEXT, {}, path=tmp_path)
        assert e.value.path == tmp_path
        assert str(tmp_path) in str(e.value)

    def test_the_error_names_the_list_and_what_is_available(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            substitute_reference_placeholders(
                self.TEXT, {"other_list": ["A"]}, path=tmp_path)
        message = str(e.value)
        assert "gauge_list" in message
        assert "other_list" in message

    def test_a_caller_holding_no_bundle_path_still_gets_the_message(self):
        # The locator is optional: a render with no bundle behind it names the
        # list and the placeholder, and invents no directory.
        with pytest.raises(ConfigBundleError) as e:
            substitute_reference_placeholders(self.TEXT, {})
        assert e.value.path is None
        assert "gauge_list" in str(e.value)

    def test_a_resolvable_placeholder_is_substituted(self, tmp_path):
        out = substitute_reference_placeholders(
            self.TEXT, {"gauge_list": ["WDS-9"]}, path=tmp_path)
        assert "- WDS-9" in out
        assert "{reference:" not in out


class TestRenderersCarryTheBundleLocation:
    """The three prompt renderers pass the bundle root through, so the locator
    is present where the failure actually happens rather than only in a unit
    test of the substituter."""

    def _prompt(self, tmp_path, name):
        prompts = tmp_path / "prompts"
        prompts.mkdir(exist_ok=True)
        path = prompts / name
        path.write_text("Choose from:\n{reference:gauge_list}\n",
                        encoding="utf-8")
        return path

    def test_extractor_system_render(self, tmp_path):
        from meltiro.prompt_builder import build_system_message
        path = self._prompt(tmp_path, "extractor_system.md")
        with pytest.raises(ConfigBundleError) as e:
            build_system_message(system_prompt_path=path,
                                 reference_lists={})
        assert e.value.path == tmp_path

    def test_review_system_render(self, tmp_path):
        from meltiro.prompt_builder import build_review_system_message
        path = self._prompt(tmp_path, "review_system.md")
        with pytest.raises(ConfigBundleError) as e:
            build_review_system_message(system_prompt_path=path,
                                        reference_lists={})
        assert e.value.path == tmp_path

    def test_checker_system_render(self, tmp_path):
        from meltiro.checker_prompts import build_checker_system_text
        path = self._prompt(tmp_path, "checker_system.md")
        with pytest.raises(ConfigBundleError) as e:
            build_checker_system_text(system_prompt_path=path,
                                      max_checks_per_field=2,
                                      reference_lists={})
        assert e.value.path == tmp_path
