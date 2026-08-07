"""Reference-list key allowlists: entry keys, entry shape, top-level keys.

A reference list is how a review pins the controlled vocabulary a field must
resolve to. `entry_canonical_name` reads exactly one key, `tool_name`, and
what it returns is used three ways at once: it is rendered into the prompt the
model reads (`render_reference_block`), it is the spelling a matched value is
stored as (`resolve_reference_value`), and it is hashed into
`reference_lists_hash` and through it `config_fp` / `checker_fp`.

That makes a mistyped key a data-integrity fault, not a cosmetic one. Without
the allowlist, `tool_nmae:` leaves the entry with no canonical name, and the
bare-string fallback stringifies the whole mapping — so a Python repr becomes
the vocabulary term in all three places and the run completes clean. A
mistyped `aliases:` is quieter still: it disables that entry's
canonicalisation and signals nothing at all.

These tests pin the load-time refusal of both, the entry-key and top-level-key
allowlists around them, the survival of the bare-string entry form, and the
property that motivates the whole thing: no canonical name of a loaded list is
ever text the review did not write.

`test_reference_lists.py` covers the accessors, the resolution index, and the
`aliases:` VALUE rules (non-empty strings, no duplicates); this file covers
only which keys and shapes may exist.
"""

import pytest
import yaml

from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import reference_lists_hash
from meltiro.reference_lists import (
    entry_canonical_name,
    load_reference_lists,
    render_reference_block,
)


def _load(tmp_path, body, name="gauge_list.yaml"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text(body, encoding="utf-8")
    return load_reference_lists(tmp_path)


def _name_is_authored(entry):
    """True when this entry's canonical name is text the review literally
    wrote, rather than something `str(entry)` synthesised.

    This is the property the entry allowlist buys, stated as a predicate so it
    can be asserted directly in both directions: it must hold for every entry a
    load returns, and it must NOT hold for any shape the loader refuses.
    """
    name = entry_canonical_name(entry)
    if isinstance(entry, str):
        return name == entry
    if isinstance(entry, dict) and isinstance(entry.get("tool_name"), str):
        return name == entry["tool_name"]
    return False


class TestCanonicalNameKey:
    """A mapping entry must carry `tool_name:`; a misspelling is a load error.

    This is the severe case. It is not a silent default -- there is no default
    to fall back to -- it is a corrupted vocabulary term written into the
    prompt, the output artefact, and the fingerprint.
    """

    def test_misspelt_tool_name_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - toolname: WDS-9\n"
                "    aliases: [Widget Durability Scale 9]\n"
            ))
        msg = str(e.value)
        # Names the file, the entry, the offending key, and the legal set.
        assert "gauge_list.yaml" in msg
        assert "entry #1" in msg
        assert "['toolname']" in msg
        assert "'aliases', 'search_terms', 'tool_name'" in msg
        assert "has no `tool_name:`" in msg

    def test_mapping_without_tool_name_rejected(self, tmp_path):
        # Not a misspelling, just an omission: same fault, same refusal.
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - aliases: [WDS9]\n"
            ))
        assert "has no `tool_name:`" in str(e.value)

    def test_non_string_tool_name_rejected(self, tmp_path):
        # A mapping-valued `tool_name:` would stringify to a repr just as a
        # mistyped key did, so the value is constrained as well as the key.
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - tool_name: {name: WDS-9}\n"
            ))
        assert "`tool_name:` that is not a non-empty string" in str(e.value)

    def test_empty_tool_name_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - tool_name: '   '\n"
            ))
        assert "not a non-empty string" in str(e.value)

    def test_every_offending_entry_reported(self, tmp_path):
        # One bad list reports all its problems, matching `_validate_aliases`.
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - toolname: WDS-9\n"
                "  - tool_nmae: SRI-7\n"
            ))
        assert "['toolname']" in str(e.value)
        assert "['tool_nmae']" in str(e.value)


class TestNoReprCanBecomeACanonicalName:
    """The property the allowlist exists to guarantee.

    `entry_canonical_name` ends in `str(entry)`. For a bare string that is the
    identity; for anything with a container in it, it is a Python repr. The
    guarantee is that no file that LOADS can put a repr in that position --
    asserted here directly, both by refusing every shape that would produce
    one and by checking the property over what does load.
    """

    # Each of these, passed to `entry_canonical_name` unvalidated, returns a
    # Python repr or another string the review never wrote.
    SYNTHESISED_SHAPES = [
        ("mapping with a mistyped name key", "  - toolname: WDS-9\n"),
        ("mapping with no recognised key at all", "  - foo: bar\n"),
        ("mapping with a non-string tool_name", "  - tool_name: [WDS-9]\n"),
        ("nested list", "  - [WDS-9, SRI-7]\n"),
        ("blank list item", "  -\n"),
        ("bare number", "  - 2020\n"),
        ("bare boolean", "  - true\n"),
    ]
    _IDS = [s[0] for s in SYNTHESISED_SHAPES]

    @pytest.mark.parametrize("label,item", SYNTHESISED_SHAPES, ids=_IDS)
    def test_shape_is_refused_at_load(self, tmp_path, label, item):
        with pytest.raises(ConfigBundleError):
            _load(tmp_path, "list:\n" + item)

    @pytest.mark.parametrize("label,item", SYNTHESISED_SHAPES, ids=_IDS)
    def test_shape_would_otherwise_corrupt_the_name(self, label, item):
        # Proves the refusals above are load-bearing rather than incidental:
        # unvalidated, every one of these shapes yields a canonical name that
        # is not authored text. Parsed directly, bypassing the loader that now
        # refuses it.
        entry = yaml.safe_load("list:\n" + item)["list"][0]
        assert not _name_is_authored(entry)

    def test_bare_empty_string_rejected(self, tmp_path):
        # Not a repr, but the same class of fault: a canonical name that is
        # not a name. It would render as an empty bullet and resolve nothing.
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, "list:\n  - ''\n")
        assert "is an empty string" in str(e.value)

    def test_canonical_names_of_a_loaded_list_are_authored_text(self,
                                                               tmp_path):
        # The property, stated positively: for every entry a load returns, the
        # canonical name is either the bare string itself or the exact
        # `tool_name` the review wrote -- never a rendering of the entry.
        lists = _load(tmp_path, (
            "label: Gauge Reference List\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
            "    search_terms: Widget Durability Scale; WDS9\n"
            "    aliases: [WDS9]\n"
            "  - Bare Tool\n"
        ))
        assert all(_name_is_authored(e) for e in lists["gauge_list"])

    def test_shipped_fixture_has_no_synthesised_name(self, config_dir):
        # The fixture the golden fingerprints are computed against.
        lists = load_reference_lists(config_dir / "reference")
        assert lists
        for entries in lists.values():
            assert all(_name_is_authored(e) for e in entries)


class TestBareStringEntriesSurvive:
    """The bare-string entry form is supported and must keep working.

    `render_reference_block` renders it, `entry_canonical_name` returns it
    verbatim, and it is the one legitimate user of the `str(entry)` path.
    """

    def test_bare_strings_load(self, tmp_path):
        lists = _load(tmp_path, (
            "list:\n"
            "  - WDS-9\n"
            "  - Composite Rig Test (Heavy Duty)\n"
        ))
        assert lists["gauge_list"] == \
            ["WDS-9", "Composite Rig Test (Heavy Duty)"]

    def test_bare_strings_render_and_canonicalise(self, tmp_path):
        lists = _load(tmp_path, (
            "list:\n"
            "  - WDS-9\n"
            "  - SRI-7\n"
        ))
        entries = lists["gauge_list"]
        assert [entry_canonical_name(e) for e in entries] == ["WDS-9", "SRI-7"]
        assert render_reference_block(entries) == "- WDS-9\n- SRI-7"

    def test_bare_strings_mixed_with_mappings(self, tmp_path):
        lists = _load(tmp_path, (
            "list:\n"
            "  - tool_name: WDS-9\n"
            "    aliases: [WDS9]\n"
            "  - Bare Tool\n"
        ))
        assert render_reference_block(lists["gauge_list"]) == \
            "- WDS-9\n- Bare Tool"

    def test_top_level_yaml_list_of_bare_strings(self, tmp_path):
        # The other legal file shape: no mapping wrapper at all.
        lists = _load(tmp_path, "- WDS-9\n- SRI-7\n")
        assert lists["gauge_list"] == ["WDS-9", "SRI-7"]


class TestEntryKeyAllowlist:
    def test_misspelt_aliases_rejected(self, tmp_path):
        # Was silent: `entry_aliases` returned [] and the entry simply stopped
        # canonicalising, with nothing said and the run completing clean.
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "    alises: [WDS9]\n"
            ))
        msg = str(e.value)
        assert "entry 'WDS-9'" in msg
        assert "unknown entry key(s) ['alises']" in msg

    def test_misspelt_search_terms_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "    search_term: WDS9\n"
            ))
        assert "unknown entry key(s) ['search_term']" in str(e.value)

    def test_unknown_entry_key_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "list:\n"
                "  - tool_name: WDS-9\n"
                "    notes: legacy scale\n"
            ))
        assert "unknown entry key(s) ['notes']" in str(e.value)

    def test_all_three_legal_entry_keys_accepted(self, tmp_path):
        # The complete legal entry key set: `tool_name` (read by
        # `entry_canonical_name`), `aliases` (read by `entry_aliases`), and
        # `search_terms` (read by nothing here, but part of the shape).
        lists = _load(tmp_path, (
            "list:\n"
            "  - tool_name: WDS-9\n"
            "    search_terms: Widget Durability Scale; WDS9\n"
            "    aliases: [WDS9]\n"
        ))
        assert entry_canonical_name(lists["gauge_list"][0]) == "WDS-9"


class TestTopLevelKeyAllowlist:
    def test_misspelt_label_rejected(self, tmp_path):
        # Was silent: any scalar sibling that was not the entries key got
        # dropped, so the list lost its display name and the render fell back
        # to the raw file stem with no signal.
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "labell: Gauge Reference List\n"
                "gauge_reference_list:\n"
                "  - tool_name: WDS-9\n"
            ))
        msg = str(e.value)
        assert "gauge_list.yaml" in msg
        assert "unknown top-level key(s) ['labell']" in msg
        assert "'gauge_reference_list'" in msg
        assert "['label']" in msg

    def test_unknown_top_level_key_rejected(self, tmp_path):
        with pytest.raises(ConfigBundleError) as e:
            _load(tmp_path, (
                "gauge_reference_list:\n"
                "  - tool_name: WDS-9\n"
                "version: 2\n"
            ))
        assert "unknown top-level key(s) ['version']" in str(e.value)

    def test_label_and_entries_key_accepted(self, tmp_path):
        lists = _load(tmp_path, (
            "label: Gauge Reference List\n"
            "gauge_reference_list:\n"
            "  - tool_name: WDS-9\n"
        ))
        assert entry_canonical_name(lists["gauge_list"][0]) == "WDS-9"


class TestFingerprintCannotRecordACorruptedName:
    """`reference_lists_hash` hashes `entry_canonical_name` of every entry.

    A mis-keyed entry therefore reached the fingerprint preimage as its own
    Python repr -- and a repr follows YAML key order, so re-ordering two keys
    in the file moved `config_fp` while changing nothing the pipeline does.
    A published fingerprint could describe a vocabulary nobody wrote. The load
    error is what makes that unreachable.
    """

    def test_mis_keyed_entry_cannot_reach_a_preimage(self, tmp_path):
        with pytest.raises(ConfigBundleError):
            _load(tmp_path, (
                "list:\n"
                "  - toolname: WDS-9\n"
                "    aliases: [Widget Durability Scale]\n"
            ))

    def test_key_order_does_not_move_the_hash_for_legal_entries(self,
                                                                tmp_path):
        # The corollary, and the sharpest reason a repr must never become a
        # canonical name: for entries that load, the hash is a function of the
        # canonical names and aliases, never of the order the YAML keys were
        # written in. A repr carries that order, and so would break this.
        a = _load(tmp_path / "a", (
            "list:\n"
            "  - tool_name: WDS-9\n"
            "    aliases: [WDS9]\n"
        ))
        b = _load(tmp_path / "b", (
            "list:\n"
            "  - aliases: [WDS9]\n"
            "    tool_name: WDS-9\n"
        ))
        assert reference_lists_hash(a) == reference_lists_hash(b)

    def test_shipped_fixture_still_loads_and_hashes(self, config_dir):
        # The frozen fixture must survive the new allowlists unchanged: the
        # golden fingerprints in test_fingerprint_goldens.py are computed
        # against it.
        lists = load_reference_lists(config_dir / "reference")
        assert [entry_canonical_name(e) for e in lists["gauge_list"]] == [
            "Widget Durability Scale 9 (WDS-9)",
            "Surface Resistance Index 7 (SRI-7)",
            "Composite Rig Test (Heavy Duty)",
            "Bracket Load Rating (BLR)",
            "Coupling Wear Score (CWS)",
        ]
        assert len(reference_lists_hash(lists)) == 64
