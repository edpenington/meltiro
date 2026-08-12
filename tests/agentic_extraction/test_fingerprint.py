"""Fingerprints: what a run's identity is made of, and what it ignores.

A fingerprint answers one question. Two runs carry the same fingerprint if and
only if they were produced by the same configuration, so a difference in their
output is a difference in the model's behaviour rather than a difference in
what it was asked. Everything else follows from that, and every test in this
module is one of two claims about it.

An axis that CHANGES the model's inputs must move the hash. A field's
description, its extraction instruction, its type, its required flag, a
reference list's aliases, the record entity's own wording: each of these
reaches the model, through a tool schema or a rendered prompt, so a run made
after editing one is not comparable with a run made before. A hash that failed
to move here would let two incomparable runs be compared silently, which is
the expensive failure.

An axis that does NOT change the model's inputs must leave the hash alone. A
presentation-only `label`, a reference list's `search_terms`, a retired key
left in a template dict: these are invisible to the model, so folding them in
would refuse a resume, invalidate a cached comparison, or fork a run's identity
over an edit that changed nothing the model saw. That failure is cheaper but
far more common, so both directions are asserted for every axis.

The stage fingerprints are separable on purpose. `config_fp`, `checker_fp` and
`review_fp` each cover one stage, `instrument_fp` is deliberately model-free so
it survives a repoint, and `engine_fp` covers the code itself. Editing the
review prompt must move `review_fp` and ONLY `review_fp`; a component that
leaks across stages makes every fingerprint mean the same thing, which is to
say it makes none of them mean anything. The decoding parameters actually sent
on the wire come from `direktoro`'s call-identity block rather than being
recomposed here, so what is fingerprinted is what is sent.
"""

import hashlib

from meltiro.fingerprint import (
    ABSENT_STAGE,
    engine_fingerprint,
    checker_config_fingerprint,
    config_fingerprint,
    structure_hash,
    field_catalogue_hash,
    reference_lists_hash,
    review_config_fingerprint,
    run_fingerprint,
    tool_set_hash,
)
from direktoro import (
    call_identity_fields, canonical_json, resolved_decoding_params)
from meltiro.tools import get_tool_definitions


class TestToolSetHash:
    def test_stable_across_runs(self, synthetic_template):
        a = tool_set_hash(get_tool_definitions(synthetic_template))
        b = tool_set_hash(get_tool_definitions(synthetic_template))
        assert a == b

    def test_stable_under_dict_key_reordering(self):
        # The canonical serialisation sorts keys, so two tool lists that
        # differ only in dict-key insertion order should hash identically.
        a = [{"name": "x", "input_schema": {"type": "object",
                                            "properties": {}}}]
        b = [{"input_schema": {"properties": {}, "type": "object"},
              "name": "x"}]
        assert tool_set_hash(a) == tool_set_hash(b)

    def test_changes_with_description(self):
        tools_a = [{"name": "x", "description": "first"}]
        tools_b = [{"name": "x", "description": "second"}]
        assert tool_set_hash(tools_a) != tool_set_hash(tools_b)

    def test_changes_with_field_extraction_instruction(
            self, synthetic_template):
        # A field's extraction_instruction is composed into its tool-schema
        # property description (via _field_description), so toggling it must
        # move the tool-set hash. Every fixture leaves it None, so this test
        # sets it explicitly rather than relying on a fixture to cover it.
        a = tool_set_hash(get_tool_definitions(synthetic_template))
        synthetic_template["study_fields"][0]["fields"][0][
            "extraction_instruction"] = "Record the primary aim verbatim."
        b = tool_set_hash(get_tool_definitions(synthetic_template))
        assert a != b

    def test_field_extraction_instruction_renders_into_tool_description(
            self, synthetic_template):
        # The instruction text itself reaches the model: it renders into the
        # field's property description inside the update_study tool schema.
        synthetic_template["study_fields"][0]["fields"][0][
            "extraction_instruction"] = "Count each study arm separately."
        update_study = next(
            t for t in get_tool_definitions(synthetic_template)
            if t["name"] == "update_study")
        desc = update_study["input_schema"]["properties"]["study"][
            "properties"]["primary_aim"]["description"]
        assert "Count each study arm separately." in desc


class TestFieldCatalogueHash:
    def test_stable(self, synthetic_template):
        a = field_catalogue_hash(synthetic_template)
        b = field_catalogue_hash(synthetic_template)
        assert a == b

    def test_changes_with_description(self, synthetic_template):
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["description"] = \
            "different description"
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_changes_with_extraction_instruction(self, synthetic_template):
        # A field's extraction_instruction is model-facing guidance folded
        # into the field catalogue, so toggling it must move the hash. Every
        # fixture leaves it None, so this test sets it explicitly rather than
        # relying on a fixture to cover it.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0][
            "extraction_instruction"] = "Record the primary aim verbatim."
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_changes_with_field_type(self, synthetic_template):
        # field_type alters the checker's rendered `_Type_` line, so it reaches
        # the model and must move the hash.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][1]["field_type"] = \
            "number"
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_changes_with_allow_other(self, synthetic_template):
        # allow_other switches a categorical between a hard enum and an open
        # list (changing both the checker wording and the validator's
        # acceptance rule), so it must move the hash.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["record_fields"][0]["fields"][2]["allow_other"] = \
            True
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_changes_with_evidence(self, synthetic_template):
        # evidence (required vs optional) changes what the checker demands.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["evidence"] = \
            "optional"
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_ignores_dead_allowed_values_key(self, synthetic_template):
        # `allowed_values` never exists in a parsed field dict; it is no
        # longer hashed, so setting it must NOT move the hash.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["allowed_values"] = \
            "some stray value"
        b = field_catalogue_hash(synthetic_template)
        assert a == b

    def test_changes_with_required(self, synthetic_template):
        # `required` is now template-declared and drives the mark_complete
        # gate, so toggling it must move the hash (unlike the presentation
        # -only `label`).
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["required"] = True
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_ignores_label(self, synthetic_template):
        # `label` is presentation-only and must NOT move the field-catalogue
        # hash, so the checker fingerprint ignores it. (It does still move
        # config_fp, which folds in the whole-file template_hash; this test
        # covers ONLY the field-catalogue hash.)
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["label"] = \
            "A totally different label"
        synthetic_template["study_fields"][0]["label"] = "Renamed section"
        b = field_catalogue_hash(synthetic_template)
        assert a == b

    def test_changes_with_canonical_reference(self, synthetic_template):
        # canonical_reference changes a field's validation surface (strict
        # reference matching), so it must move the hash.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["record_fields"][0]["fields"][0][
            "canonical_reference"] = "gauge_list"
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_changes_with_role(self, synthetic_template):
        # `role` decides WHICH field supplies a value the checker is shown, not
        # merely how a field renders. The `role: summary` field's value is the
        # second-precedence source of the study-identity context handed to
        # EVERY checker call, so moving the role changes what every study-level
        # check is judged against — and it selects the value the
        # summary-mismatch tripwire compares, and so what a run records in
        # meta.warnings. Excluding it would let a template edit change every
        # checker call while checker_fp reported the instrument unchanged.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["role"] = "summary"
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_moving_the_role_to_another_field_moves_it(
            self, synthetic_template):
        # The direction that matters most and the one a presence check misses:
        # the same role, declared on a different field, is a different
        # identity context for every check in the run.
        synthetic_template["study_fields"][0]["fields"][0]["role"] = "summary"
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["study_fields"][0]["fields"][0]["role"] = None
        synthetic_template["study_fields"][0]["fields"][1]["role"] = "summary"
        b = field_catalogue_hash(synthetic_template)
        assert a != b

    def test_ignores_an_unrecognised_multivalue_key(self, synthetic_template):
        # `multivalue` is not a field key: list-valued reference fields are
        # string_list, covered by field_type. A stray key in a field dict must
        # NOT move the hash.
        a = field_catalogue_hash(synthetic_template)
        synthetic_template["record_fields"][0]["fields"][0]["multivalue"] = \
            True
        b = field_catalogue_hash(synthetic_template)
        assert a == b


class TestEntityFingerprintCoverage:
    """The record entity's name, plural and description are rendered into the
    tool descriptions, so editing any one of them moves the tool-set hash.
    The entity's wording is model-facing text, not template bookkeeping."""

    def _hash(self, template):
        return tool_set_hash(get_tool_definitions(template))

    def test_singular_moves_tool_set_hash(self, synthetic_template):
        before = self._hash(synthetic_template)
        synthetic_template["record_entity"]["singular"] = "finding"
        assert self._hash(synthetic_template) != before

    def test_plural_moves_tool_set_hash(self, synthetic_template):
        before = self._hash(synthetic_template)
        synthetic_template["record_entity"]["plural"] = "findings"
        assert self._hash(synthetic_template) != before

    def test_description_moves_tool_set_hash(self, synthetic_template):
        before = self._hash(synthetic_template)
        synthetic_template["record_entity"]["description"] = "something else"
        assert self._hash(synthetic_template) != before

    def test_extraction_instruction_moves_tool_set_hash(
            self, synthetic_template):
        # The record-level extraction_instruction leads the add_record tool
        # description (what counts as one record), so setting it must move the
        # tool-set hash. The fixture entity omits it, so this test sets it
        # explicitly rather than relying on a fixture to cover it.
        before = self._hash(synthetic_template)
        synthetic_template["record_entity"][
            "extraction_instruction"] = "One row per reported relationship."
        assert self._hash(synthetic_template) != before


class TestReferenceListsHash:
    _A = {"gauge_list": [{"tool_name": "WDS-9", "aliases": ["WDS9"]},
                         {"tool_name": "SRI-7"}]}
    # Same names, one alias edited.
    _B = {"gauge_list": [{"tool_name": "WDS-9", "aliases": ["WDS-9 short"]},
                         {"tool_name": "SRI-7"}]}

    def test_alias_edit_moves_reference_hash(self):
        assert reference_lists_hash(self._A) != reference_lists_hash(self._B)

    def test_list_name_order_is_inert(self):
        # The canonical serialisation sorts the payload's keys, so the order
        # the lists are iterated in cannot reach the digest. Two mappings with
        # the same lists in different insertion order are one instrument.
        first = {"gauge_list": [{"tool_name": "WDS-9"}],
                 "outcome_list": [{"tool_name": "Failure"}]}
        second = {"outcome_list": [{"tool_name": "Failure"}],
                  "gauge_list": [{"tool_name": "WDS-9"}]}
        assert list(first) != list(second)
        assert reference_lists_hash(first) == reference_lists_hash(second)

    def test_entry_order_within_a_list_is_content(self):
        # The ordering the hash must NOT flatten. Entries are rendered into a
        # prompt in file order, so reordering them is a genuine edit to what a
        # model reads and has to move the hash.
        forwards = {"gauge_list": [{"tool_name": "WDS-9"},
                                   {"tool_name": "SRI-7"}]}
        backwards = {"gauge_list": [{"tool_name": "SRI-7"},
                                    {"tool_name": "WDS-9"}]}
        assert reference_lists_hash(forwards) != reference_lists_hash(backwards)

    def test_alias_order_within_an_entry_is_content(self):
        first = {"gauge_list": [
            {"tool_name": "WDS-9", "aliases": ["WDS9", "WDS 9"]}]}
        second = {"gauge_list": [
            {"tool_name": "WDS-9", "aliases": ["WDS 9", "WDS9"]}]}
        assert reference_lists_hash(first) != reference_lists_hash(second)

    def test_search_terms_do_not_move_reference_hash(self):
        # search_terms are neither rendered nor used to canonicalise, so an
        # edit to them must NOT move the hash.
        with_terms = {"gauge_list": [
            {"tool_name": "WDS-9", "aliases": ["WDS9"],
             "search_terms": "one; two"}]}
        without = {"gauge_list": [
            {"tool_name": "WDS-9", "aliases": ["WDS9"],
             "search_terms": "three; four"}]}
        assert reference_lists_hash(with_terms) == \
            reference_lists_hash(without)

    def test_list_label_does_not_move_reference_hash(self, tmp_path):
        # A reference list's presentation-only `label:` is inert to the content
        # hash: it is not an entry, so load_reference_lists never returns it and
        # reference_lists_hash cannot see it. Mirror of the field-level `label`
        # inertness test above: adding a display label moves NO fingerprint.
        # Exercised through the loader so the whole path is real.
        from meltiro.reference_lists import load_reference_lists

        entries = ("list:\n"
                   "  - tool_name: WDS-9\n"
                   "    aliases: [WDS9]\n"
                   "  - tool_name: SRI-7\n")
        without = tmp_path / "without"
        with_label = tmp_path / "with"
        without.mkdir()
        with_label.mkdir()
        (without / "reflist.yaml").write_text(entries, encoding="utf-8")
        (with_label / "reflist.yaml").write_text(
            "label: Gauge Reference List\n" + entries, encoding="utf-8")
        assert reference_lists_hash(load_reference_lists(without)) == \
            reference_lists_hash(load_reference_lists(with_label))

    def test_alias_edit_moves_config_fp(self):
        # Aliases are not rendered into prompts, so config_fp only moves via
        # the reference-content component. Same prompt/template/tool hashes,
        # different alias -> different config_fp.
        fp_a = config_fingerprint(
            "opus", "prompt", "template", tool_set_hash="tools",
            reference_hash=reference_lists_hash(self._A))
        fp_b = config_fingerprint(
            "opus", "prompt", "template", tool_set_hash="tools",
            reference_hash=reference_lists_hash(self._B))
        assert fp_a != fp_b

    def test_alias_edit_moves_checker_fp(self):
        fp_a = checker_config_fingerprint(
            "sonnet", "sys", "user", field_catalogue_hash_str="cat",
            reference_hash=reference_lists_hash(self._A))
        fp_b = checker_config_fingerprint(
            "sonnet", "sys", "user", field_catalogue_hash_str="cat",
            reference_hash=reference_lists_hash(self._B))
        assert fp_a != fp_b

    def test_alias_edit_moves_review_fp(self):
        # The reviewer drives the same ToolDispatcher the extractor does, so an
        # alias edit changes which values its tool calls may write and what
        # they canonicalise to. Aliases reach no prompt, so the content hash is
        # the only route by which review_fp can see it.
        fp_a = review_config_fingerprint(
            "opus", "revsys", tool_set_hash="tools",
            reference_hash=reference_lists_hash(self._A))
        fp_b = review_config_fingerprint(
            "opus", "revsys", tool_set_hash="tools",
            reference_hash=reference_lists_hash(self._B))
        assert fp_a != fp_b

    def test_an_alias_edit_moves_all_three_stage_fingerprints(self):
        # Stated as the property that matters: no stage is left behind. A run
        # in which two of the three moved and one did not would report the
        # reviewer as unchanged while it wrote under a different vocabulary.
        moved = [
            config_fingerprint("opus", "prompt", "template",
                               reference_hash=reference_lists_hash(a))
            != config_fingerprint("opus", "prompt", "template",
                                  reference_hash=reference_lists_hash(b))
            for a, b in [(self._A, self._B)]
        ]
        moved.append(
            checker_config_fingerprint(
                "sonnet", "sys", "user",
                reference_hash=reference_lists_hash(self._A))
            != checker_config_fingerprint(
                "sonnet", "sys", "user",
                reference_hash=reference_lists_hash(self._B)))
        moved.append(
            review_config_fingerprint(
                "opus", "revsys",
                reference_hash=reference_lists_hash(self._A))
            != review_config_fingerprint(
                "opus", "revsys",
                reference_hash=reference_lists_hash(self._B)))
        assert moved == [True, True, True]


class TestCheckerConfigFingerprint:
    def test_prefix(self):
        fp = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
        )
        assert fp.startswith("checker_fp:")

    def test_stable(self):
        a = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
            structure_hash="d", field_catalogue_hash_str="cat",
        )
        b = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
            structure_hash="d", field_catalogue_hash_str="cat",
        )
        assert a == b

    def test_changes_with_model(self):
        a = checker_config_fingerprint("claude-sonnet-4-6", "sys", "user")
        b = checker_config_fingerprint("claude-opus-4-7", "sys", "user")
        assert a != b

    def test_changes_with_system_prompt(self):
        a = checker_config_fingerprint("claude-sonnet-4-6", "sys A", "user")
        b = checker_config_fingerprint("claude-sonnet-4-6", "sys B", "user")
        assert a != b

    def test_changes_with_checker_context_fields(self):
        # The checker context fields feed every per-record checker call's
        # context label, so editing them must move checker_fp.
        a = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
            checker_context_fields=["gauge", "outcome_variable"])
        b = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
            checker_context_fields=["gauge", "outcome_category"])
        assert a != b

    def test_checker_context_fields_order_moves_fingerprint(self):
        # The list is an ORDERED label, so reordering it is a genuine edit
        # (the checker's record label changes) and must move the fingerprint.
        a = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
            checker_context_fields=["gauge", "outcome_variable"])
        b = checker_config_fingerprint(
            "claude-sonnet-4-6", "sys", "user",
            checker_context_fields=["outcome_variable", "gauge"])
        assert a != b


class TestStructureHash:
    """`structure_hash` carries ONLY meltiro's own structure toggles: the
    per-field check budget, the reviewer on/off switch, whether the reviewer's
    own writes are checked, and the image-capability flag. The RESOLVED
    decoding dict belongs to direktoro's provider-call identity block (tested
    below) and is deliberately absent here: carrying it in both places would
    double-count it and duplicate the wire dialect meltiro must not own."""

    def test_string_encodes_the_per_field_check_budget(self):
        s = structure_hash(3)
        assert "checks3" in s

    def test_carries_no_decoding_dict(self):
        # The wire decoding dict is direktoro's: structure_hash takes only
        # structure toggles and never sees a resolved dict, so no decoding key
        # can leak into it.
        s = structure_hash(3)
        assert "max_tokens" not in s
        assert "temperature" not in s
        assert "dec" not in s

    def test_tool_call_cap_and_bonus_are_not_hashed(self):
        # The tool-call cap and its post-mark_complete cleanup bonus are an
        # operational budget, not config identity: they ride in no fingerprint,
        # so the structure string carries no calls/capbonus term.
        s = structure_hash(3)
        assert "calls" not in s
        assert "capbonus" not in s

    def test_noreview_suffix_only_when_reviewer_off(self):
        on = structure_hash(1, final_review=True)
        off = structure_hash(1, final_review=False)
        assert "_noreview" not in on
        assert off.endswith("_noreview")

    def test_noimages_suffix_only_when_text_only(self):
        on = structure_hash(1, supports_images=True)
        off = structure_hash(1, supports_images=False)
        assert "_noimages" not in on
        assert off.endswith("_noimages")

    def test_the_check_budget_moves_the_structure(self):
        assert structure_hash(1) != structure_hash(2)

    def test_checkreview_suffix_only_when_reviewer_edits_are_checked(self):
        off = structure_hash(1, check_reviewer_edits=False)
        on = structure_hash(1, check_reviewer_edits=True)
        assert "_checkreview" not in off
        assert on.endswith("_checkreview")

    def test_every_structure_combination_is_distinct(self):
        # Three independent booleans plus the budget: no two combinations may
        # collapse, or two runs with different pipelines would share a stage
        # fingerprint.
        seen = {
            structure_hash(checks, final_review=review,
                           check_reviewer_edits=check_review,
                           supports_images=images)
            for checks in (0, 1, 2)
            for review in (True, False)
            for check_review in (True, False)
            for images in (True, False)
        }
        assert len(seen) == 3 * 2 * 2 * 2


class TestResolvedDecodingFoldedIntoCallIdentity:
    """Both directions through the call-identity block: it folds in exactly
    what the adapter sends, so a temperature a model rejects never moves
    config_fp, and a registry quirk that changes the sent params always does.
    direktoro keys the params under the wire's own parameter name inside the
    block, so meltiro never writes a wire key itself."""

    def _identity(self, model, decoding):
        return canonical_json(call_identity_fields(
            model, decoding_params=decoding))

    def _config_fp(self, model, decoding):
        return config_fingerprint(self._identity(model, decoding), "p", "t")

    def test_temperature_ignored_for_a_no_temperature_model(self):
        # gpt-5.6-sol rejects temperature: two config temperatures resolve to
        # the same sent dict, so neither the block nor config_fp moves. A
        # spurious split here would fork the identity of two runs the provider
        # cannot tell apart.
        a = resolved_decoding_params("gpt-5.6-sol", sampling={"temperature": 0.0},
                                     max_tokens=100)
        b = resolved_decoding_params("gpt-5.6-sol", sampling={"temperature": 0.9},
                                     max_tokens=100)
        assert a == b
        assert "temperature" not in a
        assert self._config_fp("gpt-5.6-sol", a) == \
            self._config_fp("gpt-5.6-sol", b)

    def test_temperature_moves_fp_for_an_accepting_model(self):
        # z-ai/glm-5v-turbo (routed, Chat Completions) accepts temperature:
        # changing it changes the sent dict, the identity block, and config_fp.
        a = resolved_decoding_params("z-ai/glm-5v-turbo", sampling={"temperature": 0.0},
                                     max_tokens=100)
        b = resolved_decoding_params("z-ai/glm-5v-turbo", sampling={"temperature": 0.9},
                                     max_tokens=100)
        assert a != b
        assert self._config_fp("z-ai/glm-5v-turbo", a) != \
            self._config_fp("z-ai/glm-5v-turbo", b)

    def test_reasoning_effort_is_folded_in(self):
        # gpt-5.6-sol sends reasoning effort, so it appears in the sent dict and
        # in the identity block: bumping it moves config_fp.
        dec = resolved_decoding_params("gpt-5.6-sol", sampling={"temperature": 0.0},
                                       max_tokens=100)
        assert dec["reasoning"] == {"effort": "medium"}
        bumped = dict(dec, reasoning={"effort": "high"})
        assert self._config_fp("gpt-5.6-sol", dec) != \
            self._config_fp("gpt-5.6-sol", bumped)

    def test_output_cap_key_follows_the_wire(self):
        # Responses uses max_output_tokens; Chat Completions (routed GLM) uses
        # max_tokens. The cap rides under whatever key the wire uses, and
        # direktoro applies that keying inside the block.
        assert "max_output_tokens" in resolved_decoding_params(
            "gpt-5.6-sol", sampling={"temperature": 0.0}, max_tokens=100)
        assert "max_tokens" in resolved_decoding_params(
            "z-ai/glm-5v-turbo", sampling={"temperature": 0.0}, max_tokens=100)


class TestRunFingerprint:
    """The whole-run fingerprint folds the three stage fingerprints AND the
    engine identity into one identity for the full run-producing
    configuration. It exists because config_fp alone identifies only the
    extractor, so two runs sharing an extractor config but assigning different
    checker/reviewer configs would otherwise collapse together downstream.
    The engine component is what stops two runs of an identical config under
    different meltiro versions sharing a run_fp: no stage fingerprint covers
    engine prose, so without it run_fp claimed more than it could support."""

    # Representative stage fingerprints (already self-prefixed, as they are in
    # run.json). Their exact content is irrelevant here; run_fp hashes them
    # verbatim.
    CFG = config_fingerprint("claude-opus-4-8", "ph", "th", tool_set_hash="t")
    CHK = checker_config_fingerprint("claude-sonnet-4-6", "sys", "usr")
    REV = review_config_fingerprint("claude-opus-4-8", "revsys")
    ENG = engine_fingerprint("1.2.3", "srchash", "4.5.6", "dsrchash")

    def test_prefix_and_format(self):
        fp = run_fingerprint(self.CFG, self.CHK, self.REV, self.ENG)
        assert fp.startswith("run_fp:")
        digest = fp.split(":", 1)[1]
        # Full, untruncated SHA-256 hex, matching the stage fingerprints.
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_preimage_is_pinned(self):
        # The preimage is the three stage fingerprints and the engine
        # fingerprint joined by "|" in the order extractor, checker, reviewer,
        # engine, each hashed verbatim. Pinned by
        # reconstructing it by hand: a change to the join, the order, or the
        # component form is a downstream-visible break (consumers build
        # `llm:<run_fp>` producer strings from this) and must fail here.
        preimage = f"{self.CFG}|{self.CHK}|{self.REV}|{self.ENG}"
        expected = "run_fp:" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()
        assert run_fingerprint(self.CFG, self.CHK, self.REV, self.ENG) == expected

    def test_stable(self):
        assert run_fingerprint(self.CFG, self.CHK, self.REV, self.ENG) == \
            run_fingerprint(self.CFG, self.CHK, self.REV, self.ENG)

    def test_moves_with_each_stage(self):
        base = run_fingerprint(self.CFG, self.CHK, self.REV, self.ENG)
        other_cfg = config_fingerprint("claude-sonnet-4-6", "ph", "th",
                                       tool_set_hash="t")
        other_chk = checker_config_fingerprint("claude-opus-4-8", "sys", "usr")
        other_rev = review_config_fingerprint("claude-sonnet-4-6", "revsys")
        # Each stage moving on its own moves run_fp: this is the whole point,
        # since config_fp alone would hold two of these three fixed.
        assert run_fingerprint(other_cfg, self.CHK, self.REV, self.ENG) != base
        assert run_fingerprint(self.CFG, other_chk, self.REV, self.ENG) != base
        assert run_fingerprint(self.CFG, self.CHK, other_rev, self.ENG) != base

    def test_disabled_stage_uses_the_documented_sentinel(self):
        # A disabled stage is passed as None and folded in as the ABSENT_STAGE
        # sentinel, not Python's str(None). Pinned against the hand-built
        # preimage so the sentinel token and its placement are load-bearing.
        assert ABSENT_STAGE == "none"
        preimage = f"{self.CFG}|{self.CHK}|{ABSENT_STAGE}|{self.ENG}"
        expected = "run_fp:" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()
        assert run_fingerprint(self.CFG, self.CHK, None, self.ENG) == expected

    def test_all_four_ablation_shapes_are_distinct(self):
        # extractor+checker+reviewer, +checker, +reviewer, extractor-only: the
        # four pipeline structures must yield four distinct run_fps even from
        # the same extractor and stage configs. This is the ablation guarantee
        # (extractor-only, no-reviewer, no-checker all well-defined and
        # mutually distinct).
        both = run_fingerprint(self.CFG, self.CHK, self.REV, self.ENG)
        no_review = run_fingerprint(self.CFG, self.CHK, None, self.ENG)
        no_checker = run_fingerprint(self.CFG, None, self.REV, self.ENG)
        extractor_only = run_fingerprint(self.CFG, None, None, self.ENG)
        assert len({both, no_review, no_checker, extractor_only}) == 4

    def test_absent_checker_and_absent_reviewer_do_not_collide(self):
        # The sentinel is shared but position-tagged: "checker off, reviewer on"
        # and "checker on, reviewer off" are distinct even when the present
        # stage fingerprint happens to be identical, because the sentinel sits
        # in a fixed slot.
        same = self.CHK  # deliberately reuse one value in different slots
        no_checker = run_fingerprint(self.CFG, None, same, self.ENG)
        no_review = run_fingerprint(self.CFG, same, None, self.ENG)
        assert no_checker != no_review


class TestReviewConfigFingerprint:
    def test_prefix(self):
        fp = review_config_fingerprint("claude-opus-4-7", "review system")
        assert fp.startswith("review_fp:")

    def test_stable(self):
        a = review_config_fingerprint(
            "claude-opus-4-7", "sys", tool_set_hash="t",
            structure_hash="d")
        b = review_config_fingerprint(
            "claude-opus-4-7", "sys", tool_set_hash="t",
            structure_hash="d")
        assert a == b

    def test_changes_with_model(self):
        a = review_config_fingerprint("claude-opus-4-7", "sys")
        b = review_config_fingerprint("claude-sonnet-4-6", "sys")
        assert a != b

    def test_changes_with_system_prompt(self):
        a = review_config_fingerprint("claude-opus-4-7", "sys A")
        b = review_config_fingerprint("claude-opus-4-7", "sys B")
        assert a != b

    def test_changes_with_tool_set(self):
        a = review_config_fingerprint("claude-opus-4-7", "sys",
                                      tool_set_hash="t1")
        b = review_config_fingerprint("claude-opus-4-7", "sys",
                                      tool_set_hash="t2")
        assert a != b

    def test_changes_with_structure(self):
        a = review_config_fingerprint("claude-opus-4-7", "sys",
                                      structure_hash="s1")
        b = review_config_fingerprint("claude-opus-4-7", "sys",
                                      structure_hash="s2")
        assert a != b

    def test_changes_with_call_identity(self):
        # The decoding params (and the model, provider, base_url, route) all
        # ride in the call-identity block now, so a different block (here two
        # different first-positional values) moves review_fp.
        a = review_config_fingerprint("identity-a", "sys")
        b = review_config_fingerprint("identity-b", "sys")
        assert a != b

    def test_changes_with_reference_content(self):
        a = review_config_fingerprint("claude-opus-4-7", "sys",
                                      reference_hash="r1")
        b = review_config_fingerprint("claude-opus-4-7", "sys",
                                      reference_hash="r2")
        assert a != b
