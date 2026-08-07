"""Verbatim-quote checking: what counts as the paper actually saying it.

`quote_check` is what stands between an extracted value and a fabricated one.
Every value the model records carries evidence, and evidence is only evidence
if the span it quotes is in the paper. So this module pins where the line sits,
and the line has to sit in exactly one place: too strict and the model is
refused for a quote the paper does contain, which burns turns and teaches it to
stop citing; too loose and a fabricated quote passes, which is the failure the
whole check exists to prevent.

Normalisation is therefore narrow and enumerated. `normalise_quote_text` folds
the artefacts a PDF conversion introduces and the model cannot see: ligatures,
smart quotes, the dash family, and the soft, thin and no-break spaces NFKC
resolves. It does NOT transliterate letters, so Greek stays Greek and an accent
stays an accent, and it does not fold symbols that carry meaning, so the
multiplication sign is not an ASCII x. Each of those refusals is a character a
fabricated quote would otherwise be free to differ by.

Two of those refusals are about what a quote CLAIMS rather than what it
contains. An elision marker claims its fragments are one passage of the paper
read forwards, so they have to appear in that order and not overlap; located
independently, they would let one `<q>` assemble a sentence out of phrases the
paper never puts together. And a parenthetical is only skippable when it looks
like an attribution, because a parenthetical carrying a year is as likely to
carry a baseline, an adjustment or a subgroup, and a rule keyed on the year
alone would let a quote delete the qualifier that decides what its number
means.

Where a tolerance is worth keeping but weakens the claim, the match records
it instead of hiding it. A fragment that only matches once both sides are
lowercased is reported as `case_folded`, and a quote is reported at the
weakest tier any of its fragments needed, so nothing downstream can read a
match as stronger than the evidence for it.

Bracketed interpolation is the one sanctioned way to quote something the paper
does not literally contain. A table whose percent sign lives in the column
header rather than the cell is the case that forces it: the model wants to
write `35.6%` and the paper says `35.6`. The literal reading is ALWAYS tried
first, so a paper that genuinely contains square brackets is unaffected, and
the interpolated reading is recorded as such rather than passed off as literal.

Evidence tags are checked structurally, not by counting. `<q>` and `<img>`
must be flat and correctly paired; a scan that only balanced opens against
closes would accept `<q>a<q>b</q></q>`, which then fails downstream as a
verbatim mismatch and sends the model looking for a quoting error it did not
make. A malformed tag is reported as malformed.

A refused quote comes back with the paper's own nearest wording attached.
Refusal alone tells the model it was wrong; the suggestion tells it what the
paper says, and that is the difference between a corrected value and a
retry loop.
"""

import time

from meltiro.quote_check import (
    _check_evidence_tags,
    find_quote,
    locate_quote,
    normalise_quote_text,
    strip_interpolations,
    suggest_nearest_text,
    validate_evidence,
)


PAPER = """
Methods. We used the WDS-9 to assess fatigue severity. Regres-
sion analysis estimated the association between WDS-9 score and
unplanned removal.

--- PAGE BREAK ---

Results. The odds ratio was 1.34 (95% CI 1.10–1.62).
""".strip()


class TestNormalisation:
    def test_idempotent(self):
        # Running normalisation twice gives the same result as once.
        for raw in [PAPER, "ﬁnal results", "“smart” quotes", "regres-\nsion"]:
            once = normalise_quote_text(raw)
            twice = normalise_quote_text(once)
            assert once == twice

    def test_ligatures_decompose(self):
        # NFKC turns the ligature ﬁ into the two-letter sequence fi.
        assert "final" in normalise_quote_text("ﬁnal results")

    def test_smart_quotes_become_straight(self):
        result = normalise_quote_text("“hello” ‘world’")
        assert '"hello"' in result
        assert "'world'" in result

    def test_en_em_dash_become_hyphen(self):
        # 1.10–1.62 (en dash) becomes 1.10-1.62.
        assert "1.10-1.62" in normalise_quote_text("1.10–1.62")
        assert "a-b" in normalise_quote_text("a" + chr(0x2014) + "b")

    def test_hyphen_line_break_healed(self):
        # The hyphen + line break is removed entirely.
        assert "regression" in normalise_quote_text("regres-\nsion")

    def test_page_break_marker_removed(self):
        # The page-break marker is collapsed away (whitespace fills the gap).
        result = normalise_quote_text("foo\n--- PAGE BREAK ---\nbar")
        assert "foo bar" in result
        assert "page break" not in result

    def test_whitespace_collapses(self):
        assert normalise_quote_text("a   \t\n  b") == "a b"

    def test_lowercase(self):
        assert normalise_quote_text("WDS-9") == "wds-9"

    def test_soft_hyphen_stripped(self):
        # U+00AD soft hyphen is invisible; remove it before matching.
        assert normalise_quote_text("regres­sion") == "regression"

    def test_none_input(self):
        assert normalise_quote_text(None) == ""


class TestFindQuote:
    def test_exact_substring_after_normalisation(self):
        assert find_quote("WDS-9 to assess fatigue severity", PAPER)

    def test_quote_spanning_hyphenated_line_break(self):
        # Paper has "regres-\nsion analysis"; quote uses healed form.
        assert find_quote("regression analysis", PAPER)

    def test_quote_with_en_dash_finds_hyphen_in_paper(self):
        # Paper has 1.10–1.62 (en dash); quote uses hyphen.
        assert find_quote("1.10-1.62", PAPER)

    def test_quote_with_internal_smart_quotes(self):
        # If the paper contains smart-quote characters and the model copies
        # them as straight quotes (or vice versa), the match still works
        # because both sides get normalised to straight quotes.
        assert find_quote(
            'the term "microfracture"',
            'the term “microfracture” was used',
        )

    def test_quote_spans_page_break(self):
        assert find_quote("unplanned removal. Results", PAPER)

    def test_missing_quote_returns_false(self):
        assert not find_quote("this phrase is not in the paper", PAPER)

    def test_no_quote_text_is_not_locatable(self):
        # `find_quote` locates text; there is nothing to locate here. The
        # empty `<q></q>` BLOCK is a separate concern, rejected by
        # `validate_evidence` with its own `empty_quote` code rather than
        # reported as a verbatim miss (see TestValidateEvidence).
        assert not find_quote("", PAPER)
        assert not find_quote(None, PAPER)

    def test_ellipsis_joins_two_real_fragments(self):
        # Convention: "first fragment ... second fragment" with both
        # halves verbatim in the paper. Both fragments live in PAPER:
        # "WDS-9 to assess fatigue severity" and "regression analysis"
        # (the latter heals from the hyphenated line break).
        assert find_quote(
            "WDS-9 to assess fatigue severity ... regression analysis",
            PAPER,
        )

    def test_ellipsis_with_one_fragment_missing_fails(self):
        # If one of the pinned fragments isn't in the paper, the whole
        # quote fails; we don't accept partial elision matches.
        assert not find_quote(
            "WDS-9 to assess fatigue severity ... this phrase is not in the paper",
            PAPER,
        )

    def test_unicode_ellipsis_separator_accepted(self):
        # The Unicode horizontal-ellipsis character should be treated
        # the same as three dots.
        assert find_quote(
            "WDS-9 to assess fatigue severity … regression analysis",
            PAPER,
        )

    def test_multiple_ellipses_all_checked(self):
        # Three pinned fragments, all verbatim → match.
        assert find_quote(
            "WDS-9 ... fatigue severity ... regression analysis",
            PAPER,
        )

    def test_ellipsis_inside_word_not_split(self):
        # Ellipsis without surrounding whitespace shouldn't be treated as
        # a fragment separator; it's just a literal triple-dot inside a
        # token. (Vanishingly rare in real papers, but the regex requires
        # whitespace on both sides for safety.)
        assert not find_quote("foo...bar...baz", PAPER)

    # -- Editorial annotations the extractor adds to a quote ----------

    def test_quote_with_sic_annotation_accepted(self):
        # Paper has "WDS-9 to assess fatigue severity"; quote adds
        # an editorial [sic] which obviously isn't in the paper.
        assert find_quote(
            "WDS-9 [sic] to assess fatigue severity", PAPER,
        )

    def test_quote_with_emphasis_added_annotation_accepted(self):
        assert find_quote(
            "WDS-9 to assess fatigue [emphasis added] severity",
            PAPER,
        )

    def test_quote_with_bracketed_ellipsis_accepted(self):
        # Single-token [...] is a standard academic elision marker
        # distinct from our ` ... ` separator.
        assert find_quote(
            "WDS-9 [...] fatigue severity", PAPER,
        )

    def test_non_editorial_bracket_content_read_as_interpolation(self):
        # Bracket content outside the closed editorial list is an
        # interpolation: text the extractor inserted into the quote, marked
        # the way a printed quotation marks it. The literal reading misses,
        # the bracket-stripped reading lands. (This replaces an earlier rule
        # that refused any bracket it did not recognise. See
        # TestBracketedInterpolations for the full contract.)
        match = locate_quote(
            "WDS-9 [random stuff] to assess fatigue severity", PAPER)
        assert match
        assert match.reading == "interpolated"

    def test_interpolation_does_not_rescue_text_absent_from_the_paper(self):
        # Stripping brackets is not a licence to invent the rest: the text
        # left behind still has to be in the paper.
        assert not find_quote(
            "WDS-9 [random stuff] to assess corrosion severity", PAPER,
        )

    # -- Inline citations in the paper, omitted by the extractor ------

    def test_numeric_citation_in_paper_stripped(self):
        # Paper has a citation marker the LLM legitimately omitted.
        paper = "The use of routine CRT-HD scores [12] has been validated."
        assert find_quote(
            "The use of routine CRT-HD scores has been validated", paper,
        )

    def test_numeric_citation_range_in_paper_stripped(self):
        paper = "Multiple studies [7-12] have reported this."
        assert find_quote(
            "Multiple studies have reported this", paper,
        )

    def test_parenthetical_author_year_citation_stripped(self):
        paper = (
            "The WDS-9 has been validated (Smith et al., 2019) "
            "in multiple settings."
        )
        assert find_quote(
            "The WDS-9 has been validated in multiple settings", paper,
        )

    def test_parenthetical_without_year_not_stripped(self):
        # Only citation-shaped parentheticals are treated as citations.
        # A parenthetical that just clarifies content should not be
        # silently dropped.
        paper = "The WDS-9 (the screening tool) was used."
        assert not find_quote(
            "The WDS-9 was used", paper,
        )

    def test_a_year_alone_does_not_make_a_parenthetical_a_citation(self):
        # The qualifier a citation-stripping rule must not eat. Dropping
        # this parenthetical from the paper before comparison would let a
        # quote delete the baseline the figure is measured from and still
        # be certified verbatim; the same goes for an adjustment set, a
        # subgroup restriction, or any other parenthetical that happens to
        # carry a year.
        paper = ("Maintenance costs rose (from the 2019 baseline) by 12 per "
                 "cent in the widget fleet.")
        assert not find_quote("Maintenance costs rose by 12 per cent", paper)
        assert find_quote(
            "Maintenance costs rose (from the 2019 baseline) by 12 per cent",
            paper)

    def test_a_capitalised_word_before_a_year_is_not_an_attribution(self):
        # A date reads as an author-year citation to any pattern that asks
        # only for a capital before the year, and a date is exactly the
        # kind of qualifier this must not delete.
        quote = "Recruitment closed at 1,200 units"
        for dated in ("(March 2019)", "(Q4 2020)", "(Wave 2021)"):
            paper = f"Recruitment {dated} closed at 1,200 units."
            assert not find_quote(quote, paper), dated

    def test_the_citation_forms_a_paper_actually_writes_are_stripped(self):
        # The other side of the bargain: an attribution carries a signal a
        # date does not, and the model is not made to copy it back.
        for citation in ("(Smith, 2019)", "(Smith et al., 2019)",
                         "(Smith et al. 2019)", "(Smith and Jones 2019)",
                         "(Smith & Jones, 2019)", "(van der Berg 2020)",
                         "(Smith 2019; Brown 2020)",
                         "(Smith et al., 2019, p. 12)"):
            paper = f"The WDS-9 has been validated {citation} in the fleet."
            assert find_quote(
                "The WDS-9 has been validated in the fleet", paper), citation

    def test_an_unrecognised_citation_form_is_refused_not_guessed_at(self):
        # A form the pattern does not recognise costs a turn and comes back
        # with the paper's own wording. That is the direction the cost is
        # meant to fall in: nothing is accepted on the strength of a
        # parenthetical the pattern was unsure about.
        paper = "The WDS-9 has been validated (2019) in the fleet."
        assert not find_quote("The WDS-9 has been validated in the fleet",
                              paper)
        # The suggested span is the length of the quote, so it stops short
        # of the full sentence; what matters is that the parenthetical the
        # quote dropped is in it.
        assert suggest_nearest_text(
            "The WDS-9 has been validated in the fleet", paper) == (
            "The WDS-9 has been validated (2019) in the")

    # -- Trailing punctuation differences -----------------------------

    def test_trailing_period_on_quote_not_in_paper(self):
        # Extractor added a sentence-ending period that doesn't exist
        # in the paper's run-on form.
        paper = "fatigue severity from the WDS-9 was high"
        assert find_quote("fatigue severity.", paper)


class TestValidateEvidence:
    """The envelope shape is `{value, evidence: string|null}`; `evidence`
    is a single string with optional `<q>...</q>` quote blocks,
    `<img>label</img>` image references, and free-text prose outside the
    tags. See meltiro.quote_check.parse_evidence_string.
    """

    def test_passes_for_null_value(self):
        errors = validate_evidence(
            evidence=None, paper_text=PAPER, image_labels=set(),
            value=None, field_path="study.foo",
        )
        assert errors == []

    def test_null_value_with_prose_reasoning_passes(self):
        # Prose explaining WHY the field is null is allowed alongside a
        # null value; the validator only complains about fabricated
        # quote content, not the existence of reasoning text.
        errors = validate_evidence(
            evidence="Paper does not state X; leaving null.",
            paper_text=PAPER, image_labels=set(),
            value=None, field_path="study.foo",
        )
        assert errors == []

    def test_null_value_with_fabricated_quote_still_rejected(self):
        # Even when value is null, any <q> block must be verbatim.
        errors = validate_evidence(
            evidence="<q>fake quote not in paper</q>",
            paper_text=PAPER, image_labels=set(),
            value=None, field_path="study.foo",
        )
        assert any(e["code"] == "quote_not_in_text" for e in errors)

    def test_text_sourced_passes_with_verbatim_quote(self):
        errors = validate_evidence(
            evidence="<q>WDS-9 to assess fatigue severity</q>",
            paper_text=PAPER, image_labels=set(),
            value="WDS-9", field_path="study.gauge",
        )
        assert errors == []

    def test_text_sourced_fails_when_quote_missing(self):
        errors = validate_evidence(
            evidence="<q>this phrase is not in the paper</q>",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        codes = [e["code"] for e in errors]
        assert "quote_not_in_text" in codes
        # Error message names the offending block index.
        assert any("evidence[<q>0]" in e["path"] for e in errors
                   if e["code"] == "quote_not_in_text")

    def test_text_sourced_multi_quote_one_missing(self):
        errors = validate_evidence(
            evidence=(
                "<q>WDS-9 to assess fatigue severity</q>"
                "<q>definitely not in the paper</q>"
            ),
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="rel.relationship_1.gauge",
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "rel.relationship_1.gauge.evidence[<q>1]"

    def test_image_sourced_no_quote_check(self):
        # <img> reference satisfies the required-evidence gate; no
        # quote needed.
        errors = validate_evidence(
            evidence="<img>table_02</img>",
            paper_text=PAPER, image_labels={"table_02"},
            value="0.42", field_path="rel.relationship_1.effect_size",
        )
        assert errors == []

    def test_image_label_match_case_insensitive(self):
        errors = validate_evidence(
            evidence="<img>Table_02</img>",
            paper_text=PAPER, image_labels={"table_02"},
            value="0.42", field_path="rel.relationship_1.effect_size",
        )
        assert errors == []

    def test_required_with_prose_only_fails(self):
        # `evidence_required=True` (default); required fields need at
        # least one <q> or <img> when value is non-null. Pure prose is
        # rejected.
        errors = validate_evidence(
            evidence="I think this is X.",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
            evidence_required=True,
        )
        assert any(e["code"] == "evidence_required" for e in errors)

    def test_required_with_empty_evidence_fails(self):
        errors = validate_evidence(
            evidence=None,
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
            evidence_required=True,
        )
        assert any(e["code"] == "evidence_required" for e in errors)

    def test_optional_with_prose_only_passes(self):
        # `evidence_required=False`; pure prose is acceptable.
        errors = validate_evidence(
            evidence="My judgement about the sample.",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
            evidence_required=False,
        )
        assert errors == []

    def test_malformed_tags_rejected(self):
        # Mismatched <q> open/close counts trigger malformed_tags.
        errors = validate_evidence(
            evidence="<q>quote without close",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        assert any(e["code"] == "malformed_tags" for e in errors)

    def test_empty_quote_block_rejected_as_empty_not_as_a_miss(self):
        # `<q></q>` is correctly paired, so the tag scan passes it through.
        # An empty block is its own fault and gets its own code: reporting it
        # as `quote_not_in_text` would tell the model to go and find text it
        # never wrote, and would drag in a nearest-text suggestion for a
        # needle that does not exist.
        errors = validate_evidence(
            evidence="<q></q>",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        assert [e["code"] for e in errors] == ["empty_quote"]
        assert errors[0]["path"] == "study.foo.evidence[<q>0]"

    def test_whitespace_only_quote_block_rejected_as_empty(self):
        # Whitespace is not quote content, so it is the same fault.
        errors = validate_evidence(
            evidence="<q>   </q>",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        assert [e["code"] for e in errors] == ["empty_quote"]

    def test_an_empty_block_does_not_mask_a_fabricated_one_beside_it(self):
        # Each block is judged on its own, and the path indexes say which.
        errors = validate_evidence(
            evidence="<q></q> and <q>fake quote not in paper</q>",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        assert [(e["code"], e["path"]) for e in errors] == [
            ("empty_quote", "study.foo.evidence[<q>0]"),
            ("quote_not_in_text", "study.foo.evidence[<q>1]"),
        ]

    def test_unknown_image_label_rejected(self):
        errors = validate_evidence(
            evidence="<img>nonexistent_label</img>",
            paper_text=PAPER, image_labels={"table_01"},
            value="X", field_path="study.foo",
        )
        codes = [e["code"] for e in errors]
        assert "unknown_image_label" in codes

    def test_empty_image_block_rejected_as_empty_not_as_unknown(self):
        # `<img></img>` names no label at all, which is a different fault
        # from naming one that does not exist: the remedy is to write a
        # label, not to correct one. Reporting it as `unknown_image_label`
        # would say the empty string is not in the available set, which is
        # true and useless.
        errors = validate_evidence(
            evidence="<img></img>",
            paper_text=PAPER, image_labels={"table_01"},
            value="X", field_path="study.foo",
        )
        assert [e["code"] for e in errors] == ["empty_image_label"]
        assert errors[0]["path"] == "study.foo.evidence[<img>0]"

    def test_whitespace_only_image_block_rejected_as_empty(self):
        errors = validate_evidence(
            evidence="<img> </img>",
            paper_text=PAPER, image_labels={"table_01"},
            value="X", field_path="study.foo",
        )
        assert [e["code"] for e in errors] == ["empty_image_label"]

    def test_error_message_lists_available_image_labels(self):
        errors = validate_evidence(
            evidence="<q>missing quote</q>",
            paper_text=PAPER,
            image_labels={"table_01", "figure_02"},
            value="X", field_path="study.foo",
        )
        # The error message tells the model exactly which image labels exist.
        msg = errors[0]["message"]
        assert "table_01" in msg
        assert "figure_02" in msg

    def test_non_string_evidence_rejected(self):
        # The validator only accepts str | None.
        errors = validate_evidence(
            evidence=42, paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        assert any(e["code"] == "type_mismatch" for e in errors)


class TestEvidenceTagStructure:
    """`_check_evidence_tags` enforces flat, correctly-paired <q>/<img> tags
    by scanning the sequence, NOT by counting opens against closes. A bare
    count accepts balanced-but-nested input like `<q>a<q>b</q></q>`, which
    then leaks downstream as a confusing verbatim mismatch. The scan reports
    nested, interleaved, orphaned and unclosed tags as malformed_tags, which
    is the error a caller can act on.
    """

    def _codes(self, s):
        return [e["code"] for e in _check_evidence_tags(s, "study.foo")]

    def test_correctly_paired_multi_quote_ok(self):
        assert _check_evidence_tags("<q>a</q> and <q>b</q>", "study.foo") == []

    def test_correctly_paired_quote_then_image_ok(self):
        assert _check_evidence_tags(
            "<q>a</q> see <img>table_01</img>", "study.foo") == []

    def test_no_tags_ok(self):
        assert _check_evidence_tags("just interpretive prose", "study.foo") == []

    def test_none_ok(self):
        assert _check_evidence_tags(None, "study.foo") == []

    def test_nested_same_tag_rejected(self):
        # Balanced counts (2 opens, 2 closes) but nested: must be malformed.
        assert "malformed_tags" in self._codes("<q>a<q>b</q></q>")

    def test_nested_image_in_quote_rejected(self):
        assert "malformed_tags" in self._codes("<q>a<img>t</img>b</q>")

    def test_interleaved_tags_rejected(self):
        # Overlap: a second tag opens before the first is closed.
        assert "malformed_tags" in self._codes("<q>a<img>t</q>b</img>")

    def test_orphan_close_rejected(self):
        assert "malformed_tags" in self._codes("</q>a")

    def test_unclosed_open_rejected(self):
        assert "malformed_tags" in self._codes("<q>a")

    def test_mismatched_close_kind_rejected(self):
        # <q> closed by </img>.
        assert "malformed_tags" in self._codes("<q>a</img>")

    def test_nested_surfaces_via_validate_evidence(self):
        # End-to-end: nested tags surface as malformed_tags, not as a
        # confusing quote_not_in_text mismatch.
        errors = validate_evidence(
            evidence="<q>a<q>b</q></q>",
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        codes = [e["code"] for e in errors]
        assert "malformed_tags" in codes
        assert "quote_not_in_text" not in codes


# The table shape that forces the interpolation rule: the percent sign lives
# in the column header, so the cells read `35.6`, not `35.6%`.
TABLE_PAPER = """
Table 1. Characteristics of the sampled units.

| Characteristic | No. (%) |
| --- | --- |
| Brackets | 448 (35.6) |
| Couplings | 88 (7.0) |

Surface corrosion was recorded on 634 [50.4%] of units, with a
median [IQR] score of 4.0 [1.0-7.0].
""".strip()


class TestBracketedInterpolations:
    """Square brackets mark text the extractor inserted into a quote, and
    are stripped before verbatim checking.

    The rule cannot be unconditional, because real paper text is full of
    brackets. So a quote gets two readings, literal first, and either one
    matching is enough.
    """

    def test_strip_interpolations_removes_the_insertion(self):
        assert strip_interpolations("448 (35.6[%])") == "448 (35.6)"

    def test_strip_interpolations_takes_the_space_with_the_bracket(self):
        # An insertion between words leaves the words one space apart, and
        # an insertion tight against a number leaves the number as written.
        assert strip_interpolations(
            "the brackets [in the trial] received") == "the brackets received"
        assert strip_interpolations("35.6 [%])") == "35.6)"

    def test_no_brackets_offers_no_second_reading(self):
        assert strip_interpolations("448 (35.6)") is None

    def test_nested_brackets_offer_no_second_reading(self):
        assert strip_interpolations("scores [high [very]] overall") is None

    def test_unbalanced_brackets_offer_no_second_reading(self):
        assert strip_interpolations("scores [high overall") is None
        assert strip_interpolations("scores high] overall") is None

    def test_whole_quote_as_one_insertion_offers_no_second_reading(self):
        # A quote that is nothing but a bracketed insertion carries no paper
        # text at all: a mistake, not an interpolation.
        assert strip_interpolations("[the whole thing]") is None
        assert not find_quote("[WDS-9 to assess fatigue severity]", PAPER)

    def test_interpolated_percent_sign_matches(self):
        # The live failure, stated as the convention would have written it.
        match = locate_quote("448 (35.6[%])", TABLE_PAPER)
        assert match
        assert match.reading == "interpolated"
        assert TABLE_PAPER[match.start:match.end] == "448 (35.6)"

    def test_quote_containing_real_brackets_still_matches_literally(self):
        # The trap: this passage genuinely contains brackets. The literal
        # reading is tried first, so it matches as written and the
        # bracket-stripped reading never runs.
        match = locate_quote("634 [50.4%] of units", TABLE_PAPER)
        assert match
        assert match.reading == "literal"
        assert TABLE_PAPER[match.start:match.end] == "634 [50.4%] of units"

    def test_interquartile_range_brackets_still_match_literally(self):
        match = locate_quote("median [IQR] score of 4.0 [1.0-7.0]", TABLE_PAPER)
        assert match
        assert match.reading == "literal"

    def test_editorial_marker_still_accepted_under_either_reading(self):
        # The closed editorial list composes with the general rule rather
        # than being replaced by it.
        assert find_quote("WDS-9 [sic] to assess fatigue severity", PAPER)

    def test_reading_reported_per_fragment(self):
        # One elided quote, one fragment literal and one interpolated. The
        # quote-level reading reports that an interpolation was involved.
        match = locate_quote(
            "448 (35.6[%]) ... 634 [50.4%] of units", TABLE_PAPER)
        assert match
        assert match.reading == "interpolated"
        assert [f.reading for f in match.fragments] == [
            "interpolated", "literal"]

    def test_interpolation_reading_used_by_validate_evidence(self):
        errors = validate_evidence(
            evidence="<q>448 (35.6[%])</q>",
            paper_text=TABLE_PAPER, image_labels=set(),
            value="35.6%", field_path="study.bracket_pct",
        )
        assert errors == []


class TestMatchPosition:
    """`locate_quote` reports where the quote landed, as offsets into the
    paper text that was passed in. A later stage shows the checker each
    quote in context, and a context window needs that position.
    """

    def test_direct_match_span_is_the_paper_text(self):
        quote = "WDS-9 to assess fatigue severity"
        match = locate_quote(quote, PAPER)
        assert PAPER[match.start:match.end] == quote
        assert match.tier == "direct"

    def test_tolerant_match_span_covers_the_omitted_citation(self):
        # The quote omits the paper's `[12]`; the span is the paper's own
        # run of text, citation included.
        paper = "The use of routine CRT-HD scores [12] has been validated."
        match = locate_quote(
            "The use of routine CRT-HD scores has been validated", paper)
        assert match.tier == "tolerant"
        assert paper[match.start:match.end] == (
            "The use of routine CRT-HD scores [12] has been validated")

    def test_normalised_match_span_covers_the_healed_hyphenation(self):
        # PAPER breaks `Regression` across a line as `Regres-\nsion`. The
        # span is the paper's own characters, which normalise back to the
        # quote. The quote copies the paper's capital, so healing the
        # hyphenation is the only thing the match rests on and the tier
        # says exactly that.
        quote = "Regression analysis"
        match = locate_quote(quote, PAPER)
        assert match.tier == "normalised"
        assert "Regres-\nsion analysis" in PAPER[match.start:match.end]
        assert normalise_quote_text(
            PAPER[match.start:match.end]) == normalise_quote_text(quote)

    def test_each_fragment_carries_its_own_span(self):
        match = locate_quote(
            "WDS-9 to assess fatigue severity ... unplanned removal",
            PAPER)
        assert len(match.fragments) == 2
        first, second = match.fragments
        assert PAPER[first.start:first.end] == (
            "WDS-9 to assess fatigue severity")
        assert PAPER[second.start:second.end] == "unplanned removal"
        # The quote-level span is the first fragment's.
        assert (match.start, match.end) == (first.start, first.end)

    def test_case_folding_is_reported_as_its_own_tier(self):
        # A quote that matches only once both sides are lowercased is a
        # weaker match than one the paper writes that way, and the tier is
        # where that difference is recorded. Accepting it silently under
        # `normalised` would leave nothing downstream able to tell the two
        # apart.
        exact = locate_quote("Methods. We used the WDS-9", PAPER)
        assert exact.tier == "direct"
        folded = locate_quote("METHODS. WE USED THE WDS-9", PAPER)
        assert folded
        assert folded.tier == "case_folded"
        assert PAPER[folded.start:folded.end] == "Methods. We used the WDS-9"

    def test_case_folding_and_hyphen_healing_are_separate_tiers(self):
        # `regression` differs from the paper's `Regres-\nsion` in two
        # ways. Copying the capital leaves only the hyphenation, which is
        # the converter's doing; changing it as well is the model's.
        assert locate_quote("Regression analysis", PAPER).tier == "normalised"
        assert locate_quote("regression analysis", PAPER).tier == "case_folded"

    def test_quote_tier_is_the_weakest_its_fragments_needed(self):
        # The first fragment is verbatim and the second is not. Reporting
        # the first fragment's tier would describe the quote as a stronger
        # match than any reader of the whole quote could verify.
        match = locate_quote(
            "Methods. We used the WDS-9 ... UNPLANNED REMOVAL", PAPER)
        assert match
        assert [f.tier for f in match.fragments] == ["direct", "case_folded"]
        assert match.tier == "case_folded"

    def test_miss_reports_no_position(self):
        match = locate_quote("this phrase is not in the paper", PAPER)
        assert not match
        assert match.matched is False
        assert match.start is None and match.end is None
        assert match.reading is None and match.fragments == ()

    def test_find_quote_is_the_boolean_face(self):
        assert find_quote("WDS-9", PAPER) is True
        assert find_quote("not here at all", PAPER) is False


def _realistic_paper(paragraphs=250):
    """A paper of realistic length (about 70k characters), built
    deterministically.

    Every paragraph is the same shape with different numbers, which is the
    hard case for the suggestion: hundreds of near-identical windows, one
    of which is the right answer.
    """
    parts = []
    for i in range(paragraphs):
        parts.append(
            f"Section {i}. Of the {1000 + i} widgets enrolled in batch "
            f"{i}, {448 + i} ({35.6 + i / 10:.1f}) reported signs of "
            f"fatigue and {88 + i} ({7.0 + i / 10:.1f}) reported severe "
            f"corrosion. Scores on the durability gauge were analysed with "
            f"a regression model adjusted for load, batch, and test site "
            f"[{i % 40 + 1}]."
        )
    return "\n\n".join(parts)


class TestElisionOrder:
    """` ... ` says the fragments either side of it are one passage of the
    paper read forwards, with words dropped out of the middle. So each
    fragment has to land at or after the end of the one before it.

    Without that, a single `<q>` can reverse two phrases, or stitch
    together phrases from unrelated sections in any order, and be certified
    verbatim: every fragment is in the paper, and nothing asks whether the
    sentence the quote builds out of them is.
    """

    def test_fragments_in_the_papers_order_match(self):
        assert find_quote(
            "WDS-9 to assess fatigue severity ... unplanned removal", PAPER)

    def test_the_same_fragments_reversed_do_not(self):
        # The pair above, written the other way round. Both fragments are
        # still in the paper; the passage the quote claims is not.
        assert not find_quote(
            "unplanned removal ... WDS-9 to assess fatigue severity", PAPER)

    def test_a_later_occurrence_satisfies_the_order(self):
        # `WDS-9` occurs twice. The second fragment is matched from the end
        # of the first, not from the top of the paper, so a repeated phrase
        # is quotable rather than a false ordering failure.
        match = locate_quote("WDS-9 ... WDS-9 score", PAPER)
        assert match
        first, second = match.fragments
        assert first.end <= second.start
        assert PAPER[second.start:second.end] == "WDS-9 score"

    def test_fragments_may_not_overlap(self):
        # `severity` appears only inside the first fragment. An elision
        # that folds back onto text it has already quoted is not a passage
        # read forwards.
        assert not find_quote("fatigue severity ... severity", PAPER)

    def test_order_holds_across_three_fragments(self):
        assert find_quote(
            "WDS-9 ... fatigue severity ... regression analysis", PAPER)
        assert not find_quote(
            "WDS-9 ... regression analysis ... fatigue severity", PAPER)

    def test_distance_between_fragments_is_not_bounded(self):
        # A deliberate decision, pinned so that adding a bound is a choice
        # rather than a drift. Any threshold would be a number with no
        # source: this module sees plain text with no section structure,
        # and the checker's context budget belongs to the caller. Distance
        # is disclosed instead of limited, by `quote_context` giving each
        # fragment its own window rather than one spanning the gap.
        paper = _realistic_paper()
        quote = "Section 0. Of the 1000 widgets ... Section 249. Of the 1249"
        assert find_quote(quote, paper)
        first, second = locate_quote(quote, paper).fragments
        assert second.start - first.end > 60_000

    def test_a_quote_with_no_elision_marker_is_unaffected(self):
        # One fragment has nothing to be out of order with. The ordering
        # rule must not cost anything to the ordinary quote.
        assert find_quote("WDS-9 to assess fatigue severity", PAPER)
        assert find_quote("unplanned removal", PAPER)

    def test_order_must_be_established_not_assumed(self):
        # Matching runs against transformed copies of the paper, and a copy
        # whose index map could not be tracked reports no position (see the
        # offset-tracking section of the module). A decomposed accent is
        # enough to lose the map, because NFKC composes it across two
        # characters. A fragment landing there can still carry a quote of
        # one fragment, where there is no order to check; it cannot carry a
        # quote of several, because certifying an order nothing established
        # is the failure the rule exists to prevent.
        decomposed = (
            # Escaped, because the difference between the two forms of
            # the accent is invisible in a source file and this test
            # turns on exactly that difference.
            "Ferre\u0301 measured 1,204 wid-\ngets in the fleet. Later the "
            "depot recorded unplanned re-\nmoval of 96 units."
        )
        # Both fragments need the hyphenation healed, so both are matched
        # in the untracked copy.
        assert locate_quote("1,204 widgets", decomposed).start is None
        assert find_quote("1,204 widgets", decomposed)
        assert find_quote("unplanned removal", decomposed)
        assert not find_quote("1,204 widgets ... unplanned removal",
                              decomposed)
        # The same paper with the accent composed keeps its map, and the
        # same quote is fine: what is refused is the unverifiable order,
        # not the elision.
        composed = decomposed.replace("e\u0301", "\u00e9")
        assert find_quote("1,204 widgets ... unplanned removal", composed)

    def test_the_refusal_says_the_order_is_what_is_wrong(self):
        # Refusing a reversal while showing the model the text it just
        # wrote would send it back to fix a quote that is not misquoted.
        # The remedy is a different one and the message has to name it.
        errors = validate_evidence(
            evidence=(
                "<q>unplanned removal ... WDS-9 to assess fatigue "
                "severity</q>"
            ),
            paper_text=PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        assert [e["code"] for e in errors] == ["quote_not_in_text"]
        message = errors[0]["message"]
        assert "not in the order the quote puts them" in message
        assert "separate <q>...</q> blocks" in message
        assert "closest text" not in message

    def test_a_genuinely_absent_fragment_still_gets_the_nearest_text(self):
        # The ordering message must not displace the suggestion for the
        # ordinary case, where one fragment is simply not in the paper.
        errors = validate_evidence(
            evidence="<q>Table 1 ... 448 (35.7)</q>",
            paper_text=TABLE_PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        message = errors[0]["message"]
        assert "The closest text in the paper reads: '448 (35.6)'." in message
        assert "not in the order the quote puts them" not in message


class TestNearestTextSuggestion:
    """A rejected quote comes back with the paper's own wording at the
    point it was aiming for, so a near miss can be corrected rather than
    resubmitted. Nothing close enough means no suggestion at all.
    """

    def test_suggests_the_table_cell_as_the_paper_writes_it(self):
        assert suggest_nearest_text("448 (35.6%)", TABLE_PAPER) == "448 (35.6)"
        assert suggest_nearest_text("88 (7.0%)", TABLE_PAPER) == "88 (7.0)"

    def test_suggestion_appears_in_the_validation_error(self):
        errors = validate_evidence(
            evidence="<q>448 (35.6%)</q>",
            paper_text=TABLE_PAPER, image_labels={"table_01"},
            value="35.6%", field_path="study.bracket_pct",
        )
        assert [e["code"] for e in errors] == ["quote_not_in_text"]
        message = errors[0]["message"]
        assert "The closest text in the paper reads: '448 (35.6)'." in message
        assert "Did you mean that?" in message
        # The rest of the guidance is still there.
        assert "table_01" in message

    def test_suggestion_is_the_papers_own_spelling_not_the_normalised_form(self):
        # Matching is case-folded and whitespace-collapsed; the suggestion is
        # not, because the model has to copy what the paper writes.
        paper = "The Trial Enrolled 1,227 Widgets In Total."
        assert suggest_nearest_text(
            "the trial enrolled 1227 widgets", paper) == (
            "The Trial Enrolled 1,227 Widgets")

    def test_suggestion_spans_a_line_break_as_one_line(self):
        paper = "reported signs of\nfatigue in most batches"
        assert suggest_nearest_text(
            "reported signs of fatigue in some batches", paper) == (
            "reported signs of fatigue in most batches")

    def test_nothing_close_gives_no_suggestion(self):
        assert suggest_nearest_text(
            "the flux capacitor requires 1.21 gigawatts", TABLE_PAPER) is None

    def test_no_suggestion_leaves_the_message_as_it_was(self):
        errors = validate_evidence(
            evidence="<q>the flux capacitor requires 1.21 gigawatts</q>",
            paper_text=TABLE_PAPER, image_labels=set(),
            value="X", field_path="study.foo",
        )
        message = errors[0]["message"]
        assert "closest text" not in message
        assert message.startswith(
            "Quote not found in paper text after normalisation:")
        assert "Copy a verbatim substring including punctuation" in message

    def test_very_short_quote_gives_no_suggestion(self):
        # Below the minimum length there is nothing distinctive to anchor
        # on, and any "nearest text" would be noise.
        assert suggest_nearest_text("7.0", TABLE_PAPER) is None

    def test_matching_quote_gives_no_suggestion(self):
        assert suggest_nearest_text("448 (35.6)", TABLE_PAPER) is None

    def test_suggests_for_the_first_failing_fragment_of_an_elided_quote(self):
        # The first fragment is fine; the second is the one that missed.
        assert suggest_nearest_text(
            "Table 1 ... 448 (35.7)", TABLE_PAPER) == "448 (35.6)"

    def test_deterministic_across_repeated_calls(self):
        # The message ends up in a tool result that a resumed run replays,
        # so the same inputs must always produce the same suggestion.
        paper = _realistic_paper()
        quote = "612 (52.0%) reported signs of fatigue"
        answers = {suggest_nearest_text(quote, paper) for _ in range(8)}
        assert len(answers) == 1
        assert answers.pop() == "612 (52.0) reported signs of fatigue"

    def test_finds_the_right_window_late_in_a_long_paper(self):
        # The candidate caps must not bias the answer towards the start of
        # the document. This target is in the last dozen paragraphs.
        paper = _realistic_paper()
        assert suggest_nearest_text(
            "685 (59.3%) reported signs", paper) == (
            "685 (59.3) reported signs")

    def test_cost_on_a_realistic_paper(self):
        # Suggestion runs on every failed quote, one tool call can carry
        # many fields, and paper texts are tens of thousands of characters.
        # The budget is deliberately loose (the measured cost on this text
        # is a couple of milliseconds a quote): it is here to catch a
        # change that makes the work quadratic in the paper length, not to
        # police small regressions on a busy machine.
        paper = _realistic_paper()
        assert len(paper) > 60_000
        quotes = [f"{448 + i} ({35.6 + i / 10:.1f}%) reported signs"
                  for i in range(25)]
        started = time.perf_counter()
        for quote in quotes:
            assert not find_quote(quote, paper)
            assert suggest_nearest_text(quote, paper) is not None
        elapsed = time.perf_counter() - started
        assert elapsed < 5.0, f"{len(quotes)} quotes took {elapsed:.2f}s"

    def test_cost_of_long_quotes_on_a_realistic_paper(self):
        # The expensive shape. A long quote means a long scoring window, and
        # the fuzzy comparison is quadratic in the window length, so this is
        # where an unbounded implementation would fall over. Each quote here
        # is a genuine 300-character span of the paper with one character
        # the paper does not contain stuck on the end: as close to matching
        # as a failing quote can be, and long enough to hit the length cap.
        paper = _realistic_paper()
        quotes = [paper[5000 + i:5300 + i] + "%" for i in range(20)]
        started = time.perf_counter()
        for quote in quotes:
            assert not find_quote(quote, paper)
            assert suggest_nearest_text(quote, paper) is not None
        elapsed = time.perf_counter() - started
        assert elapsed < 5.0, f"{len(quotes)} long quotes took {elapsed:.2f}s"
