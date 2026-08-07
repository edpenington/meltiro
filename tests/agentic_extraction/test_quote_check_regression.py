"""Quote checking against the text shapes real papers actually produce."""

from meltiro.quote_check import (
    find_quote,
    locate_quote,
    suggest_nearest_text,
    validate_evidence,
)


# The common table shape a model misquotes: the percent sign is in the column
# header, so every cell reads `35.6`, never `35.6%`.
TABLE_WITH_PERCENT_IN_HEADER = """
| Characteristic | No. (%) |
| --- | --- |
| Brackets | 448 (35.6) |
| Couplings | 88 (7.0) |
""".strip()


def test_a_percent_sign_from_the_column_header_is_refused():
    # What a model submits, over and over, against what the paper says.
    # Refusing it is correct. Refusing it while naming ONLY the rejected quote
    # leaves no way to see that one character is the whole problem, so the
    # nearest paper text is offered alongside the refusal.
    assert not find_quote("448 (35.6%)", TABLE_WITH_PERCENT_IN_HEADER)
    assert not find_quote("88 (7.0%)", TABLE_WITH_PERCENT_IN_HEADER)
    assert suggest_nearest_text("448 (35.6%)", TABLE_WITH_PERCENT_IN_HEADER) == "448 (35.6)"
    assert suggest_nearest_text("88 (7.0%)", TABLE_WITH_PERCENT_IN_HEADER) == "88 (7.0)"


def test_the_refusal_message_shows_the_papers_own_text():
    errors = validate_evidence(
        evidence="<q>88 (7.0%)</q>",
        paper_text=TABLE_WITH_PERCENT_IN_HEADER, image_labels={"table_01"},
        value="7.0%", field_path="rel.record_1.proportion",
    )
    assert [e["code"] for e in errors] == ["quote_not_in_text"]
    assert errors[0]["message"] == (
        "Quote not found in paper text after normalisation: '88 (7.0%)'. "
        "The closest text in the paper reads: '88 (7.0)'. Did you mean that? "
        "Copy a verbatim substring including punctuation, or, if the "
        "information is in a figure or table, use <img>label</img> with one "
        "of the available labels: table_01."
    )


def test_the_same_cell_stated_as_an_interpolation_is_accepted():
    # The other half of the answer: the interpolation convention, which lets
    # the model say "the paper writes 448 (35.6) and the unit is a percentage".
    match = locate_quote("448 (35.6[%])", TABLE_WITH_PERCENT_IN_HEADER)
    assert match
    assert match.reading == "interpolated"
    assert TABLE_WITH_PERCENT_IN_HEADER[match.start:match.end] == "448 (35.6)"


def test_soft_hyphen_at_line_break_heals():
    # PDF text extraction routinely emits `com\xad\nponents` (soft hyphen
    # followed by newline). The quote in the LLM response will read
    # "1128 components" with the word fully joined; the normaliser must
    # match across the soft-hyphen + newline boundary.
    paper = "Methods: ... rig batch of 1128 com­\nponents including 281"
    assert find_quote("1128 components including 281", paper)


def test_soft_hyphen_in_middle_of_word_in_paper(tmp_path):
    # Less common but seen: a soft hyphen mid-word with no line break.
    # Should still match a quote without the soft hyphen.
    paper = "the term micro­fracture is used"
    assert find_quote("the term microfracture is used", paper)


def test_lexically_hyphenated_word_split_across_pdf_line():
    # `self-assessment` is a real hyphenated compound. PDF rendering
    # broke it at the hyphen: `self- \nassessment`. The model copied
    # `self-assessment` (the canonical English form). Should match.
    paper = "AIC is a 5-item self- \nassessment checklist"
    assert find_quote("5-item self-assessment checklist", paper)


def test_pdf_broken_unhyphenated_word_still_matches():
    # Same gotcha in the opposite direction: PDF broke `components`
    # at a discretionary point as `com-\nponents`. The model copied
    # `components` (no hyphen). Should still match.
    paper = "study of 1128 com-\nponents including 281 controls"
    assert find_quote("1128 components including 281 controls", paper)


# Two sections of one paper, far enough apart that nothing in the text ties
# them together. The elision marker is the only thing claiming they are one
# passage, which is what makes it worth checking.
TWO_SECTIONS = """
Methods. The fleet comprised 1,204 load-bearing widgets across nine depots.
Each was inspected quarterly against the durability gauge.

Results. Of the widgets inspected, 96 met the threshold for unplanned
removal within the first year.
""".strip()


def test_an_elided_quote_may_not_run_backwards_through_the_paper():
    # Both phrases are verbatim, and in this order they are one passage of
    # the paper with the middle elided.
    assert find_quote("load-bearing widgets ... unplanned removal",
                      TWO_SECTIONS)
    # Reversed, they are a sentence the paper does not contain, assembled
    # out of parts that it does. An elision marker asserts the order; if
    # only membership were checked, this would certify as verbatim.
    assert not find_quote("unplanned removal ... load-bearing widgets",
                          TWO_SECTIONS)


def test_the_two_phrases_are_still_quotable_as_separate_blocks():
    # Refusing the reversal costs nothing that the model cannot say
    # another way: phrases from unrelated passages belong in a block each,
    # which claims no ordering and needs none.
    errors = validate_evidence(
        evidence="<q>unplanned removal</q> <q>load-bearing widgets</q>",
        paper_text=TWO_SECTIONS, image_labels=set(),
        value="X", field_path="study.foo",
    )
    assert errors == []


# The qualifier that decides what the number means. A citation-stripping
# rule keyed on the year alone deletes this from the paper before the
# comparison, and the quote below then reads as verbatim.
QUALIFIED_FIGURE = (
    "Maintenance costs rose (from the 2019 baseline) by 12 per cent in the "
    "widget fleet."
)


def test_a_quote_may_not_drop_a_qualifier_that_happens_to_carry_a_year():
    assert not find_quote("Maintenance costs rose by 12 per cent",
                          QUALIFIED_FIGURE)
    # And the refusal shows the model the phrase it dropped, so the
    # correction is one edit away. The span offered is the length of the
    # quote, so it reaches into the parenthetical rather than past it.
    assert suggest_nearest_text(
        "Maintenance costs rose by 12 per cent", QUALIFIED_FIGURE) == (
        "Maintenance costs rose (from the 2019")
    # Quoted with the qualifier, it is the paper's own sentence.
    assert find_quote(
        "Maintenance costs rose (from the 2019 baseline) by 12 per cent",
        QUALIFIED_FIGURE)


def test_a_real_citation_is_still_the_models_to_omit():
    # The tolerance the narrowing has to preserve: an author-year citation
    # carries an attribution, and a quote that steps over one is quoting
    # the sentence, not editing it.
    paper = ("Maintenance costs rose (Okonkwo et al., 2019) by 12 per cent "
             "in the widget fleet.")
    assert find_quote("Maintenance costs rose by 12 per cent", paper)


def test_a_quote_that_needs_case_folding_says_so():
    # Case is accepted, because a converter's capitalisation is not worth a
    # retry loop, but it is the weakest thing a match can rest on and the
    # record of the match names it rather than reporting the quote as text
    # the paper writes that way.
    match = locate_quote("MAINTENANCE COSTS ROSE", QUALIFIED_FIGURE)
    assert match
    assert match.tier == "case_folded"
    assert QUALIFIED_FIGURE[match.start:match.end] == "Maintenance costs rose"
