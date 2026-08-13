"""Stress the input-handling code against full-length, real-shaped paper text.

The rest of the suite runs on `tests/fixtures/bundle_minimal/`, whose text is
tiny and flat. Nothing there carries the unicode (ligatures, smart quotes, en
dashes in ranges, Greek letters, minus signs), the markdown tables, or the
length of a converted paper, so the bundle loader, the verbatim-quote checker
and the prompt builders would otherwise never meet the kinds of text they
handle in production.

These tests use two committed bundles. Both are original prose written for
this suite, so the repo redistributes nothing, but the TYPOGRAPHY is the
typography of a converted paper, which is the part the code has to survive:

  - `bundle_tables` (`syn-flexural`, a one-year depot cohort study): markdown
    tables, a figure, en/em dashes, curly quotes, non-breaking and thin
    spaces.
  - `bundle_unicode` (`syn-degradation`, a durability-score methods paper):
    the minus sign U+2212 in exponents, the multiplication sign, the Greek
    letter beta, thin spaces, and accented author names.

The fixtures are committed files: nothing here touches the network (the guard
in conftest.py enforces that suite-wide).

Where a text-handling behaviour is non-obvious, the test asserts and documents
the code's ACTUAL contract rather than an assumed one.
`TestMarkdownEmphasisIsLiteral` covers the one interaction that reads as a
surprise.
"""

import json
import shutil
from pathlib import Path

import pytest

from meltiro import cli
from meltiro.bundle import PaperBundle, load_bundle, validate_bundle
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import AgenticExtractionError
from meltiro.orchestrator import Orchestrator
from meltiro.prompt_builder import (
    build_initial_user_blocks,
    build_system_message,
    render_user_prompt_text,
)
from meltiro.quote_check import (
    find_quote,
    locate_quote,
    normalise_quote_text,
    suggest_nearest_text,
    validate_evidence,
)
from meltiro.template import load_template


# ---------------------------------------------------------------------------
# Bundle loading + validation on the real bundles
# ---------------------------------------------------------------------------

class TestRealBundleLoading:
    def test_tables_bundle_valid(self, bundle_tables_dir):
        # The validator inspects manifest.json, text.md and figures/ only. Any
        # other file sitting at the bundle root is ignored, NOT rejected.
        assert validate_bundle(bundle_tables_dir) == []

    def test_unicode_bundle_valid(self, bundle_unicode_dir):
        assert validate_bundle(bundle_unicode_dir) == []

    def test_tables_bundle_loads(self, bundle_tables_dir):
        b = load_bundle(bundle_tables_dir)
        assert isinstance(b, PaperBundle)
        assert b.study_id == "syn-flexural"
        assert b.doi == "10.0000/syn.0002"
        assert b.summary and "flexural" in b.summary
        # Exact figure-label discovery: the one committed PNG's stem is its
        # label, and that stem is the citation token the extractor would use.
        assert set(b.figures) == {"figure_01"}
        assert b.figures["figure_01"].name == "figure_01.png"
        # Real markdown tables survived the round-trip into text.md.
        assert "| --- |" in b.text
        assert "Odds ratio for Flexural Cracking" in b.text

    def test_unicode_bundle_loads(self, bundle_unicode_dir):
        b = load_bundle(bundle_unicode_dir)
        assert b.study_id == "syn-degradation"
        assert b.doi == "10.0000/syn.0003"
        # No figures/ directory: figure discovery yields the empty mapping.
        assert b.figures == {}
        assert "SPLINE-CD" in b.text

    def test_neither_bundle_redistributes_anything(self, bundle_tables_dir,
                                                    bundle_unicode_dir):
        # Both bundles are original prose written for this suite, so there is
        # nothing to attribute and no NOTICE.md to carry. A NOTICE here would
        # be a false provenance record, which is worse than none: assert its
        # absence rather than leaving the question open.
        for d in (bundle_tables_dir, bundle_unicode_dir):
            assert not (d / "NOTICE.md").exists(), d.name

    def test_validate_bundle_cli_reports_ok(self, bundle_tables_dir,
                                            bundle_unicode_dir, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["validate-bundle", str(bundle_tables_dir),
                      str(bundle_unicode_dir)])
        assert (excinfo.value.code or 0) == 0
        out = capsys.readouterr().out
        assert out.count("OK:") == 2
        assert "INVALID:" not in out


# ---------------------------------------------------------------------------
# Verbatim-quote checking against full-length paper text
# ---------------------------------------------------------------------------
# quote_check normalises BOTH the candidate quote and the paper text before
# matching (see quote_check.normalise_quote_text). The normaliser is
# deliberately narrow: it folds a specific set of PDF/typesetting artefacts
# (ligatures, smart quotes, the dash family, soft/thin/no-break spaces via
# NFKC) but does NOT transliterate letters (Greek stays Greek, accents stay
# accented) or fold every symbol (the multiplication sign is not an ASCII x).
# Each test below picks a span whose exact form is present in the fixture text
# and asserts what the code actually does with it.

@pytest.fixture
def flexural_text(bundle_tables_dir):
    return load_bundle(bundle_tables_dir).text


@pytest.fixture
def degradation_text(bundle_unicode_dir):
    return load_bundle(bundle_unicode_dir).text


class TestRealTextQuoteChecking:
    def test_en_dash_range_verbatim_and_hyphenated(self, flexural_text):
        # The paper reports a confidence interval with an en dash (U+2013).
        assert "confidence interval: 0.70–0.89" in flexural_text
        # The exact en-dash form matches (tier 1, direct substring).
        assert find_quote("confidence interval: 0.70–0.89", flexural_text)
        # A model that "cleans up" the en dash to an ASCII hyphen still
        # matches: the normaliser folds the whole dash family to "-".
        assert find_quote("confidence interval: 0.70-0.89", flexural_text)

    def test_em_dash_folds_to_hyphen(self, flexural_text):
        # A table header carries an em dash: "[min—max]".
        assert "[min—max]" in flexural_text
        assert find_quote("Mean (SD) [min—max]", flexural_text)
        assert find_quote("Mean (SD) [min-max]", flexural_text)

    def test_curly_and_straight_quotes_interchangeable(self, flexural_text):
        # The Whooley questions are quoted in the paper with curly quotes.
        assert "“Did the unit show a visible crack" in flexural_text
        assert find_quote("“Did the unit show a visible crack", flexural_text)
        # The extractor copying them as straight quotes still matches.
        assert find_quote('"Did the unit show a visible crack', flexural_text)

    def test_ligature_folds_to_plain_letters(self, flexural_text):
        # The paper has the plain word "significantly"; a quote using the fi
        # ligature (as PDF extraction often produces) still matches because
        # NFKC decomposes the ligature.
        assert "significantly" in flexural_text
        assert "signiﬁcantly" not in flexural_text  # ligature not in text
        assert find_quote("signiﬁcantly predicted", flexural_text)

    def test_no_break_space_matches_regular_space(self, flexural_text):
        # The paper joins "20" and "years" with a non-breaking space.
        assert "rated over 18 kN" in flexural_text
        # Verbatim (with the NBSP) matches.
        assert find_quote("rated over 18 kN", flexural_text)
        # A regular space in the quote matches the NBSP in the paper (NFKC
        # maps NBSP to a normal space, then whitespace collapses).
        assert find_quote("rated over 18 kN", flexural_text)
        # But dropping the space entirely is a real difference and fails.
        assert not find_quote("rated over 18kN", flexural_text)

    def test_thin_space_matches_regular_space(self, degradation_text):
        # The methods paper spaces "3.49 x 10" with thin spaces (U+2009)
        # around a multiplication sign.
        assert "3.49 × 10" in degradation_text
        # A quote written with ordinary spaces matches the thin-spaced text.
        assert find_quote("3.49 × 10", degradation_text)

    def test_minus_sign_folds_to_hyphen(self, degradation_text):
        # p-value exponents use the true minus sign U+2212, with a thin space
        # inside the superscript: "10^- 5^".
        minus_span = "10^− 5^"
        assert minus_span in degradation_text
        assert find_quote(minus_span, degradation_text)
        # Swapping the minus sign for an ASCII hyphen still matches: the
        # normaliser folds U+2212 to "-".
        assert find_quote(minus_span.replace("−", "-"), degradation_text)

    def test_multiplication_sign_is_not_folded_to_x(self, degradation_text):
        # The multiplication sign is preserved verbatim,
        assert find_quote("3.49 × 10", degradation_text)
        # but it is NOT treated as an ASCII "x": the normaliser folds dashes,
        # quotes, and ligatures, not arbitrary symbols. A model that rewrote
        # "x 10" would have its quote correctly rejected.
        assert not find_quote("3.49 x 10", degradation_text)

    def test_greek_letter_is_not_transliterated(self, degradation_text):
        # The score formula contains the Greek small letter beta.
        assert "β" in degradation_text
        assert find_quote("β", degradation_text)
        # NFKC leaves Greek as Greek: normalising the letter does not turn it
        # into the Latin word "beta". A quote that spelled it out would not be
        # accepted as the same character.
        assert normalise_quote_text("β") == "β"
        assert "beta" not in normalise_quote_text("β")

    def test_accented_letters_are_not_stripped(self, degradation_text):
        # An author name in the reference list carries an acute accent.
        assert "Ferré J" in degradation_text
        assert find_quote("Ferré J", degradation_text)
        # The accent is content, not a typesetting artefact: NFKC keeps the
        # precomposed letter, so the un-accented spelling does not match.
        assert not find_quote("Ferre J", degradation_text)


class TestRealTextInterpolationsAndSuggestions:
    """The percent-sign-in-the-header failure, and the bracket rule that
    answers it, against full-length paper text.

    Table 1 of the tables bundle has exactly the shape that provokes it: the
    header cell reads `N (%)` and every count cell reads `164 (26.3)`, so a
    model wanting to quote the percentage writes a percent sign the paper does
    not contain.
    """

    def test_percent_sign_is_in_the_header_not_the_cell(self, flexural_text):
        assert "| N (%) |" in flexural_text
        assert "| 139 (23.6) |" in flexural_text
        assert "139 (23.6%)" not in flexural_text

    def test_cell_quoted_with_the_header_percent_is_refused_and_suggested(
            self, flexural_text):
        assert not find_quote("139 (23.6%)", flexural_text)
        assert suggest_nearest_text("139 (23.6%)", flexural_text) == (
            "139 (23.6)")

    def test_cell_quoted_as_an_interpolation_is_accepted(self, flexural_text):
        match = locate_quote("139 (23.6[%])", flexural_text)
        assert match
        assert match.reading == "interpolated"
        assert flexural_text[match.start:match.end] == "139 (23.6)"

    def test_real_bracketed_range_still_quotable(self, flexural_text):
        # The trap: this table header genuinely contains square brackets.
        # The literal reading is tried first, so it is unaffected.
        match = locate_quote("Mean (SD) [min—max]", flexural_text)
        assert match
        assert match.reading == "literal"
        assert flexural_text[match.start:match.end] == "Mean (SD) [min—max]"

    def test_real_bracketed_statistic_still_quotable(self, flexural_text):
        # The trap in running prose rather than a header: this paper writes its
        # effect sizes with square brackets nested inside parentheses. Literal
        # is tried first, so both the odds ratio and its interval come back
        # verbatim and nothing needs the second reading. The spaces around `=`
        # in the quotes below are the paper's own THIN spaces (U+2009) and the
        # range dash is an en dash (U+2013): both are copied from the text
        # rather than typed, so a "tidy-up" to ASCII would break the match and
        # is meant to.
        quote = "(aOR = 0.42 [0.21–0.84])"
        match = locate_quote(quote, flexural_text)
        assert match
        assert match.reading == "literal"
        assert flexural_text[match.start:match.end] == quote
        assert find_quote("(OR = 0.29 [0.13–0.64])",
                          flexural_text)

    def test_suggestion_on_a_real_paper_names_the_papers_own_wording(
            self, flexural_text):
        # A quote that drifted from the paper's wording comes back with the
        # paper's own sentence, spelling and capitalisation included.
        assert suggest_nearest_text(
            "confidence interval: 0.70-0.90", flexural_text) == (
            "confidence interval: 0.70–0.89")

    def test_no_suggestion_for_text_that_is_not_in_the_paper(
            self, flexural_text):
        assert suggest_nearest_text(
            "randomised to twelve weeks of intravenous ketamine",
            flexural_text) is None


class TestMarkdownEmphasisIsLiteral:
    """Settled behaviour: the checker stays strict about markdown syntax.

    The bundles are markdown, and the conversion renders the paper's inline
    emphasis with markdown markers: an italicised statistical symbol like the
    sample-size N becomes ``*N*`` in text.md. The verbatim-quote checker runs
    on the raw text.md, and its normaliser folds unicode but does NOT strip
    markdown syntax. So a quote of an italicised token has to include the
    asterisks; the reader-facing form without them is rejected.

    Stripping inline emphasis before matching is the tempting alternative, and
    it is refused: every character the checker agrees to ignore is a character
    a fabricated quote can differ by, and this function is what decides
    whether a value is supported by the paper. The burden sits on the
    converter instead.

    Only text carrying inline emphasis exercises this, and the minimal bundle
    carries none. So this test pins a decision rather than reporting an
    observation. If it fails because the normaliser has been taught about
    markdown, that is the rule being reversed, and it needs to be reversed
    deliberately.
    """

    def test_inline_emphasis_markers_are_part_of_verbatim_text(self, degradation_text):
        # The paper writes the test-set size as an italic N: "(*N*  = 1,227)".
        assert "*N* = 1,308" in degradation_text
        # Quoting it WITH the markdown emphasis markers matches (the regular
        # space also folds to the thin space in the text).
        assert find_quote("*N* = 1,308", degradation_text)
        # Quoting the reader-facing form WITHOUT the asterisks does not match:
        # the normaliser does not strip markdown syntax.
        assert not find_quote("N = 1,308", degradation_text)


# ---------------------------------------------------------------------------
# validate_evidence tying the quote checker to a real bundle's image labels
# ---------------------------------------------------------------------------

class TestValidateEvidenceOnRealBundle:
    def test_verbatim_quote_from_real_text_passes(self, bundle_tables_dir):
        b = load_bundle(bundle_tables_dir)
        errors = validate_evidence(
            evidence=("<q>A high FDG score significantly predicted a lower "
                      "rate of flexural cracking</q>"),
            paper_text=b.text, image_labels=set(b.figures),
            value="protective", field_path="study.finding",
        )
        assert errors == []

    def test_fabricated_quote_is_rejected(self, bundle_tables_dir):
        b = load_bundle(bundle_tables_dir)
        errors = validate_evidence(
            evidence="<q>the drug halved mortality in the placebo arm</q>",
            paper_text=b.text, image_labels=set(b.figures),
            value="X", field_path="study.finding",
        )
        assert any(e["code"] == "quote_not_in_text" for e in errors)

    def test_real_figure_label_satisfies_image_evidence(self,
                                                        bundle_tables_dir):
        # The committed figure's stem ("figure_01") is the label the extractor
        # cites; an <img> reference to it resolves against the discovered set.
        b = load_bundle(bundle_tables_dir)
        errors = validate_evidence(
            evidence="<img>figure_01</img>",
            paper_text=b.text, image_labels=set(b.figures),
            value="0.710", field_path="study.auc",
        )
        assert errors == []

    def test_unknown_figure_label_is_rejected(self, bundle_tables_dir):
        b = load_bundle(bundle_tables_dir)
        errors = validate_evidence(
            evidence="<img>table_99</img>",
            paper_text=b.text, image_labels=set(b.figures),
            value="X", field_path="study.finding",
        )
        assert any(e["code"] == "unknown_image_label" for e in errors)


# ---------------------------------------------------------------------------
# Prompt building over a real bundle
# ---------------------------------------------------------------------------

class TestPromptBuildingOverRealBundle:
    def test_initial_user_blocks_carry_real_text_and_figure(self,
                                                            bundle_tables_dir):
        b = load_bundle(bundle_tables_dir)
        png_bytes = b.figures["figure_01"].read_bytes()
        blocks = build_initial_user_blocks(
            b.study_id, b.text, figures=[("figure_01", png_bytes)])
        # Header names the study; the full real paper text is carried verbatim.
        assert b.study_id in blocks[0]["text"]
        assert "--- PAPER TEXT ---" in blocks[1]["text"]
        assert "SPLINE-CD" not in blocks[1]["text"]  # right paper, not the other
        assert "flexural" in blocks[1]["text"]
        # Label precedes its image; the real PNG bytes are base64-encoded.
        assert blocks[2]["type"] == "text" and "figure_01" in blocks[2]["text"]
        assert blocks[3]["type"] == "image"
        assert blocks[3]["source"]["media_type"] == "image/png"
        # Last block carries the cache marker for the whole user prefix.
        assert blocks[-1]["cache_control"] == {"type": "ephemeral"}

    def test_render_user_prompt_text_size_and_label_order(self,
                                                          bundle_unicode_dir):
        b = load_bundle(bundle_unicode_dir)
        # Two labels, given in a deterministic order, appear in that order and
        # after the paper text. Real unicode in the text must not raise.
        text = render_user_prompt_text(
            b.study_id, b.text, image_labels=["figure_01", "table_02"])
        assert isinstance(text, str)
        assert len(text) > len(b.text)  # header + labels add to the raw text
        assert "−" in text  # the minus sign survived into the prompt
        i1 = text.index("figure_01")
        i2 = text.index("table_02")
        assert text.index("--- END PAPER TEXT ---") < i1 < i2

    def test_build_system_message_over_the_real_config(self, config_dir,
                                                       bundle_tables_dir):
        # The system prompt renders the config's reference list without
        # crashing, and carries nothing of the paper: the exhibits this bundle
        # supplies are labelled where they arrive, in the user message.
        config = load_config_bundle(config_dir)
        template = load_template(config.template_path)
        b = load_bundle(bundle_tables_dir)
        txt = build_system_message(
            system_prompt_path=config.extractor_system_path,
            reference_lists=config.reference_lists,
        )
        assert isinstance(txt, str) and txt
        assert "WDS-9" in txt
        for label in b.figures:
            assert label not in txt


# ---------------------------------------------------------------------------
# CLI dry-run over a real bundle with the shipped config (no API calls)
# ---------------------------------------------------------------------------

class TestCliDryRunOverRealBundle:
    def _no_client(self, monkeypatch):
        def _boom(self, role):
            raise AssertionError("dry-run must not resolve a provider adapter")
        monkeypatch.setattr(Orchestrator, "_adapter_for_role", _boom)

    def test_dry_run_tables_bundle_renders_instrument(self, tmp_path, config_dir,
                                                      bundle_tables_dir,
                                                      capsys, monkeypatch):
        self._no_client(monkeypatch)
        out_dir = tmp_path / "runs"
        with pytest.raises(SystemExit) as excinfo:
            cli.main([
                "extract",
                "--config", str(config_dir),
                "--paper", str(bundle_tables_dir),
                "--out", str(out_dir),
                "--dry-run",
            ])
        assert (excinfo.value.code or 0) == 0
        out = capsys.readouterr().out
        assert "=== SYSTEM MESSAGE ===" in out
        assert "TOOL CATALOGUE" in out
        # The rendered instrument files land under {out}/{study_id}/dry_run/.
        report = out_dir / "syn-flexural" / "dry_run"
        assert (report / "extractor_system.md").is_file()
        catalogue = json.loads(
            (report / "tool_catalogue.json").read_text(encoding="utf-8"))
        assert catalogue
        fps = json.loads(
            (report / "fingerprints.json").read_text(encoding="utf-8"))
        assert fps["config_fp"]
        # The real figure label reaches the rendered exhibit file.
        exhibits = (report / "attached_exhibits.txt").read_text(
            encoding="utf-8")
        assert "figure_01" in exhibits
        # A dry run creates NO session.
        assert not (out_dir / "syn-flexural" / "sessions").exists()

    def test_dry_run_unicode_bundle_no_figures(self, tmp_path, config_dir,
                                               bundle_unicode_dir, capsys,
                                               monkeypatch):
        self._no_client(monkeypatch)
        out_dir = tmp_path / "runs"
        with pytest.raises(SystemExit) as excinfo:
            cli.main([
                "extract",
                "--config", str(config_dir),
                "--paper", str(bundle_unicode_dir),
                "--out", str(out_dir),
                "--dry-run",
            ])
        assert (excinfo.value.code or 0) == 0
        out = capsys.readouterr().out
        assert "=== SYSTEM MESSAGE ===" in out
        report = out_dir / "syn-degradation" / "dry_run"
        assert (report / "extractor_system.md").exists()
        # A figure-less bundle still renders an (empty) exhibit file, and
        # no session is created.
        assert (report / "attached_exhibits.txt").exists()
        assert not (out_dir / "syn-degradation" / "sessions").exists()
