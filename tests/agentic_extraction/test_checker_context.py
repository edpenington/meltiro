"""The checker sees each quote in its surrounding paper text.

A quote alone can be unjudgeable. The case these tests are built around is a
severity-table cell: `88 (7.0)` is a count of 88 and a percentage of 7.0 ONLY
because a column header says so, and a checker shown the cell on its own cannot
tell a count from a percentage.

A bundle renders each table once, as a cropped image, and its `text.md` carries
a sentinel line in place of the markdown, so no paper here offers a wide
markdown table to quote from. The wide-table case is therefore built rather
than borrowed: `_wide_table_paper` composes a synthetic paper these tests own.

The ORDINARY-PROSE half of the bound (`TestWindowIsBounded`) needs a
full-length paper with a paragraph line thousands of characters long, because
that is what makes the line snap overrun its budget. It reads
`tests/fixtures/bundle_tables/`, a committed fixture, so no test here depends
on which papers a distribution happens to carry.

The window is `checker_context_chars` characters of paper text on each side of
the matched span, snapped outward to whole lines, with one window per fragment
of an ellipsed quote and overlapping windows merged. A quote inside a markdown
table also drags in that table's header row, whatever the budget says, and its
caption line when the paper sets that caption on its own line above the table
(`quote_context._caption_line`; a caption laid out otherwise arrives only if it
falls inside the ordinary window). `checker_context_chars: 0` renders exactly
what a call carrying no paper text renders, and a quote that cannot be located
degrades with an explicit statement rather than silently showing less.
"""

from pathlib import Path

import pytest

from meltiro.checker_prompts import (
    _render_evidence_block,
    build_checker_user_message,
)
from meltiro.fingerprint import checker_config_fingerprint
from meltiro.quote_context import (
    QUOTE_CLOSE_MARKER,
    QUOTE_OPEN_MARKER,
    _LINE_SNAP_SLACK as LINE_SLACK,
    quote_context_windows,
    render_window,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# A committed fixture paper with paragraph lines running past two thousand
# characters, which is what the prose half of the bound needs. A fixture, so
# these tests depend on no paper outside `tests/fixtures/`.
PROSE_PAPER_TEXT = (REPO_ROOT / "tests" / "fixtures" / "bundle_tables"
                    / "text.md")

# The cell that motivated the whole feature, and the two lines above the table
# that make it readable.
TABLE_CELL = "88 (7.0)"
TABLE_CAPTION = "**Table 2. Severity Categories of Fatigue"
TABLE_HEADER = "| Severity category | Total, No. (%) |"


# ---------------------------------------------------------------------------
# The wide table these tests own
# ---------------------------------------------------------------------------
#
# The table rule is about a cell whose column header is too far above it to fit
# in the character budget, and only a WIDE table puts a header there: narrow
# rows keep the header inside the budget, where the ordinary line snap would
# have reached it anyway. No paper in this repo carries such a table as
# markdown, and none needs to, so these tests build the case they are about
# instead of depending on whichever paper happens to have one. The
# columns, the row labels and every number below are invented. Only the shape
# is borrowed: a caption, a header row naming counts and percentages, and rows
# wide enough that a cell in the body sits thousands of characters below both.

_TABLE_COLUMNS = (
    "Brackets, No. (%)", "Couplings, No. (%)",
    "Fasteners, No. (%)", "Housings, No. (%)",
    "Cast units, No. (%)", "Forged units, No. (%)",
    "Frontline duty, No. (%)", "Second line duty, No. (%)",
    "Tertiary test rig, No. (%)", "Secondary test rig, No. (%)",
    "Community bench, No. (%)", "Field rig, No. (%)",
    "Region A, No. (%)", "Region B, No. (%)",
    "Region C, No. (%)", "Region D, No. (%)",
)

_TABLE_ROW_LABELS = (
    "Fatigue, none or minimal", "Fatigue, mild",
    "Fatigue, moderate", "Fatigue, moderately severe",
    "Fatigue, severe", "Corrosion, none or minimal",
    "Corrosion, mild", "Corrosion, moderate",
    "Corrosion, severe", "Wear, none or minimal",
    "Wear, mild", "Wear, moderate",
    "Wear, severe", "Deflection, none or minimal",
    "Deflection, mild", "Deflection, severe",
)

# Which row's total column holds `TABLE_CELL`. Far enough down the table that
# the header is past any budget these tests use.
_TABLE_CELL_ROW = 8


def _wide_table_paper():
    """A synthetic paper whose one markdown table is wide enough to put a
    column header outside the budget.

    `TABLE_CELL` appears exactly once, in the total column of one body row,
    and every other cell in the table is a distinct invented number, so a
    quote of any cell resolves to the cell it was taken from.
    """
    header = ("| Severity category | Total, No. (%) | "
              + " | ".join(_TABLE_COLUMNS) + " |")
    delimiter = "| " + " | ".join(["---"] * (2 + len(_TABLE_COLUMNS))) + " |"

    rows = []
    for i, label in enumerate(_TABLE_ROW_LABELS):
        total = (TABLE_CELL if i == _TABLE_CELL_ROW
                 else f"{900 + i} ({(900 + i) / 10:.1f})")
        cells = []
        for j in range(len(_TABLE_COLUMNS)):
            n = 100 + i * len(_TABLE_COLUMNS) + j
            cells.append(f"{n} ({n / 10:.1f})")
        rows.append(f"| {label} | {total} | " + " | ".join(cells) + " |")

    opening = (
        "A synthetic batch of 1257 units was put through every gauge. "
        "The numbers here are invented and describe no real study. "
        "Sampling ran across sixteen sites over eleven months.")
    closing = (
        "Each cell above is a count with its percentage in parentheses. "
        "Read alone a cell carries neither its subgroup nor its denominator. "
        "That is the whole reason a window reaches up to the header row.")

    return "\n".join([
        "# A synthetic paper for the table rule",
        "",
        opening,
        "",
        f"{TABLE_CAPTION}, Corrosion, Wear, and Deflection in a Batch**",
        "",
        header,
        delimiter,
        *rows,
        "",
        closing,
        "",
    ])


@pytest.fixture(scope="module")
def prose_paper():
    return PROSE_PAPER_TEXT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def table_paper():
    return _wide_table_paper()


def _spec():
    """A minimal field spec: these tests are about the evidence block, not the
    field briefing."""
    return {"variable": "prevalence", "description": "A prevalence estimate."}


def _quote_candidates(paper):
    """Plausible quotes from a paper: every table cell of a usable length, and
    every sentence-sized fragment of prose. Used to sweep a rule over real
    text rather than over one hand-picked example."""
    out = []
    for line in paper.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            out.extend(cell.strip() for cell in stripped.split("|")
                       if len(cell.strip()) >= 8)
        else:
            for sentence in stripped.split(". "):
                sentence = sentence.strip()
                if 20 <= len(sentence) <= 90:
                    out.append(sentence)
    return out


def _sweep(paper, budget):
    """Window every locatable candidate quote in `paper` and classify what
    came back.

    Returns `(checked, exempt, overruns)`: how many quotes produced windows,
    how many windows ran past `budget + LINE_SLACK` from inside a markdown
    table, which is the one deliberate exemption, and every window that ran
    past it from outside a table, which is a bug. Shared so that the sweep of
    the repo's papers and the sweep of the synthetic wide table are the same
    measurement made twice.
    """
    checked = 0
    exempt = 0
    overruns = []
    for quote in _quote_candidates(paper):
        windows = quote_context_windows(quote, paper, budget)
        if not windows:
            continue
        checked += 1
        for window in windows:
            for span_start, span_end in window.spans:
                lead = span_start - window.start
                trail = window.end - span_end
                if (lead <= budget + LINE_SLACK
                        and trail <= budget + LINE_SLACK):
                    continue
                line_start = paper.rfind("\n", 0, span_start) + 1
                if paper[line_start:].lstrip().startswith("|"):
                    exempt += 1
                else:
                    overruns.append((quote[:60], lead, trail))
    return checked, exempt, overruns


# ---------------------------------------------------------------------------
# The table rule
# ---------------------------------------------------------------------------

class TestTableHeaderAndCaption:
    def test_a_cell_gets_its_header_and_caption_past_the_budget(
            self, table_paper):
        # 200 characters each side reaches nowhere near the top of this table:
        # the cell sits several wide rows below the header. The table rule is
        # what puts the header and caption in the window anyway.
        windows = quote_context_windows(TABLE_CELL, table_paper, 200)
        assert len(windows) == 1
        rendered = render_window(table_paper, windows[0])
        assert TABLE_CAPTION in rendered
        assert TABLE_HEADER in rendered

        # The header really was out of budget: it starts more than 200
        # characters above the match, so a fixed window would have cut it.
        cell_at = table_paper.index(TABLE_CELL)
        header_at = table_paper.index(TABLE_HEADER)
        assert cell_at - header_at > 200

    def test_the_header_carries_the_percent_the_cell_does_not(
            self, table_paper):
        # The whole point: the cell says `88 (7.0)` and the column header says
        # `%`. Without the header the second number could be anything.
        windows = quote_context_windows(TABLE_CELL, table_paper, 200)
        rendered = render_window(table_paper, windows[0])
        assert "Total, No. (%)" in rendered

    def test_the_cell_is_marked_and_the_header_is_not(self, table_paper):
        windows = quote_context_windows(TABLE_CELL, table_paper, 200)
        rendered = render_window(table_paper, windows[0])
        marked = f"{QUOTE_OPEN_MARKER}{TABLE_CELL}{QUOTE_CLOSE_MARKER}"
        assert marked in rendered
        # Exactly one marked span, and the header is outside it.
        assert rendered.count(QUOTE_OPEN_MARKER) == 1
        assert rendered.count(QUOTE_CLOSE_MARKER) == 1
        quoted = rendered.split(QUOTE_OPEN_MARKER)[1].split(
            QUOTE_CLOSE_MARKER)[0]
        assert quoted == TABLE_CELL
        assert "Severity category" not in quoted

    def test_prose_outside_a_table_does_not_reach_for_a_header(self):
        # The table rule fires on a match inside a table, not on any match.
        paper = (
            "**Table 1. Something**\n"
            "\n"
            "| Group | Rate (%) |\n"
            "| --- | --- |\n"
            "| Brackets | 12 (4.1) |\n"
            "\n"
            "A later paragraph reports that recruitment ran for 24 months.\n"
        )
        windows = quote_context_windows("ran for 24 months", paper, 20)
        rendered = render_window(paper, windows[0])
        assert "Table 1" not in rendered
        assert "Rate (%)" not in rendered


# ---------------------------------------------------------------------------
# Ellipsed quotes: one window per fragment
# ---------------------------------------------------------------------------

def _two_passage_paper():
    """A paper whose two quotable sentences are far apart, with short lines
    between them so nothing bridges the gap."""
    filler = "\n".join(
        f"Filler line {i} of the methods section." for i in range(60))
    return (
        "The mean age was 41 years in the whole sample.\n"
        "\n"
        f"{filler}\n"
        "\n"
        "The response rate was 62% across all sites.\n"
    )


class TestEllipsedFragments:
    def test_two_fragments_get_two_windows_not_one(self):
        paper = _two_passage_paper()
        quote = "The mean age was 41 years ... The response rate was 62%"
        windows = quote_context_windows(quote, paper, 100)
        assert len(windows) == 2
        assert windows[0].end < windows[1].start

    def test_the_elided_middle_is_not_shown_as_if_it_were_quoted(self):
        paper = _two_passage_paper()
        quote = "The mean age was 41 years ... The response rate was 62%"
        windows = quote_context_windows(quote, paper, 100)
        bodies = [render_window(paper, w) for w in windows]
        # The bulk of the elided material stays out: one window spanning the
        # gap would carry every filler line.
        joined = "\n".join(bodies)
        assert "Filler line 30" not in joined
        # Each window marks exactly its own fragment.
        assert f"{QUOTE_OPEN_MARKER}The mean age was 41 years" in bodies[0]
        assert f"{QUOTE_OPEN_MARKER}The response rate was 62%" in bodies[1]

    def test_each_window_is_labelled_as_a_separate_passage(self):
        paper = _two_passage_paper()
        evidence = ("<q>The mean age was 41 years ... "
                    "The response rate was 62%</q>")
        text, _ = _render_evidence_block(
            evidence, set(), paper_text=paper, context_chars=100)
        assert "Passage 1 of 2:" in text
        assert "Passage 2 of 2:" in text

    def test_overlapping_windows_are_merged_into_one(self):
        # Two fragments a few characters apart: their windows overlap, so they
        # become a single window carrying both marked spans rather than the
        # same text rendered twice.
        paper = (
            "Intro line.\n"
            "\n"
            "The mean age was 41 years in the whole sample.\n"
            "\n"
            "Closing line.\n"
        )
        windows = quote_context_windows("The mean age ... 41 years", paper, 40)
        assert len(windows) == 1
        assert len(windows[0].spans) == 2
        rendered = render_window(paper, windows[0])
        assert rendered.count(QUOTE_OPEN_MARKER) == 2
        # The shared surrounding text appears once, not twice.
        assert rendered.count("in the whole sample") == 1


# ---------------------------------------------------------------------------
# Zero: exactly what no paper text renders
# ---------------------------------------------------------------------------

class TestZeroContextChars:
    @pytest.mark.parametrize("evidence", [
        "<q>The mean age was 41 years</q>",
        "<q>The mean age was 41 years</q><q>Intro line.</q>",
        ["The mean age was 41 years"],
        ["The mean age was 41 years", "Intro line."],
        "an untagged prose evidence string",
    ])
    def test_zero_renders_exactly_what_no_paper_text_renders(self, evidence):
        paper = (
            "Intro line.\n\nThe mean age was 41 years in the whole sample.\n")
        without, _ = _render_evidence_block(evidence, set())
        with_zero, _ = _render_evidence_block(
            evidence, set(), paper_text=paper, context_chars=0)
        assert with_zero == without
        assert QUOTE_OPEN_MARKER not in with_zero
        assert "in context" not in with_zero

    def test_zero_is_the_default_of_the_message_builder(
            self, synthetic_template, checker_user_template_path,
            table_paper):
        # The renderer asks for no context unless a caller supplies both the
        # paper text and a width, so a call site supplying neither gets the
        # bare rendering.
        blocks = build_checker_user_message(
            field_path="study.primary_aim",
            field_spec=_spec(),
            envelope={"value": "X", "evidence": f"<q>{TABLE_CELL}</q>"},
            identity_context="ctx",
            image_labels=set(),
            user_prompt_path=checker_user_template_path,
            paper_text=table_paper,
        )
        assert QUOTE_OPEN_MARKER not in blocks[0]["text"]

    def test_quote_context_windows_returns_nothing_at_zero(self, table_paper):
        # The quote is locatable in this paper, so the None is the width
        # talking and not a failed match.
        assert quote_context_windows(TABLE_CELL, table_paper, 200) is not None
        assert quote_context_windows(TABLE_CELL, table_paper, 0) is None


# ---------------------------------------------------------------------------
# Paper edges
# ---------------------------------------------------------------------------

class TestPaperEdges:
    def test_a_quote_at_the_very_start_does_not_overrun(self):
        paper = "First words of the paper.\n\nAnd then some more text here.\n"
        windows = quote_context_windows("First words", paper, 1000)
        assert windows[0].start == 0
        assert windows[0].end <= len(paper)
        assert render_window(paper, windows[0]).startswith(QUOTE_OPEN_MARKER)

    def test_a_quote_at_the_very_end_does_not_overrun(self):
        paper = "Some earlier text here.\n\nThe last words of the paper."
        windows = quote_context_windows("last words of the paper", paper, 1000)
        assert windows[0].end == len(paper)
        assert windows[0].start >= 0

    def test_a_budget_wider_than_the_paper_yields_the_whole_paper(self):
        paper = "A very short paper indeed.\n"
        windows = quote_context_windows("short paper", paper, 100000)
        assert (windows[0].start, windows[0].end) == (0, len(paper))


# ---------------------------------------------------------------------------
# The window is bounded, except where the table rule says otherwise
# ---------------------------------------------------------------------------

class TestWindowIsBounded:
    """Line snapping is bounded by `_LINE_SNAP_SLACK`; the table rule is not.

    Both halves are pinned against each other on purpose. A later tidy-up that
    bounds the table reach fails the table tests, and one that unbounds line
    snapping fails the prose tests. Neither half can be quietly given up to
    satisfy the other.
    """

    # Ordinary prose, in a paper whose longest line runs to more than two
    # thousand characters. Without the bound, snapping this quote's window out
    # to its own paragraph's boundaries costs far more than the slack allows.
    PROSE_QUOTE = "Flexural failure at follow-up was determined"

    def test_ordinary_prose_stays_within_budget_plus_slack(self, prose_paper):
        for context_chars in (200, 500, 1000):
            windows = quote_context_windows(
                self.PROSE_QUOTE, prose_paper, context_chars)
            assert len(windows) == 1
            window = windows[0]
            span_start, span_end = window.spans[0]
            # Each side is the budget plus at most the line-snap slack.
            assert span_start - window.start <= context_chars + LINE_SLACK
            assert window.end - span_end <= context_chars + LINE_SLACK

    def test_the_bound_actually_bites_on_a_long_line(self, prose_paper):
        # The guard is only meaningful if this paper would otherwise overrun.
        # It does: the quote sits in a paragraph line long enough that snapping
        # to its boundaries costs more than the slack.
        at = prose_paper.index(self.PROSE_QUOTE)
        line_start = prose_paper.rfind("\n", 0, at) + 1
        line_end = prose_paper.find("\n", at)
        assert line_end - line_start > 2 * LINE_SLACK

        window = quote_context_windows(self.PROSE_QUOTE, prose_paper, 1000)[0]
        # The forward edge is the one that would otherwise overrun: the rest
        # of that paragraph runs far past the budget. It stops off a line
        # boundary, which is what the bound buys, and still not mid-word.
        assert window.end < len(prose_paper)
        assert prose_paper[window.end] != "\n"
        assert not (prose_paper[window.end - 1].isalnum()
                    and prose_paper[window.end].isalnum())
        # The backward edge sits close to the paragraph's start, so it is
        # still within slack and snaps whole. Both branches, one window.
        assert prose_paper[window.start - 1] == "\n"

    def test_a_table_quote_reaches_its_header_past_budget_plus_slack(
            self, table_paper):
        # The exemption. 200 characters of budget plus 250 of slack is 450,
        # and the header sits further above the cell than that, so a bounded
        # table reach would lose it. It must not.
        context_chars = 200
        windows = quote_context_windows(TABLE_CELL, table_paper, context_chars)
        window = windows[0]
        span_start, _ = window.spans[0]
        assert span_start - window.start > context_chars + LINE_SLACK

        rendered = render_window(table_paper, window)
        assert TABLE_CAPTION in rendered
        assert TABLE_HEADER in rendered

    def test_a_short_line_still_snaps_to_the_whole_line(self):
        # The common case: where the line boundary is within slack, the
        # window is whole lines.
        paper = ("Intro line.\n"
                 "\n"
                 "The mean age was 41 years in the whole sample.\n"
                 "\n"
                 "Closing line.\n")
        window = quote_context_windows("41 years", paper, 30)[0]
        assert window.start == 0 or paper[window.start - 1] == "\n"
        assert window.end == len(paper) or paper[window.end] == "\n"

    def test_the_invariant_holds_across_every_paper_in_the_repo(self):
        """No window over budget plus slack, in any real paper text here.

        The two tests above pin one quote each. This one sweeps every
        locatable quote in every paper text in the repo, which is what would
        catch a regression on a paper shaped unlike the ones those tests use.
        It is cheap, about a tenth of a second.

        The companion guard, that the table exemption fires somewhere, does
        NOT belong here; it lives in
        `test_the_table_exemption_is_genuinely_exercised`, against a table
        these tests build for themselves. A bundle renders each table once as
        an image and carries a sentinel line in the text, so this corpus may
        legitimately hold no markdown table at all, and certainly none wide
        enough to push a header out of budget. A guard a legitimate corpus can
        switch off is not a guard, and it fails as an accusation against code
        that did nothing wrong.

        What this sweeps is the committed fixtures under `tests/fixtures/`.
        The floor below is a floor, not a measurement: it is set well under
        what those fixtures actually yield, so that it keeps catching a sweep
        that found nothing (a glob that stops matching, a corpus that empties)
        without failing every time a fixture is added or retired.
        """
        budget = 1000
        checked = 0
        overruns = []
        papers = sorted(REPO_ROOT.glob("tests/fixtures/**/text.md"))
        assert papers, "no paper fixtures found"

        for path in papers:
            paper = path.read_text(encoding="utf-8")
            seen, _, over = _sweep(paper, budget)
            checked += seen
            overruns.extend((path.name,) + entry for entry in over)

        assert checked > 400, f"sweep found too few quotes: {checked}"
        assert not overruns, f"prose windows over budget+slack: {overruns[:5]}"

    def test_the_table_exemption_is_genuinely_exercised(self, table_paper):
        """The exemption fires, so bounding the table reach cannot pass by
        doing nothing.

        This is the half of the sweep the repo's own papers cannot guarantee,
        for the reason given on the test above. Here it is guaranteed: the
        same measurement, over a table built wide enough that reaching its
        header costs more than budget plus slack. If the table reach is ever
        bounded, `exempt` goes to zero and this fails.
        """
        budget = 1000
        checked, exempt, overruns = _sweep(table_paper, budget)
        assert checked > 0, "the synthetic paper produced no locatable quotes"
        assert not overruns, f"prose windows over budget+slack: {overruns[:5]}"
        assert exempt > 0

    def test_no_window_edge_ever_lands_mid_word(self, prose_paper, table_paper):
        # Whichever branch the snap takes, the guarantee holds. Both papers
        # are here because only the second one takes the table branch.
        cases = (
            (prose_paper, self.PROSE_QUOTE),
            (prose_paper,
             "nothing here establishes reach beyond the Northmoor network"),
            (table_paper, TABLE_CELL),
            (table_paper, "Sampling ran across sixteen sites"),
        )
        edges = 0
        for paper, quote in cases:
            for context_chars in (50, 250, 1000):
                windows = quote_context_windows(quote, paper, context_chars)
                assert windows, f"quote no longer locatable: {quote}"
                for w in windows:
                    if w.start > 0:
                        edges += 1
                        assert not (paper[w.start - 1].isalnum()
                                    and paper[w.start].isalnum())
                    if w.end < len(paper):
                        edges += 1
                        assert not (paper[w.end - 1].isalnum()
                                    and paper[w.end].isalnum())
        # Both assertions sit behind an edge-is-interior test, which a window
        # spanning the whole paper would fail. Windows like that are the
        # regression this test exists to catch, so count the edges actually
        # inspected rather than let the guards silently skip them all.
        assert edges >= len(cases) * 3, f"only {edges} interior edges inspected"


# ---------------------------------------------------------------------------
# Honest degradation
# ---------------------------------------------------------------------------

class TestDegradation:
    def test_an_unlocatable_quote_yields_no_windows(self):
        paper = "Nothing here resembles the quote at all.\n"
        assert quote_context_windows("a fabricated quote", paper, 500) is None

    def test_the_message_says_so_rather_than_showing_nothing(self):
        paper = "Nothing here resembles the quote at all.\n"
        text, _ = _render_evidence_block(
            "<q>a fabricated quote</q>", set(),
            paper_text=paper, context_chars=500)
        assert "a fabricated quote" in text
        assert "no surrounding context could be resolved" in text.lower()
        assert QUOTE_OPEN_MARKER not in text

    def test_one_missing_quote_does_not_suppress_the_others(self):
        paper = "Intro line.\n\nThe mean age was 41 years in the sample.\n"
        text, _ = _render_evidence_block(
            "<q>The mean age was 41 years</q><q>a fabricated quote</q>",
            set(), paper_text=paper, context_chars=200)
        assert "Quote 1:" in text
        assert "Quote 2: no surrounding context could be resolved" in text
        assert QUOTE_OPEN_MARKER in text


# ---------------------------------------------------------------------------
# Context is not evidence, and images are untouched
# ---------------------------------------------------------------------------

class TestContextIsDistinguished:
    def test_the_lead_in_names_the_markers_and_disclaims_the_context(
            self, table_paper):
        text, _ = _render_evidence_block(
            f"<q>{TABLE_CELL}</q>", set(),
            paper_text=table_paper, context_chars=200)
        assert QUOTE_OPEN_MARKER in text
        assert QUOTE_CLOSE_MARKER in text
        lead_in = text.lower()
        assert "the quoted span is wrapped in" in lead_in
        assert "surrounding paper text" in lead_in
        # The context is the paper's words, never the extractor's: the message
        # says so, so the checker cannot read it as an argument for the value.
        assert "never the extractor's" in text
        assert "not itself the evidence offered" in text

    def test_the_quote_is_still_shown_as_the_evidence_first(self,
                                                            table_paper):
        text, _ = _render_evidence_block(
            f"<q>{TABLE_CELL}</q>", set(),
            paper_text=table_paper, context_chars=200)
        # The bare quote leads, before any context. The context is really
        # there: a quote that failed to resolve would say so under the same
        # heading and satisfy the ordering trivially.
        assert QUOTE_OPEN_MARKER in text
        assert text.startswith(f'"{TABLE_CELL}"')
        assert text.index(f'"{TABLE_CELL}"') < text.index("in context")

    def test_evidence_prose_is_still_withheld(self, table_paper):
        # Context opens the paper, not the extractor's argument: prose written
        # into the evidence string stays out.
        text, _ = _render_evidence_block(
            f"<q>{TABLE_CELL}</q> I reasoned my way to this from the table.",
            set(), paper_text=table_paper, context_chars=200)
        assert QUOTE_OPEN_MARKER in text
        assert "I reasoned my way" not in text

    def test_image_evidence_is_unaffected(self, tmp_path, prose_paper):
        png = tmp_path / "table_02.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        text, labels = _render_evidence_block(
            "<img>table_02</img>", {"table_02"},
            paper_text=prose_paper, context_chars=1000)
        assert labels == ["table_02"]
        assert "treat it AS the evidence" in text
        assert QUOTE_OPEN_MARKER not in text
        assert "in context" not in text


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestFingerprint:
    def _fp(self, chars):
        return checker_config_fingerprint(
            "call-identity", "system prompt", "user template",
            checker_context_chars=chars)

    def test_checker_fp_moves_with_the_width(self):
        assert self._fp(0) != self._fp(1000)
        assert self._fp(1000) != self._fp(1001)

    def test_the_same_width_is_the_same_fingerprint(self):
        assert self._fp(1000) == self._fp(1000)

    def test_the_checker_config_folds_its_width_in(self, synthetic_template,
                                                   checker_system_path,
                                                   checker_user_template_path):
        from meltiro.checker import CheckerConfig
        from meltiro.prompt_partials import stage_predicates
        cfg = CheckerConfig(max_tokens=1024, 
            checker_model="claude-sonnet-4-6",
            system_prompt_path=str(checker_system_path),
            user_prompt_template_path=str(checker_user_template_path),
        )
        # Structure held fixed: the width is the only thing that varies here,
        # and it is the checker's own knob rather than a pipeline toggle.
        predicates = stage_predicates(2, True)
        wide = cfg.fingerprint(synthetic_template, {"gauge_list": []},
                               predicates=predicates)
        cfg.context_chars = 0
        narrow = cfg.fingerprint(synthetic_template, {"gauge_list": []},
                                 predicates=predicates)
        assert wide != narrow


# ---------------------------------------------------------------------------
# The pipeline.yaml key
# ---------------------------------------------------------------------------

def _cli_args():
    from types import SimpleNamespace
    return SimpleNamespace(
        max_tool_calls=None, max_checks_per_field=None, final_review=None,
        extractor_model=None, review_model=None, checker_model=None,
        diagnostics="standard", dry_run=True)


def _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg):
    from meltiro import cli
    from meltiro.bundle import load_bundle
    from meltiro.config_bundle import load_config_bundle
    config = load_config_bundle(config_dir)
    bundle = load_bundle(str(bundle_minimal_dir))
    return cli._build_orchestrator(
        config, bundle, tmp_path / "runs", loop_cfg, _cli_args())


def _pipeline(config_dir):
    from meltiro.config_bundle import load_config_bundle
    return dict(load_config_bundle(config_dir).pipeline)


class TestPipelineKey:
    def test_the_key_is_on_the_allowlist(self):
        from meltiro.config_bundle import KNOWN_PIPELINE_KEYS
        assert "checker_context_chars" in KNOWN_PIPELINE_KEYS

    def test_every_config_fixture_sets_the_default(self):
        # Discovered rather than named, so adding a config fixture cannot
        # leave this checking a subset, or silently checking nothing.
        from meltiro.config_bundle import load_config_bundle
        fixtures = REPO_ROOT / "tests" / "fixtures"
        configs = sorted(p.parent for p in fixtures.glob("*/pipeline.yaml"))
        assert configs, "no config fixture found under tests/fixtures/"
        for config in configs:
            pipeline = load_config_bundle(config).pipeline
            assert pipeline["checker_context_chars"] == 1000, config.name

    def test_the_value_reaches_the_checker_config(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_context_chars"] = 250
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.context_chars == 250

    def test_zero_is_accepted_and_not_swallowed_as_absent(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_context_chars"] = 0
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.context_chars == 0

    def test_the_default_stands_when_the_key_is_absent(
            self, config_dir, bundle_minimal_dir, tmp_path):
        loop_cfg = _pipeline(config_dir)
        loop_cfg.pop("checker_context_chars", None)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert orch.checker_config.context_chars == 1000

    @pytest.mark.parametrize("bad", [-1, -1000, 1.5, "1000", True, None])
    def test_a_bad_value_fails_loudly_at_startup(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys, bad):
        # None is the absent case and must NOT fail; every other value here is
        # a config error caught before any spend. Kept in one parametrisation
        # so the absent case is pinned next to the rejected ones.
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_context_chars"] = bad
        if bad is None:
            orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
            assert orch.checker_config.context_chars == 1000
            return
        with pytest.raises(SystemExit) as excinfo:
            _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        assert excinfo.value.code == 1
        assert "checker_context_chars" in capsys.readouterr().err

    def test_it_is_recorded_in_the_dry_run_report(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["checker_context_chars"] = 400
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        report = orch.dry_run_report()
        assert report["fingerprints"]["checker_context_chars"] == 400

    def test_an_off_checker_records_no_width(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        loop_cfg = _pipeline(config_dir)
        loop_cfg["max_checks_per_field"] = 0
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path, loop_cfg)
        report = orch.dry_run_report()
        assert report["fingerprints"]["checker_context_chars"] is None


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------

class TestRunJson:
    """The width is config identity, so a finished session says what it was
    without anyone re-reading the config bundle."""

    def _session_meta(self, config_dir, bundle_dir, out_dir, *,
                      context_chars, max_checks_per_field=2):
        from meltiro.bundle import load_bundle
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator
        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(str(bundle_dir)),
            out_dir,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024, 
                checker_model=("claude-sonnet-4-6"
                               if max_checks_per_field else None),
                context_chars=context_chars, api_key="x"),
            review_model=None,
            max_checks_per_field=max_checks_per_field,
            final_review=False,
            extractor_max_tokens=4096,
            api_key="x",
        )
        orch.prepare_new_session()
        return dict(orch.session.meta)

    def test_the_width_is_recorded(self, config_dir, bundle_minimal_dir,
                                   tmp_path):
        meta = self._session_meta(
            config_dir, bundle_minimal_dir, tmp_path / "on", context_chars=750)
        assert meta["checker_context_chars"] == 750

    def test_an_off_checker_records_null(self, config_dir, bundle_minimal_dir,
                                         tmp_path):
        meta = self._session_meta(
            config_dir, bundle_minimal_dir, tmp_path / "off",
            context_chars=750, max_checks_per_field=0)
        assert meta["checker_context_chars"] is None
        assert meta["checker_fp"] is None

    def test_two_widths_give_two_checker_fps(self, config_dir,
                                             bundle_minimal_dir, tmp_path):
        wide = self._session_meta(
            config_dir, bundle_minimal_dir, tmp_path / "wide",
            context_chars=1000)
        narrow = self._session_meta(
            config_dir, bundle_minimal_dir, tmp_path / "narrow",
            context_chars=0)
        assert wide["checker_fp"] != narrow["checker_fp"]
        # And only the checker moves: the extractor was asked the same thing.
        assert wide["config_fp"] == narrow["config_fp"]
        assert wide["run_fp"] != narrow["run_fp"]
