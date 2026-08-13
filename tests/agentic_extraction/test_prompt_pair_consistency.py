"""The extractor and checker prompts must not contradict each other.

A contradiction between them is not a wording defect. The extractor decides
whether to overrule a challenge, so a prompt telling it the checker is blind
to the table a number was read from teaches it to weigh table-sourced
challenges as if made blind, in the very case where the checker is given the
MOST context. A challenge that should have stuck gets dismissed, and the
number ships recorded as checked. In a calibration run whose whole output is
extraction accuracy, that is a measurement error.

No test that reads one prompt alone can catch it. This file reads them
together, in two halves that have to agree:

1. GROUND TRUTH, from the engine, not from either prompt. What the checker is
   actually handed is decided by `quote_context._window_bounds` (the table
   rule) and `checker_prompts._render_context_block` (whether any paper text
   is rendered at all). These tests probe that code directly, so a change to
   what the checker sees breaks them first.

2. THE PAIR. Every claim below is checked against BOTH system prompts of every
   config fixture AS THEY RENDER — the engine's prompt for that role plus the
   bundle's own text with its partials expanded, which is what a model is
   actually shown — never against the one that happened to carry it. A false
   claim that migrates from one prompt to the other still fails, a prompt that
   quietly drops a fact its partner asserts still fails, and it makes no
   difference whether the sentence was written in the bundle or composed by the
   engine.

The forbidden patterns are the specific false claims, each named beside the
fact and the code that refutes it. They are not a style check: rewording a
true sentence is free, and only re-asserting a refuted one costs.
"""

import re
from pathlib import Path

import pytest

from meltiro.checker_prompts import (
    _render_evidence_block,
    build_record_context,
)
from meltiro.prompt_partials import (
    CHECKER_SYSTEM,
    EXPAND_ALL_BRANCHES,
    EXTRACTOR_SYSTEM,
    compose_engine_prompt,
    join_blocks,
    substitute_include_placeholders,
)
from meltiro.quote_context import (
    QUOTE_CLOSE_MARKER,
    QUOTE_OPEN_MARKER,
    quote_context_windows,
    render_window,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Synthetic papers. Built here rather than borrowed: a shipped bundle's
# `text.md` carries a sentinel line where each table was, so the markdown-table
# case has to be constructed (see test_checker_context's module docstring).
# ---------------------------------------------------------------------------

CAPTION = "Table 2. WDS-9 score and unplanned removal, n (%)."
HEADER = "| Subgroup | Admitted, n (%) | OR (95% CI) |"
CELL = "118 (35.6)"
ROW = f"| Whole sample | {CELL} | 1.34 (1.02-1.76) |"
PROSE_BEFORE = "An earlier paragraph of ordinary prose about the batch."

# Enough filler rows that the header sits far outside any sane character
# budget: the table rule has to reach it anyway, or it buys nothing.
FILLER = "".join(f"| Filler {i} | {i} (1.0) | 1.00 |\n" for i in range(300))

CAPTION_ABOVE = (
    f"{PROSE_BEFORE}\n\n{CAPTION}\n\n{HEADER}\n| --- | --- | --- |\n"
    f"{FILLER}{ROW}\n\nA following paragraph.\n"
)
# The quoted cell sits at the TOP of this one, so the caption below it is well
# outside the forward budget. A caption a line or two under a short table does
# reach the checker, but by ordinary proximity, not by the table rule.
CAPTION_BELOW = (
    f"{PROSE_BEFORE}\n\n{HEADER}\n| --- | --- | --- |\n{ROW}\n{FILLER}"
    f"\n{CAPTION}\n"
)
CAPTION_IN_PARAGRAPH = (
    f"We report the results below in the next display.\n{CAPTION}\n\n"
    f"{HEADER}\n| --- | --- | --- |\n{ROW}\n"
)
CAPTION_FAR_ABOVE = (
    f"{CAPTION}\n\n\n\n\n{HEADER}\n| --- | --- | --- |\n{ROW}\n"
)
PLAIN_PROSE = (
    "The batch was drawn from two test sites. "
    "Fatigue was measured with the WDS-9 at commissioning. "
    "Unplanned removal was ascertained from service records over "
    "twelve months.\n"
)

# A budget far too small to reach the header on its own, so any test that
# still finds the header found it through the table rule and not by luck.
TIGHT = 40


def _window_text(paper, quote, chars=TIGHT):
    windows = quote_context_windows(quote, paper, chars)
    assert windows, f"no window resolved for {quote!r}"
    return "\n\n".join(render_window(paper, w) for w in windows)


# ---------------------------------------------------------------------------
# 1. Ground truth: what the engine actually hands the checker
# ---------------------------------------------------------------------------

class TestWhatTheCheckerIsActuallyShown:
    """Probes over the code that decides it, so the facts below cannot rot."""

    def test_a_quote_arrives_inside_the_papers_own_surrounding_text(self):
        # checker_prompts._render_context_block. This is what refutes any
        # claim that the checker is shown "nothing else: not the paper".
        rendered, _ = _render_evidence_block(
            f"<q>{CELL}</q>", set(), paper_text=CAPTION_ABOVE,
            context_chars=1000)
        assert QUOTE_OPEN_MARKER in rendered
        assert QUOTE_CLOSE_MARKER in rendered
        assert HEADER in rendered

    def test_a_table_quote_reaches_the_header_row_past_the_budget(self):
        # quote_context._window_bounds: the table probe lowers the window start
        # with `ws = min(ws, top)` and nothing clamps it back. This is what
        # refutes any claim that the checker "cannot see the table a number
        # was read from".
        text = _window_text(CAPTION_ABOVE, CELL)
        assert HEADER in text
        assert len(text) > TIGHT * 100, (
            "the window did not overrun the budget, so the table rule did not "
            "fire and this test is no longer testing it")

    def test_the_header_row_comes_even_with_context_turned_down_to_one(self):
        assert HEADER in _window_text(CAPTION_ABOVE, CELL, chars=1)

    def test_a_bracketed_insertion_still_resolves_to_the_papers_own_cell(self):
        # The prompts teach `<q>118 (35.6[%])</q>` for a cell whose percent
        # sign is in the column header. That reading must still window, or the
        # advice would silently cost the checker its context.
        text = _window_text(CAPTION_ABOVE, "118 (35.6[%])")
        assert f"{QUOTE_OPEN_MARKER}{CELL}{QUOTE_CLOSE_MARKER}" in text
        assert HEADER in text

    def test_the_caption_arrives_when_it_stands_alone_above_the_table(self):
        assert CAPTION in _window_text(CAPTION_ABOVE, CELL)

    @pytest.mark.parametrize("paper,why", [
        (CAPTION_BELOW, "printed below the table"),
        (CAPTION_IN_PARAGRAPH, "folded into a running paragraph"),
        (CAPTION_FAR_ABOVE, "held off by more than two blank lines"),
    ])
    def test_the_caption_does_not_always_arrive(self, paper, why):
        # quote_context._caption_line. This is what refutes a prompt that
        # promises the caption unconditionally, "however far above the cell
        # they sit".
        assert CAPTION not in _window_text(paper, CELL), why

    def test_a_prose_quote_gets_a_window_but_no_table_expansion(self):
        text = _window_text(PLAIN_PROSE, "measured with the WDS-9", chars=60)
        assert "drawn from two test sites" in text
        assert "|" not in text

    def test_a_record_check_shows_other_fields_values(self):
        # checker_prompts.build_record_context, fed by the template's
        # `checker_context_fields`. This is what refutes any claim that the
        # checker "does not see other extracted fields".
        label = build_record_context(
            {"record_id": "relationship_3",
             "gauge": {"value": "WDS-9"},
             "outcome_variable": {"value": "Unplanned removal"}},
            ["gauge", "outcome_variable"])
        assert "WDS-9" in label and "Unplanned removal" in label

    def test_image_evidence_is_the_cropped_image_and_carries_no_window(self):
        rendered, attach = _render_evidence_block(
            "<img>table_02</img>", {"table_02"}, paper_text=CAPTION_ABOVE,
            context_chars=1000)
        assert attach == ["table_02"]
        assert QUOTE_OPEN_MARKER not in rendered

    def test_zero_context_really_does_show_the_quote_alone(self):
        rendered, _ = _render_evidence_block(
            f"<q>{CELL}</q>", set(), paper_text=CAPTION_ABOVE,
            context_chars=0)
        assert rendered == f'"{CELL}"'

    def test_prose_beside_a_tag_is_stripped_from_the_evidence(self):
        # checker_prompts._render_evidence_block, tagged branch: the string is
        # parsed into quotes and images and the prose between them is
        # discarded, so an argument written around a quote reaches the checker
        # nowhere. This is what a claim about stripping may be made about.
        rendered, _ = _render_evidence_block(
            f"<q>{CELL}</q> the denominator is the whole sample",
            set(), paper_text=CAPTION_ABOVE, context_chars=1000)
        assert CELL in rendered
        assert "denominator" not in rendered

    def test_untagged_prose_only_evidence_is_shown_whole(self):
        # The same function's other branch, and the one an absolute claim gets
        # wrong: an evidence string carrying no tag at all is treated as a
        # single quote, so the checker IS shown it — under a statement that it
        # could not be located in the paper, which is what the window machinery
        # can say about it and all it can say.
        prose = "The denominator is the whole sample rather than the subgroup."
        rendered, _ = _render_evidence_block(
            prose, set(), paper_text=CAPTION_ABOVE, context_chars=1000)
        assert prose in rendered
        assert "could not be located in the paper text" in rendered


# ---------------------------------------------------------------------------
# 2. The pair: every claim checked against both prompts of every example
# ---------------------------------------------------------------------------

def _rendered(role, prompt_path):
    """A system prompt as a model receives it: the whole composed message.

    The claims below are about what reaches a model, and most of the checker's
    contract is in the engine's prompt rather than in a bundle's covering text.
    Reading the bundle's file alone would check the wrapper and skip the body,
    so the engine's half is composed here exactly as the builders compose it.
    Every branch is expanded, so a stage a bundle happens to disable today
    cannot hide a false claim that surfaces when someone switches it on.
    """
    partials = prompt_path.parent / "partials"
    return join_blocks(
        compose_engine_prompt(role, partials, predicates=EXPAND_ALL_BRANCHES),
        substitute_include_placeholders(
            prompt_path.read_text(encoding="utf-8"), partials,
            predicates=EXPAND_ALL_BRANCHES),
    )


def _example_prompt_pairs():
    """(bundle name, extractor text, checker text) for every config bundle.

    Discovered rather than named, so adding a config fixture cannot leave
    this file silently checking a subset of what the repository ships.
    """
    pairs = []
    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    for pipeline in sorted(fixtures.glob("*/pipeline.yaml")):
        prompts = pipeline.parent / "prompts"
        extractor = prompts / "extractor_system.md"
        checker = prompts / "checker_system.md"
        if extractor.exists() and checker.exists():
            pairs.append(pytest.param(
                _rendered(EXTRACTOR_SYSTEM, extractor),
                _rendered(CHECKER_SYSTEM, checker),
                id=pipeline.parent.name))
    assert pairs, "no config fixture with both system prompts"
    return pairs


EXAMPLE_PROMPT_PAIRS = _example_prompt_pairs()

# Each entry: (what the engine does, the patterns that assert its negation).
# A prompt may say any of this any way it likes; it may not say the opposite.
REFUTED_CLAIMS = [
    (
        "the checker is shown the paper text around each quote "
        "(checker_prompts._render_context_block)",
        [
            r"shown nothing else",
            r"nothing else[:,]?\s*not the paper",
            r"sees nothing (?:else )?but the (?:quote|evidence)",
            r"(?:does not|doesn't|never) see (?:any of )?the paper\b",
        ],
    ),
    (
        "a table quote arrives with that table's header row, past the budget "
        "(quote_context._window_bounds)",
        [
            r"(?:cannot|can ?not|can't) see the table",
            r"blind to the table's",
            r"without the (?:table's )?(?:header row|column headings)",
        ],
    ),
    (
        "the caption arrives only when it stands alone above the table "
        "(quote_context._caption_line)",
        [
            r"caption,? however far above",
            r"always (?:arrives with|carries) (?:its|the) caption",
        ],
    ),
    (
        "a record check shows the template's context-field values "
        "(checker_prompts.build_record_context)",
        [
            r"(?:does not|doesn't|never) see (?:any )?other extracted fields",
            r"(?:does not|doesn't|never) see the other fields",
        ],
    ),
    (
        "prose is stripped only where the evidence carries a tag beside it; "
        "untagged prose-only evidence is shown to the checker whole, as a "
        "quote the paper does not locate "
        "(checker_prompts._render_evidence_block)",
        [
            r"prose[^.]*\b(?:is |are )?not shown\b",
            r"prose[^.]*\bnever (?:shown|reaches|reads)\b",
            r"prose[^.]*\bnot shown (?:to the checker )?at all\b",
            r"(?:does not|doesn't|never) (?:see|read)s? [^.]*\bprose\b",
        ],
    ),
]

# Facts both prompts have to carry, so one cannot quietly drop what the other
# asserts. Deliberately loose: they pin the subject matter, not the wording.
#
# They are about the SHAPE of a check, which is what the extractor weighs a
# challenge by: one field at a time, on a narrow slice of context. How wide
# that slice is, is described to the checker, which reads its own material,
# and left to the checker's own prompt; a second description of it in the
# extractor's would be a second copy free to drift from the code, which is
# the failure the refuted claims above exist to catch.
SHARED_FACTS = [
    ("the checker judging one field at a time", [r"one field at a time"]),
    ("how little of the extraction the checker is given",
     [r"limited (?:amount of )?context", r"deliberately narrow",
      r"narrower model"]),
]


@pytest.mark.parametrize("extractor,checker", EXAMPLE_PROMPT_PAIRS)
class TestTheTwoPromptsAgree:

    @pytest.mark.parametrize("fact,patterns", REFUTED_CLAIMS,
                             ids=lambda v: None if isinstance(v, list) else v)
    def test_neither_prompt_asserts_a_refuted_claim(
            self, extractor, checker, fact, patterns):
        for role, text in (("extractor", extractor), ("checker", checker)):
            for pattern in patterns:
                hit = re.search(pattern, text, re.IGNORECASE)
                assert hit is None, (
                    f"{role}_system.md says {hit.group(0)!r}, but {fact}. "
                    "See TestWhatTheCheckerIsActuallyShown."
                )

    @pytest.mark.parametrize("fact,patterns", SHARED_FACTS,
                            ids=lambda v: None if isinstance(v, list) else v)
    def test_both_prompts_describe_the_checkers_context(
            self, extractor, checker, fact, patterns):
        def states(text):
            return any(re.search(p, text, re.IGNORECASE) for p in patterns)
        assert states(extractor) and states(checker), (
            f"only one of the two prompts mentions {fact}; they have to "
            "describe the same checker."
        )

    def test_the_question_the_checker_answers_is_the_same_in_both(
            self, extractor, checker):
        # The one thing the two prompts must agree on word for word in
        # substance: what a verdict is a verdict ABOUT. An extractor briefed
        # that the checker rules on the field as a whole, or on the extraction,
        # would weigh every challenge against the wrong question.
        for role, text in (("extractor", extractor), ("checker", checker)):
            sentences = re.split(r"(?<=[.!?])\s+", text)
            assert any(
                re.search(r"\bevidence\b", s, re.I)
                and re.search(r"\bsupports?\b", s, re.I)
                and re.search(r"\bvalue\b", s, re.I)
                for s in sentences), (
                f"{role}_system.md never says, in one sentence, that the "
                "checker judges whether the evidence supports the value")
