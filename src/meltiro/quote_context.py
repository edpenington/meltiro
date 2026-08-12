"""Surrounding paper text for a quote the checker is shown.

A quote read alone can be unreadable. A table cell holding `88 (7.0)` cannot
be judged without the column header that carries the `%`: on its own it could
be a count, a percentage, or both. So the checker is shown a WINDOW of the
paper around each matched quote, and asked whether the evidence, read in that
context, supports the value.

This module computes the windows. It does no rendering decisions of its own
beyond marking the quoted span: how the windows are introduced to the checker
is engine framing and lives in `checker_prompts`.

Four rules define a window.

1. `context_chars` characters on EACH side of the matched span. Zero means no
   window at all, and this module returns nothing.
2. The window is snapped OUTWARD to whole lines, so it never starts or ends
   mid-word, but only while that costs no more than `_LINE_SNAP_SLACK`; past
   that it snaps to a word boundary instead. A line here is usually a whole
   paragraph (markdown-converted papers routinely carry two-thousand-
   character lines), and the checker's worth is that it is uncorrelated with
   the extractor — a property that decays the more of the paper it reads —
   so the snap is bounded (see `_LINE_SNAP_SLACK`). An ordinary window is at
   most `context_chars + _LINE_SNAP_SLACK` per side.
3. A quote written with an elision marker (` ... `) matches the paper in
   several fragments, and each fragment gets its own window. One window
   spanning the gap between two fragments would show the checker text the
   extractor deliberately elided, presented as if it sat between the quoted
   parts. Windows that overlap after the snap are merged, so text is never
   shown twice. This is also where the distance between fragments is
   disclosed rather than bounded, which is the half of the bargain
   `quote_check.locate_quote` relies on when it constrains an elision to
   run forwards and leaves how far it runs unconstrained: fragments a page
   apart arrive here as separate windows and reach the checker as separate
   passages, never spliced into one run of text.
4. When the matched span sits inside a markdown table, the window is extended
   upward to the top of that table and to the table's caption line, REGARDLESS
   of the character budget and of the slack in rule 2. This is the rule that
   resolves the motivating case: a fixed window cuts a wide table mid-row, and
   the header row is exactly what says whether a number is a count or a
   percentage. It is the one deliberate exemption: bounding it would give back
   the only thing the window was built to buy.

Positions come from `quote_check.locate_quote`, which reports where each
fragment landed in the paper text as the paper actually writes it. A quote
whose position cannot be established (it does not match at all, or it matched
in a transformed copy whose index map could not be tracked) yields no windows,
and the caller says so rather than showing less than it promised.
"""

from bisect import bisect_right
from collections import namedtuple
from functools import lru_cache

from meltiro.quote_check import locate_quote


# The markers wrapping the quoted span inside a rendered window. The checker
# must never mistake surrounding paper text for the evidence it was handed, so
# the span is delimited explicitly. The forms are chosen to be absent from
# research-paper prose (as `<q>` is on the extractor's side) and to survive
# inside a markdown table cell.
QUOTE_OPEN_MARKER = "[[QUOTE]]"
QUOTE_CLOSE_MARKER = "[[/QUOTE]]"

# Blank lines tolerated between a markdown table and the caption line above it.
# Two is enough for the usual "caption, blank, table" layout with a stray blank
# line; beyond that the line above is not this table's caption.
_MAX_CAPTION_GAP_LINES = 2

# How far past the character budget a window edge may travel to reach a line
# boundary. Beyond this it snaps to a word boundary instead, so the window
# never begins or ends mid-word but also never grows without limit. Same idea
# and same purpose as `quote_check._WORD_SNAP_SLACK`: widen a span to
# something readable, but not so far that it drags in text nobody asked for.
#
# Why 250, from measured paper text rather than from taste. What the constant
# governs is the SNAP DISTANCE at a window edge, not the length of a line: an
# edge landing uniformly inside a line of length L travels a distance uniform
# on [0, L), so long lines are over-represented in the distances actually paid,
# and a slack chosen off the line-length median would be optimistic. Measured
# over every offset in the three bundled paper texts, the median snap distance
# is 175 in `bundle_unicode`, 163 in `bundle_tables` and 35 in
# `bundle_minimal`; the figures are reproducible from the fixtures in this
# repo, which is the point of quoting them. 250 clears all three, so in each of
# them MORE THAN HALF of window edges still reach a whole line boundary: 61%,
# 60% and 100%. The first two are what set the constant and the reason it is
# not smaller, since both carry the long markdown-converted paragraphs a real
# paper carries; a slack justified against `bundle_minimal` alone would only
# flatter itself, because nothing in that text is long enough to test the
# bound. 250 is also a quarter of the default 1000-character budget, which is
# the other half of the choice: the realised window can exceed the configured
# budget by at most 25% per side, whatever the paper's longest line happens to
# be. Without the bound a single 20,000-character line would hand the checker
# half the document, which is the failure this guards against.
_LINE_SNAP_SLACK = 250

# How far an edge may travel to get off the middle of a word once the line snap
# has been refused. Bounded for the same reason `quote_check._WORD_SNAP_SLACK`
# is, and to the same value: an edge inside a pathologically long token is left
# where it is rather than dragged somewhere unrecognisable.
_WORD_SNAP_SLACK = 24


# One window of paper text. `start`/`end` are offsets into the paper text
# (`end` exclusive); `spans` is the tuple of `(start, end)` quoted spans that
# fall inside it, in document order. A merged window carries more than one.
ContextWindow = namedtuple("ContextWindow", "start end spans")

# Line offsets for one paper text: `starts[i]` is the offset of line i, and
# `lines[i]` is that line without its newline.
_LineIndex = namedtuple("_LineIndex", "starts lines")


@lru_cache(maxsize=4)
def _line_index(paper_text):
    """Line starts and line texts for `paper_text`.

    Cached on the same terms as `quote_check._paper_haystacks`: one tool call
    can carry many fields, each with several quotes, and every one of them
    windows into the same paper.
    """
    starts = []
    lines = []
    pos = 0
    for line in paper_text.split("\n"):
        starts.append(pos)
        lines.append(line)
        pos += len(line) + 1
    return _LineIndex(tuple(starts), tuple(lines))


def _line_of(index, offset):
    """Index of the line containing `offset`, clamped into range."""
    if offset <= 0:
        return 0
    line_no = bisect_right(index.starts, offset) - 1
    return max(0, min(line_no, len(index.lines) - 1))


def _line_end(index, line_no):
    """Offset one past the last character of `line_no`, excluding its
    newline."""
    return index.starts[line_no] + len(index.lines[line_no])


def _is_table_line(line):
    """Whether `line` is a row of a markdown table.

    A pipe in the first non-space column is the whole test. It accepts the
    header row, the `| --- |` delimiter, and every body row, and it rejects
    prose that merely contains a pipe.
    """
    return line.lstrip().startswith("|")


def _table_block_start(index, line_no):
    """First line of the markdown table containing `line_no`, or None when
    that line is not part of a table.

    The table is the run of contiguous lines beginning with a pipe, walked
    backwards from `line_no`. A blank line ends the table, which is what
    markdown means by it too.
    """
    if not _is_table_line(index.lines[line_no]):
        return None
    first = line_no
    while first > 0 and _is_table_line(index.lines[first - 1]):
        first -= 1
    return first


def _caption_line(index, table_first):
    """The caption line above the table starting at `table_first`, or None.

    The caption is the nearest non-blank line above the table, across at most
    `_MAX_CAPTION_GAP_LINES` blank lines. It must stand on its own (the line
    above it is blank, or it is the first line of the paper): the last line of
    a running paragraph is not a caption, and dragging half a sentence into
    the window would be worse than showing no caption at all.
    """
    i = table_first - 1
    gap = 0
    while i >= 0 and not index.lines[i].strip():
        gap += 1
        if gap > _MAX_CAPTION_GAP_LINES:
            return None
        i -= 1
    if i < 0:
        return None
    if not index.lines[i].strip() or _is_table_line(index.lines[i]):
        return None
    if i > 0 and index.lines[i - 1].strip():
        return None
    return i


def _table_top(index, offset):
    """Offset of the top of the table (its caption line where there is one)
    containing `offset`, or None when `offset` is not inside a table."""
    line_no = _line_of(index, offset)
    table_first = _table_block_start(index, line_no)
    if table_first is None:
        return None
    caption = _caption_line(index, table_first)
    return index.starts[table_first if caption is None else caption]


def _word_start(paper_text, at):
    """`at` moved backward off the middle of a word, by at most the word
    slack."""
    floor = max(0, at - _WORD_SNAP_SLACK)
    while (at > floor and paper_text[at - 1].isalnum()
            and paper_text[at].isalnum()):
        at -= 1
    return at


def _word_end(paper_text, at):
    """`at` moved forward off the middle of a word, by at most the word
    slack."""
    ceiling = min(len(paper_text), at + _WORD_SNAP_SLACK)
    while (at < ceiling and paper_text[at - 1].isalnum()
            and paper_text[at].isalnum()):
        at += 1
    return at


def _snap_start(paper_text, index, ws):
    """The window's start, snapped outward to the start of its line when that
    costs no more than `_LINE_SNAP_SLACK`, and to a word boundary otherwise."""
    line_start = index.starts[_line_of(index, ws)]
    if ws - line_start <= _LINE_SNAP_SLACK:
        return line_start
    return _word_start(paper_text, ws)


def _snap_end(paper_text, index, we):
    """The window's end, snapped outward to the end of its line when that
    costs no more than `_LINE_SNAP_SLACK`, and to a word boundary
    otherwise."""
    if we >= len(paper_text):
        return len(paper_text)
    line_end = min(_line_end(index, _line_of(index, we)), len(paper_text))
    if line_end - we <= _LINE_SNAP_SLACK:
        return line_end
    return _word_end(paper_text, we)


def _window_bounds(paper_text, index, start, end, context_chars):
    """`(window_start, window_end)` around the span `[start, end)`.

    Two rules widen the budget, and only one of them is bounded. Ordinary line
    snapping is bounded by `_LINE_SNAP_SLACK`, so the window is at most
    `context_chars + _LINE_SNAP_SLACK` per side. The TABLE rule is deliberately
    not: reaching a table's header row and caption is the point of the feature,
    and a header out of budget is exactly the case that motivated it.
    """
    ws = _snap_start(paper_text, index,
                     max(0, start - context_chars))
    we = _snap_end(paper_text, index,
                   min(len(paper_text), end + context_chars))

    # The table rule, exempt from the budget by design. Both ends of the span
    # are probed, so a quote that starts in prose and runs into a table is
    # handled as well as the ordinary case of a quote wholly inside one cell.
    for probe in (start, max(start, end - 1)):
        top = _table_top(index, probe)
        if top is not None:
            ws = min(ws, top)
    return ws, we


def _merge(windows):
    """Merge overlapping or touching windows, unioning their quoted spans.

    Two fragments of one elided quote can sit close enough that their windows
    overlap. Showing the overlap twice would read as two passages of the paper
    where there is one, so they become a single window carrying both spans.
    """
    out = []
    for window in sorted(windows, key=lambda w: (w.start, w.end)):
        if out and window.start <= out[-1].end:
            prev = out[-1]
            out[-1] = ContextWindow(
                prev.start,
                max(prev.end, window.end),
                prev.spans + window.spans,
            )
        else:
            out.append(window)
    return out


def quote_context_windows(quote, paper_text, context_chars):
    """Windows of surrounding paper text for `quote`, or None.

    None means no context could be resolved and the caller must say so: the
    quote does not match the paper (an output written against different text,
    or a paper edited after the run), or it matches at a position that could
    not be tracked back to
    the paper's own text. None is also returned when the caller asked for no
    context (`context_chars` of 0), where there is nothing to say.

    Otherwise: one window per matched fragment, in document order, overlapping
    windows merged.
    """
    if context_chars <= 0 or not quote or not paper_text:
        return None
    match = locate_quote(quote, paper_text)
    if not match:
        return None

    spans = [(f.start, f.end) for f in match.fragments]
    # A fragment whose position could not be tracked makes the whole quote
    # unresolvable rather than partially resolvable: showing windows for some
    # fragments and silently dropping the rest is exactly the quiet shortfall
    # this path exists to avoid.
    if any(s is None or e is None or e <= s for s, e in spans):
        return None

    index = _line_index(paper_text)
    windows = []
    for start, end in spans:
        ws, we = _window_bounds(paper_text, index, start, end, context_chars)
        windows.append(ContextWindow(ws, we, ((start, end),)))
    return _merge(windows)


def render_window(paper_text, window):
    """One window as text, with each quoted span wrapped in the markers.

    Leading and trailing whitespace left by the line snap is stripped; the
    markers themselves are never touched by that.
    """
    parts = []
    cursor = window.start
    for start, end in sorted(window.spans):
        if start < cursor:
            # Two fragments of one quote landing on overlapping text. The
            # first marking wins; marking the overlap twice would produce
            # nested markers and no extra information.
            continue
        parts.append(paper_text[cursor:start])
        parts.append(QUOTE_OPEN_MARKER)
        parts.append(paper_text[start:end])
        parts.append(QUOTE_CLOSE_MARKER)
        cursor = end
    parts.append(paper_text[cursor:window.end])
    return "".join(parts).strip()
