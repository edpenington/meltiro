"""Verbatim-quote validation for evidence envelopes.

`normalise_quote_text` is the single source of truth for the canonical form
used in comparisons. It is applied identically to the candidate quote and to
the paper text before substring search, so it must tolerate the differences
PDF-to-text conversion introduces (ligatures, hyphenation across line
breaks, smart quotes) and the model's light clean-up when copying.

Matching is exact-substring after normalisation; there is no fuzzy
tolerance. Fuzzy scoring appears in exactly one place and never accepts
anything: `suggest_nearest_text` uses it to show a model what the paper says
at the point it was aiming for, so a near miss can be corrected.

Markdown syntax is NOT normalised away. Where a converter renders an
italicised statistic as `*N* = 1,227`, that is the quote that passes; the
reader-facing `N = 1,227` is rejected. This function decides whether a value
is supported by the paper, so every character it agrees to ignore is a
character a fabricated quote can differ by. The burden sits on the
converter: the paper-bundle contract in the README asks it to keep inline
emphasis out of running text. Anything added to the normalisation list below
must clear the same bar — it cannot make a wrong quote look right.

Every quote gets two readings, and either matching is enough: the literal
reading, tried first, and the reading with bracketed interpolations removed
(`strip_interpolations`).

An elision marker asserts that the fragments either side of it appear in the
paper in the order written, one passage running on into the next with words
elided between them, so that is what is checked: each fragment must land at
or after the end of the fragment before it. Unordered fragments would let
one `<q>` stitch phrases from unrelated sections into a sentence the paper
never writes and certify the result as verbatim.

Case is the weakest thing a match can rest on and is folded last, so it
gets a tier of its own: a fragment that matches only once both sides are
lowercased is reported as `case_folded`, not passed off as text the paper
writes that way.

`locate_quote` reports which reading matched, which tier accepted it, and
where in the paper the quote landed; `find_quote` is its boolean wrapper.
"""

import difflib
import re
import unicodedata
from bisect import bisect_left
from collections import namedtuple
from functools import lru_cache

from meltiro.bundle import normalise_label


# Page-break markers inserted by the upstream text-extraction step. Removed before
# normalisation so quotes that span a page boundary still match.
PAGE_BREAK_PATTERN = re.compile(r"---\s*PAGE BREAK\s*---", re.IGNORECASE)

# Hyphenation across a line break: "regres-\nsion" -> "regression". The
# hyphen is dropped along with the surrounding whitespace/newline. Same
# treatment for soft hyphens (U+00AD) at end-of-line: PDF text extraction
# routinely produces `par­\nticipants` which must heal to `participants`.
HYPHEN_LINE_BREAK_PATTERN = re.compile(r"[­-]\s*\n\s*")

# Soft hyphen (U+00AD): invisible hint for line breaking; the line-break
# regex above handles the end-of-line case, but stray in-line soft hyphens
# (rare) still get stripped.
SOFT_HYPHEN = "­"

# Smart quotes and dashes that the extractor or PDF parser might emit
# instead of the ASCII versions in the paper text (or vice versa).
SMART_QUOTE_MAP = str.maketrans({
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "′": "'",  # prime
    "″": '"',  # double prime
    "–": "-",  # en dash
    chr(0x2014): "-",  # em dash (U+2014)
    "−": "-",  # minus sign
})

# Final pass: collapse any run of whitespace (incl. tabs and remaining
# newlines) to a single ASCII space.
WHITESPACE_RUN_PATTERN = re.compile(r"\s+")

# --------------------------------------------------------------------------
# Tolerance patterns
# --------------------------------------------------------------------------
# Editorial annotations the extractor sometimes adds inside a quote, as a
# CLOSED list of standard academic forms. Stripped from the extractor's
# quote before matching against the paper. Two look-alike bracket forms
# route elsewhere: `[...]` is a fragment separator (_ELLIPSIS_SEPARATOR),
# and any other bracketed text is an interpolation (strip_interpolations).
_EDITORIAL_BRACKET = re.compile(
    r"\[(?:"
    r"sic"
    r"|emphasis (?:added|mine|ours|in original)"
    r"|our emphasis|original emphasis"
    r")\]",
    re.IGNORECASE,
)

# Inline citation markers in the PAPER that the extractor often (and
# reasonably) omits from a quote. Two flavours, both intentionally
# conservative to avoid eating real content:
#
#   - Numeric reference brackets: [12], [12,15], [12-15], [12–17].
#     Pattern is "[" + digits (optionally with comma/dash/en-dash
#     separators between further digit groups) + "]". Does NOT match
#     [the patients] or [sic]; those go through the editorial path.
#   - Parenthetical author-year citations: (Smith, 2019),
#     (Smith et al., 2019), (Smith and Jones 2019; Brown 2020).
_NUMERIC_CITATION = re.compile(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]")

# A citation is recognised by its SHAPE, attribution included, and the whole
# parenthetical has to be citations and nothing else. A year alone is not
# enough: `(from the 2019 baseline)` qualifies its sentence, and deleting it
# from the paper before comparison would let a quote drop the qualifier and
# still be certified verbatim. A capitalised word before the year is not
# enough either (`(March 2019)`, `(Q4 2020)`), so an attribution has to
# carry a signal a date cannot:
#
#   - `et al.`;
#   - a co-author joined by `and` or `&`;
#   - a name particle (`van der Berg 2020`);
#   - the comma APA puts between author and year (`Smith, 2019`);
#   - or, for a bare `Smith 2019`, a second entry after a semicolon
#     (`Smith 2019; Brown 2020`), a list form prose does not use.
#
# A citation form this does not recognise is simply not stripped: the quote
# is refused and comes back with the paper's own wording attached, never
# accepted on the strength of a parenthetical the pattern was unsure about.
_YEAR = r"(?:19|20)\d{2}[a-z]?"
# Page or section locator trailing the year: `, p. 12`, `, 45`, `: 45-47`.
_LOCATOR = r"(?:\s*[,:]\s*(?:pp?\.\s*)?\d+(?:\s*[-–]\s*\d+)?)?"
_NAME_PARTICLE = r"(?:van|von|de[nrl]?|della|di|da|dos|du|la|le|ten|ter)"
# A surname starts capitalised and carries at least two letters, so `Q4` and
# other capitalised label-plus-digit tokens are not surnames.
_SURNAME = r"[A-Z][^\W\d_][^\s;,()]*"
_ET_AL = r"et\s+al\.?"
# An author part that could not be an ordinary capitalised noun.
_ATTRIBUTED = (
    rf"(?:{_NAME_PARTICLE}\s+)+{_SURNAME}"
    rf"|{_SURNAME}(?:\s+(?:and|&)\s+{_SURNAME})+(?:\s+{_ET_AL})?"
    rf"|{_SURNAME}\s+{_ET_AL}"
)
# One entry carrying its own signal, and one relying on the semicolon list.
_CITATION_ENTRY = (
    rf"(?:(?:{_ATTRIBUTED}),?\s*|{_SURNAME},\s*){_YEAR}{_LOCATOR}"
)
_LISTED_ENTRY = (
    rf"(?:{_CITATION_ENTRY}|{_SURNAME}\s+{_YEAR}{_LOCATOR})"
)
_PAREN_CITATION = re.compile(
    rf"\(\s*{_LISTED_ENTRY}(?:\s*;\s*{_LISTED_ENTRY})+\s*\)"
    rf"|\(\s*{_CITATION_ENTRY}\s*\)"
)

# Trailing sentence punctuation the extractor might add when it ended a
# quote at a sentence break but the paper's run-on continues without
# punctuation, or vice versa. Stripped from BOTH sides during the
# tolerant pass.
_TRAILING_PUNCT = re.compile(r"[.,;:!?]+\s*$")

# Bracketed interpolations: `448 (35.6[%])` says the paper writes
# `448 (35.6)` and the percent sign is the quoter's. But square brackets also
# occur in real paper text (`(634 [50.4%])`, `[1.0-7.0]`, `[12]`), and
# stripping them unconditionally would make those passages unquotable — hence
# the two readings, literal tried first (see the module docstring).
#
# Whitespace before the bracket goes with it, so an insertion between words
# (`the patients [in the trial] received`) and an insertion tight against a
# number (`35.6 [%]`) both leave the surrounding text as the paper writes it.
_BRACKET_INTERPOLATION = re.compile(r"\s*\[[^\[\]]*\]")


def _strip_editorial(text):
    """Remove the closed list of `[sic]` / `[emphasis added]` editorial
    annotations from `text` (typically the extractor's quote)."""
    return _EDITORIAL_BRACKET.sub("", text)


def _brackets_are_flat(text):
    """True when every `[` closes before the next one opens.

    Nesting is not part of the interpolation convention, and an unbalanced
    bracket is as likely to be a truncated quote as an insertion, so
    neither gets a bracket-stripped reading.
    """
    open_bracket = False
    for ch in text:
        if ch == "[":
            if open_bracket:
                return False
            open_bracket = True
        elif ch == "]":
            if not open_bracket:
                return False
            open_bracket = False
    return not open_bracket


def strip_interpolations(text):
    """`text` with bracketed interpolations removed, or None when `text`
    offers no second reading: no bracket to strip, nested or unbalanced
    brackets, or stripping would consume the whole quote — a quote that is
    nothing but a bracketed insertion carries no paper text at all.
    """
    if not text or "[" not in text:
        return None
    if not _brackets_are_flat(text):
        return None
    stripped = _BRACKET_INTERPOLATION.sub("", text)
    if stripped == text:
        return None
    stripped = WHITESPACE_RUN_PATTERN.sub(" ", stripped).strip()
    if not stripped:
        return None
    return stripped


def _strip_citations(text):
    """Remove inline citation markers: numeric `[12]` brackets and
    parenthetical `(Smith et al., 2019)` forms, from `text` (typically
    the paper text)."""
    text = _NUMERIC_CITATION.sub("", text)
    text = _PAREN_CITATION.sub("", text)
    return text


def _strip_trailing_punct(text):
    return _TRAILING_PUNCT.sub("", text)


def _normalise_common(text):
    """Steps shared by both line-break-hyphen handling strategies."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(SMART_QUOTE_MAP)
    text = PAGE_BREAK_PATTERN.sub(" ", text)
    return text


def _finalise(text, fold_case=True):
    """Steps applied after the chosen hyphen-line-break strategy.

    DELETED: soft hyphens wherever they occur, and leading and trailing
    whitespace. FOLDED: every run of whitespace to one ASCII space and,
    when `fold_case`, every character to lower case. Nothing else.

    `fold_case` is what separates the two normalised tiers in
    `_locate_one_fragment`. Case is folded last and matters least to a PDF
    converter, so a fragment that needs it is a weaker match than one that
    does not, and the tier it lands in says so.
    """
    text = text.replace(SOFT_HYPHEN, "")
    text = WHITESPACE_RUN_PATTERN.sub(" ", text)
    if fold_case:
        text = text.lower()
    return text.strip()


def _normalise(text, keep_break_hyphens=False, fold_case=True):
    """The normalisation both public forms are made of.

    `keep_break_hyphens` chooses the hyphen-line-break strategy and
    `fold_case` chooses whether case survives; see `normalise_quote_text`
    for what each step does and why the order is what it is.
    """
    if text is None:
        return ""
    text = _normalise_common(text)
    text = HYPHEN_LINE_BREAK_PATTERN.sub(
        "-" if keep_break_hyphens else "", text)
    return _finalise(text, fold_case)


def normalise_quote_text(text):
    """Aggressively normalise text for verbatim-quote comparison.

    This is the operative definition of "verbatim" in this tool, so it is
    stated in full rather than left to be read off the code.

    DELETED: page-break markers, the hyphen at a line break, soft hyphens,
    and leading and trailing whitespace.
    FOLDED: NFKC (ligatures, the space family, other compatibility forms),
    smart quotes and the dash family to their ASCII forms, runs of
    whitespace to one space, and case.
    UNTOUCHED: everything else. Letters keep their script and their
    accents, symbols keep their identity (the multiplication sign is not an
    ASCII x), and markdown syntax stays exactly where the converter put it.

    Case folding is the weakest step and the matcher does not accept it
    silently: `_locate_one_fragment` reaches this form only in its last
    tier and reports a fragment that needed it as `case_folded`. This
    function folds case anyway because it is also the canonical comparison
    form for callers outside the matcher (`reference_lists`,
    `orchestrator`), where two spellings of one label differing only in
    case are the same label.

    Drops the hyphen at line breaks entirely, so `par-\\nticipants` heals
    to `participants`. For lexically-hyphenated words split across a line
    (`self-\\nassessment`) see `_normalise`'s `keep_break_hyphens`: the
    matcher searches a copy of the paper under each strategy, so a quote
    written either way lands.

    Order matters in two places:
      - Page-break stripping (in _normalise_common) must run BEFORE the
        hyphen-line-break regex; otherwise the regex eats the final
        `-\\n` of the `---` page-break marker, mangling it before it can
        be matched.
      - Hyphenation healing must run before whitespace collapse so the
        newline between the hyphen and the next character is still
        detectable.
    """
    return _normalise(text)


# --------------------------------------------------------------------------
# Offset tracking
# --------------------------------------------------------------------------
# Matching runs against transformed copies of the paper text, but a caller
# that wants to SHOW a match needs a position in the paper as the paper
# actually writes it. Each copy is therefore carried with an index map: a
# list as long as the transformed text, where entry i is the index in the
# original paper text that transformed character i came from.
#
# The maps are best effort. Two steps are not safe to track character by
# character in general (NFKC can compose across characters, `str.lower` has
# one context-dependent rule), so each is rebuilt only when it actually
# changes the text and the rebuild is checked against the plain result. On
# any disagreement the plain result wins and the map is dropped, so a
# position comes back as unavailable rather than wrong. Whether a quote
# matches never depends on the map.

def _tracked_sub(pattern, repl, text, index_map):
    """`pattern.sub(repl, text)`, keeping `index_map` in step.

    `repl` is a literal string, no group references. Characters it inserts
    are attributed to the first character of the match they replace.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return text, index_map
    if index_map is None:
        return pattern.sub(repl, text), None
    parts = []
    out_map = []
    pos = 0
    for match in matches:
        parts.append(text[pos:match.start()])
        out_map.extend(index_map[pos:match.start()])
        if repl:
            parts.append(repl)
            out_map.extend([index_map[match.start()]] * len(repl))
        pos = match.end()
    parts.append(text[pos:])
    out_map.extend(index_map[pos:])
    return "".join(parts), out_map


def _tracked_per_char(func, expected, text, index_map):
    """Rebuild `expected` one character at a time, keeping `index_map` in
    step. `func` maps one character to zero or more characters.

    Falls back to `expected` with no map when the per-character route does
    not reproduce it exactly, which is the guard described above.
    """
    if index_map is None:
        return expected, None
    parts = []
    out_map = []
    for ch, src in zip(text, index_map):
        out = func(ch)
        if not out:
            continue
        parts.append(out)
        out_map.extend([src] * len(out))
    rebuilt = "".join(parts)
    if rebuilt != expected:
        return expected, None
    return rebuilt, out_map


def _tracked_strip(text, index_map):
    stripped = text.strip()
    if stripped == text or index_map is None:
        return stripped, index_map
    lead = len(text) - len(text.lstrip())
    return stripped, index_map[lead:lead + len(stripped)]


def _tracked_common(text, index_map):
    """Tracked counterpart of `_normalise_common`."""
    if not unicodedata.is_normalized("NFKC", text):
        expected = unicodedata.normalize("NFKC", text)
        text, index_map = _tracked_per_char(
            lambda ch: unicodedata.normalize("NFKC", ch),
            expected, text, index_map)
    translated = text.translate(SMART_QUOTE_MAP)
    # Every entry in SMART_QUOTE_MAP replaces one character with one
    # character, so a translation of unchanged length is one for one and the
    # map still holds. The length check is belt and braces against a future
    # multi-character entry.
    if len(translated) == len(text):
        text = translated
    else:
        text, index_map = _tracked_per_char(
            lambda ch: ch.translate(SMART_QUOTE_MAP),
            translated, text, index_map)
    return _tracked_sub(PAGE_BREAK_PATTERN, " ", text, index_map)


def _tracked_finalise(text, index_map, fold_case=True):
    """Tracked counterpart of `_finalise`."""
    if SOFT_HYPHEN in text:
        text, index_map = _tracked_per_char(
            lambda ch: "" if ch == SOFT_HYPHEN else ch,
            text.replace(SOFT_HYPHEN, ""), text, index_map)
    text, index_map = _tracked_sub(
        WHITESPACE_RUN_PATTERN, " ", text, index_map)
    if fold_case:
        lowered = text.lower()
        # `str.lower` never deletes a character, so unchanged length means
        # one character in, one out, and the map still holds.
        if len(lowered) == len(text):
            text = lowered
        else:
            text, index_map = _tracked_per_char(
                str.lower, lowered, text, index_map)
    return _tracked_strip(text, index_map)


# One transformed copy of the paper text, with the index map back into the
# original. `index_map` is None when the map could not be tracked; the text
# is authoritative either way.
_Haystack = namedtuple("_Haystack", "text index_map")

# The copies the four matching tiers search, in the order they search them.
_PaperHaystacks = namedtuple(
    "_PaperHaystacks", "plain tolerant normalised case_folded")


def _checked_haystack(tracked, authoritative):
    """Keep the map only if the tracked build reproduced the plain result.

    The plain result is what decides whether a quote matches, so it is what
    gets stored. This is the last line of the guard: the tiers below behave
    identically whether or not tracking worked.
    """
    text, index_map = tracked
    if text != authoritative:
        return _Haystack(authoritative, None)
    return _Haystack(text, index_map)


def _tracked_citation_strip(haystack):
    """Tracked counterpart of `_strip_citations` over a haystack."""
    text, index_map = haystack
    text, index_map = _tracked_sub(_NUMERIC_CITATION, "", text, index_map)
    return _tracked_sub(_PAREN_CITATION, "", text, index_map)


def _tracked_normalise(text, index_map, keep_break_hyphens=False,
                       fold_case=True):
    """Tracked counterpart of `_normalise`, and so of the two public forms
    built on it."""
    text, index_map = _tracked_common(text, index_map)
    text, index_map = _tracked_sub(
        HYPHEN_LINE_BREAK_PATTERN, "-" if keep_break_hyphens else "",
        text, index_map)
    return _tracked_finalise(text, index_map, fold_case)


@lru_cache(maxsize=4)
def _paper_haystacks(paper_text):
    """Every transformed copy of `paper_text` the tiers search, each with
    its index map.

    Cached: one tool call can carry many fields, each with several quotes,
    and every one of them is checked against the same paper.
    """
    identity = list(range(len(paper_text)))

    plain = _checked_haystack(
        _tracked_sub(PAGE_BREAK_PATTERN, " ", paper_text, identity),
        PAGE_BREAK_PATTERN.sub(" ", paper_text),
    )

    tolerant = []
    seen = set()
    for text, index_map in (plain,
                            _tracked_citation_strip(plain)):
        text, index_map = _tracked_sub(
            WHITESPACE_RUN_PATTERN, " ", text, index_map)
        text, index_map = _tracked_strip(text, index_map)
        if text in seen:
            continue
        seen.add(text)
        tolerant.append(_Haystack(text, index_map))

    # Both hyphen strategies, once with case preserved and once folded. The
    # folded copies are the tier of last resort, so they are built as their
    # own pair rather than merged into the pair above: a fragment that only
    # matches here is reported as having needed case folding.
    def _normalised_pair(fold_case):
        return tuple(
            _checked_haystack(
                _tracked_normalise(paper_text, identity,
                                   keep_break_hyphens=keep,
                                   fold_case=fold_case),
                _normalise(paper_text, keep_break_hyphens=keep,
                           fold_case=fold_case))
            for keep in (False, True)
        )

    return _PaperHaystacks(plain, tuple(tolerant),
                           _normalised_pair(False), _normalised_pair(True))


def _source_span(index_map, start, length):
    """Translate a hit in a transformed haystack back to a span in the
    original paper text.

    Returns `(start, end)` as offsets into the paper text passed to
    `locate_quote`, or `(None, None)` when the map is unavailable. `end` is
    one past the source character the last matched character came from, so
    whitespace that normalisation collapsed at the end of the match sits
    outside the span.
    """
    if index_map is None or length <= 0:
        return (None, None)
    return (index_map[start], index_map[start + length - 1] + 1)


# Quote fragment separator: the standard academic elision marker.
# `locate_quote` splits on this and matches each fragment separately but
# not independently — each must land at or after the end of the one before
# it, because the marker asserts a single passage read forwards with words
# dropped out of the middle (see the module docstring). Separate <q>
# blocks remain the right form for phrases from unrelated passages.
#
# Three forms are recognised:
#   - ` ... ` (whitespace + 3+ dots + whitespace)
#   - ` … ` (whitespace + Unicode horizontal ellipsis + whitespace)
#   - `[...]` / `[ ... ]` / `[…]` (bracketed elision marker, with any
#     internal whitespace; surrounding whitespace optional)
_ELLIPSIS_SEPARATOR = re.compile(
    r"\s*\[\s*(?:\.{3,}|…)\s*\]\s*"
    r"|\s+(?:\.{3,}|…)\s+"
)


def _split_fragments(quote):
    """`quote` split on the elision marker, empties dropped.

    One list, one definition: what `locate_quote` orders and what
    `_failing_fragment` reports on have to be the same fragments.
    """
    return [f for f in
            (f.strip() for f in _ELLIPSIS_SEPARATOR.split(quote.strip()))
            if f]


def _tolerant_needles(needle):
    """The needle forms the tolerant tier accepts, in a fixed order.

    As written, with editorial annotations removed, with trailing sentence
    punctuation removed, with both. Whitespace is collapsed on each so a
    swallowed citation doesn't leave a double space that breaks the match,
    duplicates are dropped, and the order is fixed: the first hit decides
    the reported position, and a resumed run replays the message that
    position ends up in.
    """
    forms = (
        needle,
        _strip_editorial(needle),
        _strip_trailing_punct(needle),
        _strip_trailing_punct(_strip_editorial(needle)),
    )
    out = []
    for form in forms:
        form = WHITESPACE_RUN_PATTERN.sub(" ", form).strip()
        if form and form not in out:
            out.append(form)
    return out


def _find_from(haystack, needle, min_start):
    """Offset of the first occurrence of `needle` in `haystack` that starts
    at or after `min_start` in the PAPER's own coordinates, or -1.

    `min_start` of None means unconstrained, which is the first fragment of
    every quote and the whole of a quote without an elision marker.

    Under a constraint the search skips to the first haystack offset whose
    index map points at or past `min_start`. The map is non-decreasing (no
    transformation reorders text), so bisecting it is exact: every hit from
    there on satisfies the constraint and every hit before it does not.

    A haystack whose map could not be tracked cannot answer the question
    and is skipped rather than guessed at; only the last two tiers can
    lose their map.
    """
    if min_start is None:
        return haystack.text.find(needle)
    if haystack.index_map is None:
        return -1
    return haystack.text.find(needle, bisect_left(haystack.index_map,
                                                  min_start))


def _locate_one_fragment(fragment, paper_text, min_start=None):
    """Verbatim-or-tolerant-or-normalised-or-case-folded match for a single
    quote fragment, as one reading (no bracket stripping here).

    Returns `(start, end, tier)` on a hit, where start and end are offsets
    into `paper_text` (both None when the position could not be tracked),
    or None on a miss. With `min_start` set, only a hit starting at or
    after that offset counts; see `_find_from`.

    Four tiers, from strictest to most permissive:

    1. Direct substring match after stripping page-break markers.
    2. Tolerant pass: strip `[sic]` / `[emphasis added]` style editorial
       annotations from the quote, strip `[12]` /
       `(Smith et al., 2019)` style inline citations from the paper,
       strip trailing sentence punctuation from both. The model
       legitimately omits citations and adds editorial markers; this
       tier accepts those forms without forcing a retry.
    3. Aggressive normalisation fallback (ligatures, smart quotes,
       hyphenation across line breaks, whitespace collapse) to handle a
       text.txt the upstream cleaner never processed. Logged at debug
       level.
    4. The same normalisation with case folded as well. Kept separate
       because case is the one difference in tier 3's list that a PDF
       converter does not introduce: a fragment reaching this tier differs
       from the paper in a way the model chose, and is reported as
       `case_folded`.
    """
    haystacks = _paper_haystacks(paper_text)

    # Tier 1: direct substring.
    at = _find_from(haystacks.plain, fragment, min_start)
    if at >= 0:
        return _source_span(
            haystacks.plain.index_map, at, len(fragment)) + ("direct",)

    # Tier 2: tolerant, academic-quote conventions.
    for needle in _tolerant_needles(fragment):
        for haystack in haystacks.tolerant:
            at = _find_from(haystack, needle, min_start)
            if at >= 0:
                return _source_span(
                    haystack.index_map, at, len(needle)) + ("tolerant",)

    # Tiers 3 and 4: normalisation fallback, case preserved and then folded.
    # The log says which, because the two point at different culprits: the
    # first at a text.txt the upstream cleaner never processed, the second
    # at a quote that did not copy the paper's capitalisation.
    for fold_case, tier, note in (
            (False, "normalised",
             "upstream cleaner may have missed something"),
            (True, "case_folded", "the quote differs from the paper in case")):
        needle = _normalise(fragment, fold_case=fold_case)
        if not needle:
            return None
        pool = haystacks.case_folded if fold_case else haystacks.normalised
        for haystack in pool:
            at = _find_from(haystack, needle, min_start)
            if at >= 0:
                import logging
                logging.getLogger(__name__).debug(
                    "Quote fragment matched via %s fallback (%s): %r",
                    tier, note, fragment[:80],
                )
                return _source_span(
                    haystack.index_map, at, len(needle)) + (tier,)
    return None


# One fragment's match. `fragment` is the text as the model wrote it,
# `searched` is the text that actually matched (the same, unless the
# bracket-stripped reading was the one that landed), `reading` is
# "literal" or "interpolated", `tier` is "direct", "tolerant",
# "normalised" or "case_folded", and `start`/`end` are offsets into the
# paper text.
QuoteFragmentMatch = namedtuple(
    "QuoteFragmentMatch", "fragment searched reading tier start end")

# The tiers weakest last, which is the order `_locate_one_fragment` tries
# them in and the order a quote's own tier is aggregated by.
_TIER_ORDER = ("direct", "tolerant", "normalised", "case_folded")


def _weakest_tier(tiers):
    """The weakest of `tiers`, which is the one a whole quote rests on."""
    return max(tiers, key=_TIER_ORDER.index)


def _locate_fragment(fragment, paper_text, min_start=None):
    """Locate one fragment under both readings, literal first.

    Returns a `QuoteFragmentMatch`, or None if neither reading lands.
    `min_start` constrains where the fragment may land; see `_find_from`.
    """
    found = _locate_one_fragment(fragment, paper_text, min_start)
    if found is not None:
        start, end, tier = found
        return QuoteFragmentMatch(
            fragment, fragment, "literal", tier, start, end)

    stripped = strip_interpolations(fragment)
    if stripped is not None:
        found = _locate_one_fragment(stripped, paper_text, min_start)
        if found is not None:
            start, end, tier = found
            return QuoteFragmentMatch(
                fragment, stripped, "interpolated", tier, start, end)
    return None


class QuoteMatch(namedtuple(
        "QuoteMatch", "matched reading tier start end fragments")):
    """Where a quote landed in the paper, and how it got there.

    - `matched`: True iff every fragment matched. The object is falsy when
      it did not, so `if locate_quote(...)` reads the way it should.
    - `reading`: "literal" when every fragment matched as the model wrote
      it, "interpolated" when at least one fragment needed its bracketed
      insertions removed first. None on a miss.
    - `tier`: the WEAKEST tier any fragment needed: "direct", "tolerant",
      "normalised" or "case_folded", aggregated the way `reading` is, so a
      quote is never reported as a stronger match than its weakest part.
      None on a miss.
    - `start`, `end`: the first fragment's span, as offsets into the
      `paper_text` argument, so `paper_text[start:end]` is the paper's own
      wording of the quote and a caller wanting the quote IN CONTEXT can
      widen that span. Both are None on a miss, and both are None on a
      match whose position could not be tracked (see the offset-tracking
      section above; the match itself is unaffected).
    - `fragments`: one `QuoteFragmentMatch` per fragment, in the order the
      model wrote them, each carrying its own reading, tier and span. A
      quote with no elision marker has exactly one.
    """

    __slots__ = ()

    def __bool__(self):
        return self.matched


_NO_QUOTE_MATCH = QuoteMatch(False, None, None, None, None, ())


def locate_quote(quote, paper_text):
    """Locate `quote` in `paper_text`. Returns a `QuoteMatch`.

    Strategy:

    1. Split on the standard academic-elision marker ` ... ` (or the
       Unicode ellipsis ` … `) into one or more fragments. A quote
       without an ellipsis is one fragment.
    2. For each fragment, try the literal reading first: direct substring
       match (after stripping page-break markers from the haystack).
       Because the upstream text-extraction step runs deterministic
       cleaning (smart quotes, ligatures, dashes, soft hyphens, most
       hyphenation), a faithful fragment from the model matches directly.
    3. If the literal reading misses, fall back to the tolerant pass, then
       to the normalised match, then to the case-folded one. Those last
       two catch a text.txt the upstream cleaner never processed, and the
       rare case where the extractor or the PDF retained a subtlety the
       upstream cleaning does not strip. A debug log fires when either
       succeeds.
    4. If no tier accepts the fragment as written, try the same four
       tiers again on the bracket-stripped reading (`strip_interpolations`).

    Every fragment must match, IN ORDER, for the quote to pass: each one
    has to land at or after the end of the fragment before it, so the
    fragments read down the paper the way they read across the quote and
    never overlap. That is what an elision marker asserts; without it a
    single quote could reverse two phrases, or stitch together phrases
    from unrelated sections, and still be certified verbatim.

    Ordering is the whole of the constraint: there is deliberately no
    bound on the distance between fragments, since this module sees plain
    text with no section structure to measure against. Distance gets
    disclosure instead — each fragment carries its own span, and
    `quote_context` shows widely separated fragments as separate labelled
    passages, never as one run of text.

    Ordering also has to be established, not assumed: a fragment matched
    in a copy whose index map could not be tracked has no position to
    order by, so it can satisfy a quote of one fragment but not carry a
    quote of several.
    """
    if not quote or paper_text is None:
        return _NO_QUOTE_MATCH

    fragments = _split_fragments(quote)
    if not fragments:
        return _NO_QUOTE_MATCH

    matches = []
    min_start = None
    for fragment in fragments:
        match = _locate_fragment(fragment, paper_text, min_start)
        if match is None:
            return _NO_QUOTE_MATCH
        matches.append(match)
        if len(matches) < len(fragments):
            if match.end is None:
                return _NO_QUOTE_MATCH
            min_start = match.end

    first = matches[0]
    reading = ("literal" if all(m.reading == "literal" for m in matches)
               else "interpolated")
    return QuoteMatch(True, reading, _weakest_tier([m.tier for m in matches]),
                      first.start, first.end, tuple(matches))


def find_quote(quote, paper_text):
    """Return True iff `quote` appears in `paper_text`.

    True means every fragment of the quote is in the paper AND, for a
    quote written with an elision marker, that the fragments appear there
    in the order the quote writes them, one at or after the end of the
    last. The boolean face of `locate_quote`, which is the one to call
    when the position of the match, the reading or the tier matters.
    """
    return bool(locate_quote(quote, paper_text))


# --------------------------------------------------------------------------
# Nearest-text suggestion
# --------------------------------------------------------------------------
# A rejection that never shows the model what the paper says at the point it
# was aiming for gives it nothing to correct against: a model that writes
# `448 (35.6%)` for a table row reading `448 (35.6)` resubmits the same
# class of quote until the repeated-failure guard fires. So a failed quote
# comes back with the paper's own nearest wording.
#
# Anchor and score, rather than scan: scoring every window of the paper
# against every failed quote is quadratic. Instead, cut the quote into short
# overlapping anchors, find each one verbatim in the paper (a C-level
# substring scan), let each occurrence vote for the window it implies, and
# run the fuzzy comparison over only the best few dozen windows. Cost is
# linear in the paper length; the fuzzy work is bounded by constants.
#
# A quote so mangled that no anchor occurs anywhere gets no suggestion, and
# neither does one whose best window scores below the threshold: a wrong
# suggestion is worse than none, because the model will act on it.

# Below this many characters a quote has nothing distinctive enough to
# anchor on, and any "nearest text" would be noise.
_SUGGEST_MIN_CHARS = 6
# Only this much of a long quote is used for anchoring and scoring. Bounds
# the fuzzy comparison; the suggested span is the same length.
_SUGGEST_MAX_CHARS = 300
# Anchor lengths in characters, longest first, and the floor none of them
# may go below. Tried in turn until one produces a window worth offering:
# `448 (35.6%)` has no 12-character run in common with `448 (35.6)`, but
# it has several 5-character runs.
_ANCHOR_CHAR_STEPS = (12, 6, 4)
_MIN_ANCHOR_CHARS = 4
# Caps that keep the work bounded on a long quote in a long paper.
_MAX_ANCHORS = 48
_MAX_ANCHOR_HITS = 40
_MAX_CANDIDATE_WINDOWS = 400
_MAX_SCORED_WINDOWS = 24
# difflib ratio a window must reach to be offered at all. Same cutoff as
# `tools._suggest_closest_field` uses for field names.
_SUGGEST_MIN_RATIO = 0.6
# How far the suggested span may grow to avoid starting or ending mid-word.
_WORD_SNAP_SLACK = 24
# Longest suggestion shown in an error message.
_SUGGEST_PREVIEW_CHARS = 160


def _anchor_lengths(needle):
    """The anchor lengths to try for `needle`, longest first, with
    duplicates dropped. Never longer than half the needle (an anchor has to
    be able to miss the wrong part of the quote) and never shorter than
    `_MIN_ANCHOR_CHARS`."""
    lengths = []
    for step in _ANCHOR_CHAR_STEPS:
        length = min(len(needle),
                     max(_MIN_ANCHOR_CHARS, min(step, len(needle) // 2)))
        if length not in lengths:
            lengths.append(length)
    return lengths


def _anchor_votes(needle, haystack, anchor_chars):
    """Candidate window starts in `haystack`, each with the number of
    anchors that voted for it.

    An anchor is a short slice of `needle`. Every verbatim occurrence of it
    in the haystack implies a window start (the occurrence, less the
    anchor's offset within the needle), and several anchors agreeing on one
    start is good evidence that the window is the passage the quote was
    aiming at. An anchor with more than `_MAX_ANCHOR_HITS` occurrences is
    too common to be evidence of anything and is dropped, which is also
    what stops a common anchor from filling the candidate list with the
    first few hundred paragraphs of the paper.
    """
    last_offset = len(needle) - anchor_chars
    stride = max(anchor_chars // 2, 1,
                 -(-len(needle) // _MAX_ANCHORS))
    offsets = list(range(0, last_offset + 1, stride))
    if offsets[-1] != last_offset:
        offsets.append(last_offset)

    votes = {}
    for offset in offsets:
        anchor = needle[offset:offset + anchor_chars]
        hits = []
        at = haystack.find(anchor)
        while at >= 0 and len(hits) <= _MAX_ANCHOR_HITS:
            hits.append(at)
            at = haystack.find(anchor, at + 1)
        if len(hits) > _MAX_ANCHOR_HITS:
            continue
        for at in hits:
            start = max(0, at - offset)
            if start in votes:
                votes[start] += 1
            elif len(votes) < _MAX_CANDIDATE_WINDOWS:
                votes[start] = 1
    return votes


def _best_window(needle, haystack, votes):
    """The highest-scoring candidate window, or None if none reaches the
    threshold. Returns `(start, ratio)`.

    Windows are considered in a fixed order (most votes first, earliest
    position breaking the tie) and only the first `_MAX_SCORED_WINDOWS` are
    scored, so the answer is the same on every call with the same inputs.

    `quick_ratio` is an upper bound on `ratio` and costs a fraction of it,
    so it prefilters against the bar a window has to clear: the threshold
    to begin with, then the best score found so far. Since the ordering
    puts the likeliest window first, that usually retires the rest for the
    price of the bound. The pruning is exact. A window it skips could not
    have beaten the one already held, and a tie still goes to the window
    that came first, so the answer is the one the unpruned loop would give.
    """
    ranked = sorted(votes.items(),
                    key=lambda item: (-item[1], item[0]))[:_MAX_SCORED_WINDOWS]
    matcher = difflib.SequenceMatcher(None, "", needle, autojunk=False)
    best = None
    bar = _SUGGEST_MIN_RATIO
    for start, _votes in ranked:
        matcher.set_seq1(haystack[start:start + len(needle)])
        if matcher.quick_ratio() < bar:
            continue
        ratio = matcher.ratio()
        if ratio < bar:
            continue
        if best is None or ratio > best[1]:
            best = (start, ratio)
            bar = ratio
    return best


def _snap_to_word_bounds(text, start, end):
    """Widen a span so it does not start or end in the middle of a word.

    Bounded by `_WORD_SNAP_SLACK` characters at each edge: a span that
    lands inside a very long token is left as it is rather than dragged
    somewhere unrecognisable.
    """
    floor = max(0, start - _WORD_SNAP_SLACK)
    while (start > floor and text[start - 1].isalnum()
            and text[start].isalnum()):
        start -= 1
    ceiling = min(len(text), end + _WORD_SNAP_SLACK)
    while end < ceiling and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    return start, end


def _failing_fragment(quote, paper_text):
    """The first fragment of `quote` that neither reading locates ANYWHERE
    in the paper, or None when each of them is somewhere in it.

    Position is deliberately not constrained here: a quote that fails only
    because its fragments are out of order has no failing fragment and
    gets no suggestion, since the paper's nearest wording for such a
    fragment is the fragment itself. `validate_evidence` names the
    ordering instead.
    """
    for fragment in _split_fragments(quote):
        if _locate_fragment(fragment, paper_text) is None:
            return fragment
    return None


def _is_out_of_order_elision(quote, paper_text):
    """True when every fragment of `quote` is in the paper but the quote
    does not match, which leaves the order they were written in as the only
    thing wrong with it."""
    fragments = _split_fragments(quote)
    if len(fragments) < 2:
        return False
    return all(_locate_fragment(f, paper_text) is not None
               for f in fragments)


def suggest_nearest_text(quote, paper_text):
    """The paper's own wording closest to `quote`, or None.

    The quote is scored as the model wrote it, against the same normalised
    form the matcher compares, but what comes back is the text as the PAPER
    writes it, because that is what the model has to copy. Whitespace is
    collapsed to one line and long spans are truncated for use in an error
    message.

    None means no honest suggestion exists: nothing scored above
    `_SUGGEST_MIN_RATIO`, the quote is too short to anchor on, every
    fragment actually matched, or the position could not be tracked back to
    the paper's own text. Callers say nothing rather than guessing.
    """
    if not quote or not paper_text:
        return None

    fragment = _failing_fragment(quote, paper_text)
    if fragment is None:
        return None

    needle = normalise_quote_text(fragment)[:_SUGGEST_MAX_CHARS]
    if len(needle) < _SUGGEST_MIN_CHARS:
        return None

    # Scored against the fully-folded copy, case included: the quote being
    # scored has already failed every tier, so nothing is being certified
    # here, and folding case keeps a suggestion available for a near miss
    # that also differs in capitalisation.
    haystack = _paper_haystacks(paper_text).case_folded[0]
    if haystack.index_map is None:
        return None

    best = None
    for anchor_chars in _anchor_lengths(needle):
        votes = _anchor_votes(needle, haystack.text, anchor_chars)
        if not votes:
            continue
        best = _best_window(needle, haystack.text, votes)
        if best is not None:
            break
    if best is None:
        return None

    start = best[0]
    length = len(haystack.text[start:start + len(needle)])
    span_start, span_end = _source_span(haystack.index_map, start, length)
    if span_start is None or span_end <= span_start:
        return None

    span_start, span_end = _snap_to_word_bounds(
        paper_text, span_start, span_end)
    shown = WHITESPACE_RUN_PATTERN.sub(
        " ", paper_text[span_start:span_end]).strip()
    if not shown:
        return None
    if len(shown) > _SUGGEST_PREVIEW_CHARS:
        shown = shown[:_SUGGEST_PREVIEW_CHARS] + "..."
    return shown


# Inline tag patterns for the unified evidence string. The model produces
# evidence as ONE string per field; tagged blocks identify the verbatim
# quotes and image references, and everything outside the tags is
# interpretive prose (reasoning). Tags are short for token economy and
# unambiguous against research-paper content (no peer-reviewed paper
# uses </q> as part of normal prose).
_QUOTE_TAG_PATTERN = re.compile(r"<q>(.*?)</q>", re.DOTALL)
_IMAGE_TAG_PATTERN = re.compile(r"<img>(.*?)</img>", re.DOTALL)
# Any <q>/<img> open or close tag, matched in document order so the
# structural check below can enforce proper sequencing, not just balanced
# counts.
_ANY_TAG_RE = re.compile(r"</?(?:q|img)>")


def parse_evidence_string(s):
    """Split a tagged evidence string into (quotes, images, prose).

    `quotes` is the list of inner texts of <q>...</q> blocks (in order).
    `images` is the list of inner texts of <img>...</img> blocks (image
    labels, whitespace-stripped). `prose` is what remains after stripping
    out all tags; the model's reasoning / synthesis text. Either may be
    empty.

    Pure parser; does no validation. Counterpart validation in
    `_check_evidence_tags`.
    """
    if not s:
        return [], [], ""
    quotes = _QUOTE_TAG_PATTERN.findall(s)
    images = [m.strip() for m in _IMAGE_TAG_PATTERN.findall(s)]
    prose = _QUOTE_TAG_PATTERN.sub("", s)
    prose = _IMAGE_TAG_PATTERN.sub("", prose).strip()
    return quotes, images, prose


def _check_evidence_tags(s, field_path):
    """Reject malformed tag structure loudly.

    Scans the <q>/<img> tags in document order and enforces that they form
    a flat, correctly-paired sequence: no nesting, no overlap, no orphan
    closing tag, and no unclosed opening tag. Quotes and images are
    siblings, never nested, so only one tag may be open at a time; opening
    a second tag while one is open, or closing a tag that is not the one
    currently open, is malformed.

    Stricter than a bare open/close count: `<q>a<q>b</q></q>` has matching
    counts yet is malformed, and a count-only check would let the bad
    structure surface downstream as a confusing verbatim mismatch instead
    of a malformed_tags error.
    """
    errors = []
    if s is None:
        return errors

    def _malformed(message):
        errors.append({
            "path": f"{field_path}.evidence",
            "code": "malformed_tags",
            "message": message,
        })

    open_kind = None            # None, "q", or "img": only one open at a time
    for match in _ANY_TAG_RE.finditer(s):
        tag = match.group(0)
        kind = "img" if "img" in tag else "q"
        is_close = tag.startswith("</")
        if not is_close:
            if open_kind is not None:
                _malformed(
                    f"Malformed evidence tags: <{kind}> opens inside an "
                    f"unclosed <{open_kind}> block. Quote and image tags "
                    "must be flat siblings; do not nest or overlap them. "
                    "Close the open block before starting the next."
                )
                return errors
            open_kind = kind
        else:
            if open_kind is None:
                _malformed(
                    f"Malformed evidence tags: </{kind}> has no matching "
                    f"<{kind}> before it. Each closing tag must follow its "
                    "own opening tag; split overlapping quotes into separate "
                    "blocks."
                )
                return errors
            if open_kind != kind:
                _malformed(
                    f"Malformed evidence tags: </{kind}> closes while a "
                    f"<{open_kind}> block is still open. Tags must be flat "
                    f"siblings; close </{open_kind}> before </{kind}>."
                )
                return errors
            open_kind = None

    if open_kind is not None:
        _malformed(
            f"Malformed evidence tags: unclosed <{open_kind}> block. Each "
            f"<{open_kind}> needs a matching </{open_kind}>; split "
            "overlapping quotes into separate blocks."
        )
    return errors


def validate_evidence(evidence, paper_text, image_labels, value,
                      field_path, evidence_required=True,
                      weak_matches=None):
    """Validate one field's unified evidence string.

    Args:
        evidence: str | None, the field's evidence string. May contain
            any combination of <q>...</q> quote blocks, <img>label</img>
            image references, and interpretive prose outside the tags.
        paper_text: full normalised-on-the-fly paper text
        image_labels: set[str] of figure filename stems
            (e.g. {"table_01", "figure_02"})
        value: the field's extracted value (or None)
        field_path: dotted path used in error messages
        evidence_required: True for fields whose YAML spec sets
            `evidence: required` (default); False for `evidence: optional`
            (judgement fields where the value is the extractor's
            reasoning rather than paper content). Required fields need at
            least one <q> or <img> when value is non-null; optional
            fields may carry pure prose, empty string, or None.
        weak_matches: an optional caller-supplied accumulator, on the same
            terms as `validators.validate_envelope`'s `canonicalisations`.
            Every quote that PASSED, but only on the `case_folded` tier, is
            appended as a `{path, quote, tier}` dict. The fold is the one
            tier whose difference is the model's own rather than the
            paper's rendering (see the module docstring); it passes, but
            without this sink nothing downstream ever learns the tier,
            because a passing quote produces no errors.

    Returns a list of {path, code, message} dicts. Empty list = passes.
    """
    if evidence is not None and not isinstance(evidence, str):
        return [{
            "path": f"{field_path}.evidence",
            "code": "type_mismatch",
            "message": (
                f"evidence must be a string or null, got "
                f"{type(evidence).__name__}. Example of correct shape: "
                '{"value": "Cohort study", "evidence": '
                '"<q>verbatim quote from paper</q> brief interpretive note."}'
            ),
        }]

    errors = []

    # Structural tag check runs regardless of value/required; malformed
    # tags are always an error.
    errors.extend(_check_evidence_tags(evidence, field_path))
    if errors:
        return errors

    quotes, images, _prose = parse_evidence_string(evidence or "")

    # Quote verbatim check + image-label resolution always run. Catches
    # fabricated quotes even on optional-evidence or null-value fields.
    for i, quote in enumerate(quotes):
        if not quote.strip():
            errors.append({
                "path": f"{field_path}.evidence[<q>{i}]",
                "code": "empty_quote",
                "message": "Each <q>...</q> block must contain non-empty text.",
            })
            continue
        located = locate_quote(quote, paper_text)
        if located:
            # Recorded only for the fold; see the weak_matches doc above.
            if located.tier == "case_folded" and weak_matches is not None:
                weak_matches.append({
                    "path": f"{field_path}.evidence[<q>{i}]",
                    "quote": quote[:120] + ("..." if len(quote) > 120 else ""),
                    "tier": located.tier,
                })
        else:
            preview = quote[:120] + ("..." if len(quote) > 120 else "")
            labels_hint = ", ".join(sorted(image_labels)) or "(none available)"
            # Show the paper's own wording where there is one worth
            # showing; silence when nothing is close (a wrong suggestion
            # is worse than none).
            nearest = suggest_nearest_text(quote, paper_text)
            if nearest:
                nearest_hint = (
                    f" The closest text in the paper reads: '{nearest}'. "
                    f"Did you mean that?"
                )
            elif _is_out_of_order_elision(quote, paper_text):
                # Every fragment is in the paper, so a suggestion would
                # point at itself; what is wrong is the order.
                nearest_hint = (
                    " Each part of this quote is in the paper, but not in "
                    "the order the quote puts them: the text either side of "
                    "an elision marker must run forwards through one "
                    "passage. Reorder them to follow the paper, or, if they "
                    "come from different passages, quote them as separate "
                    "<q>...</q> blocks."
                )
            else:
                nearest_hint = ""
            errors.append({
                "path": f"{field_path}.evidence[<q>{i}]",
                "code": "quote_not_in_text",
                "message": (
                    f"Quote not found in paper text after normalisation: "
                    f"'{preview}'.{nearest_hint} Copy a verbatim substring "
                    f"including punctuation, or, if the information is in a "
                    f"figure or table, use <img>label</img> with one of the "
                    f"available labels: {labels_hint}."
                ),
            })

    for i, img in enumerate(images):
        if not img.strip():
            errors.append({
                "path": f"{field_path}.evidence[<img>{i}]",
                "code": "empty_image_label",
                "message": "Each <img>...</img> block must contain a label.",
            })
            continue
        if normalise_label(img) not in image_labels:
            labels_hint = ", ".join(sorted(image_labels)) or "(none available)"
            errors.append({
                "path": f"{field_path}.evidence[<img>{i}]",
                "code": "unknown_image_label",
                "message": (
                    f"Image label '{img.strip()}' is not in the available "
                    f"set: {labels_hint}."
                ),
            })

    # Evidence-required gate: applies only when value is non-null. When
    # value is null, the evidence string is optional (often a prose
    # explanation of WHY the field is null), regardless of `evidence:`
    # configuration. When value is non-null AND the field is
    # `evidence: required`, at least one quote or image is demanded;
    # prose alone isn't acceptable for those fields.
    if value is not None and evidence_required and not quotes and not images:
        labels_hint = ", ".join(sorted(image_labels)) or "(none available)"
        errors.append({
            "path": f"{field_path}.evidence",
            "code": "evidence_required",
            "message": (
                "value is set but the evidence string has no <q>...</q> "
                "quote or <img>...</img> reference. Required-evidence "
                "fields need at least one. Available image labels: "
                f"{labels_hint}."
            ),
        })

    return errors
