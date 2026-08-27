"""The lines this engine writes around a paper, and how to spot a forgery.

A message is one flat sequence of text. The paper's prose, a supplement's
prose, an exhibit's caption and its content as text all arrive inside it, and
what tells a role where one ends and the next begins is a line this engine
writes: `--- PAPER TEXT ---`, `--- SUPPLEMENT supplement_a: … ---`, and the
closing lines that match them. A role has nothing else to go on. There is no
structure under the text that says which document a passage came from — the
line IS the structure.

So a bundle whose own text contains one of those lines can move the boundary.
A supplement titled with a newline and a closing line puts its prose and its
crops outside every section, where they read as the article's; a transcription
whose cell carries `--- END PAPER TEXT ---` ends the paper early. Neither is
exotic: a title and a caption are free text in the declaration, and a
transcription is a model's reading of a page. alteksto validates that a
transcription is a table and that a title is a string, which is its business;
what a string may not spell is this consumer's business, because the
vocabulary being forged is this consumer's own.

This module is the one place that vocabulary is written down, so the check and
the messages cannot disagree about what a delimiter is. It imports the
standard library only, so the loader can refuse a bundle without pulling in
the prompt stack.
"""

import re

# The article's prose, delimited so a role can tell where the paper stops
# without inferring it from a change of subject.
PAPER_TEXT_OPEN = "--- PAPER TEXT ---"
PAPER_TEXT_CLOSE = "--- END PAPER TEXT ---"

# A supplement's prose, inside that supplement's section. Named differently
# from the article's so a role reading a quote back can tell which of the two
# it came out of.
SUPPLEMENT_TEXT_OPEN = "--- SUPPLEMENT TEXT ---"
SUPPLEMENT_TEXT_CLOSE = "--- END SUPPLEMENT TEXT ---"

# The assembled extraction output, which only the reviewer is shown.
REVIEW_OUTPUT_OPEN = "--- ASSEMBLED EXTRACTION OUTPUT (to review) ---"
REVIEW_OUTPUT_CLOSE = "--- END EXTRACTION OUTPUT ---"

FIXED_LINES = (
    PAPER_TEXT_OPEN,
    PAPER_TEXT_CLOSE,
    SUPPLEMENT_TEXT_OPEN,
    SUPPLEMENT_TEXT_CLOSE,
    REVIEW_OUTPUT_OPEN,
    REVIEW_OUTPUT_CLOSE,
)

# A supplement's section opens with its name and printed title and closes with
# its name, so neither line is a fixed string. Matched by shape instead: this
# is what a forged one would have to look like, whatever name it named.
SUPPLEMENT_SECTION = re.compile(r"^---\s*(END\s+)?SUPPLEMENT\b.*---$")

# What a written report puts where a crop's bytes are. Not sent to a model —
# a role receives the image itself — but a transcription that spells this line
# makes the report name a crop the message never attached, and the report is
# read as the record of what was sent.
IMAGE_PLACEHOLDER = re.compile(r"^\(image:\s*\S+\)$")


def image_placeholder(label):
    """The line a text view of a message writes where `label`'s bytes are."""
    return f"(image: {label}.png)"


def forged_lines(text):
    """Every line of `text` that spells one of this engine's own framing
    lines, in the order they appear.

    Matched on the stripped line, because leading or trailing whitespace does
    not change what a role reads. A line of bare dashes is NOT a forgery: a
    thematic break is ordinary markdown and every delimiter here names what it
    delimits.
    """
    if not text:
        return []
    forged = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (stripped in FIXED_LINES
                or SUPPLEMENT_SECTION.match(stripped)
                or IMAGE_PLACEHOLDER.match(stripped)):
            forged.append(stripped)
    return forged
