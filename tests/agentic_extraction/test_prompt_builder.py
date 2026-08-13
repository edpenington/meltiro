"""Smoke tests for the extractor prompt builder."""

import re

import pytest

from meltiro.errors import ConfigBundleError
from meltiro.prompt_builder import (
    build_initial_user_blocks,
    build_review_system_message,
    build_review_user_blocks,
    build_system_message,
    render_user_prompt_text,
    system_message_blocks,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

CAPTIONS = {
    "table_01": "Table 1. Unit characteristics",
    "figure_01": "Figure 1. Study flow",
}


def test_system_message_includes_key_sections(synthetic_template,
                                               extractor_system_path):
    # The field catalogue lives in the tool input_schemas (each property's
    # description carries the field's description + allowed values), NOT in
    # the system message, which names no specific field at all. What the
    # system message DOES carry is: the workflow (with a GENERIC tool-call
    # budget and the interpolated per-field check budget) and the reference
    # list substituted for the {reference:gauge_list} placeholder.
    txt = build_system_message(
        system_prompt_path=extractor_system_path,
        max_checks_per_field=3,
        reference_lists={"gauge_list": [
            {"tool_name": "WDS-9"}, {"tool_name": "CRT-HD"}]},
    )
    # The tool-call budget is described generically, with NO number: the cap is
    # an operational bound, kept out of the prompt (and so out of prompt_hash
    # and config identity) so raising it on resume cannot change provenance.
    assert "tool-call budget" in txt
    assert "50 tool calls" not in txt
    # The per-field check budget is still interpolated (a genuine structure
    # signal), and it reads as a count of checks per field, not of rounds.
    assert "check budget for this run is 3" in txt
    # Whole word. The prompt legitimately says "surrounding text" and "the
    # ground on which", and a bare substring test reads both of those as the
    # check budget being described in rounds.
    assert re.search(r"\brounds?\b", txt, re.IGNORECASE) is None
    # Reference list rendered.
    assert "WDS-9" in txt
    assert "CRT-HD" in txt


def test_the_system_message_carries_nothing_of_the_paper(
        synthetic_template, extractor_system_path, review_system_path):
    # Both system messages are one string per config: the exhibits a study
    # supplies arrive labelled in the user message, so neither text varies
    # from paper to paper and both cache across a whole review.
    for build, path in ((build_system_message, extractor_system_path),
                        (build_review_system_message, review_system_path)):
        txt = build(system_prompt_path=path,
                    reference_lists={"gauge_list": []})
        for label in CAPTIONS:
            assert label not in txt


def test_each_attached_exhibit_is_labelled_with_its_caption():
    # A bare `table_01` says nothing about what the image holds, so the
    # bundle's declared caption follows the label in the block that introduces
    # the attachment. The label stays first and stays alone in the brackets,
    # because it is what an <img>label</img> citation carries.
    blocks = build_initial_user_blocks(
        "376", "Methods.",
        figures=[("table_01", PNG), ("figure_01", PNG)],
        image_captions=CAPTIONS,
    )
    texts = [b.get("text") for b in blocks if b["type"] == "text"]
    assert "[table_01] Table 1. Unit characteristics" in texts
    assert "[figure_01] Figure 1. Study flow" in texts


def test_the_reviewer_reads_the_same_labels_and_captions():
    blocks = build_review_user_blocks(
        "376", "Methods.", [("table_01", PNG)], {"study": {}},
        CAPTIONS,
    )
    texts = [b.get("text") for b in blocks if b["type"] == "text"]
    assert "[table_01] Table 1. Unit characteristics" in texts


def test_a_label_with_no_caption_arrives_bare():
    # An exhibits manifest that declares no caption for a crop, or a caller
    # with no caption map at all: the label arrives alone, with no trailing
    # space.
    for captions in (None, {"figure_01": "Figure 1. Study flow"}):
        blocks = build_initial_user_blocks(
            "376", "Methods.", figures=[("table_01", PNG)],
            image_captions=captions)
        texts = [b.get("text") for b in blocks if b["type"] == "text"]
        assert "[table_01]" in texts

    blocks = build_review_user_blocks(
        "376", "Methods.", [("table_01", PNG)], {"study": {}})
    texts = [b.get("text") for b in blocks if b["type"] == "text"]
    assert "[table_01]" in texts


def test_the_captured_user_prompt_mirrors_the_message():
    # `render_user_prompt_text` is what the session records as "the user
    # prompt", so its text blocks are the message's own, captions included.
    text = render_user_prompt_text(
        "376", "Methods.", ["table_01", "figure_01"], CAPTIONS)
    assert "[table_01] Table 1. Unit characteristics" in text
    assert "[figure_01] Figure 1. Study flow" in text


def test_system_message_blocks_carries_cache_control(synthetic_template,
                                                     extractor_system_path):
    txt = build_system_message(
        reference_lists={"gauge_list": []},
        system_prompt_path=extractor_system_path,
    )
    blocks = system_message_blocks(txt)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_unresolvable_reference_placeholder_fails_loudly(
        synthetic_template, extractor_system_path):
    # The shipped prompt cites {reference:gauge_list}; if the config bundle
    # does not provide that list, rendering must fail loudly rather than
    # ship a prompt with a dangling placeholder.
    with pytest.raises(ConfigBundleError) as excinfo:
        build_system_message(
            reference_lists={},
            system_prompt_path=extractor_system_path,
        )
    assert "gauge_list" in str(excinfo.value)


def test_initial_user_blocks_structure():
    blocks = build_initial_user_blocks(
        "376",
        "Methods. WDS-9 administered.",
        figures=[
            ("table_01", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
            ("figure_01", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
        ],
    )
    # Header + paper text + (label + image) per figure.
    assert blocks[0]["type"] == "text"
    assert "study 376" in blocks[0]["text"]
    assert "#" not in blocks[0]["text"]  # neutral wording, no markdown heading
    assert "WDS-9 administered" in blocks[1]["text"]
    # Labels precede images.
    assert blocks[2]["type"] == "text" and "table_01" in blocks[2]["text"]
    assert blocks[3]["type"] == "image"
    assert blocks[4]["type"] == "text" and "figure_01" in blocks[4]["text"]
    assert blocks[5]["type"] == "image"
    # Last block carries cache_control.
    assert blocks[-1].get("cache_control") == {"type": "ephemeral"}


def test_initial_user_blocks_no_figures():
    blocks = build_initial_user_blocks(
        "376", "Methods. WDS-9 administered.", figures=[],
    )
    # Just header + paper text.
    assert len(blocks) == 2
    assert blocks[-1].get("cache_control") == {"type": "ephemeral"}


def _section_text(haystack, heading):
    """Pull out the text under a `## heading` for assertion purposes."""
    needle = "## " + heading
    if needle not in haystack:
        return ""
    after = haystack.split(needle, 1)[1]
    # Stop at next `## ` heading.
    return after.split("\n## ", 1)[0]
