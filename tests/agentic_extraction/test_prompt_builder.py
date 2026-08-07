"""Smoke tests for the extractor prompt builder."""

import re

import pytest

from meltiro.errors import ConfigBundleError
from meltiro.prompt_builder import (
    build_initial_user_blocks,
    build_review_system_message,
    build_system_message,
    system_message_blocks,
)


def test_system_message_includes_key_sections(synthetic_template,
                                               extractor_system_path):
    # The field catalogue lives in the tool input_schemas (each property's
    # description carries the field's description + allowed values), NOT in
    # the system message, which names no specific field at all. What the
    # system message DOES carry is: the workflow (with a GENERIC tool-call
    # budget and the interpolated per-field check budget), image labels, and
    # the reference list substituted for the {reference:gauge_list}
    # placeholder.
    txt = build_system_message(
        synthetic_template,
        image_labels={"table_01", "figure_01"},
        system_prompt_path=extractor_system_path,
        max_checks_per_field=3,
        reference_lists={"gauge_list": [
            {"tool_name": "WDS-9"}, {"tool_name": "CRT-HD"}]},
    )
    # Image labels rendered.
    assert "table_01" in txt
    assert "figure_01" in txt
    # The tool-call budget is described generically, with NO number: the cap is
    # an operational bound, kept out of the prompt (and so out of prompt_hash
    # and config identity) so raising it on resume cannot change provenance.
    assert "tool-call budget" in txt
    assert "50 tool calls" not in txt
    # The per-field check budget is still interpolated (a genuine structure
    # signal), and it reads as a count of checks per field, not of rounds.
    assert "checked at most 3 times" in txt
    # Whole word. The prompt legitimately says "surrounding text" and "the
    # ground on which", and a bare substring test reads both of those as the
    # check budget being described in rounds.
    assert re.search(r"\brounds?\b", txt, re.IGNORECASE) is None
    # Reference list rendered.
    assert "WDS-9" in txt
    assert "CRT-HD" in txt


def test_image_labels_render_with_their_captions(synthetic_template,
                                                  extractor_system_path):
    # A bare `table_01` says nothing about what the image holds. The bundle's
    # declared caption is rendered beside it, with the label kept alone in
    # backticks because it is what an <img>label</img> citation carries.
    txt = build_system_message(
        synthetic_template,
        image_labels={"table_01", "figure_01"},
        system_prompt_path=extractor_system_path,
        reference_lists={"gauge_list": []},
        image_captions={
            "table_01": "Table 1. Unit characteristics",
            "figure_01": "Figure 1. Study flow",
        },
    )
    assert "- `table_01`: Table 1. Unit characteristics" in txt
    assert "- `figure_01`: Figure 1. Study flow" in txt


def test_reviewer_sees_the_same_captions(synthetic_template,
                                          review_system_path):
    txt = build_review_system_message(
        synthetic_template,
        image_labels={"table_01"},
        system_prompt_path=review_system_path,
        reference_lists={"gauge_list": []},
        image_captions={"table_01": "Table 1. Unit characteristics"},
    )
    assert "- `table_01`: Table 1. Unit characteristics" in txt


def test_label_without_a_caption_renders_bare(synthetic_template,
                                               extractor_system_path):
    # No caption map at all (a fingerprint render, or a caller with no
    # bundle): each label renders bare, with no trailing colon.
    txt = build_system_message(
        synthetic_template,
        image_labels={"table_01"},
        system_prompt_path=extractor_system_path,
        reference_lists={"gauge_list": []},
    )
    assert "- `table_01`\n" in txt
    assert "- `table_01`:" not in txt


def test_system_message_blocks_carries_cache_control(synthetic_template,
                                                     extractor_system_path):
    txt = build_system_message(
        synthetic_template,
        image_labels=set(), reference_lists={"gauge_list": []},
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
            synthetic_template,
            image_labels=set(), reference_lists={},
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
