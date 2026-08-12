"""Tests for the final-review prompt assembly."""

from meltiro.prompt_builder import (
    build_review_system_message,
    build_review_user_blocks,
)


def test_review_system_message_renders(synthetic_template, review_system_path):
    txt = build_review_system_message(
        image_labels={"table_01", "figure_01"},
        system_prompt_path=review_system_path,
        reference_lists={"gauge_list": [{"tool_name": "WDS-9"}]},
    )
    # Role framing.
    assert "final reviewer" in txt.lower() or "reviewer" in txt.lower()
    # Image labels are mentioned so the reviewer knows what to cite.
    # (The field catalogue lives in the extraction output JSON the reviewer
    # reads as part of the user message, not in the system prompt;
    # mirrors the extractor's tool-schema-based approach.)
    assert "table_01" in txt
    assert "figure_01" in txt


def test_review_user_blocks_includes_paper_images_and_extraction_record():
    paper = "Methods: WDS-9 administered."
    figures = [
        ("table_01", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
        ("figure_01", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
    ]
    extraction_record = {
        "initial_check": {"text_readable": True},
        "study": {"primary_aim": {"value": "Assess X", "evidence": ["q"],
                                  "source": "Abstract"}},
        "records": [
            {"record_id": "relationship_1",
             "gauge": {"value": "WDS-9", "evidence": ["q"], "source": "M"}},
        ],
        "quality_check": {},
    }
    blocks = build_review_user_blocks("376", paper, figures, extraction_record)
    # Header mentions reviewing the study (neutral wording, no "#").
    assert "study 376" in blocks[0]["text"]
    # Paper text included.
    assert any("WDS-9 administered" in b.get("text", "") for b in blocks)
    # Two image blocks.
    assert sum(1 for b in blocks if b.get("type") == "image") == 2
    # Extraction output JSON included.
    assert any("Assess X" in b.get("text", "") for b in blocks)
    assert any('"relationship_1"' in b.get("text", "") for b in blocks)
    # Last block carries cache_control.
    assert blocks[-1].get("cache_control") == {"type": "ephemeral"}


def test_review_user_blocks_with_no_figures():
    extraction_record = {"initial_check": {}, "study": {}, "records": [],
                "quality_check": {}}
    blocks = build_review_user_blocks(
        "376", "Methods.", figures=[], extraction_record_dict=extraction_record)
    # Header + paper + extraction output = 3 blocks; no images.
    assert sum(1 for b in blocks if b.get("type") == "image") == 0
    assert blocks[-1].get("cache_control") == {"type": "ephemeral"}
