"""Smoke tests for the extractor prompt builder.

The builders own one contract beyond what they each render: the text-only
render a session captures carries the exact strings the message carries. Both
sides are here, and so is the orchestrator call site that has to feed the
capture the message's own figure sequence for the contract to mean anything.
"""

import json
import re
import shutil

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import ConfigBundleError
from meltiro.orchestrator import Orchestrator
from meltiro.prompt_builder import (
    NO_EXHIBITS_NOTICE,
    build_initial_user_blocks,
    build_review_system_message,
    build_review_user_blocks,
    build_system_message,
    system_message_blocks,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

CAPTIONS = {
    "table_01": "Table 1. Unit characteristics",
    "figure_01": "Figure 1. Study flow",
}

# One of the two exhibits prints a footnote and the other does not, which is
# the ordinary case: the maps are separate because only some exhibits have one.
NOTES = {
    "table_01": "SD, standard deviation. Percentages are of the 588 units "
                "carried through both rounds.",
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


def test_the_captured_user_prompt_mirrors_the_message(rendered_user_message):
    # `render_user_prompt_text` is what the session records as "the user
    # prompt", so its text blocks are the message's own, captions included.
    text = rendered_user_message(
        "376", "Methods.", ["table_01", "figure_01"], CAPTIONS)
    assert "[table_01] Table 1. Unit characteristics" in text
    assert "[figure_01] Figure 1. Study flow" in text


class TestAnExhibitsPrintedFootnote:
    """A crop's printed footnote reaches the model as text, under its label.

    It is the smallest print on the exhibit and the crop carries it as pixels;
    `text.md` carries it nowhere, which is also why the engine prompts tell a
    model to cite what it reads there as `<img>label</img>` rather than quote
    it. The line follows the caption rather than replacing it, because the two
    say different things about the same crop.
    """

    def _texts(self, blocks):
        return [b.get("text") for b in blocks if b["type"] == "text"]

    def test_it_follows_the_caption_under_the_same_label(self):
        blocks = build_initial_user_blocks(
            "376", "Methods.", figures=[("table_01", PNG)],
            image_captions=CAPTIONS, image_notes=NOTES)
        assert ("[table_01] Table 1. Unit characteristics\n"
                "Footnote: SD, standard deviation. Percentages are of the "
                "588 units carried through both rounds.") in self._texts(
                    blocks)

    def test_the_reviewer_reads_the_same_line(self):
        blocks = build_review_user_blocks(
            "376", "Methods.", [("table_01", PNG)], {"study": {}},
            CAPTIONS, NOTES)
        assert any(t.startswith("[table_01] Table 1.") and "Footnote: SD" in t
                   for t in self._texts(blocks))

    def test_the_captured_prompt_mirrors_it(self, rendered_user_message):
        # The session's record of the message, so the footnote is in the
        # record exactly as it went on the wire.
        text = rendered_user_message(
            "376", "Methods.", ["table_01", "figure_01"], CAPTIONS, NOTES)
        assert "Footnote: SD, standard deviation." in text

    def test_an_exhibit_that_prints_none_carries_no_footnote_line(self):
        # The absence is silent: no empty `Footnote:` under a crop whose paper
        # printed nothing under it, and none at all for a caller with no map.
        blocks = build_initial_user_blocks(
            "376", "Methods.",
            figures=[("figure_01", PNG), ("table_01", PNG)],
            image_captions=CAPTIONS, image_notes=NOTES)
        texts = self._texts(blocks)
        assert "[figure_01] Figure 1. Study flow" in texts
        assert sum("Footnote:" in t for t in texts) == 1

        bare = build_initial_user_blocks(
            "376", "Methods.", figures=[("table_01", PNG)],
            image_captions=CAPTIONS)
        assert "[table_01] Table 1. Unit characteristics" in self._texts(bare)

    def test_a_footnote_on_a_crop_with_no_caption_still_arrives(self):
        # The two are independent: a manifest may record a footnote for an
        # exhibit whose caption the paper never printed.
        blocks = build_initial_user_blocks(
            "376", "Methods.", figures=[("table_01", PNG)],
            image_notes=NOTES)
        assert any(t.startswith("[table_01]\nFootnote: SD")
                   for t in self._texts(blocks))


# A figure list that is neither alphabetical nor lower-cased, so a capture
# built from a sorted, normalised label set differs from the message in BOTH
# ways at once and either divergence fails the comparisons below.
MIXED_CASE_FIGURES = [("Table_02", PNG), ("figure_01", PNG), ("TABLE_01", PNG)]
MIXED_CASE_CAPTIONS = {
    "table_02": "Table 2. Removals by subgroup",
    "figure_01": "Figure 1. Study flow",
    "table_01": "Table 1. Unit characteristics",
}


def _capture_the_message_implies(blocks, labels):
    """The capture a message of `blocks` has to produce, built from `blocks`.

    The message's own text blocks in the message's own order, with the
    `(image: LABEL.png)` stand-in written where each attachment's bytes are.
    Constructed rather than re-rendered, so a comparison against it is a
    comparison with the message and not with a second call of the function
    under test.
    """
    texts = [b["text"] for b in blocks if b["type"] == "text"]
    lead = len(texts) - len(labels)
    parts = texts[:lead]
    for label, text in zip(labels, texts[lead:]):
        parts += [text, f"(image: {label}.png)"]
    return "\n\n".join(parts)


def test_the_capture_and_the_message_emit_the_same_strings(rendered_user_message):
    # The function's own contract: "the exact text strings match those
    # `build_initial_user_blocks` emits as text content blocks". A capture
    # built from the dispatcher's normalised label set would satisfy every
    # substring assertion above and still record `[table_01]` where the
    # message sent `[TABLE_01]`, in an order the message never used — so the
    # whole emitted text is compared, not membership of it.
    labels = [label for label, _ in MIXED_CASE_FIGURES]
    blocks = build_initial_user_blocks(
        "376", "Methods.", MIXED_CASE_FIGURES, MIXED_CASE_CAPTIONS)
    assert rendered_user_message(
        "376", "Methods.", labels, MIXED_CASE_CAPTIONS) == \
        _capture_the_message_implies(blocks, labels)


def _mixed_case_bundle(tmp_path, source_dir):
    """A copy of a real bundle whose crops are neither lower-cased nor
    alphabetical in the order the message attaches them.

    `PaperBundle.figures` is sorted by the label the filesystem holds, so
    `TABLE_01` leads and `figure_01` follows, while the dispatcher's
    normalised set sorts the other way round and in another case. A capture
    built from that set therefore diverges from the message on BOTH axes at
    once, and a comparison over this bundle catches either.
    """
    root = tmp_path / "mixed"
    shutil.copytree(source_dir, root)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8"))
    original = manifest["exhibits"][0]
    png = (root / "figures" / f"{original['label']}.png").read_bytes()
    (root / "figures" / f"{original['label']}.png").unlink()
    manifest["exhibits"] = []
    for label in ("TABLE_01", "figure_01"):
        (root / "figures" / f"{label}.png").write_bytes(png)
        entry = {"label": label,
                 "caption": f"Exhibit {label}. Depot readings"}
        if label == "TABLE_01":
            # One of the two prints a footnote, and it is the capitalised one:
            # the maps this is looked up in are keyed on the normalised label,
            # so an unnormalised map loses the footnote here and nowhere else.
            entry["notes"] = "Note. Readings are depot means over the round."
        manifest["exhibits"].append(entry)
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return root


def test_the_orchestrator_captures_the_prompt_it_sends(
        tmp_path, config_dir, bundle_tables_dir, stage_keys):
    # The call site the contract above rests on. `_render_user_prompt_text`
    # has two label sources to choose between and they disagree on case and
    # order, so a capture built from the wrong one records a prompt naming the
    # same exhibits in a form the message never used — and a reader auditing
    # what an `<img>` citation could legally name reads the wrong strings.
    bundle_dir = _mixed_case_bundle(tmp_path, bundle_tables_dir)
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir),
        tmp_path / "runs",
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=1024,
                                     checker_model="claude-sonnet-4-6"),
        review_model="claude-opus-4-8",
        extractor_max_tokens=4096, review_max_tokens=4096,
    )
    orch.prepare_new_session()

    labels = list(load_bundle(bundle_dir).figures)
    captured = (orch.session.instrument_dir / "user_prompt.txt").read_text(
        encoding="utf-8")
    assert captured == _capture_the_message_implies(
        orch.messages[0]["content"], labels)
    # The bundle really is mixed-case and out of normalised order, or the
    # equality above would hold for a capture that agreed by accident.
    assert labels == ["TABLE_01", "figure_01"]
    assert "[TABLE_01] Exhibit TABLE_01. Depot readings" in captured
    # And the footnote the manifest records for that exhibit is in both, which
    # the equality above cannot show on its own: a message and a capture that
    # both lost it agree with each other.
    assert "Footnote: Note. Readings are depot means" in captured


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
    # Header + paper text + the statement that none accompany the study.
    assert len(blocks) == 3
    assert blocks[-1]["text"] == NO_EXHIBITS_NOTICE
    assert blocks[-1].get("cache_control") == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# What the message says when nothing is attached
# ---------------------------------------------------------------------------

class TestTheMessageStatesWhenNoExhibitsAccompanyTheStudy:
    """Exhibit presence is the message's fact to state, on both stages.

    A system prompt is one string per config, so it cannot know whether this
    study ships crops or whether this role's model can accept one. The message
    can: it carries the labelled attachments, or it carries the statement that
    there are none. Neither role is left inferring the absence from a message
    that simply stops after the paper text.
    """

    def test_the_extractors_message_carries_it(self):
        blocks = build_initial_user_blocks("376", "Methods.", figures=[])
        assert NO_EXHIBITS_NOTICE in [
            b.get("text") for b in blocks if b["type"] == "text"]

    def test_the_reviewers_message_carries_it(self):
        blocks = build_review_user_blocks(
            "376", "Methods.", [], {"study": {}}, CAPTIONS)
        assert NO_EXHIBITS_NOTICE in [
            b.get("text") for b in blocks if b["type"] == "text"]

    def test_the_captured_prompt_mirrors_it(self, rendered_user_message):
        # `render_user_prompt_text` is the session's record of the message, so
        # the statement has to be in the record exactly as it is on the wire.
        text = rendered_user_message("376", "Methods.", [], CAPTIONS)
        assert NO_EXHIBITS_NOTICE in text

    @pytest.mark.parametrize("build,extra", [
        (build_initial_user_blocks, ()),
        (build_review_user_blocks, ({"study": {}},)),
    ], ids=["extractor", "reviewer"])
    def test_a_message_that_does_carry_exhibits_does_not(self, build, extra):
        # The other half of the pair: the statement is the empty case's, not a
        # standing preamble, so a study with a crop reads its label instead.
        blocks = build("376", "Methods.", [("table_01", PNG)], *extra,
                       CAPTIONS)
        texts = [b.get("text") for b in blocks if b["type"] == "text"]
        assert NO_EXHIBITS_NOTICE not in texts
        assert "[table_01] Table 1. Unit characteristics" in texts


def _section_text(haystack, heading):
    """Pull out the text under a `## heading` for assertion purposes."""
    needle = "## " + heading
    if needle not in haystack:
        return ""
    after = haystack.split(needle, 1)[1]
    # Stop at next `## ` heading.
    return after.split("\n## ", 1)[0]


def _footnote_lines(blocks):
    """Every footnote line in a message's content blocks."""
    return [line
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
            for line in block.get("text", "").split("\n")
            if line.startswith("Footnote: ")]


def test_a_resumed_session_rebuilds_the_message_it_paused_on(
        tmp_path, config_dir, bundle_tables_dir, stage_keys):
    """The resume path builds the initial message a second time.

    It is a separate call site from the one that opens a session, and the
    conversation it rebuilds is prepended to every replayed turn — so a
    resumed run that dropped part of the message would carry the extractor
    through the rest of its work on a paper described differently from the
    one it started on.
    """
    bundle_dir = _mixed_case_bundle(tmp_path, bundle_tables_dir)
    out = tmp_path / "runs"

    def _orch():
        return Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
        )

    first = _orch()
    first.prepare_new_session()
    opened = _footnote_lines(first.initial_user_blocks)
    assert opened == ["Footnote: Note. Readings are depot means over the "
                      "round."]

    resumed = _orch()
    resumed.resume_session(first.session.session_dir)
    assert _footnote_lines(resumed.initial_user_blocks) == opened
    assert resumed.messages[0]["content"] == first.initial_user_blocks


def test_the_reviewer_is_shown_the_exhibits_the_extractor_was(
        tmp_path, config_dir, bundle_tables_dir, stage_keys, monkeypatch):
    """The review stage builds its own user message, from its own call site.

    The reviewer reads the paper and the crops to form an independent view,
    so an exhibit that reaches the extractor labelled and footnoted and the
    reviewer bare would leave the two stages looking at the same image with
    different amounts of it legible.
    """
    from meltiro import orchestrator as orch_mod

    bundle_dir = _mixed_case_bundle(tmp_path, bundle_tables_dir)
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir),
        tmp_path / "runs",
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=1024,
                                     checker_model="claude-sonnet-4-6"),
        review_model="claude-opus-4-8",
        extractor_max_tokens=4096, review_max_tokens=4096,
    )
    orch.prepare_new_session()

    # Spy rather than stub: the real builder still runs, so what is asserted
    # is the message the reviewer would actually be sent.
    built = {}
    real = orch_mod.build_review_user_blocks

    def _spy(*args, **kwargs):
        blocks = real(*args, **kwargs)
        built["blocks"] = blocks
        return blocks

    monkeypatch.setattr(orch_mod, "build_review_user_blocks", _spy)
    # The review loop needs an adapter and a model reply; neither is the
    # subject here, so the loop is cut short after the message is built.
    monkeypatch.setattr(orch_mod, "get_tool_definitions",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    orch._adapter_for_role = lambda role: object()
    with pytest.raises(RuntimeError):
        orch._final_review()

    assert _footnote_lines(built["blocks"]) == _footnote_lines(
        orch.initial_user_blocks)


def test_the_session_records_the_exhibits_the_message_carried(
        tmp_path, config_dir, bundle_tables_dir, stage_keys):
    """`instrument/image_labels.json` is the durable copy of the manifest's
    wording, so it is written from the same normalised maps the message is.

    The capitalised label is what makes this decidable: those maps are keyed
    on the normalised label, so a record that looked a label up as the
    manifest spells it would write `null` for an exhibit that printed a
    footnote, and the transcript would report "none printed" for it.
    """
    bundle_dir = _mixed_case_bundle(tmp_path, bundle_tables_dir)
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir),
        tmp_path / "runs",
        extractor_model="claude-opus-4-8",
        checker_config=CheckerConfig(max_tokens=1024,
                                     checker_model="claude-sonnet-4-6"),
        review_model="claude-opus-4-8",
        extractor_max_tokens=4096, review_max_tokens=4096,
    )
    orch.prepare_new_session()

    recorded = json.loads(
        (orch.session.instrument_dir / "image_labels.json").read_text(
            encoding="utf-8"))
    assert recorded == [
        {"label": "TABLE_01",
         "caption": "Exhibit TABLE_01. Depot readings",
         "notes": "Note. Readings are depot means over the round.",
         # This bundle transcribes neither exhibit, so both record the
         # absence. It is a fact about the MESSAGE, on the same terms as the
         # caption beside it: what the model was shown, not what the bundle
         # could have supplied.
         "transcribed": False},
        {"label": "figure_01",
         "caption": "Exhibit figure_01. Depot readings",
         "notes": None,
         "transcribed": False},
    ]
