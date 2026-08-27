"""Declared image capability per role.

A role whose model is marked `supports_images=False` is sent no image content
blocks and none of the label blocks that introduce them, validates `<img>`
citations against an empty label set, and records the omission loudly (a
per-role meta.images_omitted flag, a run-start stderr warning, and a
fingerprint that moves with the flag). Every test here is offline: no network,
no API key, no live client.

No text-only model exists in the shared registry: EVERY entry (Claude, GPT,
and the routed GLM/Qwen vision slugs) supports images. So these tests inject a
synthetic text-only entry (`TEXT_ONLY_MODEL`) via an autouse monkeypatch
fixture, hermetic and shaped like a text-only GLM chat endpoint
(OpenAI-compatible, Chat Completions, `supports_images=False`).
"""

import dataclasses
import json
import shutil
from types import SimpleNamespace

import pytest
from alteksto.bundle import SCHEMA_VERSION

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import AgenticExtractionError
from meltiro.extraction_record import ExtractionRecord
from meltiro.fingerprint import structure_hash
from meltiro.instrument import Instrument
from meltiro.template import load_template
from meltiro import orchestrator as orch_mod
from meltiro.orchestrator import Orchestrator
from meltiro.prompt_builder import NO_EXHIBITS_NOTICE, image_label_text
from direktoro.registry import (
    MODEL_REGISTRY, Model, PROVIDER_OPENAI, WIRE_CHAT_COMPLETIONS,
    known_models, model_info, model_supports_images)
from meltiro.tools import ToolDispatcher


# A synthetic text-only model, injected into the registry by the autouse
# fixture below. Direct (unrouted) OpenAI-compatible Chat Completions entry with
# supports_images=False, mirroring a text-only OpenAI-compatible endpoint
# reached at its own base URL. Hermetic:
# never called, never keyed, only its registry metadata is read.
TEXT_ONLY_MODEL = "synthetic-text-only"
_SYNTHETIC_TEXT_ONLY = Model(
    PROVIDER_OPENAI, "https://synthetic.invalid/v1", "OPENAI_API_KEY",
    wire_api=WIRE_CHAT_COMPLETIONS, supports_images=False,
    forced_tool_choice=True)


@pytest.fixture(autouse=True)
def _register_text_only(monkeypatch):
    """Inject the synthetic text-only model for the duration of each test."""
    monkeypatch.setitem(MODEL_REGISTRY, TEXT_ONLY_MODEL, _SYNTHETIC_TEXT_ONLY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orch(config_dir, bundle_dir, out_dir, *, extractor_model,
          checker_model="claude-sonnet-4-6", review_model="claude-opus-4-7",
          max_checks_per_field=3, final_review=True, dry_run=False):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    return Orchestrator(
        config, bundle, out_dir,
        extractor_model=extractor_model,
        checker_config=CheckerConfig(
            max_tokens=1024, checker_model=checker_model),
        review_model=review_model,
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        extractor_max_tokens=4096,
        review_max_tokens=4096,
        dry_run=dry_run,
    )


def _instrument(config_dir, **kwargs):
    """The instrument alone, with no run around it.

    The two fingerprint claims below are about an argument, not about a
    configuration that could be run: a text-only model has no session to read
    a fingerprint out of, because the run is refused first.
    """
    config = load_config_bundle(config_dir)
    return Instrument(
        config, load_template(config.template_path), config.reference_lists,
        max_checks_per_field=kwargs.get("max_checks_per_field", 3),
        final_review=kwargs.get("final_review", True),
        check_reviewer_edits=kwargs.get("check_reviewer_edits", False),
    )


def _no_figures_bundle(tmp_path):
    """Write a minimal bundle (manifest + text, no figures/ dir) and return
    its dir. A no-figures bundle withholds nothing from any model, so the
    capability guard must be a no-op over it."""
    root = tmp_path / "no_figs"
    root.mkdir()
    # `exhibits: []` is the manifest's explicit assertion that the paper has
    # no tables and no figures, which is what makes this a no-figures bundle
    # rather than a bundle whose crops were forgotten.
    (root / "manifest.json").write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "id": "nofig-001",
            "title": "T",
            "exhibits": [],
            "summary": "A synthetic study used only to exercise the suite.",
        }),
        encoding="utf-8")
    (root / "text.md").write_text("Methods. N=10.", encoding="utf-8")
    return root


def _canonical_image_blocks(messages):
    out = []
    for m in messages:
        for block in m.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "image":
                out.append(block)
    return out


def _message_text(messages):
    return "\n".join(
        block.get("text", "")
        for m in messages
        for block in (m.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text")


# ---------------------------------------------------------------------------
# Registry flag + helper
# ---------------------------------------------------------------------------

class TestRegistryFlag:
    def test_synthetic_model_is_text_only(self):
        assert model_supports_images(TEXT_ONLY_MODEL) is False

    def test_only_the_synthetic_model_is_text_only(self):
        # EVERY model in the shared registry supports images; the only
        # text-only id in view is the synthetic one this module injects.
        for mid in known_models():
            expected = mid != TEXT_ONLY_MODEL
            assert model_info(mid).supports_images is expected, mid
            assert model_supports_images(mid) is expected, mid

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            model_supports_images("no-such-model")


# ---------------------------------------------------------------------------
# Fingerprint folding (the capability enters the stage fingerprints)
# ---------------------------------------------------------------------------

class TestFingerprintFolding:
    def test_true_is_byte_identical_to_omitting_the_arg(self):
        # The image-capable path adds no term: passing the default is
        # byte-identical to omitting the argument, so ONLY a text-only model
        # moves a fingerprint.
        assert structure_hash(3) == structure_hash(3, supports_images=True)

    def test_text_only_appends_noimages_and_moves_the_hash(self):
        on = structure_hash(3, supports_images=True)
        off = structure_hash(3, supports_images=False)
        assert "_noimages" not in on
        assert off.endswith("_noimages")
        assert on != off

    def test_capability_flag_moves_config_fp(self, config_dir):
        # Same model id, prompts, template, tools, decoding, provider: only the
        # registry capability flag differs. config_fp must move, proving the
        # flag itself (not just the model id) folds into the fingerprint, so a
        # registry edit that turns a text-only model into a vision one can
        # never happen silently.
        #
        # Asserted on the instrument rather than through a run, because a run
        # configured with a text-only model is refused before it has a session
        # to read a fingerprint out of (see TestATextOnlyModelIsRefused). The
        # flag stays in the recipe for the runs already recorded under it: a
        # consumer holding a config_fp from one still resolves it here.
        instrument = _instrument(config_dir)
        shared = dict(prompt_hash=instrument.extractor_prompt_hash(),
                      tool_hash=instrument.tool_set_hash())
        identity = ("claude-opus-4-7", {"max_tokens": 4096})
        assert instrument.extractor_fingerprint(
            identity, supports_images=False, **shared) != \
            instrument.extractor_fingerprint(
                identity, supports_images=True, **shared)


# ---------------------------------------------------------------------------
# Extractor: no image parts sent, none-available prompt, meta + warning
# ---------------------------------------------------------------------------

class TestATextOnlyModelIsRefused:
    """Image input is not optional in this pipeline, so a model that cannot
    take an image is refused before the run starts.

    This pipeline reads the paper AND its exhibits: a table's value is read
    off a crop, `<img>label</img>` names one, and the checker verifies a value
    against the exhibit it was read from. A role that cannot see an image
    cannot do that, and a run that proceeded anyway would answer with the same
    `run_fp` a full run answers with — the same question, apparently asked and
    actually not.

    The refusal is per ENABLED stage, so an ablation that turns the checker or
    the reviewer off is not refused for the model it is no longer using.
    """

    def _refusal(self, config_dir, bundle_dir, out_dir, **kwargs):
        orch = _orch(config_dir, bundle_dir, out_dir, **kwargs)
        with pytest.raises(AgenticExtractionError) as excinfo:
            orch.prepare_new_session()
        return orch, str(excinfo.value)

    @pytest.mark.parametrize("role,key,kwargs", [
        ("extractor", "extractor_model",
         {"extractor_model": TEXT_ONLY_MODEL}),
        ("checker", "checker_model",
         {"extractor_model": "claude-opus-4-7",
          "checker_model": TEXT_ONLY_MODEL}),
        ("review", "review_model",
         {"extractor_model": "claude-opus-4-7",
          "review_model": TEXT_ONLY_MODEL}),
    ])
    def test_each_enabled_role_is_refused_by_name(
            self, config_dir, bundle_minimal_dir, tmp_path, role, key,
            kwargs):
        # The message names the role, the model and the pipeline.yaml key,
        # because that key is the line an operator edits.
        _, message = self._refusal(
            config_dir, bundle_minimal_dir, tmp_path / role, **kwargs)
        assert role in message
        assert TEXT_ONLY_MODEL in message
        assert key in message

    def test_nothing_is_created_and_nothing_is_called(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Refused BEFORE spend: the guard runs ahead of session creation, so
        # there is no session on disk to resume and no provider was reached.
        out_dir = tmp_path / "runs"
        orch, _ = self._refusal(config_dir, bundle_minimal_dir, out_dir,
                                extractor_model=TEXT_ONLY_MODEL)
        assert orch.session is None
        assert not out_dir.exists()

    def test_a_dry_run_is_refused_too(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A dry run exists to show what a run would send. For a configuration
        # that cannot run, the honest report is the refusal, not a preview of
        # a message no role could read.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL, dry_run=True)
        with pytest.raises(AgenticExtractionError) as excinfo:
            orch.dry_run_report()
        assert TEXT_ONLY_MODEL in str(excinfo.value)

    @pytest.mark.parametrize("stage_off", ["checker", "review"])
    def test_a_disabled_stage_is_not_refused_for_its_model(
            self, config_dir, bundle_minimal_dir, tmp_path, stage_off):
        # The guard asks what this run USES. A pipeline that names a
        # text-only checker and then runs `max_checks_per_field: 0` never
        # sends it anything, and refusing it would make a documented ablation
        # unrunnable on a model it does not reach.
        kwargs = {"extractor_model": "claude-opus-4-7"}
        if stage_off == "checker":
            kwargs.update(checker_model=TEXT_ONLY_MODEL,
                          max_checks_per_field=0)
        else:
            kwargs.update(review_model=TEXT_ONLY_MODEL, final_review=False)
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / stage_off,
                     **kwargs)
        orch.prepare_new_session()
        assert orch.session is not None

    def test_an_image_capable_run_is_unaffected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        orch.prepare_new_session()
        # The crop rides, the dispatcher carries the real label, and nothing
        # is said about withholding anything.
        assert len(_canonical_image_blocks(orch.messages)) == 1
        assert orch.dispatcher.image_labels == {"table_01"}
        assert "images-omitted" not in capsys.readouterr().err
        # And the record carries no omission key: no run can populate one.
        assert "images_omitted" not in orch.session.meta


# ---------------------------------------------------------------------------
# The dispatcher-level evidence semantics (validator unchanged)
# ---------------------------------------------------------------------------

def test_empty_label_set_rejects_img_citation(
        synthetic_template, paper_text):
    # A dispatcher with no image labels (what a text-only extractor gets)
    # rejects an <img> citation as an unknown label. That is ordinary
    # validator behaviour; the guard ONLY supplies the empty set.
    record = ExtractionRecord()
    # The initial-check ordering gate would refuse the write before the
    # evidence validator ever saw it, and it is not what is on trial here, so
    # open it directly rather than scripting the call that opens it.
    record.initial_check_recorded = True
    disp = ToolDispatcher(record, synthetic_template, paper_text,
                          image_labels=set())
    res = disp.dispatch("update_study", {"study": {
        "primary_aim": {"value": "An aim", "evidence": "<img>table_01</img>"},
    }})
    assert res["status"] == "validation_failed"
    codes = {e["code"] for errs in res["failed_fields"].values() for e in errs}
    assert "unknown_image_label" in codes


# ---------------------------------------------------------------------------
# No-figures bundle: the guard is a no-op (nothing to withhold)
# ---------------------------------------------------------------------------

class TestNoFiguresBundle:
    def test_the_message_says_the_paper_supplies_none(
            self, config_dir, tmp_path, capsys):
        # The bundle's `exhibits: []` is an assertion that the paper has no
        # tables and no figures, and the message passes it on: a role is told
        # there are none rather than left to infer it from a message that
        # stops after the paper text. This is the ONLY thing that produces
        # that notice now — a role that cannot read a crop is refused, so the
        # notice always describes the paper and never the model.
        bundle_dir = _no_figures_bundle(tmp_path)
        orch = _orch(config_dir, bundle_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        orch.prepare_new_session()
        assert _canonical_image_blocks(orch.messages) == []
        assert NO_EXHIBITS_NOTICE in _message_text(orch.messages)
        assert "images-omitted" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Checker: per-field figure attachment guarded on the checker's own model
# ---------------------------------------------------------------------------

def _checker_orch(config_dir, bundle_dir, tmp_path, checker_model,
                  cited="table_01"):
    """A real orchestrator, with only what a session would have supplied.

    Built through the constructor rather than assembled attribute by
    attribute, because the four maps the checker resolves a citation through
    are built there and a hand-made stand-in gets to disagree with them: one
    that carried the article's notes where the run carries the whole bundle's
    would show a supplement's crop arriving with no footnote and call it
    correct.

    Only the extraction record is planted, which is the one thing a run
    produces rather than derives.
    """
    orch = Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir),
        tmp_path / "runs",
        extractor_model="claude-opus-4-7",
        checker_config=CheckerConfig(max_tokens=1024,
                                     checker_model=checker_model,
                                     context_chars=0),
        review_model="claude-opus-4-7",
        max_checks_per_field=2,
        extractor_max_tokens=4096, review_max_tokens=4096,
    )
    record = ExtractionRecord()
    record.apply_update_study(study={
        "sample_size": {"value": 402, "evidence": f"<img>{cited}</img>"},
    })
    orch.extraction_record = record
    return orch


def _call_image_blocks(calls):
    blocks = []
    for c in calls:
        blocks += [b for b in c["user_message_blocks"]
                   if isinstance(b, dict) and b.get("type") == "image"]
    return blocks


class TestCheckerAttachment:
    def test_image_capable_checker_attaches_the_cropped_png(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Positive control: an image-capable checker attaches the PNG cited by
        # the field's <img> evidence.
        orch = _checker_orch(config_dir, bundle_minimal_dir,
                             tmp_path, "claude-sonnet-4-6")
        calls, _ = orch._build_checker_calls(["study.sample_size"])
        assert len(_call_image_blocks(calls)) == 1

    def test_a_supplements_crop_attaches_like_the_articles(
            self, config_dir, bundle_supplemented_dir, tmp_path):
        """The evidence a role is steered toward for a supplement's exhibit.

        A label is one exhibit across a whole bundle, so `<img>` on a
        supplement's exhibit is an ordinary citation and the extractor's
        briefing recommends it. The checker resolves it through the same four
        maps as the article's: it validated the label, so it owes the crop.

        The alternative is the worst shape available — the evidence block
        telling the checker "the cropped image is attached below; treat it AS
        the evidence" with nothing attached, on a route the briefing
        recommends, leaving it to challenge or invent.
        """
        orch = _checker_orch(config_dir, bundle_supplemented_dir,
                             tmp_path, "claude-sonnet-4-6",
                             cited="supplement_a_table_01")
        calls, _ = orch._build_checker_calls(["study.sample_size"])
        assert len(_call_image_blocks(calls)) == 1
        texts = [b["text"] for b in calls[0]["user_message_blocks"]
                 if b.get("type") == "text"]
        # The promise the evidence block makes, and the label block that keeps
        # it, carrying the exhibit's own footnote and content as text.
        assert any("attached below" in t for t in texts)
        label_block = next(
            t for t in texts if t.startswith("[supplement_a_table_01]"))
        assert "Footnote: IQR, interquartile range." in label_block
        assert "<table>" in label_block

    def test_no_citable_label_is_left_without_its_crop(
            self, config_dir, bundle_supplemented_dir, tmp_path):
        # The property behind the case above, stated over the whole bundle:
        # the labels the checker accepts and the crops it can attach are one
        # set. A citation it validates and cannot illustrate is the shape that
        # sends a verdict on an exhibit nobody saw.
        orch = _checker_orch(config_dir, bundle_supplemented_dir,
                             tmp_path, "claude-sonnet-4-6")
        assert orch.image_labels == set(orch.image_figures)

    def test_the_attached_crops_footnote_arrives_under_its_label(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The checker's context is deliberately narrow, and this stays inside
        # it: the footnote is printed on the crop the checker is already
        # holding, so reading it as text adds nothing the attachment did not
        # carry. The paper's caption is a different matter and stays out.
        orch = _checker_orch(config_dir, bundle_minimal_dir,
                             tmp_path, "claude-sonnet-4-6")
        calls, _ = orch._build_checker_calls(["study.sample_size"])
        texts = [b["text"] for b in calls[0]["user_message_blocks"]
                 if b.get("type") == "text"]
        label_block = next(t for t in texts if t.startswith("[table_01]"))
        assert label_block.startswith("[table_01]\nFootnote: CI, confidence")
        assert "Primary and secondary associations" not in label_block

    def test_review_fp_moves_with_review_model_capability(
            self, config_dir, monkeypatch):
        # The reviewer's stage fingerprint folds its own model's capability,
        # so the flag cannot flip under a recorded run without the number
        # moving. Asserted on the instrument, for the reason the extractor's
        # twin is: a run naming a text-only reviewer is refused before it has
        # a session.
        identity = (TEXT_ONLY_MODEL, {"max_tokens": 4096})
        fp_text_only = _instrument(config_dir).review_fingerprint(
            identity, review_model=TEXT_ONLY_MODEL, tool_hash="t")
        patched = dataclasses.replace(
            MODEL_REGISTRY[TEXT_ONLY_MODEL], supports_images=True)
        monkeypatch.setitem(MODEL_REGISTRY, TEXT_ONLY_MODEL, patched)
        assert fp_text_only != _instrument(config_dir).review_fingerprint(
            identity, review_model=TEXT_ONLY_MODEL, tool_hash="t")


# ---------------------------------------------------------------------------
# The dry run previews the message, not the bundle
# ---------------------------------------------------------------------------

def _mixed_case_label_bundle(tmp_path, bundle_minimal_dir):
    """A copy of the fixture whose exhibit label carries a capital.

    The format's label rule admits one (`^[A-Za-z0-9._-]+$`), and a capital is
    the shape that tells a preview built from the message apart from one built
    from the dispatcher's normalised label set.
    """
    root = tmp_path / "mixed"
    shutil.copytree(bundle_minimal_dir, root)
    # Unlink before writing: a case-insensitive filesystem (macOS by default)
    # treats the two names as one file, so writing first would delete what was
    # just written.
    crop = (root / "figures" / "table_01.png").read_bytes()
    (root / "figures" / "table_01.png").unlink()
    (root / "figures" / "Table_01.png").write_bytes(crop)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["exhibits"][0]["label"] = "Table_01"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _preview_entries(text):
    """The exhibits a dry-run report names, back in the form the message
    carries them.

    The file's rule is that an entry starts at column 0 and every later line
    of it is indented, so a footnote of any shape stays inside its exhibit.
    Reading it back is how a test compares the file against the message
    without either of them being the other's definition.
    """
    entries = []
    for line in text.split("\n"):
        if line.startswith("  "):
            entries[-1] += "\n" + line[2:]
        elif line:
            entries.append(line)
    return entries


class TestTheDryRunPreviewIsTheMessage:
    """`attached_exhibits` is a preview of the extractor's user message, so it
    is built from that message's own figure sequence.

    The normalised label set is the dispatcher's, lower-cased and covering the
    whole bundle. A preview built from it would name a label the message never
    sent, in a case the message never used, and a paper that capitalises its
    own label is a valid bundle.
    """

    def _preview(self, orch, tmp_path):
        orch.dry_run_report(tmp_path / "report")
        return (tmp_path / "report" / "attached_exhibits.txt").read_text(
            encoding="utf-8")

    def _message_label_blocks(self, orch):
        return [image_label_text(label, orch.image_captions, orch.image_notes)
                for label, _ in orch.figures]

    def test_it_spells_a_label_the_way_the_message_does(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        bundle_dir = _mixed_case_label_bundle(tmp_path, bundle_minimal_dir)
        orch = _orch(config_dir, bundle_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        preview = self._preview(orch, tmp_path)
        capsys.readouterr()
        assert "[Table_01]" in preview
        assert "[table_01]" not in preview
        # Entry for entry, the message's own blocks.
        assert _preview_entries(preview) == self._message_label_blocks(orch)

    def test_the_printed_preview_indents_an_exhibit_whole(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # stdout is read by eye rather than parsed, so what it owes the reader
        # is that a footnote sits under the exhibit it belongs to instead of
        # hanging at the margin as if it were one.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        orch.dry_run_report(tmp_path / "report")
        printed = capsys.readouterr().out
        block = printed.split("=== ATTACHED EXHIBITS (1) ===")[1]
        # To the next section, whatever follows this one.
        block = block.split("\n=== ")[0]
        lines = [ln for ln in block.split("\n") if ln.strip()]
        assert lines[0].startswith("  [table_01] Table 1.")
        assert lines[1].startswith("  Footnote: CI, confidence")

    def test_a_footnote_of_any_shape_stays_inside_its_exhibit(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        """Two exhibits, the first carrying a footnote of two paragraphs.

        The format asks a footnote only to be a non-empty string, so a blank
        line inside one is a valid bundle, and a file that separated exhibits
        by a blank line would read this as three. The reader's rule is column
        0, which no footnote line can reach.
        """
        root = tmp_path / "shaped"
        shutil.copytree(bundle_minimal_dir, root)
        (root / "figures" / "figure_02.png").write_bytes(
            (root / "figures" / "table_01.png").read_bytes())
        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8"))
        manifest["exhibits"][0]["notes"] = (
            "SD, standard deviation.\n\nUnits withdrawn before the first "
            "round are excluded from every column.")
        manifest["exhibits"].append(
            {"label": "figure_02", "caption": "Figure 2. Fleet flow"})
        (root / "manifest.json").write_text(json.dumps(manifest),
                                            encoding="utf-8")

        orch = _orch(config_dir, root, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        preview = self._preview(orch, tmp_path)
        capsys.readouterr()
        entries = _preview_entries(preview)
        assert len(entries) == 2
        assert entries == self._message_label_blocks(orch)
        assert "\n\nUnits withdrawn" in entries[1]
