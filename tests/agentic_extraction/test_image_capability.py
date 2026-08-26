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
from meltiro.extraction_record import ExtractionRecord
from meltiro.fingerprint import structure_hash
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
          max_checks_per_field=3, final_review=True):
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

    def test_capability_flag_moves_config_fp(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # Same model id, prompts, template, tools, decoding, provider: only the
        # registry capability flag differs. config_fp must move, proving the
        # flag itself (not just the model id) folds into the fingerprint, so a
        # registry edit that turns a text-only model into a vision one can
        # never happen silently.
        a = _orch(config_dir, bundle_minimal_dir, tmp_path / "a",
                  extractor_model=TEXT_ONLY_MODEL)
        a.prepare_new_session()
        fp_text_only = a.session.meta["config_fp"]

        patched = dataclasses.replace(
            MODEL_REGISTRY[TEXT_ONLY_MODEL], supports_images=True)
        monkeypatch.setitem(MODEL_REGISTRY, TEXT_ONLY_MODEL, patched)
        b = _orch(config_dir, bundle_minimal_dir, tmp_path / "b",
                  extractor_model=TEXT_ONLY_MODEL)
        b.prepare_new_session()
        fp_capable = b.session.meta["config_fp"]

        assert fp_text_only != fp_capable


# ---------------------------------------------------------------------------
# Extractor: no image parts sent, none-available prompt, meta + warning
# ---------------------------------------------------------------------------

class TestTextOnlyExtractor:
    def test_no_image_blocks_in_the_canonical_messages(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # bundle_minimal carries one figure (table_01). A text-only
        # extractor must put zero image parts into the canonical messages.
        # That is the whole of meltiro's responsibility here: a wire
        # translation renders what it is given and cannot invent an image,
        # so asserting against a provider's translator would test the
        # provider layer's job through its private surface. The companion
        # test below pins the same property at the adapter boundary, which
        # is the last point meltiro owns.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()

        assert _canonical_image_blocks(orch.messages) == []

    def test_fake_adapter_receives_no_image_parts(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # A fake adapter captures exactly what the extractor turn would send.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()

        captured = {}

        class _Capture:
            def create_message(self, **kwargs):
                captured["messages"] = kwargs["messages"]
                captured["system"] = kwargs["system"]
                return SimpleNamespace(
                    content=[], usage=SimpleNamespace(),
                    resolved_model=TEXT_ONLY_MODEL, provider="openai_compat",
                    base_url="x", raw_request={}, raw_response={},
                    wire_request=None, decoding_params={})

        orch._call_extractor(_Capture(), tool_defs=[])
        assert _canonical_image_blocks(captured["messages"]) == []
        # No label block either: a label with no image behind it would invite
        # a citation of an exhibit the model was never shown.
        assert "table_01" not in _message_text(captured["messages"])
        # And nothing about the paper is in the system prompt, whatever the
        # model's capability.
        sys_text = "".join(b["text"] for b in captured["system"])
        assert "table_01" not in sys_text

    def test_the_labels_go_with_the_images(
            self, config_dir, bundle_minimal_dir, tmp_path):
        text_only = _orch(config_dir, bundle_minimal_dir, tmp_path / "glm",
                          extractor_model=TEXT_ONLY_MODEL)
        text_only.prepare_new_session()
        # The captured user prompt lists no image label, and the system
        # prompt never carried one.
        user_prompt = (text_only.session.instrument_dir /
                       "user_prompt.txt").read_text(encoding="utf-8")
        assert "table_01" not in user_prompt
        assert "table_01" not in text_only.system_text

        capable = _orch(config_dir, bundle_minimal_dir, tmp_path / "opus",
                        extractor_model="claude-opus-4-7")
        capable.prepare_new_session()
        # Positive control: an image-capable extractor is shown the real
        # label, in the message the crop itself arrives in.
        assert "table_01" in _message_text(capable.messages)
        assert "table_01" not in capable.system_text

    def test_the_message_says_that_none_accompany_the_study(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The other half of withholding the images. The system prompt is one
        # string per config and cannot know this role's capability, so the
        # statement belongs to the message — and a text-only role reads the
        # same one a no-crops bundle produces, rather than a message that
        # simply stops after the paper text. It is in the capture too, because
        # the capture is the record of the message.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()
        assert NO_EXHIBITS_NOTICE in _message_text(orch.messages)
        assert NO_EXHIBITS_NOTICE in (
            orch.session.instrument_dir / "user_prompt.txt").read_text(
                encoding="utf-8")

        # Positive control: the same bundle under an image-capable extractor
        # carries the crop instead, and says nothing of the kind.
        capable = _orch(config_dir, bundle_minimal_dir, tmp_path / "opus",
                        extractor_model="claude-opus-4-7")
        capable.prepare_new_session()
        assert NO_EXHIBITS_NOTICE not in _message_text(capable.messages)

    def test_the_exhibits_record_is_empty_for_a_text_only_extractor(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # `instrument/image_labels.json` records what the message carried, so
        # a role sent no crop records none. `meta.images_omitted` beside it is
        # what separates this from a bundle that ships no crops at all.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()
        assert json.loads(
            (orch.session.instrument_dir / "image_labels.json").read_text(
                encoding="utf-8")) == []
        assert orch.session.meta["images_omitted"] == {"extractor": True}

    def test_meta_flag_and_warning(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()
        assert orch.session.meta["images_omitted"] == {"extractor": True}
        err = capsys.readouterr().err
        assert "images-omitted" in err
        assert "extractor" in err
        assert TEXT_ONLY_MODEL in err
        assert "1 bundle figure" in err  # one figure withheld

    def test_img_citation_fails_validation_for_text_only_extractor(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The dispatcher a text-only extractor drives carries an empty label
        # set, so an <img> citation of a figure it never saw fails as an
        # unknown label. The validator is not special-cased on capability; the
        # empty set is the whole of what makes the citation unknown.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()
        assert orch.dispatcher.image_labels == set()

    def test_image_capable_extractor_is_unaffected(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        orch.prepare_new_session()
        # Image block present, dispatcher carries the real label, no omission.
        assert len(_canonical_image_blocks(orch.messages)) == 1
        assert orch.dispatcher.image_labels == {"table_01"}
        assert orch.session.meta["images_omitted"] == {}
        assert "images-omitted" not in capsys.readouterr().err


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
    def test_image_capable_extractor_records_no_omission(
            self, config_dir, tmp_path, capsys):
        bundle_dir = _no_figures_bundle(tmp_path)
        orch = _orch(config_dir, bundle_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7")
        orch.prepare_new_session()
        assert _canonical_image_blocks(orch.messages) == []
        # The bundle's `exhibits: []` is an assertion that the paper has no
        # tables and no figures, and the message passes it on: an
        # image-capable model is told there are none rather than left to infer
        # it from a message that stops after the paper text.
        assert NO_EXHIBITS_NOTICE in _message_text(orch.messages)
        assert orch.session.meta["images_omitted"] == {}
        assert "images-omitted" not in capsys.readouterr().err

    def test_text_only_extractor_withholds_nothing(
            self, config_dir, tmp_path, capsys):
        # No figures means nothing is withheld even for a text-only model, so
        # no omission is recorded and no warning fires (the fingerprint still
        # moves via the capability flag, tested separately).
        bundle_dir = _no_figures_bundle(tmp_path)
        orch = _orch(config_dir, bundle_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()
        assert orch.session.meta["images_omitted"] == {}
        assert "images-omitted" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Checker: per-field figure attachment guarded on the checker's own model
# ---------------------------------------------------------------------------

def _checker_orch(config_dir, bundle_dir, template, checker_model):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_dir)
    orch = Orchestrator.__new__(Orchestrator)
    orch.template = template
    orch.image_labels = {"table_01"}
    orch.image_notes = bundle.exhibit_notes
    orch.config = config
    orch.bundle = bundle  # figures={"table_01": Path(.../table_01.png)}
    # context_chars=0: this is the image-attachment path, which has no text
    # position to window into, so the quote-context machinery is out of scope.
    orch.checker_config = SimpleNamespace(checker_model=checker_model,
                                          context_chars=0)
    orch.paper_text = bundle.text
    # The instrument the per-field template renders its conditional blocks
    # against is built from these three toggles plus the config bundle above.
    orch.reference_lists = config.reference_lists
    orch.max_checks_per_field = 2
    orch.final_review = True
    orch.check_reviewer_edits = False
    orch._check_counts = {}
    orch._study_identity_context = lambda: "Summary: ctx"
    record = ExtractionRecord()
    record.apply_update_study(study={
        "primary_aim": {"value": "An aim", "evidence": "<img>table_01</img>"},
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
            self, config_dir, bundle_minimal_dir, synthetic_template):
        # Positive control: an image-capable checker attaches the PNG cited by
        # the field's <img> evidence.
        orch = _checker_orch(config_dir, bundle_minimal_dir,
                             synthetic_template, "claude-sonnet-4-6")
        calls, _ = orch._build_checker_calls(["study.primary_aim"])
        assert len(_call_image_blocks(calls)) == 1

    def test_text_only_checker_attaches_nothing(
            self, config_dir, bundle_minimal_dir, synthetic_template):
        # Mixed case: image-capable extractor produced an <img> citation, but
        # the checker is text-only. It must attach no PNG (and produce no
        # error), because its own model cannot accept images.
        orch = _checker_orch(config_dir, bundle_minimal_dir,
                             synthetic_template, TEXT_ONLY_MODEL)
        calls, _ = orch._build_checker_calls(["study.primary_aim"])
        assert _call_image_blocks(calls) == []
        # The call is still built (no error), just without the attachment.
        assert any(c["field_path"] == "study.primary_aim" for c in calls)

    def test_the_attached_crops_footnote_arrives_under_its_label(
            self, config_dir, bundle_minimal_dir, synthetic_template):
        # The checker's context is deliberately narrow, and this stays inside
        # it: the footnote is printed on the crop the checker is already
        # holding, so reading it as text adds nothing the attachment did not
        # carry. The paper's caption is a different matter and stays out.
        orch = _checker_orch(config_dir, bundle_minimal_dir,
                             synthetic_template, "claude-sonnet-4-6")
        calls, _ = orch._build_checker_calls(["study.primary_aim"])
        texts = [b["text"] for b in calls[0]["user_message_blocks"]
                 if b.get("type") == "text"]
        label_block = next(t for t in texts if t.startswith("[table_01]"))
        assert label_block.startswith("[table_01]\nFootnote: CI, confidence")
        assert "Primary and secondary associations" not in label_block

    def test_a_text_only_checker_is_handed_no_footnote_either(
            self, config_dir, bundle_minimal_dir, synthetic_template,
            monkeypatch):
        """The footnote describes an attachment, so it is withheld with the
        attachment rather than becoming a text-only consolation for the crop.

        Asserted on what the trigger HANDS the message builder, because the
        rendered message cannot tell the two apart: with no figures map the
        builder emits no label block at all, so a footnote passed there would
        be invisible in the output and the guard untested.
        """
        handed = {}
        real = orch_mod.build_checker_user_message
        monkeypatch.setattr(
            orch_mod, "build_checker_user_message",
            lambda **kw: handed.update(kw) or real(**kw))

        orch = _checker_orch(config_dir, bundle_minimal_dir,
                             synthetic_template, TEXT_ONLY_MODEL)
        orch._build_checker_calls(["study.primary_aim"])
        assert handed["figures"] is None
        assert handed["exhibit_notes"] is None

        handed.clear()
        capable = _checker_orch(config_dir, bundle_minimal_dir,
                                synthetic_template, "claude-sonnet-4-6")
        capable._build_checker_calls(["study.primary_aim"])
        assert handed["figures"] == capable.bundle.figures
        assert handed["exhibit_notes"] == capable.image_notes
        assert handed["exhibit_notes"]  # the fixture records one


# ---------------------------------------------------------------------------
# Review: guarded on the reviewer's own model, folded into review_fp
# ---------------------------------------------------------------------------

class TestReviewCapability:
    def test_review_fp_moves_with_review_model_capability(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        base = _orch(config_dir, bundle_minimal_dir, tmp_path / "a",
                     extractor_model="claude-opus-4-7",
                     review_model=TEXT_ONLY_MODEL)
        base.prepare_new_session()
        fp_text_only = base.session.meta["review_fp"]

        patched = dataclasses.replace(
            MODEL_REGISTRY[TEXT_ONLY_MODEL], supports_images=True)
        monkeypatch.setitem(MODEL_REGISTRY, TEXT_ONLY_MODEL, patched)
        capable = _orch(config_dir, bundle_minimal_dir, tmp_path / "b",
                        extractor_model="claude-opus-4-7",
                        review_model=TEXT_ONLY_MODEL)
        capable.prepare_new_session()
        assert fp_text_only != capable.session.meta["review_fp"]

    def test_text_only_review_records_omission(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model="claude-opus-4-7",
                     review_model=TEXT_ONLY_MODEL)
        orch.prepare_new_session()
        assert orch.session.meta["images_omitted"] == {"review": True}


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
    whole bundle. A preview built from it would differ from the message in two
    ways at once, and each is reachable from a valid bundle: a label the paper
    capitalised, and a role that was sent no images at all.
    """

    def _preview(self, orch, tmp_path):
        orch.dry_run_report(tmp_path / "report")
        return (tmp_path / "report" / "attached_exhibits.txt").read_text(
            encoding="utf-8")

    def _message_label_blocks(self, orch):
        figures, _ = orch._extractor_image_inputs()
        return [image_label_text(label, orch.image_captions, orch.image_notes)
                for label, _ in figures]

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
        block = block.split("=== FINGERPRINTS ===")[0]
        lines = [ln for ln in block.split("\n") if ln.strip()]
        assert lines[0].startswith("  [table_01] Table 1.")
        assert lines[1].startswith("  Footnote: CI, confidence")

    def test_a_text_only_extractor_previews_no_exhibits(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        # Its message states that none accompany the study, so a preview
        # listing the bundle's crops would contradict the message and the
        # `images_omitted` flag in the same report.
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     extractor_model=TEXT_ONLY_MODEL)
        preview = self._preview(orch, tmp_path)
        capsys.readouterr()
        assert preview == ""
        assert self._message_label_blocks(orch) == []

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
