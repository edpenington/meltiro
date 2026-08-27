"""Supplementary material as a document of its own, carried into a run.

The format keeps a supplement out of the article: its prose never joins
`text.md`, its exhibits never join the manifest, and it is declared in a file
of its own. That separation is the thing this has to preserve, because it is
what the format bought — a paper's identity does not move when a supplement
lands, and a value quoted out of a supplement is a claim about a document
that is often not reviewed to the article's standard and can be revised after
publication.

So two claims are tested together, and they pull against each other:

  - a supplement REACHES a run. Its prose, its crops and its transcriptions
    are all in the message, in a section of its own that says which document
    they belong to.
  - it stays SEPARATE. `text.md` is untouched, no `<q>` is ever checked
    against a supplement's prose, and the article's own fingerprint axes do
    not move when a supplement arrives.

Between them sits the one thing that IS shared: the exhibit maps. A label is
unique across a whole bundle by the format's rule, so `<img>label</img>`
resolves to one exhibit wherever it sits, and the dispatcher and the checker
read one flat map. What the message groups, the citation does not have to.
"""

import shutil

import pytest
from alteksto.bundle import validate_bundle

from meltiro.bundle import load_bundle
from meltiro.checker_prompts import build_checker_system_text
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import bundle_fingerprint
from meltiro.prompt_partials import stage_predicates
from meltiro.prompt_builder import (
    NO_SUPPLEMENT_TEXT_NOTICE,
    build_initial_user_blocks,
    build_review_system_message,
    build_review_user_blocks,
    build_system_message,
    render_user_prompt_text,
    supplement_close,
    supplement_open,
)
from meltiro.quote_check import find_quote


def _text_blocks(blocks):
    return [b["text"] for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"]


def _joined(blocks):
    return "\n".join(_text_blocks(blocks))


@pytest.fixture
def supplemented(bundle_supplemented_dir):
    return load_bundle(bundle_supplemented_dir)


@pytest.fixture
def sections(supplemented):
    """The supplement payloads the message builders take."""
    return [
        {"name": s.name, "title": s.title, "text": s.text,
         "figures": [(label, b"png") for label in s.figures]}
        for s in supplemented.supplements.values()
    ]


class TestTheFixtureIsWhatItClaims:
    def test_it_is_a_bundle(self, bundle_supplemented_dir):
        assert validate_bundle(bundle_supplemented_dir) == []

    def test_the_supplement_carries_prose_a_crop_and_a_transcription(
            self, supplemented):
        s = supplemented.supplements["supplement_a"]
        assert s.text is not None
        assert set(s.figures) == {"supplement_a_table_01"}
        assert set(s.tables) == {"supplement_a_table_01"}


class TestTheLoaderCarriesIt:
    def test_supplements_are_keyed_by_name_in_order(self, supplemented):
        assert list(supplemented.supplements) == ["supplement_a"]

    def test_the_title_is_the_one_the_paper_prints(self, supplemented):
        # `name` is a directory and a token; `title` is what a reader
        # choosing between supplements chooses on, so it is what the message
        # shows.
        assert supplemented.supplements["supplement_a"].title.startswith(
            "Supplement A.")

    def test_an_ordinary_paper_carries_none(self, bundle_transcribed_dir):
        assert load_bundle(bundle_transcribed_dir).supplements == {}

    def test_a_supplement_that_prints_no_prose_carries_None(
            self, bundle_supplemented_dir, tmp_path):
        # Optional here where the article's is required, and absent is None
        # rather than "": a supplement that is a run of data tables prints no
        # prose, and inventing one would mean inventing the prose.
        dst = tmp_path / "no_prose"
        shutil.copytree(bundle_supplemented_dir, dst)
        (dst / "supplements" / "supplement_a" / "text.md").unlink()

        assert validate_bundle(dst) == []
        assert load_bundle(dst).supplements["supplement_a"].text is None


class TestTheExhibitMapsAreTheWholeBundles:
    """One flat map, because one label means one exhibit bundle-wide."""

    def test_all_figures_carries_both_documents(self, supplemented):
        assert set(supplemented.all_figures()) == {
            "table_01", "supplement_a_table_01"}

    def test_the_article_maps_stay_the_articles(self, supplemented):
        # The merged view is a view. The article's own map is still the
        # article's, which is what lets the message attach each document's
        # exhibits in its own section.
        assert set(supplemented.figures) == {"table_01"}

    def test_captions_notes_and_tables_merge_the_same_way(self, supplemented):
        for view in (supplemented.all_exhibits(),
                     supplemented.all_exhibit_notes(),
                     supplemented.all_tables()):
            assert set(view) == {"table_01", "supplement_a_table_01"}

    def test_a_citation_resolves_without_knowing_the_document(
            self, supplemented):
        # What the flat map is for: the dispatcher validates `<img>` against
        # labels alone and never asks which document an exhibit sits in.
        labels = {label.lower() for label in supplemented.all_figures()}
        assert "supplement_a_table_01" in labels


class TestItReachesTheMessageAsItsOwnDocument:
    def test_the_extractor_gets_a_delimited_section(
            self, supplemented, sections):
        blocks = build_initial_user_blocks(
            supplemented.study_id, supplemented.text,
            [("table_01", b"png")], supplemented.all_exhibits(),
            supplemented.all_exhibit_notes(), {}, sections)
        joined = _joined(blocks)
        assert supplement_open(
            "supplement_a",
            supplemented.supplements["supplement_a"].title) in joined
        assert supplement_close("supplement_a") in joined

    def test_the_supplements_prose_is_inside_that_section(
            self, supplemented, sections):
        blocks = build_initial_user_blocks(
            supplemented.study_id, supplemented.text,
            [("table_01", b"png")], supplemented.all_exhibits(),
            supplemented.all_exhibit_notes(), {}, sections)
        joined = _joined(blocks)
        prose = supplemented.supplements["supplement_a"].text.strip()
        opening = joined.index(supplement_open(
            "supplement_a", supplemented.supplements["supplement_a"].title))
        closing = joined.index(supplement_close("supplement_a"))
        assert opening < joined.index(prose.splitlines()[-1]) < closing

    def test_its_crop_attaches_inside_the_section_too(
            self, supplemented, sections):
        # An exhibit's document is a fact about the message, so the crop sits
        # between the delimiters rather than among the article's.
        blocks = build_initial_user_blocks(
            supplemented.study_id, supplemented.text,
            [("table_01", b"png")], supplemented.all_exhibits(),
            supplemented.all_exhibit_notes(), {}, sections)
        texts = _text_blocks(blocks)
        opening = next(i for i, t in enumerate(texts)
                       if t.startswith("--- SUPPLEMENT supplement_a:"))
        closing = next(i for i, t in enumerate(texts)
                       if t.startswith("--- END SUPPLEMENT supplement_a"))
        label_at = next(i for i, t in enumerate(texts)
                        if t.startswith("[supplement_a_table_01]"))
        assert opening < label_at < closing
        # And the article's own exhibit is NOT inside it.
        article_at = next(i for i, t in enumerate(texts)
                          if t.startswith("[table_01]"))
        assert article_at < opening

    def test_the_reviewer_gets_it_before_the_output_it_reviews(
            self, supplemented, sections):
        blocks = build_review_user_blocks(
            supplemented.study_id, supplemented.text,
            [("table_01", b"png")], {"study": {}},
            supplemented.all_exhibits(), supplemented.all_exhibit_notes(),
            {}, sections)
        joined = _joined(blocks)
        assert joined.index(supplement_close("supplement_a")) < joined.index(
            "--- ASSEMBLED EXTRACTION OUTPUT (to review) ---")

    def test_a_supplement_with_no_prose_says_so(self, supplemented):
        section = {"name": "supplement_a", "title": "S A", "text": None,
                   "figures": []}
        blocks = build_initial_user_blocks(
            supplemented.study_id, supplemented.text, [("table_01", b"png")],
            {}, {}, {}, [section])
        assert NO_SUPPLEMENT_TEXT_NOTICE in _joined(blocks)

    def test_the_recorded_prompt_matches_the_message(
            self, supplemented, sections):
        sent = build_initial_user_blocks(
            supplemented.study_id, supplemented.text,
            [("table_01", b"png")], supplemented.all_exhibits(),
            supplemented.all_exhibit_notes(), {}, sections)
        recorded = render_user_prompt_text(
            supplemented.study_id, supplemented.text, ["table_01"],
            supplemented.all_exhibits(), supplemented.all_exhibit_notes(),
            {}, sections)
        for block in _text_blocks(sent):
            assert block in recorded


class TestItStaysSeparateFromTheArticle:
    def test_the_articles_text_is_untouched(
            self, supplemented, bundle_transcribed_dir):
        # `text.md` stays the article's, byte for byte. The two fixtures share
        # one, so a supplement landing must not have changed it.
        assert supplemented.text == load_bundle(bundle_transcribed_dir).text

    def test_the_supplements_prose_is_not_quotable(self, supplemented):
        # The claim that makes "verbatim" mean something: a quote certified
        # against the paper is always a claim about the ARTICLE. Reading a
        # supplement is what `<img>` on its exhibits is for.
        prose = supplemented.supplements["supplement_a"].text
        sentence = "Turnaround is given per shift rather than pooled"
        assert sentence in prose
        assert not find_quote(sentence, supplemented.text)

    def test_a_supplements_cell_is_not_quotable_either(self, supplemented):
        markup = supplemented.supplements["supplement_a"].tables[
            "supplement_a_table_01"].read_text(encoding="utf-8")
        assert "7.2 (5.1-9.8)" in markup
        assert not find_quote("7.2 (5.1-9.8)", supplemented.text)


class TestThePapersIdentity:
    def test_supplements_fp_is_reported(self, supplemented):
        assert bundle_fingerprint(supplemented)["supplements_fp"].startswith(
            "supplements_fp:")

    def test_a_paper_with_none_still_reports_one(self, bundle_transcribed_dir):
        fp = bundle_fingerprint(load_bundle(bundle_transcribed_dir))
        assert fp["supplements_fp"].startswith("supplements_fp:")

    def test_a_supplement_landing_moves_no_article_axis(
            self, supplemented, bundle_transcribed_dir):
        # The property the whole shape was built for: a consumer that
        # identifies a paper by the article's own bytes — the screening side
        # does — is untouched by supplementary material arriving later, while
        # one that reads the whole bundle sees it.
        article = bundle_fingerprint(load_bundle(bundle_transcribed_dir))
        withsupp = bundle_fingerprint(supplemented)

        assert withsupp["text_fp"] == article["text_fp"]
        assert withsupp["figures_fp"] == article["figures_fp"]
        assert withsupp["tables_fp"] == article["tables_fp"]
        assert withsupp["supplements_fp"] != article["supplements_fp"]
        assert withsupp["bundle_fp"] != article["bundle_fp"]

    def test_the_manifest_axis_moves_only_because_the_id_differs(
            self, supplemented, bundle_transcribed_dir):
        # The two fixtures are the same article under different ids, so
        # manifest_fp is expected to differ. Asserted rather than left
        # ambiguous, so the test above is not read as proving more than it
        # does.
        assert supplemented.study_id != load_bundle(
            bundle_transcribed_dir).study_id

    @pytest.mark.parametrize("edit", ["prose", "crop", "transcription",
                                      "title", "withdrawn"])
    def test_every_part_of_a_supplement_moves_it(
            self, bundle_supplemented_dir, tmp_path, edit):
        import json

        dst = tmp_path / edit
        shutil.copytree(bundle_supplemented_dir, dst)
        supp = dst / "supplements" / "supplement_a"

        if edit == "prose":
            path = supp / "text.md"
            path.write_text(path.read_text(encoding="utf-8").replace(
                "per shift", "per rota shift"), encoding="utf-8")
        elif edit == "crop":
            path = supp / "figures" / "supplement_a_table_01.png"
            path.write_bytes(path.read_bytes() + b"\x00")
        elif edit == "transcription":
            path = supp / "tables" / "supplement_a_table_01.html"
            path.write_text(path.read_text(encoding="utf-8").replace(
                "<td>7.2 (5.1-9.8)</td>", "<td>7.3 (5.1-9.8)</td>"),
                encoding="utf-8")
        elif edit == "title":
            path = dst / "supplements.json"
            declared = json.loads(path.read_text(encoding="utf-8"))
            declared["supplements"][0]["title"] = "Supplement A. Renamed"
            path.write_text(json.dumps(declared), encoding="utf-8")
        elif edit == "withdrawn":
            shutil.rmtree(dst / "supplements")
            (dst / "supplements.json").unlink()

        before = bundle_fingerprint(load_bundle(bundle_supplemented_dir))
        after = bundle_fingerprint(load_bundle(dst))
        assert after["supplements_fp"] != before["supplements_fp"], edit
        assert after["bundle_fp"] != before["bundle_fp"], edit
        # And none of it is mistaken for an edit to the article.
        assert after["text_fp"] == before["text_fp"], edit
        assert after["figures_fp"] == before["figures_fp"], edit
        assert after["tables_fp"] == before["tables_fp"], edit

    def test_it_is_a_resume_axis(self):
        from meltiro.session import BUNDLE_AXES

        assert "supplements_fp" in BUNDLE_AXES


class TestTheDryRunPreviewIsTheMessage:
    """The preview's headline count is the one number a reader trusts."""

    def _orch(self, config_dir, bundle_dir, out_dir):
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator

        return Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
            dry_run=True,
        )

    def test_it_previews_every_crop_the_message_attaches(
            self, config_dir, bundle_supplemented_dir, tmp_path, capsys):
        orch = self._orch(config_dir, bundle_supplemented_dir,
                          tmp_path / "runs")
        orch.dry_run_report()
        out = capsys.readouterr().out

        # Two crops: the article's and the supplement's. Counting the
        # article's alone would under-report a supplemented study.
        assert "=== ATTACHED EXHIBITS (2) ===" in out
        assert "[table_01]" in out
        assert "[supplement_a_table_01]" in out

    def test_a_supplements_crop_is_previewed_under_its_supplement(
            self, config_dir, bundle_supplemented_dir, tmp_path, capsys):
        # The message puts it in that supplement's section, and a label alone
        # does not say which document a crop came out of.
        orch = self._orch(config_dir, bundle_supplemented_dir,
                          tmp_path / "runs")
        orch.dry_run_report()
        out = capsys.readouterr().out
        assert "(supplement_a) [supplement_a_table_01]" in out

    def test_the_preview_carries_the_section_the_message_carries(
            self, config_dir, bundle_supplemented_dir, tmp_path, capsys):
        """The supplement's own framing, previewed before anything is spent.

        A dry run exists to show what a run would send, and a supplement's
        prose rides in the same cached user message as the paper's, which is
        most of what that run pays for. The delimiters are the part an
        operator cannot read off the bundle: they are the engine's, and they
        are what tells a role which document a value came out of.
        """
        supplemented = load_bundle(bundle_supplemented_dir)
        supplement = next(iter(supplemented.supplements.values()))
        orch = self._orch(config_dir, bundle_supplemented_dir,
                          tmp_path / "runs")
        report = orch.dry_run_report()
        out = capsys.readouterr().out

        for previewed in (report["user_message"], out):
            assert supplement_open(
                supplement.name, supplement.title) in previewed
            assert supplement_close(supplement.name) in previewed
            # Its prose in full, and the article's beside it: the message
            # holds both, so a preview of one is a preview of half the spend.
            assert supplement.text.strip() in previewed
            assert supplemented.text.strip() in previewed

    def test_the_preview_reports_the_axis_a_supplement_moves(
            self, config_dir, bundle_supplemented_dir, tmp_path, capsys):
        # A dry run is per-paper, so whether supplementary material moved the
        # input's identity is answerable without paying for the run that
        # would record it.
        orch = self._orch(config_dir, bundle_supplemented_dir,
                          tmp_path / "runs")
        report = orch.dry_run_report()
        capsys.readouterr()
        fps = report["fingerprints"]
        expected = bundle_fingerprint(load_bundle(bundle_supplemented_dir))
        for axis, value in expected.items():
            assert fps[axis] == value, (
                f"the preview's {axis} is not the one a run would record")


class TestEveryRoleIsSentEveryOne:
    """No role is sent a subset of the supplementary material.

    A supplement reaches a role as a document — prose, crops and
    transcriptions together — and there is no configuration in which a role
    gets part of one or none of them: a run whose model cannot read a crop is
    refused before it starts (see
    tests/agentic_extraction/test_image_capability.py).

    Asserted against the bundle rather than against the orchestrator's own
    attribute, so it is the loader's answer being compared with the message's.
    """

    def _orch(self, config_dir, bundle_dir, out_dir):
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator

        return Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
        )

    def test_the_sections_are_the_bundles_supplements(
            self, config_dir, bundle_supplemented_dir, tmp_path):
        bundle = load_bundle(bundle_supplemented_dir)
        orch = self._orch(config_dir, bundle_supplemented_dir,
                          tmp_path / "runs")
        sections = orch._supplements_for()
        assert [s["name"] for s in sections] == list(bundle.supplements)
        for section, supplement in zip(sections, bundle.supplements.values()):
            assert section["title"] == supplement.title
            assert section["text"] == supplement.text
            assert [label for label, _ in section["figures"]] == list(
                supplement.figures)

    def test_both_message_building_roles_get_them(
            self, config_dir, bundle_supplemented_dir, tmp_path):
        bundle = load_bundle(bundle_supplemented_dir)
        supplement = next(iter(bundle.supplements.values()))
        orch = self._orch(config_dir, bundle_supplemented_dir,
                          tmp_path / "runs")
        orch.prepare_new_session()
        extractor = _joined(orch.messages[0]["content"])
        reviewer = _joined(orch._review_message({"study": {}})[0])
        for message in (extractor, reviewer):
            assert supplement_open(
                supplement.name, supplement.title) in message
            assert supplement.text.strip() in message
            assert supplement_close(supplement.name) in message


class TestEveryRoleShownASectionIsBriefedOnIt:
    """The briefing parity, and its other half: the role shown none is told
    none.

    The extractor and the reviewer are both sent supplement sections, so both
    have to be told what a section is and what a value read out of one is a
    claim about — a delimited block naming a document is exactly the shape a
    role will otherwise read as more paper text.

    The checker is sent no section. It is shown one exhibit at a time, looked
    up in a flat map that spans the whole bundle, so its narrow view never
    reaches a supplement's prose and never has to place a crop in a document.
    Naming supplementary material to it would describe a part of the message
    it cannot see, which is the silence the engine's prompts keep everywhere
    else.
    """

    @pytest.fixture
    def briefings(self, config_dir):
        bundle = load_config_bundle(config_dir)
        return {
            "extractor": build_system_message(
                system_prompt_path=bundle.extractor_system_path,
                reference_lists=bundle.reference_lists),
            "reviewer": build_review_system_message(
                system_prompt_path=bundle.review_system_path,
                reference_lists=bundle.reference_lists),
            "checker": build_checker_system_text(
                system_prompt_path=bundle.checker_system_path,
                reference_lists=bundle.reference_lists,
                predicates=stage_predicates(2, True, False),
                max_checks_per_field=2),
        }

    @pytest.mark.parametrize("role", ["extractor", "reviewer"])
    def test_the_briefing_says_a_supplement_arrives_in_its_own_section(
            self, briefings, role):
        text = briefings[role].lower()
        assert "supplementary material" in text, (
            f"the {role} is sent supplement sections but its briefing does "
            "not say supplementary material arrives")
        assert "follows the paper" in text and "section" in text, (
            f"the {role} is not told where a supplement sits in the message "
            "or that it is delimited")

    @pytest.mark.parametrize("role", ["extractor", "reviewer"])
    def test_the_briefing_keeps_a_supplement_out_of_a_quote(
            self, briefings, role):
        # The one rule a role cannot discover except by a refusal: a `<q>` is
        # checked against `text.md` alone, so a supplement's prose supports a
        # field through `notes` or through `<img>` on one of its exhibits.
        text = briefings[role].lower()
        assert "not the paper text" in text, (
            f"the {role} is not told a supplement's prose is not the paper "
            "text, so it will quote from it and be refused")
        assert "<q>" in briefings[role]

    def test_the_checker_is_told_nothing_about_supplements(self, briefings):
        assert "supplement" not in briefings["checker"].lower(), (
            "the checker is shown no supplement section, so naming one to it "
            "describes a part of the message it cannot see")


class TestThePreviewIsTheMessageBlockForBlock:
    """`--dry-run` exists to show what a run would send before it spends, so
    the preview is compared with the message a run actually starts from.

    The comparison is built by reading the SENT blocks: each text block's text
    verbatim, and where an attachment's bytes sit, the label read off the
    block above it — which is how a person checking this report would do it,
    and is not a second call of the code under test. A crop's bytes are the
    only thing the preview cannot hold, and it names each one where it
    attaches.
    """

    def _orch(self, config_dir, bundle_dir, out_dir, **kwargs):
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator

        return Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
            **kwargs)

    @staticmethod
    def _read_back(blocks):
        """What a reader of these blocks would write down as their text view.

        An image block carries the crop's bytes and no label, so the label is
        taken from the `[label]` the block above it opens with — the same
        pairing a role makes when it reads the message.
        """
        parts = []
        for index, block in enumerate(blocks):
            if block["type"] == "text":
                parts.append(block["text"])
                continue
            assert block["type"] == "image"
            above = blocks[index - 1]["text"]
            assert above.startswith("[")
            parts.append(f"(image: {above[1:above.index(']')]}.png)")
        return "\n\n".join(parts)

    def test_the_extractor_preview_is_what_the_run_would_send(
            self, config_dir, bundle_supplemented_dir, tmp_path, capsys):
        previewed = self._orch(config_dir, bundle_supplemented_dir,
                               tmp_path / "preview", dry_run=True)
        report = previewed.dry_run_report()
        printed = capsys.readouterr().out

        run = self._orch(config_dir, bundle_supplemented_dir,
                         tmp_path / "runs")
        run.prepare_new_session()
        sent = run.messages[0]["content"]

        assert report["user_message"] == self._read_back(sent)
        assert report["user_message"] in printed
        # Both crops ride as bytes and are named, not silently dropped.
        assert sum(1 for b in sent if b["type"] == "image") == 2
        assert report["user_message"].count("(image: ") == 2

    def test_the_reviewer_preview_is_what_the_review_would_send(
            self, config_dir, bundle_supplemented_dir, tmp_path, capsys):
        # The reviewer's message differs from the extractor's in its framing
        # and in carrying the extraction output, and its exhibits are guarded
        # on its own model, so it is previewed from its own construction.
        previewed = self._orch(config_dir, bundle_supplemented_dir,
                               tmp_path / "preview", dry_run=True)
        report = previewed.dry_run_report()
        capsys.readouterr()

        run = self._orch(config_dir, bundle_supplemented_dir,
                         tmp_path / "runs")
        sent, _ = run._review_message(run.REVIEW_OUTPUT_PLACEHOLDER)

        assert report["review_user_message"] == self._read_back(sent)
        # Everything but the output is previewed: the supplement's section
        # reaches the reviewer as it reaches the extractor.
        supplement = next(
            iter(load_bundle(bundle_supplemented_dir).supplements.values()))
        assert supplement_open(
            supplement.name, supplement.title) in report[
                "review_user_message"]
        assert supplement.text.strip() in report["review_user_message"]
        # And the one part a dry run cannot know says so rather than showing
        # an empty record as if it were what the reviewer will be sent.
        assert "not previewable" in report["review_user_message"]
