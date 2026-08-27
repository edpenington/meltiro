"""An exhibit's content as text, carried from the bundle into the messages.

The format lets a table exhibit carry its content as `tables/{label}.html`
beside the crop, and `text.md` carries no table content at all — only a
sentinel where the exhibit sits. So without this the only reading of a table
available to any role is the pixels of its crop, and the transcription is what
makes a cell legible as text.

What is tested here is the whole path and the two claims that path rests on:

  - the markup reaches every role that is shown the exhibit, VERBATIM. It is
    not rendered, reflowed or flattened on the way, because a pipe table
    cannot express a header that spans columns or a stub that spans rows, and
    those are what say which column a number sits under.
  - it changes no citation. The crop is still what the exhibit IS, so a fact
    taken from an exhibit is still `<img>label</img>`; nothing here adds a way
    to quote a cell, and a role that tries gets the same refusal it always
    did.

`bundle_transcribed` is the fixture, and the only one shaped the way the
format actually specifies an exhibit. Its transcription carries a spanning
group header, a stub spanning rows, all four `scope` values and two cells the
source leaves empty, so a rendering that dropped structure could not pass the
verbatim assertions below by accident.
"""

import json
import shutil

import pytest
from alteksto.bundle import validate_bundle

from meltiro.bundle import load_bundle, read_transcription
from meltiro.checker_prompts import (
    build_checker_system_text,
    build_checker_user_message,
)
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import bundle_fingerprint
from meltiro.prompt_partials import stage_predicates
from meltiro.prompt_builder import (
    EXHIBIT_TRANSCRIPTION_PREFIX,
    build_initial_user_blocks,
    build_review_system_message,
    build_review_user_blocks,
    build_system_message,
    render_message_text,
    image_label_text,
)
from meltiro.quote_check import find_quote


def _text_blocks(blocks):
    return [b["text"] for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"]


@pytest.fixture
def transcribed(bundle_transcribed_dir):
    return load_bundle(bundle_transcribed_dir)


@pytest.fixture
def markup(bundle_transcribed_dir):
    return (bundle_transcribed_dir / "tables" / "table_01.html").read_text(
        encoding="utf-8").strip()


class TestTheFixtureIsWhatItClaims:
    """Guards on the fixture, so the tests below rest on a real shape."""

    def test_it_is_a_bundle(self, bundle_transcribed_dir):
        assert validate_bundle(bundle_transcribed_dir) == []

    def test_text_md_carries_the_sentinel_and_not_the_content(
            self, transcribed):
        # The format's rule, and the reason a transcription is worth carrying:
        # the table's content is nowhere in the prose a quote is checked
        # against.
        assert "[TABLE 1." in transcribed.text
        assert "|" not in transcribed.text
        for cell in ("5.8 (4.1-7.6)", "212 (38.4)", "76.4"):
            assert cell not in transcribed.text

    def test_the_transcription_carries_what_a_pipe_table_cannot(self, markup):
        assert 'colspan="2"' in markup
        assert 'rowspan="2"' in markup
        # A cell the source leaves empty is empty here, never dropped: dropping
        # it is what slides the rest of the row sideways.
        assert "<td></td>" in markup


class TestTheLoaderCarriesIt:
    def test_tables_maps_label_to_path(self, transcribed):
        assert set(transcribed.tables) == {"table_01"}
        assert transcribed.tables["table_01"].name == "table_01.html"

    def test_a_bundle_with_no_transcription_carries_an_empty_map(
            self, bundle_minimal_dir):
        # Absence is the format's only signal and it is not a defect: it means
        # the crop is the content, which is what every exhibit meant before the
        # directory existed.
        assert load_bundle(bundle_minimal_dir).tables == {}

    def test_the_declared_exhibit_is_unchanged_by_it(self, transcribed):
        # A transcription is a reading of the exhibit, not a second exhibit.
        assert set(transcribed.figures) == {"table_01"} == set(
            transcribed.exhibits)


class TestItReachesEveryRoleVerbatim:
    def test_the_extractor_message_carries_the_markup(
            self, transcribed, markup):
        blocks = build_initial_user_blocks(
            transcribed.study_id, transcribed.text,
            [("table_01", b"png-bytes")],
            transcribed.exhibits, transcribed.exhibit_notes,
            {"table_01": markup})
        assert any(markup in t for t in _text_blocks(blocks))

    def test_the_reviewer_message_carries_the_markup(
            self, transcribed, markup):
        blocks = build_review_user_blocks(
            transcribed.study_id, transcribed.text,
            [("table_01", b"png-bytes")], {"study": {}},
            transcribed.exhibits, transcribed.exhibit_notes,
            {"table_01": markup})
        assert any(markup in t for t in _text_blocks(blocks))

    def test_the_checker_message_carries_the_markup(
            self, transcribed, markup, config_dir):
        blocks = build_checker_user_message(
            field_path="study.turnaround",
            field_spec={"variable": "turnaround", "description": "d",
                        "extraction_instruction": None},
            envelope={"value": "3.9 hours",
                      "evidence": "<img>table_01</img>", "notes": None},
            identity_context="Summary: ctx",
            image_labels={"table_01"},
            partials_dir=config_dir / "prompts" / "partials",
            figures=transcribed.figures,
            exhibit_notes=transcribed.exhibit_notes,
            exhibit_tables={"table_01": markup},
        )
        assert any(markup in t for t in _text_blocks(blocks))

    def test_it_sits_under_the_label_it_belongs_to(self, transcribed, markup):
        # Inside the exhibit's own block, after the label and the footnote, so
        # a role reading several attachments cannot attach it to the wrong one.
        block = image_label_text(
            "table_01", transcribed.exhibits, transcribed.exhibit_notes,
            {"table_01": markup})
        assert block.startswith("[table_01]")
        assert block.index(EXHIBIT_TRANSCRIPTION_PREFIX) > block.index(
            "[table_01]")
        assert block.endswith(markup)

    def test_the_recorded_prompt_matches_the_message(self, transcribed, markup, rendered_user_message):
        # The session captures a text-only render of the initial user message.
        # A transcription in one and not the other would make the record a
        # description of a message that was never sent.
        sent = build_initial_user_blocks(
            transcribed.study_id, transcribed.text,
            [("table_01", b"png-bytes")],
            transcribed.exhibits, transcribed.exhibit_notes,
            {"table_01": markup})
        recorded = rendered_user_message(
            transcribed.study_id, transcribed.text, ["table_01"],
            transcribed.exhibits, transcribed.exhibit_notes,
            {"table_01": markup})
        exhibit_block = [t for t in _text_blocks(sent)
                         if t.startswith("[table_01]")]
        assert len(exhibit_block) == 1
        assert exhibit_block[0] in recorded

    def test_an_exhibit_with_no_transcription_reads_as_it_always_did(
            self, transcribed):
        block = image_label_text(
            "table_01", transcribed.exhibits, transcribed.exhibit_notes, {})
        assert EXHIBIT_TRANSCRIPTION_PREFIX not in block


class TestItChangesNoCitation:
    def test_a_cell_is_still_not_quotable(self, transcribed, markup):
        # The claim the whole feature rests beside: a transcription is shown,
        # and `text.md` remains the only thing a <q> is checked against. A
        # model that copies a cell out of the markup gets the ordinary
        # refusal, which is what sends it to <img> instead.
        assert "212 (38.4)" in markup
        assert not find_quote("212 (38.4)", transcribed.text)

    def test_the_markup_itself_is_not_quotable_either(
            self, transcribed, markup):
        assert not find_quote("<td>212 (38.4)</td>", transcribed.text)


class TestItIsPartOfThePapersIdentity:
    def test_tables_fp_is_reported(self, transcribed):
        fp = bundle_fingerprint(transcribed)
        assert fp["tables_fp"].startswith("tables_fp:")

    def test_a_bundle_with_no_transcription_still_reports_one(
            self, bundle_minimal_dir):
        # "This paper supplies no transcriptions" is a hashed fact, not the
        # digest of an empty payload.
        fp = bundle_fingerprint(load_bundle(bundle_minimal_dir))
        assert fp["tables_fp"].startswith("tables_fp:")

    def test_absent_and_present_do_not_share_a_fingerprint(
            self, transcribed, bundle_minimal_dir):
        other = bundle_fingerprint(load_bundle(bundle_minimal_dir))
        assert bundle_fingerprint(transcribed)["tables_fp"] != \
            other["tables_fp"]

    def test_editing_a_cell_moves_tables_fp_and_bundle_fp_alone(
            self, bundle_transcribed_dir, tmp_path):
        # The property every part of this fingerprint is built for: one input
        # moves one part. A re-transcribed cell is not a re-crop and not an
        # edit to the prose, so it must move neither of those.
        dst = tmp_path / "edited"
        shutil.copytree(bundle_transcribed_dir, dst)
        before = bundle_fingerprint(load_bundle(bundle_transcribed_dir))

        path = dst / "tables" / "table_01.html"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "<td>212 (38.4)</td>", "<td>213 (38.4)</td>"),
            encoding="utf-8")
        after = bundle_fingerprint(load_bundle(dst))

        assert after["tables_fp"] != before["tables_fp"]
        assert after["bundle_fp"] != before["bundle_fp"]
        assert after["text_fp"] == before["text_fp"]
        assert after["figures_fp"] == before["figures_fp"]
        assert after["manifest_fp"] == before["manifest_fp"]

    def test_withdrawing_a_transcription_moves_it_too(
            self, bundle_transcribed_dir, tmp_path):
        # Removing the file is a real change to what the run is shown, and the
        # bundle stays valid without it, so nothing else would report this.
        dst = tmp_path / "withdrawn"
        shutil.copytree(bundle_transcribed_dir, dst)
        before = bundle_fingerprint(load_bundle(bundle_transcribed_dir))
        shutil.rmtree(dst / "tables")

        assert validate_bundle(dst) == []
        after = bundle_fingerprint(load_bundle(dst))
        assert after["tables_fp"] != before["tables_fp"]
        assert after["bundle_fp"] != before["bundle_fp"]
        assert after["figures_fp"] == before["figures_fp"]

    def test_it_is_recorded_as_a_resume_axis(self):
        # A transcription is shown to the model, so re-transcribing between a
        # pause and a resume changes what the run was reading. It is compared
        # on resume for the reason a re-crop is.
        from meltiro.session import BUNDLE_AXES

        assert "tables_fp" in BUNDLE_AXES


class TestTheSessionRecordsWhatTheMessageCarried:
    """End to end, through a real session, over the transcribed fixture."""

    def _orch(self, config_dir, bundle_dir, out_dir):
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator

        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
        )
        orch.prepare_new_session()
        return orch

    def test_the_exhibit_record_says_the_content_rode_with_it(
            self, config_dir, bundle_transcribed_dir, tmp_path):
        # The record kept at every diagnostics level. Without this a lean run
        # could not say afterwards whether a cell was read as text or off
        # pixels, because the rendered prompt that holds the markup is
        # captured only from `standard` up.
        orch = self._orch(config_dir, bundle_transcribed_dir,
                          tmp_path / "runs")
        recorded = json.loads(
            (orch.session.instrument_dir / "image_labels.json").read_text(
                encoding="utf-8"))
        assert [(e["label"], e["transcribed"]) for e in recorded] == [
            ("table_01", True)]

    def test_the_captured_prompt_holds_the_markup(
            self, config_dir, bundle_transcribed_dir, tmp_path, markup):
        orch = self._orch(config_dir, bundle_transcribed_dir,
                          tmp_path / "runs")
        captured = (orch.session.instrument_dir / "user_prompt.txt").read_text(
            encoding="utf-8")
        assert markup in captured

    def test_the_message_actually_sent_holds_it_too(
            self, config_dir, bundle_transcribed_dir, tmp_path, markup):
        # The capture above is a render. This is the conversation the run
        # would send, so the two cannot be asserted apart.
        orch = self._orch(config_dir, bundle_transcribed_dir,
                          tmp_path / "runs")
        sent = "\n".join(
            b.get("text", "")
            for m in orch.messages
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text")
        assert markup in sent


class TestTheTextOnlyRoleIsUnchanged:
    def test_a_role_sent_no_exhibits_is_sent_no_transcription(
            self, transcribed, markup):
        # A transcription rides with the crop it transcribes. A role that is
        # sent no images has no `<img>` label that resolves, so content it
        # could not cite would only invite evidence it cannot supply.
        blocks = build_initial_user_blocks(
            transcribed.study_id, transcribed.text, [],
            transcribed.exhibits, transcribed.exhibit_notes,
            {"table_01": markup})
        joined = "\n".join(_text_blocks(blocks))
        assert markup not in joined
        assert EXHIBIT_TRANSCRIPTION_PREFIX not in joined


class TestEveryRoleShownOneIsBriefedOnIt:
    """A role sent the markup is told in its own briefing that it arrives.

    Three roles are shown the transcription by three different builders, and
    the parity is what is asserted here rather than any one role's wording: a
    role shown a table's content under a label its briefing describes as a
    crop and a footnote has to infer what the extra markup is, and the reading
    nearest to hand is that a cell has become quotable.

    The rule against that inference is asserted only for the two roles that
    WRITE evidence. The checker writes none — it reads one field's evidence
    and answers whether it supports the value — so a rule about what may go
    in a `<q>` is not its business, and its briefing frames the content as the
    exhibit's own words instead.
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

    @pytest.mark.parametrize("role", ["extractor", "reviewer", "checker"])
    def test_the_briefing_says_the_content_arrives_as_text(
            self, briefings, role):
        text = briefings[role].lower()
        assert "content as text" in text, (
            f"the {role} is sent an exhibit's transcription but its briefing "
            "does not say the content arrives")
        assert "html table" in text, (
            f"the {role} is not told what form the content arrives in")

    @pytest.mark.parametrize("role", ["extractor", "reviewer"])
    def test_the_briefing_keeps_a_cell_out_of_a_quote(self, briefings, role):
        # The instruction and its mechanism, and no claim about what a
        # particular bundle's text.md contains: inline one table row into it
        # and a cell quote validates, so a briefing that said a cell CANNOT
        # be quoted would be asserting something the validator does not
        # enforce.
        text = briefings[role]
        assert "cited as `<img>label</img>` and never quoted" in text, (
            f"the {role} writes evidence and is shown a table's content as "
            "text, so it has to be told how to cite it")
        assert "a `<q>` is resolved against the paper text alone" in text, (
            f"the {role} is not told what a quote is checked against")


class TestTheHashCoversWhatTheRoleReads:
    """`tables_fp` digests the transcription as a role is shown it.

    A crop's bytes ARE the crop, so `figures_fp` hashes the file. A
    transcription's file has surrounding whitespace that no message carries
    and no role can observe, so hashing the file would move the paper's
    identity — and refuse a resume — for a change to something nobody read,
    while two bundles a role cannot tell apart would carry different numbers.
    """

    def test_whitespace_around_the_markup_moves_nothing(
            self, bundle_transcribed_dir, tmp_path):
        dst = tmp_path / "padded"
        shutil.copytree(bundle_transcribed_dir, dst)
        path = dst / "tables" / "table_01.html"
        path.write_text("\n\n" + path.read_text(encoding="utf-8") + "\n   \n",
                        encoding="utf-8")

        assert validate_bundle(dst) == []
        before = bundle_fingerprint(load_bundle(bundle_transcribed_dir))
        after = bundle_fingerprint(load_bundle(dst))
        assert after["tables_fp"] == before["tables_fp"]
        assert after["bundle_fp"] == before["bundle_fp"]

    def test_a_cell_edited_inside_the_markup_still_moves_it(
            self, bundle_transcribed_dir, tmp_path):
        # The control: the reader strips, it does not normalise.
        dst = tmp_path / "edited"
        shutil.copytree(bundle_transcribed_dir, dst)
        path = dst / "tables" / "table_01.html"
        path.write_text(
            path.read_text(encoding="utf-8").replace("<td>", "<td> ", 1),
            encoding="utf-8")
        assert bundle_fingerprint(load_bundle(dst))["tables_fp"] != \
            bundle_fingerprint(load_bundle(bundle_transcribed_dir))["tables_fp"]

    def test_the_digest_is_over_the_string_the_message_carries(
            self, bundle_transcribed_dir):
        # Stated directly, so the two cannot drift apart through their
        # separate call sites.
        import hashlib

        bundle = load_bundle(bundle_transcribed_dir)
        text = read_transcription(bundle.tables["table_01"])
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        blocks = build_initial_user_blocks(
            bundle.study_id, bundle.text, [("table_01", b"png")],
            bundle.exhibits, bundle.exhibit_notes, {"table_01": text})
        # The message carries that string, and the pair the axis is built
        # from carries its digest — the two ends of the same reading.
        from meltiro.fingerprint import _transcription_digests

        assert any(text in b["text"] for b in blocks
                   if b.get("type") == "text")
        assert _transcription_digests(bundle.tables) == [
            ("table_01", expected)]


class TestEachRolesWiringIsPinnedAtTheRun:
    """The markup reaches each role from the ORCHESTRATOR, not just from a
    builder called with a hand-made map.

    The builders are tested above with an `exhibit_tables` argument supplied by
    the test. That proves the builder emits what it is given and says nothing
    about whether a run gives it anything: dropping the map at any of the three
    call sites left the whole suite green. These are the three call sites.
    """

    def _orch(self, config_dir, bundle_dir, out_dir):
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator

        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            max_checks_per_field=2,
            extractor_max_tokens=4096, review_max_tokens=4096,
        )
        orch.prepare_new_session()
        return orch

    def test_the_extractor_is_sent_it(self, config_dir,
                                     bundle_transcribed_dir, tmp_path, markup):
        orch = self._orch(config_dir, bundle_transcribed_dir,
                          tmp_path / "runs")
        assert markup in "\n".join(
            b.get("text", "") for b in orch.messages[0]["content"]
            if isinstance(b, dict))

    def test_the_reviewer_is_sent_it(self, config_dir, bundle_transcribed_dir,
                                    tmp_path, markup):
        orch = self._orch(config_dir, bundle_transcribed_dir,
                          tmp_path / "runs")
        blocks, _ = orch._review_message({"study": {}})
        assert markup in "\n".join(
            b.get("text", "") for b in blocks if isinstance(b, dict))

    def test_the_checker_is_sent_it_for_the_exhibit_it_checks(
            self, config_dir, bundle_transcribed_dir, tmp_path, markup):
        from meltiro.extraction_record import ExtractionRecord

        orch = self._orch(config_dir, bundle_transcribed_dir,
                          tmp_path / "runs")
        record = ExtractionRecord()
        record.apply_update_study(study={
            "sample_size": {"value": 402,
                            "evidence": "<img>table_01</img>"},
        })
        orch.extraction_record = record
        calls, _ = orch._build_checker_calls(["study.sample_size"])
        assert markup in "\n".join(
            b.get("text", "") for b in calls[0]["user_message_blocks"]
            if isinstance(b, dict))

    def test_the_briefings_name_the_marker_the_message_writes(
            self, config_dir):
        # The briefings tell a role the content arrives "under `Content as
        # text:`". That is the constant the builder writes, and nothing tied
        # the two: renaming the constant left every test passing while three
        # briefings went on naming a marker no message contained.
        from meltiro.config_bundle import load_config_bundle

        bundle = load_config_bundle(config_dir)
        for text in (
            build_system_message(
                system_prompt_path=bundle.extractor_system_path,
                reference_lists=bundle.reference_lists),
            build_review_system_message(
                system_prompt_path=bundle.review_system_path,
                reference_lists=bundle.reference_lists),
        ):
            assert EXHIBIT_TRANSCRIPTION_PREFIX in text


class TestTheAbsentStageSentinelIsLoadBearing:
    """"This paper supplies none" is a hashed FACT, not the digest of an empty
    payload.

    Removing the sentinel and hashing `canonical_json([])` instead passed
    every test, because the only comparison was against a bundle that DOES
    transcribe — and an empty list hashes differently from a full one either
    way. What the sentinel is for is that "none" is stated rather than
    computed, so the preimage is the same fixed token for every paper that
    supplies nothing, whatever its shape.
    """

    def test_no_transcriptions_hashes_the_sentinel_itself(
            self, bundle_minimal_dir):
        import hashlib

        from meltiro.fingerprint import ABSENT_STAGE

        fps = bundle_fingerprint(load_bundle(bundle_minimal_dir))
        assert load_bundle(bundle_minimal_dir).tables == {}
        expected = hashlib.sha256(ABSENT_STAGE.encode()).hexdigest()
        assert fps["tables_fp"] == f"tables_fp:{expected}"

    def test_no_supplements_hashes_it_too(self, bundle_minimal_dir):
        import hashlib

        from meltiro.fingerprint import ABSENT_STAGE

        fps = bundle_fingerprint(load_bundle(bundle_minimal_dir))
        expected = hashlib.sha256(ABSENT_STAGE.encode()).hexdigest()
        assert fps["supplements_fp"] == f"supplements_fp:{expected}"

    def test_the_sentinel_is_not_what_an_empty_payload_hashes_to(self):
        # The distinction the sentinel buys: the two preimages differ, so a
        # bundle that supplies none can never collide with one whose payload
        # happens to serialise to nothing.
        import hashlib

        from meltiro.fingerprint import ABSENT_STAGE, canonical_json

        assert hashlib.sha256(ABSENT_STAGE.encode()).hexdigest() != \
            hashlib.sha256(canonical_json([]).encode()).hexdigest()


class TestTheProjectionRefusesAMismatchedPair:
    """`render_message_text` pairs a label to each image block by position, so
    a caller handing it the wrong sequence would name the wrong crop.

    Both guards were deletable with the suite green: nothing called the
    function directly, and no caller inside this package can supply a
    mismatched pair — which is exactly why the guard is what stops the next
    one from doing it silently. Without them a short sequence writes
    `(image: None.png)`.
    """

    def _blocks(self, count):
        return [{"type": "text", "text": "before"}] + [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": ""}}
            for _ in range(count)]

    def test_too_few_labels_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            render_message_text(self._blocks(2), ["only_one"])
        assert "more images than the figure sequence" in str(excinfo.value)

    def test_leftover_labels_are_refused(self):
        with pytest.raises(ValueError) as excinfo:
            render_message_text(self._blocks(1), ["one", "two"])
        assert "labels the message does not attach" in str(excinfo.value)

    def test_an_unrenderable_block_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            render_message_text([{"type": "tool_use"}], [])
        assert "unrenderable content block" in str(excinfo.value)

    def test_a_matched_pair_names_each_crop_where_it_attaches(self):
        text = render_message_text(self._blocks(2), ["first", "second"])
        assert text.index("(image: first.png)") < text.index(
            "(image: second.png)")


class TestTheMessageAndTheHashReadOneString:
    """The single-reader invariant, guarded on the side that decides what the
    model sees.

    Reverting the FINGERPRINT to a raw read fails two tests. Reverting the
    MESSAGE to a raw read passed the whole suite, because every wiring
    assertion is `markup in message` against a fixture that is already
    stripped — a substring a raw read satisfies too. That is the dangerous
    direction: the roles read bytes the axis does not cover, so a resume
    admits material they will read differently.
    """

    def test_the_orchestrator_holds_exactly_what_the_axis_digests(
            self, config_dir, bundle_transcribed_dir, tmp_path):
        import hashlib

        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.fingerprint import _transcription_digests
        from meltiro.orchestrator import Orchestrator

        # Padded on disk, so a raw read and the reader disagree.
        dst = tmp_path / "padded"
        shutil.copytree(bundle_transcribed_dir, dst)
        path = dst / "tables" / "table_01.html"
        raw = "\n\n" + path.read_text(encoding="utf-8") + "\n   \n"
        path.write_text(raw, encoding="utf-8")

        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(dst),
            tmp_path / "runs",
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
        )
        held = orch.image_tables["table_01"]
        assert held != raw, "the orchestrator read the file raw"
        digested = dict(_transcription_digests(load_bundle(dst).tables))
        assert digested["table_01"] == hashlib.sha256(
            held.encode("utf-8")).hexdigest()

    def test_the_message_carries_that_string_and_not_the_file(
            self, config_dir, bundle_transcribed_dir, tmp_path):
        from meltiro.checker import CheckerConfig
        from meltiro.config_bundle import load_config_bundle
        from meltiro.orchestrator import Orchestrator

        dst = tmp_path / "padded"
        shutil.copytree(bundle_transcribed_dir, dst)
        path = dst / "tables" / "table_01.html"
        path.write_text("\n\n" + path.read_text(encoding="utf-8") + "\n \n",
                        encoding="utf-8")

        orch = Orchestrator(
            load_config_bundle(config_dir), load_bundle(dst),
            tmp_path / "runs",
            extractor_model="claude-opus-4-8",
            checker_config=CheckerConfig(max_tokens=1024,
                                         checker_model="claude-sonnet-4-6"),
            review_model="claude-opus-4-8",
            extractor_max_tokens=4096, review_max_tokens=4096,
        )
        orch.prepare_new_session()
        block = next(
            b["text"] for b in orch.messages[0]["content"]
            if isinstance(b, dict) and b.get("type") == "text"
            and b["text"].startswith("[table_01]"))
        # The block ends where the transcription ends: a raw read would leave
        # the file's trailing blank lines inside the message.
        assert block == block.rstrip()
        assert block.endswith("</table>")
