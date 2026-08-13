"""The checker round a dry run previews.

The checker is the one stage of a run whose question an operator cannot read
before paying for it. Its system prompt renders like any other, but the message
that carries the actual question is assembled per field at call time, out of a
scaffold, one template field, and whatever the extractor wrote into that field.

So a dry run renders both halves of it. The SCAFFOLD is shown as itself, slot
tokens and all: it is the text an author overrides, and an override is written
against those tokens. The SPECIMEN ROUND is that scaffold filled in for a real
field of the loaded template, so the shape of a whole check can be read at
once. Everything in the specimen that a live check would read off the
extraction or the paper is written by the engine and says so, because a preview
whose sample content could be mistaken for a paper's own is worse than no
preview.

The two are not the same kind of artefact, and a session keeps only one of
them: the scaffold, which its checks really were rendered from. Nothing in a
run was ever asked through the specimen.

Everything here is offline: no provider call is made or stubbed, because a dry
run makes none and a prepared session makes none before its first turn.
"""

import re
import shutil

import pytest

from meltiro import checker_prompts
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.checker_prompts import SAMPLE_MARKER, sample_checker_field
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.template import load_template


# The orchestrator's pre-spend key preflight runs before a session is created,
# so the module that prepares one needs every stage's key variable present.
pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"

SCAFFOLD_FILE = "checker_user_scaffold.md"
SPECIMEN_FILE = "checker_round_sample.md"

# The scaffold slots a live check fills from the extraction and the paper, and
# that the specimen therefore fills with engine-written content. Every other
# slot takes the template's own text — the field, its description, what it
# may hold — which no preview fabricates.
FABRICATED_SLOTS = ("identity_context", "evidence_block", "value",
                    "notes_block")

_SLOT = re.compile(r"\{([a-z_]+)\}")

# What may follow the marker at the end of a fabricated string: the quote marks
# a string value is rendered inside, and the sentence punctuation the sample
# paper text carries. Never a bracket — the marker itself ends in one.
_CLOSERS = " \t\n\"'`.,;:!?"


def _rendered_slots(scaffold, specimen):
    """What the specimen holds where the scaffold holds each slot.

    Carved out of the two artefacts themselves: the scaffold's literal text
    between its slots is the anchor, so this reads whatever slots the scaffold
    declares rather than a list kept here that could fall behind it.
    """
    pattern, names, pos = [], [], 0
    for slot in _SLOT.finditer(scaffold):
        pattern.append(re.escape(scaffold[pos:slot.start()]))
        pattern.append(f"(?P<{slot.group(1)}>.*?)")
        names.append(slot.group(1))
        pos = slot.end()
    pattern.append(re.escape(scaffold[pos:]))
    filled = re.fullmatch("".join(pattern), specimen, re.DOTALL)
    assert filled, ("the specimen is not this scaffold with its slots filled "
                    "in, so nothing here is reading the slot it names")
    return {name: filled.group(name) for name in names}


def _orch(config_dir, bundle_dir, out_dir, *, max_checks_per_field=2,
          dry_run=False):
    """An orchestrator with the reviewer off, so the checker is the subject."""
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(max_tokens=1024, checker_model=CHECKER),
        review_model=None,
        max_checks_per_field=max_checks_per_field, final_review=False,
        extractor_max_tokens=4096, dry_run=dry_run,
    )


def _report(config_dir, bundle_dir, tmp_path, **kwargs):
    """Run a dry run into `tmp_path/dry_run`; return `(report, report_dir)`."""
    report_dir = tmp_path / "dry_run"
    orch = _orch(config_dir, bundle_dir, tmp_path / "runs", dry_run=True,
                 **kwargs)
    return orch.dry_run_report(report_dir=report_dir), report_dir


def _copy_bundle(tmp_path, config_dir):
    dest = tmp_path / "config"
    shutil.copytree(config_dir, dest)
    return dest


# ---------------------------------------------------------------------------
# What a dry run writes
# ---------------------------------------------------------------------------

class TestBothArtefactsArePartOfTheReport:
    def test_they_land_beside_the_prompts(self, config_dir,
                                          bundle_minimal_dir, tmp_path):
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        names = {p.name for p in report_dir.iterdir()}
        assert {SCAFFOLD_FILE, SPECIMEN_FILE} <= names

    def test_a_checker_off_run_writes_neither(self, config_dir,
                                              bundle_minimal_dir, tmp_path):
        # The report is one run's, and a stage that does not run contributes
        # nothing to it: a scaffold on disk for a run with no checker would
        # describe a message nothing sends.
        report, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path,
                                     max_checks_per_field=0)
        names = {p.name for p in report_dir.iterdir()}
        assert SCAFFOLD_FILE not in names
        assert SPECIMEN_FILE not in names
        assert report["checker_user_scaffold"] is None
        assert report["checker_round_sample"] is None

    def test_both_are_printed_untruncated(self, config_dir,
                                          bundle_minimal_dir, tmp_path,
                                          capsys):
        report, _ = _report(config_dir, bundle_minimal_dir, tmp_path)
        out = capsys.readouterr().out
        assert "=== CHECKER USER SCAFFOLD ===" in out
        assert "=== CHECKER ROUND, ONE SPECIMEN FIELD ===" in out
        assert report["checker_user_scaffold"] in out
        assert report["checker_round_sample"] in out

    def test_nothing_is_printed_for_a_checker_that_does_not_run(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        _report(config_dir, bundle_minimal_dir, tmp_path,
                max_checks_per_field=0)
        out = capsys.readouterr().out
        assert "CHECKER USER SCAFFOLD" not in out
        assert "CHECKER ROUND" not in out


# ---------------------------------------------------------------------------
# The scaffold, shown as itself
# ---------------------------------------------------------------------------

class TestTheScaffoldIsTheTextAnAuthorOverrides:
    def test_its_slots_are_left_standing(self, config_dir,
                                         bundle_minimal_dir, tmp_path):
        # Deliberately not a filled message: the slots are the interface an
        # override is written against, so the artefact that exists to be
        # overridden has to show them.
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        text = (report_dir / SCAFFOLD_FILE).read_text(encoding="utf-8")
        for slot in ("{field_path}", "{field_description}", "{value}",
                     "{evidence_block}", "{identity_context}"):
            assert slot in text

    def test_an_override_is_what_the_artefact_shows(self, config_dir,
                                                    bundle_minimal_dir,
                                                    tmp_path):
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        override_dir = bundle_dir / "prompts" / "partials" / "meltiro"
        override_dir.mkdir(parents=True, exist_ok=True)
        (override_dir / "checker_user.md").write_text(
            "FIELD {field_path}\nVALUE {value}", encoding="utf-8")
        _, report_dir = _report(bundle_dir, bundle_minimal_dir,
                                tmp_path / "run")
        text = (report_dir / SCAFFOLD_FILE).read_text(encoding="utf-8")
        assert text.startswith("FIELD {field_path}")
        assert "## Field under review" not in text

    def test_the_specimen_is_built_from_that_same_override(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The two artefacts are one text and one filling of it, so an override
        # that changes the first has to change the second.
        bundle_dir = _copy_bundle(tmp_path, config_dir)
        override_dir = bundle_dir / "prompts" / "partials" / "meltiro"
        override_dir.mkdir(parents=True, exist_ok=True)
        (override_dir / "checker_user.md").write_text(
            "FIELD {field_path}\nVALUE {value}", encoding="utf-8")
        _, report_dir = _report(bundle_dir, bundle_minimal_dir,
                                tmp_path / "run")
        specimen = (report_dir / SPECIMEN_FILE).read_text(encoding="utf-8")
        assert specimen.startswith("FIELD study.title")
        assert "## Field under review" not in specimen


# ---------------------------------------------------------------------------
# The specimen round
# ---------------------------------------------------------------------------

class TestTheSpecimenUsesARealField:
    def test_the_field_and_its_definition_are_the_templates_own(
            self, config_dir, bundle_minimal_dir, tmp_path):
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        text = (report_dir / SPECIMEN_FILE).read_text(encoding="utf-8")
        assert "`study.title`" in text
        assert "Full paper title" in text

    def test_the_choice_is_the_first_field_that_must_carry_evidence(
            self, config_dir):
        # Not simply the first field: the fixture's first study field takes
        # optional evidence, and a check is about evidence, so the preview
        # opens on a field that has to have some.
        template = load_template(config_dir / "extraction_template.yaml")
        assert template["study_fields"][0]["fields"][0]["variable"] == \
            "study_label"
        assert sample_checker_field(template)[0] == "study.title"

    def test_it_falls_back_when_a_template_requires_evidence_nowhere(self):
        template = {
            "study_fields": [{"section": "S", "fields": [
                {"variable": "verdict", "description": "A judgement",
                 "evidence": "optional"}]}],
            "record_fields": [],
            "record_entity": {"singular": "record"},
            "checker_context_fields": [],
        }
        assert sample_checker_field(template)[0] == "study.verdict"

    def test_a_record_scoped_choice_reads_as_a_real_record_would(self):
        # The label form is the engine's and the template's: the id the engine
        # would mint for a first record, and the context fields the template
        # names beside it.
        template = {
            "study_fields": [],
            "record_fields": [{"section": "S", "fields": [
                {"variable": "effect_size", "description": "The estimate",
                 "evidence": "required"}]}],
            "record_entity": {"singular": "relationship"},
            "checker_context_fields": ["outcome"],
        }
        path, spec, record = sample_checker_field(template)
        assert path == "record.relationship_1.effect_size"
        assert spec["description"] == "The estimate"
        assert record["record_id"] == "relationship_1"
        assert SAMPLE_MARKER in record["outcome"]["value"]

    def test_a_template_with_no_envelope_field_previews_no_round(self):
        template = {"study_fields": [], "record_fields": [],
                    "record_entity": {"singular": "record"},
                    "checker_context_fields": []}
        assert sample_checker_field(template) is None


class TestNothingInTheSpecimenReadsAsRealContent:
    """The guarantee is about the ARTEFACT, not about the constants behind it.

    A preview whose sample content could be mistaken for a paper's own is
    worse than no preview, and the constants are only one route into it: text
    composed at the render site, or added around a constant there, reaches an
    operator's screen having passed through no `_SAMPLE_*` name at all. So the
    marker is asserted over the rendered round — slot by slot, and paragraph
    by paragraph — with the constants kept as the fast first line.
    """

    def test_every_value_the_engine_writes_says_it_is_a_sample(self):
        # The property at its source, so a value added later without a marker
        # fails here rather than shipping into a preview an operator reads as
        # a paper's own words.
        written = {name: value for name, value in vars(checker_prompts).items()
                   if name.startswith("_SAMPLE_") and isinstance(value, str)}
        assert written
        for name, value in written.items():
            assert SAMPLE_MARKER in value, name

    @pytest.mark.parametrize("slot", FABRICATED_SLOTS)
    def test_each_fabricated_slot_ends_in_the_marker(
            self, config_dir, bundle_minimal_dir, tmp_path, slot):
        """Where a live check would carry the paper's content, the specimen
        carries the engine's, and the last thing an operator reads in that slot
        says so. Ending with it rather than merely containing it is what
        catches content written around a marked constant: a sentence appended
        after the marker leaves the slot finishing on words nothing labels."""
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        slots = _rendered_slots(
            (report_dir / SCAFFOLD_FILE).read_text(encoding="utf-8"),
            (report_dir / SPECIMEN_FILE).read_text(encoding="utf-8"))
        assert slot in slots, "the scaffold no longer carries this slot"
        rendered = slots[slot].rstrip(_CLOSERS)
        assert rendered, f"{slot} rendered empty, so it asserts nothing"
        assert rendered.endswith(SAMPLE_MARKER), (
            f"the specimen's {slot} ends on content nothing marks as the "
            f"engine's: ...{slots[slot][-120:]!r}")

    def test_no_marked_paragraph_trails_off_into_unmarked_content(
            self, config_dir, bundle_minimal_dir, tmp_path):
        """The same property inside a slot. A slot holds engine framing as
        well as sample content (the quote-context lead-in, the note's
        preamble), and framing carries no marker and needs none. What may not
        happen is a paragraph that IS sample content running on into content
        that is not: once a paragraph carries the marker, the marker is the
        last thing in it."""
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        specimen = (report_dir / SPECIMEN_FILE).read_text(encoding="utf-8")
        marked = [p for p in specimen.split("\n\n") if SAMPLE_MARKER in p]
        assert len(marked) >= 4, "too few sample paragraphs to be reading one"
        for paragraph in marked:
            assert paragraph.rstrip(_CLOSERS).endswith(SAMPLE_MARKER), (
                "a paragraph of sample content runs on into content nothing "
                f"marks as the engine's: {paragraph!r}")

    @pytest.mark.parametrize("name", ["_SAMPLE_SUMMARY", "_SAMPLE_QUOTE",
                                      "_SAMPLE_VALUE", "_SAMPLE_NOTE"])
    def test_each_of_them_reaches_the_rendered_round(
            self, config_dir, bundle_minimal_dir, tmp_path, name):
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        text = (report_dir / SPECIMEN_FILE).read_text(encoding="utf-8")
        assert getattr(checker_prompts, name) in text

    def test_the_quote_is_shown_in_a_window_of_sample_paper_text(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The window is half of what a checker is really given, so a preview
        # that skipped it would show a check narrower than the one that runs.
        _, report_dir = _report(config_dir, bundle_minimal_dir, tmp_path)
        text = (report_dir / SPECIMEN_FILE).read_text(encoding="utf-8")
        assert "_The quote in context_" in text
        assert checker_prompts._SAMPLE_PAPER_TEXT.split(". ")[0] in text


# ---------------------------------------------------------------------------
# What a session keeps
# ---------------------------------------------------------------------------

class TestTheSessionRecordsTheScaffoldAndNotTheSpecimen:
    def _session(self, config_dir, bundle_dir, tmp_path, **kwargs):
        orch = _orch(config_dir, bundle_dir, tmp_path / "runs", **kwargs)
        orch.prepare_new_session()
        return orch

    def test_the_scaffold_is_captured_beside_the_system_prompts(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = self._session(config_dir, bundle_minimal_dir, tmp_path)
        captured = (orch.session.instrument_dir /
                    "checker_user_scaffold.txt").read_text(encoding="utf-8")
        assert captured == orch.instrument.render_checker_user_scaffold()
        assert "{field_path}" in captured

    def test_the_specimen_is_not_captured(self, config_dir,
                                          bundle_minimal_dir, tmp_path):
        # A run record holds what the run asked. The specimen is an aid to
        # reading the scaffold, and no check was ever asked through it.
        orch = self._session(config_dir, bundle_minimal_dir, tmp_path)
        names = {p.name for p in orch.session.instrument_dir.iterdir()}
        assert not [n for n in names if "round" in n or "sample" in n]

    def test_a_checker_off_run_captures_no_scaffold(self, config_dir,
                                                    bundle_minimal_dir,
                                                    tmp_path):
        orch = self._session(config_dir, bundle_minimal_dir, tmp_path,
                             max_checks_per_field=0)
        assert not (orch.session.instrument_dir /
                    "checker_user_scaffold.txt").exists()
