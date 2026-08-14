"""`diagnostics/transcript.md`, and the `meltiro transcript` subcommand.

The transcript is the instrument an operator reads to check the pipeline is
working, so what is on trial here is the DOCUMENT rather than the
renderer's internals: a real session is driven through the offline
harness (a real Session, dispatcher and extraction record; the provider
adapter, `_call_extractor` and the checker fan-out stubbed), rendered,
and then read.

Three claims carry the most weight.

First, completeness: every turn in order, every tool result's applied AND
rejected fields, and every checker verdict beside the field it judged, with its
rationale. A field's life has to be legible where it happened.

Second, the checker's system prompt appears exactly ONCE. It is one string for
the whole run and a run can make hundreds of checks, so reprinting it per check
would bury the document.

Third, honest degradation. A session that kept less says what it cannot show,
in place of the section it would have rendered, rather than failing or quietly
omitting it.

And one property that keeps the two copies from drifting: what `extract`
writes at a stop is byte-identical to what the subcommand renders from the same
session afterwards.
"""

import copy
import json
import re
from types import SimpleNamespace

import pytest

from direktoro import NormalisedResponse, NormalisedUsage
from meltiro import checker as checker_mod
from meltiro import cli
from meltiro import orchestrator as orch_mod
from meltiro.bundle import load_bundle
from meltiro.checker import CHECKER_TOOL_REPROMPT, CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import SessionError
from meltiro.orchestrator import Orchestrator
from meltiro.transcript import render_transcript


# Every stage's key variable is present for this module: these tests
# reach the orchestrator's pre-spend key preflight, and the provider
# calls behind it are stubbed.
pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"
CHECKER = "claude-sonnet-4-6"
REVIEWER = "claude-opus-4-8"

# A verbatim substring of tests/fixtures/bundle_minimal/text.md, so evidence
# citing it passes the quote check and the write lands.
QUOTE = "A synthetic study of baseline CRT-HD scores"
SHORT_TITLE = "A synthetic study"
FULL_TITLE = "A synthetic study of baseline CRT-HD scores"
REVIEWED_TITLE = "A synthetic study of baseline CRT-HD scores (reviewed)"

# The shipped template's one REQUIRED quality-check variable. `mark_complete`
# takes the caller's quality check as a required argument, so the scripted
# reviewer conclusion below has to carry one.
QUALITY_CHECK = {"deviation_from_expectations":
                 "one relationship, as expected"}


# ---------------------------------------------------------------------------
# Offline harness
# ---------------------------------------------------------------------------

def _tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name,
                           input=tool_input)


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _resp(*blocks):
    return SimpleNamespace(content=list(blocks))


def _review_resp(*blocks):
    return NormalisedResponse(
        content=list(blocks), usage=NormalisedUsage(),
        resolved_model=REVIEWER, provider="anthropic", base_url=None,
        raw_request={"model": REVIEWER}, raw_response={},
        wire_request={"model": REVIEWER}, decoding_params={"max_tokens": 1024},
    )


class _ScriptedAdapter:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create_message(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


def _orch(config_dir, bundle_dir, out_dir, *, max_checks_per_field=2,
          final_review=False, review_model=None, diagnostics="standard"):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(checker_model=CHECKER,
                                     max_tokens=1024),
        review_model=review_model,
        max_checks_per_field=max_checks_per_field,
        final_review=final_review,
        max_tool_calls=50,
        diagnostics=diagnostics,
        extractor_max_tokens=4096,
        review_max_tokens=4096,
    )


def _drive(orch, turns):
    """Play `turns` in order, latching mark_complete on the last one so the
    extractor loop ends cleanly. The last turn must be read-only: a write
    clears the flag as it lands.

    The extractor's initial-check ordering gate is opened directly rather than
    by scripting the `record_initial_check` turn that opens it. What is on
    trial here is the DOCUMENT, and these tests read it by call number and turn
    number ("#### Call 1. `update_study`", "### Turn 1"); an extra opening call
    would renumber all of it without exercising anything the renderer does
    differently.
    """
    orch.extraction_record.initial_check_recorded = True
    played = []

    def _call(adapter, tool_defs):
        idx = min(len(played), len(turns) - 1)
        played.append(idx)
        if idx == len(turns) - 1:
            orch.extraction_record.mark_complete()
        return turns[idx]

    orch._call_extractor = _call


def _stub_fanout(monkeypatch, verdicts_per_call):
    """One scripted (verdict, rationale) per fan-out call; the last repeats."""
    run = [0]

    def _batch(*, calls, config, on_complete=None, api_logger=None, **kw):
        idx = min(run[0], len(verdicts_per_call) - 1)
        run[0] += 1
        verdict, rationale = verdicts_per_call[idx]
        return {c["field_path"]: {
            "verdict": verdict, "rationale": rationale,
            "notes": "Judged against the quoted header only.",
            "error_origin": False, "input_tokens": 812, "output_tokens": 96,
            "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "cost_usd": 0.001234,
        } for c in calls}

    monkeypatch.setattr(orch_mod, "run_checker_batch", _batch)


# The extractor's script: a partial write (one field applied, one rejected on
# a quote that is not in the paper), a revision answering the challenge, then a
# read-only call that ends the loop.
_BAD_QUOTE_CALL = {"study": {
    "title": {"value": SHORT_TITLE, "evidence": f"<q>{QUOTE}</q>",
              "notes": "Shortened from the header for brevity."},
    "doi": {"value": "not-a-doi", "evidence": "<q>nowhere in the paper</q>"},
}}

_REVISION_CALL = {"study": {
    "title": {"value": FULL_TITLE, "evidence": f"<q>{QUOTE}</q>"},
}}


def _run_session(config_dir, bundle_dir, tmp_path, monkeypatch, *,
                 diagnostics="standard", final_review=True):
    """Drive one full offline session and return its Orchestrator."""
    orch = _orch(config_dir, bundle_dir, tmp_path / "runs",
                 max_checks_per_field=2, final_review=final_review,
                 review_model=REVIEWER if final_review else None,
                 diagnostics=diagnostics)
    orch.prepare_new_session()
    _drive(orch, [
        _resp(_text("Reading the paper. I will start with the study block."),
              _tool_use("t1", "update_study", _BAD_QUOTE_CALL)),
        _resp(_text("The checker is right: the header gives the full title."),
              _tool_use("t2", "update_study", _REVISION_CALL)),
        _resp(_tool_use("t3", "view_summary", {})),
    ])
    _stub_fanout(monkeypatch, [
        ("challenge", "The quote gives the full title; the stored value is a "
                      "shortened paraphrase of it."),
        ("ok", "The value now matches the quoted header exactly."),
    ])
    review_adapter = _ScriptedAdapter([
        _review_resp(
            _text("The title can be more precise. Revising, then confirming."),
            _tool_use("r1", "update_study", {"study": {
                "title": {"value": REVIEWED_TITLE,
                          "evidence": f"<q>{QUOTE}</q>"}}})),
        _review_resp(_tool_use("r2", "mark_complete",
                               {"quality_check": dict(QUALITY_CHECK)})),
    ])
    orch._adapter_for_role = lambda role: review_adapter
    assert orch.run() == "complete"
    return orch


@pytest.fixture
def session(config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
    return _run_session(config_dir, bundle_minimal_dir, tmp_path, monkeypatch)


@pytest.fixture
def document(session):
    return (session.session.session_dir /
            "diagnostics" / "transcript.md").read_text(encoding="utf-8")


def _order(document, *needles):
    """Assert each needle appears, and in the order given."""
    positions = []
    for needle in needles:
        idx = document.find(needle)
        assert idx != -1, f"not in the document: {needle!r}"
        positions.append(idx)
    assert positions == sorted(positions), (
        f"out of order: {list(zip(needles, positions))}")
    return positions


def _between(document, start, end):
    """The slice of the document between two markers."""
    a = document.index(start)
    b = document.index(end, a)
    return document[a:b]


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

class TestTheDocument:
    def test_every_turn_is_present_in_order(self, document):
        _order(
            document,
            "## 1. The run",
            "## 2. The instrument",
            "## 3. The extraction, turn by turn",
            "### Turn 1",
            "### Turn 2",
            "### Turn 3",
            "## 4. The review",
            "### Reviewer turn 4",
            "## 5. The extraction output",
            "## 6. What happened to each field",
            "## 7. The tool definitions in full",
        )

    def test_each_turns_prose_sits_with_its_turn(self, document):
        turn_one = _between(document, "### Turn 1", "### Turn 2")
        assert "> Reading the paper. I will start with the study block." \
            in turn_one
        turn_two = _between(document, "### Turn 2", "### Turn 3")
        assert "> The checker is right: the header gives the full title." \
            in turn_two

    def test_a_tool_call_shows_what_applied_and_what_was_rejected(
            self, document):
        call = _between(document, "#### Call 1. `update_study`",
                        "### Turn 2")
        # The call as sent.
        assert '"value": "A synthetic study"' in call
        # Applied, with the value either side.
        assert "**Applied: 1 field.**" in call
        assert f'| `study.title` | `null` | `"{SHORT_TITLE}"` | yes |' in call
        # Rejected, with the dispatcher's reason, next to the same call.
        assert "**Rejected: 1 field.**" in call
        assert "- `study.doi`" in call
        assert "`quote_not_in_text`" in call
        assert "Quote not found in paper text" in call

    def test_a_partial_dispatch_is_labelled_honestly(self, document):
        assert "Status `partial`, applied in part." in document

    def test_each_checker_verdict_sits_beside_the_field_it_judged(
            self, document):
        call = _between(document, "#### Call 1. `update_study`",
                        "### Turn 2")
        assert "##### Check 1. `study.title`: challenge" in call
        # Its rationale, in full, in the same block as the field.
        assert ("> The quote gives the full title; the stored value is a "
                "shortened paraphrase of it.") in call
        # And what it was scoring, so the verdict can be read against it.
        assert f'| Value it scored | `"{SHORT_TITLE}"` |' in call
        assert "| Verdict | `challenge` |" in call
        assert ("| Extractor's note, as it saw it | "
                '`"Shortened from the header for brevity."` |') in call

    def test_a_re_check_is_numbered_as_that_fields_second(self, document):
        assert ("##### Check 2. `study.title`: ok "
                "(check 2 of this field)") in document
        assert "> The value now matches the quoted header exactly." in document

    def test_the_checker_system_prompt_appears_exactly_once(self, session):
        """It is one string for the whole run and a run can make hundreds of
        checks, so it is printed once in the instrument and referenced from
        each check."""
        prompt = (session.session.instrument_dir /
                  "checker_system_prompt.txt").read_text(encoding="utf-8")
        assert prompt.strip()
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        assert document.count(prompt) == 1
        # And each check points back at it rather than repeating it.
        assert document.count(
            "[printed once in section 2.4](#instrument-checker-system)") == 2

    def test_the_instrument_prints_each_stages_prompt_once(self, session):
        instrument = session.session.instrument_dir
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        for name in ("system_prompt.txt", "user_prompt.txt",
                     "review_system_prompt.txt", "checker_system_prompt.txt"):
            text = (instrument / name).read_text(encoding="utf-8")
            assert document.count(text) == 1, name

    def test_the_scaffold_the_checks_were_rendered_from_is_printed(
            self, session):
        """The other half of every check. The system prompt above it is one
        string for the whole run and so is this, so the document prints it
        once, beside the prompts, with its slots standing as the tokens they
        are."""
        scaffold = (session.session.instrument_dir /
                    "checker_user_scaffold.txt").read_text(encoding="utf-8")
        assert "{field_path}" in scaffold
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        assert "### 2.7 The checker's per-field scaffold" in document
        assert document.count(scaffold) == 1

    def test_a_session_that_captured_no_scaffold_renders_without_one(
            self, session):
        """A session recorded before the capture existed has every other
        instrument file and not this one. The document is then the document it
        would have been: the subsection is skipped outright rather than
        reported absent, and nothing above it moves."""
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        (session.session.instrument_dir /
         "checker_user_scaffold.txt").unlink()
        without = render_transcript(session.session.session_dir)

        assert "2.7 The checker's per-field scaffold" not in without
        assert "instrument-checker-scaffold" not in without
        # No note in its place either: an absent file that the level never
        # promised is not a degradation to report.
        assert "checker_user_scaffold" not in without
        # And the rest is untouched, byte for byte, on both sides of where the
        # subsection sat.
        head = document[:document.index(
            '<a id="instrument-checker-scaffold"></a>')]
        tail = document[document.index('<a id="sec-extraction"></a>'):]
        assert without == head + tail

    def test_the_captured_instrument_is_what_the_run_actually_sends(
            self, session):
        """The document is only honest if the captured prompt is the one that
        went on the wire, so this reads the request the reviewer actually made
        rather than re-rendering the prompt and comparing a function with
        itself.

        The reviewer is the stage this fixture drives through a real adapter:
        the extractor's turn function and the checker's fan-out are scripted,
        so their wire calls are not made here and only the render identity is
        available for them.
        """
        instrument = session.session.instrument_dir
        adapter = session._adapter_for_role("review")
        assert adapter.calls, "the reviewer made no call to read"
        sent = [b["text"] for b in adapter.calls[0]["system"]]
        assert sent == [
            (instrument / "review_system_prompt.txt").read_text()]

        assert (instrument / "checker_system_prompt.txt").read_text() == \
            session._render_checker_system_text()
        assert (instrument / "system_prompt.txt").read_text() == \
            session.system_text

    def test_the_review_is_a_separate_section_on_a_fresh_context(
            self, document):
        review = _between(document, "## 4. The review",
                          "## 5. The extraction output")
        assert "**A separate stage on a fresh context.**" in review
        assert "It never saw the extractor's turns" in review
        # The reviewer's own turn, its prose, and its edit, all inside it.
        assert "### Reviewer turn 4" in review
        assert "> The title can be more precise." in review
        assert f'`"{REVIEWED_TITLE}"`' in review
        # The reviewer's `mark_complete` IS dispatched: it carries the
        # reviewer's own quality check, and that has to be recorded. So the
        # turn that ends the review renders as an ordinary call with its own
        # result, rather than as a turn that appears to have done nothing.
        # (Its presence in the batch is what ends the review; the run
        # finalising `complete` in `_run_session` is that.)
        assert re.search(r"#### Call \d+\. `mark_complete`", review)
        # And no extractor turn leaked into it.
        assert "### Turn 1" not in review

    def test_the_final_extraction_output_is_included_in_full(self, session):
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        output = json.loads(session.session.extraction_record_path.read_text())
        assert json.dumps(output, indent=2, ensure_ascii=False) in document
        assert REVIEWED_TITLE in document

    def test_the_field_history_aggregate_is_summarised(self, session):
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        history = json.loads(session.session.field_history_path.read_text())
        aggregate = history["aggregate"]
        summary = _between(document, "## 6. What happened to each field",
                           "### Every field")
        assert f"| Challenges raised | {aggregate['challenges_raised']} |" \
            in summary
        assert ("| Challenges answered by changing the value | "
                f"{aggregate['challenges_revised']} |") in summary

    def test_a_field_can_be_followed_to_every_check_it_received(
            self, document):
        """The navigability claim: one place per field, linking to every write
        and every check, each of which carries the anchor it links to."""
        entry = _between(document, "#### `study.title`", "#### `study.doi`")
        assert ("Trail: [call 1 applied](#call-1), "
                "[check 1 challenge](#check-1), [call 2 applied](#call-2), "
                "[check 2 ok](#check-2)") in entry
        for anchor in ('<a id="call-1"></a>', '<a id="check-1"></a>',
                       '<a id="call-2"></a>', '<a id="check-2"></a>'):
            assert anchor in document
        # A rejected proposal is on the trail too, so a field that never
        # landed is still followable.
        rejected = document[document.index("#### `study.doi`"):]
        assert "Trail: [call 1 rejected](#call-1)." in rejected

    def test_the_contents_links_resolve_to_anchors_in_the_document(
            self, document):
        for anchor in ("sec-run", "sec-instrument", "sec-extraction",
                       "sec-review", "sec-output", "sec-field-history",
                       "sec-tool-definitions"):
            assert f"](#{anchor})" in document
            assert f'<a id="{anchor}"></a>' in document

    def test_the_header_reports_the_run_and_its_provenance(self, session):
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        meta = session.session.meta
        for value in (meta["study_id"], meta["session_id"], meta["status"],
                      meta["config_fp"], meta["checker_fp"],
                      meta["review_fp"], meta["run_fp"],
                      meta["meltiro_version"]):
            assert value in document
        assert "| Extractor | `claude-opus-4-8` |" in document.replace(
            " | *(not recorded)* | *(not recorded)* |", " |")


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------

class TestItReadsStartToFinish:
    """The document's opening promise is that it can be read start to finish,
    so the ORDER is a property in its own right and not a consequence of the
    sections existing.

    The tool schemas carry the whole field catalogue, twice over for the
    record tools. Printed inside the instrument they ran to roughly three
    quarters of the document, so a reader met them before the run they exist
    to narrate. They are reference material, so they are moved to the end.
    """

    def test_the_run_begins_before_the_full_tool_schemas_do(self, document):
        run = document.index("## 3. The extraction, turn by turn")
        schemas = document.index("## 7. The tool definitions in full")
        assert run < schemas
        # And the first schema BODY, not merely the heading above it: a
        # section that opened with the definitions would satisfy a
        # heading-only check.
        assert run < document.index('<a id="tool-update-study"></a>')
        assert run < document.index("Input schema:")

    def test_the_whole_narrative_precedes_the_reference_material(
            self, document):
        """Sections 1 to 6 are the run and come first; 7 is what a reader
        jumps to."""
        _order(
            document,
            "## 1. The run",
            "### 2.5 The tool catalogue",
            "## 3. The extraction, turn by turn",
            "## 4. The review",
            "## 5. The extraction output",
            "## 6. What happened to each field",
            "## 7. The tool definitions in full",
            '<a id="tool-update-study"></a>',
        )

    def test_the_run_is_reached_early_in_the_document(self, document):
        """The measurable form of the same claim. Before the schemas moved,
        91% of the document sat ahead of the run; a check that the sections
        merely exist would not have caught that."""
        run = document.index("## 3. The extraction, turn by turn")
        assert run < len(document) / 2, (
            f"the run starts {run / len(document):.0%} into the document")

    def test_2_5_indexes_the_tools_rather_than_printing_them(self, session):
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        index = _between(document, "### 2.5 The tool catalogue",
                         "### 2.6 The figures")
        tools = json.loads((session.session.instrument_dir /
                            "tool_definitions.json").read_text())
        # The captured catalogue is the EXTRACTOR's: ten tools, the reviewer's
        # nine plus `record_initial_check`. The count is here so the loop
        # below cannot pass vacuously on an empty file.
        assert len(tools) == 10
        for tool in tools:
            name = tool["name"]
            slug = name.replace("_", "-")
            # One row per tool, linking to its full entry.
            assert f"[`{name}`](#tool-{slug})" in index, name
            # The opening of the description, verbatim: the index abbreviates
            # by stopping early, never by rewriting.
            assert tool["description"][:30] in index, name
            # And never the whole of a long one; that is what section 7 is
            # for.
            if len(tool["description"]) > 250:
                assert tool["description"] not in index, name
        # No schema in the index at all.
        assert "Input schema:" not in index
        assert "[section 7](#sec-tool-definitions)" in index
        assert "only moved out of the way of the run" in index

    def test_2_6_names_each_exhibit_and_the_text_printed_with_it(self,
                                                                 session):
        # The exhibits record is the only durable copy of the manifest's
        # wording: a label says which crop an `<img>` citation named, the
        # caption says which table that was, and the footnote says what the
        # exhibit's own small print qualified — none of which survives the
        # paper bundle being re-cropped or moved. All of it comes off
        # `instrument/image_labels.json`, so the section and the file agree.
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        figures = _between(document, "### 2.6 The figures",
                           "### 2.7 The checker's per-field scaffold")
        exhibits = json.loads((session.session.instrument_dir /
                               "image_labels.json").read_text())
        assert exhibits, "the fixture bundle attached no exhibit to record"
        assert "| Label | Caption | Footnote |" in figures
        recorded_notes = [e["notes"] for e in exhibits if e.get("notes")]
        assert recorded_notes, "the fixture recorded no exhibit footnote"
        for exhibit in exhibits:
            assert f"`{exhibit['label']}`" in figures
            assert exhibit["caption"] in figures
        for note in recorded_notes:
            assert note in figures

    def test_2_6_reads_an_exhibits_record_that_carries_labels_alone(
            self, session):
        """The section's tolerance, matching the rest of the document's: an
        exhibits record whose entries are bare labels still renders. Each
        label is named, and the caption cell reports the caption as not
        recorded rather than the render failing or a caption being invented
        for it.
        """
        path = session.session.instrument_dir / "image_labels.json"
        exhibits = json.loads(path.read_text())
        captions = [e["caption"] for e in exhibits if e.get("caption")]
        assert captions, "the fixture recorded no caption to do without"
        path.write_text(json.dumps([e["label"] for e in exhibits]),
                        encoding="utf-8")

        figures = _between(render_transcript(session.session.session_dir),
                           "### 2.6 The figures",
                           "### 2.7 The checker's per-field scaffold")
        assert "| Label | Caption | Footnote |" in figures
        for exhibit in exhibits:
            assert f"`{exhibit['label']}`" in figures
        assert "*(not recorded)*" in figures
        # Nothing stood in for the captions the record does not carry.
        for caption in captions:
            assert caption not in figures

    def test_the_appendix_prints_every_definition_in_full(self, session):
        """Moved, not summarised: every description and every schema is still
        in the document, byte for byte."""
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        appendix = document[
            document.index("## 7. The tool definitions in full"):]
        tools = json.loads((session.session.instrument_dir /
                            "tool_definitions.json").read_text())
        for tool in tools:
            assert tool["description"] in appendix
            assert json.dumps(tool["input_schema"], indent=2,
                              ensure_ascii=False) in appendix

    def test_the_per_tool_anchors_are_unchanged_by_the_move(self, session):
        """An existing link into a tool still resolves: the anchors moved with
        the definitions rather than being renamed."""
        document = (session.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        tools = json.loads((session.session.instrument_dir /
                            "tool_definitions.json").read_text())
        for tool in tools:
            anchor = f'<a id="tool-{tool["name"].replace("_", "-")}"></a>'
            assert document.count(anchor) == 1, tool["name"]
            # The anchor sits inside the definitions section, not before it.
            assert document.index(anchor) > document.index(
                "## 7. The tool definitions in full")


# ---------------------------------------------------------------------------
# Degrading by diagnostics level
# ---------------------------------------------------------------------------

class TestDegradesHonestly:
    def test_minimal_says_what_it_cannot_show_rather_than_omitting_it(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        orch = _run_session(config_dir, bundle_minimal_dir, tmp_path,
                            monkeypatch, diagnostics="minimal")
        document = (orch.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        assert not orch.session.instrument_dir.exists()

        # It rendered rather than failing, and the run is all still there.
        _order(document, "## 2. The instrument", "## 3. The extraction, "
               "turn by turn", "### Turn 1", "## 4. The review",
               "## 5. The extraction output")
        assert "##### Check 1. `study.title`: challenge" in document

        # The instrument section says why it is empty, naming the level.
        instrument = _between(document, "## 2. The instrument",
                              "## 3. The extraction")
        assert "kept its diagnostics at `minimal`" in instrument
        assert "captures no instrument" in instrument
        assert "never gains one" in instrument
        # And it does not pretend it could rebuild it from the config bundle.
        assert "reconstruction dressed as a capture" in instrument
        # Nothing was quietly dropped: no prompt heading claims content that
        # is not there.
        assert "2.4 The checker's system prompt" not in instrument

        # The header says the same thing up front.
        assert "This run kept its diagnostics at `minimal`." in document

    def test_below_full_each_check_says_its_message_was_never_written(
            self, document):
        assert document.count(
            "The rendered user message for this check is not in the session"
        ) == 2
        assert "`standard` does not keep" in document

    def test_full_prints_each_checks_rendered_user_message_verbatim(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        """At `full` the wire log is kept, and it is the only place a check's
        rendered user message exists, so the document prints it in full."""
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     max_checks_per_field=1, diagnostics="full")
        orch.prepare_new_session()
        _drive(orch, [
            _resp(_tool_use("t1", "update_study", {"study": {
                "title": {"value": SHORT_TITLE,
                          "evidence": f"<q>{QUOTE}</q>"}}})),
            _resp(_tool_use("t2", "view_summary", {})),
        ])
        # The REAL fan-out, with only the provider adapter stubbed, so the
        # api_logger fires at its own call site with the genuine per-field
        # request and its field_path / check_index.
        _use_real_fanout(monkeypatch)
        orch._adapter_for_role = lambda role: object()
        assert orch.run() == "complete"

        document = (orch.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        # The stub answers in prose rather than by calling the verdict tool,
        # so this check was asked twice — and BOTH asks were sent and billed.
        # Each is printed and labelled: keyed on the field and the check
        # alone, the second would overwrite the first, and the document would
        # show the re-ask as if it were the whole of what was sent.
        assert "This check took 2 asks" in document
        assert "Ask 1 of 2, the first ask:" in document
        assert ("Ask 2 of 2, re-asked, correcting the reply that recorded no "
                "verdict:" in document)
        assert "The rendered user message for this check is not in the " \
            "session" not in document
        # The verbatim message, slot by slot: the field, its value, and the
        # evidence the checker was scoring.
        check = _between(document, "##### Check 1. `study.title`",
                         "| Property | Value |")
        assert "`study.title`" in check
        assert f'"{SHORT_TITLE}"' in check
        assert f'"{QUOTE}"' in check
        # The correction rides on the second ask and nothing else, so its one
        # appearance is what separates the two messages.
        assert check.count(CHECKER_TOOL_REPROMPT) == 1
        # The re-ask replayed the prose reply, and the document shows it as
        # the model's own words rather than as more of what it was sent: three
        # fences under Ask 2, and the middle one named for what it is.
        assert "the reply being corrected:" in check
        assert "the correction, as a new user turn:" in check
        assert "`full` is the top level" in document
        # The re-ask is on the outcome, not only in the message dump.
        assert "##### Check 1. `study.title`" in document
        assert "(re-asked once)" in document
        assert "| Re-asks before a verdict | 1 |" in document

    def test_full_labels_no_replay_when_the_reply_could_not_be_replayed(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        """A reply holding a tool call is corrected without being replayed, so
        the re-ask is one message and the document shows one, unlabelled."""
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     max_checks_per_field=1, diagnostics="full")
        orch.prepare_new_session()
        _drive(orch, [
            _resp(_tool_use("t1", "update_study", {"study": {
                "title": {"value": SHORT_TITLE,
                          "evidence": f"<q>{QUOTE}</q>"}}})),
            _resp(_tool_use("t2", "view_summary", {})),
        ])
        # The checker calls a tool that is not the verdict tool: no verdict,
        # and nothing the re-ask can quote back.
        _use_real_fanout(monkeypatch,
                         [_tool_use("c1", "mark_complete", {})])
        orch._adapter_for_role = lambda role: object()
        assert orch.run() == "complete"

        document = (orch.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        check = _between(document, "##### Check 1. `study.title`",
                         "| Property | Value |")
        assert "This check took 2 asks" in check
        assert check.count(CHECKER_TOOL_REPROMPT) == 1
        # Nothing was replayed, so nothing is labelled as replayed.
        assert "the reply being corrected:" not in check
        assert "the correction, as a new user turn:" not in check

    def test_a_session_stopped_before_the_review_says_so(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        orch = _run_session(config_dir, bundle_minimal_dir, tmp_path,
                            monkeypatch, final_review=False)
        document = (orch.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        review = _between(document, "## 4. The review",
                          "## 5. The extraction output")
        assert "The reviewer stage was off for this run" in review
        assert "The reviewer stage was off for this run, so it has no " \
            "system prompt." in document


def _use_real_fanout(monkeypatch, content=None):
    """Run the genuine checker fan-out against a stubbed provider adapter.

    `content` is the reply every checker call gets. The default is prose — the
    verdict written out rather than called in — which records no verdict, so
    the check is re-asked and both asks reach the wire log the document is
    rendered from.
    """
    if content is None:
        content = [_text(json.dumps({
            "verdict": "challenge",
            "rationale": "The quote gives the full title.",
            "notes": None,
        }))]

    class _Adapter:
        def create_message(self, **kwargs):
            # One dict under both names, as the Anthropic adapter records it:
            # the canonical format IS that wire, so the two are the same
            # object and the audit log stores the request once.
            request = dict(kwargs)
            return NormalisedResponse(
                content=list(content),
                usage=NormalisedUsage(input_tokens=812, output_tokens=96),
                resolved_model=CHECKER, provider="anthropic", base_url=None,
                raw_request=request, raw_response={"id": "msg_stub"},
                wire_request=request,
                decoding_params={"temperature": 0.0},
            )

    real = checker_mod.run_checker_batch
    # The orchestrator now supplies its own cached adapter (one per run, so a
    # fan-out reuses the connection pool the last one left warm), so the stub
    # REPLACES that argument rather than adding one.
    monkeypatch.setattr(
        orch_mod, "run_checker_batch",
        lambda **kw: real(**{**kw, "adapter": _Adapter()}))


# ---------------------------------------------------------------------------
# The two copies never diverge
# ---------------------------------------------------------------------------

class TestStatusGloss:
    """The transcript glosses every tool-call status the dispatcher emits,
    and glosses nothing it does not.

    The document's job is to be readable without the source beside it, so a
    status that reaches it unglossed is a bare token in the one artefact meant
    to explain itself — and a gloss for a status that cannot occur is text no
    reader will ever see, kept alive by nothing.
    """

    def _dispatcher_statuses(self):
        """Every status literal `ToolDispatcher._result` is called with."""
        import ast
        import inspect

        from meltiro import tools as tools_mod

        tree = ast.parse(inspect.getsource(tools_mod))
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "_result"):
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                found.add(node.args[0].value)
        # The two branches that compute the status into a local first, so it
        # is assigned rather than passed as a literal.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "status"):
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Constant) and isinstance(
                            sub.value, str):
                        found.add(sub.value)
        return found

    def test_gloss_keys_are_exactly_the_dispatcher_statuses(self):
        from meltiro.transcript import _STATUS_GLOSS

        assert self._dispatcher_statuses() == set(_STATUS_GLOSS), (
            "the transcript's status glossary and the dispatcher's statuses "
            "have drifted: a status the dispatcher emits renders bare in "
            "every transcript, or a gloss is kept for a status no run can "
            "produce.")

    @pytest.mark.parametrize("status,gloss", [
        ("ok", "Every field in the call applied."),
        ("partial", "Some fields applied, some were rejected."),
        ("validation_failed", "Every field was rejected; nothing applied."),
    ])
    def test_each_status_is_glossed_in_the_document(
            self, tmp_path, status, gloss):
        # `validation_failed` is the one an operator most needs explained, and
        # the one that reached the document bare.
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "session_started"},
            {"ts": "T1", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "tool_use", "id": "t1",
                          "name": "update_study", "input": {}}]},
            {"ts": "T2", "event": "tool_call_failed", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_study", "args": {},
             "result": {"status": status, "errors": []}},
            {"ts": "T3", "event": "terminate", "status": "failed_validation"},
        ])
        document = render_transcript(session_dir)
        assert status in document
        assert gloss in document


class TestOneCodePath:
    def test_a_run_writes_the_transcript_at_finalisation(self, session):
        path = session.session.session_dir / "diagnostics" / "transcript.md"
        assert path.exists()
        assert path.read_text().startswith("# Transcript: demo-001")

    def test_a_paused_run_writes_one_too(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        """A pause is where an operator most needs to read what happened
        before deciding whether to raise the cap."""
        orch = _orch(config_dir, bundle_minimal_dir, tmp_path / "runs",
                     max_checks_per_field=1)
        orch.max_tool_calls = 1
        orch.prepare_new_session()
        _drive(orch, [_resp(_tool_use("t1", "update_study", _REVISION_CALL))])
        _stub_fanout(monkeypatch, [("ok", "fine")])
        orch._adapter_for_role = lambda role: object()
        assert orch.run() == "in_progress"

        document = (orch.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        assert "| Status | `in_progress` |" in document
        assert "| Paused because | `tool_cap_hit` |" in document
        assert "the run paused (tool_cap_hit)" in document
        assert "#### Call 1. `update_study`" in document

    def test_the_subcommand_reproduces_the_run_written_copy_byte_for_byte(
            self, session, tmp_path, capsys):
        """The only guarantee that the two can never disagree: they are the
        same function over the same finished files."""
        session_dir = session.session.session_dir
        written = (session_dir / "diagnostics" / "transcript.md").read_bytes()
        out = tmp_path / "elsewhere" / "transcript.md"
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["transcript", str(session_dir), "--out", str(out)])
        assert exit_info.value.code == 0
        assert out.read_bytes() == written
        assert "Wrote transcript to" in capsys.readouterr().out

    def test_re_rendering_is_stable_and_changes_nothing_in_the_session(
            self, session):
        session_dir = session.session.session_dir
        before = {p.name: p.read_bytes()
                  for p in (session_dir / "diagnostics").iterdir()
                  if p.is_file()}
        assert render_transcript(session_dir) == render_transcript(session_dir)
        after = {p.name: p.read_bytes()
                 for p in (session_dir / "diagnostics").iterdir()
                 if p.is_file()}
        assert before == after

    def test_the_document_carries_no_rendering_time_state(self, session):
        """Byte-identity depends on the document being a pure function of the
        session, so it must not stamp itself with anything from the render."""
        session_dir = session.session.session_dir
        document = render_transcript(session_dir)
        # Both timestamps in the document are the session's own, read off
        # run.json, and no other appears.
        meta = json.loads(
            (session_dir / "diagnostics" / "run.json").read_text())
        assert meta["started_at"] in document
        assert meta["updated_at"] in document
        assert "generated" not in document.lower().split("## 2.")[0]


# ---------------------------------------------------------------------------
# The events a happy run never produces
# ---------------------------------------------------------------------------

def _hand_built_session(tmp_path, events, *, meta=None, output=None):
    """A session directory assembled by hand, so the rendering of events a
    clean run never emits can be exercised without contriving a failure for
    each. The shapes are the orchestrator's own, copied from its append sites.
    """
    session_dir = tmp_path / "sessions" / "20260101_000000_000000_abcdef"
    diagnostics = session_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    base = {
        "session_id": "20260101_000000_000000_abcdef",
        "study_id": "demo-001",
        "status": "failed_validation",
        "current_phase": "done",
        "started_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:10:00+00:00",
        "meltiro_version": "0.1.0.dev0",
        "git_commit": "abc1234", "git_dirty": False,
        "config_fp": "config_fp:aaa", "checker_fp": "checker_fp:bbb",
        "review_fp": None, "run_fp": "run_fp:ccc",
        "extractor_model": "claude-opus-4-8",
        "checker_model": "claude-sonnet-4-6", "review_model": None,
        "tool_set_hash": "ts", "template_hash": "th", "prompt_hash": "ph",
        "tool_call_count": 1, "checker_calls_run": 1, "warnings": [],
        "caps": {"max_tool_calls": 50, "max_review_tool_calls": 30,
                 "max_checks_per_field": 2},
        "diagnostics": "minimal",
        "structure": {"checker": True, "review": False,
                      "max_checks_per_field": 2,
                      "check_reviewer_edits": False},
        "images_omitted": {},
        "failure_reason": "surrendered",
        "failed_validation_reason": "the paper reports no usable outcome",
    }
    base.update(meta or {})
    (diagnostics / "run.json").write_text(json.dumps(base))
    (diagnostics / "tool_calls.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))
    (session_dir / "extraction_output.json").write_text(
        json.dumps(output or {"study": {}, "records": []}))
    return session_dir


class TestTheUnhappyPaths:
    def test_run_level_events_are_described_in_place(self, tmp_path):
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "session_started"},
            {"ts": "T1", "event": "resumed", "max_tool_calls": 200,
             "previous": 50, "max_review_tool_calls": 30,
             "previous_max_review_tool_calls": 30, "diagnostics": "minimal",
             "previous_diagnostics": "minimal", "git_dirty": True},
            {"ts": "T2", "event": "provider_retry", "stage": "extractor",
             "attempt": 1, "delay_seconds": 2, "error": "429 rate limited"},
            {"ts": "T3", "event": "assistant_text", "turn_id": 1,
             "text": "I cannot find any usable outcome."},
            {"ts": "T4", "event": "extractor_reprompt", "turn_id": 1,
             "text": "Call a tool."},
            {"ts": "T5", "event": "extractor_abandoned", "turn_id": 1,
             "reason": "the paper reports no usable outcome"},
            {"ts": "T6", "event": "terminate", "status": "failed_validation"},
        ])
        document = render_transcript(session_dir)
        _order(
            document,
            "*Run note: the session was created.*",
            "the run was resumed. Extractor tool-call cap 50 to 200",
            "The code tree was dirty at the start of this segment.",
            "the extractor stage hit a transient provider failure and retried",
            "> I cannot find any usable outcome.",
            "the turn called no tool, so *meltiro* sent this back",
            "the extractor called `abandon_extraction` and gave up",
            "the run finished with status `failed_validation`.",
        )
        # The outcome, and the stated reason, in the header.
        assert "| Failure reason | `surrendered` |" in document
        assert ("| Stated surrender reason | the paper reports no usable "
                "outcome |") in document

    def test_a_canonicalised_value_is_reported_next_to_its_field(
            self, tmp_path):
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "tool_call_applied", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_study",
             "args": {"study": {"instrument": {"value": "WDS9"}}},
             "result": {
                 "status": "ok", "applied_changes": {"study_fields": ["x"]},
                 "_field_diffs": {"study.instrument": {
                     "before": None, "after": "WDS-9"}},
                 "errors": [], "warnings": [], "failed_fields": {},
                 "canonicalisation_notes": ["study.instrument: recorded as "
                                            "'WDS-9' (entered 'WDS9')"],
                 "_canonicalisations": [{"path": "study.instrument",
                                         "entered": "WDS9",
                                         "stored": "WDS-9"}],
             }},
            {"ts": "T1", "event": "value_canonicalised", "turn_id": 1,
             "field_path": "study.instrument", "entered": "WDS9",
             "stored": "WDS-9"},
            {"ts": "T2", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "text", "text": ""}]},
        ])
        document = render_transcript(session_dir)
        assert "**Canonicalised.**" in document
        assert ('- `study.instrument`: entered `"WDS9"`, stored `"WDS-9"`'
                in document)
        # Once, next to the field: the standalone event restates the result
        # key and is not rendered a second time.
        assert document.count("entered `\"WDS9\"`") == 1

    def test_a_call_level_error_is_reported_apart_from_the_field_ones(
            self, tmp_path):
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "tool_call_failed", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_record",
             "args": {"record_id": "relationship_9"},
             "result": {
                 "status": "error", "applied_changes": {}, "_field_diffs": {},
                 "errors": [
                     {"path": None, "code": "unknown_record",
                      "message": "No record with id 'relationship_9'."},
                     {"path": "record.relationship_1.gauge",
                      "code": "missing_evidence", "message": "Needs a quote."},
                 ],
                 "warnings": [],
                 "failed_fields": {"record.relationship_1.gauge": [
                     {"path": "record.relationship_1.gauge",
                      "code": "missing_evidence",
                      "message": "Needs a quote."}]},
             }},
            {"ts": "T1", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "text", "text": ""}]},
        ])
        document = render_transcript(session_dir)
        assert "**Call-level errors.**" in document
        assert "- `unknown_record`: No record with id 'relationship_9'." \
            in document
        # The per-field error stays with its field, and is not repeated in the
        # call-level list, which `errors` flattens it into.
        assert document.count("Needs a quote.") == 1
        assert "- `record.relationship_1.gauge`" in document

    def test_a_failed_check_is_not_shown_as_a_judgement(
            self, tmp_path):
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "tool_call_applied", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_study", "args": {},
             "result": {
                 "status": "ok", "applied_changes": {}, "errors": [],
                 "warnings": [], "failed_fields": {},
                 "_field_diffs": {"study.title": {"before": None,
                                                  "after": "T"}},
                 "_checker_verdicts": {"study.title": {
                     "verdict": "challenge",
                     "rationale": "(checker error: rate limited)",
                     "notes": None, "value_checked": "T",
                     "evidence_checked": "<q>T</q>", "note_checked": None,
                     "error_origin": True, "stage": "extractor",
                     "input_tokens": 0, "output_tokens": 0,
                     "cache_creation_tokens": 0, "cache_read_tokens": 0,
                     "cost_usd": 0.0}},
             }},
            {"ts": "T1", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "text", "text": ""}]},
        ])
        document = render_transcript(session_dir)
        # The headline names no cause. This verdict's was a rate limit, and
        # the ways a check ends without one also include a reply that called
        # no tool, a cut-off reply, a verdict outside the vocabulary and a
        # fault in the plumbing; the rationale below is where the actual
        # cause is recorded.
        assert ("##### Check 1. `study.title`: challenge, but from a failed "
                "check — not a checker judgement") in document
        assert "exhausted retry" not in document
        assert "This is not a judgement." in document
        # What the extractor was shown is read off this call's own
        # `checker_challenges`, which this event does not carry.
        assert "its text was not put to the extractor" in document
        # What the paragraph must NOT claim: a failed check's calls may have
        # reached the provider and been billed, and whatever they sent is in
        # the wire log when the session kept one. Stating otherwise would tell
        # a reader their run was cheaper than it was and send them looking for
        # a log entry the document said was absent.
        assert "billed at zero" not in document.lower()
        assert "no request reached the wire log" not in document.lower()
        assert "may well have completed and been billed" in document
        assert "[check 1 check error](#check-1)" in document

    def test_a_failed_check_that_was_shown_says_it_was_shown(self, tmp_path):
        # The same failure, in a session where the challenge DID go into the
        # tool result. What reached the extractor is a fact about the run
        # being rendered, read off this call's own `checker_challenges`;
        # asserting the engine's current rule instead would misdescribe every
        # session recorded under a different one.
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "tool_call_applied", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_study", "args": {},
             "result": {
                 "status": "ok", "applied_changes": {}, "errors": [],
                 "warnings": [], "failed_fields": {},
                 "checker_challenges": {
                     "study.title": "(checker error: rate limited)"},
                 "_checker_verdicts": {"study.title": {
                     "verdict": "challenge",
                     "rationale": "(checker error: rate limited)",
                     "notes": None, "value_checked": "T",
                     "evidence_checked": "<q>T</q>", "note_checked": None,
                     "error_origin": True, "stage": "extractor",
                     "input_tokens": 0, "output_tokens": 0,
                     "cache_creation_tokens": 0, "cache_read_tokens": 0,
                     "cost_usd": 0.0}},
             }},
            {"ts": "T1", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "text", "text": ""}]},
        ])
        document = render_transcript(session_dir)
        assert "its text WAS put to the extractor" in document
        assert "its text was not put to the extractor" not in document

    def test_a_total_that_covers_less_than_the_run_says_so(self, tmp_path):
        # A call billed and not priced leaves the sum a floor. Printed bare it
        # would read as the whole bill, and nothing else in the document would
        # contradict it.
        session_dir = _hand_built_session(
            tmp_path,
            [{"ts": "T0", "event": "session_started"}],
            meta={"cost_usd": 0.004, "cost_incomplete": True,
                  "unreceipted_calls": 2, "input_tokens": 10,
                  "output_tokens": 2})
        document = render_transcript(session_dir)
        assert "at least $0.004000" in document
        assert "2 call(s) returned no receipt" in document

    def test_a_fully_receipted_total_is_printed_plainly(self, tmp_path):
        session_dir = _hand_built_session(
            tmp_path,
            [{"ts": "T0", "event": "session_started"}],
            meta={"cost_usd": 0.004, "input_tokens": 10, "output_tokens": 2})
        document = render_transcript(session_dir)
        assert "$0.004000" in document
        assert "at least" not in document

    def test_a_run_nothing_priced_states_no_floor_to_be_at_least_of(
            self, tmp_path):
        # Two separate gaps, and stacking them produces a claim nobody can
        # make: "at least *(not priced)*" is a floor over no number at all.
        # The coverage still has to be stated — an unpriced run is the one
        # place a missing receipt could hide behind a louder fault.
        session_dir = _hand_built_session(
            tmp_path,
            [{"ts": "T0", "event": "session_started"}],
            meta={"cost_usd": None, "cost_incomplete": True,
                  "unreceipted_calls": 3, "input_tokens": 10,
                  "output_tokens": 2})
        document = render_transcript(session_dir)
        assert ("| Cost | *(not priced)* — 3 call(s) returned no receipt and "
                "are missing from any figure |") in document
        assert "at least" not in document

    def test_a_receipted_sum_of_zero_is_not_dressed_up_as_a_floor(
            self, tmp_path):
        # Every receipt there was is in the figure and it is still nothing, so
        # "at least $0.0000" would invite a reader to imagine a bill just above
        # zero. What is true is that nothing receipted was charged, over calls
        # nobody can price.
        session_dir = _hand_built_session(
            tmp_path,
            [{"ts": "T0", "event": "session_started"}],
            meta={"cost_usd": 0.0, "cost_incomplete": True,
                  "unreceipted_calls": 2, "input_tokens": 10,
                  "output_tokens": 2})
        document = render_transcript(session_dir)
        spend = document.split("### Spend", 1)[1].split("###", 1)[0]
        assert ("| Cost | no receipted charge (2 call(s) returned no "
                "receipt) |") in spend
        assert "$" not in spend

    def test_a_roles_own_figure_carries_its_own_coverage(self, tmp_path):
        # The per-role rows are what a reader adds up for one stage's bill, and
        # the run-wide qualifier is a table away by then. A floor arriving here
        # bare would be summed as a total.
        session_dir = _hand_built_session(
            tmp_path,
            [{"ts": "T0", "event": "session_started"}],
            meta={"cost_usd": 0.004, "cost_incomplete": True,
                  "unreceipted_calls": 2, "input_tokens": 10,
                  "output_tokens": 2,
                  "usage_by_role": {
                      "extractor": {"input_tokens": 10, "output_tokens": 2,
                                    "cache_read_tokens": 0,
                                    "cache_write_tokens": 0,
                                    "cost_usd": 0.004},
                      "checker": {"input_tokens": 5, "output_tokens": 1,
                                  "cache_read_tokens": 0,
                                  "cache_write_tokens": 0,
                                  "cost_usd": 0.0,
                                  "cost_incomplete": True,
                                  "unreceipted_calls": 2}}})
        document = render_transcript(session_dir)
        assert "| extractor | $0.004000 |" in document
        assert ("| checker | no receipted charge (2 call(s) returned no "
                "receipt) |") in document

    def test_a_check_and_the_checkers_total_both_state_their_coverage(
            self, tmp_path):
        # One check can make more than one call (a re-ask is a second), so a
        # verdict states what its own figure covers, and the checker's total in
        # the field history is a floor over the same gap. Neither is derivable
        # from the other, and a reader meets them in different sections.
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "tool_call_applied", "turn_id": 1,
             "tool_use_id": "t1", "tool": "update_study", "args": {},
             "result": {
                 "status": "ok", "applied_changes": {}, "errors": [],
                 "warnings": [], "failed_fields": {},
                 "_field_diffs": {"study.title": {"before": None,
                                                  "after": "T"}},
                 "_checker_verdicts": {"study.title": {
                     "verdict": "ok", "rationale": "matches the quote",
                     "notes": None, "value_checked": "T",
                     "evidence_checked": "<q>T</q>", "note_checked": None,
                     "error_origin": False, "stage": "extractor",
                     "input_tokens": 5, "output_tokens": 1,
                     "cache_creation_tokens": 0, "cache_read_tokens": 0,
                     "cost_usd": 0.004, "cost_incomplete": True,
                     "unreceipted_responses": 1}},
             }},
            {"ts": "T1", "event": "assistant_message", "turn_id": 1,
             "content": [{"type": "text", "text": ""}]},
        ])
        document = render_transcript(session_dir)
        assert ("| Cost | at least $0.004000 (1 call(s) returned no "
                "receipt) |") in document
        assert ("| What the checker cost | at least $0.004000 (1 call(s) "
                "returned no receipt) |") in document

    def test_a_session_with_no_field_history_file_says_it_derived_one(
            self, tmp_path):
        """A session killed mid-run has no field_history.json: the file is
        written at every stop. The document derives one and says so."""
        session_dir = _hand_built_session(tmp_path, [
            {"ts": "T0", "event": "session_started"},
        ], meta={"status": "in_progress", "current_phase": "extracting"})
        document = render_transcript(session_dir)
        assert "was killed mid-run rather than stopped" in document
        assert "## 6. What happened to each field" in document


# ---------------------------------------------------------------------------
# Strict inputs
# ---------------------------------------------------------------------------

class TestStrictInputs:
    def test_a_missing_session_directory_is_an_error(self, tmp_path):
        with pytest.raises(SessionError, match="no such session directory"):
            render_transcript(tmp_path / "nope")

    def test_a_directory_that_is_not_a_session_is_an_error(self, tmp_path):
        (tmp_path / "notasession").mkdir()
        with pytest.raises(SessionError, match="run.json"):
            render_transcript(tmp_path / "notasession")

    def test_a_session_recording_no_level_is_an_error(self, session):
        """A session whose run.json does not say what it kept cannot be
        described honestly, and guessing would put claims in the document
        that no file backs."""
        meta_path = session.session.meta_path
        meta = json.loads(meta_path.read_text())
        del meta["diagnostics"]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(SessionError, match="Unknown diagnostics level"):
            render_transcript(session.session.session_dir)

    def test_a_corrupt_event_log_is_an_error_not_a_partial_document(
            self, session):
        log = session.session.tool_calls_path
        lines = log.read_text().splitlines()
        lines.insert(1, "{not json")
        log.write_text("\n".join(lines) + "\n")
        with pytest.raises(SessionError, match="malformed JSON on line 2"):
            render_transcript(session.session.session_dir)

    def test_a_missing_extraction_output_is_an_error(self, session):
        session.session.extraction_record_path.unlink()
        with pytest.raises(SessionError, match="extraction_output.json"):
            render_transcript(session.session.session_dir)

    def test_the_subcommand_fails_loudly_and_writes_nothing(
            self, tmp_path, capsys):
        out = tmp_path / "transcript.md"
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["transcript", str(tmp_path / "nope"), "--out", str(out)])
        assert exit_info.value.code == 1
        assert not out.exists()
        assert "no such session directory" in capsys.readouterr().err
