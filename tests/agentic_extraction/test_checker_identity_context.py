"""Checker study-identity context resolution.

Covers the resolution chain the orchestrator assembles for the checker:
  a. manifest summary (highest precedence),
  b. the extracted role:summary field value,
  c. startup guard when neither a manifest summary nor a role:summary field
     exists (before any API spend),
  d. runtime degradation to title + DOI (with a warning) when relying on (b)
     but the field is empty,
  e. the mismatch tripwire when both summaries exist and diverge.

These use lightweight stubs (mirroring the cap-bonus tests) so the pure
resolution logic can be exercised without constructing a live session or
touching the network.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from meltiro.bundle import load_bundle
from meltiro.errors import AgenticExtractionError
from meltiro.orchestrator import Orchestrator, _summaries_match


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self):
        self.warnings = []
        self.events = []
        self.meta = {"current_phase": "extracting"}

    def add_warning(self, message):
        self.warnings.append(message)

    def append_event(self, event):
        self.events.append(event)


def _bare_orch():
    """A real Orchestrator instance with __init__ bypassed, so its own
    methods resolve via `self` while ONLY the attributes the identity-context
    path reads are set."""
    return Orchestrator.__new__(Orchestrator)


def _ctx_stub(*, manifest_summary, role_summary_var="abstract",
              extracted_value, title="A synthetic study", doi="10.0/x"):
    """Build a bare Orchestrator carrying just what _study_identity_context
    reads.

    `role_summary_var=None` models a template with no role:summary field.
    `extracted_value` is the current value of the role:summary study field
    (None models a null/empty extraction).
    """
    role_fields = {}
    study = {}
    if role_summary_var is not None:
        role_fields["summary"] = {"variable": role_summary_var}
        study[role_summary_var] = {"value": extracted_value}
    stub = _bare_orch()
    stub.bundle = SimpleNamespace(
        summary=manifest_summary, title=title, doi=doi)
    stub.template = {"role_fields": role_fields, "checker_context_fields": []}
    stub.extraction_record = SimpleNamespace(study=study)
    stub.study_id = "demo-001"
    stub.session = _FakeSession()
    stub._identity_degradation_warned = False
    stub._summary_mismatch_advised = False
    return stub


# ---------------------------------------------------------------------------
# Record-field identity context: study context + record label
# ---------------------------------------------------------------------------

class TestRecordIdentityContext:
    def test_record_context_includes_study_context_and_record_id(self):
        stub = _ctx_stub(
            manifest_summary="A synthetic bench-test study.",
            extracted_value=None)
        stub.template["checker_context_fields"] = ["gauge", "outcome_variable"]
        record = {
            "record_id": "relationship_3",
            "gauge": {"value": "WDS-9"},
            "outcome_variable": {"value": "Unplanned removal"},
        }
        ctx = Orchestrator._record_identity_context(stub, record)
        # Study identity context is present (the checker knows which paper) ...
        assert "Summary: A synthetic bench-test study." in ctx
        # ... alongside the record label, which leads with the engine record id.
        assert "relationship_3" in ctx
        assert "WDS-9" in ctx
        assert "Unplanned removal" in ctx
        # Drift guard: pin the FULL composed string, not just membership. The
        # newline JOIN between the study-identity line and the record label is
        # the checker's actual wording; membership alone would let the
        # orchestrator drift to a different separator (say ` || `) from
        # render_record_identity_context while every `in` assertion above stayed
        # green and checker_fp did not move. The ` — ` record-id lead-in and the
        # ` | ` context-field separator come from build_record_context.
        assert ctx == (
            "Summary: A synthetic bench-test study.\n"
            "relationship_3 — WDS-9 | Unplanned removal")

    def test_record_context_is_bare_id_when_no_context_fields(self):
        stub = _ctx_stub(
            manifest_summary="A synthetic bench-test study.",
            extracted_value=None)
        stub.template["checker_context_fields"] = []
        record = {"record_id": "relationship_9", "gauge": {"value": "WDS-9"}}
        ctx = Orchestrator._record_identity_context(stub, record)
        assert "Summary: A synthetic bench-test study." in ctx
        assert "relationship_9" in ctx
        # No context fields declared: the record's field values are not shown.
        assert "WDS-9" not in ctx


# ---------------------------------------------------------------------------
# Startup guard (c)
# ---------------------------------------------------------------------------

class TestStartupGuard:
    def _guard_stub(self, summary, has_role, checker_enabled=True):
        return SimpleNamespace(
            bundle=SimpleNamespace(summary=summary),
            template={"role_fields": {"summary": {"variable": "abstract"}}
                      if has_role else {}},
            checker_enabled=checker_enabled,
        )

    def test_manifest_summary_only_ok(self):
        Orchestrator._startup_identity_guard(
            self._guard_stub("some summary", has_role=False))

    def test_role_field_only_ok(self):
        Orchestrator._startup_identity_guard(
            self._guard_stub(None, has_role=True))

    def test_both_ok(self):
        Orchestrator._startup_identity_guard(
            self._guard_stub("some summary", has_role=True))

    def test_neither_raises(self):
        with pytest.raises(AgenticExtractionError,
                           match="No study-identity context"):
            Orchestrator._startup_identity_guard(
                self._guard_stub(None, has_role=False))

    def test_blank_manifest_summary_counts_as_absent(self):
        with pytest.raises(AgenticExtractionError,
                           match="No study-identity context"):
            Orchestrator._startup_identity_guard(
                self._guard_stub("   ", has_role=False))

    def test_guard_names_both_remedies(self):
        try:
            Orchestrator._startup_identity_guard(
                self._guard_stub(None, has_role=False))
        except AgenticExtractionError as e:
            msg = str(e)
            assert "summary" in msg and "role: summary" in msg
        else:
            pytest.fail("guard did not raise")

    def test_an_extractor_only_run_is_admitted(self):
        # The context this guards exists to fill a slot in a CHECKER call.
        # With `max_checks_per_field: 0` no checker call is ever made, so a
        # bundle with no summary and a template with no `role: summary` field
        # starves nothing. The extractor-only ablation is a documented
        # configuration with a run_fp shape of its own, and refusing to start
        # it on behalf of a stage it switches off makes that arm unrunnable.
        Orchestrator._startup_identity_guard(
            self._guard_stub(None, has_role=False, checker_enabled=False))

    def test_the_guard_still_fires_when_the_checker_runs(self):
        # The gate is on the stage, not on the check: with the checker on, the
        # same inputs are still refused before any spend.
        with pytest.raises(AgenticExtractionError,
                           match="No study-identity context"):
            Orchestrator._startup_identity_guard(
                self._guard_stub(None, has_role=False, checker_enabled=True))

    def test_an_extractor_only_run_starts_via_prepare_new_session(
            self, tmp_path, config_dir, bundle_minimal_dir):
        # Integration for the ablation: the same bundle+template that a
        # checker-on run refuses builds a session when the checker is off.
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = dataclasses.replace(
            load_bundle(bundle_minimal_dir), summary=None)
        orch = Orchestrator(
            config, bundle, tmp_path / "runs",
            extractor_model="claude-opus-4-7",
            review_model="claude-opus-4-7",
            max_checks_per_field=0,
            api_key="")
        orch.template["role_fields"] = {}  # no role:summary field either
        orch.prepare_new_session()
        assert orch.session.meta["status"] == "in_progress"
        assert orch.session.meta["checker_fp"] is None

    def test_guard_fires_via_prepare_new_session(self, tmp_path, config_dir,
                                                  bundle_minimal_dir):
        # Integration: the guard is wired into prepare_new_session and fires
        # BEFORE any API spend. Construct the missing-both case in the test
        # (bundle with no summary + template with the role field removed),
        # never by breaking the fixture on disk.
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        bundle = dataclasses.replace(bundle, summary=None)
        # Models are required keyword args (no hardcoded default); the guard
        # fires before any API call, so the specific ids are immaterial here.
        orch = Orchestrator(
            config, bundle, tmp_path / "runs",
            extractor_model="claude-opus-4-7",
            review_model="claude-opus-4-7",
            api_key="")
        orch.template["role_fields"] = {}  # simulate no role:summary field
        with pytest.raises(AgenticExtractionError,
                           match="No study-identity context"):
            orch.prepare_new_session()


# ---------------------------------------------------------------------------
# Resolution chain (a, b) + degradation (d)
# ---------------------------------------------------------------------------

class TestResolutionChain:
    def test_manifest_summary_wins(self):
        # Manifest and extracted values agree (extracted is a substring of
        # the manifest, so the mismatch tripwire stays quiet), which lets
        # the returned text prove the manifest value took precedence.
        stub = _ctx_stub(
            manifest_summary="Durability gauge study in a synthetic batch.",
            extracted_value="Durability gauge study")
        ctx = Orchestrator._study_identity_context(stub)
        assert ctx == "Summary: Durability gauge study in a synthetic batch."
        assert stub.session.warnings == []

    def test_extracted_value_used_when_no_manifest(self):
        stub = _ctx_stub(manifest_summary=None,
                         extracted_value="Extracted abstract text.")
        ctx = Orchestrator._study_identity_context(stub)
        assert ctx == "Summary: Extracted abstract text."
        assert stub.session.warnings == []

    def test_degrades_to_title_and_doi_with_warning(self, capsys):
        stub = _ctx_stub(manifest_summary=None, extracted_value=None,
                         title="A synthetic study", doi="10.0/demo")
        ctx = Orchestrator._study_identity_context(stub)
        assert "Title: A synthetic study" in ctx
        assert "DOI: 10.0/demo" in ctx
        # Drift guard: pin the FULL degraded string, including the newline JOIN
        # between the Title and DOI lines that render_degraded_identity_context
        # composes. Membership alone would let the orchestrator drift to a
        # different separator without moving checker_fp.
        assert ctx == "Title: A synthetic study\nDOI: 10.0/demo"
        # Warning recorded on the session and on stderr.
        assert len(stub.session.warnings) == 1
        assert "identity-degradation" in stub.session.warnings[0]
        assert "identity-degradation" in capsys.readouterr().err

    def test_degradation_warning_only_fires_once(self):
        stub = _ctx_stub(manifest_summary=None, extracted_value=None)
        Orchestrator._study_identity_context(stub)
        Orchestrator._study_identity_context(stub)
        assert len(stub.session.warnings) == 1

    def test_the_degradation_warning_names_the_field_that_was_empty(self):
        # A reader of run.json has to know WHICH field to go and populate, and
        # a template may declare any variable name for the summary role. So the
        # warning quotes the name from the template rather than describing the
        # role in the abstract, and names the study alongside it.
        stub = _ctx_stub(manifest_summary=None, extracted_value=None,
                         role_summary_var="paper_abstract")
        Orchestrator._study_identity_context(stub)
        warning = stub.session.warnings[0]
        assert "'paper_abstract'" in warning
        assert "demo-001" in warning


# ---------------------------------------------------------------------------
# Mismatch tripwire (e)
# ---------------------------------------------------------------------------

class TestMismatchAdvisory:
    """At checker time the tripwire only ADVISES: it prints to stderr and
    records a session event, and never persists to meta.warnings. The value it
    sees is mid-run and may still change."""

    def test_divergent_summaries_advise_but_do_not_persist(self, capsys):
        stub = _ctx_stub(
            manifest_summary="A study of widgets in the automotive sector.",
            extracted_value=("This paper measures the Fictional Durability "
                             "Gauge against synthetic lifecycle outcomes."),
        )
        Orchestrator._study_identity_context(stub)
        assert stub.session.warnings == []
        assert "summary-mismatch" in capsys.readouterr().err
        events = [e for e in stub.session.events
                  if e["event"] == "summary_mismatch_advisory"]
        assert len(events) == 1
        # Honestly worded: the event says what it observed and when, and does
        # not claim the finished artefact is wrong.
        assert "Mid-run observation only" in events[0]["message"]
        assert events[0]["phase"] == "extracting"

    def test_matching_summaries_say_nothing(self, capsys):
        # The extracted abstract contains the manifest summary verbatim
        # (a truncated search-index abstract): containment => no advisory.
        full = ("This paper measures the Fictional Durability Gauge against "
                "synthetic lifecycle outcomes in an invented batch.")
        stub = _ctx_stub(
            manifest_summary="Fictional Durability Gauge against synthetic "
                             "lifecycle outcomes",
            extracted_value=full,
        )
        Orchestrator._study_identity_context(stub)
        assert stub.session.warnings == []
        assert stub.session.events == []
        assert capsys.readouterr().err == ""

    def test_advisory_only_fires_once(self):
        stub = _ctx_stub(
            manifest_summary="Completely unrelated widget study text here.",
            extracted_value="Fictional Durability Gauge synthetic outcomes.",
        )
        Orchestrator._study_identity_context(stub)
        Orchestrator._study_identity_context(stub)
        assert len(stub.session.events) == 1

    def test_a_divergence_after_a_matching_check_is_still_advised(self, capsys):
        # The latch counts advisories, not comparisons. The context is rebuilt
        # for every checked field, so the first check routinely sees a matching
        # value; if that spent the segment's one advisory, a summary later
        # overwritten with the wrong paper's would reach the operator only at
        # finalisation, which is the one case the mid-run heads-up exists for.
        full = ("This paper measures the Fictional Durability Gauge against "
                "synthetic lifecycle outcomes in an invented batch.")
        stub = _ctx_stub(
            manifest_summary="Fictional Durability Gauge against synthetic "
                             "lifecycle outcomes",
            extracted_value=full,
        )
        Orchestrator._study_identity_context(stub)
        assert stub.session.events == []
        assert capsys.readouterr().err == ""

        # A later tool call rewrites the role:summary field with a value that
        # describes a different paper.
        stub.extraction_record.study["abstract"]["value"] = (
            "A study of widgets in the automotive sector.")
        Orchestrator._study_identity_context(stub)
        assert "summary-mismatch" in capsys.readouterr().err
        assert len(stub.session.events) == 1

        # Still at most one per segment: the ceiling the latch exists for.
        Orchestrator._study_identity_context(stub)
        assert len(stub.session.events) == 1


class TestShippedMismatchWarning:
    """The persisted warning is decided at finalisation, against the value the
    run ships, so meta.warnings describes the finished artefact."""

    def test_divergent_shipped_summary_persists_a_warning(self, capsys):
        stub = _ctx_stub(
            manifest_summary="A study of widgets in the automotive sector.",
            extracted_value=("This paper measures the Fictional Durability "
                             "Gauge against synthetic lifecycle outcomes."),
        )
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert len(stub.session.warnings) == 1
        assert "summary-mismatch" in stub.session.warnings[0]
        assert "shipped role:summary value" in stub.session.warnings[0]
        assert "summary-mismatch" in capsys.readouterr().err

    def test_matching_shipped_summary_persists_nothing(self):
        stub = _ctx_stub(
            manifest_summary="Fictional Durability Gauge against synthetic "
                             "lifecycle outcomes",
            extracted_value=("This paper measures the Fictional Durability "
                             "Gauge against synthetic lifecycle outcomes in "
                             "an invented batch."),
        )
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert stub.session.warnings == []

    def test_no_manifest_summary_cannot_compare(self):
        stub = _ctx_stub(manifest_summary=None,
                         extracted_value="Anything at all.")
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert stub.session.warnings == []

    def test_empty_shipped_summary_cannot_compare(self):
        # Nothing to compare against, on any status.
        stub = _ctx_stub(manifest_summary="A study of widgets.",
                         extracted_value=None)
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert stub.session.warnings == []

    def test_no_role_summary_field_cannot_compare(self):
        stub = _ctx_stub(manifest_summary="A study of widgets.",
                         role_summary_var=None, extracted_value=None)
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert stub.session.warnings == []

    def test_a_run_carrying_an_unresolved_challenge_is_still_judged(self):
        # A challenge is advisory and never changes the status, so a run that
        # shipped a field the checker was still unhappy with finalises
        # `complete` like any other: an answer the run stands behind, and one
        # this tripwire judges.
        stub = _ctx_stub(
            manifest_summary="A study of widgets in the automotive sector.",
            extracted_value=("This paper measures the Fictional Durability "
                             "Gauge against synthetic lifecycle outcomes."),
        )
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert len(stub.session.warnings) == 1

    @pytest.mark.parametrize("status", ["failed_validation", "error"])
    def test_an_aborted_run_is_not_judged(self, status):
        # An aborted run stopped part-way, so its role:summary field is a
        # work-in-progress snapshot, not an answer: a PARTIAL abstract (not
        # empty, just unfinished) diverges for the obvious reason, and every
        # hypothesis the warning offers would be false of it. This is the same
        # mid-run state the advisory refuses to persist, reached by a
        # different door.
        #
        # The partial value is in the PAPER's wording, not a prefix of the
        # manifest summary, so it genuinely diverges (a prefix would match by
        # containment and the test would pass without the status gate).
        manifest = ("A study of widget durability in automotive "
                    "manufacturing, measured over ten years.")
        partial = "Background. Fatigue testing was performed on"
        assert not _summaries_match(manifest, partial)
        stub = _ctx_stub(manifest_summary=manifest, extracted_value=partial)
        Orchestrator._check_shipped_summary_mismatch(stub, status)
        assert stub.session.warnings == []

        # The same value on a considered answer DOES warn: the status gate is
        # what silences it above, not the comparison.
        considered = _ctx_stub(manifest_summary=manifest,
                               extracted_value=partial)
        Orchestrator._check_shipped_summary_mismatch(considered, "complete")
        assert len(considered.session.warnings) == 1

    def test_the_warning_names_a_mis_extraction_as_a_possibility(self):
        # On a considered answer a divergence has a third honest explanation
        # besides the two about the bundle: the extractor got the field wrong.
        stub = _ctx_stub(
            manifest_summary="A study of widgets in the automotive sector.",
            extracted_value=("This paper measures the Fictional Durability "
                             "Gauge against synthetic lifecycle outcomes."),
        )
        Orchestrator._check_shipped_summary_mismatch(stub, "complete")
        assert "extraction may have got the field wrong" in \
            stub.session.warnings[0]


# ---------------------------------------------------------------------------
# A mid-run mismatch that resolves leaves no false warning
# ---------------------------------------------------------------------------

class TestResolvedMismatchLeavesNoFalseWarning:
    """A mismatch seen mid-run and gone by finalisation persists NOTHING.

    An abstract still truncated when the first check assembles its identity
    context scores below threshold, so the fuzzy comparison diverges. The
    checker challenges the truncated value, the extractor completes it, and
    the shipped value scores comfortably above threshold. A warning latched
    at that first reading would be true of a transient mid-run state and
    false of the finished artefact, and both of its stated hypotheses (wrong
    paper, distrusted search-index abstract) would be untrue of the run. So
    the persisted warning is decided against the value the run ships.

    A real session is used (offline: no adapter, no API call) so the wiring
    through `_finalise` is exercised, not just the comparison.

    These assertions read `_mismatch_warnings`, not the whole list: a real
    session records whatever else the run legitimately warned about (a dirty
    code tree, say, which depends on the developer's working copy), and this
    class is about one warning.
    """

    # A first-round value truncated part-way through the abstract scores far
    # below the threshold (0.6) against the fixture's manifest summary, and
    # the completed value scores comfortably above it, on the ratio and NOT
    # on containment.
    #
    # TRUNCATED is written out, because its job is to be the paper's own
    # opening wording and nothing like the manifest summary. The completed
    # value is DERIVED from the fixture bundle's summary rather than
    # transcribed from it: what it has to be is "the same paper's abstract,
    # finished, in slightly different words", and re-wording the tail keeps
    # that true — and keeps it a ratio match rather than a containment one —
    # whatever wording the fixture itself carries. Both relationships are
    # asserted at every use site below, never assumed.
    TRUNCATED = ("Background. Load-bearing brackets in heavy-duty service "
                 "account for a substantial share of maintenance spend.")

    def _completed(self, orch):
        """The finished abstract: the bundle's own summary with a re-worded
        tail, so neither string contains the other and the tripwire has to
        decide on the difflib ratio."""
        return orch.bundle.summary.rstrip(". ") + \
            ", as recorded in the fabricated dataset."

    def _orch(self, config_dir, bundle_minimal_dir, tmp_path):
        from meltiro.config_bundle import load_config_bundle
        config = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        orch = Orchestrator(
            config, bundle, tmp_path / "runs",
            extractor_model="claude-opus-4-8",
            review_model=None,
            max_checks_per_field=0, final_review=False,
            api_key="x")
        orch.prepare_new_session()
        return orch

    def _mismatch_warnings(self, orch):
        return [w for w in orch.session.meta["warnings"]
                if w.startswith("summary-mismatch")]

    def _set_abstract(self, orch, text):
        orch.extraction_record.study["abstract"] = {
            "value": text, "evidence": [], "source": "Abstract"}

    def test_mismatch_at_checker_time_that_resolves_persists_no_warning(
            self, config_dir, bundle_minimal_dir, tmp_path, capsys):
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        # The first check sees a truncated abstract and advises.
        self._set_abstract(orch, self.TRUNCATED)
        assert not _summaries_match(orch.bundle.summary, self.TRUNCATED)
        orch._study_identity_context()
        assert "summary-mismatch" in capsys.readouterr().err

        # The checker challenges the truncated value, the extractor completes
        # it, and the run finalises on a value that passes comfortably.
        completed = self._completed(orch)
        self._set_abstract(orch, completed)
        assert _summaries_match(orch.bundle.summary, completed)
        orch._finalise("complete")

        # meta.warnings describes the finished artefact, which matches.
        assert self._mismatch_warnings(orch) == []
        # The trace of the mid-run divergence survives in the transcript,
        # honestly labelled.
        advisories = [e for e in orch.session.read_events()
                      if e.get("event") == "summary_mismatch_advisory"]
        assert len(advisories) == 1

    def test_mismatch_that_survives_to_the_output_persists_a_warning(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        self._set_abstract(
            orch, "A study of widget durability in automotive manufacturing.")
        orch._finalise("complete")
        warnings = self._mismatch_warnings(orch)
        assert len(warnings) == 1

    def test_check_runs_with_the_checker_stage_off(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The comparison is a property of the bundle and the shipped output,
        # not of the checker stage: this orchestrator never runs a checker
        # round (max_checks_per_field=0), and the warning still lands.
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        assert orch.checker_enabled is False
        self._set_abstract(
            orch, "A study of widget durability in automotive manufacturing.")
        orch._finalise("complete")
        assert len(self._mismatch_warnings(orch)) == 1

    def test_an_aborted_run_ships_a_partial_abstract_without_a_warning(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The check runs at `_finalise`, so it runs on the terminal paths too
        # (`failed_validation`, `error`). A surrendered run's half-written
        # abstract diverges because it never finished, not because anything
        # the warning hypothesises is true, so the check declines to judge it.
        # The status and failure_reason are what tell the consumer not to
        # trust this output.
        #
        # PARTIAL, not empty, and in the paper's own wording rather than a
        # prefix of the manifest summary, so it genuinely diverges: this is
        # the case the empty-value guard does not cover.
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        partial = self.TRUNCATED
        assert not _summaries_match(orch.bundle.summary, partial)
        self._set_abstract(orch, partial)
        orch._finalise("failed_validation", failure_reason="surrendered")
        assert self._mismatch_warnings(orch) == []

    def test_the_same_partial_abstract_on_a_considered_answer_does_warn(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The pair to the test above: the status is the only difference, so it
        # is the status gate doing the work, not the comparison.
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        self._set_abstract(orch, self.TRUNCATED)
        orch._finalise("complete")
        assert len(self._mismatch_warnings(orch)) == 1

    def test_the_persisted_warning_reaches_the_run_json_and_the_transcript(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # meta.warnings is only worth writing if a reader meets it. The two
        # places a reader looks are the machine-readable run.json and the
        # rendered transcript, and the transcript is a SEPARATE render over the
        # finished session rather than a view of the same dict, so it can stop
        # carrying the warning without run.json changing.
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        self._set_abstract(
            orch, "A study of widget durability in automotive manufacturing.")
        orch._finalise("complete")
        expected = self._mismatch_warnings(orch)
        assert len(expected) == 1

        session_dir = orch.session.session_dir
        on_disk = json.loads(
            (session_dir / "diagnostics" / "run.json").read_text())
        assert expected[0] in on_disk["warnings"]

        document = (session_dir / "diagnostics" / "transcript.md").read_text()
        assert "### Warnings the run recorded" in document
        assert expected[0] in document

    def test_a_matching_run_leaves_no_mismatch_warning_in_the_transcript(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The pair to the test above: the transcript carries the run's own
        # warning list rather than printing that sentence unconditionally.
        # Keyed on the mismatch warning alone, because a real session records
        # whatever else the run legitimately warned about.
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        self._set_abstract(orch, self._completed(orch))
        orch._finalise("complete")
        assert self._mismatch_warnings(orch) == []
        document = (orch.session.session_dir /
                    "diagnostics" / "transcript.md").read_text()
        assert "### Warnings the run recorded" in document
        assert "summary-mismatch" not in document

    def test_evaluated_on_a_resumed_run_that_never_re_enters_the_checker(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # The check must not hang off an in-memory latch or a stage a resumed
        # segment may skip: a fresh Orchestrator reattached to the session
        # evaluates it at finalisation from persisted state alone.
        orch = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        self._set_abstract(
            orch, "A study of widget durability in automotive manufacturing.")
        orch.session.write_extraction_record(orch.extraction_record)
        session_dir = orch.session.session_dir

        orch2 = self._orch(config_dir, bundle_minimal_dir, tmp_path)
        orch2.resume_session(session_dir)
        assert orch2._summary_mismatch_advised is False
        orch2._finalise("complete")
        assert len(self._mismatch_warnings(orch2)) == 1


# ---------------------------------------------------------------------------
# Fuzzy-match rule
# ---------------------------------------------------------------------------

class TestSummariesMatch:
    def test_identical_match(self):
        assert _summaries_match("Same text here.", "Same text here.")

    def test_containment_matches(self):
        assert _summaries_match("durability gauge outcomes",
                                "the durability gauge outcomes in a batch")

    def test_case_and_whitespace_insensitive(self):
        assert _summaries_match("Hello   World", "hello world")

    def test_wholly_different_does_not_match(self):
        assert not _summaries_match(
            "A study of widget durability in automotive manufacturing.",
            "Measuring the durability gauge against lifecycle outcomes.")

    def test_empty_side_treated_as_uncomparable(self):
        # Cannot compare against an empty normalised form: do not flag.
        assert _summaries_match("", "anything at all")
        assert _summaries_match("anything at all", "   ")
