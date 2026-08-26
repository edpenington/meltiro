"""Shared fixtures for agentic_extraction tests.

Provides a small synthetic template carrying the structure of a full
extraction template while staying small enough to reason about. Used by the
tools, orchestrator and checker tests.

Both scopes carry a quality-assessment section (`qa: true`). Those sections are
ordinary sections of `study_fields` / `record_fields`, so the fixture exercises
the path where a QA field reaches the tool schemas, the validator, and the
checker's per-field fan-out. Holding QA in a separate block and leaving it
empty is the tempting simplification, and it hides a checker that never looks
at a QA field at all.

It also provides the `stage_keys` fixture, which a module reaching the
orchestrator's pre-spend key preflight opts into.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def stage_keys(monkeypatch):
    """Every API key variable the registry names, set to a placeholder.

    `Orchestrator._preflight_keys` asks the environment for the variable each
    enabled stage's model resolves to, which is the same question
    `direktoro.build_adapter` answers. A test that reaches `run()` therefore
    needs those variables present, and a test that took them from the
    developer's shell would pass or fail by whose machine it ran on. Every
    provider call in these tests is stubbed, so no value here is ever sent
    anywhere.

    The set is read from the registry rather than listed, so a new provider's
    variable is covered the day the registry names it. A test that is ABOUT a
    missing key deletes the one it means in its own body, which runs after
    this.

    Opted into per module (`pytestmark = pytest.mark.usefixtures(
    "stage_keys")`) rather than autouse, so a module that says nothing about
    keys is not silently given any.
    """
    from direktoro import known_models, model_info

    for env in sorted({model_info(m).api_key_env for m in known_models()}):
        monkeypatch.setenv(env, "not-a-real-key")


# The synthetic template's two check blocks, answered in full, in the flat
# `variable -> value` shape the two check-recording tools take (no envelopes).
# Every `required: true` variable is present, so a call carrying one of these
# satisfies that block's completeness gate outright.
INITIAL_CHECK_FIELDS = {
    "text_readable": True,
    "figure_tables_included": True,
    "expected_relationships": 1,
}
QUALITY_CHECK_FIELDS = {"deviation_from_expectations": "None"}


def open_initial_check_gate(extraction_record):
    """Latch the ordering gate without dispatching `record_initial_check`.

    The dispatcher refuses every extractor mutation until the initial check
    has landed. `initial_check_recorded` is a plain attribute, so a test whose
    subject is something else (envelope validation, scope notes,
    canonicalisation, unknown-field hints) opens the gate in one line here
    rather than prefixing every call with an unrelated tool call. Tests that
    are ABOUT the ordering rule make the real call instead; see
    test_tools.py::TestTheInitialCheckGate.

    Returns the record, so it composes inside a fixture.
    """
    extraction_record.initial_check_recorded = True
    return extraction_record


def checker_trigger_orch(template, extraction_record, *,
                         max_checks_per_field=2, final_review=True,
                         check_counts=None):
    """A bare Orchestrator carrying exactly what `_build_checker_calls` reads.

    The trigger is pure given a template, an extraction record, and the
    per-field counts, so a test can exercise it without a session, a config
    bundle, or a provider. `build_checker_user_message` still runs for real
    unless the caller stubs it, so the caller supplies a checker user template
    path when it wants the rendered message.
    """
    from meltiro.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.template = template
    orch.extraction_record = extraction_record
    orch.image_labels = set()
    orch.image_notes = {}
    orch.image_tables = {}
    orch.bundle = SimpleNamespace(figures={}, tables={})
    orch.config = SimpleNamespace(partials_dir="/unused")
    orch.checker_config = SimpleNamespace(
        checker_model="claude-sonnet-4-6", context_chars=0)
    # No paper text and no context width: these tests exercise the trigger,
    # not the quote-context window, so the rendered evidence block is the
    # quote alone.
    orch.paper_text = None
    # The checker's user template may carry `{include_if:review:...}`, so the
    # trigger renders it against the run's instrument, which is what owns the
    # structure toggles. These are the fields that instrument is built from.
    orch.reference_lists = {}
    orch.max_checks_per_field = max_checks_per_field
    orch.final_review = final_review
    orch.check_reviewer_edits = False
    orch._check_counts = dict(check_counts or {})
    orch._study_identity_context = lambda: "Summary: ctx"
    return orch


def _field(variable, field_type="free_text", options=None, allow_other=False,
           description="", extraction_instruction=None,
           canonical_reference=None, required=False):
    return {
        "variable": variable,
        "field_type": field_type,
        "options": options,
        "allow_other": allow_other,
        "description": description,
        "extraction_instruction": extraction_instruction,
        "canonical_reference": canonical_reference,
        "required": required,
    }


def _section(name, fields, extraction_instruction=None, qa=False):
    return {
        "section": name,
        "fields": fields,
        "extraction_instruction": extraction_instruction,
        "qa": qa,
    }


@pytest.fixture
def synthetic_template():
    """A trimmed-down extraction template suitable for unit tests."""
    return {
        "record_entity": {
            "singular": "relationship",
            "plural": "relationships",
            "description": "a reported relationship between a gauge score "
                           "and a lifecycle outcome",
        },
        "study_fields": [
            _section("Identity", [
                _field("primary_aim", field_type="free_text"),
                _field("sample_size", field_type="integer"),
                _field("publication_type", field_type="categorical",
                       options=["Academic paper", "Government report"],
                       allow_other=True),
            ]),
            _section("Reporting", [
                _field("qa_reporting", field_type="categorical",
                       options=["Compliant", "Non-compliant", "Not reported"]),
            ], qa=True),
        ],
        "record_fields": [
            _section("Core", [
                _field("gauge", field_type="free_text", required=True),
                _field("outcome_variable", field_type="free_text",
                       required=True),
                _field("outcome_category", field_type="categorical",
                       options=["Cost or resource use", "Service life",
                                "Failure state"], required=True),
                _field("index_tariff", field_type="free_text"),
                _field("cost_source", field_type="free_text"),
                _field("failure_state_definition", field_type="free_text"),
                _field("effect_size", field_type="free_text"),
                _field("statistical_method", field_type="free_text"),
                _field("effect_type", field_type="free_text"),
                _field("subgroup", field_type="free_text"),
                _field("gauge_score_format", field_type="free_text"),
            ]),
            _section("Sample and Selection", [
                _field("rqa_sample_adequate", field_type="categorical",
                       options=["Adequate", "Inadequate", "Unclear"]),
            ], qa=True),
        ],
        "checker_context_fields": [
            "gauge", "gauge_score_format", "outcome_variable",
            "outcome_category", "statistical_method", "effect_type",
            "subgroup",
        ],
        # Cross-field gate rules mirror the worked config: each gated field is
        # only expected under the matching outcome_category value.
        "gates": [
            {"when_field": "outcome_category", "field": "index_tariff",
             "allowed_values": ["Service life"]},
            {"when_field": "outcome_category", "field": "cost_source",
             "allowed_values": ["Cost or resource use"]},
            {"when_field": "outcome_category", "field": "failure_state_definition",
             "allowed_values": ["Failure state"]},
        ],
        "initial_check_fields": [
            _section("Initial Check", [
                _field("text_readable", field_type="boolean", required=True),
                _field("figure_tables_included", field_type="boolean",
                       required=True),
                _field("expected_relationships", field_type="integer",
                       required=True),
            ]),
        ],
        "quality_check_fields": [
            _section("Quality Check", [
                _field("deviation_from_expectations", field_type="free_text",
                       required=True),
            ]),
        ],
        "template_hash": "test-template-hash",
        "template_path": "/tmp/test-template.yaml",
    }


@pytest.fixture
def paper_text():
    return (
        "Methods. The WDS-9 was administered to 348 units under "
        "load. Total Fleet Service Costs were estimated using "
        "2019/20 unit costs.\n\n"
        "Results. The odds ratio for unplanned removal was 1.34 "
        "(95% CI 1.10-1.62). DI-4 index was used for service-life outcomes."
    )


@pytest.fixture
def image_labels():
    return {"table_01", "table_02", "figure_01"}
