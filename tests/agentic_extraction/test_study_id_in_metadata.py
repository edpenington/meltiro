"""The bundle study id lives in the run's output metadata, not in a record.

A downstream converter reads the study id from run.json's top-level
`study_id`, sourced from the bundle manifest, and NOT from a copy stamped on
each record: one run has one study, so one home for the id cannot disagree with
itself. This test pins that location so a change that drops it fails loudly.
"""

from meltiro.bundle import load_bundle
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator


def test_meta_json_carries_manifest_study_id(
        config_dir, bundle_minimal_dir, tmp_path):
    config = load_config_bundle(config_dir)
    bundle = load_bundle(bundle_minimal_dir)
    orch = Orchestrator(
        config, bundle, tmp_path / "runs",
        extractor_model="claude-opus-4-8",
        review_model=None,
        max_checks_per_field=0, final_review=False,
        extractor_max_tokens=4096,
        api_key="x")
    orch.prepare_new_session()
    # The output metadata carries the bundle manifest id at the top level.
    assert orch.session.meta["study_id"] == str(bundle.study_id)
    assert orch.session.meta["study_id"]  # non-empty
