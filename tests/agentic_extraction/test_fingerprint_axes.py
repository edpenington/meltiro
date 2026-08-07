"""The three orthogonal fingerprint axes.

A run records `instrument_fp`, a per-role `*_call_fp`, and `engine_fp` beside
the three stage fingerprints. The stage fingerprints answer "did this stage's
inputs change?", which is the right question for refusing a drifted resume and
the wrong one for comparing runs, because each blends what the config author
wrote with which model was called and says nothing at all about which engine
ran.

These tests pin the two comparisons the axes exist to make possible, each of
which the stage fingerprints cannot answer:

  - same instrument, different model. Every stage fingerprint moves at once,
    so they cannot say the question was unchanged. `instrument_fp` holds.
  - same instrument, same models, different *meltiro*. No stage fingerprint
    moves at all, so they cannot say the engine changed. `engine_fp` does.

Everything here is offline: dry-run orchestrators, no network, no API key.
"""

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig, DEFAULT_CONTEXT_CHARS
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import (
    engine_fingerprint,
    instrument_fingerprint,
    instrument_structure_hash,
    run_fingerprint,
)
from meltiro.orchestrator import Orchestrator
from meltiro.rates import parse_rates


def _orch(config, bundle, out_dir, **over):
    """A prepared dry-run orchestrator, with pipeline.yaml overridable."""
    loop = dict(config.pipeline)
    loop.update(over)
    checker_config = CheckerConfig.from_env(
        model_override=loop["checker_model"])
    orch = Orchestrator(
        config, bundle, out_dir,
        # Parsed from the (overridable) pipeline mapping exactly as the CLI
        # parses it, so a `rates` override travels the real path into the run.
        rates=parse_rates(loop),
        extractor_model=loop["extractor_model"],
        checker_config=checker_config,
        review_model=loop["review_model"],
        max_tool_calls=int(loop["max_tool_calls"]),
        max_checks_per_field=int(loop["max_checks_per_field"]),
        final_review=bool(loop.get("final_review", True)),
        check_reviewer_edits=bool(loop.get("check_reviewer_edits", False)),
        temperature=float(loop["temperature"]),
        extractor_max_tokens=int(loop["extractor_max_tokens"]),
        review_max_tokens=int(loop["review_max_tokens"]),
        dry_run=True,
    )
    orch.prepare_new_session()
    return orch


@pytest.fixture
def meta(tmp_path, config_dir, bundle_minimal_dir):
    def _build(name="runs", **over):
        return _orch(load_config_bundle(config_dir),
                     load_bundle(bundle_minimal_dir),
                     tmp_path / name, **over).session.meta
    return _build


class TestSameInstrumentDifferentModel:
    """The model-comparison case: hold the config, swap the API."""

    def test_instrument_holds_while_every_stage_fingerprint_moves(self, meta):
        # Swapping the extractor model is the A/B a model comparison runs. The
        # instrument is untouched by it, so instrument_fp must not move, and
        # that is exactly what makes the two runs comparable.
        a = meta("a")
        b = meta("b", extractor_model="claude-sonnet-4-6")
        assert a["instrument_fp"] == b["instrument_fp"]
        # ... while the stage fingerprint for that role does move, which is
        # why config_fp cannot answer "same question?" on its own.
        assert a["config_fp"] != b["config_fp"]
        assert a["extractor_call_fp"] != b["extractor_call_fp"]

    def test_untouched_roles_keep_their_call_fingerprints(self, meta):
        # Only the role whose model changed moves. A single run-wide call
        # fingerprint would hide which stage was swapped; per-role values say.
        a = meta("a")
        b = meta("b", extractor_model="claude-sonnet-4-6")
        assert a["checker_call_fp"] == b["checker_call_fp"]
        assert a["review_call_fp"] == b["review_call_fp"]

    def test_swapping_only_the_checker_moves_only_the_checker(self, meta):
        a = meta("a")
        b = meta("b", checker_model="claude-opus-4-8")
        assert a["instrument_fp"] == b["instrument_fp"]
        assert a["checker_call_fp"] != b["checker_call_fp"]
        assert a["extractor_call_fp"] == b["extractor_call_fp"]
        assert a["review_call_fp"] == b["review_call_fp"]

    def test_review_call_fp_reads_the_reviewer_s_own_params(
            self, tmp_path, config_dir, bundle_minimal_dir):
        # The shipped config runs the same model as extractor and reviewer
        # with the same resolved params, so their two call fingerprints are
        # legitimately EQUAL. That equality would also be produced by plumbing
        # that fed the extractor's params into the reviewer's fingerprint, so
        # pin the thing that tells the two apart: moving review_max_tokens
        # alone must move review_call_fp and nothing else.
        cb = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        a = _orch(cb, bundle, tmp_path / "a").session.meta
        b = _orch(cb, bundle, tmp_path / "b",
                  review_max_tokens=4096).session.meta
        assert a["review_call_fp"] != b["review_call_fp"]
        assert a["extractor_call_fp"] == b["extractor_call_fp"]
        assert a["instrument_fp"] == b["instrument_fp"]

    def test_call_fingerprints_are_null_for_a_disabled_stage(self, meta):
        # A disabled stage records a null call fingerprint, mirroring its null
        # stage fingerprint, rather than a fingerprint of a call never made.
        m = meta("off", max_checks_per_field=0)
        assert m["checker_call_fp"] is None
        assert m["checker_fp"] is None
        assert m["extractor_call_fp"] is not None


class TestSameEverythingDifferentEngine:
    """The engine case: nothing about the question changed, the code did."""

    def test_engine_fingerprint_moves_on_version_and_source(self):
        base = engine_fingerprint("1.2.3", "src-a")
        assert base != engine_fingerprint("1.2.4", "src-a")
        assert base != engine_fingerprint("1.2.3", "src-b")

    def test_an_edited_copy_never_shares_a_fingerprint_with_the_release(self):
        # The axis identifies the code by content, so a working edit, a patched
        # site-packages copy and the release they started from are three
        # distinct engines and the fingerprint says so. Nothing about where the
        # copy sits enters it, which is what makes the answer the same from a
        # checkout and from a wheel built out of one.
        release = engine_fingerprint("1.2.3", "src-a", "4.5.6", "dsrc-a")
        patched = engine_fingerprint("1.2.3", "src-a-edited", "4.5.6", "dsrc-a")
        assert release != patched

    def test_unreadable_source_still_fingerprints_on_the_version(self):
        # A frozen or zipimported copy folds in the `nosource` token: the
        # version is the best available answer, and must still produce a usable
        # value rather than collapsing two releases onto the shared token.
        assert (engine_fingerprint("1.2.3", "nosource")
                != engine_fingerprint("1.2.4", "nosource"))

    def test_run_fingerprint_moves_with_the_engine_alone(self):
        # The point of folding the engine into run_fp. Identical config on both
        # sides: every stage fingerprint is held fixed, so without the engine
        # term the two runs would share a run_fp while asking materially
        # different questions.
        args = ("config_fp:x", "checker_fp:y", "review_fp:z")
        assert (run_fingerprint(*args, "engine_fp:one")
                != run_fingerprint(*args, "engine_fp:two"))

    def test_a_run_records_an_engine_fingerprint(self, meta):
        m = meta()
        assert m["engine_fp"].startswith("engine_fp:")
        assert m["run_fp"] == run_fingerprint(
            m["config_fp"], m["checker_fp"], m["review_fp"], m["engine_fp"])


class TestInstrumentCoversWhatTheAuthorWrote:
    """Everything a config author can edit must move instrument_fp."""

    def test_structure_toggles_move_it(self, meta):
        # All THREE toggles `instrument_structure_hash` carries. Leaving one
        # out would let a config author change what the pipeline does while
        # the axis reported the same instrument, which is the one thing this
        # axis must never do.
        base = meta("base")
        assert meta("a", max_checks_per_field=0)["instrument_fp"] \
            != base["instrument_fp"]
        assert meta("b", final_review=False)["instrument_fp"] \
            != base["instrument_fp"]
        assert meta("c", check_reviewer_edits=True)["instrument_fp"] \
            != base["instrument_fp"]

    def test_check_reviewer_edits_moves_the_structure_component_itself(self):
        # Direct on the component, not only through a run: the toggle extends
        # the checker to the reviewer's own tool calls, so a config setting it
        # runs a different pipeline and the structure word has to differ.
        assert (instrument_structure_hash(2, check_reviewer_edits=True)
                != instrument_structure_hash(2, check_reviewer_edits=False))

    def test_the_three_structure_toggles_stay_mutually_distinct(self):
        # Each toggle needs its own mark in the word. Two that produced the
        # same suffix would make a pair of genuinely different pipelines share
        # an instrument, and the collision would be invisible in every
        # fingerprint built on it.
        words = {
            instrument_structure_hash(2),
            instrument_structure_hash(0),
            instrument_structure_hash(2, final_review=False),
            instrument_structure_hash(2, check_reviewer_edits=True),
            instrument_structure_hash(2, final_review=False,
                                      check_reviewer_edits=True),
        }
        assert len(words) == 5

    def test_checker_context_width_moves_it(self, tmp_path, config_dir,
                                            bundle_minimal_dir):
        # The width changes the question the checker is asked (a table cell
        # judged with its column header is a different question), so it is
        # instrument, not an operational knob.
        cb = load_config_bundle(config_dir)
        bundle = load_bundle(bundle_minimal_dir)
        a = _orch(cb, bundle, tmp_path / "a")
        b = _orch(cb, bundle, tmp_path / "b")
        b.checker_config.context_chars = 5000
        assert (a._build_fingerprints()["instrument_fp"]
                != b._build_fingerprints()["instrument_fp"])

    def test_no_checker_is_not_a_zero_width_window(self):
        # Two different instruments: one asks the checker nothing at all, the
        # other asks it about every quote with none of the paper around it.
        # Every other component is held equal here, so the distinction rests
        # on this one and cannot be borrowed from the structure component that
        # happens to sit beside it in a real run.
        assert (instrument_fingerprint("ph", "th", checker_context_chars=None)
                != instrument_fingerprint("ph", "th",
                                          checker_context_chars=0))

    def test_an_omitted_width_reads_as_no_checker(self):
        # The absent case is the default, matching the `="none"` sentinel the
        # other optional components take.
        assert (instrument_fingerprint("ph", "th")
                == instrument_fingerprint("ph", "th",
                                          checker_context_chars=None))

    def test_prompt_and_template_and_reference_edits_move_it(self):
        # Component-level, so each input is pinned independently rather than
        # relying on one composite happening to change.
        base = instrument_fingerprint("ph", "th", tool_set_hash="ts",
                                      reference_hash="rh")
        assert base != instrument_fingerprint("PH2", "th", tool_set_hash="ts",
                                              reference_hash="rh")
        assert base != instrument_fingerprint("ph", "TH2", tool_set_hash="ts",
                                              reference_hash="rh")
        assert base != instrument_fingerprint("ph", "th", tool_set_hash="TS2",
                                              reference_hash="rh")
        assert base != instrument_fingerprint("ph", "th", tool_set_hash="ts",
                                              reference_hash="RH2")

    def test_checker_context_fields_are_ordered(self):
        # They render into the checker's per-record label, so a reordering is a
        # genuine edit to what the checker sees.
        a = instrument_fingerprint("ph", "th",
                                   checker_context_fields=["x", "y"])
        b = instrument_fingerprint("ph", "th",
                                   checker_context_fields=["y", "x"])
        assert a != b

    def test_image_capability_is_excluded(self):
        # instrument_structure_hash must not carry supports_images: that is a
        # property of the assigned model, and folding it in would make the
        # instrument axis move on a model swap, destroying the whole point.
        assert instrument_structure_hash(2) == "checks2"
        assert instrument_structure_hash(2, final_review=False) \
            == "checks2_noreview"
        assert instrument_structure_hash(2, check_reviewer_edits=True) \
            == "checks2_checkreview"


class TestBundleAndRunAgree:
    """`meltiro fingerprint` must print the value a run records."""

    # The tool catalogue is per role, and BOTH sides hash
    # `all_tool_definitions`: a `config_bundle` that hashed the extractor's
    # catalogue alone would make `meltiro fingerprint` print an instrument_fp
    # no run could ever record.
    def test_bundle_instrument_fp_equals_the_run_it_describes(
            self, tmp_path, config_dir, bundle_minimal_dir):
        # The bundle computes instrument_fp from the directory, applying its own
        # defaults for absent pipeline keys; the run computes it from the values
        # the orchestrator honoured. If those defaults ever drift apart, the
        # printed fingerprint becomes a lie about the run it claims to describe.
        cb = load_config_bundle(config_dir)
        orch = _orch(cb, load_bundle(bundle_minimal_dir), tmp_path / "runs")
        assert cb.instrument_fp == orch.session.meta["instrument_fp"]

    def test_they_agree_when_the_bundle_disables_the_checker(
            self, tmp_path, config_dir, bundle_minimal_dir):
        # A bundle with no checker has no quote-context window, whatever width
        # its `checker_context_chars` key names, and both sides have to read it
        # that way. A bundle that folded the width in regardless would print an
        # identity no run of it records — for the one configuration where the
        # key describes nothing that happens.
        import shutil
        import yaml
        root = tmp_path / "no-checker"
        shutil.copytree(config_dir, root)
        path = root / "pipeline.yaml"
        pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
        pipeline["max_checks_per_field"] = 0
        path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
        cb = load_config_bundle(root)
        orch = _orch(cb, load_bundle(bundle_minimal_dir), tmp_path / "runs")
        assert cb.instrument_fp == orch.session.meta["instrument_fp"]

    def test_a_cli_override_makes_the_run_differ_from_the_bundle(
            self, tmp_path, config_dir, bundle_minimal_dir):
        # An override changes the instrument, so the run's value is the true
        # one and the two SHOULD differ. Pinned so the divergence stays a
        # documented consequence rather than an unnoticed bug.
        cb = load_config_bundle(config_dir)
        orch = _orch(cb, load_bundle(bundle_minimal_dir), tmp_path / "runs",
                     max_checks_per_field=0)
        assert cb.instrument_fp != orch.session.meta["instrument_fp"]

    def test_default_context_chars_is_what_the_checker_actually_uses(self):
        # The bundle's default for an absent `checker_context_chars` has to be
        # the CheckerConfig default, or the two computations diverge on every
        # bundle that omits the key.
        assert CheckerConfig.from_env().context_chars == DEFAULT_CONTEXT_CHARS


class TestAxesReachTheRunLog:
    """The run log is the index a consumer sweeps, so the axes must be in it."""

    def test_entry_carries_every_axis(self, tmp_path, config_dir,
                                      bundle_minimal_dir):
        from meltiro.run_entry import build_entry
        cb = load_config_bundle(config_dir)
        orch = _orch(cb, load_bundle(bundle_minimal_dir), tmp_path / "runs")
        entry = build_entry(orch.session)
        meta = orch.session.meta
        for key in ("instrument_fp", "extractor_call_fp", "checker_call_fp",
                    "review_call_fp", "engine_fp", "run_fp"):
            assert entry[key] == meta[key], key


# A rate card per role, and a second set differing in every number. Two runs
# that differ only in these asked exactly the same questions of exactly the same
# models over exactly the same paper: the operator typed different prices, and
# nothing else.
_CARD = {role: {"input_per_1m": 3.0, "output_per_1m": 15.0,
                "cache_read_per_1m": 0.3, "cache_write_per_1m": 3.75}
         for role in ("extractor", "checker", "review")}
_OTHER_CARD = {role: {"input_per_1m": 1.25, "output_per_1m": 10.0,
                      "cache_read_per_1m": 0.125,
                      "cache_write_per_1m": 1.5625}
               for role in ("extractor", "checker", "review")}


def _identities(meta):
    """Every content identity a run records: each `*_fp` and each `*_hash`.

    Collected by SHAPE rather than by a hand-written list, so a fingerprint or
    hash added later is covered by the assertions below the moment it appears in
    a run's meta, instead of quietly escaping them.
    """
    return {k: v for k, v in meta.items()
            if k.endswith("_fp") or k.endswith("_hash")}


class TestRatesReachNoFingerprint:
    """A rate card says what a run COST, never what it asked.

    Prices are commercial: they belong to the operator's account and its
    invoices, and they move without anything about the extraction changing. Two
    runs differing only in the rates typed into `pipeline.yaml` are therefore
    the SAME instrument, the same call to the same model, and the same engine —
    and a comparison across them is a legitimate comparison, not a comparison of
    two different things.

    A rate card in any preimage would break that twice over. It would split one
    instrument into as many fingerprints as there are price lists, so runs that
    should group would not; and it would refuse a resume as config drift merely
    because a provider changed a price mid-run. Both are silent when they
    happen, so the guarantee is pinned here.
    """

    def test_adding_a_rate_card_moves_no_identity(self, meta):
        unpriced = _identities(meta("unpriced"))
        priced = _identities(meta("priced", rates=_CARD))
        assert unpriced, "no fingerprints found in the run's meta"
        assert priced == unpriced

    def test_changing_the_rates_moves_no_identity(self, meta):
        cheap = _identities(meta("cheap", rates=_CARD))
        dear = _identities(meta("dear", rates=_OTHER_CARD))
        assert cheap == dear

    def test_the_named_axes_are_among_them(self, meta):
        # The shape-based sweep above is only as strong as what it catches, so
        # the axes a consumer actually keys on are named here too. A rename that
        # dropped one out of the `_fp` / `_hash` shape would fail here rather
        # than silently shrinking the guarantee above.
        found = set(_identities(meta("named")))
        for key in ("instrument_fp", "config_fp", "checker_fp", "review_fp",
                    "run_fp", "extractor_call_fp", "checker_call_fp",
                    "review_call_fp", "engine_fp", "prompt_hash",
                    "template_hash", "tool_set_hash"):
            assert key in found, key
