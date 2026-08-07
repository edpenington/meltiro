"""What identifies the ENGINE behind a run, and what a fingerprint covers.

A stage fingerprint's promise is that the same fingerprint plus the same input
means the model was asked the same question. Two things are outside it, and
this module pins both halves of that boundary:

  - the config side is fingerprinted, the ENGINE side is not. *meltiro* writes
    prose of its own around the config's prompts (the user-block headers, the
    reviewer's unresolved-challenge preamble, the tool re-prompts, the wording
    filling the checker template's slots) and returns more of it from every
    tool call. None of it moves a stage fingerprint. What identifies it is the
    ENGINE AXIS: both packages' versions and a content hash of each one's
    source, recorded with every run, with the documented rule that runs from
    different *meltiro* versions are compared deliberately rather than assumed
    equivalent;
  - the INPUT is not fingerprinted into any of those either: the same config on
    another paper is the same question asked, so all three stage fingerprints
    hold. The paper's own identity is recorded separately (see
    `test_bundle_fingerprint.py`).

`source_hash` is what makes the first half exact: engine prose lives in the
package's source files, and hashing them names the code that ran wherever it
was installed from.
"""

import json
import shutil

from meltiro import __version__, prompt_builder
from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.orchestrator import Orchestrator
from meltiro.run_entry import build_entry


def _orch(config_dir, bundle_dir, out_dir, **kwargs):
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model="claude-opus-4-7",
        checker_config=CheckerConfig(
            checker_model="claude-sonnet-4-6", api_key="x"),
        review_model="claude-opus-4-7",
        api_key="x",
        **kwargs,
    )


def _prepared_orch(config_dir, bundle_dir, out_dir):
    """An Orchestrator with prepare_new_session run (no network)."""
    orch = _orch(config_dir, bundle_dir, out_dir)
    orch.prepare_new_session()
    return orch


def _fps(orch):
    m = orch.session.meta
    return m["config_fp"], m["checker_fp"], m["review_fp"]


# ---------------------------------------------------------------------------
# Engine prose rides in no fingerprint
# ---------------------------------------------------------------------------

class TestEngineProseMovesNoFingerprint:
    """The documented exclusion. Rewriting a piece of engine wording changes
    what a model reads, and deliberately moves no stage fingerprint: engine
    identity is carried by `meltiro_version` + `git_commit` instead, kept as a
    value separate from config identity."""

    def test_extractor_reprompt_edit_moves_no_stage_fingerprint(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        before = _fps(_prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "a"))
        monkeypatch.setattr(
            prompt_builder, "EXTRACTOR_TOOL_REPROMPT",
            "You must call a tool to proceed.")
        after = _fps(_prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "b"))
        assert after == before

    def test_review_user_block_edit_moves_no_stage_fingerprint(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        before = _fps(_prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "a"))
        real = prompt_builder.build_review_user_blocks

        def restyled(*args, **kwargs):
            blocks = real(*args, **kwargs)
            for block in blocks:
                if block.get("type") == "text":
                    block["text"] = block["text"].replace(
                        "Do NOT make stylistic changes", "Feel free to restyle")
            return blocks

        monkeypatch.setattr(
            prompt_builder, "build_review_user_blocks", restyled)
        after = _fps(_prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "b"))
        assert after == before


# ---------------------------------------------------------------------------
# The recorded engine identity
# ---------------------------------------------------------------------------

class TestMeltiroVersionIsRecorded:
    def test_meta_records_the_running_version(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _prepared_orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        assert orch.session.meta["meltiro_version"] == __version__
        # Persisted, not just in-memory: a consumer reads the file.
        with open(orch.session.meta_path, encoding="utf-8") as f:
            assert json.load(f)["meltiro_version"] == __version__

    def test_meta_records_it_beside_the_git_anchor(
            self, config_dir, bundle_minimal_dir, tmp_path):
        # Version and commit are the pair that identifies the engine; the
        # version is present even where git is not.
        meta = _prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "runs").session.meta
        assert set(meta) >= {"meltiro_version", "git_commit", "git_dirty"}

    def test_run_log_entry_records_the_running_version(
            self, config_dir, bundle_minimal_dir, tmp_path):
        orch = _prepared_orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        entry = build_entry(orch.session)
        assert entry["meltiro_version"] == __version__


class TestDirektoroVersionIsRecorded:
    """The other half of the engine is recorded too.

    direktoro builds the provider-call identity block that is the FIRST
    component of every stage fingerprint, resolves the decoding params that
    decide what is sent, and declares each model's image capability. A run
    recording only meltiro's version would leave the package most able to move
    a published fingerprint unnamed in the artefact, so two runs sharing a
    run_fp could have come from different engines with nothing to say so."""

    def test_meta_records_the_running_direktoro_version(
            self, config_dir, bundle_minimal_dir, tmp_path):
        import direktoro

        orch = _prepared_orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        assert orch.session.meta["direktoro_version"] == direktoro.__version__
        # Persisted, not just in-memory: a consumer reads the file.
        with open(orch.session.meta_path, encoding="utf-8") as f:
            assert json.load(f)["direktoro_version"] == direktoro.__version__

    def test_run_log_entry_records_it(
            self, config_dir, bundle_minimal_dir, tmp_path):
        import direktoro

        orch = _prepared_orch(config_dir, bundle_minimal_dir, tmp_path / "runs")
        entry = build_entry(orch.session)
        assert entry["direktoro_version"] == direktoro.__version__

    def test_it_is_recorded_beside_the_meltiro_version(
            self, config_dir, bundle_minimal_dir, tmp_path):
        meta = _prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "runs").session.meta
        assert set(meta) >= {"meltiro_version", "direktoro_version"}

    def test_it_moves_the_engine_fingerprint(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # The point of recording it: a run under a different direktoro is a
        # run under a different engine, and engine_fp has to say so.
        before = _prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "a").session.meta
        # Patched where each caller looks it up: `Session.create` imported the
        # name into meltiro.session, while `engine_identity` calls it inside
        # meltiro.run_log. Both have to move together or the recorded version
        # and the fingerprint would disagree.
        monkeypatch.setattr(
            "meltiro.run_log.direktoro_version", lambda: "0.0.0-not-real")
        monkeypatch.setattr(
            "meltiro.session.direktoro_version", lambda: "0.0.0-not-real")
        after = _prepared_orch(
            config_dir, bundle_minimal_dir, tmp_path / "b").session.meta
        assert after["direktoro_version"] == "0.0.0-not-real"
        assert after["engine_fp"] != before["engine_fp"]
        assert after["run_fp"] != before["run_fp"]
        # And only the engine axis moves: the instrument and the call are
        # untouched by which engine version composed them.
        assert after["instrument_fp"] == before["instrument_fp"]
        assert after["extractor_call_fp"] == before["extractor_call_fp"]

    def test_an_absent_direktoro_records_a_marker_rather_than_raising(
            self, monkeypatch):
        # `import meltiro` must work with direktoro absent, so reading the
        # version must not be the thing that raises. A --no-deps consumer that
        # reads and validates bundles gets a null and a distinct engine_fp.
        import builtins

        from meltiro.run_log import direktoro_version

        real_import = builtins.__import__

        def _no_direktoro(name, *args, **kwargs):
            if name == "direktoro":
                raise ImportError("no direktoro here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_direktoro)
        assert direktoro_version() is None

    def test_an_absent_direktoro_yields_no_source_digest_either(
            self, monkeypatch):
        # The source half answers to the same rule as the version half, and by
        # the same route: a --no-deps install gets a null rather than an
        # ImportError escaping into `import meltiro`.
        import builtins

        from meltiro.run_log import direktoro_source_hash

        real_import = builtins.__import__

        def _no_direktoro(name, *args, **kwargs):
            if name == "direktoro":
                raise ImportError("no direktoro here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_direktoro)
        assert direktoro_source_hash() is None


# ---------------------------------------------------------------------------
# The source digest: which code, as opposed to which checkout
# ---------------------------------------------------------------------------

class TestSourceHash:
    """`run_log.source_hash` is the engine axis's content half.

    A version names a release and a commit names a tree in one repository.
    Neither survives the ordinary ways code reaches a machine: an editable
    checkout with a working edit, a wheel installed into site-packages, a
    vendored copy. The digest over the package's own source files names the
    bytes, so the same code answers the same wherever it sits and different
    code always answers differently."""

    def test_it_is_a_sha256_of_the_running_package(self):
        from meltiro.run_log import source_hash

        digest = source_hash()
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_it_is_stable_across_calls(self):
        from meltiro.run_log import source_hash

        assert source_hash() == source_hash()

    def test_it_covers_the_imported_package_not_the_repository(self):
        # Anchored to `meltiro.__file__`'s directory, so an installed copy and
        # the checkout it was built from produce the same digest. Nothing
        # outside the package's own directory may reach it — README, tests and
        # packaging metadata are not the engine.
        from pathlib import Path

        import meltiro
        from meltiro.run_log import _hash_tree, source_hash

        assert source_hash() == _hash_tree(Path(meltiro.__file__).parent)

    def test_a_source_edit_moves_the_digest(self, tmp_path):
        # The property everything else rests on, exercised on a tree this test
        # owns: one byte changed in one file is a different engine.
        from meltiro.run_log import _hash_tree

        (tmp_path / "a.py").write_text("x = 1\n")
        before = _hash_tree(tmp_path)
        (tmp_path / "a.py").write_text("x = 2\n")
        assert _hash_tree(tmp_path) != before

    def test_a_rename_moves_the_digest(self, tmp_path):
        # The relative path is hashed beside the bytes, so moving code between
        # modules is a change even when every byte survives it.
        from meltiro.run_log import _hash_tree

        (tmp_path / "a.py").write_text("x = 1\n")
        before = _hash_tree(tmp_path)
        (tmp_path / "a.py").rename(tmp_path / "b.py")
        assert _hash_tree(tmp_path) != before

    def test_compiled_artefacts_are_excluded(self, tmp_path):
        # `__pycache__` and `.pyc` files are derived from the source and differ
        # between interpreters, so hashing them would make one checkout report
        # two digests depending on what had run there.
        from meltiro.run_log import _hash_tree

        (tmp_path / "a.py").write_text("x = 1\n")
        before = _hash_tree(tmp_path)
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-313.pyc").write_bytes(b"\x00compiled")
        (cache / "a.py").write_text("not source\n")
        assert _hash_tree(tmp_path) == before

    def test_nothing_to_hash_is_none(self, tmp_path):
        # A digest over zero files is a fixed constant that would claim to
        # identify whatever produced it, so the absence is reported instead and
        # `source_hash` turns it into the `nosource` token.
        from meltiro.run_log import _hash_tree

        assert _hash_tree(tmp_path) is None
        assert _hash_tree(tmp_path / "does-not-exist") is None

    def test_unreadable_source_is_the_nosource_token(self, monkeypatch):
        # A frozen or zipimported copy: the package's files cannot be walked.
        # The token keeps the version identifying the release, which is the
        # best available answer there.
        from meltiro.run_log import source_hash

        monkeypatch.setattr("meltiro.run_log._hash_tree", lambda _: None)
        assert source_hash() == "nosource"


# ---------------------------------------------------------------------------
# A resume under a different engine
# ---------------------------------------------------------------------------

class TestResumeUnderADifferentEngine:
    """A session can outlive the tree it started against.

    `meta.engine_fp` is written once, at creation, and `meta.run_fp` is derived
    from it, so both name the engine that STARTED the run. A run that pauses on
    the tool-call cap, is left while the operator checks out another commit or
    upgrades a package, and is then resumed executes its remainder under
    different engine prose. The stage fingerprints cannot see it (no engine
    text enters them) and the creation-time fields cannot either, so the
    per-segment identity has to be recorded where a reader will find it, and
    the divergence has to be said out loud.

    Recorded rather than refused: `engine_fp` moves on every commit, so
    refusing would refuse the documented cap-hit recovery to anyone whose tree
    moved between segments."""

    def _pin(self, monkeypatch, *, commit, direktoro="9.9.9", dirty=False):
        """Pin every engine-identity reading the RESUME path takes."""
        monkeypatch.setattr(
            "meltiro.orchestrator.git_state", lambda: (commit, dirty))
        monkeypatch.setattr(
            "meltiro.run_log.direktoro_version", lambda: direktoro)

    def _resumed_events(self, orch):
        return [e for e in orch.session.read_events()
                if e.get("event") == "resumed"]

    def test_the_segment_engine_identity_is_recorded(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        out = tmp_path / "runs"
        orch = _prepared_orch(config_dir, bundle_minimal_dir, out)
        self._pin(monkeypatch, commit="beefcaf", direktoro="9.9.9")
        resumed = _orch(config_dir, bundle_minimal_dir, out)
        resumed.resume_session(orch.session.session_dir)

        event = self._resumed_events(resumed)[0]
        # Everything the engine axis is made of, per segment: both package
        # versions, the commit, the tree state and the fingerprint over them.
        assert event["meltiro_version"] == __version__
        assert event["direktoro_version"] == "9.9.9"
        assert event["git_commit"] == "beefcaf"
        assert event["engine_fp"].startswith("engine_fp:")
        assert "git_dirty" in event

    def test_the_creation_time_identity_is_left_alone(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # run.json's engine fields are the state at session START, and
        # meta.run_fp is derived from meta.engine_fp. Rewriting either would
        # restate the run's identity halfway through it.
        out = tmp_path / "runs"
        orch = _prepared_orch(config_dir, bundle_minimal_dir, out)
        before = dict(orch.session.meta)
        self._pin(monkeypatch, commit="beefcaf", direktoro="9.9.9")
        resumed = _orch(config_dir, bundle_minimal_dir, out)
        resumed.resume_session(orch.session.session_dir)

        for key in ("meltiro_version", "direktoro_version", "git_commit",
                    "git_dirty", "engine_fp", "run_fp"):
            assert resumed.session.meta[key] == before[key], key

    def test_a_moved_engine_warns_and_names_the_two_fingerprints(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch,
            capsys):
        out = tmp_path / "runs"
        orch = _prepared_orch(config_dir, bundle_minimal_dir, out)
        started_under = orch.session.meta["engine_fp"]
        capsys.readouterr()

        self._pin(monkeypatch, commit="beefcaf", direktoro="9.9.9")
        resumed = _orch(config_dir, bundle_minimal_dir, out)
        resumed.resume_session(orch.session.session_dir)

        warnings = [w for w in resumed.session.meta["warnings"]
                    if w.startswith("engine-drift")]
        assert len(warnings) == 1
        assert started_under in warnings[0]
        assert self._resumed_events(resumed)[0]["engine_fp"] in warnings[0]
        assert "WARNING: engine-drift" in capsys.readouterr().err

    def test_a_resume_onto_the_same_engine_says_nothing(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch,
            capsys):
        # The ordinary case, and the reason the check is a comparison rather
        # than an announcement: a cap-hit resume on an unchanged tree is the
        # documented recovery and must stay quiet.
        out = tmp_path / "runs"
        monkeypatch.setattr(
            "meltiro.session.git_state", lambda: ("abc1234", False))
        monkeypatch.setattr(
            "meltiro.run_log.git_state", lambda: ("abc1234", False))
        monkeypatch.setattr(
            "meltiro.orchestrator.git_state", lambda: ("abc1234", False))
        orch = _prepared_orch(config_dir, bundle_minimal_dir, out)
        capsys.readouterr()

        resumed = _orch(config_dir, bundle_minimal_dir, out)
        resumed.resume_session(orch.session.session_dir)
        assert [w for w in resumed.session.meta["warnings"]
                if w.startswith("engine-drift")] == []
        assert "engine-drift" not in capsys.readouterr().err

    def test_the_resume_is_not_refused(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # The decision, asserted as behaviour: a moved engine is disclosed and
        # the run continues. Refusing would break the cap-hit recovery for
        # anyone whose tree moved between segments.
        out = tmp_path / "runs"
        orch = _prepared_orch(config_dir, bundle_minimal_dir, out)
        self._pin(monkeypatch, commit="beefcaf", direktoro="9.9.9")
        resumed = _orch(config_dir, bundle_minimal_dir, out)
        resumed.resume_session(orch.session.session_dir)
        assert resumed.session.meta["status"] == "in_progress"

    def test_each_segment_records_its_own_identity(
            self, config_dir, bundle_minimal_dir, tmp_path, monkeypatch):
        # Per-segment, not once: a run resumed across two different engines
        # carries both, in order, so the whole run's engine history is
        # readable from the event log. The two segments run under different
        # direktoro versions, which is what moves the engine axis; the commits
        # ride along beside them, naming each segment's checkout.
        out = tmp_path / "runs"
        orch = _prepared_orch(config_dir, bundle_minimal_dir, out)
        session_dir = orch.session.session_dir

        for commit, direktoro in (("beefcaf", "9.9.9"), ("d00dfee", "9.9.10")):
            self._pin(monkeypatch, commit=commit, direktoro=direktoro)
            resumed = _orch(config_dir, bundle_minimal_dir, out)
            resumed.resume_session(session_dir)

        events = self._resumed_events(resumed)
        assert [e["git_commit"] for e in events] == ["beefcaf", "d00dfee"]
        assert [e["direktoro_version"] for e in events] == ["9.9.9", "9.9.10"]
        assert events[0]["engine_fp"] != events[1]["engine_fp"]


# ---------------------------------------------------------------------------
# The paper is input, not instrument
# ---------------------------------------------------------------------------

class TestPaperDoesNotMoveStageFingerprints:
    """Input is not fingerprinted: the same config on another paper is the
    same question asked.

    The papers must differ in the FIGURE SET, not only in text. The figure
    set is what feeds the image-label list, the one per-paper thing besides
    the text that reaches a prompt render, so a test that holds it constant
    leaves the image-label path unexercised."""

    def _bundle(self, src, dst, *, extra_text="", figures=None):
        """Copy a bundle, then set its text, its figure set, and the
        `exhibits` declaration that must match it."""
        shutil.copytree(src, dst)
        text = dst / "text.md"
        text.write_text(
            text.read_text(encoding="utf-8") + extra_text, encoding="utf-8")
        fig_dir = dst / "figures"
        png = sorted(fig_dir.glob("*.png"))[0].read_bytes()
        for existing in fig_dir.glob("*.png"):
            existing.unlink()
        for label in (figures or []):
            (fig_dir / f"{label}.png").write_bytes(png)
        if not figures:
            fig_dir.rmdir()  # a bundle with no figures at all
        # The declaration moves with the figure set, so each paper carries its
        # own labels AND its own captions: both are per-paper prompt content,
        # and neither may move a fingerprint.
        manifest_path = dst / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["exhibits"] = [
            {"label": label, "caption": f"The caption of {label}"}
            for label in (figures or [])
        ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return dst

    def test_papers_differing_in_text_and_figures_share_every_fingerprint(
            self, config_dir, bundle_minimal_dir, tmp_path):
        one_figure = self._bundle(
            bundle_minimal_dir, tmp_path / "b_one",
            extra_text="\n\nA different results section.\n",
            figures=["table_01"])
        many_figures = self._bundle(
            bundle_minimal_dir, tmp_path / "b_many",
            extra_text="\n\nAnother, longer results section entirely.\n",
            figures=["table_01", "table_02", "figure_09"])
        no_figures = self._bundle(
            bundle_minimal_dir, tmp_path / "b_none",
            extra_text="\n\nA third paper, text only.\n", figures=[])

        # Sanity: the bundles really do present different image sets, so the
        # image-label render this test is meant to exercise is exercised.
        assert len(load_bundle(many_figures).figures) == 3
        assert load_bundle(no_figures).figures == {}

        fps = [
            _fps(_prepared_orch(config_dir, b, tmp_path / f"out_{i}"))
            for i, b in enumerate([one_figure, many_figures, no_figures])
        ]
        assert fps[0] == fps[1] == fps[2]


# ---------------------------------------------------------------------------
# Consumer-pinned content hashes
# ---------------------------------------------------------------------------

class TestConsumerPinnedHashesAreUnaffected:
    def test_engine_prose_edit_moves_no_content_hash(
            self, config_dir, monkeypatch):
        """template_hash and reference_lists_hash are config-bundle content.
        A consumer pins them to decide whether a stored value is still legal,
        so engine prose must never move them, and neither must the config
        bundle's own instrument fingerprint."""
        cb_a = load_config_bundle(config_dir)

        monkeypatch.setattr(
            prompt_builder, "EXTRACTOR_TOOL_REPROMPT", "Call a tool.")
        monkeypatch.setattr(
            prompt_builder, "REVIEW_TOOL_REPROMPT", "Call a tool.")

        cb_b = load_config_bundle(config_dir)

        assert cb_a.template_hash == cb_b.template_hash
        assert cb_a.reference_lists_hash == cb_b.reference_lists_hash
        assert cb_a.instrument_fp == cb_b.instrument_fp
