"""A resume must be handed the paper the session started with.

The three config axes a resume is refused on — config_fp, checker_fp, review_fp
— fold in no part of the paper, by design: the config axes say what was asked
and the bundle axes say what it was asked of (see
`fingerprint.bundle_fingerprint`). So an edited `text.md`, a re-cropped figure
or a rewritten manifest moves none of them, and without a gate of its own a
resume would replay a conversation whose earlier turns quote the OLD paper into
a new one, verifying fresh evidence against different text and shipping one
extraction citing two sources.

`Session.resume` therefore compares the paper's own axes too, through the same
`bundle_fingerprint` recipe that recorded them at session start, and names
whichever moved.

Offline: a real Session backs a real Orchestrator, the extractor loop is
stubbed, and no provider is reached.
"""

import shutil

import pytest

from meltiro.bundle import load_bundle
from meltiro.checker import CheckerConfig
from meltiro.config_bundle import load_config_bundle
from meltiro.errors import ResumeRefused
from meltiro.orchestrator import Orchestrator
from meltiro.session import Session

pytestmark = pytest.mark.usefixtures("stage_keys")

EXTRACTOR = "claude-opus-4-8"


def _orch(config_dir, bundle_dir, out_dir):
    """An extractor-only Orchestrator (checker and reviewer off)."""
    return Orchestrator(
        load_config_bundle(config_dir), load_bundle(bundle_dir), out_dir,
        extractor_model=EXTRACTOR,
        checker_config=CheckerConfig(
            max_tokens=1024, checker_model="claude-sonnet-4-6"),
        review_model=None,
        max_checks_per_field=0, final_review=False,
        max_tool_calls=50,
        extractor_max_tokens=4096,
    )


@pytest.fixture
def paper(tmp_path, bundle_minimal_dir):
    """A writable copy of the paper bundle, so a test can edit it mid-run."""
    dst = tmp_path / "paper"
    shutil.copytree(bundle_minimal_dir, dst)
    return dst


def _paused(config_dir, paper, out):
    """A session paused mid-extraction, ready to be resumed."""
    orch = _orch(config_dir, paper, out)
    orch.prepare_new_session()
    return orch.session.session_dir


def test_an_unchanged_paper_resumes(config_dir, paper, tmp_path):
    out = tmp_path / "runs"
    session_dir = _paused(config_dir, paper, out)

    orch = _orch(config_dir, paper, out)
    orch.resume_session(session_dir)  # must not raise
    assert orch.session.meta["status"] == "in_progress"


def test_an_edited_text_refuses_and_names_the_axis(
        config_dir, paper, tmp_path):
    out = tmp_path / "runs"
    session_dir = _paused(config_dir, paper, out)

    text = paper / "text.md"
    text.write_text(text.read_text(encoding="utf-8") +
                    "\n\nAn appendix nobody had read.\n", encoding="utf-8")

    orch = _orch(config_dir, paper, out)
    with pytest.raises(ResumeRefused) as caught:
        orch.resume_session(session_dir)
    message = str(caught.value)
    assert "the paper bundle changed" in message
    assert "text_fp" in message
    # The axes that did NOT move are not named, so the message points at one
    # file rather than at the bundle in general.
    assert "figures_fp" not in message
    assert "manifest_fp" not in message


def test_an_edited_manifest_refuses_and_names_its_own_axis(
        config_dir, paper, tmp_path):
    out = tmp_path / "runs"
    session_dir = _paused(config_dir, paper, out)

    import json
    manifest_path = paper / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["title"] = "A different title entirely"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    orch = _orch(config_dir, paper, out)
    with pytest.raises(ResumeRefused) as caught:
        orch.resume_session(session_dir)
    message = str(caught.value)
    assert "the paper bundle changed" in message
    assert "manifest_fp" in message
    assert "text_fp" not in message


def test_the_paper_gate_is_not_the_config_gate(config_dir, paper, tmp_path):
    """The point of the gate: an edited paper moves NO config axis, so nothing
    the drift gate compares could have caught this."""
    out = tmp_path / "runs"
    orch1 = _orch(config_dir, paper, out)
    orch1.prepare_new_session()
    before = orch1._build_fingerprints()

    text = paper / "text.md"
    text.write_text(text.read_text(encoding="utf-8") + "\nmore text\n",
                    encoding="utf-8")

    after = _orch(config_dir, paper, out)._build_fingerprints()
    assert before == after


class TestAnAxisTheSessionNeverRecorded:
    """A session started before an axis existed cannot answer for it, and the
    refusal says so.

    This is what an upgrade actually looks like: the axes a run records grow,
    and a session paused under the previous engine has no value for the new
    ones. Read as a mismatch, that reports a paper that changed and prescribes
    a fix that cannot work — no bundle hashes to a missing value, so the
    operator re-points `--paper` at the original directory and gets the same
    refusal, with nothing naming the real boundary.
    """

    def _paused_without(self, config_dir, paper, out, axes):
        import json

        session_dir = _paused(config_dir, paper, out)
        meta_path = Session.meta_path_for(session_dir)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for axis in axes:
            meta.pop(axis, None)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return session_dir

    @pytest.mark.parametrize("axes", [
        ["tables_fp"], ["supplements_fp"], ["tables_fp", "supplements_fp"]])
    def test_it_is_named_as_unsettled_rather_than_moved(
            self, config_dir, paper, tmp_path, axes):
        out = tmp_path / "runs"
        session_dir = self._paused_without(config_dir, paper, out, axes)

        orch = _orch(config_dir, paper, out)
        with pytest.raises(ResumeRefused) as caught:
            orch.resume_session(session_dir)
        message = str(caught.value)
        # The paper did not change, and the message does not say it did.
        assert "the paper bundle changed" not in message
        assert "records no" in message
        for axis in axes:
            assert axis in message
        # Nor does it prescribe the fix that cannot work.
        assert "Point --paper at the original bundle" not in message

    def test_a_paper_that_really_moved_is_still_reported_as_moved(
            self, config_dir, paper, tmp_path):
        # The control: an absent axis does not mask a real change to an axis
        # the session DID record.
        out = tmp_path / "runs"
        session_dir = self._paused_without(
            config_dir, paper, out, ["supplements_fp"])
        text = paper / "text.md"
        text.write_text(text.read_text(encoding="utf-8") + "\n\nAdded.\n",
                        encoding="utf-8")

        orch = _orch(config_dir, paper, out)
        with pytest.raises(ResumeRefused) as caught:
            orch.resume_session(session_dir)
        message = str(caught.value)
        assert "the paper bundle changed" in message
        assert "text_fp" in message


class TestTheTwoNewestAxesAreCompared:
    """A re-transcribed cell and an edited supplement each refuse a resume.

    Both were asserted only as membership of the `BUNDLE_AXES` tuple — the
    constant, not the comparison — so excluding them from the comparison
    itself passed the whole suite. Both are shown to the model, which is why
    they are axes at all: the conversation being replayed was read against
    them.
    """

    @pytest.fixture
    def supplemented_paper(self, tmp_path, bundle_supplemented_dir):
        dst = tmp_path / "supplemented"
        shutil.copytree(bundle_supplemented_dir, dst)
        return dst

    def test_a_re_transcribed_cell_refuses(
            self, config_dir, supplemented_paper, tmp_path):
        out = tmp_path / "runs"
        session_dir = _paused(config_dir, supplemented_paper, out)

        path = supplemented_paper / "tables" / "table_01.html"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "<td>5.8 (4.1-7.6)</td>", "<td>5.9 (4.1-7.6)</td>"),
            encoding="utf-8")

        orch = _orch(config_dir, supplemented_paper, out)
        with pytest.raises(ResumeRefused) as caught:
            orch.resume_session(session_dir)
        message = str(caught.value)
        assert "the paper bundle changed" in message
        assert "tables_fp" in message
        # And it names that axis alone: the article's text and crops did not
        # move, so a reader is pointed at one file.
        assert "text_fp" not in message
        assert "figures_fp" not in message

    def test_an_edited_supplement_refuses(
            self, config_dir, supplemented_paper, tmp_path):
        import json

        out = tmp_path / "runs"
        session_dir = _paused(config_dir, supplemented_paper, out)

        path = supplemented_paper / "supplements.json"
        declared = json.loads(path.read_text(encoding="utf-8"))
        declared["supplements"][0]["exhibits"][0]["caption"] = (
            "Table S1. MEAN turnaround time by shift")
        path.write_text(json.dumps(declared), encoding="utf-8")

        orch = _orch(config_dir, supplemented_paper, out)
        with pytest.raises(ResumeRefused) as caught:
            orch.resume_session(session_dir)
        message = str(caught.value)
        assert "the paper bundle changed" in message
        assert "supplements_fp" in message
        assert "text_fp" not in message

    def test_an_untouched_supplemented_bundle_resumes(
            self, config_dir, supplemented_paper, tmp_path):
        out = tmp_path / "runs"
        session_dir = _paused(config_dir, supplemented_paper, out)
        orch = _orch(config_dir, supplemented_paper, out)
        orch.resume_session(session_dir)  # must not raise
        assert orch.session.meta["status"] == "in_progress"
