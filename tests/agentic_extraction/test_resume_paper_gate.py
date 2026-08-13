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
