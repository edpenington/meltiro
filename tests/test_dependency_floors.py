"""The SDK floors, checked against the SDKs and against direktoro.

meltiro hands direktoro a `Thinking` spec, direktoro emits it as `thinking=`
and `output_config=` on `anthropic`'s `messages.stream(...)`, and an SDK
without those keyword parameters raises TypeError the first time a bundle names
a thinking key. That is a packaging fault which no amount of engine testing
catches, because the installed SDK in a dev tree is always new enough.

So the floor is pinned three ways, none of them a hand-copied number:

  1. the SDK actually installed here accepts both parameters, inspected off
     `anthropic`'s real signature;
  2. the installed SDK satisfies meltiro's own declared floor;
  3. meltiro's declared floor is at least direktoro's, read from direktoro's
     real `pyproject.toml` rather than from its installed metadata (an editable
     install's metadata is a snapshot from install time and goes stale the
     moment the dependency's floor moves).

Offline: signature inspection and file reads only, no client, no network.
"""

import inspect
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# The two keyword parameters direktoro's Anthropic adapter puts on the wire for
# a thinking spec. Named here because they are what the floor EXISTS for; the
# versions they arrived in are recorded in pyproject.toml beside the floor.
SEAM_PARAMS = ("thinking", "output_config")


def _declared_floor(pyproject_path, package):
    """The lower bound `pyproject_path` declares for `package`, as a string."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    for spec in data["project"]["dependencies"]:
        name = spec.split(">")[0].split("=")[0].split("[")[0].strip()
        if name == package:
            return spec.split(">=")[1].strip()
    raise AssertionError(f"{package} not declared in {pyproject_path}")


def _version_tuple(text):
    return tuple(int(part) for part in text.split(".")[:3])


def test_installed_sdk_accepts_the_parameters_the_seam_sends():
    # The whole reason for the floor. If this goes red, a bundle that names a
    # thinking key would TypeError on its first call rather than at install.
    import anthropic

    client = anthropic.Anthropic(api_key="not-a-real-key")  # no call is made
    params = inspect.signature(client.messages.stream).parameters
    for name in SEAM_PARAMS:
        assert name in params, (
            f"anthropic {anthropic.__version__} has no `{name}` parameter on "
            f"messages.stream; raise the floor in pyproject.toml")


def test_installed_sdk_satisfies_our_own_declared_floor():
    import anthropic

    floor = _declared_floor(REPO_ROOT / "pyproject.toml", "anthropic")
    assert _version_tuple(anthropic.__version__) >= _version_tuple(floor)


def test_floor_is_at_least_direktoros():
    # direktoro is the package that actually passes these parameters, so its
    # floor is the binding one; declaring a lower one here would describe an
    # install pip can never produce. Read from direktoro's real pyproject, NOT
    # from `importlib.metadata.requires("direktoro")`: an editable install's
    # metadata is frozen at install time, so it reports whatever floor
    # direktoro declared back then and this test would pass while asserting
    # nothing.
    import direktoro

    source_root = Path(direktoro.__file__).resolve().parents[2]
    pyproject = source_root / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip(
            "direktoro is installed as a built wheel, so its pyproject.toml is "
            "not on disk; the floor cannot be read from the counterpart repo "
            "here")
    theirs = _declared_floor(pyproject, "anthropic")
    ours = _declared_floor(REPO_ROOT / "pyproject.toml", "anthropic")
    assert _version_tuple(ours) >= _version_tuple(theirs), (
        f"meltiro declares anthropic>={ours} but direktoro, which sends the "
        f"thinking parameters, declares anthropic>={theirs}")


def test_installed_openai_sdk_has_the_namespace_the_adapter_calls():
    # meltiro constructs `openai.OpenAI(...)` and hands the client to
    # direktoro, whose adapter calls `client.responses.create`. The `responses`
    # namespace is what the floor exists for: an older SDK installs and imports
    # cleanly and then fails with AttributeError at the first OpenAI call, once
    # the run is already under way.
    import openai

    client = openai.OpenAI(api_key="not-a-real-key")  # no call is made
    assert hasattr(client, "responses"), (
        f"openai {openai.__version__} has no `responses` namespace; direktoro's "
        f"adapter calls client.responses.create")
    assert hasattr(client.responses, "create")


def test_installed_openai_sdk_satisfies_the_declared_openai_floor():
    import openai

    floor = _declared_floor(REPO_ROOT / "pyproject.toml", "openai")
    assert _version_tuple(openai.__version__) >= _version_tuple(floor)


def test_the_openai_floor_is_at_least_direktoros():
    # Same reasoning as the anthropic floor: direktoro makes the call, so its
    # floor is the binding one and meltiro must not declare a lower bound that
    # describes an install pip can never produce.
    import direktoro

    source_root = Path(direktoro.__file__).resolve().parents[2]
    pyproject = source_root / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip(
            "direktoro is installed as a built wheel, so its pyproject.toml is "
            "not on disk; the floor cannot be read from the counterpart repo "
            "here")
    theirs = _declared_floor(pyproject, "openai")
    ours = _declared_floor(REPO_ROOT / "pyproject.toml", "openai")
    assert _version_tuple(ours) >= _version_tuple(theirs), (
        f"meltiro declares openai>={ours} but direktoro, which calls "
        f"responses.create, declares openai>={theirs}")
