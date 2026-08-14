"""The SDK floors, checked against the SDKs and against direktoro.

meltiro builds no provider client: `direktoro.build_adapter` resolves the
endpoint, the key variable and the SDK from a model id, and direktoro is the
only package here that imports `anthropic` or `openai`. So the SDK floors are
direktoro's to declare, and meltiro's job is to declare neither — a copy of a
number it does not own could only go stale, and a stale copy of a floor is
worse than none, because it describes an install pip can never produce.

What can still go wrong is a packaging fault no amount of engine testing
catches, because the installed SDK in a dev tree is always new enough: an SDK
below the floor resolves, imports, and then fails at CALL time, on a run that
has already started spending. So this pins the floors four ways, none of them a
hand-copied number:

  1. the SDK actually installed here accepts the thinking parameters direktoro
     puts on the wire, inspected off `anthropic`'s real signature, and exposes
     the `responses` namespace direktoro's OpenAI adapter calls;
  2. the installed SDKs satisfy direktoro's declared floors;
  3. direktoro declares both floors at all;
  4. meltiro declares neither SDK, so there is nothing here to drift.

Where direktoro's declaration is read from depends on the install (see
`_direktoro_requirements`). A source checkout is read from its real
`pyproject.toml`, because an editable install's metadata is a snapshot from
install time and goes stale the moment a floor moves. A wheel install has no
pyproject on disk, and is exactly the shape CI and a stranger's `pip install`
have, so its declaration is read from the distribution metadata — which is
what pip resolved against, and therefore the right number to check there.

direktoro carries a floor of its own here, checked the same way against the
version actually resolved. It is the layer that splits a decoding block,
resolves what reaches the wire, and supplies the shapes every fingerprint
folds in, so an install below the floor does not merely compute different
numbers — it raises on the call meltiro makes.

alteksto carries one for the same reason on the input side: it owns the paper
bundle format, and the version it validates against is the version a bundle
has to be built to. Beside it sits the other half of that ownership — that
this package restates no rule of the format anywhere, so there is no second
copy to drift from the one that decides.

Offline: signature inspection and file reads only, no client, no network.
"""

import inspect
import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# The two keyword parameters direktoro's Anthropic adapter puts on the wire for
# a thinking spec. Named here because they are what the floor EXISTS for; the
# versions they arrived in are recorded in direktoro's pyproject.toml beside
# the floor.
SEAM_PARAMS = ("thinking", "output_config")

# The SDKs direktoro imports and meltiro does not.
PROVIDER_SDKS = ("anthropic", "openai")


def _requirement(specs, package):
    """`package`'s requirement string among `specs`, or None when it is not a
    dependency at all — the two answers a floor test needs to tell apart.

    A spec carrying an `extra ==` marker is skipped: an extra's dependency is
    not installed by a plain `pip install`, so it pins nothing about the
    runtime under test.
    """
    for spec in specs:
        requirement, _, marker = spec.partition(";")
        if "extra ==" in marker:
            continue
        requirement = requirement.strip()
        name = re.split(r"[<>=!~\[ ]", requirement, maxsplit=1)[0]
        if name == package:
            return requirement
    return None


def _declared_floor(specs, package, source):
    """The lower bound `specs` declares for `package`, as a string."""
    requirement = _requirement(specs, package)
    if requirement is None:
        raise AssertionError(f"{package} not declared in {source}")
    if ">=" not in requirement:
        raise AssertionError(
            f"{package} is declared in {source} with no lower bound "
            f"({requirement!r})")
    return requirement.split(">=")[1].split(",")[0].strip()


def _version_tuple(text):
    return tuple(int(part) for part in text.split(".")[:3])


def _pyproject_requirements(pyproject_path):
    """The `project.dependencies` list a pyproject.toml declares."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def _meltiro_requirements():
    return _pyproject_requirements(REPO_ROOT / "pyproject.toml")


def _direktoro_requirements():
    """direktoro's declared dependencies, and a name for where they came from.

    A source checkout is read from its real `pyproject.toml`; a wheel install,
    which has none on disk, is read from the distribution metadata pip
    resolved against. Both are direktoro's own declaration, so the floors are
    checked under either install shape, and the caller skips only when neither
    source exists at all.
    """
    import direktoro

    source_root = Path(direktoro.__file__).resolve().parents[2]
    pyproject = source_root / "pyproject.toml"
    if pyproject.is_file():
        return _pyproject_requirements(pyproject), str(pyproject)
    try:
        requires = metadata.requires("direktoro")
    except metadata.PackageNotFoundError:
        requires = None
    if requires is None:
        pytest.skip(
            "direktoro declares no dependencies this build can read: its "
            "pyproject.toml is not on disk and its installed distribution "
            "metadata names no requirements, so there is no declaration to "
            "check the installed SDKs against")
    return requires, "direktoro's installed distribution metadata"


def test_installed_sdk_accepts_the_parameters_the_seam_sends():
    # The whole reason for the anthropic floor. If this goes red, a bundle that
    # names a thinking key would TypeError on its first call rather than at
    # install.
    import anthropic

    client = anthropic.Anthropic(api_key="not-a-real-key")  # no call is made
    params = inspect.signature(client.messages.stream).parameters
    for name in SEAM_PARAMS:
        assert name in params, (
            f"anthropic {anthropic.__version__} has no `{name}` parameter on "
            f"messages.stream; raise the floor in direktoro's pyproject.toml")


def test_installed_sdk_satisfies_direktoros_declared_floor():
    import anthropic

    specs, source = _direktoro_requirements()
    floor = _declared_floor(specs, "anthropic", source)
    assert _version_tuple(anthropic.__version__) >= _version_tuple(floor), (
        f"anthropic {anthropic.__version__} is below the floor {floor} "
        f"declared in {source}")


def test_installed_openai_sdk_has_the_namespace_the_adapter_calls():
    # direktoro's OpenAI adapter calls `client.responses.create`. The
    # `responses` namespace is what the floor exists for: an older SDK installs
    # and imports cleanly and then fails with AttributeError at the first
    # OpenAI-family call, once the run is already under way.
    import openai

    client = openai.OpenAI(api_key="not-a-real-key")  # no call is made
    assert hasattr(client, "responses"), (
        f"openai {openai.__version__} has no `responses` namespace; direktoro's "
        f"adapter calls client.responses.create")
    assert hasattr(client.responses, "create")


def test_installed_openai_sdk_satisfies_direktoros_declared_floor():
    import openai

    specs, source = _direktoro_requirements()
    floor = _declared_floor(specs, "openai", source)
    assert _version_tuple(openai.__version__) >= _version_tuple(floor), (
        f"openai {openai.__version__} is below the floor {floor} declared in "
        f"{source}")


def test_direktoro_declares_both_sdk_floors():
    # The floors have to exist somewhere for an install to be reproducible, and
    # direktoro is where: it is the package that imports and calls both SDKs.
    specs, source = _direktoro_requirements()
    for package in PROVIDER_SDKS:
        assert _requirement(specs, package) is not None, (
            f"direktoro declares no floor for {package} in {source}, which it "
            f"imports and calls; meltiro declares none either, so nothing "
            f"would pin it")


def test_meltiro_declares_neither_sdk():
    # meltiro constructs no provider client, so a declaration here would pin a
    # floor for calls it does not make — a number that goes stale the moment
    # direktoro's own moves, and one pip would then have to reconcile against a
    # requirement nothing in this package can justify.
    #
    # Read from this repo's pyproject.toml, which is always on disk when the
    # suite runs: it is the file under test, and the one a release is built
    # from.
    for package in PROVIDER_SDKS:
        assert _requirement(_meltiro_requirements(), package) is None, (
            f"meltiro declares a floor for {package}, but builds no client "
            f"with it; direktoro owns that floor")


def test_installed_alteksto_satisfies_the_declared_floor():
    # The package that owns the paper bundle format. Its floor is a FORMAT
    # floor: `alteksto.bundle.SCHEMA_VERSION` is the version a bundle must
    # declare, so an install below the floor rejects every bundle built to the
    # current specification, before any spend and with a message about a
    # schema version rather than about a stale install.
    import alteksto

    floor = _declared_floor(
        _meltiro_requirements(), "alteksto", "meltiro's pyproject.toml")
    assert _version_tuple(alteksto.__version__) >= _version_tuple(floor), (
        f"alteksto {alteksto.__version__} is below meltiro's declared floor "
        f"of {floor}; the bundle format it validates is not the one this "
        f"release consumes")


def test_meltiro_states_no_rule_of_the_bundle_format_itself():
    # The format is declared, not copied. A schema version, a manifest key set
    # or a label pattern restated in this package would be a second
    # implementation of someone else's specification: it can only drift, and
    # the drift would show up as a bundle one package accepts and the other
    # refuses.
    src_dir = REPO_ROOT / "src" / "meltiro"
    offenders = [path.name for path in sorted(src_dir.glob("*.py"))
                 if "SCHEMA_VERSION" in path.read_text(encoding="utf-8")]
    assert offenders == [], (
        f"{offenders} name a bundle schema version; alteksto declares it")


def test_installed_direktoro_satisfies_the_declared_floor():
    # The same read as the SDK floors, against the package that owns
    # `build_adapter`, `split_decoding_config`, `resolved_decoding_params` and
    # `call_identity_fields`. `direktoro.__version__` is the distribution's
    # own version attribute and the figure a run records as
    # `direktoro_version`, so what is checked here is what a run would report.
    import direktoro

    floor = _declared_floor(
        _meltiro_requirements(), "direktoro", "meltiro's pyproject.toml")
    assert _version_tuple(direktoro.__version__) >= _version_tuple(floor), (
        f"direktoro {direktoro.__version__} is below meltiro's declared "
        f"floor of {floor}; the decoding seam meltiro calls is not there")
