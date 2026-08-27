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
bundle format, and `load_bundle` calls into it both for the verdict on a
bundle and for the enumeration of what the bundle's `figures/` holds. Beside
it sits the other half of that ownership — that the loader restates no rule of
the format itself, so there is no second copy to drift from the one that
decides.

Offline: signature inspection and file reads only, no client, no network.
"""

import inspect
import json
import re
import tomllib
from importlib import metadata
from pathlib import Path

import importlib

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


SISTER_PINS = REPO_ROOT / "requirements" / "sisters.txt"


def _pinned_sisters():
    """`{name: version}` from the pin file, which names an exact tag each.

    The file is the development environment's answer to "which release", where
    pyproject's floor is the consumer's answer to "at least which release".
    """
    assert SISTER_PINS.is_file(), (
        f"{SISTER_PINS} is missing, so what this tree was tested against "
        f"cannot be read")
    pins = {}
    for line in SISTER_PINS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, ref = line.partition("@")
        pins[name.strip()] = ref.rsplit("@", 1)[1].strip().lstrip("v")
    return pins


def _installed_version(package):
    """What pip actually resolved, from the distribution's own metadata.

    NOT the module's `__version__`. An editable install keeps the metadata it
    was built with, so a checkout that moves under one reports a version it no
    longer holds — this machine held `alteksto` dist-info at 0.2.0 while the
    module said 0.5.0, below meltiro's own floor, with every check passing.
    """
    import importlib.metadata as metadata

    return metadata.version(package)


@pytest.mark.parametrize("package", ["alteksto", "direktoro"])
def test_the_installed_sister_is_the_pinned_release(package):
    # What a green suite was green against. A sister resolved from a local
    # checkout or from a default branch's tip is not this release, and the
    # difference is not cosmetic: alteksto validates a manifest against
    # exactly its own schema version, so its tip moving refuses every fixture
    # here.
    installed = _installed_version(package)
    pinned = _pinned_sisters()[package]
    assert installed == pinned, (
        f"{package} {installed} is installed but {SISTER_PINS.name} pins "
        f"{pinned}. Install the pin — `pip install -r "
        f"requirements/sisters.txt` — or move the pin deliberately, with the "
        f"fixtures it affects in the same commit.")


@pytest.mark.parametrize("package", ["alteksto", "direktoro"])
def test_a_sisters_metadata_agrees_with_its_module(package):
    # The two disagree exactly when a checkout has moved under an editable
    # install, which is the state that hides everything else here.
    module = importlib.import_module(package)
    assert module.__version__ == _installed_version(package), (
        f"{package} reports __version__ {module.__version__} while its "
        f"installed distribution says {_installed_version(package)}: the "
        f"install is a checkout that has moved since it was made. Reinstall "
        f"it so what is imported and what is recorded are one version.")


def test_installed_alteksto_satisfies_the_declared_floor():
    # The package that owns the paper bundle format. Its floor buys two
    # things: `figure_files`, which `load_bundle` calls to enumerate the
    # crops, and the packaging split that lets this dependency weigh what the
    # format contract weighs. Below the floor the call is missing outright,
    # which is why it is checked against what pip RESOLVED — the
    # distribution's metadata — rather than against the module attribute,
    # which an install that has drifted still reports happily.
    from alteksto.bundle import figure_files, validate_bundle  # noqa: F401

    floor = _declared_floor(
        _meltiro_requirements(), "alteksto", "meltiro's pyproject.toml")
    installed = _installed_version("alteksto")
    assert _version_tuple(installed) >= _version_tuple(floor), (
        f"alteksto {installed} is below meltiro's declared floor "
        f"of {floor}; the enumeration `load_bundle` calls is not there")


def test_the_declared_floor_admits_the_pin():
    # The two answers have to be compatible: a floor above the pin would
    # declare a release this tree has never run against.
    floor = _declared_floor(
        _meltiro_requirements(), "alteksto", "meltiro's pyproject.toml")
    assert _version_tuple(_pinned_sisters()["alteksto"]) >= \
        _version_tuple(floor), (
        f"requirements/sisters.txt pins alteksto below meltiro's own floor "
        f"of {floor}")


def test_the_fixture_bundles_declare_the_resolved_formats_version():
    # The floor above is satisfied by ANY alteksto at or past it, and the
    # format moves faster than the floor: alteksto validates a manifest
    # against the one schema version its release declares, so a fixture built
    # to a superseded version stops being a bundle the moment the format
    # moves, while the floor still resolves and still passes its own check.
    #
    # That failure is not self-describing. It arrives as a manifest error on
    # every fixture at once, in whatever test happened to load one, saying a
    # number is wrong and nothing about why it moved or what to do. So it is
    # asserted here instead, once, against the version actually resolved: the
    # fixtures either declare the format this tree reads, or this says so and
    # names the bump.
    from alteksto.bundle import SCHEMA_VERSION

    fixtures = sorted((REPO_ROOT / "tests" / "fixtures").glob(
        "*/manifest.json"))
    assert fixtures, "no fixture bundles found to check"
    declared = {
        path.parent.name: json.loads(path.read_text(
            encoding="utf-8")).get("schema_version")
        for path in fixtures
    }
    stale = {name: v for name, v in declared.items() if v != SCHEMA_VERSION}
    assert not stale, (
        f"fixture bundles {stale} declare a schema version the resolved "
        f"alteksto ({SCHEMA_VERSION}) does not accept; the format has moved, "
        f"so raise the alteksto floor in pyproject.toml and rebuild these "
        f"fixtures to what the new version specifies")


def test_the_loader_states_no_rule_of_the_bundle_format():
    # The format is declared, not copied. A schema version, a manifest key
    # set, a label pattern or a rule about which files under `figures/` are
    # exhibits, restated in this package, would be a second implementation of
    # someone else's specification: it can only drift, and the drift would
    # show up as a bundle one package accepts and the other reads differently.
    #
    # A cheap guard over the source, and the narrow one: it sees the loader
    # itself, so a rule restated in a helper beside it would pass here. What
    # catches that is behavioural and lives with the loader's own tests
    # (`test_bundle.py::test_the_loader_reads_the_directory_the_format_reads`,
    # over a directory built to make two readings disagree).
    src_dir = REPO_ROOT / "src" / "meltiro"
    versioned = [path.name for path in sorted(src_dir.glob("*.py"))
                 if "SCHEMA_VERSION" in path.read_text(encoding="utf-8")]
    assert versioned == [], (
        f"{versioned} name a bundle schema version; alteksto declares it")

    from meltiro.bundle import load_bundle

    loader = inspect.getsource(load_bundle)
    for rule in (".png", "startswith(\".\")", "is_dir()"):
        assert rule not in loader, (
            f"load_bundle names {rule!r}, which is the format's rule about "
            f"what a crop is; alteksto.bundle.figure_files answers that")


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
