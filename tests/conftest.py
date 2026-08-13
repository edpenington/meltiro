"""Shared test configuration.

The pipeline is generic; everything review-specific lives in a config bundle.
Tests point at an explicit bundle under `tests/fixtures/` rather than at any
CWD-relative default, because there are no CWD-relative defaults: the prompt
builders require paths. These constants and fixtures supply them.

This module also installs a network guard (see `_block_network` below) so
the suite's zero-network property is enforced mechanically, not just left
as a consequence of lazy client construction.
"""

import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# The config bundle the engine is exercised against: a wholly invented review
# (made-up gauges, made-up outcomes, no real paper) built to cover every
# template feature the engine supports. It is a fixture, not a worked example,
# which is why it lives here rather than anywhere a reader might mistake it
# for guidance.
CONFIG_DIR = Path(__file__).resolve().parent / "fixtures" / "config_synthetic"


# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------
# The suite must never touch the network or need an API key: every LLM call
# is stubbed. This guard makes that property fail loudly instead of relying
# on it as an accident of the code path. We patch the connect primitives on
# `socket.socket` (which `socket.create_connection`, httpx, and the Anthropic
# SDK all funnel through), the name-resolution functions (`socket.getaddrinfo`,
# `socket.gethostbyname`, `socket.gethostbyname_ex`), and the connectionless
# UDP `socket.socket.sendto` path, so every route to a non-local destination
# raises. Loopback and AF_UNIX addresses are allowed so anything genuinely
# local still works, though nothing in the suite needs even that today.
#
# A hand-rolled guard is preferred over the pytest-socket dependency: it is a
# dozen lines, keeps the test dependencies at zero third-party packages, and
# raises a message that points straight at this file. pytest-socket would add
# an install for no capability this does not already cover.

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto
_real_getaddrinfo = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname
_real_gethostbyname_ex = socket.gethostbyname_ex


def _guarded_resolver(resolver):
    # Block name resolution of non-local hosts: without this, a test reaching
    # for a real hostname performs a real DNS lookup before the connect guard
    # fires, leaking the hostname to the resolver. getaddrinfo, gethostbyname,
    # and gethostbyname_ex all take the host as their first argument.
    def _guard(host, *args, **kwargs):
        if host is None or (isinstance(host, (str, bytes)) and (
                host.decode() if isinstance(host, bytes) else host)
                in _ALLOWED_HOSTS):
            return resolver(host, *args, **kwargs)
        raise RuntimeError(
            "network access is disabled during tests (attempted DNS "
            f"resolution of {host!r}). The suite must never touch the network "
            "or need an API key: stub the client instead. See the network "
            "guard in tests/conftest.py."
        )
    return _guard


def _is_local_address(address):
    # AF_UNIX addresses are str/bytes filesystem paths: always local.
    if isinstance(address, (str, bytes)):
        return True
    # AF_INET / AF_INET6 addresses are (host, port[, ...]) tuples.
    if isinstance(address, tuple) and address:
        return address[0] in _ALLOWED_HOSTS
    return False


def _blocked(operation):
    def _guard(self, address, *args, **kwargs):
        if _is_local_address(address):
            return operation(self, address, *args, **kwargs)
        raise RuntimeError(
            "network access is disabled during tests (attempted connection "
            f"to {address!r}). The suite must never touch the network or need "
            "an API key: stub the client instead. See the network guard in "
            "tests/conftest.py."
        )
    return _guard


def _blocked_sendto(operation):
    # UDP is connectionless: sendto carries the destination directly and never
    # touches connect, so it would bypass the connect guard. sendto(data,
    # address) and sendto(data, flags, address) both put the address last
    # positionally.
    def _guard(self, *args, **kwargs):
        address = args[-1] if args else None
        if address is None or _is_local_address(address):
            return operation(self, *args, **kwargs)
        raise RuntimeError(
            "network access is disabled during tests (attempted datagram send "
            f"to {address!r}). The suite must never touch the network or need "
            "an API key: stub the client instead. See the network guard in "
            "tests/conftest.py."
        )
    return _guard


@pytest.fixture(scope="session", autouse=True)
def _block_network():
    """Fail any test that opens a non-local network connection.

    Session-scoped and autouse so it wraps every test without opt-in. The
    patched primitives are restored afterwards so the guard does not leak
    out of the pytest process.
    """
    socket.socket.connect = _blocked(_real_connect)
    socket.socket.connect_ex = _blocked(_real_connect_ex)
    socket.socket.sendto = _blocked_sendto(_real_sendto)
    socket.getaddrinfo = _guarded_resolver(_real_getaddrinfo)
    socket.gethostbyname = _guarded_resolver(_real_gethostbyname)
    socket.gethostbyname_ex = _guarded_resolver(_real_gethostbyname_ex)
    try:
        yield
    finally:
        socket.socket.connect = _real_connect
        socket.socket.connect_ex = _real_connect_ex
        socket.socket.sendto = _real_sendto
        socket.getaddrinfo = _real_getaddrinfo
        socket.gethostbyname = _real_gethostbyname
        socket.gethostbyname_ex = _real_gethostbyname_ex


# ---------------------------------------------------------------------------
# dotenv guard
# ---------------------------------------------------------------------------
# cli.main() calls load_dotenv(), which reads a developer's real .env into the
# process environment. The suite must never ingest real API keys, so the loader
# is neutralised for every test regardless of what the developer has on disk.
# Patching both the dotenv source and the name cli already imported at module
# load covers either call site.

@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    def _noop(*args, **kwargs):
        return False
    monkeypatch.setattr("meltiro.cli.load_dotenv", _noop)
    monkeypatch.setattr("dotenv.load_dotenv", _noop)


EXTRACTOR_SYSTEM_PATH = CONFIG_DIR / "prompts" / "extractor_system.md"
REVIEW_SYSTEM_PATH = CONFIG_DIR / "prompts" / "review_system.md"
CHECKER_SYSTEM_PATH = CONFIG_DIR / "prompts" / "checker_system.md"
CHECKER_PARTIALS_DIR = CONFIG_DIR / "prompts" / "partials"
TEMPLATE_PATH = CONFIG_DIR / "extraction_template.yaml"
REFERENCE_DIR = CONFIG_DIR / "reference"

# Synthetic paper bundle used by bundle + CLI tests. Entirely invented
# content (see its text.md / manifest.json); no real paper is bundled.
BUNDLE_MINIMAL_DIR = Path(__file__).resolve().parent / "fixtures" / \
    "bundle_minimal"

# Two paper bundles built to exercise the input-handling code against text
# that behaves like a real paper: unicode the engine did not choose, markdown
# tables whose headers carry meaning the cells do not, and a figure. Both are
# invented, like every other fixture here, and are built to a measured
# specification -- line-length distribution and median snap distance included,
# because quote_context's window bound is justified against them.
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
BUNDLE_TABLES_DIR = _FIXTURES / "bundle_tables"    # tables + one figure
BUNDLE_UNICODE_DIR = _FIXTURES / "bundle_unicode"  # unicode-heavy, no figures


@pytest.fixture
def config_dir():
    return CONFIG_DIR


@pytest.fixture
def bundle_minimal_dir():
    return BUNDLE_MINIMAL_DIR


@pytest.fixture
def bundle_tables_dir():
    return BUNDLE_TABLES_DIR


@pytest.fixture
def bundle_unicode_dir():
    return BUNDLE_UNICODE_DIR


@pytest.fixture
def extractor_system_path():
    return EXTRACTOR_SYSTEM_PATH


@pytest.fixture
def review_system_path():
    return REVIEW_SYSTEM_PATH


@pytest.fixture
def checker_system_path():
    return CHECKER_SYSTEM_PATH


@pytest.fixture
def checker_partials_dir():
    """The fixture bundle's `prompts/partials/`.

    Where a checker call looks for a bundle's override of the engine's
    per-field scaffold; this bundle ships none, so the engine's own wording is
    what a test renders.
    """
    return CHECKER_PARTIALS_DIR
