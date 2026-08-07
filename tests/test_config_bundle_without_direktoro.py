"""Reading a config bundle must not require the provider layer.

A consumer that only READS bundles — parse the YAML, compare hashes, verify a
pin — installs meltiro with `--no-deps` and has no `direktoro` at all. So NO
read path may acquire a transitive direktoro import: a bundle load that needs
the provider layer breaks that consumer outright, and a re-vendor step that
fails quietly leaves it validating against a stale template while its health
check stays green. `import meltiro` being lazy is not enough on its own; every
level below it must stay lazy too.

Each test runs in a FRESH subprocess with an import hook that makes `direktoro`
unimportable, which is the only honest way to check this: an in-process test
passes on an already-imported direktoro no matter how eager the imports are.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CONFIG_BUNDLE = Path(__file__).resolve().parent / "fixtures" / "config_synthetic"

# Installed ahead of everything else, so any direktoro import — at module scope
# or inside a function — raises exactly as it does in a `--no-deps` install.
_BLOCKER = """
import sys
class _NoDirektoro:
    def find_module(self, name, path=None):
        return self if name == "direktoro" or name.startswith("direktoro.") else None
    def find_spec(self, name, path=None, target=None):
        if name == "direktoro" or name.startswith("direktoro."):
            raise ImportError("No module named 'direktoro' (blocked by test)")
        return None
sys.meta_path.insert(0, _NoDirektoro())
"""


# This repo's own source. A bare `python -c "import meltiro"` resolves to
# whatever meltiro is INSTALLED in the interpreter, which need not be this
# checkout, so every subprocess below is pointed here explicitly and then
# asserts what it actually loaded. Without both halves the file passes on a
# tree it never read.
_SRC = Path(__file__).resolve().parent.parent / "src"

_ASSERT_THIS_TREE = f"""
import meltiro, pathlib
_want = pathlib.Path({str(_SRC)!r}).resolve()
_got = pathlib.Path(meltiro.__file__).resolve()
if _want not in _got.parents:
    raise SystemExit(
        f"BLOCKER FAILED: imported meltiro from {{_got}}, not from {{_want}}")
"""


def _run(body):
    code = _BLOCKER + _ASSERT_THIS_TREE + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=str(_SRC))
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120, env=env)


def test_load_config_bundle_works_without_direktoro():
    """The headline claim: a whole bundle parses, hashes and fingerprints with
    the provider layer unimportable."""
    r = _run(f"""
        import sys
        try:
            import direktoro
        except ImportError:
            pass
        else:
            raise SystemExit("BLOCKER FAILED: direktoro was importable")

        from meltiro.config_bundle import load_config_bundle
        b = load_config_bundle({str(CONFIG_BUNDLE)!r})
        assert b.template_hash, "no template_hash on the loaded bundle"
        assert b.reference_lists_hash, "no reference_lists_hash"
        assert b.instrument_fp, "no instrument_fp"
        assert "direktoro" not in sys.modules, "direktoro was imported after all"
        print("OK")
    """)
    assert "OK" in r.stdout, (
        f"loading a config bundle still needs direktoro.\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")


def test_importing_config_bundle_does_not_pull_direktoro():
    """The import alone, before any bundle is touched."""
    r = _run("""
        import sys
        import meltiro.config_bundle  # noqa: F401
        assert "direktoro" not in sys.modules
        print("OK")
    """)
    assert "OK" in r.stdout, f"stdout: {r.stdout}\nstderr: {r.stderr}"


def test_importing_checker_does_not_pull_direktoro():
    """`config_bundle` imports `checker`, so that edge carries the whole
    contract. `CheckerConfig` must be constructible too — its `thinking`
    annotation names a direktoro type."""
    r = _run("""
        import sys
        from meltiro.checker import CheckerConfig, DEFAULT_CONTEXT_CHARS
        assert "direktoro" not in sys.modules
        c = CheckerConfig(checker_model="whatever")
        assert c.thinking is None
        assert DEFAULT_CONTEXT_CHARS
        print("OK")
    """)
    assert "OK" in r.stdout, f"stdout: {r.stdout}\nstderr: {r.stderr}"


def test_default_max_checks_per_field_is_reachable_from_both_modules():
    """It lives in `config_bundle`; the orchestrator re-exports it. Both import
    paths must resolve, which is what the imports below assert: dropping the
    re-export is an ImportError here. The equality is the weaker half — the
    orchestrator binds `config_bundle`'s object, so it cannot differ."""
    from meltiro.config_bundle import DEFAULT_MAX_CHECKS_PER_FIELD as from_cfg
    from meltiro.orchestrator import DEFAULT_MAX_CHECKS_PER_FIELD as from_orch
    assert from_cfg == from_orch


def test_reading_the_constant_does_not_need_the_orchestrator():
    """Reading one integer must not pull the orchestrator in behind it: that
    import is the one that would drag the provider layer into the read path."""
    r = _run("""
        import sys
        from meltiro.config_bundle import DEFAULT_MAX_CHECKS_PER_FIELD
        assert isinstance(DEFAULT_MAX_CHECKS_PER_FIELD, int)
        assert "meltiro.orchestrator" not in sys.modules
        assert "direktoro" not in sys.modules
        print("OK")
    """)
    assert "OK" in r.stdout, f"stdout: {r.stdout}\nstderr: {r.stderr}"
