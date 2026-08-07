"""The curated public API surface (`from meltiro import ...`).

The integration is library-first: a consumer imports functions rather than
shelling out to the CLI. This pins that EVERY name the package advertises in
`__all__` is importable from the package root and resolves to the object its
defining module exports, so a consumer never has to reach into a deep module
path and never binds to an internal one.
"""

import importlib

import meltiro


EXPECTED = {
    "__version__",
    "validate_value",
    "validate_extraction_output",
    "ValidationResult",
    "load_config_bundle",
    "ConfigBundle",
    "load_bundle",
    "PaperBundle",
    "RUN_STATUSES",
    "TERMINAL_STATUSES",
    "VALIDATED_STATUSES",
}


def test_all_matches_expected_surface():
    assert set(meltiro.__all__) == EXPECTED


def test_no_duplicate_names_in_all():
    assert len(meltiro.__all__) == len(set(meltiro.__all__))


def test_every_exported_name_is_importable():
    ns = importlib.import_module("meltiro")
    for name in meltiro.__all__:
        assert hasattr(ns, name), f"meltiro.{name} is missing"


def test_star_import_binds_every_name():
    ns = {}
    exec("from meltiro import *", ns)
    for name in meltiro.__all__:
        assert name in ns, f"`from meltiro import *` did not bind {name}"


def test_exports_are_the_module_objects():
    # Each re-export must be the same object its defining module exposes, so
    # the package root and the internal path never drift.
    from meltiro import bundle, config_bundle, statuses, validators

    assert meltiro.validate_value is validators.validate_value
    assert meltiro.validate_extraction_output is \
        validators.validate_extraction_output
    assert meltiro.ValidationResult is validators.ValidationResult
    assert meltiro.load_config_bundle is config_bundle.load_config_bundle
    assert meltiro.ConfigBundle is config_bundle.ConfigBundle
    assert meltiro.load_bundle is bundle.load_bundle
    assert meltiro.PaperBundle is bundle.PaperBundle
    assert meltiro.RUN_STATUSES is statuses.RUN_STATUSES
    assert meltiro.TERMINAL_STATUSES is statuses.TERMINAL_STATUSES
    assert meltiro.VALIDATED_STATUSES is statuses.VALIDATED_STATUSES


def test_the_model_registry_is_not_re_exported():
    # The registry belongs to direktoro, a declared dependency, and a consumer
    # that wants it imports it from there. Forwarding it through this package
    # would give the same objects a second name to keep in step, and would tie
    # the advertised surface to a package a --no-deps consumer does not have.
    for name in ("cost_of", "is_known_model", "known_models", "model_info"):
        assert not hasattr(meltiro, name), (
            f"meltiro.{name} is back; the registry is direktoro's surface")
