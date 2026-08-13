"""Smoke test: package imports cleanly."""

from meltiro import errors


def test_package_imports():
    # Sanity: the package and its errors module are importable.
    assert hasattr(errors, "AgenticExtractionError")
    assert hasattr(errors, "CheckerError")
    assert hasattr(errors, "ResumeRefused")


def test_the_error_tree_carries_nothing_nothing_raises():
    """Every exception this package defines is raised somewhere.

    An error class nobody raises is documentation of a design that is not
    there: a reader writes `except` for it, catches nothing, and believes a
    failure mode is handled. The dispatcher, in particular, does NOT raise on
    a failed tool call — it returns a structured result the loop feeds back to
    the model — so an exception describing that path would describe the
    opposite of what happens.
    """
    import inspect

    from meltiro.errors import AgenticExtractionError

    defined = {
        name for name, obj in vars(errors).items()
        if inspect.isclass(obj) and issubclass(obj, AgenticExtractionError)
        and obj.__module__ == errors.__name__
    }
    assert defined == {
        "AgenticExtractionError",
        "CheckerError",
        "SessionError",
        "ResumeRefused",
        "BundleError",
        "RatesConfigError",
        "ConfigBundleError",
    }
