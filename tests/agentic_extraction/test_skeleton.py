"""Smoke test: package imports cleanly."""

from meltiro import errors


def test_package_imports():
    # Sanity: the package and its errors module are importable.
    assert hasattr(errors, "AgenticExtractionError")
    assert hasattr(errors, "ValidationFailure")
    assert hasattr(errors, "ResumeRefused")


def test_validation_failure_carries_errors():
    err = errors.ValidationFailure([
        {"path": "x", "code": "y", "message": "z"},
    ])
    assert err.errors == [{"path": "x", "code": "y", "message": "z"}]
    assert "1 validation error" in str(err)
