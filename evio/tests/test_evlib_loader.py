import pytest
from pathlib import Path
from evio.evlib_loader import load_events_with_evlib
from inspect import signature
from typing import get_type_hints
import polars as pl


def test_evlib_loader_exists():
    """Test that the loader function exists and is callable."""
    assert callable(load_events_with_evlib)


def test_evlib_loader_signature():
    """Test the loader function has the correct signature."""
    sig = signature(load_events_with_evlib)
    params = list(sig.parameters.keys())

    # Verify required parameters
    assert 'path' in params, "Missing required parameter 'path'"
    assert 'format' in params, "Missing parameter 'format'"

    # Verify parameter types
    assert sig.parameters['path'].annotation == str
    assert 'Optional[str]' in str(sig.parameters['format'].annotation) or \
           sig.parameters['format'].annotation.__class__.__name__ == '_UnionGenericAlias'


def test_evlib_loader_return_type():
    """Test the loader function has correct return type annotation."""
    sig = signature(load_events_with_evlib)

    # Check return annotation is pl.LazyFrame
    assert sig.return_annotation == pl.LazyFrame, \
        f"Expected return type pl.LazyFrame, got {sig.return_annotation}"


def test_evlib_loader_raises_import_error_when_evlib_missing():
    """Test that loader raises ImportError with helpful message when evlib not installed."""
    # This test will only work if evlib is not installed
    # If evlib IS installed, this test will be skipped
    try:
        import evlib
        pytest.skip("evlib is installed, cannot test ImportError case")
    except ImportError:
        # Good - evlib is not installed, we can test the error
        with pytest.raises(ImportError, match="evlib is required but not installed"):
            load_events_with_evlib("/nonexistent/path.dat")


def test_evlib_loader_raises_file_not_found():
    """Test that loader raises appropriate error for nonexistent files."""
    # This test requires evlib to be installed
    try:
        import evlib
    except ImportError:
        pytest.skip("evlib not installed, cannot test file operations")

    # Test with a nonexistent file
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        # evlib may raise different errors for missing files
        result = load_events_with_evlib("/nonexistent/path/to/events.dat")
        # Force evaluation if it's a LazyFrame
        if isinstance(result, pl.LazyFrame):
            result.collect()
