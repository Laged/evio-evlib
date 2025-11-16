import importlib
import sys
import types
from inspect import signature

import polars as pl
import pytest

import evio.evlib_loader as evlib_loader


def test_evlib_loader_exists():
    """Test that the loader function exists and is callable."""
    assert callable(evlib_loader.load_events_with_evlib)


def test_evlib_loader_signature():
    """Test the loader function has the correct signature."""
    sig = signature(evlib_loader.load_events_with_evlib)
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
    sig = signature(evlib_loader.load_events_with_evlib)

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
            evlib_loader.load_events_with_evlib("/nonexistent/path.dat")


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
        result = evlib_loader.load_events_with_evlib("/nonexistent/path/to/events.dat")
        # Force evaluation if it's a LazyFrame
        if isinstance(result, pl.LazyFrame):
            result.collect()


def test_evlib_loader_normalizes_duration_to_int(monkeypatch):
    """Loader should convert evlib Duration timestamps back to microsecond ints."""
    duration_events = pl.DataFrame(
        {
            "t": pl.duration(microseconds=[0, 50_000]),
            "x": [1, 2],
            "y": [3, 4],
            "polarity": [0, 1],
        }
    ).lazy()

    fake_evlib = types.SimpleNamespace(load_events=lambda path: duration_events)
    monkeypatch.setitem(sys.modules, "evlib", fake_evlib)

    lazy_frame = evlib_loader.load_events_with_evlib("dummy.h5")
    collected = lazy_frame.collect()
    assert collected.schema["t"] == pl.Int64
    assert collected["t"].to_list() == [0, 50_000]
