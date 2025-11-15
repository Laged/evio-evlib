"""Automated tests for evlib performance characteristics."""

import pytest
from pathlib import Path
import sys
from importlib.util import find_spec

# Add benchmarks to path for importing
benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
sys.path.insert(0, str(benchmarks_dir))


@pytest.mark.skip(reason="Custom .dat format not supported by evlib - need standard format test data")
def test_evlib_faster_than_custom():
    """Verify evlib is at least 2x faster than custom loader.

    Note: This test is currently skipped because the available test data uses a custom
    binary .dat format that evlib doesn't support. To enable this test:
    1. Add test data in a standard format (e.g., .aedat4, .h5)
    2. Update the test_file path below
    3. Remove the @pytest.mark.skip decorator
    """
    from bench_loading import benchmark_custom_loader, benchmark_evlib_loader

    test_file = "data/events.aedat4"  # Update with actual standard format file

    if not Path(test_file).exists():
        pytest.skip(f"Test data not available: {test_file}")

    custom = benchmark_custom_loader(test_file)
    evlib = benchmark_evlib_loader(test_file)

    if evlib is None:
        pytest.skip("evlib doesn't support this file format")

    speedup = custom['time_sec'] / evlib['time_sec']

    print(f"\nBenchmark results:")
    print(f"  Custom: {custom['time_sec']:.4f}s ({custom['events']:,} events)")
    print(f"  evlib:  {evlib['time_sec']:.4f}s ({evlib['events']:,} events)")
    print(f"  Speedup: {speedup:.1f}x")

    # Assert at least 2x speedup (conservative - evlib typically 10-50x)
    assert speedup >= 2.0, f"evlib only {speedup:.1f}x faster, expected >= 2x"

    # Sanity check: same number of events
    assert custom['events'] == evlib['events'], "Loaders returned different event counts"


@pytest.mark.skip(reason="Custom .dat format not supported by evlib")
def test_evlib_loader_returns_expected_columns():
    """Verify evlib returns standard schema.

    Note: Skipped for same reason as test_evlib_faster_than_custom - need standard format test data.
    """
    from evio.evlib_loader import load_events_with_evlib

    # Use standard format test file
    test_file = "data/events.aedat4"  # Update with actual standard format file

    if not Path(test_file).exists():
        pytest.skip("Test data not available")

    events = load_events_with_evlib(test_file)
    events_df = events.collect()

    # Check expected columns
    expected_cols = {'t', 'x', 'y', 'polarity'}
    actual_cols = set(events_df.columns)

    assert expected_cols == actual_cols, f"Expected {expected_cols}, got {actual_cols}"

    # Check data types are reasonable
    assert len(events_df) > 0, "No events loaded"
    assert events_df['t'].min() >= 0, "Timestamps should be non-negative"
    assert events_df['x'].min() >= 0, "X coordinates should be non-negative"
    assert events_df['y'].min() >= 0, "Y coordinates should be non-negative"
    assert set(events_df['polarity'].unique()) <= {0, 1}, "Polarity should be 0 or 1"


def test_custom_loader_baseline():
    """Test that custom .dat loader works as baseline benchmark."""
    from bench_loading import benchmark_custom_loader

    test_file = "data/fan_const_rpm.dat"

    if not Path(test_file).exists():
        pytest.skip("Test data not available")

    result = benchmark_custom_loader(test_file)

    # Verify result structure
    assert 'method' in result
    assert 'events' in result
    assert 'time_sec' in result
    assert 'events_per_sec' in result

    # Sanity checks
    assert result['method'] == 'custom'
    assert result['events'] > 0, "Should load some events"
    assert result['time_sec'] > 0, "Should take some time"
    assert result['events_per_sec'] > 0, "Should have positive throughput"

    print(f"\nCustom loader baseline: {result['time_sec']:.4f}s, "
          f"{result['events_per_sec']/1e6:.1f}M events/s")
