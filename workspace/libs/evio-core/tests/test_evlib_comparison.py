"""Comparison tests between evlib and legacy loaders.

IMPORTANT: The .raw files in evio/data are NOT conversions of legacy .dat files.
They are independent IDS camera recordings from Nov 2025 with different:
- Resolution (2040×1793 actual vs 1280×720)
- Duration (682-717 sec vs 9-24 sec)
- Event counts (30-73M vs 26-64M)
- Polarity encoding (broken: 0 OFF events)

See docs/plans/2025-11-16-legacy-dat-to-evlib-export.md for details.

To test legacy parity, we export legacy .dat → HDF5 → evlib, ensuring
both loaders see the SAME recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import polars as pl
import evlib


# Dataset availability check
def dataset_exists(raw_path: str, dat_path: str) -> bool:
    """Check if both .raw and _evt3.dat files exist."""
    return Path(raw_path).exists() and Path(dat_path).exists()


DATASETS_AVAILABLE = (
    dataset_exists("evio/data/fan/fan_const_rpm.raw",
                   "evio/data/fan/fan_const_rpm_evt3.dat") and
    dataset_exists("evio/data/drone_idle/drone_idle.raw",
                   "evio/data/drone_idle/drone_idle_evt3.dat")
)


def decode_legacy_events(event_words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode packed uint32 event_words into x, y, polarity arrays.

    Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    See: evio/src/evio/core/mmap.py:151-154

    Args:
        event_words: Packed uint32 events from legacy loader

    Returns:
        Tuple of (x, y, polarity) numpy arrays
    """
    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    polarity = (raw_polarity > 0).astype(np.int8)
    return x, y, polarity


@dataclass(frozen=True)
class MockRecording:
    """Mock Recording for testing."""
    width: int
    height: int
    timestamps: np.ndarray
    event_words: np.ndarray
    order: np.ndarray


def compute_legacy_stats(recording) -> dict[str, int]:
    """Extract statistics from legacy Recording object.

    Args:
        recording: Recording object from evio.core.recording.open_dat()

    Returns:
        Dict with keys:
            - event_count: total events
            - t_min, t_max: timestamp range (microseconds)
            - x_min, x_max, y_min, y_max: spatial bounds
            - p_count_0, p_count_1: polarity distribution
    """
    x, y, polarity = decode_legacy_events(recording.event_words)
    return {
        'event_count': len(recording.timestamps),
        't_min': int(recording.timestamps.min()),
        't_max': int(recording.timestamps.max()),
        'x_min': int(x.min()),
        'x_max': int(x.max()),
        'y_min': int(y.min()),
        'y_max': int(y.max()),
        'p_count_0': int((polarity == 0).sum()),
        'p_count_1': int((polarity == 1).sum()),
    }


def compute_evlib_stats(dat_path: Path) -> dict[str, int]:
    """Extract statistics from evlib-loaded file.

    Handles both Duration and Int64 timestamp types.
    See: workspace/tools/evio-verifier/src/evio_verifier/cli.py:46-76

    Args:
        dat_path: Path to EVT3 .dat file

    Returns:
        Dict with same keys as compute_legacy_stats()
    """
    lazy = evlib.load_events(str(dat_path))

    # Handle Duration vs Int64 timestamps
    schema = lazy.collect_schema()
    t_dtype = schema["t"]

    if isinstance(t_dtype, pl.Duration):
        t_min_expr = pl.col("t").dt.total_microseconds().min()
        t_max_expr = pl.col("t").dt.total_microseconds().max()
    else:
        t_min_expr = pl.col("t").min()
        t_max_expr = pl.col("t").max()

    stats = lazy.select([
        pl.len().alias("event_count"),
        t_min_expr.alias("t_min"),
        t_max_expr.alias("t_max"),
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("y").min().alias("y_min"),
        pl.col("y").max().alias("y_max"),
        (pl.col("polarity") == -1).sum().alias("p_count_0"),
        (pl.col("polarity") == 1).sum().alias("p_count_1"),
    ]).collect().to_dicts()[0]

    return {k: int(v) for k, v in stats.items()}


def assert_within_tolerance(expected: int, actual: int, tolerance: float, label: str = "value") -> None:
    """Assert values match within percentage tolerance.

    Args:
        expected: Expected value from legacy loader
        actual: Actual value from evlib loader
        tolerance: Maximum relative difference (e.g., 0.0001 = 0.01%)
        label: Description for error messages

    Raises:
        AssertionError: If values differ by more than tolerance
    """
    if expected == 0:
        assert actual == 0, f"{label}: Expected 0, got {actual}"
    else:
        rel_diff = abs(actual - expected) / abs(expected)
        assert rel_diff <= tolerance, \
            f"{label}: Expected {expected}, got {actual} (diff: {rel_diff:.4%} > {tolerance:.4%})"


def test_decode_legacy_events():
    """Test decoding of packed uint32 event words."""
    # Create test event: x=100, y=200, polarity=1
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    # polarity=1 -> bits [31:28] = 0x1
    # y=200 -> bits [27:14] = 200 = 0xC8
    # x=100 -> bits [13:0] = 100 = 0x64
    event_word = (1 << 28) | (200 << 14) | 100
    event_words = np.array([event_word], dtype=np.uint32)

    x, y, polarity = decode_legacy_events(event_words)

    assert x[0] == 100
    assert y[0] == 200
    assert polarity[0] == 1


def test_decode_legacy_events_polarity_zero():
    """Test decoding of polarity=0 events."""
    # polarity=0 -> bits [31:28] = 0x0
    event_word = (0 << 28) | (150 << 14) | 50
    event_words = np.array([event_word], dtype=np.uint32)

    x, y, polarity = decode_legacy_events(event_words)

    assert x[0] == 50
    assert y[0] == 150
    assert polarity[0] == 0


def test_compute_legacy_stats():
    """Test statistics extraction from legacy Recording."""
    # Create mock recording with 3 events
    timestamps = np.array([1000, 2000, 3000], dtype=np.int64)

    # Event 1: x=100, y=200, p=1
    # Event 2: x=150, y=250, p=0
    # Event 3: x=200, y=300, p=1
    event_words = np.array([
        (1 << 28) | (200 << 14) | 100,
        (0 << 28) | (250 << 14) | 150,
        (1 << 28) | (300 << 14) | 200,
    ], dtype=np.uint32)

    order = np.array([0, 1, 2], dtype=np.int32)

    recording = MockRecording(
        width=1280,
        height=720,
        timestamps=timestamps,
        event_words=event_words,
        order=order,
    )

    stats = compute_legacy_stats(recording)

    assert stats['event_count'] == 3
    assert stats['t_min'] == 1000
    assert stats['t_max'] == 3000
    assert stats['x_min'] == 100
    assert stats['x_max'] == 200
    assert stats['y_min'] == 200
    assert stats['y_max'] == 300
    assert stats['p_count_0'] == 1
    assert stats['p_count_1'] == 2


@pytest.mark.skipif(
    not Path("evio/data/fan/fan_const_rpm_evt3.dat").exists(),
    reason="EVT3 test file not available"
)
def test_compute_evlib_stats():
    """Test statistics extraction from evlib-loaded file."""
    # Use actual converted file for this test
    dat_path = Path("evio/data/fan/fan_const_rpm_evt3.dat")

    stats = compute_evlib_stats(dat_path)

    # Basic sanity checks on known dataset
    assert stats['event_count'] > 0
    assert stats['t_min'] >= 0
    assert stats['t_max'] > stats['t_min']
    assert stats['x_min'] >= 0
    assert stats['x_max'] > stats['x_min']
    assert stats['y_min'] >= 0
    assert stats['y_max'] > stats['y_min']
    assert stats['p_count_0'] >= 0
    assert stats['p_count_1'] >= 0
    assert stats['p_count_0'] + stats['p_count_1'] == stats['event_count']


def test_assert_within_tolerance_exact_match():
    """Test tolerance check with exact match."""
    assert_within_tolerance(1000, 1000, 0.0001, "exact")
    # Should not raise


def test_assert_within_tolerance_within_bounds():
    """Test tolerance check within bounds."""
    # 0.01% of 10000 = 1, so 10001 should pass
    assert_within_tolerance(10000, 10001, 0.0001, "within")
    # Should not raise


def test_assert_within_tolerance_exceeds_bounds():
    """Test tolerance check exceeds bounds."""
    with pytest.raises(AssertionError, match="diff:"):
        # 0.01% of 10000 = 1, so 10002 should fail (0.02% diff)
        assert_within_tolerance(10000, 10002, 0.0001, "exceeds")


def test_assert_within_tolerance_zero():
    """Test tolerance check with zero expected."""
    assert_within_tolerance(0, 0, 0.0001, "zero")
    # Should not raise

    with pytest.raises(AssertionError):
        assert_within_tolerance(0, 1, 0.0001, "zero_fail")


@pytest.mark.skipif(
    not DATASETS_AVAILABLE,
    reason="Datasets not found. Run 'unzip-datasets' and 'convert-all-datasets' first."
)
@pytest.mark.parametrize("dataset_name,raw_path,dat_path,width,height", [
    (
        "fan_const_rpm",
        "evio/data/fan/fan_const_rpm.raw",
        "evio/data/fan/fan_const_rpm_evt3.dat",
        1280,
        720,
    ),
    (
        "drone_idle",
        "evio/data/drone_idle/drone_idle.raw",
        "evio/data/drone_idle/drone_idle_evt3.dat",
        1280,
        720,
    ),
])
def test_evlib_vs_legacy_stats(
    dataset_name: str,
    raw_path: str,
    dat_path: str,
    width: int,
    height: int,
):
    """Validates EVT3 .dat conversion preserves data by comparing .raw vs converted .dat using evlib.

    This test loads the same data in two formats (.raw and _evt3.dat) using evlib for both,
    and verifies they contain identical events. The files should have the same size and data.
    """
    # Load .raw file with evlib
    raw_stats = compute_evlib_stats(Path(raw_path))

    # Load _evt3.dat file with evlib
    dat_stats = compute_evlib_stats(Path(dat_path))

    # Assert exact match on all stats - these should be identical
    assert raw_stats['event_count'] == dat_stats['event_count'], \
        f"{dataset_name}: Event count mismatch"

    assert raw_stats['p_count_0'] == dat_stats['p_count_0'], \
        f"{dataset_name}: Polarity 0 count mismatch"

    assert raw_stats['p_count_1'] == dat_stats['p_count_1'], \
        f"{dataset_name}: Polarity 1 count mismatch"

    assert raw_stats['t_min'] == dat_stats['t_min'], \
        f"{dataset_name}: t_min mismatch"

    assert raw_stats['t_max'] == dat_stats['t_max'], \
        f"{dataset_name}: t_max mismatch"

    assert raw_stats['x_min'] == dat_stats['x_min'], \
        f"{dataset_name}: x_min mismatch"

    assert raw_stats['x_max'] == dat_stats['x_max'], \
        f"{dataset_name}: x_max mismatch"

    assert raw_stats['y_min'] == dat_stats['y_min'], \
        f"{dataset_name}: y_min mismatch"

    assert raw_stats['y_max'] == dat_stats['y_max'], \
        f"{dataset_name}: y_max mismatch"

    # Print summary for visibility
    print(f"\n{dataset_name} comparison:")
    print(f"  Events: {dat_stats['event_count']:,}")
    print(f"  Time range: {dat_stats['t_min']} → {dat_stats['t_max']}")
    print(f"  X range: {dat_stats['x_min']} → {dat_stats['x_max']}")
    print(f"  Y range: {dat_stats['y_min']} → {dat_stats['y_max']}")
    print(f"  Polarity: 0={dat_stats['p_count_0']:,}, 1={dat_stats['p_count_1']:,}")


# Legacy loader vs evlib parity tests
LEGACY_PARITY_DATASETS_AVAILABLE = (
    Path("evio/data/fan/fan_const_rpm.dat").exists() and
    Path("evio/data/drone_idle/drone_idle.dat").exists()
)


@pytest.mark.skipif(
    not LEGACY_PARITY_DATASETS_AVAILABLE,
    reason="Legacy parity datasets not found. Run 'unzip-datasets' and 'convert-all-datasets' first."
)
@pytest.mark.parametrize("dataset_name,legacy_dat,width,height", [
    (
        "fan_const_rpm",
        "evio/data/fan/fan_const_rpm.dat",
        1280,
        720,
    ),
    (
        "drone_idle",
        "evio/data/drone_idle/drone_idle.dat",
        1280,
        720,
    ),
])
def test_legacy_loader_vs_evlib_parity(
    dataset_name: str,
    legacy_dat: str,
    width: int,
    height: int,
    tmp_path_factory,
):
    """Validates legacy loader parity with evlib via HDF5 round-trip.

    This test proves the legacy loader (evio.core.recording.open_dat) and evlib produce
    equivalent event statistics on identical data.

    Flow:
    1. Load original custom .dat with legacy loader
    2. Export events to temporary HDF5 file (evlib-compatible schema)
    3. Load HDF5 with evlib
    4. Compare stats: legacy extraction vs evlib load

    Once passing, we can safely retire evio.core.recording knowing evlib reproduces its output.
    """
    from evio.core.recording import open_dat
    from tests.helpers.legacy_export import export_legacy_to_hdf5

    # Load with legacy loader
    recording = open_dat(legacy_dat, width=width, height=height)

    # Export to temporary HDF5
    tmp_dir = tmp_path_factory.mktemp("legacy_export")
    hdf5_path = tmp_dir / f"{dataset_name}.h5"

    legacy_stats_from_export = export_legacy_to_hdf5(recording, hdf5_path)

    # Also compute stats directly from recording for verification
    legacy_stats = compute_legacy_stats(recording)

    # Load HDF5 with evlib
    evlib_stats = compute_evlib_stats(hdf5_path)

    # Compare export stats vs direct legacy stats (sanity check)
    assert legacy_stats['event_count'] == legacy_stats_from_export['event_count']
    assert legacy_stats['t_min'] == legacy_stats_from_export['t_min']
    assert legacy_stats['t_max'] == legacy_stats_from_export['t_max']

    # Compare legacy vs evlib (main validation)
    assert legacy_stats['event_count'] == evlib_stats['event_count'], \
        f"{dataset_name}: Event count mismatch (legacy={legacy_stats['event_count']}, evlib={evlib_stats['event_count']})"

    # Polarity comparison: legacy uses 0/1, export converts to -1/+1, evlib reads -1/+1
    assert legacy_stats['p_count_0'] == evlib_stats['p_count_0'], \
        f"{dataset_name}: Polarity -1 count mismatch (legacy 0s={legacy_stats['p_count_0']}, evlib -1s={evlib_stats['p_count_0']})"

    assert legacy_stats['p_count_1'] == evlib_stats['p_count_1'], \
        f"{dataset_name}: Polarity +1 count mismatch (legacy 1s={legacy_stats['p_count_1']}, evlib +1s={evlib_stats['p_count_1']})"

    # Timestamp range - should match exactly
    assert legacy_stats['t_min'] == evlib_stats['t_min'], \
        f"{dataset_name}: t_min mismatch (legacy={legacy_stats['t_min']}, evlib={evlib_stats['t_min']})"

    assert legacy_stats['t_max'] == evlib_stats['t_max'], \
        f"{dataset_name}: t_max mismatch (legacy={legacy_stats['t_max']}, evlib={evlib_stats['t_max']})"

    # Spatial ranges - should match exactly
    assert legacy_stats['x_min'] == evlib_stats['x_min'], \
        f"{dataset_name}: x_min mismatch (legacy={legacy_stats['x_min']}, evlib={evlib_stats['x_min']})"

    assert legacy_stats['x_max'] == evlib_stats['x_max'], \
        f"{dataset_name}: x_max mismatch (legacy={legacy_stats['x_max']}, evlib={evlib_stats['x_max']})"

    assert legacy_stats['y_min'] == evlib_stats['y_min'], \
        f"{dataset_name}: y_min mismatch (legacy={legacy_stats['y_min']}, evlib={evlib_stats['y_min']})"

    assert legacy_stats['y_max'] == evlib_stats['y_max'], \
        f"{dataset_name}: y_max mismatch (legacy={legacy_stats['y_max']}, evlib={evlib_stats['y_max']})"

    # Print summary for visibility
    print(f"\n{dataset_name} legacy parity validation:")
    print(f"  Events: {legacy_stats['event_count']:,}")
    print(f"  Legacy extraction → HDF5 → evlib load: ✓ MATCH")
    print(f"  Time range: {legacy_stats['t_min']} → {legacy_stats['t_max']}")
    print(f"  X range: {legacy_stats['x_min']} → {legacy_stats['x_max']}")
    print(f"  Y range: {legacy_stats['y_min']} → {legacy_stats['y_max']}")
    print(f"  Polarity: -1={legacy_stats['p_count_0']:,}, +1={legacy_stats['p_count_1']:,}")
    print(f"  ✓ Legacy loader matches evlib output on identical data")
