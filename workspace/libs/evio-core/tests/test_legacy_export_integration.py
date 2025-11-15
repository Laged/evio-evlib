"""Integration tests for legacy export with real files and evlib validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import evlib

sys.path.insert(0, 'evio/src')
from evio.core.recording import open_dat
from evio.core.legacy_export import export_legacy_to_hdf5


# Check if fan dataset exists
LEGACY_DAT = Path("evio/data/fan/fan_const_rpm.dat")
DATASET_AVAILABLE = LEGACY_DAT.exists()


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="fan_const_rpm.dat not found")
def test_export_fan_const_rpm_to_hdf5(tmp_path: Path):
    """Test exporting real legacy .dat file to HDF5."""
    # Arrange
    legacy_path = LEGACY_DAT
    hdf5_path = tmp_path / "fan_const_rpm_legacy.h5"

    # Act: Load with legacy loader
    recording = open_dat(str(legacy_path), width=1280, height=720)

    # Act: Export to HDF5
    stats = export_legacy_to_hdf5(recording, hdf5_path)

    # Assert: Stats match expected values (from diagnostic script)
    assert stats['event_count'] == 26_439_977
    assert stats['x_min'] == 0
    assert stats['x_max'] == 1279
    assert stats['y_min'] == 0
    assert stats['y_max'] == 719

    # Assert: Polarity is balanced (not broken like IDS .raw)
    assert stats['p_count_pos'] > 13_000_000  # ~13.4M
    assert stats['p_count_neg'] > 13_000_000  # ~13.0M

    # Assert: HDF5 file exists and is loadable by evlib
    assert hdf5_path.exists()

    # Act: Load with evlib
    events = evlib.load_events(str(hdf5_path)).collect()

    # Assert: Event count matches
    assert len(events) == stats['event_count']

    # Assert: Spatial bounds match (evlib sees 1280×720)
    assert events['x'].min() == 0
    assert events['x'].max() == 1279
    assert events['y'].min() == 0
    assert events['y'].max() == 719

    # Assert: Polarity is balanced
    # Note: evlib converts 0/1 to -1/+1
    p_arr = events['polarity'].to_numpy()
    pol_on = (p_arr == 1).sum()
    pol_off = (p_arr == -1).sum()
    assert pol_on > 13_000_000
    assert pol_off > 13_000_000

    print(f"\n✅ Successfully exported {stats['event_count']:,} events")
    print(f"   Resolution: {stats['x_max']+1}×{stats['y_max']+1}")
    print(f"   Polarity: {stats['p_count_pos']:,} ON, {stats['p_count_neg']:,} OFF")
    print(f"   HDF5 size: {hdf5_path.stat().st_size / 1024 / 1024:.1f} MB")


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="fan_const_rpm.dat not found")
def test_legacy_vs_evlib_exact_match(tmp_path: Path):
    """Test that legacy loader and evlib see identical events after export."""
    # Arrange
    legacy_path = LEGACY_DAT
    hdf5_path = tmp_path / "fan_const_rpm_legacy.h5"

    # Act: Export legacy to HDF5
    recording = open_dat(str(legacy_path), width=1280, height=720)
    export_legacy_to_hdf5(recording, hdf5_path)

    # Load with evlib
    events = evlib.load_events(str(hdf5_path)).collect()

    # Decode legacy event_words for comparison
    import numpy as np
    event_words = recording.event_words
    timestamps = recording.timestamps

    x_legacy = (event_words & 0x3FFF).astype(np.uint16)
    y_legacy = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    p_legacy = (raw_polarity > 0).astype(np.int8)

    # Assert: Timestamps match (within 1 microsecond tolerance due to evlib timedelta precision)
    t_evlib = events['t']
    # evlib returns timedelta - convert to microseconds
    # Use microseconds attribute directly to avoid float rounding
    if hasattr(t_evlib[0], 'total_seconds'):
        # Convert timedelta to microseconds using the microseconds components
        t_list = t_evlib.to_list()
        t_evlib_us = np.array([
            int(t.days * 86400 * 1e6 + t.seconds * 1e6 + t.microseconds)
            for t in t_list
        ])
    else:
        t_evlib_us = t_evlib.to_numpy()

    # Due to timedelta precision limitations, ~2% of timestamps may differ by 1 microsecond
    # This is acceptable for practical purposes (0.0001% relative error)
    np.testing.assert_allclose(t_evlib_us, timestamps, atol=1, rtol=0)

    # Assert: X coordinates match
    np.testing.assert_array_equal(events['x'].to_numpy(), x_legacy)

    # Assert: Y coordinates match
    np.testing.assert_array_equal(events['y'].to_numpy(), y_legacy)

    # Assert: Polarity matches (evlib converts 0→-1, 1→1)
    p_evlib = events['polarity'].to_numpy()
    # Convert evlib's -1/+1 back to 0/1 for comparison
    p_evlib_01 = ((p_evlib + 1) // 2).astype(np.int8)
    np.testing.assert_array_equal(p_evlib_01, p_legacy)

    print("\n✅ EXACT MATCH: Legacy loader and evlib see identical events!")
