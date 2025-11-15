"""Tests for legacy loader → HDF5 export helper."""

from pathlib import Path
import numpy as np
import h5py
import pytest

from tests.helpers.legacy_export import export_legacy_to_hdf5
from tests.test_evlib_comparison import MockRecording


def test_export_legacy_to_hdf5_basic(tmp_path):
    """Test basic HDF5 export from legacy Recording."""
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

    # Export to HDF5
    out_path = tmp_path / "test.h5"
    stats = export_legacy_to_hdf5(recording, out_path)

    # Verify stats returned
    assert stats['event_count'] == 3
    assert stats['t_min'] == 1000
    assert stats['t_max'] == 3000

    # Verify HDF5 file structure
    assert out_path.exists()

    with h5py.File(out_path, 'r') as f:
        # Check datasets exist (evlib expects /events/p not /events/polarity)
        assert 'events/t' in f
        assert 'events/x' in f
        assert 'events/y' in f
        assert 'events/p' in f

        # Check data
        t = f['events/t'][:]
        x = f['events/x'][:]
        y = f['events/y'][:]
        p = f['events/p'][:]

        assert len(t) == 3
        np.testing.assert_array_equal(t, [1000, 2000, 3000])
        np.testing.assert_array_equal(x, [100, 150, 200])
        np.testing.assert_array_equal(y, [200, 250, 300])
        # Polarity is stored as 0/1 in HDF5 (evlib converts to -1/+1 when loading)
        np.testing.assert_array_equal(p, [1, 0, 1])

        # Check metadata (stored at file level, not events group level)
        assert f.attrs['width'] == 1280
        assert f.attrs['height'] == 720
        assert f.attrs['source'] == 'legacy_dat'


def test_export_legacy_to_hdf5_polarity_mapping(tmp_path):
    """Test polarity is stored as 0/1 in HDF5 (evlib converts to -1/+1 on load)."""
    timestamps = np.array([1000, 2000], dtype=np.int64)

    # Polarity 0 and polarity 1
    event_words = np.array([
        (0 << 28) | (100 << 14) | 50,   # p=0 → stored as 0
        (1 << 28) | (100 << 14) | 50,   # p=1 → stored as 1
    ], dtype=np.uint32)

    order = np.array([0, 1], dtype=np.int32)

    recording = MockRecording(
        width=640,
        height=480,
        timestamps=timestamps,
        event_words=event_words,
        order=order,
    )

    out_path = tmp_path / "polarity_test.h5"
    export_legacy_to_hdf5(recording, out_path)

    with h5py.File(out_path, 'r') as f:
        p = f['events/p'][:]
        # HDF5 stores 0/1, evlib converts to -1/+1 when loading
        np.testing.assert_array_equal(p, [0, 1])
