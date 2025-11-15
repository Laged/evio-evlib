"""Unit tests for legacy .dat to HDF5 export."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
import pytest


@dataclass(frozen=True)
class MockRecording:
    """Mock Recording for testing."""
    width: int
    height: int
    event_words: np.ndarray
    timestamps: np.ndarray


def test_export_creates_hdf5_with_correct_schema(tmp_path: Path):
    """Test that export creates HDF5 with evlib-compatible schema."""
    # Arrange: Create mock recording
    event_words = np.array([
        0x10000000,  # polarity=1, y=0, x=0
        0x10004001,  # polarity=1, y=1, x=1
        0x00008002,  # polarity=0, y=2, x=2
    ], dtype=np.uint32)

    timestamps = np.array([1000, 2000, 3000], dtype=np.int64)

    recording = MockRecording(
        width=1280,
        height=720,
        event_words=event_words,
        timestamps=timestamps,
    )

    out_path = tmp_path / "test.h5"

    # Act: Export to HDF5
    from evio.core.legacy_export import export_legacy_to_hdf5
    stats = export_legacy_to_hdf5(recording, out_path)

    # Assert: File exists
    assert out_path.exists()

    # Assert: Stats returned
    assert stats['event_count'] == 3
    assert stats['x_min'] == 0
    assert stats['x_max'] == 2
    assert stats['y_min'] == 0
    assert stats['y_max'] == 2
    assert stats['p_count_pos'] == 2
    assert stats['p_count_neg'] == 1

    # Assert: HDF5 structure matches evlib schema
    with h5py.File(out_path, 'r') as f:
        assert 'events/t' in f
        assert 'events/x' in f
        assert 'events/y' in f
        assert 'events/p' in f

        assert f.attrs['width'] == 1280
        assert f.attrs['height'] == 720
        assert f.attrs['source'] == 'legacy_dat'

        # Check data types
        assert f['events/t'].dtype == np.int64
        assert f['events/x'].dtype == np.uint16
        assert f['events/y'].dtype == np.uint16
        assert f['events/p'].dtype == np.int8


def test_export_decodes_polarity_correctly(tmp_path: Path):
    """Test that polarity is decoded from bits 31:28."""
    # Arrange: Events with different polarities
    event_words = np.array([
        0x00000000,  # polarity=0 (raw_polarity=0)
        0x10000000,  # polarity=1 (raw_polarity=1)
        0x20000000,  # polarity=1 (raw_polarity=2, any non-zero → 1)
        0xF0000000,  # polarity=1 (raw_polarity=15)
    ], dtype=np.uint32)

    timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.int64)

    recording = MockRecording(
        width=1280,
        height=720,
        event_words=event_words,
        timestamps=timestamps,
    )

    out_path = tmp_path / "polarity_test.h5"

    # Act
    from evio.core.legacy_export import export_legacy_to_hdf5
    stats = export_legacy_to_hdf5(recording, out_path)

    # Assert: Polarity counts
    assert stats['p_count_neg'] == 1  # Only first event is 0
    assert stats['p_count_pos'] == 3  # Last three are 1

    # Assert: HDF5 polarity values
    with h5py.File(out_path, 'r') as f:
        polarity = f['events/p'][:]
        assert polarity[0] == 0
        assert polarity[1] == 1
        assert polarity[2] == 1
        assert polarity[3] == 1


def test_export_decodes_xy_coordinates_correctly(tmp_path: Path):
    """Test that x/y are decoded from correct bit positions."""
    # Arrange: Events with known x/y positions
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    event_words = np.array([
        0x00000000,  # x=0, y=0
        0x00004001,  # x=1, y=1 (y at bits 14-27, x at bits 0-13)
        0x000140FF,  # x=255, y=5 (5 << 14 = 0x14000)
        0x00000500,  # x=1280, y=0 (1280 = 0x500)
    ], dtype=np.uint32)

    timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.int64)

    recording = MockRecording(
        width=1280,
        height=720,
        event_words=event_words,
        timestamps=timestamps,
    )

    out_path = tmp_path / "xy_test.h5"

    # Act
    from evio.core.legacy_export import export_legacy_to_hdf5
    export_legacy_to_hdf5(recording, out_path)

    # Assert: Coordinates decoded correctly
    with h5py.File(out_path, 'r') as f:
        x = f['events/x'][:]
        y = f['events/y'][:]

        assert x[0] == 0 and y[0] == 0
        assert x[1] == 1 and y[1] == 1
        assert x[2] == 255 and y[2] == 5
        assert x[3] == 1280 and y[3] == 0
