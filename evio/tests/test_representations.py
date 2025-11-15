"""Tests for evlib representation wrappers."""

import pytest
import polars as pl
import numpy as np


def test_create_voxel_grid():
    """Test voxel grid creation using evlib."""
    from evio.representations import create_voxel_grid

    # Create sample events
    events = pl.DataFrame({
        't': [1000, 2000, 3000, 11000, 12000, 13000],
        'x': [100, 100, 100, 100, 100, 100],
        'y': [50, 50, 50, 50, 50, 50],
        'polarity': [1, 0, 1, 1, 0, 1],
    }).lazy()

    voxel = create_voxel_grid(
        events,
        height=720,
        width=1280,
        n_time_bins=2,
    )

    # Verify it's a Polars DataFrame
    assert isinstance(voxel, pl.DataFrame)

    # Verify schema (evlib returns: x, y, time_bin, contribution)
    assert 'time_bin' in voxel.columns
    assert 'y' in voxel.columns
    assert 'x' in voxel.columns
    assert 'contribution' in voxel.columns


def test_create_stacked_histogram():
    """Test stacked histogram creation (RVT input format)."""
    from evio.representations import create_stacked_histogram

    events = pl.DataFrame({
        't': list(range(1000, 50000, 100)),  # 490 events over 49ms
        'x': [100] * 490,
        'y': [50] * 490,
        'polarity': [1, 0] * 245,  # Alternating polarity
    }).lazy()

    hist = create_stacked_histogram(
        events,
        height=720,
        width=1280,
        bins=10,
        window_duration_ms=50.0,
    )

    # Verify schema
    assert isinstance(hist, pl.DataFrame)
    assert 'time_bin' in hist.columns
    assert 'polarity' in hist.columns
    assert 'y' in hist.columns
    assert 'x' in hist.columns
    assert 'count' in hist.columns


def test_voxel_to_numpy_array():
    """Test conversion of voxel grid to NumPy array."""
    from evio.representations import create_voxel_grid, voxel_to_numpy

    events = pl.DataFrame({
        't': [1000, 2000, 11000, 12000],
        'x': [100, 200, 100, 200],
        'y': [50, 60, 50, 60],
        'polarity': [1, 0, 1, 0],
    }).lazy()

    voxel = create_voxel_grid(events, height=720, width=1280, n_time_bins=2)

    array = voxel_to_numpy(voxel, n_time_bins=2, height=720, width=1280)

    assert array.shape == (2, 720, 1280)
    assert array.dtype == np.float32  # evlib uses float contributions

    # Verify events are in correct bins (contributions should be > 0)
    assert array[0, 50, 100] > 0  # First bin, pixel (100, 50)
    assert array[1, 50, 100] > 0  # Second bin, pixel (100, 50)


def test_histogram_to_numpy():
    """Test conversion of stacked histogram to NumPy 4D array."""
    from evio.representations import create_stacked_histogram, histogram_to_numpy

    events = pl.DataFrame({
        't': list(range(1000, 50000, 100)),
        'x': [100] * 490,
        'y': [50] * 490,
        'polarity': [1, 0] * 245,
    }).lazy()

    hist = create_stacked_histogram(
        events,
        height=720,
        width=1280,
        bins=10,
        window_duration_ms=50.0,
    )

    array = histogram_to_numpy(hist, bins=10, height=720, width=1280)

    assert array.shape == (10, 2, 720, 1280)
    assert array.dtype == np.float32
