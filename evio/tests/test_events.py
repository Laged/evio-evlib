"""Tests for evio event data structures."""

import pytest
import numpy as np
import polars as pl


def test_event_data_from_polars():
    """Test EventData creation from Polars LazyFrame."""
    from evio.events import EventData

    # Create sample Polars DataFrame
    df = pl.DataFrame({
        't': [100, 200, 300],
        'x': [10, 20, 30],
        'y': [50, 60, 70],
        'polarity': [1, 0, 1],
    })

    events = EventData.from_polars(df.lazy())

    assert events.num_events == 3
    assert events.width is None  # Not set yet
    assert events.height is None


def test_event_data_to_numpy():
    """Test conversion to NumPy arrays for MVP compatibility."""
    from evio.events import EventData

    df = pl.DataFrame({
        't': [100, 200, 300],
        'x': [10, 20, 30],
        'y': [50, 60, 70],
        'polarity': [1, 0, 1],
    })

    events = EventData.from_polars(df.lazy())

    t, x, y, p = events.to_numpy()

    assert isinstance(t, np.ndarray)
    assert len(t) == 3
    assert t[0] == 100
    assert x[1] == 20
    assert p[2] == 1


def test_event_data_with_resolution():
    """Test EventData with sensor resolution metadata."""
    from evio.events import EventData

    df = pl.DataFrame({
        't': [100],
        'x': [10],
        'y': [50],
        'polarity': [1],
    })

    events = EventData.from_polars(df.lazy(), width=1280, height=720)

    assert events.width == 1280
    assert events.height == 720


def test_event_data_filter_roi():
    """Test ROI filtering (lazy operation)."""
    from evio.events import EventData

    df = pl.DataFrame({
        't': [100, 200, 300, 400],
        'x': [10, 50, 90, 150],
        'y': [20, 60, 100, 200],
        'polarity': [1, 0, 1, 0],
    })

    events = EventData.from_polars(df.lazy(), width=1280, height=720)

    # Filter to ROI: x=[40, 100], y=[50, 110]
    roi_events = events.filter_roi(x_min=40, x_max=100, y_min=50, y_max=110)

    # Should keep events at (50, 60) and (90, 100)
    t, x, y, p = roi_events.to_numpy()
    assert len(t) == 2
    assert list(x) == [50, 90]
    assert list(y) == [60, 100]
