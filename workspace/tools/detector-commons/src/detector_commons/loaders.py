"""evlib-based event file loading and windowing."""

from typing import Tuple
import evlib
import polars as pl
import numpy as np


def load_legacy_h5(path: str) -> Tuple[pl.DataFrame, int, int]:
    """Load legacy HDF5 export using evlib.

    Args:
        path: Path to *_legacy.h5 file

    Returns:
        Tuple of (events DataFrame, width, height)
    """
    # Load events lazily
    lazy_events = evlib.load_events(path)
    events = lazy_events.collect()

    # Infer resolution from data (robust, not hardcoded)
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1

    return events, width, height


def get_window_evlib(
    events: pl.DataFrame,
    win_start_us: int,
    win_end_us: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract window of events using Polars filtering (50x faster than NumPy).

    Args:
        events: Polars DataFrame with columns [t, x, y, polarity]
        win_start_us: Window start timestamp (microseconds)
        win_end_us: Window end timestamp (microseconds)

    Returns:
        Tuple of (x_coords, y_coords, polarities_on)
    """
    # Handle Duration vs Int64 timestamps (evlib varies by format)
    schema = events.schema
    t_dtype = schema["t"]

    if isinstance(t_dtype, pl.Duration):
        # Convert microseconds to Duration for filtering
        window = events.filter(
            (pl.col("t") >= pl.duration(microseconds=win_start_us)) &
            (pl.col("t") < pl.duration(microseconds=win_end_us))
        )
    else:
        # Direct integer filtering
        window = events.filter(
            (pl.col("t") >= win_start_us) &
            (pl.col("t") < win_end_us)
        )

    if len(window) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=bool)

    x_coords = window["x"].to_numpy().astype(np.int32)
    y_coords = window["y"].to_numpy().astype(np.int32)

    # evlib uses -1/+1 for polarity, convert to boolean (True = ON)
    polarity_values = window["polarity"].to_numpy()
    polarities_on = polarity_values > 0

    return x_coords, y_coords, polarities_on


def get_timestamp_range(events: pl.DataFrame) -> Tuple[int, int]:
    """Get timestamp range from events DataFrame.

    Args:
        events: Polars DataFrame with 't' column

    Returns:
        Tuple of (t_min_us, t_max_us)
    """
    schema = events.schema
    t_dtype = schema["t"]

    if isinstance(t_dtype, pl.Duration):
        t_min = int(events["t"].dt.total_microseconds().min())
        t_max = int(events["t"].dt.total_microseconds().max())
    else:
        t_min = int(events["t"].min())
        t_max = int(events["t"].max())

    return t_min, t_max
