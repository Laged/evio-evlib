"""evlib-based event representations."""

from typing import Tuple
import polars as pl
import numpy as np


def build_accum_frame_evlib(
    events: pl.DataFrame,
    width: int,
    height: int,
) -> np.ndarray:
    """Build accumulated frame using Polars (fast vectorized counting).

    Args:
        events: Polars DataFrame with [t, x, y, polarity]
        width: Sensor width
        height: Sensor height

    Returns:
        Grayscale frame (uint8) with event counts per pixel
    """
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Extract coordinates as numpy arrays
    x_coords = events["x"].to_numpy().astype(np.int32)
    y_coords = events["y"].to_numpy().astype(np.int32)

    # Accumulate into frame
    frame = np.zeros((height, width), dtype=np.uint16)
    np.add.at(frame, (y_coords, x_coords), 1)

    # Normalize to uint8
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def pretty_event_frame_evlib(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polarities_on: np.ndarray,
    width: int,
    height: int,
    *,
    base_color: Tuple[int, int, int] = (127, 127, 127),
    on_color: Tuple[int, int, int] = (255, 255, 255),
    off_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create polarity-separated visualization frame.

    Args:
        x_coords: X coordinates
        y_coords: Y coordinates
        polarities_on: Boolean array (True = ON event)
        width: Sensor width
        height: Sensor height
        base_color: Background color (gray)
        on_color: ON event color (white)
        off_color: OFF event color (black)

    Returns:
        RGB frame for visualization
    """
    frame = np.full((height, width, 3), base_color, np.uint8)

    if len(x_coords) == 0:
        return frame

    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame
