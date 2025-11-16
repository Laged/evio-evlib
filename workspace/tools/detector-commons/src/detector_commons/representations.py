"""evlib-based event representations."""

from typing import Tuple
import polars as pl
import numpy as np


def build_accum_frame_evlib(
    events: pl.DataFrame,
    width: int,
    height: int,
    clip_to_uint8: bool = False,
) -> np.ndarray:
    """Build accumulated frame using Polars (fast vectorized counting).

    Args:
        events: Polars DataFrame with [t, x, y, polarity]
        width: Sensor width
        height: Sensor height
        clip_to_uint8: If True, clip to [0, 255] and return uint8 (for visualization).
                      If False, return uint16 with raw counts (for geometry detection).

    Returns:
        Grayscale frame (uint8 if clip_to_uint8=True, else uint16)

    Notes:
        - Use clip_to_uint8=True for visualization (matches original build_accum_frame)
        - Use clip_to_uint8=False for geometry detection (allows proper normalization)
    """
    if len(events) == 0:
        dtype = np.uint8 if clip_to_uint8 else np.uint16
        return np.zeros((height, width), dtype=dtype)

    # Extract coordinates as numpy arrays
    x_coords = events["x"].to_numpy().astype(np.int32)
    y_coords = events["y"].to_numpy().astype(np.int32)

    # Accumulate into frame (keep as uint16 to avoid clipping)
    frame = np.zeros((height, width), dtype=np.uint16)
    np.add.at(frame, (y_coords, x_coords), 1)

    if clip_to_uint8:
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
