"""Playback script using evlib for EVT3 .dat files.

This script mirrors play_dat.py but uses evlib.load_events() instead of
the legacy DatFileSource, demonstrating the evlib integration path.
"""

import argparse
import time
from pathlib import Path

import cv2
import evlib
import numpy as np
import polars as pl

from evio.core.pacer import Pacer, BatchRange


def load_evt3_file(path: str) -> tuple[pl.DataFrame, int, int]:
    """Load EVT3 file using evlib.

    Args:
        path: Path to EVT3 .dat file

    Returns:
        Tuple of (events DataFrame, width, height)
    """
    # Load events with evlib
    lazy_events = evlib.load_events(path)
    events = lazy_events.collect()

    # Determine resolution from max coordinates
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1

    return events, width, height


def get_window_evlib(
    events: pl.DataFrame,
    win_start_us: int,
    win_end_us: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract window of events using polars filtering.

    Args:
        events: Polars DataFrame with columns [t, x, y, polarity]
        win_start_us: Window start timestamp (microseconds)
        win_end_us: Window end timestamp (microseconds)

    Returns:
        Tuple of (x_coords, y_coords, polarities_on)
    """
    # Handle Duration vs Int64 timestamps
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


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render window of events to frame."""
    x_coords, y_coords, polarities_on = window

    if len(x_coords) == 0:
        return np.full((height, width, 3), base_color, np.uint8)

    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    current_time_us: int,
    start_time_us: int,
    wall_start: float,
    speed: float,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Overlay timing info."""
    wall_time_s = time.perf_counter() - wall_start
    rec_time_s = (current_time_us - start_time_us) / 1e6

    first_row_str = f"speed={speed:.2f}x (evlib loader)"
    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    cv2.putText(frame, first_row_str, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, second_row_str, (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play EVT3 .dat files using evlib loader"
    )
    parser.add_argument("dat", help="Path to EVT3 .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Window duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    args = parser.parse_args()

    # Load file
    print(f"Loading {args.dat} with evlib...")
    events, width, height = load_evt3_file(args.dat)
    print(f"Loaded {len(events):,} events, resolution: {width}x{height}")

    # Get timestamp range
    schema = events.schema
    t_dtype = schema["t"]

    if isinstance(t_dtype, pl.Duration):
        t_min = int(events["t"].dt.total_microseconds().min())
        t_max = int(events["t"].dt.total_microseconds().max())
    else:
        t_min = int(events["t"].min())
        t_max = int(events["t"].max())

    # Playback parameters
    window_us = int(args.window * 1000)
    current_time_us = t_min
    wall_start = time.perf_counter()

    cv2.namedWindow("Evlib Player", cv2.WINDOW_NORMAL)

    try:
        while current_time_us < t_max:
            win_start = current_time_us
            win_end = min(current_time_us + window_us, t_max)

            # Get window events
            window = get_window_evlib(events, win_start, win_end)

            # Render frame
            frame = get_frame(window, width, height)
            draw_hud(frame, current_time_us, t_min, wall_start, args.speed)

            cv2.imshow("Evlib Player", frame)

            # Check for quit
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

            # Advance time
            current_time_us += window_us

            # Simple speed control (sleep to match target speed)
            expected_wall_time = (current_time_us - t_min) / (1e6 * args.speed)
            actual_wall_time = time.perf_counter() - wall_start
            sleep_time = expected_wall_time - actual_wall_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
