from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import evlib
import polars as pl


def load_events(path: Path) -> pl.DataFrame:
    """Load entire file into a Polars DataFrame (example-scale)."""
    print(f"[evlib-raw-player] Loading {path} ...")
    lazy = evlib.load_events(str(path))
    df = lazy.collect()
    schema = df.schema
    if isinstance(schema["t"], pl.Duration):
        df = df.with_columns(pl.col("t").dt.total_microseconds().alias("_t_us"))
    else:
        df = df.with_columns(pl.col("t").alias("_t_us"))
    return df


def play_events(
    events: pl.DataFrame,
    window_ms: float,
    speed: float,
) -> None:
    """Stream frames in windowed chunks."""
    t_min = int(events["_t_us"].min())
    t_max = int(events["_t_us"].max())
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1

    print(
        f"[evlib-raw-player] Events={len(events):,}, "
        f"resolution={width}x{height}, "
        f"duration={(t_max - t_min) / 1e6:.2f}s"
    )

    window_us = int(window_ms * 1_000)
    current = t_min
    wall_start = time.perf_counter()

    cv2.namedWindow("evlib raw player", cv2.WINDOW_NORMAL)
    try:
        while current < t_max:
            window = events.filter(
                (pl.col("_t_us") >= current)
                & (pl.col("_t_us") < current + window_us)
            )
            if window.is_empty():
                current += window_us
                continue

            x = window["x"].to_numpy()
            y = window["y"].to_numpy()
            pol = window["polarity"].to_numpy()
            frame = _render_frame(x, y, pol, width, height)
            _draw_hud(frame, current, t_min, wall_start, speed)

            cv2.imshow("evlib raw player", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

            current += window_us
            _sleep_to_speed(current - t_min, wall_start, speed)
    finally:
        cv2.destroyAllWindows()


def _render_frame(
    x: "np.ndarray",
    y: "np.ndarray",
    polarity: "np.ndarray",
    width: int,
    height: int,
) -> "np.ndarray":
    import numpy as np

    frame = np.full((height, width, 3), 127, np.uint8)
    on_mask = polarity > 0
    frame[y[on_mask], x[on_mask]] = (255, 255, 255)
    frame[y[~on_mask], x[~on_mask]] = (0, 0, 0)
    return frame


def _draw_hud(
    frame: "np.ndarray",
    current_time_us: int,
    start_time_us: int,
    wall_start: float,
    speed: float,
) -> None:
    rec_time = (current_time_us - start_time_us) / 1e6
    wall_time = time.perf_counter() - wall_start
    cv2.putText(
        frame,
        f"speed={speed:.2f}x  evlib raw player",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"wall={wall_time:7.3f}s  rec={rec_time:7.3f}s",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _sleep_to_speed(elapsed_us: int, wall_start: float, speed: float) -> None:
    target = elapsed_us / 1e6 / speed
    delta = target - (time.perf_counter() - wall_start)
    if delta > 0:
        time.sleep(delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time playback of EVT3 .raw files via evlib.")
    parser.add_argument("path", type=Path, help="Path to EVT3 .raw/.dat file")
    parser.add_argument("--window", type=float, default=10.0, help="Window duration in ms (default: 10)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = load_events(args.path)
    if events.is_empty():
        print("No events loaded. Check file path/format.")
        return 1
    play_events(events, args.window, args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
