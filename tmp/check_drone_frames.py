"""Quick diagnostic: check drone dataset frame characteristics."""

import sys
import numpy as np
import polars as pl
sys.path.insert(0, "/Users/laged/Codings/laged/evio-evlib/workspace/tools/detector-commons/src")
from detector_commons import load_legacy_h5, get_timestamp_range, build_accum_frame_evlib

# Load data
h5_path = "/Users/laged/Codings/laged/evio-evlib/evio/data/drone_idle/drone_idle_legacy.h5"
print(f"Loading {h5_path}...")
events, width, height = load_legacy_h5(h5_path)
print(f"Loaded {len(events):,} events, resolution: {width}x{height}")

t_min, t_max = get_timestamp_range(events)
window_us = 30000  # 30ms

# Check first 3 frames
for i in range(3):
    win_start = t_min + i * window_us
    win_end = min(win_start + window_us, t_max)

    # Filter window
    schema = events.schema
    if isinstance(schema["t"], pl.Duration):
        window_events = events.filter(
            (pl.col("t") >= pl.duration(microseconds=win_start)) &
            (pl.col("t") < pl.duration(microseconds=win_end))
        )
    else:
        window_events = events.filter(
            (pl.col("t") >= win_start) &
            (pl.col("t") < win_end)
        )

    # Build frame
    frame = build_accum_frame_evlib(window_events, width, height)

    print(f"\n=== Frame {i} (t={win_start}-{win_end} us) ===")
    print(f"Events in window: {len(window_events)}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame range: [{frame.min()}, {frame.max()}]")
    print(f"Non-zero pixels: {np.count_nonzero(frame)}")
    print(f"Pixels >10: {np.count_nonzero(frame > 10)}")
    print(f"Pixels >50: {np.count_nonzero(frame > 50)}")
    print(f"Pixels >100: {np.count_nonzero(frame > 100)}")
    print(f"Pixels >1000: {np.count_nonzero(frame > 1000)}")

    # Find hot pixels
    hot_pixels = np.where(frame > 1000)
    if len(hot_pixels[0]) > 0:
        print(f"Hot pixel locations (>1000 counts):")
        for j in range(min(5, len(hot_pixels[0]))):
            y, x = hot_pixels[0][j], hot_pixels[1][j]
            print(f"  ({x},{y}): {frame[y, x]} events")
