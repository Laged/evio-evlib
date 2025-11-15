"""Drone propeller detector demo - evlib version.

This is the evlib-migrated version of drone_detector_demo.py.
Demonstrates multi-propeller detection using detector-commons utilities.

Key differences from fan-rpm-demo:
- Detects MULTIPLE ellipses (up to 2 propellers) instead of single
- Uses pick_propellers_at_time (not pick_geom_at_time) for multi-ellipse lookup
- Tracks per-propeller RPM separately (propeller_data dict)
- Shows stacked view with warning overlay (NO matplotlib plots)
"""

import argparse
from typing import List, Tuple, Dict
import numpy as np
import cv2
import polars as pl

from detector_commons import (
    load_legacy_h5,
    get_window_evlib,
    get_timestamp_range,
    build_accum_frame_evlib,
    pretty_event_frame_evlib,
    cluster_blades_dbscan_elliptic,
    pick_propellers_at_time,
)
from .geometry import propeller_mask_from_frame, ellipse_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drone propeller detector using evlib"
    )
    parser.add_argument("h5", help="Path to *_legacy.h5 file")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=30.0,
        help="Window duration in ms for event accumulation (default: 30 ms)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit on number of windows to show (0 = no limit)",
    )
    parser.add_argument(
        "--cluster-window-ms",
        type=float,
        default=0.5,
        help="Window duration in ms for DBSCAN cluster visualization",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=10.0,
        help="DBSCAN eps (cluster radius in pixels)",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=15,
        help="DBSCAN min_samples (minimum points per cluster)",
    )
    parser.add_argument(
        "--max-ellipses",
        type=int,
        default=2,
        help="Maximum number of propellers to detect (default: 2)",
    )
    args = parser.parse_args()

    print(f"Loading {args.h5} with evlib...")
    events, width, height = load_legacy_h5(args.h5)
    print(f"Loaded {len(events):,} events, resolution: {width}x{height}")

    t_min, t_max = get_timestamp_range(events)
    window_us = int(args.window_ms * 1000)

    # PASS 1: Collect multi-ellipse geometry (coarse windows)
    print(f"\nPass 1: Collecting geometry ({args.window_ms}ms windows)...")
    ell_times = []
    ell_ellipses_per_window: List[List[Tuple[int, int, float, float, float]]] = []

    current_time = t_min
    frame_count = 0

    while current_time < t_max:
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

        win_start = current_time
        win_end = min(current_time + window_us, t_max)

        # Filter window using Polars - handle Duration timestamps
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
        frame_accum = build_accum_frame_evlib(window_events, width, height)

        # Detect multiple ellipses (propellers)
        ellipses = propeller_mask_from_frame(frame_accum, max_ellipses=args.max_ellipses)

        if not ellipses:
            # Skip frames where we didn't detect propellers
            current_time += window_us
            continue

        # Store geometry
        t_s = (win_start + win_end) * 0.5 * 1e-6
        ell_times.append(t_s)
        ell_ellipses_per_window.append(ellipses)

        # Visualize
        x, y, p = get_window_evlib(events, win_start, win_end)
        vis = pretty_event_frame_evlib(x, y, p, width, height)

        # Draw all detected ellipses
        for (cx, cy, a, b, phi) in ellipses:
            xs_ring, ys_ring = ellipse_points(cx, cy, a, b, phi, 360, width, height)
            for xi, yi in zip(xs_ring, ys_ring):
                vis[yi, xi] = (0, 255, 0)  # green

            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)  # red center

        cv2.imshow("Pass 1: Multi-ellipse detection", vis)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

        current_time += window_us
        frame_count += 1

    cv2.destroyAllWindows()

    if not ell_times:
        print("No geometry collected. Exiting.")
        return

    # Convert to arrays
    ell_times_arr = np.array(ell_times)

    print(f"Pass 1 complete: {len(ell_times)} frames with propellers detected")
    print(f"  Average propellers per frame: {np.mean([len(e) for e in ell_ellipses_per_window]):.1f}")

    # PASS 2: Per-propeller blade tracking (fine windows)
    print(f"\nPass 2: Blade tracking ({args.cluster_window_ms}ms windows)...")
    cluster_window_us = int(args.cluster_window_ms * 1000)

    # Track per-propeller data
    propeller_data: Dict[int, Dict[str, List[float]]] = {}

    current_time = t_min
    frame_count = 0

    while current_time < t_max:
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

        win_start = current_time
        win_end = min(current_time + cluster_window_us, t_max)

        x, y, p = get_window_evlib(events, win_start, win_end)
        if x.size == 0:
            current_time += cluster_window_us
            continue

        # Time of this window
        t_s = (win_start + win_end) * 0.5 * 1e-6

        # Pick closest ellipses from pass 1
        ellipses_t = pick_propellers_at_time(
            t_s, ell_times_arr, ell_ellipses_per_window
        )

        # Visualize: Create stacked view (top: events, bottom: warning overlay)
        vis = pretty_event_frame_evlib(x, y, p, width, height)

        # Create bottom half with dark background for overlay
        bottom_half = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)

        # Track each propeller separately
        for prop_idx, (cx_t, cy_t, a_t, b_t, phi_t) in enumerate(ellipses_t):
            # Cluster blades for this propeller
            centers = cluster_blades_dbscan_elliptic(
                x, y, cx_t, cy_t, a_t, b_t, phi_t,
                eps=args.dbscan_eps,
                min_samples=args.dbscan_min_samples,
                r_min=0.8,
                r_max=5.0,
            )

            # Draw ellipse on both halves
            xs_ring, ys_ring = ellipse_points(cx_t, cy_t, a_t, b_t, phi_t, 360, width, height)
            for xr, yr in zip(xs_ring, ys_ring):
                vis[yr, xr] = (0, 255, 0)  # green on top
                bottom_half[yr, xr] = (0, 255, 0)  # green on bottom

            cv2.circle(vis, (cx_t, cy_t), 5, (0, 0, 255), -1)
            cv2.circle(bottom_half, (cx_t, cy_t), 5, (0, 0, 255), -1)

            # Draw cluster centers and compute angle
            if centers:
                xc, yc = centers[0]  # Use largest cluster
                theta = np.arctan2(yc - cy_t, xc - cx_t)

                # Draw cluster center on both halves
                cv2.circle(vis, (int(round(xc)), int(round(yc))), 6, (255, 0, 0), 2)
                cv2.circle(bottom_half, (int(round(xc)), int(round(yc))), 6, (255, 0, 0), 2)

                # Track angle for this propeller
                if prop_idx not in propeller_data:
                    propeller_data[prop_idx] = {"times": [], "angles": []}
                propeller_data[prop_idx]["times"].append(t_s)
                propeller_data[prop_idx]["angles"].append(theta)

        # Add warning overlay to bottom half
        if ellipses_t:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Main warning text
            text_warn = "WARNING: DRONE DETECTED"
            cv2.putText(
                bottom_half,
                text_warn,
                (10, 30),
                font,
                font_scale,
                (0, 0, 255),  # red
                thickness,
                lineType=cv2.LINE_AA,
            )

            # Show RPM per propeller
            y_offset = 60
            for prop_idx in sorted(propeller_data.keys()):
                data = propeller_data[prop_idx]
                if len(data["times"]) >= 2:
                    # Estimate RPM using unwrap and polyfit
                    angles_unwrapped = np.unwrap(data["angles"])
                    times = np.array(data["times"])
                    coeffs = np.polyfit(times, angles_unwrapped, 1)
                    omega = coeffs[0]  # rad/s
                    rpm = (omega / (2.0 * np.pi)) * 60.0

                    text_rpm = f"Propeller {prop_idx}: {abs(rpm):.1f} RPM"
                    cv2.putText(
                        bottom_half,
                        text_rpm,
                        (10, y_offset),
                        font,
                        0.6,
                        (0, 255, 0),  # green
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    y_offset += 30

        # Stack top and bottom
        stacked = np.vstack([vis, bottom_half])
        cv2.imshow("Events + Propeller mask + Speed", stacked)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

        current_time += cluster_window_us
        frame_count += 1

    cv2.destroyAllWindows()

    # Final summary
    print("\nDetection complete!")
    print(f"Tracked {len(propeller_data)} propeller(s)")
    for prop_idx in sorted(propeller_data.keys()):
        data = propeller_data[prop_idx]
        if len(data["times"]) >= 2:
            angles_unwrapped = np.unwrap(data["angles"])
            times = np.array(data["times"])
            coeffs = np.polyfit(times, angles_unwrapped, 1)
            omega = coeffs[0]
            rpm = (omega / (2.0 * np.pi)) * 60.0
            print(f"  Propeller {prop_idx}: {abs(rpm):.1f} RPM (ω = {omega:.3f} rad/s)")


if __name__ == "__main__":
    main()
