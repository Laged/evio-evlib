"""Drone propeller detector demo - evlib version.

This is the evlib-migrated version of drone_detector_demo.py.
Demonstrates multi-propeller detection using detector-commons utilities.

Key differences from fan-rpm-demo:
- Detects MULTIPLE ellipses (up to 2 propellers) instead of single
- Uses pick_propellers_at_time (not pick_geom_at_time) for multi-ellipse lookup
- Tracks per-propeller RPM separately (propeller_data dict)
- Shows stacked view with warning overlay (NO matplotlib plots)

Known Limitations:
- Sparse datasets (e.g., drone_idle early frames) may fail detection
- Try --pre-threshold 50 --window-ms 60 for sparse data
- Try --skip-seconds N to find active segments
- See docs/DRONE_DETECTOR_DATASET_LIMITATIONS.md for details
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
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=0.0,
        help="Skip first N seconds of recording (useful for finding drone)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for geometry detection",
    )
    parser.add_argument(
        "--show-pass1",
        action="store_true",
        help="Show Pass 1 visualization window (hidden by default)",
    )
    parser.add_argument(
        "--pre-threshold",
        type=int,
        default=250,
        help="Pre-threshold for geometry detection (0-255). Default: 250 (original). HDF5 exports may need lower values (50-100).",
    )
    args = parser.parse_args()

    print(f"Loading {args.h5} with evlib...")
    events, width, height = load_legacy_h5(args.h5)
    print(f"Loaded {len(events):,} events, resolution: {width}x{height}")

    t_min, t_max = get_timestamp_range(events)
    window_us = int(args.window_ms * 1000)

    # Apply skip offset
    skip_us = int(args.skip_seconds * 1e6)
    t_min += skip_us
    if t_min >= t_max:
        print(f"Error: skip_seconds ({args.skip_seconds}s) exceeds recording duration")
        return

    # PASS 1: Collect multi-ellipse geometry (coarse windows)
    print(f"\nPass 1: Collecting geometry ({args.window_ms}ms windows)...")
    if args.skip_seconds > 0:
        print(f"Skipping first {args.skip_seconds}s of recording")
    ell_times = []
    ell_ellipses_per_window: List[List[Tuple[int, int, float, float, float]]] = []

    current_time = t_min
    frame_count = 0
    processed_count = 0  # Track how many frames we actually processed (vs. detected)

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

        # Match original: ALWAYS clip to uint8 (even in Pass 1)
        frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=True)

        # Detect multiple ellipses (propellers)
        # Use --debug flag to enable diagnostic output
        if args.debug and processed_count < 5:
            print(f"\n=== Frame {processed_count} (t={win_start}-{win_end} us) ===")

        # Detection parameters (tunable via CLI for dataset-specific characteristics)
        ellipses = propeller_mask_from_frame(
            frame_accum,
            max_ellipses=args.max_ellipses,
            pre_threshold=args.pre_threshold,  # CLI override for different datasets
            min_area=145.0,     # Original line 171: minimum contour size
            debug=args.debug and processed_count < 5,
        )

        processed_count += 1

        if not ellipses:
            # Skip frames where we didn't detect propellers
            current_time += window_us
            continue

        # Store geometry
        t_s = (win_start + win_end) * 0.5 * 1e-6
        ell_times.append(t_s)
        ell_ellipses_per_window.append(ellipses)

        # Optional visualization (use --show-pass1 to enable)
        if args.show_pass1:
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

    # Close Pass 1 window if it was shown
    if args.show_pass1:
        cv2.destroyAllWindows()

    if not ell_times:
        print(f"\nNo geometry collected. Exiting.")
        print(f"Processed {processed_count} frames, detected 0 propellers.")
        print(f"Check debug output above to see why detection failed.")
        return

    # Convert to arrays
    ell_times_arr = np.array(ell_times)

    print(f"\nPass 1 complete: {len(ell_times)} frames with propellers detected (out of {processed_count} processed)")
    print(f"  Detection rate: {len(ell_times)/processed_count*100:.1f}%")
    print(f"  Average propellers per frame: {np.mean([len(e) for e in ell_ellipses_per_window]):.1f}")

    # PASS 2: Per-propeller blade tracking (fine windows)
    print(f"\nPass 2: Blade tracking ({args.cluster_window_ms}ms windows)...")
    cluster_window_us = int(args.cluster_window_ms * 1000)

    # Track per-propeller data
    propeller_data: Dict[int, Dict[str, List[float]]] = {}

    # Track global RPM values for running mean
    rpm_values: List[float] = []

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

        # Create bottom half with accumulated frame as background
        # Get window events for accumulation
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
        # Build frame with uint8 clipping for visualization (matches original)
        frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=True)
        bottom_half = cv2.cvtColor(frame_accum, cv2.COLOR_GRAY2BGR)

        # Track each propeller separately
        for prop_idx, (cx_t, cy_t, a_t, b_t, phi_t) in enumerate(ellipses_t):
            # Cluster blades for this propeller (use HARDCODED params like original)
            centers = cluster_blades_dbscan_elliptic(
                x, y, cx_t, cy_t, a_t, b_t, phi_t,
                eps=5.0,  # HARDCODED (not args.dbscan_eps)
                min_samples=15,  # HARDCODED (not args.dbscan_min_samples)
                r_min=0.8,
                r_max=1.2,
            )

            # Draw ellipse on both halves using cv2.ellipse for smooth rendering
            cv2.ellipse(
                vis,
                (int(cx_t), int(cy_t)),
                (int(a_t), int(b_t)),
                np.rad2deg(phi_t),
                0,
                360,
                (0, 255, 0),  # green
                2,
            )
            cv2.ellipse(
                bottom_half,
                (int(cx_t), int(cy_t)),
                (int(a_t), int(b_t)),
                np.rad2deg(phi_t),
                0,
                360,
                (0, 255, 0),  # green
                2,
            )

            cv2.circle(vis, (int(cx_t), int(cy_t)), 4, (0, 0, 255), -1)
            cv2.circle(bottom_half, (int(cx_t), int(cy_t)), 4, (0, 0, 255), -1)

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

            # Main warning text with black background box
            text_warn = "WARNING: DRONE DETECTED"
            (tw1, th1), _ = cv2.getTextSize(text_warn, font, font_scale, thickness)
            # Match original lines 512-518: bottom-right positioning
            H, W = bottom_half.shape[:2]
            x1 = W - tw1 - 10
            y1 = H - 10
            # Draw black background box
            cv2.rectangle(
                bottom_half,
                (x1 - 5, y1 - th1 - 5),
                (x1 + tw1 + 5, y1 + 5),
                (0, 0, 0),
                thickness=-1,
            )
            # Draw text
            cv2.putText(
                bottom_half,
                text_warn,
                (x1, y1),
                font,
                font_scale,
                (0, 0, 255),  # red
                thickness,
                lineType=cv2.LINE_AA,
            )

            # Calculate RPM per propeller (using 2-point instantaneous velocity like original)
            rpm_frame: List[float] = []  # RPM values for this frame

            for prop_idx in sorted(propeller_data.keys()):
                data = propeller_data[prop_idx]
                if len(data["times"]) >= 2:
                    # Use last 2 measurements for instantaneous velocity
                    last_th = np.unwrap(np.array(data["angles"][-2:]))
                    last_t = np.array(data["times"][-2:])
                    dt = last_t[1] - last_t[0]
                    if dt > 0:
                        dtheta = last_th[1] - last_th[0]  # rad
                        omega = dtheta / dt  # rad/s
                        rpm = omega / (2.0 * np.pi) * 60.0  # RPM
                        rpm_abs = abs(rpm)

                        # Track for frame and global means
                        rpm_frame.append(rpm_abs)
                        rpm_values.append(rpm_abs)

            # Compute frame mean and global mean
            frame_mean_rpm = None
            global_mean_rpm = None
            if rpm_frame:
                frame_mean_rpm = float(np.mean(rpm_frame))
            if rpm_values:
                global_mean_rpm = float(np.mean(rpm_values))

            # Display frame mean RPM
            if frame_mean_rpm is not None:
                text_rpm = f"RPM: {frame_mean_rpm:5.1f}"  # Match original line 539
                (tw2, th2), _ = cv2.getTextSize(text_rpm, font, font_scale, thickness)
                x2 = W - tw2 - 10
                y2 = y1 - th1 - 10
                cv2.rectangle(
                    bottom_half,
                    (x2 - 5, y2 - th2 - 5),
                    (x2 + tw2 + 5, y2 + 5),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    bottom_half,
                    text_rpm,
                    (x2, y2),
                    font,
                    font_scale,
                    (0, 255, 0),  # green
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            # Display global mean RPM
            if global_mean_rpm is not None:
                text_mean = f"Avg RPM: {global_mean_rpm:5.1f}"  # Match original line 562
                (tw3, th3), _ = cv2.getTextSize(text_mean, font, font_scale, thickness)
                x3 = W - tw3 - 10
                y3 = y2 - th2 - 10
                cv2.rectangle(
                    bottom_half,
                    (x3 - 5, y3 - th3 - 5),
                    (x3 + tw3 + 5, y3 + 5),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    bottom_half,
                    text_mean,
                    (x3, y3),
                    font,
                    font_scale,
                    (0, 255, 0),  # green
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # Stack top and bottom and scale for display
        stacked = np.vstack([vis, bottom_half])
        scale = 0.7
        stacked_small = cv2.resize(stacked, None, fx=scale, fy=scale)
        cv2.imshow("Events + Propeller mask + Speed", stacked_small)

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
