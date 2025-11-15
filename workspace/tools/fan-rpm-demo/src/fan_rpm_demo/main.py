"""Fan RPM detector demo - evlib version.

This is the evlib-migrated version of fan-example-detector.py.
Demonstrates speedup using detector-commons utilities.
"""

import argparse
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import polars as pl

from detector_commons import (
    load_legacy_h5,
    get_window_evlib,
    get_timestamp_range,
    build_accum_frame_evlib,
    pretty_event_frame_evlib,
    cluster_blades_dbscan_elliptic,
    pick_geom_at_time,
)
from .geometry import ellipse_from_frame, ellipse_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fan RPM detector using evlib"
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
    args = parser.parse_args()

    print(f"Loading {args.h5} with evlib...")
    events, width, height = load_legacy_h5(args.h5)
    print(f"Loaded {len(events):,} events, resolution: {width}x{height}")

    t_min, t_max = get_timestamp_range(events)
    window_us = int(args.window_ms * 1000)

    # PASS 1: Collect ellipse geometry (coarse windows)
    print(f"\nPass 1: Collecting geometry ({args.window_ms}ms windows)...")
    ell_times = []
    ell_cx = []
    ell_cy = []
    ell_a = []
    ell_b = []
    ell_phi = []

    prev_ellipse: Optional[Tuple[int, int, float, float, float]] = None
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

        # Fit ellipse
        cx, cy, a, b, phi, mask = ellipse_from_frame(frame_accum, prev_params=prev_ellipse)
        prev_ellipse = (cx, cy, a, b, phi)

        # Store geometry
        t_s = (win_start + win_end) * 0.5 * 1e-6
        ell_times.append(t_s)
        ell_cx.append(cx)
        ell_cy.append(cy)
        ell_a.append(a)
        ell_b.append(b)
        ell_phi.append(phi)

        # Visualize
        x, y, p = get_window_evlib(events, win_start, win_end)
        vis = pretty_event_frame_evlib(x, y, p, width, height)

        # Draw ellipse
        xs_ring, ys_ring = ellipse_points(cx, cy, a, b, phi, 360, width, height)
        for xi, yi in zip(xs_ring, ys_ring):
            vis[yi, xi] = (0, 255, 0)  # green

        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)  # red center

        # Show mask too
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(mask_color, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Ellipse on events", vis)
        cv2.imshow("Mask", mask_color)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

        current_time += window_us
        frame_count += 1

    cv2.destroyAllWindows()

    if not ell_times:
        print("No geometry collected. Exiting.")
        return

    # Convert to arrays
    ell_times = np.array(ell_times)
    ell_cx = np.array(ell_cx)
    ell_cy = np.array(ell_cy)
    ell_a = np.array(ell_a)
    ell_b = np.array(ell_b)
    ell_phi = np.array(ell_phi)

    print(f"Pass 1 complete: {len(ell_times)} frames")

    # PASS 2: Blade tracking (fine windows)
    print(f"\nPass 2: Blade tracking ({args.cluster_window_ms}ms windows)...")
    cluster_window_us = int(args.cluster_window_ms * 1000)

    times_small = []
    angles_tracked = []
    prev_angle = None

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

        # Pick closest ellipse from pass 1
        cx_t, cy_t, a_t, b_t, phi_t = pick_geom_at_time(
            t_s, ell_times, ell_cx, ell_cy, ell_a, ell_b, ell_phi
        )

        # Cluster blades
        centers = cluster_blades_dbscan_elliptic(
            x, y, cx_t, cy_t, a_t, b_t, phi_t,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            r_min=0.8,
            r_max=5.0,
        )

        # Visualize
        vis = pretty_event_frame_evlib(x, y, p, width, height)
        cv2.circle(vis, (cx_t, cy_t), 5, (0, 0, 255), -1)

        # Draw ellipse
        xs_ring, ys_ring = ellipse_points(cx_t, cy_t, a_t, b_t, phi_t, 360, width, height)
        for xr, yr in zip(xs_ring, ys_ring):
            vis[yr, xr] = (0, 255, 0)

        # Draw cluster centers and compute angles
        blade_angles = []
        for (xc, yc) in centers:
            cv2.circle(vis, (int(round(xc)), int(round(yc))), 6, (255, 0, 0), 2)
            theta = np.arctan2(yc - cy_t, xc - cx_t)
            blade_angles.append(theta)

        cv2.imshow("DBSCAN clusters (fast window)", vis)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

        # Track one blade angle
        if blade_angles:
            blade_angles = np.array(blade_angles)

            if prev_angle is None:
                chosen_theta = blade_angles[0]
            else:
                # Pick closest to previous
                diffs = np.arctan2(
                    np.sin(blade_angles - prev_angle),
                    np.cos(blade_angles - prev_angle),
                )
                idx_best = np.argmin(np.abs(diffs))
                chosen_theta = blade_angles[idx_best]

            prev_angle = chosen_theta
            times_small.append(t_s)
            angles_tracked.append(chosen_theta)

        current_time += cluster_window_us
        frame_count += 1

    cv2.destroyAllWindows()

    # Estimate angular velocity
    if len(times_small) < 2:
        print("Not enough data to estimate velocity.")
        return

    times_small = np.array(times_small)
    angles_tracked = np.array(angles_tracked)

    # Sort by time
    order = np.argsort(times_small)
    times_small = times_small[order]
    angles_tracked = angles_tracked[order]

    # Unwrap angles
    angles_unwrapped = np.unwrap(angles_tracked)

    # Fit line: angle(t) ≈ omega * t + phi
    coeffs = np.polyfit(times_small, angles_unwrapped, 1)
    omega = coeffs[0]  # rad/s
    phi0 = coeffs[1]

    rot_per_sec = omega / (2.0 * np.pi)
    rpm = rot_per_sec * 60.0

    print("\nEstimated mean angular velocity from blade tracking:")
    print(f"  ω ≈ {omega:.3f} rad/s")
    print(f"  ≈ {rot_per_sec:.3f} rotations/s")
    print(f"  ≈ {rpm:.1f} RPM")

    # Instantaneous velocity
    omega_inst = np.gradient(angles_unwrapped, times_small)

    # Plots
    plt.figure()
    plt.plot(times_small, angles_unwrapped)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad, unwrapped)")
    plt.title("Tracked blade angle vs time")
    plt.grid(True)

    plt.figure()
    plt.plot(times_small, omega_inst, label="ω(t)")
    plt.axhline(omega, linestyle="--", label=f"mean ω ≈ {omega:.2f} rad/s")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("Blade angular velocity vs time")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
