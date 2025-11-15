#!/usr/bin/env python3
"""
Visualize per-frame ellipse fit of a rotating fan in an event-camera .dat file.

For each time window:
  1) Accumulate events into a grayscale frame (event count per pixel).
  2) Threshold that frame to create a binary mask (where the fan appears).
  3) Fit an ellipse to the largest blob in the mask -> center (cx, cy),
     semi-axes (a, b), and orientation phi.
  4) Draw that ellipse and center on a "pretty" event frame so you can
     visually check if the fit looks correct.
  5) Run DBSCAN on events near the blade radius to find cluster centers.
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from evio.source.dat_file import DatFileSource


# -------------------------------------------------------------------
#  Helpers to decode events and build frames
# -------------------------------------------------------------------

def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw words in [win_start, win_stop) into x, y, polarity arrays."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)

    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def build_accum_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """Build a grayscale frame: event count per pixel in the window."""
    x_coords, y_coords, _ = window
    frame = np.zeros((height, width), dtype=np.uint16)
    frame[y_coords, x_coords] += 1
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def pretty_event_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """Nice gray/white/black event visualization frame (for debugging)."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), (127, 127, 127), np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = (255, 255, 255)
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = (0, 0, 0)
    return frame

def cluster_blades_dbscan_elliptic(
    x: np.ndarray,
    y: np.ndarray,
    cx: int,
    cy: int,
    a: float,
    b: float,
    phi: float,
    eps: float = 5.0,
    min_samples: int = 10,
    r_min: float = 0.8,
    r_max: float = 1.2,
) -> list[Tuple[float, float]]:
    """
    Cluster events near the blade *ellipse* using DBSCAN.

    Uses elliptical radius:
      - translate by (cx, cy)
      - rotate by -phi
      - normalize by (a, b)
      r_ell = sqrt((x'/a)^2 + (y'/b)^2)

    Keeps points with r_ell in [r_min, r_max] and clusters them.
    """
    if a <= 0 or b <= 0:
        return []

    # translate
    dx = x.astype(np.float32) - float(cx)
    dy = y.astype(np.float32) - float(cy)

    # rotate into ellipse-aligned frame
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    x_rot =  dx * cos_p + dy * sin_p
    y_rot = -dx * sin_p + dy * cos_p

    # elliptical radius
    r_ell = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)

    mask = (r_ell >= r_min) & (r_ell <= r_max)
    if not np.any(mask):
        return []

    pts = np.column_stack([x[mask], y[mask]])
    if pts.shape[0] < min_samples:
        return []

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pts)

    unique_labels = [lab for lab in np.unique(labels) if lab != -1]
    if not unique_labels:
        return []

    clusters = []
    for lab in unique_labels:
        pts_lab = pts[labels == lab]
        if pts_lab.shape[0] == 0:
            continue
        xc = pts_lab[:, 0].mean()
        yc = pts_lab[:, 1].mean()
        clusters.append((xc, yc, pts_lab.shape[0]))

    clusters.sort(key=lambda t: t[2], reverse=True)
    return [(xc, yc) for (xc, yc, n) in clusters[:3]]

# -------------------------------------------------------------------
#  Ellipse fitting from a single accumulated frame
# -------------------------------------------------------------------

def ellipse_from_frame(
    accum_frame: np.ndarray,
    prev_params: Optional[Tuple[int, int, float, float, float]] = None,
) -> tuple[int, int, float, float, float, np.ndarray]:
    """
    Estimate an ellipse from a single accumulated frame:
      - Normalize & blur
      - Threshold to create a mask
      - Find largest contour
      - Fit an ellipse: center (cx, cy), axes (a, b), angle phi (rad)

    If fitting fails, fall back to prev_params (if any),
    otherwise return image center + small circle.

    Returns: (cx, cy, a, b, phi_rad, mask)
    """
    h, w = accum_frame.shape

    f = accum_frame.astype(np.float32)
    if f.max() > 0:
        img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        # no events at all
        if prev_params is not None:
            cx_prev, cy_prev, a_prev, b_prev, phi_prev = prev_params
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return cx_prev, cy_prev, a_prev, b_prev, phi_prev, empty_mask
        cx0, cy0 = w // 2, h // 2
        r0 = min(w, h) * 0.25
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return cx0, cy0, r0, r0, 0.0, empty_mask

    img_blur = cv2.GaussianBlur(img8, (5, 5), 0)

    _, mask = cv2.threshold(
        img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        if prev_params is not None:
            cx_prev, cy_prev, a_prev, b_prev, phi_prev = prev_params
            return cx_prev, cy_prev, a_prev, b_prev, phi_prev, mask
        cx0, cy0 = w // 2, h // 2
        r0 = min(w, h) * 0.25
        return cx0, cy0, r0, r0, 0.0, mask

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        if prev_params is not None:
            cx_prev, cy_prev, a_prev, b_prev, phi_prev = prev_params
            return cx_prev, cy_prev, a_prev, b_prev, phi_prev, mask
        cx0, cy0 = w // 2, h // 2
        r0 = min(w, h) * 0.25
        return cx0, cy0, r0, r0, 0.0, mask

    (cx_f, cy_f), (major, minor), angle_deg = cv2.fitEllipse(cnt)
    a = major * 0.5  # semi-major
    b = minor * 0.5  # semi-minor
    phi = np.deg2rad(angle_deg)

    return int(cx_f), int(cy_f), float(a), float(b), float(phi), mask


def ellipse_points(
    cx: int,
    cy: int,
    a: float,
    b: float,
    phi: float,
    n_pts: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameterize an ellipse:
        (x', y') = (a cos θ, b sin θ) in local coords
        (x, y)   = rotation by phi + translation (cx, cy)

    Returns xs, ys (integer pixel indices).
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    x_local = a * cos_t
    y_local = b * sin_t

    x = cx + x_local * cos_p - y_local * sin_p
    y = cy + x_local * sin_p + y_local * cos_p

    xs = np.clip(np.round(x).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(y).astype(np.int32), 0, height - 1)
    return xs, ys

# -------------------------------------------------------------------
#  Helper: pick ellipse params closest in time
# -------------------------------------------------------------------

def pick_geom_at_time(
    t: float,
    times: np.ndarray,
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    phi_arr: np.ndarray,
) -> tuple[int, int, float, float, float]:
    """
    Given a time t, pick the ellipse geometry from pass 1
    whose timestamp is closest to t.
    """
    if times.size == 0:
        raise RuntimeError("No ellipse geometry stored from pass 1")

    idx = np.searchsorted(times, t)
    if idx == 0:
        j = 0
    elif idx >= times.size:
        j = times.size - 1
    else:
        if abs(times[idx] - t) < abs(times[idx - 1] - t):
            j = idx
        else:
            j = idx - 1

    return (
        int(cx_arr[j]),
        int(cy_arr[j]),
        float(a_arr[j]),
        float(b_arr[j]),
        float(phi_arr[j]),
    )

# -------------------------------------------------------------------
#  Main: visualize per-frame ellipse on real event frames
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat event file")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=30.0,
        help="Window duration in ms for event accumulation (default: 2 ms).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit on number of windows to show (0 = no limit).",
    )
    parser.add_argument(
        "--cluster-window-ms",
        type=float,
        default=0.5,
        help="Window duration in ms for DBSCAN cluster visualization.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=10.0,
        help="DBSCAN eps (cluster radius in pixels).",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=15,
        help="DBSCAN min_samples (minimum points per cluster).",
    )
    args = parser.parse_args()

    width, height = 1280, 720
    window_us = args.window_ms * 1000.0

    src = DatFileSource(
        args.dat,
        width=width,
        height=height,
        window_length_us=window_us,
    )

    prev_ellipse: Optional[Tuple[int, int, float, float, float]] = None
    n_samples = 360

    ell_times = []
    ell_cx = []
    ell_cy = []
    ell_r = []
    ell_a = []
    ell_b = []
    ell_phi = []


    for i, batch_range in enumerate(src.ranges()):
        if args.max_frames > 0 and i >= args.max_frames:
            break

        # 1) Decode events for this time window
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        # 2) Accumulated grayscale frame (for mask & ellipse fit)
        frame_accum = build_accum_frame(window, width, height)

        # 3) Fit ellipse from this frame's mask
        cx, cy, a, b, phi, mask = ellipse_from_frame(frame_accum, prev_params=prev_ellipse)
        prev_ellipse = (cx, cy, a, b, phi)

        r_pix = max(a, b)

        # store geometry with mid timestamp (in seconds)
        t_us = 0.5 * (batch_range.start_ts_us + batch_range.end_ts_us)
        t_s = t_us * 1e-6
        ell_times.append(t_s)
        ell_cx.append(cx)
        ell_cy.append(cy)
        ell_r.append(r_pix)
        ell_a.append(a)
        ell_b.append(b)
        ell_phi.append(phi)

        # 4) Create "real" event frame and draw ellipse + center on it
        vis = pretty_event_frame(window, width, height)

        xs_ring, ys_ring = ellipse_points(cx, cy, a, b, phi,
                                          n_pts=n_samples,
                                          width=width,
                                          height=height)
        # draw ellipse
        for x, y in zip(xs_ring, ys_ring):
            vis[y, x] = (0, 255, 0)  # green

        # draw center
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

        # optional: show mask as well in a tiny window
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(mask_color, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Ellipse on events", vis)
        cv2.imshow("Mask", mask_color)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or 'q' to quit
            break

    cv2.destroyAllWindows()

    ell_times = np.array(ell_times)
    ell_cx = np.array(ell_cx)
    ell_cy = np.array(ell_cy)
    ell_r = np.array(ell_r)
    ell_a = np.array(ell_a)
    ell_b = np.array(ell_b)
    ell_phi = np.array(ell_phi)


    # ==========================
    # Stage 2: CLUSTERS (fast ms)
    # ==========================
    cluster_window_us = args.cluster_window_ms * 1000.0

    src2 = DatFileSource(
        args.dat,
        width=width,
        height=height,
        window_length_us=cluster_window_us,
    )

    times_small = []
    angles_tracked = []
    prev_angle = None  # will track angle of one blade over time

    for i, batch_range in enumerate(src2.ranges()):
        if args.max_frames > 0 and i >= args.max_frames:
            break

        x_coords, y_coords, polarities_on = get_window(
            src2.event_words,
            src2.order,
            batch_range.start,
            batch_range.stop,
        )
        if x_coords.size == 0:
            continue

        # time of this small window
        t_us = 0.5 * (batch_range.start_ts_us + batch_range.end_ts_us)
        t_s = t_us * 1e-6

        # pick closest ellipse geometry from pass 1
        cx_t, cy_t, a_t, b_t, phi_t = pick_geom_at_time(
            t_s, ell_times, ell_cx, ell_cy, ell_a, ell_b, ell_phi
        )

        centers = cluster_blades_dbscan_elliptic(
            x_coords,
            y_coords,
            cx_t,
            cy_t,
            a_t,
            b_t,
            phi_t,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            r_min=0.8,
            r_max=5.0,
        )

        vis = pretty_event_frame((x_coords, y_coords, polarities_on), width, height)

        # center
        cv2.circle(vis, (cx_t, cy_t), 5, (0, 0, 255), -1)

        # ellipse from pass 1
        xs_ring, ys_ring = ellipse_points(
            cx_t, cy_t, a_t, b_t, phi_t,
            n_pts=360,
            width=width,
            height=height,
        )
        for xr, yr in zip(xs_ring, ys_ring):
            vis[yr, xr] = (0, 255, 0)  # green

        # draw cluster centers (blue) and compute angles
        blade_angles = []

        # clusters
        for (xc, yc) in centers:
            cv2.circle(vis, (int(round(xc)), int(round(yc))), 6, (255, 0, 0), 2)
            # angle of this cluster around center
            theta = np.arctan2(yc - cy_t, xc - cx_t)
            blade_angles.append(theta)

        cv2.imshow("DBSCAN clusters (fast window)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        # ---- track one blade angle over time ----
        if not blade_angles:
            continue  # no clusters this window

        blade_angles = np.array(blade_angles)

        if prev_angle is None:
            # first frame: just pick any (e.g. the first)
            chosen_theta = blade_angles[0]
        else:
            # pick cluster whose angle is closest (mod 2π) to previous
            # compute wrapped differences to prev_angle in [-pi, pi]
            diffs = np.arctan2(
                np.sin(blade_angles - prev_angle),
                np.cos(blade_angles - prev_angle),
            )
            idx_best = np.argmin(np.abs(diffs))
            chosen_theta = blade_angles[idx_best]

        prev_angle = chosen_theta
        times_small.append(t_s)
        angles_tracked.append(chosen_theta)

    cv2.destroyAllWindows()

    # ==========================
    # Angular velocity estimation
    # ==========================
    if len(times_small) < 2:
        print("Not enough data to estimate velocity.")
        return

    times_small = np.array(times_small)
    angles_tracked = np.array(angles_tracked)

    # sort by time, just in case
    order = np.argsort(times_small)
    times_small = times_small[order]
    angles_tracked = angles_tracked[order]

    # unwrap angle over time
    angles_unwrapped = np.unwrap(angles_tracked)

    # fit line: angle(t) ≈ omega * t + phi
    coeffs = np.polyfit(times_small, angles_unwrapped, 1)
    omega = coeffs[0]  # rad/s
    phi0 = coeffs[1]

    rot_per_sec = omega / (2.0 * np.pi)
    rpm = rot_per_sec * 60.0

    print("\nEstimated mean angular velocity from blade tracking:")
    print(f"  ω ≈ {omega:.3f} rad/s")
    print(f"  ≈ {rot_per_sec:.3f} rotations/s")
    print(f"  ≈ {rpm:.1f} RPM")

    # optional: instantaneous angular velocity
    omega_inst = np.gradient(angles_unwrapped, times_small)

    # optional plots
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
