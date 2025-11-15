#!/usr/bin/env python3
"""

"""

import argparse
from typing import Optional, Tuple, List

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from evio.source.dat_file import DatFileSource
import math


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

def propeller_mask_from_frame(
    accum_frame: np.ndarray,
    min_area: float = 145.0,
    max_area_frac: float = 0.01,
    top_k: int = 2,
    max_pair_distance: float = 15.0,   # max distance (in px) to consider them "close"
) -> tuple[List[Tuple[int, int, float, float, float]], np.ndarray]:
    """
    Detect multiple 'propeller-like' blobs in an accumulated frame and
    return:
        - a list of ellipse params for each blob: (cx, cy, a, b, phi_rad)
        - a binary mask where those ellipses are filled with 255.
    """
    h, w = accum_frame.shape
    prop_mask = np.zeros((h, w), dtype=np.uint8)
    # (cx, cy, a, b, phi, area)
    candidates: List[Tuple[int, int, float, float, float, float]] = []

    f = accum_frame.astype(np.float32)
    if f.max() <= 0:
        return [], prop_mask

    img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, image_binary = cv2.threshold(img8, 250, 255, cv2.THRESH_BINARY)


    # --- threshold (Otsu) ---
    _, mask = cv2.threshold(
        image_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- morphology to clean up noise ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- find all contours, not just the largest ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], prop_mask

    max_area = max_area_frac * (w * h)

    for cnt in contours:
        if len(cnt) < 50:
            continue  # fitEllipse needs at least 50 points

        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        (cx_f, cy_f), (major, minor), angle_deg = cv2.fitEllipse(cnt)

        if minor <= 0:
            continue

        # angle_deg is in [0, 180)
        # propellers are typically near 0 deg or 180 deg (horizontal), not 90 deg (vertical)
        if 70 > angle_deg:
            continue   # reject vertical-ish ellipses
        if 110 < angle_deg:
            continue   # reject vertical-ish ellipses

        a = major * 0.5
        b = minor * 0.5
        phi = np.deg2rad(angle_deg)

        cx_i = int(cx_f)
        cy_i = int(cy_f)

        candidates.append((cx_i, cy_i, float(a), float(b), float(phi), float(area)))

    # sort by area and keep only top_k
    candidates.sort(key=lambda t: t[5], reverse=True)
    candidates = candidates[:top_k]

    if not candidates:
        return [], prop_mask

    # decide which ellipses to keep based on distance
    selected: List[Tuple[int, int, float, float, float, float]] = []

    if len(candidates) == 1 or top_k == 1:
        selected = [candidates[0]]
    else:
        c0 = candidates[0]
        c1 = candidates[1]
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        dist = math.hypot(dx, dy)

        if dist <= max_pair_distance:
            selected = [c0, c1]
        else:
            selected = [c0]

    ellipses: List[Tuple[int, int, float, float, float]] = []

    for (cx_i, cy_i, a, b, phi, area) in candidates:
        ellipses.append((cx_i, cy_i, a, b, phi))

        cv2.ellipse(
            prop_mask,
            (cx_i, cy_i),
            (int(a), int(b)),
            np.rad2deg(phi),
            0,
            360,
            255,
            thickness=-1,
        )

    return ellipses, prop_mask

def pass1_collect_geometry(
    dat_path: str,
    width: int,
    height: int,
    window_ms: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    First pass:
      - use a coarse window (e.g. 30 ms)
      - detect propellers
      - store, for each window where we see props:
          * mid timestamp (seconds)
          * list of ellipses for that window
    Returns:
      times (np.ndarray of shape [N]),
      ellipses_per_window: list of length N,
        each entry is a list of (cx, cy, a, b, phi).
    """
    window_us = window_ms * 1000.0

    src = DatFileSource(
        dat_path,
        width=width,
        height=height,
        window_length_us=window_us,
    )

    times: List[float] = []
    ellipses_per_window: List[List[Tuple[int, int, float, float, float]]] = []

    for i, batch_range in enumerate(src.ranges()):
        # 1) Decode events for this time window
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        # 2) Accumulated grayscale frame (for mask & prop detection)
        frame_accum = build_accum_frame(window, width, height)

        # 3) Detect propellers (coarse but robust)
        prop_ellipses, _ = propeller_mask_from_frame(frame_accum)

        if len(prop_ellipses) == 0:
            # skip frames where we didn't see the props clearly
            continue

        # store as an "ellipse": center (cx,cy), a=b=r, phi=0
        t_us = 0.5 * (batch_range.start_ts_us + batch_range.end_ts_us)
        t_s = t_us * 1e-6

        times.append(t_s)
        ellipses_per_window.append(prop_ellipses)


    if not times:
        raise RuntimeError("Pass 1: no geometry collected (no propellers detected).")

    return np.array(times, dtype=np.float32), ellipses_per_window

def pick_propellers_at_time(
    t: float,
    times: np.ndarray,
    ellipses_per_window: List[List[Tuple[int, int, float, float, float]]],
) -> List[Tuple[int, int, float, float, float]]:
    """
    Given a time t, return the list of ellipses from pass 1
    whose timestamp is closest to t.
    """
    if times.size == 0:
        return []

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

    return ellipses_per_window[j]

def blade_angle_for_propeller(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: Tuple[int, int, float, float, float],
    eps: float = 5.0,
    min_samples: int = 15,
    r_min: float = 0.8,
    r_max: float = 1.2,
) -> Optional[float]:
    """
    From a given fine-window of events and a rotor geometry (cx,cy,a,b,phi),
    estimate the blade angle (in radians) using the events near the ring.
    """
    x, y, _ = window
    cx, cy, a, b, phi = geom

    clusters = cluster_blades_dbscan_elliptic(
        x, y, cx, cy, a, b, phi,
        eps=eps,
        min_samples=min_samples,
        r_min=r_min,
        r_max=r_max,
    )
    if not clusters:
        return None

    xc, yc = clusters[0]
    dx = xc - cx
    dy = yc - cy
    theta = np.arctan2(dy, dx)   # [-pi, pi]
    return float(theta)


def pass2_estimate_velocity(
    dat_path: str,
    width: int,
    height: int,
    window_ms: float,
    times: np.ndarray,
    ellipses_per_window: List[List[Tuple[int, int, float, float, float]]],
) -> None:
    """
    Second pass:
      - use a fine window (e.g. 2 ms)
      - for each window:
          * get propeller ellipses from pass 1 closest in time
          * for each propeller, estimate blade angle (DBSCAN on ring)
          * keep angle history per propeller, compute angular velocity
          * show mean RPM + warning
      - visualize: events (top) + overlay (bottom).
    """
    window_us = window_ms * 1000.0
    src = DatFileSource(
        dat_path,
        width=width,
        height=height,
        window_length_us=window_us,
    )

    # up to 2 propellers → two histories
    angle_hist_per_prop: List[List[float]] = [[], []]
    time_hist_per_prop: List[List[float]] = [[], []]

    rpm_values: List[float] = []

    for i, batch_range in enumerate(src.ranges()):
        # decode events
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )


        # mid timestamp
        t_us = 0.5 * (batch_range.start_ts_us + batch_range.end_ts_us)
        t_s = t_us * 1e-6

        # get the propellers from pass 1 closest in time
        prop_ellipses = pick_propellers_at_time(t_s, times, ellipses_per_window)

        # fine-window accumulated frame
        frame_accum = build_accum_frame(window, width, height)
        # convert to color for drawing colored stuff
        overlay = cv2.cvtColor(frame_accum, cv2.COLOR_GRAY2BGR)

        rpm_frame: List[float] = []

        # --- estimate angle & RPM per propeller ---
        for idx, (cx, cy, a, b, phi) in enumerate(prop_ellipses[:2]):  # max 2
            theta = blade_angle_for_propeller(
                window,
                (cx, cy, a, b, phi),
                eps=5.0,
                min_samples=15,
                r_min=0.8,
                r_max=1.2,
            )

            if theta is None:
                continue

            angle_hist_per_prop[idx].append(theta)
            time_hist_per_prop[idx].append(t_s)

            if len(angle_hist_per_prop[idx]) >= 2:
                last_th = np.unwrap(np.array(angle_hist_per_prop[idx][-2:]))
                last_t = np.array(time_hist_per_prop[idx][-2:])
                dt = last_t[1] - last_t[0]
                if dt > 0:
                    dtheta = last_th[1] - last_th[0]   # rad
                    omega = dtheta / dt                 # rad/s
                    rpm = omega / (2.0 * np.pi) * 60.0  # RPM
                    rpm_abs = abs(rpm)
                    rpm_frame.append(rpm_abs)   # for this frame
                    rpm_values.append(rpm_abs)  # global history

        frame_mean_rpm = None
        global_mean_rpm = None
        if rpm_frame:
            frame_mean_rpm = float(np.mean(rpm_frame))   # current frame mean over props
        if rpm_values:
            global_mean_rpm = float(np.mean(rpm_values)) # running mean since first

        # draw ellipses and centers from pass 1
        for (cx, cy, a, b, phi) in prop_ellipses:
            cv2.ellipse(
                overlay,
                (int(cx), int(cy)),
                (int(a), int(b)),
                np.rad2deg(phi),
                0,
                360,
                (0, 255, 0),   # green ellipse
                2,
            )
            cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # if we have any props, add a warning text
        if prop_ellipses:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            H, W = frame_accum.shape[:2]

            # warning text
            text1 = "WARNING: DRONE DETECTED"
            (tw1, th1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            x1 = W - tw1 - 10
            y1 = H - 10
            cv2.rectangle(
                overlay,
                (x1 - 5, y1 - th1 - 5),
                (x1 + tw1 + 5, y1 + 5),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                overlay,
                text1,
                (x1, y1),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

            # RPM text if we have it
            if frame_mean_rpm is not None:
                text_rpm = f"RPM: {frame_mean_rpm:5.1f}"
                (tw2, th2), _ = cv2.getTextSize(text_rpm, font, font_scale, thickness)
                x2 = W - tw2 - 10
                y2 = y1 - th1 - 10
                cv2.rectangle(
                    overlay,
                    (x2 - 5, y2 - th2 - 5),
                    (x2 + tw2 + 5, y2 + 5),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    overlay,
                    text_rpm,
                    (x2, y2),
                    font,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            if global_mean_rpm is not None:
                text_mean = f"Avg RPM: {global_mean_rpm:5.1f}"
                (tw3, th3), _ = cv2.getTextSize(text_mean, font, font_scale, thickness)
                x3 = W - tw3 - 10
                y3 = y2 - th3 - 10
                cv2.rectangle(
                    overlay,
                    (x3 - 5, y3 - th3 - 5),
                    (x3 + tw3 + 5, y3 + 5),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    overlay,
                    text_mean,
                    (x3, y3),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # events vis
        vis = pretty_event_frame(window, width, height)
        combined = np.vstack([vis, overlay])
        scale = 0.7
        combined_small = cv2.resize(combined, None, fx=scale, fy=scale)
        cv2.imshow("Events + Propeller mask + Speed", combined_small)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()

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
    # --- PASS 1: collect rotor geometry ---
    (
        times,
        ellipses_per_window,
    ) = pass1_collect_geometry(
        args.dat,
        width=width,
        height=height,
        window_ms=args.window_ms,
    )

    # --- PASS 2: estimate velocity using fine windows + pass1 geometry ---
    pass2_estimate_velocity(
        args.dat,
        width=width,
        height=height,
        window_ms=args.cluster_window_ms,
        times=times,
        ellipses_per_window=ellipses_per_window,
    )

if __name__ == "__main__":
    main()
