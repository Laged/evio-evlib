"""Temporal geometry lookup utilities."""

from typing import List, Tuple
import numpy as np


def pick_geom_at_time(
    t: float,
    times: np.ndarray,
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    phi_arr: np.ndarray,
) -> Tuple[int, int, float, float, float]:
    """Pick ellipse geometry from pass 1 closest to time t.

    Args:
        t: Query time (seconds)
        times: Array of timestamps from pass 1
        cx_arr: Array of center X coordinates
        cy_arr: Array of center Y coordinates
        a_arr: Array of semi-major axes
        b_arr: Array of semi-minor axes
        phi_arr: Array of rotation angles

    Returns:
        Tuple of (cx, cy, a, b, phi) for closest time
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


def pick_propellers_at_time(
    t: float,
    times: np.ndarray,
    ellipses_per_window: List[List[Tuple[int, int, float, float, float]]],
) -> List[Tuple[int, int, float, float, float]]:
    """Pick list of propeller ellipses from pass 1 closest to time t.

    Args:
        t: Query time (seconds)
        times: Array of timestamps from pass 1
        ellipses_per_window: List of ellipse lists per window

    Returns:
        List of ellipses [(cx, cy, a, b, phi), ...] for closest time
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
