"""Fan-specific ellipse fitting (single largest blob)."""

from typing import Optional, Tuple
import numpy as np
import cv2


def ellipse_from_frame(
    accum_frame: np.ndarray,
    prev_params: Optional[Tuple[int, int, float, float, float]] = None,
) -> Tuple[int, int, float, float, float, np.ndarray]:
    """Estimate single ellipse from accumulated frame.

    - Normalize & blur
    - Threshold to create mask
    - Find largest contour
    - Fit ellipse: center (cx, cy), axes (a, b), angle phi (rad)

    If fitting fails, fall back to prev_params (if any),
    otherwise return image center + small circle.

    Args:
        accum_frame: Grayscale accumulated frame
        prev_params: Previous ellipse parameters for fallback

    Returns:
        Tuple of (cx, cy, a, b, phi_rad, mask)
    """
    h, w = accum_frame.shape

    f = accum_frame.astype(np.float32)
    if f.max() > 0:
        img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        # No events at all
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Parameterize an ellipse for visualization.

    (x', y') = (a cos θ, b sin θ) in local coords
    (x, y)   = rotation by phi + translation (cx, cy)

    Args:
        cx, cy: Ellipse center
        a, b: Semi-major and semi-minor axes
        phi: Rotation angle (radians)
        n_pts: Number of points to generate
        width, height: Image bounds for clipping

    Returns:
        Tuple of (xs, ys) integer pixel indices
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
