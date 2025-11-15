"""Multi-ellipse geometry detection for drone propellers.

Key differences from fan detector:
- Detects MULTIPLE ellipses (up to 2 propellers) instead of single largest
- Filters by HORIZONTAL orientation (rejects vertical ellipses)
- Returns LIST of ellipses instead of single ellipse
"""

from typing import List, Tuple
import numpy as np
import cv2


def propeller_mask_from_frame(
    accum_frame: np.ndarray,
    max_ellipses: int = 2,
    min_area: float = 145.0,
    max_area_frac: float = 0.01,
    horizontal_tolerance: float = 20.0,
) -> List[Tuple[int, int, float, float, float]]:
    """Detect multiple ellipses (propellers) from accumulated frame.

    Process:
    1. Normalize & threshold (Otsu)
    2. Morphological operations to clean noise
    3. Find ALL contours (not just largest)
    4. Fit ellipse to each valid contour
    5. Filter by HORIZONTAL orientation (drone propellers are horizontal)
    6. Sort by area and return top max_ellipses

    Args:
        accum_frame: Grayscale accumulated frame (h, w)
        max_ellipses: Maximum number of ellipses to detect (typically 2 for drones)
        min_area: Minimum contour area to consider
        max_area_frac: Maximum contour area as fraction of image size
        horizontal_tolerance: Angle tolerance for horizontal filter (degrees)
                            Keeps ellipses within ±tolerance of 0° or 180°

    Returns:
        List of (cx, cy, a, b, phi) for each detected propeller
        - cx, cy: Ellipse center (pixels)
        - a, b: Semi-major and semi-minor axes (pixels)
        - phi: Rotation angle (radians)
    """
    h, w = accum_frame.shape
    max_area = max_area_frac * (w * h)

    # Normalize and threshold
    f = accum_frame.astype(np.float32)
    if f.max() <= 0:
        return []

    img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, image_binary = cv2.threshold(img8, 250, 255, cv2.THRESH_BINARY)

    # Apply Otsu threshold
    _, mask = cv2.threshold(
        image_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find ALL contours (key difference from fan detector)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Collect valid ellipse candidates
    candidates: List[Tuple[int, int, float, float, float, float]] = []

    for cnt in contours:
        # fitEllipse needs at least 5 points, but use 50 to match legacy implementation
        # and prevent fitting ellipses to noise
        if len(cnt) < 50:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Fit ellipse to contour
        (cx_f, cy_f), (major, minor), angle_deg = cv2.fitEllipse(cnt)

        if minor <= 0:
            continue

        # ORIENTATION FILTER: Drone propellers are HORIZONTAL
        # angle_deg is in [0, 180) from OpenCV
        # We want ellipses near 0° or 180° (horizontal), NOT 90° (vertical)
        # Reject if too close to vertical (90°)
        #
        # NOTE: This fixes a bug in the legacy implementation which incorrectly
        # kept vertical ellipses (70-110°) but should keep horizontal (0°/180°)
        if 90.0 - horizontal_tolerance < angle_deg < 90.0 + horizontal_tolerance:
            continue  # Reject vertical-ish ellipses

        # Convert to standard ellipse parameters
        a = major * 0.5   # semi-major axis
        b = minor * 0.5   # semi-minor axis
        phi = np.deg2rad(angle_deg)

        cx_i = int(cx_f)
        cy_i = int(cy_f)

        # Store (cx, cy, a, b, phi, area) - keep area for sorting
        candidates.append((cx_i, cy_i, float(a), float(b), float(phi), float(area)))

    if not candidates:
        return []

    # Sort by area (largest first) and keep top max_ellipses
    candidates.sort(key=lambda t: t[5], reverse=True)
    top_candidates = candidates[:max_ellipses]

    # Return ellipse parameters (drop area)
    ellipses: List[Tuple[int, int, float, float, float]] = []
    for (cx_i, cy_i, a, b, phi, _) in top_candidates:
        ellipses.append((cx_i, cy_i, a, b, phi))

    return ellipses


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

    Generates points along the ellipse boundary using parametric form:
    - (x', y') = (a cos θ, b sin θ) in local coords
    - (x, y) = rotation by phi + translation (cx, cy)

    Args:
        cx, cy: Ellipse center (pixels)
        a, b: Semi-major and semi-minor axes (pixels)
        phi: Rotation angle (radians)
        n_pts: Number of points to generate
        width, height: Image bounds for clipping

    Returns:
        Tuple of (xs, ys) integer pixel indices, clipped to image bounds
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    # Local ellipse coordinates
    x_local = a * cos_t
    y_local = b * sin_t

    # Rotate and translate to global coordinates
    x = cx + x_local * cos_p - y_local * sin_p
    y = cy + x_local * sin_p + y_local * cos_p

    # Clip to image bounds
    xs = np.clip(np.round(x).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(y).astype(np.int32), 0, height - 1)
    return xs, ys
