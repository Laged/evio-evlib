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
    max_pair_distance: float = 15.0,
    pre_threshold: int = 250,
    hot_pixel_threshold: float = float('inf'),  # Match original: no hot pixel filtering
    debug: bool = False,
) -> List[Tuple[int, int, float, float, float]]:
    """Detect multiple ellipses (propellers) from accumulated frame.

    Process:
    0. Filter hot pixels (pixels with abnormally high counts)
    1. Normalize & threshold (pre-threshold at 250, then Otsu)
    2. Morphological operations to clean noise
    3. Find ALL contours (not just largest)
    4. Fit ellipse to each valid contour
    5. Filter by orientation (keeps angles in [70, 110] degrees to match legacy)
    6. Sort by area and return top max_ellipses
    7. Apply pair distance filter: if top 2 are >15px apart, keep only largest

    Args:
        accum_frame: Grayscale accumulated frame (h, w)
        max_ellipses: Maximum number of ellipses to detect (typically 2 for drones)
        min_area: Minimum contour area to consider
        max_area_frac: Maximum contour area as fraction of image size
        horizontal_tolerance: DEPRECATED - not used (kept for API compatibility)
        max_pair_distance: Maximum distance between top 2 ellipses (pixels, default 15.0)
        hot_pixel_threshold: Pixels with count > this are considered hot pixels (default: 10000)
        debug: If True, print diagnostic information

    Returns:
        List of (cx, cy, a, b, phi) for each detected propeller
        - cx, cy: Ellipse center (pixels)
        - a, b: Semi-major and semi-minor axes (pixels)
        - phi: Rotation angle (radians)
    """
    h, w = accum_frame.shape
    max_area = max_area_frac * (w * h)

    if debug:
        print(f"\n[DEBUG] Frame shape: {h}x{w}")
        print(f"[DEBUG] Max area threshold: {max_area:.1f} px")
        print(f"[DEBUG] Min area threshold: {min_area:.1f} px")

    # Normalize and threshold
    f = accum_frame.astype(np.float32)
    if f.max() <= 0:
        if debug:
            print("[DEBUG] Frame is empty (max <= 0)")
        return []

    # FILTER HOT PIXELS: Set to 0 before normalization
    hot_pixel_mask = f > hot_pixel_threshold
    num_hot = np.count_nonzero(hot_pixel_mask)
    if num_hot > 0:
        if debug:
            print(f"[DEBUG] Filtering {num_hot} hot pixels (>{hot_pixel_threshold:.0f} counts)")
            hot_locs = np.where(hot_pixel_mask)
            for i in range(min(3, num_hot)):
                y, x = hot_locs[0][i], hot_locs[1][i]
                print(f"[DEBUG]   Hot pixel ({x},{y}): {f[y,x]:.0f} events -> 0")
        f[hot_pixel_mask] = 0

    if debug:
        print(f"[DEBUG] Raw frame range: [{f.min():.1f}, {f.max():.1f}]")
        print(f"[DEBUG] Non-zero pixels in raw frame: {np.count_nonzero(f)}")
        print(f"[DEBUG] Raw frame value distribution:")
        print(f"[DEBUG]   >0: {np.count_nonzero(f > 0)}")
        print(f"[DEBUG]   >10: {np.count_nonzero(f > 10)}")
        print(f"[DEBUG]   >50: {np.count_nonzero(f > 50)}")
        print(f"[DEBUG]   >100: {np.count_nonzero(f > 100)}")
        print(f"[DEBUG]   >200: {np.count_nonzero(f > 200)}")
        print(f"[DEBUG]   ==255 (clipped): {np.count_nonzero(f == 255)}")

        # Check hot pixels (potential noise)
        hot_pixels = np.where(f > 1000)
        if len(hot_pixels[0]) > 0:
            print(f"[DEBUG] Hot pixels (>1000 counts): {len(hot_pixels[0])}")
            for i in range(min(5, len(hot_pixels[0]))):
                y, x = hot_pixels[0][i], hot_pixels[1][i]
                count = f[y, x]
                print(f"[DEBUG]   Pixel ({x},{y}): {count:.0f} events")

    img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if debug:
        print(f"[DEBUG] Normalized frame range: [{img8.min()}, {img8.max()}]")
        print(f"[DEBUG] Pixels above pre-threshold ({pre_threshold}): {np.count_nonzero(img8 > pre_threshold)}")

    # Pre-threshold to filter noise (keeps only very bright pixels)
    _, image_binary = cv2.threshold(img8, pre_threshold, 255, cv2.THRESH_BINARY)

    if debug:
        print(f"[DEBUG] After pre-threshold: {np.count_nonzero(image_binary)} white pixels")

    # Apply Otsu threshold to pre-thresholded result
    otsu_thresh, mask = cv2.threshold(image_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        print(f"[DEBUG] Otsu threshold value: {otsu_thresh}")
        print(f"[DEBUG] After Otsu: {np.count_nonzero(mask)} white pixels")

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug:
        print(f"[DEBUG] After morphology: {np.count_nonzero(mask)} white pixels")

    # Find ALL contours (key difference from fan detector)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        if debug:
            print("[DEBUG] No contours found")
        return []

    if debug:
        print(f"[DEBUG] Found {len(contours)} contours")

    # Collect valid ellipse candidates
    candidates: List[Tuple[int, int, float, float, float, float]] = []
    rejected_counts = {
        "too_few_points": 0,
        "area_too_small": 0,
        "area_too_large": 0,
        "minor_axis_zero": 0,
        "angle_too_low": 0,
        "angle_too_high": 0,
    }

    for idx, cnt in enumerate(contours):
        # fitEllipse needs at least 5 points
        # Use 50 points minimum (match original line 212 for accuracy)
        if len(cnt) < 50:
            rejected_counts["too_few_points"] += 1
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            rejected_counts["area_too_small"] += 1
            if debug and idx < 5:  # Only show first few
                print(f"[DEBUG]   Contour {idx}: area={area:.1f} < {min_area} (rejected: too small)")
            continue
        if area > max_area:
            rejected_counts["area_too_large"] += 1
            if debug and idx < 5:
                print(f"[DEBUG]   Contour {idx}: area={area:.1f} > {max_area:.1f} (rejected: too large)")
            continue

        # Fit ellipse to contour
        (cx_f, cy_f), (major, minor), angle_deg = cv2.fitEllipse(cnt)

        if minor <= 0:
            rejected_counts["minor_axis_zero"] += 1
            continue

        # ORIENTATION FILTER: Match legacy behavior
        # angle_deg is in [0, 180) from OpenCV
        # Legacy keeps ellipses in range [70, 110] degrees
        # This appears to work empirically for drone propellers in the dataset
        if angle_deg < 70.0:
            rejected_counts["angle_too_low"] += 1
            if debug and idx < 5:
                print(f"[DEBUG]   Contour {idx}: angle={angle_deg:.1f}° < 70° (rejected)")
            continue
        if angle_deg > 110.0:
            rejected_counts["angle_too_high"] += 1
            if debug and idx < 5:
                print(f"[DEBUG]   Contour {idx}: angle={angle_deg:.1f}° > 110° (rejected)")
            continue

        # Convert to standard ellipse parameters
        a = major * 0.5   # semi-major axis
        b = minor * 0.5   # semi-minor axis
        phi = np.deg2rad(angle_deg)

        cx_i = int(cx_f)
        cy_i = int(cy_f)

        if debug and idx < 5:
            print(f"[DEBUG]   Contour {idx}: area={area:.1f}, angle={angle_deg:.1f}°, center=({cx_i},{cy_i}), axes=({a:.1f},{b:.1f}) ✓ ACCEPTED")

        # Store (cx, cy, a, b, phi, area) - keep area for sorting
        candidates.append((cx_i, cy_i, float(a), float(b), float(phi), float(area)))

    if debug:
        print(f"[DEBUG] Rejection summary:")
        for reason, count in rejected_counts.items():
            if count > 0:
                print(f"[DEBUG]   {reason}: {count}")
        print(f"[DEBUG] Valid candidates: {len(candidates)}")

    if not candidates:
        if debug:
            print("[DEBUG] No valid candidates after filtering")
        return []

    # Sort by area (largest first)
    candidates.sort(key=lambda t: t[5], reverse=True)

    if debug:
        print(f"[DEBUG] Top candidates (by area):")
        for i, (cx, cy, a, b, phi, area) in enumerate(candidates[:3]):
            print(f"[DEBUG]   {i}: area={area:.1f}, center=({cx},{cy}), axes=({a:.1f},{b:.1f})")

    # Apply pair distance filter (match legacy behavior)
    # If top 2 candidates are far apart (>15px), only keep the largest
    # This prevents detecting unrelated objects as second propeller
    selected: List[Tuple[int, int, float, float, float, float]] = []

    if len(candidates) == 1 or max_ellipses == 1:
        selected = [candidates[0]]
        if debug:
            print(f"[DEBUG] Selected 1 ellipse (only candidate or max_ellipses=1)")
    else:
        c0 = candidates[0]
        c1 = candidates[1]
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        dist = np.sqrt(dx * dx + dy * dy)

        if debug:
            print(f"[DEBUG] Distance between top 2: {dist:.1f}px (threshold: {max_pair_distance}px)")

        if dist <= max_pair_distance:
            # Close enough - keep both
            selected = [c0, c1]
            if debug:
                print(f"[DEBUG] Selected 2 ellipses (close together)")
        else:
            # Too far apart - keep only largest
            selected = [c0]
            if debug:
                print(f"[DEBUG] Selected 1 ellipse (pair too far apart)")

    # Return ellipse parameters (drop area)
    ellipses: List[Tuple[int, int, float, float, float]] = []
    for (cx_i, cy_i, a, b, phi, _) in selected:
        ellipses.append((cx_i, cy_i, a, b, phi))

    if debug:
        print(f"[DEBUG] Returning {len(ellipses)} ellipse(s)")

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
