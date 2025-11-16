"""Detector utilities - refactored from fan_detector_demo.py and drone_detector_demo.py."""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import cv2

# Try to import sklearn - gracefully degrade if not available
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, DBSCAN clustering disabled", flush=True)

# ============================================================================
# Color Palette: Detection Overlays
# ============================================================================
# All colors in BGR format (OpenCV convention)

DETECTION_SUCCESS = (153, 255, 0)   # #00ff99 - green (BGR) for ellipses/RPM
DETECTION_WARNING = (204, 102, 255)   # #ff66cc - pink Y2K (BGR) for warnings
DETECTION_BOX = (255, 255, 0)       # #00ffff - cyan (BGR) for bounding boxes
DETECTION_CLUSTER = (0, 255, 255)   # #ffff00 - yellow (BGR) for blade clusters


# ============================================================================
# Fan Detector
# ============================================================================

@dataclass
class FanDetection:
    """Fan ellipse detection result."""
    cx: int
    cy: int
    a: float
    b: float
    phi: float
    clusters: List[Tuple[float, float]]
    rpm: float


def build_accum_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """Build grayscale accumulation frame from events.

    Args:
        window: Tuple of (x_coords, y_coords, polarities) event arrays
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        Grayscale accumulation frame (uint8)
    """
    x_coords, y_coords, _ = window
    frame = np.zeros((height, width), dtype=np.uint16)
    if len(x_coords) > 0:
        frame[y_coords, x_coords] += 1
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def fit_ellipse_from_frame(
    accum_frame: np.ndarray,
    prev_params: Optional[Tuple[int, int, float, float, float]] = None,
) -> Tuple[int, int, float, float, float]:
    """Fit ellipse to largest contour in accumulated frame.

    Args:
        accum_frame: Grayscale accumulation frame
        prev_params: Previous ellipse parameters (cx, cy, a, b, phi) for fallback

    Returns:
        Tuple of (cx, cy, a, b, phi) - center, semi-axes, and orientation
    """
    h, w = accum_frame.shape

    f = accum_frame.astype(np.float32)
    if f.max() <= 0:
        if prev_params is not None:
            return prev_params
        return w // 2, h // 2, min(w, h) * 0.25, min(w, h) * 0.25, 0.0

    img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img8, (5, 5), 0)

    _, mask = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        if prev_params is not None:
            return prev_params
        return w // 2, h // 2, min(w, h) * 0.25, min(w, h) * 0.25, 0.0

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        if prev_params is not None:
            return prev_params
        return w // 2, h // 2, min(w, h) * 0.25, min(w, h) * 0.25, 0.0

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (minor_axis, major_axis), angle_deg = ellipse

    a = major_axis / 2.0
    b = minor_axis / 2.0
    phi = np.deg2rad(angle_deg)

    return int(cx), int(cy), a, b, phi


def cluster_blades_dbscan(
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
) -> List[Tuple[float, float]]:
    """Cluster events near blade ellipse using DBSCAN.

    Uses elliptical radius:
      - translate by (cx, cy)
      - rotate by -phi
      - normalize by (a, b)
      r_ell = sqrt((x'/a)^2 + (y'/b)^2)

    Keeps points with r_ell in [r_min, r_max] and clusters them.

    Args:
        x, y: Event coordinates
        cx, cy: Ellipse center
        a, b: Ellipse semi-axes
        phi: Ellipse orientation (radians)
        eps: DBSCAN cluster radius
        min_samples: Minimum points per cluster
        r_min, r_max: Elliptical radius range for filtering

    Returns:
        List of (xc, yc) cluster centers (up to 3 largest clusters)
    """
    if not SKLEARN_AVAILABLE:
        return []

    if a <= 0 or b <= 0:
        return []

    dx = x.astype(np.float32) - float(cx)
    dy = y.astype(np.float32) - float(cy)

    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    x_rot = dx * cos_p + dy * sin_p
    y_rot = -dx * sin_p + dy * cos_p

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


def estimate_rpm_from_clusters(
    clusters: List[Tuple[float, float]],
    window_duration_us: int,
) -> float:
    """Estimate RPM from number of blade clusters.

    Args:
        clusters: List of (xc, yc) cluster centers
        window_duration_us: Window duration in microseconds

    Returns:
        Estimated RPM (normalized and smoothed)
    """
    if not clusters:
        return 0.0

    # The RPM should be constant regardless of window size
    # Key insight: Smaller windows see fewer clusters, but represent shorter time
    #
    # Example: Fan at 1200 RPM = 20 rev/sec
    # - 10ms window: might see 3 clusters → 3 blades in 10ms
    # - 1ms window: might see 0-1 clusters → fewer blades in 1ms
    # - 10μs window: might see 0 clusters → almost no blades in 10μs
    #
    # We need to scale UP the estimate for smaller windows

    num_clusters = len(clusters)
    if num_clusters == 0:
        return 0.0

    # Assume 3 blades per rotation (typical fan)
    # If we see N clusters in window_us, how many clusters per second?
    window_s = window_duration_us / 1e6
    if window_s <= 0:
        return 0.0

    clusters_per_second = num_clusters / window_s
    rotations_per_second = clusters_per_second / 3.0
    rpm = rotations_per_second * 60.0

    # Clamp to reasonable range and return
    return max(0.0, min(rpm, 20000.0))  # Allow up to 20k RPM


def detect_fan(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    window_us: int,
    prev_params: Optional[Tuple[int, int, float, float, float]] = None,
) -> FanDetection:
    """Run fan detection: ellipse fit + blade clustering + RPM estimate.

    Args:
        window: Tuple of (x_coords, y_coords, polarities) event arrays
        width: Frame width
        height: Frame height
        window_us: Window duration in microseconds
        prev_params: Previous ellipse parameters for temporal smoothing

    Returns:
        FanDetection result
    """
    x_coords, y_coords, polarities = window

    # Build accumulation frame
    accum = build_accum_frame(window, width, height)

    # Fit ellipse
    cx, cy, a, b, phi = fit_ellipse_from_frame(accum, prev_params)

    # Cluster blades
    clusters = []
    if len(x_coords) > 0:
        clusters = cluster_blades_dbscan(x_coords, y_coords, cx, cy, a, b, phi)

    # Estimate RPM
    rpm = estimate_rpm_from_clusters(clusters, window_us)

    return FanDetection(cx=cx, cy=cy, a=a, b=b, phi=phi, clusters=clusters, rpm=rpm)


def render_fan_overlay(
    base_frame: np.ndarray,
    detection: FanDetection,
) -> np.ndarray:
    """Draw fan detection overlay on base frame.

    Args:
        base_frame: Base BGR frame to draw on
        detection: FanDetection result

    Returns:
        Frame with overlays drawn
    """
    frame = base_frame.copy()

    # Draw ellipse (green)
    center = (detection.cx, detection.cy)
    axes = (int(detection.a), int(detection.b))
    angle_deg = np.rad2deg(detection.phi)
    cv2.ellipse(frame, center, axes, angle_deg, 0, 360, DETECTION_SUCCESS, 2)  # Green

    # Draw center
    cv2.circle(frame, center, 5, DETECTION_SUCCESS, -1)

    # Draw blade clusters (yellow circles)
    for xc, yc in detection.clusters:
        cv2.circle(frame, (int(xc), int(yc)), 8, DETECTION_CLUSTER, 2)  # Yellow clusters

    # Draw RPM text (green)
    rpm_text = f"RPM: {detection.rpm:.0f}"
    cv2.putText(frame, rpm_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, DETECTION_SUCCESS, 2, cv2.LINE_AA)  # Green

    return frame


# ============================================================================
# Drone Detector (placeholder - to be extracted from drone_detector_demo.py)
# ============================================================================

@dataclass
class DroneDetection:
    """Drone detection result (placeholder)."""
    boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    warning: bool


def propeller_mask_from_frame(
    accum_frame: np.ndarray,
    min_area: float = 145.0,
    max_area_frac: float = 0.01,
    top_k: int = 2,
) -> Tuple[List[Tuple[int, int, float, float, float]], np.ndarray]:
    """Detect propeller-like blobs in accumulated frame.

    Args:
        accum_frame: Grayscale accumulation frame
        min_area: Minimum blob area
        max_area_frac: Maximum blob area as fraction of frame
        top_k: Number of top candidates to return

    Returns:
        Tuple of (ellipse_list, mask)
        ellipse_list: List of (cx, cy, a, b, phi) tuples
        mask: Binary mask with detected propellers
    """
    h, w = accum_frame.shape
    prop_mask = np.zeros((h, w), dtype=np.uint8)
    candidates: List[Tuple[int, int, float, float, float, float]] = []

    f = accum_frame.astype(np.float32)
    if f.max() <= 0:
        return [], prop_mask

    img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, image_binary = cv2.threshold(img8, 250, 255, cv2.THRESH_BINARY)

    _, mask = cv2.threshold(image_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = (h * w) * max_area_frac
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (minor, major), angle = ellipse
            a = major / 2.0
            b = minor / 2.0
            phi = np.deg2rad(angle)
            candidates.append((int(cx), int(cy), a, b, phi, area))

    # Sort by area descending, take top K
    candidates.sort(key=lambda x: x[5], reverse=True)
    top_candidates = candidates[:top_k]

    # Draw mask
    for cx, cy, a, b, phi, _ in top_candidates:
        axes = (int(a), int(b))
        angle_deg = np.rad2deg(phi)
        cv2.ellipse(prop_mask, (cx, cy), axes, angle_deg, 0, 360, 255, -1)

    return [(cx, cy, a, b, phi) for cx, cy, a, b, phi, _ in top_candidates], prop_mask


def detect_drone(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> DroneDetection:
    """Run drone detection - detect propeller blobs.

    Args:
        window: Tuple of (x_coords, y_coords, polarities) event arrays
        width: Frame width
        height: Frame height

    Returns:
        DroneDetection result
    """
    # Build accumulation frame
    accum = build_accum_frame(window, width, height)

    # Detect propellers
    propellers, mask = propeller_mask_from_frame(accum, min_area=145.0, top_k=2)

    # Convert ellipses to bounding boxes
    boxes = []
    for cx, cy, a, b, phi in propellers:
        # Bounding box around ellipse
        x1 = int(cx - a)
        y1 = int(cy - b)
        w = int(2 * a)
        h = int(2 * b)
        boxes.append((max(0, x1), max(0, y1), w, h))

    # Warning if any propellers detected
    warning = len(propellers) > 0

    return DroneDetection(boxes=boxes, warning=warning)


def render_drone_overlay(
    base_frame: np.ndarray,
    detection: DroneDetection,
) -> np.ndarray:
    """Draw drone detection overlay.

    Args:
        base_frame: Base BGR frame to draw on
        detection: DroneDetection result

    Returns:
        Frame with overlays drawn
    """
    frame = base_frame.copy()

    # Draw bounding boxes (cyan)
    for x, y, w, h in detection.boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), DETECTION_BOX, 2)  # Cyan boxes

    # Draw warning if present (orange)
    if detection.warning:
        h_frame, w_frame = frame.shape[:2]

        # Warning text at top
        cv2.putText(frame, "DRONE DETECTED", (w_frame // 2 - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, DETECTION_WARNING, 2, cv2.LINE_AA)  # Orange warning

        # Count at bottom-right (white, smaller, positioned above branding bar)
        count_text = f"Propellers: {len(detection.boxes)}"
        cv2.putText(frame, count_text, (w_frame - 200, h_frame - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White, smaller

    return frame
