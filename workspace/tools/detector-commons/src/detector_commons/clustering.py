"""DBSCAN clustering utilities for blade/propeller detection."""

from typing import List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN


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
) -> List[Tuple[float, float]]:
    """Cluster events near the blade ellipse using DBSCAN.

    Uses elliptical radius:
      - translate by (cx, cy)
      - rotate by -phi
      - normalize by (a, b)
      r_ell = sqrt((x'/a)^2 + (y'/b)^2)

    Keeps points with r_ell in [r_min, r_max] and clusters them.

    Args:
        x: X coordinates of events
        y: Y coordinates of events
        cx: Ellipse center X
        cy: Ellipse center Y
        a: Ellipse semi-major axis
        b: Ellipse semi-minor axis
        phi: Ellipse rotation angle (radians)
        eps: DBSCAN eps parameter (cluster radius in pixels)
        min_samples: DBSCAN min_samples parameter
        r_min: Min elliptical radius (fraction of a, b)
        r_max: Max elliptical radius (fraction of a, b)

    Returns:
        List of cluster centers [(xc, yc), ...] sorted by size (largest first)
    """
    if a <= 0 or b <= 0:
        return []

    # Translate to ellipse center
    dx = x.astype(np.float32) - float(cx)
    dy = y.astype(np.float32) - float(cy)

    # Rotate into ellipse-aligned frame
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    x_rot = dx * cos_p + dy * sin_p
    y_rot = -dx * sin_p + dy * cos_p

    # Compute elliptical radius
    r_ell = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)

    # Filter to ring region
    mask = (r_ell >= r_min) & (r_ell <= r_max)
    if not np.any(mask):
        return []

    pts = np.column_stack([x[mask], y[mask]])
    if pts.shape[0] < min_samples:
        return []

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pts)

    unique_labels = [lab for lab in np.unique(labels) if lab != -1]
    if not unique_labels:
        return []

    # Compute cluster centers
    clusters = []
    for lab in unique_labels:
        pts_lab = pts[labels == lab]
        if pts_lab.shape[0] == 0:
            continue
        xc = pts_lab[:, 0].mean()
        yc = pts_lab[:, 1].mean()
        clusters.append((xc, yc, pts_lab.shape[0]))

    # Sort by size, return top 3
    clusters.sort(key=lambda t: t[2], reverse=True)
    return [(xc, yc) for (xc, yc, n) in clusters[:3]]
