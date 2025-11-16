"""Shared utilities for event camera detectors using evlib."""

__version__ = "0.1.0"

# evlib loaders
from .loaders import (
    load_legacy_h5,
    get_window_evlib,
    get_timestamp_range,
)

# evlib representations
from .representations import (
    build_accum_frame_evlib,
    pretty_event_frame_evlib,
)

# Clustering
from .clustering import cluster_blades_dbscan_elliptic

# Temporal lookup
from .temporal import (
    pick_geom_at_time,
    pick_propellers_at_time,
)

__all__ = [
    # Loaders
    "load_legacy_h5",
    "get_window_evlib",
    "get_timestamp_range",
    # Representations
    "build_accum_frame_evlib",
    "pretty_event_frame_evlib",
    # Clustering
    "cluster_blades_dbscan_elliptic",
    # Temporal
    "pick_geom_at_time",
    "pick_propellers_at_time",
]
