#!/usr/bin/env python3
"""Generate thumbnail previews for event camera datasets.

Scans evio/data/ for *_legacy.h5 files and generates 300x150 PNG thumbnails
from the first 1 second of events. Thumbnails cached to evio/data/.cache/thumbnails/

Usage:
    nix develop
    generate-thumbnails
    generate-thumbnails --force  # Regenerate existing

See docs/plans/2025-11-16-visual-polish-thumbnails-palette-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import polars as pl
import evlib


def discover_datasets() -> list[Path]:
    """Scan evio/data/ for *_legacy.h5 files.

    Returns:
        List of Path objects to HDF5 files
    """
    data_dir = Path("evio/data")
    if not data_dir.exists():
        print(f"Error: {data_dir} not found", file=sys.stderr)
        print("Please run from repository root", file=sys.stderr)
        return []

    # Find all *_legacy.h5 files, skip _evt3
    h5_files = [
        f for f in data_dir.rglob("*_legacy.h5")
        if '_evt3' not in f.stem
    ]

    # Sort by parent dir then name
    h5_files.sort(key=lambda p: (p.parent.name, p.name))

    return h5_files


def extract_metadata(lazy_events: pl.LazyFrame) -> Tuple[int, int, int, int]:
    """Extract width, height, t_min, t_max from lazy events.

    Args:
        lazy_events: Lazy polars dataframe of events

    Returns:
        Tuple of (width, height, t_min, t_max)
    """
    metadata = lazy_events.select([
        pl.col("x").max().alias("max_x"),
        pl.col("y").max().alias("max_y"),
        pl.col("t").min().alias("t_min"),
        pl.col("t").max().alias("t_max"),
    ]).collect()

    width = int(metadata["max_x"][0]) + 1
    height = int(metadata["max_y"][0]) + 1

    # Handle Duration vs Int64
    import datetime
    t_min_val = metadata["t_min"][0]
    t_max_val = metadata["t_max"][0]

    if isinstance(t_min_val, datetime.timedelta):
        t_min = int(t_min_val.total_seconds() * 1e6)
        t_max = int(t_max_val.total_seconds() * 1e6)
    elif isinstance(t_min_val, pl.Duration):
        t_min = int(t_min_val.total_microseconds())
        t_max = int(t_max_val.total_microseconds())
    else:
        t_min = int(t_min_val)
        t_max = int(t_max_val)

    return width, height, t_min, t_max


def render_polarity_frame(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polarities: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render polarity events to BGR frame.

    Args:
        x_coords: Event x coordinates
        y_coords: Event y coordinates
        polarities: Event polarities (>0 = ON, <=0 = OFF)
        width: Frame width
        height: Frame height

    Returns:
        BGR frame (numpy array, uint8)
    """
    # Base gray, white for ON, black for OFF
    frame = np.full((height, width, 3), (127, 127, 127), dtype=np.uint8)

    if len(x_coords) > 0:
        polarities_on = polarities > 0
        # ON events = white
        frame[y_coords[polarities_on], x_coords[polarities_on]] = (255, 255, 255)
        # OFF events = black
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = (0, 0, 0)

    return frame


def resize_with_letterbox(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Resize frame to target size with letterboxing (preserve aspect ratio).

    Args:
        frame: Input BGR frame
        target_w: Target width
        target_h: Target height

    Returns:
        Resized frame with letterboxing (black bars)
    """
    h, w = frame.shape[:2]

    # Calculate scaling to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create letterboxed frame (black background)
    letterboxed = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center resized frame
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    letterboxed[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return letterboxed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate thumbnail previews for event camera datasets",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate existing thumbnails',
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Generate Dataset Thumbnails")
    print("=" * 60)
    print()

    datasets = discover_datasets()

    if not datasets:
        print("No *_legacy.h5 files found in evio/data/")
        print("Run: convert-all-legacy-to-hdf5")
        return 1

    print(f"Found {len(datasets)} dataset(s)")
    print()

    # TODO: Generate thumbnails
    print("TODO: Thumbnail generation not yet implemented")

    return 0


if __name__ == '__main__':
    sys.exit(main())
