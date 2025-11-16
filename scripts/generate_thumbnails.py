#!/usr/bin/env python3
"""Generate thumbnail previews for event camera datasets.

Scans evio/data/ for *_legacy.h5 files and generates 870x435 PNG thumbnails
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


def generate_thumbnail(
    h5_path: Path,
    force: bool = False,
) -> Path | None:
    """Generate thumbnail for a single dataset.

    Args:
        h5_path: Path to *_legacy.h5 file
        force: If True, regenerate even if thumbnail exists

    Returns:
        Path to generated PNG, or None if failed
    """
    # Determine output path
    cache_dir = Path("evio/data/.cache/thumbnails")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Output filename: remove "_legacy" suffix
    output_name = h5_path.stem.replace("_legacy", "") + ".png"
    output_path = cache_dir / output_name

    # Skip if exists and not forcing
    if output_path.exists() and not force:
        print(f"  ⏭️  Skipping {h5_path.name} (thumbnail exists)")
        return output_path

    try:
        # Load dataset metadata (lazy - no full collect!)
        lazy_events = evlib.load_events(str(h5_path))
        width, height, t_min, t_max = extract_metadata(lazy_events)

        # Render first 1 second window (1,000,000 microseconds)
        window_end = min(t_min + 1_000_000, t_max)

        # Filter and collect ONLY the window
        schema = lazy_events.schema
        t_dtype = schema["t"]

        if isinstance(t_dtype, pl.Duration):
            window = lazy_events.filter(
                (pl.col("t") >= pl.duration(microseconds=t_min)) &
                (pl.col("t") < pl.duration(microseconds=window_end))
            ).collect()
        else:
            window = lazy_events.filter(
                (pl.col("t") >= t_min) &
                (pl.col("t") < window_end)
            ).collect()

        # Extract event data
        x_coords = window["x"].to_numpy().astype(np.int32)
        y_coords = window["y"].to_numpy().astype(np.int32)
        polarities = window["polarity"].to_numpy()

        # Render polarity frame
        frame = render_polarity_frame(x_coords, y_coords, polarities, width, height)

        # Resize to 870x435 with letterboxing (matches fullscreen tile size)
        thumbnail = resize_with_letterbox(frame, target_w=870, target_h=435)

        # Save PNG
        cv2.imwrite(str(output_path), thumbnail)

        print(f"  ✅ Generated {output_name} ({len(x_coords):,} events, {width}x{height})")
        return output_path

    except Exception as e:
        print(f"  ❌ Failed to generate {h5_path.name}: {e}", file=sys.stderr)
        return None


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

    # Generate thumbnails
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, dataset_path in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {dataset_path.name}")

        # Check if thumbnail exists before attempting generation
        cache_dir = Path("evio/data/.cache/thumbnails")
        output_name = dataset_path.stem.replace("_legacy", "") + ".png"
        output_path = cache_dir / output_name
        already_exists = output_path.exists()

        result = generate_thumbnail(dataset_path, force=args.force)

        if result is None:
            fail_count += 1
        elif already_exists and not args.force:
            skip_count += 1
        else:
            success_count += 1

    # Summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    if success_count > 0:
        print(f"✅ Generated: {success_count}")
    if skip_count > 0:
        print(f"⏭️  Skipped: {skip_count} (already exist)")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count}")
    print()
    print(f"Thumbnails saved to: evio/data/.cache/thumbnails/")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
