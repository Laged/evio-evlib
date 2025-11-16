# Visual Polish: Thumbnails + Palette Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add thumbnail previews and Y2K color palette to MVP launcher for polished visual experience.

**Architecture:** Pre-generate 300x150 PNG thumbnails from first 1 second of datasets using evlib lazy loading. Update launcher menu to display thumbnails with minimal text overlays. Apply Sensofusion military gray + pink accent palette throughout UI.

**Tech Stack:** Python 3.11, evlib (Polars LazyFrame), OpenCV (cv2), NumPy, Nix (shell scripts)

---

## Task 1: Create Thumbnail Generation Script

**Files:**
- Create: `scripts/generate_thumbnails.py`

**Step 1: Write thumbnail script skeleton**

Create `scripts/generate_thumbnails.py`:

```python
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
```

**Step 2: Run script to verify skeleton**

Run:
```bash
nix develop --command python scripts/generate_thumbnails.py
```

Expected output:
```
==========================================
  Generate Dataset Thumbnails
==========================================

Found 5 dataset(s)

TODO: Thumbnail generation not yet implemented
```

**Step 3: Commit skeleton**

```bash
git add scripts/generate_thumbnails.py
git commit -m "feat(thumbnails): add generation script skeleton

Discovers *_legacy.h5 files and extracts metadata using evlib.
Thumbnail rendering not yet implemented.
"
```

---

## Task 2: Implement Polarity Frame Rendering

**Files:**
- Modify: `scripts/generate_thumbnails.py`

**Step 1: Add polarity rendering function**

Add this function to `scripts/generate_thumbnails.py` after `extract_metadata`:

```python
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
```

**Step 2: Add letterbox resize function**

Add this function after `render_polarity_frame`:

```python
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
```

**Step 3: Test functions manually**

Add temporary test code at end of `main()` before `return 0`:

```python
    # Test rendering with first dataset
    if datasets:
        print("Testing polarity rendering...")
        test_path = datasets[0]
        print(f"Loading {test_path.name}...")

        lazy_events = evlib.load_events(str(test_path))
        width, height, t_min, t_max = extract_metadata(lazy_events)

        # Get first 100ms for quick test
        window_end = min(t_min + 100_000, t_max)

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

        x = window["x"].to_numpy().astype(np.int32)
        y = window["y"].to_numpy().astype(np.int32)
        p = window["polarity"].to_numpy()

        frame = render_polarity_frame(x, y, p, width, height)
        thumbnail = resize_with_letterbox(frame, 300, 150)

        print(f"✓ Rendered {len(x)} events")
        print(f"✓ Frame shape: {frame.shape}")
        print(f"✓ Thumbnail shape: {thumbnail.shape}")
```

**Step 4: Run test**

Run:
```bash
nix develop --command python scripts/generate_thumbnails.py
```

Expected output should include:
```
Testing polarity rendering...
Loading fan_const_rpm_legacy.h5...
✓ Rendered XXXXX events
✓ Frame shape: (720, 1280, 3)
✓ Thumbnail shape: (150, 300, 3)
```

**Step 5: Remove test code and commit**

Remove the test code block from `main()`, then:

```bash
git add scripts/generate_thumbnails.py
git commit -m "feat(thumbnails): add polarity rendering and letterbox resize

Implements render_polarity_frame() for ON/OFF event visualization.
Adds resize_with_letterbox() to preserve aspect ratio with black bars.
"
```

---

## Task 3: Implement Thumbnail Generation and Caching

**Files:**
- Modify: `scripts/generate_thumbnails.py`

**Step 1: Add thumbnail generation function**

Add this function before `main()`:

```python
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

        # Resize to 300x150 with letterboxing
        thumbnail = resize_with_letterbox(frame, target_w=300, target_h=150)

        # Save PNG
        cv2.imwrite(str(output_path), thumbnail)

        print(f"  ✅ Generated {output_name} ({len(x_coords):,} events, {width}x{height})")
        return output_path

    except Exception as e:
        print(f"  ❌ Failed to generate {h5_path.name}: {e}", file=sys.stderr)
        return None
```

**Step 2: Update main() to use generate_thumbnail**

Replace the `# TODO: Generate thumbnails` section in `main()` with:

```python
    # Generate thumbnails
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, dataset_path in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {dataset_path.name}")

        result = generate_thumbnail(dataset_path, force=args.force)

        if result is None:
            fail_count += 1
        elif result.exists() and not args.force:
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
```

**Step 3: Test thumbnail generation**

Run:
```bash
nix develop --command python scripts/generate_thumbnails.py
```

Expected output:
```
==========================================
  Generate Dataset Thumbnails
==========================================

Found 5 dataset(s)

[1/5] fan_const_rpm_legacy.h5
  ✅ Generated fan_const_rpm.png (XXXXX events, 1280x720)
[2/5] fan_varying_rpm_legacy.h5
  ✅ Generated fan_varying_rpm.png (XXXXX events, 1280x720)
...

==========================================
  Summary
==========================================
✅ Generated: 5

Thumbnails saved to: evio/data/.cache/thumbnails/
```

**Step 4: Verify PNG files created**

Run:
```bash
ls -lh evio/data/.cache/thumbnails/
```

Expected: 5 PNG files, each 300x150 pixels.

**Step 5: Test --force flag**

Run:
```bash
nix develop --command python scripts/generate_thumbnails.py --force
```

Expected: All thumbnails regenerated.

**Step 6: Commit**

```bash
git add scripts/generate_thumbnails.py
git add evio/data/.cache/thumbnails/*.png
git commit -m "feat(thumbnails): implement generation and caching

Generates 300x150 PNG thumbnails from first 1 second of datasets.
Caches to evio/data/.cache/thumbnails/ with --force option.
Tested on 5 datasets successfully.
"
```

---

## Task 4: Add Nix Integration

**Files:**
- Modify: `flake.nix`

**Step 1: Add generateThumbnailsScript to flake.nix**

Find the section with `convertAllLegacyToHdf5Script` (around line 222) and add after it:

```nix
        # Generate thumbnails script
        generateThumbnailsScript = pkgs.writeShellScriptBin "generate-thumbnails" ''
          set -euo pipefail
          exec ${pkgs.uv}/bin/uv run --package evio python scripts/generate_thumbnails.py "$@"
        '';
```

**Step 2: Add to buildInputs**

Find the `buildInputs` array (around line 227) and add `generateThumbnailsScript` to the list:

```nix
          buildInputs = [
            # Core tools
            python
            pkgs.uv                 # UV package manager
            pkgs.gdown              # Google Drive downloader
            pkgs.unzip              # ZIP extraction
            convertEvt3Script       # convert-evt3-raw-to-dat command
            convertAllDatasetsScript # convert-all-datasets command
            unzipDatasetsScript     # unzip-datasets command
            convertLegacyDatToHdf5Script # convert-legacy-dat-to-hdf5 command
            convertAllLegacyToHdf5Script # convert-all-legacy-to-hdf5 command
            generateThumbnailsScript     # generate-thumbnails command (NEW)

            # Rust toolchain (for evlib compilation)
            ...
```

**Step 3: Add shell alias**

Find the shell aliases section (around line 339) and add:

```nix
            alias generate-thumbnails='uv run --package evio python scripts/generate_thumbnails.py'
```

**Step 4: Update shellHook help text**

Find the `🎯 Detector Demos:` section (around line 318) and add before it:

```nix
            echo "🎨 Thumbnails:"
            echo "  generate-thumbnails    : Generate PNG previews for launcher menu"
            echo "  generate-thumbnails --force : Regenerate all thumbnails"
            echo ""
```

**Step 5: Test Nix integration**

Exit and re-enter nix shell:
```bash
exit
nix develop
```

Expected: New help text shows `generate-thumbnails` command.

**Step 6: Test command**

Run:
```bash
generate-thumbnails --force
```

Expected: Thumbnails regenerated successfully.

**Step 7: Commit**

```bash
git add flake.nix
git commit -m "feat(thumbnails): add Nix command and shell integration

Adds generate-thumbnails command to Nix shell.
Updates shellHook help text with thumbnail section.
"
```

---

## Task 5: Define Color Palette Constants

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add color constants at top of file**

After the imports (around line 15), add these color constant definitions:

```python
# ============================================================================
# Color Palette: Sensofusion Military Gray + Y2K Pink Accents
# ============================================================================
# All colors in BGR format (OpenCV convention)

# Menu colors
BG_COLOR = (43, 43, 43)           # #2b2b2b - dark gray background
TILE_COLOR = (58, 58, 58)         # #3a3a3a - tile default (fallback)
TILE_SELECTED = (204, 102, 255)   # #ff66cc - pink Y2K accent
TEXT_PRIMARY = (245, 245, 245)    # #f5f5f5 - white
TEXT_SECONDARY = (192, 192, 192)  # #c0c0c0 - light gray
OVERLAY_BAND = (0, 0, 0)          # Black (used with alpha=0.6)

# Status bar
STATUS_BAR_BG = (30, 30, 30)      # #1e1e1e - darker gray
STATUS_TEXT = (200, 200, 200)     # Light gray

# Playback HUD
HUD_PANEL_BG = (0, 0, 0)          # Black (used with alpha=0.6)
HUD_TEXT = (245, 245, 245)        # White

# Help overlay
HELP_BG = (30, 30, 30)            # Dark gray (used with alpha=0.8)
HELP_TITLE = (245, 245, 245)      # White
HELP_TEXT = (200, 200, 200)       # Light gray
```

**Step 2: Update existing color references in _render_menu**

Find `_render_menu()` method (line 268) and replace hardcoded colors:

**Before:**
```python
frame = np.full((frame_height, frame_width, 3), (43, 43, 43), dtype=np.uint8)
```

**After:**
```python
frame = np.full((frame_height, frame_width, 3), BG_COLOR, dtype=np.uint8)
```

**Before (line 304-309):**
```python
if is_selected:
    tile_color = (226, 144, 74)  # Blue accent (BGR: #4a90e2)
    border_thickness = 3
else:
    tile_color = (64, 64, 64)  # Medium gray
    border_thickness = 1
```

**After:**
```python
if is_selected:
    tile_color = TILE_SELECTED  # Pink Y2K accent
    border_thickness = 3
else:
    tile_color = TILE_COLOR  # Medium gray
    border_thickness = 1
```

**Before (line 323-324):**
```python
text_color = (255, 255, 255)
meta_color = (180, 180, 180)
```

**After:**
```python
text_color = TEXT_PRIMARY
meta_color = TEXT_SECONDARY
```

**Before (line 343-344):**
```python
cv2.rectangle(frame, (0, status_y), (frame_width, frame_height),
              (30, 30, 30), -1)
```

**After:**
```python
cv2.rectangle(frame, (0, status_y), (frame_width, frame_height),
              STATUS_BAR_BG, -1)
```

**Before (line 350):**
```python
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
```

**After:**
```python
cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_TEXT, 1, cv2.LINE_AA)
```

**Step 3: Update _draw_hud colors**

Find `_draw_hud()` method (line 461) and replace:

**Before (line 480-482):**
```python
overlay = frame.copy()
cv2.rectangle(overlay, (panel_x, panel_y),
              (panel_x + panel_w, panel_y + panel_h),
              (0, 0, 0), -1)
```

**After:**
```python
overlay = frame.copy()
cv2.rectangle(overlay, (panel_x, panel_y),
              (panel_x + panel_w, panel_y + panel_h),
              HUD_PANEL_BG, -1)
```

**Before (line 497-499):**
```python
cv2.putText(frame, line, (panel_x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
            1, cv2.LINE_AA)
```

**After:**
```python
cv2.putText(frame, line, (panel_x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_TEXT,
            1, cv2.LINE_AA)
```

**Step 4: Update _draw_help_overlay colors**

Find `_draw_help_overlay()` method (line 502) and replace:

**Before (line 509):**
```python
cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (30, 30, 30), -1)
```

**After:**
```python
cv2.rectangle(overlay, (0, h - overlay_h), (w, h), HELP_BG, -1)
```

**Before (line 526-527):**
```python
color = (255, 255, 255) if i == 0 else (200, 200, 200)
```

**After:**
```python
color = HELP_TITLE if i == 0 else HELP_TEXT
```

**Step 5: Update _show_error_and_return_to_menu colors**

Find `_show_error_and_return_to_menu()` method (line 533) and replace:

**Before (line 535):**
```python
frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)
```

**After:**
```python
frame = np.full((480, 640, 3), HELP_BG, dtype=np.uint8)
```

**Before (line 558-559):**
```python
cv2.putText(frame, line, (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
```

**After:**
```python
cv2.putText(frame, line, (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, HELP_TEXT, 1, cv2.LINE_AA)
```

**Step 6: Test palette changes**

Run launcher:
```bash
nix develop --command run-mvp-demo
```

Expected:
- Menu has pink selection border (not blue)
- Background is dark gray
- Status bar has darker gray background
- Press Enter to play, verify HUD has correct colors
- Press 'h' to verify help overlay has correct colors

**Step 7: Commit**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(palette): apply Sensofusion + Y2K color scheme

Defines color constants at top of mvp_launcher.py.
Updates menu, status bar, HUD, and help overlay to use palette.
Pink accent (#ff66cc) replaces blue selection border.
"
```

---

## Task 6: Implement Thumbnail Loading in Menu

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add thumbnail loading helper method**

Add this method to `MVPLauncher` class, before `_render_menu()`:

```python
    def _load_thumbnail(self, dataset: Dataset) -> np.ndarray | None:
        """Load cached thumbnail for dataset.

        Args:
            dataset: Dataset metadata

        Returns:
            BGR thumbnail image (300x150), or None if not found
        """
        # Thumbnail path: evio/data/.cache/thumbnails/<stem>.png
        # Remove "_legacy" suffix from dataset filename
        thumbnail_name = dataset.path.stem.replace("_legacy", "") + ".png"
        thumbnail_path = Path("evio/data/.cache/thumbnails") / thumbnail_name

        if not thumbnail_path.exists():
            return None

        try:
            thumbnail = cv2.imread(str(thumbnail_path))
            if thumbnail is None:
                return None

            # Verify size (should be 300x150)
            if thumbnail.shape[:2] != (150, 300):
                print(f"Warning: Invalid thumbnail size for {dataset.name}: {thumbnail.shape}", file=sys.stderr)
                return None

            return thumbnail
        except Exception as e:
            print(f"Warning: Failed to load thumbnail for {dataset.name}: {e}", file=sys.stderr)
            return None
```

**Step 2: Add thumbnail rendering with text overlay**

Add this method to `MVPLauncher` class, after `_load_thumbnail()`:

```python
    def _render_thumbnail_tile(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        tile_width: int,
        tile_height: int,
        thumbnail: np.ndarray,
        dataset_name: str,
        is_selected: bool,
    ) -> None:
        """Render thumbnail tile with text overlay.

        Args:
            frame: Frame to draw on
            x, y: Top-left corner of tile
            tile_width, tile_height: Tile dimensions (300x150)
            thumbnail: BGR thumbnail image (300x150)
            dataset_name: Dataset name for text overlay
            is_selected: Whether tile is selected
        """
        # Draw thumbnail as background
        frame[y:y+tile_height, x:x+tile_width] = thumbnail

        # Draw selection border if selected
        if is_selected:
            cv2.rectangle(frame, (x, y), (x + tile_width, y + tile_height),
                          TILE_SELECTED, 3)

        # Draw semi-transparent black band at bottom
        band_height = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y + tile_height - band_height),
                      (x + tile_width, y + tile_height), OVERLAY_BAND, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw dataset name (centered, white)
        text_size = cv2.getTextSize(dataset_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (tile_width - text_size[0]) // 2
        text_y = y + tile_height - 15  # 15px from bottom
        cv2.putText(frame, dataset_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_PRIMARY, 2, cv2.LINE_AA)
```

**Step 3: Add fallback tile rendering**

Add this method to `MVPLauncher` class, after `_render_thumbnail_tile()`:

```python
    def _render_fallback_tile(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        tile_width: int,
        tile_height: int,
        dataset_name: str,
        category: str,
        size_mb: float,
        is_selected: bool,
    ) -> None:
        """Render fallback tile (no thumbnail available).

        Args:
            frame: Frame to draw on
            x, y: Top-left corner of tile
            tile_width, tile_height: Tile dimensions (300x150)
            dataset_name: Dataset name
            category: Dataset category
            size_mb: Dataset size in MB
            is_selected: Whether tile is selected
        """
        # Tile background color
        if is_selected:
            tile_color = TILE_SELECTED
            border_thickness = 3
        else:
            tile_color = TILE_COLOR
            border_thickness = 1

        # Draw tile rectangle
        cv2.rectangle(frame, (x, y), (x + tile_width, y + tile_height),
                      tile_color, border_thickness)

        # Fill if not selected
        if not is_selected:
            cv2.rectangle(frame, (x + border_thickness, y + border_thickness),
                          (x + tile_width - border_thickness, y + tile_height - border_thickness),
                          tile_color, -1)

        # Draw text (centered)
        # Line 1: Dataset name
        name_size = cv2.getTextSize(dataset_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        name_x = x + (tile_width - name_size[0]) // 2
        name_y = y + 50
        cv2.putText(frame, dataset_name, (name_x, name_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_PRIMARY, 2, cv2.LINE_AA)

        # Line 2: "No preview"
        no_preview = "No preview"
        no_preview_size = cv2.getTextSize(no_preview, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        no_preview_x = x + (tile_width - no_preview_size[0]) // 2
        no_preview_y = y + 90
        cv2.putText(frame, no_preview, (no_preview_x, no_preview_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_SECONDARY, 1, cv2.LINE_AA)

        # Line 3: Category + size
        meta_text = f"{category} | {size_mb:.1f} MB"
        meta_size = cv2.getTextSize(meta_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        meta_x = x + (tile_width - meta_size[0]) // 2
        meta_y = y + 120
        cv2.putText(frame, meta_text, (meta_x, meta_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_SECONDARY, 1, cv2.LINE_AA)
```

**Step 4: Update _render_menu to use thumbnails**

Find the tile rendering loop in `_render_menu()` (around line 294-340) and replace the entire tile rendering section with:

```python
        # Draw tiles
        for i, dataset in enumerate(self.datasets):
            row = i // cols
            col = i % cols

            x = col * tile_width + (col + 1) * margin
            y = row * tile_height + (row + 1) * margin

            is_selected = (i == self.selected_index)

            # Try to load thumbnail
            thumbnail = self._load_thumbnail(dataset)

            if thumbnail is not None:
                # Render thumbnail tile with minimal overlay
                self._render_thumbnail_tile(
                    frame, x, y, tile_width, tile_height,
                    thumbnail, dataset.name, is_selected
                )
            else:
                # Render fallback tile
                self._render_fallback_tile(
                    frame, x, y, tile_width, tile_height,
                    dataset.name, dataset.category, dataset.size_mb, is_selected
                )
```

**Step 5: Test thumbnail rendering**

Run launcher:
```bash
nix develop --command run-mvp-demo
```

Expected:
- Menu shows thumbnail previews with dataset names at bottom
- Selection shows pink border around thumbnail
- Navigate with arrows to verify selection works
- If you delete thumbnails: `rm -rf evio/data/.cache/thumbnails/`, menu shows fallback tiles

**Step 6: Regenerate thumbnails and test again**

```bash
generate-thumbnails
nix develop --command run-mvp-demo
```

Expected: Thumbnails displayed with minimal text overlay.

**Step 7: Commit**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(thumbnails): implement menu rendering with previews

Adds thumbnail loading and rendering to menu tiles.
Fallback to text tiles if thumbnail missing.
Minimal text overlay (dataset name only) on thumbnails.
Pink selection border applied to thumbnail tiles.
"
```

---

## Task 7: Update Detector Overlay Colors

**Files:**
- Modify: `evio/scripts/detector_utils.py`

**Step 1: Add color constants to detector_utils.py**

Add these constants after the imports (around line 16):

```python
# ============================================================================
# Color Palette: Detection Overlays
# ============================================================================
# All colors in BGR format (OpenCV convention)

DETECTION_SUCCESS = (153, 255, 0)   # #00ff99 - green (BGR) for ellipses/RPM
DETECTION_WARNING = (0, 136, 255)   # #ff8800 - orange (BGR) for warnings
DETECTION_BOX = (255, 255, 0)       # #00ffff - cyan (BGR) for bounding boxes
DETECTION_CLUSTER = (0, 255, 255)   # #ffff00 - yellow (BGR) for blade clusters
```

**Step 2: Update render_fan_overlay colors**

Find `render_fan_overlay()` function (line 260) and update colors:

**Before (line 279):**
```python
cv2.ellipse(frame, center, axes, angle_deg, 0, 360, (255, 255, 0), 2)  # Cyan in BGR
```

**After:**
```python
cv2.ellipse(frame, center, axes, angle_deg, 0, 360, DETECTION_SUCCESS, 2)  # Green
```

**Before (line 282):**
```python
cv2.circle(frame, center, 5, (255, 255, 0), -1)
```

**After:**
```python
cv2.circle(frame, center, 5, DETECTION_SUCCESS, -1)
```

**Before (line 286):**
```python
cv2.circle(frame, (int(xc), int(yc)), 8, (0, 255, 255), 2)  # Yellow in BGR
```

**After:**
```python
cv2.circle(frame, (int(xc), int(yc)), 8, DETECTION_CLUSTER, 2)  # Yellow clusters
```

**Before (line 290-291):**
```python
cv2.putText(frame, rpm_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)  # Green
```

**After:**
```python
cv2.putText(frame, rpm_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, DETECTION_SUCCESS, 2, cv2.LINE_AA)  # Green
```

**Step 3: Update render_drone_overlay colors**

Find `render_drone_overlay()` function (line 408) and update colors:

**Before (line 425):**
```python
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

**After:**
```python
cv2.rectangle(frame, (x, y), (x + w, y + h), DETECTION_BOX, 2)  # Cyan boxes
```

**Before (line 432-433):**
```python
cv2.putText(frame, "DRONE DETECTED", (w_frame // 2 - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 136, 255), 2, cv2.LINE_AA)
```

**After:**
```python
cv2.putText(frame, "DRONE DETECTED", (w_frame // 2 - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, DETECTION_WARNING, 2, cv2.LINE_AA)  # Orange warning
```

**Before (line 437-438):**
```python
cv2.putText(frame, count_text, (w_frame - 200, h_frame - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 136, 255), 2, cv2.LINE_AA)
```

**After:**
```python
cv2.putText(frame, count_text, (w_frame - 200, h_frame - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, DETECTION_WARNING, 2, cv2.LINE_AA)  # Orange
```

**Step 4: Test detector colors**

Run launcher with fan dataset:
```bash
nix develop --command run-mvp-demo
```

Expected:
- Select fan dataset, press Enter
- Fan detector shows **green** ellipse and RPM text (not cyan/green mix)
- Blade clusters show yellow circles

Run launcher with drone dataset:
```bash
nix develop --command run-mvp-demo
```

Expected:
- Select drone_idle dataset, press Enter
- Drone detector shows **cyan** boxes (not red)
- Warning text shows **orange** (not red/orange mix)

**Step 5: Commit**

```bash
git add evio/scripts/detector_utils.py
git commit -m "feat(palette): apply detection overlay colors

Updates fan detector: green ellipse/RPM, yellow clusters.
Updates drone detector: cyan boxes, orange warnings.
Consistent with Sensofusion/Y2K palette design.
"
```

---

## Task 8: Documentation and Final Testing

**Files:**
- Modify: `docs/plans/ui-polishing.md`
- Create: `.gitignore` entry for thumbnail cache

**Step 1: Add .gitignore entry for thumbnail cache**

Check if `evio/data/.cache/` is already in `.gitignore`:

```bash
grep -r "\.cache" .gitignore
```

If not found, add to `.gitignore`:

```
# Thumbnail cache
evio/data/.cache/
```

**Step 2: Update ui-polishing.md to mark Phase 1 complete**

Add this section at the top of `docs/plans/ui-polishing.md`:

```markdown
## Implementation Status

**Phase 1: Visual Polish (Thumbnails + Palette) - ✅ COMPLETE**
- Implemented: 2025-11-16
- See: `docs/plans/2025-11-16-visual-polish-implementation.md`
- Thumbnails: Pre-generation script, 300x150 PNG cache
- Palette: Sensofusion gray + Y2K pink accents applied throughout
- Detector colors: Green/yellow (fan), cyan/orange (drone)

**Phase 2: Performance (Not Yet Started)**
...
```

**Step 3: Run comprehensive test suite**

Test 1: Generate thumbnails from scratch
```bash
rm -rf evio/data/.cache/thumbnails/
generate-thumbnails
```

Expected: All thumbnails generated successfully.

Test 2: Run launcher with thumbnails
```bash
run-mvp-demo
```

Expected:
- Menu shows thumbnail previews with pink selection
- All text uses correct palette colors
- Status bar has darker gray background

Test 3: Test playback with fan detector
- Select fan_const_rpm dataset, press Enter
- Verify green ellipse, yellow clusters, green RPM text
- Press 'h' to verify help overlay colors
- Press '2' to toggle HUD, verify colors

Test 4: Test playback with drone detector
- ESC to return to menu
- Select drone_idle dataset, press Enter
- Verify cyan boxes, orange warning text
- Press 'h' to verify help overlay colors

Test 5: Test fallback tiles
```bash
rm -rf evio/data/.cache/thumbnails/
run-mvp-demo
```

Expected: Menu shows fallback tiles with "No preview" text.

Test 6: Regenerate with --force
```bash
generate-thumbnails --force
run-mvp-demo
```

Expected: Thumbnails regenerated and displayed.

**Step 4: Commit documentation updates**

```bash
git add .gitignore
git add docs/plans/ui-polishing.md
git commit -m "docs(visual-polish): mark Phase 1 complete, add .gitignore

Thumbnail generation and palette application completed.
Added .cache/ to .gitignore for thumbnail cache.
Updated ui-polishing.md with implementation status.
"
```

---

## Task 9: Create Final Summary and Tag

**Files:**
- Create: `docs/plans/2025-11-16-visual-polish-completion.md`

**Step 1: Write completion summary**

Create `docs/plans/2025-11-16-visual-polish-completion.md`:

```markdown
# Visual Polish Phase 1: Completion Summary

**Date:** 2025-11-16
**Status:** ✅ COMPLETE

## Implemented Features

### 1. Thumbnail Generation
- **Script:** `scripts/generate_thumbnails.py`
- **Command:** `generate-thumbnails` (Nix shell alias)
- **Output:** 300x150 PNG files in `evio/data/.cache/thumbnails/`
- **Window:** First 1 second of events per dataset
- **Features:** Letterboxing, lazy loading, --force flag

### 2. Menu Thumbnail Rendering
- **Tile Size:** 300x150 pixels
- **Text Overlay:** Semi-transparent band with dataset name (white)
- **Selection:** Pink border (#ff66cc) on selected tile
- **Fallback:** Text-only tiles if thumbnail missing

### 3. Color Palette Application
- **Base:** Sensofusion military gray (#2b2b2b background)
- **Accent:** Y2K pink (#ff66cc selection border)
- **Menu:** Dark gray tiles, white text, light gray metadata
- **Status Bar:** Darker gray background (#1e1e1e)
- **HUD:** Semi-transparent black panel, white text
- **Help Overlay:** Dark gray background, white/gray text

### 4. Detector Overlay Colors
- **Fan Detector:** Green ellipse/RPM (#00ff99), yellow clusters (#ffff00)
- **Drone Detector:** Cyan boxes (#00ffff), orange warnings (#ff8800)

## Files Modified

- `scripts/generate_thumbnails.py` (NEW)
- `evio/scripts/mvp_launcher.py` (palette + thumbnails)
- `evio/scripts/detector_utils.py` (detection colors)
- `flake.nix` (Nix command integration)
- `.gitignore` (thumbnail cache)
- `docs/plans/ui-polishing.md` (status update)

## Usage

### Generate Thumbnails

```bash
nix develop
generate-thumbnails          # Generate missing thumbnails
generate-thumbnails --force  # Regenerate all
```

### Run Launcher

```bash
run-mvp-demo
```

## Testing Checklist

- [x] Thumbnail generation for all datasets
- [x] Menu displays thumbnails with pink selection
- [x] Fallback tiles work when thumbnails missing
- [x] All palette colors applied correctly
- [x] Fan detector shows green/yellow overlays
- [x] Drone detector shows cyan/orange overlays
- [x] HUD panel uses correct colors
- [x] Help overlay uses correct colors
- [x] --force flag regenerates thumbnails
- [x] Nix command integration works

## Next Steps (Phase 2)

See `docs/plans/ui-polishing.md` for Phase 2:
- Schema/metadata caching
- Preallocated frame buffers
- FPS throttling (~60 FPS)
- Async thumbnail generation

## Next Steps (Phase 3)

See `docs/plans/ui-polishing.md` for Phase 3:
- Playback speed control (↑/↓ arrow keys)
- Screenshot export (press 's')
- Frame export with overlays
```

**Step 2: Commit completion summary**

```bash
git add docs/plans/2025-11-16-visual-polish-completion.md
git commit -m "docs(visual-polish): add Phase 1 completion summary

Comprehensive summary of implemented features:
- Thumbnail generation and caching
- Menu rendering with thumbnails
- Sensofusion + Y2K palette application
- Detector overlay color updates

All tests passing, ready for Phase 2 or user demos.
"
```

**Step 3: Create git tag (optional)**

```bash
git tag -a visual-polish-phase1 -m "Visual Polish Phase 1: Thumbnails + Palette Complete

Features:
- Thumbnail generation (300x150 PNG cache)
- Menu rendering with preview images
- Sensofusion military gray + Y2K pink palette
- Updated detector colors (green/yellow fan, cyan/orange drone)

See docs/plans/2025-11-16-visual-polish-completion.md
"
```

**Step 4: Push commits**

```bash
git log --oneline -10
```

Expected: 9 commits from this implementation.

**Step 5: Final verification**

Run full test sequence:

```bash
# Clean slate
rm -rf evio/data/.cache/thumbnails/

# Generate thumbnails
generate-thumbnails

# Run launcher
run-mvp-demo
```

Expected: Polished launcher with thumbnails, pink accents, and correct colors throughout.

---

## Success Criteria

- ✅ `generate-thumbnails` command available in Nix shell
- ✅ Thumbnails generated for all `*_legacy.h5` datasets
- ✅ Menu displays thumbnail previews with minimal text overlay
- ✅ Pink Y2K accent (#ff66cc) used for selection border
- ✅ Sensofusion gray palette applied to menu, HUD, help, status bar
- ✅ Fan detector uses green ellipse, yellow clusters
- ✅ Drone detector uses cyan boxes, orange warnings
- ✅ Fallback tiles work when thumbnails missing
- ✅ --force flag regenerates thumbnails
- ✅ All text uses anti-aliased rendering (LINE_AA)

---

## Notes for Engineer

**Nix Environment:**
- ALWAYS run commands via `nix develop` for correct dependencies
- UV package manager only (never pip)
- OpenCV and HDF5 provided by Nix

**Color Format:**
- OpenCV uses BGR format (not RGB!)
- All color constants defined as (B, G, R) tuples
- Example: Pink #ff66cc = (204, 102, 255) in BGR

**Testing:**
- Test both with and without thumbnail cache
- Verify fallback tiles display correctly
- Test all detector types (fan and drone)
- Verify keyboard shortcuts (arrows, Enter, h, ESC, q)

**Common Issues:**
- If thumbnails don't show: check cache path exists
- If colors look wrong: verify BGR format (not RGB)
- If launcher won't start: ensure running in nix shell
- If evlib import fails: run `uv sync` in nix shell

**Commit Style:**
- Prefix: `feat(thumbnails):`, `feat(palette):`, `docs(visual-polish):`
- Keep commits atomic (one logical change per commit)
- Include Claude Code attribution footer

---

## References

- Design: `docs/plans/2025-11-16-visual-polish-thumbnails-palette-design.md`
- Original plan: `docs/plans/ui-polishing.md`
- MVP launcher: `evio/scripts/mvp_launcher.py`
- Detector utils: `evio/scripts/detector_utils.py`
- Nix config: `flake.nix`
