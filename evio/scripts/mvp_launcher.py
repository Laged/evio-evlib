#!/usr/bin/env python3
"""MVP Rendering Demo Launcher - Menu-driven event camera dataset explorer."""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import time
import evlib
import polars as pl


class AppMode(Enum):
    """Application state."""
    MENU = "menu"
    PLAYBACK = "playback"


@dataclass
class Dataset:
    """Dataset metadata."""
    path: Path
    name: str
    category: str
    size_mb: float


@dataclass
class PlaybackState:
    """Playback session state."""
    dataset: Dataset
    lazy_events: pl.LazyFrame  # CRITICAL: Lazy, not collected!
    width: int
    height: int
    t_min: int
    t_max: int
    current_t: int
    detector_type: str = "none"
    window_us: int = 10_000  # 10ms window
    speed: float = 1.0
    overlay_flags: Dict[str, bool] = None
    prev_fan_params: Optional[Tuple[int, int, float, float, float]] = None

    def __post_init__(self):
        if self.overlay_flags is None:
            self.overlay_flags = {
                "detector": True,
                "hud": True,
                "help": False,
            }


class MVPLauncher:
    """Main launcher application."""

    def __init__(self):
        self.mode = AppMode.MENU
        self.datasets: List[Dataset] = []
        self.selected_index = 0
        self.window_name = "Event Camera Demo"
        self.playback_state: Optional[PlaybackState] = None

        # Print banner
        print("=" * 60)
        print("  Event Camera MVP Launcher")
        print("=" * 60)
        print()
        print("Environment: Must run via 'nix develop' for HDF5/OpenGL deps")
        print("Command: uv run --package evio python evio/scripts/mvp_launcher.py")
        print()

        # Discover datasets on startup
        self.datasets = self.discover_datasets()
        print()

        if self.datasets:
            print(f"✓ Discovered {len(self.datasets)} datasets:")
            for i, ds in enumerate(self.datasets):
                print(f"  [{i}] {ds.name} ({ds.category}, {ds.size_mb:.1f} MB)")
        else:
            print("✗ No datasets found!")
            print()
            print("Prerequisites:")
            print("  1. Extract datasets: unzip-datasets")
            print("  2. Convert to HDF5: convert-all-legacy-to-hdf5")
            print()
            print("Or manually:")
            print("  convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat")
            print("  convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat")

        print()
        print("=" * 60)
        print()

    def discover_datasets(self) -> List[Dataset]:
        """Scan evio/data/ for *_legacy.h5 files."""
        data_dir = Path("evio/data")
        datasets = []

        if not data_dir.exists():
            print(f"Error: {data_dir} not found", file=sys.stderr)
            print("Please ensure you're running from repository root", file=sys.stderr)
            return datasets

        # Explicitly look for *_legacy.h5 files only (NOT _evt3.dat)
        h5_files = list(data_dir.rglob("*_legacy.h5"))

        if not h5_files:
            print("Warning: No *_legacy.h5 files found", file=sys.stderr)
            print("Run: convert-all-legacy-to-hdf5 to create HDF5 exports", file=sys.stderr)

        for h5_file in h5_files:
            # Skip if filename contains '_evt3' (defensive check)
            if '_evt3' in h5_file.stem:
                continue

            try:
                name = h5_file.stem.replace("_legacy", "").replace("_", " ").title()
                category = h5_file.parent.name
                size_mb = h5_file.stat().st_size / (1024 * 1024)

                datasets.append(Dataset(
                    path=h5_file,
                    name=name,
                    category=category,
                    size_mb=size_mb,
                ))
            except Exception as e:
                print(f"Warning: Failed to process {h5_file}: {e}", file=sys.stderr)
                continue

        # Sort by category then name
        datasets.sort(key=lambda d: (d.category, d.name))
        return datasets

    def _init_playback(self, dataset: Dataset) -> PlaybackState:
        """Load dataset and prepare playback state using LAZY windowing."""
        print(f"Loading {dataset.path}...")

        try:
            # CRITICAL: DON'T collect() here - keep lazy!
            lazy_events = evlib.load_events(str(dataset.path))

            # Only collect metadata needed for initialization
            # Use SQL-style aggregation - this is more efficient than collecting everything
            metadata = lazy_events.select([
                pl.col("x").max().alias("max_x"),
                pl.col("y").max().alias("max_y"),
                pl.col("t").min().alias("t_min"),
                pl.col("t").max().alias("t_max"),
            ]).collect()

            width = int(metadata["max_x"][0]) + 1
            height = int(metadata["max_y"][0]) + 1

            # Get time range (handle Duration vs Int64 vs timedelta)
            import datetime
            t_min_val = metadata["t_min"][0]
            t_max_val = metadata["t_max"][0]

            if isinstance(t_min_val, datetime.timedelta):
                # Python timedelta object
                t_min = int(t_min_val.total_seconds() * 1e6)
                t_max = int(t_max_val.total_seconds() * 1e6)
            elif isinstance(t_min_val, pl.Duration):
                # Polars Duration
                t_min = int(t_min_val.total_microseconds())
                t_max = int(t_max_val.total_microseconds())
            else:
                # Direct integer
                t_min = int(t_min_val)
                t_max = int(t_max_val)

            print(f"Resolution: {width}x{height}, "
                  f"Duration: {(t_max - t_min) / 1e6:.2f}s")

            # Determine detector type (will be implemented in Phase 3)
            detector_type = "none"  # TODO: Add _map_detector_type in Task 7
            print(f"Detector: {detector_type}")

            return PlaybackState(
                dataset=dataset,
                lazy_events=lazy_events,  # Store lazy reference, NOT collected!
                width=width,
                height=height,
                t_min=t_min,
                t_max=t_max,
                current_t=t_min,
                detector_type=detector_type,
            )

        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            raise

    def run(self) -> None:
        """Main application loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                if self.mode == AppMode.MENU:
                    if not self._menu_loop():
                        break
                elif self.mode == AppMode.PLAYBACK:
                    if not self._playback_loop():
                        break
        finally:
            cv2.destroyAllWindows()

    def _render_menu(self) -> np.ndarray:
        """Render menu grid with text tiles."""
        if not self.datasets:
            # No datasets - show message
            frame = np.full((480, 640, 3), (43, 43, 43), dtype=np.uint8)
            msg1 = "No datasets found"
            msg2 = "Run: convert-all-legacy-to-hdf5"
            cv2.putText(frame, msg1, (150, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, msg2, (100, 250), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1, cv2.LINE_AA)
            return frame

        # Grid parameters
        tile_width, tile_height = 300, 150
        margin = 20
        cols = 2
        rows = (len(self.datasets) + cols - 1) // cols

        # Calculate frame size
        frame_width = cols * tile_width + (cols + 1) * margin
        frame_height = rows * tile_height + (rows + 1) * margin + 60  # +60 for status bar

        # Dark gray background
        frame = np.full((frame_height, frame_width, 3), (43, 43, 43), dtype=np.uint8)

        # Draw tiles
        for i, dataset in enumerate(self.datasets):
            row = i // cols
            col = i % cols

            x = col * tile_width + (col + 1) * margin
            y = row * tile_height + (row + 1) * margin

            # Tile background color
            is_selected = (i == self.selected_index)
            if is_selected:
                tile_color = (226, 144, 74)  # Blue accent (BGR: #4a90e2)
                border_thickness = 3
            else:
                tile_color = (64, 64, 64)  # Medium gray
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
            # Line 1: Dataset name (white, larger)
            text_color = (255, 255, 255)
            meta_color = (180, 180, 180)

            name_text = dataset.name
            name_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            name_x = x + (tile_width - name_size[0]) // 2
            name_y = y + 50
            cv2.putText(frame, name_text, (name_x, name_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

            # Line 2: Category + size
            meta_text = f"{dataset.category} | {dataset.size_mb:.1f} MB"
            meta_size = cv2.getTextSize(meta_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            meta_x = x + (tile_width - meta_size[0]) // 2
            meta_y = y + 90
            cv2.putText(frame, meta_text, (meta_x, meta_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, meta_color, 1, cv2.LINE_AA)

        # Draw status bar at bottom
        status_y = frame_height - 30
        cv2.rectangle(frame, (0, status_y), (frame_width, frame_height),
                      (30, 30, 30), -1)

        status_text = "↑/↓/j/k: Navigate | Enter/Space: Play | Q/ESC: Quit"
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        status_x = (frame_width - status_size[0]) // 2
        cv2.putText(frame, status_text, (status_x, status_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def _menu_loop(self) -> bool:
        """Menu mode loop. Returns False to exit app."""
        frame = self._render_menu()
        cv2.imshow(self.window_name, frame)

        # Use shorter waitKey for more responsive input
        # 30ms = ~33 FPS polling rate
        key = cv2.waitKey(30) & 0xFF

        # Debug: print key code when pressed (except 255 which is no key)
        if key != 255:
            print(f"Key pressed: {key}")

        # Handle quit keys
        if key == ord('q'):
            print("Quitting application...")
            return False
        elif key == 27:  # ESC
            print("Quitting application...")
            return False

        # Handle navigation (check multiple key codes for cross-platform compatibility)
        elif key in (82, 0, ord('k')):  # Up arrow or 'k'
            if self.datasets:
                self.selected_index = (self.selected_index - 1) % len(self.datasets)
                print(f"Navigation: UP → Selected index {self.selected_index}")
        elif key in (84, 1, ord('j')):  # Down arrow or 'j'
            if self.datasets:
                self.selected_index = (self.selected_index + 1) % len(self.datasets)
                print(f"Navigation: DOWN → Selected index {self.selected_index}")

        # Handle selection
        elif key == 13 or key == ord(' '):  # Enter or Space
            if self.datasets:
                selected_dataset = self.datasets[self.selected_index]
                print(f"\nSelected: {selected_dataset.name}")
                try:
                    self.playback_state = self._init_playback(selected_dataset)
                    self.mode = AppMode.PLAYBACK
                except Exception as e:
                    error_msg = f"Failed to load dataset: {str(e)}"
                    print(error_msg, file=sys.stderr)
                    # Stay in menu mode on error

        return True

    def _playback_loop(self) -> bool:
        """Playback mode loop. Returns False to exit app."""
        if self.playback_state is None:
            self.mode = AppMode.MENU
            return True

        # TODO: Implement in Task 5 (Phase 2)
        # For now, just return to menu to prove init worked
        print("\nPlayback initialized successfully!")
        print("Returning to menu (full playback in Task 5)...")
        self.mode = AppMode.MENU
        self.playback_state = None
        return True


def main() -> None:
    """Entry point."""
    # Check environment
    import os
    if 'NIX_STORE' not in os.environ.get('PATH', ''):
        print("=" * 60)
        print("  WARNING: Not running in Nix environment!")
        print("=" * 60)
        print()
        print("This launcher requires HDF5 and OpenGL libraries.")
        print("Please run via:")
        print()
        print("  nix develop")
        print("  run-mvp-demo")
        print()
        print("Or:")
        print("  nix develop")
        print("  uv run --package evio python evio/scripts/mvp_launcher.py")
        print()
        print("Attempting to run anyway, but expect import errors...")
        print("=" * 60)
        print()

    launcher = MVPLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
