#!/usr/bin/env python3
"""MVP Rendering Demo Launcher - Menu-driven event camera dataset explorer."""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


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


class MVPLauncher:
    """Main launcher application."""

    def __init__(self):
        self.mode = AppMode.MENU
        self.datasets: List[Dataset] = []
        self.selected_index = 0
        self.window_name = "Event Camera Demo"

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

    def _menu_loop(self) -> bool:
        """Menu mode loop. Returns False to exit app."""
        # TODO: Implement in next task
        return False

    def _playback_loop(self) -> bool:
        """Playback mode loop. Returns False to exit app."""
        # TODO: Implement in Phase 2
        return False


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
