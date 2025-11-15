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
