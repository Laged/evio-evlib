#!/usr/bin/env python3
"""MVP Rendering Demo Launcher - Menu-driven event camera dataset explorer."""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import time
import evlib
import polars as pl
import psutil

# ============================================================================
# Performance Timing Instrumentation
# ============================================================================
# Enable to debug performance bottlenecks
ENABLE_TIMING = True

def log_timing(label: str, start_time: float) -> None:
    """Log timing information if ENABLE_TIMING is True."""
    if ENABLE_TIMING:
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        print(f"⏱️  {label}: {elapsed:.1f}ms")

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

# Try to import detector utilities - degrade gracefully if missing
try:
    # Try absolute import first (when running via uv run --package)
    try:
        from evio.scripts.detector_utils import (
            detect_fan,
            detect_drone,
            render_fan_overlay,
            render_drone_overlay,
            FanDetection,
            DroneDetection,
        )
    except ImportError:
        # Fall back to relative import (when running script directly)
        from detector_utils import (
            detect_fan,
            detect_drone,
            render_fan_overlay,
            render_drone_overlay,
            FanDetection,
            DroneDetection,
        )
    DETECTORS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Detector utilities not available: {e}", file=sys.stderr)
    print("Running in base playback mode only (no detector overlays)", file=sys.stderr)
    DETECTORS_AVAILABLE = False

    # Define stub functions to avoid NameError
    def detect_fan(*args, **kwargs):
        return None

    def detect_drone(*args, **kwargs):
        return None

    def render_fan_overlay(frame, *args, **kwargs):
        return frame

    def render_drone_overlay(frame, *args, **kwargs):
        return frame


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
    # PERFORMANCE: Cache metadata to skip expensive aggregation on load
    width: Optional[int] = None
    height: Optional[int] = None
    t_min: Optional[int] = None
    t_max: Optional[int] = None
    duration_sec: Optional[float] = None
    # PERFORMANCE: Optional LazyFrame cache (enabled with --enable-cache)
    lazy_events_cache: Optional[pl.LazyFrame] = None


@dataclass
class PlaybackState:
    """Playback session state."""
    dataset: Dataset
    lazy_events: pl.LazyFrame  # CRITICAL: Lazy, not collected!
    schema: dict  # CACHED: Schema resolved once at load time
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
    smoothed_rpm: float = 0.0  # Exponentially smoothed RPM for stable display

    def __post_init__(self):
        if self.overlay_flags is None:
            self.overlay_flags = {
                "detector": False,  # OFF by default - press '1' to enable
                "hud": False,       # OFF by default - press '2' to enable
                "help": False,      # OFF by default - press 'h' to show
            }


class MVPLauncher:
    """Main launcher application."""

    def __init__(self, enable_cache: bool = False, skip_frames: bool = False):
        self.mode = AppMode.MENU
        self.datasets: List[Dataset] = []
        self.selected_index = 0
        self.window_name = "Event Camera Demo"
        self.playback_state: Optional[PlaybackState] = None
        self.cache_enabled = enable_cache
        self.skip_frames = skip_frames
        self.frame_count = 0  # For frame skipping logic

        # Print banner
        print("=" * 60)
        print("  Event Camera MVP Launcher")
        print("=" * 60)
        print()
        print("Environment: Must run via 'nix develop' for HDF5/OpenGL deps")
        print("Command: uv run --package evio python evio/scripts/mvp_launcher.py")
        print()

        # Check caching configuration
        if self.cache_enabled:
            # Check available RAM
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            available_ram_gb = psutil.virtual_memory().available / (1024**3)

            print(f"Cache Mode: ENABLED")
            print(f"System RAM: {total_ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available")

            if available_ram_gb < 8:
                print("⚠️  WARNING: Low available RAM ({available_ram_gb:.1f} GB < 8 GB)")
                print("   Caching may cause memory pressure. Disabling cache.")
                self.cache_enabled = False
            else:
                print("✓ Sufficient RAM for caching (requires ~6.5 GB for all datasets)")
            print()
        else:
            print("Cache Mode: DISABLED (use --enable-cache to enable)")
            print()

        # Check frame skipping configuration
        if self.skip_frames:
            print("Frame Skipping: ENABLED")
            print("✓ High-speed playback optimized (render ~60 FPS, process at full speed)")
            print()
        else:
            print("Frame Skipping: DISABLED (use --skip-frames for high-speed playback)")
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

    def _extract_metadata(self, h5_path: Path) -> tuple[int, int, int, int, float]:
        """Extract metadata from HDF5 file (cached to avoid re-extraction on load).

        Returns:
            Tuple of (width, height, t_min, t_max, duration_sec)
        """
        lazy_events = evlib.load_events(str(h5_path))

        # Quick metadata extraction (same as in _init_playback)
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
            t_min = int(t_min_val.total_seconds() * 1e6)
            t_max = int(t_max_val.total_seconds() * 1e6)
        elif isinstance(t_min_val, pl.Duration):
            t_min = int(t_min_val.total_microseconds())
            t_max = int(t_max_val.total_microseconds())
        else:
            t_min = int(t_min_val)
            t_max = int(t_max_val)

        duration_sec = (t_max - t_min) / 1e6

        return width, height, t_min, t_max, duration_sec

    def discover_datasets(self) -> List[Dataset]:
        """Scan evio/data/ for *_legacy.h5 files (metadata extracted lazily on load)."""
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

                # PERFORMANCE: Don't extract metadata upfront (too slow at startup)
                # Metadata will be extracted on first load and cached in the Dataset object
                datasets.append(Dataset(
                    path=h5_file,
                    name=name,
                    category=category,
                    size_mb=size_mb,
                    # Leave metadata None - will be populated on first load
                ))
            except Exception as e:
                print(f"Warning: Failed to process {h5_file}: {e}", file=sys.stderr)
                continue

        # Sort by demo order: events, fan_const, fan_varying, fan_varying_turning, drone_idle, drone_moving
        demo_order = {
            "events": 0,
            "fan const rpm": 1,
            "fan varying rpm": 2,
            "fan varying rpm turning": 3,
            "drone idle": 4,
            "drone moving": 5,
        }

        def sort_key(d):
            # Use demo order if found, otherwise put at end (999)
            return demo_order.get(d.name.lower(), 999)

        datasets.sort(key=sort_key)
        return datasets

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

    def _map_detector_type(self, category: str) -> str:
        """Map dataset category to detector type."""
        if not DETECTORS_AVAILABLE:
            return "none"

        mapping = {
            "fan": "fan_rpm",
            "drone_idle": "drone",
            "drone_moving": "drone",
        }
        return mapping.get(category, "none")

    def _init_playback(self, dataset: Dataset) -> PlaybackState:
        """Load dataset and prepare playback state using LAZY windowing."""
        print(f"Loading {dataset.path}...")

        try:
            # PERFORMANCE: Check cache first (if enabled)
            if self.cache_enabled and dataset.lazy_events_cache is not None:
                t_step = time.perf_counter()
                lazy_events = dataset.lazy_events_cache
                log_timing("  └─ load from CACHE (instant!)", t_step)
            else:
                # CRITICAL: DON'T collect() here - keep lazy!
                t_step = time.perf_counter()
                lazy_events = evlib.load_events(str(dataset.path))
                log_timing("  └─ evlib.load_events()", t_step)

                # Cache if enabled
                if self.cache_enabled:
                    dataset.lazy_events_cache = lazy_events
                    print(f"  └─ Cached LazyFrame for future loads")

            # PERFORMANCE: Resolve schema ONCE and cache it
            t_step = time.perf_counter()
            schema = lazy_events.collect_schema()
            log_timing("  └─ collect_schema()", t_step)

            # PERFORMANCE: Use cached metadata if available (Option B optimization)
            if dataset.width is not None:
                # Metadata already cached from previous load
                width = dataset.width
                height = dataset.height
                t_min = dataset.t_min
                t_max = dataset.t_max
                print(f"Using cached metadata: {width}x{height}, {dataset.duration_sec:.2f}s")
            else:
                # First load: Extract and cache metadata
                t_step = time.perf_counter()
                metadata = lazy_events.select([
                    pl.col("x").max().alias("max_x"),
                    pl.col("y").max().alias("max_y"),
                    pl.col("t").min().alias("t_min"),
                    pl.col("t").max().alias("t_max"),
                ]).collect()
                log_timing("  └─ extract metadata (aggregation)", t_step)

                width = int(metadata["max_x"][0]) + 1
                height = int(metadata["max_y"][0]) + 1

                # Get time range (handle Duration vs Int64 vs timedelta)
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

                duration_sec = (t_max - t_min) / 1e6

                # Cache metadata back into Dataset object for next load
                dataset.width = width
                dataset.height = height
                dataset.t_min = t_min
                dataset.t_max = t_max
                dataset.duration_sec = duration_sec

                print(f"Resolution: {width}x{height}, Duration: {duration_sec:.2f}s")

            # Determine detector type
            detector_type = self._map_detector_type(dataset.category)
            print(f"Detector: {detector_type}")

            return PlaybackState(
                dataset=dataset,
                lazy_events=lazy_events,  # Store lazy reference, NOT collected!
                schema=schema,  # CACHED: Resolved once, reused every frame
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
            frame = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
            msg1 = "No datasets found"
            msg2 = "Run: convert-all-legacy-to-hdf5"
            cv2.putText(frame, msg1, (150, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, TEXT_PRIMARY, 2, cv2.LINE_AA)
            cv2.putText(frame, msg2, (100, 250), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, TEXT_SECONDARY, 1, cv2.LINE_AA)
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
        frame = np.full((frame_height, frame_width, 3), BG_COLOR, dtype=np.uint8)

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

        # Draw status bar at bottom
        status_y = frame_height - 30
        cv2.rectangle(frame, (0, status_y), (frame_width, frame_height),
                      STATUS_BAR_BG, -1)

        status_text = "↑/↓/j/k: Navigate | Enter/Space: Play | Q/ESC: Quit"
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        status_x = (frame_width - status_size[0]) // 2
        cv2.putText(frame, status_text, (status_x, status_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_TEXT, 1, cv2.LINE_AA)

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
                t_start_transition = time.perf_counter()
                selected_dataset = self.datasets[self.selected_index]
                print(f"\nSelected: {selected_dataset.name}")
                try:
                    t_start_init = time.perf_counter()
                    self.playback_state = self._init_playback(selected_dataset)
                    log_timing("Total _init_playback()", t_start_init)

                    t_start_mode_switch = time.perf_counter()
                    self.mode = AppMode.PLAYBACK
                    log_timing("Mode switch to PLAYBACK", t_start_mode_switch)

                    log_timing("TOTAL Enter → Playback Ready", t_start_transition)
                except Exception as e:
                    error_msg = f"Failed to load dataset: {str(e)}"
                    print(error_msg, file=sys.stderr)
                    self._show_error_and_return_to_menu(error_msg)

        return True

    def _get_event_window(
        self,
        lazy_events: pl.LazyFrame,
        schema: dict,
        win_start_us: int,
        win_end_us: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract window of events using LAZY polars filtering.

        CRITICAL: This uses lazy_events.filter().collect() to collect ONLY the window,
        NOT the full dataset. This prevents OOM on large files.

        Args:
            lazy_events: Lazy polars dataframe
            schema: CACHED schema (resolved once at load time)
            win_start_us: Window start time (microseconds)
            win_end_us: Window end time (microseconds)
        """
        # Use cached schema to determine time column type
        t_dtype = schema["t"]

        # Apply filter lazily, then collect ONLY the filtered window
        if isinstance(t_dtype, pl.Duration):
            # Convert microseconds to Duration
            window = lazy_events.filter(
                (pl.col("t") >= pl.duration(microseconds=win_start_us)) &
                (pl.col("t") < pl.duration(microseconds=win_end_us))
            ).collect()  # Collect ONLY the filtered window
        else:
            # Direct integer filtering
            window = lazy_events.filter(
                (pl.col("t") >= win_start_us) &
                (pl.col("t") < win_end_us)
            ).collect()  # Collect ONLY the filtered window

        if len(window) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=bool),
            )

        x_coords = window["x"].to_numpy().astype(np.int32)
        y_coords = window["y"].to_numpy().astype(np.int32)
        polarity_values = window["polarity"].to_numpy()
        polarities_on = polarity_values > 0

        return x_coords, y_coords, polarities_on

    def _render_polarity_frame(
        self,
        window: tuple[np.ndarray, np.ndarray, np.ndarray],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Render polarity events to frame."""
        x_coords, y_coords, polarities_on = window

        # Base gray, white for ON, black for OFF
        frame = np.full((height, width, 3), (127, 127, 127), dtype=np.uint8)

        if len(x_coords) > 0:
            frame[y_coords[polarities_on], x_coords[polarities_on]] = (255, 255, 255)
            frame[y_coords[~polarities_on], x_coords[~polarities_on]] = (0, 0, 0)

        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        state: PlaybackState,
        fps: float,
        wall_start: float,
    ) -> None:
        """Draw HUD overlay on frame (bottom-right)."""
        if not state.overlay_flags.get("hud", True):
            return

        h, w = frame.shape[:2]

        # Larger semi-transparent panel background
        panel_w, panel_h = 350, 180
        panel_x, panel_y = w - panel_w - 10, h - panel_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      HUD_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text content - all text same size (0.7), FPS at bottom
        wall_time_s = time.perf_counter() - wall_start
        rec_time_s = (state.current_t - state.t_min) / 1e6

        # Smart window formatting: show μs for < 1ms, ms otherwise
        if state.window_us < 1000:
            window_display = f"{state.window_us}μs"
        else:
            window_display = f"{state.window_us/1000:.1f}ms"

        # All text same size (0.7 scale, thickness 2) with FPS at bottom
        lines = [
            ("SPEED:", f"{state.speed:.2f}x", 0.7, 2),
            ("WINDOW:", window_display, 0.7, 2),
            ("Time:", f"{rec_time_s:.2f}s", 0.7, 2),
            ("Dataset:", f"{state.dataset.category}", 0.7, 2),
            ("FPS:", f"{fps:.1f}", 0.7, 2),  # Moved to bottom
        ]

        y_offset = panel_y + 25
        for label, value, font_scale, thickness in lines:
            # Draw label
            cv2.putText(frame, label, (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, HUD_TEXT,
                        thickness, cv2.LINE_AA)
            # Draw value (right-aligned)
            text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            value_x = panel_x + panel_w - text_size[0] - 10
            cv2.putText(frame, value, (value_x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, HUD_TEXT,
                        thickness, cv2.LINE_AA)
            y_offset += int(30 * font_scale)

    def _draw_help_overlay(self, frame: np.ndarray) -> None:
        """Draw help overlay with keybindings."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay (bottom third)
        overlay_h = h // 3
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - overlay_h), (w, h), HELP_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Help text
        help_lines = [
            "KEYBOARD SHORTCUTS",
            "",
            "1 - Toggle detector overlay",
            "2 - Toggle HUD",
            "h - Toggle this help",
            "",
            "Up/Down - Increase/Decrease speed",
            "Left/Right - Smaller/Larger event window",
            "",
            "ESC - Return to menu",
            "q - Quit application",
        ]

        y_start = h - overlay_h + 30
        for i, line in enumerate(help_lines):
            font_scale = 0.7 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            color = HELP_TITLE if i == 0 else HELP_TEXT

            cv2.putText(frame, line, (30, y_start + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                        thickness, cv2.LINE_AA)

    def _show_error_and_return_to_menu(self, error_msg: str) -> None:
        """Show error message for 3 seconds then return to menu."""
        frame = np.full((480, 640, 3), HELP_BG, dtype=np.uint8)

        # Error title
        cv2.putText(frame, "ERROR", (270, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Error message (word wrap)
        words = error_msg.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            if size[0] > 580:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)

        y = 250
        for line in lines[:4]:  # Max 4 lines
            cv2.putText(frame, line, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, HELP_TEXT, 1, cv2.LINE_AA)
            y += 30

        # Show for 3 seconds
        for _ in range(30):
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(100) & 0xFF in (ord('q'), 27):
                break

        self.mode = AppMode.MENU

    def _playback_loop(self) -> bool:
        """Playback mode loop. Returns False to exit app."""
        if self.playback_state is None:
            self.mode = AppMode.MENU
            return True

        state = self.playback_state

        # Track wall time for FPS and speed control
        if not hasattr(self, '_playback_wall_start'):
            self._playback_wall_start = time.perf_counter()
            self._last_frame_time = self._playback_wall_start
            self._fps = 0.0
            self._first_frame = True
        else:
            self._first_frame = False

        # Track first frame rendering time
        if self._first_frame:
            t_first_frame_start = time.perf_counter()

        # Extract events for current window
        t_step = time.perf_counter()
        win_end = min(state.current_t + state.window_us, state.t_max)
        window = self._get_event_window(state.lazy_events, state.schema, state.current_t, win_end)
        if self._first_frame:
            log_timing("  └─ get_event_window() [first frame]", t_step)

        # Render base polarity frame
        frame = self._render_polarity_frame(window, state.width, state.height)

        # Apply detector overlays if enabled AND available
        if state.overlay_flags.get("detector", True) and DETECTORS_AVAILABLE:
            try:
                if state.detector_type == "fan_rpm":
                    detection = detect_fan(
                        window,
                        state.width,
                        state.height,
                        state.window_us,
                        state.prev_fan_params,
                    )
                    state.prev_fan_params = (detection.cx, detection.cy,
                                              detection.a, detection.b, detection.phi)
                    frame = render_fan_overlay(frame, detection)

                elif state.detector_type == "drone":
                    detection = detect_drone(window, state.width, state.height)
                    frame = render_drone_overlay(frame, detection)

            except Exception as e:
                # Detector crashed - disable overlays and show warning
                print(f"Detector error: {e}", file=sys.stderr)
                state.overlay_flags["detector"] = False
                cv2.putText(frame, "Detector disabled (error)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        elif state.overlay_flags.get("detector", True) and not DETECTORS_AVAILABLE:
            # Show warning that detectors aren't available
            cv2.putText(frame, "Detectors not available", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)

        # Draw HUD
        now = time.perf_counter()
        frame_delta = now - self._last_frame_time
        if frame_delta > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / frame_delta)
        self._last_frame_time = now

        self._draw_hud(frame, state, self._fps, self._playback_wall_start)

        # Draw help overlay if enabled
        if state.overlay_flags.get("help", False):
            self._draw_help_overlay(frame)

        # Frame skipping for high-speed playback
        # When enabled, only render every Nth frame to maintain ~60 FPS display
        # while processing at full speed
        should_render = True
        if self.skip_frames:
            # Calculate skip interval based on speed
            # Target: ~60 FPS display regardless of processing speed
            if state.speed > 10:
                skip_interval = max(1, int(state.speed / 2))  # e.g., 100x speed -> skip 50 frames
            else:
                skip_interval = 1  # No skipping at low speeds

            should_render = (self.frame_count % skip_interval == 0)
            self.frame_count += 1

        # Display (only if should_render or frame skipping disabled)
        if should_render or not self.skip_frames:
            t_step = time.perf_counter()
            cv2.imshow(self.window_name, frame)
            if self._first_frame:
                log_timing("  └─ cv2.imshow() [first frame]", t_step)
                log_timing("TOTAL First Frame Render", t_first_frame_start)

        # Handle input with 1ms waitKey for responsiveness (critical-fixes.md section 4)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return False  # Quit app
        elif key == 27:  # ESC
            print("\nReturning to menu...")
            self.mode = AppMode.MENU
            self.playback_state = None
            if hasattr(self, '_playback_wall_start'):
                delattr(self, '_playback_wall_start')
            return True
        elif key == ord('1'):
            state.overlay_flags["detector"] = not state.overlay_flags["detector"]
            print(f"Detector overlay: {'ON' if state.overlay_flags['detector'] else 'OFF'}")
        elif key == ord('2'):
            state.overlay_flags["hud"] = not state.overlay_flags["hud"]
            print(f"HUD: {'ON' if state.overlay_flags['hud'] else 'OFF'}")
        elif key == ord('h') or key == ord('H'):
            state.overlay_flags["help"] = not state.overlay_flags["help"]
            print(f"Help: {'ON' if state.overlay_flags['help'] else 'OFF'}")

        # Playback speed control (↑/↓ arrows)
        # Sensible steps: 0.25x, 0.5x, 1x, 2x, 5x, 10x, 20x, 50x, 100x
        elif key in (82, 0):  # Up arrow - increase speed
            speed_steps = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            # Find next higher step
            for step in speed_steps:
                if state.speed < step:
                    state.speed = step
                    break
            print(f"Playback speed: {state.speed:.2f}x")
        elif key in (84, 1):  # Down arrow - decrease speed
            speed_steps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25]
            # Find next lower step
            for step in speed_steps:
                if state.speed > step:
                    state.speed = step
                    break
            print(f"Playback speed: {state.speed:.2f}x")

        # Event window size control (←/→ arrows)
        # Smooth 10-step transition: 10μs -> 1ms, then coarser steps up to 100ms
        # 10 steps: 10μs, 20μs, 50μs, 100μs, 200μs, 500μs, 1ms, 3ms, 10ms, 100ms
        elif key in (83, 2):  # Right arrow - larger window
            window_us = state.window_us
            # Predefined smooth steps from 10μs to 100ms
            steps = [10, 20, 50, 100, 200, 500, 1000, 3000, 10000, 100000]

            # Find next larger step
            for step in steps:
                if window_us < step:
                    window_us = step
                    break

            state.window_us = window_us
            # Smart formatting: show μs for < 1ms, ms otherwise
            if window_us < 1000:
                print(f"Event window: {window_us}μs")
            else:
                print(f"Event window: {window_us/1000:.1f}ms")
        elif key in (81, 3):  # Left arrow - smaller window
            window_us = state.window_us
            # Predefined smooth steps from 100ms to 10μs
            steps = [100000, 10000, 3000, 1000, 500, 200, 100, 50, 20, 10]

            # Find next smaller step
            for step in steps:
                if window_us > step:
                    window_us = step
                    break

            state.window_us = window_us
            # Smart formatting: show μs for < 1ms, ms otherwise
            if window_us < 1000:
                print(f"Event window: {window_us}μs")
            else:
                print(f"Event window: {window_us/1000:.1f}ms")

        # Advance time
        state.current_t += state.window_us

        # Auto-loop at end
        if state.current_t >= state.t_max:
            state.current_t = state.t_min
            self._playback_wall_start = time.perf_counter()
            print("Auto-looping to start...")

        # Speed control (simple sleep-based)
        expected_wall_time = (state.current_t - state.t_min) / (1e6 * state.speed)
        actual_wall_time = time.perf_counter() - self._playback_wall_start
        sleep_time = expected_wall_time - actual_wall_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        return True


def main() -> None:
    """Entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Event Camera MVP Launcher - Menu-driven dataset explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default mode (no caching, 3-19s load time)
  uv run --package evio python evio/scripts/mvp_launcher.py

  # Enable caching (requires ~6.5 GB RAM, <100ms re-load time)
  uv run --package evio python evio/scripts/mvp_launcher.py --enable-cache

Notes:
  - Caching stores LazyFrames in memory for instant re-loads
  - First load still takes 3-19s (evlib limitation)
  - Subsequent loads are nearly instant with caching
  - Requires 8+ GB available RAM for safe operation
        """
    )
    parser.add_argument(
        '--enable-cache',
        action='store_true',
        help='Cache LazyFrames in memory for faster re-loads (requires ~6.5 GB RAM)'
    )
    parser.add_argument(
        '--skip-frames',
        action='store_true',
        help='Enable frame skipping for high-speed playback (maintains ~60 FPS display)'
    )
    args = parser.parse_args()

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

    launcher = MVPLauncher(enable_cache=args.enable_cache, skip_frames=args.skip_frames)
    launcher.run()


if __name__ == "__main__":
    main()
