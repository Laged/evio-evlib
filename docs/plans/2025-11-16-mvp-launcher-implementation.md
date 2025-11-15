# MVP Rendering Demo Launcher - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a polished, menu-driven launcher for event camera dataset exploration with detector overlays

**Architecture:** Top-down implementation (menu → playback → detectors) using cv2-only rendering. State machine with MENU and PLAYBACK modes in a single window. Auto-discovery of HDF5 datasets, hardcoded detector mapping.

**Tech Stack:** Python 3.11, cv2 (OpenCV), evlib, polars, numpy, scikit-learn (existing dependencies)

**Reference:** `docs/plans/2025-11-16-mvp-rendering-demo-design.md`

---

## Phase 1: Menu with Auto-Discovery

**Goal:** See the app structure with dataset selection grid

### Task 1: Create menu module skeleton

**Files:**
- Create: `evio/scripts/mvp_launcher.py`

**Step 1: Create basic application skeleton**

Create `evio/scripts/mvp_launcher.py`:

```python
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
    launcher = MVPLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
```

**Step 2: Test skeleton runs**

Run from repo root:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Window opens briefly and closes (returns False immediately)

**Step 3: Commit skeleton**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add launcher skeleton with state machine"
```

---

### Task 2: Implement dataset auto-discovery

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add dataset discovery function**

Add to `MVPLauncher` class in `evio/scripts/mvp_launcher.py`:

```python
def discover_datasets(self) -> List[Dataset]:
    """Scan evio/data/ for *_legacy.h5 files."""
    data_dir = Path("evio/data")
    datasets = []

    if not data_dir.exists():
        print(f"Warning: {data_dir} not found", file=sys.stderr)
        return datasets

    for h5_file in data_dir.rglob("*_legacy.h5"):
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
```

**Step 2: Load datasets on init**

Modify `__init__` in `MVPLauncher`:

```python
def __init__(self):
    self.mode = AppMode.MENU
    self.datasets: List[Dataset] = []
    self.selected_index = 0
    self.window_name = "Event Camera Demo"

    # Discover datasets on startup
    self.datasets = self.discover_datasets()
    print(f"Discovered {len(self.datasets)} datasets")
    for i, ds in enumerate(self.datasets):
        print(f"  [{i}] {ds.name} ({ds.category}, {ds.size_mb:.1f} MB)")
```

**Step 3: Test discovery**

Run:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected output:
```
Discovered 4 datasets
  [0] Drone Idle (drone_idle, XXX.X MB)
  [1] Fan Const Rpm (fan, XXX.X MB)
  [2] Fan Varying Rpm (fan, XXX.X MB)
  [3] Fan Varying Rpm Turning (fan, XXX.X MB)
```

**Step 4: Commit discovery**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add dataset auto-discovery from evio/data/"
```

---

### Task 3: Implement menu rendering (text tiles)

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add menu rendering function**

Add to `MVPLauncher` class:

```python
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

    status_text = "↑/↓ Navigate | Enter: Play | Q: Quit"
    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    status_x = (frame_width - status_size[0]) // 2
    cv2.putText(frame, status_text, (status_x, status_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
```

**Step 2: Implement menu loop with rendering**

Replace `_menu_loop` in `MVPLauncher`:

```python
def _menu_loop(self) -> bool:
    """Menu mode loop. Returns False to exit app."""
    frame = self._render_menu()
    cv2.imshow(self.window_name, frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q') or key == 27:  # q or ESC
        return False
    elif key == 82 or key == 0:  # Up arrow (key codes vary by platform)
        if self.datasets:
            self.selected_index = (self.selected_index - 1) % len(self.datasets)
    elif key == 84 or key == 1:  # Down arrow
        if self.datasets:
            self.selected_index = (self.selected_index + 1) % len(self.datasets)
    elif key == 13:  # Enter
        if self.datasets:
            print(f"Selected: {self.datasets[self.selected_index].name}")
            self.mode = AppMode.PLAYBACK

    return True
```

**Step 3: Test menu rendering**

Run:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Menu grid appears with dataset tiles. Arrow keys navigate (highlight changes). Enter prints selected dataset. Q/ESC exits.

**Step 4: Commit menu rendering**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add menu rendering with text tiles and navigation"
```

---

## Phase 2: Basic Playback (No Detectors)

**Goal:** Prove playback loop works with state transitions

### Task 4: Add playback state and initialization

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add playback imports**

Add to top of `evio/scripts/mvp_launcher.py`:

```python
import time
import evlib
import polars as pl
```

**Step 2: Add playback state dataclass**

Add after `Dataset` dataclass:

```python
@dataclass
class PlaybackState:
    """Playback session state."""
    dataset: Dataset
    events: pl.DataFrame
    width: int
    height: int
    t_min: int
    t_max: int
    current_t: int
    window_us: int = 10_000  # 10ms window
    speed: float = 1.0
    overlay_flags: Dict[str, bool] = None

    def __post_init__(self):
        if self.overlay_flags is None:
            self.overlay_flags = {
                "detector": True,
                "hud": True,
                "help": False,
            }
```

**Step 3: Add playback state to launcher**

Modify `MVPLauncher.__init__`:

```python
def __init__(self):
    self.mode = AppMode.MENU
    self.datasets: List[Dataset] = []
    self.selected_index = 0
    self.window_name = "Event Camera Demo"
    self.playback_state: Optional[PlaybackState] = None

    # Discover datasets on startup
    self.datasets = self.discover_datasets()
    print(f"Discovered {len(self.datasets)} datasets")
    for i, ds in enumerate(self.datasets):
        print(f"  [{i}] {ds.name} ({ds.category}, {ds.size_mb:.1f} MB)")
```

**Step 4: Add playback initialization function**

Add to `MVPLauncher` class:

```python
def _init_playback(self, dataset: Dataset) -> PlaybackState:
    """Load dataset and prepare playback state."""
    print(f"Loading {dataset.path}...")

    try:
        # Load events with evlib
        lazy_events = evlib.load_events(str(dataset.path))
        events = lazy_events.collect()

        # Get resolution
        width = int(events["x"].max()) + 1
        height = int(events["y"].max()) + 1

        # Get time range (handle Duration vs Int64)
        schema = events.schema
        t_dtype = schema["t"]

        if isinstance(t_dtype, pl.Duration):
            t_min = int(events["t"].dt.total_microseconds().min())
            t_max = int(events["t"].dt.total_microseconds().max())
        else:
            t_min = int(events["t"].min())
            t_max = int(events["t"].max())

        print(f"Loaded {len(events):,} events, {width}x{height}, "
              f"{(t_max - t_min) / 1e6:.2f}s duration")

        return PlaybackState(
            dataset=dataset,
            events=events,
            width=width,
            height=height,
            t_min=t_min,
            t_max=t_max,
            current_t=t_min,
        )

    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        raise
```

**Step 5: Update menu loop to initialize playback**

Modify `_menu_loop` Enter key handler:

```python
elif key == 13:  # Enter
    if self.datasets:
        selected_dataset = self.datasets[self.selected_index]
        print(f"Selected: {selected_dataset.name}")
        try:
            self.playback_state = self._init_playback(selected_dataset)
            self.mode = AppMode.PLAYBACK
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            # Stay in menu mode
```

**Step 6: Test playback initialization**

Run and select a dataset:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Menu shows, press Enter → dataset loads, prints event count and resolution, then returns to menu (playback loop still returns False).

**Step 7: Commit playback init**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add playback initialization with evlib loading"
```

---

### Task 5: Implement basic event rendering

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add event window extraction function**

Add to `MVPLauncher` class:

```python
def _get_event_window(
    self,
    events: pl.DataFrame,
    win_start_us: int,
    win_end_us: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract window of events using polars filtering."""
    schema = events.schema
    t_dtype = schema["t"]

    if isinstance(t_dtype, pl.Duration):
        # Convert microseconds to Duration
        window = events.filter(
            (pl.col("t") >= pl.duration(microseconds=win_start_us)) &
            (pl.col("t") < pl.duration(microseconds=win_end_us))
        )
    else:
        # Direct integer filtering
        window = events.filter(
            (pl.col("t") >= win_start_us) &
            (pl.col("t") < win_end_us)
        )

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
```

**Step 2: Add base frame rendering**

Add to `MVPLauncher` class:

```python
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
```

**Step 3: Add simple HUD**

Add to `MVPLauncher` class:

```python
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

    # Semi-transparent panel background
    panel_w, panel_h = 280, 100
    panel_x, panel_y = w - panel_w - 10, h - panel_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text content
    wall_time_s = time.perf_counter() - wall_start
    rec_time_s = (state.current_t - state.t_min) / 1e6

    lines = [
        f"FPS: {fps:.1f}",
        f"Speed: {state.speed:.2f}x",
        f"Recording: {rec_time_s:.2f}s",
        f"Dataset: {state.dataset.category}",
    ]

    y_offset = panel_y + 25
    for line in lines:
        cv2.putText(frame, line, (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        y_offset += 22
```

**Step 4: Implement playback loop**

Replace `_playback_loop` in `MVPLauncher`:

```python
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

    # Extract events for current window
    win_end = min(state.current_t + state.window_us, state.t_max)
    window = self._get_event_window(state.events, state.current_t, win_end)

    # Render base polarity frame
    frame = self._render_polarity_frame(window, state.width, state.height)

    # TODO: Add detector overlays in Phase 3

    # Draw HUD
    now = time.perf_counter()
    frame_delta = now - self._last_frame_time
    if frame_delta > 0:
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / frame_delta)
    self._last_frame_time = now

    self._draw_hud(frame, state, self._fps, self._playback_wall_start)

    # Display
    cv2.imshow(self.window_name, frame)

    # Handle input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        return False  # Quit app
    elif key == 27:  # ESC
        self.mode = AppMode.MENU
        self.playback_state = None
        delattr(self, '_playback_wall_start')
        return True
    elif key == ord('1'):
        state.overlay_flags["detector"] = not state.overlay_flags["detector"]
    elif key == ord('2'):
        state.overlay_flags["hud"] = not state.overlay_flags["hud"]
    elif key == ord('h'):
        state.overlay_flags["help"] = not state.overlay_flags["help"]

    # Advance time
    state.current_t += state.window_us

    # Auto-loop at end
    if state.current_t >= state.t_max:
        state.current_t = state.t_min
        self._playback_wall_start = time.perf_counter()

    # Speed control (simple sleep-based)
    expected_wall_time = (state.current_t - state.t_min) / (1e6 * state.speed)
    actual_wall_time = time.perf_counter() - self._playback_wall_start
    sleep_time = expected_wall_time - actual_wall_time
    if sleep_time > 0:
        time.sleep(sleep_time)

    return True
```

**Step 5: Test basic playback**

Run and play a dataset:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Select dataset → playback shows polarity events (white/black on gray), HUD in bottom-right, auto-loops at end. ESC returns to menu.

**Step 6: Commit basic playback**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add basic playback loop with polarity rendering and HUD"
```

---

## Phase 3: Detector Overlays

**Goal:** Integrate existing detector work

### Task 6: Create detector utilities module

**Files:**
- Create: `evio/scripts/detector_utils.py`

**Step 1: Extract fan detector core functions**

Create `evio/scripts/detector_utils.py`:

```python
"""Detector utilities - refactored from fan_detector_demo.py and drone_detector_demo.py."""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import cv2
from sklearn.cluster import DBSCAN


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
    """Build grayscale accumulation frame from events."""
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
    """Fit ellipse to largest contour in accumulated frame."""
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
    """Cluster events near blade ellipse using DBSCAN."""
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
    """Estimate RPM from number of blade clusters."""
    if not clusters:
        return 0.0

    # Assume each cluster is a blade
    num_blades = len(clusters)
    if num_blades == 0:
        return 0.0

    # Simple estimate: if we see N blades in window, assume full rotation
    # This is a rough approximation - real RPM tracking needs temporal tracking
    window_s = window_duration_us / 1e6
    if window_s <= 0:
        return 0.0

    # Assume 3 blades per rotation (typical fan)
    rotations_per_sec = num_blades / 3.0 / window_s
    rpm = rotations_per_sec * 60.0

    return max(0.0, min(rpm, 10000.0))  # Clamp to reasonable range


def detect_fan(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    window_us: int,
    prev_params: Optional[Tuple[int, int, float, float, float]] = None,
) -> FanDetection:
    """Run fan detection: ellipse fit + blade clustering + RPM estimate."""
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
    """Draw fan detection overlay on base frame."""
    frame = base_frame.copy()

    # Draw ellipse (cyan)
    center = (detection.cx, detection.cy)
    axes = (int(detection.a), int(detection.b))
    angle_deg = np.rad2deg(detection.phi)
    cv2.ellipse(frame, center, axes, angle_deg, 0, 360, (255, 255, 0), 2)  # Cyan in BGR

    # Draw center
    cv2.circle(frame, center, 5, (255, 255, 0), -1)

    # Draw blade clusters (yellow circles)
    for xc, yc in detection.clusters:
        cv2.circle(frame, (int(xc), int(yc)), 8, (0, 255, 255), 2)  # Yellow in BGR

    # Draw RPM text (green)
    rpm_text = f"RPM: {detection.rpm:.0f}"
    cv2.putText(frame, rpm_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)  # Green

    return frame


# ============================================================================
# Drone Detector (placeholder - to be extracted similarly)
# ============================================================================

@dataclass
class DroneDetection:
    """Drone detection result (placeholder)."""
    boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    warning: bool


def detect_drone(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> DroneDetection:
    """Run drone detection (placeholder - simplified from drone_detector_demo.py)."""
    # TODO: Extract real logic from drone_detector_demo.py
    # For MVP, return empty detection
    return DroneDetection(boxes=[], warning=False)


def render_drone_overlay(
    base_frame: np.ndarray,
    detection: DroneDetection,
) -> np.ndarray:
    """Draw drone detection overlay (placeholder)."""
    frame = base_frame.copy()

    # Draw bounding boxes (red)
    for x, y, w, h in detection.boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red

    # Draw warning if present (orange)
    if detection.warning:
        h, w = frame.shape[:2]
        cv2.putText(frame, "DRONE DETECTED", (w // 2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3, cv2.LINE_AA)

    return frame
```

**Step 2: Test imports**

Run:
```bash
python3 -c "from evio.scripts.detector_utils import detect_fan, detect_drone; print('OK')"
```

Expected: `OK`

**Step 3: Commit detector utils**

```bash
git add evio/scripts/detector_utils.py
git commit -m "feat(mvp): extract fan detector utilities (ellipse, DBSCAN, RPM)"
```

---

### Task 7: Wire up fan detector

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add detector imports and mapping**

Add to top of `evio/scripts/mvp_launcher.py`:

```python
from evio.scripts.detector_utils import (
    detect_fan,
    detect_drone,
    render_fan_overlay,
    render_drone_overlay,
    FanDetection,
    DroneDetection,
)
```

Add to `MVPLauncher` class:

```python
def _map_detector_type(self, category: str) -> str:
    """Map dataset category to detector type."""
    mapping = {
        "fan": "fan_rpm",
        "drone_idle": "drone",
        "drone_moving": "drone",
    }
    return mapping.get(category, "none")
```

**Step 2: Add detector state tracking**

Modify `PlaybackState` to add detector type and prev params:

```python
@dataclass
class PlaybackState:
    """Playback session state."""
    dataset: Dataset
    events: pl.DataFrame
    width: int
    height: int
    t_min: int
    t_max: int
    current_t: int
    detector_type: str = "none"
    window_us: int = 10_000
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
```

**Step 3: Set detector type on init**

Modify `_init_playback` to set detector type:

```python
def _init_playback(self, dataset: Dataset) -> PlaybackState:
    """Load dataset and prepare playback state."""
    print(f"Loading {dataset.path}...")

    try:
        # Load events with evlib
        lazy_events = evlib.load_events(str(dataset.path))
        events = lazy_events.collect()

        # Get resolution
        width = int(events["x"].max()) + 1
        height = int(events["y"].max()) + 1

        # Get time range
        schema = events.schema
        t_dtype = schema["t"]

        if isinstance(t_dtype, pl.Duration):
            t_min = int(events["t"].dt.total_microseconds().min())
            t_max = int(events["t"].dt.total_microseconds().max())
        else:
            t_min = int(events["t"].min())
            t_max = int(events["t"].max())

        # Determine detector type
        detector_type = self._map_detector_type(dataset.category)

        print(f"Loaded {len(events):,} events, {width}x{height}, "
              f"{(t_max - t_min) / 1e6:.2f}s duration, detector: {detector_type}")

        return PlaybackState(
            dataset=dataset,
            events=events,
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
```

**Step 4: Add detector overlay rendering to playback loop**

Modify `_playback_loop` to run detectors (replace `# TODO: Add detector overlays`):

```python
# Render base polarity frame
frame = self._render_polarity_frame(window, state.width, state.height)

# Apply detector overlays if enabled
if state.overlay_flags.get("detector", True):
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

# Draw HUD
# ... (rest of existing code)
```

**Step 5: Test fan detector**

Run and select a fan dataset:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Fan dataset shows cyan ellipse, yellow blade clusters, green RPM text. Press `1` to toggle detector overlay on/off.

**Step 6: Commit fan detector integration**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): integrate fan detector with ellipse and RPM overlay"
```

---

### Task 8: Extract and wire up drone detector

**Files:**
- Modify: `evio/scripts/detector_utils.py`

**Step 1: Extract drone detector logic from drone_detector_demo.py**

This step requires extracting the propeller detection logic from `drone_detector_demo.py`. The file is 659 lines, so we'll create a simplified version that captures the key detection logic.

Replace the placeholder `detect_drone` and `render_drone_overlay` in `detector_utils.py`:

```python
def propeller_mask_from_frame(
    accum_frame: np.ndarray,
    min_area: float = 145.0,
    max_area_frac: float = 0.01,
    top_k: int = 2,
) -> Tuple[List[Tuple[int, int, float, float, float]], np.ndarray]:
    """Detect propeller-like blobs in accumulated frame."""
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
    """Run drone detection - detect propeller blobs."""
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
    """Draw drone detection overlay."""
    frame = base_frame.copy()

    # Draw bounding boxes (red)
    for x, y, w, h in detection.boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw warning if present (orange)
    if detection.warning:
        h_frame, w_frame = frame.shape[:2]

        # Warning text at top
        cv2.putText(frame, "⚠ DRONE DETECTED", (w_frame // 2 - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 136, 255), 2, cv2.LINE_AA)

        # Count at bottom-right
        count_text = f"Propellers: {len(detection.boxes)}"
        cv2.putText(frame, count_text, (w_frame - 200, h_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 136, 255), 2, cv2.LINE_AA)

    return frame
```

**Step 2: Test drone detector**

Run and select drone dataset:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Drone dataset shows red bounding boxes around propellers, orange warning text if detected. Press `1` to toggle.

**Step 3: Commit drone detector**

```bash
git add evio/scripts/detector_utils.py
git commit -m "feat(mvp): add drone detector with propeller blob detection"
```

---

## Phase 4: Polish & Help Overlay

**Goal:** Final touches for professional appearance

### Task 9: Add help overlay

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add help overlay rendering function**

Add to `MVPLauncher` class:

```python
def _draw_help_overlay(self, frame: np.ndarray) -> None:
    """Draw help overlay with keybindings."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay (bottom third)
    overlay_h = h // 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Help text
    help_lines = [
        "KEYBOARD SHORTCUTS",
        "",
        "1 - Toggle detector overlay",
        "2 - Toggle HUD",
        "h - Toggle this help",
        "ESC - Return to menu",
        "q - Quit application",
    ]

    y_start = h - overlay_h + 30
    for i, line in enumerate(help_lines):
        font_scale = 0.7 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        color = (255, 255, 255) if i == 0 else (200, 200, 200)

        cv2.putText(frame, line, (30, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    thickness, cv2.LINE_AA)
```

**Step 2: Call help overlay in playback loop**

Modify `_playback_loop` to draw help overlay after HUD:

```python
self._draw_hud(frame, state, self._fps, self._playback_wall_start)

# Draw help overlay if enabled
if state.overlay_flags.get("help", False):
    self._draw_help_overlay(frame)

# Display
cv2.imshow(self.window_name, frame)
```

**Step 3: Test help overlay**

Run playback and press `h`:
```bash
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected: Press `h` → help overlay appears at bottom. Press `h` again → disappears.

**Step 4: Commit help overlay**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add help overlay with keyboard shortcuts"
```

---

### Task 10: Add error handling and polish

**Files:**
- Modify: `evio/scripts/mvp_launcher.py`

**Step 1: Add error overlay for playback failures**

Add to `MVPLauncher` class:

```python
def _show_error_and_return_to_menu(self, error_msg: str) -> None:
    """Show error message for 3 seconds then return to menu."""
    frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        y += 30

    # Show for 3 seconds
    for _ in range(30):
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(100) & 0xFF in (ord('q'), 27):
            break

    self.mode = AppMode.MENU
```

**Step 2: Wrap playback init with error handling**

Modify `_menu_loop` Enter key handler:

```python
elif key == 13:  # Enter
    if self.datasets:
        selected_dataset = self.datasets[self.selected_index]
        print(f"Selected: {selected_dataset.name}")
        try:
            self.playback_state = self._init_playback(selected_dataset)
            self.mode = AppMode.PLAYBACK
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            print(error_msg, file=sys.stderr)
            self._show_error_and_return_to_menu(error_msg)
```

**Step 3: Add graceful detector failure handling**

Modify `_playback_loop` detector overlay section:

```python
# Apply detector overlays if enabled
if state.overlay_flags.get("detector", True):
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
```

**Step 4: Test error handling**

Test with non-existent file (should be caught earlier), or force an error in detector utils.

**Step 5: Commit error handling**

```bash
git add evio/scripts/mvp_launcher.py
git commit -m "feat(mvp): add error handling and graceful degradation"
```

---

### Task 11: Add Nix alias and finalize

**Files:**
- Modify: `flake.nix`

**Step 1: Add run-mvp-demo alias**

Add to `shellHook` in `flake.nix` after existing aliases (around line 349):

```bash
alias run-mvp-demo='uv run --package evio python evio/scripts/mvp_launcher.py'
```

**Step 2: Update help text in shellHook**

Modify the demo aliases section in `shellHook` (around line 318):

```bash
echo "🎯 Detector Demos:"
echo "  run-mvp-demo           : MVP launcher (menu + detectors) - NEW!"
echo "  run-fan-rpm-demo       : Fan RPM (evlib, detector-commons)"
echo "  run-drone-detector-demo: Drone detection (evlib, detector-commons)"
echo "  run-fan-detector       : Fan RPM (legacy loader)"
echo "  run-drone-detector     : Drone detection (legacy loader)"
```

**Step 3: Test alias**

Exit and re-enter nix develop:
```bash
exit
nix develop
run-mvp-demo
```

Expected: Launcher runs via alias.

**Step 4: Commit Nix alias**

```bash
git add flake.nix
git commit -m "feat(mvp): add run-mvp-demo alias to Nix shell"
```

---

### Task 12: Final verification and documentation

**Files:**
- Modify: `docs/plans/2025-11-16-mvp-rendering-demo-design.md`

**Step 1: Test complete workflow**

```bash
nix develop
run-mvp-demo
```

Verify:
- [ ] Menu shows all datasets with navigation
- [ ] Select fan dataset → playback with ellipse, RPM, clusters
- [ ] Select drone dataset → playback with bounding boxes, warning
- [ ] `1` toggles detector overlay
- [ ] `2` toggles HUD
- [ ] `h` toggles help
- [ ] ESC returns to menu
- [ ] `q` quits
- [ ] Auto-loop works at end of dataset
- [ ] HUD shows accurate FPS and timing

**Step 2: Update design doc with completion status**

Add to top of `docs/plans/2025-11-16-mvp-rendering-demo-design.md`:

```markdown
**Status:** ✅ COMPLETE - Implementation verified 2025-11-16
```

**Step 3: Commit design update**

```bash
git add docs/plans/2025-11-16-mvp-rendering-demo-design.md
git commit -m "docs(mvp): mark design as complete"
```

**Step 4: Final commit**

```bash
git commit --allow-empty -m "feat(mvp): MVP rendering demo launcher complete

Unified launcher with menu-driven dataset selection and detector overlays.

Features:
- Auto-discovery of *_legacy.h5 datasets
- Text-tile menu with keyboard navigation
- Seamless auto-looping playback
- Fan detector: ellipse fit, blade clusters, RPM
- Drone detector: propeller detection, warning overlay
- Toggle controls for overlays (1/2/h keys)
- Error handling and graceful degradation
- Nix alias: run-mvp-demo

Implementation: 4 phases (menu → playback → detectors → polish)
Files: mvp_launcher.py, detector_utils.py

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Notes

**Total estimated time:** 2-3 hours for all 12 tasks

**Dependencies:**
- All dependencies already available in UV workspace
- No new packages needed
- Nix environment provides all system libraries

**Testing strategy:**
- Visual verification after each task
- Incremental testing ensures each phase works before moving on
- Final verification checklist ensures all features work together

**Key files created:**
1. `evio/scripts/mvp_launcher.py` (~400 lines)
2. `evio/scripts/detector_utils.py` (~350 lines)

**Key files modified:**
1. `flake.nix` (1 alias + help text)
2. `docs/plans/2025-11-16-mvp-rendering-demo-design.md` (status update)

---

## Future Enhancements (Post-MVP)

After MVP completion, consider:
1. Thumbnail generation and caching
2. Playback speed control (↑/↓ keys)
3. Frame export (screenshot feature)
4. Better RPM tracking (temporal smoothing)
5. More sophisticated drone detection
6. Custom detector plugin system
7. Multi-window support
8. Video export capability

See `docs/plans/2025-11-16-mvp-rendering-demo-design.md` for detailed future roadmap.
