# Interactive Event Camera Detection Workbench - Architecture Design

**Date:** 2025-11-15
**Status:** Approved Design
**Target:** Sensofusion Junction Hackathon Challenge

---

## Executive Summary

Transform evio into a **collaborative, production-ready event camera detection platform** using:
- **UV workspaces** for multi-team development
- **evlib integration** for 10-200x performance gains
- **Plugin architecture** for extensible detection algorithms
- **Adapter pattern** for seamless file → live stream transition
- **Interactive UI** for rapid experimentation (hot-swap data & detectors)

**Key Innovation:** Single app that runs multiple detection pipelines on both offline .dat files and live camera streams, with runtime plugin/data switching for rapid iteration.

---

## Design Principles

1. **Start Small, Scale Smart**: Begin with minimal PoC, expand to production features
2. **Preserve APIs**: Keep familiar evio interfaces, accelerate internals with evlib
3. **Plugin Extensibility**: New detectors = new Python packages, zero app changes
4. **Adapter Abstraction**: File and stream sources implement same protocol
5. **Interactive Development**: Hot-swap everything (data, plugins) for fast iteration
6. **Collaborative Structure**: UV workspaces enable parallel team development

---

## Repository Structure

```
evio-evlib/                        # Root repo
├── evio/                          # PRESERVED: Original evio (reference)
│   ├── src/evio/
│   ├── scripts/                   # Legacy MVP demos
│   └── pyproject.toml
│
├── workspace/                     # NEW: UV workspace root
│   ├── pyproject.toml            # UV workspace configuration
│   ├── uv.lock                   # Unified lockfile (all packages)
│   │
│   ├── libs/                     # Shared libraries
│   │   └── evio-core/           # Core event processing library
│   │       ├── src/evio_core/
│   │       │   ├── __init__.py
│   │       │   ├── loaders.py        # evlib file loading
│   │       │   ├── adapters.py       # EventSource protocol + adapters
│   │       │   ├── representations.py # evlib wrappers (time surface, voxel)
│   │       │   ├── playback.py       # Looping playback controller
│   │       │   ├── plugin.py         # DetectorPlugin protocol
│   │       │   └── compat.py         # Backward compat with evio API
│   │       ├── pyproject.toml
│   │       └── tests/
│   │
│   ├── plugins/                  # Detection plugins (teams work here)
│   │   ├── fan-bbox/            # Challenge 1: Static fan bbox
│   │   │   ├── src/fan_bbox/
│   │   │   │   ├── __init__.py
│   │   │   │   └── detector.py      # FanBBoxDetector class
│   │   │   ├── pyproject.toml
│   │   │   └── README.md
│   │   │
│   │   ├── fan-rpm/             # Challenge 2: RPM detection
│   │   │   ├── src/fan_rpm/
│   │   │   │   └── detector.py      # FanRPMDetector class
│   │   │   └── pyproject.toml
│   │   │
│   │   └── drone-tracker/       # Challenge 3: Drone tracking (future)
│   │       ├── src/drone_tracker/
│   │       │   └── detector.py
│   │       └── pyproject.toml
│   │
│   ├── apps/                     # Applications
│   │   └── detector-ui/         # Interactive detection workbench
│   │       ├── src/detector_ui/
│   │       │   ├── __init__.py
│   │       │   ├── main.py          # Main UI loop + plugin management
│   │       │   ├── controls.py      # Keyboard controls (hot-swap)
│   │       │   └── visualizer.py    # OpenCV visualization
│   │       ├── pyproject.toml
│   │       └── README.md
│   │
│   └── tools/                    # Development utilities
│       ├── benchmark.py         # evlib vs manual performance comparison
│       └── plugin-template/     # Template for new detector plugins
│
├── flake.nix                     # Nix dev environment (enhanced for UV)
├── README.md                     # Updated with workspace instructions
└── data/                         # Event camera datasets
    ├── fan/
    │   ├── fan_const_rpm.dat
    │   └── fan_varying_rpm.dat
    └── drone/                   # Future datasets
```

### Why This Structure?

**UV Workspaces:**
- One `git` repo, multiple Python packages
- Single lockfile (`uv.lock`) for dependency consistency
- Teams work in isolated packages without conflicts
- Fast: `uv sync` installs everything in seconds

**Separation of Concerns:**
- **libs/evio-core**: Data loading, adapters, plugin protocol (stable)
- **plugins/***: Detection algorithms (high churn, independent)
- **apps/detector-ui**: User interface (consumes libs + plugins)

**Collaborative Workflow:**
- Team A: Works in `plugins/fan-bbox/`, adds sklearn clustering
- Team B: Works in `plugins/fan-rpm/`, adds scipy frequency analysis
- Team C: Works in `apps/detector-ui/`, improves visualization
- No merge conflicts, shared dependencies via workspace

---

## Core Abstractions

### 1. EventSource Protocol

**Purpose:** Unified interface for file and stream sources.

```python
# libs/evio-core/src/evio_core/adapters.py

from typing import Protocol
import polars as pl

class EventSource(Protocol):
    """Protocol that all event sources must implement."""

    def get_window(self, duration_ms: float = 50) -> pl.DataFrame:
        """
        Get next window of events.

        For files: Returns next duration_ms of events, loops at end.
        For streams: Returns latest buffered events.
        """
        ...

    def get_resolution(self) -> tuple[int, int]:
        """Get sensor resolution (width, height)."""
        ...

    def is_live(self) -> bool:
        """True if live stream, False if file playback."""
        ...
```

**Implementations:**

**File Adapter** (available now):
```python
class FileEventAdapter:
    """File playback using evlib (10x faster than manual parsing)."""

    def __init__(self, path: str):
        # evlib auto-detects format (.dat, .aedat, .h5)
        self.events = evlib.load_events(path).collect()
        self.width = int(self.events["x"].max()) + 1
        self.height = int(self.events["y"].max()) + 1
        self.t_start = int(self.events["t"].min())
        self.t_end = int(self.events["t"].max())
        self.current_position = self.t_start

    def get_window(self, duration_ms: float = 50) -> pl.DataFrame:
        """Get next window, auto-loop at end."""
        t_window_start = self.current_position
        t_window_end = t_window_start + int(duration_ms * 1000)

        window = self.events.filter(
            (pl.col("t") >= t_window_start) &
            (pl.col("t") < t_window_end)
        )

        self.current_position = t_window_end

        # Loop playback
        if self.current_position >= self.t_end:
            self.current_position = self.t_start

        return window

    def get_resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def is_live(self) -> bool:
        return False
```

**Stream Adapter** (add at hackathon venue):
```python
class StreamEventAdapter:
    """Live camera using Metavision SDK or neuromorphic-drivers."""

    def __init__(self, device_id: int = 0):
        # Open camera, start background capture thread
        # Buffer events in queue for smooth playback
        ...

    def get_window(self, duration_ms: float = 50) -> pl.DataFrame:
        """Get latest buffered events from camera."""
        return self.buffer.get_nowait()

    def is_live(self) -> bool:
        return True
```

**Key Benefit:** App and plugins don't know source type. Same code works for both.

### 2. DetectorPlugin Protocol

**Purpose:** Standardized interface for all detection algorithms.

```python
# libs/evio-core/src/evio_core/plugin.py

from typing import Protocol
import polars as pl

class DetectorPlugin(Protocol):
    """All detector plugins implement this interface."""

    name: str         # Display name: "Fan BBox", "Fan RPM", etc.
    key: str          # Keyboard shortcut: "1", "2", "3", etc.
    description: str  # Short description for UI

    def process(self, events: pl.DataFrame, width: int, height: int) -> dict:
        """
        Process events and return detection results.

        Args:
            events: Polars DataFrame with columns [t, x, y, polarity]
            width: Sensor width
            height: Sensor height

        Returns:
            dict with detection-specific results:
                Common keys:
                - bbox: (x_min, y_min, x_max, y_max) or None
                - confidence: float 0-1
                - debug_info: dict with internal state

                Plugin-specific keys:
                - rpm: float (for RPM detectors)
                - tracks: list[dict] (for tracking detectors)
                - overlays: list of visualization elements
        """
        ...
```

**Example Plugin:**

```python
# plugins/fan-bbox/src/fan_bbox/detector.py

import polars as pl
import evlib.representations as evr

class FanBBoxDetector:
    """Minimal fan bounding box detector."""

    name = "Fan BBox"
    key = "1"
    description = "Time surface + spatial bounds"

    def process(self, events: pl.DataFrame, width: int, height: int) -> dict:
        # Create time surface (evlib - 50x faster than manual!)
        time_surface = evr.create_timesurface(
            events, height=height, width=width,
            dt=33000, tau=50000
        )

        # Find active pixels
        active_pixels = [
            (row['x'], row['y'])
            for row in time_surface.iter_rows(named=True)
            if row['value'] > 0.3
        ]

        if not active_pixels:
            return {"bbox": None, "confidence": 0.0}

        # Compute bounding box
        xs, ys = zip(*active_pixels)
        bbox = (min(xs), min(ys), max(xs), max(ys))

        return {
            "bbox": bbox,
            "confidence": len(active_pixels) / (width * height),
        }
```

**Key Benefit:** Adding new detector = new Python package, implement protocol. App auto-discovers and presents as mode.

### 3. Interactive Hot-Swapping

**User Experience:**
```
1. Start app: detector-ui data/fan_const_rpm.dat
2. Fan rotating, BBox detector active (plugin 1)
3. Press '2' → switches to RPM detector, same data
4. Press 'd' → file picker appears
5. Select data/drone.dat → loads new data
6. Press '3' → switches to drone tracker
7. All while playback continues smoothly
```

**Implementation:**

```python
# apps/detector-ui/src/detector_ui/main.py

class DetectorUI:
    def __init__(self, source: EventSource, plugins: list[DetectorPlugin]):
        self.source = source
        self.plugins = {p.key: p for p in plugins}
        self.current_plugin = plugins[0]
        self.visualizer = EventVisualizer(
            source.get_resolution()[0],
            source.get_resolution()[1]
        )
        self.running = True

    def handle_keypress(self, key: str):
        """Handle keyboard input for hot-swapping."""

        # Switch plugin (1, 2, 3, ...)
        if key in self.plugins:
            self.current_plugin = self.plugins[key]
            print(f"→ Switched to: {self.current_plugin.name}")

        # Switch data file (d)
        elif key == 'd':
            new_file = self.file_picker()
            self.source = FileEventAdapter(new_file)
            print(f"→ Loaded: {new_file}")

        # Quit (q)
        elif key == 'q':
            self.running = False

    def run(self):
        """Main loop."""
        print("Interactive Event Camera Detector")
        print("Controls:")
        for plugin in self.plugins.values():
            print(f"  {plugin.key} = {plugin.name}")
        print("  d = Load different data")
        print("  q = Quit")

        while self.running:
            # Get next window of events (auto-loops for files)
            events = self.source.get_window(duration_ms=50)

            # Run current plugin
            results = self.current_plugin.process(
                events,
                self.source.get_resolution()[0],
                self.source.get_resolution()[1]
            )

            # Visualize
            self.visualizer.draw(events, results)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_keypress(chr(key))

        self.visualizer.cleanup()


def main():
    import sys

    # Determine source type
    if sys.argv[1].endswith('.dat'):
        source = FileEventAdapter(sys.argv[1])
    elif sys.argv[1].startswith('camera:'):
        device_id = int(sys.argv[1].split(':')[1])
        source = StreamEventAdapter(device_id)
    else:
        print("Usage: detector-ui <file.dat|camera:0>")
        return

    # Load all plugins
    from fan_bbox.detector import FanBBoxDetector
    from fan_rpm.detector import FanRPMDetector

    plugins = [
        FanBBoxDetector(),
        FanRPMDetector(),
    ]

    # Run UI
    ui = DetectorUI(source, plugins)
    ui.run()
```

**Key Benefits:**
- Zero restart for plugin changes
- Zero restart for data changes
- Rapid experimentation loop
- Compare detectors on same data
- Same UI for files and live streams

---

## evlib Integration Strategy

### Performance Gains

| Operation | Manual (old evio) | evlib | Speedup |
|-----------|-------------------|-------|---------|
| Load .dat file (1GB) | 1200ms | 120ms | **10x** |
| Create voxel grid (10M events) | 2500ms | 45ms | **55x** |
| Create histogram (540M events) | ~600s | ~3s | **200x** |
| ROI filtering (10M events) | 800ms | 15ms | **53x** |

### Backward Compatibility

**Old evio users:**
```python
# This still works (compat layer)
from evio_core.compat import open_dat
rec = open_dat("file.dat", width=1280, height=720)
events = rec.get_events(t_start, t_end)
```

**New evio-core users:**
```python
# New API (cleaner, faster)
from evio_core.loaders import FileEventSource
source = FileEventSource("file.dat")  # Auto-detects resolution
events = source.get_events(t_start, t_end)
```

Both get evlib speed underneath!

### evlib Wrappers

```python
# libs/evio-core/src/evio_core/representations.py

import evlib.representations as evr

def create_time_surface(events: pl.DataFrame, width: int, height: int,
                       tau_ms: float = 50.0):
    """Wrapper with sensible defaults."""
    return evr.create_timesurface(
        events, height=height, width=width,
        dt=33000.0, tau=tau_ms * 1000.0
    )

def create_voxel_grid(events: pl.DataFrame, width: int, height: int,
                     n_bins: int = 10):
    """Wrapper for voxel grids."""
    return evr.create_voxel_grid(
        events, height=height, width=width, n_time_bins=n_bins
    )
```

**Philosophy:** Preserve evio's educational API, accelerate with evlib internals.

---

## Development Environment

### UV + Nix Integration

**Nix provides:** System dependencies (Python 3.11, Rust toolchain, OpenCV, pkg-config)
**UV provides:** Python package management, workspace resolution, fast installs

**Root Workspace:** (`workspace/pyproject.toml`)
```toml
[tool.uv.workspace]
members = [
    "libs/evio-core",
    "plugins/fan-bbox",
    "plugins/fan-rpm",
    "apps/detector-ui",
]
```

**Each package has:** `pyproject.toml` with dependencies
**Shared lockfile:** `workspace/uv.lock` (committed to git)

### Developer Workflow

**Initial Setup (once):**
```bash
nix develop          # Enters Nix shell
cd workspace
uv sync              # Installs all workspace packages
```

**Daily Development:**
```bash
# Team member working on fan-bbox plugin
cd workspace/plugins/fan-bbox
uv add scikit-learn  # Add dependency
# Edit detector.py
cd ../..
uv run pytest plugins/fan-bbox/tests

# Run the app
uv run detector-ui ../data/fan/fan_const_rpm.dat
```

**Adding New Plugin:**
```bash
cd workspace/plugins
cp -r fan-bbox my-detector
# Edit my-detector/pyproject.toml (change name)
# Add "plugins/my-detector" to workspace/pyproject.toml
cd ..
uv sync  # Installs new plugin
```

### Enhanced flake.nix

```nix
{
  devShells.default = pkgs.mkShell {
    buildInputs = [
      python311
      pkgs.uv           # UV package manager
      pkgs.rustc        # For evlib compilation
      pkgs.cargo
      pkgs.pkg-config
      pkgs.opencv4      # OpenCV system libs
    ];

    shellHook = ''
      echo "Event Camera Detection Workbench"
      echo "Quick Start:"
      echo "  cd workspace && uv sync"
      echo "  uv run detector-ui ../data/fan/fan_const_rpm.dat"
    '';
  };
}
```

---

## First PoC: Fan Bounding Box

### Goal

Prove evlib integration works, demonstrate plugin system.

**Deliverable:** ~100 line detector that draws bounding box around rotating fan.

### Implementation (Minimal)

```python
# plugins/fan-bbox/src/fan_bbox/detector.py

import polars as pl
import numpy as np
import evlib.representations as evr

class FanBBoxDetector:
    name = "Fan BBox (Minimal)"
    key = "1"
    description = "Time surface + spatial bounds"

    def __init__(self, tau_ms: float = 50.0, threshold: float = 0.3):
        self.tau_ms = tau_ms
        self.threshold = threshold

    def process(self, events: pl.DataFrame, width: int, height: int) -> dict:
        if len(events) == 0:
            return {"bbox": None, "confidence": 0.0}

        # evlib time surface (exponential temporal decay)
        events_typed = events.with_columns([
            pl.col("t").cast(pl.Float64),
            pl.col("x").cast(pl.Int64),
            pl.col("y").cast(pl.Int64),
            pl.col("polarity").cast(pl.Int64)
        ])

        time_surface = evr.create_timesurface(
            events_typed,
            height=height, width=width,
            dt=33000.0,
            tau=self.tau_ms * 1000.0
        )

        # Find active pixels
        active_pixels = [
            (row['x'], row['y'])
            for row in time_surface.iter_rows(named=True)
            if row['value'] > self.threshold
        ]

        if not active_pixels:
            return {"bbox": None, "confidence": 0.0}

        # Bounding box from active pixel bounds
        xs, ys = zip(*active_pixels)
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        # Confidence: active area / total area
        active_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        confidence = min(1.0, active_area / (width * height * 0.5))

        return {
            "bbox": bbox,
            "confidence": float(confidence),
            "debug_info": {
                "active_pixels": len(active_pixels),
                "tau_ms": self.tau_ms,
            }
        }
```

### Enhancement Path

**Phase 2:** Add sklearn DBSCAN for robust clustering (handles multiple objects, outliers)
**Phase 3:** Add Kalman filtering for stable tracking across frames
**Phase 4:** Add confidence scoring based on cluster quality

---

## Real-Time Streaming Integration

### Strategy

**Develop with files, deploy with streams.** Adapter pattern makes this seamless.

### Implementation Timeline

**Before Hackathon (Weeks 1-2):**
- Build entire system with `FileEventAdapter`
- Develop all plugins using .dat files
- Test, iterate, optimize

**At Hackathon Venue (Day 1, ~30 minutes):**
```bash
# Install Metavision SDK (provided by Sensofusion)
# Update dependencies
cd workspace/libs/evio-core
uv add metavision-sdk-core metavision-sdk-driver

# Test with live camera
cd ../..
uv run detector-ui camera:0
# Same app, same plugins, now with live data!
```

**Stream Adapter Implementation:**
```python
# libs/evio-core/src/evio_core/adapters.py

from queue import Queue
import threading
from metavision_core.event_io import EventsIterator

class StreamEventAdapter:
    def __init__(self, device_id: int = 0):
        self.camera = EventsIterator(f"camera:{device_id}")
        self.buffer = Queue(maxsize=100)
        self.running = True

        # Background capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Camera → buffer (runs in background)."""
        for events in self.camera:
            if not self.running:
                break
            try:
                events_df = self._to_polars(events)
                self.buffer.put_nowait(events_df)
            except:
                pass  # Buffer full, skip

    def get_window(self, duration_ms: float = 50) -> pl.DataFrame:
        """Get latest events from buffer."""
        try:
            return self.buffer.get_nowait()
        except:
            return pl.DataFrame({"x": [], "y": [], "t": [], "polarity": []})

    def is_live(self) -> bool:
        return True
```

**Zero App Changes:** DetectorUI doesn't know source type, works identically for files and streams.

---

## Summary: Three Collaborative Work Streams

### 1. Monorepo Setup (Day 1-2, 1 person)

**Tasks:**
- Create `workspace/` directory structure
- Write UV workspace `pyproject.toml`
- Enhance `flake.nix` with UV support
- Create `libs/evio-core/` skeleton
- Write setup documentation

**Deliverable:** Team can run `nix develop && cd workspace && uv sync` and have working environment.

### 2. evlib Abstraction + First PoC (Day 2-4, 1-2 people)

**Tasks:**
- Implement `FileEventAdapter` using evlib
- Implement `DetectorPlugin` protocol
- Create minimal `FanBBoxDetector` (time surface → bbox)
- Build basic `detector-ui` with visualization
- Test end-to-end: load .dat → detect fan → draw bbox

**Deliverable:** Working demo showing fan bounding box detection using evlib.

### 3. Real-Time Preparation (Ongoing, async)

**Tasks:**
- Design `StreamEventAdapter` interface (stub implementation)
- Document Metavision SDK integration points
- Create test plan for hackathon venue
- Write quick-start guide for venue deployment

**Deliverable:** Plan to add live streaming in 30 minutes at hackathon.

---

## Success Criteria

### Must Have (Week 1)
- ✅ UV workspace structure with evio-core, fan-bbox plugin, detector-ui app
- ✅ FileEventAdapter using evlib (10x faster file loading)
- ✅ DetectorPlugin protocol implemented
- ✅ Minimal FanBBoxDetector drawing bounding box on fan
- ✅ Interactive UI with plugin hot-swapping (press 1, 2, 3...)
- ✅ Looping .dat file playback

### Should Have (Week 2)
- ✅ Enhanced FanBBoxDetector with DBSCAN clustering
- ✅ FanRPMDetector plugin (voxel grid + scipy frequency analysis)
- ✅ Data file hot-swapping (press 'd' to load different .dat)
- ✅ Backward compatibility layer (evio_core.compat)

### Nice to Have (At Hackathon)
- ✅ StreamEventAdapter with Metavision SDK
- ✅ Live camera demonstration
- ✅ Drone tracker plugin
- ✅ Performance benchmarks (evlib vs manual)

---

## Next Steps

1. **Review this design document** with team
2. **Set up git worktree** for isolated development
3. **Create implementation plan** with detailed tasks
4. **Begin Phase 1:** Monorepo setup

**Ready to create the implementation plan?**
