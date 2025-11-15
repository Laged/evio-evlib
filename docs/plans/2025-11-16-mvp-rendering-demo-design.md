# MVP Rendering Demo - Design Document

**Date:** 2025-11-16
**Owner:** Claude (via brainstorming with user)
**Status:** Ready for implementation
**Parent Plan:** `docs/plans/mvp-rendering-demo.md`

---

## Overview

Build a unified launcher application that provides a polished, menu-driven interface for exploring event camera datasets with detector overlays. Focus on top-down implementation (menu → playback → detectors) for fast visual feedback.

**Goals:**
- Professional "app" feeling with dataset selection menu
- Seamless playback with auto-looping
- Integration of existing detector work (fan RPM, drone detection)
- cv2-only rendering (no new GUI dependencies)
- Incremental, visually testable implementation

---

## Architecture

### File Structure

```
evio/scripts/
├── mvp_launcher.py        # Main app: menu + playback loop
└── detector_utils.py      # Refactored detector functions
```

**Two-file approach:**
- `mvp_launcher.py` - Menu mode + playback mode state machine
- `detector_utils.py` - Extracted detector logic from existing demos

### Application Flow

```
START
  ↓
MENU MODE (auto-discover datasets)
  ↓ [Enter on selection]
PLAYBACK MODE (load dataset, run detector overlays)
  ↓ [ESC]
MENU MODE (return to selection)
  ↓ [Q]
EXIT
```

**State machine:**
- `mode ∈ {MENU, PLAYBACK}`
- Single cv2 window reused for both modes
- Clean state transitions with proper cleanup

---

## Component Design

### 1. Menu Mode - Dataset Discovery & Selection

**Auto-discovery:**
```python
def discover_datasets() -> list[dict]:
    """Scan evio/data/ for *_legacy.h5 files."""
    data_dir = Path("evio/data")
    datasets = []

    for h5_file in data_dir.rglob("*_legacy.h5"):
        name = h5_file.stem.replace("_legacy", "").replace("_", " ").title()
        category = h5_file.parent.name  # "fan", "drone_idle", etc.
        size_mb = h5_file.stat().st_size / (1024 * 1024)

        datasets.append({
            "path": h5_file,
            "name": name,
            "category": category,
            "size_mb": size_mb,
        })

    return sorted(datasets, key=lambda d: (d["category"], d["name"]))
```

**Menu rendering (cv2 grid):**
- **Layout:** 2-column grid, 300x150px tiles
- **Tile content (text-only MVP):**
  - Line 1: Dataset name (larger font, white)
  - Line 2: Category + file size (smaller font, gray)
  - Line 3: Duration if available (gray)
- **Selection:** Highlighted tile with blue border (#4a90e2, 3px)
- **Status bar:** Bottom overlay with keybindings

**Color scheme:**
- Background: `#2b2b2b` (dark gray)
- Tiles: `#404040` (medium gray)
- Selected: `#4a90e2` (blue accent)
- Text: White (names), light gray (metadata)

**Keyboard controls:**
- Arrow Up/Down: Navigate selection (wrap around)
- Enter: Start playback with selected dataset
- Q/ESC: Exit application

**Error handling:**
- No datasets found → Show message: "No datasets found. Run: convert-all-legacy-to-hdf5"
- Invalid HDF5 → Skip with warning, continue loading others

---

### 2. Playback Mode - Event Rendering & Loop

**Initialization:**
```python
def init_playback(dataset: dict) -> PlaybackState:
    """Load dataset and prepare playback."""
    events = evlib.load_events(str(dataset["path"])).collect()

    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1
    t_min, t_max = get_time_range(events)  # Handle Duration vs Int64

    detector_type = map_category_to_detector(dataset["category"])

    return PlaybackState(
        events=events,
        width=width,
        height=height,
        t_min=t_min,
        t_max=t_max,
        current_t=t_min,
        detector_type=detector_type,
        overlay_flags={"detector": True, "hud": True, "help": False},
    )
```

**Playback loop:**
1. Extract events in current window (10ms default)
2. Render base polarity frame (white=ON, black=OFF, gray=bg)
3. Apply detector overlays if enabled
4. Add HUD if enabled
5. Display frame
6. Advance time
7. **At end:** Reset to `t_min` (seamless auto-loop)

**Speed control:**
- Default: 1.0x real-time
- Sleep-based throttling (reuse `play_evlib.py` logic)

**Keyboard controls:**
- `1` - Toggle detector overlay
- `2` - Toggle HUD
- `h` - Toggle help overlay
- `ESC` - Return to menu
- `q` - Quit application
- `SPACE` - Pause/resume (optional)

**Error handling:**
- Invalid HDF5 → Error overlay, return to menu after 3s
- Detector crash → Disable overlays, show warning, continue
- OOM → Reduce window size, show warning

---

### 3. Detector Integration

**Dataset → Detector mapping:**
```python
def map_category_to_detector(category: str) -> str:
    """Map dataset category to detector type."""
    mapping = {
        "fan": "fan_rpm",
        "drone_idle": "drone",
        "drone_moving": "drone",
    }
    return mapping.get(category, "none")
```

**Refactored detector functions (detector_utils.py):**

Extract from existing `fan_detector_demo.py`:
```python
def detect_fan_ellipse(
    events_window: tuple,
    width: int,
    height: int
) -> tuple[EllipseParams, float]:
    """Run ellipse fit and RPM estimation."""
    # Extract existing logic: accumulation, threshold, fit, DBSCAN
    return ellipse_params, rpm_estimate

def render_fan_overlay(
    base_frame: np.ndarray,
    ellipse_params: EllipseParams,
    rpm: float
) -> np.ndarray:
    """Draw ellipse, clusters, RPM text on frame."""
    # Cyan ellipse, yellow clusters, green RPM text
    return frame_with_overlay
```

Extract from existing `drone_detector_demo.py`:
```python
def detect_drones(
    events_window: tuple,
    width: int,
    height: int
) -> list[Detection]:
    """Run drone detection logic."""
    # Extract existing logic
    return detections

def render_drone_overlay(
    base_frame: np.ndarray,
    detections: list[Detection]
) -> np.ndarray:
    """Draw bounding boxes, warnings on frame."""
    # Red boxes, orange warnings
    return frame_with_overlay
```

**Overlay composition in launcher:**
```python
base_frame = render_polarity_frame(events_window, width, height)

if detector_type == "fan_rpm" and overlay_flags["detector"]:
    ellipse_params, rpm = detect_fan_ellipse(events_window, width, height)
    frame = render_fan_overlay(base_frame, ellipse_params, rpm)
elif detector_type == "drone" and overlay_flags["detector"]:
    detections = detect_drones(events_window, width, height)
    frame = render_drone_overlay(base_frame, detections)
else:
    frame = base_frame

if overlay_flags["hud"]:
    draw_hud(frame, current_t, fps, dataset_name, detector_type)
```

**Graceful degradation:**
- Detector import fails → Run without overlays (base playback only)
- Detector crashes → Disable, show warning, continue

---

### 4. Visual Polish

**Overlay color palette:**
- Fan ellipse: Cyan `#00ffff`
- Blade clusters: Yellow circles
- RPM text: Green `#00ff00`
- Drone boxes: Red `#ff0000`
- Warning indicators: Orange `#ff8800`
- All overlays: Semi-transparent (alpha=0.7)

**HUD design (bottom-right corner):**
- Black background panel (alpha=0.6)
- White text showing:
  - FPS (actual render rate)
  - Playback speed (1.0x)
  - Recording time (timestamp in seconds)
  - Dataset name
  - Detector type

**Help overlay ('h' key):**
- Semi-transparent overlay (bottom third)
- Lists all keybindings
- Press 'h' again to dismiss

**Fonts:**
- cv2.FONT_HERSHEY_SIMPLEX throughout
- Consistent sizing: Large (0.8) for titles, medium (0.6) for labels, small (0.4) for metadata

---

## Implementation Strategy (Top-Down)

### Phase 1: Menu with Auto-Discovery
**Goal:** See the app structure immediately
- Implement dataset discovery
- Render text-only grid with navigation
- Test with existing `*_legacy.h5` files
- **Visual test:** Navigate menu, see all datasets listed

### Phase 2: Basic Playback (No Detectors)
**Goal:** Prove playback loop and state transitions
- Implement playback initialization
- Render base polarity frames
- Auto-loop at end
- ESC returns to menu
- **Visual test:** Play any dataset, see events render, verify loop

### Phase 3: Detector Overlays
**Goal:** Showcase existing detector work
- Refactor fan detector into `detector_utils.py`
- Wire up fan overlay rendering
- Add toggle controls (1/2 keys)
- **Visual test:** Play fan dataset, see ellipse/RPM, toggle overlays
- Repeat for drone detector
- **Visual test:** Play drone dataset, see boxes/warnings, toggle overlays

### Phase 4: Polish & HUD
**Goal:** Professional appearance
- Implement HUD overlay
- Add help overlay ('h' key)
- Tune colors, fonts, transparency
- Add error handling
- **Visual test:** All overlays look clean, keybindings work

---

## Nix Integration

**Add alias to `flake.nix` shellHook:**
```bash
alias run-mvp-demo='uv run --package evio python evio/scripts/mvp_launcher.py'
```

**Run command:**
```bash
nix develop
run-mvp-demo
```

**Dependencies (already available):**
- cv2 (opencv-python) ✓
- evlib ✓
- polars ✓
- numpy ✓
- scikit-learn ✓

**No new dependencies required.**

---

## Future Enhancements (Post-MVP)

### Thumbnail Generation
1. On first launch, generate thumbnails:
   - Load first 100ms of each dataset
   - Render to 300x150 image
   - Cache as PNG in `evio/data/.cache/thumbnails/`
2. Menu checks for cached thumbnails, falls back to text
3. Async/background generation to avoid blocking

### Additional Features
- Playback speed control (↑/↓ keys)
- Frame export (screenshot current frame)
- Recording session (export video)
- Metadata viewer (show full dataset info)
- Multiple window sizes/layouts
- Custom detector plugins via convention

---

## Success Criteria

**MVP is complete when:**
1. Menu shows all discovered datasets with navigation
2. Playback runs smoothly with auto-loop
3. Fan detector overlays render correctly (ellipse, RPM, clusters)
4. Drone detector overlays render correctly (boxes, warnings)
5. All hotkeys work (1/2/h/ESC/q)
6. HUD displays accurate stats
7. State transitions (menu ↔ playback) are clean
8. Error handling prevents crashes

**Visual quality bar:**
- Consistent color scheme and fonts
- No flicker or rendering artifacts
- Smooth transitions between modes
- Professional, polished appearance

---

## References

- Parent plan: `docs/plans/mvp-rendering-demo.md`
- Rendering pipeline: `docs/plans/mvp-rendering-pipeline.md`
- UI features: `docs/plans/mvp-ui-features.md`
- Existing code: `evio/scripts/play_evlib.py`, `fan_detector_demo.py`, `drone_detector_demo.py`
