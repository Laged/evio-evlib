# MVP Launcher Implementation - Critical Fixes & Blockers

**Date:** 2025-11-16
**Status:** Required addendum to `2025-11-16-mvp-launcher-implementation.md`
**Purpose:** Address critical implementation blockers before execution

---

## Critical Blockers to Address

### 1. Data Discovery - Filter Out IDS Files

**Problem:** Discovery may pick up `_evt3.dat` (IDS camera files) instead of `*_legacy.h5` files.

**Fix for Task 2 - Step 1:** Update `discover_datasets()`:

```python
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
```

**Updated Step 2:** Add clear banner if no datasets found:

```python
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
```

---

### 2. Memory Usage - Lazy Windowing

**Problem:** `evlib.load_events(...).collect()` loads entire dataset into RAM, causing OOM on large files.

**Fix for Task 4 - Step 4:** Use lazy filtering per window instead of preloading:

```python
def _init_playback(self, dataset: Dataset) -> PlaybackState:
    """Load dataset and prepare playback state."""
    print(f"Loading {dataset.path}...")

    try:
        # DON'T collect() here - keep lazy!
        lazy_events = evlib.load_events(str(dataset.path))

        # Only collect metadata needed for initialization
        # Use limit(1) to avoid loading full dataset
        sample = lazy_events.limit(1).collect()

        # For resolution, we need to scan - but use SQL-style aggregation
        # This is more efficient than collecting everything
        metadata = lazy_events.select([
            pl.col("x").max().alias("max_x"),
            pl.col("y").max().alias("max_y"),
            pl.col("t").min().alias("t_min"),
            pl.col("t").max().alias("t_max"),
        ]).collect()

        width = int(metadata["max_x"][0]) + 1
        height = int(metadata["max_y"][0]) + 1

        # Get time range (handle Duration vs Int64)
        t_min_val = metadata["t_min"][0]
        t_max_val = metadata["t_max"][0]

        if isinstance(t_min_val, pl.Duration):
            t_min = int(t_min_val.total_microseconds())
            t_max = int(t_max_val.total_microseconds())
        else:
            t_min = int(t_min_val)
            t_max = int(t_max_val)

        print(f"Resolution: {width}x{height}, "
              f"Duration: {(t_max - t_min) / 1e6:.2f}s")

        # Determine detector type
        detector_type = self._map_detector_type(dataset.category)
        print(f"Detector: {detector_type}")

        return PlaybackState(
            dataset=dataset,
            lazy_events=lazy_events,  # Store lazy reference, not collected!
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

**Update PlaybackState dataclass:**

```python
@dataclass
class PlaybackState:
    """Playback session state."""
    dataset: Dataset
    lazy_events: pl.LazyFrame  # Changed from events: pl.DataFrame
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

**Update _get_event_window to use lazy frame:**

```python
def _get_event_window(
    self,
    lazy_events: pl.LazyFrame,
    win_start_us: int,
    win_end_us: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract window of events using lazy polars filtering."""
    # Apply filter lazily, then collect only the window
    # This avoids loading the entire dataset into memory

    # Peek at schema to determine time column type
    schema = lazy_events.schema
    t_dtype = schema["t"]

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
```

**Update playback loop call:**

```python
# In _playback_loop, change:
# window = self._get_event_window(state.events, state.current_t, win_end)
# To:
window = self._get_event_window(state.lazy_events, state.current_t, win_end)
```

---

### 3. Detector Import Failures - Graceful Degradation

**Fix for Task 6 - Step 1:** Add try/except around detector imports at module level:

```python
# At top of evio/scripts/mvp_launcher.py, after base imports:

# Try to import detector utilities - degrade gracefully if missing
try:
    from evio.scripts.detector_utils import (
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
```

**Update _map_detector_type to check availability:**

```python
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
```

**Update playback loop to show warning if detectors unavailable:**

```python
# In _playback_loop, before detector overlay section:

# Apply detector overlays if enabled AND available
if state.overlay_flags.get("detector", True) and DETECTORS_AVAILABLE:
    try:
        if state.detector_type == "fan_rpm":
            # ... fan detector code
        elif state.detector_type == "drone":
            # ... drone detector code
    except Exception as e:
        # Detector crashed
        print(f"Detector error: {e}", file=sys.stderr)
        state.overlay_flags["detector"] = False
        cv2.putText(frame, "Detector disabled (error)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

elif state.overlay_flags.get("detector", True) and not DETECTORS_AVAILABLE:
    # Show warning that detectors aren't available
    cv2.putText(frame, "Detectors not available", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)
```

---

### 4. cv2 Keyboard Handling - Robust Polling

**Problem:** `cv2.waitKey()` can miss keypresses if frame processing is slow.

**Fix for all loop implementations:** Use consistent, non-blocking key handling:

```python
def _menu_loop(self) -> bool:
    """Menu mode loop. Returns False to exit app."""
    frame = self._render_menu()
    cv2.imshow(self.window_name, frame)

    # Use shorter waitKey for more responsive input
    # 30ms = ~33 FPS polling rate
    key = cv2.waitKey(30) & 0xFF

    # Handle quit keys
    if key == ord('q'):
        return False
    elif key == 27:  # ESC
        return False

    # Handle navigation (check multiple key codes for cross-platform compatibility)
    elif key in (82, 0, ord('k')):  # Up arrow or 'k'
        if self.datasets:
            self.selected_index = (self.selected_index - 1) % len(self.datasets)
    elif key in (84, 1, ord('j')):  # Down arrow or 'j'
        if self.datasets:
            self.selected_index = (self.selected_index + 1) % len(self.datasets)

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
                self._show_error_and_return_to_menu(error_msg)

    return True
```

**Update playback loop similarly:**

```python
# In _playback_loop, use waitKey(1) for lowest latency:
key = cv2.waitKey(1) & 0xFF

# Explicit key handling with no conflicts
if key == ord('q'):
    return False  # Quit app entirely
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
```

---

### 5. HUD/Render Scale - Make Optional

**Fix for HUD rendering:** Add note about scale in design, don't enforce 70% parity:

The current design uses native resolution. If drone parity requires 70% scale, add this as an OPTIONAL feature:

```python
# In PlaybackState, add optional scale:
@dataclass
class PlaybackState:
    # ... existing fields
    render_scale: float = 1.0  # Optional: set to 0.7 for drone parity

# In _render_polarity_frame, apply scale if needed:
def _render_polarity_frame(
    self,
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Render polarity events to frame."""
    x_coords, y_coords, polarities_on = window

    # Base gray, white for ON, black for OFF
    frame = np.full((height, width, 3), (127, 127, 127), dtype=np.uint8)

    if len(x_coords) > 0:
        frame[y_coords[polarities_on], x_coords[polarities_on]] = (255, 255, 255)
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = (0, 0, 0)

    # Optional scaling (for drone parity)
    if scale != 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    return frame
```

**DON'T enforce scale by default** - let user see native resolution unless they specifically need parity mode.

---

### 6. Runtime Environment - Documentation

**Fix for Task 1 - Step 2:** Add clear environment check and banner:

```python
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
```

---

## Updated Task Order with Fixes

**Phase 1: Menu (Tasks 1-3)**
- Task 1: Add environment check and banner
- Task 2: Use improved discovery with `*_legacy.h5` filter and clear error messages
- Task 3: Use robust keyboard handling with multiple key codes

**Phase 2: Playback (Tasks 4-5)**
- Task 4: Use lazy windowing instead of `collect()` for memory efficiency
- Task 5: Keep render scale at 1.0 by default (native resolution)

**Phase 3: Detectors (Tasks 6-8)**
- Task 6: Add graceful import handling with `DETECTORS_AVAILABLE` flag
- Task 7: Add detector error handling in playback loop
- Task 8: Ensure detector utils don't have hard dependencies

**Phase 4: Polish (Tasks 9-12)**
- Task 9: Update help overlay with environment info
- Task 10: Comprehensive error handling
- Task 11: Add Nix alias with documentation
- Task 12: Final verification checklist

---

## Pre-Execution Checklist

Before starting implementation:

1. [ ] Verify datasets exist: `ls evio/data/*//*_legacy.h5`
2. [ ] If missing, run: `convert-all-legacy-to-hdf5`
3. [ ] Confirm nix environment: `echo $NIX_STORE` (should output path)
4. [ ] Test evlib import: `uv run --package evio python -c "import evlib; print('OK')"`
5. [ ] Test detector imports work: `uv run --package evio python -c "from sklearn.cluster import DBSCAN; print('OK')"`

If any check fails, address before proceeding with implementation.

---

## Summary of Critical Changes

| Blocker | Original Plan Issue | Fix Applied |
|---------|---------------------|-------------|
| Data discovery | May pick up `_evt3.dat` files | Explicit `*_legacy.h5` glob, defensive filtering |
| Memory usage | `collect()` loads full dataset | Lazy filtering with per-window `collect()` |
| Detector imports | Hard dependency assumed | Try/except with `DETECTORS_AVAILABLE` flag |
| cv2 keyboard | Single key code, blocking wait | Multiple key codes, 1ms/30ms wait |
| Render scale | None | Optional scale parameter (default 1.0) |
| Environment | No validation | Banner check for `NIX_STORE` in PATH |

These fixes ensure the implementation is robust against the identified blockers.
