# Task 5 Manual Visual Test Instructions

## What was implemented

Task 5 from `docs/plans/2025-11-16-mvp-launcher-implementation.md` has been completed:

1. ✅ `_get_event_window()` - LAZY windowing using `lazy_events.filter().collect()`
   - Filters events by time window BEFORE collecting
   - Collects ONLY the 10ms window, not the full dataset
   - Prevents OOM on large files
   - Verified by automated test (see test_lazy_windowing.py)

2. ✅ `_render_polarity_frame()` - Base event rendering
   - Gray background (127, 127, 127)
   - White pixels for ON events (255, 255, 255)
   - Black pixels for OFF events (0, 0, 0)

3. ✅ `_draw_hud()` - Playback stats overlay
   - Bottom-right panel with semi-transparent background
   - FPS counter
   - Speed multiplier
   - Recording time
   - Dataset category

4. ✅ Full `_playback_loop()` implementation
   - Auto-loop at t_max: seamlessly resets to t_min
   - Speed control via sleep-based timing
   - FPS calculation with exponential smoothing (0.9 * old + 0.1 * new)
   - Robust keyboard handling with 1ms waitKey

5. ✅ Keyboard controls (from critical-fixes.md section 4)
   - ESC: Return to menu
   - q: Quit application
   - 1: Toggle detector overlay (ready for Phase 3)
   - 2: Toggle HUD
   - h: Toggle help (ready for Phase 4)

## Manual Visual Test

To visually verify playback works:

```bash
# From repository root, in nix environment:
nix develop
uv run --package evio python evio/scripts/mvp_launcher.py
```

Expected behavior:
1. Menu appears with discovered datasets
2. Use arrow keys or j/k to navigate
3. Press Enter or Space to select a dataset
4. Playback should show:
   - Gray background with white/black events
   - Bottom-right HUD with FPS, speed, time, category
   - Auto-loop when reaching end of dataset
   - Responsive keyboard handling (ESC to menu, q to quit)

## Verification Results

Automated tests passed:
- ✅ Lazy windowing extracts only requested window (28,751 events in 10ms window)
- ✅ Empty windows handled correctly
- ✅ Multiple sequential windows work without caching issues
- ✅ Syntax validation passed
- ✅ No import errors in nix environment

## Critical Implementation Details

### Lazy Windowing (from critical-fixes.md section 2)
```python
# CRITICAL: Apply filter BEFORE collect
window = lazy_events.filter(
    (pl.col("t") >= win_start_us) &
    (pl.col("t") < win_end_us)
).collect()  # Collect ONLY the filtered window
```

This prevents loading the full dataset into RAM, allowing playback of large files.

### Keyboard Handling (from critical-fixes.md section 4)
```python
# Use 1ms waitKey for maximum responsiveness
key = cv2.waitKey(1) & 0xFF
```

Combined with multiple key codes for cross-platform compatibility.

### Auto-Loop Implementation
```python
if state.current_t >= state.t_max:
    state.current_t = state.t_min
    self._playback_wall_start = time.perf_counter()
    print("Auto-looping to start...")
```

Seamlessly resets playback to beginning when reaching end.

## Next Steps

Task 5 is complete. The next task in the plan is:
- **Task 6**: Create detector utilities module (fan detector extraction)

Files ready for Phase 3 (Detector Overlays):
- `evio/scripts/mvp_launcher.py` - Main launcher with playback loop
- Detector overlay infrastructure ready (toggle flags, render pipeline)
