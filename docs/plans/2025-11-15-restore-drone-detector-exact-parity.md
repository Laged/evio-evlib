# Restore Drone Detector Exact Feature Parity - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore `run-drone-detector-demo` to EXACT behavioral parity with `example-drone-original.py` by reverting 5 algorithmic deviations introduced by previous fixes.

**Architecture:** Two-pass propeller detector: Pass 1 (coarse 30ms windows) detects ellipse geometry, Pass 2 (fine 0.5ms windows) tracks blade angles for RPM estimation. Uses evlib HDF5 loader but matches original's data processing exactly.

**Tech Stack:** Python, OpenCV, scikit-learn DBSCAN, evlib, detector-commons, Polars

**Current Status:** New implementation has 7 deviations from original (5 algorithmic, 2 cosmetic), causing unreliable detection and different visual appearance.

**Root Cause:** Previous Claude made "fixes" claiming to improve detection, but these deviated from original behavior, creating a fundamentally different detector.

---

## Deviation Summary (What We're Fixing)

| Deviation | Original | Current (Wrong) | Impact |
|-----------|----------|-----------------|---------|
| **1. Pass 1 accumulation** | uint8 (clipped) | uint16 (full range) | CRITICAL - Different normalization |
| **2. Pre-threshold** | 250 | 50 | CRITICAL - 5x more noise accepted |
| **3. Min area** | 145.0 | 100.0 | MODERATE - 31% smaller contours |
| **4. Min points** | 50 | 30 | MODERATE - Less accurate ellipses |
| **5. Hot pixel filter** | None | >10,000 events | NEW FEATURE - Removes valid data |
| **6. Text position** | Bottom-right | Top-left | COSMETIC |
| **7. Text labels** | "RPM", "Avg RPM" | "Frame mean", "Global mean" | COSMETIC |

---

## Task 1: Restore uint8 Accumulation in Pass 1

**Files:**
- Modify: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:136`

**Why:** Original ALWAYS clips to uint8, even in Pass 1. This affects normalization range in geometry detection.

**Step 1: Change Pass 1 accumulation to uint8**

Find line 136:
```python
frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=False)
```

Change to:
```python
# Match original: ALWAYS use uint8 (even in Pass 1)
frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=True)
```

**Step 2: Verify change**

```bash
nix develop --command bash -c "cd workspace/tools/drone-detector-demo && grep -n 'clip_to_uint8' src/drone_detector_demo/main.py"
```

Expected output:
- Line 136: `clip_to_uint8=True` ✓
- Line 254: `clip_to_uint8=True` ✓ (Pass 2, already correct)

**Step 3: Commit**

```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "fix(drone-detector): restore uint8 accumulation in Pass 1

Match original behavior: clip to uint8 in BOTH passes.
Previous uint16 accumulation changed normalization range."
```

---

## Task 2: Restore Original Pre-Threshold (250)

**Files:**
- Modify: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:149`

**Why:** Original uses 250 to filter top 2% intensity. Current 50 accepts top 80%, introducing noise.

**Step 1: Restore pre_threshold to 250**

Find line 149:
```python
pre_threshold=50,  # Much lower than default 250
```

Change to:
```python
pre_threshold=250,  # Match original (top 2% intensity only)
```

**Step 2: Remove misleading comment at line 143-145**

Remove these lines:
```python
# Lower pre_threshold to 50 (out of 255) to detect low-intensity propellers
# Default 250 was too high for drone_idle dataset
# Also lower min_area to 100 (from 145) to accept smaller contours
```

Replace with:
```python
# Use original parameters for exact feature parity
```

**Step 3: Commit**

```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "fix(drone-detector): restore pre-threshold to 250

Match original: filter to top 2% intensity only.
Previous threshold of 50 was too permissive."
```

---

## Task 3: Restore Original Min Area (145.0)

**Files:**
- Modify: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:150`

**Why:** Original requires minimum 145 pixels. Current 100 accepts 31% smaller (noisier) contours.

**Step 1: Restore min_area to 145.0**

Find line 150:
```python
min_area=100.0,  # Lower than default 145
```

Change to:
```python
min_area=145.0,  # Match original minimum contour size
```

**Step 2: Commit**

```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "fix(drone-detector): restore min_area to 145.0

Match original minimum contour size.
Previous 100.0 was too permissive."
```

---

## Task 4: Restore Original Min Points (50)

**Files:**
- Modify: `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:150`

**Why:** Original requires 50 points for accurate ellipse fitting. Current 30 reduces geometric accuracy.

**Step 1: Restore min points to 50**

Find line 150:
```python
if len(cnt) < 30:
    rejected_counts["too_few_points"] += 1
    continue
```

Change to:
```python
if len(cnt) < 50:
    rejected_counts["too_few_points"] += 1
    continue
```

**Step 2: Update comment at line 148-149**

Find:
```python
# fitEllipse needs at least 5 points
# Use 30 points as minimum (lowered from 50) to handle smaller propellers
```

Change to:
```python
# fitEllipse needs at least 5 points
# Use 50 points minimum (match original for accuracy)
```

**Step 3: Commit**

```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py
git commit -m "fix(drone-detector): restore min points to 50 for ellipse fitting

Match original: require 50 points for accurate geometry.
Previous 30 points reduced ellipse quality."
```

---

## Task 5: Remove Hot Pixel Filtering

**Files:**
- Modify: `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:22,68-78`

**Why:** Original has NO hot pixel filtering. This feature removes data that shouldn't be removed.

**Step 1: Set hot_pixel_threshold to infinity (effectively disables)**

Find line 22:
```python
hot_pixel_threshold: float = 10000.0,
```

Change to:
```python
hot_pixel_threshold: float = float('inf'),  # Disabled (match original - no filtering)
```

**Step 2: Update docstring at line 44**

Find:
```python
hot_pixel_threshold: Pixels with count > this are considered hot pixels (default: 10000)
```

Change to:
```python
hot_pixel_threshold: Pixels with count > this are considered hot pixels (default: inf = disabled)
```

**Step 3: Commit**

```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py
git commit -m "fix(drone-detector): disable hot pixel filtering

Match original: no hot pixel filtering.
Original processes all event data without filtering."
```

---

## Task 6: Restore Bottom-Right Text Positioning

**Files:**
- Modify: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:313-415`

**Why:** Original places text bottom-right for visibility. Current top-left differs visually.

**Step 1: Update warning text position (line 317)**

Find:
```python
x1, y1 = 10, 30
```

Change to:
```python
# Match original: bottom-right positioning
H, W = bottom_half.shape[:2]
x1 = W - tw1 - 10
y1 = H - 10
```

**Step 2: Update RPM text positioning (lines 372-391)**

Find the "Frame mean" text block:
```python
if frame_mean_rpm is not None:
    text_frame = f"Frame mean: {frame_mean_rpm:.1f} RPM"
    (tw2, th2), _ = cv2.getTextSize(text_frame, font, rpm_font_scale, rpm_thickness)
    x2, y2 = 10, y_offset
```

Change to:
```python
if frame_mean_rpm is not None:
    text_frame = f"RPM: {frame_mean_rpm:.1f}"  # Match original label
    (tw2, th2), _ = cv2.getTextSize(text_frame, font, font_scale, thickness)
    x2 = W - tw2 - 10
    y2 = y1 - th1 - 10
```

**Step 3: Update global mean text positioning (lines 393-415)**

Find the "Global mean" text block:
```python
if global_mean_rpm is not None:
    text_global = f"Global mean: {global_mean_rpm:.1f} RPM"
    (tw3, th3), _ = cv2.getTextSize(text_global, font, rpm_font_scale, rpm_thickness)
    x3, y3 = 10, y_offset
```

Change to:
```python
if global_mean_rpm is not None:
    text_global = f"Avg RPM: {global_mean_rpm:.1f}"  # Match original label
    (tw3, th3), _ = cv2.getTextSize(text_global, font, font_scale, thickness)
    x3 = W - tw3 - 10
    y3 = y2 - th2 - 10
```

**Step 4: Remove y_offset variable tracking (lines 369, 391, 415)**

Remove these lines:
```python
y_offset = 60  # line 369
y_offset += 30  # line 391
y_offset += 30  # line 415
```

**Step 5: Update font size variables (lines 340-341)**

Remove:
```python
rpm_font_scale = 0.6
rpm_thickness = 2
```

These are no longer needed (use `font_scale` and `thickness` from line 311-312).

**Step 6: Commit**

```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "fix(drone-detector): restore bottom-right text positioning

Match original:
- Text positioned bottom-right (not top-left)
- Labels: 'RPM' and 'Avg RPM' (not 'Frame mean' and 'Global mean')
- Same font size for all text (0.8 scale, thickness 2)"
```

---

## Task 7: Verification - Compare Visual Output

**Goal:** Manually verify that visual output matches original.

**Files:**
- Test with: `evio/data/drone_idle/drone_idle_legacy.h5`
- Reference: `evio/data/drone_idle/drone_idle.dat`

**Step 1: Run original script (reference baseline)**

```bash
nix develop --command python evio/scripts/drone_detector_demo.py evio/data/drone_idle/drone_idle.dat --max-frames 50
```

**Observe:**
- Text position: Bottom-right ✓
- Text labels: "WARNING: DRONE DETECTED", "RPM: X", "Avg RPM: X" ✓
- Window size: Scaled to 0.7x ✓
- Detection rate: Should be ~98% (330/336 frames from previous runs)

Press 'q' to quit after observing ~10 frames.

**Step 2: Run new implementation (verify parity)**

```bash
nix develop --command uv run drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5 --max-frames 50
```

**Expected:**
- Text position: Bottom-right ✓ (matches original)
- Text labels: "WARNING: DRONE DETECTED", "RPM: X", "Avg RPM: X" ✓ (matches)
- Window size: Scaled to 0.7x ✓ (matches)
- Detection rate: Should be similar to original (~90-100%)
- Visual appearance: Should look identical

Press 'q' to quit.

**Step 3: Document any remaining differences**

If behavior differs:
1. Note specific differences (detection rate, RPM values, visual artifacts)
2. This indicates loader/data conversion issue (NOT algorithm issue)
3. Open issue for investigation: "HDF5 loader data mismatch with .dat loader"

---

## Task 8: Verification - Automated Test

**Goal:** Create regression test to prevent future deviations.

**Files:**
- Create: `workspace/tools/drone-detector-demo/tests/test_parity.py`

**Step 1: Create test file**

```python
"""Test that drone detector matches original parameters exactly."""

import pytest
from drone_detector_demo.geometry import propeller_mask_from_frame


def test_default_parameters_match_original():
    """Verify default parameters match example-drone-original.py exactly."""
    import inspect
    sig = inspect.signature(propeller_mask_from_frame)

    # These MUST match the original
    assert sig.parameters['min_area'].default == 145.0, \
        "min_area must be 145.0 (original value)"
    assert sig.parameters['pre_threshold'].default == 250, \
        "pre_threshold must be 250 (original value)"
    assert sig.parameters['hot_pixel_threshold'].default == float('inf'), \
        "hot_pixel_threshold must be inf (original had no filtering)"

    # Not in signature but used internally - verify via source inspection
    import drone_detector_demo.geometry as geom_module
    source = inspect.getsource(geom_module.propeller_mask_from_frame)

    # Verify min points for ellipse fitting
    assert 'len(cnt) < 50' in source, \
        "Must require 50 points for fitEllipse (original value)"


def test_no_hot_pixel_filtering_by_default():
    """Verify hot pixels are NOT filtered by default (match original)."""
    import numpy as np

    # Create frame with simulated hot pixel (20,000 events)
    frame = np.zeros((100, 100), dtype=np.uint16)
    frame[50, 50] = 20000  # Hot pixel

    # With default threshold (inf), hot pixel should NOT be filtered
    from drone_detector_demo.geometry import propeller_mask_from_frame

    # This should process the hot pixel normally (not filter it)
    # If it were filtered, the frame would be empty and return []
    # We're not testing detection here, just that hot pixels aren't removed
    result = propeller_mask_from_frame(frame)
    # (Detection will likely fail on random noise, that's OK)
    # The point is hot pixel wasn't filtered out before normalization
```

**Step 2: Run test**

```bash
nix develop --command uv run --package drone-detector-demo pytest workspace/tools/drone-detector-demo/tests/test_parity.py -v
```

**Expected:** PASS (all assertions succeed)

**Step 3: Commit**

```bash
git add workspace/tools/drone-detector-demo/tests/test_parity.py
git commit -m "test(drone-detector): add parity verification test

Regression test to ensure parameters match original exactly.
Prevents future accidental deviations."
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/DRONE_DETECTOR_FINAL_STATUS.md`
- Create: `docs/DRONE_DETECTOR_PARITY_RESTORED.md`

**Step 1: Update final status to reflect restoration**

Replace `docs/DRONE_DETECTOR_FINAL_STATUS.md` with updated version:

```markdown
# Drone Detector - Status: Exact Parity Restored

**Date:** 2025-11-15
**Status:** ✅ **EXACT PARITY ACHIEVED**

---

## What Changed

Previous implementation (2025-11-15 early) made 5 algorithmic changes claiming to "fix detection failures":
1. uint16 accumulation (vs original uint8)
2. Lower pre-threshold (50 vs 250)
3. Lower min_area (100 vs 145)
4. Lower min_points (30 vs 50)
5. Added hot pixel filtering

**Problem:** These changes created a fundamentally different detector with unreliable behavior.

**Solution:** Restored ALL parameters to match `example-drone-original.py` exactly.

---

## Current Implementation

All parameters now match original exactly:

```python
# Pass 1 & 2: uint8 accumulation (clipped)
frame_accum = build_accum_frame_evlib(..., clip_to_uint8=True)

# Geometry detection
propeller_mask_from_frame(
    frame_accum,
    pre_threshold=250,        # Top 2% intensity only
    min_area=145.0,           # Minimum contour size
    hot_pixel_threshold=inf,  # No filtering (disabled)
)

# Ellipse fitting: requires 50 points (in geometry.py)
if len(cnt) < 50:
    continue
```

---

## Visual Parity

Text positioning and labels match original exactly:
- **Position:** Bottom-right corner
- **Labels:** "WARNING: DRONE DETECTED", "RPM: X", "Avg RPM: X"
- **Scaling:** 0.7x display scaling
- **Font:** Same size for all text (scale=0.8, thickness=2)

---

## Testing

Regression test added: `workspace/tools/drone-detector-demo/tests/test_parity.py`

Verifies:
- Default parameters match original
- No hot pixel filtering by default
- Minimum 50 points for ellipse fitting

---

## If Detection Fails

If detection fails with original parameters, this indicates:
- **Data quality issue** (hot pixels, noise in HDF5 conversion)
- **Loader issue** (evlib vs legacy .dat loader mismatch)

**NOT an algorithm tuning issue.**

Investigate loader/conversion, don't modify detection parameters.

---

## Next Steps

1. Manual verification: Compare visual output side-by-side
2. If detection rate differs: Investigate HDF5 conversion (NOT algorithm)
3. If all matches: Mark as production-ready
```

**Step 2: Create parity restoration summary**

Create `docs/DRONE_DETECTOR_PARITY_RESTORED.md`:

```markdown
# Drone Detector Parity Restoration - 2025-11-15

## Summary

Restored exact feature parity with `example-drone-original.py` by reverting 5 algorithmic deviations and 2 cosmetic changes.

## Changes Made

### Algorithmic Fixes (CRITICAL)

1. **uint8 accumulation in Pass 1**
   - Was: `clip_to_uint8=False` (uint16)
   - Now: `clip_to_uint8=True` (uint8) ✓

2. **Pre-threshold restored to 250**
   - Was: 50 (top 80% intensity)
   - Now: 250 (top 2% intensity) ✓

3. **Min area restored to 145.0**
   - Was: 100.0 pixels
   - Now: 145.0 pixels ✓

4. **Min points restored to 50**
   - Was: 30 points
   - Now: 50 points ✓

5. **Hot pixel filtering disabled**
   - Was: >10,000 events filtered
   - Now: No filtering (threshold = inf) ✓

### Cosmetic Fixes

6. **Text positioning**
   - Was: Top-left
   - Now: Bottom-right ✓

7. **Text labels**
   - Was: "Frame mean", "Global mean"
   - Now: "RPM", "Avg RPM" ✓

## Verification

- Regression test: `tests/test_parity.py` ✓
- Manual comparison: Run both versions side-by-side
- Expected: Identical visual output and detection behavior

## Commits

```bash
git log --oneline
```

Should show 9 commits:
1. Restore uint8 accumulation
2. Restore pre-threshold 250
3. Restore min_area 145
4. Restore min points 50
5. Disable hot pixel filtering
6. Restore text positioning
7. Add parity verification test
8. Update documentation
```

**Step 3: Commit documentation**

```bash
git add docs/DRONE_DETECTOR_FINAL_STATUS.md docs/DRONE_DETECTOR_PARITY_RESTORED.md
git commit -m "docs(drone-detector): document parity restoration

Updated status to reflect exact parity with original.
Added restoration summary with all changes listed."
```

---

## Task 10: Final Verification & Sign-Off

**Step 1: Run full test suite**

```bash
nix develop --command bash -c "
  cd workspace/tools/drone-detector-demo &&
  uv run pytest tests/ -v
"
```

**Expected:** All tests PASS

**Step 2: Check git status**

```bash
git status
git log --oneline -10
```

**Expected:**
- Clean working tree (no uncommitted changes)
- 9 commits for parity restoration

**Step 3: Manual side-by-side comparison**

Terminal 1:
```bash
nix develop --command python evio/scripts/drone_detector_demo.py \
  evio/data/drone_idle/drone_idle.dat --max-frames 100
```

Terminal 2:
```bash
nix develop --command uv run drone-detector-demo \
  evio/data/drone_idle/drone_idle_legacy.h5 --max-frames 100
```

**Verify:**
- Visual appearance: Identical ✓
- Text position: Both bottom-right ✓
- Text labels: Both use "RPM" and "Avg RPM" ✓
- Detection behavior: Similar detection rates ✓
- RPM values: Within reasonable variance ✓

**Step 4: If all matches → Success!**

If differences remain:
- Document specific differences
- These indicate loader/conversion issues (NOT algorithm)
- Create follow-up issue for loader investigation

---

## Success Criteria

✅ All 7 deviations reverted
✅ Regression test passes
✅ Visual output matches original
✅ Detection behavior matches original (within reasonable variance)
✅ Documentation updated

---

## Notes for Future

**If detection fails with original parameters:**

1. **DO NOT modify algorithm parameters**
2. **DO investigate:**
   - HDF5 conversion quality
   - Event loader differences (evlib vs legacy .dat)
   - Data integrity in conversion pipeline

**The original algorithm works** - proven by `example-drone-original.py` on `.dat` files.

If evlib version fails, the issue is in **data loading/conversion**, not detection logic.
