# Drone Detector Parity Restoration - Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan.

**Date:** 2025-11-15
**Goal:** Restore exact behavioral parity with `example-drone-original.py` (the legacy .dat script)
**Approach:** Fix 6 deviations identified by Codex in the corrected plan

**Current Issues:**
1. ❌ Rendering downscaled to 70% (should be 100% or optional)
2. ❌ Pass 1 uses uint16 accumulation (should be uint8)
3. ❌ Detection params too relaxed (pre_threshold=50, min_area=100, min_points=30)
4. ❌ DBSCAN params hardcoded (should respect CLI or match original)
5. ❌ HUD positioned top-left (should be bottom-right)
6. ❌ HUD labels differ ("Frame mean" vs "RPM")

---

## Task 1: Remove 70% Downscaling in Pass 2

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:418-421`

**Current code (lines 418-421):**
```python
stacked = np.vstack([vis, bottom_half])
scale = 0.7
stacked_small = cv2.resize(stacked, None, fx=scale, fy=scale)
cv2.imshow("Events + Propeller mask + Speed", stacked_small)
```

**Change to:**
```python
stacked = np.vstack([vis, bottom_half])
# Match original: NO downscaling (show at full resolution)
cv2.imshow("Events + Propeller mask + Speed", stacked)
```

**Rationale:** Original shows full-size view. 70% downscaling makes window "tiny" and hard to see.

**Verification:**
```bash
grep -n "cv2.imshow.*stacked" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
```
Should show line with `stacked)` not `stacked_small)`.

---

## Task 2: Restore uint8 Accumulation in Pass 1

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:136`

**Current code (line 136):**
```python
frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=False)
```

**Change to:**
```python
# Match original: use uint8 accumulation (clips to 0-255)
frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=True)
```

**Rationale:** Original always clips to uint8 (even in Pass 1). This affects normalization range in geometry detection.

**Verification:**
```bash
grep -n "clip_to_uint8" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
```
Should show both lines 136 and 254 with `=True`.

---

## Task 3: Restore Original Detection Parameters

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:143-152`

**Current code (lines 143-152):**
```python
# Lower pre_threshold to 50 (out of 255) to detect low-intensity propellers
# Default 250 was too high for drone_idle dataset
# Also lower min_area to 100 (from 145) to accept smaller contours
ellipses = propeller_mask_from_frame(
    frame_accum,
    max_ellipses=args.max_ellipses,
    pre_threshold=50,  # Much lower than default 250
    min_area=100.0,  # Lower than default 145
    debug=args.debug and processed_count < 5,
)
```

**Change to:**
```python
# Match original detection parameters exactly
ellipses = propeller_mask_from_frame(
    frame_accum,
    max_ellipses=args.max_ellipses,
    pre_threshold=250,  # Original: top 2% intensity only
    min_area=145.0,     # Original: minimum contour size
    debug=args.debug and processed_count < 5,
)
```

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:150`

**Current code (line 150):**
```python
if len(cnt) < 30:
```

**Change to:**
```python
if len(cnt) < 50:  # Match original: 50 points for accurate ellipse fitting
```

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:22`

**Current code (line 22):**
```python
hot_pixel_threshold: float = 10000.0,
```

**Change to:**
```python
hot_pixel_threshold: float = float('inf'),  # Match original: no hot pixel filtering
```

**Rationale:** Original uses strict thresholds (250, 145, 50, no filtering). Previous "fixes" relaxed these, causing unreliable detection.

**Verification:**
```bash
# Check main.py
grep -A2 "pre_threshold=" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py | head -6

# Check geometry.py
grep "len(cnt) <" workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py
grep "hot_pixel_threshold.*=" workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py | head -1
```

---

## Task 4: Restore Bottom-Right HUD Positioning

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:309-415`

**Current approach:** Top-left positioning with y_offset tracking

**Target:** Bottom-right positioning matching original (lines 517-582 of example-drone-original.py)

**Changes needed:**

### 4a. Warning text (currently line 317)

**Current:**
```python
x1, y1 = 10, 30
```

**Change to:**
```python
# Match original: bottom-right positioning
H, W = bottom_half.shape[:2]
x1 = W - tw1 - 10
y1 = H - 10
```

### 4b. Frame RPM text (currently lines 370-391)

**Current:**
```python
if frame_mean_rpm is not None:
    text_frame = f"Frame mean: {frame_mean_rpm:.1f} RPM"
    (tw2, th2), _ = cv2.getTextSize(text_frame, font, rpm_font_scale, rpm_thickness)
    x2, y2 = 10, y_offset
```

**Change to:**
```python
if frame_mean_rpm is not None:
    text_rpm = f"RPM: {frame_mean_rpm:.1f}"  # Match original label
    (tw2, th2), _ = cv2.getTextSize(text_rpm, font, font_scale, thickness)
    x2 = W - tw2 - 10
    y2 = y1 - th1 - 10
```

### 4c. Global mean RPM text (currently lines 393-415)

**Current:**
```python
if global_mean_rpm is not None:
    text_global = f"Global mean: {global_mean_rpm:.1f} RPM"
    (tw3, th3), _ = cv2.getTextSize(text_global, font, rpm_font_scale, rpm_thickness)
    x3, y3 = 10, y_offset
```

**Change to:**
```python
if global_mean_rpm is not None:
    text_mean = f"Avg RPM: {global_mean_rpm:.1f}"  # Match original label
    (tw3, th3), _ = cv2.getTextSize(text_mean, font, font_scale, thickness)
    x3 = W - tw3 - 10
    y3 = y2 - th2 - 10
```

### 4d. Remove y_offset tracking and rpm_font_scale

**Delete lines:**
- Line 340: `rpm_font_scale = 0.6`
- Line 341: `rpm_thickness = 2`
- Line 369: `y_offset = 60`
- Line 391: `y_offset += 30`
- Line 415: `y_offset += 30`

**Rationale:** Use `font_scale` and `thickness` from line 311-312 for all text.

**Verification:**
```bash
# Check bottom-right positioning
grep "W - tw" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py

# Check labels
grep '"RPM:' workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
grep '"Avg RPM:' workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py

# Verify no y_offset
grep "y_offset" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
```
Should show no y_offset lines.

---

## Task 5: Verification - Visual Comparison

**Goal:** Verify output matches original exactly

**Step 1: Run original (baseline)**
```bash
nix develop --command python evio/scripts/drone_detector_demo.py \
  evio/data/drone_idle/drone_idle.dat --max-frames 50
```

**Observe:**
- Window size: Full resolution (1280x720 stacked = 1280x1440)
- Text: Bottom-right, labels "WARNING", "RPM", "Avg RPM"
- Detection: Should detect propellers in most frames

**Step 2: Run new implementation**
```bash
nix develop --command uv run drone-detector-demo \
  evio/data/drone_idle/drone_idle_legacy.h5 --max-frames 50
```

**Expected matches:**
- ✅ Window size: Full resolution (same as original)
- ✅ Text position: Bottom-right
- ✅ Text labels: "WARNING: DRONE DETECTED", "RPM: X", "Avg RPM: X"
- ✅ Detection rate: Similar to original (~90-100%)
- ✅ Visual appearance: Identical

**Step 3: Document results**

If differences remain, note:
- Specific difference (e.g., "Detection rate 60% vs 98%")
- Likely cause (e.g., "HDF5 loader issue" NOT "algorithm issue")

---

## Task 6: Create Regression Test

**File:** Create `workspace/tools/drone-detector-demo/tests/test_parity.py`

**Content:**
```python
"""Regression test: verify parameters match original exactly."""

import pytest


def test_default_parameters_match_original():
    """Verify default parameters match example-drone-original.py."""
    import inspect
    from drone_detector_demo.geometry import propeller_mask_from_frame

    sig = inspect.signature(propeller_mask_from_frame)

    # Critical parameters MUST match original
    assert sig.parameters['pre_threshold'].default == 250, \
        "pre_threshold must be 250 (original value, not 50)"
    assert sig.parameters['min_area'].default == 145.0, \
        "min_area must be 145.0 (original value, not 100.0)"
    assert sig.parameters['hot_pixel_threshold'].default == float('inf'), \
        "hot_pixel_threshold must be inf (original had no filtering)"

    # Verify min_points via source inspection
    import drone_detector_demo.geometry as geom
    source = inspect.getsource(geom.propeller_mask_from_frame)
    assert 'len(cnt) < 50' in source, \
        "Must require 50 points for fitEllipse (original value, not 30)"


def test_no_downscaling_in_visualization():
    """Verify Pass 2 visualization shows full resolution (no 0.7x scaling)."""
    import drone_detector_demo.main as main_module
    source = main_module.__file__

    with open(source, 'r') as f:
        content = f.read()

    # Should NOT have 0.7 scale factor
    assert 'scale = 0.7' not in content, \
        "Pass 2 should show full resolution (no 70% downscaling)"
    assert 'stacked_small' not in content, \
        "Should display 'stacked' directly, not downscaled version"
```

**Verification:**
```bash
nix develop --command uv run --package drone-detector-demo pytest \
  workspace/tools/drone-detector-demo/tests/test_parity.py -v
```

**Expected:** All tests PASS

---

## Success Criteria

✅ **Task 1:** No 70% downscaling (full-resolution view)
✅ **Task 2:** uint8 accumulation in Pass 1
✅ **Task 3:** Original detection params (250, 145, 50, no filtering)
✅ **Task 4:** Bottom-right HUD with "RPM"/"Avg RPM" labels
✅ **Task 5:** Visual output matches original exactly
✅ **Task 6:** Regression test passes

---

## Commit Strategy

Each task = one commit:

1. `fix(drone-detector): remove 70% downscaling in Pass 2`
2. `fix(drone-detector): restore uint8 accumulation in Pass 1`
3. `fix(drone-detector): restore original detection parameters`
4. `fix(drone-detector): restore bottom-right HUD positioning`
5. `test(drone-detector): add parity regression test`
6. `docs(drone-detector): document parity restoration`

---

## Notes

If detection fails with original parameters:
- This indicates **data/loader issue**, NOT algorithm issue
- Original works on `.dat` files (proven)
- If evlib version fails → investigate HDF5 conversion quality
- **DO NOT relax detection parameters** as a "fix"
