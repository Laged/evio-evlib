# Drone Detector Parity Restoration - VERIFIED Plan

> **CRITICAL:** This plan was created after SYSTEMATIC VERIFICATION against `example-drone-original.py`.
> Previous plan (2025-11-15-drone-detector-parity-execution.md) had FALSE CLAIMS and has been superseded.

**Date:** 2025-11-16
**Goal:** Restore exact behavioral parity with `example-drone-original.py`
**Verification Method:** Every claim below verified by grep/inspection of original file

---

## VERIFIED Deviations (Against Original)

### ✅ Task 1: VERIFIED - Scaling is CORRECT (NO CHANGE NEEDED)
**Claim:** Original uses 0.7x scaling
**Verification:** `example-drone-original.py:587` → `scale = 0.7` ✓
**Status:** **ALREADY MATCHES** - No fix needed
**Previous error:** Earlier plan incorrectly claimed scaling should be removed

---

### ✅ Task 2: VERIFIED - uint8 Accumulation Needed
**Claim:** Original clips to uint8 in Pass 1
**Verification:** `example-drone-original.py:45-47`:
```python
frame = np.zeros((height, width), dtype=np.uint16)
frame[y_coords, x_coords] += 1
frame = np.clip(frame, 0, 255).astype(np.uint8)  # ← ALWAYS uint8
return frame
```
**Current implementation:** `clip_to_uint8=False` (line 136 of main.py)
**Status:** **NEEDS FIX** - Change to `clip_to_uint8=True`

---

### ✅ Task 3: VERIFIED - Detection Parameters Need Restoration
**Claim:** Original uses pre_threshold=250, min_area=145, min_points=50
**Verification:**
- `example-drone-original.py:192` → `threshold(img8, 250, 255, cv2.THRESH_BINARY)` ✓
- `example-drone-original.py:171` → `min_area: float = 145.0,` ✓
- `example-drone-original.py:212` → `if len(cnt) < 50:` ✓

**Current implementation:**
- `pre_threshold=50` (main.py:149)
- `min_area=100.0` (main.py:150)
- `if len(cnt) < 30:` (geometry.py:150)

**Status:** **NEEDS FIX** - Restore to 250, 145.0, 50

---

### ✅ Task 4: VERIFIED - Text Positioning Needs Restoration
**Claim:** Original uses bottom-right positioning
**Verification:** `example-drone-original.py`:
- Line 517: `x1 = W - tw1 - 10` (bottom-right)
- Line 518: `y1 = H - 10`
- Line 541: `x2 = W - tw2 - 10`
- Line 542: `y2 = y1 - th1 - 10`
- Line 564: `x3 = W - tw3 - 10`
- Line 565: `y3 = y2 - th2 - 10`

**Current implementation:** `x1, y1 = 10, 30` (top-left, main.py:317)
**Status:** **NEEDS FIX** - Change to bottom-right

---

### ✅ Task 5: VERIFIED - Text Labels Need Restoration
**Claim:** Original uses "RPM:" and "Avg RPM:"
**Verification:** `example-drone-original.py`:
- Line 539: `text_rpm = f"RPM: {frame_mean_rpm:5.1f}"`
- Line 562: `text_mean = f"Avg RPM: {global_mean_rpm:5.1f}"`

**Current implementation:**
- `f"Frame mean: {frame_mean_rpm:.1f} RPM"` (main.py:371)
- `f"Global mean: {global_mean_rpm:.1f} RPM"` (main.py:395)

**Status:** **NEEDS FIX** - Change labels to match original

---

## Task 1: Restore uint8 Accumulation in Pass 1

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:136`

**Current:**
```python
frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=False)
```

**Change to:**
```python
# Match original: ALWAYS clip to uint8 (even in Pass 1)
frame_accum = build_accum_frame_evlib(window_events, width, height, clip_to_uint8=True)
```

**Verification:**
```bash
grep -n "clip_to_uint8" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
```
Should show both lines 136 and 254 with `=True`.

**Commit:**
```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "fix(drone-detector): restore uint8 accumulation in Pass 1

Match original (line 47): clip to uint8 in BOTH passes.
Previous uint16 accumulation changed normalization range."
```

---

## Task 2: Restore Original Detection Parameters

### 2a. Update main.py parameters

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:143-152`

**Current:**
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
# Match original detection parameters (verified against example-drone-original.py)
ellipses = propeller_mask_from_frame(
    frame_accum,
    max_ellipses=args.max_ellipses,
    pre_threshold=250,  # Original line 192: top 2% intensity only
    min_area=145.0,     # Original line 171: minimum contour size
    debug=args.debug and processed_count < 5,
)
```

### 2b. Update geometry.py min_points

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:150`

**Current:**
```python
if len(cnt) < 30:
```

**Change to:**
```python
if len(cnt) < 50:  # Match original line 212: 50 points for accurate ellipse fitting
```

### 2c. Update geometry.py comment

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:148-149`

**Current:**
```python
# fitEllipse needs at least 5 points
# Use 30 points as minimum (lowered from 50) to handle smaller propellers
```

**Change to:**
```python
# fitEllipse needs at least 5 points
# Use 50 points minimum (match original line 212 for accuracy)
```

### 2d. Disable hot pixel filtering

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py:22`

**Current:**
```python
hot_pixel_threshold: float = 10000.0,
```

**Change to:**
```python
hot_pixel_threshold: float = float('inf'),  # Match original: no hot pixel filtering
```

**Verification:**
```bash
# Check main.py
grep "pre_threshold=" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
grep "min_area=" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py

# Check geometry.py
grep "len(cnt) <" workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py
grep "hot_pixel_threshold.*=" workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py | head -1
```

**Commit:**
```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py \
        workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py
git commit -m "fix(drone-detector): restore original detection parameters

Match example-drone-original.py exactly:
- pre_threshold: 250 (line 192)
- min_area: 145.0 (line 171)
- min_points: 50 (line 212)
- hot_pixel_threshold: inf (no filtering in original)"
```

---

## Task 3: Restore Bottom-Right Text Positioning and Labels

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py:309-415`

### 3a. Warning text (line 317)

**Current:**
```python
x1, y1 = 10, 30
```

**Change to:**
```python
# Match original lines 512-518: bottom-right positioning
H, W = bottom_half.shape[:2]
x1 = W - tw1 - 10
y1 = H - 10
```

### 3b. Frame RPM text (lines 370-391)

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
    text_rpm = f"RPM: {frame_mean_rpm:5.1f}"  # Match original line 539
    (tw2, th2), _ = cv2.getTextSize(text_rpm, font, font_scale, thickness)
    x2 = W - tw2 - 10
    y2 = y1 - th1 - 10
```

### 3c. Global mean RPM text (lines 393-415)

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
    text_mean = f"Avg RPM: {global_mean_rpm:5.1f}"  # Match original line 562
    (tw3, th3), _ = cv2.getTextSize(text_mean, font, font_scale, thickness)
    x3 = W - tw3 - 10
    y3 = y2 - th2 - 10
```

### 3d. Remove y_offset and rpm_font variables

**Delete these lines:**
- Line 340: `rpm_font_scale = 0.6`
- Line 341: `rpm_thickness = 2`
- Line 369: `y_offset = 60`
- Line 391: `y_offset += 30`
- Line 415: `y_offset += 30`

**Verification:**
```bash
# Check bottom-right positioning
grep "W - tw" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py

# Check labels match original
grep '"RPM:' workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
grep '"Avg RPM:' workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py

# Verify no y_offset
grep "y_offset" workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
```

**Commit:**
```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "fix(drone-detector): restore bottom-right HUD positioning and labels

Match example-drone-original.py exactly:
- Text positioned bottom-right (lines 517-565)
- Labels: 'RPM:' (line 539) and 'Avg RPM:' (line 562)
- Format: {value:5.1f} (matches original)
- Same font size for all text"
```

---

## Task 4: Create Regression Test

**File:** Create `workspace/tools/drone-detector-demo/tests/test_parity.py`

**Content:**
```python
"""Regression test: verify parameters match example-drone-original.py exactly."""

import pytest


def test_default_parameters_match_original():
    """Verify default parameters match example-drone-original.py."""
    import inspect
    from drone_detector_demo.geometry import propeller_mask_from_frame

    sig = inspect.signature(propeller_mask_from_frame)

    # Verified against example-drone-original.py
    assert sig.parameters['pre_threshold'].default == 250, \
        "pre_threshold must be 250 (original line 192, not 50)"
    assert sig.parameters['min_area'].default == 145.0, \
        "min_area must be 145.0 (original line 171, not 100.0)"
    assert sig.parameters['hot_pixel_threshold'].default == float('inf'), \
        "hot_pixel_threshold must be inf (original has no filtering)"

    # Verify min_points via source inspection
    import drone_detector_demo.geometry as geom
    source = inspect.getsource(geom.propeller_mask_from_frame)
    assert 'len(cnt) < 50' in source, \
        "Must require 50 points for fitEllipse (original line 212, not 30)"


def test_text_labels_match_original():
    """Verify HUD text labels match example-drone-original.py."""
    import drone_detector_demo.main as main_module

    with open(main_module.__file__, 'r') as f:
        content = f.read()

    # Verified against example-drone-original.py lines 539, 562
    assert '"RPM:' in content, \
        "Must use 'RPM:' label (original line 539, not 'Frame mean')"
    assert '"Avg RPM:' in content, \
        "Must use 'Avg RPM:' label (original line 562, not 'Global mean')"


def test_scaling_matches_original():
    """Verify 0.7x scaling is preserved (matches example-drone-original.py line 587)."""
    import drone_detector_demo.main as main_module

    with open(main_module.__file__, 'r') as f:
        content = f.read()

    # Original DOES use 0.7x scaling (verified line 587)
    assert 'scale = 0.7' in content, \
        "Must preserve 0.7x scaling (matches original line 587)"
```

**Verification:**
```bash
nix develop --command uv run --package drone-detector-demo pytest \
  workspace/tools/drone-detector-demo/tests/test_parity.py -v
```

**Commit:**
```bash
git add workspace/tools/drone-detector-demo/tests/test_parity.py
git commit -m "test(drone-detector): add verified parity regression test

Tests verify parameters match example-drone-original.py exactly:
- Detection params (250, 145, 50, inf)
- Text labels ('RPM:', 'Avg RPM:')
- 0.7x scaling (verified original DOES scale)"
```

---

## Task 5: Visual Verification

**Step 1: Run original**
```bash
nix develop --command python evio/scripts/drone_detector_demo.py \
  evio/data/drone_idle/drone_idle.dat --max-frames 50
```

**Step 2: Run new implementation**
```bash
nix develop --command uv run drone-detector-demo \
  evio/data/drone_idle/drone_idle_legacy.h5 --max-frames 50
```

**Expected matches:**
- ✅ Window size: Scaled 0.7x (both)
- ✅ Text position: Bottom-right (both)
- ✅ Text labels: "WARNING: DRONE DETECTED", "RPM: X", "Avg RPM: X" (both)
- ✅ Detection rate: Similar (~90-100%)
- ✅ Visual appearance: Identical

---

## Success Criteria

✅ **Task 1:** uint8 accumulation in Pass 1
✅ **Task 2:** Original detection params (250, 145, 50, inf)
✅ **Task 3:** Bottom-right HUD with "RPM:"/"Avg RPM:" labels
✅ **Task 4:** Regression test passes
✅ **Task 5:** Visual output matches original

---

## Verification Summary

All claims verified against `example-drone-original.py`:
- Line 47: uint8 clipping ✓
- Line 192: pre_threshold=250 ✓
- Line 171: min_area=145.0 ✓
- Line 212: len(cnt)<50 ✓
- Line 587: scale=0.7 ✓
- Lines 517-565: bottom-right text ✓
- Lines 539, 562: "RPM:", "Avg RPM:" ✓
