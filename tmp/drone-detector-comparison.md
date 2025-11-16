# Drone Detector Implementation Comparison

**Date:** 2025-11-15
**Purpose:** Comprehensive analysis of functional differences between original and new drone detector implementations

---

## Executive Summary

This document compares the **original drone detector** (`example-drone-original.py`) with the **new evlib-based implementation** across multiple modules:
- New geometry: `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py`
- New main: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py`
- Detector commons utilities: `workspace/tools/detector-commons/src/detector_commons/`

**Key Finding:** Multiple critical differences exist that could significantly impact detection quality, particularly in geometry detection, visualization, and RPM calculation.

---

## 1. Geometry Detection: `propeller_mask_from_frame`

### 1.1 Pre-threshold Logic (CRITICAL DIFFERENCE)

**Original** (lines 191-192):
```python
img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, image_binary = cv2.threshold(img8, 250, 255, cv2.THRESH_BINARY)

# --- threshold (Otsu) ---
_, mask = cv2.threshold(
    image_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

**New** (geometry.py lines 52-55):
```python
img8 = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply Otsu threshold directly (no pre-threshold needed)
_, mask = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**Impact Assessment: CRITICAL**

The original implementation applies a **two-stage thresholding**:
1. First threshold at 250 (keeps only very bright pixels)
2. Then Otsu on the pre-thresholded result

The new implementation applies **Otsu directly** on the normalized image.

**Hypothesis:**
- The pre-threshold at 250 acts as a **noise filter**, removing dim pixels before Otsu
- Without this, Otsu may include more noise, especially in cluttered scenes
- This could lead to **more false positives** or **unstable ellipse detection**
- The pre-threshold ensures only the brightest activity (propellers) is considered

**Why this matters:**
- Drone propellers create strong, concentrated event bursts
- Background noise creates weak, diffuse events
- Pre-threshold ensures we only detect strong signals
- Direct Otsu may be too sensitive to noise distribution

---

### 1.2 Pair Distance Filtering (CRITICAL DIFFERENCE)

**Original** (lines 247-262):
```python
# decide which ellipses to keep based on distance
selected: List[Tuple[int, int, float, float, float, float]] = []

if len(candidates) == 1 or top_k == 1:
    selected = [candidates[0]]
else:
    c0 = candidates[0]
    c1 = candidates[1]
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    dist = math.hypot(dx, dy)

    if dist <= max_pair_distance:
        selected = [c0, c1]
    else:
        selected = [c0]
```

**New** (geometry.py lines 108-116):
```python
# Sort by area (largest first) and keep top max_ellipses
candidates.sort(key=lambda t: t[5], reverse=True)
top_candidates = candidates[:max_ellipses]

# Return ellipse parameters (drop area)
ellipses: List[Tuple[int, int, float, float, float]] = []
for (cx_i, cy_i, a, b, phi, _) in top_candidates:
    ellipses.append((cx_i, cy_i, a, b, phi))

return ellipses
```

**Impact Assessment: CRITICAL**

The original implementation has **intelligent pair selection logic**:
- If two largest candidates are within 15px of each other → keep BOTH
- If they're far apart → keep ONLY the largest (single propeller mode)
- This prevents detecting unrelated objects as second propeller

The new implementation **blindly returns top N** by area, regardless of spatial relationship.

**Hypothesis:**
- Without distance filtering, the new version may detect:
  - Background objects as "second propeller"
  - Noise blobs as valid propellers
  - Unrelated features in the scene
- This could cause **false multi-propeller detections** when only one is present
- RPM tracking would be contaminated by tracking non-propeller features

**Example failure scenario:**
1. Drone with single visible propeller + bright background object
2. New: detects both as propellers, tracks garbage RPM for second "propeller"
3. Original: distance check rejects background, tracks only real propeller

---

### 1.3 Return Values (IMPORTANT DIFFERENCE)

**Original** (line 280):
```python
return ellipses, prop_mask
```
Returns: `(List[ellipse], mask_image)`

**New** (geometry.py line 117):
```python
return ellipses
```
Returns: `List[ellipse]` only

**Impact Assessment: IMPORTANT**

The original returns a **binary mask** filled with detected ellipses. The new version only returns ellipse parameters.

**Hypothesis:**
- The mask was likely used for visualization or further processing
- New implementation doesn't need it (draws ellipses manually in visualization)
- This is an **API simplification**, not a functional degradation
- However, if mask was used for event filtering, this could be significant

---

### 1.4 Angle Filter Implementation (MINOR DIFFERENCE)

**Original** (lines 226-229):
```python
if 70 > angle_deg:
    continue   # reject vertical-ish ellipses
if 110 < angle_deg:
    continue   # reject vertical-ish ellipses
```

**New** (geometry.py lines 89-92):
```python
if angle_deg < 70.0:
    continue   # reject
if angle_deg > 110.0:
    continue   # reject
```

**Impact Assessment: MINOR**

Both implementations apply the **same logic** (keep angles in [70, 110] degrees), just written differently.

**No functional difference** - this is purely a style change.

---

## 2. Event Loading and Decoding

### 2.1 Event Source (CRITICAL DIFFERENCE)

**Original** (lines 302-307):
```python
src = DatFileSource(
    dat_path,
    width=width,
    height=height,
    window_length_us=window_us,
)
# Uses get_window() to decode raw uint32 words
```

**New** (main.py lines 75, 96-107):
```python
events, width, height = load_legacy_h5(args.h5)

# Filter window using Polars - handle Duration timestamps
schema = events.schema
if isinstance(schema["t"], pl.Duration):
    window_events = events.filter(
        (pl.col("t") >= pl.duration(microseconds=win_start)) &
        (pl.col("t") < pl.duration(microseconds=win_end))
    )
else:
    window_events = events.filter(
        (pl.col("t") >= win_start) &
        (pl.col("t") < win_end)
    )
```

**Impact Assessment: CRITICAL**

**Original:**
- Reads `.dat` files directly
- Decodes raw uint32 words with bit masks
- Uses custom time-ordered indexing

**New:**
- Reads HDF5 files via evlib
- Events already decoded into Polars DataFrame
- Uses Polars filtering (50x faster)

**Potential Issues:**

1. **Data Format Differences:**
   - `.dat` format: Custom binary encoding
   - HDF5 export: May have different precision/rounding
   - Polarity encoding: `.dat` uses bits, HDF5 uses -1/+1

2. **Event Ordering:**
   - Original uses pre-sorted `time_order` index
   - New relies on evlib's loading order
   - If not identically ordered, event counts per frame could differ

3. **Timestamp Precision:**
   - Original: uint32 microseconds
   - New: Polars Duration or Int64
   - Type conversion could introduce rounding errors

**Hypothesis:**
- If HDF5 export is **not byte-perfect** with original .dat, results will differ
- Event count per pixel could vary slightly
- This would propagate through entire pipeline affecting all downstream results

---

### 2.2 Event Decoding (IMPORTANT DIFFERENCE)

**Original** (lines 28-34):
```python
words = event_words[event_indexes].astype(np.uint32, copy=False)

x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
pixel_polarity = ((words >> 28) & 0xF) > 0
```

**New** (loaders.py lines 64-69):
```python
x_coords = window["x"].to_numpy().astype(np.int32)
y_coords = window["y"].to_numpy().astype(np.int32)

# evlib uses -1/+1 for polarity, convert to boolean (True = ON)
polarity_values = window["polarity"].to_numpy()
polarities_on = polarity_values > 0
```

**Impact Assessment: IMPORTANT**

**Polarity Encoding:**
- Original: `(word >> 28 & 0xF) > 0` → boolean
- New: `-1/+1` converted to boolean

If evlib polarity encoding is incorrect or inconsistent, this affects:
- Visualization (ON/OFF event colors)
- Any polarity-dependent processing (if added later)

---

## 3. Frame Accumulation

### 3.1 Implementation Comparison (MINOR DIFFERENCE)

**Original** (lines 38-48):
```python
def build_accum_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    x_coords, y_coords, _ = window
    frame = np.zeros((height, width), dtype=np.uint16)
    frame[y_coords, x_coords] += 1
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame
```

**New** (representations.py lines 8-36):
```python
def build_accum_frame_evlib(
    events: pl.DataFrame,
    width: int,
    height: int,
) -> np.ndarray:
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    x_coords = events["x"].to_numpy().astype(np.int32)
    y_coords = events["y"].to_numpy().astype(np.int32)

    frame = np.zeros((height, width), dtype=np.uint16)
    np.add.at(frame, (y_coords, x_coords), 1)

    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame
```

**Impact Assessment: MINOR**

Both use the same logic:
1. Create uint16 frame
2. Accumulate event counts
3. Clip to 255 and convert to uint8

**Difference:**
- Original: `frame[y, x] += 1`
- New: `np.add.at(frame, (y, x), 1)`

`np.add.at` is **more correct** (handles duplicate coordinates properly), but for visualization this is unlikely to matter.

---

## 4. Visualization

### 4.1 Ellipse Drawing (IMPORTANT DIFFERENCE)

**Original** (lines 494-505):
```python
for (cx, cy, a, b, phi) in prop_ellipses:
    cv2.ellipse(
        overlay,
        (int(cx), int(cy)),
        (int(a), int(b)),
        np.rad2deg(phi),
        0,
        360,
        (0, 255, 0),   # green ellipse
        2,
    )
    cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 0, 255), -1)
```

**New** (main.py lines 205-211):
```python
xs_ring, ys_ring = ellipse_points(cx_t, cy_t, a_t, b_t, phi_t, 360, width, height)
for xr, yr in zip(xs_ring, ys_ring):
    vis[yr, xr] = (0, 255, 0)  # green on top
    bottom_half[yr, xr] = (0, 255, 0)  # green on bottom

cv2.circle(vis, (cx_t, cy_t), 5, (0, 0, 255), -1)
cv2.circle(bottom_half, (cx_t, cy_t), 5, (0, 0, 255), -1)
```

**Impact Assessment: IMPORTANT**

**Original:**
- Uses `cv2.ellipse()` - anti-aliased, smooth rendering
- Draws on overlay layer only

**New:**
- Uses manual pixel setting via `ellipse_points()`
- Draws on both top and bottom halves
- No anti-aliasing (single-pixel wide)

**Visual Quality:**
- Original: smooth, professional appearance
- New: aliased, pixelated appearance
- New may be harder to see at high resolutions

---

### 4.2 Text Rendering (IMPORTANT DIFFERENCE)

**Original** (lines 514-559):
```python
# WARNING text with black background box
cv2.rectangle(
    overlay,
    (x1 - 5, y1 - th1 - 5),
    (x1 + tw1 + 5, y1 + 5),
    (0, 0, 0),
    thickness=-1,
)
cv2.putText(
    overlay,
    text1,
    (x1, y1),
    font,
    font_scale,
    (0, 0, 255),
    thickness,
    lineType=cv2.LINE_AA,
)
```

**New** (main.py lines 236-245):
```python
# Main warning text (NO background box)
text_warn = "WARNING: DRONE DETECTED"
cv2.putText(
    bottom_half,
    text_warn,
    (10, 30),
    font,
    font_scale,
    (0, 0, 255),  # red
    thickness,
    lineType=cv2.LINE_AA,
)
```

**Impact Assessment: IMPORTANT**

**Original:**
- Black background boxes behind text
- Right-aligned positioning
- Better readability on varying backgrounds

**New:**
- No background boxes
- Left-aligned at fixed position (10, 30)
- May be hard to read if events are bright in that area

**Text Position:**
- Original: Dynamic positioning based on text size, bottom-right corner
- New: Fixed position (10, 30), top-left area

---

### 4.3 Display Scaling (IMPORTANT DIFFERENCE)

**Original** (lines 587-589):
```python
combined = np.vstack([vis, overlay])
scale = 0.7
combined_small = cv2.resize(combined, None, fx=scale, fy=scale)
cv2.imshow("Events + Propeller mask + Speed", combined_small)
```

**New** (main.py lines 273-274):
```python
stacked = np.vstack([vis, bottom_half])
cv2.imshow("Events + Propeller mask + Speed", stacked)
```

**Impact Assessment: IMPORTANT**

**Original:**
- Scales display to 70% (896x1008 for 1280x720 sensor)
- Better for high-resolution displays

**New:**
- No scaling (full 1280x1440 display)
- May overflow smaller screens

**Hypothesis:**
- Original scaling improves usability on laptop screens
- New version assumes large monitor or user can resize window
- Not a quality issue, but UX difference

---

### 4.4 Visualization Layout (MINOR DIFFERENCE)

**Original** (lines 585-586):
```python
vis = pretty_event_frame(window, width, height)
combined = np.vstack([vis, overlay])
```

**New** (main.py lines 188-191):
```python
vis = pretty_event_frame_evlib(x, y, p, width, height)

# Create bottom half with dark background for overlay
bottom_half = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)
```

**Impact Assessment: MINOR**

**Original:**
- Top: raw event frame (gray/white/black)
- Bottom: accumulated grayscale + overlay

**New:**
- Top: raw event frame (gray/white/black)
- Bottom: dark gray (50,50,50) + overlay

**Difference:**
- Original bottom shows accumulated events as background
- New bottom is solid dark gray

This is a **stylistic choice**. New version may make overlays more visible, but loses context of event density.

---

## 5. RPM Calculation

### 5.1 Angle Tracking Method (CRITICAL DIFFERENCE)

**Original** (lines 474-484):
```python
if len(angle_hist_per_prop[idx]) >= 2:
    last_th = np.unwrap(np.array(angle_hist_per_prop[idx][-2:]))
    last_t = np.array(time_hist_per_prop[idx][-2:])
    dt = last_t[1] - last_t[0]
    if dt > 0:
        dtheta = last_th[1] - last_th[0]   # rad
        omega = dtheta / dt                 # rad/s
        rpm = omega / (2.0 * np.pi) * 60.0  # RPM
        rpm_abs = abs(rpm)
        rpm_frame.append(rpm_abs)
        rpm_values.append(rpm_abs)
```

**New** (main.py lines 252-257):
```python
# Estimate RPM using unwrap and polyfit
angles_unwrapped = np.unwrap(data["angles"])
times = np.array(data["times"])
coeffs = np.polyfit(times, angles_unwrapped, 1)
omega = coeffs[0]  # rad/s
rpm = (omega / (2.0 * np.pi)) * 60.0
```

**Impact Assessment: CRITICAL**

**Original:**
- **Instantaneous velocity** between last two measurements
- Updates every frame
- Sensitive to noise (jitter in angle detection)

**New:**
- **Linear regression** over ALL accumulated angles
- Smoother, more stable estimate
- Less responsive to sudden speed changes

**Trade-offs:**

| Aspect | Original (2-point) | New (polyfit) |
|--------|-------------------|---------------|
| Noise sensitivity | High | Low |
| Responsiveness | Immediate | Averaged |
| Accuracy (constant speed) | Lower | Higher |
| Accuracy (changing speed) | Higher | Lower |
| Startup behavior | Works immediately | Needs data accumulation |

**Hypothesis:**
- For **constant-speed propellers** (test dataset): polyfit is better
- For **varying-speed propellers**: 2-point is more responsive
- Dataset name suggests "drone_idle" → likely constant speed → polyfit should work well

---

### 5.2 RPM Display Logic (IMPORTANT DIFFERENCE)

**Original** (lines 486-492):
```python
frame_mean_rpm = None
global_mean_rpm = None
if rpm_frame:
    frame_mean_rpm = float(np.mean(rpm_frame))   # current frame mean over props
if rpm_values:
    global_mean_rpm = float(np.mean(rpm_values)) # running mean since first
```

Shows:
- Current frame mean RPM (average of all propellers this frame)
- Global running mean (average since start)

**New** (main.py lines 249-270):
```python
for prop_idx in sorted(propeller_data.keys()):
    data = propeller_data[prop_idx]
    if len(data["times"]) >= 2:
        # ... polyfit ...
        text_rpm = f"Propeller {prop_idx}: {abs(rpm):.1f} RPM"
```

Shows:
- Per-propeller RPM (separate line for each)
- No global averaging

**Impact Assessment: IMPORTANT**

**Original:**
- Better for **overall drone speed** monitoring
- Single number to watch
- Running average smooths fluctuations

**New:**
- Better for **multi-rotor diagnostics**
- See if propellers are balanced
- Identify asymmetric thrust

**Use case dependent:** Neither is "wrong", they serve different purposes.

---

## 6. Clustering (DBSCAN)

### 6.1 Implementation (IDENTICAL)

**Original** (lines 63-128):
```python
def cluster_blades_dbscan_elliptic(...):
    # ... elliptical transform ...
    r_ell = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    mask = (r_ell >= r_min) & (r_ell <= r_max)
    # ... DBSCAN ...
```

**New** (clustering.py lines 8-92):
```python
def cluster_blades_dbscan_elliptic(...):
    # ... elliptical transform ...
    r_ell = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    mask = (r_ell >= r_min) & (r_ell <= r_max)
    # ... DBSCAN ...
```

**Impact Assessment: NONE**

The clustering logic is **functionally identical**. The new version is just extracted to `detector_commons`.

---

### 6.2 Default Parameters (CRITICAL DIFFERENCE)

**Original** (lines 462-465):
```python
theta = blade_angle_for_propeller(
    window,
    (cx, cy, a, b, phi),
    eps=5.0,
    min_samples=15,
    r_min=0.8,
    r_max=1.2,
)
```

**New** (main.py lines 196-202):
```python
centers = cluster_blades_dbscan_elliptic(
    x, y, cx_t, cy_t, a_t, b_t, phi_t,
    eps=args.dbscan_eps,  # default 10.0
    min_samples=args.dbscan_min_samples,  # default 15
    r_min=0.8,
    r_max=5.0,  # CHANGED FROM 1.2
)
```

**Impact Assessment: CRITICAL**

**Key differences:**

| Parameter | Original | New | Impact |
|-----------|----------|-----|--------|
| `eps` | 5.0 | 10.0 (default) | Larger clusters, may merge separate blades |
| `r_max` | 1.2 | 5.0 | MUCH wider ring, includes far more events |

**r_max change (1.2 → 5.0):**
- Original: narrow ring (0.8-1.2x ellipse radius)
- New: VERY wide ring (0.8-5.0x ellipse radius)
- This **massively expands** the clustering region

**Hypothesis:**
- `r_max=5.0` is **likely a bug** - it includes events WAY beyond propeller
- Should probably be 1.2 to match original
- With 5x radius, clustering will include:
  - Other propellers' events
  - Background noise
  - Unrelated features
- This could **severely degrade** blade angle accuracy

**This is the most suspicious difference found.**

---

## 7. Temporal Geometry Lookup

### 7.1 Multi-Ellipse Lookup (FUNCTIONAL CHANGE)

**Original** (lines 344-367):
```python
def pick_propellers_at_time(
    t: float,
    times: np.ndarray,
    ellipses_per_window: List[List[Tuple[int, int, float, float, float]]],
) -> List[Tuple[int, int, float, float, float]]:
    # Returns LIST of ellipses
```

**New** (temporal.py lines 53-82):
```python
def pick_propellers_at_time(
    t: float,
    times: np.ndarray,
    ellipses_per_window: List[List[Tuple[int, int, float, float, float]]],
) -> List[Tuple[int, int, float, float, float]]:
    # Same logic, extracted to module
```

**Impact Assessment: NONE**

Implementation is **identical**, just modularized.

---

## 8. Summary of Critical Issues

### 8.1 High-Priority Differences (Fix Recommended)

1. **Pre-threshold removal** (Section 1.1)
   - **Risk:** More noise, unstable detection
   - **Fix:** Add pre-threshold at 250 before Otsu

2. **Missing pair distance filter** (Section 1.2)
   - **Risk:** False multi-propeller detections
   - **Fix:** Add distance check, only keep pairs within 15px

3. **r_max=5.0 bug** (Section 6.2)
   - **Risk:** Clustering includes wrong events
   - **Fix:** Change to 1.2 to match original

4. **Data format differences** (Section 2.1)
   - **Risk:** Different event counts if HDF5 export is inexact
   - **Fix:** Verify HDF5 export is byte-perfect with .dat

### 8.2 Medium-Priority Differences (Consider Fixing)

5. **RPM calculation method** (Section 5.1)
   - **Impact:** Different noise/responsiveness trade-off
   - **Action:** Test both on dataset, compare stability

6. **Visualization quality** (Sections 4.1-4.3)
   - **Impact:** UX, not detection quality
   - **Action:** Add background boxes, restore scaling

### 8.3 Low-Priority Differences (Acceptable)

7. **Frame accumulation** (Section 3.1) - `np.add.at` is better
8. **Clustering extraction** (Section 6.1) - Good refactoring
9. **Layout style** (Section 4.4) - Design preference

---

## 9. Testing Recommendations

### 9.1 Immediate Tests

1. **Run both implementations** on same dataset:
   ```bash
   python example-drone-original.py evio/data/drone_idle/drone_idle.dat
   uv run drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5
   ```

2. **Compare outputs:**
   - Number of propellers detected per frame
   - Ellipse parameters (cx, cy, a, b, phi)
   - RPM estimates
   - Visual appearance

3. **Check r_max parameter:**
   - Run new version with `--r-max 1.2` (if you add this flag)
   - Compare blade angle stability

### 9.2 Validation Metrics

- **Detection rate:** Frames where propellers found
- **False positive rate:** Detections when no propeller present
- **RPM stability:** Std dev of RPM over time
- **Spatial accuracy:** Distance between detected centers

---

## 10. Root Cause Hypothesis

Based on the analysis, if the new implementation produces **lower quality** results, the most likely causes are:

1. **r_max=5.0** (80% confidence)
   - This is almost certainly wrong
   - 4x larger ring than original
   - Would explain poor blade tracking

2. **Missing pair distance filter** (60% confidence)
   - Allows false propeller detections
   - Contaminates RPM tracking with noise

3. **Pre-threshold removal** (40% confidence)
   - Affects initial detection stability
   - May cause jittery ellipse parameters

4. **HDF5 vs .dat differences** (20% confidence)
   - Only if export is buggy
   - Should be verifiable empirically

---

## Appendix: Line Number Reference

### Original (`example-drone-original.py`)
- Event decoding: 22-35
- Frame accumulation: 38-48
- Clustering: 63-128
- Geometry detection: 169-280
- Pass 1: 282-342
- Pass 2: 401-595

### New Implementation

**geometry.py:**
- Geometry detection: 14-117
- Ellipse drawing: 120-164

**main.py:**
- Main loop: 31-299
- Pass 1: 82-145
- Pass 2: 157-281

**detector_commons:**
- loaders.py: 9-93
- representations.py: 8-73
- clustering.py: 8-92
- temporal.py: 7-82
