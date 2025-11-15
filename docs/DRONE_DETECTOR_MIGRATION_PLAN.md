# Drone Detector Migration Plan - evlib Integration

> **For Claude with fresh context:** This plan migrates the drone detector to evlib using the existing detector-commons utilities. Follow the same pattern as fan-rpm-demo.

**Goal:** Create `drone-detector-demo` package using detector-commons (same as fan-rpm-demo, but with multi-ellipse detection)

**Prerequisites:**
- ✅ detector-commons package complete (loaders, representations, clustering, temporal)
- ✅ fan-rpm-demo working as reference implementation
- ✅ Drone HDF5 files: `evio/data/drone_idle/drone_idle_legacy.h5`, `evio/data/drone_moving/drone_moving_legacy.h5`

---

## Background: Drone vs Fan Differences

### Shared (96% identical → use detector-commons)
- Event loading and windowing → `detector_commons.load_legacy_h5`, `get_window_evlib`
- Accumulation → `detector_commons.build_accum_frame_evlib`
- Visualization → `detector_commons.pretty_event_frame_evlib`
- DBSCAN clustering → `detector_commons.cluster_blades_dbscan_elliptic`
- Temporal lookup → **NEW function needed:** `pick_propellers_at_time` (already in detector-commons!)

### Drone-Specific (in geometry.py)
- **Multi-ellipse detection** (up to 2 propellers) vs single ellipse (fan)
- **Orientation filtering** (horizontal propellers only) vs no filtering (fan)
- **Dual RPM tracking** (one per propeller) vs single RPM (fan)
- **Warning overlay** ("DRONE DETECTED") vs plots (fan)

---

## Implementation Plan

### Task 1: Create drone-detector-demo Package

**Files to create:**
- `workspace/tools/drone-detector-demo/pyproject.toml`
- `workspace/tools/drone-detector-demo/src/drone_detector_demo/__init__.py`
- `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py`
- `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py`

**Step 1: Create directory structure**
```bash
mkdir -p workspace/tools/drone-detector-demo/src/drone_detector_demo
```

**Step 2: Write pyproject.toml**

Copy from fan-rpm-demo and adjust:
```toml
[project]
name = "drone-detector-demo"
version = "0.1.0"
description = "Drone propeller detector using evlib and detector-commons"
requires-python = ">=3.11"
dependencies = [
    "detector-commons",
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
]

[project.scripts]
drone-detector-demo = "drone_detector_demo.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/drone_detector_demo"]

[tool.uv.sources]
detector-commons = { workspace = true }
```

**Step 3: Create __init__.py**
```python
"""Drone propeller detector demo using evlib."""

__version__ = "0.1.0"
```

**Step 4: Run `uv sync`**

Verify package installs.

**Step 5: Commit**
```bash
git add workspace/tools/drone-detector-demo/
git commit -m "feat(drone-detector-demo): create package structure"
```

---

### Task 2: Implement Multi-Ellipse Geometry Module

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py`

**Reference:** Extract from `evio/scripts/drone_detector_demo.py` (lines ~150-300)

**Key Functions:**

#### 1. `propeller_mask_from_frame()`
```python
def propeller_mask_from_frame(
    accum_frame: np.ndarray,
    max_ellipses: int = 2,
) -> List[Tuple[int, int, float, float, float]]:
    """Detect multiple ellipses (propellers) from accumulated frame.

    - Normalize & blur
    - Threshold to create mask
    - Find multiple contours
    - Fit ellipse to each
    - Filter by orientation (horizontal propellers only)
    - Return up to max_ellipses

    Args:
        accum_frame: Grayscale accumulated frame
        max_ellipses: Maximum number of ellipses to detect

    Returns:
        List of (cx, cy, a, b, phi) for each detected propeller
    """
```

**Key difference from fan:**
- Find **all contours** (not just largest)
- Fit ellipse to **each contour**
- **Filter by orientation**: `abs(angle_deg) < 30 or abs(angle_deg - 180) < 30`
- Return **list of ellipses** (up to 2)

#### 2. `ellipse_points()` (same as fan)
Reuse from fan_rpm_demo or copy to drone_detector_demo.

**Step: Commit**
```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/geometry.py
git commit -m "feat(drone-detector-demo): add multi-ellipse geometry module"
```

---

### Task 3: Implement Two-Pass Drone Detector

**File:** `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py`

**Structure:** Nearly identical to fan_rpm_demo/main.py with these changes:

#### Imports
```python
from detector_commons import (
    load_legacy_h5,
    get_window_evlib,
    get_timestamp_range,
    build_accum_frame_evlib,
    pretty_event_frame_evlib,
    cluster_blades_dbscan_elliptic,
    pick_propellers_at_time,  # Use this instead of pick_geom_at_time
)
from .geometry import propeller_mask_from_frame, ellipse_points
```

#### Pass 1: Collect Multi-Ellipse Geometry
```python
# Instead of storing single ellipse per window:
ell_times = []
ell_ellipses_per_window = []  # List[List[Tuple[cx, cy, a, b, phi]]]

# In loop:
ellipses = propeller_mask_from_frame(frame_accum, max_ellipses=2)
ell_times.append(t_s)
ell_ellipses_per_window.append(ellipses)

# Visualize: Draw all ellipses
for (cx, cy, a, b, phi) in ellipses:
    xs_ring, ys_ring = ellipse_points(cx, cy, a, b, phi, 360, width, height)
    for xi, yi in zip(xs_ring, ys_ring):
        vis[yi, xi] = (0, 255, 0)  # green
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
```

#### Pass 2: Per-Propeller Blade Tracking
```python
# Pick closest ellipses from pass 1
ellipses_t = pick_propellers_at_time(
    t_s, ell_times_arr, ell_ellipses_per_window
)

# Track each propeller separately
propeller_data = {}  # {propeller_id: {"times": [...], "angles": [...]}}

for prop_idx, (cx_t, cy_t, a_t, b_t, phi_t) in enumerate(ellipses_t):
    # Cluster blades for this propeller
    centers = cluster_blades_dbscan_elliptic(
        x, y, cx_t, cy_t, a_t, b_t, phi_t,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        r_min=0.8,
        r_max=5.0,
    )

    # Track angle for this propeller
    if centers:
        xc, yc = centers[0]  # Use largest cluster
        theta = np.arctan2(yc - cy_t, xc - cx_t)

        if prop_idx not in propeller_data:
            propeller_data[prop_idx] = {"times": [], "angles": []}
        propeller_data[prop_idx]["times"].append(t_s)
        propeller_data[prop_idx]["angles"].append(theta)
```

#### Visualization: Warning Overlay
```python
# Create stacked view (top: events, bottom: overlay)
top_half = vis[:height//2, :]
bottom_half = np.full((height//2, width, 3), (50, 50, 50), dtype=np.uint8)

# Draw ellipses on bottom half
for (cx, cy, a, b, phi) in ellipses_t:
    # Shift y coordinate to bottom half
    cy_shifted = cy - height//2
    if 0 <= cy_shifted < height//2:
        xs_ring, ys_ring = ellipse_points(cx, cy_shifted, a, b, phi, 360, width, height//2)
        for xr, yr in zip(xs_ring, ys_ring):
            bottom_half[yr, xr] = (0, 255, 0)
        cv2.circle(bottom_half, (cx, cy_shifted), 5, (0, 0, 255), -1)

# Add warning text
cv2.putText(bottom_half, "WARNING: DRONE DETECTED", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Show RPM per propeller
for prop_idx, data in propeller_data.items():
    if len(data["times"]) >= 2:
        # Estimate RPM (same unwrap/polyfit as fan)
        angles_unwrapped = np.unwrap(data["angles"])
        coeffs = np.polyfit(data["times"], angles_unwrapped, 1)
        rpm = (coeffs[0] / (2.0 * np.pi)) * 60.0

        cv2.putText(bottom_half, f"RPM {prop_idx}: {rpm:.1f}",
                    (10, 60 + prop_idx*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Stack
stacked = np.vstack([top_half, bottom_half])
cv2.imshow("Events + Propeller mask + Speed", stacked)
```

#### No Matplotlib (Different from Fan)
Drone detector shows **live overlay** instead of plots at the end.

**Step: Commit**
```bash
git add workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py
git commit -m "feat(drone-detector-demo): implement two-pass detector with overlay"
```

---

### Task 4: Add Nix Alias

**File:** `flake.nix`

**Update shellHook banner:**
```nix
echo "🎯 Detector Demos:"
echo "  run-fan-rpm-demo       : Fan RPM (evlib, detector-commons)"
echo "  run-drone-detector-demo: Drone detection (evlib, detector-commons) - NEW!"
echo "  run-fan-detector       : Fan RPM (legacy loader)"
echo "  run-drone-detector     : Drone detection (legacy loader)"
```

**Add alias:**
```nix
alias run-drone-detector-demo='uv run drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5'
```

**Step: Commit**
```bash
git add flake.nix
git commit -m "feat(nix): add run-drone-detector-demo alias"
```

---

### Task 5: Test on Both Drone Datasets

**Test 1: drone_idle**
```bash
nix develop
uv run drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5 --max-frames 50
```

Expected:
- Detects 1-2 propellers
- Green ellipses (horizontal orientation)
- Warning overlay appears
- RPM estimates per propeller

**Test 2: drone_moving**
```bash
uv run drone-detector-demo evio/data/drone_moving/drone_moving_legacy.h5 --max-frames 50
```

Expected: Same as idle, but moving drone

**Step: Document results**

Record RPM estimates, number of propellers detected, any issues.

---

## Key Differences from Fan-RPM-Demo

| Aspect | Fan | Drone |
|--------|-----|-------|
| **Ellipses** | Single | Multiple (up to 2) |
| **Orientation** | Any | Horizontal only (filter) |
| **Geometry storage** | 5 arrays (cx, cy, a, b, phi) | List of ellipse lists |
| **Temporal lookup** | `pick_geom_at_time` | `pick_propellers_at_time` |
| **Tracking** | Single RPM | Per-propeller RPM dict |
| **Visualization** | Matplotlib plots | Live warning overlay |
| **Output** | Terminal + plots | Stacked view window |

---

## Code to Extract from Legacy Drone Detector

**Source:** `evio/scripts/drone_detector_demo.py`

### Multi-Ellipse Detection (~lines 150-250)
```python
def propeller_mask_from_frame(...):
    # Threshold + morphology
    # findContours (all, not just max)
    # fitEllipse to each
    # Filter by orientation: abs(angle) < 30 or abs(angle-180) < 30
    # Return list of (cx, cy, a, b, phi)
```

### Pass 1 Storage (~lines 400-450)
```python
# Store list of ellipses per window
ell_ellipses_per_window = []
ellipses = propeller_mask_from_frame(frame_accum)
ell_ellipses_per_window.append(ellipses)
```

### Pass 2 Per-Propeller Tracking (~lines 500-600)
```python
# Pick ellipses from pass 1
ellipses_t = pick_propellers_at_time(...)

# Track each propeller
for prop_idx, (cx, cy, a, b, phi) in enumerate(ellipses_t):
    centers = cluster_blades_dbscan_elliptic(...)
    # Track angle, store in propeller_data[prop_idx]
```

### Overlay Visualization (~lines 650-700)
```python
# Stacked view (top/bottom)
# Warning text: "DRONE DETECTED"
# RPM text per propeller
cv2.imshow("Events + Propeller mask + Speed", stacked)
```

---

## Success Criteria

✅ **drone-detector-demo package installs**
✅ **Detects 1-2 propellers on both datasets**
✅ **Horizontal orientation filter works**
✅ **Warning overlay displays correctly**
✅ **Per-propeller RPM estimates are reasonable** (3000-10000 RPM)
✅ **Nix alias works:** `run-drone-detector-demo`
✅ **No crashes or import errors**

---

## Estimated Effort

- **Task 1** (Package structure): 10 minutes
- **Task 2** (Multi-ellipse geometry): 30 minutes (extract + adapt from legacy)
- **Task 3** (Two-pass detector): 45 minutes (adapt from fan-rpm-demo)
- **Task 4** (Nix alias): 5 minutes
- **Task 5** (Testing): 15 minutes

**Total:** ~2 hours

---

## Notes for Fresh Context

1. **detector-commons is complete** - All utilities ready (loaders, representations, clustering, temporal)
2. **fan-rpm-demo is the reference** - Copy structure, adjust for multi-ellipse
3. **pick_propellers_at_time already exists** - In detector-commons/temporal.py
4. **HDF5 files ready** - Both drone datasets converted to legacy.h5
5. **Duration timestamp handling** - Already solved in detector-commons
6. **No matplotlib** - Drone uses live overlay instead of plots

---

## Quick Start for Fresh Claude

```bash
# 1. Review existing work
cat docs/PHASE1_SUMMARY.md  # Understand what's built
cat workspace/tools/fan-rpm-demo/src/fan_rpm_demo/main.py  # Reference implementation

# 2. Check legacy drone detector
cat evio/scripts/drone_detector_demo.py  # Code to extract

# 3. Start implementation
mkdir -p workspace/tools/drone-detector-demo/src/drone_detector_demo
# Follow Task 1-5 above

# 4. Test
nix develop
run-drone-detector-demo
```

---

**Status:** Ready to implement
**Dependencies:** None (all prerequisites met)
**Risk:** Low (proven pattern from fan-rpm-demo)
