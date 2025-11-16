# MVP Demos First - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make fan-example-detector.py and drone-example-detector.py immediately runnable via Nix aliases

**Architecture:** Minimal nixification, no refactoring yet - prove demos work visually before building detector-commons

**Tech Stack:** Nix shell aliases, existing scripts, convert-legacy-dat-to-hdf5

---

## Critical Path: Prove → Refactor → Integrate

**Phase 0 (THIS PLAN):** Prove demos work as-is ✅
**Phase 1:** Refactor to detector-commons (after visual validation)
**Phase 2:** Integrate into plugin architecture (after performance proven)

**Rationale:** Can't validate a refactor if we don't know what "correct" looks like!

---

## Task 1: Move Demo Scripts to Nix-Accessible Location

**Files:**
- Move: `fan-example-detector.py` → `evio/scripts/fan_detector_demo.py`
- Move: `drone-example-detector.py` → `evio/scripts/drone_detector_demo.py`

**Step 1: Move fan detector**

Run: `git mv fan-example-detector.py evio/scripts/fan_detector_demo.py`
Expected: File moved

**Step 2: Move drone detector**

Run: `git mv drone-example-detector.py evio/scripts/drone_detector_demo.py`
Expected: File moved

**Step 3: Verify imports still work**

Run: `python evio/scripts/fan_detector_demo.py --help`
Expected: Shows help message

Run: `python evio/scripts/drone_detector_demo.py --help`
Expected: Shows help message

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move detector demos to evio/scripts/"
```

---

## Task 2: Add Nix Shell Aliases

**Files:**
- Modify: `flake.nix`

**Step 1: Read current demo aliases section**

Current location: `flake.nix:318-323` (Demo Aliases section)

**Step 2: Add new aliases after run-demo-fan-ev3**

```nix
# In shellHook, Demo Aliases section:
alias download-datasets='uv run --package downloader download-datasets'
alias run-evlib-tests='uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -v -s'
alias run-demo-fan='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat'
alias run-demo-fan-ev3='uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_legacy.h5'
alias run-mvp-1='uv run --package evio python evio/scripts/mvp_1_density.py evio/data/fan/fan_const_rpm.dat'
alias run-mvp-2='uv run --package evio python evio/scripts/mvp_2_voxel.py evio/data/fan/fan_varying_rpm.dat'
alias run-evlib-raw-demo='uv run --package evlib-examples evlib-raw-demo'
alias run-evlib-raw-player='uv run --package evlib-examples evlib-raw-player'

# NEW: Detector demos (legacy loaders)
alias run-fan-detector='uv run --package evio python evio/scripts/fan_detector_demo.py evio/data/fan/fan_const_rpm.dat'
alias run-drone-detector='uv run --package evio python evio/scripts/drone_detector_demo.py evio/data/drone_idle/drone_idle.dat'
```

**Step 3: Update shellHook banner**

Add new section before existing demo aliases:

```nix
echo ""
echo "🎯 Detector Demos (legacy loaders):"
echo "  run-fan-detector     : Fan RPM estimation with ellipse fitting"
echo "  run-drone-detector   : Drone propeller detection and RPM"
echo ""
```

**Step 4: Test flake syntax**

Run: `nix flake check`
Expected: No errors

**Step 5: Reload environment**

Run: `exit` (if in nix develop)
Run: `nix develop`
Expected: New aliases shown in banner

**Step 6: Test aliases exist**

Run: `type run-fan-detector`
Expected: `run-fan-detector is aliased to ...`

Run: `type run-drone-detector`
Expected: `run-drone-detector is aliased to ...`

**Step 7: Commit**

```bash
git add flake.nix
git commit -m "feat(nix): add detector demo aliases"
```

---

## Task 3: Convert Drone Datasets to HDF5

**Files:**
- None (generates new .h5 files)

**Step 1: Check drone datasets exist**

Run: `ls -lh evio/data/drone_idle/drone_idle.dat`
Expected: Shows file (~736 MB)

Run: `ls -lh evio/data/drone_moving/drone_moving.dat`
Expected: Shows file (~1.5 GB)

**Step 2: Convert drone_idle**

Run: `convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat`
Expected:
```
Converting legacy .dat to HDF5...
Output: evio/data/drone_idle/drone_idle_legacy.h5
✅ Successfully converted 92,048,384 events
```

**Step 3: Convert drone_moving**

Run: `convert-legacy-dat-to-hdf5 evio/data/drone_moving/drone_moving.dat`
Expected:
```
✅ Successfully converted 148,773,889 events
```

**Step 4: Verify HDF5 files**

Run: `ls -lh evio/data/drone_*/drone_*_legacy.h5`
Expected: Shows both HDF5 files

**Step 5: Verify with evlib**

Run: `uv run python -c "import evlib; events = evlib.load_events('evio/data/drone_idle/drone_idle_legacy.h5').collect(); print(f'{len(events):,} events')"`
Expected: `92,048,384 events` (matches conversion output)

**Step 6: Document conversion**

No commit needed (HDF5 files are gitignored, generated on-demand)

---

## Task 4: Test Fan Detector Demo (Legacy)

**Files:**
- None (visual verification)

**Step 1: Run with minimal args**

Run: `run-fan-detector`
Expected: Opens two OpenCV windows showing:
- "Ellipse on events" - Event frame with green ellipse + red center
- "Mask" - Binary mask from thresholding

**Step 2: Verify ellipse tracking**

Observation: Green ellipse should follow fan rotation
Observation: Red center should stay relatively stable
Expected: Press 'q' or ESC to quit after Pass 1

**Step 3: Observe Pass 2 (blade tracking)**

Expected: Shows "DBSCAN clusters (fast window)" with:
- Green ellipse (from Pass 1)
- Blue circles (DBSCAN cluster centers on blades)

**Step 4: Wait for matplotlib plots**

After Pass 2 completes, matplotlib should show:
- Plot 1: "Tracked blade angle vs time" (unwrapped angle, linear trend)
- Plot 2: "Blade angular velocity vs time" (instantaneous ω)

Expected: Terminal prints:
```
Estimated mean angular velocity from blade tracking:
  ω ≈ 31.xxx rad/s
  ≈ 5.xxx rotations/s
  ≈ 3xx.x RPM
```

**Step 5: Test with parameters**

Run: `uv run --package evio python evio/scripts/fan_detector_demo.py evio/data/fan/fan_varying_rpm.dat --max-frames 50`
Expected: Works on different dataset, stops after 50 frames

**Step 6: Document success criteria**

✅ Pass if:
- Ellipse tracks fan rotation visually
- RPM estimate is reasonable (200-400 range for fan)
- No crashes or import errors

---

## Task 5: Test Drone Detector Demo (Legacy)

**Files:**
- None (visual verification)

**Step 1: Run with minimal args**

Run: `run-drone-detector`
Expected: Opens OpenCV window "Events + Propeller mask + Speed" showing:
- Top half: Event frame (white/black on gray)
- Bottom half: Overlay with:
  - Green ellipses (up to 2 propellers)
  - Red centers
  - Text overlays: "WARNING: DRONE DETECTED", "RPM: xxx", "Avg RPM: xxx"

**Step 2: Verify dual propeller detection**

Observation: Should detect 2 propellers if drone has dual rotors
Observation: Each propeller gets separate ellipse
Expected: Green ellipses should be roughly horizontal (orientation filtering)

**Step 3: Observe RPM estimation**

Expected: Text overlays show:
- "WARNING: DRONE DETECTED" (red, top-right)
- "RPM: xxx.x" (green, below warning)
- "Avg RPM: xxx.x" (cyan, below RPM)

**Step 4: Test with drone_moving dataset**

Run: `uv run --package evio python evio/scripts/drone_detector_demo.py evio/data/drone_moving/drone_moving.dat --max-frames 50`
Expected: Works on moving drone dataset

**Step 5: Test parameters**

Run: `uv run --package evio python evio/scripts/drone_detector_demo.py evio/data/drone_idle/drone_idle.dat --window-ms 50 --cluster-window-ms 1.0 --max-frames 30`
Expected: Different window sizes work

**Step 6: Document success criteria**

✅ Pass if:
- Detects 1-2 propellers visually
- RPM estimates are reasonable (drone props typically 3000-10000 RPM)
- Warning overlay appears
- No crashes

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/setup.md`
- Create: `docs/demo-workflow.md`

**Step 1: Update setup.md with demo section**

Add after "Verification" section:

```markdown
## Running Detector Demos

### Prerequisites

```bash
nix develop
unzip-datasets  # Extract datasets (one-time)

# Convert legacy .dat to HDF5 for evlib demos
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat
```

### Fan Detector Demo

**Legacy loader:**
```bash
run-fan-detector
```

Shows:
- Pass 1: Ellipse fitting on rotating fan (30ms windows)
- Pass 2: DBSCAN blade tracking (0.5ms windows)
- Matplotlib plots: Angle tracking and RPM estimation

**Controls:** Press 'q' or ESC to quit

### Drone Detector Demo

**Legacy loader:**
```bash
run-drone-detector
```

Shows:
- Dual propeller detection (up to 2 ellipses)
- Per-propeller RPM estimation
- "WARNING: DRONE DETECTED" overlay

**Controls:** Press 'q' or ESC to quit

### Expected Outputs

**Fan Detector:**
- RPM: ~250-400 (depending on dataset)
- Single ellipse tracking fan rotation

**Drone Detector:**
- RPM: ~3000-10000 per propeller
- 1-2 ellipses (horizontal orientation)
- Warning overlay

---

## Next Steps

See `docs/demo-workflow.md` for evlib migration path.
```

**Step 2: Create demo-workflow.md**

```markdown
# Demo Workflow - Legacy to evlib Migration

**Current State:** Both demos use legacy DatFileSource (works, but slow)

**Migration Path:** Detector-commons → evlib (20-50x speedup)

---

## Phase 0: Visual Validation ✅ (YOU ARE HERE)

**Commands:**
```bash
run-fan-detector       # Visual check: ellipse tracks fan?
run-drone-detector     # Visual check: detects propellers?
```

**Success Criteria:**
- Ellipse/propeller detection works visually
- RPM estimates are reasonable
- No crashes

---

## Phase 1: Detector-Commons Refactor (NEXT)

**Plan:** `docs/plans/2025-11-16-detector-commons-evlib-integration.md`

**Goal:** Extract shared code to detector-commons, migrate to evlib

**Deliverables:**
- `workspace/tools/detector-commons/` - Shared evlib utilities
- `workspace/tools/fan-rpm-demo/` - evlib-powered fan detector
- `workspace/tools/drone-detector-demo/` - evlib-powered drone detector

**Benefits:**
- 55x faster accumulation (evlib representations)
- 50x faster filtering (Polars)
- Shared code maintenance

---

## Phase 2: Plugin Architecture (FUTURE)

**Blocked on:** evio-core implementation (EventSource, DetectorPlugin)

**Goal:** Hot-swappable detectors in interactive UI

**Commands (future):**
```bash
uv run detector-ui evio/data/fan/fan_const_rpm_legacy.h5
# Press '1' → Fan BBox detector
# Press '2' → Fan RPM detector
# Press 'd' → Load drone dataset
# Press '3' → Drone tracker
```

---

## Current Status

✅ Phase 0 complete - demos proven working
⏳ Phase 1 ready - awaiting go-ahead for refactor
⏸️ Phase 2 blocked - needs evio-core
```

**Step 3: Commit documentation**

```bash
git add docs/setup.md docs/demo-workflow.md
git commit -m "docs: add detector demo workflow and setup"
```

---

## Task 7: Create Quick Start Guide

**Files:**
- Create: `DETECTOR_DEMOS.md` (repo root)

**Step 1: Write quick start guide**

```markdown
# Detector Demos Quick Start

**Want to see fan/drone detection in action?** This guide gets you running in 2 minutes.

---

## Prerequisites

```bash
# Install Nix with flakes enabled
# See: https://nixos.org/download.html

# Clone repo
git clone <repo-url> evio-evlib
cd evio-evlib
```

---

## Setup (One Time)

```bash
# Enter Nix environment
nix develop

# Initialize workspace
uv sync

# Extract datasets
unzip-datasets

# Convert to HDF5 for evlib demos
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat
```

---

## Run Demos

### Fan RPM Detector

```bash
run-fan-detector
```

**What you'll see:**
- Pass 1: Green ellipse tracking rotating fan
- Pass 2: Blue circles on blade tips (DBSCAN clusters)
- Matplotlib: Angle tracking and RPM estimate (~300 RPM)

**Controls:** Press 'q' to quit

### Drone Propeller Detector

```bash
run-drone-detector
```

**What you'll see:**
- Green ellipses on 1-2 propellers
- Red "WARNING: DRONE DETECTED" overlay
- RPM estimates per propeller (~3000-10000 RPM)

**Controls:** Press 'q' to quit

---

## Troubleshooting

**"No module named evio"**
→ Run `uv sync` from repo root

**"File not found: evio/data/..."**
→ Run `unzip-datasets` to extract datasets

**"uv: command not found"**
→ Run `nix develop` to enter environment

**Window doesn't show**
→ Check X11/XQuartz on macOS, or DISPLAY on Linux

---

## Next Steps

- **See implementation:** `fan-example-detector.py`, `drone-example-detector.py`
- **evlib migration:** `docs/plans/2025-11-16-detector-commons-evlib-integration.md`
- **Architecture:** `docs/architecture.md`

---

**Questions?** Open an issue or check `docs/setup.md`
```

**Step 2: Commit quick start**

```bash
git add DETECTOR_DEMOS.md
git commit -m "docs: add detector demos quick start guide"
```

---

## Task 8: Final Verification Checklist

**Files:**
- None (checklist only)

**Step 1: Clean environment test**

```bash
# Exit and re-enter nix shell
exit
nix develop

# Verify aliases are loaded
type run-fan-detector
type run-drone-detector
```

Expected: Both aliases defined

**Step 2: End-to-end fan test**

```bash
run-fan-detector
```

Expected:
- ✅ Opens 2 OpenCV windows
- ✅ Ellipse tracks fan
- ✅ Pass 2 shows clusters
- ✅ Matplotlib plots appear
- ✅ Prints RPM estimate

**Step 3: End-to-end drone test**

```bash
run-drone-detector
```

Expected:
- ✅ Opens stacked view window
- ✅ Detects 1-2 propellers
- ✅ Shows warning overlay
- ✅ Prints RPM per propeller

**Step 4: Documentation check**

- ✅ `docs/setup.md` has demo section
- ✅ `docs/demo-workflow.md` explains migration path
- ✅ `DETECTOR_DEMOS.md` quick start works

**Step 5: Git status check**

Run: `git status`
Expected:
- All new files committed
- No uncommitted changes to tracked files
- HDF5 files not tracked (gitignored)

**Step 6: Create summary commit**

```bash
git add -A
git commit -m "feat: complete MVP detector demos setup

- Move detector scripts to evio/scripts/
- Add Nix aliases: run-fan-detector, run-drone-detector
- Convert drone datasets to HDF5
- Add comprehensive documentation
- Verify visual output from both demos

Ready for Phase 1: detector-commons refactor"
```

---

## Success Criteria

✅ **Phase 0 Complete** when:
1. `run-fan-detector` shows working fan detection
2. `run-drone-detector` shows working drone detection
3. Documentation explains demo → refactor → plugin path
4. All changes committed to git

✅ **Ready for Phase 1** (detector-commons refactor):
- Visual "ground truth" established
- Know what "working" looks like
- Can validate refactor against baseline

---

## Next Steps

**After Phase 0 passes:**

Use `docs/plans/2025-11-16-detector-commons-evlib-integration.md` to:
1. Create detector-commons with evlib utilities
2. Migrate fan detector to workspace/tools/fan-rpm-demo
3. Benchmark evlib vs legacy (prove 20-50x speedup)
4. Migrate drone detector to workspace/tools/drone-detector-demo
5. Build toward plugin architecture

**DO NOT proceed with refactor until Phase 0 visual validation passes!**
