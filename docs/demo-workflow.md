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
