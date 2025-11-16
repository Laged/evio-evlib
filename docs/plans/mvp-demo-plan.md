# MVP Demo Plan – Fan & Drone (Legacy vs evlib)

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Goal:** Add minimal Nix shell aliases to run the existing fan/drone demo scripts “as-is” so we can visually preview both detectors before doing deeper refactors. Keep legacy + evlib paths discoverable.

---

## 1) Keep It Simple – Reuse Existing Scripts

- **Fan (legacy loader):** `run-demo-fan` already points to `evio/scripts/play_dat.py` with the legacy loader on `.dat`.
- **Fan (evlib on legacy data):** Update `run-demo-fan-ev3` to point at `fan_const_rpm_legacy.h5` (already exported via `convert-legacy-dat-to-hdf5`).
- **Drone (legacy loader):** Add `run-demo-drone` → `python drone-example-detector.py evio/data/drone_idle/drone_idle.dat`.
- **Drone (evlib path):** Add `run-demo-drone-ev3` using the same `drone-example-detector.py` but swapping input to `drone_idle_legacy.h5` once evlib wiring is done. For now, keep it legacy to confirm visual behavior.

---

## 2) Nix shell aliases (flake.nix)

Add to the shellHook under “Demo Aliases”:
```bash
alias run-demo-fan='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat'
alias run-demo-fan-ev3='uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_legacy.h5'
alias run-demo-drone='uv run --package evio python drone-example-detector.py evio/data/drone_idle/drone_idle.dat'
# Optional placeholder for evlib path once wired:
# alias run-demo-drone-ev3='uv run --package evio python drone-example-detector.py evio/data/drone_idle/drone_idle_legacy.h5'
```

Prereqs to communicate:
- `unzip-datasets`
- `convert-legacy-dat-to-hdf5` (for fan/drone HDF5)

---

## 3) Visual Preview Steps

From repo root:
```bash
nix develop
unzip-datasets
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat

run-demo-fan          # legacy loader, fan
run-demo-fan-ev3      # evlib loader on fan_const_rpm_legacy.h5
run-demo-drone        # legacy loader, drone (visual check only)
```
For drone evlib, keep as a follow-up once the script is evlib-enabled.

---

## 4) Scope & Constraints
- **No refactors**: just add aliases to quickly launch the existing demos.
- **Known limitations**: drone script is legacy-only today; evlib path to follow.
- **Don’t touch**: core loaders or data pipelines; this is for visual validation before merging.

---

## 5) Next Steps (post-MVP)
- Wire `drone-example-detector.py` to evlib/HDF5 and update `run-demo-drone-ev3`.
- Integrate a file/data selector for fan vs drone in a shared player harness.
- Fold detectors into plugins once evlib ingestion is stable.
