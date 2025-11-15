# Plan – Visual Smoke Test for evlib Migration

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Prerequisite:** Complete `docs/plans/2025-11-15-legacy-evlib-parity-hdf5.md` (data-level parity proving evlib matches the legacy loader on identical event streams).

Once the parity tests pass, we still need to ensure the **user-facing experience** remains unchanged. This plan defines a visual smoke-test workflow that compares the original `evio` demo output (legacy loader + custom `.dat`) with the new evlib-driven path (`_evt3.dat` or HDF5) to catch regressions that raw statistics might miss.

---

## 1. Objectives

1. **Visual parity:** Confirm that the legacy playback scripts and the evlib-based playback show indistinguishable behavior for key datasets (`fan_const_rpm`, `drone_idle`).
2. **Automated capture:** Create repeatable commands that render short sequences to images/videos, enabling side-by-side comparison.
3. **Tolerance logging:** Quantify differences between legacy and evlib renders (e.g., per-frame mean absolute difference) so small numerical jitter doesn’t trigger blockers.
4. **Documentation:** Record the procedure and results so we can prove the migration doesn’t change the operator experience.

---

## 2. Test Matrix

| Dataset | Legacy Input | evlib Input | Demo Script | Expected Output |
| --- | --- | --- | --- | --- |
| fan_const_rpm | `evio/data/fan/fan_const_rpm.dat` | `evio/data/fan/fan_const_rpm_evt3.dat` | `evio/scripts/play_dat.py` (legacy) vs new evlib replay (`workspace/apps/detector-ui` or a lightweight adapter) | Event stream visualization gif/mp4 |
| drone_idle | `evio/data/drone_idle/drone_idle.dat` | `evio/data/drone_idle/drone_idle_evt3.dat` | same as above | gif/mp4 |

---

## 3. Workflow Overview

1. **Legacy capture:** run original `uv run --package evio python evio/scripts/play_dat.py <legacy.dat>` with a `--dump-frames /tmp/legacy/<dataset>` flag (extend script if needed) to emit PNGs or an mp4 clip (using ffmpeg).
2. **evlib capture:** run the new evlib-backed script (`uv run --package evio-core python -m evio_core.tools.play_evlib <evt3.dat>`) with the same `--dump-frames /tmp/evlib/<dataset>`.
3. **Frame comparison:** use a helper script to compute per-frame stats:
   - Mean absolute difference
   - Percentage of pixels differing > threshold
4. **Visual diff artifacts:** combine frames side-by-side (legacy vs evlib vs absolute diff) for manual inspection. Tools: `ffmpeg hstack`, `imagemagick compare`.
5. **Report:** record metrics and attach sample frames to `docs/evlib-integration.md` or `docs/plans/wip-evlib-integration.md`.

---

## 4. Detailed Task Breakdown

### Task A – Extend Legacy Demo for Frame Dumps
- Modify `evio/scripts/play_dat.py` to accept:
  - `--output-dir PATH`: when set, save each rendered frame as PNG (e.g., `frame_000001.png`) using existing OpenCV/Matplotlib rendering.
  - `--max-frames N`: stop after N frames for quicker smoke tests (default: 300 frames ≈ 10 seconds).
- Ensure it works with both legacy loader and evlib-based loader (depending on input path/flag).

### Task B – evlib Playback Tool
- Add a simple script under `workspace/libs/evio-core/tools/play_evlib.py` (or `workspace/apps/detector-ui` CLI) that:
  - Loads events via `evlib.load_events`.
  - Reuses the same rendering pipeline as `play_dat.py`.
  - Supports the same `--output-dir` / `--max-frames`.
- Goal: identical rendering logic so differences stem purely from data ingestion.

### Task C – Frame Comparison Utility
- New helper script `scripts/compare_frames.py`:
  - Inputs: `--legacy-dir`, `--evlib-dir`, optional `--diff-dir`.
  - For each frame index present in both directories:
    - Load grayscale images as NumPy arrays.
    - Compute mean absolute difference (MAD).
    - Count percentage of pixels with difference > 5.
    - Optionally write diff image.
  - Output summary table (per frame + overall average).

### Task D – Automation Command
- Provide a wrapper command in `flake.nix` shellHook instructions (not necessarily alias) describing the workflow:
  1. `run-legacy-demo fan_const_rpm --max-frames 600 --output-dir /tmp/legacy/fan`
  2. `run-evlib-demo fan_const_rpm_evt3 --max-frames 600 --output-dir /tmp/evlib/fan`
  3. `python scripts/compare_frames.py --legacy-dir /tmp/legacy/fan --evlib-dir /tmp/evlib/fan --diff-dir /tmp/diff/fan`

### Task E – Documentation & Sign-off
- Update `docs/evlib-integration.md` with:
  - Purpose of the visual smoke test.
  - Commands to run.
  - Expected metrics (e.g., “Average MAD ≤ 0.5 gray levels”).
  - Instructions to attach sample frames or gifs to PRs when verifying new datasets.
- Add a checklist item in `docs/plans/wip-evlib-integration.md` to mark the test complete.

---

## 5. Acceptance Criteria

1. Running the legacy + evlib playback with `--output-dir` produces synchronized frame sequences for at least two datasets.
2. `compare_frames.py` outputs numeric difference metrics and optional diff images.
3. Documentation clearly states how to execute the smoke test and what “pass” looks like (subjective visual check + objective thresholds).
4. Smoke-test results are archived (e.g., commit attachments or entries in `docs/evlib-integration.md`) so future reviewers can see the visual parity evidence.

When this plan is implemented after the parity/HDF5 validation, we will have both data-level and visual-level assurance that evlib fully replaces the legacy loader without user-visible regressions.
