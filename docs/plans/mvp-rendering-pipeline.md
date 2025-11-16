# MVP Rendering Pipeline – Simple, Robust Loop

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Goal:** Minimal, reliable improvements to the current cv2-based renderer so we can: (1) loop a selected data file, (2) render existing overlays/text cleanly, (3) toggle between fan/drone datasets at runtime. Keep dependencies minimal; evlib ingestion and detectors remain headless. Target quick validation before deeper UI work.

---

## Objectives
1) **Loop playback of a selected file** (HDF5 exports): fan/drone `_legacy.h5` with evlib loader.  
2) **Cleaner overlays**: consistent fonts, small HUD panel, optional sparkline (RPM) blended into frame.  
3) **Runtime dataset toggle**: switch fan↔drone without restarting; reinitialize loader when selection changes.

---

## Implementation Outline (cv2 + NumPy only)

1. **Player harness**
   - Load HDF5 via `evlib.load_events` (re-use current `play_evlib.py` logic).
   - Add `--dataset` or hotkeys to swap between `fan_const_rpm_legacy.h5` and `drone_idle_legacy.h5` (extendable via CLI path).
   - Loop playback: when `t` reaches end, restart from `t_min`.

2. **Renderer helper**
   - Layers: `raw` (polarity color), `overlay` (detections/masks/ellipses), `hud_text`.
   - Composition: `np.vstack/hstack` or simple overlay alpha-blends.
   - Hotkeys: `d` toggle detections/overlays; `h` toggle HUD; `q` quit.

3. **HUD polish**
   - Single text block (fps, drops, rpm/omega) with consistent font (cv2) on a semi-transparent panel.
   - Optional sparkline: render a tiny matplotlib plot of the last N RPM samples every ~1 s; blend into a corner of the frame.

4. **Dataset selector**
   - Maintain a small map: `{ "fan": path_to_fan_h5, "drone": path_to_drone_h5 }`.
   - Hotkey (e.g., `1`=fan, `2`=drone) reinitializes the loader and resets playback.

---

## Commands (expected)
```bash
nix develop
unzip-datasets
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle.dat

# Run the MVP player (new script or enhanced play_evlib.py)
uv run --package evio python evio/scripts/play_evlib.py --dataset fan   # or --dataset drone
```

---

## Constraints / Scope
- No new GUI deps; cv2/matplotlib only.
- No shader/particle effects; keep it simple and robust.
- Detectors remain headless; renderer consumes their outputs (overlays/text).

---

## Next (after MVP)
- If stable, this becomes the default “preview” path while we prototype the richer dashboard/shader options separately.
