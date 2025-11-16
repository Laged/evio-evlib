# MVP UI Features – Unified Demo Launcher (cv2-based)

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Dependence:** `docs/plans/mvp-rendering-pipeline.md` (cv2-based renderer with layer toggles), rendering stack selection remains cv2/NumPy/matplotlib (no new GUI deps).

---

## Goals (MVP)
1) Unified UI to run our demos (fan/drone) with a single launcher.  
2) Initial menu/grid showing available datasets (one tile per file; minimal text now, image preview later).  
3) Keyboard navigation: arrows to move selection, Enter to open playback, ESC to go back to the menu.  
4) Playback loops raw data (evlib on `_legacy.h5`); fast, minimal rendering.  
5) At runtime, quick switching between datasets without restart.  
6) Overlays controllable via hotkeys (numeric toggles) and composable:
   - `1` – mask overlay
   - `2` – bounding boxes / ellipses (detections)
   - `3` – textual debug (RPM, stats)
   - All overlays can be on/off and layered together.
7) Stretch (optional): shader-based layer on top of others, only if a GL path exists; otherwise skip.

---

## Implementation Outline (stay within MVP rendering pipeline)

- **Menu screen (cv2 window)**:
  - Build a simple grid of tiles (e.g., fixed-size colored rectangles with filename text).
  - Maintain a list of datasets (fan/drone HDF5 outputs) with labels and paths.
  - Arrow keys move highlighting; Enter selects and launches playback; ESC exits.
  - Start minimal: text + border; later: add cached thumbnails.

- **Playback screen (reuse mvp-rendering-pipeline)**:
  - Load selected HDF5 via evlib (`evlib.load_events`), loop from `t_min` to `t_max`.
  - Layers: raw (polarity color), optional mask/detections/DEBUG overlays (as provided by detectors).
  - Hotkeys:
    - `1` toggle mask
    - `2` toggle boxes/ellipses
    - `3` toggle debug text (RPM, drop rates)
    - `←/→` previous/next dataset
    - `ESC` return to menu
    - `q` quit
  - Composition: `np.vstack/hstack` or alpha-blend of overlays on the raw frame.

- **Data selection**:
  - Maintain a dict of dataset entries: `{ "fan": "evio/data/fan/fan_const_rpm_legacy.h5", "drone": "evio/data/drone_idle_legacy.h5", ... }`.
  - Playback loop should reinitialize loader when the selection changes (without closing the app).

- **Reuse existing detectors/overlays**:
  - Use detectors as headless functions that emit overlays (mask polygons, boxes, text).
  - The renderer applies/turns on/off overlay layers based on hotkeys.

- **Stretch (shader layer)**:
  - Only if GL renderer exists (per `full-rendering-pipeline.md`). Otherwise, keep a placeholder toggle that does nothing in MVP.

---

## Commands (expected flow)
```bash
nix develop
unzip-datasets
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle.dat

# Launch the unified UI (new script, cv2-based)
uv run --package evio python evio/scripts/mvp_ui_launcher.py
```

---

## Scope & Constraints
- Stay within cv2/NumPy/matplotlib (no Qt/GL deps for MVP).  
- Keep detectors headless; UI only draws overlays based on their outputs.  
- Minimal architecture changes; do not alter `docs/architecture.md` for MVP.  
- Thumbnails optional; start with text tiles.  
- Looping playback required; smooth dataset switching preferred but can be simple re-init.

---

## Artifacts to produce for Claude
- A small “launcher” script design (cv2 menu + playback mode) that wraps the mvp-rendering-pipeline renderer.  
- Keybindings/hotkeys documented as above.  
- Dataset map with labels/paths.  
- Notes on optional shader layer: only if GL stack added; otherwise a no-op toggle.
