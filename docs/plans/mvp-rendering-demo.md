# MVP Rendering Demo – Unified Launcher (cv2-based)

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Scope:** Minimal, reliable demo launcher using the cv2-based pipeline (per `docs/plans/mvp-rendering-pipeline.md` and `docs/plans/mvp-ui-features.md`). Rendering library research (`docs/libraries/rendering/*`) is optional background; stick to cv2/NumPy/matplotlib for this MVP.

---

## Goals
- Single entrypoint (`nix run .#demo`) that:
  1. Shows a menu/grid of available datasets (fan/drone HDF5 exports).
  2. Lets the user navigate with arrows, Enter to play, ESC to return/exit.
  3. Loops playback for the selected dataset via evlib (HDF5 legacy exports).
  4. Supports overlay toggles (mask/detections/debug HUD) via hotkeys.
  5. Allows runtime dataset switching (fan ↔ drone) without restart.

---

## Prereqs
- Data: `unzip-datasets`, then convert legacy `.dat` → HDF5:
  ```bash
  convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
  convert-legacy-dat-to-hdf5 evio/data/drone_idle.dat   # and/or drone_moving
  ```
- Ingestion: evlib (`*_legacy.h5`), no raw→EVT3 path.
- Rendering: cv2 + NumPy (+ matplotlib for optional sparkline); no Qt/GL deps.

---

## Implementation outline
1. **Launcher script** (`evio/scripts/mvp_ui_launcher.py` or similar):
   - Build a simple cv2 grid of tiles (filename text to start; thumbnails later).
   - Dataset map: e.g., `fan_const_rpm_legacy.h5`, `drone_idle_legacy.h5`, `drone_moving_legacy.h5`.
   - Hotkeys: arrows to move selection, Enter to play, ESC to quit.
2. **Playback** (reuse `mvp-rendering-pipeline`):
   - Load selected HDF5 via `evlib.load_events`, loop from `t_min` to `t_max`.
   - Layers: raw (polarity), optional mask/detections/DEBUG overlays from detectors.
   - Hotkeys:
     - `1` mask
     - `2` detections (ellipses/boxes)
     - `3` debug text (RPM/stats)
     - `←/→` previous/next dataset
     - `ESC` back to menu, `q` quit
   - Composition: `np.vstack/hstack` or alpha-blend overlays on raw frame.
3. **HUD polish**
   - Single text panel (fps, rpm/omega if available) with consistent cv2 font.
   - Optional sparkline (matplotlib) blended once per second; otherwise keep overlays lightweight.
4. **Keep it cv2-only**
   - Do not add Qt/GL deps; the rendering research docs are for future upgrades.

---

## Commands (draft)
```bash
nix develop
unzip-datasets
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle.dat

nix run .#demo   # launches the cv2 menu and playback
```

---

## References
- Rendering pipeline (cv2-based): `docs/plans/mvp-rendering-pipeline.md`
- UI behavior: `docs/plans/mvp-ui-features.md`
- Rendering options (for future, not used here): `docs/libraries/rendering/*`
***
