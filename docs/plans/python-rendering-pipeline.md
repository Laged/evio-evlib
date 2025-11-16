# Plan – Python Rendering Pipeline Upgrade (Multi-View, High-Perf)

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Goal:** Explore higher-performance rendering options and UI patterns to visualize multi-layer event data (raw events, masks, detections, RVT outputs) and control overlays via hotkeys. Keep alignment with `docs/architecture.md`: run from repo root, evlib ingestion, plugin detectors.

---

## Quick assessment of current renderer
- Current demos (`play_dat.py`, `play_evlib.py`, `fan-example-detector.py`, `drone-example-detector.py`) use **OpenCV + cv2.imshow** loops for per-frame rendering.
- Strengths: simple, zero extra deps.
- Limits: single-window, basic text overlays, CPU-bound drawing; resizing multi-views is clunky; no built-in hotkey/toggle system beyond `waitKey`.

---

## Candidate rendering stacks (Python)
1. **VisPy / PyQtGraph**  
   - GPU-accelerated 2D/3D; great for high-rate point clouds/images; easy multi-view layouts.  
   - Pros: real-time speed, Qt widget embedding, good for heatmaps/overlays.  
   - Cons: adds Qt dependency (already present via OpenCV’s Qt stack).
2. **DearPyGui**  
   - ImGui-like immediate-mode GUI; supports textures/images/plots; good hotkeys/menus.  
   - Pros: fast, minimal boilerplate, built-in docking/layouts.  
   - Cons: another UI toolkit; styling less “native”.
3. **PyGame/SDL**  
   - Lightweight windowing; suitable for blitting RGB frames; manual overlay handling.  
   - Pros: simple; no Qt.  
   - Cons: less ergonomic for multi-pane dashboards/plots.
4. **Plotting overlays (matplotlib + cv2)**  
   - Suitable for small inline charts (RPM/time) rendered to images and blended onto frames.  
   - Pros: minimal change; already used.  
   - Cons: not high-performance for live multi-graph dashboards.

**Recommendation:** Start with **cv2 + minimal compositing** for MVP, and prototype a **DearPyGui** or **VisPy/PyQtGraph** dashboard for multi-view/high-rate use. Both can coexist; keep the core pipeline (evlib + plugins) separate from the UI layer.

---

## UI/UX features to target
- **Multi-layer toggles (hotkeys):**  
  - Raw events (polarity color), masks (ellipses/propeller blobs), detection overlays (bboxes, ellipses, blade centers), RVT outputs (voxel/timesurface heatmaps), debug text (fps, drops, counts).
  - Hotkeys: `m`=mask on/off, `d`=detections, `h`=heatmap, `r`=raw-only, `g`=graphs pane toggle, `q`=quit.
- **Multi-view layouts:**  
  - Side-by-side or grid: raw frame, overlay frame, heatmap/timesurface, small plot of RPM/angle/time (or RVT confidence).  
  - Simple composition: `np.hstack`/`np.vstack` on same-sized buffers; for performance, pre-allocate and blit.
- **Real-time graphs:**  
  - Option A: render tiny matplotlib plots to images every N frames; alpha-blend onto the main frame.  
  - Option B (faster): DearPyGui/PyQtGraph embedded plots in separate docking panel.
- **Concurrency:**  
  - Reader thread (evlib/Polars windowing) + renderer thread to keep UI responsive.  
  - Use queues for frames/overlays; drop frames if renderer lags (respecting `Pacer` semantics).

---

## Implementation ideas (phased)

**Phase 1 – Minimal upgrades (cv2-based, low risk, robust)**
- Add a `Renderer` helper:  
  - Accepts a dict of overlay layers (`raw`, `mask`, `detections`, `heatmap`, `hud_text`).  
  - Handles hotkeys to toggle layers.  
  - Composes layers into a single frame via `np.vstack/hstack` and shows with `cv2.imshow`.
- Add a small “sparkline” graph (e.g., RPM over last N frames) rendered via matplotlib to a PNG once per second and blended into a corner of the frame.
- Keep everything in the current scripts; no new dependencies beyond matplotlib (already present).

**Phase 2 – Experimental GPU/shader path (optional)**
- Build a standalone dashboard app (e.g., `workspace/apps/event-dashboard`) that:
  - Subscribes to a frame/overlay queue produced by the detector plugin runner.
- **GPU renderer**: ModernGL / VisPy / PyQtGraph (OpenGL) or DearPyGui with texture APIs; optionally explore light-weight game engines (see below) if they align with licensing/packaging.
- Support custom shaders/particles: e.g., emit sparks/droplets seeded from detector outputs (ellipse center + angular velocity), update in a vertex/compute shader, composite over the evlib texture.
- UI: checkboxes/hotkeys to toggle textures (raw, overlay, heatmap, particles), dockable plots (RPM/time, detection confidence, RVT scores).
- This app remains optional; keep cv2 path as fallback.

**Phase 3 – Data selector integration**
- CLI/UI selector to pick dataset: fan/drone (HDF5 exports).  
  - `--dataset fan` → `fan_const_rpm_legacy.h5`  
  - `--dataset drone` → `drone_idle_legacy.h5`  
  - Loop playback, reuse the same renderer interface.
- Shared “player harness” so both detectors (fan/drone plugins) plug into the same rendering pipeline.

---

## Compatibility notes (architecture)
- Keep ingestion via `evlib.load_events` (HDF5 legacy exports) to align with `docs/architecture.md` Layer 1/2.
- Renderers should be UI-only; detectors/modules remain headless and emit overlays/metadata. UI decides how to draw.
- If adding DearPyGui/VisPy, gate them behind extras/optional deps to avoid breaking minimal environments.

---

## Optional: game-engine style renderers
- Lightweight options: **Pyglet** (OpenGL windowing) or **Arcade** (built on Pyglet) for sprite/particle systems; **Panda3D** for more complete engine features. These can host custom shaders/particles with a simpler API than raw ModernGL, but add their own event loops and packaging overhead.
- Fit: only consider if we want richer particle/shader effects and are comfortable adding a heavier UI stack; otherwise prefer ModernGL/VisPy/DearPyGui for tighter control.

---

## Suggested next steps
1. Implement Phase 1 Renderer helper (cv2-based), add hotkeys/toggles, and a small blended sparkline plot.
2. Add dataset selector flag to the demo runner to switch HDF5 sources (fan/drone) and keep looping.
3. Prototype a DearPyGui dashboard separately (optional), keeping evlib/plugin pipeline unchanged.
4. Document commands and hotkeys in `docs/evlib-integration.md` once stabilized.
