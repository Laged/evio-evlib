# Full Rendering Pipeline – Advanced / Stretch Goals

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Goal:** Design a richer, GPU-capable dashboard with multi-view layouts, hotkeys, and optional shader/particle effects that map detector outputs (e.g., fan/drone) into visually compelling overlays. Keep evlib ingestion and detectors headless; the renderer is pluggable.

---

## Stretch Feature Set
1) **Multi-view dashboard**: side-by-side or grid (raw, overlay, RVT heatmap/timesurface, plots).  
2) **Hotkeys/UI controls**: toggle layers (raw/mask/detections/RVT), switch datasets (fan/drone), pause/step, loop.  
3) **Real-time plots**: RPM/omega, detection confidence, RVT scores.  
4) **Shader/particle effects**: optional “spark/droplet” emitters driven by detector outputs (ellipse center, angular velocity) with depth cues.  
5) **Data selector**: a UI selector for dataset paths/aliases; live reload without restart.

---

## Candidate Stacks
- **PyQtGraph / VisPy (Qt + OpenGL)**: good for fast image/heatmap and plots; supports multi-view layouts via Qt.  
- **DearPyGui (ImGui)**: easy docking, checkboxes, hotkeys; can display textures/plots; simpler GUI wiring.  
- **ModernGL (raw OpenGL)**: for custom shaders/particles; pair with a lightweight window (Qt/Pyglet).  
- **Optional engines**: Pyglet/Arcade/Panda3D only if we want engine-like scenes; otherwise stick to the above.

Recommendation: Start with a Qt-backed (PyQtGraph/VisPy) or DearPyGui dashboard; add a ModernGL-based particle pass if/when needed.

---

## Architecture Sketch
```
evlib ingestion (HDF5)
  ↓
Detector plugins (fan/drone/RVT) → overlays/metadata (ellipses, bboxes, RPM, scores, masks)
  ↓
Render queue (frames + overlays + plots data)
  ↓
GPU/GUI renderer (PyQtGraph/VisPy or DearPyGui) with optional ModernGL shader pass
  ↓
UI controls/hotkeys + dataset selector
```

---

## Implementation Steps
1) **Renderer shell (dashboard app)**  
   - Create a standalone app (e.g., `workspace/apps/event-dashboard`) that hosts multiple views and a control panel.  
   - Support dataset selection: dropdown or hotkeys to pick fan/drone HDF5; reload evlib loader on change.  

2) **Layer management**  
   - Maintain textures for raw, overlay (detections/masks), optional RVT heatmap/time surface.  
   - Toggle visibility via UI/hotkeys; compose views in panes.  

3) **Plots**  
   - Embed plots (RPM/time, confidence) using native plotting widgets (PyQtGraph/VisPy) or DearPyGui plots.  
   - Update at a throttled rate (e.g., 10 Hz) to avoid UI lag.  

4) **Optional shader/particle pass**  
   - If chosen stack supports it (ModernGL or VisPy custom shader):  
     - Emit particles from detector outputs (e.g., fan ellipse center + angular velocity).  
     - Apply a simple depth/projection to give 3D-ish motion; depth controls size/alpha.  
   - Gate behind a flag; keep cv2 path as fallback.  

5) **Playback control**  
   - Loop, pause, step, speed control; handle frame drops gracefully (drop frames on slow UI).  

6) **Packaging**  
   - Keep GUI deps optional (extras flag) so core users aren’t forced to install Qt/GL stacks.  
   - Document requirements and commands in `docs/evlib-integration.md` once stable.

---

## Acceptance Criteria
- Multi-pane dashboard runs from repo root, loads fan/drone HDF5 via evlib, shows togglable overlays, and renders basic plots.
- Dataset selector works without restart; playback loops/pause works.
- Optional shader/particle mode demonstrated (flagged as experimental) without breaking the default renderer.
- Clear docs and commands; cv2-based MVP remains as a fallback.***
