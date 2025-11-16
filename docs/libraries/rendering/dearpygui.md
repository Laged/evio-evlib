# DearPyGui

**What:** Immediate-mode GUI toolkit (ImGui-inspired) with textures, plots, docking, and straightforward hotkeys.

**Pros**
- Easy to build tool panels, checkboxes, hotkeys, and multi-pane layouts.
- Supports image/texture drawing and simple overlay primitives.
- Docking and split views for raw/overlay/plot panes without heavy boilerplate.

**Cons**
- Not as GPU-tuned for large point clouds as VisPy/PyQtGraph; still performant for typical camera frames.
- Different styling paradigm (ImGui) vs native Qt.

**Fit for our architecture**
- Keep evlib/detector headless; push NumPy frames into DearPyGui textures.
- Useful for a live dashboard with toggles for layers (raw, mask, detection, RVT heatmap) and embedded plots (RPM/time).
- Optional dependency; keep cv2 path as fallback.

**Typical use**
- Create a main window with docking; render textures from NumPy data (converted to RGBA).
- Add menus/hotkeys to toggle overlays and debug info.
