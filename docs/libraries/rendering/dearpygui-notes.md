# DearPyGui – Additional Notes

**Docs:** https://dearpygui.readthedocs.io/en/latest/index.html

**Highlights**
- Immediate-mode GUI (ImGui style) with docking, panels, textures, and plots.
- Good for building custom tool panels and hotkey-driven dashboards without Qt.
- Supports image/texture rendering from NumPy buffers and embedding small plots.

**Pros**
- Quick to wire up checkboxes, buttons, hotkeys, and multi-pane layouts.
- Cross-platform, pure Python, fast enough for typical CV frame rates.
- Docking makes it easy to arrange multiple views (raw, overlay, plots).

**Cons**
- Styling is ImGui-like (not native); less specialized for large point clouds than VisPy/PyQtGraph.
- Still CPU-bound for heavy image copies; not a replacement for a full GL shader pipeline.

**Fit for us**
- Good candidate for a richer dashboard if we move beyond cv2 (see `docs/plans/python-rendering-pipeline.md`).
- Keep as an optional UI path; default MVP remains cv2-only.

**Getting started**
- Create a viewport, define a texture registry, upload frames as RGBA textures, and draw them in an image widget; add plots for RPM/confidence, and hotkeys for overlay toggles.
