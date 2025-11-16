# Rendering Library Options – Quick Scan

**Purpose:** Pick a UI/renderer to visualize evlib-fed event data (raw frames, overlays, RVT outputs) with optional hotkeys and multi-view dashboards. All options keep evlib ingestion unchanged; the renderer is UI-only.

## Shortlist
- **cv2 + NumPy (current)**: simplest, zero extra deps; limited to basic overlays and single-window UI.
- **PyQtGraph / VisPy (OpenGL, Qt-backed)**: GPU-accelerated 2D/3D plots, images, point clouds; good for multi-view dashboards and fast redraws.
- **DearPyGui (ImGui-style)**: immediate-mode GUI with textures, plots, docking, and easy hotkeys; moderate performance.
- **ctk-opencv (CustomTkinter bridge)**: modern Tk UI wrapper to embed OpenCV frames; lightweight but less suited for high-FPS dashboards.
- **ModernGL (raw OpenGL)**: minimal wrapper over GL; maximum control for custom shaders/particles; requires building the UI (window management) separately.
- **Pyglet / Arcade**: simple OpenGL windowing (Pyglet) and 2D game-oriented toolkit (Arcade) for sprites/particles; lightweight engine flavor.
- **Panda3D**: full-featured engine (scenegraph, shaders, windowing); heavier but capable for custom effects.

## Selection Guidance
- For minimal change: keep **cv2**; add layer toggles and simple compositing.
- For fast multi-view dashboards: **PyQtGraph** or **VisPy**.
- For easy GUI layout/hotkeys: **DearPyGui**.
- For custom shaders/particles with tight control: **ModernGL** (or engine variants like Pyglet/Arcade).
- For engine-like needs: consider **Panda3D** only if we accept the heavier stack.
