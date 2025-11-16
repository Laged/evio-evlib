# Panda3D

**What:** Full-featured 3D engine with scenegraph, windowing, shaders, and asset management.

**Pros**
- Supports custom shaders, cameras, and complex scenes.
- Handles windowing, input, and rendering loop out of the box.

**Cons**
- Heavier than plotting/GUI libraries; overkill for simple overlays.
- Packaging footprint and dependency set are larger than our current needs.

**Fit for our architecture**
- Consider only if we need engine-level features (3D scenes, advanced camera control) beyond simple dashboards.
- Likely excessive for event-camera overlays; other lighter options (VisPy/ModernGL/DearPyGui) are preferable.
- If explored, keep it as an optional, isolated path so it doesn’t impact standard workflows.
