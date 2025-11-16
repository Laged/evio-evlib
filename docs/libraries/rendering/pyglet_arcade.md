# Pyglet / Arcade

**What:** Pyglet provides OpenGL windowing/event loop; Arcade is a 2D game framework built on Pyglet for sprites/particles.

**Pros**
- Lightweight compared to full engines; easy to get an OpenGL window and draw textured quads/sprites.
- Arcade includes particle helpers and simple scene management for 2D effects.

**Cons**
- Less geared toward high-rate scientific plotting; more manual work to build multi-pane dashboards.
- Adds another event loop/toolkit alongside existing Qt/cv2 flows.

**Fit for our architecture**
- Optional for experimental particle/shader effects if we prefer a game-style loop over raw ModernGL.
- Not the best choice for multi-pane analytics dashboards; better suited for a single-scene visualization with optional particles.
- Should remain an optional extra to avoid bloating the core toolchain.

**Typical use**
- Create a window, draw textured sprites from NumPy frames, add particle systems driven by detector outputs (e.g., sparks).
