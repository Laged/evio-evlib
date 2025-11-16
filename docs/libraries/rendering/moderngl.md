# ModernGL

**What:** Lightweight wrapper over OpenGL in Python, exposing shaders, buffers, and framebuffers with minimal overhead.

**Pros**
- Maximum control for custom shaders, particles, and compositing.
- Suited for advanced effects (e.g., sparks/droplets driven by detector outputs).
- Can be paired with simple window/toolkit (e.g., PyQt, Pyglet) to host the GL context.

**Cons**
- Requires writing GLSL and managing the render loop/windowing yourself.
- More boilerplate than higher-level plotting libs.

**Fit for our architecture**
- Use only for an optional “experimental renderer” path; keep cv2 as default.
- Keep detectors/headless pipeline untouched; pass frame/overlay data + emitter parameters into the GL layer.
- Good when we specifically need shader-driven visuals; otherwise overkill for basic overlays.

**Typical use**
- Create a GL context (via a small window helper), upload textures from NumPy frames, run custom vertex/fragment shaders for overlays/particles, and blit to screen.
