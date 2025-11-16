# ctk-opencv (CustomTkinter + OpenCV bridge)

**What:** A helper library (`ctk-opencv`) that bridges OpenCV video frames into CustomTkinter widgets, enabling modern-looking Tk GUIs for CV apps.

**Pros**
- Lightweight GUI: leverages CustomTkinter for a more modern Tk look without heavy Qt deps.
- Easy embedding of OpenCV frames into Tk widgets; suitable for simple UIs with buttons/menus.
- Minimal overhead if you want a “desktop app” feel without moving to Qt/ImGui.

**Cons**
- Tk/CustomTkinter isn’t as performant as Qt/GL for high-FPS or multi-pane dashboards.
- Less flexible for advanced overlays/plots than PyQtGraph/VisPy or DearPyGui.
- Another GUI stack to manage; might overlap with existing cv2 flows.

**Fit with our architecture**
- Could be used for a basic menu/launcher (dataset selection) with embedded OpenCV frames.
- Not ideal for high-rate rendering or shader/particle effects.
- If adopted, keep it optional to avoid adding Tk deps to the core pipeline.

**Typical use**
- Build a window with CustomTkinter, embed a canvas/image widget, and update it with OpenCV frames converted to PhotoImage.
