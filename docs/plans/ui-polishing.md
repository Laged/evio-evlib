# UI Polishing Plan – Performance & Visuals

**Owner:** Codex (handoff for Claude)**
**Date:** 2025-11-16
**Scope:** Small, high-impact tweaks for the MVP launcher/demo: reduce UI hangs, define a cohesive color palette (Sensofusion military gray + Y2K accents), and add thumbnail support with readable text overlays.

---

## Implementation Status

**Phase 1: Visual Polish (Thumbnails + Palette) - ✅ COMPLETE**
- Implemented: 2025-11-16
- See: `docs/plans/2025-11-16-visual-polish-implementation.md`
- Thumbnails: Pre-generation script, 300x150 PNG cache
- Palette: Sensofusion gray + Y2K pink accents applied throughout
- Detector colors: Green/yellow (fan), cyan/orange (drone)
- All manual tests passing

**Phase 2: Performance - ✅ COMPLETE**
- Implemented: 2025-11-16
- See: `PERFORMANCE_OPTIMIZATION_SUMMARY.md`
- Lazy windowing: Schema/metadata caching, optional LazyFrame cache
- HDF5 chunking: 3.7x compression, 2.5x faster loads
- Frame skipping: Optional --skip-frames for 6000+ FPS processing
- All optimizations verified and tested

---

## 1) Performance boosts (evlib/cv2 path)
- **Lazy windowing:** Avoid `.collect()` on full datasets; filter per window and `collect()` only that slice. This reduces startup lag and RAM spikes:
  ```python
  window_events = events.filter((pl.col("t") >= t0) & (pl.col("t") < t1))
  df = window_events.collect()
  ```
- **Warm cache:** On first load, precompute `width/height`, `t_min/t_max`, and keep `schema` cached. Skip repeated `collect_schema()` calls.
- **Frame timer throttling:** Cap UI refresh at ~60 FPS with `waitKey`/sleep; drop frames if processing lags.
- **Preallocate buffers:** Reuse NumPy arrays for compositing; avoid reallocating per frame.
- **Async thumbnails:** If generating thumbnails, do it once per dataset in a background thread/process; store under `evio/data/.cache/thumbnails/`.

---

## 2) Palette (Sensofusion-inspired + Y2K accents)
- **Base:** Military gray/charcoal background `#2b2b2b`; tiles `#3a3a3a`; borders `#4a4a4a`.
- **Text:** Primary white `#f5f5f5` (titles), secondary gray `#c0c0c0` (metadata).
- **Accents (Y2K):** Pink `#ff66cc`, Neon blue `#66ccff`. Use sparingly for selection highlights and badges.
- **Overlay colors:** Keep detections consistent: green text/ellipses `#00ff99`, warning orange `#ff8800`, box/ellipse cyan `#00ffff`.
- **HUD panel:** Semi-transparent black (alpha ~0.6), consistent font sizes (HERSHEY_SIMPLEX 0.8/0.6/0.4).

---

## 3) Thumbnails & overlaid text
- **Thumbnail generation:** For each dataset, render a short window (e.g., first 100–200 ms) to a 300x150 PNG in `evio/data/.cache/thumbnails/<stem>.png`. Use evlib load + simple polarity frame; no detectors needed.
- **Tile rendering:** If a thumbnail exists, draw it as the tile background; then overlay a semi-transparent band at the bottom with the dataset name (white) and metadata (gray). Ensure text remains legible (alpha band ~0.6).
- **Fallback:** If no thumbnail, draw colored rectangle with text only (current MVP).

---

## 4) Controls & layering (quick UX hygiene)
- Keep keybindings consistent across menu/playback (arrows/Enter/ESC/q; 1/2/3 for overlays; h for help).
- Ensure overlays can stack (mask + detections + HUD); avoid overdraw hiding critical visuals.
- Keep the default 70% scaling for the stacked drone view (matches original) but gate with a CLI flag if experimentation is needed.

---

## 5) Suggested sequence for Claude
1. Add evlib windowed filtering in playback to avoid full `.collect()` stalls; cache schema/meta.
2. Implement optional thumbnail generation/caching; update tiles to draw thumbnails with text bands.
3. Apply the palette to menu tiles, borders, selections, and HUD.
4. Keep a CLI flag for scaling (default 0.7 for drone parity), and ensure overlays remain readable at that scale.

These tweaks should make the MVP feel snappier and more polished without new GUI dependencies.***
