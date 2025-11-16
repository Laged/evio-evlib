# Visual Polish: Thumbnails + Palette Design

**Date:** 2025-11-16
**Status:** Design Complete - Ready for Implementation
**Priority:** High Impact (Visual Polish)

## Overview

Complete visual overhaul of the MVP launcher with thumbnail previews and Sensofusion/Y2K color palette. This enhancement transforms the text-based menu into a visually compelling dataset explorer while maintaining the existing playback functionality.

## Goals

1. **Thumbnail Previews:** Replace text tiles with actual dataset preview images
2. **Y2K Palette:** Apply Sensofusion military gray + pink accent color scheme
3. **Clean Overlays:** Minimal text overlays (dataset name only)
4. **Manual Generation:** Pre-generate thumbnails via Nix command (not on-demand)

## Non-Goals

- Performance optimizations (caching, throttling) - deferred to future work
- Playback speed control - deferred to future work
- Screenshot export - deferred to future work
- Advanced rendering pipeline (OpenGL/shaders) - separate project

---

## Architecture

### Component 1: Thumbnail Generation Script

**Location:** `scripts/generate_thumbnails.py`

**Functionality:**
- Scans `evio/data/` for `*_legacy.h5` files (same discovery as launcher)
- For each dataset, generates 300x150 PNG preview
- Renders first 1 second of events (1,000,000 microseconds)
- Saves to `evio/data/.cache/thumbnails/<dataset_stem>.png`
- Uses evlib lazy loading (no full collect)

**Key Implementation Details:**

```python
def generate_thumbnail(h5_path: Path) -> Path:
    """Generate thumbnail for a single dataset."""

    # Load metadata (lazy - no collect!)
    lazy_events = evlib.load_events(str(h5_path))
    width, height, t_min, t_max = extract_metadata(lazy_events)

    # Render first 1 second window
    window_end = min(t_min + 1_000_000, t_max)  # Cap at dataset end
    events = lazy_events.filter(t >= t_min, t < window_end).collect()

    # Render polarity frame at native resolution
    frame = render_polarity_frame(events, width, height)

    # Resize to 300x150 with letterboxing (preserve aspect ratio)
    thumbnail = resize_with_letterbox(frame, target_w=300, target_h=150)

    # Save to cache
    cache_dir = Path("evio/data/.cache/thumbnails")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{h5_path.stem.replace('_legacy', '')}.png"
    cv2.imwrite(str(output_path), thumbnail)

    return output_path
```

**Error Handling:**
- Skip datasets that fail to load (print warning, continue)
- Auto-create cache directory if missing
- Support `--force` flag to regenerate existing thumbnails
- Show progress: "Generating thumbnails... [2/6] fan_const_rpm.h5"

**Nix Integration:**

```nix
generateThumbnailsScript = pkgs.writeShellScriptBin "generate-thumbnails" ''
  set -euo pipefail
  exec ${pkgs.uv}/bin/uv run --package evio python scripts/generate_thumbnails.py "$@"
'';
```

Add to `buildInputs` and shell aliases.

---

### Component 2: Menu Tile Rendering

**Location:** `evio/scripts/mvp_launcher.py` - `_render_menu()` method

**Rendering Logic:**

For each dataset tile:

1. **Check for cached thumbnail:** `evio/data/.cache/thumbnails/{dataset_stem}.png`

2. **If thumbnail exists:**
   - Load PNG with `cv2.imread()`
   - Draw as tile background (already sized to 300x150)
   - Add semi-transparent black band at bottom (40px height, alpha=0.6)
   - Overlay dataset name in white (centered, bold)
   - Draw pink border if selected (3px thickness)

3. **If thumbnail missing (fallback):**
   - Draw solid gray tile (current behavior)
   - Show dataset name + "No preview" text
   - User should run `generate-thumbnails`

**Text Overlay Rendering:**

```python
# Semi-transparent band at bottom of thumbnail
overlay = tile_region.copy()
band_height = 40
cv2.rectangle(overlay, (x, y + tile_height - band_height),
              (x + tile_width, y + tile_height), (0, 0, 0), -1)
cv2.addWeighted(overlay, 0.6, tile_region, 0.4, 0, tile_region)

# Centered dataset name (white, bold)
name_text = dataset.name
text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
text_x = x + (tile_width - text_size[0]) // 2
text_y = y + tile_height - 15  # 15px from bottom
cv2.putText(tile_region, name_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_PRIMARY, 2, cv2.LINE_AA)
```

---

### Component 3: Color Palette

**Sensofusion Military Gray + Y2K Pink Accents**

All color constants added to `mvp_launcher.py` (and `detector_utils.py` where applicable):

```python
# Menu colors (BGR format for OpenCV)
BG_COLOR = (43, 43, 43)           # #2b2b2b - dark gray background
TILE_COLOR = (58, 58, 58)         # #3a3a3a - tile default (fallback)
TILE_SELECTED = (204, 102, 255)   # #ff66cc - pink Y2K accent
TEXT_PRIMARY = (245, 245, 245)    # #f5f5f5 - white
TEXT_SECONDARY = (192, 192, 192)  # #c0c0c0 - light gray
OVERLAY_BAND = (0, 0, 0)          # Black (used with alpha=0.6)

# Status bar
STATUS_BAR_BG = (30, 30, 30)      # #1e1e1e - darker gray
STATUS_TEXT = (200, 200, 200)     # Light gray

# Playback HUD
HUD_PANEL_BG = (0, 0, 0)          # Black (used with alpha=0.6)
HUD_TEXT = (245, 245, 245)        # White

# Help overlay
HELP_BG = (30, 30, 30)            # Dark gray (used with alpha=0.8)
HELP_TITLE = (245, 245, 245)      # White
HELP_TEXT = (200, 200, 200)       # Light gray

# Detector overlays (detector_utils.py)
DETECTION_SUCCESS = (153, 255, 0)   # #00ff99 - green (BGR) for ellipses/RPM
DETECTION_WARNING = (0, 136, 255)   # #ff8800 - orange (BGR) for warnings
DETECTION_BOX = (255, 255, 0)       # #00ffff - cyan (BGR) for bounding boxes
```

**Application Points:**

- Menu background: `BG_COLOR`
- Fallback tiles: `TILE_COLOR`
- Selected border: `TILE_SELECTED` (pink!)
- Thumbnail text overlay: `TEXT_PRIMARY` on `OVERLAY_BAND`
- Status bar: `STATUS_BAR_BG` + `STATUS_TEXT`
- HUD panel: `HUD_PANEL_BG` (alpha blend) + `HUD_TEXT`
- Help overlay: `HELP_BG` (alpha blend) + `HELP_TITLE` / `HELP_TEXT`
- Fan detector: Green ellipse + RPM text (`DETECTION_SUCCESS`)
- Drone detector: Cyan boxes (`DETECTION_BOX`) + orange warnings (`DETECTION_WARNING`)

---

## Data Flow

### Thumbnail Generation (Manual Step)

1. User runs: `nix develop` → `generate-thumbnails`
2. Script scans `evio/data/` for `*_legacy.h5`
3. For each dataset:
   - Load with evlib (lazy)
   - Filter first 1 second
   - Render polarity frame
   - Resize to 300x150
   - Save PNG to cache
4. Output: `evio/data/.cache/thumbnails/*.png`

### Menu Rendering (Runtime)

1. Launcher discovers datasets (existing logic)
2. For each tile:
   - Check if `thumbnails/{stem}.png` exists
   - If yes: load PNG, draw with text overlay + pink border if selected
   - If no: draw fallback tile with "No preview" text
3. Display grid

---

## File Structure

```
evio-evlib/
├── scripts/
│   └── generate_thumbnails.py          # NEW - thumbnail generator
├── evio/
│   ├── data/
│   │   └── .cache/
│   │       └── thumbnails/             # NEW - cached PNGs
│   │           ├── fan_const_rpm.png
│   │           ├── drone_idle.png
│   │           └── ...
│   └── scripts/
│       ├── mvp_launcher.py             # MODIFY - palette + thumbnail rendering
│       └── detector_utils.py           # MODIFY - detection colors
├── flake.nix                           # MODIFY - add generate-thumbnails command
└── docs/
    └── plans/
        └── 2025-11-16-visual-polish-thumbnails-palette-design.md  # THIS FILE
```

---

## Implementation Sequence

1. **Create thumbnail generation script**
   - Reuse launcher's polarity rendering logic
   - Add letterbox resizing function
   - Implement dataset scanning + caching

2. **Add Nix integration**
   - Add script to `flake.nix`
   - Test generation: `generate-thumbnails`
   - Verify PNGs created in cache

3. **Update launcher menu rendering**
   - Add thumbnail loading logic
   - Implement text overlay rendering
   - Keep fallback for missing thumbnails

4. **Apply palette constants**
   - Define all color constants at top
   - Update menu rendering (background, tiles, borders, text)
   - Update status bar
   - Update HUD panel
   - Update help overlay

5. **Update detector overlays**
   - Apply green/orange/cyan to `detector_utils.py`
   - Test fan detector (green ellipse + RPM)
   - Test drone detector (cyan boxes + orange warning)

6. **Testing**
   - Generate thumbnails for all datasets
   - Verify menu shows thumbnails with names
   - Delete cache, verify fallback works
   - Regenerate with `--force`, verify updates
   - Check playback mode uses new palette
   - Verify detector colors

7. **Documentation**
   - Update `flake.nix` shellHook to mention `generate-thumbnails`
   - Add cache location comment in launcher
   - Mark ui-polishing.md Phase 1 complete

---

## Testing Strategy

### Thumbnail Generation

- Run `generate-thumbnails` with no datasets → expect warning
- Run with datasets → verify PNGs created
- Run again without `--force` → expect skip message
- Run with `--force` → verify regeneration
- Check thumbnail quality (first 1s visible, correct aspect ratio)

### Menu Rendering

- Launch with thumbnails → verify images shown
- Delete cache → verify fallback tiles
- Navigate with arrows → verify pink selection border
- Check text overlays are readable (white on semi-transparent black)

### Palette Application

- Menu: Verify dark gray background, pink selection
- Status bar: Verify darker gray + light text
- Playback HUD: Verify semi-transparent black panel + white text
- Help overlay: Verify dark background + legible text
- Fan detector: Verify green ellipse + RPM text
- Drone detector: Verify cyan boxes + orange warning

### Edge Cases

- Dataset shorter than 1 second → verify caps at t_max
- Very wide/tall datasets → verify letterboxing preserves aspect ratio
- Missing HDF5 file → verify error handling + skip
- Corrupted thumbnail PNG → verify fallback

---

## Success Criteria

- ✅ Thumbnails generated for all `*_legacy.h5` files
- ✅ Menu displays thumbnail previews with dataset names
- ✅ Fallback tiles work when thumbnails missing
- ✅ Pink Y2K accent applied to selection border
- ✅ Sensofusion gray palette applied throughout (menu, HUD, help)
- ✅ Detector overlays use green/orange/cyan colors
- ✅ All text uses anti-aliased rendering
- ✅ `generate-thumbnails` command available in Nix shell

---

## Future Enhancements (Out of Scope)

**Phase 2 - Performance:**
- Schema/metadata caching (avoid repeated collect_schema)
- Preallocated frame buffers
- FPS throttling (~60 FPS cap)
- Async thumbnail generation (background thread)

**Phase 3 - User Features:**
- Playback speed control (↑/↓ arrow keys, 0.1x to 10x)
- Screenshot export (press 's' to save current frame)
- Frame export with overlays
- Video recording (export session to MP4)

**Phase 4 - Advanced:**
- OpenGL/shader-based rendering pipeline
- Real-time particle effects
- Multiple window layouts
- Custom detector plugins

---

## Notes

- Thumbnail window duration (1 second) balances visual richness vs. generation speed
- Letterboxing ensures all thumbnails are exactly 300x150 regardless of dataset aspect ratio
- Pink accent (#ff66cc) chosen for bold Y2K aesthetic (vs. neon blue)
- Manual generation (vs. on-demand) keeps launcher startup fast and predictable
- Fallback tiles ensure launcher works without thumbnails (useful for new datasets)
- BGR color format used throughout (OpenCV convention)

---

## References

- Original plan: `docs/plans/ui-polishing.md`
- Current implementation: `evio/scripts/mvp_launcher.py`
- Detector utilities: `evio/scripts/detector_utils.py`
- Nix infrastructure: `flake.nix`
