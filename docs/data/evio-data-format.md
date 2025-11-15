# Event File Audit – `data/` directory

**Date:** 2025-11-15  
**Scope:** `data/` and `data/fan/` assets currently sitting outside Git tracking. Goal is to record their provenance, confirm encoding, and outline how to make them usable through `evlib`.

---

## 1. Inventory & Git Status

| Path | Size | Notes | Tracked? |
| --- | --- | --- | --- |
| `data/fan/fan_const_rpm.dat` | 202 MB | Original `.dat` export with custom header | No (`git ls-files data` returns empty) |
| `data/fan/fan_varying_rpm.dat` | 489 MB | Custom `.dat` | No |
| `data/fan/fan_varying_rpm_turning.dat` | 367 MB | Custom `.dat` | No |
| `data/fan_const_rpm.raw` | 119 MB | EVT3 raw example with explicit metadata header | No |
| `data/fan/fan_varying_rpm_turning.raw` | 224 MB | EVT3 raw export | No |

**Recommendation:** keep these datasets out of Git history unless we wire up Git LFS or an external artifact bucket. Right now they are untracked, which is preferable: each file is 100–500 MB and would bloat clones (the repo already documents download steps instead).

---

## 2.3 Dataset Sources in ZIP Archive

**IMPORTANT:** The `junction-sensofusion.zip` archive contains TWO distinct dataset collections:

### Legacy Sensofusion .dat Files
- Source: Original Sensofusion event camera recordings
- Format: Custom binary format with minimal header (% Height, % Version 2, % end)
- Resolution: 1280×720 (hardcoded in loader)
- Examples: `fan/fan_const_rpm.dat`, `fan/fan_varying_rpm.dat`
- Loader: `evio.core.recording.open_dat(path, width=1280, height=720)`
- **NOT compatible with evlib** - requires conversion (see Section 5)

### IDS Camera .raw Files (EVT3 Format)
- Source: NEW IDS Imaging Development Systems captures (Nov 2025)
- Format: Standard Prophesee EVT3 with full ASCII header
- Resolution: Header claims 1280×720, but actual events span 2040×1793
- Duration: 682-717 seconds (much longer than legacy files)
- Examples: `fan/fan_const_rpm.raw`, `fan/fan_varying_rpm.raw`
- Loader: `evlib.load_events(path)` (native support)
- **NOT equivalent to legacy .dat files** - different recording sessions

**These are NOT the same recordings!** Do not compare metrics between legacy .dat and _evt3.dat files derived from .raw - they capture different sessions with different hardware.

---

## 2. Format Inspection

Commands run from repo root (`nix develop` → `uv run` not required for inspection):

```bash
hexdump -C data/fan_const_rpm.raw | head
hexdump -C data/fan/fan_const_rpm.dat | head
strings -a data/fan/fan_const_rpm.dat | head
```

### 2.1 `.raw` Files – Confirmed EVT3

`data/fan_const_rpm.raw` (and `data/fan/fan_varying_rpm_turning.raw`) begin with the standard Prophesee ASCII header:

```
% camera_integrator_name IDS Imaging Development Systems GmbH
% date 2025-11-10 14:26:47
% evt 3.0
% format EVT3;height=720;width=1280
...
```

This matches the EVT3 spec (`evt 3.0`, `format EVT3`). These files **should** be ingestible by `evlib.load_events()` directly once copied into place, assuming we use the `.raw` extension (evlib auto-detects based on header, not just the suffix).

### 2.2 `.dat` Files – Custom Encoding

All three `.dat` files share a much shorter header:

```
% Height 720
% Version 2
% Width 1280
% date 2025-11-10 15:30:23
% end
```

There is no `evt`/`format`/`generation` metadata, and version “2” does not match EVT2/EVT3 canonical identifiers. After the `% end` marker, the byte stream is tightly packed event data that our legacy `evio.core.recording.DatRecording` understands, but evlib does not recognize. This confirms the earlier finding in `evio/docs/dat-format-compatibility.md`: these `.dat` files were produced by the legacy tooling and remain incompatible with evlib.

---

## 3. Making the Data Work with evlib

### 3.1 Preferred Path – Use EVT3 `.raw`

1. Place EVT3 `.raw` files under `data/<scene>/…`.
2. Verify with:
   ```bash
   uv run python - <<'PY'
   import evlib
   df = evlib.load_events("data/fan_const_rpm.raw")
   print(df.head())
   PY
   ```
3. Feed the returned Polars LazyFrame into `evlib.representations.*` or our adapters.

### 3.2 Convert Legacy `.dat` → evlib-friendly Format

Until all recordings are re-exported as EVT3, convert the custom `.dat` payloads into an evlib-supported container (HDF5 or EVT3):

```python
# scripts/convert_dat_to_h5.py  (to be added under workspace/tools)
from pathlib import Path
import h5py
import polars as pl
from evio.core.recording import open_dat

def dat_to_h5(path: Path, width: int, height: int):
    rec = open_dat(path, width=width, height=height)
    events = pl.DataFrame(rec.as_arrays())  # produces columns t,x,y,polarity
    out = path.with_suffix(".h5")
    with h5py.File(out, "w") as f:
        f.create_dataset("events/t", data=events["t"])
        f.create_dataset("events/x", data=events["x"])
        f.create_dataset("events/y", data=events["y"])
        f.create_dataset("events/p", data=events["polarity"])
        f.attrs["width"] = width
        f.attrs["height"] = height
    print(f"Converted {path} → {out}")
```

Usage (after adding the helper script):

```bash
uv run python evio/scripts/convert_dat_to_h5.py \
  --input evio/data/fan/fan_const_rpm.dat \
  --width 1280 --height 720
```

`evlib.load_events("evio/data/fan/fan_const_rpm.h5")` will then work exactly like the EVT3 `.raw` example.

### 3.3 Optional – Contribute Decoder to evlib

If we want evlib to read the custom `.dat` files directly, we would need to upstream a decoder that understands the `% Height / % Version 2` header and event packing scheme. That requires Rust changes plus documentation of the binary layout; for hackathon cadence, conversion is faster.

---

## Experimental: IDS Camera Recordings

The repository contains .raw files from IDS Imaging cameras (Nov 2025) in EVT3 format.
These are **separate recordings** from different hardware, not conversions of legacy .dat files.

**Key differences from legacy data:**
- Resolution: Header claims 1280×720, events span ~2040×1793
- Duration: 682-717 seconds (vs 9-24 seconds for legacy)
- Polarity: All show 0 OFF events (encoding issue)

**Usage:** Experimental evlib testing only. NOT for legacy parity validation.

See `docs/data/datasets.md` for complete manifest.

---

## 4. Recommendations

1. **Do not commit the large `.dat`/`.raw` files** to Git unless we adopt Git LFS or an artifact host; keep instructions in `docs/setup.md`/`NEXT_STEPS.md` for fetching them.
2. **Standardize on EVT3 or HDF5** going forward. When recording new data, export directly as EVT3 (the `.raw` example already proves it works).
3. **Add a conversion script** as noted above and track conversions in `docs/data/` so everyone knows which files are evlib-ready.
4. **Update `docs/setup.md`** once the converter lands, so new contributors immediately run it before testing `evlib`.

This audit should help unblock evlib usage while keeping our data workflow reproducible.

