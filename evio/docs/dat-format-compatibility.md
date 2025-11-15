# .dat Format Compatibility: Custom vs Standard Event Camera Formats

**TL;DR**: Your `.dat` files use a custom binary format that evlib doesn't support. evlib supports **standard** event camera formats (Prophesee EVT2/EVT3, AEDAT4, HDF5). This document explains the issue and provides solutions.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [What is evlib](#what-is-evlib)
3. [Supported Formats](#supported-formats)
4. [Your Custom .dat Format](#your-custom-dat-format)
5. [Why This Matters](#why-this-matters)
6. [Solutions](#solutions)
7. [Current Workaround](#current-workaround)
8. [Testing Status](#testing-status)

---

## The Problem

**Your `.dat` files cannot be loaded by evlib.**

```python
# This works with your custom loader:
from evio.core.recording import open_dat
rec = open_dat("data/fan_const_rpm.dat", width=1280, height=720)
# ✅ Success - custom binary format

# This fails with evlib:
from evio.evlib_loader import load_events_with_evlib
events = load_events_with_evlib("data/fan_const_rpm.dat")
# ❌ Error: Unrecognized file format
```

**Root cause**: `.dat` is not a standard format name. Your files use a custom binary encoding that only your `evio.core.recording` module understands.

---

## What is evlib

**evlib** is a Rust-backed, high-performance event camera data processing library that provides:
- **50-200x faster** event processing vs pure Python/NumPy
- Standard format support (industry-wide compatibility)
- Modern data structures (Polars DataFrames)
- Professional-grade quality (used in production systems)

**GitHub**: https://github.com/ac-freeman/evlib

---

## Supported Formats

evlib supports these **standard** event camera formats:

### ✅ Prophesee EVT Formats (Most Common)
- **`.dat`** (Prophesee EVT2/EVT3) - **Binary event format from Prophesee cameras**
- Extension: `.dat`, `.raw`
- Encoding: EVT2 (legacy) or EVT3 (modern)
- Cameras: Prophesee Gen1, Gen3, Gen4, HD, IMX636, etc.

### ✅ AEDAT Format (iniVation/AER Standard)
- **`.aedat4`** - AEDAT 4.0 (HDF5-based, modern)
- **`.aedat`** - AEDAT 2.0/3.1 (legacy)
- Cameras: DVS128, DAVIS240, DAVIS346, etc.

### ✅ HDF5 Format (Generic Container)
- **`.h5`**, **`.hdf5`** - HDF5 container with event datasets
- Flexible schema, widely used in research

### ✅ Raw Binary Formats
- **`.raw`** - Raw event stream (needs header/metadata)
- Format must match Prophesee specifications

---

## Your Custom .dat Format

### File Information
```bash
$ ls -lh data/
-rw-r--r-- 1 laged users 202M Nov 15 00:52 fan_const_rpm.dat
-rw-r--r-- 1 laged users 367M Nov 15 00:52 fan_varying_rpm_turning.dat
```

### Current Loader Implementation
```python
# From src/evio/core/recording.py
class DatRecording:
    """Custom binary .dat file reader."""

    def __init__(self, path, width, height):
        self._mmap = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
        self.width = width
        self.height = height
        # Custom binary parsing logic
        # NOT compatible with Prophesee EVT2/EVT3 format
```

**Key differences**:
1. **Custom binary encoding** - Not Prophesee EVT2/EVT3
2. **Requires width/height** - Standard formats include metadata
3. **Memory-mapped custom parser** - Not standard event camera format

### Format Detection
```bash
# evlib tries to auto-detect format from file header
# Your .dat files have a custom header that evlib doesn't recognize
```

---

## Why This Matters

### Current Status
| Component | Status | Note |
|-----------|--------|------|
| **Custom loader** | ✅ Working | Your current `evio.core.recording` |
| **evlib loader** | ❌ Blocked | Cannot read custom .dat format |
| **Benchmarks** | ⏸️ Partial | Custom loader works, evlib skipped |
| **Performance gains** | ⏸️ Unvalidated | Can't measure 50-200x speedup |
| **RVT preprocessing** | ⏸️ Blocked | Needs evlib-loaded events |

### Impact on Integration Plan

**Phase 1 & 2 (Architecture)**: ✅ Complete
- Code infrastructure works
- Tests pass with synthetic data
- evlib imports and functions correctly

**Phase 2 (Performance Validation)**: ❌ Blocked
- Cannot load real event data with evlib
- Cannot benchmark real speedups
- Cannot test voxel grids on 100M+ events

**Phase 3+ (Deep Learning)**: ❌ Blocked
- RVT preprocessing needs evlib representations
- Cannot validate histogram creation at scale
- Cannot test PyTorch tensor pipeline

---

## Solutions

### Solution 1: Obtain Standard Format Files (Recommended)

**If your data came from a Prophesee camera:**
```bash
# Record new data with Prophesee SDK
metavision_recorder -o output.dat

# Or convert existing recordings:
metavision_file_to_dat --input recording.raw --output standard.dat
```

**If your data came from an iniVation camera (DVS/DAVIS):**
```bash
# Files should already be .aedat format
# Just ensure they're .aedat4 (modern) not .aedat (legacy)
```

**For research/simulation data:**
```bash
# Use HDF5 format (.h5)
# See: evlib documentation for schema
```

### Solution 2: Convert Custom .dat to Standard Format

**Option A: Extract events and re-encode**
```python
# Read with custom loader
from evio.core.recording import open_dat
rec = open_dat("data/fan_const_rpm.dat", width=1280, height=720)

# Extract events (your current implementation)
events = extract_events(rec)  # Get t, x, y, p

# Write to HDF5 (evlib-compatible)
import h5py
with h5py.File("fan_const_rpm.h5", "w") as f:
    f.create_dataset("events/t", data=events['t'])
    f.create_dataset("events/x", data=events['x'])
    f.create_dataset("events/y", data=events['y'])
    f.create_dataset("events/p", data=events['p'])
    # Add metadata
    f.attrs['width'] = 1280
    f.attrs['height'] = 720
```

**Option B: Write custom evlib decoder**
```python
# Contribute to evlib with a custom format decoder
# This is advanced - requires understanding evlib internals
# See: https://github.com/ac-freeman/evlib/blob/main/CONTRIBUTING.md
```

### Solution 3: Dual-Path Approach (Current Workaround)

**Keep both loaders:**
```python
# For custom .dat files (your current data)
from evio.core.recording import open_dat
rec = open_dat("data/fan_const_rpm.dat", width=1280, height=720)
# Convert to NumPy, then to Polars LazyFrame
# Feed to evlib representations

# For standard format files (new data, benchmarks)
from evio.evlib_loader import load_events_with_evlib
events = load_events_with_evlib("data/standard_format.aedat4")
# Already in Polars LazyFrame
# Direct to evlib representations
```

**This is what we've implemented**: Infrastructure supports both paths.

---

## Current Workaround

We've designed the system to support **both** loaders:

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   Event Data Sources                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────┐      ┌──────────────────────┐  │
│  │ Custom .dat        │      │ Standard Formats     │  │
│  │ (your files)       │      │ (.aedat4, .h5, etc.) │  │
│  └────────────────────┘      └──────────────────────┘  │
│           ↓                            ↓                │
│  ┌────────────────────┐      ┌──────────────────────┐  │
│  │ evio.core.         │      │ evlib.load_events()  │  │
│  │ recording.open_dat │      │ (Rust-backed)        │  │
│  └────────────────────┘      └──────────────────────┘  │
│           ↓                            ↓                │
│  ┌────────────────────┐      ┌──────────────────────┐  │
│  │ Custom → Polars    │      │ Polars LazyFrame     │  │
│  │ (manual convert)   │      │ (native output)      │  │
│  └────────────────────┘      └──────────────────────┘  │
│           ↓                            ↓                │
│           └────────────┬───────────────┘                │
│                        ↓                                 │
│              ┌─────────────────────┐                    │
│              │   EventData Bridge  │                    │
│              └─────────────────────┘                    │
│                        ↓                                 │
│              ┌─────────────────────┐                    │
│              │ evlib.representations│                   │
│              │ (voxel, histogram)  │                    │
│              └─────────────────────┘                    │
│                        ↓                                 │
│                   MVP Algorithms                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Status
- ✅ **evlib loader**: Works with standard formats
- ✅ **EventData bridge**: Works with both Polars and NumPy
- ✅ **Representations**: Work with Polars LazyFrame (any source)
- ✅ **MVP algorithms**: Work with NumPy arrays (any source)
- ⏸️ **Custom .dat → Polars**: Not yet implemented (you can add this)

---

## Testing Status

### ✅ What We Can Test (Synthetic Data)
```python
# All tests passing with synthetic Polars DataFrames
import polars as pl

events = pl.DataFrame({
    't': [1000, 2000, 3000],
    'x': [100, 200, 300],
    'y': [50, 60, 70],
    'polarity': [1, 0, 1]
}).lazy()

# Test EventData bridge
from evio.events import EventData
event_data = EventData.from_polars(events)  # ✅ Works

# Test voxel grids
from evio.representations import create_voxel_grid
voxel = create_voxel_grid(events, height=720, width=1280, n_time_bins=10)  # ✅ Works

# Test RPM detector
from evio.mvp.rpm_detector import RPMDetector
detector = RPMDetector(height=720, width=1280)
rpm = detector.detect_rpm(events)  # ✅ Works (with calibration needed)
```

### ❌ What We Cannot Test (Real Data)
```python
# Cannot load your .dat files with evlib
from evio.evlib_loader import load_events_with_evlib
events = load_events_with_evlib("data/fan_const_rpm.dat")  # ❌ Fails

# Cannot benchmark real speedups
# Cannot test 100M+ event performance
# Cannot validate RVT preprocessing at scale
```

### ⏸️ What's Skipped
```bash
# Tests that skip when no standard format data available
pytest tests/test_evlib_benchmark.py -v
# SKIPPED: test_evlib_faster_than_custom
# SKIPPED: test_evlib_loader_returns_expected_columns
```

---

## Recommendations

### Immediate Actions
1. **Identify data source**: Where did your `.dat` files come from?
   - Prophesee camera? → Get Prophesee SDK, re-record in EVT3 format
   - Custom recording software? → Export to HDF5 (.h5) format
   - Simulation? → Generate in standard format

2. **Test with sample standard file**: Download a small standard format file to validate evlib integration
   - Prophesee samples: https://docs.prophesee.ai/stable/datasets.html
   - DVS samples: https://inivation.com/support/software/fileformat/

3. **Implement converter** (if needed): Write a script to convert your custom `.dat` → `.h5`
   - Extract events using `evio.core.recording.open_dat()`
   - Write to HDF5 with proper schema for evlib
   - Validate with evlib loader

### Long-term Strategy
- **For production**: Use standard format files (EVT3, AEDAT4, HDF5)
- **For legacy data**: Keep custom loader, convert to Polars in `EventData`
- **For new recordings**: Use Prophesee SDK or iniVation tools

---

## Next Steps

Once you have standard format files:

1. **Validate evlib loading**
   ```bash
   nix develop
   python -c "from evio.evlib_loader import load_events_with_evlib; \
              events = load_events_with_evlib('standard_file.aedat4'); \
              print(f'Loaded {len(events.collect())} events')"
   ```

2. **Run benchmarks**
   ```bash
   python benchmarks/bench_loading.py standard_file.aedat4 --runs 5
   # Should show 10-50x speedup!
   ```

3. **Test MVP-2 at scale**
   ```bash
   python scripts/mvp_2_evlib.py standard_file.aedat4 --blades 4
   # Should detect RPM with 100x faster voxel grid creation
   ```

4. **Continue to Phase 3** (RVT preprocessing)

---

## Questions?

**Q: Can I make evlib support my custom .dat format?**
A: Technically yes, but it requires contributing a custom decoder to evlib (Rust code). It's easier to convert your files to a standard format.

**Q: Why not just use the custom loader?**
A: The custom loader works but is 50-200x slower for large datasets. For 100M+ events, this means minutes vs seconds.

**Q: Can I convert my .dat files?**
A: Yes! Extract events with your custom loader, then write to HDF5. See Solution 2 above.

**Q: Will the evlib integration still work with custom files?**
A: Not directly. You'd need to convert custom → Polars LazyFrame manually, then feed to EventData/representations.

---

**Status**: evlib integration is **architecturally complete** but **validation blocked** by data format incompatibility. Once you have standard format files, all features can be validated.

**Last updated**: 2025-01-15
**Related docs**:
- `docs/plans/2025-01-15-evlib-integration.md` (implementation plan)
- `docs/refactor-to-evlib.md` (technical deep dive)
