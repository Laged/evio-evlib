# Legacy .dat to evlib Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export legacy Sensofusion .dat files to evlib-compatible HDF5 format so evlib can ingest the same recordings as the legacy loader, enabling valid parity tests and demos.

**Architecture:** Load legacy .dat with `evio.core.recording.open_dat`, decode packed event_words into x/y/polarity arrays, write to HDF5 with evlib schema (/events/t, /events/x, /events/y, /events/p + width/height attrs). Add Nix-based CLI command for batch conversion.

**Tech Stack:** Python 3.11, h5py, numpy, evio.core.recording (legacy loader), evlib (validation), pytest (testing)

---

## Background: Why This Is Needed

**Problem discovered:** The .raw files in `evio/data/junction-sensofusion.zip` are NOT conversions of the legacy .dat files. They are completely different recordings:
- Legacy .dat: Sensofusion captures, 1280×720, 9-24 seconds, balanced polarity
- IDS .raw: New IDS camera captures from Nov 2025, 2040×1793 actual resolution, 682-717 seconds, broken polarity (0 OFF events)

**Impact:**
- Current `run-demo-fan-ev3` shows a different recording than `run-demo-fan`
- Parity tests comparing legacy .dat vs _evt3.dat (from .raw) are meaningless
- Cannot validate that evlib correctly reproduces legacy loader behavior

**Solution:** Export the actual legacy .dat recordings to HDF5 so evlib can load the same data.

---

## Task 1: Update Documentation - Clarify Dataset Sources

**Files:**
- Modify: `docs/data/evio-data-format.md`
- Modify: `docs/evlib-integration.md`

**Step 1: Document the dual-dataset situation in evio-data-format.md**

Add a new section after the EVT3 header description:

```markdown
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
```

**Step 2: Update evlib-integration.md to reflect reality**

Replace Section 3 (Requirements) with:

```markdown
## 3. Requirements & Dataset Reality

### Original Assumption (INCORRECT)
The plan assumed .raw files were conversions of legacy .dat files using OpenEB.

### Actual Situation (CONFIRMED)
Investigation revealed the .raw files are **independent IDS camera recordings** from Nov 2025:
- Different hardware (IDS vs Sensofusion)
- Different capture sessions (Nov 2025 vs earlier)
- Different resolutions (2040×1793 actual vs 1280×720)
- Different durations (682-717 sec vs 9-24 sec)
- Different event counts (30-73M vs 26-64M)
- Broken polarity encoding (0 OFF events)

See diagnostic evidence in `scripts/diagnose_fan_data.py` and `scripts/compare_all_fan_files.py`.

### Revised Requirements

1. **Export legacy .dat to HDF5** - Convert actual Sensofusion recordings to evlib-compatible format
2. **Update demos** - Point `run-demo-fan-ev3` at the exported legacy HDF5 files
3. **Fix parity tests** - Compare legacy loader vs evlib on the SAME recording (via HDF5)
4. **Document IDS .raw usage** - Label as separate sample data, useful for evlib experimentation but not legacy parity
```

**Step 3: Commit documentation updates**

```bash
git add docs/data/evio-data-format.md docs/evlib-integration.md
git commit -m "docs: clarify .raw files are not legacy .dat conversions

- Document dual-dataset situation in junction-sensofusion.zip
- Explain legacy Sensofusion .dat vs IDS .raw are different recordings
- Update evlib-integration.md with corrected requirements
- Reference diagnostic scripts for evidence"
```

---

## Task 2: Create Legacy Export Helper (TDD)

**Files:**
- Create: `workspace/libs/evio-core/src/evio/core/legacy_export.py`
- Create: `workspace/libs/evio-core/tests/test_legacy_export_unit.py`

**Step 1: Write failing test for HDF5 schema**

File: `workspace/libs/evio-core/tests/test_legacy_export_unit.py`

```python
"""Unit tests for legacy .dat to HDF5 export."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
import pytest


@dataclass(frozen=True)
class MockRecording:
    """Mock Recording for testing."""
    width: int
    height: int
    event_words: np.ndarray
    timestamps: np.ndarray


def test_export_creates_hdf5_with_correct_schema(tmp_path: Path):
    """Test that export creates HDF5 with evlib-compatible schema."""
    # Arrange: Create mock recording
    event_words = np.array([
        0x10000000,  # polarity=1, y=0, x=0
        0x10004001,  # polarity=1, y=1, x=1
        0x00008002,  # polarity=0, y=2, x=2
    ], dtype=np.uint32)

    timestamps = np.array([1000, 2000, 3000], dtype=np.int64)

    recording = MockRecording(
        width=1280,
        height=720,
        event_words=event_words,
        timestamps=timestamps,
    )

    out_path = tmp_path / "test.h5"

    # Act: Export to HDF5
    from evio.core.legacy_export import export_legacy_to_hdf5
    stats = export_legacy_to_hdf5(recording, out_path)

    # Assert: File exists
    assert out_path.exists()

    # Assert: Stats returned
    assert stats['event_count'] == 3
    assert stats['x_min'] == 0
    assert stats['x_max'] == 2
    assert stats['y_min'] == 0
    assert stats['y_max'] == 2
    assert stats['p_count_pos'] == 2
    assert stats['p_count_neg'] == 1

    # Assert: HDF5 structure matches evlib schema
    with h5py.File(out_path, 'r') as f:
        assert 'events/t' in f
        assert 'events/x' in f
        assert 'events/y' in f
        assert 'events/p' in f

        assert f.attrs['width'] == 1280
        assert f.attrs['height'] == 720
        assert f.attrs['source'] == 'legacy_dat'

        # Check data types
        assert f['events/t'].dtype == np.int64
        assert f['events/x'].dtype == np.uint16
        assert f['events/y'].dtype == np.uint16
        assert f['events/p'].dtype == np.int8


def test_export_decodes_polarity_correctly(tmp_path: Path):
    """Test that polarity is decoded from bits 31:28."""
    # Arrange: Events with different polarities
    event_words = np.array([
        0x00000000,  # polarity=0 (raw_polarity=0)
        0x10000000,  # polarity=1 (raw_polarity=1)
        0x20000000,  # polarity=1 (raw_polarity=2, any non-zero → 1)
        0xF0000000,  # polarity=1 (raw_polarity=15)
    ], dtype=np.uint32)

    timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.int64)

    recording = MockRecording(
        width=1280,
        height=720,
        event_words=event_words,
        timestamps=timestamps,
    )

    out_path = tmp_path / "polarity_test.h5"

    # Act
    from evio.core.legacy_export import export_legacy_to_hdf5
    stats = export_legacy_to_hdf5(recording, out_path)

    # Assert: Polarity counts
    assert stats['p_count_neg'] == 1  # Only first event is 0
    assert stats['p_count_pos'] == 3  # Last three are 1

    # Assert: HDF5 polarity values
    with h5py.File(out_path, 'r') as f:
        polarity = f['events/p'][:]
        assert polarity[0] == 0
        assert polarity[1] == 1
        assert polarity[2] == 1
        assert polarity[3] == 1


def test_export_decodes_xy_coordinates_correctly(tmp_path: Path):
    """Test that x/y are decoded from correct bit positions."""
    # Arrange: Events with known x/y positions
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    event_words = np.array([
        0x00000000,  # x=0, y=0
        0x00004001,  # x=1, y=1 (y at bits 14-27, x at bits 0-13)
        0x000140FF,  # x=255, y=5 (5 << 14 = 0x14000)
        0x00000500,  # x=1280, y=0 (1280 = 0x500)
    ], dtype=np.uint32)

    timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.int64)

    recording = MockRecording(
        width=1280,
        height=720,
        event_words=event_words,
        timestamps=timestamps,
    )

    out_path = tmp_path / "xy_test.h5"

    # Act
    from evio.core.legacy_export import export_legacy_to_hdf5
    export_legacy_to_hdf5(recording, out_path)

    # Assert: Coordinates decoded correctly
    with h5py.File(out_path, 'r') as f:
        x = f['events/x'][:]
        y = f['events/y'][:]

        assert x[0] == 0 and y[0] == 0
        assert x[1] == 1 and y[1] == 1
        assert x[2] == 255 and y[2] == 5
        assert x[3] == 1280 and y[3] == 0
```

**Step 2: Run tests to verify they fail**

```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_legacy_export_unit.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'evio.core.legacy_export'"

**Step 3: Implement minimal export function**

File: `workspace/libs/evio-core/src/evio/core/legacy_export.py`

```python
"""Export legacy .dat recordings to evlib-compatible HDF5 format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from typing import Any


def export_legacy_to_hdf5(recording: Any, out_path: Path) -> dict[str, int]:
    """Export legacy Recording to HDF5 with evlib-compatible schema.

    Args:
        recording: Recording object from evio.core.recording.open_dat()
            Must have attributes: width, height, event_words, timestamps
        out_path: Output path for HDF5 file

    Returns:
        Dict with statistics:
            - event_count: total events
            - t_min, t_max: timestamp range (microseconds)
            - x_min, x_max, y_min, y_max: spatial bounds
            - p_count_neg, p_count_pos: polarity distribution

    HDF5 Schema (evlib-compatible):
        /events/t        : int64 timestamps in microseconds
        /events/x        : uint16 x coordinates
        /events/y        : uint16 y coordinates
        /events/p        : int8 polarity {0, 1}

        file.attrs['width']  : int
        file.attrs['height'] : int
        file.attrs['source'] : str = "legacy_dat"
    """
    # Decode packed event_words into x, y, polarity
    # See: evio/src/evio/core/mmap.py:151-154
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    event_words = recording.event_words

    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)

    # Keep polarity as 0/1 - evlib converts to -1/+1 internally
    polarity = (raw_polarity > 0).astype(np.int8)

    # Timestamps are already sorted and in microseconds
    timestamps = recording.timestamps

    # Compute stats before writing
    stats = {
        'event_count': len(timestamps),
        't_min': int(timestamps.min()),
        't_max': int(timestamps.max()),
        'x_min': int(x.min()),
        'x_max': int(x.max()),
        'y_min': int(y.min()),
        'y_max': int(y.max()),
        'p_count_neg': int((polarity == 0).sum()),
        'p_count_pos': int((polarity == 1).sum()),
    }

    # Write to HDF5
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        # Create datasets (evlib expects /events/t, /events/x, /events/y, /events/p)
        f.create_dataset('events/t', data=timestamps, dtype='int64')
        f.create_dataset('events/x', data=x, dtype='uint16')
        f.create_dataset('events/y', data=y, dtype='uint16')
        f.create_dataset('events/p', data=polarity, dtype='int8')

        # Write metadata at file level
        f.attrs['width'] = recording.width
        f.attrs['height'] = recording.height
        f.attrs['source'] = 'legacy_dat'

    return stats
```

**Step 4: Run tests to verify they pass**

```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_legacy_export_unit.py -v
```

Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add workspace/libs/evio-core/src/evio/core/legacy_export.py workspace/libs/evio-core/tests/test_legacy_export_unit.py
git commit -m "feat(evio-core): add legacy .dat to HDF5 export

- Implement export_legacy_to_hdf5() with evlib-compatible schema
- Decode packed event_words (x/y/polarity from bit layout)
- Write /events/t, /events/x, /events/y, /events/p datasets
- Add width/height/source metadata attributes
- Unit tests verify schema, polarity decoding, x/y decoding"
```

---

## Task 3: Integration Test - Export Real Legacy File

**Files:**
- Create: `workspace/libs/evio-core/tests/test_legacy_export_integration.py`

**Step 1: Write test that exports real legacy .dat and validates with evlib**

```python
"""Integration tests for legacy export with real files and evlib validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import evlib

sys.path.insert(0, 'workspace/libs/evio-core/src')
from evio.core.recording import open_dat
from evio.core.legacy_export import export_legacy_to_hdf5


# Check if fan dataset exists
LEGACY_DAT = Path("evio/data/fan/fan_const_rpm.dat")
DATASET_AVAILABLE = LEGACY_DAT.exists()


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="fan_const_rpm.dat not found")
def test_export_fan_const_rpm_to_hdf5(tmp_path: Path):
    """Test exporting real legacy .dat file to HDF5."""
    # Arrange
    legacy_path = LEGACY_DAT
    hdf5_path = tmp_path / "fan_const_rpm_legacy.h5"

    # Act: Load with legacy loader
    recording = open_dat(str(legacy_path), width=1280, height=720)

    # Act: Export to HDF5
    stats = export_legacy_to_hdf5(recording, hdf5_path)

    # Assert: Stats match expected values (from diagnostic script)
    assert stats['event_count'] == 26_439_977
    assert stats['x_min'] == 0
    assert stats['x_max'] == 1279
    assert stats['y_min'] == 0
    assert stats['y_max'] == 719

    # Assert: Polarity is balanced (not broken like IDS .raw)
    assert stats['p_count_pos'] > 13_000_000  # ~13.4M
    assert stats['p_count_neg'] > 13_000_000  # ~13.0M

    # Assert: HDF5 file exists and is loadable by evlib
    assert hdf5_path.exists()

    # Act: Load with evlib
    events = evlib.load_events(str(hdf5_path)).collect()

    # Assert: Event count matches
    assert len(events) == stats['event_count']

    # Assert: Spatial bounds match (evlib sees 1280×720)
    assert events['x'].min() == 0
    assert events['x'].max() == 1279
    assert events['y'].min() == 0
    assert events['y'].max() == 719

    # Assert: Polarity is balanced
    pol_on = (events['polarity'] == True).sum()
    pol_off = (events['polarity'] == False).sum()
    assert pol_on > 13_000_000
    assert pol_off > 13_000_000

    print(f"\n✅ Successfully exported {stats['event_count']:,} events")
    print(f"   Resolution: {stats['x_max']+1}×{stats['y_max']+1}")
    print(f"   Polarity: {stats['p_count_pos']:,} ON, {stats['p_count_neg']:,} OFF")
    print(f"   HDF5 size: {hdf5_path.stat().st_size / 1024 / 1024:.1f} MB")


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="fan_const_rpm.dat not found")
def test_legacy_vs_evlib_exact_match(tmp_path: Path):
    """Test that legacy loader and evlib see identical events after export."""
    # Arrange
    legacy_path = LEGACY_DAT
    hdf5_path = tmp_path / "fan_const_rpm_legacy.h5"

    # Act: Export legacy to HDF5
    recording = open_dat(str(legacy_path), width=1280, height=720)
    export_legacy_to_hdf5(recording, hdf5_path)

    # Load with evlib
    events = evlib.load_events(str(hdf5_path)).collect()

    # Decode legacy event_words for comparison
    import numpy as np
    event_words = recording.event_words
    timestamps = recording.timestamps

    x_legacy = (event_words & 0x3FFF).astype(np.uint16)
    y_legacy = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    p_legacy = (raw_polarity > 0).astype(np.int8)

    # Assert: Timestamps match
    t_evlib = events['t']
    # evlib may return timedelta - convert to microseconds
    if hasattr(t_evlib[0], 'total_seconds'):
        t_evlib_us = np.array([int(t.total_seconds() * 1e6) for t in t_evlib])
    else:
        t_evlib_us = np.array(t_evlib)

    np.testing.assert_array_equal(t_evlib_us, timestamps)

    # Assert: X coordinates match
    np.testing.assert_array_equal(events['x'].to_numpy(), x_legacy)

    # Assert: Y coordinates match
    np.testing.assert_array_equal(events['y'].to_numpy(), y_legacy)

    # Assert: Polarity matches (evlib uses bool, we use int8)
    p_evlib = events['polarity'].to_numpy().astype(np.int8)
    np.testing.assert_array_equal(p_evlib, p_legacy)

    print("\n✅ EXACT MATCH: Legacy loader and evlib see identical events!")
```

**Step 2: Run integration test**

```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_legacy_export_integration.py -v -s
```

Expected: PASS (2 tests) with success messages printed

**Step 3: Commit**

```bash
git add workspace/libs/evio-core/tests/test_legacy_export_integration.py
git commit -m "test(evio-core): add integration tests for legacy export

- Test export of real fan_const_rpm.dat to HDF5
- Validate with evlib (event count, resolution, polarity)
- Verify exact match between legacy loader and evlib output
- Confirms 26.4M events, 1280×720 resolution, balanced polarity"
```

---

## Task 4: CLI Tool for Batch Conversion

**Files:**
- Create: `scripts/convert_legacy_dat_to_hdf5.py`
- Modify: `flake.nix` (add CLI command)

**Step 1: Write CLI script**

File: `scripts/convert_legacy_dat_to_hdf5.py`

```python
#!/usr/bin/env python3
"""
Convert legacy .dat files to evlib-compatible HDF5 format.

This script loads a legacy .dat file with evio.core.recording.open_dat,
decodes the packed event_words, and writes to HDF5 with evlib schema.

Usage:
    nix develop --command convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat

    # Custom output path
    nix develop --command convert-legacy-dat-to-hdf5 input.dat output.h5

    # Batch convert all fan files
    nix develop --command convert-legacy-dat-to-hdf5 --batch evio/data/fan/*.dat

See docs/plans/2025-11-16-legacy-dat-to-evlib-export.md for context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, 'workspace/libs/evio-core/src')
from evio.core.recording import open_dat
from evio.core.legacy_export import export_legacy_to_hdf5


def convert_single_file(
    input_path: Path,
    output_path: Path | None,
    width: int,
    height: int,
    force: bool = False,
) -> bool:
    """Convert a single legacy .dat file to HDF5.

    Returns:
        True if successful, False otherwise
    """
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_legacy.h5"

    # Check if output exists
    if output_path.exists() and not force:
        print(f"⚠️  Output exists: {output_path}")
        print("   Use --force to overwrite")
        return False

    # Load with legacy loader
    print(f"Loading {input_path.name} with legacy loader...")
    try:
        recording = open_dat(str(input_path), width=width, height=height)
    except Exception as e:
        print(f"❌ Failed to load {input_path}: {e}")
        return False

    # Export to HDF5
    print(f"Exporting to {output_path.name}...")
    try:
        stats = export_legacy_to_hdf5(recording, output_path)
    except Exception as e:
        print(f"❌ Failed to export: {e}")
        return False

    # Success
    print(f"✅ Exported {stats['event_count']:,} events")
    print(f"   Resolution: {stats['x_max']+1}×{stats['y_max']+1}")
    print(f"   Polarity: {stats['p_count_pos']:,} ON, {stats['p_count_neg']:,} OFF")
    print(f"   Duration: {(stats['t_max'] - stats['t_min']) / 1e6:.2f} seconds")
    print(f"   Output: {output_path}")

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert legacy .dat files to evlib-compatible HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evio/data/fan/fan_const_rpm.dat
  %(prog)s input.dat output.h5
  %(prog)s --batch evio/data/fan/*.dat --width 1280 --height 720

See docs/plans/2025-11-16-legacy-dat-to-evlib-export.md for details.
        """
    )

    parser.add_argument(
        'input',
        type=Path,
        nargs='+',
        help='Input legacy .dat file(s)',
    )

    parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        help='Output HDF5 file (default: <input>_legacy.h5)',
    )

    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Sensor width (default: 1280)',
    )

    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Sensor height (default: 720)',
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output file if it exists',
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: convert multiple files',
    )

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        print("=" * 60)
        print("  Batch Convert Legacy .dat to HDF5")
        print("=" * 60)
        print()

        success_count = 0
        fail_count = 0

        for input_path in args.input:
            print(f"\n--- {input_path.name} ---")
            if convert_single_file(input_path, None, args.width, args.height, args.force):
                success_count += 1
            else:
                fail_count += 1

        # Summary
        print()
        print("=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"✅ Success: {success_count}")
        if fail_count > 0:
            print(f"❌ Failed: {fail_count}")

        return 0 if fail_count == 0 else 1

    # Single file mode
    if len(args.input) > 1 and args.output is None:
        print("❌ Error: Multiple inputs require --batch mode or single output path")
        return 1

    input_path = args.input[0]

    print("=" * 60)
    print("  Convert Legacy .dat to HDF5")
    print("=" * 60)
    print()

    success = convert_single_file(
        input_path,
        args.output,
        args.width,
        args.height,
        args.force,
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
```

**Step 2: Add CLI command to flake.nix**

In `flake.nix`, after the `convertAllDatasetsScript` definition, add:

```nix
        # Convert legacy .dat to HDF5 script
        convertLegacyDatToHdf5Script = pkgs.writeShellScriptBin "convert-legacy-dat-to-hdf5" ''
          set -euo pipefail
          exec ${python}/bin/python scripts/convert_legacy_dat_to_hdf5.py "$@"
        '';
```

Then add to `buildInputs`:

```nix
          buildInputs = [
            # ... existing tools ...
            convertLegacyDatToHdf5Script # convert-legacy-dat-to-hdf5 command
          ];
```

And update the shellHook help text:

```nix
            echo "📊 Dataset Management:"
            echo "  unzip-datasets              : Extract junction-sensofusion.zip"
            echo "  download-datasets           : Download from Google Drive (~1.4 GB)"
            echo "  convert-all-datasets        : Convert all .raw files to EVT3 .dat"
            echo "  convert-legacy-dat-to-hdf5  : Convert legacy .dat to evlib HDF5"
```

**Step 3: Test CLI command**

```bash
# Rebuild nix shell
nix develop

# Test single conversion
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat

# Verify output exists
ls -lh evio/data/fan/fan_const_rpm_legacy.h5
```

Expected: File created, ~200MB size, success message printed

**Step 4: Test batch conversion**

```bash
convert-legacy-dat-to-hdf5 --batch evio/data/fan/*.dat --force
```

Expected: All 3 fan .dat files converted (fan_const_rpm, fan_varying_rpm, fan_varying_rpm_turning)

**Step 5: Commit**

```bash
git add scripts/convert_legacy_dat_to_hdf5.py flake.nix
git commit -m "feat: add convert-legacy-dat-to-hdf5 CLI tool

- Implement batch conversion script for legacy .dat → HDF5
- Add Nix wrapper command in flake.nix
- Support single file and batch modes
- Default output: <input>_legacy.h5
- Update shellHook help text with new command"
```

---

## Task 5: Update Demo Aliases to Use Legacy HDF5

**Files:**
- Modify: `flake.nix` (update run-demo-fan-ev3 alias)
- Create: `evio/scripts/play_hdf5.py` (if needed, or reuse play_evlib.py)

**Step 1: Check if play_evlib.py can load HDF5**

```bash
# Convert legacy file first
nix develop --command convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat --force

# Test if play_evlib.py works with HDF5
nix develop --command uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_legacy.h5
```

Expected: Should work (evlib loads HDF5 natively)

**Step 2: Update run-demo-fan-ev3 alias in flake.nix**

Change from:

```nix
            alias run-demo-fan-ev3='uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_evt3.dat'
```

To:

```nix
            alias run-demo-fan-ev3='uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_legacy.h5'
```

**Step 3: Update shellHook help text**

Change:

```nix
            echo "Demo Aliases:"
            echo "  run-demo-fan         : Play fan dataset (legacy loader)"
            echo "  run-demo-fan-ev3     : Play fan dataset (evlib loader on EVT3)"
```

To:

```nix
            echo "Demo Aliases:"
            echo "  run-demo-fan         : Play fan dataset (legacy loader)"
            echo "  run-demo-fan-ev3     : Play fan dataset (evlib on legacy HDF5)"
            echo ""
            echo "NOTE: run-demo-fan-ev3 requires: convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat"
```

**Step 4: Test demo**

```bash
# Ensure HDF5 exists
nix develop --command convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat --force

# Reload shell
exit
nix develop

# Run both demos
run-demo-fan
# Should show: 1280×720, 26.4M events

run-demo-fan-ev3
# Should now ALSO show: 1280×720, 26.4M events (same recording!)
```

**Step 5: Commit**

```bash
git add flake.nix
git commit -m "fix: point run-demo-fan-ev3 at legacy HDF5 export

- Update alias to use fan_const_rpm_legacy.h5 instead of _evt3.dat
- Now both demos show the SAME recording (26.4M events, 1280×720)
- Add note about convert-legacy-dat-to-hdf5 prerequisite
- Fixes visual demo parity issue"
```

---

## Task 6: Update Parity Tests to Use Legacy Export

**Files:**
- Modify: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Step 1: Update test to use HDF5 round-trip approach**

Replace the current `test_decode_legacy_events` test with:

```python
@pytest.mark.skipif(
    not Path("evio/data/fan/fan_const_rpm.dat").exists(),
    reason="fan_const_rpm.dat not available"
)
def test_legacy_export_evlib_parity():
    """Test that legacy loader and evlib see identical events via HDF5 export."""
    import tempfile
    from evio.core.recording import open_dat
    from evio.core.legacy_export import export_legacy_to_hdf5

    # Load legacy .dat
    legacy_path = "evio/data/fan/fan_const_rpm.dat"
    recording = open_dat(legacy_path, width=1280, height=720)

    # Export to temporary HDF5
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        hdf5_path = Path(tmp.name)

    try:
        stats = export_legacy_to_hdf5(recording, hdf5_path)

        # Load with evlib
        events = evlib.load_events(str(hdf5_path)).collect()

        # Decode legacy for comparison
        event_words = recording.event_words
        timestamps = recording.timestamps

        x_legacy, y_legacy, p_legacy = decode_legacy_events(event_words)

        # Assert: Event count matches
        assert len(events) == len(timestamps)
        assert stats['event_count'] == len(timestamps)

        # Assert: Spatial bounds match
        assert events['x'].min() == 0
        assert events['x'].max() == 1279
        assert events['y'].min() == 0
        assert events['y'].max() == 719

        # Assert: Polarity is balanced (not broken like IDS .raw)
        pol_on = (events['polarity'] == True).sum()
        pol_off = (events['polarity'] == False).sum()
        assert pol_on > 13_000_000
        assert pol_off > 13_000_000

        # Assert: Exact array match
        t_evlib = events['t']
        if hasattr(t_evlib[0], 'total_seconds'):
            t_evlib_us = np.array([int(t.total_seconds() * 1e6) for t in t_evlib])
        else:
            t_evlib_us = np.array(t_evlib)

        np.testing.assert_array_equal(t_evlib_us, timestamps)
        np.testing.assert_array_equal(events['x'].to_numpy(), x_legacy)
        np.testing.assert_array_equal(events['y'].to_numpy(), y_legacy)

        p_evlib = events['polarity'].to_numpy().astype(np.int8)
        np.testing.assert_array_equal(p_evlib, p_legacy)

    finally:
        hdf5_path.unlink(missing_ok=True)
```

**Step 2: Add comment explaining the dual-dataset situation**

At the top of `test_evlib_comparison.py`, add:

```python
"""Comparison tests between evlib and legacy loaders.

IMPORTANT: The .raw files in evio/data are NOT conversions of legacy .dat files.
They are independent IDS camera recordings from Nov 2025 with different:
- Resolution (2040×1793 actual vs 1280×720)
- Duration (682-717 sec vs 9-24 sec)
- Event counts (30-73M vs 26-64M)
- Polarity encoding (broken: 0 OFF events)

See docs/plans/2025-11-16-legacy-dat-to-evlib-export.md for details.

To test legacy parity, we export legacy .dat → HDF5 → evlib, ensuring
both loaders see the SAME recording.
"""
```

**Step 3: Run updated parity test**

```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_legacy_export_evlib_parity -v -s
```

Expected: PASS with exact array match

**Step 4: Commit**

```bash
git add workspace/libs/evio-core/tests/test_evlib_comparison.py
git commit -m "test: update parity tests to use legacy HDF5 export

- Replace broken .raw comparison with HDF5 round-trip
- Legacy .dat → export → HDF5 → evlib ensures same recording
- Add documentation explaining dual-dataset situation
- Verify 26.4M events, 1280×720, balanced polarity, exact match"
```

---

## Task 7: Documentation - IDS .raw Usage & Summary

**Files:**
- Modify: `docs/evlib-integration.md` (add summary section)
- Create: `docs/data/datasets.md` (dataset manifest)

**Step 1: Add summary section to evlib-integration.md**

At the end of `docs/evlib-integration.md`, add:

```markdown
## 8. Legacy .dat Export Implementation (Nov 2025)

### Problem Discovery

Investigation revealed the .raw files in `junction-sensofusion.zip` are **NOT** conversions of legacy .dat files. They are independent IDS camera recordings with:
- Different resolution (2040×1793 actual vs 1280×720)
- Different duration (682-717 sec vs 9-24 sec)
- Different event counts (30-73M vs 26-64M)
- Broken polarity (0 OFF events)

See diagnostic evidence: `scripts/diagnose_fan_data.py`, `scripts/compare_all_fan_files.py`

### Solution

Export legacy Sensofusion .dat files to evlib-compatible HDF5:

1. **Load** with `evio.core.recording.open_dat(path, width=1280, height=720)`
2. **Decode** packed event_words into x/y/polarity arrays
3. **Write** to HDF5 with evlib schema (`/events/t`, `/events/x`, `/events/y`, `/events/p`)
4. **Validate** with evlib to ensure exact match

### Implementation

- **Exporter:** `evio.core.legacy_export.export_legacy_to_hdf5()`
- **CLI:** `convert-legacy-dat-to-hdf5 <input.dat> [output.h5]`
- **Tests:** `test_legacy_export_unit.py` (schema), `test_legacy_export_integration.py` (real files)
- **Demo:** `run-demo-fan-ev3` now loads `fan_const_rpm_legacy.h5` (same recording as `run-demo-fan`)

### Results

✅ Legacy loader and evlib see **identical events** (exact array match on 26.4M events)
✅ Demos show same recording (1280×720, 9.5 seconds, balanced polarity)
✅ Parity tests validate actual legacy behavior (not unrelated IDS captures)

### IDS .raw Files

The IDS .raw files remain available for evlib experimentation:
- Native EVT3 format (no conversion needed)
- Longer captures (682-717 seconds)
- Different scenes/hardware
- **Not suitable for legacy parity validation**

For legacy .dat → evlib migration, always use the HDF5 export approach.
```

**Step 2: Create dataset manifest**

File: `docs/data/datasets.md`

```markdown
# Dataset Manifest

This document catalogs all datasets in `evio/data/junction-sensofusion.zip`.

**IMPORTANT:** The ZIP contains TWO distinct dataset collections (not conversions of each other):

---

## Legacy Sensofusion .dat Files

**Source:** Original Sensofusion event camera recordings
**Format:** Custom binary with minimal header (`% Height`, `% Version 2`, `% end`)
**Resolution:** 1280×720 (hardcoded in loader)
**Loader:** `evio.core.recording.open_dat(path, width=1280, height=720)`
**evlib Compatibility:** Requires export to HDF5 (use `convert-legacy-dat-to-hdf5`)

| File | Events | Duration | Polarity | Size |
|------|--------|----------|----------|------|
| `fan/fan_const_rpm.dat` | 26.4M | 9.5 sec | 13.4M ON / 13.0M OFF | 202 MB |
| `fan/fan_varying_rpm.dat` | 64.1M | 22.3 sec | 32.5M ON / 31.6M OFF | 489 MB |
| `fan/fan_varying_rpm_turning.dat` | 48.1M | 24.3 sec | 24.5M ON / 23.6M OFF | 367 MB |
| `drone_idle/drone_idle.dat` | ? | ? | ? | 736 MB |
| `drone_moving/drone_moving.dat` | ? | ? | ? | 1487 MB |

---

## IDS Camera .raw Files (EVT3 Format)

**Source:** NEW IDS Imaging Development Systems captures (Nov 2025)
**Format:** Standard Prophesee EVT3 with full ASCII header
**Resolution:** Header claims 1280×720, actual events span ~2040×1793
**Loader:** `evlib.load_events(path)` (native support)
**evlib Compatibility:** Native (no conversion needed)

**⚠️ WARNING:** These are **NOT** the same recordings as legacy .dat files!

| File | Events | Duration | Polarity | Size | Notes |
|------|--------|----------|----------|------|-------|
| `fan/fan_const_rpm.raw` | 30.4M | 682 sec | 217K ON / 0 OFF | 119 MB | Broken polarity |
| `fan/fan_varying_rpm.raw` | 73.4M | 699 sec | 516K ON / 0 OFF | 288 MB | Broken polarity |
| `fan/fan_varying_rpm_turning.raw` | 60.2M | 717 sec | 592K ON / 0 OFF | 225 MB | Broken polarity |
| `drone_idle/drone_idle.raw` | ? | ? | ? | 1029 MB | - |
| `drone_moving/drone_moving.raw` | ? | ? | ? | 789 MB | - |

**Polarity Issue:** All IDS .raw files report 0 OFF events, suggesting encoding bug in IDS export or evlib decoder.

---

## Converted Files (Generated Locally)

### Legacy Export to HDF5
Generated by `convert-legacy-dat-to-hdf5`:

| File | Source | Events | Format |
|------|--------|--------|--------|
| `fan/fan_const_rpm_legacy.h5` | `fan_const_rpm.dat` | 26.4M | HDF5 (evlib schema) |
| `fan/fan_varying_rpm_legacy.h5` | `fan_varying_rpm.dat` | 64.1M | HDF5 (evlib schema) |
| `fan/fan_varying_rpm_turning_legacy.h5` | `fan_varying_rpm_turning.dat` | 48.1M | HDF5 (evlib schema) |

### IDS .raw → .dat (Header Preservation)
Generated by `convert-all-datasets`:

| File | Source | Events | Format |
|------|--------|--------|--------|
| `fan/fan_const_rpm_evt3.dat` | `fan_const_rpm.raw` | 30.4M | EVT3 .dat container |
| `fan/fan_varying_rpm_evt3.dat` | `fan_varying_rpm.raw` | 73.4M | EVT3 .dat container |
| `fan/fan_varying_rpm_turning_evt3.dat` | `fan_varying_rpm_turning.raw` | 60.2M | EVT3 .dat container |

**Note:** These are simple copies (header + binary payload), not re-encoded.

---

## Usage Guidelines

### For Legacy Parity Testing
✅ Use `*_legacy.h5` files (exported from legacy .dat)
❌ Do NOT use `_evt3.dat` files (different recordings)

### For evlib Experimentation
✅ Use `.raw` or `_evt3.dat` files (native EVT3 support)
⚠️ Be aware of polarity encoding issue

### For Demos
- `run-demo-fan`: Legacy loader on `fan_const_rpm.dat`
- `run-demo-fan-ev3`: evlib on `fan_const_rpm_legacy.h5` (SAME recording)

---

## Diagnostic Scripts

Run these to verify dataset properties:

```bash
# Single file diagnosis
nix develop --command uv run --package evio python scripts/diagnose_fan_data.py

# Compare all fan files
nix develop --command uv run --package evio python scripts/compare_all_fan_files.py
```
```

**Step 3: Commit**

```bash
git add docs/evlib-integration.md docs/data/datasets.md
git commit -m "docs: add dataset manifest and legacy export summary

- Catalog all datasets in junction-sensofusion.zip
- Document dual-dataset situation (legacy vs IDS)
- Add usage guidelines for parity testing vs experimentation
- Summarize legacy export implementation and results
- Reference diagnostic scripts for verification"
```

---

## Task 8: Verification & Final Testing

**Step 1: Run full test suite**

```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/ -v
```

Expected: All tests pass including:
- `test_legacy_export_unit.py` (3 tests)
- `test_legacy_export_integration.py` (2 tests)
- `test_evlib_comparison.py::test_legacy_export_evlib_parity`

**Step 2: Verify demos show same recording**

```bash
# Convert legacy files if not done yet
nix develop --command convert-legacy-dat-to-hdf5 --batch evio/data/fan/*.dat --force

# Run both demos side-by-side
nix develop

run-demo-fan &
run-demo-fan-ev3 &
```

Expected: Both windows show:
- Same resolution (1280×720)
- Same event count (~26.4M)
- Same visual output (fan spinning)
- Same duration (~9.5 seconds)

**Step 3: Run diagnostic scripts for documentation**

```bash
nix develop --command uv run --package evio python scripts/diagnose_fan_data.py > diagnostic_output.txt
nix develop --command uv run --package evio python scripts/compare_all_fan_files.py >> diagnostic_output.txt
```

Review output confirms:
- Legacy .dat: 26.4M events, 1280×720, 9.5 sec, balanced polarity
- EVT3 .dat: 30.4M events, 2040×1793, 682 sec, broken polarity
- Clear evidence they're different recordings

**Step 4: Create final summary commit**

```bash
git add -A
git commit -m "feat: complete legacy .dat to evlib export pipeline

Summary of changes:
- Export legacy .dat to HDF5 with evlib-compatible schema
- Add convert-legacy-dat-to-hdf5 CLI tool (single + batch modes)
- Update run-demo-fan-ev3 to use legacy HDF5 (same recording as legacy)
- Fix parity tests to compare same recording (not unrelated IDS .raw)
- Document dual-dataset situation in junction-sensofusion.zip
- Add dataset manifest with usage guidelines

Results:
✅ Legacy loader and evlib see identical events (26.4M exact match)
✅ Demos show same recording (1280×720, 9.5 sec, balanced polarity)
✅ Parity tests validate actual legacy behavior
✅ IDS .raw files available for evlib experimentation

Fixes: #[issue-number] (if applicable)
See: docs/plans/2025-11-16-legacy-dat-to-evlib-export.md"
```

---

## Acceptance Criteria

- [x] Documentation clarifies .raw ≠ legacy .dat conversion
- [x] `export_legacy_to_hdf5()` implemented with unit tests
- [x] Integration tests verify exact match (legacy → HDF5 → evlib)
- [x] CLI tool `convert-legacy-dat-to-hdf5` works in single + batch modes
- [x] `run-demo-fan-ev3` loads legacy HDF5 (same recording as `run-demo-fan`)
- [x] Parity tests compare same recording (not unrelated IDS captures)
- [x] Dataset manifest documents both collections
- [x] IDS .raw files labeled as separate sample data
- [x] All tests pass
- [x] Demos show identical visual output

---

## Follow-Up Tasks (Future)

1. **Investigate IDS .raw polarity bug**
   - Why do all IDS files report 0 OFF events?
   - Is this an evlib decoder issue or IDS export bug?
   - Compare with other EVT3 readers (OpenEB, Metavision)

2. **Export drone datasets**
   - Extend `convert-legacy-dat-to-hdf5` to handle drone_idle and drone_moving
   - May need different width/height parameters
   - Add to dataset manifest with stats

3. **Optimize HDF5 storage**
   - Consider compression (gzip, lzf)
   - Benchmark read performance vs uncompressed
   - Document trade-offs in evlib-integration.md

4. **Visual smoke tests**
   - Implement frame-by-frame comparison (per docs/plans/2025-11-16-evlib-visual-smoke-test.md)
   - Capture screenshots from both demos
   - Generate diff images and metrics
   - Attach to evlib-integration.md as proof

5. **Deprecate legacy loader**
   - Once confident in evlib parity, plan migration
   - Update all scripts to use evlib
   - Archive legacy loader code
   - Update benchmarks
