# Unzip Datasets Command Design

**Date:** 2025-11-15
**Status:** Approved for implementation

## Overview

Create an `unzip-datasets` command that provides a simpler alternative to `download-datasets` for hackathon participants who receive the `junction-sensofusion.zip` file directly from mentors.

## Background

Users currently have two options for getting datasets:
1. **Automated download** via `download-datasets` - Downloads from Google Drive (may hit quota limits)
2. **Manual unzip** - Users manually extract ZIP file (requires knowing exact commands)

This design adds a third option: a guided unzip command that provides the same user experience as the automated downloader but works with the pre-distributed ZIP file.

## Requirements

1. Simple workflow: copy ZIP to `evio/data/`, run `unzip-datasets`
2. Confirm before overwriting existing datasets
3. Verify extraction was successful using existing inventory system
4. Keep ZIP file after extraction (don't auto-delete)
5. Ensure ZIP file is in `.gitignore`
6. Consistent UX with `download-datasets`

## Design

### Command Structure

**Command:** `unzip-datasets` (shell alias defined in flake.nix)

**Implementation:** Bash script at `workspace/tools/downloader/src/downloader/unzip_datasets.sh`

### User Flow

1. **Prerequisite Check:**
   - Check if `evio/data/junction-sensofusion.zip` exists
   - If missing: Show error with instructions to copy ZIP file
   - If present: Continue to step 2

2. **Inventory Check:**
   - Call `downloader.verification.check_inventory()` to detect existing datasets
   - If datasets exist: Show warning with list of what will be overwritten
   - Prompt: "Continue with extraction? (y/N)"
   - If user declines: Exit gracefully
   - If no datasets exist: Skip confirmation (nothing to overwrite)

3. **Extraction:**
   - Run `unzip -o evio/data/junction-sensofusion.zip -d evio/data/`
   - The `-o` flag overwrites without prompting (we already confirmed in step 2)
   - Capture exit code to detect failures

4. **Verification:**
   - Call `downloader.verification.check_inventory()` again
   - Compare before/after to confirm extraction worked
   - Detect missing expected files

5. **Summary:**
   - Show inventory table (same format as `download-datasets`)
   - Show demo commands if fan datasets present
   - ZIP file remains in `evio/data/` for future use

### File Changes

#### 1. New Script: `workspace/tools/downloader/src/downloader/unzip_datasets.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Unzip Datasets Command
# Extracts junction-sensofusion.zip with confirmation and verification

ZIP_PATH="evio/data/junction-sensofusion.zip"
DATA_DIR="evio/data"

# Check ZIP exists
if [ ! -f "$ZIP_PATH" ]; then
    echo "❌ Error: junction-sensofusion.zip not found"
    echo ""
    echo "Please copy the ZIP file to evio/data/ first:"
    echo "  cp /path/to/junction-sensofusion.zip evio/data/"
    echo ""
    echo "Then run: unzip-datasets"
    exit 1
fi

# Check current inventory
echo "Checking existing datasets..."
uv run --package downloader python -c "
from downloader.verification import check_inventory, print_inventory
inventory = check_inventory()
print_inventory(inventory)
" || true

# Check if datasets exist (simple heuristic: check for fan/ directory)
if [ -d "$DATA_DIR/fan" ] || [ -d "$DATA_DIR/drone_idle" ] || [ -d "$DATA_DIR/drone_moving" ] || [ -d "$DATA_DIR/fred-0" ]; then
    echo ""
    echo "⚠️  WARNING: Existing datasets found"
    echo ""
    echo "The following will be overwritten:"
    [ -d "$DATA_DIR/fan" ] && echo "  ✓ fan/ (6 files)"
    [ -d "$DATA_DIR/drone_idle" ] && echo "  ✓ drone_idle/ (2 files)"
    [ -d "$DATA_DIR/drone_moving" ] && echo "  ✓ drone_moving/ (2 files)"
    [ -d "$DATA_DIR/fred-0" ] && echo "  ✓ fred-0/ (events + frames)"
    echo ""
    read -p "Continue with extraction? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Extraction cancelled."
        exit 0
    fi
fi

# Extract ZIP
echo ""
echo "Extracting datasets..."
if ! unzip -o "$ZIP_PATH" -d "$DATA_DIR"; then
    echo ""
    echo "❌ Error: Failed to extract datasets"
    echo ""
    echo "The ZIP file may be corrupted. Try:"
    echo "  1. Re-download junction-sensofusion.zip"
    echo "  2. Verify file integrity"
    echo "  3. Extract manually: cd evio/data && unzip junction-sensofusion.zip"
    exit 1
fi

# Verify extraction
echo ""
echo "Verifying extraction..."
uv run --package downloader python -c "
from downloader.verification import check_inventory, print_inventory

inventory = check_inventory()

# Check for expected datasets
expected = ['fan', 'drone_idle', 'drone_moving', 'fred-0']
found = [name for name in expected if name in inventory and inventory[name].get('dat', 0) > 0 or inventory[name].get('raw', 0) > 0]
missing = [name for name in expected if name not in found]

if missing:
    print('⚠️  Warning: Extraction incomplete')
    print('')
    print('Missing expected datasets:', ', '.join(missing))
    print('')
    print('Please check:')
    print('  - ZIP file integrity')
    print('  - Available disk space')
    print('')
    print('Run download-datasets as fallback if needed.')
    exit(1)

print('=' * 50)
print('  Extraction Summary')
print('=' * 50)
print('')
print(f'✅ Successfully extracted {len(found)} dataset groups')
print('')
print_inventory(inventory)

# Show demo commands if fan datasets present
if inventory.get('fan', {}).get('dat', 0) > 0:
    print('')
    print('Ready to run:')
    print('  run-demo-fan')
    print('  run-mvp-1')
    print('  run-mvp-2')
"
```

**Make executable:** `chmod +x workspace/tools/downloader/src/downloader/unzip_datasets.sh`

#### 2. Update `flake.nix`

Add `unzip` to buildInputs (line 36):
```nix
buildInputs = [
  # Core tools
  python
  pkgs.uv                 # UV package manager
  pkgs.gdown              # Google Drive downloader
  pkgs.unzip              # ZIP extraction
  ...
```

Add alias to shellHook (line 113):
```nix
# Shell aliases for convenience
alias download-datasets='uv run --package downloader download-datasets'
alias unzip-datasets='bash workspace/tools/downloader/src/downloader/unzip_datasets.sh'
alias run-demo-fan='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat'
...
```

Update help text (line 100):
```nix
echo "📊 Dataset Management:"
echo "  download-datasets    : Download from Google Drive (~1.4 GB)"
echo "  unzip-datasets       : Extract junction-sensofusion.zip"
```

#### 3. Update `.gitignore`

Add to existing `evio/data/` section:
```
# Event camera datasets (large binary files)
evio/data/*.dat
evio/data/*.raw
evio/data/*.h5
evio/data/*.aedat
evio/data/junction-sensofusion.zip
```

#### 4. Update `evio/data/README.md`

Reorder sections to make `unzip-datasets` the primary recommended method:

```markdown
## Downloading Datasets

**These files are not committed to git** (they are large binary files ~5.6 GB total, listed in .gitignore).

### Option 1: Extract with unzip-datasets (Recommended for Hackathon)

If you have the `junction-sensofusion.zip` file:

1. Copy the ZIP file to this directory:
   ```bash
   cp /path/to/junction-sensofusion.zip evio/data/
   ```

2. From the `nix develop` shell, run:
   ```bash
   unzip-datasets
   ```

This will extract and verify all datasets automatically.

### Option 2: Manual Extraction

If you prefer manual control:

```bash
cd evio/data/
unzip junction-sensofusion.zip
```

### Option 3: Automated Download

If you don't have the ZIP file, download from Google Drive:

```bash
download-datasets
```

Note: May hit quota limits during high-traffic periods.

...
```

## Error Handling

### Error: ZIP file not found
```
❌ Error: junction-sensofusion.zip not found

Please copy the ZIP file to evio/data/ first:
  cp /path/to/junction-sensofusion.zip evio/data/

Then run: unzip-datasets
```

**Exit code:** 1

### Error: Unzip command fails
```
❌ Error: Failed to extract datasets

The ZIP file may be corrupted. Try:
  1. Re-download junction-sensofusion.zip
  2. Verify file integrity
  3. Extract manually: cd evio/data && unzip junction-sensofusion.zip
```

**Exit code:** 1

### Warning: Verification fails
```
⚠️  Warning: Extraction incomplete

Missing expected datasets: drone_moving, fred-0

Please check:
  - ZIP file integrity
  - Available disk space

Run download-datasets as fallback if needed.
```

**Exit code:** 1

### Success
```
==================================================
  Extraction Summary
==================================================

✅ Successfully extracted 4 dataset groups

Inventory:
  fan/: 6 files (1.7 GB)
  drone_idle/: 2 files (1.7 GB)
  drone_moving/: 2 files (2.2 GB)
  fred-0/: events + 50 frames (400 MB)

Ready to run:
  run-demo-fan
  run-mvp-1
  run-mvp-2
```

**Exit code:** 0

## Testing Plan

1. **Test: ZIP not found**
   - Remove ZIP file
   - Run `unzip-datasets`
   - Verify error message and exit code 1

2. **Test: Fresh extraction**
   - Remove all datasets
   - Place ZIP in evio/data/
   - Run `unzip-datasets`
   - Verify no confirmation prompt (nothing to overwrite)
   - Verify successful extraction and inventory

3. **Test: Overwrite with confirmation (accept)**
   - With datasets already present
   - Run `unzip-datasets`
   - Answer 'y' to confirmation
   - Verify extraction proceeds

4. **Test: Overwrite with confirmation (decline)**
   - With datasets already present
   - Run `unzip-datasets`
   - Answer 'n' to confirmation
   - Verify graceful exit, no extraction

5. **Test: Corrupted ZIP**
   - Create invalid ZIP file
   - Run `unzip-datasets`
   - Verify error message

6. **Test: ZIP remains after extraction**
   - Run successful extraction
   - Verify `junction-sensofusion.zip` still exists in evio/data/

## Implementation Notes

- Script uses `set -euo pipefail` for strict error handling
- Uses `uv run --package downloader python -c "..."` to call Python inventory functions
- Simple directory checks for confirmation prompt (avoids complex Python invocation)
- Reuses existing `check_inventory()` and `print_inventory()` for consistency
- Exit codes: 0 = success, 1 = error/incomplete

## Migration Path

No migration needed. This is a new alternative command that complements existing workflows:
- `download-datasets` - Still works for Google Drive downloads
- Manual unzip - Still works as documented
- `unzip-datasets` - New guided workflow for ZIP files

Users can choose whichever method suits their situation.
