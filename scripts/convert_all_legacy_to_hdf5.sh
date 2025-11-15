#!/usr/bin/env bash
#
# Convert all legacy .dat files to evlib-compatible HDF5 format.
#
# This script finds all legacy .dat files in evio/data/ and converts them
# to HDF5 using the convert-legacy-dat-to-hdf5 tool.
#
# Usage:
#   nix develop --command convert-all-legacy-to-hdf5
#
# See docs/plans/2025-11-16-legacy-dat-to-evlib-export.md for context.

set -euo pipefail

DATA_DIR="evio/data"

echo "=========================================="
echo "  Convert All Legacy .dat to HDF5"
echo "=========================================="
echo ""

# Find all .dat files (excluding _evt3.dat files)
echo "Scanning for legacy .dat files..."
DAT_FILES=$(find "$DATA_DIR" -name "*.dat" -type f ! -name "*_evt3.dat" 2>/dev/null || true)

if [ -z "$DAT_FILES" ]; then
    echo "❌ No legacy .dat files found in $DATA_DIR"
    echo ""
    echo "Run 'unzip-datasets' first to extract datasets."
    exit 1
fi

# Count files
FILE_COUNT=$(echo "$DAT_FILES" | wc -l | tr -d ' ')
echo "Found $FILE_COUNT legacy .dat files to convert"
echo ""

# Convert each file
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_SIZE=0

for DAT_FILE in $DAT_FILES; do
    BASENAME=$(basename "$DAT_FILE")
    echo "Converting: $BASENAME"

    # Determine output path
    DIR=$(dirname "$DAT_FILE")
    STEM="${BASENAME%.dat}"
    OUTPUT="$DIR/${STEM}_legacy.h5"

    # Check if already exists
    if [ -f "$OUTPUT" ]; then
        echo "  ⚠️  Output exists: ${STEM}_legacy.h5"
        echo "  Skipping (use --force to overwrite)"
        echo ""
        continue
    fi

    # Convert (use Python directly since we're already in nix develop)
    if python scripts/convert_legacy_dat_to_hdf5.py "$DAT_FILE" "$OUTPUT" --width 1280 --height 720; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Get file size
        if [ -f "$OUTPUT" ]; then
            SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null || echo 0)
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
        fi
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  ⚠️  Failed to convert $BASENAME"
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "  Conversion Summary"
echo "=========================================="
echo ""
echo "✅ Successfully converted: $SUCCESS_COUNT files"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "❌ Failed: $FAIL_COUNT files"
fi

# Show total size in human-readable format
if [ $TOTAL_SIZE -gt 0 ]; then
    TOTAL_GB=$(echo "scale=2; $TOTAL_SIZE / 1024 / 1024 / 1024" | bc)
    TOTAL_MB=$(echo "scale=1; $TOTAL_SIZE / 1024 / 1024" | bc)

    if (( $(echo "$TOTAL_GB > 1" | bc -l) )); then
        echo "📦 Total size: ${TOTAL_GB} GB"
    else
        echo "📦 Total size: ${TOTAL_MB} MB"
    fi
fi

echo ""
echo "Verify converted files with:"
echo "  nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_legacy_export_integration.py -v"
echo ""
echo "Run demos:"
echo "  run-demo-fan         # Legacy loader"
echo "  run-demo-fan-ev3     # evlib on exported HDF5"
