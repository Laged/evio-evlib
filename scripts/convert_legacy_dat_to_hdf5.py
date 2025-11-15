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
