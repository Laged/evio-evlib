#!/usr/bin/env python3
"""
Convert EVT3 .raw files to .dat format by preserving the header and binary payload.

This script performs a simple container format conversion without re-encoding events.
The .raw files already contain valid EVT3 headers (% evt 3.0, % format EVT3, etc.)
and binary event data that evlib can parse. We simply copy this to a .dat file.

Usage:
    python scripts/convert_evt3_raw_to_dat.py <input.raw> [output.dat]

If output is not specified, creates <input>_evt3.dat in the same directory.

See docs/evlib-integration.md for context and requirements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def validate_evt3_header(raw_path: Path) -> tuple[int, bytes]:
    """
    Validate that the .raw file contains an EVT3 header.

    Returns:
        (header_size, header_bytes) - size in bytes and the actual header content

    Raises:
        ValueError if the file doesn't have a valid EVT3 header
    """
    with open(raw_path, 'rb') as f:
        # Read first 4KB to find header (should be much smaller)
        chunk = f.read(4096)

    # Check for required EVT3 markers
    if b'% evt 3.0' not in chunk:
        raise ValueError(
            f"{raw_path.name} does not contain '% evt 3.0' marker. "
            "Only EVT3 .raw files can be converted."
        )

    if b'% format EVT3' not in chunk:
        raise ValueError(
            f"{raw_path.name} does not contain '% format EVT3' marker. "
            "Ensure this is a valid EVT3 recording."
        )

    # Find end of header (marked by "% end\n")
    header_end_marker = b'% end\n'
    header_end = chunk.find(header_end_marker)

    if header_end == -1:
        raise ValueError(
            f"{raw_path.name} header does not contain '% end' terminator. "
            "File may be corrupted."
        )

    # Header size includes the end marker
    header_size = header_end + len(header_end_marker)
    header_bytes = chunk[:header_size]

    return header_size, header_bytes


def convert_raw_to_dat(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 1024 * 1024  # 1MB chunks
) -> dict[str, int]:
    """
    Convert .raw to .dat by copying header and binary payload.

    Args:
        input_path: Source .raw file
        output_path: Destination .dat file
        chunk_size: Size of chunks for streaming copy

    Returns:
        Dictionary with conversion stats (header_bytes, event_bytes, total_bytes)
    """
    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate EVT3 header and get its size
    header_size, header_bytes = validate_evt3_header(input_path)

    # Get total file size
    total_size = input_path.stat().st_size
    event_bytes = total_size - header_size

    # Copy file: header + binary events
    with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
        # Write header
        dst.write(header_bytes)

        # Skip to end of header in source
        src.seek(header_size)

        # Stream-copy binary event data
        bytes_copied = 0
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            bytes_copied += len(chunk)

    return {
        'header_bytes': header_size,
        'event_bytes': event_bytes,
        'total_bytes': total_size,
        'output_path': str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert EVT3 .raw files to .dat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evio/data/fan/fan_const_rpm.raw
  %(prog)s evio/data/fan/fan_const_rpm.raw evio/data/fan/fan_evt3.dat

See docs/evlib-integration.md for details.
        """
    )

    parser.add_argument(
        'input',
        type=Path,
        help='Input .raw file (must be EVT3 format)'
    )

    parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        help='Output .dat file (default: <input>_evt3.dat)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output file if it exists'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine output path
    if args.output is None:
        # Create default output name: <basename>_evt3.dat
        output_path = args.input.parent / f"{args.input.stem}_evt3.dat"
    else:
        output_path = args.output

    # Check if output exists
    if output_path.exists() and not args.force:
        print(f"❌ Error: Output file already exists: {output_path}")
        print("   Use --force to overwrite")
        return 1

    # Convert
    print(f"Converting {args.input.name} to EVT3 .dat format...")
    print(f"Input : {args.input}")
    print(f"Output: {output_path}")
    print()

    try:
        stats = convert_raw_to_dat(args.input, output_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Success
    print("✅ Conversion successful!")
    print()
    print(f"Header size : {stats['header_bytes']:,} bytes")
    print(f"Event data  : {stats['event_bytes']:,} bytes")
    print(f"Total size  : {stats['total_bytes']:,} bytes")
    print()
    print(f"Output: {stats['output_path']}")
    print()
    print("Verify with: uv run --package evio-verifier verify-dat", output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
