#!/usr/bin/env python3
"""
⚠️ DEPRECATION WARNING ⚠️

This script operates on IDS camera .raw files (Nov 2025) which are SEPARATE recordings
from the legacy Sensofusion .dat files. They cannot be used for legacy parity testing.

For legacy data export, use: scripts/convert_legacy_dat_to_hdf5.py

This tool is preserved for IDS-specific experimentation only.

---

Convert EVT3 .raw files to .dat format while patching metadata to match actual stats.

This script now inspects the raw events via evlib before copying bytes so we can
log true resolution/duration/polarity counts and optionally fix the header fields
(`height`, `width`, `geometry`) when they are incorrect.

Usage:
    python scripts/convert_evt3_raw_to_dat.py <input.raw> [output.dat]

If output is not specified, creates <input>_evt3.dat in the same directory.

See docs/plans/raw-to-evt3-deprecation.md for evidence this approach was abandoned.
"""

from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path

import evlib
import polars as pl


def collect_evt3_stats(raw_path: Path) -> dict[str, int]:
    """Use evlib to gather x/y/t/p stats without loading the full dataset eagerly."""
    lazy = evlib.load_events(str(raw_path))
    schema = lazy.collect_schema()
    if isinstance(schema["t"], pl.Duration):
        t_expr = pl.col("t").dt.total_microseconds()
    else:
        t_expr = pl.col("t")

    stats = (
        lazy.select(
            pl.len().alias("events"),
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
            t_expr.min().alias("t_min"),
            t_expr.max().alias("t_max"),
            (pl.col("polarity") == 1).sum().alias("pol_on"),
            (pl.col("polarity") == -1).sum().alias("pol_off"),
        )
        .collect()
        .to_dicts()[0]
    )
    return {k: int(v) for k, v in stats.items()}


def patch_header_dimensions(header: bytes, width: int, height: int) -> bytes:
    """Rewrite width/height metadata in the ASCII header if present."""
    text = header.decode("ascii", errors="ignore")
    patched = text

    def _replace(pattern: str, replacement, src: str) -> tuple[str, bool]:
        new, count = re.subn(pattern, replacement, src, count=1, flags=re.IGNORECASE)
        return new, count > 0

    patched, _ = _replace(
        r"(format\s+EVT3;)([^\\n]*)",
        lambda m: f"{m.group(1)}height={height};width={width}",
        patched,
    )
    patched, _ = _replace(
        r"(height=)\d+",
        f"\\1{height}",
        patched,
    )
    patched, _ = _replace(
        r"(width=)\d+",
        f"\\1{width}",
        patched,
    )
    if "geometry" in patched:
        patched, _ = _replace(
            r"(geometry\s+)\d+x\d+",
            f"\\1{width}x{height}",
            patched,
        )
    else:
        patched = patched.replace("% end", f"% geometry {width}x{height}\n% end", 1)

    if patched != text:
        return patched.encode("ascii")
    return header


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
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    patch_header: bool = True,
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
    stats = collect_evt3_stats(input_path)
    width = stats["x_max"] + 1
    height = stats["y_max"] + 1
    if patch_header:
        header_bytes = patch_header_dimensions(header_bytes, width, height)

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
        'stats': stats,
        'width': width,
        'height': height,
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
    parser.add_argument(
        '--no-patch-header',
        action='store_true',
        help='Skip automatic width/height header patching',
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    # Deprecation warning
    print("=" * 70)
    print("⚠️  DEPRECATION WARNING")
    print("=" * 70)
    print()
    print("This tool converts IDS camera .raw files (separate recordings).")
    print("It does NOT convert legacy Sensofusion .dat files.")
    print()
    print("For legacy data export, use:")
    print("  nix develop --command convert-legacy-dat-to-hdf5 <file.dat>")
    print()
    print("Continuing with experimental IDS conversion...")
    print("=" * 70)
    print()

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
        stats = convert_raw_to_dat(
            args.input,
            output_path,
            patch_header=not args.no_patch_header,
        )
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
    print("Event stats (via evlib):")
    print(
        f"Events: {stats['stats']['events']:,} | "
        f"X range: {stats['stats']['x_min']}–{stats['stats']['x_max']} | "
        f"Y range: {stats['stats']['y_min']}–{stats['stats']['y_max']}"
    )
    duration_ms = (stats['stats']['t_max'] - stats['stats']['t_min']) / 1000
    print(
        f"Duration: {duration_ms / 1000:.2f}s "
        f"(t_min={stats['stats']['t_min']}, t_max={stats['stats']['t_max']})"
    )
    print(
        f"Polarity: ON={stats['stats']['pol_on']:,} "
        f"OFF={stats['stats']['pol_off']:,}"
    )
    print(f"Metadata resolution set to: {stats['width']}x{stats['height']}")
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
