"""Diagnose fan dataset file differences."""

import sys
from pathlib import Path

import evlib
import numpy as np

sys.path.insert(0, 'workspace/libs/evio-core/src')
from evio.core.recording import open_dat


def analyze_legacy_dat(dat_path: Path, width: int, height: int):
    """Analyze legacy .dat file."""
    print(f"=== LEGACY .dat: {dat_path.name} ===")
    print(f"File size: {dat_path.stat().st_size / 1024 / 1024:.1f} MB")

    rec = open_dat(str(dat_path), width=width, height=height)
    event_words = rec.event_words
    timestamps = rec.timestamps

    # Decode events
    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    polarity = (raw_polarity > 0).astype(np.int8)

    print(f"Event count: {len(timestamps):,}")
    print(f"Resolution (specified): {width}x{height}")
    print(f"X range: {x.min()}-{x.max()}")
    print(f"Y range: {y.min()}-{y.max()}")
    print(f"Polarity: {(polarity == 1).sum():,} ON, {(polarity == 0).sum():,} OFF")
    print(f"Time range: {timestamps.min():,} - {timestamps.max():,} µs")
    print(f"Duration: {(timestamps.max() - timestamps.min()) / 1e6:.2f} seconds")
    print()

    return {
        'event_count': len(timestamps),
        'x_min': int(x.min()),
        'x_max': int(x.max()),
        'y_min': int(y.min()),
        'y_max': int(y.max()),
    }


def analyze_evt3_dat(dat_path: Path):
    """Analyze EVT3 .dat file."""
    print(f"=== EVT3 .dat: {dat_path.name} ===")
    print(f"File size: {dat_path.stat().st_size / 1024 / 1024:.1f} MB")

    events = evlib.load_events(str(dat_path)).collect()

    print(f"Event count: {len(events):,}")
    print(f"X range: {events['x'].min()}-{events['x'].max()}")
    print(f"Y range: {events['y'].min()}-{events['y'].max()}")
    print(f"Polarity: {(events['polarity'] == True).sum():,} ON, {(events['polarity'] == False).sum():,} OFF")
    # evlib returns timedelta - convert to int for display
    t_min = int(events['t'].min().total_seconds() * 1e6) if hasattr(events['t'].min(), 'total_seconds') else events['t'].min()
    t_max = int(events['t'].max().total_seconds() * 1e6) if hasattr(events['t'].max(), 'total_seconds') else events['t'].max()
    print(f"Time range: {t_min:,} - {t_max:,} µs")
    print(f"Duration: {(t_max - t_min) / 1e6:.2f} seconds")
    print()

    return {
        'event_count': len(events),
        'x_min': int(events['x'].min()),
        'x_max': int(events['x'].max()),
        'y_min': int(events['y'].min()),
        'y_max': int(events['y'].max()),
    }


def main():
    data_dir = Path("evio/data/fan")

    print("=" * 60)
    print("  FAN DATASET DIAGNOSTIC")
    print("=" * 60)
    print()

    # Analyze legacy .dat
    legacy_stats = analyze_legacy_dat(
        data_dir / "fan_const_rpm.dat",
        width=1280,
        height=720
    )

    # Analyze EVT3 .dat
    evt3_stats = analyze_evt3_dat(
        data_dir / "fan_const_rpm_evt3.dat"
    )

    # Compare
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print()

    print(f"Event count match: {legacy_stats['event_count'] == evt3_stats['event_count']}")
    print(f"  Legacy: {legacy_stats['event_count']:,}")
    print(f"  EVT3:   {evt3_stats['event_count']:,}")
    print()

    print(f"Spatial range match: {legacy_stats['x_max'] == evt3_stats['x_max'] and legacy_stats['y_max'] == evt3_stats['y_max']}")
    print(f"  Legacy: x={legacy_stats['x_min']}-{legacy_stats['x_max']}, y={legacy_stats['y_min']}-{legacy_stats['y_max']}")
    print(f"  EVT3:   x={evt3_stats['x_min']}-{evt3_stats['x_max']}, y={evt3_stats['y_min']}-{evt3_stats['y_max']}")
    print()

    print("=" * 60)
    print("  CONCLUSION")
    print("=" * 60)
    print()

    if legacy_stats['event_count'] != evt3_stats['event_count']:
        print("❌ DIFFERENT RECORDINGS - Event counts don't match")
        print()
        print("The fan_const_rpm_evt3.dat was created from fan_const_rpm.raw,")
        print("which is a DIFFERENT recording than fan_const_rpm.dat (legacy).")
        print()
        print("To fix: Need to export legacy .dat → HDF5 → evlib-compatible format")
    else:
        print("✅ Same recording")


if __name__ == "__main__":
    main()
