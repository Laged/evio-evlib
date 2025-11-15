"""Compare ALL fan dataset files to understand .dat vs .raw relationship."""

import sys
from pathlib import Path

import evlib
import numpy as np

sys.path.insert(0, 'workspace/libs/evio-core/src')
from evio.core.recording import open_dat


def get_legacy_stats(dat_path: Path, width: int, height: int):
    """Get stats from legacy .dat file."""
    rec = open_dat(str(dat_path), width=width, height=height)
    event_words = rec.event_words
    timestamps = rec.timestamps

    # Decode events
    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    polarity = (raw_polarity > 0).astype(np.int8)

    return {
        'path': str(dat_path.name),
        'size_mb': dat_path.stat().st_size / 1024 / 1024,
        'event_count': len(timestamps),
        'x_range': f"{x.min()}-{x.max()}",
        'y_range': f"{y.min()}-{y.max()}",
        'duration_sec': (timestamps.max() - timestamps.min()) / 1e6,
        'pol_on': (polarity == 1).sum(),
        'pol_off': (polarity == 0).sum(),
    }


def get_evt3_stats(dat_path: Path):
    """Get stats from EVT3 .dat file."""
    events = evlib.load_events(str(dat_path)).collect()

    # Handle timedelta for time
    t_min = events['t'].min()
    t_max = events['t'].max()
    if hasattr(t_min, 'total_seconds'):
        t_min = int(t_min.total_seconds() * 1e6)
        t_max = int(t_max.total_seconds() * 1e6)

    duration_sec = (t_max - t_min) / 1e6

    return {
        'path': str(dat_path.name),
        'size_mb': dat_path.stat().st_size / 1024 / 1024,
        'event_count': len(events),
        'x_range': f"{events['x'].min()}-{events['x'].max()}",
        'y_range': f"{events['y'].min()}-{events['y'].max()}",
        'duration_sec': duration_sec,
        'pol_on': (events['polarity'] == True).sum(),
        'pol_off': (events['polarity'] == False).sum(),
    }


def compare_pair(name: str, legacy_stats: dict, evt3_stats: dict):
    """Compare legacy vs EVT3 stats."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    print(f"\n{'Metric':<20} {'Legacy .dat':<25} {'EVT3 .dat':<25} {'Match':<10}")
    print("-" * 80)

    # File size
    print(f"{'File size (MB)':<20} {legacy_stats['size_mb']:>24.1f} {evt3_stats['size_mb']:>24.1f}")

    # Event count
    match_events = '✓' if legacy_stats['event_count'] == evt3_stats['event_count'] else '✗ DIFF'
    print(f"{'Event count':<20} {legacy_stats['event_count']:>24,} {evt3_stats['event_count']:>24,} {match_events:<10}")

    # X range
    match_x = '✓' if legacy_stats['x_range'] == evt3_stats['x_range'] else '✗ DIFF'
    print(f"{'X range':<20} {legacy_stats['x_range']:>24} {evt3_stats['x_range']:>24} {match_x:<10}")

    # Y range
    match_y = '✓' if legacy_stats['y_range'] == evt3_stats['y_range'] else '✗ DIFF'
    print(f"{'Y range':<20} {legacy_stats['y_range']:>24} {evt3_stats['y_range']:>24} {match_y:<10}")

    # Duration
    dur_diff = abs(legacy_stats['duration_sec'] - evt3_stats['duration_sec'])
    match_dur = '✓' if dur_diff < 0.1 else '✗ DIFF'
    print(f"{'Duration (sec)':<20} {legacy_stats['duration_sec']:>24.2f} {evt3_stats['duration_sec']:>24.2f} {match_dur:<10}")

    # Polarity ON
    match_pol_on = '✓' if legacy_stats['pol_on'] == evt3_stats['pol_on'] else '✗ DIFF'
    print(f"{'Polarity ON':<20} {legacy_stats['pol_on']:>24,} {evt3_stats['pol_on']:>24,} {match_pol_on:<10}")

    # Polarity OFF
    match_pol_off = '✓' if legacy_stats['pol_off'] == evt3_stats['pol_off'] else '✗ DIFF'
    print(f"{'Polarity OFF':<20} {legacy_stats['pol_off']:>24,} {evt3_stats['pol_off']:>24,} {match_pol_off:<10}")

    # Overall verdict
    all_match = all([
        legacy_stats['event_count'] == evt3_stats['event_count'],
        legacy_stats['x_range'] == evt3_stats['x_range'],
        legacy_stats['y_range'] == evt3_stats['y_range'],
        dur_diff < 0.1,
    ])

    print(f"\n{'VERDICT':<20} {('✅ SAME RECORDING' if all_match else '❌ DIFFERENT RECORDINGS')}")

    return all_match


def main():
    data_dir = Path("evio/data/fan")

    print("=" * 80)
    print("  FAN DATASET COMPREHENSIVE COMPARISON")
    print("  Checking if .raw → _evt3.dat represents the same recording as legacy .dat")
    print("=" * 80)

    # All fan datasets
    datasets = [
        ("fan_const_rpm", 1280, 720),
        ("fan_varying_rpm", 1280, 720),
        ("fan_varying_rpm_turning", 1280, 720),
    ]

    results = []

    for name, width, height in datasets:
        legacy_path = data_dir / f"{name}.dat"
        evt3_path = data_dir / f"{name}_evt3.dat"

        if not legacy_path.exists():
            print(f"\n⚠️  Skipping {name}: {legacy_path.name} not found")
            continue

        if not evt3_path.exists():
            print(f"\n⚠️  Skipping {name}: {evt3_path.name} not found")
            continue

        try:
            legacy_stats = get_legacy_stats(legacy_path, width, height)
            evt3_stats = get_evt3_stats(evt3_path)
            match = compare_pair(name, legacy_stats, evt3_stats)
            results.append((name, match))
        except Exception as e:
            print(f"\n❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, None))

    # Summary
    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print("=" * 80)
    print()

    for name, match in results:
        if match is None:
            status = "⚠️  ERROR"
        elif match:
            status = "✅ MATCH"
        else:
            status = "❌ MISMATCH"
        print(f"{status}  {name}")

    print()
    print("CONCLUSION:")

    matches = [m for m in results if m[1] is True]
    mismatches = [m for m in results if m[1] is False]

    if len(mismatches) == len(results):
        print("ALL files show different recordings between legacy .dat and EVT3 .dat")
        print("This suggests .raw files are NOT derived from legacy .dat files.")
    elif len(matches) == len(results):
        print("ALL files match! The .raw files ARE the same recordings as legacy .dat")
        print("The differences might be encoding/polarity mapping issues.")
    else:
        print(f"{len(matches)}/{len(results)} files match.")
        print("Mixed results suggest selective conversion or different sources.")


if __name__ == "__main__":
    main()
