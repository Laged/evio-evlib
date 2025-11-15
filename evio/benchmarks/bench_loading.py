#!/usr/bin/env python3
"""Benchmark evlib vs custom .dat loader performance."""

import time
from pathlib import Path
import argparse
import numpy as np


def benchmark_custom_loader(dat_path: str) -> dict:
    """Benchmark current evio .dat loader."""
    from evio.core.recording import open_dat

    start = time.perf_counter()
    rec = open_dat(dat_path, width=1280, height=720)

    # Force evaluation by accessing data
    event_count = len(rec.event_words)

    elapsed = time.perf_counter() - start

    return {
        'method': 'custom',
        'events': event_count,
        'time_sec': elapsed,
        'events_per_sec': event_count / elapsed if elapsed > 0 else 0,
    }


def benchmark_evlib_loader(dat_path: str) -> dict:
    """Benchmark evlib loader.

    Note: evlib currently doesn't support the custom binary .dat format used in this
    project. This function will skip evlib benchmarking for .dat files and return None.
    For proper evlib benchmarking, use standard event camera formats (.aedat, .h5, etc.)
    """
    from evio.evlib_loader import load_events_with_evlib

    # Check if file is custom .dat format
    if dat_path.endswith('.dat'):
        print(f"  Note: Skipping evlib for custom .dat format - not supported by evlib")
        print(f"        evlib supports: .aedat, .aedat4, .h5, .csv, etc.")
        print(f"        To benchmark evlib, use standard event camera file formats")
        return None

    # For supported formats, benchmark normally
    start = time.perf_counter()
    events = load_events_with_evlib(dat_path)
    events_df = events.collect()
    event_count = len(events_df)
    elapsed = time.perf_counter() - start

    return {
        'method': 'evlib',
        'events': event_count,
        'time_sec': elapsed,
        'events_per_sec': event_count / elapsed if elapsed > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dat_file', help='Path to .dat file for benchmarking')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per method')
    args = parser.parse_args()

    if not Path(args.dat_file).exists():
        print(f"Error: {args.dat_file} not found")
        return

    print(f"Benchmarking event loading: {args.dat_file}")
    print(f"Runs per method: {args.runs}\n")

    # Benchmark custom loader
    print("Custom loader:")
    custom_results = []
    for i in range(args.runs):
        result = benchmark_custom_loader(args.dat_file)
        custom_results.append(result)
        print(f"  Run {i+1}: {result['time_sec']:.4f}s "
              f"({result['events_per_sec']/1e6:.1f}M events/s)")

    # Benchmark evlib loader
    print("\nevlib loader:")
    evlib_results = []
    for i in range(args.runs):
        result = benchmark_evlib_loader(args.dat_file)
        if result is None:
            break  # Skip if format not supported
        evlib_results.append(result)
        print(f"  Run {i+1}: {result['time_sec']:.4f}s "
              f"({result['events_per_sec']/1e6:.1f}M events/s)")

    # Calculate averages
    custom_avg = np.mean([r['time_sec'] for r in custom_results])

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Custom: {custom_avg:.4f}s average")

    if evlib_results:
        evlib_avg = np.mean([r['time_sec'] for r in evlib_results])
        speedup = custom_avg / evlib_avg if evlib_avg > 0 else 0
        print(f"  evlib:  {evlib_avg:.4f}s average")
        print(f"  Speedup: {speedup:.1f}x")
    else:
        print(f"  evlib:  skipped (format not supported)")
        print(f"\nNote: To benchmark evlib, provide a standard format file:")
        print(f"  Example: python benchmarks/bench_loading.py data/events.aedat4")

    print(f"{'='*50}")


if __name__ == '__main__':
    main()
