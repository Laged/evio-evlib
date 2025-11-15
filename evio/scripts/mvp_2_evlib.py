#!/usr/bin/env python3
"""
MVP-2: evlib-accelerated voxel FFT for RPM detection.

100x faster than NumPy implementation for large datasets.
"""

import argparse
from pathlib import Path

from evio.evlib_loader import load_events_with_evlib
from evio.mvp.rpm_detector import RPMDetector


def main():
    parser = argparse.ArgumentParser(
        description="Detect RPM using evlib-accelerated voxel FFT"
    )
    parser.add_argument("dat_file", help="Path to .dat file")
    parser.add_argument(
        "--bins", type=int, default=50, help="Number of temporal bins"
    )
    parser.add_argument(
        "--blades", type=int, default=4, help="Number of fan blades"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Sensor width"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Sensor height"
    )
    args = parser.parse_args()

    if not Path(args.dat_file).exists():
        print(f"Error: {args.dat_file} not found")
        return

    print(f"Loading events from {args.dat_file}...")
    events = load_events_with_evlib(args.dat_file)

    print(f"Creating voxel grid ({args.bins} bins)...")
    detector = RPMDetector(
        height=args.height,
        width=args.width,
        n_time_bins=args.bins,
        num_blades=args.blades,
    )

    print("Detecting RPM...")
    rpm = detector.detect_rpm(events)

    print(f"\n{'='*60}")
    print(f"Detected RPM: {rpm:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
