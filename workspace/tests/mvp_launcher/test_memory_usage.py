#!/usr/bin/env python3
"""Measure memory usage of caching LazyFrames."""

import sys
import time
from pathlib import Path

import evlib
import polars as pl


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "flake.nix").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (missing flake.nix)")


REPO_ROOT = find_repo_root()

def get_object_size(obj):
    """Estimate object size in MB."""
    return sys.getsizeof(obj) / 1024 / 1024

print("=" * 60)
print("Memory Analysis: Caching LazyFrames vs Re-loading")
print("=" * 60)
print()

# Test all datasets
datasets = [
    ("fan_const_rpm_legacy.h5", REPO_ROOT / "evio" / "data" / "fan" / "fan_const_rpm_legacy.h5"),
    ("drone_idle_legacy.h5", REPO_ROOT / "evio" / "data" / "drone_idle" / "drone_idle_legacy.h5"),
    ("drone_moving_legacy.h5", REPO_ROOT / "evio" / "data" / "drone_moving" / "drone_moving_legacy.h5"),
]

print("Dataset Analysis:")
print()

total_memory = 0
cache = {}

for name, path in datasets:
    # Check if file exists
    if not path.exists():
        print(f"⚠️  {name}: SKIPPED (file not found)")
        continue

    print(f"{name}:")

    # Load LazyFrame
    t_start = time.perf_counter()
    lazy_events = evlib.load_events(str(path))
    t_load = (time.perf_counter() - t_start) * 1000

    # Get event count and schema
    count = len(lazy_events.collect())
    schema = lazy_events.collect_schema()

    # Estimate memory
    # LazyFrame itself is tiny (just metadata/pointer)
    # But the underlying data is loaded (as we proved)
    lazy_size = get_object_size(lazy_events)

    # Polars LazyFrame memory = roughly the collected DataFrame size
    # ~23 bytes/event according to evlib docs
    estimated_data_mb = (count * 23) / 1024 / 1024

    print(f"  Events: {count:,}")
    print(f"  Load time: {t_load:.1f}ms")
    print(f"  LazyFrame object: {lazy_size:.2f} MB")
    print(f"  Estimated data: {estimated_data_mb:.1f} MB")
    print()

    # Cache it
    cache[name] = lazy_events
    total_memory += estimated_data_mb

print("=" * 60)
print("Caching Strategy Analysis:")
print("=" * 60)
print()

print("If we cache ALL datasets:")
print(f"  Total memory needed: ~{total_memory:.1f} MB")
print()

print("Typical usage pattern:")
print("  - User loads 1-2 datasets per session")
print("  - Cache only when needed (lazy caching)")
print("  - Estimated: ~200-400 MB for 1-2 datasets")
print()

print("Recommendation:")
if total_memory < 500:
    print("  ✅ CACHE ALL datasets (< 500 MB total)")
    print("  - Fast switching between any dataset")
    print("  - Negligible memory overhead on modern systems")
elif total_memory < 2000:
    print("  ⚠️  LAZY CACHE (on-demand caching)")
    print("  - Cache each dataset on first load")
    print("  - Keep cache until app exit")
    print("  - Estimated: ~200-400 MB per dataset")
else:
    print("  ❌ DON'T CACHE (too much memory)")
    print("  - Re-load each time (3s penalty)")
    print("  - Alternative: optimize evlib or use different loader")

print()
print("Current system requirements:")
print("  - Minimum: 8GB RAM (per evlib docs)")
print("  - Recommended: 16GB+ for comfortable usage")
