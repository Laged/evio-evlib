#!/usr/bin/env python3
"""Test evlib.load_events() performance with chunked HDF5."""

import time
import evlib
import polars as pl

# Test newly converted chunked file
print("Testing chunked HDF5 load performance...")
print()

file_path = "evio/data/fan/fan_const_rpm_legacy.h5"

# Time the load
t_start = time.perf_counter()
lazy_events = evlib.load_events(file_path)
t_load = (time.perf_counter() - t_start) * 1000  # ms

print(f"⏱️  evlib.load_events(): {t_load:.1f}ms")
print()

# Verify it's actually a LazyFrame
print(f"Type: {type(lazy_events)}")

# Quick schema check (should be fast)
t_start = time.perf_counter()
schema = lazy_events.collect_schema()
t_schema = (time.perf_counter() - t_start) * 1000

print(f"⏱️  collect_schema(): {t_schema:.1f}ms")
print(f"Schema: {list(schema.keys())}")
print()

# Quick metadata extraction (should be fast with chunking)
t_start = time.perf_counter()
metadata = lazy_events.select([
    pl.col("x").max().alias("max_x"),
    pl.col("y").max().alias("max_y"),
    pl.col("t").min().alias("t_min"),
    pl.col("t").max().alias("t_max"),
]).collect()
t_meta = (time.perf_counter() - t_start) * 1000

print(f"⏱️  extract metadata: {t_meta:.1f}ms")
print(f"Resolution: {int(metadata['max_x'][0])+1}x{int(metadata['max_y'][0])+1}")
print()

total = t_load + t_schema + t_meta
print(f"TOTAL load + schema + metadata: {total:.1f}ms")
print()

if t_load < 500:
    print("✅ SUCCESS: Load time < 500ms (fast lazy loading working!)")
else:
    print(f"❌ FAILED: Load time {t_load:.1f}ms still too slow")
