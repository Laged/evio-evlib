#!/usr/bin/env python3
"""Test if evlib is using streaming for our HDF5 files."""

import time
import evlib
import polars as pl
import psutil
import os

def monitor_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("Testing evlib streaming behavior...")
print()

file_path = "evio/data/fan/fan_const_rpm_legacy.h5"

# Monitor memory during load
initial_mem = monitor_memory()
print(f"Initial memory: {initial_mem:.1f} MB")
print()

# Load with timing
t_start = time.perf_counter()
events = evlib.load_events(file_path)
t_load = (time.perf_counter() - t_start) * 1000
mem_after_load = monitor_memory()

print(f"⏱️  evlib.load_events(): {t_load:.1f}ms")
print(f"Memory after load: {mem_after_load:.1f} MB")
print(f"Memory increase: {mem_after_load - initial_mem:.1f} MB")
print()

# Check if it's a LazyFrame (should be)
print(f"Type: {type(events)}")
print()

# Count events (should trigger collection/streaming)
t_start = time.perf_counter()
count = events.select(pl.count()).collect()
t_count = (time.perf_counter() - t_start) * 1000
mem_after_count = monitor_memory()

print(f"Event count: {count.item():,}")
print(f"⏱️  Count time: {t_count:.1f}ms")
print(f"Memory after count: {mem_after_count:.1f} MB")
print(f"Memory increase: {mem_after_count - initial_mem:.1f} MB")
print()

# Check format detection
try:
    import evlib.formats
    format_info = evlib.formats.detect_format(file_path)
    print(f"Detected format: {format_info}")
except Exception as e:
    print(f"Could not detect format: {e}")
print()

# Expected behavior:
# - If streaming: Memory increase should be minimal (<100 MB)
# - If NOT streaming: Memory increase ~= file size (89 MB for this file)
total_mem_increase = mem_after_count - initial_mem
print(f"VERDICT:")
if total_mem_increase < 100:
    print(f"  ✅ Likely using streaming (memory increase {total_mem_increase:.1f} MB)")
else:
    print(f"  ❌ NOT streaming (memory increase {total_mem_increase:.1f} MB)")
