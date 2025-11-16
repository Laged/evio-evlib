#!/usr/bin/env python3
"""Simple load time test."""

import time
import evlib
import polars as pl

file_path = "evio/data/fan/fan_const_rpm_legacy.h5"

print("Load timing test:")
print()

# Test 1: Just load_events
t = time.perf_counter()
events = evlib.load_events(file_path)
t_load = (time.perf_counter() - t) * 1000
print(f"1. evlib.load_events(): {t_load:.1f}ms")

# Test 2: Count events (minimal operation)
t = time.perf_counter()
count = len(events.collect())
t_count = (time.perf_counter() - t) * 1000
print(f"2. Count all events: {t_count:.1f}ms ({count:,} events)")

# Test 3: Load again (cached?)
t = time.perf_counter()
events2 = evlib.load_events(file_path)
t_load2 = (time.perf_counter() - t) * 1000
print(f"3. Second load_events(): {t_load2:.1f}ms")

print()
print("VERDICT:")
if t_load < 500:
    print(f"  ✅ Fast lazy loading working ({t_load:.1f}ms)")
elif t_count > t_load * 2:
    print(f"  ⚠️  Load is lazy but slow ({t_load:.1f}ms)")
    print(f"     Collecting is slower ({t_count:.1f}ms)")
else:
    print(f"  ❌ Load is NOT lazy ({t_load:.1f}ms ~= file scan time)")
