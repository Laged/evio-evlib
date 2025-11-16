#!/usr/bin/env python3
"""Test --enable-cache flag performance."""

import time
import sys
from pathlib import Path

# Import MVPLauncher
sys.path.insert(0, str(Path(__file__).parent / "evio/scripts"))
from mvp_launcher import MVPLauncher

print("=" * 60)
print("  Caching Performance Test")
print("=" * 60)
print()

# Test with caching enabled
print("Testing WITH caching enabled:")
print("-" * 60)
launcher = MVPLauncher(enable_cache=True)

if not launcher.datasets:
    print("Error: No datasets found. Run convert-all-legacy-to-hdf5 first.")
    sys.exit(1)

# Select first dataset
test_dataset = launcher.datasets[0]
print(f"Test dataset: {test_dataset.name} ({test_dataset.size_mb:.1f} MB)")
print()

# First load (should be slow)
print("First load (cold):")
t_start = time.perf_counter()
playback_state = launcher._init_playback(test_dataset)
t_first = (time.perf_counter() - t_start) * 1000
print(f"Total time: {t_first:.1f}ms")
print()

# Simulate returning to menu (reset playback state)
launcher.playback_state = None

# Second load (should be instant with cache)
print("Second load (cached):")
t_start = time.perf_counter()
playback_state = launcher._init_playback(test_dataset)
t_second = (time.perf_counter() - t_start) * 1000
print(f"Total time: {t_second:.1f}ms")
print()

# Summary
print("=" * 60)
print("  Results")
print("=" * 60)
print(f"First load:  {t_first:.1f}ms")
print(f"Second load: {t_second:.1f}ms")
speedup = t_first / t_second if t_second > 0 else float('inf')
print(f"Speedup:     {speedup:.1f}x")
print()

if t_second < 500:
    print("✅ SUCCESS: Cache working! Second load < 500ms")
else:
    print(f"⚠️  Cache might not be working. Expected < 500ms, got {t_second:.1f}ms")
