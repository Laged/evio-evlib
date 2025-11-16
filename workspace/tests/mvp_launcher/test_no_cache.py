#!/usr/bin/env python3
"""Test default mode (no caching)."""

import sys
import time
from pathlib import Path


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "flake.nix").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (missing flake.nix)")


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "evio" / "scripts"))

from mvp_launcher import MVPLauncher

print("=" * 60)
print("  Default Mode Test (NO Caching)")
print("=" * 60)
print()

# Test with caching disabled (default)
print("Testing WITHOUT caching (default):")
print("-" * 60)
launcher = MVPLauncher(enable_cache=False)

if not launcher.datasets:
    print("Error: No datasets found. Run convert-all-legacy-to-hdf5 first.")
    sys.exit(1)

# Select first dataset
test_dataset = launcher.datasets[0]
print(f"Test dataset: {test_dataset.name} ({test_dataset.size_mb:.1f} MB)")
print()

# First load
print("First load:")
t_start = time.perf_counter()
playback_state = launcher._init_playback(test_dataset)
t_first = (time.perf_counter() - t_start) * 1000
print(f"Total time: {t_first:.1f}ms")
print()

# Simulate returning to menu
launcher.playback_state = None

# Second load (should ALSO be slow, no caching)
print("Second load (no cache):")
t_start = time.perf_counter()
playback_state = launcher._init_playback(test_dataset)
t_second = (time.perf_counter() - t_start) * 1000
print(f"Total time: {t_second:.1f}ms")
print()

# Summary
print("=" * 60)
print("  Results (No Caching)")
print("=" * 60)
print(f"First load:  {t_first:.1f}ms")
print(f"Second load: {t_second:.1f}ms")
print()

if t_second > 1000:
    print("✅ Correct: Both loads are slow (no caching)")
else:
    print(f"⚠️  Unexpected: Second load was fast ({t_second:.1f}ms)")
    print("    Cache might be enabled by mistake")
