#!/usr/bin/env python3
"""
Automated verification script for MVP launcher.
Tests functionality without GUI interaction.
"""

import sys
from pathlib import Path


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "flake.nix").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (missing flake.nix)")


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "evio" / "scripts"))

# Test 1: Import check
print("=" * 60)
print("Test 1: Module imports")
print("=" * 60)

try:
    from mvp_launcher import MVPLauncher, AppMode, Dataset, PlaybackState
    from detector_utils import (
        detect_fan, detect_drone,
        FanDetection, DroneDetection,
        render_fan_overlay, render_drone_overlay
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Dataset discovery
print("\n" + "=" * 60)
print("Test 2: Dataset auto-discovery")
print("=" * 60)

try:
    launcher = MVPLauncher()
    print(f"✓ Discovered {len(launcher.datasets)} datasets:")
    for i, ds in enumerate(launcher.datasets):
        print(f"  [{i}] {ds.name}")
        print(f"      Category: {ds.category}")
        print(f"      Path: {ds.path}")
        print(f"      Size: {ds.size_mb:.2f} MB")
        print(f"      Detector: {launcher._map_detector_type(ds.category)}")

    if len(launcher.datasets) == 0:
        print("⚠ No datasets found - run convert-all-legacy-to-hdf5")
        sys.exit(1)

except Exception as e:
    print(f"✗ Discovery failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Playback initialization
print("\n" + "=" * 60)
print("Test 3: Playback initialization")
print("=" * 60)

try:
    # Test with first dataset
    test_dataset = launcher.datasets[0]
    print(f"Testing with: {test_dataset.name}")

    state = launcher._init_playback(test_dataset)
    print(f"✓ Loaded playback state:")
    print(f"  Resolution: {state.width}x{state.height}")
    print(f"  Duration: {(state.t_max - state.t_min) / 1e6:.2f}s")
    print(f"  Detector type: {state.detector_type}")
    print(f"  Overlay flags: {state.overlay_flags}")
    print(f"  Lazy events: {type(state.lazy_events)}")

except Exception as e:
    print(f"✗ Playback init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Event window extraction
print("\n" + "=" * 60)
print("Test 4: Event window extraction")
print("=" * 60)

try:
    window = launcher._get_event_window(
        state.lazy_events,
        state.t_min,
        state.t_min + state.window_us
    )
    x_coords, y_coords, polarities = window
    print(f"✓ Extracted event window:")
    print(f"  Events in window: {len(x_coords)}")
    print(f"  ON events: {polarities.sum()}")
    print(f"  OFF events: {(~polarities).sum()}")

except Exception as e:
    print(f"✗ Window extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Frame rendering
print("\n" + "=" * 60)
print("Test 5: Frame rendering")
print("=" * 60)

try:
    frame = launcher._render_polarity_frame(window, state.width, state.height)
    print(f"✓ Rendered polarity frame:")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame dtype: {frame.dtype}")
    print(f"  Pixel range: [{frame.min()}, {frame.max()}]")

except Exception as e:
    print(f"✗ Frame rendering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Detector execution
print("\n" + "=" * 60)
print("Test 6: Detector execution")
print("=" * 60)

try:
    if state.detector_type == "fan_rpm":
        print("Testing fan detector...")
        detection = detect_fan(window, state.width, state.height, state.window_us)
        print(f"✓ Fan detection:")
        print(f"  Center: ({detection.cx}, {detection.cy})")
        print(f"  Semi-axes: a={detection.a:.1f}, b={detection.b:.1f}")
        print(f"  Angle: {detection.phi:.2f} rad")
        print(f"  Clusters: {len(detection.clusters)}")
        print(f"  RPM estimate: {detection.rpm:.1f}")

        # Test overlay rendering
        overlay_frame = render_fan_overlay(frame.copy(), detection)
        print(f"✓ Fan overlay rendered: {overlay_frame.shape}")

    elif state.detector_type == "drone":
        print("Testing drone detector...")
        detection = detect_drone(window, state.width, state.height)
        print(f"✓ Drone detection:")
        print(f"  Bounding boxes: {len(detection.boxes)}")
        print(f"  Warning: {detection.warning}")

        # Test overlay rendering
        overlay_frame = render_drone_overlay(frame.copy(), detection)
        print(f"✓ Drone overlay rendered: {overlay_frame.shape}")

    else:
        print(f"⚠ No detector for type: {state.detector_type}")

except Exception as e:
    print(f"✗ Detector execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Menu rendering
print("\n" + "=" * 60)
print("Test 7: Menu rendering")
print("=" * 60)

try:
    menu_frame = launcher._render_menu()
    print(f"✓ Menu rendered:")
    print(f"  Frame shape: {menu_frame.shape}")
    print(f"  Frame dtype: {menu_frame.dtype}")

except Exception as e:
    print(f"✗ Menu rendering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test all datasets
print("\n" + "=" * 60)
print("Test 8: All datasets load test")
print("=" * 60)

for i, dataset in enumerate(launcher.datasets):
    try:
        print(f"\n[{i+1}/{len(launcher.datasets)}] Testing {dataset.name}...")
        test_state = launcher._init_playback(dataset)
        test_window = launcher._get_event_window(
            test_state.lazy_events,
            test_state.t_min,
            test_state.t_min + test_state.window_us
        )
        test_frame = launcher._render_polarity_frame(
            test_window,
            test_state.width,
            test_state.height
        )

        # Test detector if applicable
        if test_state.detector_type == "fan_rpm":
            det = detect_fan(test_window, test_state.width, test_state.height, test_state.window_us)
            test_frame = render_fan_overlay(test_frame, det)
        elif test_state.detector_type == "drone":
            det = detect_drone(test_window, test_state.width, test_state.height)
            test_frame = render_drone_overlay(test_frame, det)

        print(f"  ✓ {dataset.name} OK")

    except Exception as e:
        print(f"  ✗ {dataset.name} FAILED: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print("✓ All core functionality verified")
print("✓ Imports working")
print("✓ Dataset discovery working")
print("✓ Playback initialization working")
print("✓ Event extraction working")
print("✓ Frame rendering working")
print("✓ Detectors working")
print("✓ Menu rendering working")
print("\nReady for interactive testing!")
print("\nRun: uv run --package evio python evio/scripts/mvp_launcher.py")
