"""Regression test: verify parameters match example-drone-original.py exactly."""

import pytest


def test_default_parameters_match_original():
    """Verify default parameters match example-drone-original.py."""
    import inspect
    from drone_detector_demo.geometry import propeller_mask_from_frame

    sig = inspect.signature(propeller_mask_from_frame)

    # Verified against example-drone-original.py
    assert sig.parameters['pre_threshold'].default == 250, \
        "pre_threshold must be 250 (original line 192, not 50)"
    assert sig.parameters['min_area'].default == 145.0, \
        "min_area must be 145.0 (original line 171, not 100.0)"
    assert sig.parameters['hot_pixel_threshold'].default == float('inf'), \
        "hot_pixel_threshold must be inf (original has no filtering)"

    # Verify min_points via source inspection
    import drone_detector_demo.geometry as geom
    source = inspect.getsource(geom.propeller_mask_from_frame)
    assert 'len(cnt) < 50' in source, \
        "Must require 50 points for fitEllipse (original line 212, not 30)"


def test_text_labels_match_original():
    """Verify HUD text labels match example-drone-original.py."""
    import drone_detector_demo.main as main_module

    with open(main_module.__file__, 'r') as f:
        content = f.read()

    # Verified against example-drone-original.py lines 539, 562
    assert '"RPM:' in content, \
        "Must use 'RPM:' label (original line 539, not 'Frame mean')"
    assert '"Avg RPM:' in content, \
        "Must use 'Avg RPM:' label (original line 562, not 'Global mean')"


def test_scaling_matches_original():
    """Verify 0.7x scaling is preserved (matches example-drone-original.py line 587)."""
    import drone_detector_demo.main as main_module

    with open(main_module.__file__, 'r') as f:
        content = f.read()

    # Original DOES use 0.7x scaling (verified line 587)
    assert 'scale = 0.7' in content, \
        "Must preserve 0.7x scaling (matches original line 587)"
