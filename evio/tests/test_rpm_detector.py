"""Tests for RPM detection using evlib-accelerated voxel FFT."""

import pytest
import numpy as np
import polars as pl


def test_rpm_detector_basic():
    """Test basic RPM detection with synthetic periodic data."""
    from evio.mvp.rpm_detector import RPMDetector

    # Generate synthetic events: 4-blade fan at 600 RPM (10 Hz)
    # Blade frequency = 10 Hz * 4 blades = 40 Hz
    duration_ms = 1000  # 1 second
    blade_freq_hz = 40

    # Create events at blade frequency
    t_values = []
    for i in range(int(duration_ms * blade_freq_hz)):
        t_values.append(int(i * (1000000 / blade_freq_hz)))  # microseconds

    events = pl.DataFrame({
        't': t_values,
        'x': [640] * len(t_values),  # Center pixel
        'y': [360] * len(t_values),
        'polarity': [1] * len(t_values),
    }).lazy()

    detector = RPMDetector(
        height=720,
        width=1280,
        n_time_bins=50,
        num_blades=4,
    )

    rpm = detector.detect_rpm(events)

    # Should detect ~600 RPM (within 10% tolerance)
    assert 540 <= rpm <= 660, f"Expected ~600 RPM, got {rpm}"


def test_voxel_fft_pipeline():
    """Test the voxel grid → FFT → RPM pipeline."""
    from evio.mvp.rpm_detector import create_temporal_signal, detect_dominant_frequency
    from evio.representations import create_voxel_grid, voxel_to_numpy

    # Create sample periodic events
    freq_hz = 25  # 25 Hz signal
    duration_ms = 1000

    t_values = []
    for i in range(freq_hz * 10):  # 10 events per cycle
        t_values.append(int(i * (1000000 / (freq_hz * 10))))

    events = pl.DataFrame({
        't': t_values,
        'x': [100] * len(t_values),
        'y': [50] * len(t_values),
        'polarity': [1] * len(t_values),
    }).lazy()

    # Create voxel grid
    voxel = create_voxel_grid(events, height=720, width=1280, n_time_bins=50)
    voxel_array = voxel_to_numpy(voxel, 50, 720, 1280)

    # Extract temporal signal
    signal = create_temporal_signal(voxel_array)
    assert len(signal) == 50

    # Detect frequency
    freq = detect_dominant_frequency(signal, duration_ms)

    # Should be close to 25 Hz (within 20% for short signal)
    assert 20 <= freq <= 30
