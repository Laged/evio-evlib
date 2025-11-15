"""
RPM detection using evlib-accelerated voxel grid + FFT.

This is MVP-2 reimplemented with evlib for 100x speedup.
Algorithm remains the same (our innovation preserved).
"""

from typing import Optional
import numpy as np
import polars as pl

from evio.representations import create_voxel_grid, voxel_to_numpy


class RPMDetector:
    """
    Detect RPM using voxel grid FFT analysis.

    Algorithm:
    1. Create voxel grid (evlib - 100x faster)
    2. Sum events across spatial dimensions → temporal signal
    3. FFT to find dominant frequency
    4. Convert frequency to RPM using num_blades
    """

    def __init__(
        self,
        height: int,
        width: int,
        n_time_bins: int = 50,
        num_blades: int = 4,
    ):
        self.height = height
        self.width = width
        self.n_time_bins = n_time_bins
        self.num_blades = num_blades

    def detect_rpm(
        self,
        events: pl.LazyFrame,
        window_duration_ms: Optional[float] = None,
    ) -> float:
        """
        Detect RPM from event stream.

        Args:
            events: Polars LazyFrame with event data
            window_duration_ms: Analysis window (auto-computed if None)

        Returns:
            Detected RPM (revolutions per minute)
        """
        # Create voxel grid (evlib - fast!)
        voxel = create_voxel_grid(
            events,
            height=self.height,
            width=self.width,
            n_time_bins=self.n_time_bins,
        )

        # Convert to NumPy for FFT
        voxel_array = voxel_to_numpy(
            voxel,
            self.n_time_bins,
            self.height,
            self.width,
        )

        # Extract temporal signal
        temporal_signal = create_temporal_signal(voxel_array)

        # Compute window duration if not provided
        if window_duration_ms is None:
            events_df = events.collect(engine='in-memory')
            t_min = events_df['t'].min()
            t_max = events_df['t'].max()
            window_duration_ms = (t_max - t_min) / 1000.0

        # Detect dominant frequency
        blade_freq_hz = detect_dominant_frequency(
            temporal_signal,
            window_duration_ms,
        )

        # Convert to RPM
        rotation_freq_hz = blade_freq_hz / self.num_blades
        rpm = rotation_freq_hz * 60.0

        return rpm


def create_temporal_signal(voxel_array: np.ndarray) -> np.ndarray:
    """
    Sum voxel grid across spatial dimensions to get temporal signal.

    Args:
        voxel_array: Shape (n_time_bins, height, width)

    Returns:
        1D array of shape (n_time_bins,) with event counts per bin
    """
    temporal_signal = voxel_array.sum(axis=(1, 2))
    return temporal_signal


def detect_dominant_frequency(
    signal: np.ndarray,
    duration_ms: float,
) -> float:
    """
    Find dominant frequency using FFT.

    Args:
        signal: 1D temporal signal
        duration_ms: Duration of signal in milliseconds

    Returns:
        Dominant frequency in Hz
    """
    # FFT
    fft_result = np.fft.fft(signal)

    # Frequency bins
    n_bins = len(signal)
    duration_sec = duration_ms / 1000.0
    freqs = np.fft.fftfreq(n_bins, d=duration_sec / n_bins)

    # Find peak in positive frequencies
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_fft = np.abs(fft_result[positive_mask])

    peak_idx = np.argmax(positive_fft)
    dominant_freq = positive_freqs[peak_idx]

    return dominant_freq
