"""evlib integration layer for evio.

This module provides high-performance event loading using evlib (Rust-backed),
replacing the custom NumPy-based .dat loader with 10-200x faster implementation.
"""

from typing import Optional
import polars as pl


def load_events_with_evlib(
    path: str,
    format: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Load event camera data using evlib.

    Args:
        path: Path to event file (.dat, .h5, .aedat, etc.)
        format: Optional format hint ('dat', 'h5', 'aedat'). Currently unused as evlib
                auto-detects format from file extension. Reserved for future use to
                support explicit format override if needed.

    Returns:
        Polars LazyFrame with columns: ['t', 'x', 'y', 'polarity']
        - t: timestamp in microseconds (i64)
        - x: x coordinate (i16)
        - y: y coordinate (i16)
        - polarity: event polarity, 0 or 1 (i8)

    Raises:
        ImportError: If evlib is not installed
        FileNotFoundError: If path does not exist

    Note:
        The `format` parameter is currently unused. evlib's load_events function
        automatically detects the file format from the extension. This parameter
        is retained for API compatibility and future extensibility.
    """
    try:
        import evlib
    except ImportError as e:
        raise ImportError(
            "evlib is required but not installed. "
            "Install with: pip install evlib"
        ) from e

    # Load events - evlib.load_events auto-detects format from file extension
    # The format parameter is currently unused but reserved for future use
    events = evlib.load_events(path)

    return events
