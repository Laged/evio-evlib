"""Helper to export legacy loader events to HDF5 for evlib parity testing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from test_evlib_comparison import MockRecording


def export_legacy_to_hdf5(recording, out_path: Path) -> dict[str, int]:
    """Export legacy Recording to HDF5 with evlib-compatible schema.

    Args:
        recording: Recording object from evio.core.recording.open_dat()
            or MockRecording for testing
        out_path: Output path for HDF5 file

    Returns:
        Dict with keys:
            - event_count: total events
            - t_min, t_max: timestamp range (microseconds)
            - x_min, x_max, y_min, y_max: spatial bounds
            - p_count_neg, p_count_pos: polarity distribution

    HDF5 Schema (evlib-compatible):
        /events/t        : int64 timestamps in microseconds
        /events/x        : uint16 x coordinates
        /events/y        : uint16 y coordinates
        /events/p        : int8 polarity {0, 1} (evlib converts to -1/+1)

        file.attrs['width']  : int
        file.attrs['height'] : int
        file.attrs['source'] : str = "legacy_dat"
    """
    # Decode packed event_words into x, y, polarity
    # See: evio/src/evio/core/mmap.py:151-154
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    event_words = recording.event_words

    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)

    # Keep polarity as 0/1 - evlib will convert to -1/+1 internally
    polarity = (raw_polarity > 0).astype(np.int8)

    # Timestamps are already sorted and in microseconds
    timestamps = recording.timestamps

    # Compute stats before writing
    stats = {
        'event_count': len(timestamps),
        't_min': int(timestamps.min()),
        't_max': int(timestamps.max()),
        'x_min': int(x.min()),
        'x_max': int(x.max()),
        'y_min': int(y.min()),
        'y_max': int(y.max()),
        'p_count_neg': int((polarity == 0).sum()),
        'p_count_pos': int((polarity == 1).sum()),
    }

    # Write to HDF5
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunking parameters for optimal lazy loading performance
    # - 100K events/chunk = ~800 KB (timestamps) - within 10 KiB - 1 MiB sweet spot
    # - Enables true lazy loading in evlib (load_events() reads only metadata)
    # - Light compression (gzip level 1) for better I/O without CPU overhead
    CHUNK_SIZE = 100_000

    with h5py.File(out_path, 'w') as f:
        # Create chunked datasets (evlib expects /events/t, /events/x, /events/y, /events/p)
        # Chunking + compression = fast lazy loading
        f.create_dataset('events/t', data=timestamps, dtype='int64',
                         chunks=(CHUNK_SIZE,), compression='gzip', compression_opts=1)
        f.create_dataset('events/x', data=x, dtype='uint16',
                         chunks=(CHUNK_SIZE,), compression='gzip', compression_opts=1)
        f.create_dataset('events/y', data=y, dtype='uint16',
                         chunks=(CHUNK_SIZE,), compression='gzip', compression_opts=1)
        f.create_dataset('events/p', data=polarity, dtype='int8',
                         chunks=(CHUNK_SIZE,), compression='gzip', compression_opts=1)

        # Write metadata at file level
        f.attrs['width'] = recording.width
        f.attrs['height'] = recording.height
        f.attrs['source'] = 'legacy_dat'

    return stats
