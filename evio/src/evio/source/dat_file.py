from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from evio.core.index_scheduler import build_windows
from evio.core.recording import open_dat


class BatchRange(NamedTuple):
    """Lightweight slice into the time-ordered event stream."""

    start: int  # inclusive index into time_ordered stream
    stop: int  # exclusive index into time_ordered stream
    start_ts_us: int  # best-effort start timestamp (microseconds)
    end_ts_us: int  # best-effort end timestamp (microseconds)


class DatFileSource:
    """
    Minimal file-backed source for event batches.

    - Opens a .dat file with fixed geometry (defaults to 1280x720).
    - Precomputes non-overlapping windows.
    - Exposes:
        * words_u32 : np.ndarray  (packed uint32 words)
        * order     : np.ndarray  (time-sorted indices)
        * width, height : ints
        * ranges()  : iterator of BatchRange
    - Does NOT decode or sleep; decoding happens downstream in the render path.
    """

    def __init__(
        self,
        path: str,
        window_length_us: int = 1000,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        rec = open_dat(path, width=width, height=height)
        self._event_words = rec.event_words
        self._order = rec.order
        self._width, self._height = int(rec.width), int(rec.height)

        t_raw = getattr(rec, "timestamps", getattr(rec, "ts_us", None))
        if t_raw is None:
            msg = "Recording missing timestamps ('timestamps' or 'ts_us')."
            raise AttributeError(msg)
        t_raw = np.asarray(t_raw)
        if t_raw.shape[0] != self._event_words.shape[0]:
            msg = "timestamps length must match event count"
            raise ValueError(msg)

        # time-sorted timestamps to match index space of build_windows
        t_sorted = t_raw[self._order].astype(np.int64, copy=False)

        # windows are [start, stop) in time-ordered index space
        win_idx = build_windows(rec, window_length_us)

        self._ranges = [
            BatchRange(int(s), int(e), int(t_sorted[s]), int(t_sorted[e - 1]))
            for s, e in win_idx
            if e > s
        ]

    # --- Public surface ----------------------------------------------------

    @property
    def event_words(self) -> np.ndarray:
        """Packed event words (uint32)."""
        return self._event_words

    @property
    def order(self) -> np.ndarray:
        """Time-sorted indices into the raw event arrays."""
        return self._order

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def ranges(self) -> Iterator[BatchRange]:
        """Iterate precomputed BatchRange slices."""
        return iter(self._ranges)

    def __len__(self) -> int:
        """Number of precomputed ranges."""
        return len(self._ranges)
