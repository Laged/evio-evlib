# evio/core/index_scheduler.py
import numpy as np

from .recording import Recording


def build_windows(recording: Recording, window_duration_us: int) -> np.ndarray:
    """
    Return an (N, 2) int32 array of [start, stop) indices into the
    *time-ordered* stream,
    using fixed-duration time slicing (no event-count logic).

    Downstream usage:
        for start, stop in build_windows(rec, 10_000):  # 10 ms
            idx = rec.order[start:stop]
            words = rec.w32[idx].astype(np.uint32, copy=False)
            ...
    """
    # Get timestamps and align to time order if available
    timestamps = recording.timestamps
    if timestamps is None:
        msg = "Recording is missing 'timestamps' array."
        raise AttributeError(msg)

    n_events = timestamps.shape[0]
    if n_events == 0:
        return np.zeros((0, 2), dtype=np.int32)

    dt = max(1, window_duration_us)
    t_start, t_end = timestamps[0], timestamps[-1]

    # Boundaries cover the entire range, including the last partial window
    boundaries = np.arange(t_start, t_end + dt, dt, dtype=np.int64)
    starts = np.searchsorted(timestamps, boundaries[:-1], side="left").astype(
        np.int32
    )
    stops = np.searchsorted(timestamps, boundaries[1:], side="left").astype(
        np.int32
    )

    # Drop empty windows but keep at least one if there are events
    non_empty = stops > starts
    if not np.any(non_empty):
        return np.array([[0, n_events]], dtype=np.int32)

    return np.stack([starts[non_empty], stops[non_empty]], axis=1)
