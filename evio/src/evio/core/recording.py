from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Recording:
    width: int
    height: int
    timestamps: np.ndarray  # int64, sorted monotonic, [microseconds]
    event_words: np.ndarray  # each word is uint32 memmap view of
    # packed XY+polarity
    order: np.ndarray  # int32 permutation mapping sort order


def open_dat(
    path: str | Path,
    *,
    width: int,
    height: int,
    data_offset: int | None = None,
) -> Recording:
    """Minimal loader: memmap [(t32,u4),(w32,u4)], build stable time order.

    If data_offset is None, tries to auto-scan header lines starting with '%'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # --- header / offset ---
    off = 0
    if data_offset is None:
        with Path.open(path, "rb") as file:
            while True:
                pos = file.tell()
                first_byte = file.read(1)
                if not first_byte or first_byte != b"%":
                    file.seek(pos)
                    break
                file.readline()  # consume the header line
            # two bytes: type + decoded size
            type_b = file.read(1)
            size_b = file.read(1)
            if len(type_b) != 1 or len(size_b) != 1:
                msg = "Invalid DAT header"
                raise RuntimeError(msg)
            if int.from_bytes(size_b, "little") != 8:
                msg = "Only 8B CD events supported"
                raise RuntimeError(msg)
            off = file.tell()
    else:
        off = int(data_offset)

    DT8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])
    events = np.memmap(path, dtype=DT8, mode="r", offset=off)

    timestamps = events["t32"].astype(np.int64, copy=False)
    if timestamps.size == 0:
        msg = "No events"
        raise RuntimeError(msg)
    order = np.argsort(timestamps, kind="stable").astype(np.int32, copy=False)
    timestamps_sorted = timestamps[order]
    event_words = events["w32"]

    return Recording(
        width=width,
        height=height,
        timestamps=timestamps_sorted,
        event_words=event_words,
        order=order,
    )
