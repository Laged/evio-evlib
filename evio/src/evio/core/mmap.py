import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# 8-byte record: t32<u4>, w32<u4>
_DTYPE_CD8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])


def _parse_header_and_data_offset(dat_path: Path) -> tuple[dict, int]:
    """Parse Prophesee/Metavision-style DAT header.

    The format:
      - ASCII header lines starting with '%'
      - Followed by 2 bytes: [event_type (1B), decoded_event_size (1B)]
      - Returns (metadata_dict, offset_to_binary_records)
    """
    metadata: dict = {}

    with dat_path.open("rb") as dat_file:
        # Read ASCII header lines (each starts with '%')
        while True:
            current_pos = dat_file.tell()
            first_byte = dat_file.read(1)
            if not first_byte:
                break
            if first_byte != b"%":
                dat_file.seek(current_pos)
                break

            line_bytes = b"%" + dat_file.readline()  # includes trailing newline
            line_text = line_bytes.decode("ascii", errors="ignore").strip()
            line_body = line_text.lstrip("%").strip()
            tokens = line_body.split()

            if tokens:
                key = tokens[0].lower()
                value = " ".join(tokens[1:])
                if key == "width":
                    with contextlib.suppress(Exception):
                        metadata["width"] = int(value.split()[0])
                elif key == "height":
                    with contextlib.suppress(Exception):
                        metadata["height"] = int(value.split()[0])
                elif key == "version":
                    with contextlib.suppress(Exception):
                        metadata["version"] = int(value.split()[0])
                elif key == "date":
                    metadata["date"] = value

        # Two bytes immediately after the header: event_type and event_size
        event_type_byte = dat_file.read(1)
        event_size_byte = dat_file.read(1)
        if len(event_type_byte) != 1 or len(event_size_byte) != 1:
            msg = "Invalid DAT: missing event type/size bytes."
            raise ValueError(msg)

        metadata["event_type"] = int.from_bytes(
            event_type_byte, "little"
        )  # usually 0 for CD
        metadata["event_size"] = int.from_bytes(
            event_size_byte, "little"
        )  # must be 8 for CD

        if metadata["event_size"] != 8:
            msg = (
                f"Unsupported DAT event size:"
                f" {metadata['event_size']} (expected 8)."
            )
            raise ValueError(msg)

        return metadata, dat_file.tell()


@dataclass(frozen=True)
class DatMemmap:
    """Memory-mapped reader for Prophesee/Metavision DAT recordings.

    Each event record is 8 bytes (64 bits):
        [63:32] = timestamp (uint32)
        [31:28] = polarity (4 bits)
        [27:14] = y (14 bits)
        [13:0]  = x (14 bits)
    """

    file_path: Path
    width: int
    height: int
    event_count: int

    # Decoded arrays (x/y/p come from bitfields, not byte-aligned views)
    _decoded_x: NDArray[np.uint16]
    _decoded_y: NDArray[np.uint16]
    _decoded_timestamps: NDArray[np.int64]
    _decoded_polarities: NDArray[np.int8]

    # ---- Public accessors ----------------------------------------------------

    @property
    def x_coords(self) -> NDArray[np.uint16]:
        """Return x coordinates for each event (decoded from bitfield)."""
        return self._decoded_x

    @property
    def y_coords(self) -> NDArray[np.uint16]:
        """Return y coordinates for each event (decoded from bitfield)."""
        return self._decoded_y

    @property
    def timestamps(self) -> NDArray[np.int64]:
        """Return timestamps as int64 (converted from 32-bit values)."""
        return self._decoded_timestamps

    @property
    def polarities(self) -> NDArray[np.int8]:
        """Return event polarities as int8 values (0 for OFF, 1 for ON)."""
        return self._decoded_polarities

    # ---- Factory method ------------------------------------------------------

    @classmethod
    def open(cls, dat_path: str | Path) -> "DatMemmap":
        """Open and memory-map a Prophesee/Metavision DAT recording."""
        dat_file = Path(dat_path)
        if not dat_file.exists():
            raise FileNotFoundError(dat_file)

        metadata, binary_offset = _parse_header_and_data_offset(dat_file)

        # Validate that the binary section is aligned to 8-byte event records
        file_size = dat_file.stat().st_size
        payload_bytes = file_size - binary_offset
        if payload_bytes < 0 or (payload_bytes % 8) != 0:
            msg = (
                f"Invalid DAT payload size {payload_bytes}"
                " (not a multiple of 8 bytes)."
            )
            raise ValueError(msg)

        # Memory-map the event records (structure: [t32<u4>, w32<u4>])
        raw_events = np.memmap(
            dat_file, dtype=_DTYPE_CD8, mode="r", offset=binary_offset
        )
        num_events = raw_events.shape[0]

        # Decode bit-packed fields
        packed_w32 = raw_events["w32"].astype(np.uint32, copy=False)

        # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
        decoded_x = (packed_w32 & 0x3FFF).astype(np.uint16, copy=False)
        decoded_y = ((packed_w32 >> 14) & 0x3FFF).astype(np.uint16, copy=False)
        raw_polarity = ((packed_w32 >> 28) & 0xF).astype(np.uint8, copy=False)
        decoded_polarity = (raw_polarity > 0).astype(np.int8, copy=False)

        # Convert timestamps to int64 for uniformity
        decoded_timestamps = raw_events["t32"].astype(np.int64, copy=False)

        # Geometry: use header values if present, otherwise infer from data
        width = int(
            metadata.get(
                "width", int(decoded_x.max()) + 1 if num_events > 0 else 0
            )
        )
        height = int(
            metadata.get(
                "height", int(decoded_y.max()) + 1 if num_events > 0 else 0
            )
        )

        return DatMemmap(
            file_path=dat_file,
            width=width,
            height=height,
            event_count=int(num_events),
            _decoded_x=decoded_x,
            _decoded_y=decoded_y,
            _decoded_timestamps=decoded_timestamps,
            _decoded_polarities=decoded_polarity,
        )
