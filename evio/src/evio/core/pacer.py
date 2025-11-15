import time
from collections.abc import Iterable, Iterator

from evio.source.dat_file import BatchRange


class Pacer:
    __slots__ = (
        "_avg_drop_rate",
        "_dropped_batches",
        "_dropped_pending",
        "_e_start",
        "_emitted_batches",
        "_inst_drop_rate",
        "_last_emit_ts_us",
        "_t_start",
        "drop_tolerance_s",
        "force_speed",
        "speed",
    )

    def __init__(
        self,
        speed: float,
        force_speed: bool,
        drop_tolerance_s: float = 0.0,
    ) -> None:
        self.speed = max(float(speed), 1e-9)
        self.force_speed = force_speed
        self.drop_tolerance_s = max(0.0, float(drop_tolerance_s))
        self._t_start: float | None = None  # wall-clock anchor
        self._e_start: int | None = None  # event-time anchor (µs)

        # Stats
        self._dropped_batches = 0
        self._emitted_batches = 0
        self._dropped_pending = 0
        self._last_emit_ts_us: int | None = None

        # Computed rates (drops / sec of event-time)
        self._inst_drop_rate = 0.0
        self._avg_drop_rate = 0.0

    # --- new helper API (optional but handy) ---
    @property
    def dropped_batches(self) -> int:
        """Number of batches skipped because they were overdue."""
        return self._dropped_batches

    @property
    def emitted_batches(self) -> int:
        """Number of batches actually yielded downstream."""
        return self._emitted_batches

    @property
    def instantaneous_drop_rate(self) -> float:
        """Drop rate for the most recently emitted window (drops / ms)."""
        return self._inst_drop_rate

    @property
    def average_drop_rate(self) -> float:
        """Mean drop rate over all emitted windows (drops / ms)."""
        return self._avg_drop_rate

    def reset_stats(self) -> None:
        self._dropped_batches = 0
        self._emitted_batches = 0
        self._dropped_pending = 0
        self._last_emit_ts_us = None
        self._inst_drop_rate = 0.0
        self._avg_drop_rate = 0.0

    # --- existing internals ---
    def _target_wall_s(self, end_ts_us: int) -> float:
        # event elapsed (s) scaled by speed factor
        return max(0.0, (end_ts_us - (self._e_start or 0)) / 1e6) / self.speed

    def _sleep_until(self, target_rel_s: float) -> None:
        # Sleep in chunks so UI can be pumped and OS timers stay accurate
        while True:
            now_s = time.perf_counter() - (self._t_start or time.perf_counter())
            dt = target_rel_s - now_s
            if dt <= 0:
                break
            time.sleep(min(0.05, dt))

    def _snapshot_and_yield(
        self, batch_range: BatchRange
    ) -> Iterator[BatchRange]:
        """Emit batch and compute instantaneous + average drop rates."""
        self._emitted_batches += 1

        # Compute drop rate (drops per second of event time)
        if self._last_emit_ts_us is not None:
            dt_ms = max(
                (batch_range.end_ts_us - self._last_emit_ts_us) / 1e3, 1e-9
            )
            self._inst_drop_rate = self._dropped_pending / dt_ms
        else:
            self._inst_drop_rate = 0.0

        # Update running average
        n = self._emitted_batches
        self._avg_drop_rate += (self._inst_drop_rate - self._avg_drop_rate) / n

        # Reset per-window accumulator and update last timestamp
        self._dropped_pending = 0
        self._last_emit_ts_us = batch_range.end_ts_us

        yield batch_range

    def pace(self, batches: Iterable[BatchRange]) -> Iterator[BatchRange]:
        iterator = iter(batches)
        try:
            batch_range = next(iterator)
        except StopIteration:
            return

        if self._t_start is None:
            self._t_start = time.perf_counter()
            self._e_start = batch_range.start_ts_us
            self._last_emit_ts_us = batch_range.start_ts_us

        while True:
            target_s = self._target_wall_s(batch_range.end_ts_us)
            now_s = time.perf_counter() - self._t_start
            lag_s = now_s - target_s
            if lag_s > self.drop_tolerance_s and self.force_speed:
                # Drop overdue batch and continue catching up
                self._dropped_batches += 1  # <-- increment here
                self._dropped_pending += 1

                try:
                    batch_range = next(iterator)
                    continue
                except StopIteration:
                    yield from self._snapshot_and_yield(batch_range)
            else:
                # We're early: block until target_s (handles long windows)
                if target_s > now_s:
                    self._sleep_until(target_s)
                yield from self._snapshot_and_yield(batch_range)

            try:
                batch_range = next(iterator)
            except StopIteration:
                return
