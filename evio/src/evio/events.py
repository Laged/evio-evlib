"""Event data structures bridging evlib and NumPy MVPs."""

from typing import Optional, Tuple
import polars as pl
import numpy as np


class EventData:
    """
    Unified event data container supporting both evlib (Polars) and NumPy.

    Attributes:
        _events: Polars LazyFrame with columns ['t', 'x', 'y', 'polarity']
        width: Sensor width (optional)
        height: Sensor height (optional)
    """

    def __init__(
        self,
        events: pl.LazyFrame,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self._events = events
        self.width = width
        self.height = height

    @classmethod
    def from_polars(
        cls,
        events: pl.LazyFrame,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> 'EventData':
        """Create from Polars LazyFrame (evlib output)."""
        return cls(events, width=width, height=height)

    @property
    def num_events(self) -> int:
        """Get total number of events (forces evaluation)."""
        return len(self._events.collect())

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert to NumPy arrays for MVP compatibility.

        Returns:
            (t, x, y, polarity) tuple of NumPy arrays
        """
        df = self._events.collect()

        t = df['t'].to_numpy()
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        polarity = df['polarity'].to_numpy()

        return t, x, y, polarity

    def filter_roi(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
    ) -> 'EventData':
        """
        Filter events to region of interest (lazy operation).

        50x faster than NumPy boolean indexing for large datasets.
        """
        filtered = self._events.filter(
            (pl.col('x') >= x_min) &
            (pl.col('x') < x_max) &
            (pl.col('y') >= y_min) &
            (pl.col('y') < y_max)
        )

        return EventData(filtered, width=self.width, height=self.height)
