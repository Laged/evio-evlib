from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

import evlib


@dataclass
class VerificationResult:
    path: Path
    event_count: int
    t_min: int
    t_max: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    sample: Optional[pl.DataFrame] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a .dat file conforms to EVT3 by loading it via evlib."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the .dat (or .raw/.h5) file to verify",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of rows to materialize as a sanity check (default: 1000). "
        "Set to 0 to skip sampling.",
    )
    return parser.parse_args()


def collect_stats(lazy_events: pl.LazyFrame) -> dict[str, int]:
    """Fetch aggregate statistics from the LazyFrame."""
    # Get schema to check timestamp dtype
    schema = lazy_events.collect_schema()
    t_dtype = schema["t"]

    # Handle both Duration and Int64 timestamp types
    if isinstance(t_dtype, pl.Duration):
        # Duration type - convert to microseconds
        t_min_expr = pl.col("t").dt.total_microseconds().min()
        t_max_expr = pl.col("t").dt.total_microseconds().max()
    else:
        # Integer type (Int64/Int32) - use directly
        t_min_expr = pl.col("t").min()
        t_max_expr = pl.col("t").max()

    stats = (
        lazy_events.select(
            pl.len().alias("events"),
            t_min_expr.alias("t_min"),
            t_max_expr.alias("t_max"),
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
        )
        .collect()
        .to_dicts()[0]
    )
    # All stats should be integers even though Polars may store them as numpy types.
    return {key: int(value) for key, value in stats.items()}


def fetch_sample(lazy_events: pl.LazyFrame, rows: int) -> Optional[pl.DataFrame]:
    """Fetch a limited number of rows to ensure parsing works."""
    if rows <= 0:
        return None
    # Use limit() to only scan the requested rows (avoids loading entire dataset)
    return lazy_events.limit(rows).collect()


def verify_file(path: Path, sample_rows: int) -> VerificationResult:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    lazy_events = evlib.load_events(str(path))
    stats = collect_stats(lazy_events)
    sample = fetch_sample(lazy_events, sample_rows)
    return VerificationResult(
        path=path,
        event_count=stats["events"],
        t_min=stats["t_min"],
        t_max=stats["t_max"],
        x_min=stats["x_min"],
        x_max=stats["x_max"],
        y_min=stats["y_min"],
        y_max=stats["y_max"],
        sample=sample,
    )


def print_success(result: VerificationResult) -> None:
    print("✅ EVT3 verification passed")
    print(f"File      : {result.path}")
    print(f"Events    : {result.event_count}")
    print(
        f"Time span : {result.t_min} → {result.t_max} (Δ {result.t_max - result.t_min})"
    )
    print(f"X range   : {result.x_min} → {result.x_max}")
    print(f"Y range   : {result.y_min} → {result.y_max}")

    if result.sample is not None:
        print("")
        print("Sample rows:")
        print(result.sample)


def main() -> int:
    args = parse_args()
    try:
        result = verify_file(args.path, args.rows)
    except Exception as err:
        print("❌ Verification failed")
        print(f"Reason: {err}")
        print(
            "Hint: ensure the file was generated via the EVT3 conversion workflow "
            "(see docs/evlib-integration.md)."
        )
        return 1

    print_success(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
