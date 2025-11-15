from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import evlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _timestamp_column(lazy_events: pl.LazyFrame) -> tuple[pl.Expr, bool]:
    """Return expression yielding integer microseconds and whether conversion needed."""
    t_dtype = lazy_events.collect_schema()["t"]
    if isinstance(t_dtype, pl.Duration):
        return pl.col("t").dt.total_microseconds(), True
    return pl.col("t"), False


def load_subset(
    path: Path,
    limit_events: Optional[int],
    duration_ms: Optional[float],
) -> pl.DataFrame:
    """Load a manageable subset of events for quick inspection."""
    lazy = evlib.load_events(str(path))
    t_expr, inserted = _timestamp_column(lazy)
    lf = lazy.with_columns(t_expr.alias("_t_us"))

    if duration_ms is not None:
        stats = lf.select(
            pl.min("_t_us").alias("t_min"),
        ).collect()
        t_min = int(stats[0, "t_min"])
        window_us = int(duration_ms * 1_000)
        lf = lf.filter(
            (pl.col("_t_us") >= t_min)
            & (pl.col("_t_us") < t_min + window_us)
        )

    if limit_events is not None:
        lf = lf.limit(limit_events)

    df = lf.collect()
    if not inserted:
        df = df.with_columns(pl.col("_t_us").cast(pl.Int64))
    return df


def summarize(df: pl.DataFrame) -> dict[str, int]:
    """Compute basic stats for display."""
    stats = df.select(
        pl.len().alias("events"),
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("y").min().alias("y_min"),
        pl.col("y").max().alias("y_max"),
        pl.col("_t_us").min().alias("t_min"),
        pl.col("_t_us").max().alias("t_max"),
        (pl.col("polarity") == 1).sum().alias("on_events"),
        (pl.col("polarity") == -1).sum().alias("off_events"),
    ).to_dicts()[0]
    return {k: int(v) for k, v in stats.items()}


def render_frame(
    df: pl.DataFrame,
    width: int,
    height: int,
    output: Path,
) -> None:
    """Save a quick intensity map of event density."""
    if df.is_empty():
        print("No events to render; skipping frame export.")
        return

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    polarity = df["polarity"].to_numpy()
    canvas = np.zeros((height, width), dtype=np.float32)
    np.add.at(canvas, (y, x), 1.0)

    # Normalize for visualization
    if canvas.max() > 0:
        canvas = canvas / canvas.max()

    plt.figure(figsize=(6, 4))
    plt.imshow(canvas, cmap="viridis", origin="lower")
    plt.title("Event density (polarity agnostic)")
    plt.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()

    # Also export a polarity scatter for quick inspection
    plt.figure(figsize=(6, 4))
    on_mask = polarity > 0
    plt.scatter(x[~on_mask], y[~on_mask], s=1, c="blue", label="OFF")
    plt.scatter(x[on_mask], y[on_mask], s=1, c="red", label="ON")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Polarity scatter (sample subset)")
    scatter_path = output.with_name(output.stem + "_scatter.png")
    plt.savefig(scatter_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output} and scatter to {scatter_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and visualize EVT3 .raw/.dat files with evlib."
    )
    parser.add_argument("path", type=Path, help="Path to EVT3 file (.raw/.dat)")
    parser.add_argument(
        "--limit-events",
        type=int,
        default=200_000,
        help="Maximum number of events to materialize (default: 200k).",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=50.0,
        help="Temporal window starting at the first event (default: 50 ms).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/raw_demo.png"),
        help="Destination PNG for the density heatmap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset = load_subset(args.path, args.limit_events, args.duration_ms)
    if subset.is_empty():
        print("No events collected – check duration/limit parameters.")
        return 1

    stats = summarize(subset)
    width = stats["x_max"] + 1
    height = stats["y_max"] + 1

    print("=== evlib raw demo ===")
    print(f"File        : {args.path}")
    print(f"Events      : {stats['events']:,}")
    print(f"Resolution  : {width} x {height}")
    print(
        f"Duration    : {stats['t_min']} → {stats['t_max']} "
        f"(Δ {(stats['t_max'] - stats['t_min']) / 1_000:.2f} ms)"
    )
    print(
        f"Polarity    : ON={stats['on_events']:,}  "
        f"OFF={stats['off_events']:,}"
    )

    render_frame(subset, width, height, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
