"""OpenCV-based event playback with RVT detections."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np
import polars as pl

from evio.evlib_loader import load_events_with_evlib
from rvt_detector import RVTDetectorPlugin
from rvt_detector.env_check import validate_assets


@dataclass
class Window:
    start_us: int
    end_us: int
    data: pl.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RVT detector demo.")
    parser.add_argument("dat", nargs="?", default=None, help="Path to an event file (.dat/.aedat/.h5)")
    parser.add_argument("--window-ms", type=float, default=50.0, help="Window duration")
    parser.add_argument("--delay-ms", type=int, default=1, help="Delay passed to cv2.waitKey")
    parser.add_argument("--width", type=int, default=1280, help="Frame width used for visualization")
    parser.add_argument("--height", type=int, default=720, help="Frame height used for visualization")
    parser.add_argument("--dataset-name", default="gen1", help="Hydra dataset config (gen1/gen4)")
    parser.add_argument("--experiment", default="small", help="Hydra experiment config (small/base/tiny)")
    parser.add_argument("--rvt-repo", type=Path, default=None, help="Override path to the RVT repo")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override path to an RVT checkpoint")
    parser.add_argument("--device", default=None, help="Torch device string (default: auto)")
    parser.add_argument("--histogram-bins", type=int, default=10, help="Temporal bins for histogram tensor")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit the number of playback windows")
    parser.add_argument("--no-loop", action="store_true", help="Disable looping playback once file ends")
    parser.add_argument("--mock", action="store_true", help="Use synthetic events instead of reading a file")
    return parser.parse_args()


def _select_device(requested: Optional[str], report) -> str:
    """Resolve the actual device we will use for inference."""

    if requested in (None, "", "auto"):
        return report.device
    if requested.lower().startswith("cuda") and not report.cuda_available:
        print("[WARN] CUDA requested but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    return requested


def main() -> None:
    args = parse_args()
    if args.mock:
        print("Generating mock events ...")
    else:
        if args.dat is None:
            raise SystemExit("The dat argument is required unless --mock is specified.")
        print(f"Loading events from {args.dat} ...")
    report = validate_assets(args.rvt_repo, args.checkpoint)
    if report.errors:
        for err in report.errors:
            print(f"[ERROR] {err}")
        return
    for warn in report.warnings:
        print(f"[WARN] {warn}")

    selected_device = _select_device(args.device, report)
    print(f"Using device: {selected_device}")

    if args.mock:
        events = _fake_events(args.width, args.height)
    else:
        events = load_events_with_evlib(args.dat).collect().sort("t")
    window_iter: Iterator[Window] = _iter_windows(
        events,
        int(args.window_ms * 1000),
        loop=not args.no_loop,
    )
    if args.max_windows is not None:
        window_iter = itertools.islice(window_iter, args.max_windows)

    preview_window = next(window_iter, None)
    if preview_window is None:
        print("No events found.")
        return
    window_iter = itertools.chain([preview_window], window_iter)

    plugin = RVTDetectorPlugin(
        rvt_repo=args.rvt_repo,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset_name,
        experiment=args.experiment,
        histogram_bins=args.histogram_bins,
        window_duration_ms=args.window_ms,
        device=selected_device,
    )
    plugin.reset_states()

    cv2.namedWindow("detector-ui", cv2.WINDOW_NORMAL)
    try:
        for window in window_iter:
            detections = plugin.process(window.data)
            frame = _events_to_frame(window.data, args.width, args.height)
            _draw_detections(frame, detections)
            cv2.imshow("detector-ui", frame)
            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()


def _iter_windows(
    df: pl.DataFrame,
    window_us: int,
    *,
    loop: bool,
) -> Iterator[Window]:
    if df.height == 0:
        return iter(())

    def generator() -> Iterator[Window]:
        timestamps = df["t"].to_numpy()
        n_events = df.height
        start_idx = 0
        current_start = int(timestamps[0])
        while True:
            end_ts = current_start + window_us
            end_idx = int(np.searchsorted(timestamps, end_ts, side="left"))
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, n_events)
            chunk = df.slice(start_idx, end_idx - start_idx)
            yield Window(start_us=current_start, end_us=end_ts, data=chunk)
            if end_idx >= n_events:
                if not loop:
                    break
                start_idx = 0
                current_start = int(timestamps[0])
            else:
                start_idx = end_idx
                current_start = int(timestamps[start_idx])

    return generator()


def _events_to_frame(chunk: pl.DataFrame, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    xs = chunk["x"].to_numpy()
    ys = chunk["y"].to_numpy()
    polarities = chunk["polarity"].to_numpy().astype(bool, copy=False)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    polarities = polarities[valid]
    frame[ys[polarities], xs[polarities]] = (255, 255, 255)
    frame[ys[~polarities], xs[~polarities]] = (0, 0, 0)
    return frame


def _draw_detections(frame: np.ndarray, detections: List) -> None:
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_id}:{det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def _fake_events(width: int, height: int, num_events: int = 2048) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "t": rng.integers(0, 50000, size=num_events).astype(int),
            "x": rng.integers(0, width, size=num_events).astype(int),
            "y": rng.integers(0, height, size=num_events).astype(int),
            "polarity": rng.choice([0, 1], size=num_events).astype(int),
        }
    ).sort("t")


if __name__ == "__main__":
    main()
