# evlib-examples

Minimal sandbox for experimenting with evlib on the standalone `.raw` datasets.

## Quick start

```
nix develop
uv sync
uv run --package evlib-examples evlib-raw-demo evio/data/fan/fan_const_rpm.raw \
  --duration-ms 25 --limit-events 100000 --output tmp/fan_raw.png
```

What it does:
1. Loads the EVT3 file via `evlib.load_events(...)`.
2. Materializes a small time window/desubsampled subset for interactive work.
3. Prints event count/resolution/polarity stats.
4. Emits a density heatmap (`tmp/fan_raw.png`) and a polarity scatter
   (`tmp/fan_raw_scatter.png`) so you can eyeball the capture without touching the
   legacy loader.

This tool never interacts with `evio.core.recording`; it is purely for evlib-based
prototyping of the newer `.raw` datasets so ongoing integration work stays isolated.

## Real-time player

```
nix develop
uv run --package evlib-examples evlib-raw-player evio/data/fan/fan_const_rpm.raw --window 5 --speed 1.0
```

Loads the entire file once (example-scale) and replays windows via `cv2.imshow()`.
Meant for quick inspection; for production, port the logic into the main evlib
pipeline.
