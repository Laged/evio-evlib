# evlib – High-Performance Event Data Backbone

Use this guide when reasoning about, prompting, or coding against [`evlib`](https://lib.rs/crates/evlib) inside the `evio-evlib` workspace. It summarizes upstream capabilities, our integration strategy, data constraints, and the commands Claude/Codex should run in this repo.

---

## 1. Upstream Snapshot

- **Project:** `evlib` Rust crate with Python bindings (PyO3)
- **Maintainer:** Andrew C. Freeman et al.
- **Language:** Rust core + Python module (`pip install evlib`)
- **Core deps:** Rust toolchain, `pyo3`, `polars`, `numpy`, `ndarray`, OpenCV (optional)
- **Source docs:** [`evio/docs/refactor-to-evlib.md`](../../evio/docs/refactor-to-evlib.md), [`evio/docs/dat-format-compatibility.md`](../../evio/docs/dat-format-compatibility.md)

Why we care: evlib provides the exact preprocessing pipeline our approved architecture expects (`docs/architecture.md`, repo root UV workspace). It replaces all custom `.dat` parsing, generates stacked histograms/voxel grids/time surfaces in seconds, and feeds both our classical detectors and RVT (see `evio/docs/evlib-rvt-architecture.md`).

---

## 2. Role Inside Our Architecture

| Layer (`docs/architecture.md`) | evlib Responsibility | Workspace Target |
| --- | --- | --- |
| Layer 1 – Data Acquisition | `evlib.load_events(...)` for `.dat`/`.aedat`/`.h5` ingestion | `workspace/libs/evio-core/src/evio_core/loaders.py` |
| Layer 2 – Event Representations | `evlib.representations` (`create_stacked_histogram`, `create_voxel_grid`, `create_timesurface`) | `workspace/libs/evio-core/src/evio_core/representations.py` |
| Layer 3 – Processing Pipelines | Supplies tensors for plugins (e.g., fan-bbox) + RVT | `workspace/plugins/*` |
| Layer 6 – Apps | Ensures `uv run detector-ui ...` can pull consistent windows | `workspace/apps/detector-ui` |

evlib is therefore the "data plane" for every plugin; detectors read Polars DataFrames/torch tensors produced here.

---

## 3. Capabilities & Key APIs

### 3.1 File Loading

```python
import evlib

events = evlib.load_events("data/gen4_1mpx.h5")
# → Polars LazyFrame (call .collect() for eager DataFrame)
```

Supported formats (per [dat-format doc](../../evio/docs/dat-format-compatibility.md)):
- EVT2/EVT3 `.dat`/`.raw`
- AEDAT 2/3/4
- HDF5 `.h5`

Custom Sensofusion `.dat` files currently **do not** parse; see §6.

### 3.2 Representations

```python
import evlib.representations as evr

hist = evr.create_stacked_histogram(
    events,
    height=480,
    width=640,
    bins=10,
    window_duration_ms=50.0,
)

timesurface = evr.create_timesurface(events, height=480, width=640, dt=33_000.0, tau=50_000.0)
voxel = evr.create_voxel_grid(events, height=480, width=640, bins=5)
```

Outputs are Polars DataFrames with `(time_bin, polarity, y, x, count/value)` columns; downstream adapters convert them to NumPy/Torch arrays (see conversion helpers in `evio/docs/refactor-to-evlib.md` §"Stacked Histogram").

### 3.3 Streaming Hooks

evlib focuses on file I/O. Live streaming stays in `neuromorphic-drivers` (see `evio/docs/evlib-rvt-architecture.md` §1.2) but can reuse evlib representations by pushing live batches through the same APIs.

---

## 4. Dependencies & Environment Setup

System packages already provided in `flake.nix` (`docs/plans/2025-11-15-nix-infrastructure-design.md`): Python 3.11, Rust toolchain, OpenCV, pkg-config. Inside the UV workspace:

```bash
# Install evlib + support libs for evio-core
uv add --package evio-core evlib polars numpy
uv sync
```

Verification (mirrors `docs/setup.md`):

```bash
uv run python -c "import evlib; print('evlib ready')"
```

Rust crates build during `uv sync`, so keep the Nix shell (with Rust) active.

---

## 5. Integration Checklist for Claude/Codex

1. **Use UV everywhere** – never pip (per `.claude/skills/dev-environment.md`).
2. **Load data lazily** – prefer `.lazy()` and `.collect()` windows to avoid blowing memory on 500M-event captures (`evio/docs/refactor-to-evlib.md` §3).
3. **Convert to tensors** – for RVT, convert histogram output to `(bins, 2, H, W)` before batching.
4. **Expose adapters in evio-core** – `FileEventSource`, `BatchGenerator`, etc., should only depend on evlib and Polars so plugins can be minimal.
5. **Document command paths** – all tooling runs from repo root using `uv run --package <member> ...` (`docs/architecture.md`, design principle #3).

---

## 6. Data Compatibility & Mitigations

- Custom Sensofusion `.dat` files (legacy binary) currently fail inside evlib (`evio/docs/dat-format-compatibility.md`). Options:
  1. Record/convert to standard EVT2/EVT3/AEDAT/HDF5 before feeding evlib.
  2. Add a conversion step in `evio-core` that re-encodes legacy `.dat` into HDF5 using the existing custom loader.
  3. Upstream a new decoder to evlib (advanced).
- Document which datasets live in `evio/data/` vs `workspace/data/` to avoid mixing incompatible sources.
- Always store `width`/`height` metadata with conversions (RVT requires consistent tensor sizes).

---

## 7. Testing & Benchmarks

- `uv run --package evio-core python -m evio_core.benchmarks.evlib_loader` (to be created) should compare `evlib.load_events` with legacy loaders using the datasets in `evio/data/fan/*`.
- For deterministic validation, use the minimal PoC script in [`evio/docs/evlib-rvt-poc.md`](../../evio/docs/evlib-rvt-poc.md) to render a time-surface heatmap and visually verify parity with `scripts/play_dat.py`.

---

## 8. LLM/Skill Notes

When turning this into a `.claude/skills/*` entry:
- Link back to this file plus `evio/docs/*` references.
- Provide ready-to-run commands (`uv add --package evio-core evlib polars numpy`).
- Include reminders about `nix develop` (Rust compiler) and data-format warnings.

With this cheat sheet, Claude/Codex can reason about evlib’s feature set and keep our architecture aligned with upstream expectations.

