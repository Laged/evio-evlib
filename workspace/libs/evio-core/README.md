# evio-core

Core event camera processing library with evlib integration.

## Features

- **evlib integration**: 10-200x faster event loading and processing
- **Legacy compatibility**: Validated parity with legacy evio.core.recording loader
- **Polars-based**: Modern DataFrame API for event manipulation
- **Type-safe**: Full type hints and protocol definitions

## Testing

### Run All Tests

```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/ -v
```

### Run Specific Test Suites

**Legacy parity tests** (legacy loader vs evlib):
```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_legacy_loader_vs_evlib_parity -v
```

**HDF5 export tests**:
```bash
nix develop --command uv run --package evio-core pytest workspace/libs/evio-core/tests/test_legacy_export.py -v
```

## Test Architecture

### Legacy Parity Tests

Validate legacy loader matches evlib output:
1. Load legacy .dat with `evio.core.recording.open_dat()`
2. Export events to temporary HDF5 (evlib-compatible schema)
3. Load HDF5 with `evlib.load_events()`
4. Compare stats (should match exactly)

This proves evlib can replace the legacy loader with confidence.

## Dependencies

- **evlib** (≥0.8.0): Rust-backed event processing
- **polars** (≥0.20.0): Fast DataFrame library
- **numpy** (≥1.24.0): Array operations
- **h5py** (≥3.0.0): HDF5 I/O (dev/test only)
- **pytest** (≥7.0.0): Testing framework (dev only)
