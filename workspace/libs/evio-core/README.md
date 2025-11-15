# evio-core

Core event camera processing library with evlib integration.

## Purpose

Provides:
- FileEventAdapter using evlib (10x faster file loading)
- EventSource protocol for file/stream abstraction
- DetectorPlugin protocol for extensible algorithms
- Representation wrappers (time surface, voxel grids)

## Testing

### Comparison Tests

**Purpose:** Validate that evlib-loaded EVT3 .dat files produce statistically equivalent results to the legacy loader.

**Run:**
```bash
run-evlib-tests
```

Or directly:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -v
```

**What it checks:**
- Event counts (exact match)
- Timestamp ranges (0.01% tolerance)
- Spatial bounds x/y min/max (0.01% tolerance)
- Polarity distribution (exact match)

**Prerequisites:**
1. Datasets extracted: `unzip-datasets`
2. EVT3 conversion: `convert-all-datasets`

**Datasets tested:**
- fan_const_rpm (30.4M events)
- drone_idle (140.7M events)

## Status

🚧 Skeleton - awaiting Work Stream 2 (hackathon-poc) implementation
