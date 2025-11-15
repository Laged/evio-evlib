# evlib Comparison Test Suite Design

**Date:** 2025-11-15
**Owner:** Claude
**Context:** Section 5.3 of `docs/plans/wip-evlib-integration.md`
**Purpose:** Validate that evlib-loaded EVT3 .dat files contain statistically equivalent data to legacy loader outputs before migrating evio-core to use evlib.

---

## 1. Test Architecture

**Test Package**: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Purpose**: Validate that evlib-loaded EVT3 .dat files contain statistically equivalent data to legacy loader outputs, ensuring safe migration.

**Test Structure**:
- Use pytest with parametrized tests for two datasets: fan_const_rpm (30M events) and drone_idle (140M events)
- Compare statistical aggregates with 0.01% tolerance
- Test both event counts (exact) and spatial/temporal ranges (percentage-based)

**Dependencies**:
- Legacy: `evio.core.recording.open_dat()` for legacy .dat files
- New: `evlib.load_events()` for EVT3 .dat files
- Comparison: numpy for unpacking, polars for evlib stats

**Why this approach**:
- Testing small + large datasets catches both correctness and performance issues
- Statistical comparison avoids bit-packing implementation details
- 0.01% tolerance handles floating-point precision without masking real bugs

---

## 2. Test Execution & Nix Integration

**Nix Wrapper Command**: Add `run-evlib-tests` alias to `flake.nix` shellHook

```bash
alias run-evlib-tests='uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -v'
```

**Why**: Matches the pattern of `run-demo-fan`, `run-mvp-1`, etc. - users can just type the alias without remembering UV commands.

**Package Dependencies**: Add to `workspace/libs/evio-core/pyproject.toml`:
- `evlib>=0.2.2` (Rust-backed event camera loader)
- `polars>=0.20.0` (for evlib LazyFrame output)
- `numpy>=1.24.0` (for legacy loader compatibility)
- `pytest>=7.0.0` (test runner)

**OS-Level Dependencies** (already in flake.nix):
- ✅ `pkgs.rustc` + `pkgs.cargo` (evlib is PyO3/Rust-backed)
- ✅ `pkgs.pkg-config` (for finding system libraries)
- ✅ `pkgs.hdf5` (required by evlib)
- ✅ `pkgs.zlib` (required by Rust packages)
- ✅ `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` configured

**Import Strategy**:
```python
from evio.core.recording import open_dat  # Legacy loader
import evlib  # New loader
import polars as pl
import numpy as np
```

**Test Discovery**: pytest will auto-discover `test_*.py` files in the tests directory. The alias makes it explicit which test to run.

---

## 3. Test Implementation Details

**Helper Function - Decode Legacy Events**:
```python
def decode_legacy_events(event_words: np.ndarray) -> tuple:
    """Decode packed uint32 event_words into x, y, polarity arrays.

    Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    See: evio/src/evio/core/mmap.py:151-154
    """
    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    polarity = (raw_polarity > 0).astype(np.int8)
    return x, y, polarity
```

**Helper Function - Compute Stats from Legacy Loader**:
```python
def compute_legacy_stats(recording: Recording) -> dict:
    """Extract statistics from legacy Recording object.

    Returns dict with keys:
        - event_count: total events
        - t_min, t_max: timestamp range (microseconds)
        - x_min, x_max, y_min, y_max: spatial bounds
        - p_count_0, p_count_1: polarity distribution
    """
    x, y, polarity = decode_legacy_events(recording.event_words)
    return {
        'event_count': len(recording.timestamps),
        't_min': int(recording.timestamps.min()),
        't_max': int(recording.timestamps.max()),
        'x_min': int(x.min()),
        'x_max': int(x.max()),
        'y_min': int(y.min()),
        'y_max': int(y.max()),
        'p_count_0': int((polarity == 0).sum()),
        'p_count_1': int((polarity == 1).sum()),
    }
```

**Helper Function - Compute Stats from evlib** (reuses verify-dat logic):
```python
def compute_evlib_stats(dat_path: Path) -> dict:
    """Extract statistics from evlib-loaded file.

    Handles both Duration and Int64 timestamp types.
    See: workspace/tools/evio-verifier/src/evio_verifier/cli.py:46-76
    """
    lazy = evlib.load_events(str(dat_path))

    # Handle Duration vs Int64 timestamps
    schema = lazy.collect_schema()
    if isinstance(schema["t"], pl.Duration):
        t_expr = pl.col("t").dt.total_microseconds()
    else:
        t_expr = pl.col("t")

    stats = lazy.select([
        pl.len().alias("event_count"),
        t_expr.min().alias("t_min"),
        t_expr.max().alias("t_max"),
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("y").min().alias("y_min"),
        pl.col("y").max().alias("y_max"),
        (pl.col("p") == 0).sum().alias("p_count_0"),
        (pl.col("p") == 1).sum().alias("p_count_1"),
    ]).collect().to_dicts()[0]

    return {k: int(v) for k, v in stats.items()}
```

---

## 4. Test Cases & Assertions

**Parametrized Test**:
```python
@pytest.mark.parametrize("dataset_name,legacy_dat,evt3_dat,width,height", [
    ("fan_const_rpm",
     "evio/data/fan/fan_const_rpm.dat",
     "evio/data/fan/fan_const_rpm_evt3.dat",
     1280, 720),
    ("drone_idle",
     "evio/data/drone_idle/drone_idle.dat",
     "evio/data/drone_idle/drone_idle_evt3.dat",
     1280, 720),
])
def test_evlib_vs_legacy_stats(dataset_name, legacy_dat, evt3_dat, width, height):
    """Compare evlib and legacy loader statistical outputs."""
    # Load with both loaders
    legacy_rec = open_dat(legacy_dat, width=width, height=height)
    legacy_stats = compute_legacy_stats(legacy_rec)
    evlib_stats = compute_evlib_stats(Path(evt3_dat))

    # Assert exact match on counts
    assert legacy_stats['event_count'] == evlib_stats['event_count'], \
        f"{dataset_name}: Event count mismatch"
    assert legacy_stats['p_count_0'] == evlib_stats['p_count_0'], \
        f"{dataset_name}: Polarity 0 count mismatch"
    assert legacy_stats['p_count_1'] == evlib_stats['p_count_1'], \
        f"{dataset_name}: Polarity 1 count mismatch"

    # Assert 0.01% tolerance on ranges
    tolerance = 0.0001
    for key in ['t_min', 't_max', 'x_min', 'x_max', 'y_min', 'y_max']:
        assert_within_tolerance(
            legacy_stats[key],
            evlib_stats[key],
            tolerance,
            label=f"{dataset_name}.{key}"
        )
```

**Tolerance Helper**:
```python
def assert_within_tolerance(expected, actual, tolerance, label="value"):
    """Assert values match within percentage tolerance.

    Args:
        expected: Expected value from legacy loader
        actual: Actual value from evlib loader
        tolerance: Maximum relative difference (e.g., 0.0001 = 0.01%)
        label: Description for error messages
    """
    if expected == 0:
        assert actual == 0, f"{label}: Expected 0, got {actual}"
    else:
        rel_diff = abs(actual - expected) / abs(expected)
        assert rel_diff <= tolerance, \
            f"{label}: Expected {expected}, got {actual} (diff: {rel_diff:.4%} > {tolerance:.4%})"
```

**Test Output**: Use pytest's verbose mode (`-v`) to show which dataset is being tested and detailed assertion messages.

---

## 5. Error Handling & Documentation

**File Existence Checks**:
- Skip tests with `pytest.mark.skipif` if datasets don't exist
- Provide helpful error message: "Run 'unzip-datasets' or 'download-datasets' first"

**Test Skipping Logic**:
```python
import pytest
from pathlib import Path

def dataset_exists(dat_path: str, evt3_path: str) -> bool:
    """Check if both legacy and EVT3 files exist."""
    return Path(dat_path).exists() and Path(evt3_path).exists()

# Apply at module level
DATASETS_AVAILABLE = (
    dataset_exists("evio/data/fan/fan_const_rpm.dat",
                   "evio/data/fan/fan_const_rpm_evt3.dat") and
    dataset_exists("evio/data/drone_idle/drone_idle.dat",
                   "evio/data/drone_idle/drone_idle_evt3.dat")
)

pytestmark = pytest.mark.skipif(
    not DATASETS_AVAILABLE,
    reason="Datasets not found. Run 'unzip-datasets' or 'convert-all-datasets' first."
)
```

**Documentation Updates**:

1. **docs/plans/wip-evlib-integration.md** - Update section 5.3:
   - Mark PoC test as complete ✅
   - Add test results summary (pass/fail, stats comparison)
   - Document any discovered discrepancies

2. **flake.nix shellHook** - Add to help text:
   ```bash
   echo "🧪 Testing:"
   echo "  run-evlib-tests      : Compare evlib vs legacy loader"
   ```

3. **workspace/libs/evio-core/README.md** - Document the test:
   - Purpose: validates evlib integration
   - Run command: `run-evlib-tests`
   - What it checks: event counts, ranges, polarity distribution
   - Prerequisites: datasets must be extracted and converted

**Success Criteria**:
- Both datasets (fan_const_rpm, drone_idle) pass with <0.01% difference on all metrics
- Event counts match exactly
- Tests complete in <30 seconds total
- Clear output showing which comparisons passed/failed

---

## 6. Expected Workflow

**Developer Setup**:
```bash
# 1. Enter Nix environment
nix develop

# 2. Get datasets (if needed)
unzip-datasets

# 3. Convert to EVT3 (if needed)
convert-all-datasets

# 4. Sync Python dependencies
uv sync

# 5. Run comparison tests
run-evlib-tests
```

**Expected Output**:
```
workspace/libs/evio-core/tests/test_evlib_comparison.py::test_evlib_vs_legacy_stats[fan_const_rpm] PASSED
workspace/libs/evio-core/tests/test_evlib_comparison.py::test_evlib_vs_legacy_stats[drone_idle] PASSED

========== 2 passed in 12.34s ==========
```

**On Failure**: Test output will show which metric failed and the percentage difference, helping debug data conversion or loading issues.

---

## 7. Future Extensions

**After PoC Success**:
1. Add remaining datasets (fan_varying_rpm, drone_moving, fred-0) if needed
2. Add event-by-event comparison tests (slower but more thorough)
3. Add performance benchmarks (.raw vs .dat vs legacy loader)
4. Integrate into CI pipeline (if/when added)

**Migration Path**:
Once tests pass consistently:
1. Update `workspace/libs/evio-core/src/evio_core/loaders.py` to wrap evlib
2. Deprecate legacy `evio.core.recording.open_dat()` path
3. Update plugins to use evlib-backed adapters
4. Keep comparison tests as regression suite

---

## References

- **Legacy loader**: `evio/src/evio/core/recording.py:17-73`
- **Bit packing format**: `evio/src/evio/core/mmap.py:151-154`
- **evlib stats extraction**: `workspace/tools/evio-verifier/src/evio_verifier/cli.py:46-76`
- **WIP tracking**: `docs/plans/wip-evlib-integration.md` section 5.3
