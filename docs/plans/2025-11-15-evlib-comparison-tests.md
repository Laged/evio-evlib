# evlib Comparison Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build comparison test suite that validates evlib-loaded EVT3 .dat files contain statistically equivalent data to legacy loader outputs.

**Architecture:** Pytest-based comparison tests that load datasets with both legacy (evio.core.recording.open_dat) and new (evlib.load_events) loaders, extract statistical aggregates (event counts, timestamp ranges, spatial bounds, polarity distribution), and assert equivalence with 0.01% tolerance for ranges and exact match for counts.

**Tech Stack:** pytest, evlib, polars, numpy, evio.core.recording

**Design Document:** `docs/plans/2025-11-15-evlib-comparison-tests-design.md`

---

## Task 1: Setup evio-core Package Structure

**Files:**
- Modify: `workspace/libs/evio-core/pyproject.toml`
- Create: `workspace/libs/evio-core/tests/__init__.py`
- Create: `workspace/libs/evio-core/README.md`

**Step 1: Add test dependencies to pyproject.toml**

Open `workspace/libs/evio-core/pyproject.toml` and update the dependencies section:

```toml
[project]
name = "evio-core"
version = "0.1.0"
description = "Core event camera processing library with evlib integration"
requires-python = ">=3.11"
dependencies = [
    "evlib>=0.2.2",
    "polars>=0.20.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 2: Create tests directory structure**

Run:
```bash
mkdir -p workspace/libs/evio-core/tests
touch workspace/libs/evio-core/tests/__init__.py
```

Expected: Directory and file created successfully

**Step 3: Create README documenting the tests**

Create `workspace/libs/evio-core/README.md`:

```markdown
# evio-core

Core event camera processing library with evlib integration.

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
```

**Step 4: Sync dependencies**

Run:
```bash
uv sync
```

Expected: Dependencies installed, including evlib, polars, numpy, pytest

**Step 5: Commit**

```bash
git add workspace/libs/evio-core/pyproject.toml
git add workspace/libs/evio-core/tests/__init__.py
git add workspace/libs/evio-core/README.md
git commit -m "feat(evio-core): add test dependencies and README"
```

---

## Task 2: Write Helper Functions (TDD - Part 1)

**Files:**
- Create: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Step 1: Write failing test for decode_legacy_events**

Create `workspace/libs/evio-core/tests/test_evlib_comparison.py`:

```python
"""Comparison tests between evlib and legacy loaders."""

from __future__ import annotations

import numpy as np
import pytest


def test_decode_legacy_events():
    """Test decoding of packed uint32 event words."""
    # Create test event: x=100, y=200, polarity=1
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    # polarity=1 -> bits [31:28] = 0x1
    # y=200 -> bits [27:14] = 200 = 0xC8
    # x=100 -> bits [13:0] = 100 = 0x64
    event_word = (1 << 28) | (200 << 14) | 100
    event_words = np.array([event_word], dtype=np.uint32)

    x, y, polarity = decode_legacy_events(event_words)

    assert x[0] == 100
    assert y[0] == 200
    assert polarity[0] == 1


def test_decode_legacy_events_polarity_zero():
    """Test decoding of polarity=0 events."""
    # polarity=0 -> bits [31:28] = 0x0
    event_word = (0 << 28) | (150 << 14) | 50
    event_words = np.array([event_word], dtype=np.uint32)

    x, y, polarity = decode_legacy_events(event_words)

    assert x[0] == 50
    assert y[0] == 150
    assert polarity[0] == 0
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_decode_legacy_events -v
```

Expected: FAIL with "NameError: name 'decode_legacy_events' is not defined"

**Step 3: Implement decode_legacy_events**

Add to `workspace/libs/evio-core/tests/test_evlib_comparison.py` at the top (after imports):

```python
def decode_legacy_events(event_words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode packed uint32 event_words into x, y, polarity arrays.

    Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    See: evio/src/evio/core/mmap.py:151-154

    Args:
        event_words: Packed uint32 events from legacy loader

    Returns:
        Tuple of (x, y, polarity) numpy arrays
    """
    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    polarity = (raw_polarity > 0).astype(np.int8)
    return x, y, polarity
```

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_decode_legacy_events -v
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_decode_legacy_events_polarity_zero -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add workspace/libs/evio-core/tests/test_evlib_comparison.py
git commit -m "feat(evio-core): add decode_legacy_events with tests"
```

---

## Task 3: Write Stats Computation Functions (TDD - Part 2)

**Files:**
- Modify: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Step 1: Write failing test for compute_legacy_stats**

Add to `workspace/libs/evio-core/tests/test_evlib_comparison.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MockRecording:
    """Mock Recording for testing."""
    width: int
    height: int
    timestamps: np.ndarray
    event_words: np.ndarray
    order: np.ndarray


def test_compute_legacy_stats():
    """Test statistics extraction from legacy Recording."""
    # Create mock recording with 3 events
    timestamps = np.array([1000, 2000, 3000], dtype=np.int64)

    # Event 1: x=100, y=200, p=1
    # Event 2: x=150, y=250, p=0
    # Event 3: x=200, y=300, p=1
    event_words = np.array([
        (1 << 28) | (200 << 14) | 100,
        (0 << 28) | (250 << 14) | 150,
        (1 << 28) | (300 << 14) | 200,
    ], dtype=np.uint32)

    order = np.array([0, 1, 2], dtype=np.int32)

    recording = MockRecording(
        width=1280,
        height=720,
        timestamps=timestamps,
        event_words=event_words,
        order=order,
    )

    stats = compute_legacy_stats(recording)

    assert stats['event_count'] == 3
    assert stats['t_min'] == 1000
    assert stats['t_max'] == 3000
    assert stats['x_min'] == 100
    assert stats['x_max'] == 200
    assert stats['y_min'] == 200
    assert stats['y_max'] == 300
    assert stats['p_count_0'] == 1
    assert stats['p_count_1'] == 2
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_compute_legacy_stats -v
```

Expected: FAIL with "NameError: name 'compute_legacy_stats' is not defined"

**Step 3: Implement compute_legacy_stats**

Add to `workspace/libs/evio-core/tests/test_evlib_comparison.py` (after decode_legacy_events):

```python
def compute_legacy_stats(recording) -> dict[str, int]:
    """Extract statistics from legacy Recording object.

    Args:
        recording: Recording object from evio.core.recording.open_dat()

    Returns:
        Dict with keys:
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

**Step 4: Run test to verify it passes**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_compute_legacy_stats -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add workspace/libs/evio-core/tests/test_evlib_comparison.py
git commit -m "feat(evio-core): add compute_legacy_stats with tests"
```

---

## Task 4: Write evlib Stats Function (TDD - Part 3)

**Files:**
- Modify: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Step 1: Add imports for evlib and polars**

Add to top of `workspace/libs/evio-core/tests/test_evlib_comparison.py`:

```python
import polars as pl
import evlib
```

**Step 2: Write failing test for compute_evlib_stats**

Add test (this will be skipped if test file doesn't exist):

```python
@pytest.mark.skipif(
    not Path("evio/data/fan/fan_const_rpm_evt3.dat").exists(),
    reason="EVT3 test file not available"
)
def test_compute_evlib_stats():
    """Test statistics extraction from evlib-loaded file."""
    # Use actual converted file for this test
    dat_path = Path("evio/data/fan/fan_const_rpm_evt3.dat")

    stats = compute_evlib_stats(dat_path)

    # Basic sanity checks on known dataset
    assert stats['event_count'] > 0
    assert stats['t_min'] >= 0
    assert stats['t_max'] > stats['t_min']
    assert stats['x_min'] >= 0
    assert stats['x_max'] > stats['x_min']
    assert stats['y_min'] >= 0
    assert stats['y_max'] > stats['y_min']
    assert stats['p_count_0'] >= 0
    assert stats['p_count_1'] >= 0
    assert stats['p_count_0'] + stats['p_count_1'] == stats['event_count']
```

**Step 3: Run test to verify it fails**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_compute_evlib_stats -v
```

Expected: FAIL with "NameError: name 'compute_evlib_stats' is not defined" (or SKIPPED if file doesn't exist)

**Step 4: Implement compute_evlib_stats**

Add to `workspace/libs/evio-core/tests/test_evlib_comparison.py` (after compute_legacy_stats):

```python
def compute_evlib_stats(dat_path: Path) -> dict[str, int]:
    """Extract statistics from evlib-loaded file.

    Handles both Duration and Int64 timestamp types.
    See: workspace/tools/evio-verifier/src/evio_verifier/cli.py:46-76

    Args:
        dat_path: Path to EVT3 .dat file

    Returns:
        Dict with same keys as compute_legacy_stats()
    """
    lazy = evlib.load_events(str(dat_path))

    # Handle Duration vs Int64 timestamps
    schema = lazy.collect_schema()
    t_dtype = schema["t"]

    if isinstance(t_dtype, pl.Duration):
        t_min_expr = pl.col("t").dt.total_microseconds().min()
        t_max_expr = pl.col("t").dt.total_microseconds().max()
    else:
        t_min_expr = pl.col("t").min()
        t_max_expr = pl.col("t").max()

    stats = lazy.select([
        pl.len().alias("event_count"),
        t_min_expr.alias("t_min"),
        t_max_expr.alias("t_max"),
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("y").min().alias("y_min"),
        pl.col("y").max().alias("y_max"),
        (pl.col("p") == 0).sum().alias("p_count_0"),
        (pl.col("p") == 1).sum().alias("p_count_1"),
    ]).collect().to_dicts()[0]

    return {k: int(v) for k, v in stats.items()}
```

**Step 5: Run test to verify it passes**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_compute_evlib_stats -v
```

Expected: PASS (or SKIPPED if dataset not available)

**Step 6: Commit**

```bash
git add workspace/libs/evio-core/tests/test_evlib_comparison.py
git commit -m "feat(evio-core): add compute_evlib_stats with tests"
```

---

## Task 5: Write Tolerance Assertion Helper

**Files:**
- Modify: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Step 1: Write test for assert_within_tolerance**

Add to `workspace/libs/evio-core/tests/test_evlib_comparison.py`:

```python
def test_assert_within_tolerance_exact_match():
    """Test tolerance check with exact match."""
    assert_within_tolerance(1000, 1000, 0.0001, "exact")
    # Should not raise


def test_assert_within_tolerance_within_bounds():
    """Test tolerance check within bounds."""
    # 0.01% of 10000 = 1, so 10001 should pass
    assert_within_tolerance(10000, 10001, 0.0001, "within")
    # Should not raise


def test_assert_within_tolerance_exceeds_bounds():
    """Test tolerance check exceeds bounds."""
    with pytest.raises(AssertionError, match="diff:"):
        # 0.01% of 10000 = 1, so 10002 should fail (0.02% diff)
        assert_within_tolerance(10000, 10002, 0.0001, "exceeds")


def test_assert_within_tolerance_zero():
    """Test tolerance check with zero expected."""
    assert_within_tolerance(0, 0, 0.0001, "zero")
    # Should not raise

    with pytest.raises(AssertionError):
        assert_within_tolerance(0, 1, 0.0001, "zero_fail")
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -k "tolerance" -v
```

Expected: FAIL with "NameError: name 'assert_within_tolerance' is not defined"

**Step 3: Implement assert_within_tolerance**

Add to `workspace/libs/evio-core/tests/test_evlib_comparison.py` (after compute_evlib_stats):

```python
def assert_within_tolerance(expected: int, actual: int, tolerance: float, label: str = "value") -> None:
    """Assert values match within percentage tolerance.

    Args:
        expected: Expected value from legacy loader
        actual: Actual value from evlib loader
        tolerance: Maximum relative difference (e.g., 0.0001 = 0.01%)
        label: Description for error messages

    Raises:
        AssertionError: If values differ by more than tolerance
    """
    if expected == 0:
        assert actual == 0, f"{label}: Expected 0, got {actual}"
    else:
        rel_diff = abs(actual - expected) / abs(expected)
        assert rel_diff <= tolerance, \
            f"{label}: Expected {expected}, got {actual} (diff: {rel_diff:.4%} > {tolerance:.4%})"
```

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -k "tolerance" -v
```

Expected: All 4 tolerance tests PASS

**Step 5: Commit**

```bash
git add workspace/libs/evio-core/tests/test_evlib_comparison.py
git commit -m "feat(evio-core): add tolerance assertion helper with tests"
```

---

## Task 6: Write Main Comparison Test

**Files:**
- Modify: `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Step 1: Add dataset existence check and imports**

Add at the top of `workspace/libs/evio-core/tests/test_evlib_comparison.py` (after other imports):

```python
from evio.core.recording import open_dat
```

Add after imports:

```python
# Dataset availability check
def dataset_exists(dat_path: str, evt3_path: str) -> bool:
    """Check if both legacy and EVT3 files exist."""
    return Path(dat_path).exists() and Path(evt3_path).exists()


DATASETS_AVAILABLE = (
    dataset_exists("evio/data/fan/fan_const_rpm.dat",
                   "evio/data/fan/fan_const_rpm_evt3.dat") and
    dataset_exists("evio/data/drone_idle/drone_idle.dat",
                   "evio/data/drone_idle/drone_idle_evt3.dat")
)
```

**Step 2: Write the parametrized comparison test**

Add to end of `workspace/libs/evio-core/tests/test_evlib_comparison.py`:

```python
@pytest.mark.skipif(
    not DATASETS_AVAILABLE,
    reason="Datasets not found. Run 'unzip-datasets' and 'convert-all-datasets' first."
)
@pytest.mark.parametrize("dataset_name,legacy_dat,evt3_dat,width,height", [
    (
        "fan_const_rpm",
        "evio/data/fan/fan_const_rpm.dat",
        "evio/data/fan/fan_const_rpm_evt3.dat",
        1280,
        720,
    ),
    (
        "drone_idle",
        "evio/data/drone_idle/drone_idle.dat",
        "evio/data/drone_idle/drone_idle_evt3.dat",
        1280,
        720,
    ),
])
def test_evlib_vs_legacy_stats(
    dataset_name: str,
    legacy_dat: str,
    evt3_dat: str,
    width: int,
    height: int,
):
    """Compare evlib and legacy loader statistical outputs.

    This test validates that the EVT3 .dat conversion preserves data integrity
    by comparing key statistics from both loaders.
    """
    # Load with legacy loader
    legacy_rec = open_dat(legacy_dat, width=width, height=height)
    legacy_stats = compute_legacy_stats(legacy_rec)

    # Load with evlib
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

    # Print summary for visibility
    print(f"\n{dataset_name} comparison:")
    print(f"  Events: {evlib_stats['event_count']:,}")
    print(f"  Time range: {evlib_stats['t_min']} → {evlib_stats['t_max']}")
    print(f"  X range: {evlib_stats['x_min']} → {evlib_stats['x_max']}")
    print(f"  Y range: {evlib_stats['y_min']} → {evlib_stats['y_max']}")
    print(f"  Polarity: 0={evlib_stats['p_count_0']:,}, 1={evlib_stats['p_count_1']:,}")
```

**Step 3: Run test to verify behavior**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py::test_evlib_vs_legacy_stats -v -s
```

Expected:
- If datasets available: PASS for both fan_const_rpm and drone_idle with printed statistics
- If datasets not available: SKIPPED with helpful message

**Step 4: Commit**

```bash
git add workspace/libs/evio-core/tests/test_evlib_comparison.py
git commit -m "feat(evio-core): add main comparison test for evlib vs legacy"
```

---

## Task 7: Add Nix Alias and Update Documentation

**Files:**
- Modify: `flake.nix`
- Modify: `docs/plans/wip-evlib-integration.md`

**Step 1: Add run-evlib-tests alias to flake.nix**

Open `flake.nix` and add to shellHook after other aliases (around line 301):

```nix
alias run-evlib-tests='uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -v -s'
```

And add to the help text (around line 296):

```nix
echo "🧪 Testing:"
echo "  run-evlib-tests      : Compare evlib vs legacy loader"
echo ""
```

**Step 2: Verify alias works**

Exit and re-enter nix develop:

```bash
exit
nix develop
```

Then test the alias:

```bash
run-evlib-tests
```

Expected: Tests run with verbose output (PASS or SKIPPED)

**Step 3: Update WIP integration tracking doc**

Open `docs/plans/wip-evlib-integration.md` and update section 5.3:

```markdown
### 5.3 Minimal PoC test before full migration ✅

**Status:** Complete

**Test Location:** `workspace/libs/evio-core/tests/test_evlib_comparison.py`

**Run Command:**
```bash
run-evlib-tests
```

**What it tests:**
- Loads fan_const_rpm and drone_idle with both legacy and evlib loaders
- Compares event counts (exact match required)
- Compares timestamp/spatial ranges (0.01% tolerance)
- Compares polarity distribution (exact match required)

**Test Results:** [To be filled after first successful run]
- fan_const_rpm: ✅ PASS
  - Events: [count]
  - Time range: [t_min] → [t_max]
  - X range: [x_min] → [x_max]
  - Y range: [y_min] → [y_max]
  - Polarity: 0=[count], 1=[count]

- drone_idle: ✅ PASS
  - Events: [count]
  - [similar details]

**Next Steps:** See section 5.4 for roll-out plan.
```

**Step 4: Update usage commands section**

Update section 3 in `docs/plans/wip-evlib-integration.md`:

```markdown
## 3. Usage Commands

Run inside `nix develop`:

```bash
# Get datasets (if not already present)
unzip-datasets   # or download-datasets

# Convert all .raw files to EVT3 .dat
convert-all-datasets

# Verify a converted file
uv run --package evio-verifier verify-dat evio/data/fan/fan_const_rpm_evt3.dat

# Run comparison tests (evlib vs legacy)
run-evlib-tests

# Or convert a single file
convert-evt3-raw-to-dat evio/data/fan/fan_const_rpm.raw
```
```

**Step 5: Commit**

```bash
git add flake.nix
git add docs/plans/wip-evlib-integration.md
git commit -m "feat: add run-evlib-tests alias and update docs"
```

---

## Task 8: Run Full Test Suite and Document Results

**Files:**
- Modify: `docs/plans/wip-evlib-integration.md`

**Step 1: Ensure datasets are ready**

Run:
```bash
# Check if datasets exist
ls -lh evio/data/fan/fan_const_rpm.dat evio/data/fan/fan_const_rpm_evt3.dat
ls -lh evio/data/drone_idle/drone_idle.dat evio/data/drone_idle/drone_idle_evt3.dat
```

Expected: All 4 files exist

If not:
```bash
unzip-datasets
convert-all-datasets
```

**Step 2: Run the test suite**

Run:
```bash
run-evlib-tests
```

Expected output:
```
workspace/libs/evio-core/tests/test_evlib_comparison.py::test_evlib_vs_legacy_stats[fan_const_rpm]
fan_const_rpm comparison:
  Events: 30,380,201
  Time range: ... → ...
  X range: ... → ...
  Y range: ... → ...
  Polarity: 0=..., 1=...
PASSED

workspace/libs/evio-core/tests/test_evlib_comparison.py::test_evlib_vs_legacy_stats[drone_idle]
drone_idle comparison:
  Events: 140,XXX,XXX
  ...
PASSED

========== 2 passed in XX.XXs ==========
```

**Step 3: Document results in wip-evlib-integration.md**

Update section 5.3 with actual test results from the output above.

**Step 4: Run all unit tests to ensure nothing broke**

Run:
```bash
uv run --package evio-core pytest workspace/libs/evio-core/tests/ -v
```

Expected: All tests PASS (both unit tests and comparison tests)

**Step 5: Commit final results**

```bash
git add docs/plans/wip-evlib-integration.md
git commit -m "docs: add evlib comparison test results"
```

---

## Task 9: Final Verification and Cleanup

**Step 1: Verify all commands work from clean shell**

Exit and re-enter nix develop:
```bash
exit
nix develop
```

Check that help text shows the new command:
```bash
# Should see in output:
# 🧪 Testing:
#   run-evlib-tests      : Compare evlib vs legacy loader
```

**Step 2: Run the test one more time**

```bash
run-evlib-tests
```

Expected: PASS for both datasets

**Step 3: Verify pytest can discover tests**

```bash
uv run --package evio-core pytest --collect-only workspace/libs/evio-core/tests/
```

Expected: Shows all test items including test_evlib_vs_legacy_stats[fan_const_rpm] and [drone_idle]

**Step 4: Check git status**

```bash
git status
```

Expected: Clean working directory (all changes committed)

**Step 5: Review commit history**

```bash
git log --oneline -10
```

Expected: See commits from all tasks in logical order

---

## Success Criteria

- [ ] `run-evlib-tests` command available in nix develop shell
- [ ] Tests run successfully on both fan_const_rpm and drone_idle datasets
- [ ] All event counts match exactly between legacy and evlib loaders
- [ ] All timestamp/spatial ranges within 0.01% tolerance
- [ ] Tests complete in <30 seconds total
- [ ] Documentation updated with test results
- [ ] All code committed with clear commit messages

## References

- Design document: `docs/plans/2025-11-15-evlib-comparison-tests-design.md`
- Legacy loader: `evio/src/evio/core/recording.py`
- Bit packing: `evio/src/evio/core/mmap.py:151-154`
- evlib stats: `workspace/tools/evio-verifier/src/evio_verifier/cli.py`
