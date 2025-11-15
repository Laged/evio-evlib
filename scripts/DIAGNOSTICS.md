# Dataset Diagnostic Scripts

These scripts were used to prove that .raw files are NOT conversions of legacy .dat files.

## Evidence Scripts (Preserved for Historical Reference)

### compare_all_fan_files.py
Compares legacy .dat files vs .raw files side-by-side.

**Key findings:**
- Event counts differ (26M vs 30M)
- Resolutions differ (1280×720 vs 2040×1793)
- Durations differ (9.5s vs 682s)
- Polarity distributions differ (balanced vs broken)

### diagnose_fan_data.py
Logs detailed header and event statistics for fan dataset files.

**Key findings:**
- Headers claim 1280×720 but events extend to 2040×1793
- .raw files show 0 OFF events (encoding bug)
- Metadata mismatch indicates separate recordings

## Usage

These scripts are preserved for documentation but not part of the normal workflow.

Run to verify the dataset distinction:

```bash
nix develop --command uv run python scripts/compare_all_fan_files.py
nix develop --command uv run python scripts/diagnose_fan_data.py
```

See `docs/plans/raw-to-evt3-deprecation.md` for complete analysis.

### convert_evt3_raw_to_dat.py
Deprecated but kept in scripts/ for compatibility.
Marked with runtime warnings. See flake.nix for shell command.
