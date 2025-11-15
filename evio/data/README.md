# Event Camera Data

This directory contains event camera dataset files (.dat, .aedat, .h5 formats).

## Expected Files

### Fan Datasets (for RPM detection challenges)
- `fan/fan_const_rpm.dat` - Constant RPM fan rotation (202 MB)
- `fan/fan_const_rpm.raw` - Raw format (119 MB)
- `fan/fan_varying_rpm.dat` - Varying RPM fan rotation (489 MB)
- `fan/fan_varying_rpm.raw` - Raw format (367 MB)
- `fan/fan_varying_rpm_turning.dat` - Fan with rotation variation (367 MB)
- `fan/fan_varying_rpm_turning.raw` - Raw format (224 MB)

### Drone Datasets (for tracking challenges)
- `drone/drone_idle.dat` - Stationary drone
- `drone/drone_idle.raw` - Raw format
- `drone/drone_moving.dat` - Moving drone
- `drone/drone_moving.raw` - Raw format

### Fred-0 Reference Dataset
- `fred-0/events/events.dat` - Event data
- `fred-0/events/events.raw` - Raw format
- `fred-0/frames/*.png` - Reference frame images

## Downloading Datasets

**These files are not committed to git** (they are large binary files ~1.4 GB total, listed in .gitignore).

### Automated Download (Recommended)

From the `nix develop` shell:

```bash
download-datasets
```

This will:
- Download all datasets from Google Drive (~1.4 GB)
- Automatically organize files into the correct directory structure
- Prompt for confirmation before downloading

### Manual Download

For the Sensofusion Junction Hackathon:
- Download dataset files from Sensofusion mentors at the venue
- Or access Google Drive folder: https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE
- Place files in the appropriate subdirectories

**Note:** The hong_kong 3D dataset is not included (requires special permissions)

## File Formats

The evio library and evlib support multiple event camera formats:
- `.dat` - Prophesee/Metavision DAT format
- `.aedat` - AER DAT format
- `.h5` - HDF5 event format

## Usage

Once data files are present, you can run demos:

```bash
# From repo root, in nix develop shell
run-demo-fan           # Play fan dataset
run-mvp-1              # MVP 1 - Event density
run-mvp-2              # MVP 2 - Voxel FFT
```

Or directly:
```bash
uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat
```
