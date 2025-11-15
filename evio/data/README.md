# Event Camera Data

This directory contains event camera dataset files (.dat, .aedat, .h5 formats).

## Expected Files

### Fan Datasets (for RPM detection challenges)
- `fan/fan_const_rpm.dat` - Constant RPM fan rotation (202 MB)
- `fan/fan_const_rpm.raw` - Raw format (119 MB)
- `fan/fan_varying_rpm.dat` - Varying RPM fan rotation (490 MB)
- `fan/fan_varying_rpm.raw` - Raw format (288 MB)
- `fan/fan_varying_rpm_turning.dat` - Fan with rotation variation (367 MB)
- `fan/fan_varying_rpm_turning.raw` - Raw format (225 MB)

### Drone Datasets (for tracking challenges)
- `drone_idle/drone_idle.dat` - Stationary drone (702 MB)
- `drone_idle/drone_idle.raw` - Raw format (982 MB)
- `drone_moving/drone_moving.dat` - Moving drone (1.4 GB)
- `drone_moving/drone_moving.raw` - Raw format (753 MB)

### Fred-0 Reference Dataset
- `fred-0/Event/events.dat` - Event data (224 MB)
- `fred-0/Event/events.raw` - Raw format (143 MB)
- `fred-0/Event/Frames/*.png` - Reference frame images (50 frames)
- `fred-0/Event_YOLO/*.txt` - YOLO annotation files

## Downloading Datasets

**These files are not committed to git** (they are large binary files ~5.6 GB total, listed in .gitignore).

### Option 1: Extract with unzip-datasets (Recommended for Hackathon)

If you have the `junction-sensofusion.zip` file:

1. Copy the ZIP file to this directory:
   ```bash
   cp /path/to/junction-sensofusion.zip evio/data/
   ```

2. From the `nix develop` shell, run:
   ```bash
   unzip-datasets
   ```

This will:
- Check if datasets already exist (prompts before overwriting)
- Extract all datasets to the correct directory structure
- Verify extraction was successful
- Show inventory summary

After extraction, you'll have:
- `evio/data/fan/` - Fan datasets (6 files, ~1.7 GB)
- `evio/data/drone_idle/` - Idle drone dataset (2 files, ~1.7 GB)
- `evio/data/drone_moving/` - Moving drone dataset (2 files, ~2.2 GB)
- `evio/data/fred-0/` - Fred-0 reference dataset (events + 50 frames, ~400 MB)

### Option 2: Manual Extraction

If you prefer manual control:

```bash
cd evio/data/
unzip junction-sensofusion.zip
```

### Option 3: Automated Download

If you don't have the ZIP file, download from Google Drive:

From the `nix develop` shell:

```bash
download-datasets
```

This will:
- Download all datasets from Google Drive (~1.4 GB)
- Automatically organize files into the correct directory structure
- Prompt for confirmation before downloading

**Note:** May hit quota limits during high-traffic periods (hackathon events).

### Option 4: Manual Download

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
