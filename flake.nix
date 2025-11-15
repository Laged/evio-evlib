{
  description = "Event Camera Detection Workbench - Nix Infrastructure";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        sharedLibPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.hdf5
          pkgs.libGL
          pkgs.libGLU
          pkgs.glib
          pkgs.xorg.libX11
          pkgs.xorg.libXext
          pkgs.xorg.libXrender
          pkgs.xorg.libxcb
          pkgs.xorg.libSM
          pkgs.xorg.libICE
          pkgs.libxkbcommon
        ];

        # Convert EVT3 raw to dat script
        convertEvt3Script = pkgs.writeShellScriptBin "convert-evt3-raw-to-dat" ''
          set -euo pipefail
          exec ${pkgs.uv}/bin/uv run python scripts/convert_evt3_raw_to_dat.py "$@"
        '';

        # Convert all datasets script
        convertAllDatasetsScript = pkgs.writeShellScriptBin "convert-all-datasets" ''
          set -euo pipefail

          DATA_DIR="evio/data"

          echo "=========================================="
          echo "  Convert All Datasets to EVT3 .dat"
          echo "=========================================="
          echo ""

          # Find all .raw files
          RAW_FILES=$(find "$DATA_DIR" -name "*.raw" -type f 2>/dev/null || true)

          if [ -z "$RAW_FILES" ]; then
              echo "❌ No .raw files found in $DATA_DIR"
              echo ""
              echo "Run 'unzip-datasets' first to extract datasets."
              exit 1
          fi

          # Count files
          FILE_COUNT=$(echo "$RAW_FILES" | wc -l | tr -d ' ')
          echo "Found $FILE_COUNT .raw files to convert"
          echo ""

          # Convert each file
          SUCCESS_COUNT=0
          FAIL_COUNT=0

          for RAW_FILE in $RAW_FILES; do
              echo "Converting: $RAW_FILE"
              if ${pkgs.uv}/bin/uv run python scripts/convert_evt3_raw_to_dat.py "$RAW_FILE" --force; then
                  SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
              else
                  FAIL_COUNT=$((FAIL_COUNT + 1))
                  echo "⚠️  Failed to convert $RAW_FILE"
              fi
              echo ""
          done

          # Summary
          echo "=========================================="
          echo "  Conversion Summary"
          echo "=========================================="
          echo ""
          echo "✅ Successfully converted: $SUCCESS_COUNT files"

          if [ $FAIL_COUNT -gt 0 ]; then
              echo "❌ Failed: $FAIL_COUNT files"
          fi

          echo ""
          echo "Verify converted files with:"
          echo "  uv run --package evio-verifier verify-dat <file>.dat"
        '';

        # Unzip datasets script
        unzipDatasetsScript = pkgs.writeShellScriptBin "unzip-datasets" ''
          set -euo pipefail

          ZIP_PATH="evio/data/junction-sensofusion.zip"
          DATA_DIR="evio/data"

          # Check ZIP exists
          if [ ! -f "$ZIP_PATH" ]; then
              echo "❌ Error: junction-sensofusion.zip not found"
              echo ""
              echo "Please copy the ZIP file to evio/data/ first:"
              echo "  cp /path/to/junction-sensofusion.zip evio/data/"
              echo ""
              echo "Then run: unzip-datasets"
              exit 1
          fi

          # Check current inventory
          echo "Checking existing datasets..."
          ${pkgs.uv}/bin/uv run --package downloader python -c '
from downloader.verification import check_inventory, print_inventory
inventory = check_inventory()
print_inventory(inventory)
' || true

          # Check if datasets exist
          if [ -d "$DATA_DIR/fan" ] || [ -d "$DATA_DIR/drone_idle" ] || [ -d "$DATA_DIR/drone_moving" ] || [ -d "$DATA_DIR/fred-0" ]; then
              echo ""
              echo "⚠️  WARNING: Existing datasets found"
              echo ""
              echo "The following will be overwritten:"
              [ -d "$DATA_DIR/fan" ] && echo "  ✓ fan/ (6 files)"
              [ -d "$DATA_DIR/drone_idle" ] && echo "  ✓ drone_idle/ (2 files)"
              [ -d "$DATA_DIR/drone_moving" ] && echo "  ✓ drone_moving/ (2 files)"
              [ -d "$DATA_DIR/fred-0" ] && echo "  ✓ fred-0/ (events + frames)"
              echo ""
              read -p "Continue with extraction? (y/N): " -n 1 -r
              echo
              if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                  echo "Extraction cancelled."
                  exit 0
              fi
          fi

          # Extract ZIP
          echo ""
          echo "Extracting datasets..."
          if ! ${pkgs.unzip}/bin/unzip -o "$ZIP_PATH" -d "$DATA_DIR"; then
              echo ""
              echo "❌ Error: Failed to extract datasets"
              echo ""
              echo "The ZIP file may be corrupted. Try:"
              echo "  1. Re-download junction-sensofusion.zip"
              echo "  2. Verify file integrity"
              echo "  3. Extract manually: cd evio/data && unzip junction-sensofusion.zip"
              exit 1
          fi

          # Verify extraction
          echo ""
          echo "Verifying extraction..."
          ${pkgs.uv}/bin/uv run --package downloader python -c '
from pathlib import Path
from downloader.verification import check_inventory, print_inventory

inventory = check_inventory()

# Check for expected dataset directories (use actual paths, not inventory keys)
expected_dirs = {
    "fan": Path("evio/data/fan"),
    "drone_idle": Path("evio/data/drone_idle"),
    "drone_moving": Path("evio/data/drone_moving"),
    "fred-0": Path("evio/data/fred-0/Event")
}

missing = []
for name, path in expected_dirs.items():
    if not path.exists():
        missing.append(name)
    elif name in ["fan", "drone_idle", "drone_moving"]:
        # Check for .dat or .raw files
        if not list(path.glob("*.dat")) and not list(path.glob("*.raw")):
            missing.append(name)
    elif name == "fred-0":
        # Check for events.dat or events.raw
        if not (path / "events.dat").exists() and not (path / "events.raw").exists():
            missing.append(name)

if missing:
    print("⚠️  Warning: Extraction incomplete")
    print("")
    print("Missing expected datasets:", ", ".join(missing))
    print("")
    print("Please check:")
    print("  - ZIP file integrity")
    print("  - Available disk space")
    print("")
    print("Run download-datasets as fallback if needed.")
    exit(1)

print("=" * 50)
print("  Extraction Summary")
print("=" * 50)
print("")
print(f"✅ Successfully extracted {len(expected_dirs) - len(missing)} dataset groups")
print("")
print_inventory(inventory)

# Show demo commands if fan datasets present
if inventory.get("fan", {}).get("dat", 0) > 0:
    print("")
    print("Ready to run:")
    print("  run-demo-fan")
    print("  run-mvp-1")
    print("  run-mvp-2")
'
        '';

        # Convert legacy .dat to HDF5 script
        convertLegacyDatToHdf5Script = pkgs.writeShellScriptBin "convert-legacy-dat-to-hdf5" ''
          set -euo pipefail
          exec ${pkgs.uv}/bin/uv run --package evio-core python scripts/convert_legacy_dat_to_hdf5.py "$@"
        '';

        # Convert all legacy .dat to HDF5 script
        convertAllLegacyToHdf5Script = pkgs.writeShellScriptBin "convert-all-legacy-to-hdf5" ''
          set -euo pipefail
          exec ${pkgs.bash}/bin/bash scripts/convert_all_legacy_to_hdf5.sh
        '';

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Core tools
            python
            pkgs.uv                 # UV package manager
            pkgs.gdown              # Google Drive downloader
            pkgs.unzip              # ZIP extraction
            convertEvt3Script       # convert-evt3-raw-to-dat command
            convertAllDatasetsScript # convert-all-datasets command
            unzipDatasetsScript     # unzip-datasets command
            convertLegacyDatToHdf5Script # convert-legacy-dat-to-hdf5 command
            convertAllLegacyToHdf5Script # convert-all-legacy-to-hdf5 command

            # Rust toolchain (for evlib compilation)
            pkgs.rustc
            pkgs.cargo
            pkgs.pkg-config

            # System libraries
            pkgs.opencv4            # OpenCV for visualization
            pkgs.libGL              # OpenGL runtime required by OpenCV
            pkgs.libGLU             # GLU for OpenCV/Qt
            pkgs.glib               # GLib for OpenCV thread helpers
            pkgs.xorg.libX11        # Qt X11 stack
            pkgs.xorg.libXext
            pkgs.xorg.libXrender
            pkgs.xorg.libxcb
            pkgs.xorg.libSM
            pkgs.xorg.libICE
            pkgs.libxkbcommon
            pkgs.zlib               # Required by some Rust packages
            pkgs.hdf5               # Required by evlib
          ];

          # Set library paths for Rust-backed libraries (evlib)
          # Use DYLD_LIBRARY_PATH on macOS to override hardcoded homebrew paths
          LD_LIBRARY_PATH = sharedLibPath;
          DYLD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.hdf5
            pkgs.zlib
          ]}";

          shellHook = ''
            echo "=========================================="
            echo "  Event Camera Detection Workbench"
            echo "=========================================="
            echo ""
            export QT_QPA_PLATFORM="xcb"
            echo "Python: $(python --version)"
            echo "UV: $(uv --version)"
            echo ""

            # Create workspace structure if missing
            if [ ! -d workspace ]; then
              echo "Creating workspace structure..."
              mkdir -p workspace/libs workspace/plugins workspace/apps
            fi

            # Check if workspace needs initialization
            if [ ! -d .venv ]; then
              if [ ! -d workspace/libs/evio-core ]; then
                echo "⚠️  Workspace members not found. This appears to be initial setup."
                echo "    See docs/setup.md for workspace initialization steps."
              else
                echo "⚠️  First time setup: Run 'uv sync' to initialize workspace"
              fi
            fi

            echo ""
            echo "📦 Package Management:"
            echo "  NEVER use pip - UV only!"
            echo "  Add dependency: uv add --package <member> <package>"
            echo "  Sync workspace: uv sync"
            echo ""
            echo "📊 Dataset Management:"
            echo "  unzip-datasets              : Extract junction-sensofusion.zip"
            echo "  download-datasets           : Download from Google Drive (~1.4 GB)"
            echo ""
            echo "📦 Legacy Data Export (RECOMMENDED):"
            echo "  convert-legacy-dat-to-hdf5  : Convert single legacy .dat to evlib HDF5"
            echo "  convert-all-legacy-to-hdf5  : Convert ALL legacy .dat to evlib HDF5"
            echo ""
            echo "🧪 Experimental (IDS Camera Data):"
            echo "  convert-all-datasets        : Convert .raw to _evt3.dat (IDS only, not legacy)"
            echo "  convert-evt3-raw-to-dat     : Manual .raw → .dat (experimental)"
            echo ""
            echo "🚀 Running Commands (from repo root):"
            echo "  uv run --package <member> <command>"
            echo ""
            echo "🧪 Testing:"
            echo "  run-evlib-tests      : Compare evlib vs legacy loader"
            echo ""
            echo "🎯 Detector Demos (legacy loaders):"
            echo "  run-fan-detector     : Fan RPM estimation with ellipse fitting"
            echo "  run-drone-detector   : Drone propeller detection and RPM"
            echo ""
            echo "Demo Aliases:"
            echo "  run-demo-fan         : Play fan dataset (legacy loader)"
            echo "  run-demo-fan-ev3     : Play fan dataset (evlib on legacy HDF5 export)"
            echo "  run-mvp-1            : MVP 1 - Event density"
            echo "  run-mvp-2            : MVP 2 - Voxel FFT"
            echo ""
            echo "NOTE: run-demo-fan-ev3 uses legacy export (requires convert-legacy-dat-to-hdf5)"
            echo ""
            echo "📈 Experimental IDS Data Sandbox:"
            echo "  run-evlib-raw-demo   : Load .raw with evlib (IDS camera, not legacy)"
            echo "  run-evlib-raw-player : Real-time .raw playback (IDS camera, not legacy)"
            echo ""

            # Shell aliases for convenience
            alias download-datasets='uv run --package downloader download-datasets'
            alias run-evlib-tests='uv run --package evio-core pytest workspace/libs/evio-core/tests/test_evlib_comparison.py -v -s'
            alias run-demo-fan='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat'
            alias run-demo-fan-ev3='uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_legacy.h5'
            alias run-mvp-1='uv run --package evio python evio/scripts/mvp_1_density.py evio/data/fan/fan_const_rpm.dat'
            alias run-mvp-2='uv run --package evio python evio/scripts/mvp_2_voxel.py evio/data/fan/fan_varying_rpm.dat'
            alias run-evlib-raw-demo='uv run --package evlib-examples evlib-raw-demo'
            alias run-evlib-raw-player='uv run --package evlib-examples evlib-raw-player'
            alias run-fan-detector='uv run --package evio python evio/scripts/fan_detector_demo.py evio/data/fan/fan_const_rpm.dat'
            alias run-drone-detector='uv run --package evio python evio/scripts/drone_detector_demo.py evio/data/drone_idle/drone_idle.dat'

            echo "Read .claude/skills/dev-environment.md for workflow guidelines"
            echo "=========================================="
            echo ""
          '';
        };
      }
    );
}
