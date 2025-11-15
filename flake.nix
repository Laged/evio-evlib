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

        # Data download script
        download-datasets = pkgs.writeShellScriptBin "download-datasets" ''
          set -e

          echo "=========================================="
          echo "  Event Camera Dataset Downloader"
          echo "=========================================="
          echo ""
          echo "⚠️  WARNING: This will download ~1.4 GB of event camera data"
          echo ""
          echo "Dataset includes:"
          echo "  - Fan datasets (constant/varying RPM)"
          echo "  - Drone datasets (idle/moving)"
          echo "  - Fred-0 reference frames"
          echo ""
          read -p "Continue with download? (y/N): " -n 1 -r
          echo

          if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Download cancelled."
            exit 0
          fi

          echo ""
          echo "Downloading datasets from Google Drive..."
          echo "(This may take several minutes and some files may fail due to permissions)"
          echo ""

          TEMP_DIR=$(mktemp -d)
          cd "$TEMP_DIR"

          # Download with error handling - continue even if some files fail
          set +e  # Don't exit on error
          ${pkgs.gdown}/bin/gdown --folder --fuzzy --remaining-ok \
            "https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE" 2>&1 | tee download.log
          DOWNLOAD_EXIT=$?
          set -e  # Re-enable exit on error

          echo ""
          echo "Download completed (exit code: $DOWNLOAD_EXIT)"

          # Check if any files were downloaded
          if [ ! -d "Event Camera Challenge" ]; then
            echo "❌ Error: No files were downloaded. Check permissions or network."
            echo "You can manually download from: https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE"
            rm -rf "$TEMP_DIR"
            exit 1
          fi

          echo ""
          echo "Organizing files into project structure..."

          # Navigate to the downloaded structure
          cd "Event Camera Challenge" || {
            echo "❌ Error: Could not find 'Event Camera Challenge' folder"
            ls -la
            exit 1
          }

          # Create target directories
          mkdir -p "$OLDPWD/evio/data/fan"
          mkdir -p "$OLDPWD/evio/data/drone"
          mkdir -p "$OLDPWD/evio/data/fred-0/events"
          mkdir -p "$OLDPWD/evio/data/fred-0/frames"

          # Track what was successfully copied
          COPIED_FILES=0
          FAILED_DATASETS=""

          # Move fan datasets
          if [ -d "fan" ]; then
            echo "  → Processing fan datasets..."
            set +e
            cp fan/*.dat fan/*.raw "$OLDPWD/evio/data/fan/" 2>/dev/null
            if [ $? -eq 0 ]; then
              COUNT=$(ls fan/*.{dat,raw} 2>/dev/null | wc -l)
              COPIED_FILES=$((COPIED_FILES + COUNT))
              echo "    ✓ Copied $COUNT fan files"
            else
              FAILED_DATASETS="$FAILED_DATASETS\n  - fan datasets"
            fi
            set -e
          else
            echo "  ⚠️  fan/ directory not found in download"
            FAILED_DATASETS="$FAILED_DATASETS\n  - fan datasets (directory missing)"
          fi

          # Move drone datasets
          if [ -d "drone_idle" ]; then
            echo "  → Processing drone_idle datasets..."
            set +e
            cp drone_idle/*.dat drone_idle/*.raw "$OLDPWD/evio/data/drone/" 2>/dev/null
            if [ $? -eq 0 ]; then
              COUNT=$(ls drone_idle/*.{dat,raw} 2>/dev/null | wc -l)
              COPIED_FILES=$((COPIED_FILES + COUNT))
              echo "    ✓ Copied $COUNT drone_idle files"
            fi
            set -e
          fi

          if [ -d "drone_moving" ]; then
            echo "  → Processing drone_moving datasets..."
            set +e
            cp drone_moving/*.dat drone_moving/*.raw "$OLDPWD/evio/data/drone/" 2>/dev/null
            if [ $? -eq 0 ]; then
              COUNT=$(ls drone_moving/*.{dat,raw} 2>/dev/null | wc -l)
              COPIED_FILES=$((COPIED_FILES + COUNT))
              echo "    ✓ Copied $COUNT drone_moving files"
            fi
            set -e
          fi

          # Move fred-0 datasets
          if [ -d "fred-0/Event" ]; then
            echo "  → Processing fred-0 event data..."
            set +e
            cp fred-0/Event/*.dat fred-0/Event/*.raw "$OLDPWD/evio/data/fred-0/events/" 2>/dev/null
            if [ $? -eq 0 ]; then
              COUNT=$(ls fred-0/Event/*.{dat,raw} 2>/dev/null | wc -l)
              COPIED_FILES=$((COPIED_FILES + COUNT))
              echo "    ✓ Copied $COUNT fred-0 event files"
            fi

            if [ -d "fred-0/Event/Frames" ]; then
              cp fred-0/Event/Frames/*.png "$OLDPWD/evio/data/fred-0/frames/" 2>/dev/null
              if [ $? -eq 0 ]; then
                COUNT=$(ls fred-0/Event/Frames/*.png 2>/dev/null | wc -l)
                COPIED_FILES=$((COPIED_FILES + COUNT))
                echo "    ✓ Copied $COUNT fred-0 frame images"
              fi
            fi
            set -e
          fi

          # Cleanup
          cd "$OLDPWD"
          rm -rf "$TEMP_DIR"

          echo ""
          echo "=========================================="
          echo "  Download Summary"
          echo "=========================================="
          echo ""
          echo "✅ Successfully copied $COPIED_FILES files"
          echo ""
          echo "Datasets installed to:"
          echo "  evio/data/fan/          - Fan rotation datasets"
          echo "  evio/data/drone/        - Drone tracking datasets"
          echo "  evio/data/fred-0/       - Fred-0 reference data"
          echo ""

          if [ -n "$FAILED_DATASETS" ]; then
            echo "⚠️  Some datasets could not be downloaded:"
            echo -e "$FAILED_DATASETS"
            echo ""
            echo "This is usually due to Google Drive permissions."
            echo "You can manually download missing files from:"
            echo "  https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE"
            echo ""
          fi

          # Check what's actually in the directories
          FAN_COUNT=$(ls evio/data/fan/*.dat 2>/dev/null | wc -l | tr -d ' ')
          DRONE_COUNT=$(ls evio/data/drone/*.dat 2>/dev/null | wc -l | tr -d ' ')
          FRED_COUNT=$(ls evio/data/fred-0/events/*.dat 2>/dev/null | wc -l | tr -d ' ')

          echo "Current dataset inventory:"
          echo "  Fan datasets:   $FAN_COUNT .dat files"
          echo "  Drone datasets: $DRONE_COUNT .dat files"
          echo "  Fred-0 events:  $FRED_COUNT .dat files"
          echo ""

          if [ "$FAN_COUNT" -gt 0 ]; then
            echo "Run demos with:"
            echo "  run-demo-fan"
            echo "  run-mvp-1"
            echo "  run-mvp-2"
            echo ""
          else
            echo "⚠️  No fan datasets found - demos may not work"
            echo ""
          fi
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Core tools
            python
            pkgs.uv                 # UV package manager
            pkgs.gdown              # Google Drive downloader

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

            # Helper scripts
            download-datasets       # Dataset download script
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
            echo "  download-datasets    : Download event camera datasets (~1.4 GB)"
            echo ""
            echo "🚀 Running Commands (from repo root):"
            echo "  uv run --package <member> <command>"
            echo ""
            echo "Demo Aliases:"
            echo "  run-demo-fan         : Play fan dataset"
            echo "  run-mvp-1            : MVP 1 - Event density"
            echo "  run-mvp-2            : MVP 2 - Voxel FFT"
            echo ""

            # Shell aliases for convenience
            alias run-demo-fan='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat'
            alias run-mvp-1='uv run --package evio python evio/scripts/mvp_1_density.py evio/data/fan/fan_const_rpm.dat'
            alias run-mvp-2='uv run --package evio python evio/scripts/mvp_2_voxel.py evio/data/fan/fan_varying_rpm.dat'

            echo "Read .claude/skills/dev-environment.md for workflow guidelines"
            echo "=========================================="
            echo ""
          '';
        };
      }
    );
}
