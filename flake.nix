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

          TEMP_DIR=$(mktemp -d)
          cd "$TEMP_DIR"

          ${pkgs.gdown}/bin/gdown --folder --fuzzy --remaining-ok \
            "https://drive.google.com/drive/folders/1DFGuRTnME-WM_r9b-ImqKToHhc1rrR3H"

          echo ""
          echo "Organizing files into project structure..."

          # Navigate to the downloaded structure
          cd "Junction Sensofusion/Event Camera Challenge"

          # Create target directories
          mkdir -p "$OLDPWD/evio/data/fan"
          mkdir -p "$OLDPWD/evio/data/drone"
          mkdir -p "$OLDPWD/evio/data/fred-0/events"
          mkdir -p "$OLDPWD/evio/data/fred-0/frames"

          # Move fan datasets
          if [ -d "fan" ]; then
            echo "  → Moving fan datasets..."
            cp fan/*.dat fan/*.raw "$OLDPWD/evio/data/fan/" 2>/dev/null || true
          fi

          # Move drone datasets
          if [ -d "drone_idle" ]; then
            echo "  → Moving drone_idle datasets..."
            cp drone_idle/*.dat drone_idle/*.raw "$OLDPWD/evio/data/drone/" 2>/dev/null || true
          fi

          if [ -d "drone_moving" ]; then
            echo "  → Moving drone_moving datasets..."
            cp drone_moving/*.dat drone_moving/*.raw "$OLDPWD/evio/data/drone/" 2>/dev/null || true
          fi

          # Move fred-0 datasets
          if [ -d "fred-0/Event" ]; then
            echo "  → Moving fred-0 event data..."
            cp fred-0/Event/*.dat fred-0/Event/*.raw "$OLDPWD/evio/data/fred-0/events/" 2>/dev/null || true

            if [ -d "fred-0/Event/Frames" ]; then
              echo "  → Moving fred-0 reference frames..."
              cp fred-0/Event/Frames/*.png "$OLDPWD/evio/data/fred-0/frames/" 2>/dev/null || true
            fi
          fi

          # Cleanup
          cd "$OLDPWD"
          rm -rf "$TEMP_DIR"

          echo ""
          echo "✅ Download complete!"
          echo ""
          echo "Datasets installed to:"
          echo "  evio/data/fan/          - Fan rotation datasets"
          echo "  evio/data/drone/        - Drone tracking datasets"
          echo "  evio/data/fred-0/       - Fred-0 reference data"
          echo ""
          echo "Run demos with:"
          echo "  run-demo-fan"
          echo "  run-mvp-1"
          echo "  run-mvp-2"
          echo ""
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
            pkgs.zlib               # Required by some Rust packages
            pkgs.hdf5               # Required by evlib

            # Helper scripts
            download-datasets       # Dataset download script
          ];

          # Set library paths for Rust-backed libraries (evlib)
          # Use DYLD_LIBRARY_PATH on macOS to override hardcoded homebrew paths
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.hdf5
          ]}";
          DYLD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.hdf5
            pkgs.zlib
          ]}";

          shellHook = ''
            echo "=========================================="
            echo "  Event Camera Detection Workbench"
            echo "=========================================="
            echo ""
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
