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
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Core tools
            python
            pkgs.uv                 # UV package manager

            # Rust toolchain (for evlib compilation)
            pkgs.rustc
            pkgs.cargo
            pkgs.pkg-config

            # System libraries
            pkgs.opencv4            # OpenCV for visualization
            pkgs.zlib               # Required by some Rust packages
            pkgs.hdf5               # Required by evlib
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
