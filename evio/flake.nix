{
  description = "Minimal Python library for standardized handling of event camera data";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        pythonPackages = python.pkgs;

        evio = pythonPackages.buildPythonPackage {
          pname = "evio";
          version = "0.4.0";
          pyproject = true;

          src = ./.;

          build-system = with pythonPackages; [
            hatchling
          ];

          dependencies = with pythonPackages; [
            numpy
            opencv4
            polars
          ];

          # Skip dependency check since opencv4 satisfies opencv-python
          pythonRemoveDeps = [ "opencv-python" ];

          # Optional: run tests during build
          # nativeCheckInputs = with pythonPackages; [
          #   pytest
          #   pytest-timeout
          # ];
          #
          # checkPhase = ''
          #   runHook preCheck
          #   pytest
          #   runHook postCheck
          # '';

          pythonImportsCheck = [ "evio" ];

          meta = with pkgs.lib; {
            description = "Minimal library for standardized handling of event camera data";
            homepage = "https://github.com/ahtihelminen/evio";
            license = licenses.mit;
            maintainers = [ ];
          };
        };

        # Script wrappers
        play-dat = pkgs.writeShellScriptBin "play-dat" ''
          exec ${python.withPackages (ps: [ evio ])}/bin/python ${./scripts/play_dat.py} "$@"
        '';

        mvp-1 = pkgs.writeShellScriptBin "mvp-1" ''
          exec ${python.withPackages (ps: [ evio ])}/bin/python ${./scripts/mvp_1_density.py} "$@"
        '';

        mvp-2 = pkgs.writeShellScriptBin "mvp-2" ''
          exec ${python.withPackages (ps: [ evio ])}/bin/python ${./scripts/mvp_2_voxel.py} "$@"
        '';

        mvp-3 = pkgs.writeShellScriptBin "mvp-3" ''
          exec ${python.withPackages (ps: [ evio ])}/bin/python ${./scripts/mvp_3_hybrid.py} "$@"
        '';

        mvp-4 = pkgs.writeShellScriptBin "mvp-4" ''
          exec ${python.withPackages (ps: [ evio ps.scipy ])}/bin/python ${./scripts/mvp_4_automatic.py} "$@"
        '';

        mvp-5 = pkgs.writeShellScriptBin "mvp-5" ''
          exec ${python.withPackages (ps: [ evio ps.scipy ])}/bin/python ${./scripts/mvp_5_blade_tracking.py} "$@"
        '';

      in
      {
        packages = {
          default = evio;
          inherit evio play-dat mvp-1 mvp-2 mvp-3 mvp-4 mvp-5;
        };

        apps = {
          default = {
            type = "app";
            program = "${play-dat}/bin/play-dat";
          };

          play-dat = {
            type = "app";
            program = "${play-dat}/bin/play-dat";
          };

          evio = {
            type = "app";
            program = "${play-dat}/bin/play-dat";
          };

          mvp-1 = {
            type = "app";
            program = "${mvp-1}/bin/mvp-1";
          };

          mvp-2 = {
            type = "app";
            program = "${mvp-2}/bin/mvp-2";
          };

          mvp-3 = {
            type = "app";
            program = "${mvp-3}/bin/mvp-3";
          };

          mvp-4 = {
            type = "app";
            program = "${mvp-4}/bin/mvp-4";
          };

          mvp-5 = {
            type = "app";
            program = "${mvp-5}/bin/mvp-5";
          };
        };

        devShells = {
          # Default development shell with evlib support
          default = pkgs.mkShell {
            buildInputs = [
              python
              pythonPackages.pip
              pythonPackages.uv
            ] ++ (with pythonPackages; [
              # Runtime dependencies
              numpy
              opencv4
              scipy  # For MVP-4 autocorrelation
              polars  # High-performance DataFrames (required by evlib)
              # Note: evlib installed via UV from pyproject.toml (not in nixpkgs yet)

              # Dev dependencies
              pytest
              pytest-timeout
              ruff
              mypy
              types-setuptools

              # Build dependencies
              hatchling
            ]);

            # Set LD_LIBRARY_PATH for Rust-backed libraries like evlib
            LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}";

            shellHook = ''
              # Add src directory to Python path for development
              export PYTHONPATH="$PWD/src:$PYTHONPATH"

              # Create UV virtual environment if it doesn't exist
              # This allows installing evlib (not in nixpkgs) without conflicts
              # Use Nix's Python as the base for the venv
              if [ ! -d .venv ]; then
                echo "Creating UV virtual environment..."
                uv venv --python $(which python) --quiet
              fi

              # Activate the virtual environment
              source .venv/bin/activate

              # Install project dependencies (including evlib from PyPI)
              echo "Syncing Python dependencies with UV..."
              uv pip install -e . --quiet 2>/dev/null || echo "Dependencies already synced"

              echo ""
              echo "evio development environment (with evlib)"
              echo "Python: $(python --version)"
              echo ""
              echo "📦 Core packages: numpy, opencv, scipy, polars (nixpkgs), evlib (UV/PyPI)"
              echo ""
              echo "Available commands:"
              echo "  - python scripts/play_dat.py <file.dat>     : View event data"
              echo "  - python scripts/mvp_1_density.py <file>    : MVP 1 - Event density"
              echo "  - python scripts/mvp_2_voxel.py <file>      : MVP 2 - Voxel FFT"
              echo "  - python scripts/mvp_3_hybrid.py <file>     : MVP 3 - Hybrid approach"
              echo ""
              echo "Or use nix run:"
              echo "  - nix run .#mvp-1 -- data/fan/fan_const_rpm.dat"
              echo "  - nix run .#mvp-2 -- data/fan/fan_varying_rpm.dat"
              echo "  - nix run .#mvp-3 -- data/fan/fan_varying_rpm_turning.dat"
              echo ""
              echo "Development tools:"
              echo "  - pytest           : Run tests"
              echo "  - ruff check       : Run linter"
              echo "  - mypy src         : Run type checker"
              echo ""
              echo "For full ML stack (PyTorch, RVT): nix develop .#hackathon"
              echo ""
            '';
          };

          # Full ML stack for hackathon
          hackathon = pkgs.mkShell {
            buildInputs = [
              python
              pythonPackages.pip
              pythonPackages.uv
            ] ++ (with pythonPackages; [
              # Core event camera library
              numpy
              opencv4

              # Event processing (NEW)
              polars  # High-performance DataFrames
              # Note: evlib installed via UV from pyproject.toml (not in nixpkgs yet)

              # Machine Learning
              torch
              torchvision
              scikit-learn

              # Scientific Computing
              scipy
              numba  # JIT compilation for performance

              # Visualization
              matplotlib

              # Data handling
              h5py
              pandas
              pillow

              # Development tools
              pytest
              pytest-timeout
              ruff
              mypy
              types-setuptools
              ipython

              # Build dependencies
              hatchling
            ]);

            shellHook = ''
              # Add src directory to Python path for development
              export PYTHONPATH="$PWD/src:$PYTHONPATH"

              # Create UV virtual environment if it doesn't exist
              # This allows installing evlib (not in nixpkgs) without conflicts
              # Use Nix's Python as the base for the venv
              if [ ! -d .venv ]; then
                echo "Creating UV virtual environment..."
                uv venv --python $(which python) --quiet
              fi

              # Activate the virtual environment
              source .venv/bin/activate

              # Install project dependencies (including evlib from PyPI)
              echo "Syncing Python dependencies with UV..."
              uv pip install -e . --quiet 2>/dev/null || echo "Dependencies already synced"

              echo "================================================================"
              echo "  evio Hackathon Environment - Sensofusion Challenge"
              echo "================================================================"
              echo ""
              echo "Python: $(python --version)"
              echo ""
              echo "📦 Installed Packages:"
              echo "  Core:        numpy, opencv, scipy"
              echo "  Event:       evlib (UV/PyPI venv), polars (nixpkgs)"
              echo "  ML:          PyTorch, scikit-learn"
              echo "  Performance: numba (JIT compilation)"
              echo "  Viz:         matplotlib"
              echo "  Data:        h5py, pandas"
              echo ""
              echo "🎯 MVP Demos (Fan Rotation Detection):"
              echo "  nix run .#mvp-1 -- data/fan/fan_const_rpm.dat     # Event Density"
              echo "  nix run .#mvp-2 -- data/fan/fan_varying_rpm.dat   # Voxel FFT"
              echo "  nix run .#mvp-3 -- data/fan/fan_const_rpm.dat     # Hybrid"
              echo ""
              echo "  Or directly:"
              echo "  python scripts/mvp_1_density.py data/fan/fan_const_rpm.dat"
              echo "  python scripts/mvp_2_voxel.py data/fan/fan_varying_rpm.dat"
              echo "  python scripts/mvp_3_hybrid.py data/fan/fan_const_rpm.dat"
              echo ""
              echo "📚 Documentation:"
              echo "  docs/hackathon-mvp.md        - MVP implementation guide"
              echo "  docs/voxel_grid_approach.md  - Voxel grids for ML"
              echo "  docs/hackathon-live-data.md  - Live data processing guide"
              echo ""
              echo "💡 Optimization Tips:"
              echo "  - Use @jit decorator from numba for hot loops"
              echo "  - Vectorize with numpy instead of Python loops"
              echo "  - Profile with: python -m cProfile script.py"
              echo "  - GPU acceleration: model.cuda() in PyTorch"
              echo ""
              echo "📝 Additional packages via UV if needed:"
              echo "  uv pip install plotly seaborn jupyterlab"
              echo ""
              echo "⚠️  Note: Prophesee Metavision SDK (live cameras) requires Linux/Windows"
              echo "    Develop with .dat files on macOS, test live at hackathon venue"
              echo ""
              echo "================================================================"
              echo ""
            '';

            # Set environment variables for ML libraries and Rust-backed packages
            LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}";
          };
        };
      }
    );
}
