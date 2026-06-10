#!/usr/bin/bash
# Create the dvanalysis conda environment on the server.
#
# Usage:
#   bash setup_env.sh
#
# Prerequisites:
#   - conda is available in PATH
#   - Run from the DVA_library root directory

set -e

ENV_NAME="dvanalysis"

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

conda activate $ENV_NAME

echo "Installing dvanalysis in editable mode with dev dependencies"
pip install -e ".[dev]"

echo "Verifying installation"
python -c "import dvanalysis; print(f'dvanalysis {dvanalysis.__version__} installed successfully')"
python -c "from dvanalysis.validation.runner import run_hybrid_evaluation; print('Validation framework OK')"

echo "Done. Activate with: conda activate $ENV_NAME"
