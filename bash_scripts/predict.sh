#!/bin/bash
# Run inference and generate 3D visualizations for a trained model.
#
# Usage:
#   bash bash_scripts/predict.sh yolov9e

set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 yolov9e"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Predicting with: $1 ==="
uv run --directory "$PROJECT_ROOT" python py_files/predict_model.py "$1"

echo "Done!"
