#!/bin/bash
# Train one or more YOLO models.
#
# Usage:
#   bash bash_scripts/train.sh --data-dir /path/to/data
#   bash bash_scripts/train.sh --data-dir /path/to/data yolov9e
#   bash bash_scripts/train.sh --data-dir /path/to/data yolov8n yolov9e
#
# /path/to/data must contain:
#   train/               (tomogram slice directories)
#   train_labels.csv

set -e

ALL_MODELS=("yolov8n" "yolov8s" "yolov8m" "yolov8l" "yolov8x"
            "yolov9t" "yolov9s" "yolov9m" "yolov9c" "yolov9e")

DATA_DIR=""
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) SELECTED_MODELS+=("$1"); shift ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "Error: --data-dir is required"
    echo "Usage: $0 --data-dir /path/to/data [model_name ...]"
    exit 1
fi

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

for model in "${SELECTED_MODELS[@]}"; do
    echo "=== Training: $model ==="
    uv run --directory "$PROJECT_ROOT" python py_files/train_model.py "$model" --data-dir "$DATA_DIR"
done

echo "All models trained!"
