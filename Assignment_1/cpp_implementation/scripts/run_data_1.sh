#!/bin/bash
# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include the build directory
export PYTHONPATH="$PROJECT_ROOT/build/bindings:$PYTHONPATH"

# Run the training script
python3 "$PROJECT_ROOT/python/train.py" --dataset "$PROJECT_ROOT/../data_1" --epochs 1
