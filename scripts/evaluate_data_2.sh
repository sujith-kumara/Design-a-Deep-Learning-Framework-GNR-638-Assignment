#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH to include the project and build directory
export PYTHONPATH="$PROJECT_ROOT/build/bindings:$PYTHONPATH"

# Run the evaluation script on the 10% test split of data_2
python3 "$PROJECT_ROOT/python/evaluate.py" \
    --dataset "$PROJECT_ROOT/data_2" \
    --weights "$PROJECT_ROOT/python/cnn_weights.npz" \
    --test_split 0.1 \
    --seed 42
