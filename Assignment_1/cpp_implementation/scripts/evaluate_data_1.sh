#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH to include the project and build directory
export PYTHONPATH="$PROJECT_ROOT/build/bindings:$PYTHONPATH"

# Run the evaluation script on the 10% test split of data_1
# (Assumes weights were trained with the default 10% test split)
python3 "$PROJECT_ROOT/python/evaluate.py" \
    --dataset "$PROJECT_ROOT/../data_1" \
    --weights "$PROJECT_ROOT/python/cnn_weights.pkl" \
    --test_split 0.1 \
    --seed 42
