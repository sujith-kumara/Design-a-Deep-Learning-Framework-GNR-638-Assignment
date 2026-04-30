#!/usr/bin/env bash

# Exit immediately on errors, treat unset variables as errors, catch pipeline failures
set -e
set -u
set -o pipefail

echo "======================================================================"
echo "Starting Automated Setup for GNR Project"
echo "Target: Linux, CUDA 12.6, L40s (48GB VRAM)"
echo "======================================================================"

# ==============================================================================
# 1. SYSTEM DEPENDENCIES (Critical for OpenCV on headless Linux)
# ==============================================================================
echo "[1/3] Checking System Dependencies..."
if command -v apt-get &> /dev/null; then
    SUDO=""
    if command -v sudo &> /dev/null; then SUDO="sudo"; fi
    if ! dpkg -l | grep -q libgl1-mesa-glx; then
        echo "Installing libGL for OpenCV..."
        $SUDO apt-get update -yqq
        $SUDO apt-get install -yqq libgl1-mesa-glx libglib2.0-0
    else
        echo "System dependencies already met."
    fi
else
    echo "Warning: apt-get not found. Assuming system dependencies (libGL) are met."
fi

# ==============================================================================
# 2. CONDA ENVIRONMENT CREATION
# ==============================================================================
ENV_NAME="gnr_project_env"

echo "[2/3] Managing Conda Environment: $ENV_NAME..."
eval "$(conda shell.bash hook)"

if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment already exists. Updating just in case..."
    conda env update -f environment.yml --prune
else
    echo "Creating environment from environment.yml..."
    conda env create -f environment.yml
fi

# ==============================================================================
# 3. FINAL VERIFICATION
# ==============================================================================
echo "[3/3] Verifying Setup & CUDA availability..."
conda activate "$ENV_NAME"

python -c "
import torch
import cv2
import os

print(f'PyTorch Version: {torch.__version__}')
print(f'OpenCV Version: {cv2.__version__}')

if not os.path.exists('./llava-1.6-mistral-offline'):
    print('WARNING: Model folder ./llava-1.6-mistral-offline not found in current directory!')
else:
    print('Model folder found successfully.')

if torch.cuda.is_available():
    print(f'CUDA is Available! GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CRITICAL ERROR: CUDA is NOT available!')
    exit(1)
"

conda deactivate
echo "======================================================================"
echo "Setup Complete! System is ready for inference."
echo "======================================================================"
exit 0