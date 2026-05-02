#!/bin/bash
set -e

echo "========================================"
echo "🚀 Starting Server Setup for VQA_OSM..."
echo "========================================"

echo "📥 1. Fetching repository files..."
REPO_URL="https://github.com/anuroopck/VQA_OSM.git"

git clone "$REPO_URL" .temp_repo

rm -rf .temp_repo/.git
cp -a .temp_repo/. ./
rm -rf .temp_repo

echo "✅ Repository files moved into the current directory."

echo "📦 2. Building Conda Environment from environment.yml..."
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
    echo "✅ Conda environment 'gnr_project_env' created successfully!"
else
    echo "❌ Error: environment.yml not found."
    exit 1
fi

echo "🧠 3. Downloading LLaVA Model directly via Conda Environment..."
conda run -n gnr_project_env pip install huggingface_hub -q

# Run Python directly from bash (No 'cat' or temporary files needed!)
conda run -n gnr_project_env python -c "
import os
from huggingface_hub import snapshot_download

MODEL_ID = 'llava-hf/llava-v1.6-mistral-7b-hf'
SAVE_DIR = './llava-v1.6-mistral-offline'

os.makedirs(SAVE_DIR, exist_ok=True)
print('Starting download for ' + MODEL_ID + '... This may take a while.')

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    local_dir_use_symlinks=False,
    ignore_patterns=['*.bin', '*.h5', '*.msgpack']
)
print('\n✅ Download complete! Files saved to: ' + SAVE_DIR)
"

echo "========================================"
echo "🎉 Setup Complete!"
echo "========================================"
