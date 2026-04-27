
# Submission Setup Guide - GNR 638 Project

This guide explains how to set up the environment and run the Geospatial Image Stitching & Analysis pipeline.

## 1. Environment Setup
You can create the required environment using the provided `requirements.txt`.

```bash
# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Note: For the final Kaggle submission, we use pre-downloaded `.whl` files to ensure offline capability.

## 2. Project Structure
- `stitcher.py`: Contains the `MapStitcher` class which reconstructs the map from patches.
- `vqa_engine.py`: Contains the `VQAEngine` class for multimodal inference on the map.
- `run_inference.py`: The main entry point that executes the full pipeline.
- `requirements.txt`: List of python dependencies.

## 3. How to Run
To perform the full reconstruction and generate answers for `test.csv`:

```bash
python3 run_inference.py
```

This will:
1. Load patches from the `./patches/` directory.
2. Reconstruct the full map as `reconstructed_map.png`.
3. Load the VQA model (or use mock mode if GPU/Weights are unavailable).
4. Save the results to `submission.csv`.

## 4. Hardware Requirements
- Final inference is optimized for **48GB L40s GPU**.
- Stitching takes approximately 2 minutes for a 225-patch grid.
- Total runtime is well within the 1-hour competition limit.
