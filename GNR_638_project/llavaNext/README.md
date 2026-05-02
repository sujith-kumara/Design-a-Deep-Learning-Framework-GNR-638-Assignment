# VQA_OSM: Visual Question Answering on OpenStreetMap

An offline, production-ready inference pipeline for map stitching and Visual Question Answering (VQA) using OpenStreetMap data. 

This project uses a custom computer vision map-stitching algorithm alongside an offline deployment of the **LLaVA 1.6 Mistral 7B** model to analyze geographical patches and answer spatial questions—entirely without internet access.

---

## Automated Setup

This project includes an automated setup script designed for Linux servers. It will build the Conda environment and safely download the multi-gigabyte LLaVA model into a local directory for offline inference.

### 1. Prerequisites
* **OS:** Linux (Ubuntu recommended)
* **Hardware:** NVIDIA GPU (Recommended: 48GB VRAM for unquantized FP16, or 16GB+ for quantized inference)
* **Software:** Conda or Miniconda installed

### 2. Run the Setup Script
bash setup.sh

### 3. Activate env and run the code
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>

