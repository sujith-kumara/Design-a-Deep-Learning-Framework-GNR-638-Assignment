# VQA_OSM: Visual Question Answering on OpenStreetMap

An offline, production-ready inference pipeline for map stitching and Visual Question Answering (VQA) using OpenStreetMap data. 

This project uses a custom computer vision map-stitching algorithm alongside an offline, quantized deployment of the **LLaVA 1.6 Mistral 7B** model to analyze geographical patches and answer spatial questions—entirely without internet access.

---

##  Quick Start (Automated Setup)

This repository includes an automated setup script for env and inference.py.

### 1. Prerequisites
* **OS:** Linux (Ubuntu recommended)
* **Hardware:** NVIDIA GPU (Recommended: 16GB+ VRAM for 8-bit, or 8GB+ for 4-bit quantization)
* **Software:** Conda or Miniconda installed

### 2. Run the Setup Script
Run the following commands in your server terminal to fetch the code and trigger the setup:
```bash
# For setting up env, run :
bash setup.bash

# Run the automated setup
bash setup.sh
