# GNR 638 Assignment 1: Deep Learning Framework

This is a custom deep learning framework built from scratch with a C++ backend and Python bindings (No NumPy, No Torch).

## Project Structure
- `Assignment_1/data_x`: Datasets
- `Assignment_1/cpp_implementation`: Core framework, bindings, and scripts

## How to Run

### 1. Build the Framework
All build commands must be run from within the `cpp_implementation` folder:
```bash
cd Assignment_1/cpp_implementation
mkdir -p build && cd build
cmake ..
make -j4
cd ..
```

### 2. Run Training
Run these from the `Assignment_1/cpp_implementation` directory:
- **Dataset 1:** `./scripts/run_data_1.sh`
- **Dataset 2:** `./scripts/run_data_2.sh`
- **Setup & Run:** `./setup_and_run.sh`

### 3. Run Evaluation
- **Dataset 1:** `./scripts/evaluate_data_1.sh`
- **Dataset 2:** `./scripts/evaluate_data_2.sh`

### 4. Visualization
To visualize feature maps (requires saved weights):
```bash
python3 python/visualize.py --image ../data_1/Abyssinian/Image_1.png --weights python/cnn_weights.pkl
```

---

## 🛠️ Prerequisites
- Python 3.12+
- `opencv-python-headless` (for image loading)
- `pybind11` (for C++ bindings)
- `cmake` (for building)

*Note: NumPy is strictly prohibited and has been completely removed.*

## 📊 Model Performance (2-Conv)
- **Epoch Time:** ~40-50 minutes (on CPU)
- **Train Accuracy:** ~75-80% (Dataset 1)
- **Target Epoch Limit:** < 3 Hours (Met)
