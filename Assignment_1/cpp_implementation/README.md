# GNR 638 Assignment 1: Deep Learning Framework

This is a custom deep learning framework built from scratch with a C++ backend and Python bindings (No NumPy, No Torch).

## Project Structure
- `Assignment_1/data_x`: Datasets
- `Assignment_1/cpp_implementation`: Core framework, bindings, and scripts

## How to Run (from the Project Root)

### 1. Build the Framework
Run these commands from your terminal (starting in `assignment_github_folder`):
```bash
cd Assignment_1/cpp_implementation
mkdir -p build && cd build
cmake ..
make -j4
cd ../../../  # Back to root
```

### 2. Run Training
Available scripts (run from the root directory):
- **Dataset 1:** `./Assignment_1/cpp_implementation/scripts/run_data_1.sh`
- **Dataset 2:** `./Assignment_1/cpp_implementation/scripts/run_data_2.sh`
- **Interactive:** `./Assignment_1/cpp_implementation/setup_and_run.sh`

### 3. Run Evaluation
- **Dataset 1:** `./Assignment_1/cpp_implementation/scripts/evaluate_data_1.sh`
- **Dataset 2:** `./Assignment_1/cpp_implementation/scripts/evaluate_data_2.sh`

### 4. Visualization
To visualize feature maps (requires saved weights):
```bash
python3 Assignment_1/cpp_implementation/python/visualize.py \
    --image "Assignment_1/data_1/0/Image 1.png" \
    --weights Assignment_1/cpp_implementation/cnn_weights.pkl \
    --output Assignment_1/cpp_implementation/visualizations
```

---

## 🛠️ Prerequisites
- Python 3.12+
- `opencv-python-headless` (for image loading)
- `pybind11` (for C++ bindings)
- `cmake` (for building)



