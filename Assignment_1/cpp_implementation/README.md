# GNR 638 Assignment 1: Deep Learning Framework

This is a custom deep learning framework built from scratch with a C++ backend and Python bindings.

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
*Note: If you have multiple Python versions, you can specify your python path during cmake:*
*`cmake -DPython3_EXECUTABLE=$(which python3) ..`*

### 2. Run Training
Use the provided scripts from the `cpp_implementation` folder:
- **Dataset 1:** `cpp_implementation/scripts/run_data_2.sh`
- **Dataset 2:** `cpp_implementation/scripts/run_data_2.sh`
- **Interactive:** `cpp_implementation/setup_and_run.sh`

### 3. Run Evaluation
- **Dataset 1:** `cpp_implementation/scripts/evaluate_data_1.sh`
- **Dataset 2:** `cpp_implementation/scripts/evaluate_data_2.sh`

### 4. Visualization
- **Generate Feature Maps:** `python3 cpp_implementation/python/visualize.py --image path/to/image.png`
