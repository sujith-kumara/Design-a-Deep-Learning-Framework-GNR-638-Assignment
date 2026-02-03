# DeepLearn Framework

A C++ deep learning framework with Python bindings and a custom autograd engine.

## Features
- **C++ Core**: Functional implementation of Tensors, Layers (Linear, Conv2D, MaxPool2D), and Optimizers (SGD).
- **Autograd Engine**: Automatic differentiation supports complex graphs.
- **Python Bindings**: Exposed via pybind11 for easy experimentation in Python.
- **Fast DataLoader**: Efficiently loads PNG datasets (e.g., CIFAR-like structures) using OpenCV.

## Prerequisites
- CMake (>= 3.10)
- Core development tools (g++, make)
- Python 3 with `numpy`, `opencv-python`, and `pybind11`

## Building the Framework
```bash
mkdir build && cd build
cmake ..
make -j4
```

## Running the Model

The framework currently supports training a CNN (Conv -> ReLU -> MaxPool -> Linear) on the provided datasets.

### 1. Training
The training script supports command-line arguments to specify the dataset, number of epochs, and output path for weights.

**Train on Dataset 1:**
```bash
python python/train.py --dataset data_1 --epochs 20 --save_path cnn_weights_data1.npz
```

**Train on Dataset 2:**
```bash
python python/train.py --dataset data_2 --epochs 20 --save_path cnn_weights_data2.npz
```

**General Usage:**
```bash
python python/train.py --dataset <DATASET_PATH> --epochs <NUM_EPOCHS> --save_path <OUTPUT_PATH>
```

### 2. Evaluation
After training, run the evaluation script to check accuracy on the dataset and measure inference performance.

```bash
python3 python/test.py
```

## Dataset Structure
The DataLoader expects datasets in a folder structure where each subdirectory represents a class name containing PNG images:
```text
data_x/
├── class_1/
│   ├── img1.png
│   └── ...
├── class_2/
│   ├── img1.png
│   └── ...
└── ...
```
