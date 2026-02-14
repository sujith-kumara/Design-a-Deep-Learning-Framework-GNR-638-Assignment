# GNR 638 Assignment 1: Deep Learning Framework

This is a custom deep learning framework built from scratch with a C++ backend and Python bindings.

## 🚀 How to Run (Local)

### 1. Build the Framework
```bash
mkdir build && cd build
cmake .. && make
cd ..
```

### 2. Run Training
Use the provided scripts for easy execution:
- **Dataset 1:** `./scripts/run_data_1.sh`
- **Dataset 2:** `./scripts/run_data_2.sh`

### 3. Run Evaluation
- **Dataset 1:** `./scripts/evaluate_data_1.sh`
- **Dataset 2:** `./scripts/evaluate_data_2.sh`

---

## 📓 How to Run (Jupyter / Google Colab)

1. Open **`training_demo.ipynb`**.
2. Run the cells sequentially to:
   - Check system information.
   - Build the C++ framework.
   - Execute a mini-verification training.

---

## 🛠️ Prerequisites
```bash
pip install numpy opencv-python-headless pybind11
```

## 📊 Model Performance (2-Conv)
- **Epoch Time:** ~40 minutes
- **Train Accuracy:** ~76% (Dataset 1)
- **Target Epoch Limit:** < 3 Hours (Met)
