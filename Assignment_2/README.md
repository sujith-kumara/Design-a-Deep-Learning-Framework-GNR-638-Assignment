# GNR638: Coding Assignment 2
## Pre-trained CNN Representation Transfer Analysis

This repository contains the implementation and technical report for an investigative study into the representation transfer capabilities, fine-tuning efficiency, and robustness of pre-trained Convolutional Neural Networks. We evaluate three distinct architectures—**ResNet-50**, **EfficientNet-B0**, and **DenseNet-121**—on the **Aerial Images Dataset (AID)**.

---

## Project Overview
The objective is to systematically analyze representation transfer, fine-tuning strategies, data efficiency under few-shot settings, and robustness to input corruption. The analysis covers five mandatory experimental scenarios:

1.  **Linear Probe Transfer:** Evaluating frozen backbones as feature extractors.
2.  **Fine-Tuning Strategies:** Comparing full fine-tuning, last-block adaptation, and selective 20 percent unfreezing.
3.  **Few-Shot Learning:** Testing data efficiency at 100 percent, 20 percent, and 5 percent training regimes.
4.  **Corruption Robustness:** Assessing sensitivity to Gaussian noise, motion blur, and brightness shifts.
5.  **Layer-Wise Probing:** Examining semantic abstraction evolution across network depth.

---

## Key Findings
* **Best Extractor:** **DenseNet-121** achieved the highest validation accuracy of **91.02 percent** in linear probing, demonstrating superior inductive bias for aerial imagery.
* **Robustness Leader:** **ResNet-50** showed the most stable behavior under input corruptions, particularly Gaussian noise, likely due to its residual learning framework.
* **The Robustness Trade-off:** While **DenseNet-121** performed best on clean data, it was the most sensitive to noise; its dense feature reuse mechanism acted as an "error amplifier" for corrupted activations.
* **Few-Shot Efficiency:** **DenseNet-121** demonstrated the strongest data efficiency, maintaining a validation accuracy of **66.28 percent** even with only 5 percent of training data.

---

## Model Efficiency Metrics
We report the computational complexity for each backbone used in the study:

| Model | Parameters (Millions) | MACs | FLOPs (Estimated) |
| :--- | :---: | :---: | :---: |
| **EfficientNet-B0** | 4.05 | 390.87 MMac | 781.74 MFLOPS |
| **ResNet-50** | 23.57 | 4.13 GMac | 8.26 GFLOPS |
| **DenseNet-121** | 6.98 | 2.87 GMac | 5.74 GFLOPS |

---

## Tech Stack and Requirements
* **Language:** Python 3.12
* **Framework:** PyTorch 2.2.0
* **Libraries:** TIMM, Matplotlib, Scikit-learn
* **Hardware:** NVIDIA T4 GPU with 16 GB VRAM
* **OS:** Tested on Linux (Google Colab environment)

---

## How to Reproduce
1.  **Clone the Repository:**
    Use the git clone command with the repository link.
2.  **Dataset Setup:**
    Place the **AID dataset** in the same directory and change the adress in the notebooks.
3.  **Run Experiments:**
    Run the notebooks.

---

## Contributors
* **Sujith A** (25d2035)
* **Anuroop C K** (24M0327)
