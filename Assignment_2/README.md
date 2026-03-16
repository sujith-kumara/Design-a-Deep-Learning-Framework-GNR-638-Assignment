# GNR638: Coding Assignment 2
## Pre-trained CNN Representation Transfer and Robustness Analysis

[cite_start]This repository contains the implementation and technical report for an investigative study into the representation transfer capabilities, fine-tuning efficiency, and robustness of pre-trained Convolutional Neural Networks (CNNs)[cite: 151]. [cite_start]We evaluate three distinct architectures—**ResNet-50**, **EfficientNet-B0**, and **DenseNet-121**—on the **Aerial Images Dataset (AID)**[cite: 152].

---

## 🛰️ Project Overview
[cite_start]The objective is to systematically analyze representation transfer, fine-tuning strategies, data efficiency under few-shot settings, and robustness to input corruption[cite: 7]. [cite_start]The analysis covers five mandatory experimental scenarios[cite: 40]:

1.  [cite_start]**Linear Probe Transfer:** Evaluating frozen backbones as feature extractors[cite: 41, 186].
2.  [cite_start]**Fine-Tuning Strategies:** Comparing full fine-tuning, last-block adaptation, and selective 20% unfreezing[cite: 54, 263].
3.  [cite_start]**Few-Shot Learning:** Testing data efficiency at 100%, 20%, and 5% training regimes[cite: 68, 399].
4.  [cite_start]**Corruption Robustness:** Assessing sensitivity to Gaussian noise, motion blur, and brightness shifts[cite: 81, 432].
5.  [cite_start]**Layer-Wise Probing:** Examining semantic abstraction evolution across network depth[cite: 97, 453].

---

##  Key Findings
* [cite_start]**Best Extractor:** **DenseNet-121** achieved the highest validation accuracy of **91.02%** in linear probing, demonstrating superior inductive bias for aerial imagery[cite: 191, 277].
* [cite_start]**Robustness Leader:** **ResNet-50** showed the most stable behavior under input corruptions, particularly Gaussian noise, likely due to its residual learning framework[cite: 436, 448].
* [cite_start]**The Robustness Trade-off:** While **DenseNet-121** performed best on clean data, it was the most sensitive to noise; its dense feature reuse mechanism acted as an "error amplifier" for corrupted activations[cite: 630, 632].
* [cite_start]**Few-Shot Efficiency:** **DenseNet-121** demonstrated the strongest data efficiency, maintaining a validation accuracy of **66.28%** even with only 5% of training data[cite: 405, 413].

---

##  Model Efficiency Metrics
[cite_start]We report the computational complexity for each backbone used in the study[cite: 180]:

| Model | Parameters (M) | MACs | FLOPs (Est.) |
| :--- | :---: | :---: | :---: |
| **EfficientNet-B0** | 4.05 | 390.87 MMac | 781.74 MFLOPS |
| **ResNet-50** | 23.57 | 4.13 GMac | 8.26 GFLOPS |
| **DenseNet-121** | 6.98 | 2.87 GMac | 5.74 GFLOPS |

[cite_start]*(Metrics cited from Table 1 [cite: 181, 183])*

---

##  Tech Stack & Requirements
* [cite_start]**Language:** Python 3.12 [cite: 127, 647]
* [cite_start]**Framework:** PyTorch 2.2.0 [cite: 647]
* [cite_start]**Libraries:** TIMM (HuggingFace), Matplotlib, Scikit-learn [cite: 31, 647]
* [cite_start]**Hardware:** NVIDIA T4 GPU (16 GB VRAM) [cite: 644]
* [cite_start]**OS:** Tested on Linux (Google Colab environment) [cite: 128, 643]

---

##  How to Reproduce
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/GNR638_Assignment_2.git](https://github.com/your-username/GNR638_Assignment_2.git)
    cd GNR638_Assignment_2
    ```
2.  **Dataset Setup:**
    Edit the file directory to training set
3.  **Run Experiments:**
    Run notebooks
---

##  Contributors
* [cite_start]**Sujith A** (25d2035) [cite: 148]
* [cite_start]**Anuroop C K** (24M0327) [cite: 148]
