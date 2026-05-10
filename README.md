# Handwritten Digit Recognition with CNN & RNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Course](https://img.shields.io/badge/Course-ANN-Galala%20University-purple)](https://gu.edu.eg)

> **Artificial Neural Networks Course Assignment**  
> Galala University — ANN Course  
> Discussion Date: Wednesday, 13 May 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Part A — Scientific Study](#part-a--scientific-study)
- [Part B — Numerical Examples](#part-b--numerical-examples)
- [Project — Python Implementation](#project--python-implementation)
- [Results](#results)
- [Team Members](#team-members)
- [License](#license)
- [References](#references)

---

## Overview

This repository contains a complete solution for the **Artificial Neural Networks** course assignment at **Galala University**. The project demonstrates two complementary deep learning approaches for **Handwritten Digit Recognition** using the classic **MNIST dataset**:

| Approach | Architecture | Strength |
|----------|-----------|----------|
| **CNN** | Convolutional Neural Network | Spatial feature extraction (edges, curves, strokes) |
| **RNN/LSTM** | Recurrent Neural Network with Long Short-Term Memory | Sequential pattern modeling (row-by-row image processing) |

Both models achieve **~98-99% accuracy** on MNIST, with CNN slightly outperforming RNN due to its native 2D spatial processing capabilities.

---

## Project Structure

```
├── 📄 README.md                          # This file
├── 📊 ANN_Presentation.pptx              # Discussion presentation
├── 📄 ANN_Scientific_Report.docx         # Full scientific report
├── 🐍 ann_project.py                     # Complete Python implementation
├── 📁 data/                              # MNIST dataset (auto-downloaded)
├── 📁 outputs/                           # Generated visualizations
│   ├── training_curves.png
│   ├── comparison.png
│   ├── cm_cnn.png
│   ├── cm_rnn_lstm.png
│   ├── predictions_cnn.png
│   └── predictions_rnn_lstm.png
└── 📄 requirements.txt                   # Python dependencies
```

---

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- torchvision
- matplotlib
- scikit-learn
- numpy

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/handwritten-recognition-cnn-rnn.git
cd handwritten-recognition-cnn-rnn
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

---

## Usage

### Run the Complete Pipeline

```bash
python ann_project.py
```

This will:
1. Print **Part B — Numerical Examples** to console (CNN & RNN)
2. Download MNIST dataset automatically
3. Train both **CNN** and **RNN/LSTM** models
4. Evaluate and compare results
5. Generate all visualization plots

### Expected Output

```
====================================================================
  ANN ASSIGNMENT — HANDWRITTEN RECOGNITION
  CNN & RNN on MNIST Dataset
====================================================================

[CNN Example 1] Convolution Operation
...

  Training CNN
  ────────────────────────────────────────────────────────────────────
  Epoch  1/5  |  Train Loss: 0.2341  Acc: 92.5%  |  Val Loss: 0.1234  Acc: 96.2%
  ...
  ✓ Best Val Accuracy: 98.50%  |  Time: 45.2s

  Saved: training_curves.png
  Saved: comparison.png
  ...

  ALL DONE — Assignment Complete!
```

---

## Part A — Scientific Study

### Chosen Topic: Handwritten Digit Recognition (Option C)

#### Why This Topic?
Handwritten digit recognition is a foundational problem in computer vision and deep learning, serving as an ideal benchmark to demonstrate:
- **CNN capabilities**: 2D spatial feature hierarchy (edges → shapes → digits)
- **RNN capabilities**: Sequential temporal modeling (row-by-row scanning)

#### Dataset: MNIST

| Property | Value |
|----------|-------|
| Total Images | 70,000 |
| Training Set | 60,000 |
| Test Set | 10,000 |
| Image Size | 28 × 28 pixels (grayscale) |
| Classes | 10 (digits 0–9) |
| Channels | 1 (grayscale) |

#### Real-World Applications

- 🏦 **Banking**: Automated cheque processing, amount verification
- 📮 **Postal Services**: ZIP code recognition for mail sorting
- 📄 **Document Digitization**: Form data extraction, OCR pipelines
- 📱 **Mobile Apps**: Google Lens, document scanning apps

---

## Part B — Numerical Examples

### CNN Examples (3 Worked Examples)

#### Example 1: Convolution Operation
**Input**: 4×4 image | **Filter**: 3×3 vertical edge detector

```
Input:          Filter:          Feature Map (2×2):
1 0 1 0         1  0 -1         0  0
0 1 0 1         1  0 -1         0  0
1 0 1 0         1  0 -1
0 1 0 1
```

The symmetric checker pattern produces **zero response** to the vertical edge detector — an expected and intuitive result.

#### Example 2: ReLU + MaxPooling
**Raw Map** → **ReLU** (zero negatives) → **MaxPool 2×2** (select maxima)

```
Raw:            ReLU:           MaxPool (2×2):
-3  2  0 -1     0  2  0  0      4  3
 4 -5  3  2     4  0  3  2      3  6
-1  0 -2  6     0  0  0  6
 3 -1  4 -3     3  0  4  0
```

#### Example 3: Fully Connected + Softmax

**Input**: x = [4, 3, 3, 6]  
**Weights**: W (3×4), **Bias**: b (3)  
**Logits**: z = [2.4, −0.5, 4.3]  
**Softmax**: [12.9%, 0.7%, **86.4%**] → **Class 2 predicted**  
**Cross-Entropy Loss**: −log(0.864) = **0.146**

### RNN Examples (3 Worked Examples)

#### Example 1: Single RNN Cell Forward Pass
**Sequence**: [1, 0, 1] | **Hidden Size**: 2

| Time | Input | Hidden State h |
|------|-------|----------------|
| t=1 | x₁ = 1 | [0.462, 0.291] |
| t=2 | x₂ = 0 | [0.161, 0.121] |
| t=3 | x₃ = 1 | [0.511, 0.334] |

#### Example 2: Output Layer + Binary Cross-Entropy

Using final hidden state h₃:
- **Output**: z_out = 0.273 → ŷ = σ(0.273) = **0.568**
- **Prediction**: Class 1 (56.8% probability)
- **BCE Loss**: −log(0.568) = **0.566**

#### Example 3: Backpropagation Through Time (BPTT)

Scalar 2-step sequence [1, 0] with learning rate η = 0.1:

| Gradient | Value | Weight Update |
|----------|-------|---------------|
| ∂L/∂Wo | −0.065 | Wo: 0.8 → **0.807** |
| ∂L/∂Wxh | −0.087 | Wxh: 0.5 → **0.509** |
| ∂L/∂Whh | −0.171 | Whh: 0.3 → **0.317** |

---

## Project — Python Implementation

### CNN Architecture

```
Input (28×28×1)
    ↓
Conv2D(1→32, 3×3) + ReLU
    ↓
MaxPool2D(2×2)
    ↓
Conv2D(32→64, 3×3) + ReLU
    ↓
MaxPool2D(2×2)
    ↓
Dropout2D(0.25)
    ↓
Flatten (64×7×7 = 3136)
    ↓
Dense(3136 → 128) + ReLU + Dropout(0.5)
    ↓
Dense(128 → 10)
    ↓
Softmax
```

**Parameters**: ~422K

### RNN/LSTM Architecture

```
Input (28×28×1)
    ↓
Squeeze → (28 time steps × 28 features)
    ↓
LSTM(128 units, 2 layers, dropout=0.3)
    ↓
Last Time Step (128-dim)
    ↓
Dense(128 → 64) + ReLU + Dropout(0.4)
    ↓
Dense(64 → 10)
    ↓
Softmax
```

**Parameters**: ~170K

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 5 (configurable) |
| Loss Function | Cross-Entropy Loss |
| Scheduler | StepLR (step=3, γ=0.5) |
| Device | CUDA (if available) / CPU |

---

## Results

### Accuracy Comparison

| Model | Best Validation Accuracy | Speed |
|-------|------------------------|-------|
| **CNN** | ~99.2% | Faster |
| **RNN/LSTM** | ~98.5% | Slower (sequential) |

### Generated Visualizations

1. **training_curves.png** — Loss & accuracy curves for both models
2. **comparison.png** — Bar chart comparing best accuracy
3. **cm_cnn.png** — Confusion matrix (CNN predictions)
4. **cm_rnn_lstm.png** — Confusion matrix (RNN predictions)
5. **predictions_cnn.png** — Sample digit predictions (CNN)
6. **predictions_rnn_lstm.png** — Sample digit predictions (RNN)



## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). **Gradient-Based Learning Applied to Document Recognition**. *Proceedings of the IEEE*, 86(11), 2278–2324.

2. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*, 9(8), 1735–1780.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. *CVPR 2016*, 770–778.

4. **MNIST Database** — Yann LeCun, Corinna Cortes, Christopher J.C. Burges. [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

5. **PyTorch Documentation** — [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

---

## Acknowledgments

- **Galala University** — Faculty of Computers and Artificial Intelligence
- **Dr. Abdelghany Fathy** — Course instructor and guidance
- **PyTorch Team** — Deep learning framework
- **Yann LeCun et al.** — MNIST dataset creators

---

> 💡 **Note**: This repository is prepared for academic purposes as part of the ANN course requirements at Galala University.
