# Artificial Neural Networks — Scientific Report

## Galala University

### CNN & RNN for Handwritten Recognition Using the MNIST Dataset

**Course:** Artificial Neural Networks
**Institution:** Galala University
**Submitted to:** [abdgyd.fathy@GU.edu.eg](mailto:abdgyd.fathy@GU.edu.eg)
**Discussion Date:** Wednesday, 13 May 2026


# Part A — Scientific Study: Handwritten Recognition

# 1. Introduction

Handwritten recognition is the task of identifying handwritten characters or digits from image or sequence input. It is one of the classic benchmark problems in deep learning and serves as an ideal application to demonstrate both Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

* **CNN** excels at spatial feature extraction from images — detecting strokes, curves, and edges.
* **RNN** excels at sequential pattern modeling — pen trajectory over time and left-to-right character sequences.

Together, they form the backbone of modern OCR (Optical Character Recognition) systems such as those used in Google's document scanning, bank cheque processing, and postal code recognition.

---

# 2. Dataset — MNIST

The MNIST database is the standard benchmark for handwritten digit recognition.

## Dataset Characteristics

* 70,000 grayscale images of handwritten digits (0–9)
* Image size: 28×28 pixels, single channel
* Training set: 60,000 images
* Test set: 10,000 images
* Classes: 10 (digits 0 through 9)
* Achievable accuracy:

  * CNN ≈ 99.3%
  * RNN/LSTM ≈ 98.5%

---

# 3. CNN Approach for Handwritten Recognition

A Convolutional Neural Network processes the entire 28×28 image simultaneously, learning a spatial hierarchy of features — from low-level edges to high-level digit shapes.

## 3.1 CNN Architecture

```text
Input (28×28×1)
 → Conv2D(32 filters, 3×3) → ReLU → MaxPool(2×2)
 → Conv2D(64 filters, 3×3) → ReLU → MaxPool(2×2)
 → Flatten
 → Dense(128) → ReLU → Dropout(0.5)
 → Dense(10) → Softmax
```

The CNN learns spatial hierarchies:

* First convolution layer:

  * Detects edges, curves, and basic shapes.
* Second convolution layer:

  * Detects complex structures such as loops in digit “0” or vertical strokes in digit “1”.

---

# 4. RNN Approach for Handwritten Recognition

Each 28×28 image is treated as a sequence of 28 rows, where each row is a vector of 28 pixel values. The RNN reads the image top-to-bottom, effectively treating the image as a time series.

## 4.1 RNN/LSTM Architecture

```text
Input → Reshape to (28 time steps × 28 features)
 → LSTM(128 units, 2 layers)
 → Dense(64) → ReLU
 → Dense(10) → Softmax
```

The LSTM (Long Short-Term Memory) variant is used to overcome the vanishing gradient problem that affects vanilla RNNs when processing long sequences.

---

# Part B — Numerical Examples: Training Process

# Section 1 — CNN Numerical Examples

# CNN Example 1 — Convolution Operation

## Setup

A 4×4 input image and a 3×3 edge-detector filter (kernel).

### Input Image (I)

```text
1 0 1 0
0 1 0 1
1 0 1 0
0 1 0 1
```

### Filter (K) — Horizontal Edge Detector

```text
1  0 -1
1  0 -1
1  0 -1
```

## Output Size Formula

\text{Output Size} = \frac{(4-3)}{1}+1 = 2

## Convolution Computation

### O[0,0] — Top-left 3×3 patch

```text
Patch:          Filter:         Product:

1 0 1           1  0 -1         1  0 -1
0 1 0     ×     1  0 -1   =     0  0  0
1 0 1           1  0 -1         1  0 -1
```

Sum:

```text
(1 + 0 -1) + (0 + 0 + 0) + (1 + 0 -1) = 0
```

## Result — Feature Map

```text
O = [ [0, 0],
      [0, 0] ]
```

The symmetric checker pattern produces zero response to this edge detector, which is the expected result.

---

# CNN Example 2 — ReLU Activation + MaxPooling

## Raw Feature Map (Pre-Activation)

```text
-3  2  0 -1
 4 -5  3  2
-1  0 -2  6
 3 -1  4 -3
```

## Step 1 — ReLU Activation

f(x)=\max(0,x)

### After ReLU

```text
0 2 0 0
4 0 3 2
0 0 0 6
3 0 4 0
```

## Step 2 — MaxPooling

Using a 2×2 window with stride = 2.

### Pooling Results

```text
Top-left  : max(0,2,4,0) = 4
Top-right : max(0,0,3,2) = 3
Bottom-left : max(0,0,3,0) = 3
Bottom-right: max(0,6,4,0) = 6
```

### Pooled Output

```text
4 3
3 6
```

Spatial dimensions are reduced while preserving dominant features.

---

# CNN Example 3 — Fully Connected Layer + Softmax + Loss

## Setup

```text
x = [4, 3, 3, 6]

W = [
 [ 0.2, -0.1,  0.4,  0.1],
 [-0.3,  0.5,  0.2, -0.2],
 [ 0.1,  0.3, -0.1,  0.5]
]

b = [0.1, -0.2, 0.3]
```

## Linear Combination

z = Wx + b

### Computation

```text
z1 = 0.2(4) + (-0.1)(3) + 0.4(3) + 0.1(6) + 0.1 = 2.4

z2 = (-0.3)(4) + 0.5(3) + 0.2(3) + (-0.2)(6) - 0.2 = -0.5

z3 = 0.1(4) + 0.3(3) + (-0.1)(3) + 0.5(6) + 0.3 = 4.3
```

### Logits

```text
z = [2.4, -0.5, 4.3]
```

## Softmax

\text{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}

### Exponentials

```text
exp(2.4)  = 11.02
exp(-0.5) = 0.607
exp(4.3)  = 73.70

Sum = 85.33
```

### Probabilities

```text
P(class 0) = 11.02 / 85.33 = 0.129
P(class 1) = 0.607 / 85.33 = 0.007
P(class 2) = 73.70 / 85.33 = 0.864
```

Predicted class: **Class 2**

## Cross-Entropy Loss

L=-\log(0.864)\approx0.146

---

# Section 2 — RNN Numerical Examples

# RNN Example 1 — Single RNN Cell Forward Pass

## Setup

Binary sequence classification using sequence:

```text
[1, 0, 1]
```

### Weight Matrices

```text
Wxh = [[0.5],
       [0.3]]

Whh = [[0.1, 0.4],
       [0.2, 0.1]]

bh = [0, 0]
h0 = [0, 0]
```

## Time Step t = 1

```text
x1 = 1
z = [0.5, 0.3]
```

## Hidden State

h_t=\tanh(z_t)

```text
h1 = tanh([0.5, 0.3]) = [0.462, 0.291]
```

## Time Step t = 2

```text
x2 = 0

z[0] = 0.1×0.462 + 0.4×0.291 = 0.163
z[1] = 0.2×0.462 + 0.1×0.291 = 0.122

h2 = tanh([0.163, 0.122]) = [0.161, 0.121]
```

## Time Step t = 3

```text
x3 = 1

z[0] = 0.5 + (0.1×0.161 + 0.4×0.121) = 0.565
z[1] = 0.3 + (0.2×0.161 + 0.1×0.121) = 0.344

h3 = tanh([0.565, 0.344]) = [0.511, 0.334]
```

### Final Hidden State

```text
h3 = [0.511, 0.334]
```

---

# RNN Example 2 — Output Layer + BCE Loss

## Output Computation

```text
Wo = [0.6, -0.4]
bo = 0.1
```

### Linear Output

```text
z_out = 0.6×0.511 + (-0.4)×0.334 + 0.1
      = 0.273
```

## Sigmoid Activation

\hat{y}=\sigma(z)=\frac{1}{1+e^{-z}}

```text
ŷ = σ(0.273) = 0.568
```

Predicted probability:

```text
56.8% → Class 1
```

## Binary Cross-Entropy Loss

L=-\log(0.568)\approx0.566

---

# RNN Example 3 — Backpropagation Through Time (BPTT)

BPTT unfolds the RNN across time and accumulates gradients from every time step.

## Setup

```text
Wxh = 0.5
Whh = 0.3
Wo  = 0.8

η = 0.1
h0 = 0
```

## Forward Pass

```text
z1 = 0.5×1 + 0.3×0 = 0.5
h1 = tanh(0.5) = 0.462

z2 = 0.5×0 + 0.3×0.462 = 0.139
h2 = tanh(0.139) = 0.138

ŷ = σ(0.8×0.138) = 0.528

Loss = -log(0.528) = 0.639
```

## Backward Pass — Output Layer

```text
∂L/∂ŷ = -1/ŷ = -1.894

∂ŷ/∂z_out = ŷ(1-ŷ)
          = 0.528×0.472
          = 0.249

∂L/∂z_out = -1.894 × 0.249
           = -0.472
```

## Weight Update

```text
∂L/∂Wo = -0.472 × 0.138 = -0.065

Wo_new = 0.8 - 0.1×(-0.065)
       = 0.807
```

## Gradient Through Time

```text
∂L/∂h2 = -0.472 × 0.8 = -0.378

∂h2/∂z2 = 1 - tanh²(0.139)
         = 0.981

∂L/∂z2 = -0.378 × 0.981
        = -0.371
```

## Final Weight Updates

```text
∂L/∂Wxh = -0.087
Wxh_new = 0.509

∂L/∂Whh = -0.171
Whh_new = 0.317
```

BPTT accumulates gradients across all time steps while propagating error backward through the unrolled network.

---

# Section 3 — CNN vs RNN Comparison Summary

| Aspect            | CNN                        | RNN/LSTM                  |
| ----------------- | -------------------------- | ------------------------- |
| Input Type        | 2D Spatial Images          | Sequential / Time-Series  |
| Key Operation     | Convolution + Pooling      | Hidden State Recurrence   |
| Parameter Sharing | Across Spatial Locations   | Across Time Steps         |
| Strength          | Spatial Feature Extraction | Temporal Pattern Modeling |
| Weakness          | No Temporal Memory         | Vanishing Gradient        |
| MNIST Processing  | Reads Full Image           | Reads Row-by-Row          |
| Typical Accuracy  | ~99.3%                     | ~98.5%                    |

---

# Project — Python Implementation

The accompanying Python program (`ann_project.py`) implements the complete CNN and RNN handwritten recognition pipeline using PyTorch.

The implementation supports both CPU and GPU execution.

---

# Running the Code

```bash
# Install dependencies
pip install torch torchvision matplotlib scikit-learn

# Run the project
python ann_project.py
```

---

# Output Files Generated

* `training_curves.png` — Loss and accuracy curves
* `comparison.png` — CNN vs RNN accuracy comparison
* `cm_cnn.png` — CNN confusion matrix
* `cm_rnn_lstm.png` — RNN/LSTM confusion matrix
* `predictions_cnn.png` — CNN predictions
* `predictions_rnn_lstm.png` — RNN predictions

---

# Expected Results

* CNN validation accuracy: approximately 96–99%
* RNN/LSTM validation accuracy: approximately 95–98%
* Numerical examples printed automatically at startup

---

# Conclusion

This report demonstrated two complementary deep learning approaches for handwritten digit recognition.

* CNN achieved superior accuracy on the static MNIST image classification task by exploiting two-dimensional spatial structure through convolution and pooling operations.
* RNN/LSTM provided a sequential perspective by processing the image row-by-row as a time series.
* Combined CNN+RNN architectures (CRNN) are widely used in modern OCR systems including document scanning, automated banking systems, and intelligent text recognition pipelines.

The numerical examples in Part B provided detailed derivations of:

* Forward propagation
* Activation functions
* Softmax classification
* Cross-entropy loss
* Backpropagation Through Time (BPTT)

These examples establish a strong mathematical foundation for understanding how neural networks learn from data.

---

# References

1. Gradient-Based Learning Applied to Document Recognition
2. Long Short-Term Memory
3. Deep Residual Learning for Image Recognition
4. [MNIST Database Official Page](http://yann.lecun.com/exdb/mnist/?utm_source=chatgpt.com)
5. [PyTorch Documentation](https://pytorch.org/docs/stable/?utm_source=chatgpt.com)
