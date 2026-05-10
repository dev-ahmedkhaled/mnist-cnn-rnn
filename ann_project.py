"""
=============================================================================
  ANN Assignment — Handwritten Digit Recognition using CNN and RNN
  Part A: Smart application using CNN & RNN on MNIST
  Part B: Numerical examples (see report)
  Project: Complete Python implementation
=============================================================================
  Requirements:
      pip install torch torchvision matplotlib scikit-learn numpy
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
EPOCHS      = 5           # increase to 10–15 for better accuracy
LR          = 0.001
NUM_CLASSES = 10
SUBSET_SIZE = 5000        # use a subset for faster demo (set None for full 60k)

print(f"Device: {DEVICE}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 0: NUMERICAL EXAMPLES (Part B — prints to console)
# ─────────────────────────────────────────────────────────────────────────────

def numerical_examples():
    print("\n" + "=" * 60)
    print("  PART B — NUMERICAL EXAMPLES")
    print("=" * 60)

    # ── CNN Example 1: Convolution ──────────────────────────────────────────
    print("\n[CNN Example 1] Convolution Operation")
    print("-" * 40)
    I = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [1,0,1,0],
                  [0,1,0,1]], dtype=float)
    K = np.array([[ 1, 0,-1],
                  [ 1, 0,-1],
                  [ 1, 0,-1]], dtype=float)
    out_size = I.shape[0] - K.shape[0] + 1
    feature_map = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            patch = I[i:i+3, j:j+3]
            feature_map[i, j] = np.sum(patch * K)
    print(f"Input:\n{I}")
    print(f"Filter (edge detector):\n{K}")
    print(f"Feature Map (after convolution):\n{feature_map}")

    # ── CNN Example 2: ReLU + MaxPool ──────────────────────────────────────
    print("\n[CNN Example 2] ReLU Activation + MaxPooling")
    print("-" * 40)
    raw = np.array([[-3, 2, 0,-1],
                    [ 4,-5, 3, 2],
                    [-1, 0,-2, 6],
                    [ 3,-1, 4,-3]], dtype=float)
    relu_out = np.maximum(0, raw)
    print(f"Raw feature map:\n{raw}")
    print(f"After ReLU:\n{relu_out}")
    # MaxPool 2×2
    pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            pooled[i, j] = relu_out[i*2:i*2+2, j*2:j*2+2].max()
    print(f"After MaxPool(2×2):\n{pooled}")

    # ── CNN Example 3: FC + Softmax ─────────────────────────────────────────
    print("\n[CNN Example 3] Fully Connected Layer + Softmax")
    print("-" * 40)
    x   = np.array([4, 3, 3, 6], dtype=float)
    W   = np.array([[ 0.2,-0.1, 0.4, 0.1],
                    [-0.3, 0.5, 0.2,-0.2],
                    [ 0.1, 0.3,-0.1, 0.5]], dtype=float)
    b   = np.array([0.1, -0.2, 0.3])
    z   = W @ x + b
    exp = np.exp(z)
    sm  = exp / exp.sum()
    loss = -np.log(sm[2])
    print(f"Input x: {x}")
    print(f"Logits z = Wx + b: {np.round(z, 3)}")
    print(f"Softmax probabilities: {np.round(sm, 3)}")
    print(f"Predicted class: {np.argmax(sm)}  (true label: 2)")
    print(f"Cross-Entropy Loss: {loss:.4f}")

    # ── RNN Example 1: Single Cell Forward Pass ────────────────────────────
    print("\n[RNN Example 1] Single RNN Cell — Forward Pass")
    print("-" * 40)
    Wxh = np.array([[0.5], [0.3]])   # (2,1)
    Whh = np.array([[0.1, 0.4],
                    [0.2, 0.1]])     # (2,2)
    bh  = np.zeros(2)
    h   = np.zeros(2)
    seq = [1, 0, 1]
    for t, xt in enumerate(seq):
        z = Wxh.flatten() * xt + Whh @ h + bh
        h = np.tanh(z)
        print(f"  t={t+1}, x={xt}, z={np.round(z,4)}, h={np.round(h,4)}")
    print(f"  Final hidden state h = {np.round(h,4)}")

    # ── RNN Example 2: Output + Loss ───────────────────────────────────────
    print("\n[RNN Example 2] Output Layer + Binary Cross-Entropy Loss")
    print("-" * 40)
    Wo     = np.array([0.6, -0.4])
    bo     = 0.1
    z_out  = Wo @ h + bo
    y_hat  = 1 / (1 + np.exp(-z_out))
    true_y = 1
    loss_rnn = -(true_y * np.log(y_hat) + (1-true_y) * np.log(1-y_hat))
    print(f"  h (from Ex.1) = {np.round(h,4)}")
    print(f"  z_out = {z_out:.4f}")
    print(f"  ŷ = σ(z_out) = {y_hat:.4f}")
    print(f"  BCE Loss (y=1): {loss_rnn:.4f}")

    # ── RNN Example 3: BPTT ────────────────────────────────────────────────
    print("\n[RNN Example 3] Backpropagation Through Time (BPTT)")
    print("-" * 40)
    # 2-step sequence, scalar hidden
    wxh_s, whh_s, wo_s = 0.5, 0.3, 0.8
    h0 = 0.0; x_seq = [1, 0]; eta = 0.1
    z1 = wxh_s * x_seq[0] + whh_s * h0;  h1 = np.tanh(z1)
    z2 = wxh_s * x_seq[1] + whh_s * h1;  h2 = np.tanh(z2)
    y_out = 1 / (1 + np.exp(-wo_s * h2))
    L_bptt = -np.log(y_out)  # true label = 1
    print(f"  Forward: h1={h1:.4f}, h2={h2:.4f}, ŷ={y_out:.4f}, L={L_bptt:.4f}")
    # Gradients
    dL_dzo   = y_out - 1          # d(BCE)/d(z_out) for sigmoid+BCE
    dL_dWo   = dL_dzo * h2
    dL_dh2   = dL_dzo * wo_s
    dL_dz2   = dL_dh2 * (1 - h2**2)
    dL_dh1   = dL_dz2 * whh_s
    dL_dz1   = dL_dh1 * (1 - h1**2)
    dL_dWxh  = dL_dz1 * x_seq[0] + dL_dz2 * x_seq[1]
    dL_dWhh  = dL_dz2 * h1
    print(f"  ∂L/∂Wo  = {dL_dWo:.4f}  →  Wo_new  = {wo_s  - eta*dL_dWo:.4f}")
    print(f"  ∂L/∂Wxh = {dL_dWxh:.4f} →  Wxh_new = {wxh_s - eta*dL_dWxh:.4f}")
    print(f"  ∂L/∂Whh = {dL_dWhh:.4f} →  Whh_new = {whh_s - eta*dL_dWhh:.4f}")

    print("\n" + "=" * 60)
    print("  NUMERICAL EXAMPLES COMPLETE")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_full = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform)
    test_full  = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform)
    if SUBSET_SIZE:
        train_data = Subset(train_full, range(SUBSET_SIZE))
        test_data  = Subset(test_full,  range(SUBSET_SIZE // 5))
    else:
        train_data, test_data = train_full, test_full

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training samples : {len(train_data)}")
    print(f"Test samples     : {len(test_data)}")
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CNNModel(nn.Module):
    """
    CNN for handwritten digit recognition.
    Input: (B, 1, 28, 28) grayscale images
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (B,32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (B,64,7,7)
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: RNN MODEL (LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class RNNModel(nn.Module):
    """
    LSTM-based RNN for handwritten digit recognition.
    Each 28×28 image is treated as 28 time steps of 28-feature vectors.
    Input: (B, 1, 28, 28) → reshape to (B, 28, 28)
    """
    def __init__(self, input_size=28, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        # x: (B, 1, 28, 28) → (B, 28, 28)
        x = x.squeeze(1)
        out, _ = self.lstm(x)       # out: (B, 28, hidden_size)
        last   = out[:, -1, :]     # take last time step
        return self.classifier(last)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4: TRAINING & EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, 100 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100 * correct / total, all_preds, all_labels


def train_model(model, name, train_loader, test_loader, epochs=EPOCHS):
    print(f"\n{'─'*60}")
    print(f"  Training {name}")
    print(f"{'─'*60}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc

        print(f"  Epoch {epoch:2d}/{epochs}  |  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.1f}%  |  "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.1f}%")

    elapsed = time.time() - t0
    print(f"\n  ✓ Best Val Accuracy: {best_acc:.2f}%  |  Time: {elapsed:.1f}s")

    # Final evaluation
    _, _, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\n  Classification Report ({name}):")
    print(classification_report(labels, preds,
                                 target_names=[str(i) for i in range(10)]))
    return history, preds, labels


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(cnn_hist, rnn_hist):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN vs RNN — Training Curves", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(
        axes,
        [("train_loss","val_loss"), ("train_acc","val_acc")],
        ["Loss", "Accuracy (%)"]
    ):
        epochs = range(1, len(cnn_hist[metric[0]]) + 1)
        ax.plot(epochs, cnn_hist[metric[0]], "b-o", label="CNN Train", linewidth=2)
        ax.plot(epochs, cnn_hist[metric[1]], "b--s", label="CNN Val",  linewidth=2)
        ax.plot(epochs, rnn_hist[metric[0]], "r-o", label="RNN Train", linewidth=2)
        ax.plot(epochs, rnn_hist[metric[1]], "r--s", label="RNN Val",  linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
    print("\n  Saved: training_curves.png")
    plt.show()


def plot_confusion_matrix(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(10), yticks=range(10),
           xticklabels=range(10), yticklabels=range(10),
           xlabel="Predicted Label", ylabel="True Label",
           title=f"Confusion Matrix — {title}")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=8)
    plt.tight_layout()
    fname = f"cm_{title.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.show()


def plot_sample_predictions(model, loader, title, n=16):
    model.eval()
    images_all, labels_all, preds_all = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(DEVICE))
            _, predicted = out.max(1)
            images_all.append(imgs.cpu())
            labels_all.append(lbls.cpu())
            preds_all.append(predicted.cpu())
            if sum(len(x) for x in images_all) >= n:
                break
    images = torch.cat(images_all)[:n]
    labels = torch.cat(labels_all)[:n]
    preds  = torch.cat(preds_all)[:n]

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 5))
    fig.suptitle(f"Sample Predictions — {title}", fontsize=13, fontweight="bold")
    for idx, ax in enumerate(axes.flat):
        img = images[idx].squeeze().numpy()
        true, pred = labels[idx].item(), preds[idx].item()
        ax.imshow(img, cmap="gray")
        color = "green" if true == pred else "red"
        ax.set_title(f"T:{true} P:{pred}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fname = f"predictions_{title.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.show()


def plot_architecture_summary(cnn_model, rnn_model):
    """Print a layer-by-layer parameter summary."""
    print("\n" + "─"*60)
    print("  MODEL ARCHITECTURES")
    print("─"*60)
    for name, model in [("CNN", cnn_model), ("RNN/LSTM", rnn_model)]:
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  {name} ({total:,} parameters):")
        print(f"  {model}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6: COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(cnn_hist, rnn_hist):
    cnn_acc = max(cnn_hist["val_acc"])
    rnn_acc = max(rnn_hist["val_acc"])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["CNN", "RNN (LSTM)"], [cnn_acc, rnn_acc],
                   color=["#2196F3", "#F44336"], width=0.5, edgecolor="black")
    for bar, val in zip(bars, [cnn_acc, rnn_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                f"{val:.2f}%", ha="center", va="top",
                fontsize=13, fontweight="bold", color="white")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
    ax.set_title("CNN vs RNN — MNIST Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120, bbox_inches="tight")
    print("  Saved: comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  ANN ASSIGNMENT — HANDWRITTEN RECOGNITION")
    print("  CNN & RNN on MNIST Dataset")
    print("=" * 60)

    # Part B — Numerical Examples
    numerical_examples()

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data()

    # Instantiate models
    cnn_model = CNNModel().to(DEVICE)
    rnn_model = RNNModel().to(DEVICE)
    plot_architecture_summary(cnn_model, rnn_model)

    # Train CNN
    cnn_history, cnn_preds, cnn_labels = train_model(
        cnn_model, "CNN", train_loader, test_loader
    )

    # Train RNN
    rnn_history, rnn_preds, rnn_labels = train_model(
        rnn_model, "RNN (LSTM)", train_loader, test_loader
    )

    # ── Visualizations ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Generating Visualizations...")
    print("─" * 60)

    plot_training_curves(cnn_history, rnn_history)
    plot_comparison(cnn_history, rnn_history)
    plot_confusion_matrix(cnn_labels, cnn_preds, "CNN")
    plot_confusion_matrix(rnn_labels, rnn_preds, "RNN LSTM")
    plot_sample_predictions(cnn_model, test_loader, "CNN")
    plot_sample_predictions(rnn_model, test_loader, "RNN LSTM")

    print("\n" + "=" * 60)
    print("  ALL DONE — Assignment Complete!")
    print("  Files saved:")
    print("    training_curves.png")
    print("    comparison.png")
    print("    cm_cnn.png")
    print("    cm_rnn_lstm.png")
    print("    predictions_cnn.png")
    print("    predictions_rnn_lstm.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
