"""
Day 14: Deep learning basics — neural network from scratch + PyTorch intro.

Covers:
1. Neural network from scratch (numpy only) — forward/backward pass
2. PyTorch basics — same network, with autograd
3. Simple transformer self-attention demo

Usage:
    python deep_learning.py
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Only numpy examples will run. Install: pip install torch")


def section(title: str):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}\n")


# ========== 1. NEURAL NETWORK FROM SCRATCH ==========

class NumpyNN:
    """
    2-layer neural network using only numpy.
    Input -> Hidden (ReLU) -> Output (Sigmoid)

    This is what's happening under the hood in PyTorch/TensorFlow.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(42)
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr=0.01):
        m = X.shape[0]

        # Output layer gradients
        dz2 = output - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dz1 = (dz2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def loss(self, y, output):
        """Binary cross-entropy loss."""
        eps = 1e-8
        return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))


def numpy_nn_demo():
    section("Neural Network from Scratch (numpy)")

    # XOR problem — not linearly separable
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NumpyNN(input_dim=2, hidden_dim=8, output_dim=1)

    print("Training on XOR problem...")
    for epoch in range(2000):
        output = nn.forward(X)
        nn.backward(X, y, output, lr=0.5)
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: loss={nn.loss(y, output):.4f}")

    print(f"\nPredictions after training:")
    predictions = nn.forward(X)
    for i in range(len(X)):
        print(f"  {X[i]} -> {predictions[i][0]:.3f} (expected {y[i][0]})")


# ========== 2. PYTORCH VERSION ==========

def pytorch_nn_demo():
    if not HAS_TORCH:
        print("Skipping PyTorch demo (not installed)")
        return

    section("Neural Network with PyTorch")

    # Same XOR problem
    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.FloatTensor([[0], [1], [1], [0]])

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(2000):
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    print(f"\nPredictions:")
    with torch.no_grad():
        preds = model(X)
        for i in range(len(X)):
            print(f"  {X[i].tolist()} -> {preds[i].item():.3f} (expected {y[i].item():.0f})")

    # Show model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")


# ========== 3. SELF-ATTENTION DEMO ==========

def self_attention_demo():
    section("Self-Attention (Transformer building block)")

    # Simulating 3 tokens with 4-dimensional embeddings
    # Think of it as: ["The", "cat", "sat"] each represented as a 4-dim vector
    np.random.seed(42)
    seq_len = 3
    d_model = 4

    X = np.random.randn(seq_len, d_model)
    print(f"Input (3 tokens, 4 dims):\n{X.round(3)}\n")

    # Learned weight matrices (in practice, these are trained)
    d_k = 4
    W_Q = np.random.randn(d_model, d_k) * 0.5
    W_K = np.random.randn(d_model, d_k) * 0.5
    W_V = np.random.randn(d_model, d_k) * 0.5

    # Compute Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    print(f"Q (queries):\n{Q.round(3)}\n")
    print(f"K (keys):\n{K.round(3)}\n")
    print(f"V (values):\n{V.round(3)}\n")

    # Attention scores: Q * K^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)
    print(f"Attention scores (Q @ K^T / sqrt(d_k)):\n{scores.round(3)}\n")

    # Softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scores)
    print(f"Attention weights (after softmax):\n{attention_weights.round(3)}")
    print(f"  Each row sums to: {attention_weights.sum(axis=1).round(3)}\n")

    # Weighted sum of values
    output = attention_weights @ V
    print(f"Output (attention_weights @ V):\n{output.round(3)}")
    print(f"\nEach output token is a weighted combination of all value vectors.")
    print(f"The weights come from how much each query 'attends to' each key.")


if __name__ == "__main__":
    numpy_nn_demo()
    pytorch_nn_demo()
    self_attention_demo()
