"""Fully-connected MLP implemented with NumPy only."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MLP:
    """784 → hidden layers (ReLU) → 10 (softmax). Mini-batch SGD + L2."""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (784, 128, 64, 10),
        seed: int = 0,
        l2: float = 1e-4,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output sizes")
        self.layer_sizes = layer_sizes
        self.l2 = l2
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization for ReLU hidden layers
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Return (probs, pre-activations, activations including input)."""
        activations = [x]
        pre_acts: list[np.ndarray] = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre_acts.append(z)
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                a = self._relu(z)
            activations.append(a)
        return a, pre_acts, activations

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        preds = []
        for start in range(0, x.shape[0], batch_size):
            probs, _, _ = self.forward(x[start : start + batch_size])
            preds.append(np.argmax(probs, axis=1))
        return np.concatenate(preds)

    def loss_and_accuracy(
        self,
        x: np.ndarray,
        y_onehot: np.ndarray,
        batch_size: int = 1024,
    ) -> tuple[float, float]:
        n = x.shape[0]
        total_loss = 0.0
        correct = 0
        for start in range(0, n, batch_size):
            xb = x[start : start + batch_size]
            yb = y_onehot[start : start + batch_size]
            probs, _, _ = self.forward(xb)
            clipped = np.clip(probs, 1e-12, 1.0)
            total_loss += float(-np.sum(yb * np.log(clipped)))
            correct += int(np.sum(np.argmax(probs, axis=1) == np.argmax(yb, axis=1)))
        l2_term = 0.5 * self.l2 * sum(float(np.sum(w * w)) for w in self.weights)
        return (total_loss / n) + l2_term, correct / n

    def train_epoch(
        self,
        x: np.ndarray,
        y_onehot: np.ndarray,
        *,
        lr: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        n = x.shape[0]
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            self._sgd_step(x[idx], y_onehot[idx], lr)

    def _sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> None:
        batch = x.shape[0]
        probs, pre_acts, activations = self.forward(x)

        # Softmax + cross-entropy: dL/dz_out = (p - y) / N
        delta = (probs - y) / batch
        dw_list: list[np.ndarray] = []
        db_list: list[np.ndarray] = []

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            dw = a_prev.T @ delta + self.l2 * self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True)
            dw_list.append(dw)
            db_list.append(db)
            if i > 0:
                relu_mask = (pre_acts[i - 1] > 0).astype(np.float32)
                delta = (delta @ self.weights[i].T) * relu_mask

        dw_list.reverse()
        db_list.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= lr * dw_list[i]
            self.biases[i] -= lr * db_list[i]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int32),
            "l2": np.array(self.l2, dtype=np.float32),
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"w{i}"] = w
            payload[f"b{i}"] = b
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> MLP:
        data = np.load(path)
        sizes = tuple(int(s) for s in data["layer_sizes"])
        model = cls(layer_sizes=sizes, l2=float(data["l2"]))
        model.weights = [data[f"w{i}"] for i in range(len(sizes) - 1)]
        model.biases = [data[f"b{i}"] for i in range(len(sizes) - 1)]
        return model
