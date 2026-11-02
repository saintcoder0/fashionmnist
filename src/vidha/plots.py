"""Matplotlib figures for dataset inspection and training results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vidha.data_loader import CLASS_NAMES


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_class_samples(
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    per_class: int = 8,
) -> None:
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(per_class * 1.15, 11.5))
    rng = np.random.default_rng(7)
    for cls in range(len(CLASS_NAMES)):
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=per_class, replace=False)
        for col, sample_i in enumerate(chosen):
            ax = axes[cls, col]
            ax.imshow(x[sample_i].reshape(28, 28), cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls], rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Fashion-MNIST training samples by class", fontsize=13)
    _save(fig, path)


def plot_class_counts(y_train: np.ndarray, y_test: np.ndarray, path: Path) -> None:
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    xs = np.arange(10)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(xs - width / 2, train_counts, width, label="train")
    ax.bar(xs + width / 2, test_counts, width, label="test")
    ax.set_xticks(xs, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Class distribution")
    ax.legend()
    _save(fig, path)


def plot_history(history: dict[str, list[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Training curves (NumPy MLP)")
    _save(fig, path)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    _save(fig, path)


def plot_predictions(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    n: int = 25,
    seed: int = 3,
) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(9, 9))
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        i = idx[k]
        ax.imshow(x[i].reshape(28, 28), cmap="gray")
        ok = y_true[i] == y_pred[i]
        color = "tab:green" if ok else "tab:red"
        ax.set_title(
            f"{CLASS_NAMES[y_pred[i]]}\n({CLASS_NAMES[y_true[i]]})",
            color=color,
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle("Test predictions (green=correct, red=wrong)")
    _save(fig, path)
