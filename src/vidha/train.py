"""Train the NumPy MLP on Fashion-MNIST and write matplotlib result plots."""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np

from vidha.data_loader import CLASS_NAMES, DATA_DIR, load_fashion_mnist, one_hot, train_val_split
from vidha.network import MLP
from vidha.paths import MODELS_DIR, PLOTS_DIR, PROJECT_ROOT
from vidha.plots import (
    confusion_matrix,
    plot_class_counts,
    plot_class_samples,
    plot_confusion_matrix,
    plot_history,
    plot_predictions,
)

def train(
    *,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 0.12,
    lr_decay: float = 0.92,
    val_size: int = 5000,
    seed: int = 42,
) -> dict:
    print(f"Loading Fashion-MNIST from {DATA_DIR}")
    x_train_full, y_train_full, x_test, y_test = load_fashion_mnist()
    print(
        f"Train {x_train_full.shape}  Test {x_test.shape}  "
        f"pixel range [{x_train_full.min():.3f}, {x_train_full.max():.3f}]"
    )

    plot_class_samples(x_train_full, y_train_full, PLOTS_DIR / "dataset_samples.png")
    plot_class_counts(y_train_full, y_test, PLOTS_DIR / "class_counts.png")
    print(f"Saved dataset plots under {PLOTS_DIR}")

    x_train, y_train, x_val, y_val = train_val_split(
        x_train_full, y_train_full, val_size=val_size, seed=seed
    )
    y_train_oh = one_hot(y_train)
    y_val_oh = one_hot(y_val)
    y_test_oh = one_hot(y_test)

    model = MLP(layer_sizes=(784, 128, 64, 10), seed=seed, l2=1e-4)
    rng = np.random.default_rng(seed)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    t0 = time.perf_counter()
    current_lr = lr
    for epoch in range(1, epochs + 1):
        model.train_epoch(x_train, y_train_oh, lr=current_lr, batch_size=batch_size, rng=rng)
        tr_loss, tr_acc = model.loss_and_accuracy(x_train, y_train_oh)
        va_loss, va_acc = model.loss_and_accuracy(x_val, y_val_oh)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)
        print(
            f"epoch {epoch:02d}/{epochs}  lr={current_lr:.4f}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}"
        )
        current_lr *= lr_decay

    elapsed = time.perf_counter() - t0
    test_loss, test_acc = model.loss_and_accuracy(x_test, y_test_oh)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = np.divide(
        np.diag(cm).astype(np.float64),
        cm.sum(axis=1).astype(np.float64),
        out=np.zeros(10, dtype=np.float64),
        where=cm.sum(axis=1) > 0,
    )

    plot_history(history, PLOTS_DIR / "training_curves.png")
    plot_confusion_matrix(cm, PLOTS_DIR / "confusion_matrix.png", f"Test confusion (acc={test_acc:.3f})")
    plot_predictions(x_test, y_test, y_pred, PLOTS_DIR / "test_predictions.png")

    weights_path = MODELS_DIR / "mlp_fashion_mnist.npz"
    model.save(weights_path)

    metrics = {
        "architecture": list(model.layer_sizes),
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_lr": lr,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "val_accuracy_last": history["val_acc"][-1],
        "train_accuracy_last": history["train_acc"][-1],
        "seconds": elapsed,
        "per_class_accuracy": {
            CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(10)
        },
        "plots": [
            "plots/dataset_samples.png",
            "plots/class_counts.png",
            "plots/training_curves.png",
            "plots/confusion_matrix.png",
            "plots/test_predictions.png",
        ],
        "weights": str(weights_path.relative_to(PROJECT_ROOT)),
    }
    metrics_path = PLOTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Test accuracy {test_acc:.4f}  loss {test_loss:.4f}  time {elapsed:.1f}s")
    print(f"Wrote {metrics_path} and {weights_path}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
