"""Load Fashion-MNIST from IDX files (or CSV fallback) with NumPy only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vidha.paths import DATA_DIR

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

def _read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, n, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        if magic != 2051:
            raise ValueError(f"Unexpected image magic {magic} in {path}")
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return pixels.reshape(int(n), int(rows) * int(cols))


def _read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, n = np.frombuffer(f.read(8), dtype=">i4")
        if magic != 2049:
            raise ValueError(f"Unexpected label magic {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != int(n):
        raise ValueError(f"Label count mismatch in {path}")
    return labels.copy()


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    labels = raw[:, 0].astype(np.uint8)
    images = raw[:, 1:]
    return images, labels


def load_fashion_mnist(
    data_dir: Path | None = None,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, y_train, X_test, y_test as float32 / uint8 arrays.

    Images are (N, 784). Pixel values are in [0, 1] when normalize=True.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    idx_files = {
        "train_images": data_dir / "train-images-idx3-ubyte",
        "train_labels": data_dir / "train-labels-idx1-ubyte",
        "test_images": data_dir / "t10k-images-idx3-ubyte",
        "test_labels": data_dir / "t10k-labels-idx1-ubyte",
    }
    if all(p.exists() for p in idx_files.values()):
        x_train = _read_idx_images(idx_files["train_images"]).astype(np.float32)
        y_train = _read_idx_labels(idx_files["train_labels"])
        x_test = _read_idx_images(idx_files["test_images"]).astype(np.float32)
        y_test = _read_idx_labels(idx_files["test_labels"])
    else:
        x_train, y_train = _read_csv(data_dir / "fashion-mnist_train.csv")
        x_test, y_test = _read_csv(data_dir / "fashion-mnist_test.csv")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    return x_train, y_train, x_test, y_test


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0
    return encoded


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_size: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
