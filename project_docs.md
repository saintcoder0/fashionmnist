# Vidha — project documentation

NumPy-from-scratch multilayer perceptron (MLP) trained on **Fashion-MNIST**, with matplotlib plots. No PyTorch, TensorFlow, or scikit-learn. Dependencies are managed with **uv**.

## What the project does

1. Loads 28×28 grayscale clothing images from `data/`.
2. Trains a fully connected network: **784 → 128 ReLU → 64 ReLU → 10 softmax**.
3. Uses mini-batch SGD, categorical cross-entropy, He initialization, and a small L2 penalty.
4. Writes plots, metrics, and weight files so you can inspect dataset and results.

Last recorded run: **87.36% test accuracy** (12 epochs, ~8.5s on CPU). See `plots/metrics.json`.

## Project structure

```
vidha/
├── data/                          Fashion-MNIST (IDX binaries + CSV copies)
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── fashion-mnist_train.csv    60,000 rows + header
│   └── fashion-mnist_test.csv     10,000 rows + header
├── src/vidha/                     Python package
│   ├── data_loader.py             IDX/CSV load, normalize, one-hot, train/val split
│   ├── network.py                 MLP forward/backward, SGD, save/load .npz
│   ├── plots.py                   matplotlib figures
│   ├── train.py                   training loop + artifact writes
│   ├── paths.py                   project/data/plots/models paths (env-aware)
│   ├── __main__.py                python -m vidha
│   └── __init__.py                exposes main()
├── plots/                         generated figures + metrics.json
├── models/                        mlp_fashion_mnist.npz (trained weights)
├── Dockerfile                     locked, non-root training image
├── docker-compose.yml             run with dropped caps + host plot/model mounts
├── .dockerignore                  keep image small (skip .venv, CSVs, plots)
├── pyproject.toml                 uv project, numpy + matplotlib
├── uv.lock                        locked dependency versions
├── context.md                     short handoff notes for another agent
└── project_docs.md                this file
```

`uv` also creates `.venv/` (gitignored). Do not edit files under `.venv`.

## How it works

### Data (`data_loader.py`)

- Prefers **IDX** files (faster). Falls back to CSV if IDX is missing.
- Pixels become `float32` in `[0, 1]` (divide by 255).
- Images are flattened to shape `(N, 784)`.
- Labels are integers `0–9`. Training uses one-hot vectors of length 10.
- A random **5,000** examples are held out from train as validation (`seed=42`).

Class names:

| id | name |
|---:|------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Network (`network.py`)

- **Forward:** each hidden layer is `relu(x @ W + b)`; output is softmax.
- **Loss:** mean categorical cross-entropy plus `0.5 * l2 * ||W||²`.
- **Backward:** softmax + CCE simplifies to `(probs - y) / batch_size`; ReLU gates gradients where pre-activation was ≤ 0.
- **Update:** vanilla SGD on weights and biases.
- **Save:** compressed `.npz` with keys `layer_sizes`, `l2`, `w0`, `b0`, `w1`, `b1`, `w2`, `b2`.

### Training (`train.py`)

Defaults:

- epochs = 12
- batch size = 128
- learning rate = 0.12, multiplied by 0.92 after each epoch
- L2 = `1e-4`

Each epoch shuffles training data, runs SGD, then reports train and val loss/accuracy. After the last epoch it evaluates the official test set, writes plots, saves weights, and writes `plots/metrics.json`. Matplotlib uses the **Agg** backend so it works without a display.

### Plots (`plots.py`)

| file | contents |
|------|----------|
| `plots/dataset_samples.png` | 8 training images per class |
| `plots/class_counts.png` | train vs test counts |
| `plots/training_curves.png` | loss and accuracy vs epoch |
| `plots/confusion_matrix.png` | test-set confusion |
| `plots/test_predictions.png` | sample predictions (green = correct, red = wrong) |

## Run with Docker (portable / isolated)

Docker is the supported way to train on any machine without installing Python packages on the host. The image is **CPU-only**, **non-root** (`uid 1000`), read-only root filesystem, and `cap_drop: ALL`. Fashion-MNIST IDX files are baked into the image; plots and weights land on the host via volumes.

You need Docker Engine and Compose v2.

```bash
cd /home/machine/code/vidha   # or wherever you cloned the repo
```

### Build and train (writes into ./plots and ./models)

```bash
docker compose up --build
```

Equivalent without Compose:

```bash
docker build -t vidha-mlp:latest .
mkdir -p plots models
docker run --rm \
  --read-only \
  --tmpfs /tmp:size=64m,mode=1777 \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --user 1000:1000 \
  -v "$(pwd)/plots:/app/plots" \
  -v "$(pwd)/models:/app/models" \
  vidha-mlp:latest
```

Copy the image to another computer (`docker save` / `docker load`) and run the same `docker run` command; no uv or NumPy install is required there.

### Inspect weights inside the container

```bash
docker compose run --rm train python -c "import numpy as np; d=np.load('/app/models/mlp_fashion_mnist.npz'); print(list(d.keys()), d['w0'].shape)"
```

### Security notes

- Process runs as `app` (uid 1000), not root.
- Root filesystem is read-only; only `/tmp`, `./plots`, and `./models` are writable.
- Linux capabilities are dropped; privilege escalation is disabled.
- Training data in the image is public Fashion-MNIST, not secrets.

If Docker is not installed on Linux:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"
# log out and back in, then: docker compose up --build
```

## How to run it again (local uv, no Docker)


You need **uv** and Python 3.12+ (uv can install CPython if needed). Run everything from the project root:

```bash
cd /home/machine/code/vidha
```

### 1. Install dependencies (first time, or after lockfile changes)

```bash
uv sync
```

This creates/updates `.venv` from `pyproject.toml` and `uv.lock`. Only **numpy** and **matplotlib** are required.

### 2. Train from scratch and regenerate plots

```bash
uv run python -m vidha
```

Equivalent:

```bash
uv run vidha
```

This overwrites:

- `plots/*.png`
- `plots/metrics.json`
- `models/mlp_fashion_mnist.npz`

Training is CPU-only and typically finishes in under a minute.

### 3. Change hyperparameters

Edit defaults in `src/vidha/train.py` (`train(...)`) or call it from a REPL:

```bash
uv run python -c "from vidha.train import train; train(epochs=5, lr=0.1)"
```

### 4. Inspect saved weights

```bash
uv run python
```

```python
import numpy as np

data = np.load("models/mlp_fashion_mnist.npz")
print(list(data.keys()))
print(data["layer_sizes"])   # [784 128 64 10]
print(data["w0"].shape)      # (784, 128)
print(data["b2"])            # 10 output biases
```

Or reload the model object:

```python
from vidha.network import MLP
model = MLP.load("models/mlp_fashion_mnist.npz")
print(model.weights[0].shape)
```

### 5. Load data without training

```python
from vidha.data_loader import load_fashion_mnist
x_train, y_train, x_test, y_test = load_fashion_mnist()
print(x_train.shape, y_train[:5])
```

## Requirements and constraints

- Do not add extra ML libraries if you want to keep this “from scratch”.
- Keep matrix ops vectorized (no Python loop over samples inside a batch).
- IDX files should stay next to the CSVs in `data/`. If only CSVs remain, loading still works but is slower.

## Related files

- `context.md` — shorter handoff for another agent (status + last metrics).
- `plots/metrics.json` — last test accuracy, per-class scores, and artifact paths.
