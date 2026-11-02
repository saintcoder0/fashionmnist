# Context: NumPy Neural Network on Fashion-MNIST

## Goal
Build a neural network **from scratch** (no PyTorch/TensorFlow/scikit-learn) using **only NumPy** for math and **matplotlib** for plots. Data lives in `/home/machine/code/vidha/data`. Package management is **uv**.

## Dataset
Fashion-MNIST (Zalando):
- 28×28 grayscale, 10 clothing classes
- Train 60,000 / Test 10,000
- Files: IDX binaries (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-*`) plus CSV copies
- Loader prefers IDX (faster). CSV fallback: header `label,pixel1..pixel784`

Classes: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot

## Project layout
- `pyproject.toml` / `uv.lock` — uv project, deps: numpy, matplotlib
- `src/vidha/data_loader.py` — IDX/CSV load, one-hot, train/val split
- `src/vidha/network.py` — MLP: He init, ReLU, softmax, CCE, mini-batch SGD, L2, save/load npz
- `src/vidha/plots.py` — samples, class counts, curves, confusion matrix, predictions
- `src/vidha/train.py` — training entrypoint (`train()` / `main()`)
- `src/vidha/__main__.py` — `uv run python -m vidha`
- `plots/` — matplotlib outputs + `metrics.json`
- `models/mlp_fashion_mnist.npz` — trained weights (gitignored)

Architecture: **784 → 128 ReLU → 64 ReLU → 10 softmax**.
Defaults: 12 epochs, batch 128, lr 0.12 decaying ×0.92/epoch, L2 1e-4, 5k val split from train.

## Commands
Local:
```bash
cd /home/machine/code/vidha
uv sync
uv run python -m vidha
```

Docker (isolated, portable):
```bash
docker compose up --build
```
Image: `Dockerfile` + `docker-compose.yml`. Non-root, read-only rootfs, cap_drop ALL. Plots/weights bind-mounted to `./plots` and `./models`.

## Status
- [x] Inspected `/data` (Fashion-MNIST CSV + IDX)
- [x] uv project + numpy/matplotlib
- [x] Code: loader, network, train, plots
- [x] Training run (CPU, ~8.5s)
- [x] Results recorded
- [x] Docker image + Compose (non-root, read-only, cap_drop ALL)

## Results (2026-11-01)
- Test accuracy: **0.8736**
- Test loss: **0.3772**
- Last train acc: 0.8974, last val acc: 0.8800
- Hardest classes: Shirt 0.671, Pullover 0.694 (confused with each other / coats / T-shirts)
- Easiest: Bag 0.971, Sneaker 0.965, Trouser 0.959

Plots:
- `plots/dataset_samples.png`
- `plots/class_counts.png`
- `plots/training_curves.png`
- `plots/confusion_matrix.png`
- `plots/test_predictions.png`
- `plots/metrics.json`

## How to resume if tokens run out
1. Read this file and `plots/metrics.json`.
2. Training already finished; re-run with `uv run python -m vidha` only if you change the network.
3. Do **not** add extra ML libraries. Keep batch ops vectorized.

## Notes for a follow-up agent
- Headless matplotlib uses `Agg` in `train.py`.
- `uv run` may need unrestricted FS if the sandbox cannot write `.venv/bin/vidha`.
- Shirt vs T-shirt/Pullover/Coat is the main remaining error; a CNN would help but is out of scope unless requested.
