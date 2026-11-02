"""Resolve project, data, plot, and model directories (local uv or Docker)."""

from __future__ import annotations

import os
from pathlib import Path


def _detect_root() -> Path:
    env = os.environ.get("VIDHA_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    if here.parent.name == "vidha" and here.parents[1].name == "src":
        return here.parents[2]
    cwd = Path.cwd()
    if (cwd / "data").is_dir():
        return cwd
    return Path("/app")


PROJECT_ROOT = _detect_root()
DATA_DIR = Path(os.environ.get("VIDHA_DATA_DIR", PROJECT_ROOT / "data")).resolve()
PLOTS_DIR = Path(os.environ.get("VIDHA_PLOTS_DIR", PROJECT_ROOT / "plots")).resolve()
MODELS_DIR = Path(os.environ.get("VIDHA_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
