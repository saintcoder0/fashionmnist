# Reproducible CPU training image. Build once, run anywhere with Docker.
# Multi-stage: uv installs a locked venv, runtime is a slim non-root Python.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY pyproject.toml uv.lock /app/
COPY src /app/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /app/data /app/plots /app/models

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src /app/src
COPY --chown=app:app \
    data/train-images-idx3-ubyte \
    data/train-labels-idx1-ubyte \
    data/t10k-images-idx3-ubyte \
    data/t10k-labels-idx1-ubyte \
    /app/data/

RUN chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    VIDHA_ROOT=/app \
    VIDHA_DATA_DIR=/app/data \
    VIDHA_PLOTS_DIR=/app/plots \
    VIDHA_MODELS_DIR=/app/models

USER app

CMD ["python", "-m", "vidha"]
