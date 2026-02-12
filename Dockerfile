FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="9.0;12.0"

# ── System deps + Python 3.13 (via deadsnakes PPA) ──────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# ── uv (fast package manager) ───────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ── Dependencies (cached layer) ─────────────────────────────────────────────
COPY pyproject.toml ./
RUN uv venv && \
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-ml-py && \
    uv pip install -e .

# ── Application code ────────────────────────────────────────────────────────
COPY . .
RUN uv pip install -e .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["gpu-test", "bench"]
