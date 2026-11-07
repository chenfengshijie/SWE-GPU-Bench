# Optional development image. Override BASE_IMAGE to match the task's CUDA ABI.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl ca-certificates build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1
# Runtime tools and evaluator dependencies come from the frozen campaign bundle.
WORKDIR /tmp
