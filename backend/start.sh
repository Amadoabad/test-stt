#!/usr/bin/env bash
set -e
export HF_HOME=/workspace/models      # persistent vast.ai storage
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /workspace/stt-eval/backend

# Install deps (uv reads pyproject.toml)
uv sync

# Launch
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
