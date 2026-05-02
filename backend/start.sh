#!/usr/bin/env bash
set -e
export HF_HOME="${HF_HOME:-.}/models"      # persistent storage (default: local)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Navigate to backend directory (handle both local and vast.ai paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg
fi

# Install deps (uv reads pyproject.toml)
echo "Installing dependencies..."
uv sync

# Launch
echo "Starting STT Evaluation API on 0.0.0.0:8000"
uv run uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --timeout-keep-alive 300
