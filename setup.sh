#!/usr/bin/env bash
set -euo pipefail

echo "=== Multimodal Agent — Vast.ai Setup ==="

# System deps
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python env
python3 -m venv /opt/venv
source /opt/venv/bin/activate
echo "source /opt/venv/bin/activate" >> ~/.bashrc

# Install deps
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Playwright browsers
playwright install chromium
playwright install-deps chromium

# CUDA check
echo "--- CUDA ---"
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0)}')"

echo "=== Setup complete ==="
