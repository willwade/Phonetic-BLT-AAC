#!/bin/bash
# Cloud GPU Training Setup Script for Meta BLT
# This script sets up the environment on a cloud GPU instance for training

set -e  # Exit on any error

echo "=== Meta BLT Cloud Training Setup ==="
echo "Setting up environment for GPU training..."

# Detect GPU and set CUDA environment variables
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected"
    nvidia-smi
    export CUDA_VISIBLE_DEVICES=0
else
    echo "[ERROR] No NVIDIA GPU detected - this setup requires GPU"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "[OK] Python version: $PYTHON_VERSION"

# Install UV package manager if not present
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[OK] UV package manager ready"

# Install project dependencies
echo "Installing project dependencies..."
uv sync --extra dev

# Uninstall CPU PyTorch and install GPU version
echo "Installing GPU-enabled PyTorch..."
uv pip uninstall torch torchvision -y
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify GPU support
echo "Verifying GPU support in PyTorch..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU device:', torch.cuda.get_device_name(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Copy your .env file with HF_TOKEN to this machine"
echo "2. Upload your training data to data/phonemized.txt"
echo "3. Run: python -m model.train --config model/blt_configs/meta_blt.yaml"
echo ""
echo "Or run the included test script to verify everything works:"
echo "  python test_meta_blt.py"
