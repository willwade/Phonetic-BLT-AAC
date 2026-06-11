#!/bin/bash
# Cloud GPU Training Setup for BLT
# Sets up the environment on a cloud GPU instance for training

set -e

echo "=== BLT Cloud Training Setup ==="
echo "Setting up environment for GPU training..."

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected"
    nvidia-smi
    export CUDA_VISIBLE_DEVICES=0
else
    echo "[ERROR] No NVIDIA GPU detected - training requires GPU"
    exit 1
fi

# Check Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "[OK] Python version: $PYTHON_VERSION"

# Install UV
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "[OK] UV ready"

# Install dependencies
echo "Installing project dependencies..."
uv sync --extra dev

# Replace CPU PyTorch with CUDA version
echo "Installing GPU-enabled PyTorch..."
uv pip uninstall torch torchvision -y
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify
echo "Verifying GPU support..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {mem:.1f} GB')
    if mem < 35:
        print(f'WARNING: {mem:.0f} GB VRAM may be insufficient for BLT (4.63B params)')
        print('Reduce batch_size in meta_blt.yaml if you get OOM errors')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

# Verify model loads
echo ""
echo "Verifying BLT model download..."
python -c "
from model.blt_transformers import MetaBLTWrapper
w = MetaBLTWrapper()
print(f'Model ID: {w.model_name}')
print('[OK] Wrapper imports correctly')
print('Model weights will download on first training run (~9 GB)')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload training data to data/phonemized.txt"
echo "  2. Run smoke test: uv run python test_blt_load.py"
echo "  3. Start training: uv run model/train.py --config model/blt_configs/meta_blt.yaml"
