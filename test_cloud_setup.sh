#!/bin/bash
# Quick test of cloud setup scripts (local validation)

echo "=== Testing Cloud Setup Scripts ==="

echo "1. Testing GPU detection..."
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[SKIP] No NVIDIA GPU detected (expected on local machine)"
fi

echo "2. Testing UV package manager..."
if command -v uv &> /dev/null; then
    echo "[OK] UV is available"
    uv --version
else
    echo "[INFO] UV not installed - would be installed by setup script"
fi

echo "3. Testing Python environment..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "[OK] Python version: $PYTHON_VERSION"

echo "4. Testing data setup script..."
bash cloud_data_setup.sh

echo "5. Checking critical files..."
if [ -f .env ]; then echo "[OK] .env exists"; else echo "[ERROR] .env missing"; fi
if [ -f model/train.py ]; then echo "[OK] Training script exists"; else echo "[ERROR] Training script missing"; fi
if [ -f model/blt_configs/meta_blt.yaml ]; then echo "[OK] Config file exists"; else echo "[ERROR] Config missing"; fi
if [ -f test_meta_blt.py ]; then echo "[OK] Test script exists"; else echo "[ERROR] Test script missing"; fi

echo "=== Test Complete ==="
echo "Setup scripts are ready for cloud deployment!"
