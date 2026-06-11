#!/bin/bash
# Data and Environment Setup for Cloud Training
# Handles data transfer and environment configuration

set -e

echo "=== Data and Environment Setup ==="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << 'EOF'
# Hugging Face authentication token
# Request access at: https://huggingface.co/facebook/blt-1b
# Then add your token below:
HF_TOKEN=your_token_here
EOF
    echo "[WARNING] .env file created - please edit it and add your HF_TOKEN"
    echo "   Get your token from: https://huggingface.co/settings/tokens"
    echo "   Request access at: https://huggingface.co/facebook/blt-1b"
else
    echo "[OK] .env file exists"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data checkpoints/meta_blt

# Check if training data exists
if [ ! -f data/phonemized.txt ]; then
    echo "[WARNING] Training data not found at data/phonemized.txt"
    echo "   Please upload your phonemized training data"
    echo "   Or run the data pipeline: uv run data/download.py && uv run data/phonemize.py"
else
    LINES=$(wc -l < data/phonemized.txt)
    echo "[OK] Training data found: $LINES lines"
fi

# Test the setup
echo ""
echo "Testing setup..."
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if hf_token and hf_token != 'your_token_here':
    print('[OK] HF_TOKEN is configured')
else:
    print('[ERROR] HF_TOKEN not set - please edit .env file')
    exit(1)
"

echo ""
echo "=== Data Setup Complete ==="
echo "Ready for training! Run:"
echo "  python -m model.train --config model/blt_configs/meta_blt.yaml"
