#!/bin/bash
# Data setup for cloud training
# Checks training data exists and creates directories

set -e

echo "=== Data Setup ==="

# Create directories
echo "Creating directories..."
mkdir -p data checkpoints/meta_blt

# Check training data
if [ ! -f data/phonemized.txt ]; then
    echo ""
    echo "[WARNING] No training data at data/phonemized.txt"
    echo ""
    echo "Option A: Upload pre-generated data"
    echo "  scp phonemized.txt <instance>:/path/to/Phonetic-BLT-AAC/data/"
    echo ""
    echo "Option B: Generate on this instance"
    echo "  uv run data/download.py --threshold 0.85"
    echo "  uv run data/phonemize.py --input data/raw_conversations.txt --output data/phonemized.txt"
    echo ""
    echo "For a quick test, use the high-density subset:"
    echo "  uv run data/download.py --output data/raw_conversations.txt"
    echo "  (then phonemize as above)"
else
    LINES=$(wc -l < data/phonemized.txt)
    SIZE=$(du -h data/phonemized.txt | cut -f1)
    echo "[OK] Training data: $LINES lines, $SIZE"
fi

echo ""
echo "=== Ready ==="
echo "Start training with:"
echo "  uv run model/train.py --config model/blt_configs/meta_blt.yaml"
