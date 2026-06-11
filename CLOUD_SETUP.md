# Cloud GPU Training Setup

Setup scripts for training BLT on a cloud GPU instance (A100, H200, etc.).

## Quick Start

```bash
# 1. Clone
git clone https://github.com/willwade/Phonetic-BLT-AAC.git
cd Phonetic-BLT-AAC

# 2. Run setup (installs uv, dependencies, CUDA PyTorch)
chmod +x cloud_setup.sh
./cloud_setup.sh

# 3. Upload or generate training data
chmod +x cloud_data_setup.sh
./cloud_data_setup.sh

# 4. Smoke test (verifies model loads, ~9 GB download)
uv run python test_blt_load.py

# 5. Train
uv run model/train.py --config model/blt_configs/meta_blt.yaml
```

## Requirements

- **GPU**: NVIDIA GPU with **40+ GB VRAM** (A100 80GB recommended)
- **CUDA**: CUDA 12.4+ compatible drivers
- **Python**: 3.12+
- **Storage**: ~50 GB (model weights ~9 GB, data + checkpoints ~40 GB)
- **No HF token needed** — model weights (`itazap/blt-1b-hf`) are public

## What the Scripts Do

### `cloud_setup.sh`
- Detects and verifies GPU
- Installs UV package manager
- Installs project dependencies via `uv sync`
- Installs CUDA-enabled PyTorch
- Verifies CUDA is working
- Checks model wrapper imports correctly

### `cloud_data_setup.sh`
- Creates `data/` and `checkpoints/` directories
- Checks if training data exists
- Shows how to generate data if missing

## Monitoring

```bash
watch -n 1 nvidia-smi    # GPU usage
```

## Troubleshooting

**Out of Memory**: Reduce `batch_size` in `model/blt_configs/meta_blt.yaml` (try 4 or 2)

**CUDA not available**: Check `nvidia-smi` shows the GPU, verify driver supports CUDA 12.4+

**Model download slow**: `itazap/blt-1b-hf` is ~9 GB. First run downloads it, subsequent runs use cache.

**Data Issues**: Verify `data/phonemized.txt` exists and contains one phonemized sequence per line
