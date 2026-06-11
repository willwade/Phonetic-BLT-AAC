# Cloud GPU Training Setup

This directory contains scripts for setting up Meta BLT training on cloud GPU instances.

## Quick Start

1. **Choose a cloud provider** (AWS, Google Cloud, Azure, etc.)
   - Select a GPU instance with ≥16GB VRAM (recommend 24GB+)
   - Examples: AWS p3.2xlarge (V100), Google Cloud n1-standard-4 with V100

2. **Rent the instance and SSH in**

3. **Clone the repository:**
   ```bash
   git clone https://github.com/willwade/Phonetic-BLT-AAC.git
   cd Phonetic-BLT-AAC
   ```

4. **Run the setup script:**
   ```bash
   chmod +x cloud_setup.sh
   ./cloud_setup.sh
   ```

5. **Configure your HF token:**
   ```bash
   chmod +x cloud_data_setup.sh
   ./cloud_data_setup.sh
   # Edit .env and add your HF_TOKEN
   nano .env
   ```

6. **Upload your training data** to `data/phonemized.txt`

7. **Start training:**
   ```bash
   python -m model.train --config model/blt_configs/meta_blt.yaml
   ```

## Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (24GB+ recommended)
- **CUDA**: CUDA 12.4+ compatible drivers
- **Python**: Python 3.12+ (will be installed by setup)
- **Storage**: ~50GB free space for data, checkpoints, and logs
- **HF Token**: Valid HuggingFace token with BLT-1B access

## What the Scripts Do

### `cloud_setup.sh`
- Detects and verifies GPU
- Installs UV package manager
- Installs project dependencies
- Installs GPU-enabled PyTorch
- Verifies CUDA support

### `cloud_data_setup.sh`
- Creates `.env` file template
- Sets up directories
- Checks for training data
- Validates HF token configuration

## Monitoring Training

While training is running, monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

**Out of Memory**: Reduce batch size in `model/blt_configs/meta_blt.yaml`

**HF Token Issues**: Ensure token has access to `facebook/blt-1b`

**Data Issues**: Verify `data/phonemized.txt` exists and contains phonemized text

## Cost Estimates

Typical cloud GPU costs (as of 2025):
- **AWS p3.2xlarge** (V100 16GB): ~$3-4/hour
- **Google Cloud V100**: ~$2-3/hour
- **AWS p3.8xlarge** (4x V100 64GB): ~$12/hour

For 10 epochs on full dataset, estimate 4-8 hours depending on dataset size.
