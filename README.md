# Phonetic-BLT-AAC

A phonetic language model for [Dasher](https://www.inference.org.uk/dasher/) built on Meta's Byte Latent Transformer (BLT), trained on EMNLP 2025 AAC-scored conversational datasets.

## What This Does

 This project trains a **phonetic** byte-level language model — meaning it predicts the next byte based on phonetic (SAMPA) representations of conversational speech — to improve prediction quality for AAC (Augmentative and Alternative Communication) users.

The pipeline:
1. **Download** AAC-scored conversational data from Hugging Face
2. **Phonemize** text to 1-byte SAMPA using Epitran (parallel G2P)
3. **Fine-tune** Meta's pre-trained BLT model on byte sequences
4. **Export** to quantized ONNX for real-time Dasher C++ integration

## Quick Start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/phonetic-blt-aac.git
cd phonetic-blt-aac

# Install all dependencies (uv creates .venv automatically)
uv sync --extra dev

# Download and filter AAC data
uv run data/download.py --threshold 0.85

# Phonemize to SAMPA
uv run data/phonemize.py --input data/raw_conversations.txt --output data/phonemized.txt
```

### GPU training (optional)

`uv sync` installs CPU-only PyTorch. For GPU training you **must** override:

```bash
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
uv run python -c "import torch; print(torch.cuda.is_available())"  # verify True

uv run model/train.py --config model/blt_configs/low_resource.yaml
```

### Linting

```bash
uv run ruff check          # lint
uv run ruff format         # format
uv run ruff check --fix    # auto-fix
```

### Model Training


For cloud GPU training setup, see [CLOUD_SETUP.md](CLOUD_SETUP.md).



## Key References

| Resource | Link |
|----------|------|
| Meta BLT Repository | [facebookresearch/blt](https://github.com/facebookresearch/blt) |
| BLT 1B Weights | [facebook/blt-1b](https://huggingface.co/facebook/blt-1b) |
| BLT Entropy Predictor | [facebook/blt-entropy1B](https://huggingface.co/facebook/blt-entropy1B) |
| AAC C4 Corpus (4.35B tokens) | [figmtu/aac_c4_deberta_classified](https://huggingface.co/datasets/figmtu/aac_c4_deberta_classified) |
| AAC Subtitles (52.6M tokens) | [figmtu/aac_subtitle_deberta_classified](https://huggingface.co/datasets/figmtu/aac_subtitle_deberta_classified) |
| High-Density Filtered | [figmtu/aac_c4_deberta_classified_0.90](https://huggingface.co/datasets/figmtu/aac_c4_deberta_classified_0.90) |
| Epitran G2P | [dmortensen/epitran](https://github.com/dmortensen/epitran) |
| EMNLP 2025 AAC Project | [OSF ajm7t](https://osf.io/ajm7t/) |


## License

MIT — see [LICENSE](LICENSE).
