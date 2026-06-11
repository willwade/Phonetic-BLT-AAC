# Phonetic-BLT-AAC — Developer Guide

This document is the working reference for anyone implementing this pipeline.
Every section tells you: what exists now, what you need to write, what tests to
add, and what could catch you out.

---

## Prerequisites

| Requirement | Why |
|---|---|
| Python 3.12 | BLT deps and type syntax (`X \| None`) require 3.10+; pinned to 3.12 |
| [uv](https://docs.astral.sh/uv/) | Package manager — do not use pip or conda directly |
| CUDA 12.1+ GPU | Training only. Data pipeline runs fine on CPU |
| 32 GB RAM | Phonemization and dataset loading are memory-heavy |
| ~50 GB disk | Raw C4 corpus is large even with streaming; phonemized output is smaller |

### Setup

```bash
git clone https://github.com/your-org/phonetic-blt-aac.git
cd phonetic-blt-aac
uv sync --extra dev        # creates .venv, installs everything
```

### GPU / CUDA setup (IMPORTANT)

`uv sync` installs **CPU-only** PyTorch. If you are training on a GPU machine,
you MUST override this before running any training:

```bash
# Replace CPU torch with CUDA nightly
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

Verify it worked:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

If this prints `False`, you are still on CPU torch and training will be
extremely slow. Check `nvidia-smi` first to confirm the driver sees the GPU.

### Linting

Run these before every commit:

```bash
uv run ruff check          # lint
uv run ruff format --check # format check
uv run ruff format         # apply formatting
uv run ruff check --fix    # auto-fix lint issues
```

---

## Step 0: Environment Validation

**Status: DONE**

- [x] `pyproject.toml` with all dependencies
- [x] `.python-version` pinned to 3.12
- [x] `uv.lock` committed
- [x] Ruff config (lint + format)
- [x] Mypy config
- [x] `.gitignore` with data artifacts, checkpoints, ONNX files

**Tests to write:**

- [ ] `tests/test_environment.py` — verify `torch`, `datasets`, `epitran`, `onnx`, `onnxruntime` all import without error

---

## Step 1: Data Download

**Files:** `data/download.py` (functional), `data/__init__.py`

**Current state:** Working streaming download from
`figmtu/aac_c4_deberta_classified` with configurable score threshold.

### What to implement

- [ ] Add `--source` flag to switch between corpora:
  - `c4` (default): `figmtu/aac_c4_deberta_classified` — 4.35B tokens
  - `subtitles`: `figmtu/aac_subtitle_deberta_classified` — 52.6M tokens
  - `c4-fast`: `figmtu/aac_c4_deberta_classified_0.90` — pre-filtered high-density
- [ ] Resume support: if `output_path` exists, count existing lines and skip
  that many rows from the streaming dataset
- [ ] Progress: add `tqdm` progress bar showing rows scanned vs kept
- [ ] Deduplication: track SHA-256 of each line, skip exact duplicates

### Gotchas

- The C4 corpus is **streaming only** — you cannot `len()` it. Always iterate.
- `aac_score` column may not exist on all splits. Use `.get()` with a default.
- Hugging Face may rate-limit unauthenticated requests. Run
  `huggingface-cli login` if downloading large amounts.
- The dataset is multilingual; you may need to filter to English only.

### Tests to write

- [ ] `tests/test_download.py`:
  - Mock `load_dataset` and verify rows below threshold are dropped
  - Verify output file contains one line per kept row
  - Verify resume skips already-written lines
  - Verify deduplication drops exact duplicates

---

## Step 2: Phonemization

**Files:** `data/phonemize.py` (functional), `data/utils.py` (functional)

**Current state:** `SampaPhonemizer` wraps Epitran for `eng-Latn`. Parallel
processing via `ProcessPoolExecutor` is wired up.

### What to implement

- [ ] Error handling: Epitran can raise on unusual Unicode. The `try/except` in
  `convert_word` silently returns `""` — add logging of skipped words with a
  count summary at the end
- [ ] Benchmarks: run on a 100K-line sample and report lines/second. Target
  should be > 1000 lines/sec on 8 cores
- [ ] Verify SAMPA output is actually 1-byte (ASCII range 0-127). If Epitran
  produces multi-byte IPA characters, add a sanitization step
- [ ] Option to output byte-encoded file directly (skip the text intermediate)

### Gotchas

- **Epitran is slow.** Each `ProcessPoolExecutor` worker creates its own
  `Epitran` instance (~2s startup). The `chunk_size=5000` amortizes this, but
  if you lower it you will see a performance cliff.
- Epitran produces **IPA**, not SAMPA. The class is called `SampaPhonemizer`
  but currently passes through Epitran's IPA output unchanged. You need to
  either: (a) add an IPA→SAMPA mapping table, or (b) confirm that BLT can
  train on IPA bytes directly and rename the class.
- The multiprocessing import means `data/phonemize.py` must be run as a script
  or via `uv run`. Do not import `parallel_phonemize` from an interactive
  session without `if __name__ == "__main__"` guard (already present).

### Tests to write

- [ ] `tests/test_phonemize.py`:
  - `SampaPhonemizer.convert_word("hello")` returns a non-empty string
  - `SampaPhonemizer.convert_word("")` returns `""`
  - `clean_text` strips control characters and collapses whitespace
  - `text_to_bytes` roundtrips: `bytes_to_text(text_to_bytes(s)) == s`
  - `train_val_test_split` returns correct proportions within 1% tolerance

---

## Step 3: Dataset & DataLoader

**Files:** `model/dataset.py` (functional)

**Current state:** `ByteSequenceDataset` loads a text file, encodes to bytes,
and creates sliding-window (input, target) pairs. `collate_fn` pads to equal
length.

### What to implement

- [ ] Add `__repr__` showing dataset size and sequence count
- [ ] Add a `subset(n)` method that returns a random `n`-sample subset for
  debugging
- [ ] Verify that the sliding window does not create trivially overlapping
  samples (same bytes appearing in both input and target of adjacent samples)
- [ ] Consider shuffling sequences (not just samples) at construction time

### Gotchas

- The dataset loads **the entire phonemized file into memory** as integer lists.
  A 10M-line file with average 40 bytes/line = ~400M integers = ~3.2 GB RAM.
  If this is too large, switch to memory-mapped or on-disk format.
- `PAD_TOKEN=256` and `EOS_TOKEN=257` are valid special tokens only because
  byte values are 0-255. Do not change these without also updating `VOCAB_SIZE`.
- `collate_fn` uses `zip(*batch, strict=True)` — this will raise if any sample
  has mismatched input/target lengths (should never happen, but good to know).

### Tests to write

- [ ] `tests/test_dataset.py`:
  - Create a small temp file with 5 lines, verify dataset length
  - Verify each sample is a (input, target) tuple of `torch.long` tensors
  - Verify target is shifted by 1 from input
  - Verify `collate_fn` pads shorter sequences to match the longest
  - Verify `PAD_TOKEN` does not appear in any target values from actual data
  - Verify empty file produces 0 samples without error

---

## Step 4: Model Architecture (THE HARD PART)

**Files:** `model/train.py` (skeleton), `model/blt_configs/low_resource.yaml`

**Current state:** `PhoneticBLT` is a **placeholder** — it uses a stock
`nn.TransformerEncoder`. This is NOT a Byte Latent Transformer. It will train
and produce logits, but the result will not have the dynamic patching behavior
that makes BLT efficient for byte-level prediction.

### What you actually need to do

This is the core work of the project. You need to implement three components:

1. **Entropy Patcher** (`model/blt_patcher.py` — new file)
   - Takes a byte sequence, computes local entropy over a sliding window
   - Inserts patch boundaries where entropy exceeds `entropy_threshold` (1.34)
   - Outputs a "patched" sequence: variable-length byte groups with boundary
   - Reference: [BLT paper §3.1](https://arxiv.org/abs/2407.02394),
     [facebookresearch/blt](https://github.com/facebookresearch/blt)

2. **Global Latent Transformer** (part of `model/train.py` or new module)
   - Cross-attention over the patched representation
   - This is where most parameters live (12 layers × 512 hidden × 8 heads)
   - Must support variable-length patches (not fixed-size)

3. **Local Decoder** (`model/blt_decoder.py` — new file)
   - Expands patches back to bytes
   - 2 layers, 128 hidden dim
   - Outputs per-byte logits over vocab_size=260

### Implementation approach

- [ ] Clone and study `facebookresearch/blt`:
  ```bash
  git clone https://github.com/facebookresearch/blt.git /tmp/blt-reference
  ```
  Focus on: `blt/models/blt.py`, `blt/models/patcher.py`
- [ ] Extract the entropy patcher, global transformer, and local decoder into
  standalone `nn.Module` subclasses in this repo
- [ ] Replace `PhoneticBLT` in `model/train.py` with the real architecture
- [ ] Verify the model runs a forward pass with a dummy batch before training
- [ ] Add a `model/blt_configs/debug.yaml` with tiny dimensions for fast CI

### Gotchas

- **Meta's BLT repo uses Fairseq**, not vanilla PyTorch. You will need to
  extract the relevant modules and remove Fairseq dependencies. This is
  non-trivial. Budget at least 2-3 days for this step.
- The entropy patcher uses a **small local model** to predict where to place
  boundaries. Meta provides pre-trained weights at
  `facebook/blt-entropy1B`. You can use these or train your own.
- BLT's `forward()` signature is different from a standard transformer. The
  `dataset.py` output (byte IDs) may need reshaping to match what BLT expects.
- The `low_resource.yaml` config assumes a 512-byte context. BLT's patching
  makes this effectively longer — verify memory usage at training time.

### Tests to write

- [ ] `tests/test_model.py`:
  - `PhoneticBLT(config)` instantiates without error
  - Forward pass with random byte input `(2, 64)` produces logits of shape
    `(2, 64, 260)`
  - Forward pass with varying sequence lengths works (no hardcoded size)
  - Parameter count matches expected order of magnitude (~10-50M for
    `low_resource.yaml`)
  - Gradients flow through all components (no detached tensors)

---

## Step 5: Training Loop

**Files:** `model/train.py` (partially functional)

**Current state:** Basic training loop runs with the placeholder model. Missing:
AMP, validation, checkpoint resumption, logging.

### What to implement

- [ ] Mixed-precision training via `torch.amp.autocast("cuda")` and
  `GradScaler`
- [ ] Validation loop: compute perplexity on held-out split every N steps
- [ ] Learning rate scheduler (cosine with warmup is standard for transformers)
- [ ] Checkpoint resumption: save optimizer state, epoch, step count alongside
  model weights
- [ ] TensorBoard logging: training loss, validation perplexity, learning rate
  (or WandB if `wandb` extra is installed)
- [ ] Gradient clipping at 1.0 (standard for transformer training)
- [ ] Early stopping on validation perplexity

### Gotchas

- The current loss computation masks `PAD_TOKEN` but does **not** mask
  `EOS_TOKEN`. Decide whether EOS should contribute to the loss.
- `num_workers=2` in the DataLoader may be too low for large datasets. Scale
  with CPU count, but watch out for shared memory limits on some systems.
- `batch_size=8` with `max_seq_len=512` is ~4K tokens per batch. On a 16 GB GPU
  this should fit; on 8 GB you may need to reduce batch size or use gradient
  accumulation.

### Tests to write

- [ ] `tests/test_train.py`:
  - `load_config("model/blt_configs/low_resource.yaml")` returns valid dict
  - `train_one_epoch` runs without error on a 10-sample dataset
  - Loss decreases over 3 epochs on trivial repeated data (sanity check)
  - Checkpoint file is created and loadable

---

## Step 6: ONNX Export

**Files:** `export/export_onnx.py` (skeleton — raises `NotImplementedError`)

**Current state:** The target export flow is documented in comments but not
wired up. Blocked on Step 4 (real BLT architecture).

### What to implement

- [ ] Remove `NotImplementedError` and wire up model loading from checkpoint
- [ ] Implement the ONNX export with dynamic axes (see commented code)
- [ ] Add INT8 dynamic quantization via `onnxruntime.quantization`
- [ ] Validate output parity: run same input through PyTorch and ONNX, verify
  cosine similarity > 0.999
- [ ] Add `--opset` flag (default 17, may need higher for attention ops)

### Gotchas

- Dynamic quantization may not support all BLT operations. If it fails, fall
  back to FP16 quantization or static quantization with calibration data.
- ONNX export of models with dynamic shapes (variable-length patches from BLT)
  can be tricky. You may need to fix the patch size at export time.
- The exported model must run on **CPU** for Dasher integration. Test on a
  machine without CUDA to confirm ONNX Runtime uses CPU providers.

### Tests to write

- [ ] `tests/test_export.py`:
  - Export produces a valid `.onnx` file (loadable by `onnx.load`)
  - ONNX model produces output of correct shape for multiple input sizes
  - INT8 quantized model is smaller than FP32 original
  - PyTorch vs ONNX output parity within tolerance

---

## Step 7: Benchmarking

**Files:** `export/benchmark.py` (skeleton)

**Current state:** Prints placeholder messages. Blocked on Steps 4 and 6.

### What to implement

- [ ] `benchmark_pytorch`: load `.pt` checkpoint, run inference at each seq
  length, measure median latency over `n_runs`
- [ ] `benchmark_onnx`: load `.onnx` with `onnxruntime.InferenceSession`,
  measure the same way
- [ ] `measure_peak_memory`: wrap inference in `tracemalloc` for accurate
  peak measurement
- [ ] Print results as a markdown table to stdout
- [ ] Latency target: **< 5ms per token on CPU** for the `low_resource` config

### Tests to write

- [ ] `tests/test_benchmark.py`:
  - Verify output dict has expected keys
  - Verify latency is > 0 (not a timing bug)

---

## Step 8: Evaluation Pipeline (new)

**Files:** `model/evaluate.py` — new file

Not yet created. Needed to measure model quality.

### What to implement

- [ ] Compute perplexity on held-out AAC test split
- [ ] Top-1 and top-5 byte prediction accuracy
- [ ] Compare against a character-level bigram baseline
- [ ] Log results to TensorBoard or JSON

### Tests to write

- [ ] `tests/test_evaluate.py`:
  - Perplexity is finite and positive on random model
  - Perfect predictions give perplexity = 1.0

---

## File Checklist

| File | Status | Tests exist | Blocked by |
|---|---|---|---|
| `data/download.py` | Functional | No | — |
| `data/phonemize.py` | Functional | No | — |
| `data/utils.py` | Functional | No | — |
| `model/dataset.py` | Functional | No | — |
| `model/train.py` | Placeholder model, basic loop | No | Step 4 |
| `model/blt_configs/low_resource.yaml` | Written | N/A | — |
| `export/export_onnx.py` | Skeleton (`NotImplementedError`) | No | Step 4 |
| `export/benchmark.py` | Skeleton | No | Steps 4, 6 |
| `model/evaluate.py` | Does not exist | No | Step 4 |
| `tests/` | Does not exist | — | — |

### New files to create

- [ ] `model/blt_patcher.py` — Entropy-based byte patching module
- [ ] `model/blt_decoder.py` — Local byte-level decoder
- [ ] `model/evaluate.py` — Evaluation metrics
- [ ] `tests/__init__.py`
- [ ] `tests/test_environment.py`
- [ ] `tests/test_download.py`
- [ ] `tests/test_phonemize.py`
- [ ] `tests/test_dataset.py`
- [ ] `tests/test_model.py`
- [ ] `tests/test_train.py`
- [ ] `tests/test_export.py`
- [ ] `tests/test_benchmark.py`
- [ ] `tests/test_evaluate.py`
- [ ] `model/blt_configs/debug.yaml` — Tiny model for CI
