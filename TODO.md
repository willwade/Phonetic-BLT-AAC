# Phonetic-BLT-AAC — Implementation Roadmap

## Phase 1: Data Pipeline

- [ ] **`data/download.py`** — Hugging Face streaming download with AAC-score filtering
  - [ ] Stream `figmtu/aac_c4_deberta_classified` with configurable threshold
  - [ ] Add optional support for subtitle corpus (`figmtu/aac_subtitle_deberta_classified`)
  - [ ] Add high-density shortcut (`figmtu/aac_c4_deberta_classified_0.90`)
  - [ ] Write to `data/raw_conversations.txt`
  - [ ] Add resume/checkpoint support for interrupted downloads

- [ ] **`data/phonemize.py`** — Parallel G2P transliteration (Epitran → SAMPA)
  - [ ] Implement `SampaPhonemizer` wrapper class in `data/utils.py`
  - [ ] Multiprocessing chunk-based pipeline using `ProcessPoolExecutor`
  - [ ] Benchmark throughput on 1M-line corpus
  - [ ] Handle OOV and Unicode edge cases

- [ ] **`data/utils.py`** — Shared data utilities
  - [ ] `SampaPhonemizer` class: wraps Epitran for `eng-Latn` → 1-byte SAMPA
  - [ ] Text cleaning: normalize whitespace, strip control chars, deduplicate
  - [ ] Byte-encoding helpers: raw text → byte sequences for BLT input
  - [ ] Train/val/test split logic

## Phase 2: Model

- [ ] **`model/dataset.py`** — PyTorch `Dataset` for byte sequences
  - [ ] Load phonemized SAMPA corpus
  - [ ] Encode to byte-level token IDs (0–255 + special tokens)
  - [ ] Sliding-window sequence generation with configurable context length
  - [ ] Collation function for variable-length batches

- [ ] **`model/train.py`** — BLT training loop
  - [ ] Load `low_resource.yaml` config
  - [ ] Instantiate BLT architecture (entropy patcher → global transformer → local decoder)
  - [ ] Custom byte-level cross-entropy loss
  - [ ] Mixed-precision training (AMP)
  - [ ] Checkpoint saving and resumption
  - [ ] Validation perplexity evaluation every N steps
  - [ ] TensorBoard or WandB logging integration

- [ ] **`model/blt_configs/low_resource.yaml`** — Compact model config
  - [x] Initial config written (see plan spec)
  - [ ] Tune hyperparameters after first training run

- [ ] **`model/blt_configs/`** — Additional configs
  - [ ] `medium_resource.yaml` — 1B-scale variant for GPU training
  - [ ] `debug.yaml` — Tiny model for CI/testing

## Phase 3: Export & Optimization

- [ ] **`export/export_onnx.py`** — ONNX export with quantization
  - [ ] Load PyTorch checkpoint
  - [ ] Export to ONNX with dynamic axes (batch, sequence length)
  - [ ] Apply INT8 dynamic quantization
  - [ ] Validate output parity between PyTorch and ONNX (cosine similarity)

- [ ] **`export/benchmark.py`** — Latency and memory profiling
  - [ ] Measure per-token latency on CPU (target: < 5ms)
  - [ ] Measure peak RSS memory usage
  - [ ] Profile with varying sequence lengths (32, 64, 128, 256, 512)
  - [ ] Compare PyTorch vs ONNX vs ONNX-INT8

## Phase 4: Integration & Validation

- [ ] **Dasher C++ integration** (separate repo)
  - [ ] ONNX Runtime inference wrapper in DasherCore
  - [ ] Byte-stream interface between Dasher UI and phonetic model
  - [ ] End-to-end latency measurement with Dasher driving

- [ ] **Evaluation pipeline**
  - [ ] Perplexity on held-out AAC conversational test set
  - [ ] Phoneme prediction accuracy
  - [ ] User study: prediction helpfulness in Dasher context
  - [ ] Compare against character-level baseline

## Phase 5: Documentation & Release

- [ ] Complete README with quickstart guide
- [ ] Add architecture diagrams
- [ ] Write contributing guidelines
- [ ] Tag v0.1.0 release with pre-trained checkpoint + ONNX bundle
