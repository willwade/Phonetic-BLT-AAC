# Meta BLT Model Access Guide

## Current Status

The project now supports **Meta's pre-trained Byte Latent Transformer (BLT)** models with intelligent fallback:

### ✅ What Works Now
- **Automatic BLT loading**: System tries to load Meta's BLT first
- **Intelligent fallback**: Falls back to custom BLT if Meta unavailable
- **Training continues**: Both approaches support full training pipeline
- **No breaking changes**: Existing code continues to work

### 🔑 Getting Access to Meta BLT Models

To use Meta's pre-trained BLT models instead of our custom implementation:

#### 1. Request Access
Visit **[Meta BLT models on HuggingFace](https://huggingface.co/facebook/blt-1b)** and:
- Click "Request Access" 
- Fill out the access request form
- Wait for approval (usually 1-2 days)

#### 2. Set Up Authentication
Once approved, your HuggingFace token will work automatically.

The token should be stored in `.env` (git-ignored for security) as:
```
HF_TOKEN=your_token_here
```

#### 3. Available Models
- **`facebook/blt-1b`**: 1B parameter BLT model (recommended for fine-tuning)
- **`facebook/blt-7b`**: 7B parameter BLT model (larger, more powerful)
- **`facebook/blt-entropy`**: Entropy prediction model for patching

## Benefits of Using Meta BLT

### Why Use Meta's Pre-trained BLT?

1. **Proven Architecture**: Trained on massive datasets (8T training bytes)
2. **Better Performance**: Matches token-based LLMs at scale
3. **Faster Training**: Start from pre-trained weights vs training from scratch
4. **Research Validation**: Battle-tested in Meta's research

### Performance Comparison

| Approach | Training Time | Final Performance | Research Validation |
|----------|---------------|-------------------|---------------------|
| **Meta BLT (fine-tuning)** | Hours-Days | State-of-the-art | ✅ Meta validated |
| **Custom BLT (from scratch)** | Days-Weeks | Unknown | ❌ Experimental |

## Current Fallback Behavior

The system automatically:
1. **Tries Meta BLT first**: Attempts to load `facebook/blt-1b`
2. **Falls back gracefully**: Uses custom BLT if unavailable
3. **Continues training**: Both approaches work with existing pipeline

### Sample Output
```
Setting up BLT model...
Attempting to load Meta's pre-trained BLT...
Could not load Meta BLT: Gated repository access required
Falling back to custom BLT implementation...
Model Type: Custom BLT implementation
Model parameters: 8,860
✅ Training continues successfully
```

## Next Steps

### Option 1: Get Meta BLT Access (Recommended)
1. Visit https://huggingface.co/facebook/blt-1b
2. Request access (you have the token ready)
3. Once approved, training will automatically use Meta BLT

### Option 2: Continue with Custom BLT
- Current implementation works for development and testing
- Can be used for prototyping and experimentation
- Will switch to Meta BLT automatically once access is approved

## Technical Details

### Model Loading
```python
from model.blt_transformers import setup_blt_model

model, model_info = setup_blt_model(config, prefer_meta=True)
# Automatically tries Meta BLT, falls back to custom BLT
```

### Architecture Comparison

**Meta BLT:**
- Dual-model architecture (entropy predictor + main model)
- Pre-trained on 8T bytes of data
- Validated scaling up to 8B parameters
- Dynamic patching based on learned entropy

**Custom BLT:**
- Single unified model
- Statistical entropy computation
- Trained from scratch on AAC data
- Experimental dynamic patching

## Configuration

The training configs automatically support both approaches:

```yaml
# model/blt_configs/low_resource.yaml
# Works with both Meta BLT and custom BLT
model:
  vocab_size: 260
  max_sequence_length: 512
  # ... configuration compatible with both approaches
```

---

**Recommendation**: Get access to Meta BLT for production use. The custom implementation is excellent for development and testing, but Meta's pre-trained models will provide superior performance.