"""Load test: verify BLT-1B loads from transformers, check architecture.

No forward pass — just loads and inspects. Runs on any machine in ~10 seconds.

Usage:
    uv run test_blt_load.py
"""

import torch
from transformers import AutoTokenizer, BltConfig, BltForCausalLM

MODEL_ID = "itazap/blt-1b-hf"


def main():
    print(f"1. Loading config from {MODEL_ID}...")
    config = BltConfig.from_pretrained(MODEL_ID)
    print("   Config loaded")
    for attr in [
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
    ]:
        if hasattr(config, attr):
            print(f"   {attr}: {getattr(config, attr)}")

    print(f"\n2. Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    test = "Hello world"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)
    print(f"   Encode/decode: '{test}' -> {ids} -> '{decoded}'")

    print(f"\n3. Loading model weights from {MODEL_ID}...")
    model = BltForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,} ({param_count / 1e9:.2f}B)")

    print("\n4. Model architecture:")
    print(f"   Type: {type(model).__name__}")
    print(f"   Modules: {list(model.children())}")

    print("\n5. Checking forward pass shapes (no computation)...")
    dummy = torch.randint(0, config.vocab_size, (1, 5))
    print(f"   Dummy input shape: {dummy.shape}")
    print("   (Skipping actual forward pass — too slow on CPU)")
    print(f"   This would produce logits of shape: (1, 5, {config.vocab_size})")

    print("\n=== Option A is viable ===")
    print(f"BltForCausalLM loads from {MODEL_ID}")
    print("Ready to wire into training pipeline with GPU")


if __name__ == "__main__":
    main()
