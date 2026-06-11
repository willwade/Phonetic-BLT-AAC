"""Quick test of Meta BLT model loading."""

from dotenv import load_dotenv

from model.blt_transformers import MetaBLTWrapper

# Load environment variables
load_dotenv()

print("Testing Meta BLT model loading...")

try:
    # Try to load Meta's BLT-1B model
    blt_wrapper = MetaBLTWrapper(model_name="facebook/blt-1b")
    model = blt_wrapper.load_model()

    print("Meta BLT model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Try a simple forward pass
    import torch

    dummy_input = torch.randint(0, 256, (1, 32))  # Batch size 1, sequence length 32
    print(f"Testing forward pass with input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Forward pass successful! Output logits shape: {output.logits.shape}")
    print(f"Output vocab size: {output.logits.shape[-1]}")

    # Verify the output makes sense
    batch_size, seq_len, vocab_size = output.logits.shape
    assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
    assert vocab_size == 260, f"Expected vocab size 260, got {vocab_size}"
    print("All assertions passed! Model is working correctly.")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    print("Note: Make sure HF_TOKEN is set in .env file and you have approved access")
