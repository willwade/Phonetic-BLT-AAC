"""Export a trained Meta BLT checkpoint to quantized ONNX for Dasher C++ integration.

Usage:
    python export/export_onnx.py --checkpoint checkpoints/best.pt --output model.onnx
"""

import argparse
import tempfile

import torch
import torch.onnx

from dotenv import load_dotenv
from model.blt_transformers import MetaBLTWrapper

# Load environment variables
load_dotenv()


def export_to_onnx(
    checkpoint_path: str,
    onnx_output_path: str = "model.onnx",
    quantize: bool = True,
    opset_version: int = 17,
) -> str:
    """Load a trained Meta BLT checkpoint and export to ONNX with optional INT8 quantization.

    Args:
        checkpoint_path: Path to the saved PyTorch checkpoint (.pt file)
        onnx_output_path: Destination path for the ONNX model
        quantize: Whether to apply INT8 quantization
        opset_version: ONNX opset version to use

    Returns:
        Path to the exported ONNX model (quantized if quantize=True)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load the Meta BLT model
    blt_wrapper = MetaBLTWrapper()
    model = blt_wrapper.load_model()

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully")

    # Move model to CPU for export
    model = model.cpu()

    # Determine input shape from config or use defaults
    config = checkpoint.get("config", {})
    max_seq_len = config.get("model", {}).get("max_sequence_length", 512)
    batch_size = 1  # Dasher needs single token prediction

    print(f"Exporting with batch_size={batch_size}, max_seq_len={max_seq_len}")

    # Create dummy input for export
    dummy_input = torch.randint(0, 256, (batch_size, max_seq_len), dtype=torch.long)

    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_bytes"],
        output_names=["output_logits"],
        dynamic_axes={
            "input_bytes": {0: "batch_size", 1: "sequence_length"},
            "output_logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"Exported to: {onnx_output_path}")

    # Verify the exported model
    print("Verifying exported model...")
    import onnx
    onnx_model = onnx.load(onnx_output_path)
    print(f"ONNX model loaded successfully")
    print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"Outputs: {[out.name for out in onnx_model.graph.output]}")

    if quantize:
        print("\nApplying INT8 quantization...")
        quantized_path = onnx_output_path.replace(".onnx", "_int8.onnx")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            # Use dynamic quantization (works without calibration data)
            quantize_dynamic(
                model_input=onnx_output_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8,
            )
            print(f"Quantized model saved to: {quantized_path}")

            # Verify quantized model
            quantized_onnx = onnx.load(quantized_path)
            print("Quantized ONNX model loaded successfully")
            return quantized_path

        except ImportError:
            print("Warning: onnxruntime not available, skipping quantization")
            print("Install with: pip install onnxruntime")
            return onnx_output_path
    else:
        print("Skipping quantization (quantize=False)")
        return onnx_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Meta BLT to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    exported_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        onnx_output_path=args.output,
        quantize=not args.no_quantize,
        opset_version=args.opset,
    )

    print(f"\nExport complete! Final model: {exported_path}")
