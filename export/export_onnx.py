"""Export a trained Phonetic BLT checkpoint to quantized ONNX for Dasher C++ integration.

Usage:
    python export/export_onnx.py --checkpoint checkpoints/best.pt --output model.onnx
"""

import argparse


def export_to_onnx(
    pytorch_model_path: str, onnx_output_path: str, config_path: str | None = None
) -> None:
    """Load a trained model and export to ONNX with INT8 quantization.

    Args:
        pytorch_model_path: Path to the saved PyTorch state dict.
        onnx_output_path: Destination path for the ONNX model.
        config_path: Optional YAML config used during training.
    """
    print(f"Loading checkpoint from {pytorch_model_path}...")

    # TODO: Load config and instantiate model properly once BLT arch is implemented
    # For now this is a skeleton — actual model class needs to be imported
    raise NotImplementedError(
        "Wire up model instantiation from model.train.PhoneticBLT once architecture is finalized."
    )

    # The following is the target export flow:

    # model = PhoneticBLT(config)
    # model.load_state_dict(torch.load(pytorch_model_path, map_location="cpu"))
    # model.eval()

    # dummy_input = torch.randint(0, 256, (1, 64), dtype=torch.long)
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     onnx_output_path,
    #     export_params=True,
    #     opset_version=17,
    #     do_constant_folding=True,
    #     input_names=["byte_history"],
    #     output_names=["next_byte_logits"],
    #     dynamic_axes={
    #         "byte_history": {0: "batch_size", 1: "sequence_length"},
    #         "next_byte_logits": {0: "batch_size", 1: "sequence_length"},
    #     },
    # )

    # print("Applying INT8 quantization...")
    # quantized_path = onnx_output_path.replace(".onnx", "_int8.onnx")
    # quantize_dynamic(
    #     model_input=onnx_output_path,
    #     model_output=quantized_path,
    #     weight_type=QuantType.QUInt8,
    # )
    # print(f"Export complete: {quantized_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BLT to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--config", type=str, default=None, help="Training config YAML")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.config)
