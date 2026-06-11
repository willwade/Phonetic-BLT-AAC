"""Latency and memory profiling for BLT model variants (PyTorch, ONNX, ONNX-INT8).

Usage:
    python export/benchmark.py --model checkpoints/best.pt
    python export/benchmark.py --model model_int8.onnx --backend onnx
"""

import argparse


def benchmark_pytorch(
    model_path: str,
    seq_lengths: list[int] | None = None,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict[str, dict]:
    """Profile a PyTorch model at various sequence lengths.

    Returns:
        Dict mapping seq_length → {"latency_ms": float, "throughput_tok_s": float}
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256, 512]

    # TODO: Load actual model
    print(f"[PyTorch] Benchmarking {model_path}...")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Warmup runs: {n_warmup}, Measurement runs: {n_runs}")
    print("  NOTE: Implement once model architecture is finalized.")

    return {}


def benchmark_onnx(
    model_path: str,
    seq_lengths: list[int] | None = None,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict[str, dict]:
    """Profile an ONNX model via ONNX Runtime at various sequence lengths.

    Returns:
        Dict mapping seq_length → {"latency_ms": float, "throughput_tok_s": float}
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256, 512]

    # TODO: Load with onnxruntime.InferenceSession
    print(f"[ONNX] Benchmarking {model_path}...")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Warmup runs: {n_warmup}, Measurement runs: {n_runs}")
    print("  NOTE: Implement once ONNX export is finalized.")

    return {}


def measure_peak_memory() -> int:
    """Return peak RSS in bytes (macOS / Linux via resource module)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BLT model variants")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model file (.pt or .onnx)"
    )
    parser.add_argument("--backend", type=str, choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    if args.backend == "pytorch":
        benchmark_pytorch(args.model, args.seq_lengths, args.warmup, args.runs)
    else:
        benchmark_onnx(args.model, args.seq_lengths, args.warmup, args.runs)
