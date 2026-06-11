"""Training monitoring utilities for Meta BLT."""

import json
import time
from pathlib import Path
from typing import Any

import torch


class TrainingMonitor:
    """Monitor training progress and log metrics."""

    def __init__(self, log_dir: Path | str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "training_metrics.jsonl"
        self.start_time = time.time()

    def log_metrics(
        self,
        epoch: int,
        step: int,
        loss: float,
        perplexity: float,
        learning_rate: float,
        gpu_memory: float = 0.0,
        **extra_metrics: Any,
    ):
        """Log training metrics to JSONL file.

        Args:
            epoch: Current epoch number
            step: Current step within epoch
            loss: Training loss value
            perplexity: Validation perplexity
            learning_rate: Current learning rate
            gpu_memory: GPU memory usage in GB
            **extra_metrics: Additional metrics to log
        """
        elapsed = time.time() - self.start_time

        metrics = {
            "timestamp": time.time(),
            "elapsed": elapsed,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "gpu_memory_gb": gpu_memory,
            **extra_metrics,
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def get_latest_metrics(self, last_n: int = 10) -> list[dict]:
        """Get the last N metrics entries.

        Args:
            last_n: Number of recent entries to retrieve

        Returns:
            List of metric dictionaries
        """
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line))

        return metrics[-last_n:]

    def print_summary(self):
        """Print a summary of recent training progress."""
        metrics = self.get_latest_metrics(5)
        if not metrics:
            print("No metrics logged yet.")
            return

        print("=== Recent Training Progress ===")
        for i, m in enumerate(metrics[-5:], 1):
            print(f"Step {m['step']} (Epoch {m['epoch']}):")
            print(f"  Loss: {m['loss']:.4f}")
            print(f"  Perplexity: {m['perplexity']:.2f}")
            print(f"  LR: {m['learning_rate']:.6f}")
            print(f"  GPU: {m['gpu_memory_gb']:.1f}GB")
            print()

    def save_checkpoint_info(self, epoch: int, checkpoint_path: Path, **info):
        """Save information about a checkpoint.

        Args:
            epoch: Epoch number when checkpoint was saved
            checkpoint_path: Path to the checkpoint file
            **info: Additional information to save
        """
        checkpoint_info = {
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": time.time(),
            **info,
        }

        info_file = self.log_dir / "checkpoints.jsonl"
        with open(info_file, "a") as f:
            f.write(json.dumps(checkpoint_info) + "\n")


def get_gpu_memory() -> float:
    """Get current GPU memory usage in GB.

    Returns:
        GPU memory usage in GB, or 0.0 if no GPU available
    """
    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            return memory_allocated
        except Exception:
            return 0.0
    return 0.0


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)