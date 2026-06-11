"""Simple monitoring dashboard for Meta BLT training."""

import json
import time
from pathlib import Path

from model.monitor import TrainingMonitor


def print_training_status(log_dir: str = "logs"):
    """Print current training status from logs.

    Args:
        log_dir: Directory containing training logs
    """
    log_path = Path(log_dir)
    metrics_file = log_path / "training_metrics.jsonl"

    if not metrics_file.exists():
        print("No training logs found.")
        return

    monitor = TrainingMonitor(log_dir)
    monitor.print_summary()

    # Show recent checkpoints
    checkpoints_file = log_path / "checkpoints.jsonl"
    if checkpoints_file.exists():
        print("\n=== Recent Checkpoints ===")
        with open(checkpoints_file) as f:
            checkpoints = [json.loads(line) for line in f]
        for cp in checkpoints[-3:]:
            epoch = cp.get("epoch", "Unknown")
            path = cp.get("checkpoint_path", "Unknown")
            print(f"Epoch {epoch}: {Path(path).name}")


def watch_training(log_dir: str = "logs", update_interval: int = 10):
    """Watch training progress in real-time.

    Args:
        log_dir: Directory containing training logs
        update_interval: Seconds between updates
    """
    log_path = Path(log_dir)
    metrics_file = log_path / "training_metrics.jsonl"

    if not metrics_file.exists():
        print("No training logs found. Start training first.")
        return

    print("Watching training progress... (Ctrl+C to stop)")
    print("=" * 60)

    try:
        last_size = 0
        while True:
            if metrics_file.exists():
                current_size = metrics_file.stat().st_size
                if current_size > last_size:
                    # New data available
                    monitor = TrainingMonitor(log_dir)
                    latest = monitor.get_latest_metrics(1)
                    if latest:
                        m = latest[0]
                        print(
                            f"\r[{m['epoch']:02d}|{m['step']:04d}] "
                            f"Loss: {m['loss']:.4f} | "
                            f"PPL: {m['perplexity']:.2f} | "
                            f"GPU: {m['gpu_memory_gb']:.1f}GB | "
                            f"LR: {m['learning_rate']:.6f}    ",
                            end="",
                            flush=True,
                        )
                    last_size = current_size

            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Meta BLT training")
    parser.add_argument("--log-dir", default="logs", help="Directory containing training logs")
    parser.add_argument("--watch", action="store_true", help="Watch training in real-time")
    parser.add_argument(
        "--interval", type=int, default=10, help="Update interval for watch mode (seconds)"
    )

    args = parser.parse_args()

    if args.watch:
        watch_training(args.log_dir, args.interval)
    else:
        print_training_status(args.log_dir)
