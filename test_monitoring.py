"""Test the monitoring system with simulated training data."""

import tempfile
from pathlib import Path

from model.monitor import TrainingMonitor, format_time, get_gpu_memory


def test_monitoring():
    """Test the monitoring system with simulated training."""
    print("=== Testing Monitoring System ===")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = TrainingMonitor(temp_dir)

        print("1. Logging simulated training metrics...")
        # Simulate some training progress
        for epoch in range(1, 4):
            for step in range(1, 4):
                loss = 5.0 - (epoch * 0.5) - (step * 0.1)  # Decreasing loss
                perplexity = 150.0 - (epoch * 10) - (step * 5)  # Decreasing perplexity
                lr = 0.0001 * (0.9 ** (epoch - 1))  # Learning rate decay

                monitor.log_metrics(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    perplexity=perplexity,
                    learning_rate=lr,
                    gpu_memory=get_gpu_memory(),
                    phase="train",
                )

        print("   Logged 12 simulated training steps")

        print("2. Testing metric retrieval...")
        metrics = monitor.get_latest_metrics(5)
        print(f"   Retrieved {len(metrics)} metrics")
        assert len(metrics) == 5, "Should get 5 metrics"

        print("3. Testing summary print...")
        monitor.print_summary()

        print("4. Testing checkpoint info saving...")
        monitor.save_checkpoint_info(
            epoch=3,
            checkpoint_path=Path("checkpoints/best.pt"),
            perplexity=95.5,
            checkpoint_type="best",
        )

        print("5. Testing format_time utility...")
        test_time = 7325  # 2h + 5m + 25s
        formatted = format_time(test_time)
        print(f"   {test_time}s -> {formatted}")
        assert "2h" in formatted, "Should contain hours"

        print("\n[OK] All monitoring tests passed!")


if __name__ == "__main__":
    test_monitoring()
