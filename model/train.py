"""BLT training loop for the phonetic byte-level language model.

Usage:
    python model/train.py --config model/blt_configs/low_resource.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore

from model.dataset import PAD_TOKEN, VOCAB_SIZE, ByteSequenceDataset, collate_fn
from model.blt_model import create_blt_model


def load_config(path: str | Path) -> dict:
    """Load a YAML training configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


class PhoneticBLT(nn.Module):
    """Factory class to create BLT model based on configuration.

    This replaces the placeholder with the actual BLT implementation.
    Uses either the full BLT or lightweight variant based on config.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Determine if we should use lightweight variant
        use_lightweight = "debug" in str(config.get("config_path", "")) or \
                         config.get("model", {}).get("global_transformer", {}).get("num_layers", 12) <= 2

        # Create the actual BLT model
        self.blt_model = create_blt_model(config, use_lightweight=use_lightweight)

        # Store reference for compatibility
        self.model_forward = self.blt_model.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: byte IDs → next-byte logits.

        Args:
            x: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        return self.model_forward(x)

    def get_model_info(self) -> dict:
        """Get model information for debugging."""
        if hasattr(self.blt_model, 'get_model_info'):
            return self.blt_model.get_model_info()
        else:
            return {"model_type": "BLT", "config": self.config}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    grad_clip: float = 1.0,
    epoch: int = 1,
    writer: SummaryWriter | None = None,
) -> float:
    """Run one training epoch with mixed precision and gradient clipping.

    Args:
        model: The model to train.
        loader: Training data loader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Training device.
        scaler: Optional GradScaler for mixed precision training.
        grad_clip: Gradient clipping threshold (0 to disable).
        epoch: Current epoch number for logging.
        writer: Optional TensorBoard writer.

    Returns:
        Average loss per batch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits = model(inputs)
                mask = targets != PAD_TOKEN
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = (loss * mask.view(-1).float()).sum() / mask.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            mask = targets != PAD_TOKEN
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = (loss * mask.view(-1).float()).sum() / mask.sum()

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Log to TensorBoard every 100 batches
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(n_batches, 1)

    if writer is not None:
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Time/epoch", elapsed, epoch)

    print(f"  Epoch {epoch} completed in {elapsed:.1f}s — avg loss: {avg_loss:.4f}")

    return avg_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validation"
) -> tuple[float, float]:
    """Run validation and return loss and perplexity.

    Args:
        model: The model to validate.
        loader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        desc: Description for progress display.

    Returns:
        Tuple of (average_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            mask = targets != PAD_TOKEN
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = (loss * mask.view(-1).float()).sum() / mask.sum()

            total_loss += loss.item()
            total_tokens += mask.sum().item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"  {desc} — loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")

    return avg_loss, perplexity


def main(config_path: str, resume_from: str | None = None):
    config = load_config(config_path)
    config["config_path"] = config_path  # Store config path for model creation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for GPU availability and mixed precision support
    use_amp = device.type == "cuda" and torch.cuda.is_available()
    if use_amp:
        print("Mixed precision training enabled (AMP)")
    else:
        print("Mixed precision disabled (CPU or CUDA unavailable)")

    model = PhoneticBLT(config).to(device)

    # Print model information
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"Model Type: {model_info.get('model_type', 'BLT')}")
        print(f"Model parameters: {model_info.get('total_parameters', sum(p.numel() for p in model.parameters())):,}")
        if 'patch_dim' in model_info:
            print(f"Patch dimension: {model_info['patch_dim']}")
        if 'global_transformer_layers' in model_info:
            print(f"Transformer layers: {model_info['global_transformer_layers']}")
        if 'entropy_threshold' in model_info:
            print(f"Entropy threshold: {model_info['entropy_threshold']}")
    else:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup datasets
    train_cfg = config.get("training", {})
    data_path = train_cfg.get("data_path", "data/phonemized.txt")
    max_seq_len = config["model"]["max_sequence_length"]
    batch_size = train_cfg.get("batch_size", 8)

    # Load and split dataset
    full_dataset = ByteSequenceDataset(data_path=data_path, max_seq_len=max_seq_len)

    # Use subset for faster testing if debug config
    if "debug" in str(config_path):
        print("Using debug config - creating small subset for testing")
        full_dataset = full_dataset.subset(n=100, seed=42)

    # Simple train/val split (90/10)
    total_samples = len(full_dataset)
    val_start = int(total_samples * 0.9)
    train_dataset = full_dataset.subset(val_start, seed=42)
    val_dataset = full_dataset.subset(total_samples - val_start, seed=43)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-4))
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Learning rate scheduler
    epochs = train_cfg.get("epochs", 10)
    warmup_epochs = max(1, epochs // 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    # TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "logs"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")

    # Checkpoint resumption
    start_epoch = 1
    best_perplexity = float("inf")
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_perplexity = checkpoint.get("best_perplexity", float("inf"))
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Training loop
    grad_clip = train_cfg.get("grad_clip", 1.0)
    patience = train_cfg.get("early_stopping_patience", 3)
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_clip=grad_clip, epoch=epoch, writer=writer
        )

        # Validation
        val_loss, val_perplexity = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        if epoch > warmup_epochs:
            scheduler.step()
            if writer:
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Perplexity/validation", val_perplexity, epoch)

        # Checkpoint saving
        save_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save best model
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            save_path = save_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_perplexity": best_perplexity,
                "config": config,
            }, save_path)
            print(f"  [*] New best model saved (perplexity: {best_perplexity:.2f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save latest checkpoint
        latest_path = save_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_perplexity": best_perplexity,
            "config": config,
        }, latest_path)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    if writer:
        writer.close()
        print(f"\nTraining completed. Best perplexity: {best_perplexity:.2f}")
        print(f"TensorBoard logs: {log_dir}")
    else:
        print(f"\nTraining completed. Best perplexity: {best_perplexity:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phonetic BLT")
    parser.add_argument(
        "--config",
        type=str,
        default="model/blt_configs/low_resource.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    main(args.config, args.resume)
