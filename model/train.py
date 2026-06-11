"""BLT training loop for the phonetic byte-level language model.

Uses the community-converted BLT weights (itazap/blt-1b-hf) loaded via
HuggingFace Transformers. Fine-tunes on phonemized AAC conversational data.

Automatically pushes best checkpoint to HuggingFace Hub when HF_REPO is set
in config or environment. This ensures you don't lose progress if a rented
GPU instance shuts down.

Usage:
    python model/train.py --config model/blt_configs/meta_blt.yaml
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = Any

from model.blt_transformers import MetaBLTWrapper
from model.dataset import PAD_TOKEN, ByteSequenceDataset, collate_fn


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def push_to_hub(
    model: nn.Module,
    repo_id: str,
    epoch: int,
    perplexity: float,
    config: dict,
) -> None:
    """Push the model checkpoint to HuggingFace Hub.

    Requires HF_TOKEN environment variable to be set.
    Creates the repo if it doesn't exist.

    Args:
        model: The trained model.
        repo_id: HF repo ID (e.g. "username/phonetic-blt-aac").
        epoch: Current epoch number.
        perplexity: Best validation perplexity.
        config: Training config dict.
    """
    try:
        from huggingface_hub import HfApi

        token = os.getenv("HF_TOKEN")
        if not token:
            print("  [SKIP] HF_TOKEN not set, skipping hub upload")
            return

        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

        save_dir = Path("checkpoints/hub_upload")
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_dir)

        meta = {
            "epoch": epoch,
            "perplexity": perplexity,
            "base_model": "itazap/blt-1b-hf",
            "config": config,
        }
        import json

        (save_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=repo_id,
            commit_message=f"Epoch {epoch} - perplexity {perplexity:.2f}",
            token=token,
        )

        print(f"  Pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  [WARN] Hub upload failed: {e}")
        print("  Checkpoints are saved locally - no data lost")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    grad_clip: float = 1.0,
    epoch: int = 1,
    writer: SummaryWriter | None = None,
) -> float:
    """Run one training epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                mask = targets != PAD_TOKEN
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
                )
                loss = (loss * mask.view(-1).float()).sum() / mask.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            mask = targets != PAD_TOKEN
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1).float()).sum() / mask.sum()

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(n_batches, 1)

    if writer is not None:
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Time/epoch", elapsed, epoch)

    print(f"  Epoch {epoch} completed in {elapsed:.1f}s - avg loss: {avg_loss:.4f}")
    return avg_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            mask = targets != PAD_TOKEN
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1).float()).sum() / mask.sum()

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"  Validation - loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")
    return avg_loss, perplexity


def main(config_path: str, resume_from: str | None = None):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # HF Hub repo for checkpoint uploads (set in config or env)
    hub_repo = config.get("hub_repo") or os.getenv("HF_REPO")
    if hub_repo:
        print(f"Checkpoint uploads: https://huggingface.co/{hub_repo}")
    else:
        print("Checkpoint uploads: disabled (set hub_repo in config or HF_REPO env)")

    # Load model
    print("Loading BLT model...")
    wrapper = MetaBLTWrapper()
    model = wrapper.load_model()
    model = model.to(device)

    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision enabled")

    # Dataset
    train_cfg = config.get("training", {})
    data_path = train_cfg.get("data_path", "data/phonemized.txt")
    max_seq_len = config["model"]["max_sequence_length"]
    batch_size = train_cfg.get("batch_size", 8)

    full_dataset = ByteSequenceDataset(data_path=data_path, max_seq_len=max_seq_len)

    if "debug" in str(config_path):
        print("Debug mode - using 100 sample subset")
        full_dataset = full_dataset.subset(n=100, seed=42)

    total_samples = len(full_dataset)
    val_start = int(total_samples * 0.9)
    train_dataset = full_dataset.subset(val_start, seed=42)
    val_dataset = full_dataset.subset(total_samples - val_start, seed=43)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-4))
    epochs = train_cfg.get("epochs", 10)
    warmup_epochs = max(1, epochs // 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    # TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "logs"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")

    # Resume
    start_epoch = 1
    best_perplexity = float("inf")
    if resume_from:
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_perplexity = ckpt.get("best_perplexity", float("inf"))
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Train
    grad_clip = train_cfg.get("grad_clip", 1.0)
    patience = train_cfg.get("early_stopping_patience", 3)
    patience_counter = 0

    print(f"\nTraining {epochs} epochs, batch_size={batch_size}...")
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            grad_clip=grad_clip,
            epoch=epoch,
            writer=writer,
        )

        val_loss, val_perplexity = validate(model, val_loader, device)

        if epoch > warmup_epochs:
            scheduler.step()

        if writer:
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Perplexity/validation", val_perplexity, epoch)

        # Save local checkpoints
        save_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)

        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_perplexity": best_perplexity,
                    "config": config,
                },
                save_dir / "best.pt",
            )
            print(f"  New best model (perplexity: {best_perplexity:.2f})")

            # Push to HF Hub on every improvement
            if hub_repo:
                push_to_hub(model, hub_repo, epoch, best_perplexity, config)

            patience_counter = 0
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_perplexity": best_perplexity,
                "config": config,
            },
            save_dir / "latest.pt",
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final push after training completes
    if hub_repo:
        print("\nPushing final checkpoint to Hub...")
        push_to_hub(model, hub_repo, epoch, best_perplexity, config)

    if writer:
        writer.close()

    print(f"\nDone. Best perplexity: {best_perplexity:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BLT for Phonetic AAC")
    parser.add_argument("--config", type=str, default="model/blt_configs/meta_blt.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.resume)
