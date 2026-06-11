"""BLT training loop for the phonetic byte-level language model.

Usage:
    python model/train.py --config model/blt_configs/low_resource.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from model.dataset import PAD_TOKEN, VOCAB_SIZE, ByteSequenceDataset, collate_fn


def load_config(path: str | Path) -> dict:
    """Load a YAML training configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


class PhoneticBLT(nn.Module):
    """Placeholder BLT architecture stub.

    The full implementation should instantiate Meta's BLT layers:
    - Entropy patcher (segments bytes into dynamic patches)
    - Global latent transformer (cross-attention over patches)
    - Local decoder (byte-level output)

    See: https://github.com/facebookresearch/blt
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        # TODO: Replace with actual BLT architecture from facebookresearch/blt
        # These are placeholder layers matching the config dimensions
        hidden = model_cfg["global_transformer"]["hidden_dim"]
        self.embed = nn.Embedding(VOCAB_SIZE, hidden)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=model_cfg["global_transformer"]["num_heads"],
                dim_feedforward=hidden * 4,
                dropout=model_cfg["global_transformer"]["dropout"],
                batch_first=True,
            ),
            num_layers=model_cfg["global_transformer"]["num_layers"],
        )
        self.head = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: byte IDs → next-byte logits.

        Args:
            x: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        h = self.embed(x)
        h = self.transformer(h)
        return self.head(h)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)

        mask = targets != PAD_TOKEN
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = (loss * mask.view(-1).float()).sum() / mask.sum()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PhoneticBLT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = ByteSequenceDataset(
        data_path=config.get("data_path", "data/phonemized.txt"),
        max_seq_len=config["model"]["max_sequence_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4))
    criterion = nn.CrossEntropyLoss(reduction="none")

    epochs = config.get("epochs", 10)
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")

    save_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phonetic BLT")
    parser.add_argument(
        "--config",
        type=str,
        default="model/blt_configs/low_resource.yaml",
        help="YAML config path",
    )
    args = parser.parse_args()
    main(args.config)
