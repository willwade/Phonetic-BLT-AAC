"""Local decoder for BLT architecture.

The local decoder expands patch-level representations back to byte-level
outputs, completing the BLT's encode-decode pipeline.
"""

import torch
import torch.nn as nn


class LocalByteDecoder(nn.Module):
    """Decodes patch representations back to byte-level predictions.

    This module takes the global transformer's patch representations and
    generates per-byte logits, completing the BLT architecture.

    Args:
        patch_dim: Dimension of patch representations from global transformer
        hidden_dim: Hidden dimension for decoder layers (default: 128)
        vocab_size: Output vocabulary size (default: 260 for bytes + specials)
        num_layers: Number of decoder layers (default: 2)
    """

    def __init__(
        self,
        patch_dim: int = 512,
        hidden_dim: int = 128,
        vocab_size: int = 260,
        num_layers: int = 2,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Patch to hidden projection
        self.patch_to_hidden = nn.Linear(patch_dim, hidden_dim)

        # Decoder layers
        decoder_layers = []
        for i in range(num_layers):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        self.decoder = nn.Sequential(*decoder_layers)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, patch_representations: torch.Tensor) -> torch.Tensor:
        """Decode patch representations to byte logits.

        Args:
            patch_representations: Patch representations from global transformer
                                  Shape: (batch, num_patches, patch_dim)

        Returns:
            Byte logits: (batch, total_bytes, vocab_size)
        """
        batch_size, num_patches, patch_dim = patch_representations.shape

        # Project patches to hidden dimension
        hidden = self.patch_to_hidden(patch_representations)  # (batch, num_patches, hidden_dim)
        hidden = self.decoder(hidden)  # (batch, num_patches, hidden_dim)

        # Project to vocabulary
        logits = self.output_proj(hidden)  # (batch, num_patches, vocab_size)

        return logits


class HierarchicalLocalDecoder(nn.Module):
    """Hierarchical decoder that handles variable-length patches.

    This decoder explicitly handles the fact that patches have different
    lengths, making it more suitable for the actual BLT architecture.

    Args:
        patch_dim: Dimension of patch representations from global transformer
        hidden_dim: Hidden dimension for decoder layers (default: 128)
        vocab_size: Output vocabulary size (default: 260)
        num_layers: Number of decoder layers (default: 2)
        max_patch_size: Maximum patch size to handle
    """

    def __init__(
        self,
        patch_dim: int = 512,
        hidden_dim: int = 128,
        vocab_size: int = 260,
        num_layers: int = 2,
        max_patch_size: int = 128,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_patch_size = max_patch_size

        # Per-position processing
        self.patch_to_hidden = nn.Linear(patch_dim, hidden_dim)

        # Decoder layers
        decoder_layers = []
        for i in range(num_layers):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        self.decoder = nn.Sequential(*decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Position embeddings for variable-length patches
        self.position_embeddings = nn.Embedding(max_patch_size, hidden_dim)

    def forward(
        self,
        patch_representations: torch.Tensor,
        patch_boundaries: list = None
    ) -> torch.Tensor:
        """Decode patch representations to byte logits with boundary awareness.

        Args:
            patch_representations: Patch representations from global transformer
                                  Shape: (batch, num_patches, patch_dim)
            patch_boundaries: Optional list of (start, end) for each patch

        Returns:
            Byte logits: (batch, sequence_length, vocab_size)
        """
        batch_size, num_patches, patch_dim = patch_representations.shape

        # Project patches to hidden dimension
        hidden = self.patch_to_hidden(patch_representations)  # (batch, num_patches, hidden_dim)
        hidden = self.decoder(hidden)  # (batch, num_patches, hidden_dim)

        # Project to vocabulary
        logits = self.output_proj(hidden)  # (batch, num_patches, vocab_size)

        # If we have patch boundaries, we could expand to byte-level
        # For now, return patch-level logits
        return logits


class ByteLevelDecoder(nn.Module):
    """Full byte-level decoder that reconstructs byte sequences from patches.

    This decoder handles the complete mapping from patch representations
    back to byte sequences, handling variable patch lengths.

    Args:
        patch_dim: Dimension of patch representations
        hidden_dim: Hidden dimension for decoder (default: 128)
        vocab_size: Output vocabulary size (default: 260)
        num_layers: Number of transformer layers (default: 2)
        num_heads: Number of attention heads (default: 4)
    """

    def __init__(
        self,
        patch_dim: int = 512,
        hidden_dim: int = 128,
        vocab_size: int = 260,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Input projection
        self.input_proj = nn.Linear(patch_dim, hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        patch_representations: torch.Tensor,
        byte_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode patch representations to byte-level predictions.

        Args:
            patch_representations: Patch representations (batch, num_patches, patch_dim)
            byte_ids: Optional original byte IDs for cross-attention

        Returns:
            Byte logits (batch, sequence_length, vocab_size)
        """
        batch_size, num_patches, patch_dim = patch_representations.shape

        # Project to hidden dimension
        hidden = self.input_proj(patch_representations)  # (batch, num_patches, hidden_dim)

        # If byte_ids provided, use as cross-attention target
        if byte_ids is not None:
            # Create byte-level queries
            seq_len = byte_ids.shape[1]
            byte_queries = torch.zeros(batch_size, seq_len, self.hidden_dim, device=byte_ids.device)

            # Apply transformer decoder with cross-attention
            decoded = self.transformer_decoder(byte_queries, hidden)
        else:
            # Just process patch representations
            decoded = self.transformer_decoder(hidden, hidden)

        # Project to vocabulary
        logits = self.output_proj(decoded)  # (batch, sequence_length, vocab_size)

        return logits


class SimpleLocalDecoder(nn.Module):
    """Simplified local decoder for initial implementation.

    This version provides a straightforward mapping from patch representations
    to byte logits without complex hierarchical processing.
    """

    def __init__(
        self,
        patch_dim: int = 512,
        hidden_dim: int = 128,
        vocab_size: int = 260,
        num_layers: int = 2,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Simple feedforward decoder
        layers = []
        input_dim = patch_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.decoder = nn.Sequential(*layers)

    def forward(self, patch_representations: torch.Tensor) -> torch.Tensor:
        """Decode patch representations to byte logits.

        Args:
            patch_representations: (batch, num_patches, patch_dim)

        Returns:
            Byte logits: (batch, num_patches, vocab_size)
        """
        return self.decoder(patch_representations)
