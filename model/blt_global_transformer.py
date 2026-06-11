"""Global latent transformer for BLT architecture.

The global transformer processes patch representations with cross-attention,
forming the core computational component of the BLT architecture.
"""

import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """Embed raw byte patches into continuous representations.

    Args:
        vocab_size: Size of byte vocabulary (260 for bytes + specials)
        patch_dim: Output dimension for patch embeddings
        max_patch_size: Maximum patch size to handle
    """

    def __init__(
        self,
        vocab_size: int = 260,
        patch_dim: int = 512,
        max_patch_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_dim = patch_dim
        self.max_patch_size = max_patch_size

        # Byte embedding
        self.byte_embedding = nn.Embedding(vocab_size, patch_dim)

        # Positional embedding for patches
        self.patch_embedding = nn.Linear(max_patch_size * patch_dim, patch_dim)

        # Position encoding
        self.pos_embedding = nn.Embedding(1000, patch_dim)  # Max sequence position

    def forward(self, byte_patches: list, positions: torch.Tensor = None) -> torch.Tensor:
        """Embed byte patches into continuous representations.

        Args:
            byte_patches: List of byte ID tensors, one per patch
            positions: Optional position IDs for patches

        Returns:
            Patch embeddings: (num_patches, patch_dim)
        """
        device = next(self.parameters()).device
        patch_embeddings = []

        for i, patch in enumerate(byte_patches):
            # Embed each byte in the patch
            patch = patch.to(device)
            byte_embeds = self.byte_embedding(patch)  # (patch_len, patch_dim)

            # Flatten patch embedding
            patch_flat = byte_embeds.view(-1)  # (patch_len * patch_dim)

            # Pad or truncate to max_patch_size
            if patch_flat.shape[0] < self.max_patch_size * self.patch_dim:
                padded = torch.zeros(self.max_patch_size * self.patch_dim, device=device)
                padded[:patch_flat.shape[0]] = patch_flat
                patch_flat = padded
            else:
                patch_flat = patch_flat[:self.max_patch_size * self.patch_dim]

            # Project to patch dimension
            patch_embed = self.patch_embedding(patch_flat)  # (patch_dim,)
            patch_embeddings.append(patch_embed)

        # Stack and add positional encoding
        patch_embeddings = torch.stack(patch_embeddings)  # (num_patches, patch_dim)

        if positions is not None:
            pos_embeds = self.pos_embedding(positions.to(device))
            patch_embeddings = patch_embeddings + pos_embeds

        return patch_embeddings


class GlobalLatentTransformer(nn.Module):
    """Global transformer with cross-attention over patch representations.

    This is the core component of BLT where most parameters live.
    It processes patch representations with transformer layers.

    Args:
        patch_dim: Dimension of patch representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for feedforward layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        patch_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process patch embeddings through transformer.

        Args:
            patch_embeddings: Patch embeddings (batch, num_patches, patch_dim)
            padding_mask: Optional mask for padding positions

        Returns:
            Processed patch representations (batch, num_patches, patch_dim)
        """
        # Apply transformer
        processed = self.transformer(patch_embeddings, src_key_padding_mask=padding_mask)
        return processed


class PatchCrossAttention(nn.Module):
    """Cross-attention between byte-level and patch-level representations.

    This module allows information flow between byte and patch hidden
    representations, as mentioned in the BLT paper.

    Args:
        byte_dim: Dimension of byte-level representations
        patch_dim: Dimension of patch-level representations
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
    """

    def __init__(
        self,
        byte_dim: int = 128,
        patch_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.byte_dim = byte_dim
        self.patch_dim = patch_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Projection layers to match dimensions
        self.byte_to_patch = nn.Linear(byte_dim, patch_dim)
        self.patch_to_byte = nn.Linear(patch_dim, byte_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=patch_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        byte_repr: torch.Tensor,
        patch_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention between byte and patch representations.

        Args:
            byte_repr: Byte-level representations (batch, num_bytes, byte_dim)
            patch_repr: Patch-level representations (batch, num_patches, patch_dim)

        Returns:
            Tuple of (enhanced_byte_repr, enhanced_patch_repr)
        """
        # Project to matching dimensions
        byte_patches = self.byte_to_patch(byte_repr)  # (batch, num_bytes, patch_dim)

        # Apply cross-attention
        enhanced_patch_repr = patch_repr
        for attention_layer in self.cross_attention_layers:
            # Query: patches, Key/Value: bytes
            attended_patches, _ = attention_layer(
                query=enhanced_patch_repr,
                key=byte_patches,
                value=byte_patches,
            )
            enhanced_patch_repr = enhanced_patch_repr + attended_patches  # Residual

        # Project back to byte dimension
        enhanced_byte_repr = self.patch_to_byte(enhanced_patch_repr)

        return enhanced_byte_repr, enhanced_patch_repr


class ByteSequenceMemory(nn.Module):
    """Byte-sequence memory for BLT as mentioned in the paper.

    This component provides a memory mechanism for byte sequences,
    improving the model's ability to handle long-range dependencies.

    Args:
        memory_dim: Dimension of memory vectors
        num_memories: Number of memory slots
    """

    def __init__(
        self,
        memory_dim: int = 512,
        num_memories: int = 16,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_memories = num_memories

        # Learnable memory vectors
        self.memory_vectors = nn.Parameter(
            torch.randn(num_memories, memory_dim) * 0.02
        )

        # Memory update network
        self.memory_update = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(
        self,
        patch_repr: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Read from and update byte-sequence memory.

        Args:
            patch_repr: Patch representations (batch, num_patches, memory_dim)
            memory_mask: Optional mask for memory slots

        Returns:
            Memory-enhanced patch representations
        """
        batch_size = patch_repr.shape[0]
        device = patch_repr.device

        # Expand memory for batch
        memory = self.memory_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_memories, memory_dim)

        # Compute attention between patches and memory
        # For simplicity, we'll just concatenate
        memory_expanded = memory.mean(dim=1, keepdim=True)  # (batch, 1, memory_dim)
        memory_broadcast = memory_expanded.expand(-1, patch_repr.shape[1], -1)  # (batch, num_patches, memory_dim)

        # Combine with patch representations
        combined = torch.cat([patch_repr, memory_broadcast], dim=-1)
        enhanced = self.memory_update(combined) + patch_repr  # Residual

        return enhanced


class CompleteGlobalTransformer(nn.Module):
    """Complete global transformer combining all BLT components.

    This integrates the patch embedding, transformer processing, cross-attention,
    and memory components into the full global transformer.

    Args:
        vocab_size: Size of byte vocabulary
        patch_dim: Dimension of patch representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for feedforward layers
        max_patch_size: Maximum patch size
        dropout: Dropout rate
        use_cross_attention: Whether to use cross-attention
        use_memory: Whether to use byte-sequence memory
    """

    def __init__(
        self,
        vocab_size: int = 260,
        patch_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        max_patch_size: int = 128,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        use_memory: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_dim = patch_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_memory = use_memory

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            vocab_size=vocab_size,
            patch_dim=patch_dim,
            max_patch_size=max_patch_size
        )

        # Global transformer
        self.global_transformer = GlobalLatentTransformer(
            patch_dim=patch_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Optional cross-attention
        if use_cross_attention:
            self.cross_attention = PatchCrossAttention(
                byte_dim=patch_dim // 4,  # Smaller dimension for byte-level
                patch_dim=patch_dim,
                num_heads=num_heads // 2,
                num_layers=2,
            )

        # Optional memory
        if use_memory:
            self.byte_memory = ByteSequenceMemory(
                memory_dim=patch_dim,
                num_memories=16,
            )

    def forward(
        self,
        byte_patches: list,
        positions: torch.Tensor = None
    ) -> torch.Tensor:
        """Process byte patches through complete global transformer.

        Args:
            byte_patches: List of byte ID tensors, one per patch
            positions: Optional position IDs for patches

        Returns:
            Processed patch representations (batch, num_patches, patch_dim)
        """
        # Embed patches
        patch_embeddings = self.patch_embedding(byte_patches, positions)  # (num_patches, patch_dim)

        # Add batch dimension
        patch_embeddings = patch_embeddings.unsqueeze(0)  # (1, num_patches, patch_dim)

        # Apply global transformer
        processed = self.global_transformer(patch_embeddings)  # (1, num_patches, patch_dim)

        # Optional memory enhancement
        if self.use_memory:
            processed = self.byte_memory(processed)

        return processed.squeeze(0)  # (num_patches, patch_dim)
