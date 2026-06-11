"""Entropy-based byte patching for BLT architecture.

The patcher dynamically groups bytes into patches based on local entropy,
allocating more compute capacity where data complexity is higher.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class EntropyPatcher(nn.Module):
    """Dynamic byte patching based on local entropy computation.

    Computes entropy over sliding windows and inserts patch boundaries
    where entropy exceeds the threshold. This allows the model to allocate
    more capacity to complex regions and be more efficient on simple regions.

    Args:
        vocab_size: Size of byte vocabulary (260 for our use: 256 bytes + PAD + EOS)
        hidden_dim: Hidden dimension for entropy prediction model
        entropy_threshold: Entropy threshold for patch boundaries (default: 1.34)
        window_size: Window size for entropy computation
        max_patch_size: Maximum size of a patch (for safety)
    """

    def __init__(
        self,
        vocab_size: int = 260,
        hidden_dim: int = 128,
        entropy_threshold: float = 1.34,
        window_size: int = 8,
        max_patch_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.max_patch_size = max_patch_size

        # Small entropy prediction model
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.entropy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output entropy probability
        )

    def compute_local_entropy(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Compute entropy predictions for each position.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)

        Returns:
            Entropy predictions (batch, seq_len)
        """
        embedded = self.embedding(byte_ids)  # (batch, seq_len, hidden)
        entropy_pred = self.entropy_predictor(embedded).squeeze(-1)  # (batch, seq_len)
        return entropy_pred

    def create_patches(
        self, byte_ids: torch.Tensor, entropy_pred: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Create dynamic patches based on entropy threshold.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)
            entropy_pred: Optional pre-computed entropy predictions

        Returns:
            Tuple of (patches, boundaries) where:
            - patches: List of tensors, one per patch
            - boundaries: List of (start, end) indices for each patch
        """
        if entropy_pred is None:
            entropy_pred = self.compute_local_entropy(byte_ids)

        patches = []
        boundaries = []
        batch_size, seq_len = byte_ids.shape

        for batch_idx in range(batch_size):
            sequence = byte_ids[batch_idx]
            entropies = entropy_pred[batch_idx]

            patch_start = 0
            current_patch_size = 0

            for pos in range(seq_len):
                current_patch_size += 1

                # Check if we should create a patch boundary
                should_split = False

                # Split if entropy exceeds threshold (high complexity)
                if entropies[pos] > self.entropy_threshold:
                    should_split = True

                # Split if we exceed max patch size (safety)
                if current_patch_size >= self.max_patch_size:
                    should_split = True

                # Split if we're at the end of sequence
                if pos == seq_len - 1:
                    should_split = True

                if should_split and current_patch_size > 1:
                    # Create a patch
                    patch_end = pos + 1
                    patch = sequence[patch_start:patch_end]
                    patches.append(patch)
                    boundaries.append((patch_start, patch_end))

                    # Reset for next patch
                    patch_start = patch_end
                    current_patch_size = 0

        return patches, boundaries

    def forward(self, byte_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Forward pass: create patches from byte sequences.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)

        Returns:
            Tuple of (patches, boundaries)
        """
        entropy_pred = self.compute_local_entropy(byte_ids)
        patches, boundaries = self.create_patches(byte_ids, entropy_pred)
        return patches, boundaries

    def get_patch_stats(self, byte_ids: torch.Tensor) -> dict:
        """Compute statistics about patching for analysis.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)

        Returns:
            Dictionary with patching statistics
        """
        patches, boundaries = self.forward(byte_ids)

        if not patches:
            return {
                "num_patches": 0,
                "avg_patch_size": 0,
                "min_patch_size": 0,
                "max_patch_size": 0,
            }

        patch_sizes = [len(patch) for patch in patches]

        return {
            "num_patches": len(patches),
            "avg_patch_size": sum(patch_sizes) / len(patch_sizes),
            "min_patch_size": min(patch_sizes),
            "max_patch_size": max(patch_sizes),
        }


class SimpleEntropyPatcher(nn.Module):
    """Simplified entropy patcher using statistical entropy instead of learned model.

    This version computes actual Shannon entropy over byte distributions
    in sliding windows, making it simpler to train and debug.
    """

    def __init__(
        self,
        entropy_threshold: float = 1.34,
        window_size: int = 16,
        max_patch_size: int = 128,
        min_patch_size: int = 2,
    ):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size

    def compute_window_entropy(self, window: List[int]) -> float:
        """Compute Shannon entropy for a window of bytes.

        Args:
            window: List of byte IDs

        Returns:
            Entropy value
        """
        if len(window) < 2:
            return 0.0

        # Count byte frequencies
        freq = {}
        for byte_id in window:
            freq[byte_id] = freq.get(byte_id, 0) + 1

        # Compute Shannon entropy
        entropy = 0.0
        window_len = len(window)
        for count in freq.values():
            if count > 0:
                probability = count / window_len
                entropy -= probability * math.log2(probability)

        return entropy

    def create_patches(self, byte_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Create patches based on statistical entropy.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)

        Returns:
            Tuple of (patches, boundaries)
        """
        patches = []
        boundaries = []
        batch_size, seq_len = byte_ids.shape

        for batch_idx in range(batch_size):
            sequence = byte_ids[batch_idx].tolist()

            patch_start = 0
            current_patch_size = 0

            for pos in range(seq_len):
                current_patch_size += 1

                # Compute entropy for current window
                window_start = max(0, pos - self.window_size // 2)
                window_end = min(seq_len, pos + self.window_size // 2 + 1)
                window = sequence[window_start:window_end]
                entropy = self.compute_window_entropy(window)

                # Check if we should create a patch boundary
                should_split = False

                # Split if entropy exceeds threshold (high complexity)
                if entropy > self.entropy_threshold and current_patch_size >= self.min_patch_size:
                    should_split = True

                # Split if we exceed max patch size (safety)
                if current_patch_size >= self.max_patch_size:
                    should_split = True

                # Split if we're at the end of sequence
                if pos == seq_len - 1:
                    should_split = True

                if should_split and current_patch_size >= self.min_patch_size:
                    # Create a patch
                    patch_end = pos + 1
                    patch_tensor = torch.tensor(sequence[patch_start:patch_end], dtype=torch.long)
                    patches.append(patch_tensor)
                    boundaries.append((patch_start, patch_end))

                    # Reset for next patch
                    patch_start = patch_end
                    current_patch_size = 0

        return patches, boundaries

    def forward(self, byte_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Forward pass: create patches from byte sequences.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)

        Returns:
            Tuple of (patches, boundaries)
        """
        return self.create_patches(byte_ids)

    def get_patch_stats(self, byte_ids: torch.Tensor) -> dict:
        """Compute statistics about patching for analysis.

        Args:
            byte_ids: Tensor of byte IDs (batch, seq_len)

        Returns:
            Dictionary with patching statistics
        """
        patches, boundaries = self.forward(byte_ids)

        if not patches:
            return {
                "num_patches": 0,
                "avg_patch_size": 0,
                "min_patch_size": 0,
                "max_patch_size": 0,
            }

        patch_sizes = [len(patch) for patch in patches]

        return {
            "num_patches": len(patches),
            "avg_patch_size": sum(patch_sizes) / len(patch_sizes),
            "min_patch_size": min(patch_sizes),
            "max_patch_size": max(patch_sizes),
        }