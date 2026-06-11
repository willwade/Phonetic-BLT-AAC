"""PyTorch Dataset for byte-level BLT training on phonemized SAMPA text."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD_TOKEN = 256
EOS_TOKEN = 257
VOCAB_SIZE = 260


class ByteSequenceDataset(Dataset):
    """Loads phonemized SAMPA text and encodes each line as a byte sequence.

    Produces sliding-window (input, target) pairs suitable for autoregressive
    training: given bytes ``[0..T-1]``, predict ``[1..T]``.

    Args:
        data_path: Path to phonemized text file (one SAMPA sequence per line).
        max_seq_len: Maximum context window length in bytes.
        stride: Sliding window stride. Defaults to ``max_seq_len // 2``.
    """

    def __init__(
        self,
        data_path: str | Path,
        max_seq_len: int = 512,
        stride: int | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2
        self.data_path = Path(data_path)

        raw = self.data_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        self.sequences: list[list[int]] = []
        for line in lines:
            byte_seq = list(line.encode("utf-8"))
            byte_seq.append(EOS_TOKEN)
            if len(byte_seq) >= 2:
                self.sequences.append(byte_seq)

        self.samples = self._build_samples()

    def __repr__(self) -> str:
        """Return string representation with dataset size and sequence count."""
        return (f"ByteSequenceDataset(path={self.data_path.name}, "
                f"sequences={len(self.sequences)}, samples={len(self.samples)}, "
                f"max_seq_len={self.max_seq_len}, stride={self.stride})")

    def subset(self, n: int, seed: int | None = None) -> "ByteSequenceDataset":
        """Return a random subset of n samples for debugging.

        Args:
            n: Number of samples to include in the subset.
            seed: Optional random seed for reproducibility.

        Returns:
            A new dataset instance with only n randomly selected samples.
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Create a new dataset instance with the same parameters
        subset_dataset = ByteSequenceDataset(
            data_path=self.data_path,
            max_seq_len=self.max_seq_len,
            stride=self.stride,
        )

        # Randomly select n samples
        if n < len(self.samples):
            subset_dataset.samples = random.sample(self.samples, n)
        else:
            subset_dataset.samples = self.samples.copy()

        return subset_dataset

    def verify_no_trivial_overlap(self) -> dict[str, int]:
        """Verify that sliding window doesn't create trivially overlapping samples.

        Check if adjacent samples have the same bytes appearing in both input
        and target positions, which would indicate the stride is too small.

        Returns:
            Dict with overlap statistics: total_checks, overlaps_found, max_overlap_ratio
        """
        overlaps = 0
        max_overlap = 0
        total_checks = 0

        for i in range(len(self.samples) - 1):
            inp1, tgt1 = self.samples[i]
            inp2, tgt2 = self.samples[i + 1]

            # Check if target of sample 1 overlaps with input of sample 2
            if len(tgt1) > 0 and len(inp2) > 0:
                # Find the maximum overlap where tgt1 ending matches inp2 beginning
                max_possible_overlap = min(len(tgt1), len(inp2))
                current_overlap = 0

                for k in range(1, max_possible_overlap + 1):
                    if tgt1[-k:] == inp2[:k]:
                        current_overlap = k
                    else:
                        break

                if current_overlap > 0:
                    overlaps += 1
                    max_overlap = max(max_overlap, current_overlap)

                total_checks += 1

        overlap_ratio = max_overlap / self.max_seq_len if self.max_seq_len > 0 else 0

        return {
            "total_checks": total_checks,
            "overlaps_found": overlaps,
            "max_overlap_bytes": max_overlap,
            "max_overlap_ratio": overlap_ratio,
        }

    def _build_samples(self) -> list[tuple[list[int], list[int]]]:
        """Create sliding-window (input, target) pairs from all sequences."""
        samples: list[tuple[list[int], list[int]]] = []
        for seq in self.sequences:
            for start in range(0, len(seq) - 1, self.stride):
                end = start + self.max_seq_len + 1
                chunk = seq[start:end]
                if len(chunk) < 2:
                    continue
                inp = chunk[:-1]
                tgt = chunk[1:]
                inp = inp[: self.max_seq_len]
                tgt = tgt[: self.max_seq_len]
                samples.append((inp, tgt))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences within a batch to equal length."""
    if not batch:
        return torch.zeros((0, 0), dtype=torch.long), torch.zeros((0, 0), dtype=torch.long)

    inputs, targets = zip(*batch, strict=True)
    max_len = max(t.shape[0] for t in inputs)

    padded_inputs = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets, strict=True)):
        padded_inputs[i, : inp.shape[0]] = inp
        padded_targets[i, : tgt.shape[0]] = tgt

    return padded_inputs, padded_targets
