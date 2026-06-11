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

        raw = Path(data_path).read_text(encoding="utf-8")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        self.sequences: list[list[int]] = []
        for line in lines:
            byte_seq = list(line.encode("utf-8"))
            byte_seq.append(EOS_TOKEN)
            if len(byte_seq) >= 2:
                self.sequences.append(byte_seq)

        self.samples = self._build_samples()

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
    inputs, targets = zip(*batch, strict=True)
    max_len = max(t.shape[0] for t in inputs)

    padded_inputs = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets, strict=True)):
        padded_inputs[i, : inp.shape[0]] = inp
        padded_targets[i, : tgt.shape[0]] = tgt

    return padded_inputs, padded_targets
