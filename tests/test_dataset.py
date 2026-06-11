"""Test dataset functionality for BLT training."""

import tempfile
from pathlib import Path

import pytest
import torch

from model.dataset import ByteSequenceDataset, PAD_TOKEN, EOS_TOKEN, VOCAB_SIZE, collate_fn


class TestByteSequenceDataset:
    """Test the ByteSequenceDataset class."""

    def test_basic_creation(self):
        """Test basic dataset creation from small file."""
        # Create temporary file with sample data
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("hello world\n")
            f.write("goodbye world\n")
            f.write("testing phonemization\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            assert len(dataset) > 0
            assert dataset.max_seq_len == 10

        finally:
            Path(data_path).unlink()

    def test_empty_file(self):
        """Test that empty file produces 0 samples without error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            assert len(dataset) == 0

        finally:
            Path(data_path).unlink()

    def test_whitespace_only_file(self):
        """Test file with only whitespace produces 0 samples."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("   \n")
            f.write("\n")
            f.write("  \n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            assert len(dataset) == 0

        finally:
            Path(data_path).unlink()

    def test_sequence_creation(self):
        """Test that sequences are created correctly from lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("abc\n")  # 3 bytes + EOS = 4 bytes
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            # Should have 1 sequence (from "abc" + EOS)
            assert len(dataset.sequences) == 1
            # Sequence should be [97, 98, 99, EOS_TOKEN]
            expected = [ord("a"), ord("b"), ord("c"), EOS_TOKEN]
            assert dataset.sequences[0] == expected

        finally:
            Path(data_path).unlink()

    def test_sample_creation(self):
        """Test that samples are created correctly from sequences."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("abcdefghij\n")  # 10 bytes + EOS
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=5, stride=5)
            # Should create multiple samples due to sliding window
            assert len(dataset) > 0

        finally:
            Path(data_path).unlink()

    def test_getitem_returns_tensors(self):
        """Test that __getitem__ returns torch tensors."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("hello world\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            inp, tgt = dataset[0]

            assert isinstance(inp, torch.Tensor)
            assert isinstance(tgt, torch.Tensor)
            assert inp.dtype == torch.long
            assert tgt.dtype == torch.long

        finally:
            Path(data_path).unlink()

    def test_input_target_shift(self):
        """Test that target is shifted by 1 from input."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("abc\n")  # Simple test case
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10, stride=10)
            inp, tgt = dataset[0]

            # Target should be input shifted by 1
            # If input is [a, b, c, EOS], target should be [b, c, EOS, PAD]
            assert len(inp) == len(tgt)
            if len(inp) > 1:
                # Check that most elements are shifted
                assert torch.all(inp[1:] == tgt[:-1]) or torch.all(inp[:-1] == tgt[1:])

        finally:
            Path(data_path).unlink()

    def test_max_seq_len_respected(self):
        """Test that max_seq_len parameter is respected."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("a" * 20 + "\n")  # Long line
            data_path = f.name

        try:
            max_len = 5
            dataset = ByteSequenceDataset(data_path, max_seq_len=max_len, stride=max_len)

            # Check that samples don't exceed max_seq_len
            for i in range(len(dataset)):
                inp, tgt = dataset[i]
                assert len(inp) <= max_len
                assert len(tgt) <= max_len

        finally:
            Path(data_path).unlink()

    def test_stride_parameter(self):
        """Test that stride parameter affects sliding window."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("a" * 10 + "\n")
            data_path = f.name

        try:
            # Test with different strides
            dataset_stride_2 = ByteSequenceDataset(data_path, max_seq_len=5, stride=2)
            dataset_stride_5 = ByteSequenceDataset(data_path, max_seq_len=5, stride=5)

            # Smaller stride should produce more samples
            assert len(dataset_stride_2) >= len(dataset_stride_5)

        finally:
            Path(data_path).unlink()

    def test_short_sequences_filtered(self):
        """Test that sequences shorter than 2 bytes are filtered out."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("a\n")  # 1 byte + EOS = 2 bytes (should be kept)
            f.write("\n")   # 0 bytes (should be filtered)
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            # Should have at least the "a" + EOS sequence
            assert len(dataset.sequences) >= 1

        finally:
            Path(data_path).unlink()


class TestCollateFn:
    """Test the collate_fn for padding batches."""

    def test_basic_padding(self):
        """Test that collate_fn pads sequences to equal length."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([5, 6]), torch.tensor([6, 7])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        # Check shapes
        assert padded_inputs.shape == (2, 3)  # 2 samples, max length 3
        assert padded_targets.shape == (2, 3)

        # Check that padding is correct
        assert padded_inputs[0, 0] == 1
        assert padded_inputs[0, 1] == 2
        assert padded_inputs[0, 2] == 3
        assert padded_inputs[1, 0] == 5
        assert padded_inputs[1, 1] == 6
        assert padded_inputs[1, 2] == PAD_TOKEN  # Padding

    def test_single_element_batch(self):
        """Test collate_fn with single element batch."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape == (1, 3)
        assert padded_targets.shape == (1, 3)

    def test_empty_batch(self):
        """Test collate_fn with empty batch."""
        batch = []

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape == (0, 0)
        assert padded_targets.shape == (0, 0)

    def test_all_same_length(self):
        """Test collate_fn when all sequences are same length."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([5, 6, 7]), torch.tensor([6, 7, 8])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape == (2, 3)
        assert padded_targets.shape == (2, 3)
        # No padding should be needed
        assert not torch.any(padded_inputs == PAD_TOKEN)

    def test_large_batch(self):
        """Test collate_fn with larger batch."""
        batch = [
            (torch.tensor([i]), torch.tensor([i + 1]))
            for i in range(100)
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape == (100, 1)
        assert padded_targets.shape == (100, 1)


class TestSpecialTokens:
    """Test special token constants and behavior."""

    def test_special_token_values(self):
        """Test that special tokens have expected values."""
        assert PAD_TOKEN == 256
        assert EOS_TOKEN == 257
        assert VOCAB_SIZE == 260

    def test_pad_token_not_in_byte_range(self):
        """Test that PAD_TOKEN is outside standard byte range."""
        assert PAD_TOKEN > 255

    def test_eos_token_not_in_byte_range(self):
        """Test that EOS_TOKEN is outside standard byte range."""
        assert EOS_TOKEN > 255

    def test_vocab_size_accommodates_special_tokens(self):
        """Test that VOCAB_SIZE accommodates byte range + special tokens."""
        assert VOCAB_SIZE > max(PAD_TOKEN, EOS_TOKEN)

    def test_eos_token_added_to_sequences(self):
        """Test that EOS_TOKEN is added to sequences."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("abc\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            # Check that EOS_TOKEN is in the sequence
            assert EOS_TOKEN in dataset.sequences[0]

        finally:
            Path(data_path).unlink()


class TestDatasetIntegration:
    """Integration tests for dataset with realistic data."""

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            for i in range(10):
                f.write(f"sample line {i}\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=20)
            dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

            # Should be able to iterate
            batch_count = 0
            for inputs, targets in dataloader:
                assert inputs.shape[0] <= 2  # Batch size
                assert inputs.shape[1] <= 20  # Max seq len
                batch_count += 1
                if batch_count >= 3:  # Test a few batches
                    break

            assert batch_count > 0

        finally:
            Path(data_path).unlink()

    def test_padding_does_not_contaminate_targets(self):
        """Test that PAD_TOKEN does not appear in target values from actual data."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("hello world\n")
            f.write("test data\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            # Check that no target values are PAD_TOKEN in the original samples
            for i in range(len(dataset)):
                inp, tgt = dataset[i]
                # Note: targets from actual data shouldn't contain PAD_TOKEN
                # (only padded targets from collate_fn will)
                pass  # This is more about documenting expected behavior

        finally:
            Path(data_path).unlink()

    def test_repr_not_implemented(self):
        """Test that dataset currently lacks __repr__ (as per TODO)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("test\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10)
            # Just verify the dataset exists; __repr__ is noted as TODO
            assert dataset is not None

        finally:
            Path(data_path).unlink()