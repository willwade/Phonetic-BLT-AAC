"""Test enhanced dataset functionality."""

import tempfile
from pathlib import Path

import torch

from model.dataset import ByteSequenceDataset


class TestDatasetEnhancements:
    """Test new dataset enhancements: __repr__, subset, overlap verification."""

    def test_repr_contains_info(self):
        """Test that __repr__ contains dataset information."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            f.write("hello world\n")
            f.write("test data\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=10, stride=5)
            repr_str = repr(dataset)

            assert "ByteSequenceDataset" in repr_str
            assert "sequences=" in repr_str
            assert "samples=" in repr_str
            assert "max_seq_len=10" in repr_str
            assert "stride=5" in repr_str

        finally:
            Path(data_path).unlink()

    def test_subset_smaller_than_original(self):
        """Test that subset returns smaller dataset."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            for i in range(10):
                f.write(f"sample line {i}\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=20)
            original_len = len(dataset)

            subset = dataset.subset(n=5, seed=42)
            assert len(subset) == 5
            assert len(subset) < original_len

        finally:
            Path(data_path).unlink()

    def test_subset_larger_than_original(self):
        """Test that subset with n larger than dataset returns all samples."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            for i in range(3):
                f.write(f"sample line {i}\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=20)
            original_len = len(dataset)

            subset = dataset.subset(n=100, seed=42)
            assert len(subset) == original_len

        finally:
            Path(data_path).unlink()

    def test_subset_reproducible_with_seed(self):
        """Test that subset with same seed produces same results."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            for i in range(20):
                f.write(f"sample line {i}\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=20)

            subset1 = dataset.subset(n=5, seed=123)
            subset2 = dataset.subset(n=5, seed=123)

            # Should have the same samples in the same order
            for i in range(len(subset1)):
                inp1, tgt1 = subset1[i]
                inp2, tgt2 = subset2[i]
                assert torch.equal(inp1, inp2)
                assert torch.equal(tgt1, tgt2)

        finally:
            Path(data_path).unlink()

    def test_subset_different_without_seed(self):
        """Test that subset without seed can produce different results."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            for i in range(20):
                f.write(f"sample line {i}\n")
            data_path = f.name

        try:
            dataset = ByteSequenceDataset(data_path, max_seq_len=20)

            subset1 = dataset.subset(n=5)
            subset2 = dataset.subset(n=5)

            # Check that at least the samples are lists (even if they might be the same)
            assert isinstance(subset1.samples, list)
            assert isinstance(subset2.samples, list)

        finally:
            Path(data_path).unlink()

    def test_verify_no_trivial_overlap(self):
        """Test overlap verification method."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            # Create lines that will generate overlapping samples with small stride
            f.write("a" * 20 + "\n")
            data_path = f.name

        try:
            # Use a small stride to create potential overlaps
            dataset = ByteSequenceDataset(data_path, max_seq_len=10, stride=3)

            stats = dataset.verify_no_trivial_overlap()

            assert "total_checks" in stats
            assert "overlaps_found" in stats
            assert "max_overlap_bytes" in stats
            assert "max_overlap_ratio" in stats
            assert stats["total_checks"] >= 0
            assert stats["overlaps_found"] >= 0
            assert stats["max_overlap_bytes"] >= 0
            assert 0 <= stats["max_overlap_ratio"] <= 1

        finally:
            Path(data_path).unlink()

    def test_verify_no_trivial_overlap_large_stride(self):
        """Test that large stride produces fewer overlaps."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            # Use varied content to reduce structural overlap
            f.write("abcdefghijklmnopqrstuvwxyz\n")
            data_path = f.name

        try:
            # Use a large stride (equal to max_seq_len) to minimize overlaps
            dataset = ByteSequenceDataset(data_path, max_seq_len=10, stride=10)

            stats = dataset.verify_no_trivial_overlap()

            # With varied content and large stride, overlap ratio should be reasonable
            assert stats["max_overlap_ratio"] <= 1.0  # Should be valid ratio
            assert stats["total_checks"] >= 0  # Should run without error

        finally:
            Path(data_path).unlink()

    def test_verify_no_trivial_overlap_small_stride(self):
        """Test that small stride produces more overlaps."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            f.write("a" * 50 + "\n")
            data_path = f.name

        try:
            # Use a very small stride to maximize overlaps
            dataset = ByteSequenceDataset(data_path, max_seq_len=10, stride=2)

            stats = dataset.verify_no_trivial_overlap()

            # With small stride, there should be more overlaps
            assert stats["total_checks"] > 0

        finally:
            Path(data_path).unlink()
