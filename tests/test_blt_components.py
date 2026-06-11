"""Test BLT architecture components."""

import torch
import pytest
from model.blt_patcher import SimpleEntropyPatcher
from model.blt_decoder import SimpleLocalDecoder
from model.blt_global_transformer import GlobalLatentTransformer
from model.blt_model import ByteLatentTransformer, LightweightByteLatentTransformer, create_blt_model


class TestEntropyPatcher:
    """Test entropy-based patching component."""

    def test_basic_patching(self):
        """Test basic patch creation."""
        patcher = SimpleEntropyPatcher()

        # Create simple byte sequence
        byte_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        patches, boundaries = patcher(byte_ids)

        assert len(patches) > 0
        assert len(boundaries) > 0
        assert len(patches) == len(boundaries)

    def test_empty_sequence(self):
        """Test patching with empty sequence."""
        patcher = SimpleEntropyPatcher()
        byte_ids = torch.tensor([[]]).view(1, 0)

        patches, boundaries = patcher(byte_ids)
        assert len(patches) == 0

    def test_patch_boundaries(self):
        """Test that patch boundaries are valid."""
        patcher = SimpleEntropyPatcher()
        byte_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        patches, boundaries = patcher(byte_ids)

        for start, end in boundaries:
            assert start < end
            assert start >= 0
            assert end <= byte_ids.shape[1]

    def test_patch_stats(self):
        """Test patch statistics computation."""
        patcher = SimpleEntropyPatcher()
        byte_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        stats = patcher.get_patch_stats(byte_ids)

        assert "num_patches" in stats
        assert "avg_patch_size" in stats
        assert stats["num_patches"] >= 0


class TestLocalDecoder:
    """Test local decoder component."""

    def test_basic_decode(self):
        """Test basic decoding."""
        decoder = SimpleLocalDecoder(
            patch_dim=128,
            hidden_dim=64,
            vocab_size=260,
        )

        patch_repr = torch.randn(2, 4, 128)  # (batch, patches, patch_dim)
        logits = decoder(patch_repr)

        assert logits.shape == (2, 4, 260)

    def test_different_dimensions(self):
        """Test decoder with different dimensions."""
        decoder = SimpleLocalDecoder(
            patch_dim=256,
            hidden_dim=128,
            vocab_size=260,
        )

        patch_repr = torch.randn(1, 3, 256)
        logits = decoder(patch_repr)

        assert logits.shape == (1, 3, 260)

    def test_single_patch(self):
        """Test decoding single patch."""
        decoder = SimpleLocalDecoder(patch_dim=64, hidden_dim=32, vocab_size=260)
        patch_repr = torch.randn(1, 1, 64)

        logits = decoder(patch_repr)
        assert logits.shape == (1, 1, 260)


class TestGlobalTransformer:
    """Test global transformer component."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        transformer = GlobalLatentTransformer(
            patch_dim=128,
            num_layers=2,
            num_heads=4,
            hidden_dim=256,
        )

        patch_repr = torch.randn(2, 8, 128)  # (batch, patches, patch_dim)
        processed = transformer(patch_repr)

        assert processed.shape == (2, 8, 128)

    def test_different_configurations(self):
        """Test with different configurations."""
        transformer = GlobalLatentTransformer(
            patch_dim=64,
            num_layers=1,
            num_heads=2,
            hidden_dim=128,
        )

        patch_repr = torch.randn(1, 4, 64)
        processed = transformer(patch_repr)

        assert processed.shape == (1, 4, 64)


class TestCompleteBLTModel:
    """Test complete BLT model integration."""

    def test_blt_model_creation(self):
        """Test BLT model creation."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 64,
                "patcher": {
                    "entropy_threshold": 1.34,
                },
                "global_transformer": {
                    "num_layers": 2,
                    "num_heads": 2,
                    "hidden_dim": 128,
                    "dropout": 0.1,
                },
                "decoder": {
                    "num_layers": 1,
                    "hidden_dim": 64,
                },
            }
        }

        model = ByteLatentTransformer(config)
        assert model is not None

    def test_lightweight_blt_forward(self):
        """Test lightweight BLT forward pass."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 32,
                "patcher": {"entropy_threshold": 1.34},
                "global_transformer": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "hidden_dim": 64,
                },
                "decoder": {"num_layers": 1, "hidden_dim": 32},
            }
        }

        model = LightweightByteLatentTransformer(config)
        x = torch.randint(0, 260, (2, 16))  # (batch, seq_len)

        logits = model(x)
        assert logits.shape == (2, 16, 260)

    def test_blt_model_info(self):
        """Test model information retrieval."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 32,
                "patcher": {"entropy_threshold": 1.34},
                "global_transformer": {
                    "num_layers": 2,
                    "num_heads": 2,
                    "hidden_dim": 128,
                },
                "decoder": {"num_layers": 1, "hidden_dim": 64},
            }
        }

        model = LightweightByteLatentTransformer(config)
        info = model.get_model_info()

        assert "total_parameters" in info
        assert info["total_parameters"] > 0

    def test_factory_function(self):
        """Test BLT model factory function."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 32,
                "patcher": {"entropy_threshold": 1.34},
                "global_transformer": {
                    "num_layers": 2,
                    "num_heads": 2,
                    "hidden_dim": 128,
                },
                "decoder": {"num_layers": 1, "hidden_dim": 64},
            }
        }

        # Test lightweight
        model = create_blt_model(config, use_lightweight=True)
        assert isinstance(model, LightweightByteLatentTransformer)

    def test_blt_forward_pass(self):
        """Test complete BLT forward pass."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 32,
                "patcher": {"entropy_threshold": 1.34},
                "global_transformer": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "hidden_dim": 64,
                },
                "decoder": {"num_layers": 1, "hidden_dim": 32},
            }
        }

        model = LightweightByteLatentTransformer(config)
        x = torch.randint(0, 260, (1, 16))

        with torch.no_grad():
            logits = model(x)

        assert logits.shape[0] == 1  # Batch size
        assert logits.shape[2] == 260  # Vocabulary size


class TestBLTIntegration:
    """Test BLT model integration with training infrastructure."""

    def test_loss_computation(self):
        """Test that BLT model can compute loss."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 32,
                "patcher": {"entropy_threshold": 1.34},
                "global_transformer": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "hidden_dim": 64,
                },
                "decoder": {"num_layers": 1, "hidden_dim": 32},
            }
        }

        model = LightweightByteLatentTransformer(config)
        criterion = torch.nn.CrossEntropyLoss()

        inputs = torch.randint(0, 260, (2, 16))
        targets = torch.randint(0, 260, (2, 16))

        logits = model(inputs)
        loss = criterion(logits.view(-1, 260), targets.view(-1))

        assert loss.item() > 0

    def test_gradient_flow(self):
        """Test that gradients flow through BLT model."""
        config = {
            "model": {
                "vocab_size": 260,
                "max_sequence_length": 32,
                "patcher": {"entropy_threshold": 1.34},
                "global_transformer": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "hidden_dim": 64,
                },
                "decoder": {"num_layers": 1, "hidden_dim": 32},
            }
        }

        model = LightweightByteLatentTransformer(config)
        inputs = torch.randint(0, 260, (1, 8))
        targets = torch.randint(0, 260, (1, 8))

        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 260),
            targets.view(-1)
        )

        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None