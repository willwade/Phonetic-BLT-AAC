"""Test that all required dependencies can be imported without error.

This validates that the environment is correctly set up with all required packages.
"""



def test_torch_import():
    """Test that PyTorch can be imported."""
    import torch

    assert torch.__version__ is not None


def test_datasets_import():
    """Test that Hugging Face datasets can be imported."""
    import datasets

    assert datasets.__version__ is not None


def test_epitran_import():
    """Test that Epitran G2P library can be imported."""
    import epitran

    assert epitran.__version__ is not None


def test_onnx_import():
    """Test that ONNX can be imported."""
    import onnx

    assert onnx.__version__ is not None


def test_onnxruntime_import():
    """Test that ONNX Runtime can be imported."""
    import onnxruntime

    assert onnxruntime.__version__ is not None


def test_huggingface_hub_import():
    """Test that Hugging Face Hub can be imported."""
    import huggingface_hub

    assert huggingface_hub.__version__ is not None


def test_all_core_imports():
    """Test that all core dependencies are available in the same environment."""
    import datasets
    import epitran
    import onnx
    import onnxruntime
    import torch

    # Verify basic functionality
    assert hasattr(torch, "nn")
    assert hasattr(datasets, "load_dataset")
    assert hasattr(epitran, "Epitran")
    assert hasattr(onnx, "load")
    assert hasattr(onnxruntime, "InferenceSession")
