"""Meta BLT model loader using HuggingFace Transformers.

This module handles loading Meta's pre-trained BLT models and fine-tuning
them for the phonetic AAC task.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import BltForCausalLM, BltConfig

# Load environment variables
load_dotenv()


class MetaBLTWrapper:
    """Wrapper for Meta's pre-trained BLT models from HuggingFace.

    This handles loading Meta's BLT-1B or BLT-7B models and adapting them
    for our phonetic AAC task.
    """

    def __init__(self, model_name: str = "facebook/blt-1b", use_auth_token: Optional[str] = None):
        """Initialize Meta BLT model wrapper.

        Args:
            model_name: HuggingFace model ID (facebook/blt-1b or facebook/blt-7b)
            use_auth_token: Optional HuggingFace authentication token
        """
        self.model_name = model_name
        self.use_auth_token = use_auth_token or os.getenv("HF_TOKEN")
        self.model = None
        self.config = None

    def load_model(self) -> BltForCausalLM:
        """Load Meta's pre-trained BLT model.

        Returns:
            Loaded BLT model

        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            return self.model

        try:
            # Authenticate with HuggingFace if token provided
            if self.use_auth_token:
                login(token=self.use_auth_token)

            print(f"Loading Meta BLT model: {self.model_name}")

            # Load configuration first
            self.config = BltConfig.from_pretrained(
                self.model_name,
                token=self.use_auth_token
            )

            print(f"BLT Config loaded:")
            print(f"  Vocab size: {self.config.vocab_size}")
            print(f"  Max sequence length: {self.config.max_position_embeddings}")
            print(f"  Number of layers: {self.config.num_hidden_layers}")
            print(f"  Number of heads: {self.config.num_attention_heads}")
            print(f"  Hidden size: {self.config.hidden_size}")

            # Load the full model
            self.model = BltForCausalLM.from_pretrained(
                self.model_name,
                token=self.use_auth_token
            )

            print(f"✅ Meta BLT model loaded successfully!")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return self.model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Meta BLT model '{self.model_name}': {e}\n"
                f"Note: Meta's BLT models require access approval.\n"
                f"Request access at: https://huggingface.co/{self.model_name}\n"
                f"If you have access, ensure HF_TOKEN is set in .env file."
            )

    def get_model(self) -> BltForCausalLM:
        """Get or load the BLT model."""
        if self.model is None:
            self.load_model()
        return self.model

    def get_config(self) -> BltConfig:
        """Get or load the BLT config."""
        if self.config is None:
            self.load_model()
        return self.config


class BLTModelFactory:
    """Factory for creating BLT models with fallback strategies.

    This provides a unified interface for getting a BLT model, with fallback
    options when Meta's pre-trained models are unavailable.
    """

    def __init__(self):
        self.meta_wrapper = None
        self.using_meta = False

    def get_blt_model(
        self,
        prefer_meta: bool = True,
        model_name: str = "facebook/blt-1b",
        fallback_config: Optional[dict] = None
    ):
        """Get a BLT model with fallback strategies.

        Args:
            prefer_meta: Whether to try Meta's pre-trained models first
            model_name: Which Meta BLT model to use
            fallback_config: Config for fallback custom BLT implementation

        Returns:
            BLT model (either Meta's or custom implementation)
        """
        if prefer_meta:
            try:
                print("Attempting to load Meta's pre-trained BLT...")
                self.meta_wrapper = MetaBLTWrapper(model_name)
                model = self.meta_wrapper.load_model()
                self.using_meta = True
                return model
            except Exception as e:
                print(f"Could not load Meta BLT: {e}")
                print("Falling back to custom BLT implementation...")
                self.using_meta = False

        # Fallback to custom implementation
        if fallback_config:
            from model.blt_model import LightweightByteLatentTransformer
            print("Using custom BLT implementation")
            return LightweightByteLatentTransformer(fallback_config)
        else:
            raise ValueError("No BLT model available and no fallback config provided")

    def is_using_meta(self) -> bool:
        """Check if currently using Meta's pre-trained BLT."""
        return self.using_meta

    def get_model_info(self) -> dict:
        """Get information about the current BLT model."""
        if self.using_meta and self.meta_wrapper:
            config = self.meta_wrapper.get_config()
            return {
                "model_type": "Meta BLT (Pre-trained)",
                "model_name": self.meta_wrapper.model_name,
                "vocab_size": config.vocab_size,
                "num_layers": config.num_hidden_layers,
                "num_heads": config.num_attention_heads,
                "hidden_size": config.hidden_size,
                "max_sequence_length": config.max_position_embeddings,
            }
        else:
            return {"model_type": "Custom BLT implementation"}


def setup_blt_model(config: dict, prefer_meta: bool = True):
    """Setup BLT model with intelligent fallback.

    Args:
        config: Training configuration
        prefer_meta: Whether to prefer Meta's pre-trained models

    Returns:
        Tuple of (model, model_info)
    """
    factory = BLTModelFactory()

    try:
        model = factory.get_blt_model(
            prefer_meta=prefer_meta,
            model_name="facebook/blt-1b",
            fallback_config=config
        )
        model_info = factory.get_model_info()
        return model, model_info
    except Exception as e:
        raise RuntimeError(f"Failed to setup BLT model: {e}")