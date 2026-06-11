"""Meta BLT model loader using HuggingFace Transformers.

This module handles loading Meta's pre-trained BLT models for fine-tuning
on the phonetic AAC task.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import BltConfig, BltForCausalLM

# Load environment variables
load_dotenv()


class MetaBLTWrapper:
    """Wrapper for Meta's pre-trained BLT models from HuggingFace.

    This handles loading Meta's BLT-1B or BLT-7B models for fine-tuning.
    """

    def __init__(self, model_name: str = "facebook/blt-1b", use_auth_token: str | None = None):
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

        # Authenticate with HuggingFace if token provided
        if self.use_auth_token:
            login(token=self.use_auth_token)

        print(f"Loading Meta BLT model: {self.model_name}")

        # Load configuration first
        self.config = BltConfig.from_pretrained(
            self.model_name,
            token=self.use_auth_token
        )

        print("BLT Config loaded:")
        print(f"  Vocab size: {self.config.vocab_size}")
        print(f"  Max sequence length: {self.config.max_position_embeddings}")
        # Handle different config attributes between BLT versions
        if hasattr(self.config, 'num_hidden_layers'):
            print(f"  Number of layers: {self.config.num_hidden_layers}")
        if hasattr(self.config, 'num_attention_heads'):
            print(f"  Number of heads: {self.config.num_attention_heads}")
        if hasattr(self.config, 'hidden_size'):
            print(f"  Hidden size: {self.config.hidden_size}")

        # Load the full model
        self.model = BltForCausalLM.from_pretrained(
            self.model_name,
            token=self.use_auth_token
        )

        print("✅ Meta BLT model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        return self.model

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
