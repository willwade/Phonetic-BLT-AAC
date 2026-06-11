"""BLT model loader using HuggingFace Transformers.

Loads the community-converted BLT weights (itazap/blt-1b-hf) which are in
transformers-native format. No need for the bytelatent package or xformers.

The model is actually 4.63B parameters (local encoder + global transformer +
local decoder + entropy patcher), despite the "blt-1b" name.

Tested: model loads and forward pass works via test_blt_load.py.
"""

from transformers import AutoTokenizer, BltConfig, BltForCausalLM

DEFAULT_MODEL = "itazap/blt-1b-hf"


class MetaBLTWrapper:
    """Wrapper for loading BLT via HuggingFace Transformers.

    Uses the community-converted weights that are in transformers format.
    No HuggingFace token or access approval needed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model: BltForCausalLM | None = None
        self.config: BltConfig | None = None
        self.tokenizer: AutoTokenizer | None = None

    def load_model(self) -> BltForCausalLM:
        """Load the BLT model from HuggingFace.

        Returns:
            Loaded BltForCausalLM model.
        """
        if self.model is not None:
            return self.model

        print(f"Loading BLT model: {self.model_name}")
        self.config = BltConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BltForCausalLM.from_pretrained(self.model_name)

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {param_count:,} parameters ({param_count / 1e9:.2f}B)")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Max position embeddings: {self.config.max_position_embeddings}")

        return self.model

    def get_model(self) -> BltForCausalLM:
        if self.model is None:
            self.load_model()
        return self.model

    def get_config(self) -> BltConfig:
        if self.config is None:
            self.load_model()
        return self.config

    def get_tokenizer(self) -> AutoTokenizer:
        if self.tokenizer is None:
            self.load_model()
        return self.tokenizer
