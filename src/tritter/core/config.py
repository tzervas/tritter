"""Configuration for Tritter multimodal model."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TritterConfig:
    """Configuration class for Tritter model architecture and training.

    Attributes:
        model_size: Model size variant ('3B' or '7B')
        hidden_size: Dimension of hidden representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: Dimension of FFN intermediate layer
        max_position_embeddings: Maximum sequence length (context window)
        vocab_size: Size of unified vocabulary
        use_bitnet: Enable BitNet 1.58-bit ternary quantization
        use_flash_attention: Enable FlashAttention2 optimization
        sliding_window_size: Size of sliding window for long context
        int4_kv_cache: Use INT4 quantization for KV cache
        modalities: Supported modalities for multimodal fusion
    """

    # Model architecture
    model_size: Literal["3B", "7B"] = "3B"
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 131072  # 128K context
    vocab_size: int = 65536

    # BitNet quantization
    use_bitnet: bool = True
    bitnet_precision: float = 1.58  # Ternary weights: {-1, 0, 1}

    # Attention optimizations
    use_flash_attention: bool = True
    sliding_window_size: int = 4096
    use_streaming_llm: bool = True

    # Memory optimizations
    int4_kv_cache: bool = True
    gradient_checkpointing: bool = True

    # Multimodal configuration
    modalities: list[str] = field(default_factory=lambda: ["text", "code", "image", "audio"])
    use_early_fusion: bool = True
    unified_embedding: bool = True

    # Hardware targeting
    target_device: str = "cuda"  # RTX 5080 optimized
    max_memory_gb: int = 16  # GDDR7

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Advanced features
    use_vsa: bool = False  # Vector Symbolic Architectures
    use_hrr: bool = False  # Holographic Reduced Representations

    def __post_init__(self) -> None:
        """Validate configuration and set derived attributes."""
        if self.model_size == "7B":
            self.hidden_size = 4096
            self.num_layers = 32
            self.num_heads = 32
            self.intermediate_size = 16384

        # Ensure head dimension is valid
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )

        # Validate modalities
        valid_modalities = {"text", "code", "image", "audio"}
        for modality in self.modalities:
            assert modality in valid_modalities, (
                f"Invalid modality: {modality}. Must be one of {valid_modalities}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_heads
