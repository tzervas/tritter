"""Configuration for Tritter multimodal model.

Defines all model hyperparameters, optimization flags, and hardware constraints.

Why: Centralized configuration as a dataclass provides type safety, validation, and single
source of truth for model architecture. This prevents mismatches between components (e.g.,
embedding dim != hidden_size) and enables easy experimentation by changing config rather
than modifying code throughout the codebase. The __post_init__ validation catches errors
at config creation time rather than runtime during training, saving compute time.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TritterConfig:
    """Configuration class for Tritter model architecture and training.

    Why: This config enables a 3-7B parameter multimodal model with 128K context on RTX 5080
    16GB VRAM through aggressive memory optimization. Key design decisions:

    - BitNet quantization (1.58 bits): Reduces 7B model weights from 14GB to 1.4GB (10x savings)
    - INT4 KV-cache: Fits 128K context in remaining VRAM (~8-12GB for KV-cache)
    - Sliding window attention: Bounds memory growth while maintaining long-range dependencies
    - Early fusion multimodality: Shared embedding space enables any-to-any generation
    - 65K vocab: Accommodates text BPE (~50K) + image VQVAE codes (~8K) + audio tokens

    The 3B/7B sizing follows proven architectures (Llama, Mistral) while BitNet quantization
    makes 7B feasible on consumer hardware. Context window of 128K enables repository-level
    code understanding and long-form multimodal documents.

    Attributes:
        model_size: Model size variant ('3B' or '7B') - auto-configures layer counts
        hidden_size: Dimension of hidden representations (2048 for 3B, 4096 for 7B)
        num_layers: Number of transformer layers (24 for 3B, 32 for 7B)
        num_heads: Number of attention heads (must divide hidden_size evenly)
        intermediate_size: FFN intermediate dimension (typically 4x hidden_size)
        max_position_embeddings: Maximum sequence length / context window (131072 = 128K)
        vocab_size: Unified vocabulary size for all modalities (default 65536 = 2^16)
        use_bitnet: Enable BitNet 1.58-bit ternary quantization {-1, 0, +1}
        use_flash_attention: Enable FlashAttention2 (reduces O(N²) to O(N) memory)
        sliding_window_size: Window size for sliding window attention (None = disabled, not yet implemented)
        int4_kv_cache: Use INT4 quantization for key-value cache
        modalities: List of enabled modalities from {text, code, image, audio}
        use_early_fusion: Enable Chameleon-style early fusion vs late fusion
        unified_embedding: Use shared embedding layer for all modalities
        target_device: Hardware target (cuda for RTX 5080)
        max_memory_gb: VRAM budget (16GB for RTX 5080 GDDR7)
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
    use_streaming_llm: bool = True

    # Attention architecture configuration
    attention_mode: str = "causal"
    """Attention mode for transformer computation.

    Why: Different attention patterns enable different use cases. Causal (autoregressive) is
    standard for language modeling and decoder-only pretraining. Bidirectional enables
    semantic embeddings. Prefix-LM (bidirectional prefix + causal suffix) enables instruction
    tuning with context. Embedding mode (from roadmap) enables Coconut-style continuous
    latent reasoning.

    Options:
        - "causal": Standard autoregressive decoder-only (default, for pretraining/generation)
        - "bidirectional": Full attention all tokens attend to all tokens (for embeddings)
        - "prefix_lm": Bidirectional prefix (instructions) + causal suffix (response)
        - "embedding": Coconut-style continuous latent reasoning in embedding space
    """

    use_sliding_window: bool = False
    """Enable sliding window attention with bounded KV-cache.

    Why: Sliding window attention bounds the context each token attends to (e.g., 4K tokens),
    reducing KV-cache from O(N²) to O(N*W) where W=window_size. This enables 128K context
    windows within 16GB VRAM while maintaining local dependencies. Global attention can be
    implemented via DistilBERT-style attention sinks (see num_sink_tokens).

    Note: Not yet implemented, placeholder for future FlexAttention patterns.
    """

    sliding_window_size: int | None = None
    """Size of attention window for sliding window attention.

    Why: Controls the recency window for attention computation. Typical values: 2K-4K tokens
    balance compute efficiency with dependency modeling. Smaller windows (2K) are faster but
    may lose long-range dependencies. Larger windows (8K) preserve more context but increase
    memory. Recommended for 128K context: 4K window (good compromise).

    Default: None (disabled)

    Note: Configured but not yet implemented in attention layers. Must be > 0 if use_sliding_window=True.
    """

    use_attention_sinks: bool = False
    """Enable attention sinks for StreamingLLM-style inference.

    Why: StreamingLLM maintains num_sink_tokens as "attention sinks" - special tokens that all
    future tokens attend to regardless of position. This preserves important early context
    (e.g., system prompt) even when KV-cache is evicted. Enables infinite-length generation
    bounded by window size instead of fixed context length.

    Note: Not yet implemented, placeholder for StreamingLLM patterns.
    """

    num_sink_tokens: int = 4
    """Number of initial tokens to retain as attention sinks in StreamingLLM.

    Why: Attention sinks typically include the BOS token and first few prompt tokens.
    4 tokens balances information retention (usually enough for system prompt start) with
    memory efficiency. Original StreamingLLM paper tested values 2-8 and found 4-8 optimal.
    """

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
        """Validate configuration and set derived attributes.

        Why: Automatic configuration based on model_size prevents manual errors and ensures
        consistent scaling. The 7B config doubles hidden_size (2048→4096) and increases layers
        (24→32) following standard scaling laws. Validation catches incompatible settings early:

        - hidden_size must divide evenly by num_heads to avoid fractional head dimensions
        - Only supported modalities are allowed to prevent integration issues

        These assertions fail fast at config creation rather than during expensive training.
        """
        # Auto-configure 7B variant
        # Why: 7B uses wider layers (4096 hidden) and deeper stack (32 layers) vs 3B.
        # This follows proven scaling from Llama-3/Mistral architectures.
        if self.model_size == "7B":
            # Only override if still at default 3B values (preserve user-specified values)
            if self.hidden_size == 2048:
                self.hidden_size = 4096
            if self.num_layers == 24:
                self.num_layers = 32
            if self.num_heads == 16:
                self.num_heads = 32
            # Only update intermediate_size if it's still at the 4x hidden_size ratio
            if self.intermediate_size == 8192:  # 4x the 3B hidden_size
                self.intermediate_size = 16384  # 4x the 7B hidden_size

        # Ensure head dimension is valid
        # Why: Multi-head attention splits hidden_size across heads. Non-divisible configs
        # would require padding or truncation, breaking the math. Standard head_dim is 64-128.
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )

        # Validate modalities
        # Why: Only these four modalities have tokenizers implemented. Invalid modalities
        # would fail at runtime when trying to encode. Fail fast here instead.
        valid_modalities = {"text", "code", "image", "audio"}
        for modality in self.modalities:
            assert modality in valid_modalities, (
                f"Invalid modality: {modality}. Must be one of {valid_modalities}"
            )

        # Validate attention configuration
        # Why: attention_mode must be one of the supported patterns to avoid architecture
        # mismatches. sliding_window_size must be positive if enabled to prevent zero-size
        # windows that would make attention degenerate.
        valid_attention_modes = {"causal", "bidirectional", "prefix_lm", "embedding"}
        assert self.attention_mode in valid_attention_modes, (
            f"Invalid attention_mode: {self.attention_mode!r}. Must be one of {valid_attention_modes}"
        )

        if self.use_sliding_window:
            assert self.sliding_window_size is not None and self.sliding_window_size > 0, (
                f"sliding_window_size must be > 0 when use_sliding_window=True, got {self.sliding_window_size}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head.

        Why: Computed property ensures head_dim stays synchronized with hidden_size and num_heads.
        Multi-head attention splits the hidden dimension across heads (e.g., 2048 / 16 = 128 per head).
        This enables parallel attention computation while keeping each head's context manageable.
        Standard values are 64-128; larger values increase expressiveness but slow attention.

        Returns:
            int: Dimension per attention head (hidden_size // num_heads)
        """
        return self.hidden_size // self.num_heads
