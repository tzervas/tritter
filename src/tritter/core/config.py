"""Configuration for Tritter multimodal model.

Defines all model hyperparameters, optimization flags, and hardware constraints.

Why: Centralized configuration as a dataclass provides type safety, validation, and single
source of truth for model architecture. This prevents mismatches between components (e.g.,
embedding dim != hidden_size) and enables easy experimentation by changing config rather
than modifying code throughout the codebase. The __post_init__ validation catches errors
at config creation time rather than runtime during training, saving compute time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tritter.core.model_specs import HardwareRecommendation, MemoryEstimate, ModelSpec


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

    Usage Modes:

    **Developer mode (simple)** - Pick a model size and everything auto-configures:
    ```python
    config = TritterConfig(model_size="7B")  # All params set automatically
    ```

    **Researcher mode (advanced)** - Full manual control over architecture:
    ```python
    config = TritterConfig.for_research(
        hidden_size=4096,
        num_layers=40,
        num_heads=32,
        intermediate_size=14336,
    )
    ```

    See TritterConfig.for_research() for full documentation on custom architectures.

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
    model_size: Literal[
        "test", "125M", "350M", "1B", "3B", "7B", "10B", "13B", "30B", "33B", "40B", "65B", "70B"
    ] = "3B"
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int | None = None  # None = MHA, set for GQA
    intermediate_size: int = 8192
    max_position_embeddings: int = 131072  # 128K context
    rope_theta: float = 500000.0  # RoPE base frequency (extended for 128K context)
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
    implemented via attention sinks (see num_sink_tokens).

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

    # Progressive Layer Loading
    use_layer_streaming: bool = False
    """Enable progressive layer loading for large model inference.

    Why: Allows running models larger than GPU VRAM by streaming layer
    groups through memory. Set to True for models that don't fit in VRAM.
    """

    layer_group_size: int = 4
    """Number of layers to load per group.

    Why: Larger groups reduce transfer overhead but require more VRAM.
    Optimal value depends on model size and available memory.

    Constraints:
        - Must be > 0
        - Should divide num_layers evenly (or remainder handled)
        - Typical range: 2-8 layers
    """

    gpu_memory_budget_gb: float = 14.0
    """Maximum GPU memory to use for inference (in GB).

    Why: Reserves headroom for CUDA overhead and unexpected allocations.
    Typically set to 85-90% of total VRAM.

    Constraints:
        - Must be > 0
        - Should be less than physical VRAM
    """

    use_pinned_memory: bool = True
    """Use CUDA pinned memory for faster host-to-device transfers.

    Why: Pinned memory enables async DMA transfers at full PCIe bandwidth.
    Disable if system RAM is limited.
    """

    prefetch_next_group: bool = True
    """Prefetch next layer group while processing current group.

    Why: Overlaps transfer with computation to hide latency.
    Requires double the layer group memory but significantly improves throughput.
    """

    # Multimodal configuration
    modalities: list[str] = field(default_factory=lambda: ["text", "code", "image", "audio"])
    use_early_fusion: bool = True
    unified_embedding: bool = True

    # Hardware targeting
    target_device: str = "cuda"  # RTX 5080 optimized
    max_memory_gb: int = 16  # GDDR7

    # Acceleration profiles
    accel_profile: str = "auto"
    """Acceleration profile selector.

    Why: Enables swapping between baseline and accelerated backends (e.g.,
    Rust bindings) without changing core model code.
    """

    accel_feature_flags: list[str] = field(default_factory=list)
    """Optional feature flags for accelerated backends.

    Why: Fine-grained switches help isolate kernel experiments or backends
    while keeping the profile name deterministic.
    """

    profile_name_override: str | None = None
    """Optional override for auto-generated profile names.

    Why: Allows custom experiment labels while retaining metadata logs.
    """

    profile_tag_overrides: dict[str, str | float] = field(default_factory=dict)
    """Optional overrides for profile naming tags.

    Why: Users can override specific tags (e.g., training mode) without
    redefining the full naming logic.
    """

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
        consistent scaling. Uses centralized model specs for all sizes (1B-70B) following
        standard scaling laws from Llama/Mistral architectures. Validation catches
        incompatible settings early:

        - hidden_size must divide evenly by num_heads to avoid fractional head dimensions
        - Only supported modalities are allowed to prevent integration issues

        These assertions fail fast at config creation rather than during expensive training.
        """
        # Auto-configure from model specs if not using explicit values
        # Why: Centralized specs ensure consistent architecture across all sizes (1B-70B).
        # We only override if values are at 3B defaults (preserving user-specified values).
        self._apply_model_spec()

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

        if self.use_attention_sinks:
            # Why: Attention sinks require a positive number of sink tokens. Zero or negative
            # values would make the StreamingLLM attention pattern degenerate or invalid.
            assert self.num_sink_tokens is not None and self.num_sink_tokens > 0, (
                f"num_sink_tokens must be > 0 when use_attention_sinks=True, got {self.num_sink_tokens}"
            )

        # Validate layer streaming configuration
        if self.use_layer_streaming:
            assert self.layer_group_size > 0, (
                f"layer_group_size must be > 0, got {self.layer_group_size}"
            )
            assert self.gpu_memory_budget_gb > 0, (
                f"gpu_memory_budget_gb must be > 0, got {self.gpu_memory_budget_gb}"
            )

        # Validate minimum vocab_size for byte-level encoding
        # Why: The _encode_text method in MultiModalTokenizer uses offset 8 for special tokens
        # (PAD, BOS, EOS, UNK, etc.) plus 256 byte values (0x00-0xFF). This minimum ensures
        # every byte can be represented without overflow, which is critical for the
        # embedding-prediction paradigm where continuous embeddings must map to discrete tokens.
        min_vocab_size = 264  # 8 special tokens + 256 byte values
        assert self.vocab_size >= min_vocab_size, (
            f"vocab_size ({self.vocab_size}) must be >= {min_vocab_size} "
            f"(8 special tokens + 256 byte values for byte-level encoding)"
        )

    def _apply_model_spec(self) -> None:
        """Apply model specification based on model_size.

        Why: Centralizes architecture configurations in model_specs.py. We only override
        values that are at their 3B defaults, preserving any user-specified overrides.

        Important: If hidden_size is customized, we don't apply spec's num_heads because
        they may be incompatible. Users who customize hidden_size should also specify num_heads.
        """
        # Import here to avoid circular dependency
        from tritter.core.model_specs import get_model_spec

        # Skip if already at non-default values (user specified)
        # Default 3B values that indicate auto-configuration is needed
        defaults = {
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 8192,
        }

        # Check if we're using 3B or need to apply a different spec
        if self.model_size == "3B":
            # Apply 3B spec values (may differ from hardcoded defaults)
            spec = get_model_spec("3B")
        else:
            spec = get_model_spec(self.model_size)

        # Track whether hidden_size was customized by user
        hidden_size_customized = self.hidden_size != defaults["hidden_size"]

        # Only override if at default 3B values
        if not hidden_size_customized:
            self.hidden_size = spec.hidden_size

        if self.num_layers == defaults["num_layers"]:
            self.num_layers = spec.num_layers

        # Only apply spec's num_heads if hidden_size wasn't customized
        # Why: Customized hidden_size may not be compatible with spec's num_heads
        if self.num_heads == defaults["num_heads"] and not hidden_size_customized:
            self.num_heads = spec.num_heads

        # Check intermediate_size default ratio
        is_intermediate_default = self.intermediate_size == defaults["intermediate_size"]
        if is_intermediate_default:
            self.intermediate_size = spec.intermediate_size

        # Apply GQA configuration if spec uses it and user didn't override
        # Only apply if hidden_size wasn't customized (GQA head counts are tied to spec)
        if (
            self.num_kv_heads is None
            and spec.num_kv_heads is not None
            and not hidden_size_customized
        ):
            self.num_kv_heads = spec.num_kv_heads

        # Apply vocab_size from spec if different and user didn't override
        # Why: Some model sizes (like "test") use smaller vocab for efficiency
        default_vocab_size = 65536
        if self.vocab_size == default_vocab_size and spec.vocab_size != default_vocab_size:
            self.vocab_size = spec.vocab_size

        # Apply max_position_embeddings from spec if different
        default_max_pos = 131072
        if (
            self.max_position_embeddings == default_max_pos
            and spec.max_position_embeddings != default_max_pos
        ):
            self.max_position_embeddings = spec.max_position_embeddings

        # Auto-enable layer streaming for large models if not explicitly set
        # Why: Models 30B+ typically won't fit on a single 16GB GPU without streaming
        large_model_sizes = {"30B", "33B", "40B", "65B", "70B"}
        if self.model_size in large_model_sizes and self.max_memory_gb <= 24:
            if not self.use_layer_streaming:
                # Note: Don't auto-enable, just set reasonable defaults if enabled
                pass

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

    @property
    def effective_num_kv_heads(self) -> int:
        """Number of key-value heads (equals num_heads for MHA).

        Why: Grouped Query Attention (GQA) uses fewer KV heads than query heads
        to reduce KV-cache memory. This property provides the effective count.
        """
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @property
    def uses_gqa(self) -> bool:
        """Whether this config uses Grouped Query Attention.

        Why: GQA reduces memory for KV-cache by sharing KV heads across query heads.
        Larger models (7B+) typically use GQA for memory efficiency.
        """
        return self.num_kv_heads is not None and self.num_kv_heads < self.num_heads

    def get_model_spec(self) -> ModelSpec:
        """Get the model specification for this config's size.

        Returns:
            ModelSpec with architecture details
        """
        from tritter.core.model_specs import get_model_spec

        return get_model_spec(self.model_size)

    def estimate_memory(self, batch_size: int = 1) -> MemoryEstimate:
        """Estimate memory requirements for this config.

        Args:
            batch_size: Batch size for KV-cache estimation

        Returns:
            MemoryEstimate with detailed breakdown
        """
        from tritter.core.model_specs import estimate_memory

        return estimate_memory(self.get_model_spec(), batch_size)

    def get_hardware_recommendation(
        self,
        target_vram_gb: float | None = None,
        target_gpus: int = 1,
        for_training: bool = False,
    ) -> HardwareRecommendation:
        """Get hardware recommendations for this config.

        Args:
            target_vram_gb: Available VRAM per GPU (defaults to max_memory_gb)
            target_gpus: Number of available GPUs
            for_training: Whether this is for training (vs inference)

        Returns:
            HardwareRecommendation with configuration guidance
        """
        from tritter.core.model_specs import recommend_hardware

        vram = target_vram_gb if target_vram_gb is not None else float(self.max_memory_gb)
        return recommend_hardware(
            self.model_size,
            target_vram_gb=vram,
            target_gpus=target_gpus,
            for_training=for_training,
        )

    def total_params(self) -> int:
        """Estimate total parameter count.

        Returns:
            Total number of parameters
        """
        return self.get_model_spec().total_params()

    def total_params_billions(self) -> float:
        """Total parameters in billions.

        Returns:
            Parameter count in billions
        """
        return self.total_params() / 1e9

    @classmethod
    def for_research(
        cls,
        *,
        # Core architecture (required for custom configs)
        hidden_size: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        # Optional architecture parameters
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        vocab_size: int = 65536,
        max_position_embeddings: int = 131072,
        # Model size reference (optional, for sensible defaults)
        model_size: Literal[
            "test",
            "125M",
            "350M",
            "1B",
            "3B",
            "7B",
            "10B",
            "13B",
            "30B",
            "33B",
            "40B",
            "65B",
            "70B",
        ]
        | None = None,
        # All other config parameters can be overridden
        **kwargs: Any,
    ) -> TritterConfig:
        """Create a research configuration with full manual control over architecture.

        Why: Researchers need fine-grained control over architecture hyperparameters to explore
        design space (e.g., wider-shallower vs narrower-deeper models, different head counts,
        intermediate size ratios). This method provides a clean API for custom architectures
        while still offering sensible defaults from standard model sizes.

        Two modes of operation:

        1. **From scratch** - Specify hidden_size, num_layers, num_heads explicitly:
           ```python
           config = TritterConfig.for_research(
               hidden_size=3072,      # Custom width
               num_layers=28,         # Custom depth
               num_heads=24,          # Custom parallelism
               intermediate_size=8192,  # Custom FFN width
           )
           ```

        2. **From reference** - Start from a model_size spec, then override specific params:
           ```python
           config = TritterConfig.for_research(
               model_size="7B",       # Use 7B as base
               num_layers=40,         # But make it deeper
               intermediate_size=14336,  # And wider FFN
           )
           ```

        Parameters to tune for research:

        **Width vs Depth Trade-offs:**
        - hidden_size: Model width. Larger = more expressiveness, more memory.
          Typical: 2048 (small), 4096 (medium), 8192 (large)
        - num_layers: Model depth. More layers = more compute, better compositionality.
          Typical: 16-32 (small), 32-60 (medium), 60-80 (large)

        **Attention Architecture:**
        - num_heads: Parallel attention heads. Must divide hidden_size evenly.
          Standard head_dim is 64-128, so num_heads = hidden_size / head_dim.
          Example: hidden_size=4096 → 32 heads (128-dim) or 64 heads (64-dim)
        - num_kv_heads: For Grouped Query Attention (GQA). Set lower than num_heads
          to reduce KV-cache memory. None = MHA (num_kv_heads == num_heads).
          Typical GQA ratios: 4:1, 8:1 (e.g., 32 query heads, 8 KV heads)

        **FFN Width:**
        - intermediate_size: FFN hidden dimension. Typically 2.5-4x hidden_size.
          Larger = more capacity but slower and more memory.
          Llama uses ~2.7x, Mistral uses ~2.67x, older models use 4x.

        **Context Window:**
        - max_position_embeddings: Maximum sequence length. 128K default.
          Longer = more memory for positional embeddings and KV-cache.
          Consider sliding_window_size for very long contexts.

        **Vocabulary:**
        - vocab_size: Must be >= 264 (8 special + 256 bytes). Larger vocabs
          improve compression but increase embedding table size.

        Constraints and validation:

        1. **hidden_size % num_heads == 0** - Head dimension must be integer.
           Standard head_dim: 64, 96, 128, or 256.

        2. **num_kv_heads divides num_heads** - For GQA, query heads must be
           evenly distributed across KV heads. Example: 32 Q heads / 8 KV heads = 4:1.

        3. **vocab_size >= 264** - Minimum for byte-level encoding.

        4. **intermediate_size > 0** - FFN must have positive width.

        Memory implications:

        - **Model parameters** ≈ 12 * num_layers * hidden_size²
          (rough estimate, actual depends on vocab_size and intermediate_size)
        - **KV-cache per token** = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        - **Activations** ≈ batch_size * seq_len * hidden_size * 12
        - **Packed ternary weights** ≈ total_params * 0.25 bytes

        Use config.estimate_memory() and config.get_hardware_recommendation() to check
        if your custom architecture fits target hardware.

        Examples:

        ```python
        # Wider-shallower model (more parameters in fewer layers)
        config = TritterConfig.for_research(
            hidden_size=5120,
            num_layers=24,
            num_heads=40,
            intermediate_size=13824,
        )

        # Narrower-deeper model (fewer parameters across more layers)
        config = TritterConfig.for_research(
            hidden_size=3072,
            num_layers=48,
            num_heads=24,
            intermediate_size=8192,
        )

        # Custom GQA ratio (16:1 for extreme KV-cache reduction)
        config = TritterConfig.for_research(
            model_size="7B",       # Base: 32 heads, 8 KV heads (4:1)
            num_kv_heads=2,        # Override: 32 heads, 2 KV heads (16:1)
        )

        # Ultra-long context with sliding window
        config = TritterConfig.for_research(
            model_size="7B",
            max_position_embeddings=1048576,  # 1M tokens
            use_sliding_window=True,
            sliding_window_size=4096,
        )

        # Minimal model for ablation studies
        config = TritterConfig.for_research(
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=1376,
            vocab_size=1024,  # Small vocab for faster experiments
        )
        ```

        Args:
            hidden_size: Hidden dimension (required if model_size not given)
            num_layers: Number of transformer layers (required if model_size not given)
            num_heads: Number of attention heads (required if model_size not given)
            num_kv_heads: Number of KV heads for GQA (None = MHA)
            intermediate_size: FFN intermediate dimension (defaults to ~2.7x hidden_size)
            vocab_size: Vocabulary size (default 65536)
            max_position_embeddings: Maximum context length (default 128K)
            model_size: Reference model size for defaults (optional)
            **kwargs: Any other TritterConfig parameters (dropout, use_flash_attention, etc.)

        Returns:
            TritterConfig with custom architecture

        Raises:
            ValueError: If required parameters are missing or constraints are violated
        """
        # Determine base configuration
        if model_size is not None:
            # Start from model spec
            from tritter.core.model_specs import get_model_spec

            spec = get_model_spec(model_size)

            # Use spec values as defaults, but allow overrides
            hidden_size = hidden_size or spec.hidden_size
            num_layers = num_layers or spec.num_layers
            num_heads = num_heads or spec.num_heads
            if num_kv_heads is None and spec.num_kv_heads is not None:
                num_kv_heads = spec.num_kv_heads
            if intermediate_size is None:
                intermediate_size = spec.intermediate_size
        else:
            # Fully manual mode - all required params must be provided
            if hidden_size is None:
                raise ValueError(
                    "hidden_size is required when model_size is not specified. "
                    "Either provide model_size as a base, or specify hidden_size/num_layers/num_heads."
                )
            if num_layers is None:
                raise ValueError(
                    "num_layers is required when model_size is not specified. "
                    "Either provide model_size as a base, or specify hidden_size/num_layers/num_heads."
                )
            if num_heads is None:
                raise ValueError(
                    "num_heads is required when model_size is not specified. "
                    "Either provide model_size as a base, or specify hidden_size/num_layers/num_heads."
                )

            # Default intermediate_size to ~2.7x hidden_size if not provided
            if intermediate_size is None:
                intermediate_size = int(hidden_size * 2.7)

        # Validate constraints early
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}). "
                f"Current head_dim would be {hidden_size / num_heads:.2f}, which is not an integer. "
                f"Try adjusting num_heads to divide evenly (e.g., {hidden_size // 64} heads for "
                f"head_dim=64, or {hidden_size // 128} heads for head_dim=128)."
            )

        if num_kv_heads is not None:
            if num_kv_heads > num_heads:
                raise ValueError(
                    f"num_kv_heads ({num_kv_heads}) cannot exceed num_heads ({num_heads})"
                )
            if num_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) "
                    f"for Grouped Query Attention. Current ratio: {num_heads / num_kv_heads:.2f}. "
                    f"Try num_kv_heads = {[n for n in [1, 2, 4, 8, 16] if num_heads % n == 0]}"
                )

        # Build config with manual overrides
        # Use a base model_size for non-architecture defaults, but override architecture params
        base_model_size = model_size if model_size is not None else "3B"

        config = cls(
            model_size=base_model_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )

        return config
