"""Model specifications for different parameter scales.

Why: Centralizes architecture specifications for all supported model sizes (1B-70B).
Provides memory estimation, compute recommendations, and automatic configuration
for layer streaming, multi-GPU, and other optimizations based on hardware targets.

Embedding-Prediction Context: All model sizes share the same embedding-prediction
paradigm - they operate in continuous embedding space with token prediction as
temporary scaffolding. The architecture scales by adjusting hidden_size, num_layers,
and num_heads while maintaining consistent head_dim (typically 128).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Supported model sizes
ModelSize = Literal["1B", "3B", "7B", "10B", "13B", "30B", "33B", "40B", "65B", "70B"]


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a model size variant.

    Why: Frozen dataclass ensures specs can't be accidentally modified after
    creation. Each spec defines the complete architecture for a model size.

    Attributes:
        name: Human-readable name (e.g., "7B")
        hidden_size: Dimension of hidden representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA, None = num_heads)
        intermediate_size: FFN intermediate dimension
        vocab_size: Default vocabulary size
        max_position_embeddings: Maximum context length
        head_dim: Dimension per attention head (computed if None)
        rope_theta: RoPE base frequency
        description: Brief description of use case
    """

    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int | None  # None = MHA (num_kv_heads == num_heads)
    intermediate_size: int
    vocab_size: int = 65536
    max_position_embeddings: int = 131072  # 128K default
    head_dim: int | None = None  # Computed from hidden_size // num_heads
    rope_theta: float = 500000.0  # Extended RoPE for 128K context
    description: str = ""

    def __post_init__(self) -> None:
        """Validate spec consistency."""
        # Validate head_dim divides evenly
        computed_head_dim = self.hidden_size // self.num_heads
        if self.head_dim is not None and self.head_dim != computed_head_dim:
            raise ValueError(f"head_dim mismatch: {self.head_dim} vs computed {computed_head_dim}")

    @property
    def effective_head_dim(self) -> int:
        """Get head dimension (computed if not specified)."""
        return self.head_dim or (self.hidden_size // self.num_heads)

    @property
    def effective_num_kv_heads(self) -> int:
        """Get KV heads (equals num_heads for MHA)."""
        return self.num_kv_heads or self.num_heads

    @property
    def uses_gqa(self) -> bool:
        """Whether this spec uses Grouped Query Attention."""
        return self.num_kv_heads is not None and self.num_kv_heads < self.num_heads

    def total_params(self) -> int:
        """Estimate total parameter count.

        Why: Accurate parameter estimation helps with memory planning
        and matching published model sizes.
        """
        # Embedding: vocab_size * hidden_size
        embedding_params = self.vocab_size * self.hidden_size

        # Per-layer params
        # Attention: Q, K, V projections + output projection
        # Q: hidden_size * hidden_size
        # K, V: hidden_size * (num_kv_heads * head_dim) each
        # O: hidden_size * hidden_size
        kv_dim = self.effective_num_kv_heads * self.effective_head_dim
        attn_params = (
            self.hidden_size * self.hidden_size  # Q
            + self.hidden_size * kv_dim  # K
            + self.hidden_size * kv_dim  # V
            + self.hidden_size * self.hidden_size  # O
        )

        # FFN: up, gate, down projections (SwiGLU style)
        ffn_params = (
            self.hidden_size * self.intermediate_size  # up
            + self.hidden_size * self.intermediate_size  # gate
            + self.intermediate_size * self.hidden_size  # down
        )

        # Layer norms (2 per layer: attention + FFN)
        norm_params = 2 * self.hidden_size * 2  # weight + bias each

        per_layer_params = attn_params + ffn_params + norm_params
        total_layer_params = self.num_layers * per_layer_params

        # Final norm + LM head
        final_norm_params = self.hidden_size * 2
        lm_head_params = self.hidden_size * self.vocab_size

        return embedding_params + total_layer_params + final_norm_params + lm_head_params

    def total_params_billions(self) -> float:
        """Total parameters in billions."""
        return self.total_params() / 1e9


# =============================================================================
# Model Specifications Registry
# =============================================================================

MODEL_SPECS: dict[ModelSize, ModelSpec] = {
    # ---------------------------------------------------------------------
    # Small models (1B-3B): Single consumer GPU
    # ---------------------------------------------------------------------
    "1B": ModelSpec(
        name="1B",
        hidden_size=2048,
        num_layers=16,
        num_heads=16,
        num_kv_heads=None,  # MHA
        intermediate_size=5632,
        description="Lightweight model for edge deployment and fast experimentation",
    ),
    "3B": ModelSpec(
        name="3B",
        hidden_size=2560,
        num_layers=26,
        num_heads=20,
        num_kv_heads=None,  # MHA
        intermediate_size=6912,
        description="Consumer GPU sweet spot (RTX 3080/4070 class)",
    ),
    # ---------------------------------------------------------------------
    # Medium models (7B-13B): Single high-end consumer GPU
    # ---------------------------------------------------------------------
    "7B": ModelSpec(
        name="7B",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,  # GQA 4:1
        intermediate_size=11008,
        description="RTX 5080/4090 optimized, production quality",
    ),
    "10B": ModelSpec(
        name="10B",
        hidden_size=4096,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,  # GQA 4:1
        intermediate_size=14336,
        description="Extended 7B with more layers and wider FFN",
    ),
    "13B": ModelSpec(
        name="13B",
        hidden_size=5120,
        num_layers=40,
        num_heads=40,
        num_kv_heads=8,  # GQA 5:1
        intermediate_size=13824,
        description="Matches LLaMA-13B class, requires 24GB+ or streaming",
    ),
    # ---------------------------------------------------------------------
    # Large models (30B-40B): Multi-GPU or aggressive streaming
    # ---------------------------------------------------------------------
    "30B": ModelSpec(
        name="30B",
        hidden_size=6656,
        num_layers=60,
        num_heads=52,
        num_kv_heads=8,  # GQA 6.5:1
        intermediate_size=17920,
        head_dim=128,  # Explicit for this config
        description="Multi-GPU recommended, matches LLaMA-30B class",
    ),
    "33B": ModelSpec(
        name="33B",
        hidden_size=6912,
        num_layers=60,
        num_heads=54,
        num_kv_heads=6,  # GQA 9:1
        intermediate_size=18432,
        head_dim=128,
        description="Optimized for 2x24GB GPU setup",
    ),
    "40B": ModelSpec(
        name="40B",
        hidden_size=8192,
        num_layers=60,
        num_heads=64,
        num_kv_heads=8,  # GQA 8:1
        intermediate_size=21760,
        description="Professional workstation (2-4 GPUs)",
    ),
    # ---------------------------------------------------------------------
    # Extra large models (65B-70B): Data center / multi-node
    # ---------------------------------------------------------------------
    "65B": ModelSpec(
        name="65B",
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,  # GQA 8:1
        intermediate_size=22016,
        description="Matches LLaMA-65B class, requires 4+ GPUs",
    ),
    "70B": ModelSpec(
        name="70B",
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,  # GQA 8:1
        intermediate_size=28672,
        description="Frontier model, matches LLaMA-2-70B class",
    ),
}


def get_model_spec(size: ModelSize) -> ModelSpec:
    """Get model specification by size name.

    Args:
        size: Model size identifier

    Returns:
        ModelSpec for the requested size

    Raises:
        KeyError: If size is not supported
    """
    if size not in MODEL_SPECS:
        supported = list(MODEL_SPECS.keys())
        raise KeyError(f"Unknown model size: {size}. Supported: {supported}")
    return MODEL_SPECS[size]


# =============================================================================
# Memory Estimation
# =============================================================================


@dataclass
class MemoryEstimate:
    """Memory requirements for a model configuration.

    All values in bytes unless otherwise noted.
    """

    # Weight storage
    weights_fp32: int
    weights_fp16: int
    weights_int8: int
    weights_packed_ternary: int

    # Training memory (with optimizer)
    training_fp32: int  # Weights + gradients + optimizer states (AdamW 2x)
    training_bf16_mixed: int  # BF16 weights/grads + FP32 optimizer

    # KV-cache for different context lengths
    kv_cache_4k_fp16: int
    kv_cache_32k_fp16: int
    kv_cache_128k_fp16: int
    kv_cache_4k_int4: int
    kv_cache_32k_int4: int
    kv_cache_128k_int4: int

    # Activation memory (approximate)
    activation_per_token: int

    def format_gb(self, value: int) -> str:
        """Format bytes as GB string."""
        return f"{value / (1024**3):.2f} GB"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Memory Estimate ===",
            f"Weights (FP32):          {self.format_gb(self.weights_fp32)}",
            f"Weights (FP16):          {self.format_gb(self.weights_fp16)}",
            f"Weights (INT8):          {self.format_gb(self.weights_int8)}",
            f"Weights (Packed 2-bit):  {self.format_gb(self.weights_packed_ternary)}",
            "",
            f"Training (FP32 AdamW):   {self.format_gb(self.training_fp32)}",
            f"Training (BF16 mixed):   {self.format_gb(self.training_bf16_mixed)}",
            "",
            "KV-Cache (FP16, batch=1):",
            f"  4K context:            {self.format_gb(self.kv_cache_4k_fp16)}",
            f"  32K context:           {self.format_gb(self.kv_cache_32k_fp16)}",
            f"  128K context:          {self.format_gb(self.kv_cache_128k_fp16)}",
            "",
            "KV-Cache (INT4, batch=1):",
            f"  4K context:            {self.format_gb(self.kv_cache_4k_int4)}",
            f"  32K context:           {self.format_gb(self.kv_cache_32k_int4)}",
            f"  128K context:          {self.format_gb(self.kv_cache_128k_int4)}",
        ]
        return "\n".join(lines)


def estimate_memory(spec: ModelSpec, batch_size: int = 1) -> MemoryEstimate:
    """Estimate memory requirements for a model specification.

    Args:
        spec: Model specification
        batch_size: Batch size for KV-cache estimation

    Returns:
        MemoryEstimate with detailed breakdown

    Why: Accurate memory estimation is critical for:
    - Choosing appropriate hardware
    - Configuring layer streaming
    - Setting context window limits
    - Planning multi-GPU distribution
    """
    total_params = spec.total_params()

    # Weight storage at different precisions
    weights_fp32 = total_params * 4
    weights_fp16 = total_params * 2
    weights_int8 = total_params * 1
    # Packed ternary: 2 bits/param + scales (~4 bytes per 4096 params)
    weights_packed = int(total_params * 0.25 + (total_params / 4096) * 4)

    # Training memory
    # FP32: weights + gradients + AdamW (2x for m and v)
    training_fp32 = weights_fp32 * 4  # 1x weights + 1x grads + 2x optimizer
    # BF16 mixed: BF16 weights/grads + FP32 optimizer states
    training_bf16 = weights_fp16 * 2 + weights_fp32 * 2  # weights+grads + optimizer

    # KV-cache calculation
    # Per layer: 2 * batch * seq_len * num_kv_heads * head_dim * dtype_bytes
    kv_heads = spec.effective_num_kv_heads
    head_dim = spec.effective_head_dim

    def calc_kv_cache(seq_len: int, dtype_bytes: float) -> int:
        per_layer = int(2 * batch_size * seq_len * kv_heads * head_dim * dtype_bytes)
        return spec.num_layers * per_layer

    kv_4k_fp16 = calc_kv_cache(4096, 2)
    kv_32k_fp16 = calc_kv_cache(32768, 2)
    kv_128k_fp16 = calc_kv_cache(131072, 2)
    kv_4k_int4 = calc_kv_cache(4096, 0.5)
    kv_32k_int4 = calc_kv_cache(32768, 0.5)
    kv_128k_int4 = calc_kv_cache(131072, 0.5)

    # Activation memory (rough estimate)
    # Peak activation per token: ~12 * hidden_size * dtype_bytes
    activation_per_token = 12 * spec.hidden_size * 2  # FP16

    return MemoryEstimate(
        weights_fp32=weights_fp32,
        weights_fp16=weights_fp16,
        weights_int8=weights_int8,
        weights_packed_ternary=weights_packed,
        training_fp32=training_fp32,
        training_bf16_mixed=training_bf16,
        kv_cache_4k_fp16=kv_4k_fp16,
        kv_cache_32k_fp16=kv_32k_fp16,
        kv_cache_128k_fp16=kv_128k_fp16,
        kv_cache_4k_int4=kv_4k_int4,
        kv_cache_32k_int4=kv_32k_int4,
        kv_cache_128k_int4=kv_128k_int4,
        activation_per_token=activation_per_token,
    )


# =============================================================================
# Hardware Recommendations
# =============================================================================


@dataclass
class HardwareRecommendation:
    """Hardware and configuration recommendations for a model.

    Why: Provides actionable guidance for deploying models on different
    hardware configurations.
    """

    model_size: ModelSize
    spec: ModelSpec

    # Minimum hardware
    min_vram_inference_gb: float
    min_vram_training_gb: float
    min_gpus_inference: int
    min_gpus_training: int

    # Recommended configuration
    use_layer_streaming: bool
    recommended_layer_group_size: int
    recommended_context_length: int
    use_gqa: bool
    use_int4_kv_cache: bool

    # Multi-GPU settings
    tensor_parallel_size: int
    pipeline_parallel_size: int

    # Notes
    notes: list[str]


def recommend_hardware(
    size: ModelSize,
    target_vram_gb: float = 16.0,
    target_gpus: int = 1,
    for_training: bool = False,
) -> HardwareRecommendation:
    """Generate hardware recommendations for a model size.

    Args:
        size: Model size
        target_vram_gb: Available VRAM per GPU
        target_gpus: Number of available GPUs
        for_training: Whether this is for training (vs inference)

    Returns:
        HardwareRecommendation with configuration guidance
    """
    spec = get_model_spec(size)
    mem = estimate_memory(spec)

    notes: list[str] = []

    # Calculate minimum VRAM needed
    packed_weights_gb = mem.weights_packed_ternary / (1024**3)
    kv_32k_int4_gb = mem.kv_cache_32k_int4 / (1024**3)
    overhead_gb = 2.0  # CUDA overhead, activations, etc.

    min_vram_inference = packed_weights_gb + kv_32k_int4_gb + overhead_gb
    min_vram_training = mem.training_bf16_mixed / (1024**3)

    # Determine if layer streaming is needed
    # Streaming is needed if:
    # 1. Packed weights alone exceed 80% of single GPU VRAM, OR
    # 2. Total inference footprint exceeds single GPU VRAM
    total_available_vram = target_vram_gb * target_gpus
    weights_exceed_threshold = packed_weights_gb > (target_vram_gb * 0.8)
    total_exceeds_vram = min_vram_inference > target_vram_gb

    use_streaming = weights_exceed_threshold or (total_exceeds_vram and target_gpus == 1)

    if use_streaming:
        if weights_exceed_threshold:
            notes.append("Layer streaming required - weights exceed single GPU capacity")
        else:
            notes.append("Layer streaming recommended - total inference footprint exceeds VRAM")

    # Recommend layer group size based on VRAM
    if target_vram_gb >= 24:
        layer_group_size = 8
    elif target_vram_gb >= 16:
        layer_group_size = 4
    elif target_vram_gb >= 8:
        layer_group_size = 2
    else:
        layer_group_size = 1

    # Recommend context length based on KV-cache fit
    if target_vram_gb >= 24:
        recommended_context = 131072  # 128K
    elif target_vram_gb >= 16:
        recommended_context = 32768  # 32K
    elif target_vram_gb >= 8:
        recommended_context = 8192  # 8K
    else:
        recommended_context = 4096  # 4K

    # Multi-GPU parallelism recommendations
    tensor_parallel = 1
    pipeline_parallel = 1

    if target_gpus >= 8 and size in ["65B", "70B"]:
        tensor_parallel = 8
        notes.append("Full tensor parallelism across 8 GPUs recommended")
    elif target_gpus >= 4 and size in ["30B", "33B", "40B", "65B", "70B"]:
        tensor_parallel = 4
        notes.append("Tensor parallelism across 4 GPUs recommended")
    elif target_gpus >= 2 and size in ["13B", "30B", "33B", "40B"]:
        tensor_parallel = 2
        notes.append("Tensor parallelism across 2 GPUs recommended")

    # GQA and INT4 KV-cache recommendations
    use_gqa = spec.uses_gqa
    use_int4_kv = recommended_context > 8192 or target_vram_gb < 24

    if use_gqa:
        ratio = spec.num_heads // spec.effective_num_kv_heads
        notes.append(f"Using GQA with {ratio}:1 ratio for KV efficiency")

    if use_int4_kv:
        notes.append("INT4 KV-cache recommended for memory efficiency")

    # Training-specific notes
    if for_training:
        if min_vram_training > total_available_vram:
            notes.append(
                f"WARNING: Training requires ~{min_vram_training:.0f}GB, "
                f"but only {total_available_vram:.0f}GB available. "
                "Consider gradient checkpointing, smaller batch, or more GPUs."
            )

    # Calculate minimum GPU requirements
    min_gpus_inference = max(1, int(packed_weights_gb / (target_vram_gb * 0.7)) + 1)
    if use_streaming and min_gpus_inference == 1:
        min_gpus_inference = 1  # Streaming can work on 1 GPU

    min_gpus_training = max(1, int(min_vram_training / target_vram_gb) + 1)

    return HardwareRecommendation(
        model_size=size,
        spec=spec,
        min_vram_inference_gb=min_vram_inference,
        min_vram_training_gb=min_vram_training,
        min_gpus_inference=min_gpus_inference,
        min_gpus_training=min_gpus_training,
        use_layer_streaming=use_streaming,
        recommended_layer_group_size=layer_group_size,
        recommended_context_length=recommended_context,
        use_gqa=use_gqa,
        use_int4_kv_cache=use_int4_kv,
        tensor_parallel_size=tensor_parallel,
        pipeline_parallel_size=pipeline_parallel,
        notes=notes,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def list_models() -> list[tuple[ModelSize, float, str]]:
    """List all supported models with parameter counts.

    Returns:
        List of (size, params_billions, description) tuples
    """
    return [
        (size, spec.total_params_billions(), spec.description) for size, spec in MODEL_SPECS.items()
    ]


def print_model_summary(size: ModelSize) -> None:
    """Print detailed summary for a model size."""
    spec = get_model_spec(size)
    mem = estimate_memory(spec)

    print(f"\n{'=' * 60}")
    print(f"Model: {spec.name} ({spec.total_params_billions():.2f}B parameters)")
    print(f"{'=' * 60}")
    print(f"Description: {spec.description}")
    print()
    print("Architecture:")
    print(f"  Hidden size:       {spec.hidden_size}")
    print(f"  Num layers:        {spec.num_layers}")
    print(f"  Num heads:         {spec.num_heads}")
    print(
        f"  Num KV heads:      {spec.effective_num_kv_heads} {'(GQA)' if spec.uses_gqa else '(MHA)'}"
    )
    print(f"  Head dim:          {spec.effective_head_dim}")
    print(f"  FFN intermediate:  {spec.intermediate_size}")
    print(f"  Vocab size:        {spec.vocab_size}")
    print(f"  Max context:       {spec.max_position_embeddings}")
    print()
    print(mem.summary())
