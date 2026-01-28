"""FlexAttention integration for efficient attention patterns.

Provides FlexAttention-based attention layer with automatic fallback to SDPA,
BlockMask factory for configuration-driven mask creation, and mask caching.

Why: Tritter's 128K context window with document-packed training requires efficient
attention patterns beyond simple causal masking. FlexAttention (PyTorch 2.5+)
compiles composable mask functions to fused Triton kernels without materializing
O(n²) mask tensors, enabling sliding windows, document boundaries, and StreamingLLM
sinks within 16GB VRAM.

Design rationale: Configuration-driven mask creation centralizes attention pattern
logic in TritterConfig instead of scattering it across model code. Automatic SDPA
fallback ensures backward compatibility with PyTorch < 2.5. BlockMask caching
avoids recompilation for same sequence structure during training.
"""

import sys
from collections import OrderedDict
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from tritter.core.config import TritterConfig
from tritter.models.attention_patterns import (
    and_masks,
    causal_mask,
    document_mask,
    prefix_lm_mask,
    sliding_window_mask,
    streamingllm_mask,
)

# Check PyTorch version for FlexAttention availability
# Why: FlexAttention API introduced in PyTorch 2.5.0. Earlier versions must
# fall back to standard SDPA. We check at module load to avoid runtime errors.
PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

# Check Python version - torch.compile not supported on Python 3.14+
# Why: PyTorch's dynamo (required by torch.compile) doesn't support Python 3.14+
_PYTHON_VERSION = sys.version_info[:2]
_TORCH_COMPILE_SUPPORTED = _PYTHON_VERSION < (3, 14)

# FlexAttention requires both PyTorch 2.5+ AND torch.compile support
HAS_FLEX_ATTENTION = PYTORCH_VERSION >= (2, 5) and _TORCH_COMPILE_SUPPORTED

if HAS_FLEX_ATTENTION:
    try:
        from torch.nn.attention.flex_attention import (
            BlockMask,
            create_block_mask,
            flex_attention,
        )
    except ImportError:
        # PyTorch 2.5+ but flex_attention not available (e.g., CPU-only build)
        HAS_FLEX_ATTENTION = False
        BlockMask = None  # type: ignore
        create_block_mask = None  # type: ignore
        flex_attention = None  # type: ignore
else:
    BlockMask = None  # type: ignore
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore


_BLOCK_MASK_CACHE: "OrderedDict[tuple, BlockMask | None]" = OrderedDict()
_BLOCK_MASK_CACHE_MAX = 32


def _config_cache_key(config: TritterConfig) -> tuple:
    return (
        config.attention_mode,
        config.use_sliding_window,
        config.sliding_window_size,
        config.use_attention_sinks,
        config.num_sink_tokens,
        config.use_streaming_llm,
    )


def _doc_ids_cache_key(doc_ids: Tensor | None) -> tuple | None:
    if doc_ids is None:
        return None
    return (
        tuple(doc_ids.shape),
        str(doc_ids.dtype),
        doc_ids.device.type,
        doc_ids.data_ptr(),
    )


def _cache_get(key: tuple) -> Optional["BlockMask"]:
    if key not in _BLOCK_MASK_CACHE:
        return None
    _BLOCK_MASK_CACHE.move_to_end(key)
    return _BLOCK_MASK_CACHE[key]


def _cache_put(key: tuple, value: Optional["BlockMask"]) -> None:
    _BLOCK_MASK_CACHE[key] = value
    _BLOCK_MASK_CACHE.move_to_end(key)
    while len(_BLOCK_MASK_CACHE) > _BLOCK_MASK_CACHE_MAX:
        _BLOCK_MASK_CACHE.popitem(last=False)


def create_attention_mask(
    config: TritterConfig,
    seq_len: int,
    doc_ids: Tensor | None = None,
    device: str = "cuda",
) -> Optional["BlockMask"]:
    """Create BlockMask based on configuration.

    Args:
        config: TritterConfig with attention settings
        seq_len: Sequence length for mask
        doc_ids: Optional document ID tensor for packed sequences, shape (batch, seq_len)
        device: Target device for mask

    Returns:
        BlockMask suitable for flex_attention(), or None if simple causal mask suffices

    Raises:
        ValueError: If attention_mode is invalid or configuration is inconsistent
        RuntimeError: If FlexAttention unavailable and complex mask needed

    Why: Configuration-driven mask creation centralizes attention pattern logic
    and enables easy experimentation (e.g., toggling sliding window via config).
    Validation here catches invalid configs before expensive training starts.

    Design: Builds mask function by composing primitives from attention_patterns.py
    based on config flags. Returns None for simple causal case to use optimized
    SDPA path without FlexAttention overhead.

    Configuration mapping:
        - attention_mode="causal" + no modifiers -> None (use SDPA is_causal=True)
        - attention_mode="causal" + sliding window -> causal + window
        - attention_mode="prefix_lm" -> prefix_lm mask (requires prefix_length)
        - attention_mode="bidirectional" -> None (full attention, no mask)
        - doc_ids provided -> add document boundary mask
        - use_attention_sinks=True -> add StreamingLLM sinks

    Example:
        >>> config = TritterConfig(
        ...     model_size="7B",
        ...     attention_mode="causal",
        ...     use_sliding_window=True,
        ...     sliding_window_size=4096,
        ... )
        >>> mask = create_attention_mask(config, seq_len=8192)
        >>> # Returns BlockMask with causal + sliding window pattern
    """
    # Validate attention mode
    valid_modes = {"causal", "bidirectional", "prefix_lm", "embedding"}
    if config.attention_mode not in valid_modes:
        raise ValueError(
            f"Invalid attention_mode: {config.attention_mode!r}. "
            f"Must be one of {valid_modes}"
        )

    # Bidirectional mode: full attention, no mask needed
    # Why: Bidirectional attention (e.g., for embeddings) allows all positions
    # to attend to all others. SDPA with is_causal=False handles this efficiently.
    if config.attention_mode == "bidirectional":
        return None

    # Validate sliding window configuration
    if config.use_sliding_window:
        if config.sliding_window_size is None or config.sliding_window_size <= 0:
            raise ValueError(
                f"sliding_window_size must be > 0 when use_sliding_window=True, "
                f"got {config.sliding_window_size}"
            )

    # Validate attention sinks configuration
    # Why: Attention sinks require sliding window to bound KV-cache. Without window,
    # sinks provide no benefit (would keep entire history anyway).
    if config.use_attention_sinks:
        if not config.use_sliding_window:
            raise ValueError(
                "use_attention_sinks=True requires use_sliding_window=True. "
                "Attention sinks only make sense with bounded KV-cache."
            )
        if config.num_sink_tokens <= 0:
            raise ValueError(
                f"num_sink_tokens must be > 0 when use_attention_sinks=True, "
                f"got {config.num_sink_tokens}"
            )

    # Validate prefix_lm mode
    # Why: Prefix-LM needs prefix_length to know where bidirectional region ends.
    # Without it, we can't construct the mask pattern.
    if config.attention_mode == "prefix_lm":
        if not hasattr(config, "prefix_length") or config.prefix_length <= 0:
            raise ValueError(
                "attention_mode='prefix_lm' requires config.prefix_length > 0. "
                "Set prefix_length to the instruction prompt length."
            )

    # Build list of mask functions to combine
    # Why: Start with empty list and conditionally add patterns based on config.
    # This enables flexible composition (e.g., causal + window + doc boundaries).
    mask_functions = []

    # Handle attention mode-specific patterns
    if config.attention_mode == "causal":
        # StreamingLLM combines sinks + sliding window
        if config.use_attention_sinks:
            # Sinks require both FlexAttention and sliding window
            if not HAS_FLEX_ATTENTION:
                raise RuntimeError(
                    "FlexAttention (PyTorch >= 2.5) required for attention sinks. "
                    f"Current version: {torch.__version__}"
                )
            mask_functions.append(
                streamingllm_mask(
                    sink_tokens=config.num_sink_tokens,
                    window_size=config.sliding_window_size,  # type: ignore
                )
            )
        # Sliding window without sinks
        elif config.use_sliding_window:
            if not HAS_FLEX_ATTENTION:
                raise RuntimeError(
                    "FlexAttention (PyTorch >= 2.5) required for sliding window. "
                    f"Current version: {torch.__version__}"
                )
            # Combine causal + sliding window
            mask_functions.append(causal_mask)
            mask_functions.append(sliding_window_mask(config.sliding_window_size))  # type: ignore
        # Pure causal: use SDPA fallback (no FlexAttention needed)
        else:
            mask_functions.append(causal_mask)

    elif config.attention_mode == "prefix_lm":
        if not HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "FlexAttention (PyTorch >= 2.5) required for prefix_lm mode. "
                f"Current version: {torch.__version__}"
            )
        mask_functions.append(prefix_lm_mask(config.prefix_length))  # type: ignore

    elif config.attention_mode == "embedding":
        # Embedding mode: future feature, not yet implemented
        # Why: Coconut-style continuous latent reasoning operates in embedding space
        # with custom attention patterns. Placeholder for future development.
        raise NotImplementedError(
            "attention_mode='embedding' not yet implemented. "
            "Planned for Coconut-style continuous latent reasoning."
        )

    # Add document boundary mask if doc_ids provided
    # Why: Document packing requires preventing cross-document attention to avoid
    # corrupting gradients. This is independent of other patterns (e.g., can have
    # causal + window + doc boundaries).
    if doc_ids is not None:
        if not HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "FlexAttention (PyTorch >= 2.5) required for document masking. "
                f"Current version: {torch.__version__}"
            )
        mask_functions.append(document_mask(doc_ids))

    # Cache lookup (only for FlexAttention-supported masks)
    cache_key = (_config_cache_key(config), seq_len, device, _doc_ids_cache_key(doc_ids))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # If only simple causal mask and no doc_ids, use SDPA is_causal=True
    # Why: FlexAttention has overhead (kernel compilation, dispatch). For simple
    # causal masking, SDPA with is_causal=True is faster and uses less memory.
    if len(mask_functions) == 1 and mask_functions[0] is causal_mask:
        return None

    # If no masks at all (shouldn't happen given validation above), return None
    if len(mask_functions) == 0:
        return None

    # Combine multiple mask functions with AND logic
    # Why: All conditions must be satisfied (e.g., must be causal AND in window
    # AND same document). Boolean AND composes constraints correctly.
    if len(mask_functions) > 1:
        combined_mask_fn = and_masks(*mask_functions)
    else:
        combined_mask_fn = mask_functions[0]

    # Create BlockMask from combined mask function
    # Why: BlockMask is PyTorch's compiled representation of the mask pattern.
    # It analyzes the mask function to determine block-sparse structure and
    # generates optimized Triton kernels.
    if not HAS_FLEX_ATTENTION:
        raise RuntimeError(
            "FlexAttention (PyTorch >= 2.5) required for complex attention patterns. "
            f"Current version: {torch.__version__}"
        )

    # create_block_mask expects shape (batch, num_heads, seq_len, seq_len)
    # but we don't know batch/num_heads here, so use 1 and 1
    # Why: BlockMask structure depends only on seq_len and mask function, not batch
    # size or number of heads (which just replicate the same pattern). We use
    # placeholders (1, 1) and the mask will broadcast correctly during attention.
    block_mask = create_block_mask(
        combined_mask_fn,
        B=None,  # Batch dimension (None = use default 1)
        H=None,  # Head dimension (None = use default 1)
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    _cache_put(cache_key, block_mask)
    return block_mask


class FlexAttentionLayer(nn.Module):
    """FlexAttention wrapper with automatic fallback to SDPA.

    Provides a unified interface for attention computation that uses FlexAttention
    when available (PyTorch >= 2.5) and falls back to SDPA otherwise. Includes
    BlockMask caching for same sequence structure to avoid recompilation overhead.

    Why: Encapsulating FlexAttention vs SDPA logic in a layer provides clean
    abstraction for TritterAttention. The automatic fallback ensures models work
    on older PyTorch versions (though without advanced masking). Caching BlockMask
    avoids expensive recompilation during training when sequence structure repeats.

    Design: The layer is stateless except for _mask_cache. It takes pre-computed
    query/key/value tensors (after projection) rather than managing projections
    itself. This matches the typical attention layer pattern in transformers.

    Attributes:
        config: TritterConfig instance controlling attention behavior
        _mask_cache: Dict mapping (seq_len, doc_ids_hash) -> BlockMask
        _use_flex: Whether FlexAttention is available for this instance

    Example:
        >>> config = TritterConfig(model_size="7B")
        >>> layer = FlexAttentionLayer(config)
        >>> q = torch.randn(2, 8, 1024, 128)  # (B, num_heads, L, head_dim)
        >>> k = torch.randn(2, 8, 1024, 128)
        >>> v = torch.randn(2, 8, 1024, 128)
        >>> output = layer(q, k, v)  # (B, num_heads, L, head_dim)
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize FlexAttention layer.

        Args:
            config: TritterConfig with attention settings

        Why: Store config for mask creation in forward(). Initialize cache as
        empty dict (will populate on first forward pass). Check FlexAttention
        availability to determine backend.
        """
        super().__init__()
        self.config = config
        self._mask_cache: dict[tuple, BlockMask | None] = {}
        self._use_flex = HAS_FLEX_ATTENTION

        # Compile flex_attention for optimal performance
        # Why: flex_attention requires torch.compile for fused Triton kernels.
        # Without compilation, it falls back to unfused implementation or raises error.
        # We compile once at layer creation to amortize compilation cost.
        if self._use_flex and flex_attention is not None:
            self._flex_attention_compiled = torch.compile(flex_attention)
        else:
            self._flex_attention_compiled = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        block_mask: Optional["BlockMask"] = None,
        score_mod: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
        doc_ids: Tensor | None = None,
    ) -> Tensor:
        """Compute attention output using FlexAttention or SDPA fallback.

        Args:
            query: Query tensor, shape (B, num_heads, L, head_dim)
            key: Key tensor, shape (B, num_heads, L, head_dim)
            value: Value tensor, shape (B, num_heads, L, head_dim)
            block_mask: Optional pre-computed BlockMask (overrides config-based creation)
            score_mod: Optional score modification function (e.g., ALiBi, not yet used)
            doc_ids: Optional document ID tensor for packed sequences, shape (B, L)

        Returns:
            Attention output tensor, shape (B, num_heads, L, head_dim)

        Why: Unified interface for attention computation abstracts backend choice.
        The block_mask parameter allows bypassing config-based creation for custom
        patterns (e.g., different mask per layer). doc_ids enables document-packed
        training without pre-creating masks.

        Design: Check cache first to avoid recompilation. Fall back to SDPA if
        FlexAttention unavailable or if simple causal mask. Use is_causal=True
        for SDPA to enable FlashAttention-2 optimizations.

        Example:
            >>> layer = FlexAttentionLayer(config)
            >>> # Simple causal attention
            >>> output = layer(query, key, value)
            >>> # With document packing
            >>> doc_ids = torch.tensor([[0, 0, 1, 1]] * batch_size)
            >>> output = layer(query, key, value, doc_ids=doc_ids)
        """
        B, num_heads, seq_len, head_dim = query.shape

        # If block_mask explicitly provided, use it directly
        # Why: Allows caller to override config-based mask creation for custom
        # patterns or pre-computed masks (e.g., for beam search).
        if block_mask is None:
            # Create cache key from sequence structure
            # Why: Cache key includes seq_len and doc_ids structure. Same sequence
            # structure can reuse BlockMask without recompilation. Hash doc_ids
            # tensor if present (can't use tensor as dict key directly).
            doc_ids_hash = None
            if doc_ids is not None:
                # Use data pointer as hash (works for same tensor instance)
                # Why: Full tensor hashing is expensive. Data pointer identifies
                # same tensor without computing content hash. This works for training
                # where same doc_ids tensor is reused across batches.
                doc_ids_hash = id(doc_ids)

            cache_key = (seq_len, doc_ids_hash)

            # Check cache first
            if cache_key in self._mask_cache:
                block_mask = self._mask_cache[cache_key]
            else:
                # Create new BlockMask based on config
                block_mask = create_attention_mask(
                    config=self.config,
                    seq_len=seq_len,
                    doc_ids=doc_ids,
                    device=query.device.type,
                )
                # Cache for future use
                self._mask_cache[cache_key] = block_mask

        # Use FlexAttention if available and we have a BlockMask
        # Why: FlexAttention requires BlockMask. If mask is None (simple causal),
        # use SDPA which is faster for that case.
        if self._use_flex and block_mask is not None and self._flex_attention_compiled is not None:
            # FlexAttention path
            # Why: flex_attention() expects (B, H, L, D) tensors and BlockMask.
            # It compiles the mask to fused Triton kernel for efficient computation.
            # We use the pre-compiled version for optimal performance.
            output = self._flex_attention_compiled(
                query=query,
                key=key,
                value=value,
                block_mask=block_mask,
                score_mod=score_mod,  # Future: ALiBi, RoPE, etc.
            )
        else:
            # SDPA fallback path
            # Why: PyTorch < 2.5 or simple causal mask. Use standard SDPA with
            # is_causal=True to enable FlashAttention-2 optimizations.
            #
            # Note: If doc_ids or complex patterns required but FlexAttention
            # unavailable, create_attention_mask() would have raised RuntimeError.
            # So reaching here with doc_ids=None is safe.
            is_causal = (
                self.config.attention_mode == "causal"
                and not self.config.use_sliding_window
                and not self.config.use_attention_sinks
            )

            # scaled_dot_product_attention expects (B, num_heads, L, head_dim)
            output = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,  # No explicit mask
                dropout_p=0.0 if not self.training else self.config.dropout,
                is_causal=is_causal,
            )

        return output  # (B, num_heads, L, head_dim)

    def clear_cache(self) -> None:
        """Clear BlockMask cache.

        Why: Cache can grow unbounded if sequence lengths vary widely during
        training (e.g., curriculum learning with increasing lengths). Clearing
        cache periodically prevents memory leaks.

        Design: Simple dict clear. Caller should invoke this between training
        phases or when memory usage becomes concerning.

        Example:
            >>> layer = FlexAttentionLayer(config)
            >>> # Train with 4K sequences
            >>> for batch in train_loader_4k:
            ...     output = layer(q, k, v)
            >>> # Switch to 8K sequences - old cache entries not useful
            >>> layer.clear_cache()
            >>> for batch in train_loader_8k:
            ...     output = layer(q, k, v)
        """
        self._mask_cache.clear()


__all__ = [
    "create_attention_mask",
    "FlexAttentionLayer",
    "HAS_FLEX_ATTENTION",
]
