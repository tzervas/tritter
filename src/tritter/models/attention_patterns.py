"""FlexAttention mask primitive functions for Tritter's 128K context window.

This module provides composable mask functions for PyTorch FlexAttention, enabling
advanced attention patterns beyond simple causal masking. These primitives support:
- Document-packed training (prevent cross-document attention)
- Sliding window attention (bound KV-cache for 128K context)
- StreamingLLM attention sinks (infinite-length generation)
- Prefix-LM instruction tuning (bidirectional prefix + causal suffix)

Why: Tritter's 128K context window with document-packed training requires efficient
attention patterns that cannot be represented with simple causal masking. FlexAttention
(PyTorch 2.5+) enables these patterns to compile to fused Triton kernels without
materializing O(n²) mask tensors, which would require 64GB for 128K sequences.

The embedding-prediction paradigm (Coconut/LCM style) operates in continuous embedding
space, but these attention masks govern how embeddings attend to each other during
transformer computation. The masks are applied during attention score calculation,
before the final embedding output.

Memory Impact: These mask functions have zero memory overhead. FlexAttention compiles
them to Triton kernels that evaluate mask conditions on-the-fly during attention
computation, avoiding materialization of O(n²) mask tensors. For 128K sequences, this
saves 64GB compared to manual masking.

Note: All mask functions follow FlexAttention's signature: (b, h, q_idx, kv_idx) -> bool.
The batch (b) and head (h) indices enable per-batch or per-head masking patterns.
"""

from collections.abc import Callable

import torch


def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Standard autoregressive causal mask for decoder-only language modeling.

    Args:
        b: Batch index (unused, for signature compatibility with FlexAttention)
        h: Head index (unused, for signature compatibility with FlexAttention)
        q_idx: Query position index in sequence (0-indexed)
        kv_idx: Key/Value position index in sequence (0-indexed)

    Returns:
        True if query at q_idx should attend to key/value at kv_idx, False otherwise.
        Returns True when q_idx >= kv_idx (query can attend to self and past tokens).

    Why: Causal masking prevents the model from attending to future tokens during
    autoregressive generation and training. This is the fundamental requirement for
    decoder-only language models: each position can only attend to itself and previous
    positions, not future ones. Without this, the model would "cheat" during training
    by looking ahead at the answer.

    The condition q_idx >= kv_idx implements "attend to self and past":
    - q_idx > kv_idx: Attend to past tokens (e.g., position 5 attends to position 3)
    - q_idx == kv_idx: Attend to self (position 5 attends to position 5)
    - q_idx < kv_idx: Do NOT attend to future (position 5 cannot attend to position 7)

    This is the most basic attention pattern for decoder-only transformers and serves
    as the foundation for more complex patterns (sliding window, prefix-LM, etc.).

    Example:
        >>> # Position 5 attending to past positions
        >>> causal_mask(0, 0, q_idx=5, kv_idx=3)  # Attend to past
        True
        >>> causal_mask(0, 0, q_idx=5, kv_idx=5)  # Attend to self
        True
        >>> causal_mask(0, 0, q_idx=5, kv_idx=7)  # Cannot attend to future
        False

    Note: This function is stateless and has zero memory overhead. FlexAttention
    compiles it to a Triton kernel that evaluates the condition on-the-fly during
    attention computation.
    """
    return q_idx >= kv_idx


def sliding_window_mask(window_size: int) -> Callable[[int, int, int, int], bool]:
    """Create sliding window mask factory for bounded KV-cache attention.

    Args:
        window_size: Maximum distance between query and key positions (must be > 0).
                    For example, window_size=4096 means each token attends to at most
                    the most recent 4096 tokens (plus itself).

    Returns:
        Mask function with signature (b, h, q_idx, kv_idx) -> bool that implements
        sliding window attention: attend only to tokens within window_size distance.

    Why: Sliding window attention bounds the context each token attends to, reducing
    KV-cache memory from O(N²) to O(N*W) where W=window_size. This is critical for
    Tritter's 128K context window on RTX 5080 16GB VRAM. Without sliding windows,
    KV-cache would grow quadratically and exhaust memory.

    The window_size controls the recency bias:
    - Smaller windows (2K): Faster computation, less memory, but lose long-range deps
    - Larger windows (8K): More context preserved, but higher memory cost
    - Recommended for 128K context: 4K window (balances efficiency and dependencies)

    The mask combines causal constraint with window constraint:
    - Causal: q_idx >= kv_idx (no future attention)
    - Window: q_idx - kv_idx <= window_size (bound recency)
    - Combined: (q_idx >= kv_idx) AND (q_idx - kv_idx <= window_size)

    This pattern maintains local dependencies while enabling very long sequences.
    Pairs well with attention sinks (see streamingllm_mask) to preserve important
    early context (e.g., system prompts) beyond the sliding window.

    Example:
        >>> mask_fn = sliding_window_mask(window_size=40)
        >>> # Position 100 with window size 40
        >>> mask_fn(0, 0, q_idx=100, kv_idx=80)  # Within window
        True
        >>> mask_fn(0, 0, q_idx=100, kv_idx=50)  # Outside window (100 - 50 = 50 > 40)
        False
        >>> mask_fn(0, 0, q_idx=100, kv_idx=105)  # Future token (violates causal)
        False

    Note: The closure over window_size enables parameterized mask creation without
    requiring BlockMask recompilation when window_size changes.
    """

    def _sliding_window_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        """Sliding window mask implementation with causal constraint.

        Returns:
            True if (1) causal constraint satisfied AND (2) within window range.

        Why: Use boolean expressions instead of if statements because FlexAttention's
        vmap compilation doesn't support data-dependent control flow with tensors.
        """
        # Must be causal AND within window distance
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)

    return _sliding_window_mask


def document_mask(doc_ids: torch.Tensor) -> Callable[[int, int, int, int], bool]:
    """Create document boundary mask for packed sequence training.

    Args:
        doc_ids: Tensor of shape (batch_size, seq_len) mapping each position to its
                document ID. Positions with the same doc_id belong to the same document.
                Document IDs should be 0-indexed integers. Multiple documents can be
                packed into a single sequence by using different IDs.

    Returns:
        Mask function with signature (b, h, q_idx, kv_idx) -> bool that prevents
        cross-document attention in packed sequences.

    Why: Document-packed training is essential for efficient pretraining with variable-
    length documents. Instead of padding each document to max_length (wasting compute
    on padding tokens), we pack multiple documents into a single sequence up to the
    context limit. However, tokens must not attend across document boundaries, as this
    would leak information between unrelated documents and corrupt the training signal.

    The document mask ensures tokens only attend within their own document:
    - Same document: doc_ids[b, q_idx] == doc_ids[b, kv_idx] → True (attend)
    - Different documents: doc_ids[b, q_idx] != doc_ids[b, kv_idx] → False (mask)

    This is critical for Tritter's training data strategy (see docs/clean-datasets.md),
    which uses document packing to maximize GPU utilization. Without this mask, the
    model would learn spurious correlations between unrelated documents packed into
    the same sequence.

    Example usage with packed documents:
        >>> # Two sequences, each with two documents packed
        >>> doc_ids = torch.tensor([
        ...     [0, 0, 0, 1, 1, 1, 1, 1],  # Batch 0: doc 0 (pos 0-2) + doc 1 (pos 3-7)
        ...     [0, 0, 1, 1, 1, 2, 2, 2],  # Batch 1: 3 docs
        ... ])
        >>> mask_fn = document_mask(doc_ids)
        >>> # Same document: attend
        >>> mask_fn(b=0, h=0, q_idx=1, kv_idx=2)  # Both in doc 0
        True
        >>> # Different documents: mask
        >>> mask_fn(b=0, h=0, q_idx=4, kv_idx=1)  # doc 1 to doc 0
        False

    Note: The doc_ids tensor is captured by closure and accessed during mask evaluation.
    This enables efficient per-position document membership checking without materializing
    the full O(n²) mask. The doc_ids tensor should be on the same device as the attention
    computation for efficient access.
    """

    def _document_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        """Document boundary mask implementation.

        Returns:
            True if query and key belong to the same document (can attend).
            False if query and key belong to different documents (mask out).

        Why: FlexAttention uses vmap to vectorize this function. We return the
        comparison result directly (either bool for direct calls or tensor for vmap).
        Note: .item() cannot be used here as it's incompatible with vmap.
        """
        # Extract document IDs for query and key positions in this batch
        # Note: When vmapped, b and indices are tensors, so indexing returns tensors
        query_doc = doc_ids[b, q_idx]
        key_doc = doc_ids[b, kv_idx]

        # Only attend within same document
        # Return comparison result directly - FlexAttention handles both bool and tensor
        return query_doc == key_doc  # type: ignore[no-any-return]

    return _document_mask


def streamingllm_mask(
    sink_tokens: int,
    window_size: int,
) -> Callable[[int, int, int, int], bool]:
    """Create StreamingLLM attention sink mask for infinite-length generation.

    Args:
        sink_tokens: Number of initial tokens to always attend to regardless of position.
                    These "attention sinks" typically include BOS token and first few
                    prompt tokens. Must be > 0. Typical values: 4-8 tokens.
        window_size: Sliding window size for recent tokens (same as sliding_window_mask).
                    Must be > 0. This bounds the recency window for non-sink tokens.

    Returns:
        Mask function combining attention sinks with sliding window: sink tokens are
        always attended to, plus sliding window of recent tokens.

    Why: StreamingLLM enables infinite-length generation by maintaining a small set of
    "attention sink" tokens that all future tokens attend to, plus a sliding window of
    recent context. This solves two problems:

    1. **KV-cache eviction**: In long generation, we must evict old KV-cache entries to
       bound memory. Naive sliding window loses important early context (e.g., system
       prompt, task description).

    2. **Attention sink phenomenon**: Research shows initial tokens become "attention
       sinks" that stabilize the attention distribution even if not semantically relevant.
       Evicting these tokens causes attention score explosion and quality degradation.

    StreamingLLM preserves the first sink_tokens positions permanently in KV-cache while
    evicting middle tokens outside the sliding window. This enables bounded memory with
    preserved quality for arbitrary generation length.

    The mask implements: (kv_idx < sink_tokens) OR sliding_window_condition
    - Sink tokens (positions 0 to sink_tokens-1): ALWAYS attend (preserved in KV-cache)
    - Recent tokens (within window_size): Attend via sliding window
    - Middle tokens (between sink and window): Evicted from KV-cache, cannot attend

    This is critical for Tritter's 128K context window: enables streaming beyond 128K
    during inference while maintaining bounded KV-cache (~8-10GB on RTX 5080).

    Example:
        >>> mask_fn = streamingllm_mask(sink_tokens=4, window_size=40)
        >>> # Query at position 100
        >>> mask_fn(0, 0, q_idx=100, kv_idx=2)   # Sink token (pos 2 < 4)
        True
        >>> mask_fn(0, 0, q_idx=100, kv_idx=80)  # Recent token (within window)
        True
        >>> mask_fn(0, 0, q_idx=100, kv_idx=50)  # Middle token (evicted)
        False

    Note: This mask pattern requires corresponding KV-cache eviction logic in the
    attention implementation. The mask alone does not evict cache entries; it only
    defines which positions can be attended to.
    """

    def _streamingllm_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        """StreamingLLM mask combining attention sinks and sliding window.

        Returns:
            True if (1) kv_idx is a sink token OR (2) within sliding window.

        Why: Use boolean expressions instead of if statements because FlexAttention's
        vmap compilation doesn't support data-dependent control flow with tensors.
        """
        # Attend to sink tokens OR recent tokens in sliding window
        is_sink = kv_idx < sink_tokens
        is_in_window = (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)
        return is_sink | is_in_window

    return _streamingllm_mask


def prefix_lm_mask(prefix_length: int) -> Callable[[int, int, int, int], bool]:
    """Create prefix-LM mask for instruction tuning with bidirectional prefix.

    Args:
        prefix_length: Length of the bidirectional prefix region (e.g., instruction
                      context). Tokens in positions [0, prefix_length) use bidirectional
                      attention. Tokens in positions [prefix_length, seq_len) use causal
                      attention. Must be > 0 and < sequence_length.

    Returns:
        Mask function implementing prefix-LM: bidirectional attention on prefix,
        causal attention on suffix (response).

    Why: Prefix-LM is the optimal attention pattern for instruction tuning. Standard
    causal attention forces the model to process the instruction/context left-to-right,
    which is inefficient because the instruction is given in full before generation
    starts. Bidirectional attention on the prefix allows the model to build a richer
    representation of the instruction before generating the response.

    The pattern enables:
    - **Efficient instruction encoding**: Bidirectional attention on the prefix
      (instruction, context, examples) allows full context integration without the
      left-to-right constraint of causal attention.
    - **Autoregressive generation**: Causal attention on the suffix (response) maintains
      proper autoregressive generation where each token only sees previous response tokens.

    This is the standard approach for instruction-tuned models (T5, UL2, Flan) and
    enables better sample efficiency during instruction tuning compared to pure causal.

    The mask logic:
    - Both q_idx and kv_idx < prefix_length: Bidirectional (True)
    - q_idx >= prefix_length: Causal attention (kv_idx <= q_idx)
    - This ensures the response can attend to the full prefix (cross-attention style)
      plus previous response tokens (causal style)

    Example:
        >>> mask_fn = prefix_lm_mask(prefix_length=10)
        >>> # Prefix region (bidirectional)
        >>> mask_fn(0, 0, q_idx=5, kv_idx=8)   # Both in prefix: attend (bidirectional)
        True
        >>> mask_fn(0, 0, q_idx=8, kv_idx=5)   # Both in prefix: attend (bidirectional)
        True
        >>> # Suffix region (causal)
        >>> mask_fn(0, 0, q_idx=15, kv_idx=12) # Response to past response: attend
        True
        >>> mask_fn(0, 0, q_idx=12, kv_idx=15) # Response to future: mask
        False
        >>> # Cross-region (response can attend to prefix)
        >>> mask_fn(0, 0, q_idx=15, kv_idx=5)  # Response to prefix: attend
        True

    Note: prefix_length should be determined from the tokenized instruction. For
    instruction tuning datasets, this is typically the length of the prompt before
    the response starts. Variable-length prefixes require per-sample prefix_length,
    which can be handled by creating per-batch masks.
    """

    def _prefix_lm_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        """Prefix-LM mask with bidirectional prefix and causal suffix.

        Returns:
            True if:
            - Both positions in prefix region (bidirectional), OR
            - Query in suffix and causal constraint satisfied
        """
        # Use boolean expressions instead of if statements for vmap compatibility
        # Why: FlexAttention's vmap compilation doesn't support data-dependent control flow.

        # Bidirectional attention within prefix region
        both_in_prefix = (q_idx < prefix_length) & (kv_idx < prefix_length)

        # Causal attention for suffix: can attend to all past tokens (prefix + suffix)
        suffix_causal = (q_idx >= prefix_length) & (kv_idx <= q_idx)

        # Combine: attend if EITHER (both in prefix) OR (suffix with causal)
        return both_in_prefix | suffix_causal

    return _prefix_lm_mask


def and_masks(*masks: Callable[[int, int, int, int], bool]) -> Callable[[int, int, int, int], bool]:
    """Compose multiple mask functions with logical AND.

    Args:
        *masks: Variable number of mask functions with signature (b, h, q_idx, kv_idx) -> bool.
               Each mask function should return True if attention is allowed.

    Returns:
        Composite mask function that returns True only if ALL input masks return True.
        Returns True (allow attention) if no masks provided (identity element for AND).

    Why: Complex attention patterns often require combining multiple constraints. For example:
    - Causal + sliding window: Autoregressive generation with bounded context
    - Causal + document boundaries: Packed training with proper causality
    - Sliding window + document + sinks: Full StreamingLLM with document packing

    Composing masks with AND enables building complex patterns from simple primitives
    without implementing every combination explicitly. This follows the compositional
    design pattern of FlexAttention where simple masks are building blocks for complex
    attention patterns.

    The AND composition means attention is allowed only when ALL constraints are satisfied:
    - If any mask returns False (mask out), the result is False
    - Only if all masks return True is the result True

    This is the correct semantics for combining attention constraints: we want to enforce
    ALL constraints simultaneously (causal AND within-window AND same-document).

    Example:
        >>> # Combine causal mask with sliding window
        >>> window_mask = sliding_window_mask(window_size=40)
        >>> combined = and_masks(causal_mask, window_mask)
        >>> # Must satisfy both causal and window constraints
        >>> combined(0, 0, q_idx=100, kv_idx=80)  # Causal OK, within window OK
        True
        >>> combined(0, 0, q_idx=100, kv_idx=105) # Violates causal
        False
        >>> combined(0, 0, q_idx=100, kv_idx=50)  # Causal OK, outside window
        False

        >>> # Combine causal + sliding window + document boundaries
        >>> doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
        >>> doc_mask = document_mask(doc_ids)
        >>> full_mask = and_masks(causal_mask, window_mask, doc_mask)
        >>> # Must satisfy all three constraints

    Note: The mask functions are evaluated in order, but short-circuit evaluation is not
    guaranteed (depends on FlexAttention compilation). For efficiency, place the most
    restrictive (likely to return False) masks first.
    """

    def _and_masks(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        """Logical AND composition of all input masks.

        Returns:
            True if all masks return True, False if any mask returns False.

        Why: We manually chain AND operations instead of using Python's all()
        because FlexAttention's vmap compilation doesn't support all() on tensors.
        We also can't use 'if not masks' check because it's data-dependent control flow.
        """
        # Manually AND all masks (can't use all() or if statements with vmap)
        # Identity element: empty AND is True
        result = True
        for mask in masks:
            result = result & mask(b, h, q_idx, kv_idx)
        return result

    return _and_masks


__all__ = [
    "causal_mask",
    "sliding_window_mask",
    "document_mask",
    "streamingllm_mask",
    "prefix_lm_mask",
    "and_masks",
]
