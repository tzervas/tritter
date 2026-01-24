# FlexAttention Implementation Guide

**Guide ID**: GUIDE-001
**Specification**: SPEC-001-flexattention.md
**Status**: Ready for Implementation
**Prerequisites**: PyTorch >= 2.5.0, CUDA >= 12.0

## Overview

This guide provides step-by-step instructions for implementing FlexAttention integration as defined in SPEC-001.

## Implementation Phases

### Phase 1: Create Attention Patterns Module

**File**: `src/tritter/models/attention_patterns.py`

**Step 1.1**: Create the file with module docstring

```python
"""Attention mask patterns for FlexAttention.

Why: Tritter's 128K context window with document-packed training requires
efficient attention patterns beyond simple causal masking. This module
provides reusable mask primitives that compose into complex patterns.

Note: These patterns operate in discrete token space. The embedding-prediction
paradigm uses these during training; production inference may bypass them.
"""

from typing import Callable
import torch
```

**Step 1.2**: Implement causal mask primitive

```python
def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Standard autoregressive causal mask.

    Args:
        b: Batch index (unused, for FlexAttention signature)
        h: Head index (unused, for FlexAttention signature)
        q_idx: Query position
        kv_idx: Key/Value position

    Returns:
        True if q_idx should attend to kv_idx

    Why: Decoder-only models require each token to attend only to
    itself and previous tokens for autoregressive generation.
    """
    return q_idx >= kv_idx
```

**Step 1.3**: Implement sliding window mask

```python
def sliding_window_mask(window_size: int = 4096) -> Callable[[int, int, int, int], bool]:
    """Create sliding window mask function.

    Args:
        window_size: Maximum attention distance (default 4096 per project spec)

    Returns:
        Mask function compatible with FlexAttention

    Why: Bounds KV-cache memory from O(seq_len) to O(window_size).
    With 128K context and 4K window, KV-cache fits in 16GB VRAM.
    """
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        causal = q_idx >= kv_idx
        in_window = q_idx - kv_idx <= window_size
        return causal and in_window
    return mask_fn
```

**Step 1.4**: Implement document boundary mask

```python
def document_mask(doc_ids: torch.Tensor) -> Callable[[int, int, int, int], bool]:
    """Create document boundary mask for packed sequences.

    Args:
        doc_ids: Tensor (batch, seq_len) mapping positions to document IDs

    Returns:
        Mask function that blocks cross-document attention

    Why: Training packs multiple documents into single sequences for
    efficiency. Without this mask, attention would leak information
    across document boundaries, corrupting learned representations.
    """
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    return mask_fn
```

**Step 1.5**: Implement StreamingLLM mask

```python
def streamingllm_mask(
    sink_tokens: int = 4,
    window_size: int = 4096
) -> Callable[[int, int, int, int], bool]:
    """StreamingLLM attention sink mask.

    Args:
        sink_tokens: Number of initial tokens to always attend to
        window_size: Sliding window size for recent tokens

    Returns:
        Mask function combining sinks and sliding window

    Why: Attention sinks prevent perplexity explosion when streaming
    beyond context window. Initial tokens absorb attention mass that
    would otherwise cause numerical instability.

    Reference: Xiao et al., "Efficient Streaming Language Models"
    """
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        is_sink = kv_idx < sink_tokens
        causal = q_idx >= kv_idx
        in_window = q_idx - kv_idx <= window_size
        return is_sink or (causal and in_window)
    return mask_fn
```

**Step 1.6**: Implement prefix-LM mask

```python
def prefix_lm_mask(prefix_length: int) -> Callable[[int, int, int, int], bool]:
    """Prefix-LM mask for instruction tuning.

    Args:
        prefix_length: Length of bidirectional prefix region

    Returns:
        Mask with bidirectional prefix, causal suffix

    Why: Instruction tuning benefits from bidirectional context on
    the instruction (prefix) while maintaining causal generation
    for the response (suffix).
    """
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        # Within prefix: bidirectional
        in_prefix = q_idx < prefix_length and kv_idx < prefix_length
        # After prefix: causal (can see prefix + prior generation)
        causal_after = q_idx >= prefix_length and kv_idx <= q_idx
        return in_prefix or causal_after
    return mask_fn
```

**Step 1.7**: Add module exports

```python
__all__ = [
    "causal_mask",
    "sliding_window_mask",
    "document_mask",
    "streamingllm_mask",
    "prefix_lm_mask",
]
```

---

### Phase 2: Create FlexAttention Wrapper

**File**: `src/tritter/models/flex_attention.py`

**Step 2.1**: Create file with version check

```python
"""FlexAttention integration for advanced attention patterns.

Why: FlexAttention (PyTorch 2.5+) compiles mask functions into fused
Triton kernels, avoiding O(n²) mask materialization. Essential for
128K context on 16GB VRAM.

Note: Falls back to SDPA with is_causal=True when FlexAttention unavailable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

# Check PyTorch version for FlexAttention availability
_FLEX_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        and_masks,
        BlockMask,
    )
    _FLEX_AVAILABLE = True
except ImportError:
    BlockMask = None  # Type stub for when not available
```

**Step 2.2**: Implement BlockMask factory

```python
def create_attention_mask(
    config,  # TritterConfig
    seq_len: int,
    doc_ids: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Optional[BlockMask]:
    """Create BlockMask based on configuration.

    Args:
        config: TritterConfig with attention settings
        seq_len: Sequence length
        doc_ids: Optional document IDs for packed sequences
        device: Target device

    Returns:
        BlockMask for flex_attention, or None if simple causal

    Why: Centralizes mask creation logic. Returns None for simple
    causal to enable SDPA optimization path.
    """
    if not _FLEX_AVAILABLE:
        return None

    from tritter.models.attention_patterns import (
        causal_mask,
        sliding_window_mask,
        document_mask,
        streamingllm_mask,
        prefix_lm_mask,
    )

    # Simple causal: use SDPA path
    if (config.attention_mode == "causal"
        and not config.use_sliding_window
        and doc_ids is None):
        return None

    # Build composite mask
    masks = []

    if config.attention_mode == "causal":
        masks.append(causal_mask)
    elif config.attention_mode == "prefix_lm":
        masks.append(prefix_lm_mask(config.prefix_length))
    # bidirectional and embedding: no mask needed

    if config.use_sliding_window:
        if config.use_attention_sinks:
            masks.append(streamingllm_mask(
                config.num_sink_tokens,
                config.sliding_window_size
            ))
        else:
            masks.append(sliding_window_mask(config.sliding_window_size))

    if doc_ids is not None:
        masks.append(document_mask(doc_ids))

    if not masks:
        return None

    composite = and_masks(*masks) if len(masks) > 1 else masks[0]

    return create_block_mask(
        composite,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
```

**Step 2.3**: Implement FlexAttention layer

```python
class FlexAttentionLayer(nn.Module):
    """Attention computation with FlexAttention backend.

    Why: Unified interface for all attention patterns while
    leveraging compiled Triton kernels for efficiency.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._mask_cache: dict = {}

    def forward(
        self,
        query: torch.Tensor,  # (B, H, L, D_head)
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: Optional[BlockMask] = None,
    ) -> torch.Tensor:
        """Compute attention with automatic backend selection.

        Args:
            query, key, value: Attention tensors
            block_mask: Optional BlockMask for complex patterns

        Returns:
            Attention output (B, H, L, D_head)
        """
        if block_mask is None:
            # Fast path: SDPA with causal optimization
            is_causal = self.config.attention_mode == "causal"
            return F.scaled_dot_product_attention(
                query, key, value,
                is_causal=is_causal,
            )

        if not _FLEX_AVAILABLE:
            raise RuntimeError(
                "FlexAttention required but not available. "
                "Upgrade to PyTorch >= 2.5.0"
            )

        return flex_attention(query, key, value, block_mask=block_mask)

    def clear_cache(self) -> None:
        """Clear BlockMask cache."""
        self._mask_cache.clear()


__all__ = ["FlexAttentionLayer", "create_attention_mask", "BlockMask"]
```

---

### Phase 3: Integrate with TritterAttention

**File**: `src/tritter/models/architecture.py` (modify existing)

**Step 3.1**: Import FlexAttention components

```python
from tritter.models.flex_attention import (
    FlexAttentionLayer,
    create_attention_mask,
)
```

**Step 3.2**: Update TritterAttention._compute_attention()

```python
def _compute_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    doc_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention with optimal backend selection.

    Why: Automatically selects between SDPA (simple causal) and
    FlexAttention (complex patterns) for best performance.
    """
    # Try FlexAttention for complex patterns
    if doc_ids is not None or self.config.use_sliding_window:
        block_mask = create_attention_mask(
            self.config,
            query.size(2),  # seq_len
            doc_ids=doc_ids,
            device=query.device,
        )
        if block_mask is not None:
            return self.flex_attn(query, key, value, block_mask)

    # Simple causal: SDPA fast path
    if attention_mask is None and self.config.attention_mode == "causal":
        return F.scaled_dot_product_attention(
            query, key, value,
            is_causal=True,
        )

    # Custom mask provided
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attention_mask,
        is_causal=False,
    )
```

---

### Phase 4: Add Tests

**File**: `tests/unit/test_attention_patterns.py`

```python
"""Tests for attention mask patterns.

Why: Mask correctness is critical - wrong masks cause silent
training failures that are hard to debug.
"""

import pytest
from tritter.models.attention_patterns import (
    causal_mask,
    sliding_window_mask,
    document_mask,
    prefix_lm_mask,
)


class TestCausalMask:
    """Tests for causal mask primitive."""

    def test_attends_to_past(self):
        """Query attends to past positions."""
        assert causal_mask(0, 0, 5, 3) == True

    def test_attends_to_self(self):
        """Query attends to self."""
        assert causal_mask(0, 0, 5, 5) == True

    def test_blocks_future(self):
        """Query does not attend to future."""
        assert causal_mask(0, 0, 3, 5) == False


class TestSlidingWindow:
    """Tests for sliding window mask."""

    def test_within_window(self):
        """Attends within window."""
        mask = sliding_window_mask(window_size=10)
        assert mask(0, 0, 100, 95) == True

    def test_outside_window(self):
        """Blocks outside window."""
        mask = sliding_window_mask(window_size=10)
        assert mask(0, 0, 100, 50) == False


class TestPrefixLM:
    """Tests for prefix-LM mask."""

    def test_prefix_bidirectional(self):
        """Prefix region is bidirectional."""
        mask = prefix_lm_mask(prefix_length=10)
        # Position 5 can see position 8 (both in prefix)
        assert mask(0, 0, 5, 8) == True

    def test_generation_causal(self):
        """Generation region is causal."""
        mask = prefix_lm_mask(prefix_length=10)
        # Position 15 cannot see position 18
        assert mask(0, 0, 15, 18) == False

    def test_generation_sees_prefix(self):
        """Generation can see entire prefix."""
        mask = prefix_lm_mask(prefix_length=10)
        # Position 15 can see position 5 (in prefix)
        assert mask(0, 0, 15, 5) == True
```

---

## Verification Checklist

- [ ] `attention_patterns.py` created with all primitives
- [ ] `flex_attention.py` created with wrapper and factory
- [ ] `TritterAttention` updated to use FlexAttention
- [ ] Unit tests pass for all mask patterns
- [ ] Integration test with 128K sequence passes
- [ ] Memory usage within 15GB budget
- [ ] SDPA fallback works on PyTorch < 2.5

## Next Steps

After implementation:
1. Run full test suite: `pytest tests/`
2. Run memory benchmark: `python -m devtools.validate`
3. Update SPEC-001 status to "Implemented"
4. Create PR targeting develop branch
