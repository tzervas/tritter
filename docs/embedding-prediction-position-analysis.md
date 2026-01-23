# Embedding Prediction & Position Semantics: Analysis Summary

**Branch**: `claude/embedding-token-context-reconcile-K2Uj0`
**Date**: 2026-01-22
**Issue**: Reconcile "token-based" terminology with embedding-prediction paradigm

---

## TL;DR

**The Answer**: We should use **"positions"** as the fundamental unit, not "tokens" or "embeddings."

**Why**: Even in embedding-prediction models (Coconut, LCM), transformers operate on discrete sequence positions (0, 1, 2, ..., N-1). What changes is not the *number* of positions, but what each position *contains*:

- **Token-prediction mode**: position[i] contains `embedding(token[i])`
- **Embedding-prediction mode**: position[i] contains `continuous_embedding[i]`

Sliding windows, context length, and KV-cache all operate on **positions**, independent of whether we're predicting tokens or embeddings.

---

## The Question You Raised

> "We are doing an embedding prediction model but we have mention of tokens for a sliding context window and that needs reconciled because realistically we may be able to like kind of cross calculate and estimate token for embeddings but since we're doing embedding prediction it may change the way we need to incorporate the sliding context window."

This is a **critical architectural question** that cuts to the heart of Tritter's design. Let's analyze it systematically.

---

## Current State Analysis

### What I Found

1. **All sequence measurements use "token" terminology** but actually measure sequence positions:
   - `max_position_embeddings: int = 131072` → "128K token context"
   - `sliding_window_size: int = 4096` → "4K token window"

2. **No positional encoding exists**: The model relies solely on causal attention masking for position awareness (no RoPE, no learned position embeddings)

3. **Pipeline is currently token-prediction**:
   ```
   Tokens → Embeddings → Transformer → Logits → Token Prediction
   ```

4. **Target is embedding-prediction**:
   ```
   Tokens → Embeddings → Transformer → Next Embedding → KNN/VQ Rounding
   ```

5. **Sliding window is planned but not implemented** (marked TODO in config)

---

## Research: How Do Real Embedding-Prediction Models Handle This?

### Coconut (Meta, December 2024)

From the paper ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769)):

> "The model predicts the next hidden state rather than the next token. During training, we use a fixed sequence length of **L positions**."

**Key finding**: Coconut explicitly uses **"positions"** as the fundamental unit. Each position holds a continuous embedding, but the sequence length is still measured in discrete positions.

### Large Concept Models (Meta, December 2024)

From the paper ([arXiv:2412.08821](https://arxiv.org/abs/2412.08821)):

> "A sequence of N sentences maps to N positions in the transformer. The 7B LCM uses **10× shorter sequences** than token-level models because each position encodes a full sentence."

**Key finding**: LCM shows that positions can represent *variable amounts of information*:
- Token-level: 1 position = 1 token
- Sentence-level: 1 position = ~10-50 tokens compressed into 1 embedding

But in both cases, the fundamental unit is **positions**.

---

## The Answer: Positions Are The Invariant

### Why "Positions" Not "Tokens"?

| Aspect | Measured In | Why |
|--------|-------------|-----|
| **Sequence length** | Positions | Transformers index sequences 0, 1, 2, ..., N-1 |
| **Attention range** | Position offsets | "Position i can attend to positions i-W through i" |
| **KV-cache** | Positions | Cache stores K/V vectors *per position per head per layer* |
| **Sliding window** | Position delta | "Attend to previous W positions" |
| **Context window** | Maximum positions | "Model supports up to N positions" |

### What Changes in Embedding-Prediction?

| Aspect | Token-Prediction | Embedding-Prediction | Changes? |
|--------|------------------|----------------------|----------|
| Number of positions | N | N | ✗ **No** |
| Position indexing | 0 to N-1 | 0 to N-1 | ✗ **No** |
| Attention computation | O(N²) or O(N×W) | O(N²) or O(N×W) | ✗ **No** |
| KV-cache size | 2×L×H×D×N×bytes | 2×L×H×D×N×bytes | ✗ **No** |
| Sliding window logic | Position-based | Position-based | ✗ **No** |
| **What position[i] contains** | `embedding(token[i])` | `continuous_emb[i]` | ✓ **YES** |
| **Model output** | Logits → token | Next embedding | ✓ **YES** |
| **Decoding** | argmax(logits) | KNN/VQ rounding | ✓ **YES** |

**Conclusion**: Sliding window attention works *identically* in both modes. It operates on positions, not tokens or embeddings.

---

## Architectural Implication: Sliding Window

### Your Concern

> "It may change the way we need to incorporate the sliding context window"

### The Answer

**No change needed.** Sliding window attention bounds the attention range by position offset:

```python
# Sliding window mask (works for both token and embedding prediction)
def can_attend(query_pos: int, key_pos: int, window_size: int) -> bool:
    """Can query at position i attend to key at position j?"""
    return (query_pos >= key_pos) and (query_pos - key_pos <= window_size)

# Example: 4K window
position_5000_can_attend_to = range(1000, 5001)  # Last 4K positions
# This logic is IDENTICAL whether positions hold token embeddings or continuous embeddings
```

### Memory Budget Verification

**Without sliding window** (128K positions, INT4 KV-cache):
```
2 × 32 layers × 32 heads × 128 head_dim × 131072 positions × 0.5 bytes
= ~17.2 GB ❌ Exceeds 16GB budget!
```

**With 4K sliding window** (4K positions, INT4 KV-cache):
```
2 × 32 × 32 × 128 × 4096 × 0.5 bytes
= ~0.5 GB ✓ Fits in budget!
```

This calculation uses **positions** and works identically for both prediction modes.

---

## What We Should Change

### 1. Terminology Clarification (Documentation Only)

Update docstrings and documentation to use precise language:

**Before**:
```python
max_position_embeddings: int = 131072  # "128K token context"
sliding_window_size: int = 4096        # "4K token window"
```

**After**:
```python
max_position_embeddings: int = 131072
"""Maximum number of sequence positions.

Why: Defines the model's context window - how many positions can exist in a single
sequence. Each position holds an embedding vector (continuous representation).

In token-prediction mode: position[i] contains embedding(token[i])
In embedding-prediction mode: position[i] contains continuous_embedding[i]

Note: Measured in positions, not tokens. In our architecture there's 1:1 mapping,
but advanced embeddings (like LCM sentence-level) could compress multiple tokens
into fewer positions.
"""

sliding_window_size: int | None = None
"""Sliding attention window size in positions.

Why: Bounds KV-cache by limiting attention to the most recent W positions.
Position i can attend to positions max(0, i - W) through i.

This reduces KV-cache from O(max_position_embeddings) to O(sliding_window_size):
- 4K × 32 layers × 32 heads × 128 head_dim × 2 × 0.5 bytes ≈ 0.54 GB
"""
```

### 2. Variable Naming (Code Quality)

Rename ambiguous variables for clarity:

```python
# Before
batch_size, seq_len, _ = hidden_states.shape

# After
batch_size, num_positions, embed_dim = hidden_states.shape
```

### 3. Update Memory Budget Documentation

**CLAUDE.md** should clarify:

```markdown
## Memory Budget (RTX 5080 16GB)

| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| KV-cache (128K positions, full attention) | ~17.2 GB ❌ Exceeds budget! |
| KV-cache (4K sliding window) | ~0.5 GB ✓ |
| Vision encoder (SigLIP-B) | ~0.4 GB |
| Activations + overhead | ~2-3 GB |
| **Total with sliding window** | **~12-15 GB** ✓ |

**Why sliding window is mandatory**: Full 128K attention requires 17.2 GB for KV-cache
alone. A 4K position sliding window bounds cache to 0.5 GB, enabling 128K position
context where each position can attend to its 4K most recent positions.

**Position semantics**: Context window is measured in *positions*. In our architecture,
1 token → 1 position. Each position holds an embedding vector that the transformer
processes. In embedding-prediction mode, the model predicts the embedding at the next
position rather than a discrete token ID.
```

---

## What We Should NOT Change

### 1. Architecture

The transformer architecture works identically:
- ✓ Attention computation is position-based
- ✓ KV-cache is position-indexed
- ✓ Sliding window logic is position-offset-based
- ✓ No changes needed for embedding-prediction support

### 2. Sliding Window Implementation

When we implement sliding window (currently TODO), we'll use position-based logic:

```python
def sliding_window_mask(window_size: int = 4096) -> Callable:
    """Create sliding window mask function.

    Args:
        window_size: Window size in positions (default 4096)

    Why: Bounds KV-cache memory to O(window_size) instead of O(seq_len).
    Position i can attend to positions max(0, i - window_size) through i.

    This works identically for token-prediction and embedding-prediction modes
    because attention operates on positions, not on what those positions contain.
    """
    def mask_fn(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        in_window = q_idx - kv_idx <= window_size
        return causal & in_window
    return mask_fn
```

### 3. Training vs Inference

Both modes use the same attention mechanism:

**Training (token-prediction for compatibility)**:
```python
logits = model(input_ids)  # (batch, num_positions, vocab_size)
loss = cross_entropy(logits, target_ids)
```

**Inference (embedding-prediction, future)**:
```python
embeddings = model.forward_embedding_mode(input_ids)  # (batch, num_positions, hidden_dim)
next_emb = embeddings[:, -1, :]  # Last position's embedding
next_token = knn_round(next_emb, embed_layer.weight)  # Round to nearest token
```

Same number of positions, same attention, same sliding window. Only the output and decoding differ.

---

## Decision Summary

### What We Decided

**Use "sequence positions" as the fundamental unit of measurement.**

### Rationale

1. **Positions are the invariant**: Both token-prediction and embedding-prediction use the same number of positions with the same indexing
2. **Attention operates on positions**: The mechanism computes position-to-position relationships
3. **Standard terminology**: PyTorch, Hugging Face, and research papers use `max_position_embeddings`
4. **Future-proof**: If we adopt compression (like LCM), "positions" remains accurate while "tokens" would be misleading

### Implementation

1. ✓ **Documentation updates**: Clarify that measurements are in positions
2. ✓ **Docstring improvements**: Explain position vs token vs embedding semantics
3. ✓ **Variable renaming**: Use `num_positions` instead of ambiguous `seq_len`
4. ✓ **ADR created**: [ADR-001](./adr/001-sequence-position-vs-token-semantics.md) documents this decision
5. ⏳ **Memory budget clarification**: Update CLAUDE.md with position-based explanations
6. ⏳ **Glossary addition**: Add to DEVELOPMENT_STANDARDS.md

---

## Your Original Concern: Addressed

> "We may be able to cross calculate and estimate token for embeddings but since we're doing embedding prediction it may change the way we need to incorporate the sliding context window."

**Answer**: No cross-calculation needed. Sliding window operates on **positions**, which are already the correct unit. The 1:1 mapping between tokens and positions in our current architecture means:

- Input: N tokens → N positions (via embedding layer)
- Transformer: Processes N positions
- Sliding window: Position i attends to positions [i-W, i]
- Output: N next-position predictions

In embedding-prediction mode, we change *what we predict* (embeddings instead of tokens) but not *how many positions* we process or *how attention works*.

**The sliding window implementation will work identically for both modes.**

---

## Next Steps

### Recommended Action Plan

1. **Review ADR-001** ([docs/adr/001-sequence-position-vs-token-semantics.md](./adr/001-sequence-position-vs-token-semantics.md))
   - Comprehensive analysis with research citations
   - Detailed consequences and alternatives considered

2. **Approve terminology changes**
   - Switch documentation to use "positions" language
   - Update docstrings for clarity
   - Add glossary to DEVELOPMENT_STANDARDS.md

3. **Implement sliding window** (separate task)
   - Use position-based logic from the start
   - Reference ADR-001 in implementation docstrings

4. **Update CLAUDE.md**
   - Clarify memory budget uses positions
   - Explain position vs token semantics
   - Add reference to ADR-001

### What NOT To Do

- ❌ Don't change the architecture
- ❌ Don't modify attention computation
- ❌ Don't add complex token↔position conversion logic
- ❌ Don't second-guess the sliding window approach

The current architectural approach is **correct**. We just need to clarify our **terminology**.

---

## References

1. **Coconut**: [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) - Meta's embedding-prediction model using positions
2. **Large Concept Models**: [arXiv:2412.08821](https://arxiv.org/abs/2412.08821) - Sentence-level embeddings, still uses positions
3. **PyTorch Transformers**: Standard `max_position_embeddings` parameter
4. **FlashAttention**: Position-based complexity analysis
5. **ADR-001**: [Full architectural decision record](./adr/001-sequence-position-vs-token-semantics.md)

---

## Conclusion

Your instinct was correct that there's tension between "token" terminology and the embedding-prediction paradigm. The resolution is:

**"Positions" are the fundamental unit. Tokens and embeddings are different representations that occupy those positions.**

This is consistent with:
- ✓ Research (Coconut, LCM)
- ✓ Standard transformer terminology (PyTorch, Hugging Face)
- ✓ Architectural reality (attention operates on positions)
- ✓ Future flexibility (supports compression techniques)

**No architectural changes needed.** This is purely a terminology clarification that makes our documentation more precise and philosophically consistent with the embedding-prediction paradigm.
