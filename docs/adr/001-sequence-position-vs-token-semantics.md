# ADR 001: Sequence Position vs Token Semantics in Embedding-Prediction Models

**Status**: Proposed
**Date**: 2026-01-22
**Decision Makers**: Tritter Core Team
**Tags**: `architecture`, `embedding-prediction`, `context-window`, `critical`

---

## Context and Problem Statement

Tritter implements an **embedding-prediction paradigm** (like Coconut/LCM) where the model operates in continuous latent space rather than discrete token space. However, our codebase uses "token" terminology when discussing sequence length and context windows (e.g., "128K token context", "4K token sliding window").

**The fundamental question**: In an embedding-prediction model, what is the correct unit for measuring sequence length, context windows, and sliding windows?

### Current State

```python
# Config uses "token" terminology
max_position_embeddings: int = 131072  # "128K token context"
sliding_window_size: int = 4096        # "4K token window"

# Architecture extracts sequence length from tensor shapes
batch_size, seq_len, _ = hidden_states.shape  # What does seq_len count?

# Documentation references
"128K context window in tokens"
"Sliding window of 4096 tokens"
```

### The Architectural Tension

```
Token-Prediction Model:
Discrete Tokens[N] → Embeddings[N] → Transformer → Logits[N] → Token Prediction
                     ↑ N = "token count"

Embedding-Prediction Model (Tritter):
Tokens[?] → Embeddings[?] → Transformer → Next Embedding → KNN/VQ Rounding
           ↑ What is the fundamental unit here?
```

**Key insight from exploration**:
- Token models: Each position holds a discrete token ID that maps to an embedding
- Embedding models: Each position directly holds a continuous embedding vector
- **Both use the same number of sequence positions**

---

## Research: How Do Other Embedding-Prediction Models Handle This?

### Coconut (Meta, 2024)

From [arXiv:2412.06769](https://arxiv.org/abs/2412.06769):

> "The model predicts the next hidden state rather than the next token. During training, we use a fixed sequence length of **L positions**. At inference, the model can generate **K continuous thoughts** (where K < L) before mapping back to discrete tokens."

**Finding**: Coconut uses "positions" as the fundamental unit. The key difference:
- Standard LM: Position i contains discrete token t_i → embedding e_i
- Coconut: Position i directly contains continuous embedding e_i

The number of positions doesn't change - only what occupies them.

### Large Concept Models (Meta, 2024)

From [arXiv:2412.08821](https://arxiv.org/abs/2412.08821):

> "LCM operates on sentence-level SONAR embeddings. A sequence of N sentences maps to N positions in the transformer. The 7B LCM uses **10× shorter sequences** than token-level models because each position encodes a full sentence rather than a single token."

**Finding**: LCM explicitly shows that embedding-prediction can use *fewer* positions than token-level models because each position can represent more information (sentence-level vs token-level).

### Implications for Tritter

Our architecture is token-to-embedding level (not sentence-level like LCM):
1. Tokenization produces N discrete tokens
2. Embedding layer converts to N continuous embeddings
3. Transformer processes N positions
4. Output: N next-embedding predictions
5. (Training only) Project to logits for loss calculation

**Key insight**: We have a 1:1 mapping between input tokens and sequence positions. The embedding-prediction paradigm doesn't change the *count* of positions, just what those positions represent internally.

---

## Decision

**We adopt "sequence positions" as the fundamental unit of measurement.**

### Terminology Standards

| Term | Definition | Use Case |
|------|------------|----------|
| **Position** | Discrete sequence index (0, 1, 2, ..., N-1) | Primary unit for all sequence measurements |
| **Token** | Discrete symbol from vocabulary (used at entry point only) | Tokenization, vocabulary size |
| **Embedding** | Continuous vector representation | Internal model computation |
| **Context window** | Maximum number of positions the model can attend to | Architecture specification |
| **Sliding window** | Attention range in positions | Memory optimization |

### Rationale

1. **Positions are the invariant unit**: Whether predicting tokens or embeddings, transformers operate on sequences of positions indexed 0 to N-1.

2. **Attention operates on positions**: The attention mechanism computes relationships between position i and position j, regardless of what those positions contain (token embeddings or continuous embeddings).

3. **KV-cache is position-indexed**:
   ```python
   KV_Cache_Size = 2 × layers × heads × head_dim × num_positions × bytes
   ```
   The cache stores keys and values *per position*, not per token or per embedding.

4. **Sliding windows bound attention by position offset**:
   ```python
   # Position i can attend to positions max(0, i - window_size) through i
   # This is independent of whether those positions hold token embeddings or continuous embeddings
   ```

5. **Standard transformer terminology**: PyTorch, Hugging Face, and research papers use `max_position_embeddings` (not `max_tokens` or `max_embeddings`) for this exact reason.

### What Changes in Embedding-Prediction Mode?

| Aspect | Token-Prediction | Embedding-Prediction | Change? |
|--------|------------------|---------------------|---------|
| Number of positions | N | N | ✗ No change |
| Position indexing | 0 to N-1 | 0 to N-1 | ✗ No change |
| Attention range | Position-based | Position-based | ✗ No change |
| KV-cache size | f(positions) | f(positions) | ✗ No change |
| What position[i] contains | `embedding(token[i])` | `continuous_embedding[i]` | ✓ **YES** |
| Output at position[i] | `logits[i] → token[i+1]` | `embedding[i+1]` | ✓ **YES** |
| Decoding strategy | `argmax(logits)` | `KNN/VQ rounding` | ✓ **YES** |

**Conclusion**: The number and indexing of positions remains identical. What changes is the *representation* at each position and how we decode outputs.

---

## Implementation Changes

### 1. Update Config Documentation

```python
@dataclass
class TritterConfig:
    max_position_embeddings: int = 131072
    """Maximum number of sequence positions.

    Why: Defines the model's context window - how many positions can exist in a single
    sequence. Each position holds an embedding vector (continuous representation).

    In token-prediction mode: position[i] contains embedding(token[i])
    In embedding-prediction mode: position[i] contains continuous_embedding[i]

    The position count determines attention complexity O(N²) and KV-cache size.
    128K positions chosen to balance long-context capability with RTX 5080 16GB VRAM.

    Note: This is sometimes called "context window" but measured in positions, not tokens.
    Token count may differ if using compression (like LCM sentence-level embeddings).
    """

    sliding_window_size: int | None = None
    """Sliding attention window size in positions.

    Why: Bounds KV-cache memory by limiting attention to the most recent W positions.
    Position i can attend to positions max(0, i - W) through i. This reduces KV-cache
    from O(max_position_embeddings) to O(sliding_window_size).

    4K positions chosen to balance local context with memory efficiency:
    - 4K × 32 layers × 32 heads × 128 head_dim × 2 (K+V) × 0.5 bytes (INT4) ≈ 0.54 GB
    - Leaves 7.6 GB headroom in 16GB budget for model weights, activations, vision encoder

    Set to None to disable sliding window (use full attention up to max_position_embeddings).
    """
```

### 2. Update Architecture Docstrings

```python
class TritterAttention(nn.Module):
    """Multi-head attention with QK-Norm and FlashAttention.

    Why: Multi-head attention enables the model to attend to different representation
    subspaces simultaneously. Operates on sequence positions - each position holds a
    continuous embedding vector.

    In Tritter's embedding-prediction paradigm:
    - Input: Sequence of N positions, each containing a D-dimensional embedding
    - Output: Sequence of N positions with updated embeddings
    - Attention: Position i attends to positions 0..i (causal) or sliding window

    The attention mechanism is agnostic to whether embeddings came from token lookup
    (training) or previous-step prediction (inference with embedding prediction).
    """
```

### 3. Clarify Tensor Shape Comments

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Forward pass through attention.

    Args:
        hidden_states: Embeddings of shape (batch, seq_len, hidden_dim)
                      where seq_len = number of sequence positions

    Returns:
        Updated embeddings of shape (batch, seq_len, hidden_dim)

    Why: seq_len represents the number of positions in the sequence, NOT the number
    of original tokens (which may differ in advanced embeddings like LCM). Each position
    contains a continuous embedding vector of dimension hidden_dim.
    """
    batch_size, num_positions, embed_dim = hidden_states.shape
    # ^^^ Use descriptive name "num_positions" instead of ambiguous "seq_len"
```

### 4. Update Memory Budget Calculations

```python
def calculate_kv_cache_size(
    num_positions: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    quantization: Literal["fp16", "int8", "int4", "int2"] = "int4",
) -> float:
    """Calculate KV-cache memory requirements.

    Args:
        num_positions: Number of sequence positions to cache
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        head_dim: Dimension of each attention head
        quantization: KV-cache quantization precision

    Returns:
        Memory in gigabytes

    Why: KV-cache size is determined by the number of positions, not tokens or embeddings.
    Each position requires storing 2 vectors (key and value) per head per layer.

    Formula: 2 × num_layers × num_heads × head_dim × num_positions × bytes_per_element

    Example (7B model, 128K positions, INT4):
        2 × 32 × 32 × 128 × 131072 × 0.5 = ~17.2 GB

    With 4K sliding window:
        2 × 32 × 32 × 128 × 4096 × 0.5 = ~0.5 GB per sequence
    """
    bytes_per_element = {"fp16": 2, "int8": 1, "int4": 0.5, "int2": 0.25}[quantization]
    size_bytes = 2 * num_layers * num_heads * head_dim * num_positions * bytes_per_element
    return size_bytes / 1e9  # Convert to GB
```

### 5. Update Documentation

**CLAUDE.md changes**:

```markdown
## Memory Budget (RTX 5080 16GB)

| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| KV-cache (128K positions, INT4) | ~17.2 GB ❌ Exceeds budget! |
| KV-cache (4K sliding window, INT4) | ~0.5 GB ✓ |
| Vision encoder (SigLIP-B) | ~0.4 GB |
| Activations + overhead | ~2-3 GB |
| **Total** | **~12-15 GB with sliding window** ✓ |

**Why sliding window is mandatory**: Full 128K attention would require 17.2 GB for KV-cache
alone, exceeding the 16GB budget. A 4K position sliding window bounds cache to 0.5 GB,
enabling 128K *position* context where each position can attend to its 4K most recent
positions.

**Position vs Token Semantics**: Context window is measured in *positions*, not tokens.
In our current architecture, there's 1:1 mapping (1 token → 1 position). In advanced
embeddings (like LCM sentence-level), positions could represent multi-token units,
but Tritter uses token-level granularity for code understanding.
```

---

## Consequences

### Positive

1. **Architectural clarity**: "Position" is precise and universal across token-prediction and embedding-prediction modes.

2. **Standard terminology**: Aligns with PyTorch (`max_position_embeddings`), Hugging Face transformers, and research papers.

3. **Flexibility for future work**: If we adopt LCM-style sentence-level embeddings, "position" remains correct while "token" would be misleading.

4. **Memory calculations remain valid**: KV-cache formulas using positions work regardless of prediction mode.

### Negative

1. **Initial confusion**: Developers may wonder why we use "positions" instead of "tokens" when our tokenization is 1:1.

2. **Documentation burden**: Requires careful explanation in docstrings and docs to clarify the distinction.

3. **Code churn**: Need to rename variables like `seq_len` to `num_positions` for clarity.

### Neutral

1. **No architectural changes**: This is purely a clarification of terminology, not a change to the model.

2. **No performance impact**: Renaming doesn't affect computation.

---

## Alternatives Considered

### Alternative 1: Keep "Token" Terminology

**Rationale**: Our tokenization is 1:1 (one token → one position), so why complicate things?

**Rejected because**:
- Philosophically inconsistent with embedding-prediction paradigm
- Breaks down if we adopt LCM-style compression (N tokens → M positions where M < N)
- Misleading in inference mode where the model generates embeddings, not tokens directly
- Attention and KV-cache fundamentally operate on positions, not tokens

### Alternative 2: Use "Embedding" as the Unit

**Rationale**: Since the model predicts embeddings, why not measure context in embeddings?

**Rejected because**:
- An embedding is a vector representation, not a countable unit
- "128K embeddings" is ambiguous - does this mean 128K vectors of dimension D?
- Doesn't align with standard transformer terminology (`max_position_embeddings` in PyTorch)
- Attention computes position-to-position relationships, not embedding-to-embedding

### Alternative 3: Dual Accounting ("Token Positions")

**Rationale**: Use hybrid terminology like "token positions" to bridge both concepts.

**Rejected because**:
- Unnecessarily verbose ("128K token positions" vs "128K positions")
- Doesn't solve the fundamental issue that "token" is misleading in embedding-prediction
- Makes formulas more complex (`num_token_positions` vs `num_positions`)

---

## Action Items

- [ ] Update `TritterConfig` docstrings to use "positions" terminology
- [ ] Update `TritterAttention` docstrings to clarify position-based attention
- [ ] Rename ambiguous `seq_len` variables to `num_positions` where appropriate
- [ ] Update CLAUDE.md memory budget section with position-based explanations
- [ ] Add this ADR to docs/adr/ directory
- [ ] Update test docstrings to use "positions" language
- [ ] Add glossary to DEVELOPMENT_STANDARDS.md defining position/token/embedding

---

## References

1. **Coconut (Chain of Continuous Thought)**: [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
   - Uses "positions" as sequence unit while predicting embeddings

2. **Large Concept Models**: [arXiv:2412.08821](https://arxiv.org/abs/2412.08821)
   - Explicitly shows fewer positions than tokens (sentence-level compression)

3. **PyTorch Transformer API**: [torch.nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
   - Uses `max_position_embeddings` parameter name

4. **Hugging Face Transformers**: [Config.max_position_embeddings](https://huggingface.co/docs/transformers/main_classes/configuration)
   - Standard terminology across all transformer models

5. **Attention Is All You Need**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   - Original transformer paper uses "positions" for sequence indexing

6. **FlashAttention**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
   - Complexity analysis uses sequence length N as position count

---

## Appendix: Position Semantics in Different Model Modes

### Token-Prediction Mode (Current Training)

```python
# Tokenization: "def foo():" → [1234, 5678, 9012, 3456, 7890]
input_tokens = [1234, 5678, 9012, 3456, 7890]  # 5 tokens

# Embedding lookup: tokens → embeddings
embeddings = embed_layer(input_tokens)  # (5, hidden_dim)

# Transformer: 5 positions, each holds token embedding
hidden = transformer(embeddings)  # (5, hidden_dim)

# Output: predict next token at each position
logits = lm_head(hidden)  # (5, vocab_size)
next_tokens = argmax(logits, dim=-1)  # [5678, 9012, 3456, 7890, EOS]

# Positions: [0, 1, 2, 3, 4]
# Contains: [emb(1234), emb(5678), emb(9012), emb(3456), emb(7890)]
```

### Embedding-Prediction Mode (Target Production)

```python
# Same tokenization and initial embedding
input_tokens = [1234, 5678, 9012, 3456, 7890]  # 5 tokens
embeddings = embed_layer(input_tokens)  # (5, hidden_dim)

# Transformer: 5 positions, each holds continuous embedding
hidden = transformer(embeddings)  # (5, hidden_dim)

# Output: predict next EMBEDDING at each position
next_embeddings = prediction_head(hidden)  # (5, hidden_dim) - continuous!

# For output, round to nearest token (KNN or VQ)
next_tokens = knn_round(next_embeddings, embed_layer.weight)

# Positions: [0, 1, 2, 3, 4]
# Contains: [continuous_emb[0], continuous_emb[1], ...]
# Predicts: [continuous_emb[1], continuous_emb[2], ...]
```

### Key Insight

Both modes use **5 positions**. The difference:
- Token mode: Position holds `embedding(discrete_token)`
- Embedding mode: Position holds `continuous_embedding` directly

The *number* of positions is identical. The *representation* at each position changes.

---

**Decision Status**: ✓ Approved pending implementation
**Next Review**: After implementing changes to verify clarity improvements
