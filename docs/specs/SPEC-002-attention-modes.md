# Attention Modes Specification

**Spec ID**: SPEC-002
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-22
**Target Module**: `src/tritter/core/config.py`, `src/tritter/models/architecture.py`

## 1. Overview

### 1.1 Purpose

This specification defines the attention mode configuration system for Tritter, enabling different attention patterns for different use cases:

- **Causal**: Standard autoregressive language modeling
- **Bidirectional**: Embedding extraction and semantic encoding
- **Prefix-LM**: Instruction tuning with context comprehension
- **Embedding**: Continuous latent space reasoning (Coconut-style)

### 1.2 Motivation

Tritter's embedding-prediction paradigm requires flexibility in attention patterns. Different training phases and inference modes benefit from different attention configurations:

| Phase | Attention Mode | Rationale |
|-------|---------------|-----------|
| Base pretraining | Causal | Standard LM objective, efficient computation |
| Embedding extraction | Bidirectional | Full context for semantic representations |
| Instruction tuning | Prefix-LM | Comprehend prompt before generating response |
| Latent reasoning | Embedding | Continuous space without discrete boundaries |

### 1.3 Dependencies

- `TritterConfig` dataclass
- `TritterAttention` module
- FlexAttention integration (SPEC-001)

---

## 2. Attention Mode Definitions

### 2.1 Causal Mode

**Configuration**: `attention_mode="causal"`

**Behavior**:
- Each token attends only to itself and all preceding tokens
- Standard decoder-only transformer pattern
- Compatible with FlashAttention-2 `is_causal=True` optimization

**Attention Matrix** (for sequence length 5):
```
     k0  k1  k2  k3  k4
q0 [  1   0   0   0   0 ]
q1 [  1   1   0   0   0 ]
q2 [  1   1   1   0   0 ]
q3 [  1   1   1   1   0 ]
q4 [  1   1   1   1   1 ]
```

**Use Cases**:
- Base pretraining on text/code corpora
- Standard autoregressive generation
- Default mode for most operations

**Memory Characteristics**:
- No mask materialization with FlashAttention-2
- KV-cache grows linearly with sequence length

### 2.2 Bidirectional Mode

**Configuration**: `attention_mode="bidirectional"`

**Behavior**:
- Every token attends to every other token
- Full self-attention without masking
- Similar to BERT-style encoders

**Attention Matrix** (for sequence length 5):
```
     k0  k1  k2  k3  k4
q0 [  1   1   1   1   1 ]
q1 [  1   1   1   1   1 ]
q2 [  1   1   1   1   1 ]
q3 [  1   1   1   1   1 ]
q4 [  1   1   1   1   1 ]
```

**Use Cases**:
- Embedding extraction for semantic search
- Function-level code understanding
- Similarity computation between code snippets

**Memory Characteristics**:
- Standard O(n²) attention cost (no causal optimization)
- Typically used with shorter sequences

### 2.3 Prefix-LM Mode

**Configuration**: `attention_mode="prefix_lm"`, `prefix_length=N`

**Behavior**:
- Bidirectional attention within prefix region (positions 0 to N-1)
- Causal attention from position N onwards
- Prefix tokens can see each other; generation tokens see all prefix + prior generation

**Attention Matrix** (prefix_length=3, sequence length 5):
```
     k0  k1  k2  k3  k4
q0 [  1   1   1   0   0 ]  <- prefix: sees all prefix
q1 [  1   1   1   0   0 ]  <- prefix: sees all prefix
q2 [  1   1   1   0   0 ]  <- prefix: sees all prefix
q3 [  1   1   1   1   0 ]  <- generation: sees prefix + self
q4 [  1   1   1   1   1 ]  <- generation: sees prefix + prior gen
```

**Use Cases**:
- Instruction following (prompt = prefix, response = generation)
- Summarization (document = prefix, summary = generation)
- Code completion with context (context = prefix, completion = generation)

**Memory Characteristics**:
- Prefix portion: O(prefix²) computation
- Generation portion: O(n) with causal optimization
- Total: More efficient than full bidirectional for long generations

### 2.4 Embedding Mode

**Configuration**: `attention_mode="embedding"`

**Behavior**:
- Bidirectional attention for operating in continuous embedding space
- Used during Coconut-style latent reasoning
- No discrete token boundaries within reasoning chain

**Attention Matrix**: Same as bidirectional

**Use Cases**:
- Continuous chain-of-thought reasoning
- Latent space exploration without token discretization
- Embedding prediction training objective

**Memory Characteristics**:
- Same as bidirectional mode
- Typically used with bounded reasoning steps

---

## 3. Configuration Schema

### 3.1 TritterConfig Fields

```python
@dataclass
class TritterConfig:
    # Attention mode configuration
    attention_mode: Literal["causal", "bidirectional", "prefix_lm", "embedding"] = "causal"
    """Primary attention pattern for the model.

    Why: Different tasks require different attention patterns. Causal is efficient
    for generation, bidirectional captures full context for embeddings, prefix_lm
    enables instruction following, embedding supports latent reasoning.
    """

    prefix_length: int = 0
    """Length of bidirectional prefix for prefix_lm mode.

    Why: Controls where the bidirectional/causal boundary falls. Set based on
    expected instruction length in training data. Ignored for non-prefix_lm modes.

    Constraints:
        - Must be > 0 when attention_mode="prefix_lm"
        - Should be <= typical prompt length
        - Values 256-512 common for instruction tuning
    """
```

### 3.2 Validation Rules

```python
def __post_init__(self) -> None:
    # Validate attention_mode
    valid_modes = {"causal", "bidirectional", "prefix_lm", "embedding"}
    assert self.attention_mode in valid_modes, (
        f"attention_mode must be one of {valid_modes}, got {self.attention_mode!r}"
    )

    # prefix_lm requires prefix_length
    if self.attention_mode == "prefix_lm":
        assert self.prefix_length > 0, (
            "prefix_length must be > 0 when attention_mode='prefix_lm'"
        )

    # Warning for non-causal modes about memory
    if self.attention_mode in {"bidirectional", "embedding"}:
        import warnings
        if self.max_position_embeddings > 8192:
            warnings.warn(
                f"Bidirectional attention with {self.max_position_embeddings} positions "
                "may have high memory usage. Consider using prefix_lm or causal mode."
            )
```

---

## 4. Implementation Interface

### 4.1 TritterAttention Integration

```python
class TritterAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass with attention mode selection.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional custom mask (overrides mode)
            is_causal: Override mode-based causal setting

        Attention Mode Behavior:
            - "causal": is_causal=True, no mask needed
            - "bidirectional": is_causal=False, no mask needed
            - "prefix_lm": Custom mask from create_prefix_lm_mask()
            - "embedding": is_causal=False (same as bidirectional)

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        if is_causal is None:
            is_causal = self.config.attention_mode == "causal"

        if attention_mask is None:
            if self.config.attention_mode == "prefix_lm":
                attention_mask = self._create_prefix_lm_mask(
                    hidden_states.size(1),
                    self.config.prefix_length,
                    hidden_states.device,
                )
                is_causal = False

        # ... rest of attention computation
```

### 4.2 Mask Creation Helpers

```python
def create_prefix_lm_mask(
    seq_len: int,
    prefix_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create prefix-LM attention mask.

    Args:
        seq_len: Total sequence length
        prefix_length: Length of bidirectional prefix
        device: Target device
        dtype: Mask dtype (float for additive mask)

    Returns:
        Attention mask of shape (1, 1, seq_len, seq_len)
        Values: 0 for attend, -inf for mask

    Example:
        >>> mask = create_prefix_lm_mask(5, 3, device="cpu")
        >>> print(mask[0, 0])
        tensor([[   0.,    0.,    0., -inf, -inf],
                [   0.,    0.,    0., -inf, -inf],
                [   0.,    0.,    0., -inf, -inf],
                [   0.,    0.,    0.,    0., -inf],
                [   0.,    0.,    0.,    0.,    0.]])
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    # Causal mask for generation region
    for i in range(prefix_length, seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = float("-inf")

    # Block prefix from seeing generation (optional, for strict prefix-LM)
    # Uncomment if prefix should not see future generation tokens:
    # for i in range(prefix_length):
    #     for j in range(prefix_length, seq_len):
    #         mask[i, j] = float("-inf")

    return mask.unsqueeze(0).unsqueeze(0)
```

---

## 5. Usage Examples

### 5.1 Standard Pretraining (Causal)

```python
config = TritterConfig(
    model_size="3B",
    attention_mode="causal",  # Default
)

model = TritterModel(config)

# Training loop
for batch in dataloader:
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), batch["labels"].view(-1))
```

### 5.2 Embedding Extraction (Bidirectional)

```python
config = TritterConfig(
    model_size="3B",
    attention_mode="bidirectional",
    max_position_embeddings=2048,  # Shorter for bidirectional
)

model = TritterModel(config)

# Extract embeddings
with torch.no_grad():
    hidden_states = model.get_embeddings(input_ids)  # (B, L, D)
    # Pool to single embedding per sequence
    embeddings = hidden_states.mean(dim=1)  # (B, D)
```

### 5.3 Instruction Tuning (Prefix-LM)

```python
config = TritterConfig(
    model_size="3B",
    attention_mode="prefix_lm",
    prefix_length=512,  # Typical instruction length
)

model = TritterModel(config)

# Training with instruction-response pairs
for batch in instruction_dataloader:
    # batch["input_ids"]: [instruction tokens | response tokens]
    # Instruction = bidirectional, Response = causal
    logits = model(batch["input_ids"])

    # Only compute loss on response tokens
    response_start = config.prefix_length
    loss = F.cross_entropy(
        logits[:, response_start:].reshape(-1, config.vocab_size),
        batch["labels"][:, response_start:].reshape(-1),
    )
```

### 5.4 Latent Reasoning (Embedding Mode)

```python
config = TritterConfig(
    model_size="3B",
    attention_mode="embedding",
)

model = TritterModel(config)

# Coconut-style continuous reasoning
hidden = model.embed(input_ids)
for _ in range(num_reasoning_steps):
    hidden = model.transformer(hidden)  # Bidirectional refinement
output_embeddings = hidden[:, -1, :]  # Final embedding
```

---

## 6. Memory and Performance Implications

### 6.1 Memory Usage by Mode

| Mode | Mask Memory | Attention Complexity | FlashAttention |
|------|-------------|---------------------|----------------|
| Causal | 0 | O(n²) compute, O(n) memory | Yes |
| Bidirectional | 0 | O(n²) compute, O(n²) memory | Limited |
| Prefix-LM | O(n²) mask | O(n²) compute | Custom mask |
| Embedding | 0 | O(n²) compute, O(n²) memory | Limited |

### 6.2 Recommended Sequence Lengths

| Mode | Recommended Max | RTX 5080 16GB Limit |
|------|-----------------|---------------------|
| Causal | 128K | 128K (with INT4 KV-cache) |
| Bidirectional | 8K | 8K (activation memory limited) |
| Prefix-LM | 128K total, 2K prefix | 128K (generation is causal) |
| Embedding | 4K | 4K (full attention cost) |

---

## 7. Test Specification

### 7.1 Configuration Validation Tests

| Test | Input | Expected |
|------|-------|----------|
| Valid causal | `attention_mode="causal"` | Config creates successfully |
| Valid prefix_lm | `attention_mode="prefix_lm", prefix_length=256` | Config creates successfully |
| Invalid prefix_lm | `attention_mode="prefix_lm", prefix_length=0` | AssertionError |
| Invalid mode | `attention_mode="invalid"` | AssertionError |

### 7.2 Attention Behavior Tests

| Test | Mode | Input | Verification |
|------|------|-------|--------------|
| Causal masking | causal | seq_len=8 | Position i only attends to positions ≤ i |
| Bidirectional | bidirectional | seq_len=8 | All positions attend to all positions |
| Prefix-LM prefix | prefix_lm | prefix=4, query=2 | Attends to positions 0-3 |
| Prefix-LM generation | prefix_lm | prefix=4, query=6 | Attends to positions 0-6 |

---

## 8. Migration Guide

### 8.1 From Default Causal

No changes needed - `attention_mode="causal"` is the default.

### 8.2 Adding Prefix-LM Support

```python
# Before
config = TritterConfig()

# After
config = TritterConfig(
    attention_mode="prefix_lm",
    prefix_length=512,
)
```

### 8.3 Switching Modes at Runtime

```python
# Create model with one mode
model = TritterModel(TritterConfig(attention_mode="causal"))

# Override for specific forward pass
output = model(input_ids, is_causal=False)  # Force bidirectional
```

---

## Appendix: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-22 | Claude | Initial draft |
