# Tritter - AI Assistant Instructions

This document provides instructions for AI assistants (GitHub Copilot, Claude Code, etc.) working on the Tritter codebase.

## Project Context

Tritter is a multimodal transformer with BitNet 1.58-bit quantization, targeting RTX 5080 16GB VRAM. Key characteristics:

- **3B/7B models** with 128K context window
- **Embedding-prediction paradigm**: Model operates in continuous embedding space (Coconut/LCM style)
- **BitNet quantization**: Ternary weights {-1, 0, +1} with straight-through estimator

## Critical Architecture Rules

### BitNet Requirements
- Use **Squared ReLU** (`x * ReLU(x)`) activation, NOT SiLU or GELU
- Apply **QK-Norm** (query-key normalization) in attention
- Use **Post-FFN LayerNorm** (normalize after MLP residual, not before)
- Maintain shadow weights in full precision for STE training

### FlashAttention Usage
```python
# CORRECT: Use is_causal=True for optimal kernel dispatch
F.scaled_dot_product_attention(q, k, v, is_causal=True)

# INCORRECT: Manual mask creation bypasses FlashAttention optimization
causal_mask = torch.triu(torch.full(..., float("-inf")), diagonal=1)
F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
```

### Memory Budget
Total must fit in 16GB:
- 7B BitNet weights: 1.4 GB
- INT4 KV-cache (128K): ~8-10 GB
- Activations: ~2-3 GB

## Code Style Requirements

### Docstrings
Every non-trivial function needs Google-style docstrings with a "Why" section:

```python
def attention_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute multi-head attention.

    Args:
        hidden_states: Input embeddings of shape (batch_size, seq_len, hidden_size)

    Returns:
        Attention output of same shape for residual connection

    Why: QK-Norm prevents attention score explosion with BitNet quantization
        and long contexts. FlashAttention provides O(N) memory complexity
        essential for 128K context on 16GB VRAM.
    """
```

### Tensor Shape Documentation
Always document tensor shapes in comments:
```python
query = self.q_proj(hidden_states)  # (B, L, D)
query = query.view(B, L, H, D_head).transpose(1, 2)  # (B, H, L, D_head)
```

### Testing
- Use `config.vocab_size`, never hardcode values
- Parameter count tests need bounds checking, not just `> 0`
- Gradient tests must verify magnitude, not just existence

## Validation Commands

```bash
# Format and lint
ruff format . && ruff check .

# Type check
mypy src/tritter

# Run tests
pytest

# Verify imports
python -c "from tritter import *; print('OK')"
```

## Key Files

- `src/tritter/core/config.py` - TritterConfig (auto-scales for 7B)
- `src/tritter/models/architecture.py` - TritterModel, TritterAttention, TritterMLP
- `src/tritter/quantization/bitnet.py` - TernaryWeight with STE
- `docs/DEVELOPMENT_STANDARDS.md` - Full coding standards
- `docs/tritter-comprehensive-implementation-plan.md` - Architecture roadmap
