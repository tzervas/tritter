# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tritter** is a multimodal transformer (text/code/image/audio) with BitNet 1.58-bit ternary quantization {-1, 0, 1}, optimized for RTX 5080 16GB VRAM. Targets 3B and 7B parameter models with 128K context window.

**Core Vision**: Embedding-prediction paradigm (Coconut/LCM style) where the model operates in continuous embedding space, not discrete token space. Token prediction is temporary scaffolding for training compatibility.

## Build & Development Commands

```bash
# Install (using uv recommended)
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/unit/test_config.py

# Run with coverage
pytest --cov=src/tritter --cov-report=html

# Linting and formatting
ruff check .
ruff format .

# Type checking (strict mode enabled)
mypy src/tritter

# Verify imports work
python -c "from tritter import *; print('OK')"
```

## Architecture

### Embedding-Prediction Paradigm

The model operates in continuous embedding space:
- **Entry point**: Tokenization converts discrete tokens → embeddings
- **Core computation**: Transformer layers operate on continuous embeddings
- **Exit point**: Output projection to logits is temporary; production will use KNN/VQ rounding

When writing model or tokenization code, docstrings **must** explicitly acknowledge this paradigm.

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **TritterConfig** | [config.py](src/tritter/core/config.py) | Single source of truth; "7B" auto-configures |
| **TernaryWeight** | [bitnet.py](src/tritter/quantization/bitnet.py) | BitNet quantization with STE gradient flow |
| **TritterAttention** | [architecture.py](src/tritter/models/architecture.py) | Multi-head attention with QK-Norm |
| **TritterMLP** | [architecture.py](src/tritter/models/architecture.py) | FFN with **Squared ReLU** (required for BitNet) |
| **TritterLayer** | [architecture.py](src/tritter/models/architecture.py) | Post-FFN LayerNorm (Chameleon-style) |

### Critical Architecture Decisions

1. **Squared ReLU** (`x * ReLU(x)`) - Required for BitNet stability, not SiLU/GELU
2. **QK-Norm** - Query-key normalization prevents attention score explosion
3. **Post-FFN LayerNorm** - Chameleon-style: normalize after MLP residual, not before
4. **FlashAttention** - Use `is_causal=True` (not manual mask) for optimal kernel dispatch

### Memory Budget (RTX 5080 16GB)

| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| KV-cache (128K, INT4) | ~8-10 GB |
| Vision encoder (SigLIP-B) | ~0.4 GB |
| Activations + overhead | ~2-3 GB |
| **Total** | **~12-15 GB** |

## Implementation Roadmap

### Attention Architecture Evolution

1. **Current**: Basic SDPA with manual causal mask
2. **Immediate Fix**: Use `is_causal=True` for FlashAttention-2 optimization
3. **Short-term**: FlexAttention for dynamic masking (document boundaries, sliding window)
4. **Medium-term**: Attention modes (causal, bidirectional, prefix-lm, embedding)
5. **Long-term**: Hybrid Mamba-2/Transformer (9:1 ratio per IBM Granite 4.0)

### Attention Modes (Planned)

| Mode | Use Case |
|------|----------|
| `causal` | Standard autoregressive LM (pretraining) |
| `bidirectional` | Embedding extraction, semantic encoding |
| `prefix_lm` | Instruction tuning (bidirectional prefix + causal response) |
| `embedding` | Coconut-style continuous latent reasoning |

### FlexAttention Patterns (Planned)

- **Causal**: Standard decoder-only
- **Sliding window**: 4K window bounds KV-cache to ~8GB
- **Document masking**: Packed sequence training without cross-doc attention
- **StreamingLLM**: Attention sinks for streaming beyond window

## Mandatory Development Standards

All contributions must follow [DEVELOPMENT_STANDARDS.md](docs/DEVELOPMENT_STANDARDS.md).

### Critical Requirements

1. **Google-style docstrings with "Why" section** - Explain design decisions
2. **Tensor shapes in comments** - Always document: `x = proj(hidden)  # (B, L, D)`
3. **Use config values in tests** - Never hardcode `vocab_size=1000`, use `config.vocab_size`
4. **Symmetric encode/decode** - Round-trip must preserve data
5. **`__all__` must match imports** - Every `__init__.py` must import what it exports
6. **vocab_size >= 264** - Minimum for byte-level encoding (8 special + 256 bytes)

### Test Requirements

- Parameter count tests need bounds checking (not just `> 0`)
- Gradient tests must verify magnitude (not just existence)
- Every test needs a docstring explaining what it validates

### Anti-Patterns to Avoid

- Hardcoded magic numbers in tests
- Weak validation (`assert params > 0`)
- Missing "Why" explanations in docstrings
- Asymmetric encode/decode (data loss)
- Manual causal mask creation (use `is_causal=True`)

## API Conventions

See [API_CONVENTIONS.md](docs/API_CONVENTIONS.md) for interface schemas:

- Config validation in `__post_init__`
- Error messages must include actual and expected values
- Type hints required for all public APIs

## Training Data Strategy

From [clean-datasets.md](docs/clean-datasets.md):

| Phase | Tokens | Primary Sources |
|-------|--------|-----------------|
| Base pretraining | 1-2T | Stack-Edu Python (45%), Rust (20%) |
| Domain continued | 200-500B | ML frameworks, papers, Kaggle |
| Instruction tuning | 2-5M samples | OSS-Instruct, Glaive |

## Future Research Directions

From [considerations.md](docs/considerations.md):

- **Hybrid architectures**: Mamba-3 hybrids show 3.3x throughput, 5x less memory
- **Universal Reasoning Model (URM)**: 53.8% on ARC-AGI with recurrent refinement
- **Latent recurrent-depth**: Hidden-state reasoning without token overhead
- **Gated DeltaNet**: 6x decoding, 75% KV-cache reduction

## Key Documentation

| Document | Purpose |
|----------|---------|
| [project-plan.md](docs/project-plan.md) | Full technical blueprint |
| [considerations.md](docs/considerations.md) | Research on transformer alternatives |
| [clean-datasets.md](docs/clean-datasets.md) | Training data strategy |
| [tritter-comprehensive-implementation-plan.md](docs/tritter-comprehensive-implementation-plan.md) | Attention architecture roadmap |
