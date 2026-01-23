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
| **StreamingInferenceEngine** | [layer_streaming.py](src/tritter/inference/layer_streaming.py) | Progressive layer loading for large models |
| **LayerLoader** | [layer_streaming.py](src/tritter/inference/layer_streaming.py) | Layer group loading/eviction with double buffering |
| **MemoryManager** | [memory_manager.py](src/tritter/inference/memory_manager.py) | GPU memory tracking and budget enforcement |
| **TransferEngine** | [transfer_engine.py](src/tritter/inference/transfer_engine.py) | Async CPU→GPU transfers with CUDA streams |

### Critical Architecture Decisions

1. **Squared ReLU** (`x * ReLU(x)`) - Required for BitNet stability, not SiLU/GELU
2. **QK-Norm** - Query-key normalization prevents attention score explosion
3. **Post-FFN LayerNorm** - Chameleon-style: normalize after MLP residual, not before
4. **FlashAttention** - Use `is_causal=True` (not manual mask) for optimal kernel dispatch
5. **Progressive Layer Loading** - Stream layer groups through GPU for unbounded model size

### Progressive Layer Loading

Run models larger than VRAM by streaming layer groups:

```python
config = TritterConfig(
    model_size="7B",
    use_layer_streaming=True,     # Enable layer streaming
    layer_group_size=4,           # 4 layers per group
    gpu_memory_budget_gb=14.0,    # Reserve 2GB headroom
    prefetch_next_group=True,     # Double buffer for latency hiding
)

engine = StreamingInferenceEngine(model, config)
output = engine.generate(input_ids, max_new_tokens=100)
```

How it works:
1. Model weights stay on CPU
2. LayerLoader moves layer groups to GPU on demand
3. TransferEngine uses async CUDA streams for compute/transfer overlap
4. MemoryManager enforces budget to prevent OOM

See [ADR-002](docs/adr/002-progressive-layer-loading.md) and [SPEC-006](docs/specs/SPEC-006-progressive-layer-loading.md).

### Memory Budget (RTX 5080 16GB)

| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| KV-cache (128K, INT4) | ~8-10 GB |
| Vision encoder (SigLIP-B) | ~0.4 GB |
| Activations + overhead | ~2-3 GB |
| **Total** | **~12-15 GB** |

## Implementation Roadmap

### Completed
- FlashAttention with `is_causal=True`
- Progressive layer loading (LayerLoader, StreamingInferenceEngine)
- Memory management (MemoryManager, TransferEngine)
- Attention mode config field
- Training loop with BitNet QAT (Trainer, TrainingConfig)
- Dataset curation pipeline (Python, Rust, Triton)
- Quality gates (security scanner, quality analyzer)

### In Progress
- BitNet-2B weight validation (validate_bitnet_weights.py)
- Triton data source curation

### Planned
- Sliding window attention
- FlexAttention mask primitives
- INT4 KV-cache quantization
- Multimodal encoders (SigLIP, EnCodec)

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

From [TRAINING_STRATEGY.md](docs/TRAINING_STRATEGY.md):

**Persona**: Single-persona AI assistant specialized in software/AI engineering (Python, Rust, Triton).

**Data mix** (targeting ~100B tokens initially):
| Source | Weight | Quality Signal |
|--------|--------|----------------|
| Stack v2 Python (deduped) | 35% | Permissive licenses, >100 stars |
| Stack v2 Rust (deduped) | 20% | Same criteria |
| Triton GPU kernels | 10% | From PyTorch, JAX, ML repos |
| High-quality repos | 20% | Curated, well-documented |
| Technical docs/papers | 10% | arXiv CS, official docs |
| Persona conversations | 5% | Synthetic, Constitutional AI |

**Quality Gates** (from [SPEC-007](docs/specs/SPEC-007-dataset-quality-gates.md)):
- Hardcoded secrets: **ALWAYS REJECT**
- Security vulnerabilities: Label as negative + explanation
- Code quality issues: Label as negative + explanation
- Contrastive learning: Model learns both good AND bad code

## Future Research Directions

From [considerations.md](docs/considerations.md):

- **Hybrid architectures**: Mamba-3 hybrids show 3.3x throughput, 5x less memory
- **Universal Reasoning Model (URM)**: 53.8% on ARC-AGI with recurrent refinement
- **Latent recurrent-depth**: Hidden-state reasoning without token overhead
- **Gated DeltaNet**: 6x decoding, 75% KV-cache reduction

## Key Documentation

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](docs/ROADMAP.md) | Current status and planned work |
| [TRAINING_STRATEGY.md](docs/TRAINING_STRATEGY.md) | Persona, data mix, alignment philosophy |
| [SPEC-007](docs/specs/SPEC-007-dataset-quality-gates.md) | Security and quality gates for datasets |
| [ADR-002](docs/adr/002-progressive-layer-loading.md) | Progressive layer loading decision |
| [SPEC-006](docs/specs/SPEC-006-progressive-layer-loading.md) | Layer streaming implementation spec |
| [project-plan.md](docs/project-plan.md) | Full technical blueprint |
| [considerations.md](docs/considerations.md) | Research on transformer alternatives |
