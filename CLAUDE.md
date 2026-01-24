# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tritter** is a multimodal transformer (text/code/image/audio) with BitNet 1.58-bit ternary quantization {-1, 0, 1}, optimized for RTX 5080 16GB VRAM. Supports model sizes from 1B to 70B with 128K context window.

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
| **TritterConfig** | [config.py](src/tritter/core/config.py) | Single source of truth; auto-configures from model_size |
| **ModelSpec** | [model_specs.py](src/tritter/core/model_specs.py) | Specifications for all model sizes (1B-70B) |
| **TernaryWeight** | [bitnet.py](src/tritter/quantization/bitnet.py) | BitNet quantization with STE gradient flow (training) |
| **PackedTernaryWeight** | [packed_ternary.py](src/tritter/quantization/packed_ternary.py) | 2-bit packed weights (inference only) |
| **TritterAttention** | [architecture.py](src/tritter/models/architecture.py) | Multi-head attention with QK-Norm |
| **TritterMLP** | [architecture.py](src/tritter/models/architecture.py) | FFN with **Squared ReLU** (required for BitNet) |
| **TritterLayer** | [architecture.py](src/tritter/models/architecture.py) | Post-FFN LayerNorm (Chameleon-style) |
| **StreamingInferenceEngine** | [layer_streaming.py](src/tritter/inference/layer_streaming.py) | Progressive layer loading for large models |
| **LayerLoader** | [layer_streaming.py](src/tritter/inference/layer_streaming.py) | Layer group loading/eviction with double buffering |
| **MemoryManager** | [memory_manager.py](src/tritter/inference/memory_manager.py) | GPU memory tracking and budget enforcement |
| **TransferEngine** | [transfer_engine.py](src/tritter/inference/transfer_engine.py) | Async CPU→GPU transfers with CUDA streams |
| **LoRAConfig** | [lora.py](src/tritter/training/lora.py) | Configuration for LoRA fine-tuning |
| **LoRALinear** | [lora.py](src/tritter/training/lora.py) | Low-rank adapter wrapping base layers |
| **LoRATrainer** | [lora.py](src/tritter/training/lora.py) | Trainer specialized for LoRA fine-tuning |

### Critical Architecture Decisions

1. **Squared ReLU** (`x * ReLU(x)`) - Required for BitNet stability, not SiLU/GELU
2. **QK-Norm** - Query-key normalization prevents attention score explosion
3. **Post-FFN LayerNorm** - Chameleon-style: normalize after MLP residual, not before
4. **FlashAttention** - Use `is_causal=True` (not manual mask) for optimal kernel dispatch
5. **Progressive Layer Loading** - Stream layer groups through GPU for unbounded model size

### Supported Model Sizes

| Size | Params | Hidden | Layers | Packed Weights | Recommended VRAM |
|------|--------|--------|--------|----------------|------------------|
| 1B   | 1.1B   | 2048   | 16     | 261 MB         | 8GB              |
| 3B   | 2.4B   | 2560   | 26     | 574 MB         | 8GB              |
| 7B   | 6.2B   | 4096   | 32     | 1.45 GB        | 16GB             |
| 10B  | 9.3B   | 4096   | 40     | 2.16 GB        | 16GB             |
| 13B  | 11.7B  | 5120   | 40     | 2.73 GB        | 24GB             |
| 30B  | 28.5B  | 6656   | 60     | 6.66 GB        | 24GB+streaming   |
| 33B  | 30.2B  | 6912   | 60     | 7.06 GB        | 24GB+streaming   |
| 40B  | 42.2B  | 8192   | 60     | 9.87 GB        | 48GB / 2x24GB    |
| 65B  | 56.4B  | 8192   | 80     | 13.2 GB        | 80GB / 4x24GB    |
| 70B  | 69.5B  | 8192   | 80     | 16.3 GB        | 80GB / 4x24GB    |

Models 7B+ use Grouped Query Attention (GQA) for KV-cache efficiency.

**View model details**: `python scripts/show_model_specs.py --model 7B`

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
| 7B packed weights | 1.45 GB |
| KV-cache (32K, INT4) | ~1 GB |
| KV-cache (128K, INT4) | ~4 GB |
| Vision encoder (SigLIP-B) | ~0.4 GB |
| Activations + overhead | ~2 GB |

**Recommended configurations**:
- 7B + 32K context: ~5 GB (comfortable on 8GB+)
- 7B + 128K context: ~8 GB (fits 16GB with headroom)

See [SPEC-009](docs/specs/SPEC-009-packed-ternary-inference.md) for packed inference details.

### LoRA/QLoRA Fine-Tuning

Full fine-tuning of 7B+ models requires 60GB+ memory. LoRA freezes base weights and adds small trainable adapters:

```python
from tritter.training.lora import LoRAConfig, apply_lora, LoRATrainer

# Apply QLoRA (LoRA on ternary base weights)
lora_config = LoRAConfig(
    rank=16,                    # Low-rank dimension
    alpha=16.0,                 # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = apply_lora(model, lora_config)

# Only LoRA params (~17M for 7B) are trained
trainer = LoRATrainer(model, lora_config, learning_rate=1e-4)
```

**QLoRA Memory Comparison**:

| Model | Full Training | QLoRA (r=16) | Fits RTX 5080? |
|-------|--------------|--------------|----------------|
| 1B    | 12.2 GB      | 1.9 GB       | ✓              |
| 7B    | 60.8 GB      | 3.7 GB       | ✓              |
| 13B   | 111.8 GB     | 5.1 GB       | ✓              |
| 40B   | 397.8 GB     | 13.7 GB      | ✓              |
| 70B   | 652.1 GB     | 20.2 GB      | ✗              |

See [SPEC-010](docs/specs/SPEC-010-lora-finetuning.md) for full LoRA documentation.

**Run feasibility analysis**: `python scripts/rtx5080_feasibility.py`

## Risk Analysis

### Memory Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| OOM during inference (large context) | High | Use layer streaming, INT4 KV-cache |
| OOM during training | High | Use QLoRA, gradient checkpointing, smaller batch |
| KV-cache overflow | High | Limit context length, use sliding window |
| Activation spikes | Medium | Gradient checkpointing already enabled |

### Quality Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quantization degradation | Medium | Per-channel scaling, QK-Norm for stability |
| LoRA underfitting | Medium | Use rank 16+, add MLP targets if needed |
| Catastrophic forgetting | Medium | Lower learning rate, early stopping |

### Integration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pack/unpack precision loss | Low | Lossless ternary encoding (exact round-trip) |
| Layer streaming latency | Medium | Double buffering, async transfers |
| LoRA merge with BitNet | Low | Keep adapters separate or zero B after merge |

### Hardware Compatibility

| GPU | Inference (7B) | Training (7B) | Notes |
|-----|---------------|---------------|-------|
| RTX 5080 16GB | ✓ 128K ctx | ✓ QLoRA only | Target hardware |
| RTX 4090 24GB | ✓ 128K ctx | ✓ QLoRA, marginal full | Extra headroom |
| A100 40GB | ✓ 128K ctx | ✓ Full or QLoRA | Best option |
| RTX 3080 10GB | ✓ 32K ctx | ✓ QLoRA r=8 | Limited context |

## Hardware Profiles

Primary development hardware: **RTX 5080 (16GB)**, secondary: **RTX 3090 Ti (24GB)**

```python
from tritter.utils import detect_gpu_profile, create_config_for_profile

# Auto-detect current GPU
profile = detect_gpu_profile()

# Generate optimized config
config_kwargs = create_config_for_profile(profile, "7B")
config = TritterConfig(**config_kwargs)
```

**Quick check**: `python scripts/hardware_profile.py --check 7B`

### OS Memory Management

Desktop environments reserve GPU memory:
- Windows 11 DWM: 0.5-1.5 GB (depends on monitors)
- macOS WindowServer: 0.8 GB
- Linux Wayland/X11: 0.4-0.5 GB

The `MemoryManager` automatically detects and accounts for this overhead:

```python
from tritter.utils import check_memory_fit, print_memory_report

# Pre-flight check
fits, message = check_memory_fit(required_gb=3.7)

# Detailed report
print_memory_report()
```

## Implementation Status

### ✅ Implemented (Production Ready)

| Feature | Location | Notes |
|---------|----------|-------|
| BitNet 1.58-bit quantization | `quantization/bitnet.py` | STE gradient flow |
| Packed ternary inference | `quantization/packed_ternary.py` | 16x weight reduction |
| Model specs 1B-70B | `quantization/model_specs.py` | GQA for 7B+ |
| FlashAttention | `models/architecture.py` | `is_causal=True` |
| FlexAttention masks | `attention/flex_attention.py` | Sliding window, etc. |
| INT4 KV-cache | `attention/kv_cache.py` | 4x cache reduction |
| Progressive layer loading | `inference/layer_streaming.py` | Run 70B on 16GB |
| OS-aware memory manager | `inference/memory_manager.py` | Auto-detects desktop overhead |
| Async transfer engine | `inference/transfer_engine.py` | CUDA stream overlap |
| BitNet QAT training | `training/trainer.py` | Gradient checkpointing |
| LoRA/QLoRA fine-tuning | `training/lora.py` | Train 40B on 16GB |
| Dataset curation | `curation/` | Quality gates, dedup |
| SigLIP vision encoder | `vision/siglip.py` | 93M params, ~0.4GB |
| Hardware profiles | `utils/hardware_profiles.py` | RTX 5080, 3090 Ti verified |

### ⏳ In Progress

| Feature | Status | Notes |
|---------|--------|-------|
| Triton data curation | Extraction done | Curation WIP |
| BitNet-2B validation | Script exists | Needs validation data |
| 128K context verification | Ready | Needs GPU testing |

### 📋 Planned (Not Yet Implemented)

| Feature | Priority | Notes |
|---------|----------|-------|
| EnCodec audio encoder | High | Required for audio modality |
| VQ-VAE image tokenizer | Medium | Alternative to SigLIP patches |
| KNN/VQ embedding rounding | Medium | Core embedding-prediction feature |
| Model expansion (DUS) | Medium | 3B→7B→13B progressive training |
| Pretrained weights | **Critical** | HuggingFace distribution |

### Attention Modes

| Mode | Status | Use Case |
|------|--------|----------|
| `causal` | ✅ Implemented | Standard autoregressive LM |
| `sliding_window` | ✅ Implemented | Long context with bounded KV-cache |
| `bidirectional` | 📋 Planned | Embedding extraction |
| `prefix_lm` | 📋 Planned | Instruction tuning |
| `embedding` | 📋 Planned | Coconut-style reasoning |

## Pretrained Weights (Planned)

Target distribution via HuggingFace Hub: `tritter-ai/tritter-{size}-{variant}`

| Model | Data | Use Case |
|-------|------|----------|
| Tritter-1B | 100B tokens | Baseline, testing |
| Tritter-3B | 200B tokens | Primary, code focus |
| Tritter-7B | 300B tokens | Flagship, reasoning |
| Tritter-7B-Code | +50B code | Python/Rust specialist |

**Formats**: Full FP32 (fine-tuning), Packed ternary (deployment), LoRA adapters (task-specific)

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
| [SPEC-006](docs/specs/SPEC-006-progressive-layer-loading.md) | Layer streaming implementation spec |
| [SPEC-007](docs/specs/SPEC-007-dataset-quality-gates.md) | Security and quality gates for datasets |
| [SPEC-009](docs/specs/SPEC-009-packed-ternary-inference.md) | Packed ternary weights (2-bit encoding) |
| [SPEC-010](docs/specs/SPEC-010-lora-finetuning.md) | LoRA/QLoRA memory-efficient fine-tuning |
| [ADR-002](docs/adr/002-progressive-layer-loading.md) | Progressive layer loading decision |
| [project-plan.md](docs/project-plan.md) | Full technical blueprint |
| [considerations.md](docs/considerations.md) | Research on transformer alternatives |
