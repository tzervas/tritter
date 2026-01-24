# Tritter Development Roadmap

## Current State

**Tests**: 84+ passing (core modules)
**Branch**: develop

### Completed

- Core transformer with BitNet 1.58-bit quantization
- QK-Norm, Squared ReLU, post-FFN LayerNorm
- TernaryWeight with STE gradient flow
- Multimodal byte-level tokenization
- FlashAttention with `is_causal=True`
- Progressive layer loading (unbounded model size)
- Streaming inference engine
- Training loop with BitNet QAT (Trainer, TrainingConfig)
- Dataset curation pipeline (Python, Rust, Triton)
- Quality gates: security scanner, quality analyzer
- SPEC-007: Dataset quality gates specification

### In Progress

- Triton data source curation
- 128K context verification

---

## Architecture

### What Works

```
src/tritter/
├── core/config.py           # TritterConfig - all hyperparameters
├── models/architecture.py   # TritterModel, TritterLayer, TritterAttention
├── quantization/bitnet.py   # TernaryWeight with STE
├── tokenization/multimodal.py
└── inference/
    ├── layer_streaming.py   # LayerLoader, StreamingInferenceEngine
    ├── memory_manager.py    # GPU memory tracking
    └── transfer_engine.py   # Async H2D transfers
```

### Progressive Layer Loading

Run 70B models on 16GB VRAM:

```python
from tritter import TritterConfig, TritterModel
from tritter.inference import StreamingInferenceEngine

config = TritterConfig(
    model_size="7B",
    use_layer_streaming=True,
    layer_group_size=4,
    gpu_memory_budget_gb=14.0,
    prefetch_next_group=True,
)

model = TritterModel(config)
engine = StreamingInferenceEngine(model, config)
output = engine.generate(input_ids, max_new_tokens=100)
```

Layers stream from CPU to GPU in groups. Double buffering overlaps transfer with compute. See [SPEC-006](specs/SPEC-006-progressive-layer-loading.md).

---

## Phase 2: Attention Modes

| Status | Task |
|--------|------|
| ✅ | FlashAttention `is_causal=True` |
| ✅ | `attention_mode` config field |
| ✅ | Sliding window attention |
| ✅ | FlexAttention mask primitives |

See [SPEC-001](specs/SPEC-001-flexattention.md).

---

## Phase 3: Memory Optimization

| Status | Task |
|--------|------|
| ✅ | MemoryManager with budget enforcement |
| ✅ | Async transfer engine |
| ✅ | INT4 KV-cache quantization |
| ⏳ | 128K context verification |

See [SPEC-005](specs/SPEC-005-memory-optimization.md).

---

## Phase 4: Multimodal

| Status | Task |
|--------|------|
| ⏳ | SigLIP-B/16 vision encoder |
| ⏳ | VQ-VAE image tokenization |
| ⏳ | EnCodec audio tokenization |
| ⏳ | Unified embedding space |

---

## Phase 5: Embedding Prediction

| Status | Task |
|--------|------|
| ✅ | Hidden state feedback loop |
| ⏳ | KNN/VQ rounding |
| ⏳ | Curriculum training |

See [SPEC-003](specs/SPEC-003-embedding-prediction.md).

---

## Phase 6: Training

| Status | Task |
|--------|------|
| ✅ | BitNet QAT trainer (Trainer, TrainingConfig) |
| ✅ | Dataset curation pipeline |
| ✅ | Quality gates (security, quality) |
| ✅ | Hash-based deduplication |
| ⏳ | BitNet-2B weight validation |
| ⏳ | MinHash near-duplicate detection |
| ⏳ | Triton kernel data sources |

See [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md) and [SPEC-007](specs/SPEC-007-dataset-quality-gates.md).

---

## Memory Budget (RTX 5080 16GB)

| Component | Size |
|-----------|------|
| 7B BitNet weights | 1.4 GB |
| KV-cache (128K, INT4) | 8-10 GB |
| Layer group buffer | 0.4 GB |
| Activations | 2 GB |
| **Total** | ~12-14 GB |

---

## Open Issues

- #58: Triton dataset curation
- #60: SigLIP vision encoder integration

---

*Last Updated: 2026-01-23*
