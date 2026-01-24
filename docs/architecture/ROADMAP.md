# Tritter Development Roadmap

## Current State

**Tests**: 495 tests passing, 3 skipped, 24 CUDA environment issues
**Branch**: develop

### Completed ✅

| Feature | Location | Status |
|---------|----------|--------|
| Core transformer with BitNet 1.58-bit | `models/architecture.py` | ✅ Implemented |
| QK-Norm, Squared ReLU, post-FFN LN | `models/architecture.py` | ✅ Implemented |
| TernaryWeight with STE gradient flow | `quantization/bitnet.py` | ✅ Implemented |
| Packed Ternary Inference (2-bit) | `quantization/packed_ternary.py` | ✅ Implemented |
| Multimodal byte-level tokenization | `tokenization/multimodal.py` | ✅ Implemented |
| BPE tokenization (tiktoken) | `tokenization/multimodal.py` | ✅ Implemented |
| AST-aware code tokenization | `tokenization/ast_tokenizer.py` | ✅ Implemented |
| FlashAttention with `is_causal=True` | `models/architecture.py` | ✅ Implemented |
| FlexAttention mask primitives | `attention/flex_attention.py` | ✅ Implemented |
| Sliding window attention | `attention/flex_attention.py` | ✅ Implemented |
| Progressive layer loading | `inference/layer_streaming.py` | ✅ Implemented |
| Streaming inference engine | `inference/layer_streaming.py` | ✅ Implemented |
| MemoryManager with budget enforcement | `inference/memory_manager.py` | ✅ Implemented |
| OS-aware memory detection | `utils/memory_utils.py` | ✅ Implemented |
| Async transfer engine | `inference/transfer_engine.py` | ✅ Implemented |
| INT4 KV-cache quantization | `attention/kv_cache.py` | ✅ Implemented |
| Training loop with BitNet QAT | `training/trainer.py` | ✅ Implemented |
| LoRA/QLoRA fine-tuning | `training/lora.py` | ✅ Implemented |
| Dataset curation pipeline | `curation/` | ✅ Implemented |
| Secret detection | `curation/secrets.py` | ✅ Implemented |
| Security scanning | `curation/security.py` | ✅ Implemented |
| Quality analysis | `curation/quality.py` | ✅ Implemented |
| MinHash deduplication | `curation/dedup.py` | ✅ Implemented |
| SigLIP-B/16 vision encoder | `vision/siglip.py` | ✅ Implemented |
| EnCodec audio tokenization | `audio/encodec.py` | ✅ Implemented |
| Model specs (1B-70B) | `core/model_specs.py` | ✅ Implemented |

### In Progress ⏳

| Feature | Location | Notes |
|---------|----------|-------|
| Triton data source curation | `curation/triton_extraction.py` | Extraction done, curation in progress |
| BitNet-2B weight validation | `scripts/validate_bitnet_weights.py` | Script exists, needs validation data |
| 128K context verification | - | Needs GPU testing |

### Planned 📋

| Feature | Priority | Target |
|---------|----------|--------|
| VQ-VAE image tokenization | Medium | Phase 4 |
| Unified multimodal embedding space | Medium | Phase 4 |
| KNN/VQ rounding for embedding prediction | Medium | Phase 5 |
| Curriculum training | Low | Phase 5 |
| Depth Up-Scaling (DUS) 3B→7B | Medium | Phase 7 |
| Width Up-Scaling (Net2Net) | Medium | Phase 7 |
| EWC forgetting prevention | Low | Phase 7 |
| **Pretrained weights** | **Critical** | **Phase 8** |

---

## Phase 8: Pretrained Weights Distribution

**Goal**: Provide pretrained Tritter weights via HuggingFace Hub sister project.

### Training Plan

| Model | Base Data | Specialization | Target |
|-------|-----------|----------------|--------|
| Tritter-1B | 100B tokens | General | Baseline |
| Tritter-3B | 200B tokens | Code focus | Primary |
| Tritter-7B | 300B tokens | Code + reasoning | Flagship |
| Tritter-7B-Code | +50B code | Python/Rust/Triton | Specialist |

### Distribution Strategy

1. **HuggingFace Hub**: `tritter-ai/tritter-{size}-{variant}`
2. **Checkpoint formats**:
   - Full FP32 (for research/fine-tuning base)
   - Packed ternary (for inference deployment)
   - LoRA adapters (for task-specific fine-tuning)
3. **Model cards** with:
   - Training data composition
   - Benchmark results
   - Usage examples
   - Hardware requirements

### Verification Before Release

```bash
# Memory verification on each target GPU
python scripts/rtx5080_feasibility.py --model 7B --verify-memory

# Quality benchmarks
python scripts/benchmark_quality.py --model checkpoints/tritter-7b/

# Inference speed
python scripts/benchmark_inference.py --model checkpoints/tritter-7b/
```

---

## Hardware Support Matrix

### Verified Target (Primary)

| GPU | VRAM | Max Model (Inference) | Max Model (QLoRA Training) |
|-----|------|----------------------|---------------------------|
| **RTX 5080** | 16 GB | 13B | 40B |

### Consumer GPUs (Planned Support)

| GPU | VRAM | Max Model (Inference) | Max Model (QLoRA Training) | Status |
|-----|------|----------------------|---------------------------|--------|
| RTX 5090 | 32 GB | 40B | 65B+ | 📋 Planned |
| RTX 4090 | 24 GB | 30B | 65B | 📋 Planned |
| RTX 4080 | 16 GB | 13B | 40B | 📋 Planned |
| RTX 4070 Ti | 12 GB | 10B | 30B | 📋 Planned |
| RTX 3090 | 24 GB | 30B | 65B | 📋 Planned |
| RTX 3080 | 10 GB | 7B | 13B | 📋 Planned |

### Enterprise GPUs (Planned Support)

| GPU | VRAM | Max Model (Inference) | Max Model (Full Training) | Status |
|-----|------|----------------------|--------------------------|--------|
| H100 | 80 GB | 70B+ | 13B | 📋 Planned |
| A100-80G | 80 GB | 70B+ | 13B | 📋 Planned |
| A100-40G | 40 GB | 40B | 7B | 📋 Planned |
| A10 | 24 GB | 30B | 3B | 📋 Planned |
| L40 | 48 GB | 65B | 7B | 📋 Planned |
| MI300X | 192 GB | 70B+ | 40B+ | 📋 Planned (ROCm) |

### Memory Budget Formula

```
Safe Budget = (Total VRAM - OS Reserved - CUDA Overhead) × 0.9 - 1GB headroom

OS Reserved:
- Windows 11 DWM: 0.5-1.5 GB (monitor-dependent)
- macOS WindowServer: 0.8 GB
- Linux Wayland/X11: 0.4-0.5 GB
- Linux Headless: 0.1 GB

CUDA Overhead: ~0.3 GB
```

### Hardware-Specific Optimizations

| Hardware Feature | Optimization | GPUs |
|-----------------|--------------|------|
| FP8 compute | INT4 KV-cache | H100, RTX 40xx, RTX 50xx |
| Large L2 cache | Better layer streaming | RTX 5090, H100 |
| High bandwidth | Larger prefetch buffers | A100, H100, MI300X |
| Multi-GPU | Tensor parallelism | All (future) |

---

## Architecture

### What's Implemented

```
src/tritter/
├── core/config.py              # TritterConfig - all hyperparameters
├── models/architecture.py      # TritterModel, TritterLayer, TritterAttention
├── quantization/
│   ├── bitnet.py               # TernaryWeight with STE
│   ├── packed_ternary.py       # 2-bit packed inference weights
│   └── model_specs.py          # 1B-70B model specifications
├── tokenization/multimodal.py  # Byte-level multimodal tokenization
├── vision/siglip.py            # SigLIP-B/16 vision encoder (93M params)
├── attention/
│   ├── flex_attention.py       # FlexAttention masks (sliding window, etc.)
│   └── kv_cache.py             # INT4 quantized KV-cache
├── inference/
│   ├── layer_streaming.py      # LayerLoader, StreamingInferenceEngine
│   ├── memory_manager.py       # GPU memory tracking (OS-aware)
│   └── transfer_engine.py      # Async H2D transfers
├── training/
│   ├── trainer.py              # BitNet QAT training
│   └── lora.py                 # LoRA/QLoRA fine-tuning
├── curation/                   # Dataset quality pipeline
│   ├── secrets.py              # Secret detection (20+ patterns)
│   ├── security.py             # Security vulnerability scanning
│   ├── quality.py              # Code quality analysis
│   ├── dedup.py                # MinHash deduplication
│   ├── pipeline.py             # Unified curation pipeline
│   └── schema.py               # Data labeling schema
├── vision/
│   └── siglip.py               # SigLIP-B/16 vision encoder
├── audio/
│   └── encodec.py              # EnCodec audio tokenization
└── utils/memory_utils.py       # OS memory detection
```

### What's Planned (Not Yet Implemented)

```
src/tritter/
├── vision/vqvae.py             # VQ-VAE image tokenizer (PLANNED)
├── embedding/                  # Embedding prediction (PLANNED)
│   ├── knn_rounding.py         # KNN token lookup
│   └── vq_rounding.py          # VQ codebook lookup
└── training/
    ├── progressive.py          # Model expansion (DUS/Net2Net) (PLANNED)
    └── curriculum.py           # Curriculum training (PLANNED)
```

---

## Quick Start

### Inference (Packed Ternary)

```python
from tritter import TritterConfig, TritterModel
from tritter.quantization import convert_to_packed

# Create model
config = TritterConfig(model_size="7B")
model = TritterModel(config)

# Convert for efficient inference (7B: 28GB → 1.4GB)
packed_model = convert_to_packed(model)

# Generate
output = packed_model(input_ids)
```

### Inference (Layer Streaming for >VRAM models)

```python
from tritter import TritterConfig, TritterModel
from tritter.inference import StreamingInferenceEngine

config = TritterConfig(
    model_size="70B",
    use_layer_streaming=True,
    layer_group_size=4,
    gpu_memory_budget_gb=14.0,
    prefetch_next_group=True,
)

model = TritterModel(config)
engine = StreamingInferenceEngine(model, config)
output = engine.generate(input_ids, max_new_tokens=100)
```

### Fine-Tuning with QLoRA (7B on 16GB)

```python
from tritter import TritterConfig, TritterModel
from tritter.training import LoRAConfig, apply_lora, LoRATrainer

# Create base model
config = TritterConfig(model_size="7B", use_bitnet=True)
model = TritterModel(config)

# Apply QLoRA (only ~17M trainable params)
lora_config = LoRAConfig(
    rank=16,
    alpha=16.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = apply_lora(model, lora_config)

# Train
trainer = LoRATrainer(model, lora_config, learning_rate=1e-4)
trainer.train(train_dataloader, epochs=3)

# Save adapters (~34MB for 7B model)
trainer.save_checkpoint("checkpoints/my_lora/")
```

### Vision (SigLIP)

```python
from tritter.vision import SigLIPEncoder, SigLIPConfig

# Create encoder (93M params, ~0.4GB)
encoder = SigLIPEncoder(SigLIPConfig())

# Encode image
image = torch.randn(1, 3, 384, 384)  # (B, C, H, W)
embeddings = encoder(image)  # (B, 576, 768)
```

### Memory Budget Check

```python
from tritter.utils.memory_utils import check_memory_fit, print_memory_report

# Check if 7B fits
fits, message = check_memory_fit(required_gb=3.7)
print(message)

# Full report
print_memory_report()
# Output:
# ============================================================
# GPU MEMORY REPORT
# ============================================================
# Platform: Linux
# Desktop: Wayland
#
# Total GPU Memory:     16.00 GB
# OS Reserved:          0.50 GB
# Currently Allocated:  0.00 GB
# PyTorch Cached:       0.00 GB
# Safe Budget:          12.88 GB
# ============================================================
```

---

## Memory Budget (RTX 5080 16GB)

| Component | Size |
|-----------|------|
| 7B Packed weights | 0.7 GB |
| Per-channel scales | 0.03 GB |
| KV-cache (128K, INT4) | 8-10 GB |
| Activations | 2-3 GB |
| OS Reserved (Wayland) | 0.5 GB |
| **Total** | ~12-14 GB |

---

## Open Issues

- #58: Triton dataset curation
- #XX: EnCodec audio encoder implementation
- #XX: Pretrained weights training pipeline

---

*Last Updated: 2026-01-23*
