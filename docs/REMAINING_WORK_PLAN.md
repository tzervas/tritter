# Tritter: Remaining Work Plan

**Created**: 2026-02-07
**Based on**: Post-PR #85 analysis of develop branch
**Current state**: ~85% implemented, 2 critical architecture gaps, 151 lint errors, 122 type errors

---

## Critical Architecture Gaps

Two fundamental issues that prevent the model from producing meaningful outputs:

### 1. No Rotary Position Embeddings (RoPE)

**File**: `src/tritter/models/architecture.py` (TritterAttention)

The model has **zero positional encoding**. `TritterAttention` applies QK projections and attention, but queries and keys carry no positional information. The config defines `rope_theta: 500000.0` and `max_position_embeddings: 131072`, but neither is used.

**Fix**:
- Add `RotaryEmbedding` module (precompute cos/sin tables for rope_theta)
- Apply to Q and K after projection, before QK-Norm
- Pass position_ids or compute from sequence length
- Reference: LLaMA, Mistral implementations

### 2. No Grouped Query Attention (GQA)

**File**: `src/tritter/models/architecture.py` (TritterAttention)

`TritterAttention` creates Q/K/V projections all of size `(hidden_size, hidden_size)`. The config supports `num_kv_heads` and model specs define GQA ratios for 7B+ (e.g., 32 query heads, 8 KV heads), but the architecture ignores this.

**Fix**:
- K and V projections output `num_kv_heads * head_dim` dimensions
- Add `repeat_kv()` to expand KV heads for attention computation
- Use `config.effective_num_kv_heads` for projection sizes

---

## Phased Development Plan

### Phase 0: Development Environment (No GPU Required) — BLOCKING

| Task | Priority | Est. Effort |
|------|----------|-------------|
| 0A: Install torch (CPU) + dev deps | Must Have | 1h |
| 0B: Fix 151 ruff lint errors | Must Have | 2-4h |
| 0C: Fix 122 mypy type errors | Should Have | 4-8h |

**Lint error categories**: F-strings without placeholders, unused imports, unused variables, missing `strict=` on zip()

**Type error categories**: "Class cannot subclass Module (has type Any)", unused type:ignore, missing annotations, incompatible types

### Phase 1: Architecture Fixes (No GPU Required) — BLOCKING

| Task | Priority | Est. Effort |
|------|----------|-------------|
| 1A: Implement RoPE | Must Have | 4-6h |
| 1B: Implement GQA | Must Have | 3-4h |
| 1C: Update tests | Must Have | 2-3h |

### Phase 2: Training Pipeline Validation (Requires GPU)

| Task | Priority | Est. Effort |
|------|----------|-------------|
| 2A: Small-scale training smoke test | Must Have | 2-4h |
| 2B: Fix issues discovered during smoke test | Must Have | Variable |
| 2C: Checkpoint format round-trip verification | Should Have | 2-3h |

**Known issues to investigate**:
- Deprecated `torch.cuda.amp.GradScaler` API (should be `torch.amp.GradScaler`)
- `is_causal=True` + `attention_mask` conflict in FlashAttention
- Memory budget validation for target model sizes

### Phase 3: Pretraining at Scale (Requires GPU) — CRITICAL PATH

| Task | Priority | Est. Effort |
|------|----------|-------------|
| 3A: Data preparation (100B token corpus) | Must Have | 1-2 days |
| 3B: Train Tritter-1B (100B tokens) | Must Have | ~4h on RTX 5080 |
| 3C: Train Tritter-3B (200B tokens) | Should Have | ~12h on RTX 5080 |
| 3D: Train Tritter-7B (300B tokens) | Nice to Have | ~36h on RTX 5080 |

### Phase 4: Distribution (Partially Without GPU)

| Task | Priority | Est. Effort |
|------|----------|-------------|
| 4A: HuggingFace upload pipeline (script exists) | Must Have | 4-6h |
| 4B: Inference engine (complete stubs) | Must Have | 1-2 days |
| 4C: Benchmark suite | Should Have | 4-8h |

### Phase 5: Feature Completion (Nice to Have)

| Task | Priority | Notes |
|------|----------|-------|
| 5A: Embedding prediction training | Medium | SPEC-003 draft exists, Coconut/LCM-style |
| 5B: Curriculum scheduler | Medium | Token→embedding transition |
| 5C: Progressive model training (DUS) | Medium | SPEC-008 draft, 3B→7B expansion |
| 5D: FlexAttention embedding mode | Low | For Coconut-style latent reasoning |
| 5E: Bidirectional/prefix-LM attention | Low | For embedding extraction, instruction tuning |
| 5F: Hyperdimensional embedding | Low | Research direction (VSA/HDC) |
| 5G: GGUF format loading | Low | Writing works, loading is stub |
| 5H: Image/audio tokenization integration | Medium | VQ-VAE and SpeechTokenizer |
| 5I: Tensor parallelism | Low | Multi-GPU only, not needed for RTX 5080 |
| 5J: FP8 support | Low | Waiting for stable PyTorch APIs |

---

## Dependency Graph

```
Phase 0A (install deps)
    ├── Phase 0B (ruff) ──────────┐
    ├── Phase 0C (mypy) ──────────┤
    ├── Phase 1A (RoPE) ──┐       │
    └── Phase 1B (GQA) ───┴── Phase 1C (tests)
                                   │
                               Phase 2A (smoke test)
                                   │
                               Phase 2B (fix issues)
                                   │
    Phase 3A (data prep) ─────── Phase 3B (train 1B)
                                   │
                              Phase 3C (train 3B) ── Phase 3D (train 7B)
                                   │
                              Phase 4A (HF upload)
                              Phase 4B (inference engine)
                              Phase 4C (benchmarks)
```

---

## Stubs and TODOs in Code

### NotImplementedError Stubs (16 total)

| Location | What | Phase |
|----------|------|-------|
| `models/flex_attention.py` | FlexAttention embedding mode | 5D |
| `embedding/hyperdimensional.py` | 5 stub methods | 5F |
| `inference/__init__.py` | InferenceEngine, KVCacheManager, EmbeddingRounder | 4B |
| `training/__init__.py` | EmbeddingPredictionLoss, CurriculumScheduler | 5A, 5B |
| `checkpoints/formats.py` | GGUF loading | 5G |

### Code TODOs (6 total)

1. `training/__init__.py:178` — Export EmbeddingPredictionLoss and CurriculumScheduler
2. `tokenization/multimodal.py:457` — Implement VQ-VAE image tokenization
3. `tokenization/multimodal.py:482` — Implement SpeechTokenizer audio tokenization
4. `utils/device_utils.py:72` — Add FP8 support
5. `inference/layer_streaming.py:423` — Add EOS token detection
6. `inference/__init__.py:51` — Implement inference engine after trained weights exist

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| BitNet QAT training instability | High | Medium | Train without BitNet first to validate |
| RoPE integration breaks existing tests | Medium | Low | Shape tests should be unaffected |
| Memory overflow during 7B training | High | Medium | QLoRA on RTX 5080, gradient checkpointing |
| Data quality insufficient | Medium | Low | Existing curation pipeline has quality gates |
| Pretrained weights underperform | Medium | Medium | Start with 1B, validate before scaling |

---

## What Can Be Done Without GPU

| Phase | GPU Required? | Notes |
|-------|---------------|-------|
| Phase 0 (all) | No | Install deps, fix lint/types |
| Phase 1 (all) | No | Implement and test on CPU |
| Phase 2A (partial) | Slow on CPU | Functional but need GPU for real validation |
| Phase 3A (data prep) | No | Curation runs on CPU |
| Phase 4A (HF upload) | No | Script testing |
| Phase 5 (implementation) | No | Implement and test on CPU |
| Phase 2-4 (full validation) | **Yes** | Training, benchmarks, memory verification |

---

*Last Updated: 2026-02-07*
