# Tritter Development Roadmap

## Current State (Phase 1 Complete)

- [x] Core transformer architecture with BitNet 1.58-bit quantization
- [x] QK-Norm, Squared ReLU, post-FFN LayerNorm (Chameleon-compliant)
- [x] TernaryWeight with STE gradient flow
- [x] Multimodal tokenization (byte-level encoding)
- [x] 39 unit tests passing
- [x] Development standards and documentation

---

## Phase 2: Attention Optimization (Current)

### 2.1 FlashAttention Fix (Priority: Critical)
- [ ] Replace manual causal mask with `is_causal=True`
- [ ] Benchmark memory/speed improvement on RTX 5080
- [ ] Update tests to verify kernel dispatch

### 2.2 Attention Mode Configuration
- [ ] Add `attention_mode` config: causal, bidirectional, prefix_lm, embedding
- [ ] Add `sliding_window_size` config (default 4096)
- [ ] Add `use_attention_sinks` for StreamingLLM

### 2.3 FlexAttention Integration
- [ ] Create `src/tritter/models/flex_attention.py`
- [ ] Implement mask primitives: causal, sliding_window, document, streamingllm
- [ ] Add BlockMask caching for efficiency

---

## Phase 3: Memory Optimization

### 3.1 KV-Cache Quantization
- [ ] Implement INT4 KV-cache quantization
- [ ] Target: 128K context in ~8GB
- [ ] Integrate with sliding window attention

### 3.2 Memory Profiling
- [ ] Add memory budget verification tests
- [ ] Profile peak memory at various context lengths
- [ ] Document RTX 5080 optimization settings

---

## Phase 4: Multimodal Integration

### 4.1 Vision Encoder
- [ ] Integrate SigLIP-B/16 (93M params)
- [ ] Implement pixel shuffle (factor 4) for token reduction
- [ ] VQ-VAE image tokenization (256-512 tokens/image)

### 4.2 Audio Tokenization
- [ ] Integrate EnCodec/SpeechTokenizer
- [ ] Semantic/acoustic disentanglement
- [ ] Unified vocabulary integration

### 4.3 Unified Embedding Space
- [ ] Early fusion architecture (Chameleon-style)
- [ ] Combined vocabulary: ~100K tokens
- [ ] Cross-modal attention patterns

---

## Phase 5: Embedding Prediction Paradigm

### 5.1 Coconut-Style Continuous Thought
- [ ] Hidden state → next embedding feedback loop
- [ ] Latent space reasoning
- [ ] KNN/VQ rounding for discrete output

### 5.2 Training Adaptation
- [ ] Curriculum-based training for embedding prediction
- [ ] Dual-mode: token prediction (training) + embedding prediction (inference)
- [ ] Fallback to token prediction where continuous degrades

---

## Phase 6: Training Pipeline

### 6.1 Data Preparation
- [ ] Stack v2 Python/Rust processing pipeline
- [ ] MinHash LSH deduplication
- [ ] Quality filtering (line length, alphabetic ratio)

### 6.2 Pretraining
- [ ] 1-2T tokens on Stack-Edu (Python 45%, Rust 20%)
- [ ] BitNet QAT via Nanotron
- [ ] Checkpoint management

### 6.3 Instruction Tuning
- [ ] OSS-Instruct (75K examples)
- [ ] Prefix-LM attention pattern
- [ ] 2-5M instruction samples

---

## Phase 7: Hybrid Architecture (Research)

### 7.1 Mamba-2 Evaluation
- [ ] Benchmark Mamba-2 blocks
- [ ] Test 9:1 Mamba:Transformer ratio (IBM Granite style)
- [ ] Memory/quality tradeoff analysis

### 7.2 Advanced Architectures
- [ ] Gated DeltaNet evaluation
- [ ] URM-style recurrent refinement
- [ ] Latent recurrent depth

---

## Memory Budget Reference (RTX 5080 16GB)

| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| Vision encoder (SigLIP-B) | 0.4 GB |
| INT4 KV-cache (128K, 4K window) | ~8.4 GB |
| Activations + overhead | ~2 GB |
| **Total** | **~12.2 GB** |

---

## Task Complexity Classification

### Opus-tier (Large Planning)
- Architecture design decisions
- Multi-phase implementation planning
- Research synthesis and evaluation

### Sonnet-tier (Medium Scope)
- Module implementation
- Test suite development
- Documentation writing

### Haiku-tier (Focused Execution)
- Bug fixes
- Config changes
- Small refactors
- Code formatting

---

*Last Updated: 2026-01-21*
