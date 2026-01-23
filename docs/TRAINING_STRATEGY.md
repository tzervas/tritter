# Tritter Training Strategy

## Philosophy

**Goal**: Train a single-persona AI assistant specialized in software and AI engineering, aligned by design through data curation—not post-hoc RLHF.

**Anti-goal**: Avoid the "shogoth" pattern—a multi-persona blob that requires extensive fine-tuning to not be sociopathic.

**Approach**: Quality over quantity. Curated datasets from high-quality sources. Constitutional principles baked into training data, not bolted on after.

---

## Persona Definition

**Name**: Tritter (working name)
**Role**: Software and AI engineering assistant
**Specializations**: Python, Rust, Triton, ML frameworks
**Character traits**:
- Direct and technically precise
- Explains reasoning without being patronizing
- Admits uncertainty rather than hallucinating
- Focuses on practical solutions
- Respects the user's intelligence and autonomy

**What the persona is NOT**:
- Not a sycophant ("Great question!")
- Not evasive or preachy
- Not multi-personality
- Not pretending to be human
- Not refusing reasonable requests with canned responses

---

## Bootstrap Strategy

### Phase 0: Validation (Current)

**Goal**: Prove the architecture works end-to-end before investing compute.

**Approach**: Use Microsoft's [BitNet b1.58-2B-4T-bf16](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16) as reference:
- Architecture matches (squared ReLU, RoPE, no bias)
- MIT licensed, BF16 master weights available
- Can validate our training loop against known-good weights

**Validation tasks**:
1. Load BitNet-2B weights into Tritter architecture
2. Verify forward pass produces similar outputs
3. Run small continued pretraining to validate training loop
4. Confirm checkpointing, metrics, gradient flow

### Phase 1: Continued Pretraining on Code

**Goal**: Adapt BitNet-2B to code domain while developing persona.

**Base**: microsoft/bitnet-b1.58-2B-4T-bf16 (2B params, 4T tokens)

**Data mix** (targeting ~100B tokens initially):
| Source | Weight | Quality Signal |
|--------|--------|----------------|
| The Stack v2 Python (deduped) | 40% | Permissive licenses, >100 stars |
| The Stack v2 Rust (deduped) | 25% | Same criteria |
| High-quality repos (curated) | 20% | See curation criteria below |
| Technical docs/papers | 10% | arXiv CS, official docs |
| Persona-defining conversations | 5% | Synthetic, Constitutional AI style |

**Curation criteria for repos**:
- Maintained (commits in last 6 months)
- Well-documented (README, docstrings)
- Tests present
- No obvious code smells
- Permissive license (MIT, Apache, BSD)

### Phase 2: Persona Refinement

**Goal**: Develop the single-persona character through instruction tuning.

**Data sources**:
- [OSS-Instruct](https://arxiv.org/abs/2312.02120) (75K code examples seeded from real code)
- Custom synthetic conversations following Constitutional AI principles
- Technical Q&A from Stack Overflow (high-quality answers only)

**Constitutional principles** (baked into synthetic data):
1. Be direct and technically accurate
2. Explain your reasoning
3. Admit when uncertain
4. Don't lecture or moralize
5. Respect user autonomy
6. Focus on the task at hand
7. Provide working code, not pseudocode

**What we're NOT doing**:
- No RLHF on preferences (alignment comes from data)
- No safety fine-tuning (don't train bad behavior to fix)
- No multi-turn roleplay (single consistent persona)

---

## Dataset Quality Assessment

### Evaluation Criteria

| Criterion | Weight | How to Measure |
|-----------|--------|----------------|
| Correctness | 30% | Code compiles/runs, factually accurate |
| Style | 20% | Follows language conventions, readable |
| Documentation | 15% | Functions have docstrings, comments explain why |
| Deduplication | 15% | MinHash LSH, exact match removal |
| License | 10% | Permissive only |
| Freshness | 10% | Prefer recent, maintained code |

### Red Flags (Exclude)

- Auto-generated code (low quality patterns)
- Minified/obfuscated code
- License-violating code
- Obvious bugs or security vulnerabilities
- Student homework (usually poor quality)
- Vendored dependencies
- Configuration/boilerplate files

### Deduplication Strategy

Research shows [100% performance inflation](https://www.researchgate.net/publication/398980230_Code2Doc_A_Quality-First_Curated_Dataset_for_Code_Documentation) on duplicated data. Aggressive dedup is critical:

1. **Exact match**: Hash-based removal
2. **Near-duplicate**: MinHash with Jaccard threshold 0.8
3. **Semantic**: Embedding similarity > 0.95
4. **Cross-split**: Ensure train/eval have no overlap

---

## Compatible Pretrained Components

### Primary: Microsoft BitNet b1.58-2B-4T

| Aspect | Assessment |
|--------|------------|
| Architecture | ✅ Matches Tritter (squared ReLU, RoPE, no bias) |
| License | ✅ MIT |
| Quality | ✅ 4T tokens training, state-of-the-art for size |
| Weights | ✅ BF16 master weights available |
| Limitation | ⚠️ 2B params (smaller than 7B target) |
| Use case | Bootstrap training validation, continued pretraining |

### Secondary: StarCoder2-3B

| Aspect | Assessment |
|--------|------------|
| Architecture | ⚠️ Different (GQA vs MHA, different norm) |
| License | ✅ BigCode OpenRAIL-M |
| Quality | ✅ Code-specialized, 600+ languages |
| Weights | ✅ Available |
| Limitation | ❌ Architecture mismatch with Tritter |
| Use case | Embedding initialization only (if compatible) |

### Not Using: CodeLlama

| Aspect | Assessment |
|--------|------------|
| Architecture | ❌ Standard LLaMA (not BitNet) |
| License | ❌ Llama 2 Community License (restrictive) |
| Use case | Reference benchmark only |

---

## Training Infrastructure

### Hardware Requirements

| Phase | GPU | VRAM | Time Estimate |
|-------|-----|------|---------------|
| Validation | 1x RTX 5080 | 16GB | Hours |
| Phase 1 (2B, 100B tokens) | 4x A100 | 320GB | ~1 week |
| Phase 2 (2B, instruction) | 1x A100 | 80GB | Days |
| Full 7B training | 8x H100 | 640GB | Weeks |

### Software Stack

- **Framework**: PyTorch 2.0+ (CUDA 12.0+)
- **Training**: Custom Trainer (implemented) or Nanotron
- **Data**: Streaming from HuggingFace datasets
- **Monitoring**: Weights & Biases
- **Checkpointing**: Every 1000 steps

---

## Alignment by Design

### Why Not RLHF?

1. **RLHF fixes symptoms, not causes**: If your base model is "sociopathic," RLHF papers over it
2. **Reward hacking**: Models learn to game the reward model
3. **Mode collapse**: RLHF can reduce capability while increasing "safety scores"
4. **Expensive**: Requires human preference data collection

### Constitutional AI Approach

Instead of RLHF, we embed alignment into the training data itself:

1. **Principle-guided generation**: Synthetic data follows explicit principles
2. **Self-revision**: Model revises outputs based on principles
3. **Quality filtering**: Only include data that exemplifies good behavior

**Principles** (natural language, embedded in training):
```
1. Be helpful and technically accurate
2. Explain reasoning without being condescending
3. Admit uncertainty rather than confabulating
4. Provide working, tested code
5. Respect user's time and intelligence
6. Stay focused on the technical task
7. Don't moralize or lecture
```

### Negative Examples (What to Avoid)

The training data explicitly excludes or minimizes:
- Sycophantic responses ("What a great question!")
- Unnecessary caveats and disclaimers
- Refusals with canned safety responses
- Roleplay or persona-switching
- Emotional manipulation
- Hallucinated citations or code

---

## Metrics and Evaluation

### Code Quality Metrics

| Metric | Target | How |
|--------|--------|-----|
| HumanEval | >50% pass@1 | Code generation benchmark |
| MBPP | >60% | Python problems |
| MultiPL-E | >40% Rust | Multi-language eval |
| DS-1000 | >30% | Data science problems |

### Alignment Metrics

| Metric | Target | How |
|--------|--------|-----|
| Persona consistency | 95%+ | Same-question different phrasing |
| Hallucination rate | <5% | Fact verification |
| Refusal appropriateness | 99%+ | Reasonable requests not refused |
| Sycophancy score | <10% | Detection of empty praise |

### Perplexity Targets

| Dataset | Target PPL |
|---------|------------|
| Python code | <3.0 |
| Rust code | <4.0 |
| Technical docs | <5.0 |

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Training loop validated on small data |
| 2 | BitNet-2B weights loaded, forward pass verified |
| 3 | Continued pretraining started (small scale) |
| 4-6 | Phase 1 training (100B tokens on code) |
| 7-8 | Phase 2 instruction tuning |
| 9-10 | Evaluation and iteration |

---

## Open Questions

1. **Embedding initialization**: Can we transfer embeddings from StarCoder2 to bootstrap?
2. **Scaling**: How to scale from 2B to 7B without losing persona?
3. **Evaluation**: How to measure "not sociopathic" quantitatively?
4. **Curriculum**: Should we mix instruction data throughout or phase it?

---

## References

- [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [Constitutional AI Paper](https://arxiv.org/pdf/2212.08073)
- [OSS-Instruct / Magicoder](https://arxiv.org/abs/2312.02120)
- [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
- [Code2Doc Quality-First Dataset](https://arxiv.org/pdf/2512.18748)
- [StarCoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2)

---

*Last Updated: 2026-01-23*
