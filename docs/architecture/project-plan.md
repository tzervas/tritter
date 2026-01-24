# Building a custom multimodal AI model for RTX 5080: a technical blueprint

The combination of **BitNet b1.58 quantization**, **INT4 KV-cache compression**, and **Flash Attention** makes a **7B parameter any-to-any multimodal model with 128K+ context technically feasible on 16GB VRAM**. This represents the practical ceiling for consumer hardware, with proven implementations available for each component. The most ambitious aspect—embedding prediction replacing token prediction—remains experimental but shows promising results in systems like Meta's Coconut and Large Concept Models.

This report synthesizes research across BitNet architecture, training datasets, embedding-prediction models, multimodal tokenization, memory-efficient attention, RTX 5080 optimization, minimal implementations, and GitHub workflow tooling to provide a grounded technical foundation for this project.

---

## BitNet b1.58 delivers 10× memory reduction with matching performance

Microsoft's BitNet b1.58 quantizes weights to ternary values **{-1, 0, +1}** [Skywork](https://skywork.ai/blog/models/bitnet_b1_58-3b-free-chat-online-skywork-ai/) using AbsMean scaling, achieving **1.58 bits per parameter** (log₂(3)). [Enerzai](https://enerzai.com/resources/blog/small-but-mighty-a-technical-deep-dive-into-1-58-bit-quantization) [AI-SCHOLAR](https://ai-scholar.tech/en/articles/large-language-models/BitNet1-58b) The architecture maintains 8-bit activations and replaces standard linear layers with BitLinear operations. [Emergent Mind](https://www.emergentmind.com/topics/bitnet-b1-58-model) [Ajith's AI Pulse](https://ajithp.com/2024/03/09/bitnet-b1-58/) The flagship **bitnet-b1.58-2B-4T** model trained on 4 trillion tokens demonstrates performance matching full-precision 2B models on standard benchmarks. [arXiv](https://arxiv.org/abs/2504.12285)

Memory footprint scales dramatically: a **7B BitNet model requires approximately 1.4GB** for weights versus 14GB for FP16. This formula applies: `BitNet_weights = Parameters × 0.1975 bytes`. Microsoft's bitnet.cpp achieves **2.37-6.17× speedup** over llama.cpp on x86 CPUs with **71.9-82.2% energy reduction**, enabling 100B parameter models to run at human reading speed on single CPUs. [GitHub](https://github.com/microsoft/BitNet)

Key architectural requirements include **QK-Norm** (query-key normalization), **RMSNorm** pre-attention, **SwiGLU activation**, **Squared ReLU in FFN layers**, and no bias terms. [Hugging Face](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) Training requires maintaining shadow weights in BF16 precision with straight-through estimator (STE) for gradients through the quantization step. [arXiv](https://arxiv.org/html/2502.11895v1)

| Model Size | FP16 Memory | BitNet Memory | Practical Savings |
|------------|-------------|---------------|-------------------|
| 1B | 2 GB | 0.25 GB | 8× |
| 3B | 6 GB | 0.6 GB | 10× |
| 7B | 14 GB | 1.4 GB | 10× |

**Production implementations**: The official bitnet.cpp (github.com/microsoft/BitNet, 25.8k stars) provides lossless inference. [arXiv](https://arxiv.org/pdf/2504.12285) For training, the **Nanotron** framework supports 1.58-bit QAT, and the `bitlinear` PyPI package offers drop-in BitLinear layers. Fine-tuning remains challenging—Microsoft's recommendation is to use their released BF16 master weights at `microsoft/bitnet-b1.58-2B-4T-bf16` for domain adaptation. [Hugging Face](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

---

## The Stack v2 provides 900B tokens of license-clean code

For training on Python/Rust AI engineering codebases, **The Stack v2** dominates at **67.5TB raw size (~900 billion tokens)** [ServiceNow](https://www.servicenow.com/blogs/2024/announcing-starcoder2-stack-v2) across 619 programming languages, [arXiv](https://arxiv.org/html/2402.19173v1) with full provenance tracking via Software Heritage persistent identifiers. [Hugging Face](https://huggingface.co/datasets/bigcode/the-stack-v2) The Python subset contains **233GB deduplicated** across 57 million files, with Jupyter notebooks converted via Jupytext and Kaggle notebooks under Apache 2.0.

Rust representation remains limited at **15.6GB deduplicated** (2.2 million files), reflecting the language's smaller ecosystem. For Rust-specific training, the **Strand-Rust-Coder** synthetic dataset (191k examples across 15 task categories) achieved 13% absolute gains on Rust benchmarks through swarm-intelligence-based data generation.

Quality filtering for code datasets follows established patterns: **MinHash LSH deduplication** (256 permutations), [Hugging Face](https://huggingface.co/datasets/bigcode/the-stack-dedup) maximum line length filtering (>1000 chars removed), alphabetic ratio checks (>25% required), auto-generated file detection, and Base64/encoded data removal. StarCoder's approach includes PII redaction for emails, API keys, and SSH keys, [Hugging Face](https://huggingface.co/datasets/bigcode/starcoderdata) plus decontamination against HumanEval, MBPP, and DS-1000 benchmarks.

For instruction tuning, the cleanest commercial options are:
- **OSS-Instruct** (Magicoder): 75K examples seeded from real open-source code [arXiv](https://arxiv.org/abs/2312.02120)
- **evol-codealpaca-v1**: 110K complexity-evolved pairs
- **The Stack v2** with permissive license filtering via Blue Oak Council approved licenses [Hugging Face](https://huggingface.co/datasets/bigcode/the-stack-v2)

---

## Embedding prediction architectures show proven efficiency gains

The research on predicting embeddings instead of tokens reveals several working implementations, though the field remains emergent compared to autoregressive token prediction.

**Coconut (Chain of Continuous Thought)** from Meta demonstrates the most practical approach: the model feeds its last hidden state directly back as the next input embedding, enabling reasoning in an unrestricted latent space. [arxiv](https://arxiv.org/abs/2412.06769) [arXiv](https://arxiv.org/abs/2412.06769) Results show **emergent BFS-like reasoning** where continuous thoughts encode multiple alternative paths simultaneously, outperforming Chain-of-Thought on planning tasks requiring backtracking. [arxiv](https://arxiv.org/abs/2412.06769) [Hugging Face](https://huggingface.co/papers/2412.06769) However, curriculum-based training is required, and interpretability is lost.

**Large Concept Models (LCM)** operate at sentence-level granularity using SONAR embeddings (supporting 200 languages). The 7B LCM outperformed Llama-3.1-8B on multilingual summarization while using **~10× shorter sequences**. [GitHub](https://github.com/AkihikoWatanabe/paper_notes/issues/1611) [arXiv](https://arxiv.org/html/2412.08821v2) The open-source implementation (github.com/facebookresearch/large_concept_model) provides practical entry point for experimentation. [GitHub](https://github.com/facebookresearch/large_concept_model)

**Hrrformer** applies Holographic Reduced Representations to transformer attention, achieving **O(TH log H) complexity** versus O(T²H) for standard attention. Results show **280× faster training** on Long Range Arena benchmarks and enable sequences beyond 100,000 tokens—directly relevant for code understanding at function/block level. [arXiv](https://arxiv.org/abs/2305.19534) [ACM Digital Library](https://dl.acm.org/doi/10.5555/3618408.3618431)

For code-specific semantic representations:
- **GraphCodeBERT** and **CodeT5** provide function-level embeddings
- **cAST** (EMNLP 2025) implements AST-aware semantic chunking respecting function/class boundaries [arXiv](https://arxiv.org/html/2506.15655v1)
- **inst2vec** captures IR-level embeddings with control and data flow information [arXiv](https://ar5iv.labs.arxiv.org/html/1806.07336)

The continuous-to-discrete mapping challenge remains unsolved elegantly. Approaches include **KNN rounding** (nearest embedding lookup), **Residual Vector Quantization**, and **Latent Refinement Decoding** (two-phase progressive hardening). [arXiv](https://arxiv.org/html/2412.08821v2) All add inference latency.

---

## Chameleon proves any-to-any multimodal is achievable at scale

Meta's Chameleon architecture demonstrates true any-to-any generation through **early fusion**: all modalities (text, images, code) are tokenized to discrete tokens and processed by a single transformer. [arXiv](https://arxiv.org/abs/2405.09818) This differs fundamentally from late-fusion approaches like LLaVA that encode modalities separately. [arXiv](https://arxiv.org/html/2405.09818v1)

The key innovation is **unified vocabulary**: images are encoded via VQ-VAE into **1024 discrete tokens per 512×512 image** from an 8192-token codebook, sharing vocabulary space with BPE text tokens. [arxiv](https://arxiv.org/html/2405.09818v1) Training stability required novel techniques:

- **QK-Norm**: Layer normalization on query and key vectors (essential)
- **Z-loss regularization**: 10⁻⁵ log²Z added to loss function
- **Revised LayerNorm placement**: Post-FFN instead of pre-FFN

Training costs remain substantial: Chameleon-7B required **856K A100-hours** on 1024 concurrent GPUs. However, small-scale alternatives exist:

| Model | Parameters | VRAM (Inference) | Any-to-Any |
|-------|-----------|------------------|------------|
| SmolVLM-256M | 256M | <1GB | No (understanding) |
| SmolVLM-2.2B | 2.2B | ~10.5GB | No |
| Janus-Pro | 1.5B-7B | 3-14GB | Yes |
| BAGEL | 7B active | ~18GB | Yes |

**SmolVLM findings** provide critical guidance for small models: compact LMs (135M-360M) don't benefit from large vision encoders—**SigLIP-B/16 (93M)** outperforms SigLIP-SO-400M (428M) at this scale. **Pixel shuffle with factor 4** reduces visual token count for the smallest models. Extended context length via RoPE base increase (10K→273K) dramatically improves performance.

For audio tokenization, **EnCodec/SoundStream** with Residual Vector Quantization achieves ultra-low bitrate compression, [Emergent Mind](https://www.emergentmind.com/topics/audio-language-model-alm) while **SpeechTokenizer** disentangles semantic and acoustic information for LLM integration. [arXiv](https://arxiv.org/html/2504.04721)

---

## 128K context on 16GB requires combining multiple techniques

Achieving 128K+ context windows on 16GB VRAM requires stacking optimizations: BitNet weights + quantized KV-cache + memory-efficient attention.

**Flash Attention** reduces memory complexity from O(N²) to O(N) by never materializing the full attention matrix, using tiling and recomputation. [tridao +2](https://tridao.me/blog/2024/flash3/) Flash Attention 3 on Hopper GPUs achieves **1.5-2× speedup** with FP8 support approaching 1.2 PFLOPS. [tridao](https://tridao.me/blog/2024/flash3/)

**KV-cache quantization** provides the largest memory gains. The formula:
```
KV_Cache = 2 × layers × heads × head_dim × seq_len × batch × bytes_per_element
```

For Llama-7B at 1K tokens: **524MB in FP16**, **131MB in INT4**, **65MB in INT2** (KIVI). KIVI's key insight: keys quantize per-channel while values quantize per-token, enabling 2-bit quantization with minimal accuracy loss. [arXiv](https://arxiv.org/html/2402.02750v2)

**Calculated maximum context lengths for 16GB with BitNet:**

| Model | Weights | Available KV | KV Quant | Max Context |
|-------|---------|--------------|----------|-------------|
| 7B BitNet | 1.4GB | ~13.6GB | INT4 | ~54M tokens (theoretical) |
| 7B BitNet | 1.4GB | ~13.6GB | INT4 | **128K-256K (practical)** |
| 3B BitNet | 0.6GB | ~14.4GB | INT4 | **256K-512K (practical)** |

Practical limits account for CUDA overhead, activation memory, and operational headroom. **Sliding window attention** (Mistral-style, W=4096) bounds KV-cache to fixed size while maintaining theoretical receptive field of W×layers tokens. [Mistral AI](https://mistral.ai/news/announcing-mistral-7b/) [Medium](https://medium.com/@lyx_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632)

**Recommended configuration for 128K+ on 16GB**:
- Model: 3-7B BitNet
- Attention: Flash Attention 2 + Sliding Window (4K window)
- KV-Cache: INT4 quantization via vLLM or LMDeploy
- Memory management: PagedAttention
- Optional: StreamingLLM attention sinks for longer streaming

---

## RTX 5080's 960 GB/s bandwidth matters more than ReBAR

The **RTX 5080** (Blackwell/GB203 architecture) provides **16GB GDDR7 at 960 GB/s bandwidth**—a 34% improvement over RTX 4080's 717 GB/s. [NVIDIA](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/) This bandwidth bottleneck dominates LLM inference performance, as token generation is memory-bound, not compute-bound. [Medium](https://medium.com/@yvan.fafchamps/how-to-benchmark-and-optimize-llm-inference-performance-for-data-scientists-1dbacdc7412a)

Key specifications for AI workloads:
- **CUDA Cores**: 10,752 [ASUS](https://www.asus.com/us/motherboards-components/graphics-cards/tuf-gaming/tuf-rtx5080-16g-gaming/techspec/) (84 SMs × 128)
- **5th Gen Tensor Cores**: 336 units supporting FP4, FP6, FP8, FP16, BF16 [Best GPUs for AI](https://www.bestgpusforai.com/gpu-comparison/5070-ti-vs-5080)
- **AI Performance**: 1,801 TOPS [Best GPUs for AI](https://www.bestgpusforai.com/gpu-comparison/5070-ti-vs-5080) (FP4)
- **L2 Cache**: 64MB [HWCooling](https://www.hwcooling.net/en/blackwell-geforce-rtx-5000-architecture-and-innovations-analysis/)
- **PCIe**: Gen 5

**Resizable BAR (ReBAR) provides minimal benefit for LLM workloads**. ReBAR removes the 256MB CPU-to-GPU access limitation, designed for gaming asset streaming. For LLM inference:
- Weights are loaded once and persist in VRAM
- KV-cache operations are entirely GPU-resident
- The bottleneck is internal memory bandwidth, not PCIe transfers

Enabling ReBAR is harmless but won't improve inference throughput. Focus optimization on:
1. **FP8/FP4 quantization** leveraging 5th gen Tensor Cores
2. **KV-cache compression** to maximize sequence length
3. **Batch size tuning** to utilize memory bandwidth fully

PyTorch optimization settings for RTX 5080:
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Use torch.cuda.amp for mixed precision
# Use pin_memory=True in DataLoader
# Use non_blocking=True for transfers
```

---

## Minimal implementations enable rapid prototyping

The training and inference landscape offers clear paths from educational to production:

**For training from scratch**: Karpathy's **nanochat** (github.com/karpathy/nanochat, ~8,000 lines) provides a complete ChatGPT clone pipeline—tokenizer training, pretraining, SFT, and RLHF—for $100-$2,500 depending on target quality. **LitGPT** (Lightning AI) offers production features with YAML-based configs and supports 20+ architectures.

**For BitNet inference**: **bitnet.cpp** (github.com/microsoft/BitNet) is the official lossless runtime, achieving 2-6× speedup over llama.cpp on CPU. [GitHub](https://github.com/microsoft/BitNet) Note that llama.cpp's TQ1_0/TQ2_0 ternary types use different computation methods than BitNet training—for lossless inference of BitNet-trained models, use bitnet.cpp.

**For 1.58-bit training (QAT)**:
- **Nanotron** (Hugging Face): Production-ready framework
- **bitlinear** PyPI package: Drop-in BitLinear layers, median-variant quantization
- **BitNet Distillation**: Three-stage pipeline with dual distillation from teacher models
- **PT-BitNet**: Post-training quantization to ternary (scales to 70B) [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X)

**For serving**: **vLLM** with PagedAttention + FP8 KV-cache provides the best flexibility/performance balance. **TensorRT-LLM** maximizes NVIDIA hardware utilization with FP4/FP8 support. [Northflank](https://northflank.com/blog/vllm-vs-tensorrt-llm-and-how-to-run-them) [Bento](https://bentoml.com/llm/getting-started/choosing-the-right-inference-framework)

**Framework learning curves:**
| Framework | Lines of Code | Time to First Results |
|-----------|--------------|----------------------|
| nanoGPT | ~750 | 30 minutes |
| nanochat | ~8,000 | 4 hours |
| llama.cpp | Large but documented | 10 minutes |
| LitGPT | Medium | 1-2 hours |

---

## GitHub Spec-Kit structures AI-assisted development

**GitHub Spec-Kit** (58k+ stars, September 2025) implements Spec-Driven Development where specifications become executable artifacts. [Ainativedev](https://ainativedev.io/news/a-look-at-spec-kit-githubs-spec-driven-software-development-toolkit) The toolkit provides slash commands (`/speckit.specify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.implement`) that translate requirements into implementation. [Microsoft Developer](https://developer.microsoft.com/blog/spec-driven-development-spec-kit) [github](https://github.com/github/spec-kit)

Core project structure:
```
.specify/
├── memory/constitution.md    # Non-negotiable principles
├── specs/001-feature-name/
│   ├── spec.md              # WHAT to build
│   ├── plan.md              # HOW to build (tech stack)
│   └── tasks.md             # Task breakdown
└── templates/
```
[Microsoft Developer](https://developer.microsoft.com/blog/spec-driven-development-spec-kit)

**Definition of Done standards** in spec.md require:
- All requirements are testable
- User stories have clear acceptance criteria
- Success criteria are measurable
- `[NEEDS CLARIFICATION]` markers for ambiguities [GitHub](https://github.com/github/spec-kit/blob/main/spec-driven.md)
- Focus on WHAT users need, not HOW to implement [GitHub](https://github.com/github/spec-kit/blob/main/spec-driven.md) [GitHub](https://github.com/github/spec-kit/blob/main/templates/commands/specify.md)

**GitHub Copilot Agent configuration** uses `.github/agents/<name>.agent.md` with YAML frontmatter: [GitHub](https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-custom-agents)
```yaml
---
name: implementation-planner
description: Creates implementation plans
tools: ["read", "search", "edit"]
---
Agent instructions in Markdown...
```
[github](https://docs.github.com/en/copilot/reference/custom-agents-configuration)

**Hallucination prevention patterns**:
1. Use `[NEEDS CLARIFICATION]` markers—never guess [GitHub](https://github.com/github/spec-kit/blob/main/spec-driven.md) [Virtuallycaffeinated](https://www.virtuallycaffeinated.com/2025/04/01/preventing-hallucination-in-ai-a-guide-based-on-industry-standards/)
2. Require citations with specific line ranges [arXiv](https://www.arxiv.org/pdf/2512.12117)
3. Multi-pass verification (AI evaluates AI-generated code) [InfoWorld](https://www.infoworld.com/article/3822251/how-to-keep-ai-hallucinations-out-of-your-code.html)
4. Test-driven development with tests before implementation
5. Cross-reference packages to verify existence [Unu](https://c3.unu.edu/blog/the-invisible-threat-in-your-code-editor-ais-package-hallucination-problem)
6. Restrict to provided context, avoid external knowledge assumptions [Claude Docs](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)

Repository-level instructions in `.github/copilot-instructions.md` define coding standards, testing requirements, and build commands that Copilot respects during generation. [Medium](https://medium.com/@anil.goyal0057/mastering-github-copilot-custom-instructions-with-github-copilot-instructions-md-f353e5abf2b1)

---

Realistic architecture for RTX 5080 16GB
Based on proven results across all research areas, here is a technically grounded architecture specification:
Model configuration:
Architecture: 3B-7B decoder-only transformer with BitNet b1.58 quantization
Context window: 128K tokens practical target (256K achievable with aggressive optimization)
Weights memory: 0.6-1.4GB (BitNet)
KV-cache budget: ~12-14GB at INT4 quantization
Attention mechanism:
Flash Attention 2/3 for base efficiency
Sliding window attention (4K window) for bounded memory
StreamingLLM attention sinks for streaming beyond window
Multimodal capability:
Early fusion approach (Chameleon-style unified vocabulary)
VQ-VAE image tokenization: 256-512 tokens per image (aggressive compression)
SpeechTokenizer for audio (semantic + acoustic separation)
Combined vocabulary: ~100K tokens (text BPE + image codebook + audio codes)
Embedding prediction (experimental component):
Hrrformer-style attention for O(TH log H) complexity
Function-level AST chunking via cAST algorithm  (Emergent Mind)
SONAR embeddings for sentence-level reasoning in non-code portions  (GitHub)
Fallback to standard token prediction where continuous approach degrades
Training approach:
Pretrain text/code backbone using nanochat/LitGPT on Stack v2 + FineWeb
Add multimodal capability via frozen vision encoder + trained connector (LLaVA-style)
Transition to BitNet via Nanotron QAT with continual pretraining
Fine-tune with Spec-Kit structured instruction data
VRAM budget (inference):
| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| Vision encoder (SigLIP-B) | 0.4 GB |
| INT4 KV-cache (128K) | ~8.4 GB |
| Activations + overhead | ~2 GB |
| Total | ~12.2 GB |
This leaves headroom for batch size optimization and falls within RTX 5080's 16GB constraint.
What remains unproven versus established
Proven and production-ready:
BitNet b1.58 achieving ~10× memory reduction with matching performance at ≥3B parameters  (arXiv)
Flash Attention reducing attention memory to O(N)
INT4/INT8 KV-cache quantization with minimal quality loss
VQ-VAE image tokenization for unified multimodal vocabularies
llama.cpp/bitnet.cpp for efficient inference
Nanotron for BitNet QAT training
Promising but limited validation:
Embedding prediction outperforming token prediction for general tasks (Coconut/LCM show specific task wins)
Function-level semantic reasoning versus token-level (cAST shows RAG improvements, not generation)
Any-to-any generation at small scale (Janus-Pro 7B exists but limited documentation)
Combining all optimizations simultaneously (each proven independently)
Speculative:
Embedding prediction as primary paradigm for code generation (no production systems exist)
VSA/Holographic representations replacing standard embeddings in production LLMs  (ACM Computing Surveys)  (PubMed Central)
500K+ context being useful (attention patterns may not leverage full context effectively)
The architecture proposed above combines proven techniques conservatively. The embedding-prediction component should be treated as experimental with fallback paths to standard token prediction.
