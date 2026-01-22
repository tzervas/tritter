# License-clean datasets for BitNet b1.58 "Tritter" training

Training a 3-7B ternary quantization model for Python/Rust AI/ML engineering requires **high-quality, deduplicated, permissively-licensed data** across multiple modalities. This report identifies production-ready datasets with verified licenses suitable for commercial use, organized by priority and immediate usability.

## The license landscape fundamentally constrains your options

**Most popular instruction-tuning datasets are commercially unusable** due to GPT-3.5/GPT-4 generation. OpenAI's Terms of Service restrict commercial use of outputs, making datasets like Magicoder, [Hugging Face](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) CodeAlpaca, WizardCoder training data, and OpenCodeInterpreter problematic for commercial training. The cleanest path forward relies on BigCode's self-generated datasets, Glaive's proprietary generation system, and raw permissively-licensed code from The Stack.

For function-level semantic understanding, no single dataset perfectly matches all requirements—you'll need to combine CodeSearchNet for documentation pairs, CodeXGLUE for summarization, and extract functions from The Stack v2 using tree-sitter parsing.

---

## Python code datasets: your primary training foundation

### The Stack v2 Python subset (production-ready)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/the-stack-v2-dedup` |
| **License** | Permissive only (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC) |
| **Size** | ~67TB total, Python subset available via `data_dir="data/python"` |
| **Token Estimate** | ~30-50B Python tokens |
| **Quality** | Near-deduplicated (MinHash 256 permutations, 0.85 Jaccard), PII removed |
| **Commercial Use** | ✅ Yes |

The Stack v2 uses Software Heritage blob IDs requiring AWS credentials for content retrieval from `s3://softwareheritage`. Each file includes `gha_license_id` metadata for filtering. License detection achieves ~90% accuracy through go-license-detector combined with GitHub Archive metadata.

### Stack-Edu Python (highest educational quality)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `HuggingFaceTB/stack-edu` (subset: "Python") |
| **License** | Per-file permissive licensing |
| **Size** | **25.3M Python files, ~15-20B tokens** [huggingface](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| **Quality** | Classifier-filtered for educational quality (score 3-5/5) [huggingface](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| **Commercial Use** | ✅ Yes |

This represents the highest-quality Python code subset, trained on Llama3-70B-Instruct annotations and filtered via StarEncoder-based classifier. [huggingface](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) **Recommended as primary Python pretraining source** for a 3-7B model where data quality matters more than quantity.

### StarCoderData Python

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/starcoderdata` |
| **Size** | 783GB total, ~250B tokens across 86 languages |
| **License** | Permissive (follows The Stack v1.2 methodology) |
| **Includes** | Python code + 13GB Jupyter notebooks + 54GB GitHub Issues + 32GB commits |
| **Commercial Use** | ✅ Yes |

```python
from datasets import load_dataset
ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train")
```

### Jupyter notebook datasets for code+markdown pairs

| Dataset | HuggingFace Path | Size | License Status |
|---------|------------------|------|----------------|
| GitHub Jupyter Text-Code Pairs | `codeparrot/github-jupyter-text-code-pairs` | 451,662 pairs | ✅ Permissive |
| BigCode Jupyter Code-Text Pairs | `bigcode/jupyter-code-text-pairs` | Part of StarCoderData | ✅ Permissive |
| GitHub Jupyter Code-to-Text | `codeparrot/github-jupyter-code-to-text` | ~59K examples | ✅ With metadata |

**Note**: Kaggle-derived notebook datasets (`HuggingFaceTB/issues-kaggle-notebooks`) require Kaggle ToS verification for commercial use.

### Python instruction tuning (license-clean options)

| Dataset | HuggingFace Path | License | Commercial Safe |
|---------|------------------|---------|-----------------|
| BigCode Self-OSS-Instruct | `bigcode/self-oss-instruct-sc2-exec-filter-500k-raw` | ✅ Uses StarCoder2 | ✅ Yes |
| Glaive-Code-Assistant-v3 | `glaiveai/glaive-code-assistant-v3` | Apache-2.0 | ✅ Likely (verify generation method) |
| Python Stack Functions | `bigcode/python-stack-v1-functions-filtered` | Permissive | ✅ Yes |

**⚠️ AVOID for commercial use**: Magicoder-OSS-Instruct-75K, Magicoder-Evol-Instruct-110K, CodeAlpaca-20K, WizardCoder training data—all GPT-generated with OpenAI ToS restrictions.

---

## Rust code datasets: upsampling required for low-resource language

### The Stack v2 Rust subset

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/the-stack-v2-dedup` (filter by language) |
| **License** | Permissive only |
| **Access** | Requires Software Heritage agreement |
| **Commercial Use** | ✅ Yes |

### Stack-Edu Rust (highest quality)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `HuggingFaceTB/stack-edu` (subset: "Rust") |
| **Size** | **1.14M files, ~1.75B tokens** [huggingface](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| **Quality** | Educational quality filtered (3-5/5 score) [huggingface](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| **Commercial Use** | ✅ Yes |

### Strandset-Rust-v1 (primary Rust instruction dataset)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `Fortytwo-Network/Strandset-Rust-v1` |
| **License** | **Apache 2.0** ✅ [huggingface](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1) |
| **Size** | **191,008 examples** across 15 task categories [huggingface](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1) |
| **Quality** | Swarm inference peer-reviewed validation |
| **Tasks** | Code generation, bug detection, refactoring, optimization, documentation, testing [huggingface](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1) |
| **Commercial Use** | ✅ Yes |

This is the most comprehensive permissively-licensed Rust instruction dataset, covering ownership, lifetimes, and trait system concepts critical for Rust expertise. [Hugging Face](https://huggingface.co/Fortytwo-Network/Strand-Rust-Coder-14B-v1) [Hugging Face](https://huggingface.co/blog/Fortytwo-Network/strand-rust-coder-tech-report)

### Rust documentation sources (require custom scraping)

| Resource | License | Notes |
|----------|---------|-------|
| The Rust Book | Apache-2.0/MIT | Official at doc.rust-lang.org/book/ [The Rust Programming Language](https://doc.rust-lang.org/book/) |
| Rust by Example | Apache-2.0/MIT | Code-heavy examples |
| The Rustonomicon | Apache-2.0/MIT | Unsafe Rust guide [Rust Programming Language](https://rust-lang.org/learn/) |
| Standard library docs | Apache-2.0/MIT | API documentation |

**Gap identified**: No pre-built docs.rs dataset exists. Creating one requires custom scraping with license filtering per-crate.

### Cross-language Rust-Python interop

For PyO3 examples, collect code from:
- `PyO3/pyo3` (Apache-2.0/MIT)
- `PyO3/rust-numpy` (BSD)
- HuggingFace tokenizers (Rust backend)
- tiktoken (OpenAI's BPE tokenizer)

No formal dataset exists—requires custom collection from these repositories.

---

## Function-level semantic understanding datasets

These datasets directly support the embedding-prediction at function/code-block semantic level requirement.

### CodeSearchNet (function + docstring pairs)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `code-search-net/code_search_net` |
| **Size** | ~2M function-docstring pairs |
| **Languages** | Python, JavaScript, Ruby, Go, Java, PHP (no Rust) |
| **Granularity** | **Function-level** |
| **License** | ⚠️ Per-repository (verify individually) |

**Format per example**:
- `func_name`, `whole_func_string`, `func_code_string`
- `func_documentation_string`, `func_code_tokens`

For Apache-2.0 licensed variants: `Nan-Do/code-search-net-python`, `Nan-Do/code-search-net-java`

### CodeXGLUE Code Summarization (C-UDA licensed)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `google/code_x_glue_*` |
| **License** | **C-UDA** (Computational Use of Data Agreement) ✅ |
| **Python Size** | 251,820 train / 13,914 dev / 14,918 test |
| **Granularity** | Function-level |
| **Commercial Use** | ✅ Yes |

**Recommended for function summarization training**—filtered CodeSearchNet subset with C-UDA permissive license.

### Scotch Dataset (19M functions)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `Samip/Scotch` |
| **License** | MIT |
| **Size** | ~19M functions; 3.2M with docstrings |
| **Languages** | Python, Java, JavaScript, Go |
| **Commercial Use** | ✅ Yes |

### AST-parsed datasets

| Dataset | Path | License | Languages |
|---------|------|---------|-----------|
| Py_AST | `py_ast` | BSD-2-Clause/MIT | Python |
| MLCPD | `jugalgajjar/MultiLang-Code-Parser-Dataset` | Academic | 10 languages |
| Code2Seq Data | S3 download | MIT (code) | Java, Python |

For function-level extraction from raw code, use **tree-sitter** parsing on The Stack v2 data.

### Clone detection for semantic similarity

| Dataset | Path | License | Commercial |
|---------|------|---------|------------|
| POJ-104 | `google/code_x_glue_cc_clone_detection_poj104` | C-UDA | ✅ Yes |
| BigCloneBench | Original source | CC-BY-NC-ND | ❌ No |

**Use POJ-104** (52K programs across 104 problems) for semantic similarity training—BigCloneBench is non-commercial. [Hugging Face](https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_poj104)

---

## AI/ML domain knowledge datasets

### Papers With Code (paper-to-implementation links)

| Field | Details |
|-------|---------|
| **HuggingFace Paths** | `pwc-archive/papers-with-abstracts`, `pwc-archive/links-between-paper-and-code` |
| **License** | **CC-BY-SA 4.0** ✅ |
| **Size** | 65K+ papers with code, daily updates |
| **Commercial Use** | ✅ Yes (with attribution + share-alike) |

### S2ORC (Semantic Scholar)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `allenai/s2orc` |
| **License** | **ODC-By 1.0** ✅ (current version permissive) |
| **Size** | 81.1M papers metadata; 8.1M with full text |
| **Commercial Use** | ✅ Yes (verify underlying paper licenses) |

### ML framework documentation (create custom datasets)

| Framework | License | Source |
|-----------|---------|--------|
| PyTorch | BSD-3 | github.com/pytorch/pytorch |
| TensorFlow | Apache-2.0 | github.com/tensorflow/docs |
| HuggingFace Transformers | Apache-2.0 | github.com/huggingface/transformers |
| scikit-learn | BSD-3 | github.com/scikit-learn/scikit-learn |
| JAX/Flax | Apache-2.0 | github.com/google/jax |

**Token estimate for combined documentation**: ~100M tokens

### ArXiv ML papers

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `CShorten/ML-ArXiv-Papers` |
| **License** | AFL-3.0 (Academic Free License) |
| **Size** | ~118K papers (titles + abstracts only) |
| **Limitation** | Does NOT include full paper text |

---

## Multimodal datasets for any-to-any capability

**Critical finding**: Truly multimodal code datasets with permissive licenses are extremely rare. Most require synthetic data generation.

### Screenshot-to-code datasets

| Dataset | HuggingFace Path | License | Size |
|---------|------------------|---------|------|
| WebSight | `HuggingFaceM4/WebSight` | Synthetic/Permissive | 2M HTML-screenshot pairs |
| WebCode2M | webcode2m.github.io | CC-BY-4.0 | 2M+ real webpage-code pairs |

### Interleaved text-image datasets

| Dataset | HuggingFace Path | License | Size |
|---------|------------------|---------|------|
| OBELICS | `HuggingFaceM4/OBELICS` | **CC-BY-4.0** ✅ | 141M docs, 353M images |

OBELICS provides the foundation for general multimodal training—filter for technical content containing code keywords.

### Document understanding

| Dataset | HuggingFace Path | License |
|---------|------------------|---------|
| DocVQA | `lmms-lab/DocVQA` | Apache-2.0 ✅ |
| Google Screen Annotation | `google-research-datasets/screen_annotation` | CC BY 4.0 ✅ |

### Critical gaps requiring synthetic generation

1. **Code-to-diagram pairs**: No large-scale UML/architecture diagram datasets exist
2. **Matplotlib/visualization code + rendered output**: Execute code to capture images
3. **Voice-to-code**: No existing datasets
4. **Jupyter notebooks with inline images**: Mine from GitHub/Kaggle
5. **IDE screenshots + code**: Generate synthetically with various themes

---

## Commit and code review datasets

### CommitPack (largest permissive commit dataset)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/commitpack` |
| **License** | Permissive (MIT, Apache, BSD per-repo) |
| **Size** | **4TB of commits** from 350 programming languages |
| **Content** | Old file, new file, commit message, diff |
| **Commercial Use** | ✅ Yes |

**Highly recommended** for teaching the model to understand code changes and write commit messages.

### CodeReviewer (Microsoft)

| Field | Details |
|-------|---------|
| **Source** | Zenodo + microsoft/codereviewer |
| **License** | Apache-2.0 |
| **Tasks** | Code change quality estimation, comment generation, code refinement |
| **Commercial Use** | ✅ Yes |

### Stack Overflow considerations

Stack Overflow data uses **CC-BY-SA** license—commercial use is permitted but derivatives must use the same license. [FossID](https://fossid.com/articles/open-source-software-license-risks-copying-code-stack-overflow/) Consider this carefully for your use case. CoNaLa dataset (`conala-corpus.github.io`) provides curated Python code/NL pairs but inherits CC-BY-SA restrictions.

---

## Quality filtering pipeline recommendations

Based on StarCoder2, DeepSeek-Coder, and RedPajama approaches:

### Recommended filtering order

**Phase 1: Pre-Processing (cheap, run first)**
1. Language detection (go-enry)
2. License filtering (permissive only via go-license-detector)
3. Extension filtering (remove data files, configs)
4. Basic length filters (50-100K lines, <100 avg line length)
5. Exact deduplication (SHA256 hash)

**Phase 2: Quality Filtering**
6. Near-deduplication (MinHash LSH, 5-grams, **Jaccard=0.7**)
7. Language-specific filters (HTML: 20% visible text; JSON/YAML: 50-5000 chars)
8. Auto-generated file detection (lock files, minified JS)
9. Comment ratio check (5-30% optimal)

**Phase 3: Safety & Compliance**
10. PII detection (StarPII model or similar NER)
11. Secret/credential scanning (regex + ML, use GitLeaks/detect-secrets)
12. Malware signature matching
13. Opt-out compliance

**Phase 4: Final Processing**
14. Benchmark decontamination (remove HumanEval, MBPP, DS-1000 samples) [arXiv](https://arxiv.org/pdf/2312.02120)
15. Optional: SemDeDup for 20-50% additional semantic deduplication

### Expected data retention rates

| Stage | Typical Retention |
|-------|-------------------|
| License filtering | 30-50% |
| Basic filters | 80-90% |
| Exact dedup | 90-95% |
| Near-dedup | 55-70% |
| Quality filters | 70-85% |
| **Overall** | **15-30%** |

### Tools stack

- **Deduplication**: allenai/duplodocus (Rust-based), NVIDIA NeMo Curator (GPU)
- **Language detection**: go-enry
- **License detection**: go-license-detector
- **PII**: StarPII model from BigCode
- **Secrets**: GitLeaks, detect-secrets, TruffleHog
- **AST parsing**: tree-sitter (multi-language)

---

## Recommended data mix for Tritter (3-7B model)

### Phase 1: Base pretraining (1-2T tokens)

| Component | Percentage | Tokens | Primary Sources |
|-----------|------------|--------|-----------------|
| Python Code | 45% | 450-900B | Stack-Edu Python, The Stack v2 |
| Rust Code | 20% | 200-400B | Stack-Edu Rust, The Stack v2 |
| Other Languages | 10% | 100-200B | JavaScript, C++, Go (transfer learning) |
| Code Documentation | 10% | 100-200B | Framework docs, docstrings |
| GitHub Issues/PRs/Commits | 5% | 50-100B | CommitPack, StarCoderData issues |
| Natural Language | 8% | 80-160B | Wikipedia, ArXiv abstracts |
| Math/Reasoning | 2% | 20-40B | OpenWebMath |

**Key insight from Phi research**: For 3-7B models, **data quality beats quantity**. [Mbrenndoerfer](https://mbrenndoerfer.com/writing/phi-models-textbook-quality-data) Stack-Edu's "textbook quality" filtering is more valuable than raw volume.

### Phase 2: AI/ML domain continued pretraining (200-500B tokens)

| Component | Percentage | Tokens |
|-----------|------------|--------|
| ML Framework Code | 30% | 60-150B |
| ML Paper Implementations | 20% | 40-100B |
| Kaggle Notebooks | 15% | 30-75B |
| General Code Replay | 20% | 40-100B |
| ML Documentation | 10% | 20-50B |
| Natural Language Replay | 5% | 10-25B |

### Phase 3: Instruction tuning (2-5M samples)

| Component | Percentage | Samples | Source |
|-----------|------------|---------|--------|
| OSS-Instruct | 30% | 600K-1.5M | `bigcode/self-oss-instruct-sc2-*` |
| Glaive Code Assistant | 35% | 700K-1.75M | `glaiveai/glaive-code-assistant-v3` |
| AI/ML Specific | 20% | 400K-1M | Domain-filtered from above |
| Code Replay | 10% | 200K-500K | Sample from pretraining |
| NL Replay | 5% | 100K-250K | Prevent forgetting |

### Phase 4: Alignment (50-100K preference pairs)

| Component | Percentage | Pairs |
|-----------|------------|-------|
| Code Correctness | 50% | 25-50K |
| Code Quality/Style | 30% | 15-30K |
| Safety | 15% | 7.5-15K |
| General Capabilities | 5% | 2.5-5K |

---

## Long context (128K) training strategy

### Repository-level data organization

Follow DeepSeek-Coder's approach:
1. Parse file dependencies within repositories
2. Rearrange files based on import/dependency order
3. Concatenate dependent files into single training examples
4. Apply repo-level deduplication

### Progressive context extension

| Stage | Context Length | Data Requirement |
|-------|----------------|------------------|
| Base pretraining | 8K | Standard data |
| Extension 1 | 32K | Long documents, repos |
| Extension 2 | 64K | Multi-file repos |
| Extension 3 | 128K | Full project contexts |

**Use YaRN-based RoPE scaling** for position interpolation during extension phases.

---

## Concrete dataset download list (priority order)

### Tier 1: Immediately ready (start here)

| Dataset | Path | Size | License |
|---------|------|------|---------|
| Stack-Edu Python | `HuggingFaceTB/stack-edu` | 25.3M files | ✅ Permissive |
| Stack-Edu Rust | `HuggingFaceTB/stack-edu` | 1.14M files | ✅ Permissive |
| Strandset-Rust-v1 | `Fortytwo-Network/Strandset-Rust-v1` | 191K examples | ✅ Apache-2.0 |
| Self-OSS-Instruct | `bigcode/self-oss-instruct-sc2-exec-filter-500k-raw` | 500K | ✅ StarCoder2 |
| Glaive-Code-Assistant-v3 | `glaiveai/glaive-code-assistant-v3` | ~1M | ✅ Apache-2.0 |
| CommitPack | `bigcode/commitpack` | 4TB | ✅ Permissive |
| CodeXGLUE Summarization | `google/code_x_glue_*` | ~300K | ✅ C-UDA |
| Papers With Code | `pwc-archive/*` | 65K+ papers | ✅ CC-BY-SA |

### Tier 2: Requires processing

| Dataset | Path | Processing Needed |
|---------|------|-------------------|
| The Stack v2 | `bigcode/the-stack-v2-dedup` | SWH agreement, S3 download |
| OBELICS | `HuggingFaceM4/OBELICS` | Filter for technical content |
| WebCode2M | webcode2m.github.io | Download and process |
| S2ORC | `allenai/s2orc` | Filter for ML papers |

### Tier 3: Create custom datasets

| Target | Source | Method |
|--------|--------|--------|
| PyO3 interop examples | GitHub PyO3 ecosystem | Scrape with license filter |
| Rust documentation | doc.rust-lang.org | Clone + parse markdown |
| ML framework docs | Framework GitHub repos | RST/MD extraction |
| Code-to-diagram | Execute Mermaid/PlantUML | Synthetic generation |
| Visualization code + output | Matplotlib/seaborn code | Execute and capture

Known limitations and gaps
Rust function-level datasets are scarce—extract from The Stack v2 using tree-sitter
No comprehensive PyO3 dataset—requires manual collection
No voice-to-code datasets exist—must create synthetically via TTS
BigCloneBench is non-commercial—use POJ-104 for clone detection
Call graph datasets don't exist pre-built—generate using pyan (Python) or callGraph (Rust)
Jupyter notebooks with inline images need mining—no curated dataset exists
Most high-quality instruction data is GPT-contaminated—rely on BigCode self-generated and Glaive
For a 3-7B model targeting RTX 5080 16GB with 128K context and multimodal capability, prioritize Stack-Edu quality over raw token volume, aggressively upsample Rust (2-3x natural distribution), and plan synthetic generation pipelines for multimodal gaps that cannot be filled with existing datasets.
