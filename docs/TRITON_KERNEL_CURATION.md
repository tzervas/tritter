# Triton GPU Kernel Dataset Curation Guide

**Issue**: #58 - Triton GPU kernel dataset curation
**Status**: Research & Documentation
**Last Updated**: 2026-01-23

This document provides exact file paths, directory structures, and extraction strategies for curating Triton GPU kernels from permissively-licensed repositories. All repositories listed have permissive licenses suitable for commercial training.

---

## Executive Summary

Triton kernel code exists across five primary sources:

| Repository | License | Estimated Tokens | Primary Use |
|------------|---------|------------------|-------------|
| `triton-lang/triton` | MIT | ~50-100K | Tutorial examples, reference implementations |
| `pytorch/pytorch` | BSD-3-Clause | ~500K | Production inductor kernels, compiler backend |
| `Dao-AILab/flash-attention` | BSD-3-Clause | ~100K | FlashAttention Triton implementations |
| `facebookresearch/xformers` | BSD-3-Clause | ~200K | Memory-efficient attention kernels |
| `google/jax` | Apache-2.0 | ~50-100K | Pallas GPU/TPU kernel definitions |

**Total estimated tokens**: ~900K-1M (primary source material)

**Recommended upsample ratio**: 5-10x relative to Python for training to ensure adequate Triton language learning.

---

## Repository 1: OpenAI/Triton (`triton-lang/triton`)

### Repository Information

| Field | Value |
|-------|-------|
| **URL** | https://github.com/triton-lang/triton |
| **License** | MIT |
| **Repository Type** | Language + Compiler + Documentation |
| **Primary Content** | Tutorial examples, compiler reference, test kernels |
| **Commercial Use** | ✅ Yes |

### Directory Structure

```
triton-lang/triton/
├── python/
│   ├── tutorials/                    # Tutorial kernel implementations
│   │   ├── 01-vector-add.py
│   │   ├── 02-fused-softmax.py
│   │   ├── 03-matrix-multiplication.py
│   │   ├── 04-low-memory-dropout.py
│   │   ├── 05-layer-norm.py
│   │   ├── 06-fused-attention.py     # ⭐ Flash Attention v2 reference
│   │   ├── 07-extern-functions.py
│   │   ├── 08-grouped-gemm.py
│   │   ├── 09-persistent-matmul.py
│   │   ├── 10-block-scaled-matmul.py
│   │   ├── 11-programmatic-dependent-launch.py
│   │   └── gluon/                    # Newer tutorials (Gluon DSL)
│   │       └── 01-intro.py
│   ├── triton/                       # Core compiler modules
│   │   └── [runtime, compiler, IR modules]
│   └── test/                         # Test kernels (may be useful)
├── docs/                             # Documentation
└── src/                              # C++ compiler backend
```

### File Extraction Strategy

**Glob patterns for extraction**:
```bash
# All tutorial kernels (recommended starting point)
python/tutorials/*.py

# All kernels including tests (comprehensive)
python/**/*.py | grep -E "(kernel|triton\.jit)"

# Specific focus: attention and memory kernels
python/tutorials/{06-fused-attention,04-low-memory-dropout,05-layer-norm}.py
```

### Git Clone Command

```bash
# Clone the repository
git clone https://github.com/triton-lang/triton.git

# Extract only Triton kernel files (minimal download)
cd triton
git sparse-checkout set --cone python/tutorials
git checkout main

# Verify file count
find python/tutorials -name "*.py" | wc -l
```

### Estimated Statistics

| Metric | Value |
|--------|-------|
| **Total Python files** | ~12 tutorial files + gluon/ |
| **Kernel files with `@triton.jit`** | ~11 (all tutorials) |
| **Estimated tokens** | 50-100K (tutorial-focused) |
| **Quality** | Reference implementations (educational) |
| **License verification** | MIT header in each file |

### Key Kernels to Extract

1. **06-fused-attention.py** - Flash Attention v2 reference implementation
2. **03-matrix-multiplication.py** - GEMM patterns
3. **04-low-memory-dropout.py** - Memory-efficient operations
4. **05-layer-norm.py** - Normalization kernels
5. **08-grouped-gemm.py** - Grouped operations

---

## Repository 2: PyTorch (`pytorch/pytorch`)

### Repository Information

| Field | Value |
|-------|-------|
| **URL** | https://github.com/pytorch/pytorch |
| **License** | BSD-3-Clause |
| **Repository Type** | ML Framework |
| **Primary Content** | TorchInductor Triton backend, generated kernels |
| **Commercial Use** | ✅ Yes |

### Directory Structure

```
pytorch/pytorch/
├── torch/
│   └── _inductor/
│       ├── codegen/
│       │   ├── triton.py                 # ⭐ Triton code generation engine
│       │   ├── triton_utils.py           # Triton utilities
│       │   ├── triton_combo_kernel.py    # Combined kernels
│       │   ├── triton_split_scan.py      # Scan operation kernels
│       │   ├── cpp.py                    # C++ backend (reference)
│       │   └── [other backends]
│       ├── kernel/
│       │   ├── mm_common.py              # Matrix multiplication kernels
│       │   ├── [various kernel definitions]
│       │   └── triton_*.py               # Triton-specific kernels
│       ├── config.py                     # Configuration
│       └── runtime/
│           └── [runtime support files]
└── benchmarks/
    └── [inductor benchmarks with Triton kernels]
```

### File Extraction Strategy

**Key paths for Triton kernels**:
```bash
# Primary codegen logic (must-have)
torch/_inductor/codegen/triton.py
torch/_inductor/codegen/triton_utils.py

# Specialized kernel implementations
torch/_inductor/codegen/triton_combo_kernel.py
torch/_inductor/codegen/triton_split_scan.py

# Kernel definitions
torch/_inductor/kernel/triton_*.py
torch/_inductor/kernel/mm_common.py

# All Triton-related Python files in inductor
find torch/_inductor -name "*triton*.py"
```

### Git Clone Command

```bash
# Clone only torch/_inductor directory (sparse checkout)
git clone --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
cd pytorch
git sparse-checkout set torch/_inductor
git checkout main

# Extract all Triton files
find torch/_inductor -name "*triton*.py" > triton_files.txt
wc -l triton_files.txt
```

### Estimated Statistics

| Metric | Value |
|--------|-------|
| **Total Inductor files** | ~300+ Python files |
| **Triton-specific files** | ~15-25 files (codegen + kernels) |
| **Estimated tokens** | 400-600K (production-quality kernels) |
| **Quality** | Production code (most complex) |
| **Primary content** | Kernel generation, optimization, autotuning |

### Key Files to Extract (Priority Order)

1. **triton.py** - Core code generation (critical)
2. **triton_utils.py** - Utility functions
3. **triton_combo_kernel.py** - Combined kernel patterns
4. **triton_split_scan.py** - Reduction/scan operations
5. **mm_common.py** - Matrix multiplication patterns

### Generated Kernels Location

**Runtime-generated kernels** (not in repo, but documented):
- Output location: `/tmp/torchinductor_<username>/`
- File pattern: `output_code.py` containing `@triton.jit` decorated functions
- Access via: `TORCH_COMPILE_DEBUG=1` environment variable

---

## Repository 3: FlashAttention (`Dao-AILab/flash-attention`)

### Repository Information

| Field | Value |
|-------|-------|
| **URL** | https://github.com/Dao-AILab/flash-attention |
| **License** | BSD-3-Clause |
| **Repository Type** | ML Algorithm Implementation |
| **Primary Content** | FlashAttention Triton reference, CUDA kernels |
| **Commercial Use** | ✅ Yes |

### Directory Structure

```
flash-attention/
├── flash_attn/
│   ├── flash_attn_triton.py                 # ⭐ Triton implementation
│   ├── flash_attn_triton_og.py              # Original/legacy version
│   ├── flash_attn_triton_amd/               # AMD GPU variant
│   │   └── [AMD-specific Triton kernels]
│   ├── flash_attn_interface.py              # Public interface
│   ├── flash_blocksparse_attention.py       # Block-sparse variant
│   ├── bert_padding.py                      # Padding utilities
│   └── [other implementations: CUDA, etc.]
├── benchmarks/
│   ├── benchmark_flash_attention.py
│   └── [performance benchmarks]
└── tests/
    └── [unit tests]
```

### File Extraction Strategy

**Glob patterns for Triton extraction**:
```bash
# Core Triton implementations
flash_attn/flash_attn_triton*.py

# All Triton files including AMD variants
find flash_attn -path "*triton*" -name "*.py"

# Exclude non-Triton implementations
grep -r "@triton.jit" flash_attn/
```

### Git Clone Command

```bash
# Clone repository (relatively small)
git clone https://github.com/Dao-AILab/flash-attention.git

# Extract Triton files only
cd flash-attention
find . -name "*triton*.py" -type f

# Get file count and token estimate
find . -name "*triton*.py" | xargs wc -l
```

### File Details

| File | Lines | Content |
|------|-------|---------|
| `flash_attn_triton.py` | ~1000-1500 | Forward/backward kernels, attention logic |
| `flash_attn_triton_og.py` | ~800-1200 | Original implementation (historical interest) |
| `flash_attn_triton_amd/` | ~500-800 | AMD GPU optimizations |

### Estimated Statistics

| Metric | Value |
|--------|-------|
| **Total files** | 3-5 main Triton files |
| **Estimated tokens** | 80-150K (kernels + utilities) |
| **Quality** | Research-grade, well-documented |
| **Primary content** | Attention mechanisms, memory optimization |
| **License verification** | BSD-3-Clause header |

### Key Kernels to Extract

1. **flash_attn_triton.py** - Primary forward/backward kernels
2. **flash_attn_triton_amd/** - Hardware-specific optimizations
3. **flash_blocksparse_attention.py** - Sparse attention patterns

---

## Repository 4: xFormers (`facebookresearch/xformers`)

### Repository Information

| Field | Value |
|-------|-------|
| **URL** | https://github.com/facebookresearch/xformers |
| **License** | BSD-3-Clause |
| **Repository Type** | ML Library |
| **Primary Content** | Optimized transformer operations |
| **Commercial Use** | ✅ Yes |

### Directory Structure

```
xformers/
├── xformers/
│   └── ops/
│       ├── fmha/                          # Flash Multi-Head Attention
│       │   ├── triton_splitk.py           # ⭐ Split-K optimization
│       │   ├── _triton/
│       │   │   ├── splitk_kernels.py      # Kernel implementations
│       │   │   ├── flash.py               # Flash kernels
│       │   │   └── [other variants]
│       │   ├── ck/                        # CutlasKernel implementations
│       │   ├── flash/                     # Flash attention variants
│       │   ├── common.py                  # Common utilities
│       │   └── [other backends]
│       └── _triton/                       # General Triton ops
│           ├── [various Triton operations]
│           └── kernels.py
├── benchmarks/
│   ├── benchmark_triton_layernorm.py
│   ├── benchmark_triton_dropout.py
│   └── [other benchmarks]
└── tests/
    └── [unit tests for Triton ops]
```

### File Extraction Strategy

**Glob patterns for Triton extraction**:
```bash
# Primary attention kernels
xformers/ops/fmha/triton_splitk.py
xformers/ops/fmha/_triton/*.py

# All Triton kernels in ops
find xformers/ops -path "*_triton*" -name "*.py"

# Triton benchmarks
find benchmarks -name "*triton*.py"

# Verify Triton markers
grep -r "@triton.jit" xformers/ops/
```

### Git Clone Command

```bash
# Clone repository
git clone https://github.com/facebookresearch/xformers.git

# Extract Triton-specific files
cd xformers
find . -path "*_triton*" -name "*.py" | sort > triton_files.txt

# Get comprehensive stats
find . -path "*_triton*" -name "*.py" | xargs wc -l
```

### Directory Details

| Path | Content |
|------|---------|
| `xformers/ops/fmha/triton_splitk.py` | Split-K attention optimization |
| `xformers/ops/fmha/_triton/` | Triton kernel implementations |
| `xformers/ops/fmha/_triton/splitk_kernels.py` | Split-K kernel definitions |
| `xformers/ops/_triton/` | Generic Triton operations |

### Estimated Statistics

| Metric | Value |
|--------|-------|
| **Total Triton files** | 15-30 files |
| **Estimated tokens** | 150-300K (comprehensive ops) |
| **Quality** | Production-grade, heavily optimized |
| **Primary content** | Attention, normalization, dropout ops |
| **License verification** | BSD-3-Clause headers |

### Key Kernels to Extract

1. **triton_splitk.py** - Split-K attention optimization
2. **_triton/splitk_kernels.py** - Core kernel implementations
3. **_triton/flash.py** - Flash attention variants
4. Benchmark files for reference patterns

---

## Repository 5: JAX (`google/jax`)

### Repository Information

| Field | Value |
|-------|-------|
| **URL** | https://github.com/google/jax (or jax-ml/jax) |
| **License** | Apache-2.0 |
| **Repository Type** | ML Framework |
| **Primary Content** | Pallas GPU/TPU kernels |
| **Commercial Use** | ✅ Yes |

### Directory Structure

```
google/jax/ (or jax-ml/jax)
├── jax/
│   └── experimental/
│       └── pallas/
│           ├── triton/
│           │   ├── pallas_call_registration.py    # ⭐ Triton lowering
│           │   ├── lowering.py                    # Lowering rules
│           │   └── [other Triton-specific files]
│           ├── ops/
│           │   ├── gpu/
│           │   │   ├── [GPU-specific ops]
│           │   │   └── matmul.py                  # GEMM examples
│           │   ├── tpu/
│           │   │   └── paged_attention/
│           │   │       └── paged_attention_kernel.py  # ⭐ TPU kernel
│           │   └── [other operations]
│           ├── core/
│           │   ├── dsl.py                         # DSL definitions
│           │   └── effects.py
│           └── examples/
│               └── [reference implementations]
├── docs/
│   └── [Pallas documentation]
└── tests/
    └── [pallas tests]
```

### File Extraction Strategy

**Glob patterns for Triton extraction**:
```bash
# Triton-specific lowering rules
jax/experimental/pallas/triton/*.py

# All Pallas kernels (broader scope)
find jax/experimental/pallas -name "*.py" | grep -v "__pycache__"

# Pallas examples and tutorials
find jax/experimental/pallas -path "*example*" -name "*.py"

# Verify Triton content
grep -r "triton" jax/experimental/pallas/
```

### Git Clone Command

```bash
# Clone JAX repository (large, consider sparse checkout)
git clone --filter=blob:none --sparse https://github.com/jax-ml/jax.git
cd jax
git sparse-checkout set jax/experimental/pallas
git checkout main

# Extract Triton files
find jax/experimental/pallas -name "*.py" | head -30
wc -l jax/experimental/pallas/**/*.py
```

### Key Directories

| Path | Content |
|------|---------|
| `jax/experimental/pallas/triton/` | Triton backend lowering |
| `jax/experimental/pallas/ops/gpu/` | GPU kernel definitions |
| `jax/experimental/pallas/ops/tpu/paged_attention/` | TPU attention kernels |
| `jax/experimental/pallas/core/` | Pallas DSL and core abstractions |

### Estimated Statistics

| Metric | Value |
|--------|-------|
| **Triton files** | 10-20 files in `triton/` |
| **Pallas kernels (broader)** | 50+ kernel definitions |
| **Estimated tokens** | 50-150K (Triton-specific) / 200-400K (all Pallas) |
| **Quality** | Production-grade, highly specialized |
| **Primary content** | Triton lowering rules, kernel specifications |
| **License verification** | Apache-2.0 header |

### Note on Pallas vs Triton

- **Pallas**: High-level kernel language (DSL)
- **Triton**: Low-level GPU programming language
- **Triton backend**: JAX's Pallas can lower to Triton for GPU execution
- **For dataset**: Extract `jax/experimental/pallas/triton/*.py` for Triton-specific content

---

## Data Extraction Methodology

### Step 1: Repository Cloning

```bash
#!/bin/bash
# Create working directory
mkdir -p triton_dataset
cd triton_dataset

# Clone all repositories with sparse checkout for efficiency
repos=(
    "triton-lang/triton:python/tutorials"
    "pytorch/pytorch:torch/_inductor"
    "Dao-AILab/flash-attention:flash_attn"
    "facebookresearch/xformers:xformers/ops"
    "jax-ml/jax:jax/experimental/pallas"
)

for repo in "${repos[@]}"; do
    IFS=':' read -r github_path sparse_path <<< "$repo"
    git clone --filter=blob:none --sparse https://github.com/${github_path}.git
    cd $(basename ${github_path})
    git sparse-checkout set --cone ${sparse_path}
    git checkout main
    cd ..
done
```

### Step 2: Extract Triton Kernels

**Universal extraction pattern**:

```bash
#!/bin/bash
# Find all files containing @triton.jit decorator

find . -name "*.py" -type f -exec grep -l "@triton.jit" {} \; > triton_kernels.txt

# Get comprehensive statistics
echo "=== Triton Kernel Files ==="
wc -l $(cat triton_kernels.txt)

# Count total lines of code
xargs wc -l < triton_kernels.txt | tail -1

# Extract license headers
for file in $(cat triton_kernels.txt); do
    head -10 "$file" | grep -E "(License|Copyright|MIT|BSD|Apache)" >> licenses.txt
done
```

### Step 3: Quality Checks for Triton Code

**Checklist for each kernel file**:

```python
# Quality validation checklist
def validate_triton_kernel(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    checks = {
        "has_triton_jit": "@triton.jit" in content,
        "has_bounds_masking": "tl.where" in content or "mask=" in content,
        "has_program_id": "tl.program_id" in content,
        "has_block_size": "BLOCK_SIZE" in content or "block_" in content,
        "has_load_store": ("tl.load" in content or "tl.store" in content),
        "has_docstring": '"""' in content or "'''" in content,
        "license_permissive": any(l in content for l in
            ["MIT", "BSD", "Apache", "Apache-2.0", "BSD-3"])
    }
    return checks
```

### Step 4: Glob Patterns for Batch Extraction

```bash
# Comprehensive extraction patterns
# Pattern 1: All Triton kernels (strict)
find . -name "*.py" -exec grep -l "@triton.jit" {} \;

# Pattern 2: Attention mechanisms
find . -name "*attention*.py" -o -name "*attn*.py"

# Pattern 3: GEMM/Matrix operations
find . -name "*matmul*.py" -o -name "*gemm*.py"

# Pattern 4: Memory/normalization operations
find . -name "*norm*.py" -o -name "*dropout*.py" -o -name "*layer*.py"

# Pattern 5: All kernel definitions (broader)
find . -name "*kernel*.py"

# Pattern 6: Benchmark files (reference patterns)
find benchmarks/ -name "*.py" | grep -i triton
```

---

## License Verification Checklist

| Repository | License | Verification Method | Commercial Safe |
|------------|---------|---------------------|-----------------|
| triton-lang/triton | MIT | Header: `Copyright (c) ... MIT` | ✅ Yes |
| pytorch/pytorch | BSD-3-Clause | Header: `BSD-3-Clause` + LICENSE file | ✅ Yes |
| Dao-AILab/flash-attention | BSD-3-Clause | License.txt root | ✅ Yes |
| facebookresearch/xformers | BSD-3-Clause | License.txt root | ✅ Yes |
| google/jax | Apache-2.0 | License.txt root | ✅ Yes |

**Verification commands**:
```bash
# Check file headers
head -20 *.py | grep -i "license\|copyright"

# Check repository LICENSE
cat LICENSE

# Check SPDX identifiers
grep -r "SPDX-License-Identifier" . | head -5
```

---

## Estimated Token Counts & Extraction Summary

### Breakdown by Source

| Repository | Files | Est. Lines | Est. Tokens | Extraction Complexity |
|------------|-------|-----------|-------------|----------------------|
| triton-lang/triton | 12 | 2-3K | 50-100K | Low (tutorials) |
| pytorch/pytorch | 20-30 | 15-25K | 400-600K | Medium (sparse checkout) |
| Dao-AILab/flash-attention | 3-5 | 3-5K | 80-150K | Low (focused) |
| facebookresearch/xformers | 15-30 | 8-15K | 150-300K | Medium (many files) |
| google/jax | 10-20 | 5-10K | 50-150K | Medium (large repo) |
| **TOTAL** | **60-97** | **33-58K** | **730K-1.3M** | **Medium** |

### Final Dataset Recommendations

1. **Tier 1 (Must-Have)**:
   - All Triton tutorials (triton-lang/triton)
   - PyTorch inductor codegen (pytorch/pytorch)
   - FlashAttention kernels (Dao-AILab/flash-attention)
   - **Total**: ~550-850K tokens

2. **Tier 2 (Highly Recommended)**:
   - xFormers attention operations
   - JAX Pallas Triton lowering rules
   - **Total**: +300-450K tokens → 850-1.3M tokens

3. **Upsample Factor**: 5-10x relative to Python training data

### Example Training Data Mix

For a 3-7B model with 128K context:

| Phase | Triton Tokens | Total Tokens | Percentage |
|-------|---------------|-------------|-----------|
| Base pretraining | 50-100B | 1-2T tokens | 2.5-5% |
| Domain continued | 10-20B | 200-500B tokens | 5-10% |
| Instruction tuning | 500K-2M | 2-5M samples | Special handling |

---

## Extraction Commands (Ready to Use)

### Quick Start (All Repositories)

```bash
#!/bin/bash
# Setup
mkdir -p triton_dataset && cd triton_dataset

# 1. Triton tutorials
git clone --depth 1 https://github.com/triton-lang/triton.git
cd triton && find python/tutorials -name "*.py" > ../triton_tutorials.txt && cd ..

# 2. PyTorch Inductor
git clone --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
cd pytorch && git sparse-checkout set torch/_inductor && \
    find torch/_inductor -name "*triton*.py" > ../pytorch_triton.txt && cd ..

# 3. FlashAttention
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && find . -name "*triton*.py" > ../flash_attn_triton.txt && cd ..

# 4. xFormers
git clone --filter=blob:none --sparse https://github.com/facebookresearch/xformers.git
cd xformers && git sparse-checkout set xformers/ops && \
    find xformers/ops -path "*_triton*" -name "*.py" > ../xformers_triton.txt && cd ..

# 5. JAX Pallas
git clone --filter=blob:none --sparse https://github.com/jax-ml/jax.git
cd jax && git sparse-checkout set jax/experimental/pallas && \
    find jax/experimental/pallas -path "*triton*" -name "*.py" > ../jax_triton.txt && cd ..

# Summary
echo "=== Extraction Summary ==="
for f in *_triton.txt *_tutorials.txt; do
    echo "$f: $(wc -l < $f) files"
done

# Combine all files
cat *triton*.txt *tutorials.txt | sort -u > all_triton_files.txt
echo "Total unique files: $(wc -l < all_triton_files.txt)"
```

### Per-Repository Extraction

#### Triton Tutorials Only (Minimal)
```bash
git clone --depth 1 https://github.com/triton-lang/triton.git
cd triton
find python/tutorials -name "*.py" | xargs wc -l
```

#### PyTorch Inductor (Focused)
```bash
git clone --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
cd pytorch
git sparse-checkout set torch/_inductor
git checkout main
find torch/_inductor -name "*triton*.py"
```

---

## Next Steps for Dataset Curation

1. **Execute extraction commands** - Clone repositories and extract Triton files
2. **Validate quality** - Run bounds checking, license verification
3. **Deduplicate** - Remove identical or near-identical kernels (MinHash)
4. **Tokenize** - Convert to token sequences for training
5. **Upsample** - Apply 5-10x weighting relative to Python training data
6. **Benchmark contamination** - Check for test/benchmark data that should be held out

---

## References

- [Triton Language Documentation](https://triton-lang.org/)
- [PyTorch Inductor Documentation](https://docs.pytorch.org/docs/stable/torch.compiler_get_started.html)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [xFormers Documentation](https://facebookresearch.github.io/xformers/)
- [JAX Pallas Documentation](https://docs.jax.dev/en/latest/pallas/index.html)
- [clean-datasets.md](./clean-datasets.md) - Related Triton section

---

## Document Maintenance

**Last verified**: 2026-01-23
**Verification status**: All repository URLs and directory structures confirmed via GitHub web search
**Next review**: Quarterly (or when new major versions of dependencies released)

To update this document:
1. Verify all URLs are still accessible
2. Check for new repositories or sub-directories
3. Update token count estimates if repository structure changes
4. Verify licenses remain permissive
