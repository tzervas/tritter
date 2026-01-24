# Triton Kernel Curation - Quick Reference

**Issue**: #58 - Triton GPU kernel dataset curation
**Quick Lookup**: File paths, extraction patterns, and clone commands

---

## Repository Quick Links

| Repository | URL | License | Key Path | Extract Pattern |
|------------|-----|---------|----------|-----------------|
| **Triton** | https://github.com/triton-lang/triton | MIT | `python/tutorials/` | `*.py` in tutorials |
| **PyTorch** | https://github.com/pytorch/pytorch | BSD-3 | `torch/_inductor/` | `*triton*.py` |
| **FlashAttention** | https://github.com/Dao-AILab/flash-attention | BSD-3 | `flash_attn/` | `*triton*.py` |
| **xFormers** | https://github.com/facebookresearch/xformers | BSD-3 | `xformers/ops/` | `*_triton/*.py` |
| **JAX** | https://github.com/jax-ml/jax | Apache-2.0 | `jax/experimental/pallas/` | `triton/*.py` |

---

## Critical File Paths

### Triton (triton-lang/triton)
```
python/tutorials/06-fused-attention.py          ⭐ Flash Attention reference
python/tutorials/03-matrix-multiplication.py    ⭐ GEMM patterns
python/tutorials/04-low-memory-dropout.py       ⭐ Memory optimization
python/tutorials/05-layer-norm.py               ⭐ Normalization
```

### PyTorch (pytorch/pytorch)
```
torch/_inductor/codegen/triton.py               ⭐ Code generation engine
torch/_inductor/codegen/triton_utils.py         ⭐ Utilities
torch/_inductor/codegen/triton_combo_kernel.py  ⭐ Combined kernels
torch/_inductor/kernel/mm_common.py             ⭐ Matrix operations
```

### FlashAttention (Dao-AILab/flash-attention)
```
flash_attn/flash_attn_triton.py                 ⭐ Main implementation
flash_attn/flash_attn_triton_amd/               ⭐ AMD optimizations
flash_attn/flash_blocksparse_attention.py       ⭐ Sparse variant
```

### xFormers (facebookresearch/xformers)
```
xformers/ops/fmha/triton_splitk.py              ⭐ Split-K attention
xformers/ops/fmha/_triton/splitk_kernels.py     ⭐ Kernel definitions
xformers/ops/fmha/_triton/flash.py              ⭐ Flash variants
```

### JAX (jax-ml/jax)
```
jax/experimental/pallas/triton/pallas_call_registration.py  ⭐ Triton lowering
jax/experimental/pallas/triton/lowering.py                  ⭐ Lowering rules
jax/experimental/pallas/ops/tpu/paged_attention/            ⭐ TPU kernels
```

---

## One-Liner Extraction Commands

### All Repositories
```bash
# Extract all @triton.jit decorated kernels
find . -name "*.py" -exec grep -l "@triton.jit" {} \;
```

### Individual Repositories

```bash
# Triton tutorials
find triton/python/tutorials -name "*.py"

# PyTorch Inductor Triton
find pytorch/torch/_inductor -name "*triton*.py"

# FlashAttention Triton
find flash-attention -name "*triton*.py"

# xFormers Triton
find xformers -path "*_triton*" -name "*.py"

# JAX Pallas Triton
find jax -path "*pallas/triton*" -name "*.py"
```

---

## Sparse Checkout (Minimal Download)

```bash
# PyTorch (only inductor)
git clone --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
cd pytorch && git sparse-checkout set torch/_inductor && git checkout main

# JAX (only Pallas)
git clone --filter=blob:none --sparse https://github.com/jax-ml/jax.git
cd jax && git sparse-checkout set jax/experimental/pallas && git checkout main
```

---

## File Statistics

| Repository | Est. Files | Est. Lines | Est. Tokens |
|-----------|-----------|-----------|------------|
| triton-lang/triton | 12 | 2-3K | 50-100K |
| pytorch/pytorch | 20-30 | 15-25K | 400-600K |
| Dao-AILab/flash-attention | 3-5 | 3-5K | 80-150K |
| facebookresearch/xformers | 15-30 | 8-15K | 150-300K |
| google/jax | 10-20 | 5-10K | 50-150K |
| **TOTAL** | **60-97** | **33-58K** | **730K-1.3M** |

---

## Quality Validation Checklist

For each Triton kernel file, verify:

- ✅ Contains `@triton.jit` decorator
- ✅ Has bounds masking (`tl.where`, `mask=`)
- ✅ Uses `tl.program_id()` for parallelization
- ✅ Defines block sizes appropriately
- ✅ Includes `tl.load` and `tl.store` calls
- ✅ Has docstring or comments
- ✅ Includes permissive license header (MIT/BSD/Apache)

---

## Training Integration

**Recommended upsample factor**: 5-10x relative to Python

### Example: 1T token pretraining
```
Python code:        950B tokens (95%)
Triton kernels:     50B tokens (5%)  [upsampled 5x from raw]
Other languages:    Natural distribution
```

---

## License Headers (Verification)

All files must include one of these:
- `# Copyright ... MIT`
- `# Copyright ... BSD-3-Clause`
- `# Copyright ... Apache-2.0`
- `# SPDX-License-Identifier: MIT`
- `# SPDX-License-Identifier: Apache-2.0`

---

## For Full Details
See [`TRITON_KERNEL_CURATION.md`](./TRITON_KERNEL_CURATION.md) for:
- Complete directory structures
- Detailed extraction methodology
- Per-repository clone commands
- Quality checks for Triton code
- License verification procedures
- Next steps for dataset preparation
