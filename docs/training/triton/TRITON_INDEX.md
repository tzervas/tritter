# Triton GPU Kernel Dataset Curation - Complete Index

**Issue**: #58 - Triton GPU kernel dataset curation
**Research Completion Date**: 2026-01-23
**All Documentation**: 4 comprehensive documents + scripts

---

## Quick Navigation

### For Quick Answers
👉 **Start here**: [`TRITON_QUICK_REFERENCE.md`](./TRITON_QUICK_REFERENCE.md) (5 min read)
- Repository URLs and licenses
- Copy-paste extraction commands
- File path quick lookup table
- Validation checklist

### For Complete Details
👉 **Full reference**: [`TRITON_KERNEL_CURATION.md`](./TRITON_KERNEL_CURATION.md) (20 min read)
- Complete directory structures
- Critical file paths explained
- Per-repository extraction strategy
- License verification procedures
- Quality validation requirements

### For Implementation
👉 **Extraction plan**: [`TRITON_EXTRACTION_PLAN.md`](./TRITON_EXTRACTION_PLAN.md) (30 min read + execution)
- 7-phase extraction pipeline
- Executable shell scripts
- Python validation scripts
- Master orchestration script
- Expected output structure

### For Context
👉 **Research summary**: [`ISSUE_58_RESEARCH_SUMMARY.md`](./ISSUE_58_RESEARCH_SUMMARY.md) (10 min read)
- Executive summary
- Key findings
- Implementation roadmap
- Verification checklist

---

## Document Matrix

| Document | Size | Time | Purpose | Audience |
|----------|------|------|---------|----------|
| **TRITON_QUICK_REFERENCE.md** | 5KB | 5 min | Fast lookup | Everyone |
| **TRITON_KERNEL_CURATION.md** | 24KB | 20 min | Complete reference | Researchers |
| **TRITON_EXTRACTION_PLAN.md** | 24KB | 30 min + execution | Executable plan | Engineers |
| **ISSUE_58_RESEARCH_SUMMARY.md** | 15KB | 10 min | Overview & roadmap | Leads/PMs |

---

## Research Summary

### Repositories Identified: 5

1. **Triton Language** (`triton-lang/triton`)
   - MIT License ✅
   - Path: `python/tutorials/`
   - Files: 12 tutorials
   - Tokens: 50-100K

2. **PyTorch Inductor** (`pytorch/pytorch`)
   - BSD-3-Clause ✅
   - Path: `torch/_inductor/codegen/`
   - Files: 20-30 Triton-specific
   - Tokens: 400-600K

3. **FlashAttention** (`Dao-AILab/flash-attention`)
   - BSD-3-Clause ✅
   - Path: `flash_attn/`
   - Files: 3-5
   - Tokens: 80-150K

4. **xFormers** (`facebookresearch/xformers`)
   - BSD-3-Clause ✅
   - Path: `xformers/ops/fmha/_triton/`
   - Files: 15-30
   - Tokens: 150-300K

5. **JAX Pallas** (`jax-ml/jax`)
   - Apache-2.0 ✅
   - Path: `jax/experimental/pallas/triton/`
   - Files: 10-20
   - Tokens: 50-150K

### Totals
- **Total files**: 60-97
- **Total tokens**: 730K-1.3M
- **Commercial use**: ✅ All permissive licenses

---

## Critical File Paths (Copy-Paste Ready)

### Priority 1: Must-Have

```
triton-lang/triton/python/tutorials/06-fused-attention.py
pytorch/torch/_inductor/codegen/triton.py
pytorch/torch/_inductor/codegen/triton_utils.py
Dao-AILab/flash-attention/flash_attn/flash_attn_triton.py
facebookresearch/xformers/xformers/ops/fmha/triton_splitk.py
```

### Priority 2: Highly Recommended

```
triton-lang/triton/python/tutorials/03-matrix-multiplication.py
triton-lang/triton/python/tutorials/04-low-memory-dropout.py
triton-lang/triton/python/tutorials/05-layer-norm.py
pytorch/torch/_inductor/codegen/triton_combo_kernel.py
pytorch/torch/_inductor/kernel/mm_common.py
facebookresearch/xformers/xformers/ops/fmha/_triton/splitk_kernels.py
```

---

## Universal Extraction Commands

### All Repositories at Once

```bash
# Find all @triton.jit decorated functions
find . -name "*.py" -type f -exec grep -l "@triton.jit" {} \;
```

### By Repository

```bash
find triton/python/tutorials -name "*.py"                    # Tutorials
find pytorch/torch/_inductor -name "*triton*.py"             # PyTorch
find flash-attention -name "*triton*.py"                     # FlashAttention
find xformers -path "*_triton*" -name "*.py"                 # xFormers
find jax -path "*pallas/triton*" -name "*.py"                # JAX
```

### Efficient Cloning (Sparse Checkout)

```bash
# PyTorch (large repo - use sparse)
git clone --filter=blob:none --sparse https://github.com/pytorch/pytorch.git
cd pytorch && git sparse-checkout set torch/_inductor && git checkout main

# JAX (large repo - use sparse)
git clone --filter=blob:none --sparse https://github.com/jax-ml/jax.git
cd jax && git sparse-checkout set jax/experimental/pallas && git checkout main

# Others (smaller - full clone)
git clone --depth 1 https://github.com/triton-lang/triton.git
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
git clone --depth 1 https://github.com/facebookresearch/xformers.git
```

---

## Implementation Phases (from TRITON_EXTRACTION_PLAN.md)

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Clone repositories | ~30 min | Script: `01_clone_repositories.sh` |
| 2 | Identify kernels | ~5 min | Script: `02_identify_kernels.sh` |
| 3 | Generate statistics | ~5 min | Script: `03_statistics.sh` |
| 4 | Organize kernels | ~10 min | Script: `04_organize_kernels.sh` |
| 5 | License verification | ~5 min | Script: `05_verify_licenses.py` |
| 6 | Quality validation | ~5 min | Script: `06_quality_checks.py` |
| 7 | Create manifest | ~5 min | Script: `07_create_manifest.py` |

**Total execution time**: ~65 minutes + download time

---

## Quality Validation Checklist

Every Triton kernel must pass:

- ✅ Has `@triton.jit` decorator
- ✅ Includes bounds masking (`tl.where`, `mask=`)
- ✅ Uses `tl.program_id()` for parallelization
- ✅ Defines block sizes (`BLOCK_SIZE`)
- ✅ Uses `tl.load()`/`tl.store()` operations
- ✅ Includes docstring or comments
- ✅ Permissive license header present

---

## Training Data Integration

### Recommended Upsample Factor

**5-10x relative to Python** (Triton is low-resource language)

### Example Distribution (1T tokens)

```
Python code:         950B (95%)
Triton kernels:      50B (5%) ← upsampled 5x from natural ~0.1%
```

### Phased Rollout

| Phase | Triton % | Size |
|-------|---------|------|
| Base pretraining | 2.5-5% | 1-2T tokens |
| Domain continued | 5-10% | 200-500B tokens |
| Instruction tuning | Special | 2-5M samples |

---

## File Organization (Expected Output)

```
triton_kernels_organized/
├── tutorial/                    # Triton tutorials (educational)
├── inductor/                    # PyTorch Inductor (production)
├── flashattention/              # FlashAttention (research)
├── xformers/                    # xFormers (optimized)
├── jax/                         # JAX Pallas (specialized)
└── MANIFEST.json                # Dataset metadata
```

---

## Key Statistics

### Kernel File Count

| Repository | Files | % |
|-----------|-------|---|
| Triton | 12 | 12% |
| PyTorch | 20-30 | 30% |
| FlashAttention | 3-5 | 5% |
| xFormers | 15-30 | 25% |
| JAX | 10-20 | 18% |
| **Total** | **60-97** | **100%** |

### Token Estimates

| Repository | Lines | Tokens |
|-----------|-------|--------|
| Triton | 2-3K | 50-100K |
| PyTorch | 15-25K | 400-600K |
| FlashAttention | 3-5K | 80-150K |
| xFormers | 8-15K | 150-300K |
| JAX | 5-10K | 50-150K |
| **Total** | **33-58K** | **730K-1.3M** |

---

## License Compliance

| Repository | License | Status |
|-----------|---------|--------|
| triton-lang/triton | MIT | ✅ Commercial OK |
| pytorch/pytorch | BSD-3-Clause | ✅ Commercial OK |
| Dao-AILab/flash-attention | BSD-3-Clause | ✅ Commercial OK |
| facebookresearch/xformers | BSD-3-Clause | ✅ Commercial OK |
| jax-ml/jax | Apache-2.0 | ✅ Commercial OK |

**All repositories**: Permissively licensed, suitable for commercial training

---

## How to Use These Documents

### Scenario 1: Quick Reference
```
1. Read: TRITON_QUICK_REFERENCE.md (5 min)
2. Copy: Extraction command for your target
3. Execute: In terminal
```

### Scenario 2: Full Implementation
```
1. Understand: TRITON_KERNEL_CURATION.md (architecture)
2. Execute: TRITON_EXTRACTION_PLAN.md (step by step)
3. Validate: Run provided Python scripts
4. Integrate: With training pipeline
```

### Scenario 3: Context for Leadership
```
1. Review: ISSUE_58_RESEARCH_SUMMARY.md
2. Understand: Key findings and statistics
3. Reference: Implementation roadmap
4. Plan: 4-week execution timeline
```

### Scenario 4: Researcher Verification
```
1. Cross-check: TRITON_KERNEL_CURATION.md
2. Verify: Repository URLs and paths
3. Validate: License statements
4. Estimate: Token counts
```

---

## Quick Links

### External Resources
- **Triton Language**: https://triton-lang.org/
- **PyTorch Compiler**: https://docs.pytorch.org/docs/stable/torch.compiler_get_started.html
- **FlashAttention Paper**: https://arxiv.org/abs/2205.14135
- **xFormers Docs**: https://facebookresearch.github.io/xformers/
- **JAX Pallas**: https://docs.jax.dev/en/latest/pallas/index.html

### Related Tritter Documentation
- [`clean-datasets.md`](./clean-datasets.md) - Original Triton section
- [`project-plan.md`](./project-plan.md) - Overall roadmap
- [`DEVELOPMENT_STANDARDS.md`](./DEVELOPMENT_STANDARDS.md) - Coding standards

---

## Verification Status

- ✅ All 5 repositories verified via GitHub (2026-01-23)
- ✅ Directory structures confirmed
- ✅ File paths validated
- ✅ License status verified
- ✅ Extraction commands tested (syntax)
- ✅ Quality validation procedures documented
- ✅ Executable scripts provided
- ✅ Token count ranges estimated

**Ready for implementation**: YES

---

## Next Steps

### Immediate (This Week)
1. [ ] Read TRITON_QUICK_REFERENCE.md
2. [ ] Review TRITON_KERNEL_CURATION.md
3. [ ] Understand TRITON_EXTRACTION_PLAN.md

### Short-term (Week 1-2)
1. [ ] Set up workspace
2. [ ] Execute 7-phase extraction pipeline
3. [ ] Generate MANIFEST.json
4. [ ] Run quality validation

### Medium-term (Week 2-3)
1. [ ] Tokenize kernel files
2. [ ] Implement deduplication
3. [ ] Create upsample metadata
4. [ ] Integrate with training

### Long-term (Week 4+)
1. [ ] Benchmark contamination check
2. [ ] Quality assurance
3. [ ] Production deployment
4. [ ] Training integration

---

## Support

### Questions?
1. Check **TRITON_QUICK_REFERENCE.md** for quick answers
2. See **TRITON_KERNEL_CURATION.md** for details
3. Follow **TRITON_EXTRACTION_PLAN.md** for step-by-step

### Implementation issues?
1. Refer to troubleshooting section in TRITON_EXTRACTION_PLAN.md
2. Check GitHub repository status for current URLs
3. Verify local Git version (2.25+) for sparse checkout

---

**Research Completion**: 2026-01-23
**Status**: ✅ COMPLETE - READY FOR IMPLEMENTATION
**Documentation**: 4 documents + 7 executable scripts
**Next Phase**: Execute TRITON_EXTRACTION_PLAN.md
