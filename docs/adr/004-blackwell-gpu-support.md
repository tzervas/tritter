# ADR-004: Blackwell GPU (RTX 50-series) Support

**Status**: Accepted
**Date**: 2026-01-23
**Deciders**: Tyler Zervas, Claude (Agentic Development)

## Context

Tritter targets the NVIDIA RTX 5080 (16GB GDDR7) as the primary development and inference hardware. The RTX 5080 uses NVIDIA's Blackwell architecture with compute capability SM_120, which requires specific software stack versions.

### Problem

Standard PyTorch releases (up to 2.5.x) do not include pre-compiled CUDA kernels for SM_120. Attempting to run on RTX 5080 results in:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Hardware Specifications

| Specification | RTX 5080 |
|--------------|----------|
| Architecture | Blackwell |
| Compute Capability | SM_120 (12.0) |
| VRAM | 16 GB GDDR7 |
| Memory Bandwidth | 960 GB/s |
| CUDA Cores | 10,752 |
| Tensor Cores | 336 |

### Software Stack Requirements

Through testing, we determined the minimum requirements for SM_120 support:

| Component | Minimum Version | Tested Version |
|-----------|-----------------|----------------|
| CUDA Toolkit | 12.8 | 12.8 |
| PyTorch | Nightly with cu128 | 2.11.0.dev20260123+cu128 |
| NVIDIA Driver | 590+ | 590.48.01 |
| Python | 3.12+ | 3.13 |
| Triton | 3.6.0+ | 3.6.0+git9844da95 |

## Decision

We will:

1. **Pin PyTorch nightly cu128** for RTX 5080/50-series development
2. **Document hardware requirements** in a dedicated file
3. **Provide installation instructions** for Blackwell GPU support
4. **Support fallback** to standard PyTorch for non-Blackwell GPUs

### Installation Method

For Blackwell GPUs (RTX 50-series):

```bash
# Using UV (recommended)
uv pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128

# Using pip
pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128
```

For other GPUs (RTX 40/30-series, A100, etc.):

```bash
# Standard PyTorch with CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Rationale

### Why PyTorch Nightly?

- SM_120 support is not yet in stable PyTorch releases
- Nightly builds include pre-compiled kernels for new architectures
- cu128 index specifically includes Blackwell support
- Alternative (building from source) is time-consuming and error-prone

### Why Not Pin in pyproject.toml?

- Nightly versions change daily, pinning breaks reproducibility
- Different GPUs require different PyTorch builds
- Installation index URL cannot be specified in pyproject.toml
- Separate installation step is clearer for users

### Why Document Separately?

- Hardware requirements are environment-specific
- Users need clear guidance before installation
- Version requirements will change as stable releases include SM_120

## Consequences

### Positive

- RTX 5080 development fully functional
- Clear documentation prevents confusion
- Flexible support for multiple GPU generations
- Future-proofed for RTX 50-series GPUs

### Negative

- Nightly PyTorch may have instability
- Extra installation step for Blackwell users
- Documentation needs updates when stable releases add SM_120

### Risks

- Nightly API changes could break code
- cu128 index might be deprecated when stable releases catch up

### Mitigations

- CI testing against nightly builds
- Clear version documentation
- Update ADR when stable PyTorch adds SM_120 support

## Verification

The following operations were verified working on RTX 5080 with PyTorch 2.11.0+cu128:

```python
# Basic tensor operations
x = torch.randn(100, 100, device='cuda')
y = x @ x.T  # Matrix multiply: OK

# Neural network layers
linear = torch.nn.Linear(256, 256).cuda()
inp = torch.randn(1, 256, device='cuda')
out = linear(inp)  # Linear layer: OK

# Attention mechanisms
out = scaled_dot_product_attention(q, k, v, is_causal=True)  # SDPA: OK
```

## References

- [PyTorch GitHub Issue #159207](https://github.com/pytorch/pytorch/issues/159207) - SM_120 support request
- [NVIDIA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [CUDA Toolkit 12.8 Release Notes](https://developer.nvidia.com/cuda-toolkit-archive)
- [PyTorch Nightly Index](https://download.pytorch.org/whl/nightly/)

## Decision Record

| Date | Action |
|------|--------|
| 2026-01-23 | Initial proposal and acceptance |
