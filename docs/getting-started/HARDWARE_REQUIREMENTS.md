# Hardware Requirements

This document specifies hardware and software requirements for running Tritter.

## Target Hardware

Tritter is optimized for the **NVIDIA RTX 5080** (16GB GDDR7) but supports other NVIDIA GPUs.

### Recommended Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5080 16GB |
| System RAM | 32GB+ |
| Storage | NVMe SSD (for model weights) |
| CPU | 8+ cores (for data loading) |

### Memory Budget (RTX 5080 16GB)

| Component | Memory |
|-----------|--------|
| 7B BitNet weights | 1.4 GB |
| KV-cache (128K, INT4) | 8-10 GB |
| Vision encoder (SigLIP-B) | 0.4 GB |
| Activations + overhead | 2-3 GB |
| **Total** | ~12-14 GB |

---

## GPU Compatibility

### RTX 50-series (Blackwell) - SM_120

**Supported GPUs**: RTX 5090, RTX 5080, RTX 5070 Ti, RTX 5070

**Required Software**:

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.8+ |
| PyTorch | Nightly with cu128 |
| NVIDIA Driver | 590+ |
| Python | 3.12+ |

**Installation**:

```bash
# Using UV (recommended)
uv pip install -e ".[dev]"
uv pip install torch triton --pre --index-url https://download.pytorch.org/whl/nightly/cu128

# Using pip
pip install -e ".[dev]"
pip install torch triton --pre --index-url https://download.pytorch.org/whl/nightly/cu128
```

Pinned nightly (SM_120 stability):

```bash
export TRITTER_BLACKWELL_TORCH_VERSION="2.11.0.dev20260123+cu128"
export TRITTER_BLACKWELL_TRITON_VERSION="3.6.0+git9844da95"
uv pip install "torch==${TRITTER_BLACKWELL_TORCH_VERSION}" --pre --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install "triton==${TRITTER_BLACKWELL_TRITON_VERSION}" --pre
```

**Verification**:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### RTX 40-series (Ada Lovelace) - SM_89

**Supported GPUs**: RTX 4090, RTX 4080, RTX 4070 Ti, RTX 4070, RTX 4060 Ti

**Required Software**:

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.1+ |
| PyTorch | 2.5.0+ (stable) |
| NVIDIA Driver | 545+ |
| Python | 3.12+ |

**Installation**:

```bash
uv pip install -e ".[dev]"
# Default PyTorch should work, or explicitly:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### RTX 30-series (Ampere) - SM_86

**Supported GPUs**: RTX 3090, RTX 3080, RTX 3070, RTX 3060

**Required Software**:

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 11.8+ |
| PyTorch | 2.0.0+ (stable) |
| NVIDIA Driver | 520+ |
| Python | 3.10+ |

**Installation**:

```bash
uv pip install -e ".[dev]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Data Center GPUs

| GPU | SM | Notes |
|-----|-----|-------|
| H100 | SM_90 | Full support with PyTorch stable |
| A100 | SM_80 | Full support with PyTorch stable |
| V100 | SM_70 | Supported but limited for BitNet |

---

## Software Dependencies

### Core Dependencies

```toml
# From pyproject.toml
torch>=2.5.0          # Or nightly for Blackwell
numpy>=1.24.0
transformers>=4.35.0
tokenizers>=0.15.0
einops>=0.7.0
triton>=3.0.0         # Or nightly for Blackwell
```

### Tested Configurations

| Configuration | PyTorch | CUDA | Status |
|--------------|---------|------|--------|
| RTX 5080 | 2.11.0.dev+cu128 | 12.8 | Verified |
| RTX 4090 | 2.5.1+cu121 | 12.1 | Expected |
| A100 | 2.5.1+cu121 | 12.1 | Expected |

---

## Troubleshooting

### "no kernel image is available for execution on the device"

**Cause**: PyTorch doesn't have pre-compiled kernels for your GPU architecture.

**Solution for RTX 50-series**:
```bash
uv pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128
```

### CUDA Out of Memory (OOM)

**Cause**: Model or context length exceeds VRAM budget.

**Solutions**:
1. Reduce context length: `config.max_position_embeddings = 32768`
2. Enable layer streaming: `config.use_layer_streaming = True`
3. Use INT4 KV-cache: `config.int4_kv_cache = True`
4. Reduce batch size: `batch_size = 1`

### PyTorch nightly instability

**Cause**: Nightly builds may have regressions.

**Solution**: Pin to a known-good nightly version:
```bash
uv pip install torch==2.11.0.dev20260123+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Performance Expectations

### RTX 5080 (16GB)

| Task | Context | Tokens/sec |
|------|---------|------------|
| 7B inference | 8K | ~50-80 |
| 7B inference | 32K | ~30-50 |
| 7B inference | 128K | ~10-20 |
| 3B training | 4K | Varies |

*Benchmarks pending full verification*

---

## References

- [ADR-004: Blackwell GPU Support](adr/004-blackwell-gpu-support.md)
- [SPEC-005: Memory Optimization](specs/SPEC-005-memory-optimization.md)
- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)

---

*Last Updated: 2026-01-23*
