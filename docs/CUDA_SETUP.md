# CUDA Setup Guide

This guide documents the CUDA configuration for Tritter development on RTX 5080 16GB.

## System Requirements

- **GPU**: NVIDIA RTX 5080 16GB VRAM
- **CUDA**: 12.1+ (Driver version 590.48.01 or higher)
- **Python**: 3.12 or 3.13 (3.14+ not yet supported by PyTorch)

## Installation

### 1. Create Virtual Environment

```bash
# Use Python 3.13 for best compatibility
uv venv --python 3.13 --clear .venv
source .venv/bin/activate  # or: overlay use .venv/bin/activate.nu
```

### 2. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.5.1 with CUDA 12.1
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Tritter Dependencies

```bash
# Install full project with dev dependencies
uv pip install -e ".[dev]"
```

## Verification

Verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5080
```

## Known Issues

### RTX 5080 Compute Capability Warning

The RTX 5080 has CUDA compute capability `sm_120`, which is newer than the capabilities in PyTorch 2.5.1 (`sm_50` through `sm_90`). You may see this warning:

```
UserWarning: NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

**Impact**: Most PyTorch operations will still work, but some optimized kernels may fall back to slower implementations.

**Solution**: Monitor PyTorch releases for RTX 5080 support updates, or try nightly builds:
```bash
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

## Memory Budget (16GB VRAM)

Target allocation for 7B model with 128K context:
- **BitNet weights (7B)**: ~1.4 GB (ternary quantization)
- **INT4 KV-cache (128K)**: ~8-10 GB
- **Activations**: ~2-3 GB
- **Buffer**: ~2 GB

## Testing CUDA Functionality

```bash
# Run unit tests with CUDA
pytest tests/unit/test_quantization.py -v

# Verify BitNet CUDA kernels
python scripts/validate_bitnet_weights.py

# Test 128K context handling
python scripts/verify_128k_context.py
```

## Troubleshooting

### PyTorch not finding CUDA

1. Ensure you're in the activated virtual environment:
   ```bash
   which python  # Should show .venv/bin/python
   ```

2. Verify PyTorch has CUDA build:
   ```bash
   python -c "import torch; print(torch.version.cuda)"  # Should print "12.1"
   ```

3. Check NVIDIA driver:
   ```bash
   nvidia-smi  # Should show GPU and CUDA version
   ```

### Python Version Issues

- Python 3.14+: Not supported by PyTorch yet (as of 2026-01)
- Python 3.12-3.13: Fully supported
- Python 3.11 and below: Compatible but not recommended

To check your environment's Python:
```bash
python --version  # Inside venv
uv python list    # All available Python versions
```

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Tritter Development Standards](DEVELOPMENT_STANDARDS.md)
