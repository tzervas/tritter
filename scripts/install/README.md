# Tritter Installation Scripts

This directory contains installation scripts for setting up the Tritter development environment on Debian/Ubuntu systems.

## Quick Start

### Full Development Setup (Recommended, UV-first)

```bash
cd scripts/install
chmod +x install-dev-debian.sh
./install-dev-debian.sh
```

This will install:
- System dependencies (build-essential, git, etc.)
- Python 3.12+ (from deadsnakes PPA if needed)
- CUDA toolkit 12.x
- PyTorch with CUDA support
- Triton
- Tritter in editable mode with dev dependencies
- Run verification tests

### Minimal Installation (Python packages only)

If you already have Python 3.12+ and CUDA installed:

```bash
cd scripts/install
chmod +x install-minimal.sh
./install-minimal.sh
```

UV-based dev install (recommended):

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e ".[dev,training,inference,curation,extras]"
```

## Script Options

### install-dev-debian.sh

Full installation with system dependencies.

```bash
./install-dev-debian.sh [OPTIONS]

Options:
  --no-cuda       Skip CUDA installation (CPU-only development)
  --venv-path     Custom virtual environment path (default: ./venv)
  --blackwell     Install PyTorch nightly for RTX 50-series support
  --skip-venv     Don't create virtual environment
  -h, --help      Show help message
```

**Examples:**

```bash
# CPU-only installation
./install-dev-debian.sh --no-cuda

# RTX 5080 (Blackwell) support
./install-dev-debian.sh --blackwell

# Custom virtual environment location
./install-dev-debian.sh --venv-path ~/envs/tritter

# Use existing/system Python environment
./install-dev-debian.sh --skip-venv
```

### install-minimal.sh

Lightweight installation for CI/containers.

```bash
./install-minimal.sh [OPTIONS]

Options:
  --no-venv       Don't create virtual environment
  --no-cuda       Install CPU-only PyTorch
  --blackwell     Install PyTorch nightly for RTX 50-series support
  -h, --help      Show help message
```

**Examples:**

```bash
# Install in system Python (CI/containers)
./install-minimal.sh --no-venv

# CPU-only
./install-minimal.sh --no-cuda

# RTX 50-series support
./install-minimal.sh --blackwell
```

## Prerequisites

### Supported Operating Systems

- Ubuntu 20.04, 22.04, 24.04
- Debian 11, 12
- Linux Mint 20+
- Pop!_OS 20+

The script will check for compatibility and exit gracefully on unsupported systems.

### Required

- Debian/Ubuntu-based Linux distribution
- `sudo` access (for full installation)
- Internet connection

### Optional

- NVIDIA GPU with driver >= 525.60.13 (for CUDA support)
- 16GB+ RAM (8GB minimum for small models)

## What Gets Installed

### System Packages (install-dev-debian.sh only)

- `build-essential` - C/C++ compiler and build tools
- `git` - Version control
- `curl`, `wget` - Download utilities
- `python3.12` or `python3.13` - From deadsnakes PPA if needed
- `python3-dev`, `python3-venv` - Python development headers
- CUDA toolkit 12.6 (if GPU detected and not `--no-cuda`)

### Python Packages (both scripts)

Core dependencies (from `pyproject.toml`):
- `torch >= 2.5.0` - PyTorch deep learning framework
- `triton >= 3.0.0` - GPU kernel compiler
- `numpy >= 1.24.0` - Numerical computing
- `transformers >= 4.35.0` - HuggingFace transformers
- `tokenizers >= 0.15.0` - Fast tokenization
- `einops >= 0.7.0` - Tensor operations

Development dependencies:
- `pytest >= 7.4.0` - Testing framework
- `pytest-cov >= 4.1.0` - Coverage reporting
- `ruff >= 0.1.0` - Linting and formatting
- `mypy >= 1.7.0` - Static type checking
- `pre-commit >= 3.6.0` - Git hooks

## Hardware-Specific Installation

### RTX 50-series (Blackwell)

RTX 5080, 5090, and other Blackwell GPUs require PyTorch nightly:

```bash
./install-dev-debian.sh --blackwell
```

Pinned nightly (for SM_120 stability):

```bash
export TRITTER_BLACKWELL_TORCH_VERSION="2.11.0.dev20260123+cu128"
# Optional Triton pin (if needed)
export TRITTER_BLACKWELL_TRITON_VERSION="3.6.0+git9844da95"
./install-dev-debian.sh --blackwell
```

This installs PyTorch from the nightly channel with SM_120 (Blackwell) support.

See [docs/adr/004-blackwell-gpu-support.md](../../docs/adr/004-blackwell-gpu-support.md) for details.

### CPU-Only Development

For systems without NVIDIA GPUs:

```bash
./install-dev-debian.sh --no-cuda
```

All code will still work, but training/inference will be slower.

### Limited RAM/VRAM

For systems with limited resources:
- Use smaller models (1B, 3B)
- Enable gradient checkpointing
- Use QLoRA instead of full fine-tuning

Pinned nightly example:

```bash
export TRITTER_BLACKWELL_TORCH_VERSION="2.11.0.dev20260123+cu128"
export TRITTER_BLACKWELL_TRITON_VERSION="3.6.0+git9844da95"
./install-minimal.sh --blackwell
```
- Enable layer streaming for inference

Check feasibility:
```bash
python scripts/rtx5080_feasibility.py
python scripts/hardware_profile.py --check 7B
```

## Verification

After installation, verify everything works:

```bash
# Activate virtual environment (if created)
source venv/bin/activate

# Test imports
python -c "from tritter import *; print('OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run test suite
pytest

# Check hardware profile
python scripts/hardware_profile.py --check 7B
```

## Troubleshooting

### Python Version Issues

**Problem:** Python 3.12+ not found

**Solution:**
```bash
# The script automatically installs from deadsnakes PPA, but if it fails:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12 python3.12-dev python3.12-venv
```

### CUDA Installation Issues

**Problem:** CUDA installation fails or driver mismatch

**Solution:**

1. Check NVIDIA driver version:
   ```bash
   nvidia-smi
   ```

2. If driver < 525.60.13, update drivers first:
   ```bash
   sudo ubuntu-drivers install
   # or manually from: https://www.nvidia.com/Download/index.aspx
   ```

3. Run installation without CUDA, then add it later:
   ```bash
   ./install-dev-debian.sh --no-cuda
   ```

### Virtual Environment Issues

**Problem:** Cannot create virtual environment

**Solution:**
```bash
# Install venv module
sudo apt-get install python3.12-venv

# Or skip venv creation and use system Python
./install-dev-debian.sh --skip-venv
```

### Import Errors

**Problem:** `ImportError: No module named 'tritter'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall in editable mode
pip install -e ".[dev]"

# Verify installation
python -c "import tritter; print(tritter.__file__)"
```

### CUDA Out of Memory

**Problem:** OOM errors during training/inference

**Solutions:**

1. **Reduce batch size:**
   ```python
   config.batch_size = 1  # or smaller
   ```

2. **Enable gradient checkpointing:**
   ```python
   config.use_gradient_checkpointing = True
   ```

3. **Use QLoRA instead of full training:**
   ```python
   from tritter.training.lora import LoRAConfig, apply_lora
   lora_config = LoRAConfig(rank=16)
   model = apply_lora(model, lora_config)
   ```

4. **Enable layer streaming:**
   ```python
   config.use_layer_streaming = True
   config.layer_group_size = 4
   ```

5. **Reduce context length:**
   ```python
   config.max_seq_len = 32768  # instead of 128K
   ```

See [scripts/rtx5080_feasibility.py](../rtx5080_feasibility.py) for memory analysis.

### Permission Denied

**Problem:** `sudo: no tty present and no askpass program specified`

**Solution:**
```bash
# Run with sudo directly
sudo ./install-dev-debian.sh

# Or add your user to sudoers (not recommended for security)
```

### Package Conflicts

**Problem:** Dependency conflicts or version mismatches

**Solution:**
```bash
# Create fresh virtual environment
rm -rf venv
./install-minimal.sh

# Or use uv (faster, better dependency resolution)
pip install uv
uv pip install -e ".[dev]"
```

## CI/Docker Usage

### GitHub Actions

```yaml
- name: Install dependencies
  run: |
    cd scripts/install
    chmod +x install-minimal.sh
    ./install-minimal.sh --no-venv --no-cuda
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and run minimal installer
COPY scripts/install/install-minimal.sh /tmp/
RUN chmod +x /tmp/install-minimal.sh && \
    /tmp/install-minimal.sh --no-venv --no-cuda

WORKDIR /app
```

## Manual Installation (Non-Debian Systems)

For Fedora, Arch, macOS, or other systems:

1. **Install Python 3.12 or 3.13:**
   - Fedora: `sudo dnf install python3.12`
   - Arch: `sudo pacman -S python`
   - macOS: `brew install python@3.12`

2. **Install CUDA (optional, for GPU):**
   - Download from: https://developer.nvidia.com/cuda-downloads

3. **Install Python packages:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate

   # PyTorch with CUDA
   pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu124

   # Or CPU-only
   pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cpu

   # Tritter
   pip install -e ".[dev]"
   ```

4. **Verify:**
   ```bash
   python -c "from tritter import *; print('OK')"
   pytest
   ```

## Next Steps

After installation:

1. **Read the documentation:**
   - [CLAUDE.md](../../CLAUDE.md) - Project overview
   - [docs/DEVELOPMENT_STANDARDS.md](../../docs/DEVELOPMENT_STANDARDS.md) - Coding standards
   - [docs/project-plan.md](../../docs/project-plan.md) - Technical blueprint

2. **Check hardware compatibility:**
   ```bash
   python scripts/hardware_profile.py --check 7B
   python scripts/rtx5080_feasibility.py
   ```

3. **Run tests:**
   ```bash
   pytest                           # All tests
   pytest -m "not slow"             # Skip slow tests
   pytest -m "not gpu"              # CPU only
   pytest --cov                     # With coverage
   ```

4. **Explore the codebase:**
   ```bash
   python scripts/show_model_specs.py --model 7B
   python -c "from tritter.core.config import TritterConfig; print(TritterConfig.__doc__)"
   ```

5. **Start developing:**
   - See [DEVELOPMENT_STANDARDS.md](../../docs/DEVELOPMENT_STANDARDS.md)
   - Use pre-commit hooks: `pre-commit install`
   - Follow the embedding-prediction paradigm

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section above
2. Review [docs/HARDWARE_REQUIREMENTS.md](../../docs/HARDWARE_REQUIREMENTS.md)
3. Run feasibility analysis: `python scripts/rtx5080_feasibility.py`
4. Open an issue on GitHub

## License

These installation scripts are part of the Tritter project and are licensed under the MIT License.
