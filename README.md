# Tritter

A multimodal transformer research project combining BitNet 1.58-bit ternary quantization with an embedding-prediction architecture, optimized for consumer GPU hardware.

## Project Status

**Current Status**: Active Development (v0.2.0)

| Component | Status |
|-----------|--------|
| Core configuration | Implemented |
| Model architecture (attention, MLP, layers) | Implemented |
| BitNet 1.58-bit quantization | Implemented |
| Multimodal tokenization | Implemented |
| Training loop (BitNet QAT) | Implemented |
| Inference engine (streaming) | Implemented |
| Progressive layer loading | Implemented |
| Dataset curation pipeline | Implemented |
| Quality gates (security, quality) | Implemented |
| Development tooling | Implemented |

## Overview

Tritter implements a decoder-only transformer with the following design goals:

- **BitNet b1.58 quantization**: Ternary weights {-1, 0, +1} for ~10x memory reduction
- **Multimodal tokenization**: Unified vocabulary for text, code, image, and audio
- **Embedding-prediction paradigm**: Operations in continuous embedding space rather than discrete token space
- **Consumer GPU target**: Designed for RTX 5080 16GB VRAM constraints

### Embedding-Prediction Paradigm

The architecture operates in continuous embedding space:

1. **Entry point**: Tokenization converts discrete tokens to embeddings
2. **Core computation**: Transformer layers operate on continuous embeddings
3. **Exit point**: Output projection to logits (temporary for training compatibility)

Production inference will use KNN/VQ rounding instead of argmax token selection, enabling continuous latent reasoning.

## Hardware Target

**Primary target**: NVIDIA RTX 5080 with 16GB GDDR7

| Component | Memory Budget |
|-----------|--------------|
| 7B BitNet weights | ~1.4 GB |
| INT4 KV-cache (128K context) | ~8-10 GB |
| Activations + overhead | ~2-3 GB |
| Vision encoder | ~0.4 GB |
| **Total** | ~12-15 GB |

## Installation

Requires Python 3.12 or 3.13, and CUDA 12.1+ for GPU acceleration.

### Basic Installation

```bash
# Clone repository
git clone https://github.com/tzervas/tritter.git
cd tritter

# Create virtual environment (Python 3.13 recommended)
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Verify CUDA Setup

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

See [`docs/CUDA_SETUP.md`](docs/CUDA_SETUP.md) for detailed CUDA configuration and troubleshooting.

## Quick Start

```python
from tritter import TritterConfig, TritterModel
from tritter.tokenization import MultiModalTokenizer, ModalityType
import torch

# Initialize configuration
config = TritterConfig(
    model_size="3B",  # Auto-configures architecture
    use_bitnet=True,
    use_flash_attention=True,
)

# Create model
model = TritterModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize tokenizer
tokenizer = MultiModalTokenizer(vocab_size=config.vocab_size)

# Encode text
text = "def hello_world():\n    print('Hello, World!')"
tokens = tokenizer.encode(text, ModalityType.CODE)
input_ids = torch.tensor([tokens])

# Forward pass
with torch.no_grad():
    logits = model(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a complete demonstration.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
pytest --cov=src/tritter --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type check (strict mode)
mypy src/tritter

# Verify imports
python -c "from tritter import *; print('OK')"
```

### Training

Prepare training data and run pretraining:

```bash
# Curate training data from source code
python scripts/prepare_pretrain_data.py \
    --input-dir /path/to/code \
    --output-dir data/pretrain

# Train model (requires GPU)
python scripts/train_pretrain.py --model 1B --data-dir data/pretrain
```

See [CLAUDE.md](CLAUDE.md#training-pipeline) for full training pipeline documentation.

### Development Tools

The `devtools/` module provides development utilities:

```bash
# Run full validation suite
python -m devtools validate

# Quick validation (skip tests)
python -m devtools validate --quick

# Project status
python -m devtools status

# Implementation roadmap
python -m devtools status --roadmap
```

### Development Standards

All contributions must follow the standards documented in [`docs/DEVELOPMENT_STANDARDS.md`](docs/DEVELOPMENT_STANDARDS.md):

- Google-style docstrings with "Why" sections explaining design decisions
- Tensor shapes documented in comments: `x = proj(hidden)  # (B, L, D)`
- Tests use config values (never hardcoded magic numbers)
- Parameter count tests include bounds checking
- `__all__` exports must match imports in `__init__.py` files

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `TritterConfig` | `src/tritter/core/config.py` | Configuration with auto-scaling for 3B/7B |
| `TritterModel` | `src/tritter/models/architecture.py` | Full transformer model |
| `TritterAttention` | `src/tritter/models/architecture.py` | Multi-head attention with QK-Norm |
| `TritterMLP` | `src/tritter/models/architecture.py` | FFN with Squared ReLU |
| `TernaryWeight` | `src/tritter/quantization/bitnet.py` | BitNet quantization with STE |
| `MultiModalTokenizer` | `src/tritter/tokenization/multimodal.py` | Unified multimodal tokenization |

### BitNet Requirements

The architecture follows BitNet b1.58 constraints:

- **Squared ReLU** (`x * ReLU(x)`) activation in MLP layers
- **QK-Norm** for attention stability
- **Post-FFN LayerNorm** (Chameleon-style placement)
- Shadow weights in full precision for STE training

### Attention Architecture

Current implementation uses PyTorch SDPA with `is_causal=True` for FlashAttention-2 optimization. Planned enhancements:

1. FlexAttention for dynamic masking (sliding window, document boundaries)
2. Multiple attention modes (causal, bidirectional, prefix-lm)
3. StreamingLLM attention sinks for streaming inference

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/project-plan.md`](docs/project-plan.md) | Technical blueprint and research foundations |
| [`docs/DEVELOPMENT_STANDARDS.md`](docs/DEVELOPMENT_STANDARDS.md) | Code standards and requirements |
| [`docs/API_CONVENTIONS.md`](docs/API_CONVENTIONS.md) | Interface patterns and conventions |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Contribution guidelines |
| [`docs/clean-datasets.md`](docs/clean-datasets.md) | Training data strategy |
| [`docs/considerations.md`](docs/considerations.md) | Research on alternative architectures |
| [`CLAUDE.md`](CLAUDE.md) | AI assistant guidelines |

## Project Structure

```
tritter/
├── src/tritter/           # Core model implementation
│   ├── core/              # Configuration
│   ├── models/            # Architecture components
│   ├── quantization/      # BitNet implementation
│   ├── tokenization/      # Multimodal tokenization
│   ├── training/          # Training loop (stub)
│   ├── inference/         # Inference engine (stub)
│   └── utils/             # Utilities
├── devtools/              # Development tooling
├── tests/                 # Test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Research Context

This project builds on published research:

- **BitNet b1.58**: Microsoft's ternary quantization achieving ~10x memory reduction
- **Chameleon**: Meta's early-fusion multimodal architecture
- **Coconut/LCM**: Embedding-prediction paradigm for continuous latent reasoning
- **FlashAttention-2**: Memory-efficient attention with tiled computation

See [`docs/project-plan.md`](docs/project-plan.md) for detailed citations and technical analysis.

## Limitations

- Training loop not yet implemented (model architecture only)
- Inference engine not yet implemented
- No pretrained weights available
- Multimodal capabilities (image, audio) require additional encoder integration
- RTX 5080 16GB memory budget has been validated on real hardware; some CUDA kernels may fall back until newer compute capability support lands in PyTorch

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Tyler Zervas (tz-dev@vectorweight.com)
