# Tritter

**Multimodal AI with BitNet 1.58-bit Quantization**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tritter is a research-driven multimodal transformer optimized for NVIDIA RTX 5080 (16GB GDDR7). It combines Microsoft's BitNet b1.58 ternary quantization with any-to-any multimodal capabilities (text/code/image/audio) using unified early-fusion tokenization.

## Features

### Core Architecture
- **BitNet 1.58-bit Quantization**: Ternary weights {-1, 0, 1} for 8x memory savings
- **128K Context Window**: Extended context via sliding window attention + StreamingLLM
- **Model Sizes**: 3B and 7B parameter variants
- **Flash Attention 2**: Optimized attention computation
- **INT4 KV Cache**: Reduced memory footprint for long sequences

### Multimodal Support
- **Text & Code**: Unified tokenization with AST-aware code processing
- **Image**: VQVAE-based visual tokenization
- **Audio**: SpeechTokenizer integration
- **Early Fusion**: Shared embedding space across modalities

### Optimizations
- RTX 5080 GDDR7 optimized
- Gradient checkpointing
- Efficient inference with vLLM support
- Training with Nanotron

## Installation

### Requirements
- Python 3.12+
- PyTorch 2.1.0+
- CUDA-capable GPU (optional but recommended)

### Using UV (Recommended)

```bash
# Install UV if not already installed
pip install uv

# Install Tritter
git clone https://github.com/tzervas/tritter.git
cd tritter
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"

# Install all optional dependencies
uv pip install -e ".[all]"
```

### Using pip

```bash
pip install -e .
```

## Quick Start

```python
import torch
from tritter import TritterConfig, TritterModel
from tritter.tokenization.multimodal import MultiModalTokenizer, ModalityType
from tritter.utils.device_utils import get_optimal_device

# Create configuration and model
config = TritterConfig(model_size="3B", use_bitnet=True)
model = TritterModel(config).to(get_optimal_device())

# Tokenize and run inference
tokenizer = MultiModalTokenizer(vocab_size=config.vocab_size)
tokens = tokenizer.encode("Hello from Tritter!", ModalityType.TEXT)
input_ids = torch.tensor([tokens]).to(get_optimal_device())

with torch.no_grad():
    logits = model(input_ids)
```

**For a complete working example with device optimization, parameter tuning, and multimodal demonstration, see [`examples/basic_usage.py`](examples/basic_usage.py).**

## Architecture

### BitNet 1.58-bit Quantization

Tritter implements ternary quantization where weights are constrained to {-1, 0, 1}:

```python
from tritter.quantization.bitnet import TernaryWeight

# Create quantized layer
layer = TernaryWeight(in_features=512, out_features=256)
```

### Multimodal Tokenization

Unified vocabulary for all modalities with special prefix tokens:

```python
from tritter.tokenization.multimodal import MultiModalTokenizer, ModalityType

tokenizer = MultiModalTokenizer()

# Text
text_tokens = tokenizer.encode("Hello world", ModalityType.TEXT)

# Code
code_tokens = tokenizer.encode("def foo(): pass", ModalityType.CODE)

# Image (requires image data)
# image_tokens = tokenizer.encode(image_data, ModalityType.IMAGE)
```

## Development

### Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"
```

### Testing

Tritter follows Test-Driven Development (TDD) principles:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/tritter --cov-report=html

# Run specific test file
pytest tests/unit/test_config.py
```

### Code Quality

```bash
# Format code with Ruff
ruff format .

# Lint code
ruff check .

# Type checking with mypy
mypy src/tritter
```

## Configuration

Key configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | "3B" | Model variant: "3B" or "7B" |
| `hidden_size` | 2048 | Hidden dimension (4096 for 7B) |
| `num_layers` | 24 | Number of transformer layers (32 for 7B) |
| `num_heads` | 16 | Number of attention heads (32 for 7B) |
| `max_position_embeddings` | 131072 | Context window (128K) |
| `use_bitnet` | True | Enable BitNet quantization |
| `use_flash_attention` | True | Enable FlashAttention2 |
| `int4_kv_cache` | True | Use INT4 KV cache |
| `sliding_window_size` | 4096 | Sliding window size |

## Project Structure

```
tritter/
├── src/tritter/
│   ├── core/              # Configuration and core components
│   ├── models/            # Model architecture
│   ├── quantization/      # BitNet quantization
│   ├── tokenization/      # Multimodal tokenizers
│   ├── training/          # Training utilities
│   ├── inference/         # Inference optimizations
│   └── utils/             # Helper utilities
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── examples/              # Usage examples
└── pyproject.toml         # Project configuration
```

## Technology Stack

- **Python 3.12**: Modern Python features
- **PyTorch**: Deep learning framework
- **UV**: Fast Python package manager
- **pytest**: Testing framework
- **Ruff**: Fast Python linter and formatter
- **mypy**: Static type checking

## Roadmap

- [ ] VQVAE image tokenization
- [ ] SpeechTokenizer audio processing
- [ ] AST-aware code tokenization (Coconut integration)
- [ ] LCM (Latent Consistency Models) support
- [ ] VSA/HRR representations
- [ ] vLLM inference backend
- [ ] Nanotron training integration
- [ ] TheStack v2 dataset integration
- [ ] SpecKit specification-driven development tools

## Contributing

Contributions are welcome! **Before contributing, please read our development standards:**

📘 **[CONTRIBUTING.md](CONTRIBUTING.md)** - Start here for contribution workflow and process

📐 **[DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md)** - Mandatory code quality standards
- Google-style docstrings with "Why" explanations required
- Embedding-prediction paradigm documentation
- Testing standards and anti-patterns to avoid
- Pre-commit checklist

📋 **[API_CONVENTIONS.md](API_CONVENTIONS.md)** - API design patterns and schemas
- Configuration, model, tokenization interfaces
- Type annotation requirements
- Error handling patterns

**Quick Start:**
1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow
2. Review [DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md) before coding
3. Follow [API_CONVENTIONS.md](API_CONVENTIONS.md) for new APIs
4. Ensure all tests pass: `pytest tests/`
5. Verify code quality: `ruff check src/ tests/ && mypy src/`
6. Submit PR following the template

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Tyler Zervas (tz-dev@vectorweight.com)

## Citation

If you use Tritter in your research, please cite:

```bibtex
@software{tritter2026,
  author = {Zervas, Tyler},
  title = {Tritter: Multimodal AI with BitNet 1.58-bit Quantization},
  year = {2026},
  url = {https://github.com/tzervas/tritter}
}
```

## Acknowledgments

- Microsoft Research for BitNet quantization
- Meta AI for Flash Attention
- The open-source AI community
