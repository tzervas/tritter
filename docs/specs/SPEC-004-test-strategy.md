# Test Strategy Specification

**Spec ID**: SPEC-004
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-22
**Target Directory**: `tests/`

## 1. Overview

### 1.1 Purpose

This specification defines the testing strategy for Tritter, ensuring:

- All components have adequate test coverage
- Tests follow DEVELOPMENT_STANDARDS.md requirements
- Test organization enables efficient CI/CD
- GPU-dependent tests are properly isolated

### 1.2 Testing Principles

Per DEVELOPMENT_STANDARDS.md:

1. **Use config values, never hardcode** - Tests use `config.vocab_size`, not magic numbers
2. **Bounds checking on parameters** - Not just `> 0`, but expected ranges
3. **Gradient magnitude verification** - Not just existence, but reasonable values
4. **Docstrings required** - Every test explains what it validates

### 1.3 Test Categories

| Category | Directory | GPU Required | CI Priority |
|----------|-----------|--------------|-------------|
| Unit tests | `tests/unit/` | No (mocked) | High |
| Integration tests | `tests/integration/` | Yes | Medium |
| Performance tests | `tests/performance/` | Yes | Low |
| Smoke tests | `tests/smoke/` | Optional | Critical |

---

## 2. Test Organization

### 2.1 Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_config.py       # TritterConfig validation
│   ├── test_model.py        # Model architecture
│   ├── test_quantization.py # BitNet quantization
│   ├── test_tokenization.py # Multimodal tokenization
│   ├── test_attention.py    # Attention patterns (NEW)
│   └── test_utils.py        # Utility functions
├── integration/
│   ├── test_forward_pass.py # Full model forward
│   ├── test_training.py     # Training loop (when implemented)
│   └── test_inference.py    # Inference pipeline (when implemented)
├── performance/
│   ├── test_memory.py       # Memory budget verification
│   └── test_throughput.py   # Token throughput benchmarks
└── smoke/
    └── test_imports.py      # Import validation
```

### 2.2 Fixture Strategy

```python
# tests/conftest.py

import pytest
from tritter import TritterConfig, TritterModel


@pytest.fixture
def minimal_config() -> TritterConfig:
    """Minimal configuration for fast unit tests.

    Why: Small hidden_size and vocab_size enable CPU testing
    without GPU memory. Tests should use config values, not
    hardcoded numbers.
    """
    return TritterConfig(
        model_size="3B",
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        vocab_size=1000,
        use_bitnet=False,  # CPU-friendly
    )


@pytest.fixture
def production_config() -> TritterConfig:
    """Production-like configuration for integration tests.

    Why: Validates real architecture at smaller scale.
    Requires GPU for reasonable performance.
    """
    return TritterConfig(
        model_size="3B",
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        vocab_size=10000,
        use_bitnet=True,
    )


@pytest.fixture
def minimal_model(minimal_config) -> TritterModel:
    """Small model for unit testing."""
    return TritterModel(minimal_config)
```

---

## 3. Test Specifications by Module

### 3.1 Configuration Tests (`test_config.py`)

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| TC-001 | Default config valid | `TritterConfig()` | No errors |
| TC-002 | 3B auto-configuration | `model_size="3B"` | hidden_size=2048 |
| TC-003 | 7B auto-configuration | `model_size="7B"` | hidden_size=4096 |
| TC-004 | Invalid modality rejected | `modalities=["invalid"]` | AssertionError |
| TC-005 | Head dimension divisibility | `hidden_size=100, num_heads=3` | AssertionError |
| TC-006 | Vocab size minimum | `vocab_size=100` | AssertionError (< 264) |
| TC-007 | Attention mode validation | `attention_mode="invalid"` | AssertionError |
| TC-008 | Sliding window requires size | `use_sliding_window=True, sliding_window_size=None` | AssertionError |

```python
class TestTritterConfig:
    """Configuration validation tests.

    Why: Config errors should fail fast at creation time,
    not during expensive training runs.
    """

    def test_default_config_valid(self):
        """Verify default configuration creates successfully.

        Why: Default config must always work for quick experimentation.
        """
        config = TritterConfig()
        assert config.hidden_size == 2048  # 3B default
        assert config.vocab_size >= 264  # Minimum for byte encoding

    def test_vocab_size_minimum_enforced(self):
        """Verify vocab_size >= 264 requirement.

        Why: Byte-level encoding requires 8 special tokens + 256 bytes.
        Values below this would cause encoding failures.
        """
        with pytest.raises(AssertionError, match="vocab_size"):
            TritterConfig(vocab_size=100)

    def test_attention_mode_validation(self):
        """Verify only valid attention modes accepted.

        Why: Invalid modes would cause undefined behavior in attention.
        """
        valid_modes = ["causal", "bidirectional", "prefix_lm", "embedding"]
        for mode in valid_modes:
            config = TritterConfig(attention_mode=mode)
            assert config.attention_mode == mode

        with pytest.raises(AssertionError, match="attention_mode"):
            TritterConfig(attention_mode="invalid")
```

### 3.2 Model Tests (`test_model.py`)

| Test ID | Description | Verification |
|---------|-------------|--------------|
| TM-001 | Forward pass shape | Output shape matches (B, L, vocab_size) |
| TM-002 | Embedding output shape | With return_embeddings, shape is (B, L, D) |
| TM-003 | Parameter count bounds | Within expected range for architecture |
| TM-004 | Gradient flow | Gradients non-zero and finite |
| TM-005 | Residual connection | Output different from input |
| TM-006 | Attention causality | Future tokens don't affect current |

```python
class TestTritterModel:
    """Model architecture tests.

    Why: Verify model produces correct shapes, maintains
    gradient flow, and respects architectural constraints.
    """

    def test_forward_pass_shape(self, minimal_config, minimal_model):
        """Verify forward pass produces correct output shape.

        Why: Shape mismatches cause runtime errors during training.
        """
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(
            0, minimal_config.vocab_size, (batch_size, seq_len)
        )

        logits = minimal_model(input_ids)

        assert logits.shape == (
            batch_size,
            seq_len,
            minimal_config.vocab_size,
        ), f"Expected ({batch_size}, {seq_len}, {minimal_config.vocab_size})"

    def test_parameter_count_bounds(self, minimal_config, minimal_model):
        """Verify parameter count within expected range.

        Why: Parameter count sanity check catches initialization bugs.
        Weak assertion (> 0) wouldn't catch missing layers.
        """
        total_params = sum(p.numel() for p in minimal_model.parameters())

        # Calculate expected bounds based on architecture
        # Embedding: vocab_size * hidden_size
        # Attention per layer: ~4 * hidden_size^2
        # MLP per layer: ~8 * hidden_size^2
        # Output: vocab_size * hidden_size
        expected_min = (
            minimal_config.vocab_size * minimal_config.hidden_size  # embed
            + minimal_config.num_layers * 4 * minimal_config.hidden_size ** 2  # attn
        )
        expected_max = expected_min * 2  # Conservative upper bound

        assert expected_min < total_params < expected_max, (
            f"Parameter count {total_params:,} outside expected range "
            f"[{expected_min:,}, {expected_max:,}]"
        )

    def test_gradient_flow(self, minimal_config, minimal_model):
        """Verify gradients flow through model with reasonable magnitude.

        Why: Zero or infinite gradients indicate broken backprop.
        Just checking grad is not None isn't sufficient.
        """
        input_ids = torch.randint(
            0, minimal_config.vocab_size, (2, 8)
        )

        logits = minimal_model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check embedding gradients
        embed_grad = minimal_model.embed_tokens.weight.grad
        assert embed_grad is not None, "Embedding gradients missing"
        assert embed_grad.abs().max() > 0, "Embedding gradients all zero"
        assert torch.isfinite(embed_grad).all(), "Embedding gradients contain NaN/Inf"
        assert embed_grad.abs().mean() < 100, "Embedding gradients suspiciously large"
```

### 3.3 Quantization Tests (`test_quantization.py`)

| Test ID | Description | Verification |
|---------|-------------|--------------|
| TQ-001 | Ternary weight values | Weights in {-1, 0, +1} |
| TQ-002 | STE gradient flow | Gradients pass through quantization |
| TQ-003 | Shadow weight update | Shadow weights differ from quantized |
| TQ-004 | Scale factor positive | Quantization scale > 0 |

```python
class TestBitNetQuantization:
    """BitNet ternary quantization tests.

    Why: Quantization must preserve gradient flow via STE
    and produce valid ternary weights.
    """

    def test_ternary_weight_values(self):
        """Verify quantized weights are ternary {-1, 0, +1}.

        Why: BitNet b1.58 requires exactly three values.
        Non-ternary weights break the efficiency assumptions.
        """
        layer = TernaryWeight(in_features=32, out_features=16)
        quantized = layer._quantize(layer.weight)

        unique_values = torch.unique(quantized)
        expected = torch.tensor([-1.0, 0.0, 1.0])

        for val in unique_values:
            assert val in expected, f"Non-ternary value: {val}"

    def test_ste_gradient_flow(self):
        """Verify straight-through estimator passes gradients.

        Why: Without STE, gradients through quantization are zero,
        making training impossible.
        """
        layer = TernaryWeight(in_features=8, out_features=4)
        x = torch.randn(2, 8, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Verify gradients exist and are reasonable
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().max() > 0, "STE not passing gradients"
        assert torch.isfinite(layer.weight.grad).all()
```

### 3.4 Tokenization Tests (`test_tokenization.py`)

| Test ID | Description | Verification |
|---------|-------------|--------------|
| TT-001 | Encode/decode symmetry | Round-trip preserves text |
| TT-002 | Special tokens present | BOS, EOS, PAD in vocabulary |
| TT-003 | Modality-specific encoding | Different modalities use different ranges |
| TT-004 | Batch encoding | Multiple strings encoded correctly |

```python
class TestMultiModalTokenizer:
    """Multimodal tokenization tests.

    Why: Tokenization is the entry/exit point for discrete data.
    Asymmetric encode/decode would corrupt training data.
    """

    def test_encode_decode_symmetry(self):
        """Verify encode/decode round-trip preserves data.

        Why: Per DEVELOPMENT_STANDARDS.md, symmetric operations are
        mandatory. Data loss during tokenization corrupts training.
        """
        tokenizer = MultiModalTokenizer(vocab_size=1000)

        test_strings = [
            "Hello, World!",
            "def foo(): return 42",
            "Unicode: 你好 🎉",
        ]

        for text in test_strings:
            tokens = tokenizer.encode(text, ModalityType.TEXT)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Round-trip failed for: {text!r}"

    def test_special_tokens_in_vocabulary(self):
        """Verify special tokens are properly defined.

        Why: Special tokens (BOS, EOS, PAD, UNK) are required for
        sequence handling. Missing tokens cause training failures.
        """
        tokenizer = MultiModalTokenizer(vocab_size=1000)

        assert tokenizer.pad_token_id is not None
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.unk_token_id is not None

        # Verify they're distinct
        special_ids = {
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        }
        assert len(special_ids) == 4, "Special tokens must be distinct"
```

---

## 4. Integration Tests

### 4.1 Full Forward Pass

```python
# tests/integration/test_forward_pass.py

@pytest.mark.gpu
class TestFullForwardPass:
    """Integration tests requiring GPU.

    Why: Validates complete model behavior at realistic scale.
    Catches issues that don't appear in small unit test configs.
    """

    def test_7b_config_forward(self):
        """Verify 7B configuration forward pass works.

        Why: Production configuration must be validated.
        Small test configs may miss dimension mismatches.
        """
        config = TritterConfig(model_size="7B")
        model = TritterModel(config).cuda()

        # Small batch for memory
        input_ids = torch.randint(
            0, config.vocab_size, (1, 256), device="cuda"
        )

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (1, 256, config.vocab_size)
```

### 4.2 Memory Budget Verification

```python
# tests/integration/test_memory.py

@pytest.mark.gpu
class TestMemoryBudget:
    """Memory budget verification tests.

    Why: RTX 5080 16GB is the target. Tests must verify
    the model fits within this constraint.
    """

    def test_7b_fits_in_16gb(self):
        """Verify 7B model fits in 16GB VRAM.

        Why: This is the core hardware constraint for Tritter.
        Exceeding 16GB makes the model unusable on target hardware.
        """
        config = TritterConfig(model_size="7B", use_bitnet=True)
        model = TritterModel(config).cuda()

        torch.cuda.reset_peak_memory_stats()

        input_ids = torch.randint(
            0, config.vocab_size, (1, 4096), device="cuda"
        )

        with torch.no_grad():
            _ = model(input_ids)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        assert peak_gb < 15.0, (
            f"Peak memory {peak_gb:.2f} GB exceeds 15 GB budget. "
            f"RTX 5080 target is 16 GB."
        )
```

---

## 5. Smoke Tests

```python
# tests/smoke/test_imports.py

class TestImports:
    """Import validation smoke tests.

    Why: Catches import errors and missing dependencies before
    running expensive tests. Should run first in CI.
    """

    def test_tritter_imports(self):
        """Verify main package imports successfully."""
        from tritter import TritterConfig, TritterModel
        assert TritterConfig is not None
        assert TritterModel is not None

    def test_submodule_imports(self):
        """Verify all submodules import successfully."""
        from tritter.core.config import TritterConfig
        from tritter.models.architecture import TritterModel
        from tritter.quantization.bitnet import TernaryWeight
        from tritter.tokenization.multimodal import MultiModalTokenizer

    def test_all_exports_match_imports(self):
        """Verify __all__ matches actual imports.

        Why: Per DEVELOPMENT_STANDARDS.md, __all__ must match imports.
        """
        import tritter
        for name in tritter.__all__:
            assert hasattr(tritter, name), f"{name} in __all__ but not defined"
```

---

## 6. Test Markers

```python
# pytest.ini or pyproject.toml

[tool.pytest.ini_options]
markers = [
    "gpu: marks test as requiring GPU (skip without CUDA)",
    "slow: marks test as slow (skip with --fast)",
    "integration: marks test as integration test",
]
```

Usage:
```bash
# Run only unit tests (fast, no GPU)
pytest tests/unit/

# Run all tests with GPU
pytest

# Skip slow tests
pytest --fast

# Run only GPU tests
pytest -m gpu
```

---

## 7. CI/CD Integration

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: pytest tests/smoke/ -v

  unit:
    needs: smoke
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=src/tritter

  integration:
    needs: unit
    runs-on: [self-hosted, gpu]  # GPU runner
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v -m gpu
```

---

## 8. Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `core/config.py` | 90% |
| `models/architecture.py` | 80% |
| `quantization/bitnet.py` | 80% |
| `tokenization/multimodal.py` | 90% |
| `training/__init__.py` | N/A (stub) |
| `inference/__init__.py` | N/A (stub) |

---

## Appendix: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-22 | Claude | Initial draft |
