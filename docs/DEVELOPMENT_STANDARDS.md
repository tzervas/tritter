# Tritter Development Standards

**Version:** 1.0
**Last Updated:** 2026-01-21
**Status:** Mandatory for all contributions

## Table of Contents

1. [Core Principles](#core-principles)
2. [Documentation Standards](#documentation-standards)
3. [Code Architecture Standards](#code-architecture-standards)
4. [Testing Standards](#testing-standards)
5. [Module Organization Standards](#module-organization-standards)
6. [Common Anti-Patterns to Avoid](#common-anti-patterns-to-avoid)
7. [Pre-Commit Checklist](#pre-commit-checklist)

---

## Core Principles

### 1. Undocumented Code is Incomplete Code

**Rule:** All non-trivial code MUST have Google-style docstrings including the "why" explanation.

**Rationale:** Understanding the reasoning behind design decisions is as important as knowing what the code does. Future maintainers need to know why alternatives were rejected.

**Trivial vs Non-Trivial:**
- **Trivial:** Simple getters, obvious property accessors, `__init__` that just assigns parameters
- **Non-Trivial:** Any logic, transformations, architectural decisions, optimizations, algorithms

### 2. Embedding-First Architecture

**Rule:** All code must acknowledge that Tritter operates in continuous embedding space, not discrete token space.

**Key Understanding:**
- Tokenization is the **entry point** (discrete → continuous)
- Model operates on **embeddings** (Coconut/LCM style)
- Model predicts **next embeddings**, not next tokens
- Token decoding is the **exit point** (continuous → discrete via KNN/VQ)

**Documentation Requirement:** When writing tokenization or model code, explicitly state this paradigm in docstrings.

### 3. Memory Constraints are Non-Negotiable

**Rule:** All architectural decisions must respect RTX 5080 16GB VRAM constraint.

**Memory Budget:**
- 7B BitNet weights: 1.4 GB
- KV-cache (128K, INT4): 8-10 GB
- Activations + overhead: 2-3 GB
- Vision encoder: ~0.4 GB
- **Total: ~12-15 GB** (2-4 GB headroom for batching)

**Enforcement:** Any feature that would exceed this budget must be explicitly discussed and justified.

---

## Documentation Standards

### Google-Style Docstrings with "Why"

**Format:**

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short one-line description.

    Longer description if needed. Explain what the function does and its
    role in the larger system.

    Args:
        param1: Description of param1 (include units, ranges, constraints)
        param2: Description of param2 (include expected shapes for tensors)

    Returns:
        Description of return value (include shapes, ranges, special values)

    Raises:
        ErrorType: When this error occurs

    Why: This is the critical part. Explain:
        - Why this approach was chosen over alternatives
        - What constraints drove the design
        - How this fits into the embedding-prediction paradigm
        - What would break without this specific implementation
        - Why certain parameters have their default values

    Example:
        >>> tokenizer = MultiModalTokenizer(vocab_size=65536)
        >>> tokens = tokenizer.encode("Hello", ModalityType.TEXT)

    Note:
        Additional implementation notes or caveats.

    TODO:
        Future improvements or known limitations.
    """
```

**Module-Level Docstrings:**

Every module must have:
1. Purpose statement
2. "Why" explanation (why this module exists)
3. Key architectural context (especially embedding-prediction paradigm)

### Parameter Documentation Requirements

**For All Functions/Methods:**

```python
def attention_forward(
    self,
    hidden_states: torch.Tensor,  # (batch_size, seq_len, hidden_size)
    attention_mask: torch.Tensor | None = None,  # (batch_size, seq_len, seq_len)
    position_ids: torch.Tensor | None = None,  # (batch_size, seq_len)
) -> torch.Tensor:
    """Multi-head attention forward pass.

    Args:
        hidden_states: Input embeddings of shape (batch_size, seq_len, hidden_size).
            Why shape constraint: Must match config.hidden_size for projection matrices.
        attention_mask: Optional mask of shape (batch_size, seq_len, seq_len).
            Values: 0 = attend, -inf = mask. Default None means full attention.
            Why optional: Inference without masking faster, training needs causal mask.
        position_ids: Optional position indices of shape (batch_size, seq_len).
            Range: [0, max_position_embeddings). Default None uses range(seq_len).
            Why needed: RoPE positional encoding requires explicit positions.

    Returns:
        Attention output of shape (batch_size, seq_len, hidden_size).
        Same shape as input for residual connection compatibility.

    Why: Multi-head attention enables parallel processing of different representational
        subspaces. Splitting hidden_size across num_heads allows each head to specialize
        in different types of relationships (e.g., syntactic vs semantic).
    """
```

**Tensor Shape Documentation:**

ALWAYS document tensor shapes in comments:

```python
# ✅ CORRECT
x = torch.randn(batch_size, seq_len, hidden_size)  # (B, L, D)
attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L)

# ❌ INCORRECT
x = torch.randn(batch_size, seq_len, hidden_size)
attn_weights = torch.matmul(q, k.transpose(-2, -1))
```

### Configuration Documentation

**Every Config Field Must Have:**

```python
@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Why: [Explain overall config purpose and constraints]
    """

    # Use inline comments with reasoning
    hidden_size: int = 2048  # Why: Proven effective for 3B models; divisible by 16 heads
    num_layers: int = 24  # Why: Balances capacity with memory (RTX 5080 16GB)
    vocab_size: int = 65536  # Why: 50K BPE + 8K VQVAE + audio codes + special tokens

    # For complex defaults, use detailed docstring
    max_position_embeddings: int = 131072  # 128K context
    """Maximum sequence length.

    Why 128K: Repository-level code understanding requires full file context.
    Achievable on RTX 5080 16GB via BitNet quantization (1.4GB weights) +
    INT4 KV-cache (8-10GB). Tested to fit within memory budget.
    """
```

---

## Code Architecture Standards

### Imports and Exports

**CRITICAL RULE:** `__all__` exports MUST match actual imports in `__init__.py`

**Pattern:**

```python
# ✅ CORRECT: src/tritter/models/__init__.py
"""Model architecture components."""

from tritter.models.architecture import TritterModel

__all__ = ["TritterModel"]


# ❌ INCORRECT: Export without import
"""Model architecture components."""

__all__ = ["TritterModel"]  # ImportError! TritterModel not defined
```

**Verification Command:**

```bash
# Every __init__.py must pass this test
python -c "from tritter.models import *; print('OK')"
```

**Enforcement:** Add this to CI/CD pipeline.

### Embedding-Prediction Architecture

**Rule:** Code operating on embeddings must explicitly acknowledge this paradigm.

```python
# ✅ CORRECT
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward pass through transformer.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)

    Returns:
        Logits of shape (batch_size, seq_len, vocab_size)

    Why: Current implementation outputs logits for compatibility with standard
    language modeling. Production model will output embeddings directly and use
    KNN/VQ rounding only at generation boundaries. The model operates in continuous
    embedding space (Coconut/LCM style) - these logits are just a temporary
    projection for training with cross-entropy loss.
    """
    embeddings = self.embed(input_ids)  # Entry: discrete → continuous
    hidden_states = self.transformer(embeddings)  # Operate in continuous space
    logits = self.output_projection(hidden_states)  # Temporary discrete projection
    return logits


# ❌ INCORRECT: No mention of embedding-prediction paradigm
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward pass."""
    embeddings = self.embed(input_ids)
    hidden_states = self.transformer(embeddings)
    logits = self.output_projection(hidden_states)
    return logits
```

### Symmetry Requirements

**Rule:** Encode/decode, quantize/dequantize, compress/decompress must be symmetric.

**Pattern:**

```python
# ✅ CORRECT: Symmetric encode/decode
def encode(self, text: str) -> list[int]:
    """Encode text to token IDs."""
    return [ord(c) % self.vocab_size for c in text]

def decode(self, token_ids: list[int]) -> str:
    """Decode token IDs to text.

    Why: Matches encode's modulo operation. Full Unicode range (0x10FFFF)
    prevents data loss during round-trip. Production needs proper BPE.
    """
    return "".join(chr(t % 0x110000) for t in token_ids
                   if 0 <= t % 0x110000 <= 0x10FFFF)


# ❌ INCORRECT: Asymmetric (data loss)
def encode(self, text: str) -> list[int]:
    """Encode text to token IDs."""
    return [ord(c) % self.vocab_size for c in text]  # Can produce IDs up to vocab_size

def decode(self, token_ids: list[int]) -> str:
    """Decode token IDs to text."""
    return "".join(chr(t) for t in token_ids if t < 128)  # Only handles ASCII!
```

**Verification Test:**

```python
def test_encode_decode_symmetry():
    """Verify encode/decode round-trip preserves data."""
    tokenizer = MultiModalTokenizer()

    # Test various Unicode ranges
    test_strings = ["Hello", "你好", "🎉", "def foo(): return 42"]

    for text in test_strings:
        tokens = tokenizer.encode(text, ModalityType.TEXT, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert decoded == text, f"Round-trip failed for: {text}"
```

---

## Testing Standards

### Use Config Values, Never Hardcode

**Rule:** Tests must use `config.vocab_size`, `config.hidden_size`, etc. Never hardcode magic numbers.

**Pattern:**

```python
# ✅ CORRECT
def test_forward_pass():
    """Test forward pass through model."""
    config = TritterConfig(
        hidden_size=128,
        num_heads=4,
        vocab_size=1000,
    )
    model = TritterModel(config)

    # Use config values for test data
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)

    assert logits.shape == (2, 8, config.vocab_size)


# ❌ INCORRECT: Hardcoded values
def test_forward_pass():
    """Test forward pass through model."""
    config = TritterConfig(
        hidden_size=128,
        num_heads=4,
        vocab_size=1000,
    )
    model = TritterModel(config)

    input_ids = torch.randint(0, 1000, (2, 8))  # Hardcoded! Breaks if config changes
    logits = model(input_ids)

    assert logits.shape == (2, 8, 1000)  # Hardcoded! Breaks if config changes
```

**Why This Matters:** If config.vocab_size changes to 2000, hardcoded tests still pass but use invalid data.

### Parameter Bounds Validation

**Rule:** Tests must validate parameter counts are within expected ranges, not just > 0.

**Pattern:**

```python
# ✅ CORRECT
def test_model_parameters():
    """Test model has expected parameter count."""
    config = TritterConfig(
        hidden_size=64,
        num_heads=2,
        num_layers=1,
        vocab_size=100,
    )
    model = TritterModel(config)
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate expected range based on architecture
    # Embedding: vocab_size * hidden_size = 100 * 64 = 6,400
    # Attention: 4 * (hidden_size * hidden_size) ≈ 16,384
    # MLP: ~2 * (hidden_size * intermediate_size) ≈ 32,768
    # Output: vocab_size * hidden_size = 6,400
    # Total expected: ~60K params

    assert 50_000 < total_params < 100_000, (
        f"Param count {total_params:,} outside expected range. "
        f"Possible initialization error or missing components."
    )


# ❌ INCORRECT: No bounds checking
def test_model_parameters():
    """Test model has parameters."""
    config = TritterConfig(hidden_size=64, num_heads=2, num_layers=1, vocab_size=100)
    model = TritterModel(config)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params > 0  # Too weak! Would pass with 1 parameter
```

### Gradient Flow Validation

**Rule:** Tests for quantization/STE MUST verify gradients flow correctly with reasonable magnitude.

**Pattern:**

```python
# ✅ CORRECT
def test_ste_gradient_flow():
    """Test straight-through estimator passes gradients correctly."""
    layer = TernaryWeight(in_features=8, out_features=4)
    x = torch.randn(2, 8, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Verify gradients exist
    assert x.grad is not None, "Input gradients missing"
    assert layer.weight.grad is not None, "Weight gradients missing"

    # Verify gradients are non-zero (STE working)
    assert x.grad.abs().max() > 0, (
        "Input gradients all zero - STE may be broken"
    )
    assert layer.weight.grad.abs().max() > 0, (
        "Weight gradients all zero - STE implementation error"
    )

    # Verify gradients are finite (no numerical issues)
    assert torch.isfinite(x.grad).all(), "Input gradients contain NaN/Inf"
    assert torch.isfinite(layer.weight.grad).all(), "Weight gradients contain NaN/Inf"

    # Verify gradients are reasonable magnitude (heuristic check)
    assert x.grad.abs().mean() < 100, "Input gradients suspiciously large"
    assert layer.weight.grad.abs().mean() < 100, "Weight gradients suspiciously large"


# ❌ INCORRECT: Only checks existence
def test_gradient_flow():
    """Test gradients flow through layer."""
    layer = TernaryWeight(in_features=8, out_features=4)
    x = torch.randn(2, 8, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None  # Not enough! Gradients could be all zeros
    assert layer.weight.requires_grad  # Not tested after backward!
```

### Test Documentation

**Rule:** Every test must have a docstring explaining what it validates and why.

```python
def test_attention_output_shape():
    """Test attention preserves input shape for residual connections.

    Why: Residual connections require identical input/output shapes.
    Shape mismatch would cause runtime errors during forward pass and
    prevent gradient flow in backward pass. This test catches dimension
    errors early before expensive training runs.
    """
    config = TritterConfig(hidden_size=128, num_heads=4)
    attention = TritterAttention(config)

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape, (
        f"Shape mismatch: input {hidden_states.shape} != output {output.shape}"
    )
```

---

## Module Organization Standards

### File Naming Conventions

```
src/tritter/
├── core/                 # Configuration and base classes
│   ├── __init__.py      # Export TritterConfig
│   └── config.py        # TritterConfig implementation
├── models/              # Model architecture
│   ├── __init__.py      # Export TritterModel
│   └── architecture.py  # TritterModel, TritterAttention, etc.
├── quantization/        # BitNet implementation
│   ├── __init__.py      # Export TernaryWeight, BitNetQuantizer
│   └── bitnet.py        # Implementation
├── tokenization/        # Multimodal tokenization
│   ├── __init__.py      # Export MultiModalTokenizer, ModalityType
│   └── multimodal.py    # Implementation
├── training/            # Training loop (stub)
│   └── __init__.py      # Documented stub explaining future implementation
├── inference/           # Inference engine (stub)
│   └── __init__.py      # Documented stub explaining future implementation
└── utils/               # Utility functions
    ├── __init__.py
    └── device_utils.py  # RTX 5080 optimizations
```

### __init__.py Pattern

**MANDATORY PATTERN:**

```python
"""Brief module description.

Longer description with architectural context.

Why: Explanation of why this module exists and its role in the system.
"""

# Import all exported classes/functions
from tritter.module_name.implementation import Class1, Class2, function1

# Explicitly declare exports
__all__ = ["Class1", "Class2", "function1"]
```

**Verification:**

Every `__init__.py` must satisfy:
1. Has module-level docstring with "Why"
2. Imports match `__all__` exactly
3. Can execute `from module import *` without errors

### Stub Module Documentation

**Rule:** Stub modules (not yet implemented) MUST have comprehensive documentation explaining:

1. **Why** the module exists
2. **What** it will do when implemented
3. **Dependencies** (what needs to exist first)
4. **Why** it's not implemented yet
5. **TODO** clear next steps

**Example:** See `src/tritter/training/__init__.py` and `src/tritter/inference/__init__.py`

---

## Common Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Silent Data Loss

```python
# BAD: Decode only handles subset of encode's output range
def encode(self, text: str) -> list[int]:
    return [ord(c) % 65536 for c in text]  # Range: [0, 65535]

def decode(self, tokens: list[int]) -> str:
    return "".join(chr(t) for t in tokens if t < 128)  # Only ASCII!
```

**Fix:** Ensure encode/decode have matching domains.

### ❌ Anti-Pattern 2: Magic Numbers

```python
# BAD: Hardcoded values scattered everywhere
input_ids = torch.randint(0, 1000, (2, 8))
assert output.shape[-1] == 1000
if hidden_size != 2048:
    raise ValueError()
```

**Fix:** Use config values consistently.

### ❌ Anti-Pattern 3: Weak Validation

```python
# BAD: Test passes with incorrect implementations
assert parameters > 0  # Could be 1 parameter!
assert grad is not None  # Could be all zeros!
assert output.shape[0] == batch_size  # Doesn't check other dimensions!
```

**Fix:** Add bounds checking and magnitude verification.

### ❌ Anti-Pattern 4: Missing "Why" Context

```python
# BAD: No explanation of design decisions
@dataclass
class Config:
    hidden_size: int = 2048
    vocab_size: int = 65536
    max_position_embeddings: int = 131072
```

**Fix:** Every non-obvious value needs "Why" explanation linking to constraints.

### ❌ Anti-Pattern 5: Incomplete Examples

```python
# BAD: Example doesn't show full workflow
model = TritterModel(config)
tokens = tokenizer.encode(text)
# ... missing tensor conversion, device movement, inference
```

**Fix:** Examples must be copy-pasteable and complete. See `examples/basic_usage.py`.

---

## Pre-Commit Checklist

**Before committing, verify:**

### Documentation
- [ ] All new functions/classes have Google-style docstrings
- [ ] All docstrings include "Why" explanation
- [ ] Tensor shapes documented in comments
- [ ] Config parameters explained with rationale
- [ ] Module `__init__.py` imports match `__all__`

### Architecture
- [ ] Embedding-prediction paradigm acknowledged where relevant
- [ ] Memory constraints respected (RTX 5080 16GB budget)
- [ ] Encode/decode operations are symmetric
- [ ] No hardcoded magic numbers (use config values)

### Testing
- [ ] Tests use `config.vocab_size` not hardcoded values
- [ ] Parameter count tests have bounds checking
- [ ] Gradient tests verify magnitude, not just existence
- [ ] Every test has explanatory docstring

### Code Quality
- [ ] All Python files compile: `python -m py_compile <file>`
- [ ] Imports work: `from tritter.module import *`
- [ ] No unused imports or exports
- [ ] Type hints present for public APIs

### Examples
- [ ] Examples are complete and runnable
- [ ] Examples demonstrate best practices
- [ ] README quick start references complete examples

---

## Enforcement

### Manual Review

Every PR must be reviewed against this document. Reviewers should cite specific sections when requesting changes.

### Automated Checks (Future)

1. **Import validation**: Script to verify `__all__` matches imports
2. **Docstring linting**: Check for "Why" presence in non-trivial code
3. **Magic number detection**: Flag hardcoded values in tests
4. **Symmetry tests**: Verify encode/decode round-trips

### Attribution

When adding this to review comments:

> This violates [DEVELOPMENT_STANDARDS.md § Testing Standards: Use Config Values](#use-config-values-never-hardcode)

---

## Version History

**v1.0 (2026-01-21):**
- Initial version based on PR#2 review feedback
- Codified lessons from export mismatches, asymmetry bugs, weak tests
- Established embedding-prediction paradigm documentation requirements
- Added comprehensive examples and anti-patterns

**Future Versions:**
- Will add sections as new patterns emerge
- Will incorporate feedback from community
