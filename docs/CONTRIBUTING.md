# Contributing to Tritter

Thank you for your interest in contributing to Tritter! This document provides guidelines for contributing code, documentation, and other improvements.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Process](#development-process)
3. [Code Standards](#code-standards)
4. [Pull Request Process](#pull-request-process)
5. [Testing Requirements](#testing-requirements)
6. [Documentation Requirements](#documentation-requirements)
7. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

- **Python 3.12+** (required for modern type hints)
- **PyTorch 2.1.0+** with CUDA support (for RTX 5080 optimization)
- **Git** for version control
- Familiarity with transformer architectures and quantization

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/tzervas/tritter.git
cd tritter

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "from tritter import TritterModel; print('Installation successful')"

# Run tests to ensure everything works
pytest tests/
```

### Pre-commit Hooks

Pre-commit hooks automatically enforce coding standards before each commit:

```bash
# Hooks run automatically on commit, but you can run manually:
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

Hooks include:
- **ruff**: Linting and formatting
- **mypy**: Static type checking
- **import-validator**: Ensures `__all__` exports match imports
- **Standard hooks**: Trailing whitespace, YAML/TOML validation, etc.

### Understanding the Project

**CRITICAL:** Before contributing, read these documents in order:

1. **[DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md)** - Mandatory code quality standards
2. **[API_CONVENTIONS.md](API_CONVENTIONS.md)** - API design patterns and schemas
3. **[project-plan.md](project-plan.md)** - Technical architecture and research context
4. **[clean-datasets.md](clean-datasets.md)** - Dataset strategy and licensing

**Key Concepts to Understand:**

- **Embedding-Prediction Paradigm**: Tritter operates in continuous embedding space (Coconut/LCM style), not discrete token space. Tokenization is just the entry/exit boundary.
- **Memory Constraints**: All code must respect RTX 5080 16GB VRAM budget. See memory breakdown in `DEVELOPMENT_STANDARDS.md`.
- **BitNet Quantization**: 1.58-bit ternary weights {-1, 0, +1} reduce 7B model from 14GB to 1.4GB.
- **Early Fusion Multimodal**: Chameleon-style unified embedding space for text/code/image/audio.

---

## Spec-Driven Development

Tritter follows spec-driven development: **define specifications before implementation**.

### Documentation Hierarchy

```
docs/
├── specs/          # Specifications (what to build)
│   └── SPEC-001-flexattention.md
├── guides/         # Implementation guides (how to build)
│   └── GUIDE-001-flexattention-implementation.md
└── adr/            # Architecture Decision Records (why we decided)
    └── ADR-001-bitnet-quantization.md
```

### Workflow

1. **Specification**: Define requirements, acceptance criteria, and test specs
2. **ADR** (if needed): Document architectural decisions with alternatives considered
3. **Implementation Guide**: Create step-by-step instructions
4. **Implementation**: Follow the guide, marking checkboxes as you go
5. **Verification**: Run tests from spec, update spec status

### Creating a New Feature

For significant features, create documentation first:

```bash
# 1. Create specification
docs/specs/SPEC-XXX-feature-name.md

# 2. If architectural decision needed
docs/adr/ADR-XXX-decision-name.md

# 3. Create implementation guide
docs/guides/GUIDE-XXX-feature-implementation.md

# 4. Then implement following the guide
```

See existing specs and ADRs for templates and examples.

---

## Development Process

### Finding Something to Contribute

**Good First Issues:**

- Look for issues labeled `good-first-issue` or `help-wanted`
- Documentation improvements (adding "Why" explanations)
- Test coverage expansion
- Example code and tutorials

**Before Starting Work:**

1. Check if an issue exists; if not, create one describing what you want to do
2. Comment on the issue to claim it and get feedback on approach
3. Wait for maintainer confirmation before significant work

### Branch Naming

Use descriptive branch names:

```
feature/add-rope-positional-encoding
fix/tokenizer-decode-asymmetry
docs/update-bitnet-explanation
test/add-gradient-magnitude-checks
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or modifying tests
- `refactor`: Code refactoring without changing behavior
- `perf`: Performance improvements
- `chore`: Build process, dependencies, etc.

**Example:**

```
fix(tokenization): Resolve encode/decode asymmetry causing data loss

Previously, decode() only handled ASCII (tokens < 128) while encode()
produced IDs up to vocab_size. This caused data loss during round-trip.

Changed decode() to handle full Unicode range (0x10FFFF) matching
encode's modulo operation. Prevents data loss while maintaining
placeholder status until proper BPE implementation.

Resolves #42
```

---

## Code Standards

### Mandatory Requirements

All code MUST comply with [DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md). Key requirements:

1. **Documentation**: Google-style docstrings with "Why" explanations
2. **Type Hints**: All public APIs must have type annotations
3. **Config Values**: Never hardcode magic numbers; use config values
4. **Validation**: Input validation with clear error messages
5. **Symmetry**: Encode/decode and similar pairs must be symmetric

### Code Style

**Linting:**

```bash
# Run before committing
ruff check src/ tests/
mypy src/
```

**Formatting:**

- Line length: 100 characters max
- Use double quotes for strings
- 4 spaces for indentation (no tabs)

**Example of Good Code:**

```python
def attention_forward(
    self,
    hidden_states: torch.Tensor,  # (batch_size, seq_len, hidden_size)
    attention_mask: torch.Tensor | None = None,  # (batch_size, seq_len, seq_len)
) -> torch.Tensor:
    """Multi-head attention forward pass.

    Args:
        hidden_states: Input embeddings of shape (batch_size, seq_len, hidden_size).
            Why shape constraint: Must match config.hidden_size for projection matrices.
        attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len).
            Values: 0 = attend, -inf = mask. Default None means full attention.
            Why optional: Inference without masking is faster; training needs causal mask.

    Returns:
        Attention output of shape (batch_size, seq_len, hidden_size).
        Same shape as input to enable residual connections.

    Why: Multi-head attention splits hidden_size across num_heads, allowing parallel
        processing of different representational subspaces. Each head can specialize
        in different relationship types (e.g., syntactic vs semantic dependencies).
    """
    batch_size, seq_len = hidden_states.shape[:2]

    # Project to Q, K, V (using config values, not hardcoded dimensions)
    q = self.q_proj(hidden_states)  # (B, L, D)
    k = self.k_proj(hidden_states)  # (B, L, D)
    v = self.v_proj(hidden_states)  # (B, L, D)

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Compute attention scores
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # Apply mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax and weighted sum
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)

    # Concatenate heads and project
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

    return self.o_proj(attn_output)  # (B, L, D)
```

### Anti-Patterns to Avoid

See [DEVELOPMENT_STANDARDS.md § Common Anti-Patterns](DEVELOPMENT_STANDARDS.md#common-anti-patterns-to-avoid) for full list.

**Common mistakes:**

- Missing "Why" in docstrings
- Hardcoded values instead of config references
- Weak test assertions (just checking existence, not bounds)
- Encode/decode asymmetry
- Missing imports in `__init__.py` but declaring in `__all__`

---

## Pull Request Process

### Before Creating PR

**Checklist:**

- [ ] Code follows [DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md)
- [ ] All functions have Google-style docstrings with "Why"
- [ ] Tests added/updated and passing (`pytest tests/`)
- [ ] Type hints present (`mypy src/` passes)
- [ ] Linting clean (`ruff check src/ tests/`)
- [ ] Documentation updated if APIs changed
- [ ] Commit messages follow conventional commits
- [ ] No hardcoded values; uses config throughout

**Run Full Validation:**

```bash
# All of these must pass
python -m pytest tests/ -v
python -m mypy src/
ruff check src/ tests/
python -m py_compile src/**/*.py
```

### Creating the PR

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub** with template:

```markdown
## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issues

Fixes #(issue number)

## Changes Made

- Bullet point list of key changes
- Include architectural decisions
- Note any breaking changes

## Testing

Describe testing performed:
- Unit tests added/modified: `tests/unit/test_*.py`
- Manual testing: describe scenarios
- Performance impact: if applicable

## Documentation

- [ ] Docstrings updated with "Why" explanations
- [ ] README.md updated if user-facing changes
- [ ] API_CONVENTIONS.md updated if new interfaces
- [ ] DEVELOPMENT_STANDARDS.md updated if new patterns

## Memory Impact

If affects model size or memory usage:
- [ ] Verified fits within RTX 5080 16GB budget
- [ ] Updated memory breakdown in docs

## Checklist

- [ ] My code follows project style guidelines
- [ ] All tests pass locally
- [ ] I have added tests covering my changes
- [ ] My docstrings include "Why" explanations
- [ ] I have updated relevant documentation
- [ ] No hardcoded magic numbers
- [ ] Type hints present on public APIs
```

### PR Review Process

**Reviewers will check:**

1. **Code Quality**: Compliance with DEVELOPMENT_STANDARDS.md
2. **Architecture**: Fits embedding-prediction paradigm
3. **Tests**: Adequate coverage with bounds/magnitude checks
4. **Documentation**: All "Why" explanations present
5. **Performance**: Respects memory budget
6. **API Stability**: Breaking changes justified and documented

**Addressing Feedback:**

- Respond to all comments, even if just "Fixed"
- Don't force-push after review starts (preserves comment context)
- Request re-review when ready
- Discuss disagreements respectfully in comments

**Merge Criteria:**

- At least one approving review from maintainer
- All CI checks passing
- All reviewer comments addressed
- No merge conflicts with main branch

---

## Testing Requirements

### Unit Tests

**Every new function/class needs tests:**

```python
def test_function_name():
    """Test [specific behavior being validated].

    Why: [Explain what this test catches and why it matters]
    """
    # Setup
    config = TritterConfig(hidden_size=64, num_heads=2, vocab_size=100)

    # Execute
    result = function_under_test(config)

    # Verify with bounds checking
    assert 50 < result < 150, f"Result {result} outside expected range"
```

**Test Coverage Requirements:**

- All public APIs: 100% coverage
- Private methods: Coverage for non-trivial logic
- Edge cases: Empty inputs, None values, boundary conditions

### Integration Tests

For new features, add integration test:

```python
def test_end_to_end_workflow():
    """Test complete workflow from tokenization to inference.

    Why: Validates that all components integrate correctly and catches
    shape mismatches or API incompatibilities.
    """
    # Full workflow test
    config = TritterConfig(...)
    model = TritterModel(config)
    tokenizer = MultiModalTokenizer(...)

    # Process input through full pipeline
    tokens = tokenizer.encode("test", ModalityType.TEXT)
    input_ids = torch.tensor([tokens])
    logits = model(input_ids)

    # Verify end-to-end
    assert logits.shape == (1, len(tokens), config.vocab_size)
```

### Testing Best Practices

See [DEVELOPMENT_STANDARDS.md § Testing Standards](DEVELOPMENT_STANDARDS.md#testing-standards) for:

- Using config values vs hardcoding
- Parameter bounds validation
- Gradient flow testing
- Symmetry testing

---

## Documentation Requirements

### Code Documentation

**Every module needs:**

1. **Module docstring** with "Why" explanation
2. **Class docstrings** explaining purpose and architecture
3. **Method docstrings** with Args, Returns, Raises, Why
4. **Inline comments** for non-obvious logic

**Example Module Docstring:**

```python
"""Multi-head attention implementation with BitNet quantization support.

This module implements the core attention mechanism for Tritter's transformer
architecture, with optional BitNet 1.58-bit ternary quantization.

Why: Multi-head attention is fundamental to transformer architectures, allowing
the model to attend to different positions simultaneously. BitNet quantization
reduces memory footprint by 10x, enabling 7B models on RTX 5080 16GB VRAM.
Projection matrices (Q, K, V, O) can be quantized to {-1, 0, +1} ternary weights
without significant quality degradation when trained with straight-through
estimator (STE) gradients.

Classes:
    TritterAttention: Main attention module

See Also:
    - quantization.bitnet: BitNet quantization implementation
    - models.architecture: Overall model architecture
"""
```

### User-Facing Documentation

When adding user-facing features:

1. **Update README.md** with usage examples
2. **Add to examples/** directory if complex
3. **Update project-plan.md** if architectural change

### API Documentation

If adding new public APIs:

1. **Update API_CONVENTIONS.md** with new schemas
2. **Add type stubs** if complex types
3. **Document breaking changes** prominently

---

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Assume good faith
- Help newcomers learn

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, questions
- **Pull Requests**: Code review and discussion
- **Discussions**: General questions and brainstorming

### Getting Help

**Stuck on something?**

1. Check documentation: README, DEVELOPMENT_STANDARDS, API_CONVENTIONS
2. Search existing issues
3. Create new issue with:
   - What you're trying to do
   - What you've tried
   - Error messages or unexpected behavior
   - Relevant code snippets

**Good Issue Example:**

```markdown
### Problem
I'm trying to add RoPE positional encoding but get shape mismatch.

### What I've Tried
```python
# Code snippet showing attempt
position_ids = torch.arange(seq_len)
# Error: RuntimeError: shape mismatch...
```

### Error Message
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (64)
```

### Question
Should position_ids be per-head or shared across heads?
```

---

## Specific Contribution Areas

### Documentation Improvements

**Always welcome:**

- Adding "Why" explanations to existing docstrings
- Clarifying confusing documentation
- Fixing typos or grammar
- Adding examples
- Improving README

**Process:**

1. Small fixes: Direct PR (no issue needed)
2. Large rewrites: Create issue first to discuss approach

### Bug Fixes

**Process:**

1. Create issue describing bug with reproduction steps
2. Wait for confirmation it's actually a bug
3. Create PR with:
   - Fix implementation
   - Test preventing regression
   - Explanation of root cause

### New Features

**MUST discuss first:**

All new features require discussion in an issue before implementation.

**Include in proposal:**

- Use case and motivation
- Proposed API design
- Memory impact estimate
- Alternative approaches considered
- Why this approach is best

**Example Good Proposal:**

```markdown
### Feature: Add FlashAttention2 Integration

**Motivation:**
FlashAttention2 reduces attention memory complexity from O(N²) to O(N),
critical for 128K context window on RTX 5080 16GB.

**Proposed API:**
```python
config = TritterConfig(use_flash_attention=True)  # Already exists
# Implementation would use actual FlashAttention2 instead of placeholder
```

**Memory Impact:**
- Current: O(N²) attention matrix stored
- With FA2: O(1) memory for attention (computed on-the-fly)
- Savings: ~8GB for 128K context (batch_size=1)

**Alternatives Considered:**
1. Memory-efficient attention (slower than FA2)
2. Sparse attention (loses some modeling capacity)
3. Keep current placeholder (doesn't scale to 128K)

**Why FlashAttention2:**
- Industry standard for long context
- Proven to scale to 128K+ on consumer GPUs
- Maintains full attention quality (no approximation)
- PyTorch integration available

**Implementation Plan:**
1. Add FA2 dependency
2. Replace placeholder in TritterAttention.forward()
3. Add configuration flag validation
4. Test memory savings with profiler
5. Benchmark throughput vs baseline
```

---

## Release Process

*(For maintainers)*

### Version Numbering

- **Major.Minor.Patch** (Semantic Versioning)
- Breaking changes: Increment Major
- New features: Increment Minor
- Bug fixes: Increment Patch

### Release Checklist

- [ ] All tests passing on main branch
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Git tag created: `v0.2.0`
- [ ] GitHub release with notes
- [ ] PyPI package published (when ready)

---

## Questions?

**Not sure where to start?**

1. Look for `good-first-issue` labels
2. Read through existing PRs to understand process
3. Ask in issue comments before starting work
4. Join discussions to learn about ongoing work

**Found this document helpful?**

Consider improving it! Documentation contributions are highly valued.

---

## License

By contributing to Tritter, you agree that your contributions will be licensed under the MIT License.

---

## Acknowledgments

Thank you for contributing to Tritter! Every contribution, whether code, documentation, or community support, helps advance the project.

**Special Recognition:**

Contributors who make significant improvements to code quality, documentation, or testing infrastructure will be acknowledged in release notes.

---

**Last Updated:** 2026-01-22
**Version:** 1.1
