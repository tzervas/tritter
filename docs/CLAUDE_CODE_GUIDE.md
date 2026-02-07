# Claude Code Guide for Tritter Development

This guide explains how to use Claude Code effectively with the Tritter project, leveraging intelligent task routing, progressive CI, and automated quality gates.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Intelligent Model Selection](#intelligent-model-selection)
- [Session Hooks](#session-hooks)
- [Quality Gates](#quality-gates)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Install Claude Code CLI (if not already installed)
npm install -g @anthropic-ai/claude-code

# Clone Tritter repository
git clone https://github.com/tzervas/tritter.git
cd tritter

# Install dependencies
uv pip install -e ".[dev]"

# Start Claude Code session
claude
```

The session start hook will automatically:
- Verify Python and dependency versions
- Check CUDA/GPU availability
- Detect branch and CI strictness level
- Provide helpful command shortcuts

### Configuration File

Tritter includes a `.clauderc` file that configures Claude Code behavior:

```ini
[project]
name = "Tritter"
type = "ml-research"
language = "python"

[model]
default = "sonnet"
prefer_opus_for = ["architecture", "optimization", "design"]
prefer_haiku_for = ["format", "lint", "typo"]

[hooks]
session_start = ".github/hooks/session-start.sh"
pre_commit = ".github/hooks/pre-commit.sh"
prompt_submit = ".github/hooks/prompt-submit.sh"
```

## Intelligent Model Selection

Claude Code automatically selects the optimal model based on task complexity.

### Model Selection Rules

| Task Type | Model | Effort | Examples |
|-----------|-------|--------|----------|
| **Architecture & Design** | Opus | High | "Design distributed caching", "Optimize memory layout" |
| **Complex Implementation** | Sonnet | Medium | "Implement attention layer", "Add LoRA training" |
| **Simple Tasks** | Haiku | Low | "Format code", "Fix typo", "Add docstring" |

### Example Prompts

**Opus-routed tasks:**
```
Design a memory-efficient KV-cache compression strategy for 128K context

Architect a progressive layer loading system for 70B models on 16GB VRAM

Optimize quantization for minimal accuracy degradation
```

**Sonnet-routed tasks:**
```
Implement FlashAttention with QK-Norm for TritterAttention

Add support for sliding window attention masks

Refactor BitNet quantization to support per-channel scaling
```

**Haiku-routed tasks:**
```
Format all Python files with ruff

Fix typo in README: "embeddings" not "embedddings"

Add type hints to config.py functions
```

### Manual Override

You can override the automatic selection:

```bash
# Force Opus for a simple task (if you want maximum quality)
claude --model opus "add docstring to attention function"

# Force Haiku for a complex task (faster, cheaper)
claude --model haiku "implement memory manager"

# Specify effort level explicitly
claude --effort high "optimize transformer layer"
```

## Session Hooks

### Session Start Hook

Runs automatically when starting a Claude Code session.

**What it does:**
- Checks Python version (requires 3.12+)
- Verifies virtual environment status
- Checks if uv is installed (recommended)
- Verifies PyTorch and CUDA availability
- Shows GPU information if available
- Detects current git branch
- Shows CI strictness level for current branch
- Lists helpful commands

**Example output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tritter - Claude Code Session Starting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Checking Python version...
Python: 3.12.1
Virtual env: /home/user/tritter/.venv
uv: 0.1.23

Checking dependencies...
PyTorch: 2.2.0+cu121
CUDA: 12.1
GPU: NVIDIA GeForce RTX 5080

Tritter: Installed

Git status...
Branch: feature/attention-optimization
CI Strictness: FEATURE
Uncommitted changes: 3 files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Quick Commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Format:       ruff format .
Lint:         ruff check .
Type check:   mypy src/tritter
Test:         pytest
Coverage:     pytest --cov=src/tritter
CI check:     .github/scripts/run-ci.sh

✅ Session ready! Review CLAUDE.md for project context.
```

### Pre-Commit Hook

Runs automatically before `git commit` (if enabled).

**What it does:**
- Detects branch strictness level
- Runs quick format check
- Runs quick lint check
- **Always** scans for hardcoded secrets
- Runs type check (for release/production branches)
- Prevents commit if critical issues found

**Behavior by strictness level:**
- **Feature (Level 0)**: Warnings only, commit allowed
- **Development (Level 1)**: Format required, lint warnings OK
- **Release (Level 2)**: Format + lint + types required
- **Production (Level 3)**: All checks must pass

**Example:**
```bash
git commit -m "Add attention optimization"

Running pre-commit quality checks...
Branch: feature/attention-opt
Strictness: FEATURE (Level 0)

Checking code format...
✅ Format OK
Checking for critical lint errors...
⚠️  Lint issues found
   Continuing (warnings allowed for FEATURE level)
Checking for hardcoded secrets...
✅ No secrets detected

✅ All pre-commit checks passed for FEATURE level
```

### Prompt Submit Hook

Analyzes your prompt and suggests optimal model before execution.

**How it works:**
- Scans prompt for keywords indicating complexity
- Suggests model (Haiku/Sonnet/Opus) and effort level
- Provides reasoning for suggestion
- You can accept or override

**Example:**
```
> Optimize memory allocation for KV-cache with INT4 quantization

💡 Suggested: Opus (high effort)
   Reason: Complex optimization task

Proceed with Opus? [Y/n]
```

## Quality Gates

### Progressive CI Integration

Claude Code integrates with Tritter's progressive CI system.

**How it works:**
1. Pre-commit hook runs quality checks
2. Checks adapt based on branch type
3. Commit blocked if critical issues found
4. Warnings shown but commit allowed (for feature branches)

**Manual quality check:**
```bash
# Run full CI locally
.github/scripts/run-ci.sh

# Check specific aspects
ruff format --check .
mypy src/tritter
pytest --cov=src/tritter

# Security scan
python -m tritter.curation.secrets scan .
```

### Automated Fixes

Many issues can be auto-fixed:

```bash
# Auto-format code
ruff format .

# Auto-fix lint issues
ruff check --fix .

# Generate missing docstrings (with Claude)
claude "add Google-style docstrings to all functions in architecture.py"
```

## Best Practices

### 1. Work on Feature Branches

```bash
# Create feature branch for experimentation
git checkout -b feature/my-experiment

# Lenient CI allows rapid iteration
# Commit frequently, refactor later
```

### 2. Use Descriptive Prompts

**Good prompts:**
```
Implement FlashAttention in TritterAttention with QK-Norm, maintaining
compatibility with existing KV-cache quantization. Ensure is_causal=True
for optimal kernel dispatch.

Add comprehensive unit tests for BitNet quantization round-trip precision,
testing all model sizes (1B, 3B, 7B) with random weight matrices.
```

**Poor prompts:**
```
fix attention

add tests
```

### 3. Let Hooks Work for You

Don't disable hooks unless necessary:
- Session start hook catches environment issues early
- Pre-commit hook prevents common mistakes
- Prompt submit hook optimizes cost and quality

### 4. Review Generated Code

Always review code Claude generates:
- Check tensor shapes match comments
- Verify docstrings include "Why" sections
- Ensure tests use `config.vocab_size`, not hardcoded values
- Confirm gradient tests check magnitude, not just existence

### 5. Use Context Wisely

Claude Code automatically includes key files in context (configured in `.clauderc`):
- `CLAUDE.md` - Project overview and guidelines
- `docs/DEVELOPMENT_STANDARDS.md` - Coding standards
- `docs/API_CONVENTIONS.md` - API design patterns

Reference these in prompts:
```
Following DEVELOPMENT_STANDARDS.md, add a Google-style docstring with "Why"
section to the quantize_weights function.
```

### 6. Leverage Auto-Memory

Claude Code learns from mistakes across sessions (if enabled in `.clauderc`):
- Corrections are remembered
- Common patterns are learned
- Architectural decisions are retained

Help it learn by providing clear feedback:
```
That's incorrect - BitNet requires Squared ReLU (x * ReLU(x)), not SiLU.
Please update the activation function and add a comment explaining why.
```

## Troubleshooting

### Model Selection Not Working

**Problem:** Always using Sonnet despite complex task.

**Solution:**
1. Check `.clauderc` configuration:
   ```ini
   [hooks]
   prompt_submit = ".github/hooks/prompt-submit.sh"
   ```

2. Test hook manually:
   ```bash
   echo "Optimize memory layout for quantization" | .github/hooks/prompt-submit.sh
   ```

3. Expected output: `opus|high|Complex optimization task`

### Pre-Commit Hook Blocking Commits

**Problem:** Can't commit despite fixing issues.

**Solution:**
1. Check branch strictness:
   ```bash
   .github/scripts/detect-branch-strictness.sh
   ```

2. Run checks manually to see failures:
   ```bash
   ruff format --check .
   ruff check .
   python -m tritter.curation.secrets scan .
   ```

3. Auto-fix if possible:
   ```bash
   ruff format .
   ruff check --fix .
   ```

4. Override hook temporarily (not recommended):
   ```bash
   git commit --no-verify -m "WIP: testing"
   ```

### Session Hook Errors

**Problem:** Session start hook fails.

**Solution:**
1. Check if hook file exists and is executable:
   ```bash
   ls -l .github/hooks/session-start.sh
   ```

2. Run hook manually to see error:
   ```bash
   bash .github/hooks/session-start.sh
   ```

3. Common issues:
   - Missing dependencies (install with `uv pip install -e ".[dev]"`)
   - Python version incompatibility (requires 3.12+)
   - Git repository issues (ensure `.git` exists)

### Type Check Failures

**Problem:** mypy reports errors on valid code.

**Solution:**
1. Check mypy configuration in `pyproject.toml`:
   ```toml
   [tool.mypy]
   strict = true
   ignore_missing_imports = true
   ```

2. Add type ignore comment for known issues:
   ```python
   result = some_function()  # type: ignore[attr-defined]
   ```

3. Add type stubs for missing libraries:
   ```bash
   uv pip install types-requests
   ```

### Coverage Too Low

**Problem:** CI fails due to insufficient test coverage.

**Solution:**
1. Check current coverage with detailed report:
   ```bash
   pytest --cov=src/tritter --cov-report=term-missing
   ```

2. Identify untested code (shown in terminal output)

3. Ask Claude to generate tests:
   ```
   Generate comprehensive unit tests for src/tritter/quantization/bitnet.py,
   covering TernaryWeight quantization, straight-through estimator gradients,
   and pack/unpack round-trip precision. Follow existing test patterns in
   tests/unit/test_quantization.py.
   ```

4. Verify coverage improved:
   ```bash
   pytest --cov=src/tritter --cov-report=html
   open htmlcov/index.html  # View detailed HTML report
   ```

## Advanced Usage

### Custom Skills

Define reusable skills in `.github/skills/`:

```bash
# Example: .github/skills/validate
#!/bin/bash
python devtools/validate.py
```

Invoke with:
```bash
claude /validate
```

### Environment Variables

Override configuration via environment:

```bash
# Force Opus regardless of task
export CLAUDE_MODEL=opus

# Disable hooks for one session
export CLAUDE_NO_HOOKS=1

# Custom CI config location
export CI_CONFIG_FILE=.github/ci-config-custom.yml
```

### Integration with Other Tools

**Git aliases:**
```bash
# Add to ~/.gitconfig
[alias]
    claude-commit = !git add -A && claude "review changes and create commit with conventional message"
    claude-review = !claude "review current diff and suggest improvements"
```

**VS Code tasks:**
```json
{
  "label": "Claude Review",
  "type": "shell",
  "command": "claude 'review current file for improvements'"
}
```

## Resources

- [Main Project Documentation](../CLAUDE.md)
- [Development Standards](DEVELOPMENT_STANDARDS.md)
- [Progressive CI Configuration](../.github/ci-config.yml)
- [GitHub Actions Workflow](../.github/workflows/progressive-ci.yml)
- [Tritter Roadmap](ROADMAP.md)

---

**Questions?** Ask Claude Code directly or review the documentation files listed above.
