# Dataset Quality Gates Specification

**Spec ID**: SPEC-007
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-23

## 1. Overview

### 1.1 Purpose

Define quality gates for training data curation that ensure:
1. No insecure code is used as positive examples
2. Bad code is explicitly labeled as negative examples
3. Development practices, methodologies, and workflows are included
4. The model learns to distinguish good from bad code

### 1.2 Philosophy

**Contrastive learning**: The model should learn both:
- What GOOD code looks like (positive examples)
- What BAD code looks like (negative examples with explicit labels)
- WHY bad code is bad (explanatory context)

This prevents the model from accidentally learning bad patterns while enabling it to identify and explain poor code.

---

## 2. Quality Gate Categories

### 2.1 Security Gates

| Gate | Tool | Action on Failure |
|------|------|-------------------|
| SQL Injection | semgrep, bandit | Label as negative + security warning |
| Command Injection | semgrep, bandit | Label as negative + security warning |
| XSS Vulnerabilities | semgrep | Label as negative + security warning |
| Hardcoded Secrets | detect-secrets, trufflehog | REJECT (no training) |
| Insecure Crypto | bandit | Label as negative + explanation |
| Path Traversal | semgrep | Label as negative + security warning |
| Unsafe Deserialization | bandit | Label as negative + security warning |

**Critical rule**: Hardcoded secrets are NEVER included, even as negative examples. Risk of leakage too high.

### 2.2 Code Quality Gates

| Gate | Metric | Positive Threshold | Negative Label If |
|------|--------|-------------------|-------------------|
| Cyclomatic Complexity | radon (Python) | < 10 per function | > 20 |
| Maintainability Index | radon | > 20 | < 10 |
| Type Coverage | mypy | > 80% | < 20% |
| Lint Score | ruff/pylint | < 5 errors | > 50 errors |
| Documentation | docstring presence | > 50% | < 10% |
| Test Coverage | pytest-cov | > 60% | < 10% |

### 2.3 Anti-Pattern Detection

| Anti-Pattern | Detection Method | Training Use |
|--------------|------------------|--------------|
| God classes | LOC > 1000, methods > 30 | Negative example |
| Spaghetti code | High coupling metrics | Negative example |
| Copy-paste code | Clone detection | Negative example |
| Magic numbers | AST analysis | Negative example |
| Deep nesting | AST depth > 5 | Negative example |
| Long functions | LOC > 100 | Negative example |
| Unused imports | Static analysis | Negative example |
| Bare except | AST pattern | Negative example |

---

## 3. Data Labeling Schema

### 3.1 Sample Format

```json
{
  "text": "<code content>",
  "language": "python",
  "quality_label": "positive|negative|educational",
  "quality_score": 0.85,
  "security_issues": [],
  "anti_patterns": [],
  "quality_issues": [],
  "explanation": "Optional: why this is good/bad",
  "source": {
    "repo": "author/repo",
    "path": "src/module.py",
    "license": "MIT",
    "stars": 1500
  },
  "metadata": {
    "complexity": 5.2,
    "maintainability": 72.0,
    "type_coverage": 0.92,
    "lint_score": 9.8
  }
}
```

### 3.2 Quality Labels

| Label | Meaning | Training Weight |
|-------|---------|-----------------|
| `positive` | High-quality, secure, well-structured | 1.0 (standard) |
| `negative` | Poor quality with explicit issues | 0.3 (reduced, contrastive) |
| `educational` | Code review, refactoring example | 1.2 (emphasized) |
| `methodology` | Process docs, workflows, practices | 1.0 (standard) |

### 3.3 Negative Example Format

Negative examples MUST include explanation:

```json
{
  "text": "def process(data):\n    return eval(data)  # DANGEROUS",
  "quality_label": "negative",
  "security_issues": ["code_injection"],
  "explanation": "Using eval() on untrusted input allows arbitrary code execution. Use ast.literal_eval() for safe parsing or proper deserialization.",
  "better_alternative": "def process(data):\n    import ast\n    return ast.literal_eval(data)"
}
```

---

## 4. Security Scanning Pipeline

### 4.1 Python Security Checks

```python
SECURITY_TOOLS = {
    "bandit": {
        "command": "bandit -r {file} -f json",
        "severity_threshold": "MEDIUM",
        "confidence_threshold": "MEDIUM",
    },
    "semgrep": {
        "command": "semgrep --config=p/python --json {file}",
        "rules": ["security", "correctness"],
    },
    "detect-secrets": {
        "command": "detect-secrets scan {file}",
        "action_on_find": "REJECT",  # Never include secrets
    },
}
```

### 4.2 Rust Security Checks

```python
RUST_SECURITY_TOOLS = {
    "cargo-audit": {
        "command": "cargo audit --json",
        "check_dependencies": True,
    },
    "cargo-clippy": {
        "command": "cargo clippy --message-format=json",
        "deny": ["clippy::unwrap_used", "clippy::expect_used"],
    },
    "semgrep": {
        "command": "semgrep --config=p/rust --json {file}",
    },
}
```

### 4.3 Triton Security Checks

```python
TRITON_SECURITY_CHECKS = {
    "patterns": [
        # Memory safety
        (r"tl\.load\([^)]*,\s*mask\s*=\s*None", "unmasked_load", "Use mask for bounds checking"),
        (r"tl\.store\([^)]*,\s*mask\s*=\s*None", "unmasked_store", "Use mask for bounds checking"),
        (r"tl\.atomic_add\([^)]*\)(?!.*mask)", "unmasked_atomic", "Atomics need bounds checking"),
        # Resource limits
        (r"BLOCK_SIZE\s*=\s*\d{5,}", "excessive_block_size", "May exceed shared memory"),
        (r"num_warps\s*=\s*(?:1[7-9]|[2-9]\d)", "excessive_warps", "Usually 1-8 is optimal"),
        # Precision/correctness
        (r"tl\.where\([^)]*,\s*[^,)]*\s*/\s*[^,)]*,", "division_in_where", "Division in both branches"),
        (r"\.to\(tl\.float16\)(?!.*tl\.float32)", "precision_loss", "Accumulate in float32"),
    ],
    "action": "label_negative",  # Triton issues are usually correctness, not security
}
```

### 4.4 Universal Checks

```python
UNIVERSAL_CHECKS = {
    "secrets": {
        "patterns": [
            r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][^'\"]+['\"]",
            r"(?i)(secret|password|passwd|pwd)\s*[=:]\s*['\"][^'\"]+['\"]",
            r"(?i)bearer\s+[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+",
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub token
            r"sk-[a-zA-Z0-9]{48}",   # OpenAI key
            r"AKIA[0-9A-Z]{16}",     # AWS key
        ],
        "action": "REJECT",
    },
    "unsafe_patterns": {
        "python": [
            (r"\beval\s*\(", "code_injection", "Use ast.literal_eval()"),
            (r"\bexec\s*\(", "code_injection", "Avoid dynamic code execution"),
            (r"subprocess.*shell\s*=\s*True", "command_injection", "Use shell=False with list args"),
            (r"pickle\.loads?\(", "unsafe_deserialization", "Use JSON or safe formats"),
            (r"yaml\.load\([^)]*\)", "unsafe_deserialization", "Use yaml.safe_load()"),
        ],
        "rust": [
            (r"unsafe\s*\{", "unsafe_block", "Document safety invariants"),
            (r"\.unwrap\(\)", "panic_risk", "Use proper error handling"),
        ],
        "triton": [
            (r"tl\.load\([^)]*mask\s*=\s*None", "unmasked_load", "Always mask loads at boundaries"),
            (r"tl\.store\([^)]*mask\s*=\s*None", "unmasked_store", "Always mask stores at boundaries"),
            (r"BLOCK_SIZE\s*=\s*\d{5,}", "resource_limit", "Block size may exceed GPU limits"),
        ],
    },
}
```

---

## 5. Development Practice Content

### 5.1 Methodology Sources

| Category | Sources | Format |
|----------|---------|--------|
| Code Reviews | GitHub PR reviews, Gerrit | Diff + comments |
| Refactoring | Before/after examples | Paired samples |
| Design Patterns | Pattern implementations | Annotated code |
| Testing | Test suites with rationale | Test + explanation |
| Documentation | Well-documented codebases | Code + docs |
| Workflows | CI/CD configs, Makefiles | Config + explanation |

### 5.2 Educational Content Schema

```json
{
  "type": "refactoring_example",
  "before": {
    "text": "<original code>",
    "issues": ["high_complexity", "no_types", "magic_numbers"]
  },
  "after": {
    "text": "<refactored code>",
    "improvements": ["extracted_functions", "added_types", "named_constants"]
  },
  "explanation": "Why these changes improve the code...",
  "principles": ["single_responsibility", "explicit_over_implicit"]
}
```

### 5.3 Workflow Content

Include examples of:
- Git workflows (branching, commits, PRs)
- CI/CD pipelines (GitHub Actions, etc.)
- Testing strategies (unit, integration, e2e)
- Documentation practices (docstrings, READMEs)
- Code review comments (constructive feedback)
- Issue templates and bug reports

---

## 6. Implementation Pipeline

### 6.1 Processing Stages

```
Raw Code
    │
    ▼
┌─────────────────┐
│ Secret Scanner  │ ──REJECT──► /dev/null (never train on secrets)
└────────┬────────┘
         │ pass
         ▼
┌─────────────────┐
│ Security Scan   │ ──issues──► Label as NEGATIVE + explanation
└────────┬────────┘
         │ clean
         ▼
┌─────────────────┐
│ Quality Metrics │ ──poor──► Label as NEGATIVE + issues
└────────┬────────┘
         │ good
         ▼
┌─────────────────┐
│ Anti-Pattern    │ ──found──► Label as NEGATIVE + patterns
│ Detection       │
└────────┬────────┘
         │ clean
         ▼
┌─────────────────┐
│ Label POSITIVE  │
│ Add metadata    │
└────────┬────────┘
         │
         ▼
    Training Data
```

### 6.2 Negative Example Quotas

To maintain balanced training:

| Dataset Split | Positive | Negative | Educational |
|---------------|----------|----------|-------------|
| Pretraining   | 85%      | 10%      | 5%          |
| Fine-tuning   | 70%      | 15%      | 15%         |

---

## 7. Metrics and Monitoring

### 7.1 Curation Metrics

| Metric | Target | Alert If |
|--------|--------|----------|
| Secret rejection rate | <1% | >5% (data source issue) |
| Security negative rate | 5-15% | >30% (filter too strict) |
| Quality negative rate | 10-20% | >40% (filter too strict) |
| Average quality score | >0.7 | <0.5 |

### 7.2 Training Metrics

Monitor during training:
- Loss on positive vs negative examples
- Model's ability to identify bad code
- Security issue detection accuracy
- Code quality assessment correlation

---

## 8. Tool Requirements

### 8.1 Python Dependencies

```
bandit>=1.7.0
semgrep>=1.0.0
detect-secrets>=1.4.0
radon>=6.0.0
ruff>=0.1.0
mypy>=1.0.0
```

### 8.2 Rust Dependencies

```
cargo-audit
cargo-clippy
```

---

## Appendix A: OWASP Top 10 Mapping

| OWASP Category | Detection Method | Training Action |
|----------------|------------------|-----------------|
| A01 Broken Access Control | semgrep rules | Negative + explanation |
| A02 Cryptographic Failures | bandit, semgrep | Negative + explanation |
| A03 Injection | bandit, regex | Negative + explanation |
| A04 Insecure Design | Manual curation | Educational examples |
| A05 Security Misconfiguration | semgrep | Negative + explanation |
| A06 Vulnerable Components | cargo-audit, safety | REJECT or negative |
| A07 Auth Failures | semgrep | Negative + explanation |
| A08 Data Integrity | semgrep | Negative + explanation |
| A09 Logging Failures | semgrep | Negative + explanation |
| A10 SSRF | semgrep | Negative + explanation |

---

*Last Updated: 2026-01-23*
