# Tritter CI/CD and Tooling

This directory contains progressive CI/CD configuration, Claude Code integration, and automated quality gates for the Tritter project.

## Overview

The Tritter project uses **progressive strictness** - CI requirements adapt based on branch type, enabling rapid development on feature branches while maintaining high standards for production.

## Directory Structure

```
.github/
├── ci-config.yml              # Progressive CI configuration
├── workflows/
│   └── progressive-ci.yml     # GitHub Actions workflow
├── scripts/
│   ├── detect-branch-strictness.sh  # Branch strictness detection
│   └── run-ci.sh              # Main CI runner
├── hooks/
│   ├── session-start.sh       # Claude Code session initialization
│   ├── pre-commit.sh          # Pre-commit quality gates
│   └── prompt-submit.sh       # Intelligent task routing
└── README.md                  # This file
```

## Progressive CI Levels

| Level | Branch Type | Philosophy | Requirements |
|-------|-------------|------------|--------------|
| **0 - FEATURE** | `feature/*`, `claude/*`, `fix/*` | 🚀 Move fast | Warnings only, 50% coverage |
| **1 - DEVELOPMENT** | `dev`, `develop` | 🔧 Integrate | Format required, 70% coverage |
| **2 - RELEASE** | `release/*`, `rc/*` | 🎯 Polish | Strict lint, 80% coverage |
| **3 - PRODUCTION** | `main`, `master` | 🔒 Perfect | Zero tolerance, 85% coverage |

## Quick Start

### Running CI Locally

```bash
# Run all checks for current branch
.github/scripts/run-ci.sh

# Check what strictness level applies
.github/scripts/detect-branch-strictness.sh

# Override strictness level
STRICTNESS_LEVEL=3 .github/scripts/run-ci.sh
```

### Claude Code Integration

The `.clauderc` file configures Claude Code CLI behavior:

- **Smart model selection**: Opus for architecture, Haiku for simple tasks
- **Session hooks**: Environment verification on startup
- **Pre-commit hooks**: Quality gates before committing
- **Auto-memory**: Learn from mistakes across sessions

**Session start hook** (runs automatically):
```bash
# Checks Python version, dependencies, CUDA, git status
# Provides helpful command shortcuts
```

**Pre-commit hook** (runs before `git commit`):
```bash
# Enforces quality gates based on branch strictness
# Always checks for hardcoded secrets
```

## Configuration

### Customizing CI Strictness

Edit `.github/ci-config.yml`:

```yaml
strictness_levels:
  feature:
    coverage_threshold: 50  # Lower to 40% for experimental work
    lint: warn              # Change to 'skip' to disable

  production:
    coverage_threshold: 90  # Raise to 90% for critical systems
```

### Adding Custom Checks

Add custom validation in `ci-config.yml`:

```yaml
custom_checks:
  my_custom_check:
    command: "python scripts/validate_something.py"
    feature: skip
    development: warn
    release: error
    production: error
```

### Branch Pattern Matching

Customize branch patterns in `ci-config.yml`:

```yaml
branch_patterns:
  feature: "feature/*, experimental/*, wip/*, claude/*"
  production: "main, master, stable, live"
```

## Tritter-Specific Checks

The CI system includes specialized checks for Tritter:

1. **Development Standards** (`devtools/validate.py`)
   - Enforces Google-style docstrings with "Why" sections
   - Validates tensor shape comments
   - Checks for config usage in tests

2. **Import Validation** (`devtools/import_validator.py`)
   - Ensures `__all__` matches actual imports
   - Prevents import inconsistencies

3. **Security Scan** (`tritter.curation.secrets`)
   - Detects hardcoded secrets, API keys
   - **Always fails** regardless of strictness level

4. **Quantization Validation**
   - Verifies BitNet pack/unpack round-trip
   - Ensures ternary weight precision

5. **Memory Estimation**
   - Validates memory calculations for RTX 5080
   - Checks hardware profile accuracy

## GitHub Actions Workflow

The `progressive-ci.yml` workflow runs on:
- All pushes to main branches
- All pull requests
- Feature and release branches

**Features**:
- Automatic strictness detection
- Coverage report uploads
- PR comments with CI results
- Optional GPU testing (self-hosted runners)
- Security scanning (always runs)

## Hooks

### Session Start Hook

Runs when Claude Code session starts:
- Verifies Python, dependencies, CUDA
- Shows current branch and CI strictness
- Provides helpful command shortcuts

### Pre-Commit Hook

Runs before committing code:
- Quick format/lint checks
- Security scan (always enforced)
- Type checking (for release/production)
- Adapts to branch strictness level

### Prompt Submit Hook

Analyzes user prompts and suggests optimal model:
- Architecture/optimization → Opus (high effort)
- Simple formatting → Haiku (low effort)
- General development → Sonnet (medium effort)

## Testing Locally

### Test CI Scripts

```bash
# Test branch detection
.github/scripts/detect-branch-strictness.sh

# Test full CI pipeline
.github/scripts/run-ci.sh

# Test specific check
ruff format --check .
mypy src/tritter
pytest --cov=src/tritter
```

### Simulate Different Branches

```bash
# Test as if on feature branch
export BRANCH="feature/test"
.github/scripts/run-ci.sh

# Test as if on production
export BRANCH="main"
.github/scripts/run-ci.sh
```

## Troubleshooting

### CI Fails on Simple Changes

Check your branch name and strictness level:
```bash
.github/scripts/detect-branch-strictness.sh
```

Feature branches allow warnings; main requires perfection.

### Format/Lint Failures

Auto-fix most issues:
```bash
ruff format .
ruff check --fix .
```

### Type Check Failures

Run mypy to see specific errors:
```bash
mypy src/tritter
```

### Coverage Too Low

Run with verbose output to see what's missing:
```bash
pytest --cov=src/tritter --cov-report=term-missing
```

### Security Scan False Positives

Review the detection logic in `src/tritter/curation/secrets.py`.
Never commit actual secrets - use environment variables.

## Best Practices

1. **Feature Development**
   - Work on `feature/*` branches for lenient CI
   - Commit frequently, iterate fast
   - Fix warnings before merging to dev

2. **Integration Testing**
   - Merge to `develop` branch
   - CI enforces formatting and 70% coverage
   - Fix issues before release

3. **Production Releases**
   - Merge to `main` only from release branches
   - CI enforces strict quality gates
   - Zero warnings or failures allowed

4. **Using Claude Code**
   - Review session start output for environment status
   - Trust the pre-commit hook to catch issues early
   - Let prompt routing suggest optimal models

## Advanced Configuration

### Custom Strictness Levels

You can define custom levels in `ci-config.yml`:

```yaml
strictness_levels:
  custom_level:
    format: error
    lint: error
    coverage_threshold: 75
    # ... other settings
```

### Environment-Specific Settings

Override settings via environment variables:

```bash
export CHECK_FORMAT=skip
export CHECK_COVERAGE_THRESHOLD=60
.github/scripts/run-ci.sh
```

### Integration with Other CI Systems

The scripts work with GitLab CI, Jenkins, CircleCI, etc.:

```yaml
# .gitlab-ci.yml
quality_gate:
  script:
    - .github/scripts/run-ci.sh
```

## Contributing

When adding new checks:

1. Add check definition to `ci-config.yml` under `custom_checks`
2. Specify behavior for each strictness level
3. Update this README with check description
4. Test locally before committing

## References

- [Progressive CI Boilerplate](https://github.com/tzervas/ci-boilerplate)
- [Claude Usage Boilerplate](https://github.com/tzervas/claude-usage-boilerplate)
- [Tritter Development Standards](../docs/DEVELOPMENT_STANDARDS.md)
- [Tritter API Conventions](../docs/API_CONVENTIONS.md)

---

**Questions or issues?** Review the workflow runs in GitHub Actions or run checks locally for debugging.
