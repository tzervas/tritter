# CI Usage Guide - Local-First Approach

## Philosophy

**Local checks first, CI for verification.**

- 💰 **Cost optimization**: CI is expensive, local is free
- ⚡ **Speed**: Local checks run in seconds, CI in minutes
- 🔄 **Iteration**: Fix issues immediately without waiting for CI
- ✅ **Parity**: Local checks are IDENTICAL to CI - same script, same results

## Quick Start

### Run All Checks Locally

```bash
# This is THE command to run before pushing
bash scripts/run-checks-local.sh
```

This runs the **exact same script** that GitHub Actions uses, guaranteeing identical results.

### What Gets Checked

The checks adapt based on your current branch:

| Branch Type | Strictness | Checks |
|-------------|------------|--------|
| `feature/*`, `claude/*` | Level 0 | Warnings only, 50% coverage |
| `develop`, `dev` | Level 1 | Format required, 70% coverage |
| `release/*` | Level 2 | Strict lint, 80% coverage |
| `main`, `master` | Level 3 | Zero tolerance, 85% coverage |

**Example output**:
```
╔═══════════════════════════════════════════════════════╗
║  Tritter Local Quality Checks                        ║
║  (Exact parity with GitHub Actions CI)               ║
╚═══════════════════════════════════════════════════════╝

Pre-flight checks...
  Python: 3.12.1
  PyTorch: 2.2.0+cu121
  Tritter: Installed

Running CI checks...
Branch: feature/my-feature
Strictness Level: Level 0 (FEATURE)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check: Format Check
Config: warn
Command: ruff format --check .

⚠️  Warning (not blocking for FEATURE level)
Would reformat: src/tritter/models/architecture.py
...

✅ All checks passed for FEATURE level

Checks Run: 12
Passed: 5
Warnings: 5
Failed: 0
```

## GitHub Actions CI (Manual Only)

CI is configured for **manual trigger only** to minimize costs.

### When to Run CI

Only run CI for:
- ✅ Final verification before merging to `main`/`develop`
- ✅ Release branches before deployment
- ✅ When you can't run checks locally (no local environment)

**DO NOT run CI for**:
- ❌ Every feature branch commit
- ❌ Experimental work in progress
- ❌ Quick fixes or typos

### How to Trigger CI Manually

1. **Via GitHub Web UI**:
   - Go to Actions tab
   - Select "Progressive CI Quality Gates"
   - Click "Run workflow"
   - Choose branch and optional strictness override
   - Click "Run workflow" button

2. **Via GitHub CLI**:
   ```bash
   # Run on current branch with auto-detected strictness
   gh workflow run progressive-ci.yml

   # Run on specific branch
   gh workflow run progressive-ci.yml --ref feature/my-branch

   # Override strictness level
   gh workflow run progressive-ci.yml -f strictness_override=3
   ```

### CI Workflow Inputs

- **branch**: Branch to run on (optional, defaults to current)
- **strictness_override**: Force specific level (0-3, optional)
  - `0` = FEATURE (warnings only)
  - `1` = DEVELOPMENT (format required)
  - `2` = RELEASE (strict lint)
  - `3` = PRODUCTION (zero tolerance)

## Local vs CI: Exact Parity

Both local and CI run the **same script**: `.github/scripts/run-ci.sh`

**Local command**:
```bash
bash scripts/run-checks-local.sh
```

**CI command** (in GitHub Actions):
```yaml
bash .github/scripts/run-ci.sh
```

The only difference is the environment setup (dependency installation).

## Common Workflows

### Feature Development

```bash
# Start feature branch
git checkout -b feature/my-feature

# Make changes
vim src/tritter/...

# Run checks (fast, local)
bash scripts/run-checks-local.sh

# If checks pass, commit
git commit -m "feat: implement my feature"

# Push (no CI runs automatically)
git push origin feature/my-feature

# Create PR (still no CI)
gh pr create

# CI NOT needed for feature branches
# Merge when approved
```

### Release Preparation

```bash
# Create release branch
git checkout -b release/v0.3.0

# Run local checks (strict mode)
bash scripts/run-checks-local.sh
# Expected: Strictness Level 2 (RELEASE)

# If all pass, manually trigger CI for verification
gh workflow run progressive-ci.yml --ref release/v0.3.0

# Wait for CI, then merge if green
```

### Production Merge

```bash
# Before merging to main, run local checks
git checkout main
git merge --no-ff release/v0.3.0 --no-commit

bash scripts/run-checks-local.sh
# Expected: Strictness Level 3 (PRODUCTION)
# All checks must pass (no warnings)

# If local passes, trigger CI for final verification
gh workflow run progressive-ci.yml --ref main

# Merge only after CI passes
git merge --no-ff release/v0.3.0
git push origin main
```

## Troubleshooting

### Local checks fail, need to debug

```bash
# Run individual checks
ruff format --check .
ruff check .
mypy src/tritter
pytest --cov=src/tritter

# Auto-fix formatting
ruff format .
ruff check --fix .
```

### CI shows different results than local

**This should never happen** - they use the same script.

If it does:
1. Check Python version matches (3.12)
2. Check dependencies match: `pip freeze > requirements-freeze.txt`
3. Compare environment variables
4. File an issue - this is a bug

### Want to skip checks temporarily

```bash
# For local development only (NOT for commits)
git commit --no-verify -m "WIP: testing"

# But run checks before pushing!
bash scripts/run-checks-local.sh
```

### Dependencies not installed

```bash
# Local checks will warn but not fail
# Install dependencies to run full checks
uv pip install -e ".[dev,training,curation]"

# Or use standard pip
pip install -e ".[dev,training,curation]"
```

## Cost Savings

**Estimated monthly costs**:

| Approach | Cost | Checks/Month |
|----------|------|--------------|
| CI on every push | ~$50-200 | 500-2000 |
| Local-first + manual CI | ~$5-20 | 50-200 |

**Savings**: ~90% reduction by running locally first.

## Summary

✅ **DO**:
- Run `bash scripts/run-checks-local.sh` before every push
- Fix issues locally, iterate fast
- Only trigger CI manually for critical merges

❌ **DON'T**:
- Push without running local checks
- Trigger CI for every commit
- Ignore local check failures

---

**Questions?** See [.github/README.md](README.md) for full CI documentation.
