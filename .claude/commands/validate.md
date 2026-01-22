# Full Validation

Run complete validation suite before committing.

## Instructions

Run all validation steps in sequence:

1. **Format check**:
```bash
ruff format --check .
```

2. **Lint check**:
```bash
ruff check .
```

3. **Type check**:
```bash
mypy src/tritter
```

4. **Import verification**:
```bash
python -c "from tritter import *; print('Imports OK')"
```

5. **Run tests**:
```bash
pytest
```

Report a summary of all checks. If any fail, provide specific guidance on fixing them.

## Pre-Commit Checklist

Also verify these DEVELOPMENT_STANDARDS.md requirements:
- All new functions/classes have Google-style docstrings with "Why" section
- Tensor shapes are documented in comments
- Tests use config values (not hardcoded numbers)
- `__all__` exports match imports in `__init__.py` files
