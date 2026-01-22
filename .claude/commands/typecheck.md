# Type Check

Run mypy type checking with strict mode.

## Instructions

Run mypy on the source directory:

```bash
mypy src/tritter
```

If there are type errors:
1. List all errors grouped by file
2. For each error, explain what's wrong and how to fix it
3. Prioritize errors that would cause runtime issues

Remember: This project uses Python 3.12+ type hints including `X | None` union syntax.
