# Tritter Development Tools

Development utilities for the Tritter project. This module is separate from the core Tritter model code and provides tooling for development workflows.

## Purpose

The `devtools/` module provides:

- **Validation Runner**: Run format, lint, typecheck, and test validation in one command
- **Project Analyzer**: Analyze project structure and implementation status
- **CLI Interface**: Command-line tools for development tasks

## Usage

### Validation

Run the full validation suite:

```bash
python -m devtools validate
```

Quick validation (skip tests):

```bash
python -m devtools validate --quick
```

### Project Status

Show implementation status:

```bash
python -m devtools status
```

JSON output for automation:

```bash
python -m devtools status --json
```

Implementation roadmap:

```bash
python -m devtools status --roadmap
```

## Structure

```
devtools/
├── __init__.py       # Package exports
├── __main__.py       # CLI entry point
├── validate.py       # Validation runner
├── project_info.py   # Project analysis
└── py.typed          # PEP 561 type marker
```

## Separation from Core

This module is intentionally separate from `src/tritter/` because:

1. **Clean boundaries**: Model code should not depend on development tooling
2. **Optional dependency**: Devtools can have different dependencies than core
3. **Portability**: Core model can be distributed without development infrastructure
4. **Clarity**: Clear distinction between the research artifact and its development tools
