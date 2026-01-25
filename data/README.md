# Tritter Training Data

This directory contains curated training datasets for the Tritter multimodal transformer.

## Philosophy

**Quality over quantity.** Following the guidance in `docs/TRAINING_STRATEGY.md`, we prioritize:

- Permissive licensing (MIT, Apache-2.0, BSD variants)
- High-quality repositories (>100 stars, well-maintained)
- Aggressive deduplication (exact hash + near-duplicate filtering)
- Domain-specific filtering (remove auto-generated code, minified files)

**Contrastive learning**: We include both positive (high-quality) and negative (poor-quality) examples. Negative examples are labeled with explanations so the model learns to identify and explain bad code.

## Directory Structure

```
data/
├── README.md                    # This file
├── curated_repos.json          # High-quality repository lists
├── sample/                     # Sample data for testing pipelines
│   ├── python_positive.jsonl   # High-quality Python examples
│   ├── python_negative.jsonl   # Low-quality examples with explanations
│   └── python_mixed.jsonl      # Mixed quality for realistic testing
├── curated/                    # Filtered datasets ready for training
│   ├── python_curated.jsonl
│   ├── rust_curated.jsonl
│   └── javascript_curated.jsonl
├── pretrain/                   # Sharded data from prepare_pretrain_data.py
│   ├── shard_00000.jsonl
│   ├── shard_00001.jsonl
│   └── ...
├── raw/                        # Unprocessed downloads (gitignored)
└── processed/                  # Intermediate processing stages (gitignored)
```

## Data Preparation

### Quick Start with Sample Data

The `data/sample/` directory contains test data to validate the training pipeline:

```bash
# Test the training pipeline with sample data
python scripts/train_pretrain.py --model 1B --data-dir data/sample --dry-run
```

### Preparing Custom Training Data

Use `prepare_pretrain_data.py` to process source code files through the curation pipeline:

```bash
# Basic usage - process a directory of code files
python scripts/prepare_pretrain_data.py \
    --input-dir /path/to/code \
    --output-dir data/pretrain \
    --shard-size 10000

# With quality filtering (only high-quality samples)
python scripts/prepare_pretrain_data.py \
    --input-dir /path/to/code \
    --output-dir data/pretrain \
    --min-quality 0.7 \
    --positive-only

# Process specific languages with parallel workers
python scripts/prepare_pretrain_data.py \
    --input-dir /path/to/code \
    --output-dir data/pretrain \
    --languages python rust \
    --workers 8
```

**Command-line options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | Required | Directory containing source code files |
| `--output-dir` | Required | Directory to write JSONL shards |
| `--shard-size` | 10000 | Number of samples per shard file |
| `--min-quality` | 0.5 | Minimum quality score for positive samples |
| `--positive-only` | False | Exclude negative examples from output |
| `--languages` | All | Languages to include (python, rust, etc.) |
| `--workers` | 1 | Number of parallel workers |
| `--dry-run` | False | Count files without processing |

### Curating from The Stack v2

For large-scale data from HuggingFace, use `curate_datasets.py`:

```bash
# List available subsets
python scripts/curate_datasets.py --list-subsets

# Download and filter Python subset
python scripts/curate_datasets.py --language python --output data/curated

# Limited sample for testing (1000 samples)
python scripts/curate_datasets.py --language python --max-samples 1000 --output data/curated

# High-quality filter (min 500 stars)
python scripts/curate_datasets.py --language python --min-stars 500 --output data/curated
```

## Output Format

All curated datasets are saved as JSONL (JSON Lines) with the following schema:

### Positive Samples (High Quality)

```json
{
  "text": "def hello():\n    \"\"\"Say hello.\"\"\"\n    print('world')",
  "language": "python",
  "quality_score": 0.95,
  "source": "examples/hello.py"
}
```

### Negative Samples (For Contrastive Learning)

```json
{
  "text": "def f(x):\n    return eval(x)",
  "language": "python",
  "quality_score": 0.15,
  "quality_label": "negative",
  "explanation": "This code has security issues:\n- eval() allows arbitrary code execution.\nFix: Use ast.literal_eval() for safe evaluation."
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Full code content |
| `language` | string | Programming language (python, rust, javascript) |
| `quality_score` | float | Quality score [0.0, 1.0] from curation pipeline |
| `quality_label` | string | "positive" or "negative" (omitted for positive) |
| `explanation` | string | For negative samples, why the code is bad |
| `source` | string | Relative file path or repository identifier |
| `license` | string | SPDX license identifier (if from external source) |
| `stars` | integer | GitHub stars (if from external source) |

## Sample Data

The `data/sample/` directory contains pre-curated examples for testing:

### python_positive.jsonl (5 samples)

High-quality Python code with proper documentation:
- Fibonacci function with full docstring
- Generic Result type (Rust-inspired error handling)
- Timing decorator with proper typing
- File walker utility
- LRU cache implementation

### python_negative.jsonl (6 samples)

Code with issues for contrastive learning:
- `eval()` vulnerability (code injection)
- `pickle.load()` and `shell=True` (security issues)
- Excessive parameters and unclear names (quality issues)
- Deep nesting (maintainability issues)
- Hardcoded credentials (CRITICAL - would be rejected in real pipeline)
- SQL injection vulnerability

### python_mixed.jsonl (5 samples)

Mix of positive and negative for realistic training:
- Deep merge utility (positive)
- Bare except clause (negative)
- Secure password hashing (positive)
- Unclear variable names (negative)
- Timeout context manager (positive)

## Quality Criteria

From `TRAINING_STRATEGY.md`, we apply these filters:

### License Filtering

Only permissive licenses are included:
- `mit`
- `apache-2.0`
- `bsd-2-clause`
- `bsd-3-clause`
- `isc`
- `unlicense`
- `cc0-1.0`

### Security Filtering (SPEC-007)

| Issue Type | Action |
|------------|--------|
| Hardcoded secrets | **ALWAYS REJECT** (never train on secrets) |
| SQL injection | Label as negative + explanation |
| Code injection (eval/exec) | Label as negative + explanation |
| Shell injection | Label as negative + explanation |
| Unsafe deserialization | Label as negative + explanation |

### Quality Filtering

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Line count | 5-10,000 | Filter trivial and auto-generated |
| File size | <500KB | Filter minified/bundled code |
| Alpha ratio | >25% | Filter binary/encoded data |
| Long lines | <10% over 500 chars | Filter minified code |
| Quality score | >0.5 (configurable) | Threshold for positive label |

### Auto-Generated Detection

Files with these markers in first 20 lines are rejected:
- "auto-generated"
- "do not edit"
- "generated by"
- "this file is generated"
- "autogenerated"

### Deduplication

1. **Exact deduplication**: MD5 hash-based removal (in-memory during processing)
2. **Near-duplicate**: Post-processing with MinHash (Jaccard 0.7-0.8)
3. **Benchmark contamination**: Post-processing removal of HumanEval, MBPP, DS-1000

## Training Data Mix

From `TRAINING_STRATEGY.md`, Phase 1 continued pretraining targets:

| Component | Percentage | Tokens | Source |
|-----------|------------|--------|--------|
| Python Code | 40% | 40B | Stack v2 Python |
| Rust Code | 25% | 25B | Stack v2 Rust |
| High-quality repos | 20% | 20B | Curated repos list |
| Technical docs | 10% | 10B | Framework docs |
| Persona conversations | 5% | 5B | Synthetic |

Total: ~100B tokens for Phase 1

## Integration with Training

Once prepared, datasets can be loaded for training:

```python
# Direct usage with train_pretrain.py
python scripts/train_pretrain.py --model 3B --data-dir data/pretrain

# Or load manually
from datasets import load_dataset

# Load curated dataset
ds = load_dataset("json", data_files="data/pretrain/shard_*.jsonl")

# Streaming mode for large datasets
ds = load_dataset(
    "json",
    data_files="data/pretrain/shard_*.jsonl",
    streaming=True,
)
```

## Storage Requirements

| Dataset | Raw Size | Filtered Size (est.) |
|---------|----------|----------------------|
| Python | 233 GB | 35-70 GB (15-30%) |
| Rust | 15.6 GB | 2.3-4.7 GB (15-30%) |
| JavaScript | 87 GB | 13-26 GB (15-30%) |

Ensure sufficient disk space before starting curation.

## References

- [TRAINING_STRATEGY.md](../docs/TRAINING_STRATEGY.md) - Overall training strategy
- [SPEC-007](../docs/specs/SPEC-007-dataset-quality-gates.md) - Quality gates specification
- [clean-datasets.md](../docs/clean-datasets.md) - Dataset sources and quality criteria
- [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) - Primary data source
- [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) - Educational quality subset

---

*Last Updated: 2026-01-24*
