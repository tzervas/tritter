#!/usr/bin/env python3
"""Dataset curation pipeline for Tritter training.

Why: Quality over quantity. This script filters and prepares datasets according
to TRAINING_STRATEGY.md criteria: permissive license, well-maintained repos,
proper deduplication.

Usage:
    # List available subsets
    python scripts/curate_datasets.py --list-subsets

    # Download and filter Python subset
    python scripts/curate_datasets.py --language python --output data/python

    # Process with quality filtering
    python scripts/curate_datasets.py --language python --min-stars 100 --output data/python_filtered
"""

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Import quality gates - these integrate security and quality analysis
try:
    from security_scanner import SecurityScanner, ScanResult
    from quality_analyzer import QualityAnalyzer, QualityResult
    QUALITY_GATES_AVAILABLE = True
except ImportError:
    QUALITY_GATES_AVAILABLE = False
    SecurityScanner = None  # type: ignore
    QualityAnalyzer = None  # type: ignore


@dataclass
class DatasetConfig:
    """Configuration for dataset curation.

    Why: Centralize all filtering parameters to ensure consistency across runs
    and make it easy to adjust quality thresholds.

    Attributes:
        language: Programming language to filter (python, rust, javascript, triton)
        min_stars: Minimum GitHub stars for repository quality signal
        max_file_size_kb: Maximum file size to prevent minified/generated files
        min_lines: Minimum lines to filter out empty/trivial files
        max_lines: Maximum lines to filter out auto-generated files
        permissive_licenses: Tuple of license identifiers we can legally use
        output_format: File format for output (jsonl)
        enable_security_scan: Whether to run security vulnerability scanning
        enable_quality_scan: Whether to run code quality analysis
        include_negative_examples: Whether to include poor code as negative examples
        min_quality_score: Minimum quality score to accept as positive (0.0-1.0)
    """

    language: str = "python"
    min_stars: int = 100
    max_file_size_kb: int = 500
    min_lines: int = 10
    max_lines: int = 10000
    permissive_licenses: tuple[str, ...] = (
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "unlicense",
        "cc0-1.0",
    )
    output_format: str = "jsonl"
    enable_security_scan: bool = True
    enable_quality_scan: bool = True
    include_negative_examples: bool = True
    min_quality_score: float = 0.5


def get_stack_v2_subsets() -> dict[str, dict]:
    """Return available Stack v2 subsets with metadata.

    Why: The Stack v2 is our primary source of permissively-licensed code.
    This provides a reference for what data is available.

    Source: https://huggingface.co/datasets/bigcode/the-stack-v2

    Returns:
        Dictionary mapping language to metadata (size, file count, description)
    """
    return {
        "python": {
            "size_gb": 233,
            "files": 57_000_000,
            "description": "Python code from Software Heritage",
        },
        "rust": {
            "size_gb": 15.6,
            "files": 2_200_000,
            "description": "Rust code from Software Heritage",
        },
        "javascript": {
            "size_gb": 87,
            "files": 25_000_000,
            "description": "JavaScript/TypeScript code",
        },
        "triton": {
            "size_gb": 0.5,
            "files": 50_000,
            "description": "Triton GPU kernel code (curated from ML repos)",
            "note": "Not in Stack v2; requires custom curation from ML frameworks",
        },
    }


def is_permissive_license(license_name: str | None, config: DatasetConfig) -> bool:
    """Check if license is permissive.

    Why: Legal compliance is critical. We can only train on permissively-licensed
    code (MIT, Apache, BSD, etc.) to ensure commercial use is allowed.

    Args:
        license_name: License identifier from dataset metadata
        config: Dataset configuration with allowed licenses

    Returns:
        True if license is in the permissive list
    """
    if not license_name:
        return False
    return license_name.lower() in config.permissive_licenses


def simple_hash(content: str) -> str:
    """Compute simple hash for deduplication.

    Why: Exact deduplication prevents the model from memorizing duplicates.
    MD5 is fast enough for this use case and collision rate is negligible.

    Args:
        content: File content to hash

    Returns:
        Hex digest of MD5 hash
    """
    return hashlib.md5(content.encode()).hexdigest()


def quality_filter(content: str, config: DatasetConfig) -> tuple[bool, str]:
    """Apply quality filters to code content.

    Why: Remove low-quality code that would hurt model performance: auto-generated
    files, minified code, binary data, and trivial snippets.

    Args:
        content: Code file content
        config: Dataset configuration with filter thresholds

    Returns:
        Tuple of (passes_filter, rejection_reason)
    """
    lines = content.split("\n")
    num_lines = len(lines)

    # Line count check
    if num_lines < config.min_lines:
        return False, f"Too few lines: {num_lines} < {config.min_lines}"
    if num_lines > config.max_lines:
        return False, f"Too many lines: {num_lines} > {config.max_lines}"

    # Size check
    size_kb = len(content.encode()) / 1024
    if size_kb > config.max_file_size_kb:
        return False, f"Too large: {size_kb:.1f}KB > {config.max_file_size_kb}KB"

    # Check for auto-generated markers
    auto_generated_markers = [
        "auto-generated",
        "do not edit",
        "generated by",
        "this file is generated",
        "autogenerated",
    ]
    first_lines = "\n".join(lines[:20]).lower()
    for marker in auto_generated_markers:
        if marker in first_lines:
            return False, f"Auto-generated: found '{marker}'"

    # Check alphabetic ratio (filter out binary/encoded data)
    alpha_count = sum(1 for c in content if c.isalpha())
    total_count = len(content)
    if total_count > 0:
        alpha_ratio = alpha_count / total_count
        if alpha_ratio < 0.25:
            return False, f"Low alpha ratio: {alpha_ratio:.2f} < 0.25"

    # Check for excessive long lines (often minified/generated)
    long_lines = sum(1 for line in lines if len(line) > 500)
    if long_lines > num_lines * 0.1:
        return False, f"Too many long lines: {long_lines}/{num_lines}"

    return True, "passed"


def process_sample(
    sample: dict,
    config: DatasetConfig,
    security_scanner: "SecurityScanner | None" = None,
    quality_analyzer: "QualityAnalyzer | None" = None,
) -> dict | None:
    """Process a single sample, applying filters and quality gates.

    Why: Single responsibility function that applies all filters to one sample.
    Returns None if rejected (e.g., contains secrets), returns labeled sample
    otherwise. Negative examples are explicitly marked with explanations.

    Args:
        sample: Raw sample from dataset with keys: content, license, max_stars_count
        config: Dataset configuration with filter criteria
        security_scanner: Optional SecurityScanner for vulnerability detection
        quality_analyzer: Optional QualityAnalyzer for anti-pattern detection

    Returns:
        Processed sample dictionary or None if filtered out (secrets = always reject)
    """
    content = sample.get("content", "")
    license_name = sample.get("license")
    stars = sample.get("max_stars_count", 0) or 0

    # License filter
    if not is_permissive_license(license_name, config):
        return None

    # Stars filter
    if stars < config.min_stars:
        return None

    # Basic quality filter (size, auto-generated, etc.)
    passes, reason = quality_filter(content, config)
    if not passes:
        return None

    # Build base result
    result = {
        "text": content,
        "language": config.language,
        "license": license_name,
        "stars": stars,
        "path": sample.get("path", ""),
        "repo": sample.get("repo_name", ""),
        "quality_label": "positive",
        "quality_score": 1.0,
        "security_issues": [],
        "anti_patterns": [],
        "explanation": "",
    }

    # Security scanning (if enabled)
    if security_scanner and config.enable_security_scan:
        scan_result = security_scanner.scan(content, config.language)

        # REJECT samples with hardcoded secrets - never train on these
        if scan_result.has_secrets:
            return None

        # Label security-vulnerable code as negative examples
        if scan_result.security_issues:
            result["security_issues"] = [
                issue.issue_type for issue in scan_result.security_issues
            ]
            if scan_result.quality_label == "negative":
                result["quality_label"] = "negative"
                explanations = [
                    f"Line {issue.line_number}: {issue.issue_type} - {issue.explanation}"
                    for issue in scan_result.security_issues
                ]
                result["explanation"] = "\n".join(explanations)

                # Don't include negative examples if disabled
                if not config.include_negative_examples:
                    return None

    # Quality analysis (if enabled)
    if quality_analyzer and config.enable_quality_scan:
        quality_result = quality_analyzer.analyze(content, config.language)

        result["quality_score"] = quality_result.quality_score
        result["anti_patterns"] = [issue.issue_type for issue in quality_result.issues]

        # Label poor quality code as negative
        if quality_result.quality_score < config.min_quality_score:
            result["quality_label"] = "negative"
            if quality_result.issues:
                quality_explanations = [
                    f"{issue.location}: {issue.issue_type} - {issue.description}"
                    for issue in quality_result.issues
                ]
                if result["explanation"]:
                    result["explanation"] += "\n" + "\n".join(quality_explanations)
                else:
                    result["explanation"] = "\n".join(quality_explanations)

            # Don't include negative examples if disabled
            if not config.include_negative_examples:
                return None

    return result


def stream_from_huggingface(language: str, split: str = "train") -> Iterator[dict]:
    """Stream samples from HuggingFace Stack v2.

    Why: Streaming avoids loading 233GB of Python data into memory. Process
    samples one at a time and write filtered output incrementally.

    Requires: pip install datasets

    Args:
        language: Programming language (python, rust, javascript)
        split: Dataset split (default: train)

    Yields:
        Raw sample dictionaries from the dataset
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "bigcode/the-stack-v2-dedup",
            data_dir=f"data/{language}",
            split=split,
            streaming=True,
        )

        yield from dataset

    except ImportError:
        print("Error: 'datasets' package required. Install with: pip install datasets")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return


def save_jsonl(samples: Iterator[dict], output_path: Path, max_samples: int | None = None) -> int:
    """Save processed samples to JSONL file.

    Why: JSONL is the standard format for large text datasets. One JSON object
    per line enables streaming processing and parallel loading.

    Args:
        samples: Iterator of processed sample dictionaries
        output_path: Path to output JSONL file
        max_samples: Optional limit on number of samples to save

    Returns:
        Number of samples saved
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen_hashes = set()

    with open(output_path, "w") as f:
        for sample in samples:
            if max_samples and count >= max_samples:
                break

            # Dedup by content hash
            content_hash = simple_hash(sample.get("text", ""))
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            f.write(json.dumps(sample) + "\n")
            count += 1

            if count % 10000 == 0:
                print(f"  Processed {count:,} samples...")

    print(f"Saved {count:,} samples to {output_path}")
    return count


def main() -> None:
    """Main entry point for dataset curation pipeline.

    Why: Command-line interface for dataset curation with flexible options
    for different filtering strategies and output limits.
    """
    parser = argparse.ArgumentParser(description="Curate datasets for Tritter training")
    parser.add_argument("--list-subsets", action="store_true", help="List available subsets")
    parser.add_argument(
        "--language",
        default="python",
        choices=["python", "rust", "javascript", "triton"],
    )
    parser.add_argument("--output", type=Path, default=Path("data/curated"))
    parser.add_argument("--min-stars", type=int, default=100)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count samples without saving",
    )
    parser.add_argument(
        "--no-security-scan",
        action="store_true",
        help="Disable security vulnerability scanning",
    )
    parser.add_argument(
        "--no-quality-scan",
        action="store_true",
        help="Disable code quality analysis",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only include positive examples (no negative/educational)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.5,
        help="Minimum quality score for positive examples (0.0-1.0)",
    )
    args = parser.parse_args()

    if args.list_subsets:
        print("Available Stack v2 subsets:")
        for name, info in get_stack_v2_subsets().items():
            print(f"  {name}:")
            print(f"    Size: {info['size_gb']} GB")
            print(f"    Files: {info['files']:,}")
            print(f"    Description: {info['description']}")
            if "note" in info:
                print(f"    Note: {info['note']}")
        return

    print(f"Curating {args.language} dataset...")
    print(f"  Min stars: {args.min_stars}")
    print(f"  Output: {args.output}")
    print(f"  Security scan: {'disabled' if args.no_security_scan else 'enabled'}")
    print(f"  Quality scan: {'disabled' if args.no_quality_scan else 'enabled'}")
    print(f"  Include negatives: {'no' if args.positive_only else 'yes'}")

    config = DatasetConfig(
        language=args.language,
        min_stars=args.min_stars,
        enable_security_scan=not args.no_security_scan,
        enable_quality_scan=not args.no_quality_scan,
        include_negative_examples=not args.positive_only,
        min_quality_score=args.min_quality_score,
    )

    # Initialize quality gates if available and enabled
    security_scanner = None
    quality_analyzer = None

    if QUALITY_GATES_AVAILABLE:
        if config.enable_security_scan:
            security_scanner = SecurityScanner()
            print("  Security scanner: initialized")
        if config.enable_quality_scan:
            quality_analyzer = QualityAnalyzer()
            print("  Quality analyzer: initialized")
    else:
        if config.enable_security_scan or config.enable_quality_scan:
            print("  Warning: Quality gates not available. Run from scripts/ directory.")

    # Process stream
    def processed_stream() -> Iterator[dict]:
        for sample in stream_from_huggingface(args.language):
            processed = process_sample(sample, config, security_scanner, quality_analyzer)
            if processed:
                yield processed

    if args.dry_run:
        count = sum(1 for _ in processed_stream())
        print(f"Would save {count:,} samples (dry run)")
    else:
        output_file = args.output / f"{args.language}_curated.jsonl"
        save_jsonl(processed_stream(), output_file, args.max_samples)


if __name__ == "__main__":
    main()
