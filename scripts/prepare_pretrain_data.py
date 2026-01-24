#!/usr/bin/env python3
"""Data preparation script for Tritter pretraining.

Processes source code files through the curation pipeline and outputs JSONL
shards ready for training. Supports parallel processing for large codebases.

Why: Training data quality directly impacts model quality. This script ensures:
1. All code passes through secret detection (ALWAYS reject secrets)
2. Security vulnerabilities are labeled as negative examples with explanations
3. Poor quality code is labeled for contrastive learning
4. Output format matches what train_pretrain.py expects

Usage:
    # Basic usage - process a directory of code files
    python scripts/prepare_pretrain_data.py \\
        --input-dir /path/to/code \\
        --output-dir data/pretrain \\
        --shard-size 10000

    # With quality filtering (only include high-quality samples)
    python scripts/prepare_pretrain_data.py \\
        --input-dir /path/to/code \\
        --output-dir data/pretrain \\
        --min-quality 0.7 \\
        --positive-only

    # Process specific languages
    python scripts/prepare_pretrain_data.py \\
        --input-dir /path/to/code \\
        --output-dir data/pretrain \\
        --languages python rust

    # Fast parallel processing
    python scripts/prepare_pretrain_data.py \\
        --input-dir /path/to/code \\
        --output-dir data/pretrain \\
        --workers 8

Output Format (JSONL):
    {"text": "...", "language": "python", "quality_score": 0.95, "source": "path/to/file.py"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import tqdm for progress reporting
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def progress_bar[T](iterable: Iterable[T], total: int | None = None, desc: str = "") -> Iterator[T]:
    """Wrap iterable with progress bar if tqdm is available.

    Why: Progress reporting is essential for long-running data processing tasks.
    This gracefully degrades to simple iteration if tqdm isn't installed.

    Args:
        iterable: Any iterable to wrap
        total: Total count for progress calculation
        desc: Description to show in progress bar

    Yields:
        Items from the iterable
    """
    if TQDM_AVAILABLE:
        yield from tqdm(iterable, total=total, desc=desc)
    else:
        if desc:
            print(f"{desc}...")
        for i, item in enumerate(iterable):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,} items...")
            yield item


# Language detection based on file extension
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".triton": "triton",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".zig": "zig",
}

# Languages we support in the curation pipeline
SUPPORTED_LANGUAGES = {"python", "rust", "triton", "javascript", "typescript"}


@dataclass
class ProcessingStats:
    """Statistics from data processing run.

    Why: Tracking statistics helps identify issues in the curation pipeline
    and provides visibility into data quality distribution.

    Attributes:
        total_files: Total number of files found
        processed: Successfully processed files
        rejected_secrets: Files rejected due to secrets (NEVER train on these)
        rejected_quality: Files rejected due to poor quality
        rejected_size: Files rejected due to size constraints
        positive_samples: High-quality samples
        negative_samples: Low-quality samples (for contrastive learning)
        by_language: Count by language
    """

    total_files: int = 0
    processed: int = 0
    rejected_secrets: int = 0
    rejected_quality: int = 0
    rejected_size: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    by_language: dict[str, int] = field(default_factory=dict)

    def add_language(self, language: str) -> None:
        """Increment count for a language."""
        self.by_language[language] = self.by_language.get(language, 0) + 1


@dataclass
class ProcessedSample:
    """A processed code sample ready for training.

    Why: Using a dataclass ensures type safety and makes serialization explicit.
    The format matches what train_pretrain.py expects.

    Attributes:
        text: The code content
        language: Programming language
        quality_score: Quality score from curation pipeline [0.0, 1.0]
        quality_label: "positive" or "negative"
        source: Source file path
        explanation: For negative samples, why it's marked negative
    """

    text: str
    language: str
    quality_score: float
    quality_label: str
    source: str
    explanation: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "text": self.text,
            "language": self.language,
            "quality_score": self.quality_score,
            "source": self.source,
        }
        if self.quality_label == "negative" and self.explanation:
            result["quality_label"] = self.quality_label
            result["explanation"] = self.explanation
        return result


def detect_language(file_path: Path) -> str | None:
    """Detect programming language from file extension.

    Why: Language detection is needed for language-specific analysis in the
    curation pipeline (e.g., Python-specific security patterns).

    Args:
        file_path: Path to the source file

    Returns:
        Language name or None if not recognized
    """
    suffix = file_path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)


def compute_file_hash(content: str) -> str:
    """Compute hash for deduplication.

    Why: Exact deduplication prevents the model from memorizing duplicates.
    MD5 is fast and collision rate is negligible for this use case.

    Args:
        content: File content

    Returns:
        Hex digest of MD5 hash
    """
    return hashlib.md5(content.encode()).hexdigest()


def find_source_files(
    input_dir: Path,
    languages: set[str] | None = None,
) -> list[Path]:
    """Find all source files in directory tree.

    Why: We need to recursively find all code files while respecting the
    language filter. This also filters out common non-code directories.

    Args:
        input_dir: Root directory to search
        languages: Optional set of languages to include (None = all supported)

    Returns:
        List of file paths
    """
    if languages is None:
        languages = SUPPORTED_LANGUAGES

    # Extensions to look for
    extensions = {ext for ext, lang in EXTENSION_TO_LANGUAGE.items() if lang in languages}

    # Directories to skip
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".tox",
        ".venv",
        "venv",
        "env",
        ".env",
        "target",  # Rust build directory
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }

    files: list[Path] = []

    for path in input_dir.rglob("*"):
        # Skip directories we don't want
        if any(skip in path.parts for skip in skip_dirs):
            continue

        # Only include files with recognized extensions
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)

    return sorted(files)


def process_file(
    file_path: Path,
    input_dir: Path,
    min_quality: float,
    include_negative: bool,
) -> tuple[ProcessedSample | None, str]:
    """Process a single file through the curation pipeline.

    Why: This function is designed to be called in parallel via ProcessPoolExecutor.
    It returns both the result and a status string for logging.

    Args:
        file_path: Path to source file
        input_dir: Root input directory (for relative path calculation)
        min_quality: Minimum quality score to accept (for positive samples)
        include_negative: Whether to include negative samples

    Returns:
        Tuple of (ProcessedSample or None, status string)
    """
    # Import here to avoid issues with multiprocessing
    try:
        from tritter.curation import CurationPipeline
    except ImportError:
        return None, "import_error"

    # Detect language
    language = detect_language(file_path)
    if language is None:
        return None, "unsupported_language"

    # Read file
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="latin-1")
        except Exception:
            return None, "read_error"
    except Exception:
        return None, "read_error"

    # Calculate relative source path
    try:
        relative_path = str(file_path.relative_to(input_dir))
    except ValueError:
        relative_path = str(file_path)

    # Process through curation pipeline
    pipeline = CurationPipeline(
        quality_threshold=min_quality,
        min_lines=5,
        max_lines=10000,
    )

    result = pipeline.process(content, language)

    # Handle rejected samples (secrets)
    if result.quality_label == "rejected":
        return None, f"rejected_{result.rejected_reason or 'unknown'}"

    # Handle negative samples
    if result.quality_label == "negative":
        if not include_negative:
            return None, "filtered_negative"

        return (
            ProcessedSample(
                text=content,
                language=language,
                quality_score=result.quality_score,
                quality_label="negative",
                source=relative_path,
                explanation=result.explanation,
            ),
            "negative",
        )

    # Positive sample
    return (
        ProcessedSample(
            text=content,
            language=language,
            quality_score=result.quality_score,
            quality_label="positive",
            source=relative_path,
        ),
        "positive",
    )


def write_shard(
    samples: list[ProcessedSample],
    output_path: Path,
) -> int:
    """Write samples to a JSONL shard file.

    Why: JSONL format enables streaming reads during training and is the
    standard format for large text datasets.

    Args:
        samples: List of processed samples
        output_path: Path to output JSONL file

    Returns:
        Number of samples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    return len(samples)


def main() -> None:
    """Main entry point for data preparation.

    Why: Command-line interface provides flexibility for different processing
    scenarios (full curation, quality filtering, parallel processing).
    """
    parser = argparse.ArgumentParser(
        description="Prepare pretraining data from source code files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process Python code from a directory
    python scripts/prepare_pretrain_data.py \\
        --input-dir ~/code/my-project \\
        --output-dir data/pretrain

    # Only include high-quality positive samples
    python scripts/prepare_pretrain_data.py \\
        --input-dir ~/code/my-project \\
        --output-dir data/pretrain \\
        --min-quality 0.7 \\
        --positive-only

    # Fast parallel processing
    python scripts/prepare_pretrain_data.py \\
        --input-dir ~/code/my-project \\
        --output-dir data/pretrain \\
        --workers 8
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source code files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write JSONL shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of samples per shard file (default: 10000)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score for positive samples (default: 0.5)",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only include positive samples (exclude negative examples)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to include (default: all supported)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files without processing",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Parse languages
    languages = None
    if args.languages:
        languages = set(args.languages)
        unsupported = languages - SUPPORTED_LANGUAGES
        if unsupported:
            print(f"Warning: Unsupported languages will be skipped: {unsupported}")
            languages = languages & SUPPORTED_LANGUAGES

    print("=" * 60)
    print("Tritter Data Preparation")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shard size:       {args.shard_size:,}")
    print(f"Min quality:      {args.min_quality}")
    print(f"Positive only:    {args.positive_only}")
    print(f"Languages:        {languages or 'all supported'}")
    print(f"Workers:          {args.workers}")
    print("=" * 60)
    print()

    # Find source files
    print("Scanning for source files...")
    files = find_source_files(args.input_dir, languages)
    print(f"Found {len(files):,} source files")

    if not files:
        print("No source files found. Check --languages filter.")
        sys.exit(0)

    if args.dry_run:
        print("\nDry run complete. No files processed.")
        # Show language breakdown
        lang_counts: dict[str, int] = {}
        for f in files:
            lang = detect_language(f)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        print("\nFiles by language:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count:,}")
        sys.exit(0)

    # Process files
    stats = ProcessingStats(total_files=len(files))
    seen_hashes: set[str] = set()
    current_shard: list[ProcessedSample] = []
    shard_index = 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nProcessing files...")

    include_negative = not args.positive_only

    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_file,
                    f,
                    args.input_dir,
                    args.min_quality,
                    include_negative,
                ): f
                for f in files
            }

            for future in progress_bar(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                sample, status = future.result()

                # Update stats
                if status == "positive":
                    stats.positive_samples += 1
                    stats.processed += 1
                elif status == "negative":
                    stats.negative_samples += 1
                    stats.processed += 1
                elif status.startswith("rejected_"):
                    if "secrets" in status:
                        stats.rejected_secrets += 1
                    elif "short" in status or "long" in status:
                        stats.rejected_size += 1
                    else:
                        stats.rejected_quality += 1

                if sample is None:
                    continue

                # Dedup check
                content_hash = compute_file_hash(sample.text)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # Track language
                stats.add_language(sample.language)

                # Add to current shard
                current_shard.append(sample)

                # Write shard if full
                if len(current_shard) >= args.shard_size:
                    shard_path = args.output_dir / f"shard_{shard_index:05d}.jsonl"
                    write_shard(current_shard, shard_path)
                    shard_index += 1
                    current_shard = []
    else:
        # Sequential processing
        for file_path in progress_bar(files, total=len(files), desc="Processing"):
            sample, status = process_file(
                file_path,
                args.input_dir,
                args.min_quality,
                include_negative,
            )

            # Update stats
            if status == "positive":
                stats.positive_samples += 1
                stats.processed += 1
            elif status == "negative":
                stats.negative_samples += 1
                stats.processed += 1
            elif status.startswith("rejected_"):
                if "secrets" in status:
                    stats.rejected_secrets += 1
                elif "short" in status or "long" in status:
                    stats.rejected_size += 1
                else:
                    stats.rejected_quality += 1

            if sample is None:
                continue

            # Dedup check
            content_hash = compute_file_hash(sample.text)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Track language
            stats.add_language(sample.language)

            # Add to current shard
            current_shard.append(sample)

            # Write shard if full
            if len(current_shard) >= args.shard_size:
                shard_path = args.output_dir / f"shard_{shard_index:05d}.jsonl"
                write_shard(current_shard, shard_path)
                shard_index += 1
                current_shard = []

    # Write final shard if not empty
    if current_shard:
        shard_path = args.output_dir / f"shard_{shard_index:05d}.jsonl"
        write_shard(current_shard, shard_path)
        shard_index += 1

    # Print statistics
    print()
    print("=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total files scanned:    {stats.total_files:,}")
    print(f"Successfully processed: {stats.processed:,}")
    print(f"  Positive samples:     {stats.positive_samples:,}")
    print(f"  Negative samples:     {stats.negative_samples:,}")
    print("Rejected files:")
    print(f"  Secrets (NEVER train):{stats.rejected_secrets:,}")
    print(f"  Quality issues:       {stats.rejected_quality:,}")
    print(f"  Size constraints:     {stats.rejected_size:,}")
    print()
    print(f"Output shards:          {shard_index}")
    print(f"Output directory:       {args.output_dir}")
    print()
    print("Samples by language:")
    for lang, count in sorted(stats.by_language.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
