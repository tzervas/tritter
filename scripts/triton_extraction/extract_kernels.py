#!/usr/bin/env python3
"""Triton kernel dataset extraction and validation.

Extracts Triton GPU kernels from permissively-licensed repositories for training data.
Supports PyTorch Inductor, FlashAttention, xFormers, JAX Pallas, and Triton tutorials.

Usage:
    python extract_kernels.py --workspace ./triton_data
    python extract_kernels.py --workspace ./triton_data --skip-clone
    python extract_kernels.py --workspace ./triton_data --validate-only

Why: Triton is a low-resource language (~1M tokens available vs Python's billions).
Custom curation from ML framework codebases provides high-quality, permissively-licensed
training data for GPU kernel generation capabilities.

Reference: docs/TRITON_EXTRACTION_PLAN.md, docs/specs/SPEC-007-dataset-quality-gates.md
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RepositoryConfig:
    """Configuration for a source repository.

    Why: Encapsulates clone URL, sparse checkout paths, and extraction patterns
    for each repository in a structured way.
    """

    name: str
    url: str
    sparse_paths: list[str] = field(default_factory=list)
    kernel_patterns: list[str] = field(default_factory=lambda: ["*triton*.py"])
    license: str = "Unknown"


# Repository configurations
REPOSITORIES = [
    RepositoryConfig(
        name="triton",
        url="https://github.com/triton-lang/triton.git",
        sparse_paths=[],  # Full clone (small repo)
        kernel_patterns=["*.py"],  # All Python in tutorials
        license="MIT",
    ),
    RepositoryConfig(
        name="pytorch",
        url="https://github.com/pytorch/pytorch.git",
        sparse_paths=["torch/_inductor"],
        kernel_patterns=["*triton*.py"],
        license="BSD-3-Clause",
    ),
    RepositoryConfig(
        name="flash-attention",
        url="https://github.com/Dao-AILab/flash-attention.git",
        sparse_paths=[],  # Full clone (small repo)
        kernel_patterns=["*triton*.py"],
        license="BSD-3-Clause",
    ),
    RepositoryConfig(
        name="xformers",
        url="https://github.com/facebookresearch/xformers.git",
        sparse_paths=["xformers/ops"],
        kernel_patterns=["*_triton*.py", "*triton*.py"],
        license="BSD-3-Clause",
    ),
    RepositoryConfig(
        name="jax",
        url="https://github.com/jax-ml/jax.git",
        sparse_paths=["jax/experimental/pallas"],
        kernel_patterns=["*triton*.py"],
        license="Apache-2.0",
    ),
]

PERMISSIVE_LICENSES = ["MIT", "BSD", "BSD-3", "Apache-2.0", "Apache 2.0", "ISC"]


@dataclass
class ExtractionResult:
    """Result of kernel extraction for a repository.

    Why: Structured return type enables programmatic processing of results.
    """

    repo_name: str
    files_found: int = 0
    files_extracted: int = 0
    total_lines: int = 0
    triton_jit_count: int = 0
    errors: list[str] = field(default_factory=list)


class TritonExtractor:
    """Extracts Triton kernels from ML framework repositories.

    Why: Centralized extraction logic with validation ensures consistent,
    high-quality training data from permissively-licensed sources.

    Attributes:
        workspace: Base directory for cloned repos and extracted kernels
        output_dir: Directory for organized kernel output
        verbose: Enable verbose logging

    Example:
        >>> extractor = TritonExtractor("./triton_data")
        >>> extractor.clone_repositories()
        >>> results = extractor.extract_all()
        >>> extractor.create_manifest()
    """

    def __init__(self, workspace: str | Path, verbose: bool = True) -> None:
        """Initialize extractor.

        Args:
            workspace: Base directory for extraction
            verbose: Enable verbose output
        """
        self.workspace = Path(workspace).resolve()
        self.output_dir = self.workspace / "triton_kernels_organized"
        self.logs_dir = self.workspace / "extraction_logs"
        self.verbose = verbose

        # Create directories
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Log message if verbose enabled."""
        if self.verbose:
            print(message)

    def clone_repository(self, config: RepositoryConfig) -> bool:
        """Clone a single repository with sparse checkout if configured.

        Args:
            config: Repository configuration

        Returns:
            True if successful, False otherwise
        """
        repo_path = self.workspace / config.name

        if repo_path.exists():
            self.log(f"  {config.name}: Already exists, skipping clone")
            return True

        self.log(f"  Cloning {config.name}...")

        try:
            if config.sparse_paths:
                # Sparse checkout for large repos
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--sparse",
                        config.url,
                        str(repo_path),
                    ],
                    check=True,
                    capture_output=True,
                )

                # Set sparse checkout paths
                subprocess.run(
                    ["git", "sparse-checkout", "set", "--cone"] + config.sparse_paths,
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )
            else:
                # Full shallow clone for small repos
                subprocess.run(
                    ["git", "clone", "--depth", "1", config.url, str(repo_path)],
                    check=True,
                    capture_output=True,
                )

            self.log(f"  ✓ {config.name} cloned successfully")
            return True

        except subprocess.CalledProcessError as e:
            self.log(f"  ✗ {config.name} clone failed: {e}")
            return False

    def clone_repositories(self) -> dict[str, bool]:
        """Clone all configured repositories.

        Returns:
            Dict mapping repo name to success status
        """
        self.log("\n=== Cloning Repositories ===")
        results = {}

        for config in REPOSITORIES:
            results[config.name] = self.clone_repository(config)

        return results

    def find_kernel_files(self, config: RepositoryConfig) -> list[Path]:
        """Find Triton kernel files in a repository.

        Args:
            config: Repository configuration

        Returns:
            List of paths to kernel files
        """
        repo_path = self.workspace / config.name

        if not repo_path.exists():
            return []

        kernel_files = []

        # Search paths based on repository
        search_paths = []
        if config.name == "triton":
            search_paths = [repo_path / "python" / "tutorials"]
        elif config.sparse_paths:
            search_paths = [repo_path / sp for sp in config.sparse_paths]
        else:
            search_paths = [repo_path]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for pattern in config.kernel_patterns:
                kernel_files.extend(search_path.rglob(pattern))

        # Filter to only files containing @triton.jit or triton imports
        triton_files = []
        for f in kernel_files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if "@triton.jit" in content or "import triton" in content:
                    triton_files.append(f)
            except Exception:
                pass

        return list(set(triton_files))

    def extract_repository(self, config: RepositoryConfig) -> ExtractionResult:
        """Extract kernels from a single repository.

        Args:
            config: Repository configuration

        Returns:
            ExtractionResult with statistics
        """
        result = ExtractionResult(repo_name=config.name)

        kernel_files = self.find_kernel_files(config)
        result.files_found = len(kernel_files)

        if not kernel_files:
            return result

        # Create output directory for this repo
        repo_output = self.output_dir / config.name
        repo_output.mkdir(parents=True, exist_ok=True)

        for src_file in kernel_files:
            try:
                content = src_file.read_text(encoding="utf-8", errors="ignore")

                # Count statistics
                lines = len(content.splitlines())
                has_jit = "@triton.jit" in content

                # Determine destination path
                rel_path = src_file.relative_to(self.workspace / config.name)
                dest_path = repo_output / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(src_file, dest_path)

                result.files_extracted += 1
                result.total_lines += lines
                if has_jit:
                    result.triton_jit_count += 1

            except Exception as e:
                result.errors.append(f"{src_file}: {e}")

        return result

    def extract_all(self) -> list[ExtractionResult]:
        """Extract kernels from all repositories.

        Returns:
            List of ExtractionResult for each repository
        """
        self.log("\n=== Extracting Triton Kernels ===")
        results = []

        for config in REPOSITORIES:
            self.log(f"\nProcessing {config.name}...")
            result = self.extract_repository(config)
            results.append(result)

            self.log(f"  Found: {result.files_found} files")
            self.log(f"  Extracted: {result.files_extracted} files")
            self.log(f"  Lines: {result.total_lines:,}")
            self.log(f"  @triton.jit: {result.triton_jit_count}")

            if result.errors:
                self.log(f"  Errors: {len(result.errors)}")

        return results

    def validate_kernel(self, filepath: Path) -> dict[str, bool]:
        """Validate a single kernel file.

        Args:
            filepath: Path to kernel file

        Returns:
            Dict of validation check results
        """
        content = filepath.read_text(encoding="utf-8", errors="ignore")

        return {
            "has_triton_jit": "@triton.jit" in content,
            "has_bounds_masking": any(p in content for p in ["tl.where", "mask=", "masked_load"]),
            "has_program_id": "tl.program_id" in content,
            "has_block_size": any(p in content for p in ["BLOCK_SIZE", "block_", "BLOCK_"]),
            "has_load_store": "tl.load" in content or "tl.store" in content,
            "has_docstring": '"""' in content or "'''" in content,
            "has_license_header": any(
                lic.upper() in content[:2000].upper() for lic in PERMISSIVE_LICENSES
            ),
        }

    def validate_all(self) -> dict[str, dict[str, int]]:
        """Validate all extracted kernels.

        Returns:
            Nested dict of category -> check -> count
        """
        self.log("\n=== Validating Kernels ===")
        results = defaultdict(lambda: defaultdict(int))

        for kernel_file in self.output_dir.rglob("*.py"):
            if kernel_file.name == "__init__.py":
                continue

            category = kernel_file.relative_to(self.output_dir).parts[0]
            checks = self.validate_kernel(kernel_file)

            results[category]["total_files"] += 1
            for check_name, passed in checks.items():
                if passed:
                    results[category][check_name] += 1

        # Log results
        for category in sorted(results.keys()):
            data = results[category]
            total = data["total_files"]
            self.log(f"\n{category.upper()} ({total} files):")

            checks = [
                ("@triton.jit decorator", "has_triton_jit"),
                ("Bounds masking", "has_bounds_masking"),
                ("Program ID usage", "has_program_id"),
                ("Block size config", "has_block_size"),
                ("Load/Store ops", "has_load_store"),
                ("Documentation", "has_docstring"),
            ]

            for check_name, check_key in checks:
                count = data[check_key]
                pct = (100 * count // total) if total > 0 else 0
                status = "✓" if pct > 80 else "~" if pct > 50 else "✗"
                self.log(f"  {status} {check_name}: {count}/{total} ({pct}%)")

        return dict(results)

    def compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(block)
        return sha256_hash.hexdigest()

    def create_manifest(self) -> dict:
        """Create manifest of extracted kernels.

        Returns:
            Manifest dict (also written to MANIFEST.json)
        """
        self.log("\n=== Creating Manifest ===")

        manifest = {
            "created": datetime.now().isoformat(),
            "extraction_dir": str(self.output_dir),
            "repositories": {r.name: {"url": r.url, "license": r.license} for r in REPOSITORIES},
            "categories": {},
            "total_files": 0,
            "total_size_bytes": 0,
            "total_lines": 0,
        }

        for kernel_file in self.output_dir.rglob("*.py"):
            if kernel_file.name == "MANIFEST.json":
                continue

            category = kernel_file.relative_to(self.output_dir).parts[0]

            if category not in manifest["categories"]:
                manifest["categories"][category] = {
                    "files": [],
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_lines": 0,
                }

            content = kernel_file.read_text(encoding="utf-8", errors="ignore")
            stats = {
                "path": str(kernel_file.relative_to(self.output_dir)),
                "size_bytes": kernel_file.stat().st_size,
                "lines": len(content.splitlines()),
                "has_triton_jit": "@triton.jit" in content,
                "hash_sha256": self.compute_file_hash(kernel_file),
            }

            manifest["categories"][category]["files"].append(stats)
            manifest["categories"][category]["total_files"] += 1
            manifest["categories"][category]["total_size_bytes"] += stats["size_bytes"]
            manifest["categories"][category]["total_lines"] += stats["lines"]

            manifest["total_files"] += 1
            manifest["total_size_bytes"] += stats["size_bytes"]
            manifest["total_lines"] += stats["lines"]

        # Write manifest
        manifest_path = self.output_dir / "MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.log(f"\nManifest: {manifest_path}")
        self.log(f"Total files: {manifest['total_files']}")
        self.log(f"Total size: {manifest['total_size_bytes'] / 1024:.1f} KB")
        self.log(f"Total lines: {manifest['total_lines']:,}")

        return manifest

    def run(self, skip_clone: bool = False, validate_only: bool = False) -> dict:
        """Run complete extraction pipeline.

        Args:
            skip_clone: Skip repository cloning
            validate_only: Only run validation on existing extraction

        Returns:
            Dict with extraction statistics
        """
        self.log("=" * 60)
        self.log("Triton Kernel Dataset Extraction")
        self.log(f"Workspace: {self.workspace}")
        self.log("=" * 60)

        if validate_only:
            validation = self.validate_all()
            return {"validation": validation}

        # Clone repositories
        if not skip_clone:
            clone_results = self.clone_repositories()
        else:
            clone_results = {r.name: True for r in REPOSITORIES}

        # Extract kernels
        extraction_results = self.extract_all()

        # Validate
        validation = self.validate_all()

        # Create manifest
        manifest = self.create_manifest()

        # Summary
        self.log("\n" + "=" * 60)
        self.log("EXTRACTION COMPLETE")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Total files: {manifest['total_files']}")
        self.log(f"Total lines: {manifest['total_lines']:,}")

        return {
            "clone_results": clone_results,
            "extraction_results": [
                {
                    "repo": r.repo_name,
                    "files": r.files_extracted,
                    "lines": r.total_lines,
                }
                for r in extraction_results
            ],
            "validation": dict(validation),
            "manifest": manifest,
        }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Triton kernels from ML framework repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_kernels.py --workspace ./triton_data
  python extract_kernels.py --workspace ./triton_data --skip-clone
  python extract_kernels.py --workspace ./triton_data --validate-only
        """,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="./triton_dataset_workspace",
        help="Base directory for extraction (default: ./triton_dataset_workspace)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip repository cloning (use existing clones)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing extracted kernels",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Write results to JSON file",
    )

    args = parser.parse_args()

    extractor = TritonExtractor(args.workspace, verbose=not args.quiet)
    results = extractor.run(
        skip_clone=args.skip_clone,
        validate_only=args.validate_only,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
