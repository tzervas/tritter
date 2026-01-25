"""Tests for Triton kernel extraction tools.

Tests validation logic without requiring network access or repository clones.

Why: The extraction logic should be tested for correctness, but actual
repository cloning is expensive and network-dependent. These tests use
mock files to validate the extraction and validation logic.
"""

# Import from scripts directory
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.triton_extraction.extract_kernels import (
    REPOSITORIES,
    ExtractionResult,
    TritonExtractor,
)


class TestRepositoryConfig:
    """Tests for RepositoryConfig."""

    def test_repository_configs_exist(self) -> None:
        """Verify all expected repositories are configured.

        Why: Ensure we're extracting from the documented sources.
        """
        repo_names = {r.name for r in REPOSITORIES}
        expected = {"triton", "pytorch", "flash-attention", "xformers", "jax"}

        assert repo_names == expected

    def test_repository_has_license(self) -> None:
        """Verify all repositories have license specified.

        Why: Only permissively-licensed code should be used for training.
        """
        for repo in REPOSITORIES:
            assert repo.license != "Unknown", f"{repo.name} missing license"


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self) -> None:
        """Verify default values are sensible."""
        result = ExtractionResult(repo_name="test")

        assert result.repo_name == "test"
        assert result.files_found == 0
        assert result.files_extracted == 0
        assert result.total_lines == 0
        assert result.errors == []


class TestTritonExtractor:
    """Tests for TritonExtractor class."""

    def test_extractor_creates_directories(self) -> None:
        """Verify extractor creates required directories.

        Why: Extraction needs workspace structure before processing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)

            assert extractor.workspace.exists()
            assert extractor.output_dir.exists()
            assert extractor.logs_dir.exists()

    def test_validate_kernel_with_triton_jit(self) -> None:
        """Verify validation detects @triton.jit decorator.

        Why: Core pattern for identifying Triton kernels.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)

            # Create test kernel file
            kernel_content = '''
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Add two vectors."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
'''
            test_file = extractor.output_dir / "test_kernel.py"
            test_file.write_text(kernel_content)

            checks = extractor.validate_kernel(test_file)

            assert checks["has_triton_jit"]
            assert checks["has_bounds_masking"]  # mask=
            assert checks["has_program_id"]
            assert checks["has_block_size"]
            assert checks["has_load_store"]
            assert checks["has_docstring"]

    def test_validate_kernel_without_triton(self) -> None:
        """Verify validation correctly identifies non-Triton files.

        Why: Need to filter out regular Python files from kernel count.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)

            # Create regular Python file
            regular_content = '''
def hello():
    """A regular function."""
    print("Hello, World!")
'''
            test_file = extractor.output_dir / "regular.py"
            test_file.write_text(regular_content)

            checks = extractor.validate_kernel(test_file)

            assert not checks["has_triton_jit"]
            assert not checks["has_program_id"]
            assert not checks["has_load_store"]

    def test_compute_file_hash(self) -> None:
        """Verify file hash computation is deterministic.

        Why: Hashes are used for deduplication and tracking.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)

            test_file = extractor.output_dir / "test.py"
            test_file.write_text("content")

            hash1 = extractor.compute_file_hash(test_file)
            hash2 = extractor.compute_file_hash(test_file)

            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex digest

    def test_create_manifest_empty(self) -> None:
        """Verify manifest creation with empty output.

        Why: Should handle edge case of no extracted files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)
            manifest = extractor.create_manifest()

            assert manifest["total_files"] == 0
            assert manifest["total_lines"] == 0
            assert "created" in manifest

    def test_create_manifest_with_files(self) -> None:
        """Verify manifest includes file statistics.

        Why: Manifest tracks all extracted kernels for reproducibility.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)

            # Create test category with kernel
            category_dir = extractor.output_dir / "test_repo"
            category_dir.mkdir()

            kernel_content = "@triton.jit\ndef kernel(): pass"
            (category_dir / "kernel.py").write_text(kernel_content)

            manifest = extractor.create_manifest()

            assert manifest["total_files"] == 1
            assert "test_repo" in manifest["categories"]
            assert manifest["categories"]["test_repo"]["total_files"] == 1


class TestValidationChecks:
    """Tests for specific validation patterns."""

    @pytest.mark.parametrize(
        "content,check,expected",
        [
            ("@triton.jit\ndef k(): pass", "has_triton_jit", True),
            ("def k(): pass", "has_triton_jit", False),
            ("mask = offsets < n", "has_bounds_masking", False),
            ("x = tl.where(mask, x, 0)", "has_bounds_masking", True),
            ("pid = tl.program_id(0)", "has_program_id", True),
            ("pid = 0", "has_program_id", False),
            ("BLOCK_SIZE = 128", "has_block_size", True),
            ("x = tl.load(ptr)", "has_load_store", True),
            ("tl.store(ptr, x)", "has_load_store", True),
            ('"""Docstring."""', "has_docstring", True),
        ],
    )
    def test_validation_patterns(
        self, content: str, check: str, expected: bool
    ) -> None:
        """Verify individual validation patterns.

        Why: Each validation check should correctly identify its pattern.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TritonExtractor(tmpdir, verbose=False)

            test_file = extractor.output_dir / "test.py"
            test_file.write_text(content)

            checks = extractor.validate_kernel(test_file)

            assert checks[check] == expected, f"Check {check} failed for: {content}"
