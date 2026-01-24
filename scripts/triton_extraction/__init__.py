"""Triton kernel dataset extraction tools.

Provides utilities for extracting Triton GPU kernels from permissively-licensed
ML framework repositories for training data curation.

Why: Triton is a low-resource language (~1M tokens available). Custom curation
from PyTorch Inductor, FlashAttention, xFormers, and JAX Pallas provides
high-quality training data for GPU kernel generation.

Usage:
    from scripts.triton_extraction import TritonExtractor

    extractor = TritonExtractor("./triton_data")
    extractor.clone_repositories()
    results = extractor.extract_all()
    extractor.create_manifest()

CLI Usage:
    python -m scripts.triton_extraction.extract_kernels --workspace ./triton_data

Reference: docs/TRITON_EXTRACTION_PLAN.md
"""

from scripts.triton_extraction.extract_kernels import (
    ExtractionResult,
    RepositoryConfig,
    TritonExtractor,
    REPOSITORIES,
)

__all__ = [
    "ExtractionResult",
    "RepositoryConfig",
    "TritonExtractor",
    "REPOSITORIES",
]
