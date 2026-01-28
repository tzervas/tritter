"""Pytest configuration and fixtures.

Why:
    Ensures devtools package is importable during test collection and
    exposes shared CUDA markers used across the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure src and project root are in path for imports - must happen at import time
project_root = Path(__file__).parent
src_root = project_root / "src"
for path in (str(src_root), str(project_root)):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    """Early hook to ensure devtools is importable before test collection."""
    # Force devtools into sys.modules early
    import devtools  # noqa: F401


def _is_cuda_usable() -> bool:
    """Check if CUDA is both available AND usable (compatible architecture)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(
    not _is_cuda_usable(),
    reason="CUDA not available or GPU architecture not compatible with PyTorch",
)
