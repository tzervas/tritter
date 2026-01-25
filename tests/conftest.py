"""Pytest configuration and shared fixtures.

Why: Provides shared test infrastructure for the tritter test suite.
The CUDA availability check is critical for handling GPU architecture
mismatches (e.g., RTX 5080/Blackwell sm_120 not supported by PyTorch).
"""

import pytest
import torch


def is_cuda_usable() -> bool:
    """Check if CUDA is both available AND usable (compatible architecture).

    Why: torch.cuda.is_available() returns True even when the GPU architecture
    is not supported by the installed PyTorch (e.g., RTX 5080/Blackwell sm_120).
    This function performs a simple CUDA operation to verify actual compatibility.

    Returns:
        True if CUDA is available and a simple operation succeeds, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Try a simple CUDA operation to verify architecture compatibility
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        # CUDA error: no kernel image is available for execution on the device
        return False


# Create reusable skip marker for CUDA-dependent tests
requires_cuda = pytest.mark.skipif(
    not is_cuda_usable(),
    reason="CUDA not available or GPU architecture not compatible with PyTorch",
)
