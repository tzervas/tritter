"""Device and hardware utilities for RTX 5080 optimization."""

import torch


def get_optimal_device() -> torch.device:
    """Get the optimal device for computation.

    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_memory_info() -> dict[str, float]:
    """Get current device memory usage information.

    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "allocated": 0.0, "cached": 0.0}

    device = torch.cuda.current_device()
    return {
        "total": torch.cuda.get_device_properties(device).total_memory / 1e9,
        "allocated": torch.cuda.memory_allocated(device) / 1e9,
        "cached": torch.cuda.memory_reserved(device) / 1e9,
    }


def optimize_for_rtx5080() -> None:
    """Configure PyTorch for optimal RTX 5080 performance."""
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere/Ada Lovelace/Blackwell+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmarking for optimal kernel selection
        torch.backends.cudnn.benchmark = True
