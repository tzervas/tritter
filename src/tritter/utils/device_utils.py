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
    """Configure PyTorch for optimal RTX 5080 Blackwell architecture performance.

    Why: RTX 5080 uses Blackwell architecture which supports advanced precision formats beyond
    Ampere/Ada Lovelace. This function enables TF32 as a compatibility optimization that works
    across Ampere+, but Blackwell's native FP8 and FP6 support could provide even better
    performance when PyTorch gains full support for these formats.

    Current optimization: TF32 (TensorFloat-32) provides ~8x speedup over FP32 on matrix
    operations while maintaining numerical stability for most deep learning workloads. This
    is the current best practice for Blackwell GPUs until FP8 support matures in PyTorch.

    Future optimization: Blackwell's FP8 (8-bit floating point) could provide additional
    2-4x speedup with proper quantization-aware training. FP6 (6-bit) offers even more extreme
    compression. These formats require explicit kernel support and careful numerical stability
    tuning, which are not yet mature in PyTorch 2.x.

    Reference: NVIDIA Blackwell Architecture White Paper, PyTorch CUDA Semantics documentation.
    """
    if torch.cuda.is_available():
        # Enable TF32 for matmul operations
        # Why: TF32 provides ~8x speedup on Ampere/Ada/Blackwell GPUs with minimal accuracy loss
        # (<0.1% typical). Uses 19-bit format (8-bit exponent + 10-bit mantissa + sign) instead
        # of FP32's 23-bit mantissa, maintaining FP32 dynamic range while reducing computation.
        torch.backends.cuda.matmul.allow_tf32 = True

        # Enable TF32 for cuDNN operations (convolutions, etc.)
        # Why: Consistent TF32 usage across all operations maximizes throughput. Critical for
        # mixed workloads where convolutions (if used) benefit from same optimization as matmul.
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmarking for optimal kernel selection
        # Why: cuDNN benchmark mode profiles available kernels at first run to select fastest
        # implementation for your specific input shapes and hardware. Small startup cost (~few
        # seconds) but provides 5-20% speedup for repeated operations. Essential for training
        # but can be disabled for inference if startup latency matters.
        torch.backends.cudnn.benchmark = True

        # TODO: Add FP8 support when PyTorch/CUDA provide stable APIs
        # Blackwell supports FP8 (E4M3 and E5M2 formats) which could provide 2-4x additional
        # speedup over TF32 with quantization-aware training. Monitor PyTorch 2.x releases
        # for torch.float8_e4m3fn and torch.float8_e5m2 dtypes reaching stable status.
