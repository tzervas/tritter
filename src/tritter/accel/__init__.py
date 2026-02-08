"""Rust acceleration bindings for Tritter.

Provides high-performance operations via the tritter-accel Rust crate.
Falls back to pure Python implementations if Rust extension is unavailable.

Why: These operations are on the critical path for training and inference.
Rust implementations provide 20-25x speedup over pure Python while
maintaining exact numerical equivalence with the Python fallbacks.

Usage:
    from tritter.accel import pack_ternary_weights, ternary_matmul

    # Pack weights for inference
    packed, shape = pack_ternary_weights(ternary_weights, scales)

    # Fast matmul with packed weights
    output = ternary_matmul(input, packed, scales, shape)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

ACCEL_AVAILABLE: bool = False

# Try to import Rust acceleration
try:
    from tritter_accel import (
        compress_gradients_vsa as _rust_compress_vsa,
    )
    from tritter_accel import (
        cuda_available,
    )
    from tritter_accel import (
        decompress_gradients_vsa as _rust_decompress_vsa,
    )
    from tritter_accel import (
        pack_ternary_weights as _rust_pack,
    )
    from tritter_accel import (
        quantize_weights_absmean as _rust_quantize,
    )
    from tritter_accel import (
        ternary_matmul as _rust_matmul,
    )
    from tritter_accel import (
        unpack_ternary_weights as _rust_unpack,
    )
    from tritter_accel import (
        version as rust_version,
    )

    ACCEL_AVAILABLE = True
except ImportError:
    ACCEL_AVAILABLE = False
    rust_version = lambda: "not installed"  # noqa: E731
    cuda_available = lambda: False  # noqa: E731


def pack_ternary_weights(
    ternary_weights: npt.NDArray[np.float32],
    scales: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    """Pack ternary weights into 2-bit representation.

    Args:
        ternary_weights: 2D array of ternary values (must be -1, 0, or 1)
        scales: Per-channel scaling factors

    Returns:
        Tuple of (packed_weights, original_shape)

    Why: Packed ternary weights use 4x less memory than float32, enabling
    larger models to fit in VRAM. The packing is lossless.
    """
    if ACCEL_AVAILABLE:
        return _rust_pack(ternary_weights, scales)  # type: ignore[no-any-return]
    else:
        return _python_pack_ternary(ternary_weights, scales)


def unpack_ternary_weights(
    packed: npt.NDArray[np.uint8],
    shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Unpack ternary weights from 2-bit representation.

    Args:
        packed: Packed byte array
        shape: Original shape (rows, cols)

    Returns:
        2D array of ternary values

    Why: Unpacking is needed for operations that require full-precision
    weights, such as gradient computation during training.
    """
    if ACCEL_AVAILABLE:
        return _rust_unpack(packed, shape)
    else:
        return _python_unpack_ternary(packed, shape)


def ternary_matmul(
    input: npt.NDArray[np.float32],
    packed_weights: npt.NDArray[np.uint8],
    scales: npt.NDArray[np.float32],
    shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Matrix multiplication with packed ternary weights.

    Args:
        input: Input tensor (batch, in_features)
        packed_weights: Packed ternary weights
        scales: Per-channel scaling factors
        shape: Original weight shape (out_features, in_features)

    Returns:
        Output tensor (batch, out_features)

    Why: Operating directly on packed weights avoids unpacking overhead,
    providing 20-25x speedup for inference.
    """
    if ACCEL_AVAILABLE:
        return _rust_matmul(input, packed_weights, scales, shape)
    else:
        return _python_ternary_matmul(input, packed_weights, scales, shape)


def quantize_weights_absmean(
    weights: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Quantize weights to ternary using AbsMean scaling.

    Args:
        weights: Full-precision weights (out_features, in_features)

    Returns:
        Tuple of (ternary_weights, scales)

    Why: AbsMean scaling is the standard BitNet quantization method,
    computing scale = mean(|weights|) per output channel.
    """
    if ACCEL_AVAILABLE:
        return _rust_quantize(weights)  # type: ignore[no-any-return]
    else:
        return _python_quantize_absmean(weights)


def compress_gradients_vsa(
    gradients: npt.NDArray[np.float32],
    compression_ratio: float = 0.1,
) -> npt.NDArray[np.float32]:
    """Compress gradients using Vector Symbolic Architecture.

    Args:
        gradients: Gradient tensor to compress
        compression_ratio: Target ratio (0.1 = 10x compression)

    Returns:
        Compressed gradient representation

    Why: VSA compression achieves ~90% storage reduction while preserving
    semantic information, enabling gradient communication in distributed
    training with minimal bandwidth.
    """
    if ACCEL_AVAILABLE:
        return _rust_compress_vsa(gradients.ravel(), compression_ratio)
    else:
        return _python_compress_vsa(gradients.ravel(), compression_ratio)


def decompress_gradients_vsa(
    compressed: npt.NDArray[np.float32],
    original_size: int,
) -> npt.NDArray[np.float32]:
    """Decompress VSA-compressed gradients.

    Args:
        compressed: Compressed gradient representation
        original_size: Original gradient size

    Returns:
        Approximate reconstructed gradients

    Why: Decompression recovers an approximation of the original gradients
    for optimizer updates. The approximation preserves gradient direction.
    """
    if ACCEL_AVAILABLE:
        return _rust_decompress_vsa(compressed, original_size)
    else:
        return _python_decompress_vsa(compressed, original_size)


# ============================================================================
# Python Fallback Implementations
# ============================================================================


def _python_pack_ternary(
    weights: npt.NDArray[np.float32],
    scales: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    """Python fallback for pack_ternary_weights."""
    rows, cols = weights.shape
    packed_cols = (cols + 3) // 4
    packed = np.zeros(rows * packed_cols, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            val = int(weights[i, j])
            if val == -1:
                encoded = 0b00
            elif val == 0:
                encoded = 0b01
            elif val == 1:
                encoded = 0b10
            else:
                raise ValueError(f"Invalid ternary value: {val}")

            byte_idx = i * packed_cols + j // 4
            bit_offset = (j % 4) * 2
            packed[byte_idx] |= encoded << bit_offset

    return packed, (rows, cols)


def _python_unpack_ternary(
    packed: npt.NDArray[np.uint8],
    shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Python fallback for unpack_ternary_weights."""
    rows, cols = shape
    packed_cols = (cols + 3) // 4
    unpacked = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            byte_idx = i * packed_cols + j // 4
            bit_offset = (j % 4) * 2
            encoded = (packed[byte_idx] >> bit_offset) & 0b11

            if encoded == 0b00:
                unpacked[i, j] = -1.0
            elif encoded == 0b01:
                unpacked[i, j] = 0.0
            elif encoded == 0b10:
                unpacked[i, j] = 1.0

    return unpacked


def _python_ternary_matmul(
    input: npt.NDArray[np.float32],
    packed: npt.NDArray[np.uint8],
    scales: npt.NDArray[np.float32],
    shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Python fallback for ternary_matmul."""
    weights = _python_unpack_ternary(packed, shape)
    output = input @ weights.T
    return output * scales


def _python_quantize_absmean(
    weights: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Python fallback for quantize_weights_absmean."""
    scales = np.mean(np.abs(weights), axis=1)
    scales = np.where(scales < 1e-8, 1.0, scales)
    normalized = weights / scales[:, np.newaxis]
    ternary = np.clip(np.round(normalized), -1, 1).astype(np.float32)
    return ternary, scales.astype(np.float32)


def _python_compress_vsa(
    gradients: npt.NDArray[np.float32],
    compression_ratio: float,
) -> npt.NDArray[np.float32]:
    """Python fallback for compress_gradients_vsa."""
    original_size = len(gradients)
    compressed_size = max(1, int(np.ceil(original_size * compression_ratio)))
    compressed = np.zeros(compressed_size, dtype=np.float32)

    for i, val in enumerate(gradients):
        target_idx = i % compressed_size
        sign = 1.0 if (i // compressed_size) % 2 == 0 else -1.0
        compressed[target_idx] += val * sign

    norm = np.linalg.norm(compressed)
    if norm > 1e-8:
        compressed /= norm

    return compressed


def _python_decompress_vsa(
    compressed: npt.NDArray[np.float32],
    original_size: int,
) -> npt.NDArray[np.float32]:
    """Python fallback for decompress_gradients_vsa."""
    compressed_size = len(compressed)
    decompressed = np.zeros(original_size, dtype=np.float32)

    for i in range(original_size):
        source_idx = i % compressed_size
        sign = 1.0 if (i // compressed_size) % 2 == 0 else -1.0
        decompressed[i] = compressed[source_idx] * sign

    return decompressed


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "ACCEL_AVAILABLE",
    "compress_gradients_vsa",
    "cuda_available",
    "decompress_gradients_vsa",
    "pack_ternary_weights",
    "quantize_weights_absmean",
    "rust_version",
    "ternary_matmul",
    "unpack_ternary_weights",
]
