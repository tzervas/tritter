"""VSA (Vector Symbolic Architecture) gradient compression.

Why: Hyperdimensional computing enables extremely efficient gradient
approximation using high-dimensional random vectors. Key properties:
- Near-orthogonality: Random vectors are almost orthogonal in high dims
- Distributed representation: Information spread across all dimensions
- Noise tolerance: Robust to errors and approximation
- Efficient operations: Binding, bundling work element-wise

This enables compressing gradients for faster communication and storage
while maintaining training accuracy through correction cycles.

References:
- Kanerva (2009): Hyperdimensional Computing
- Rahimi et al. (2016): High-Dimensional Computing as a Nanoscalable Paradigm
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


def hyperdimensional_bind(a: Tensor, b: Tensor) -> Tensor:
    """Bind two hypervectors using element-wise XOR (for binary) or multiplication.

    Args:
        a: First hypervector (..., D)
        b: Second hypervector (..., D)

    Returns:
        Bound hypervector (..., D)

    Why: Binding creates a new vector that's dissimilar to both inputs
    but can be unbound to retrieve either one. This is the key operation
    for creating structured representations in hyperdimensional space.
    """
    # For real-valued vectors, use element-wise multiplication
    return a * b


def hyperdimensional_bundle(vectors: list[Tensor], weights: list[float] | None = None) -> Tensor:
    """Bundle multiple hypervectors into one superposition.

    Args:
        vectors: List of hypervectors to bundle
        weights: Optional importance weights for each vector

    Returns:
        Bundled hypervector (same dimension as inputs)

    Why: Bundling creates a superposition that's similar to all inputs.
    The result can be queried to retrieve any bundled vector. This enables
    storing multiple gradient directions in a single compressed vector.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list")

    if weights is None:
        weights = [1.0] * len(vectors)

    # Weighted sum
    result = torch.zeros_like(vectors[0])
    for v, w in zip(vectors, weights, strict=True):
        result = result + w * v

    # Normalize to prevent magnitude explosion
    return result / len(vectors)


def ternary_quantize(x: Tensor, scale: float | None = None) -> tuple[Tensor, float]:
    """Quantize tensor to ternary {-1, 0, +1}.

    Args:
        x: Input tensor
        scale: Optional scale factor (computed if None)

    Returns:
        Tuple of (quantized tensor, scale factor)

    Why: Ternary representation enables extremely fast operations using
    only additions/subtractions (no multiplications). The scale factor
    preserves magnitude information for accurate reconstruction.
    """
    if scale is None:
        scale = x.abs().mean().item()

    if scale == 0:
        return torch.zeros_like(x), 0.0

    # Quantize to {-1, 0, +1} using threshold at scale
    quantized = torch.zeros_like(x)
    quantized[x > scale] = 1
    quantized[x < -scale] = -1

    return quantized, scale


@dataclass
class VSAConfig:
    """Configuration for VSA gradient compression."""

    dimension: int = 8192  # Hypervector dimension
    compression_ratio: float = 0.1  # Target compression ratio
    use_ternary: bool = True  # Use ternary quantization
    seed: int = 42  # Random seed for reproducibility


class VSAGradientCompressor:
    """Compress gradients using Vector Symbolic Architecture.

    Why: Gradient compression enables:
    1. Faster gradient synchronization in distributed training
    2. Lower memory for gradient accumulation
    3. Efficient storage of gradient history for prediction

    The VSA approach uses random projection followed by optional
    ternary quantization for extreme compression while maintaining
    the essential gradient direction.

    Example:
        >>> compressor = VSAGradientCompressor(param_count=1_000_000)
        >>> compressed = compressor.compress(gradients)
        >>> reconstructed = compressor.decompress(compressed)
    """

    def __init__(
        self,
        param_count: int,
        config: VSAConfig | None = None,
    ) -> None:
        """Initialize VSA compressor.

        Args:
            param_count: Total number of model parameters
            config: VSA configuration

        Why: Pre-generates random projection matrices for consistent
        compression/decompression. The matrices are stored in a memory-
        efficient way using seeds for regeneration.
        """
        self.config = config or VSAConfig()
        self.param_count = param_count

        # Compressed dimension
        self.compressed_dim = max(
            256,
            int(param_count * self.config.compression_ratio)
        )

        # Generator for reproducible random projections
        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

        # Cache for projection chunks
        self._projection_cache: dict[int, Tensor] = {}

    def _get_projection_chunk(self, chunk_idx: int, chunk_size: int, device: torch.device) -> Tensor:
        """Get or generate projection matrix for a chunk.

        Why: Generating the full projection matrix would require
        O(param_count * compressed_dim) memory. Instead, we generate
        chunks on-demand using seeded random numbers.
        """
        cache_key = (chunk_idx, chunk_size)
        if cache_key not in self._projection_cache:
            # Use deterministic seed for this chunk
            chunk_gen = torch.Generator()
            chunk_gen.manual_seed(self.config.seed + chunk_idx)

            # Random Gaussian projection (Johnson-Lindenstrauss)
            proj = torch.randn(
                chunk_size, self.compressed_dim,
                generator=chunk_gen,
                device=device,
            ) / (chunk_size ** 0.5)

            self._projection_cache[cache_key] = proj

        return self._projection_cache[cache_key]

    def compress(
        self,
        gradients: dict[str, Tensor] | Iterator[tuple[str, Tensor]],
    ) -> tuple[Tensor, dict[str, tuple[int, float]]]:
        """Compress gradients to hyperdimensional representation.

        Args:
            gradients: Dictionary or iterator of (name, gradient) pairs

        Returns:
            Tuple of:
                - Compressed gradient vector (compressed_dim,)
                - Metadata for reconstruction {name: (offset, scale)}

        Why: Compresses all gradients into a single hypervector that
        preserves the aggregate gradient direction. Individual gradient
        reconstruction uses the metadata for proper scaling.
        """
        if isinstance(gradients, dict):
            gradients = gradients.items()

        compressed = None
        metadata: dict[str, tuple[int, float]] = {}
        offset = 0

        for name, grad in gradients:
            if grad is None:
                continue

            flat = grad.flatten()
            size = flat.numel()

            # Get projection for this chunk
            proj = self._get_projection_chunk(offset, size, flat.device)

            # Project gradient
            projected = flat @ proj  # (compressed_dim,)

            # Optional ternary quantization
            if self.config.use_ternary:
                projected, scale = ternary_quantize(projected)
            else:
                scale = 1.0

            # Bundle into compressed representation
            if compressed is None:
                compressed = projected
            else:
                compressed = compressed + projected

            metadata[name] = (offset, scale)
            offset += size

        if compressed is None:
            raise ValueError("No gradients to compress")

        # Normalize bundled result
        compressed = compressed / len(metadata)

        return compressed, metadata

    def decompress(
        self,
        compressed: Tensor,
        metadata: dict[str, tuple[int, float]],
        shapes: dict[str, torch.Size],
    ) -> dict[str, Tensor]:
        """Decompress gradients from hyperdimensional representation.

        Args:
            compressed: Compressed gradient vector
            metadata: Metadata from compression {name: (offset, scale)}
            shapes: Original shapes {name: shape}

        Returns:
            Dictionary of reconstructed gradients

        Why: Reconstructs approximate gradients using inverse projection.
        The reconstruction is lossy but preserves the gradient direction
        which is sufficient for SGD-style optimization.
        """
        gradients = {}

        for name, (offset, scale) in metadata.items():
            shape = shapes[name]
            size = shape.numel()

            # Get projection for this chunk
            proj = self._get_projection_chunk(offset, size, compressed.device)

            # Inverse projection (pseudo-inverse via transpose)
            reconstructed = compressed @ proj.T  # (size,)

            # Apply scale
            if scale != 0:
                reconstructed = reconstructed * scale

            # Reshape to original
            gradients[name] = reconstructed.reshape(shape)

        return gradients

    def get_compression_stats(self) -> dict[str, float]:
        """Get compression statistics.

        Returns:
            Dictionary with compression metrics
        """
        return {
            "original_params": self.param_count,
            "compressed_dim": self.compressed_dim,
            "compression_ratio": self.compressed_dim / self.param_count,
            "memory_saving": 1 - (self.compressed_dim / self.param_count),
        }


__all__ = [
    "VSAConfig",
    "VSAGradientCompressor",
    "hyperdimensional_bind",
    "hyperdimensional_bundle",
    "ternary_quantize",
]
