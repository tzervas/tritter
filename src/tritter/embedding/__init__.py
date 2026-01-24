"""Embedding systems for Tritter.

Why: The core embedding-prediction paradigm operates in continuous embedding space.
This module provides different embedding strategies:

1. **Standard** (default): Learned token embeddings (PyTorch nn.Embedding)
2. **Hyperdimensional** (planned): VSA/HDC-based embeddings for symbolic reasoning

Extension Point: The `HyperdimensionalEmbedding` class provides a stub for
integration with external hyperdimensional computing frameworks like the
Embeddenator project (ternary VSA with engram-type data structures).

Configurable Dimensionality:
- `embedding_dim`: Standard embedding dimension (e.g., 4096)
- `hd_dimension`: Hyperdimensional space dimension (e.g., 10000)
- `hd_mode`: "none" | "ternary_vsa" | "binary_vsa" | "dense"

See SPEC-012 (planned) for hyperdimensional embedding specification.
"""

from tritter.embedding.base import (
    BaseEmbedding,
    StandardEmbedding,
)
from tritter.embedding.hyperdimensional import (
    HyperdimensionalConfig,
    HyperdimensionalEmbedding,
)

__all__ = [
    "BaseEmbedding",
    "HyperdimensionalConfig",
    "HyperdimensionalEmbedding",
    "StandardEmbedding",
]
