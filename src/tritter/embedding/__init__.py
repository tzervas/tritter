"""Embedding systems for Tritter.

Why: The core embedding-prediction paradigm operates in continuous embedding space.
This module provides different embedding strategies and the critical "exit point"
for converting continuous embeddings back to discrete tokens.

Components:

1. **Standard** (default): Learned token embeddings (PyTorch nn.Embedding)
2. **Hyperdimensional** (planned): VSA/HDC-based embeddings for symbolic reasoning
3. **KNN Rounding**: Convert embeddings to tokens via nearest neighbor search
4. **VQ Rounding**: Convert embeddings to tokens via learned vector quantization
5. **Unified Rounding**: Single interface abstracting over rounding strategies

Embedding-Prediction Paradigm:

The model operates in continuous embedding space rather than discrete token space.
Token prediction is temporary scaffolding for training compatibility. The rounding
modules (KNN/VQ) provide the "exit point" from continuous space:

- **Entry point**: tokens -> embeddings (via nn.Embedding or encode)
- **Core**: continuous transformer layers in embedding space
- **Exit point**: embeddings -> tokens (via KNN/VQ rounding)

Rounding Modes:
- `knn`: K-Nearest Neighbor using vocabulary embeddings (fast, no training)
- `vq`: Vector Quantization with learned codebook (trainable)
- `none`: Pass-through for pure continuous operation

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
from tritter.embedding.knn_rounding import (
    FAISS_AVAILABLE,
    KNNRouter,
    KNNRouterConfig,
)
from tritter.embedding.rounding import (
    EmbeddingRounder,
    EmbeddingRounderConfig,
    create_rounding_layer,
)
from tritter.embedding.vq_rounding import (
    VQRouter,
    VQRouterConfig,
)

__all__ = [
    # Base embedding classes
    "BaseEmbedding",
    "StandardEmbedding",
    # Hyperdimensional embeddings (stub)
    "HyperdimensionalConfig",
    "HyperdimensionalEmbedding",
    # KNN rounding
    "KNNRouter",
    "KNNRouterConfig",
    "FAISS_AVAILABLE",
    # VQ rounding
    "VQRouter",
    "VQRouterConfig",
    # Unified rounding interface
    "EmbeddingRounder",
    "EmbeddingRounderConfig",
    "create_rounding_layer",
]
