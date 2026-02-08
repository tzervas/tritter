"""KNN-based embedding rounding for the embedding-prediction paradigm.

Implements K-Nearest Neighbor routing to convert continuous embeddings back to
discrete tokens by finding the nearest vocabulary embeddings.

Why: In the embedding-prediction paradigm, the model operates in continuous
embedding space rather than discrete token space. KNN rounding provides the
"exit point" from this continuous space, mapping model outputs back to discrete
tokens when needed (e.g., for text generation, evaluation, or training with
cross-entropy loss).

Embedding-Prediction Context: KNN rounding is the inference-time counterpart to
the embedding lookup in forward pass. While embedding lookup converts tokens ->
embeddings, KNN rounding converts embeddings -> tokens. For hard routing (k=1),
this is equivalent to argmax over the projection to vocabulary. For soft routing,
we get a distribution over tokens that can be used for sampling or beam search.

FAISS Integration: When available, FAISS provides highly optimized nearest neighbor
search using GPU acceleration and approximate algorithms (IVF, HNSW). For vocabularies
up to ~100K tokens, exact search is fast enough; larger vocabularies benefit from
approximate methods.

Reference: docs/project-plan.md (KNN/VQ rounding in embedding-prediction paradigm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try to import FAISS for accelerated KNN
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

if TYPE_CHECKING:
    pass


@dataclass
class KNNRouterConfig:
    """Configuration for KNN-based embedding rounding.

    Why: Controls the behavior of nearest neighbor lookup. k=1 provides hard
    routing (argmax), while k>1 enables soft routing with multiple candidates.
    Temperature scaling modulates the softness of the distribution.

    Embedding-Prediction Context: The KNN router converts continuous model
    outputs back to discrete tokens. Hard routing (k=1) is used during greedy
    decoding; soft routing (k>1) enables sampling and beam search strategies.

    Attributes:
        k: Number of nearest neighbors to retrieve. k=1 is argmax (greedy),
            k>1 enables top-k sampling. Default 1 for deterministic rounding.
        temperature: Temperature for soft routing probability distribution.
            Lower values (0.1) make distribution sharper (more deterministic),
            higher values (2.0) make it flatter (more random). Default 1.0.
        use_faiss: Whether to use FAISS for accelerated KNN if available.
            FAISS provides GPU-accelerated exact and approximate search.
            Falls back to brute-force PyTorch if False or FAISS not installed.
        use_gpu_faiss: Whether to use GPU acceleration for FAISS index.
            Requires FAISS-GPU and sufficient GPU memory. Default True.
        normalize: Whether to L2-normalize embeddings before computing
            distances. When True, cosine similarity is used; when False,
            Euclidean distance is used. Default True (cosine similarity).
    """

    k: int = 1
    temperature: float = 1.0
    use_faiss: bool = True
    use_gpu_faiss: bool = True
    normalize: bool = True


class KNNRouter(nn.Module):  # type: ignore[misc]
    """KNN-based router for mapping embeddings to token IDs.

    Why: Provides the "exit point" from continuous embedding space back to
    discrete token space. Uses nearest neighbor search to find the vocabulary
    embedding(s) closest to each input embedding.

    Embedding-Prediction Context: During inference, the model produces continuous
    embeddings. KNN rounding converts these to discrete tokens for text output.
    This is a non-learned operation that uses the fixed vocabulary embeddings
    as the codebook.

    The router supports two modes:
    1. **Hard routing** (k=1): Returns the single nearest token (argmax)
    2. **Soft routing** (k>1): Returns probabilities over top-k tokens

    Attributes:
        config: KNNRouterConfig with routing parameters
        vocab_embeddings: Vocabulary embedding matrix (V, D)
        index: FAISS index if available, else None

    Example:
        >>> from tritter.embedding.knn_rounding import KNNRouter, KNNRouterConfig
        >>> vocab_embeddings = torch.randn(1000, 256)  # 1K vocab, 256-dim
        >>> router = KNNRouter(vocab_embeddings, KNNRouterConfig(k=5))
        >>> embeddings = torch.randn(2, 10, 256)  # (B, L, D)
        >>> token_ids = router(embeddings)  # (2, 10)
        >>> probs = router.forward_soft(embeddings)  # (2, 10, 1000)
    """

    def __init__(
        self,
        vocab_embeddings: Tensor,
        config: KNNRouterConfig | None = None,
    ) -> None:
        """Initialize KNN router with vocabulary embeddings.

        Args:
            vocab_embeddings: Vocabulary embedding matrix of shape (V, D) where
                V is vocabulary size and D is embedding dimension
            config: KNNRouterConfig with routing parameters. Defaults to
                KNNRouterConfig() with k=1 (argmax routing)

        Why: The vocabulary embeddings define the discrete token space. The
        router finds the nearest embedding(s) for each continuous input.
        Optionally builds a FAISS index for accelerated search.
        """
        super().__init__()
        self.config = config or KNNRouterConfig()

        # Store vocabulary embeddings as buffer (not a parameter)
        # Why: These are fixed codebook entries, not learned during training
        self.register_buffer("vocab_embeddings", vocab_embeddings.clone())

        self.vocab_size, self.embedding_dim = vocab_embeddings.shape

        # Build FAISS index if available and requested
        self._index: faiss.Index | None = None
        if self.config.use_faiss and FAISS_AVAILABLE:
            self.build_index(vocab_embeddings)

    def build_index(self, embeddings: Tensor) -> None:
        """Build or rebuild FAISS index for efficient KNN search.

        Args:
            embeddings: Embedding matrix of shape (V, D)

        Why: FAISS provides highly optimized nearest neighbor search. For
        vocabularies up to ~100K, exact search (IndexFlatIP or IndexFlatL2)
        is fast enough. Larger vocabularies may benefit from approximate
        methods (IVF, HNSW), but those require more complex configuration.
        """
        if not FAISS_AVAILABLE:
            return

        # Convert to numpy for FAISS
        embeddings_np = embeddings.detach().cpu().float().numpy()

        if self.config.normalize:
            # Normalize for cosine similarity (use inner product on normalized vectors)
            # Why: Cosine similarity = dot product of L2-normalized vectors
            faiss.normalize_L2(embeddings_np)
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Use L2 distance
            self._index = faiss.IndexFlatL2(self.embedding_dim)

        self._index.add(embeddings_np)

        # Move to GPU if available and requested
        if self.config.use_gpu_faiss and torch.cuda.is_available():
            try:
                # Get FAISS GPU resources
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except (AttributeError, RuntimeError):
                # FAISS-GPU not available or failed to initialize
                pass

    def update_index(self, new_embeddings: Tensor) -> None:
        """Update the vocabulary embeddings and rebuild index.

        Args:
            new_embeddings: New vocabulary embedding matrix (V', D)

        Why: Enables dynamic vocabulary updates (e.g., adding domain-specific
        tokens after initial training). Rebuilds the FAISS index with new
        embeddings.
        """
        # Update stored embeddings
        self.vocab_size, self.embedding_dim = new_embeddings.shape
        self.register_buffer("vocab_embeddings", new_embeddings.clone())

        # Rebuild index
        if self.config.use_faiss and FAISS_AVAILABLE:
            self.build_index(new_embeddings)

    def _knn_faiss(self, query: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Find k-nearest neighbors using FAISS.

        Args:
            query: Query embeddings (N, D)
            k: Number of neighbors

        Returns:
            Tuple of (distances, indices) each of shape (N, k)
        """
        if self._index is None:
            raise RuntimeError("FAISS index not built")

        query_np = query.detach().cpu().float().numpy()

        if self.config.normalize:
            faiss.normalize_L2(query_np)

        distances, indices = self._index.search(query_np, k)

        # Convert back to tensors on same device as query
        distances_t = torch.from_numpy(distances).to(query.device)
        indices_t = torch.from_numpy(indices).long().to(query.device)

        return distances_t, indices_t

    def _knn_torch(self, query: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Find k-nearest neighbors using brute-force PyTorch.

        Args:
            query: Query embeddings (N, D)
            k: Number of neighbors

        Returns:
            Tuple of (distances/similarities, indices) each of shape (N, k)

        Why: Fallback when FAISS is not available. Uses matrix multiplication
        for cosine similarity or pairwise L2 distance. Efficient for small
        vocabularies but scales O(V*D) per query.
        """
        vocab = self.vocab_embeddings  # (V, D)

        if self.config.normalize:
            # Cosine similarity: normalize then dot product
            query_norm = F.normalize(query, p=2, dim=-1)  # (N, D)
            vocab_norm = F.normalize(vocab, p=2, dim=-1)  # (V, D)
            similarities = query_norm @ vocab_norm.t()  # (N, V)

            # Get top-k (highest similarity)
            topk_sim, topk_idx = similarities.topk(k, dim=-1)
            return topk_sim, topk_idx
        else:
            # L2 distance: ||q - v||^2 = ||q||^2 + ||v||^2 - 2<q,v>
            # Why: Expanding the squared distance avoids explicit broadcasting
            query_sq = (query**2).sum(dim=-1, keepdim=True)  # (N, 1)
            vocab_sq = (vocab**2).sum(dim=-1, keepdim=True).t()  # (1, V)
            cross = query @ vocab.t()  # (N, V)
            distances = query_sq + vocab_sq - 2 * cross  # (N, V)

            # Get top-k (smallest distance)
            topk_dist, topk_idx = distances.topk(k, dim=-1, largest=False)
            return topk_dist, topk_idx

    def forward(self, embeddings: Tensor) -> Tensor:
        """Find nearest vocabulary token for each embedding.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Token IDs of shape (B, L) or (B,) containing indices of nearest
            vocabulary embeddings

        Why: Main inference method for hard routing. Returns the single nearest
        token (k=1 argmax) regardless of config.k setting. For soft routing
        with top-k candidates, use forward_soft() instead.

        Embedding-Prediction Context: Converts continuous model outputs to
        discrete token IDs. This is the reverse of embedding lookup.
        """
        # Handle both 2D and 3D inputs
        is_3d = embeddings.dim() == 3

        if is_3d:
            B, L, D = embeddings.shape
            embeddings_flat = embeddings.view(-1, D)  # (B*L, D)
        else:
            embeddings_flat = embeddings  # (B, D)

        # Find single nearest neighbor (k=1 for hard routing)
        if self._index is not None:
            _, indices = self._knn_faiss(embeddings_flat, k=1)
            indices = indices.squeeze(-1)  # (N,)
        else:
            _, indices = self._knn_torch(embeddings_flat, k=1)
            indices = indices.squeeze(-1)  # (N,)

        # Reshape back to original batch structure
        if is_3d:
            indices = indices.view(B, L)

        return indices

    def forward_soft(self, embeddings: Tensor) -> Tensor:
        """Compute soft probability distribution over vocabulary.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Probability distribution of shape (B, L, V) or (B, V) where V
            is vocabulary size. When config.k < vocab_size, only top-k
            entries are non-zero for efficiency.

        Why: Enables soft routing for sampling and beam search. The temperature
        parameter modulates the sharpness of the distribution.

        Embedding-Prediction Context: For sampling-based generation, we need
        a full probability distribution, not just the argmax. This method
        computes (optionally sparse) softmax over similarity/distance to
        vocabulary embeddings.
        """
        is_3d = embeddings.dim() == 3

        if is_3d:
            B, L, D = embeddings.shape
            embeddings_flat = embeddings.view(-1, D)  # (B*L, D)
        else:
            B = embeddings.shape[0]
            L = 1
            embeddings_flat = embeddings

        N = embeddings_flat.shape[0]
        k = min(self.config.k, self.vocab_size)

        # Get top-k similarities/distances
        if self._index is not None:
            scores, indices = self._knn_faiss(embeddings_flat, k)
        else:
            scores, indices = self._knn_torch(embeddings_flat, k)

        # Convert distances to similarities if using L2
        if not self.config.normalize:
            # Distance to similarity: s = -d (or exp(-d) for bounded [0,1])
            # Why: Softmax expects higher = more likely
            scores = -scores

        # Apply temperature scaling
        scores = scores / self.config.temperature

        # Full vocabulary size output
        if k == self.vocab_size:
            # Already have all scores
            probs = F.softmax(scores, dim=-1)  # (N, V)
        else:
            # Sparse output: only top-k are non-zero
            # Create full probability tensor
            probs = torch.zeros(N, self.vocab_size, device=embeddings.device)

            # Compute softmax over top-k only
            topk_probs = F.softmax(scores, dim=-1)  # (N, k)

            # Scatter into full tensor
            probs.scatter_(-1, indices, topk_probs)

        # Reshape back to original batch structure
        if is_3d:
            probs = probs.view(B, L, self.vocab_size)
        else:
            probs = probs.view(B, self.vocab_size)

        return probs

    def get_similarities(self, embeddings: Tensor) -> Tensor:
        """Compute similarities to all vocabulary embeddings.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Similarities/logits of shape (B, L, V) or (B, V)

        Why: Provides raw similarity scores without softmax normalization.
        Useful for computing cross-entropy loss against target tokens or
        for custom decoding strategies.
        """
        is_3d = embeddings.dim() == 3

        if is_3d:
            B, L, D = embeddings.shape
            embeddings_flat = embeddings.view(-1, D)  # (B*L, D)
        else:
            embeddings_flat = embeddings

        vocab = self.vocab_embeddings  # (V, D)

        if self.config.normalize:
            embeddings_norm = F.normalize(embeddings_flat, p=2, dim=-1)
            vocab_norm = F.normalize(vocab, p=2, dim=-1)
            similarities = embeddings_norm @ vocab_norm.t()  # (N, V)
        else:
            # Use negative L2 distance as similarity
            query_sq = (embeddings_flat**2).sum(dim=-1, keepdim=True)
            vocab_sq = (vocab**2).sum(dim=-1, keepdim=True).t()
            cross = embeddings_flat @ vocab.t()
            distances = query_sq + vocab_sq - 2 * cross
            similarities = -distances

        if is_3d:
            similarities = similarities.view(B, L, self.vocab_size)

        return similarities


__all__ = [
    "KNNRouterConfig",
    "KNNRouter",
    "FAISS_AVAILABLE",
]
