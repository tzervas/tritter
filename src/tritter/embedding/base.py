"""Base embedding classes.

Why: Provides a common interface for different embedding strategies,
enabling seamless swapping between standard and hyperdimensional approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEmbedding(ABC, nn.Module):
    """Abstract base class for embedding layers.

    Why: Defines the interface that all embedding strategies must implement.
    This enables the model to use different embedding types without code changes.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (B, L)

        Returns:
            Embeddings of shape (B, L, embedding_dim)
        """
        ...

    @abstractmethod
    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings back to token logits or IDs.

        Args:
            embeddings: Embedding vectors of shape (B, L, embedding_dim)

        Returns:
            Logits of shape (B, L, vocab_size) or IDs of shape (B, L)

        Why: For embedding-prediction paradigm, we need to map from
        continuous embedding space back to discrete tokens.
        """
        ...


class StandardEmbedding(BaseEmbedding):
    """Standard learned token embeddings.

    Why: Default embedding strategy using PyTorch's nn.Embedding.
    Provides the baseline for comparison with alternative approaches.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
    ):
        super().__init__(vocab_size, embedding_dim, padding_idx)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
        )
        # Output projection for inverse mapping
        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)
        # Tie weights by default
        self.output_proj.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (B, L)

        Returns:
            Embeddings of shape (B, L, embedding_dim)
        """
        return self.embedding(input_ids)  # (B, L, D)

    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to logits.

        Args:
            embeddings: Embedding vectors of shape (B, L, embedding_dim)

        Returns:
            Logits of shape (B, L, vocab_size)
        """
        return self.output_proj(embeddings)  # (B, L, V)


__all__ = [
    "BaseEmbedding",
    "StandardEmbedding",
]
