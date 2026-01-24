"""Unified embedding rounding interface for the embedding-prediction paradigm.

Provides a single interface for converting continuous embeddings back to discrete
tokens, abstracting over different rounding strategies (KNN, VQ, or none).

Why: The embedding-prediction paradigm operates in continuous embedding space.
This module provides the "exit point" back to discrete token space when needed.
The unified interface allows switching between rounding strategies without
changing downstream code.

Embedding-Prediction Context: During normal operation, the model produces
continuous embeddings. For text generation, evaluation with cross-entropy loss,
or integration with discrete systems, we need to convert back to tokens. This
module provides:

1. **KNN mode**: Use fixed vocabulary embeddings as codebook (fast, no training)
2. **VQ mode**: Use learned codebook (trainable, may capture distribution better)
3. **None mode**: Pass-through for pure continuous operation (no discretization)

Factory Function: `create_rounding_layer()` creates the appropriate rounder
based on configuration and available vocabulary embeddings.

Reference: docs/project-plan.md (embedding-prediction paradigm exit point)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch import Tensor

from tritter.embedding.knn_rounding import KNNRouter, KNNRouterConfig
from tritter.embedding.vq_rounding import VQRouter, VQRouterConfig

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig


@dataclass
class EmbeddingRounderConfig:
    """Configuration for unified embedding rounding.

    Why: Centralizes rounding configuration, allowing easy switching between
    strategies. The mode determines which underlying router is used.

    Embedding-Prediction Context: Different modes suit different use cases:
    - "knn": Fast inference, uses existing vocab embeddings (no extra training)
    - "vq": Learned codebook, may better match model's internal space
    - "none": Pure continuous mode, no discretization (for embedding tasks)

    Attributes:
        mode: Rounding strategy. "knn" uses vocabulary embeddings with nearest
            neighbor search. "vq" uses a learned codebook. "none" disables
            rounding (pass-through).
        knn_config: Configuration for KNN router (used when mode="knn")
        vq_config: Configuration for VQ router (used when mode="vq")
    """

    mode: Literal["knn", "vq", "none"] = "knn"
    knn_config: KNNRouterConfig | None = None
    vq_config: VQRouterConfig | None = None

    def __post_init__(self) -> None:
        """Set default configs if not provided."""
        if self.knn_config is None:
            self.knn_config = KNNRouterConfig()
        if self.vq_config is None:
            self.vq_config = VQRouterConfig()


class EmbeddingRounder(nn.Module):
    """Unified embedding rounding interface.

    Why: Provides a single API for converting continuous embeddings to discrete
    tokens, regardless of the underlying rounding strategy. This abstraction
    enables easy experimentation with different rounding methods.

    Embedding-Prediction Context: This is the primary "exit point" from the
    continuous embedding space. The model operates in continuous space, and
    when discrete tokens are needed (generation, evaluation, tokenization),
    the rounder converts embeddings back to token IDs.

    Modes:
    - **knn**: K-Nearest Neighbor routing using vocabulary embeddings. Fast,
        no additional training required. Best for inference.
    - **vq**: Vector Quantization with learned codebook. Trainable, may better
        match the model's internal representation. Adds VQ loss to training.
    - **none**: Pass-through mode. Returns zeros for token IDs, passes embeddings
        unchanged. Used for pure continuous operation (embedding extraction).

    Attributes:
        config: EmbeddingRounderConfig with mode and strategy-specific settings
        mode: Current rounding mode ("knn", "vq", or "none")
        router: Underlying router (KNNRouter or VQRouter), None for "none" mode

    Example:
        >>> from tritter.embedding.rounding import EmbeddingRounder, EmbeddingRounderConfig
        >>> # KNN mode with vocabulary embeddings
        >>> vocab_embeddings = torch.randn(1000, 256)
        >>> config = EmbeddingRounderConfig(mode="knn")
        >>> rounder = EmbeddingRounder(config, vocab_embeddings)
        >>> embeddings = torch.randn(2, 10, 256)
        >>> token_ids = rounder.round(embeddings)  # (2, 10)

        >>> # VQ mode with learned codebook
        >>> config = EmbeddingRounderConfig(mode="vq")
        >>> rounder = EmbeddingRounder(config)
        >>> token_ids, quantized = rounder.round_with_embeddings(embeddings)
    """

    def __init__(
        self,
        config: EmbeddingRounderConfig,
        vocab_embeddings: Tensor | None = None,
    ) -> None:
        """Initialize embedding rounder.

        Args:
            config: EmbeddingRounderConfig with mode and strategy settings
            vocab_embeddings: Vocabulary embedding matrix (V, D) required for
                KNN mode. Optional for VQ mode (uses learned codebook).
                Ignored for "none" mode.

        Raises:
            ValueError: If mode="knn" but vocab_embeddings is None

        Why: KNN mode requires vocabulary embeddings as the codebook. VQ mode
        learns its own codebook. "None" mode requires no codebook.
        """
        super().__init__()
        self.config = config
        self.mode = config.mode

        # Initialize _hidden_size first to avoid redefinition
        self._hidden_size: int | None = None

        if self.mode == "knn":
            if vocab_embeddings is None:
                raise ValueError(
                    "vocab_embeddings is required for KNN mode. "
                    "Pass the vocabulary embedding matrix (V, D) or use mode='vq' "
                    "for a learned codebook."
                )
            self.router: KNNRouter | VQRouter | None = KNNRouter(
                vocab_embeddings, config.knn_config
            )
            self._hidden_size = vocab_embeddings.shape[1]

        elif self.mode == "vq":
            assert config.vq_config is not None
            self.router = VQRouter(config.vq_config)
            self._hidden_size = config.vq_config.hidden_size

        else:  # mode == "none"
            self.router = None
            # For "none" mode, hidden_size is determined at runtime

    @property
    def hidden_size(self) -> int | None:
        """Embedding dimension, if known."""
        return self._hidden_size

    def round(self, embeddings: Tensor) -> Tensor:
        """Round embeddings to discrete token IDs.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Token IDs of shape (B, L) or (B,). For "none" mode, returns
            zeros with appropriate shape.

        Why: Main interface for converting continuous embeddings to discrete
        tokens. Used for text generation and evaluation.

        Embedding-Prediction Context: This is the "exit point" that converts
        model outputs back to the discrete token space.
        """
        if self.mode == "none":
            # Pass-through mode: return placeholder zeros
            if embeddings.dim() == 3:
                B, L, _ = embeddings.shape
                return torch.zeros(B, L, dtype=torch.long, device=embeddings.device)
            else:
                B = embeddings.shape[0]
                return torch.zeros(B, dtype=torch.long, device=embeddings.device)

        elif self.mode == "knn":
            assert isinstance(self.router, KNNRouter)
            result: Tensor = self.router(embeddings)
            return result

        else:  # mode == "vq"
            assert isinstance(self.router, VQRouter)
            codes: Tensor
            codes, _, _ = self.router(embeddings)
            return codes

    def round_with_embeddings(self, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Round embeddings and return both token IDs and quantized embeddings.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Tuple of (token_ids, quantized_embeddings, loss) where:
                - token_ids: Discrete token indices, shape (B, L) or (B,)
                - quantized_embeddings: Quantized embeddings, same shape as input
                - loss: VQ loss for mode="vq", None otherwise

        Why: Provides both discrete tokens and corresponding quantized embeddings.
        The quantized embeddings can be used for downstream processing (e.g.,
        decoder) while tokens are used for text output.

        Embedding-Prediction Context: For VQ mode, returns the commitment/codebook
        loss that should be added to the training objective. For KNN mode, the
        quantized embeddings are the looked-up vocabulary embeddings.
        """
        if self.mode == "none":
            # Pass-through: return input embeddings unchanged
            if embeddings.dim() == 3:
                B, L, _ = embeddings.shape
                token_ids = torch.zeros(B, L, dtype=torch.long, device=embeddings.device)
            else:
                B = embeddings.shape[0]
                token_ids = torch.zeros(B, dtype=torch.long, device=embeddings.device)
            return token_ids, embeddings, None

        elif self.mode == "knn":
            assert isinstance(self.router, KNNRouter)
            token_ids = self.router(embeddings)

            # Get quantized embeddings by looking up the found tokens
            # Why: For KNN, quantized = vocab_embeddings[token_ids]
            quantized: Tensor = self.router.vocab_embeddings[token_ids]

            return token_ids, quantized, None

        else:  # mode == "vq"
            assert isinstance(self.router, VQRouter)
            token_ids, quantized_vq, vq_loss = self.router(embeddings)
            return token_ids, quantized_vq, vq_loss

    def get_soft_distribution(self, embeddings: Tensor) -> Tensor:
        """Get soft probability distribution over tokens.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Probability distribution of shape (B, L, V) or (B, V) where V
            is vocabulary/codebook size. For "none" mode, returns uniform.

        Why: Enables sampling and beam search with soft distributions rather
        than hard argmax. Useful for diverse generation.
        """
        if self.mode == "none":
            # Return uniform distribution
            raise ValueError(
                "Soft distribution not available in 'none' mode. "
                "Use mode='knn' or mode='vq' for probability distributions."
            )

        elif self.mode == "knn":
            assert isinstance(self.router, KNNRouter)
            return self.router.forward_soft(embeddings)

        else:  # mode == "vq"
            # VQ doesn't have built-in soft routing; use distances
            # Why: Convert distances to probabilities via softmax
            assert isinstance(self.router, VQRouter)

            is_3d = embeddings.dim() == 3

            if is_3d:
                B, L, D = embeddings.shape
                z = embeddings.view(-1, D)
            else:
                B = embeddings.shape[0]
                L = None
                z = embeddings

            # Compute distances to codebook
            z_sq = (z**2).sum(dim=-1, keepdim=True)
            e_sq = (self.router.codebook.weight**2).sum(dim=-1, keepdim=True).t()
            cross = z @ self.router.codebook.weight.t()
            distances = z_sq + e_sq - 2 * cross

            # Convert to probabilities (negative distance -> softmax)
            probs = torch.softmax(-distances, dim=-1)

            if is_3d:
                assert L is not None  # Type narrowing for mypy
                probs = probs.view(B, L, -1)

            return probs

    def forward(self, embeddings: Tensor) -> Tensor:
        """Forward pass returning token IDs.

        Args:
            embeddings: Continuous embeddings

        Returns:
            Token IDs

        Why: Enables using EmbeddingRounder as a module in nn.Sequential.
        Equivalent to calling round().
        """
        return self.round(embeddings)


def create_rounding_layer(
    config: TritterConfig | None = None,
    vocab_embeddings: Tensor | None = None,
    mode: Literal["knn", "vq", "none"] = "knn",
    *,
    # KNN-specific parameters
    k: int | None = None,
    temperature: float | None = None,
    use_faiss: bool | None = None,
    use_gpu_faiss: bool | None = None,
    normalize: bool | None = None,
    # VQ-specific parameters
    codebook_size: int | None = None,
    hidden_size: int | None = None,
    commitment_cost: float | None = None,
    use_ema: bool | None = None,
    ema_decay: float | None = None,
) -> EmbeddingRounder:
    """Factory function to create an embedding rounding layer.

    Args:
        config: Optional TritterConfig for extracting hidden_size and vocab_size.
            If provided, uses config values for VQ codebook configuration.
        vocab_embeddings: Vocabulary embedding matrix (V, D). Required for
            KNN mode. If config is provided and vocab_embeddings is None for
            VQ mode, creates codebook matching config dimensions.
        mode: Rounding mode - "knn", "vq", or "none"
        k: Number of nearest neighbors for KNN mode
        temperature: Temperature for KNN soft routing
        use_faiss: Whether to use FAISS for KNN
        use_gpu_faiss: Whether to use GPU for FAISS
        normalize: Whether to normalize embeddings for KNN
        codebook_size: Size of VQ codebook
        hidden_size: Embedding dimension for VQ
        commitment_cost: VQ commitment loss weight
        use_ema: Whether to use EMA updates for VQ
        ema_decay: EMA decay rate for VQ

    Returns:
        Configured EmbeddingRounder instance

    Why: Convenience factory that handles common configuration patterns.
    Simplifies creating a rounder that matches model configuration.

    Embedding-Prediction Context: Creates the appropriate rounding layer
    for the model's exit point from continuous space.

    Example:
        >>> from tritter.core.config import TritterConfig
        >>> from tritter.embedding.rounding import create_rounding_layer

        >>> # Create KNN rounder with model config and vocab embeddings
        >>> config = TritterConfig(model_size="7B")
        >>> vocab_emb = torch.randn(config.vocab_size, config.hidden_size)
        >>> rounder = create_rounding_layer(config, vocab_emb, mode="knn")

        >>> # Create VQ rounder (learns codebook)
        >>> rounder = create_rounding_layer(config, mode="vq")

        >>> # Create with custom parameters
        >>> rounder = create_rounding_layer(
        ...     vocab_embeddings=vocab_emb,
        ...     mode="knn",
        ...     k=5,
        ...     temperature=0.8,
        ... )
    """
    # Build KNN config from provided arguments
    knn_kwargs: dict[str, int | float | bool] = {}
    if k is not None:
        knn_kwargs["k"] = k
    if temperature is not None:
        knn_kwargs["temperature"] = temperature
    if use_faiss is not None:
        knn_kwargs["use_faiss"] = use_faiss
    if use_gpu_faiss is not None:
        knn_kwargs["use_gpu_faiss"] = use_gpu_faiss
    if normalize is not None:
        knn_kwargs["normalize"] = normalize

    # Build VQ config from provided arguments
    vq_kwargs: dict[str, int | float | bool] = {}
    if codebook_size is not None:
        vq_kwargs["codebook_size"] = codebook_size
    if hidden_size is not None:
        vq_kwargs["hidden_size"] = hidden_size
    if commitment_cost is not None:
        vq_kwargs["commitment_cost"] = commitment_cost
    if use_ema is not None:
        vq_kwargs["use_ema"] = use_ema
    if ema_decay is not None:
        vq_kwargs["ema_decay"] = ema_decay

    # Create strategy-specific configs
    knn_config = KNNRouterConfig(**knn_kwargs) if knn_kwargs else None  # type: ignore[arg-type]
    vq_config = None

    if mode == "vq":
        # Use config values for VQ if available
        if config is not None:
            if "codebook_size" not in vq_kwargs:
                vq_kwargs["codebook_size"] = config.vocab_size
            if "hidden_size" not in vq_kwargs:
                vq_kwargs["hidden_size"] = config.hidden_size

        vq_config = VQRouterConfig(**vq_kwargs) if vq_kwargs else VQRouterConfig()  # type: ignore[arg-type]

    # Create rounder config
    rounder_config = EmbeddingRounderConfig(
        mode=mode,
        knn_config=knn_config or KNNRouterConfig(),
        vq_config=vq_config or VQRouterConfig(),
    )

    return EmbeddingRounder(rounder_config, vocab_embeddings)


__all__ = [
    "EmbeddingRounderConfig",
    "EmbeddingRounder",
    "create_rounding_layer",
]
