"""Vector Quantization (VQ) based embedding rounding with learned codebook.

Implements VQ-VAE style vector quantization for converting continuous embeddings
to discrete tokens using a learned codebook rather than fixed vocabulary embeddings.

Why: While KNN rounding uses the fixed vocabulary embeddings as the codebook,
VQ rounding learns a separate codebook optimized for the embedding distribution.
This is useful when:
1. The model operates in a different embedding space than the vocabulary
2. You want to compress the discrete token space (codebook_size < vocab_size)
3. Training end-to-end with VQ loss for better quantization quality

Embedding-Prediction Context: VQ rounding is an alternative "exit point" from
continuous embedding space. Unlike KNN (which uses fixed vocab embeddings),
VQ learns codebook entries that may better represent the model's internal
representation space. The straight-through estimator enables gradient flow
through the discrete bottleneck during training.

VQ Training: The codebook is updated via:
1. Codebook loss: moves codebook entries toward encoder outputs (sg[z])
2. Commitment loss: moves encoder outputs toward codebook entries (z vs sg[e])
3. EMA updates (optional): exponential moving average of cluster assignments

Reference: docs/project-plan.md, VQ-VAE (van den Oord et al., 2017)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    pass


@dataclass
class VQRouterConfig:
    """Configuration for VQ-based embedding rounding.

    Why: VQ routing learns a codebook for discretizing continuous embeddings.
    The codebook size can differ from vocabulary size, enabling compression
    or expansion of the discrete space.

    Embedding-Prediction Context: The VQ router provides a learned discretization
    that may better match the model's internal representation than using fixed
    vocabulary embeddings. Commitment loss encourages the encoder to produce
    embeddings close to codebook entries, enabling stable training.

    Attributes:
        codebook_size: Number of entries in the codebook. Typically matches
            vocab_size for direct token mapping, but can be smaller for
            compression or larger for finer granularity. Default 65536 (2^16).
        hidden_size: Dimension of embeddings and codebook entries. Must match
            the model's hidden dimension. Default 768.
        commitment_cost: Weight for commitment loss that encourages encoder
            outputs to stay close to codebook entries. Higher values (0.5-1.0)
            lead to more stable but potentially less expressive quantization.
            Lower values (0.1-0.25) allow more flexibility. Default 0.25.
        use_ema: Whether to use Exponential Moving Average for codebook updates
            instead of gradient descent. EMA often leads to better codebook
            utilization and more stable training. Default True.
        ema_decay: Decay rate for EMA updates. Higher values (0.99) lead to
            slower but more stable updates. Lower values (0.9) adapt faster
            but may be less stable. Default 0.99.
        epsilon: Small constant for numerical stability in EMA updates.
            Default 1e-5.
    """

    codebook_size: int = 65536
    hidden_size: int = 768
    commitment_cost: float = 0.25
    use_ema: bool = True
    ema_decay: float = 0.99
    epsilon: float = 1e-5


class VQRouter(nn.Module):  # type: ignore[misc]
    """Vector Quantization router with learned codebook.

    Why: Provides a learned discretization layer that maps continuous embeddings
    to discrete codes. Unlike KNN routing (which uses fixed vocabulary embeddings),
    VQ routing learns codebook entries optimized for the embedding distribution.

    Embedding-Prediction Context: VQ is an alternative "exit point" from continuous
    space. The codebook acts as a learned vocabulary of embedding prototypes.
    The straight-through estimator enables gradient flow during training:
    forward uses quantized embeddings, backward passes gradients to encoder.

    Features:
    - Learned codebook via gradient descent or EMA updates
    - Commitment loss for training stability
    - Straight-through gradient estimator
    - Codebook usage tracking for collapse detection

    Attributes:
        config: VQRouterConfig with quantization parameters
        codebook: nn.Embedding containing codebook entries

    Example:
        >>> from tritter.embedding.vq_rounding import VQRouter, VQRouterConfig
        >>> config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        >>> router = VQRouter(config)
        >>> embeddings = torch.randn(2, 10, 256)  # (B, L, D)
        >>> codes, quantized, vq_loss = router(embeddings)
        >>> # codes: (2, 10) - discrete indices
        >>> # quantized: (2, 10, 256) - quantized embeddings
        >>> # vq_loss: scalar - VQ training loss
    """

    def __init__(self, config: VQRouterConfig) -> None:
        """Initialize VQ router with learned codebook.

        Args:
            config: VQRouterConfig with quantization parameters

        Why: Initializes codebook with uniform distribution. EMA buffers track
        cluster usage for stable codebook updates without gradient descent.
        """
        super().__init__()
        self.config = config
        self.codebook_size = config.codebook_size
        self.hidden_size = config.hidden_size
        self.commitment_cost = config.commitment_cost
        self.use_ema = config.use_ema
        self.ema_decay = config.ema_decay

        # Codebook as embedding layer
        # Why: nn.Embedding provides efficient lookup and gradient accumulation
        self.codebook = nn.Embedding(config.codebook_size, config.hidden_size)

        # Initialize codebook uniformly in [-1/K, 1/K]
        # Why: Uniform initialization prevents initial bias toward certain codes
        nn.init.uniform_(
            self.codebook.weight,
            -1 / config.codebook_size,
            1 / config.codebook_size,
        )

        # EMA tracking buffers (if enabled)
        if self.use_ema:
            # Cluster sizes for each codebook entry
            self.register_buffer("ema_cluster_size", torch.zeros(config.codebook_size))
            # Running sum of embeddings assigned to each cluster
            self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def _quantize_core(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Core quantization: find nearest codebook entry for each embedding.

        Args:
            z: Flattened embeddings of shape (N, D)

        Returns:
            Tuple of (codes, quantized) where:
                - codes: Nearest codebook indices (N,)
                - quantized: Corresponding codebook entries (N, D)
        """
        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2<z,e>
        # Why: Expanding the squared distance avoids O(N*K*D) memory broadcast
        z_sq = (z**2).sum(dim=-1, keepdim=True)  # (N, 1)
        e_sq = (self.codebook.weight**2).sum(dim=-1, keepdim=True).t()  # (1, K)
        cross = z @ self.codebook.weight.t()  # (N, K)
        distances = z_sq + e_sq - 2 * cross  # (N, K)

        # Find nearest codebook entry
        codes = distances.argmin(dim=-1)  # (N,)

        # Look up quantized embeddings
        quantized = self.codebook(codes)  # (N, D)

        return codes, quantized

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize embeddings to discrete codes with VQ loss.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Tuple of (codes, quantized, vq_loss) where:
                - codes: Discrete codebook indices (B, L) or (B,)
                - quantized: Quantized embeddings with straight-through gradient,
                    same shape as input
                - vq_loss: Scalar VQ loss (codebook + commitment) for training

        Why: Full forward pass with VQ loss computation. The straight-through
        estimator enables gradient flow: forward uses quantized embeddings,
        backward passes gradients directly to encoder.

        Embedding-Prediction Context: During training, minimize vq_loss to learn
        good codebook entries. The quantized embeddings can be used for downstream
        tasks (decoding, reconstruction) while codes provide discrete tokens.
        """
        is_3d = embeddings.dim() == 3

        if is_3d:
            B, L, D = embeddings.shape
            z = embeddings.view(-1, D)  # (N, D) where N = B*L
        else:
            B = embeddings.shape[0]
            L = None
            z = embeddings  # (B, D)

        # Find nearest codebook entries
        codes, quantized = self._quantize_core(z)

        # Compute VQ losses
        # Why: Codebook loss moves codebook toward encoder outputs (via EMA or gradient)
        # Commitment loss moves encoder toward codebook (regularization)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: use quantized in forward, pass gradient to z
        # Why: Discrete lookup has no gradient; STE copies gradient from output to input
        quantized = z + (quantized - z).detach()

        # EMA codebook update during training
        if self.use_ema and self.training:
            self._update_ema(z.detach(), codes)

        # Reshape to original batch structure
        if is_3d:
            assert L is not None  # Type narrowing for mypy
            codes = codes.view(B, L)
            quantized = quantized.view(B, L, D)

        return codes, quantized, vq_loss

    def _update_ema(self, z: Tensor, codes: Tensor) -> None:
        """Update codebook using exponential moving average.

        Args:
            z: Encoder outputs (N, D)
            codes: Assigned codes (N,)

        Why: EMA updates provide more stable codebook learning than gradient
        descent, especially for rarely-used codes. The cluster size tracks
        usage, and Laplace smoothing prevents dead codes.
        """
        # One-hot encoding of codes
        one_hot = F.one_hot(codes, self.codebook_size).float()  # (N, K)

        # Update cluster sizes (how many embeddings assigned to each code)
        cluster_size = one_hot.sum(0)  # (K,)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

        # Update embedding sums (sum of embeddings assigned to each code)
        embed_sum = one_hot.t() @ z  # (K, D)
        self.ema_embed_sum.data.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        # Update codebook with Laplace smoothing for stability
        # Why: Smoothing prevents division by zero for unused codes
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.config.epsilon)
            / (n + self.codebook_size * self.config.epsilon)
            * n
        )

        # New codebook = sum of assigned embeddings / cluster size
        self.codebook.weight.data.copy_(self.ema_embed_sum / cluster_size_smoothed.unsqueeze(-1))

    def quantize(self, embeddings: Tensor) -> Tensor:
        """Quantize embeddings without computing loss.

        Args:
            embeddings: Continuous embeddings of shape (B, L, D) or (B, D)

        Returns:
            Quantized embeddings with same shape as input

        Why: Efficient inference-only quantization without loss computation
        or EMA updates. Use for generation or evaluation.
        """
        is_3d = embeddings.dim() == 3

        if is_3d:
            B, L, D = embeddings.shape
            z = embeddings.view(-1, D)
        else:
            z = embeddings

        _, quantized = self._quantize_core(z)

        # Straight-through for gradient flow
        quantized = z + (quantized - z).detach()

        if is_3d:
            quantized = quantized.view(B, L, D)

        return quantized

    def lookup(self, codes: Tensor) -> Tensor:
        """Look up embeddings from codebook by code indices.

        Args:
            codes: Code indices of shape (B, L) or (B,) or (N,)

        Returns:
            Corresponding codebook embeddings with shape (*codes.shape, D)

        Why: Enables decoding from discrete codes back to continuous embeddings.
        This is the inverse of quantization - used for generation and
        reconstruction.

        Embedding-Prediction Context: During generation, the model may produce
        discrete codes that need to be converted back to embeddings for
        further processing or output projection.
        """
        result: Tensor = self.codebook(codes)
        return result

    def get_codebook_usage(self, codes: Tensor | None = None) -> float:
        """Compute fraction of codebook entries that are used.

        Args:
            codes: Optional codes tensor to analyze. If None, uses EMA
                cluster sizes (requires use_ema=True and some training).

        Returns:
            Fraction of codebook entries with non-zero usage (0.0 to 1.0)

        Why: Low codebook usage indicates codebook collapse, where the model
        only uses a small subset of available codes. This is a training
        pathology that reduces effective vocabulary size.
        """
        if codes is not None:
            flat_codes = codes.flatten()
            unique_codes = torch.unique(flat_codes).numel()
            return float(unique_codes / self.codebook_size)
        elif self.use_ema:
            # Use EMA cluster sizes to estimate usage
            used_codes = (self.ema_cluster_size > 0).sum().item()
            return float(used_codes) / float(self.codebook_size)
        else:
            # Cannot compute without codes or EMA stats
            return 0.0

    def get_perplexity(self, codes: Tensor) -> float:
        """Compute codebook perplexity from code distribution.

        Args:
            codes: Code indices of shape (*, )

        Returns:
            Perplexity of the code distribution. Higher values indicate
            more uniform usage. Maximum = codebook_size (all codes equally likely).

        Why: Perplexity is a measure of effective codebook usage that accounts
        for frequency, not just presence. It equals exp(entropy), giving the
        "effective number of codes" being used.
        """
        flat_codes = codes.flatten()

        # Count occurrences of each code
        counts = torch.bincount(flat_codes, minlength=self.codebook_size).float()

        # Convert to probabilities
        probs = counts / counts.sum()

        # Compute entropy: -sum(p * log(p)), handling zeros
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum()

        # Perplexity = exp(entropy)
        return torch.exp(entropy).item()  # type: ignore[no-any-return]


__all__ = [
    "VQRouterConfig",
    "VQRouter",
]
