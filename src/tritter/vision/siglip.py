"""SigLIP vision encoder for multimodal image understanding.

Implements SigLIP-B/16 (93M params) vision encoder with projection to model embedding
space. The encoder is frozen; only the projection layer trains.

Why: SmolVLM findings show compact LMs (135M-360M) don't benefit from large vision
encoders. SigLIP-B/16 outperforms SigLIP-SO-400M (428M) at this scale while using
~0.4 GB memory, fitting comfortably in the RTX 5080 16GB budget.

Embedding-Prediction Context: Images are encoded to continuous visual embeddings,
then projected to the shared embedding space where they can attend to text tokens.
This follows the Chameleon/LLaVA approach of early fusion multimodality.

Reference: docs/project-plan.md, SPEC-005-memory-optimization.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig


@dataclass
class SigLIPConfig:
    """Configuration for SigLIP vision encoder.

    Why: Centralized configuration enables easy experimentation with different
    encoder sizes while maintaining consistent interface. The B/16 variant
    (patch_size=16, hidden_size=768) balances quality and memory efficiency.

    Attributes:
        image_size: Input image resolution (square), default 384 (SigLIP-B standard)
        patch_size: ViT patch size, default 16 (B/16 variant)
        hidden_size: Encoder hidden dimension, default 768 (ViT-B)
        intermediate_size: MLP intermediate size, default 3072 (4x hidden)
        num_layers: Number of transformer layers, default 12 (ViT-B)
        num_heads: Number of attention heads, default 12 (ViT-B)
        num_channels: Input image channels, default 3 (RGB)
        layer_norm_eps: LayerNorm epsilon, default 1e-6
    """

    image_size: int = 384
    patch_size: int = 16
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        """Number of patches per image.

        Why: Determines sequence length for vision transformer.
        384/16 = 24 patches per side, 24*24 = 576 patches total.
        """
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_positions(self) -> int:
        """Total positions including CLS token.

        Why: SigLIP uses CLS token for pooled representation.
        """
        return self.num_patches + 1


class SigLIPEmbeddings(nn.Module):
    """Patch embedding for vision transformer.

    Why: Converts image into patch embeddings suitable for transformer processing.
    Uses Conv2d for efficient patch extraction (equivalent to linear projection
    of flattened patches but faster).
    """

    def __init__(self, config: SigLIPConfig) -> None:
        """Initialize patch embeddings.

        Args:
            config: SigLIP configuration
        """
        super().__init__()
        self.config = config

        # Patch embedding via Conv2d
        # Why: Conv2d with kernel_size=stride=patch_size extracts non-overlapping
        # patches and projects them in one operation. More efficient than
        # manual reshape + linear projection.
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )

        # Learnable CLS token
        # Why: CLS token aggregates global image representation for tasks
        # requiring single-vector output (classification, retrieval).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Learnable position embeddings
        # Why: Vision transformers need position information since attention
        # is permutation-invariant. Learned embeddings capture 2D structure.
        self.position_embedding = nn.Embedding(
            config.num_positions, config.hidden_size
        )

        # Register position IDs as buffer (not parameter)
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Embed image patches with position encoding.

        Args:
            pixel_values: Image tensor (B, C, H, W), normalized to [-1, 1] or [0, 1]

        Returns:
            Patch embeddings (B, num_positions, hidden_size)

        Why: Extracts patches via conv, adds CLS token, adds position embeddings.
        The CLS token is prepended and will hold the pooled representation.
        """
        batch_size = pixel_values.shape[0]

        # Extract patches: (B, hidden_size, H/patch, W/patch)
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions: (B, hidden_size, num_patches)
        patch_embeds = patch_embeds.flatten(2)

        # Transpose for transformer: (B, num_patches, hidden_size)
        patch_embeds = patch_embeds.transpose(1, 2)

        # Expand CLS token for batch: (B, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend CLS token: (B, num_patches + 1, hidden_size)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings  # (B, num_positions, hidden_size)


class SigLIPAttention(nn.Module):
    """Multi-head self-attention for vision transformer.

    Why: Standard transformer attention enables patches to attend to each other,
    learning spatial relationships and global context.
    """

    def __init__(self, config: SigLIPConfig) -> None:
        """Initialize attention module."""
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute multi-head self-attention.

        Args:
            hidden_states: Input tensor (B, L, D)

        Returns:
            Attention output (B, L, D)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)  # (B, L, D)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head: (B, num_heads, L, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Why: Use PyTorch's optimized SDPA which dispatches to FlashAttention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0
        )  # (B, num_heads, L, head_dim)

        # Reshape back: (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        return self.out_proj(attn_output)


class SigLIPMLP(nn.Module):
    """Feed-forward network for vision transformer.

    Why: Standard transformer FFN with GELU activation for non-linear transformation.
    """

    def __init__(self, config: SigLIPConfig) -> None:
        """Initialize MLP."""
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through MLP.

        Args:
            hidden_states: Input tensor (B, L, D)

        Returns:
            Output tensor (B, L, D)
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLIPEncoderLayer(nn.Module):
    """Single vision transformer encoder layer.

    Why: Standard Pre-LN transformer block with attention and MLP.
    """

    def __init__(self, config: SigLIPConfig) -> None:
        """Initialize encoder layer."""
        super().__init__()
        self.attention = SigLIPAttention(config)
        self.mlp = SigLIPMLP(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through encoder layer.

        Args:
            hidden_states: Input tensor (B, L, D)

        Returns:
            Output tensor (B, L, D)
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SigLIPEncoder(nn.Module):
    """Stack of vision transformer encoder layers.

    Why: Hierarchical feature extraction through stacked transformer layers.
    """

    def __init__(self, config: SigLIPConfig) -> None:
        """Initialize encoder stack."""
        super().__init__()
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(config) for _ in range(config.num_layers)
        ])

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through all encoder layers.

        Args:
            hidden_states: Input tensor (B, L, D)

        Returns:
            Output tensor (B, L, D)
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SigLIPVisionEncoder(nn.Module):
    """Complete SigLIP vision encoder with projection to model embedding space.

    Why: Encapsulates the full vision pipeline from pixels to model-compatible
    embeddings. The encoder is frozen (no gradients) while the projection layer
    trains, following LLaVA-style efficient adaptation.

    Embedding-Prediction Context: Visual features are projected to the same
    embedding space as text tokens, enabling cross-modal attention. The projection
    layer learns to align visual semantics with the model's embedding space.

    Attributes:
        config: SigLIPConfig for encoder architecture
        embeddings: Patch embedding layer
        encoder: Stack of transformer layers
        post_layernorm: Final layer normalization
        projection: Trainable projection to model hidden size
        freeze_encoder: Whether encoder weights are frozen

    Example:
        >>> siglip_config = SigLIPConfig()
        >>> encoder = SigLIPVisionEncoder(siglip_config, model_hidden_size=2048)
        >>> images = torch.randn(2, 3, 384, 384)  # (B, C, H, W)
        >>> visual_embeds = encoder(images)  # (B, 576, 2048) - 576 patches
    """

    def __init__(
        self,
        config: SigLIPConfig,
        model_hidden_size: int,
        freeze_encoder: bool = True,
    ) -> None:
        """Initialize SigLIP vision encoder.

        Args:
            config: SigLIPConfig with encoder architecture settings
            model_hidden_size: Target hidden size for projection (model's hidden_size)
            freeze_encoder: If True, freeze encoder weights (only train projection)

        Why: model_hidden_size enables projection from SigLIP's 768 dimensions to
        the LM's hidden size (e.g., 2048 for 3B model). Freezing encoder is memory
        efficient and follows proven LLaVA approach.
        """
        super().__init__()
        self.config = config
        self.model_hidden_size = model_hidden_size
        self.freeze_encoder = freeze_encoder

        # Vision transformer components
        self.embeddings = SigLIPEmbeddings(config)
        self.encoder = SigLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Projection to model embedding space
        # Why: SigLIP outputs 768-dim features but model may use different hidden_size.
        # This trainable projection aligns visual features with text embedding space.
        self.projection = nn.Linear(config.hidden_size, model_hidden_size)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters (not projection).

        Why: Freezing pretrained encoder saves memory (no gradients stored)
        and preserves learned visual representations. Only projection layer
        needs to adapt for the specific LM embedding space.
        """
        for module in [self.embeddings, self.encoder, self.post_layernorm]:
            for param in module.parameters():
                param.requires_grad = False

    def get_num_patches(self) -> int:
        """Get number of visual tokens (patches) per image.

        Returns:
            Number of patches (excluding CLS token)

        Why: Useful for computing total sequence length when mixing
        visual and text tokens.
        """
        return self.config.num_patches

    def forward(
        self,
        pixel_values: Tensor,
        return_cls: bool = False,
    ) -> Tensor:
        """Encode images to visual embeddings.

        Args:
            pixel_values: Image tensor (B, C, H, W), expected range [-1, 1] or [0, 1]
            return_cls: If True, return only CLS token embedding instead of all patches

        Returns:
            If return_cls=False: Visual embeddings (B, num_patches, model_hidden_size)
            If return_cls=True: CLS embedding (B, model_hidden_size)

        Why: Full patch embeddings enable fine-grained visual attention (model can
        attend to specific image regions). CLS token provides pooled representation
        for tasks needing single-vector image embedding.

        Embedding-Prediction Context: These visual embeddings will be concatenated
        with text embeddings in the unified sequence, enabling cross-modal attention
        where text tokens can attend to visual patches and vice versa.
        """
        # Patch embedding + positional encoding
        hidden_states = self.embeddings(pixel_values)  # (B, num_positions, 768)

        # Transformer encoder
        hidden_states = self.encoder(hidden_states)  # (B, num_positions, 768)

        # Final layer norm
        hidden_states = self.post_layernorm(hidden_states)  # (B, num_positions, 768)

        # Project to model hidden size
        hidden_states = self.projection(hidden_states)  # (B, num_positions, model_hidden_size)

        if return_cls:
            # Return only CLS token (first position)
            return hidden_states[:, 0, :]  # (B, model_hidden_size)
        else:
            # Return patch embeddings (exclude CLS token)
            return hidden_states[:, 1:, :]  # (B, num_patches, model_hidden_size)

    def get_memory_usage_gb(self) -> float:
        """Estimate memory usage of the encoder.

        Returns:
            Approximate memory in GB

        Why: Memory budgeting for RTX 5080 16GB constraint.
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Assume FP32 for non-quantized encoder (4 bytes per param)
        return total_params * 4 / 1e9


def create_siglip_encoder(
    tritter_config: TritterConfig,
    siglip_config: SigLIPConfig | None = None,
    freeze_encoder: bool = True,
) -> SigLIPVisionEncoder:
    """Factory function to create SigLIP encoder compatible with Tritter model.

    Args:
        tritter_config: Tritter model configuration (provides hidden_size)
        siglip_config: Optional custom SigLIP config (defaults to B/16)
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        Configured SigLIPVisionEncoder

    Why: Convenience factory ensures projection dimension matches Tritter model.
    """
    if siglip_config is None:
        siglip_config = SigLIPConfig()

    return SigLIPVisionEncoder(
        config=siglip_config,
        model_hidden_size=tritter_config.hidden_size,
        freeze_encoder=freeze_encoder,
    )


__all__ = [
    "SigLIPConfig",
    "SigLIPVisionEncoder",
    "create_siglip_encoder",
]
