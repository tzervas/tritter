"""VQ-VAE image tokenizer for discrete image token generation.

Implements VQ-VAE image tokenization with vector quantization for converting images
to discrete tokens. This provides an alternative to SigLIP patches, enabling a unified
vocabulary approach where images can be represented as discrete tokens alongside text.

Why: VQ-VAE provides aggressive compression (256-512 tokens per image) compared to
patch-based approaches, and produces discrete tokens that can be part of a unified
vocabulary with text tokens. This enables a more seamless multimodal experience where
the model treats images as sequences of discrete symbols.

Embedding-Prediction Context: Images are encoded to continuous latent features via
the convolutional encoder, then quantized to discrete codes via vector quantization.
For the Tritter model, these codes can either be embedded via the codebook (unified
vocabulary) or projected directly to the shared embedding space. This follows the
embedding-prediction paradigm where discrete codes are temporary scaffolding.

Reference: docs/project-plan.md (256-512 tokens per image target)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig


@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE image tokenizer.

    Why: Centralized configuration enables easy experimentation with different
    encoder architectures while maintaining consistent interface. The default
    settings target 256-512 tokens per image with aggressive spatial compression.

    Embedding-Prediction Context: The codebook_size determines the discrete token
    vocabulary for images. With 8192 entries, we have fine-grained control over
    image reconstruction quality while keeping tokens within manageable vocabulary
    size for joint text-image modeling.

    Attributes:
        image_size: Input image resolution (square), default 256 for efficient
            processing while maintaining reasonable quality.
        patch_size: Effective patch size after encoding, default 8 for 32x32 = 1024
            latent positions before quantization.
        in_channels: Number of input image channels, default 3 (RGB).
        hidden_size: Latent embedding dimension, default 256 for balanced
            capacity and efficiency.
        num_layers: Number of encoder/decoder convolutional layers, default 3.
        codebook_size: Number of entries in vector quantizer codebook, default 8192
            for sufficient coverage of visual concepts.
        num_codebooks: Number of codebooks, default 1 (single VQ, not RVQ like audio).
        commitment_cost: VQ loss weight for commitment loss, default 0.25.
            Balances reconstruction quality vs codebook utilization.
        layer_norm_eps: LayerNorm epsilon, default 1e-6.
        use_ema_updates: Whether to use EMA codebook updates, default False.
            EMA can improve codebook utilization but adds complexity.
        ema_decay: Decay rate for EMA updates, default 0.99.
    """

    image_size: int = 256
    patch_size: int = 8
    in_channels: int = 3
    hidden_size: int = 256
    num_layers: int = 3
    codebook_size: int = 8192
    num_codebooks: int = 1
    commitment_cost: float = 0.25
    layer_norm_eps: float = 1e-6
    use_ema_updates: bool = False
    ema_decay: float = 0.99

    @property
    def num_patches(self) -> int:
        """Number of spatial positions in latent representation.

        Why: Determines the base number of latent positions before any
        additional compression. With image_size=256 and patch_size=8,
        we get 32x32 = 1024 positions.
        """
        return (self.image_size // self.patch_size) ** 2

    @property
    def tokens_per_image(self) -> int:
        """Number of discrete tokens per image.

        Why: For single codebook VQ-VAE, each spatial position maps to one token.
        This is the key metric for sequence length planning in multimodal models.
        Target: 256-512 tokens per image for aggressive compression.
        """
        return self.num_patches * self.num_codebooks

    @property
    def latent_size(self) -> int:
        """Spatial size of latent representation.

        Why: Useful for reshaping operations in encoder/decoder.
        """
        return self.image_size // self.patch_size


class ResidualBlock2D(nn.Module):
    """Residual block with 2D convolutions for image processing.

    Why: Residual connections enable deep networks without vanishing gradients.
    GroupNorm is used instead of BatchNorm for stability with small batch sizes
    common in multimodal training.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
    ) -> None:
        """Initialize residual block.

        Args:
            channels: Number of input/output channels
            kernel_size: Convolution kernel size, default 3
            num_groups: Number of groups for GroupNorm, default 32

        Why: GroupNorm with 32 groups works well across different hidden sizes.
        """
        super().__init__()

        # Ensure num_groups doesn't exceed channels
        actual_groups = min(num_groups, channels)

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(actual_groups, channels)
        self.norm2 = nn.GroupNorm(actual_groups, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        residual = x
        x = self.norm1(x)
        x = F.silu(x)  # SiLU works well for image tasks
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class VQVAEEncoder(nn.Module):
    """2D convolutional encoder for images.

    Why: Converts RGB images to continuous latent embeddings via progressive
    downsampling. Uses 2D convolutions (not 1D like audio) for spatial structure.
    The encoder produces a spatial grid of embeddings ready for vector quantization.

    Embedding-Prediction Context: The encoder produces continuous embeddings
    that will be quantized by VQ. These embeddings capture visual semantics
    in a compressed spatial representation.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        """Initialize encoder.

        Args:
            config: VQVAEConfig with architecture settings

        Why: Progressive downsampling via strided convolutions compresses spatial
        dimensions while expanding channel dimensions. Each layer doubles channels
        and halves spatial size (up to hidden_size limit).
        """
        super().__init__()
        self.config = config

        # Calculate number of downsample steps needed
        # image_size -> latent_size requires log2(image_size/latent_size) steps
        num_downsamples = 0
        size = config.image_size
        while size > config.latent_size:
            size //= 2
            num_downsamples += 1

        # Initial projection from RGB to hidden channels
        self.input_conv = nn.Conv2d(
            config.in_channels,
            config.hidden_size // (2 ** min(num_downsamples, config.num_layers)),
            kernel_size=3,
            padding=1,
        )

        # Encoder blocks with progressive downsampling
        self.encoder_blocks = nn.ModuleList()
        channels = config.hidden_size // (2 ** min(num_downsamples, config.num_layers))

        for _ in range(min(num_downsamples, config.num_layers)):
            out_channels = min(channels * 2, config.hidden_size)
            block = nn.Sequential(
                ResidualBlock2D(channels),
                nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
            )
            self.encoder_blocks.append(block)
            channels = out_channels

        # Additional residual blocks at final resolution
        self.residual_blocks = nn.ModuleList([
            ResidualBlock2D(channels)
            for _ in range(config.num_layers - min(num_downsamples, config.num_layers))
        ])

        # Final projection to hidden_size
        self.output_conv = nn.Conv2d(channels, config.hidden_size, kernel_size=1)

        self._output_channels = channels

    def forward(self, images: Tensor) -> Tensor:
        """Encode images to latent embeddings.

        Args:
            images: Image tensor (B, C, H, W) where C=in_channels, H=W=image_size

        Returns:
            Latent embeddings (B, hidden_size, H', W') where H'=W'=latent_size

        Why: Progressive downsampling reduces spatial dimension while residual
        blocks at final resolution refine features before quantization.
        """
        # Initial projection: (B, 3, H, W) -> (B, C, H, W)
        x = self.input_conv(images)

        # Progressive downsampling through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)  # (B, C, H, W) -> (B, 2C, H/2, W/2)

        # Additional residual blocks at final resolution
        for block in self.residual_blocks:
            x = block(x)

        # Final projection: (B, C, H', W') -> (B, hidden_size, H', W')
        embeddings: Tensor = self.output_conv(x)

        return embeddings


class VQVAEDecoder(nn.Module):
    """2D convolutional decoder for image reconstruction.

    Why: Mirror of encoder, reconstructs images from latent embeddings.
    Used for training the encoder via reconstruction loss and for image
    generation at inference time.

    Embedding-Prediction Context: Converts discrete codes back to continuous
    embeddings via codebook lookup, then upsamples to reconstruct the original
    image. Enables round-trip image -> tokens -> image for image generation.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        """Initialize decoder.

        Args:
            config: VQVAEConfig with architecture settings

        Why: Uses transposed convolutions for upsampling, mirroring the encoder's
        downsampling structure.
        """
        super().__init__()
        self.config = config

        # Calculate number of upsample steps needed
        num_upsamples = 0
        size = config.latent_size
        while size < config.image_size:
            size *= 2
            num_upsamples += 1

        # Initial projection from hidden_size
        self.input_conv = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=1)

        # Residual blocks at latent resolution
        self.residual_blocks = nn.ModuleList([
            ResidualBlock2D(config.hidden_size)
            for _ in range(config.num_layers - min(num_upsamples, config.num_layers))
        ])

        # Decoder blocks with progressive upsampling
        self.decoder_blocks = nn.ModuleList()
        channels = config.hidden_size

        for _ in range(min(num_upsamples, config.num_layers)):
            out_channels = max(channels // 2, config.hidden_size // (2 ** config.num_layers))
            block = nn.Sequential(
                ResidualBlock2D(channels),
                nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
            )
            self.decoder_blocks.append(block)
            channels = out_channels

        # Final projection to RGB
        self.output_conv = nn.Conv2d(channels, config.in_channels, kernel_size=3, padding=1)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Decode latent embeddings to images.

        Args:
            embeddings: Latent tensor (B, hidden_size, H', W')

        Returns:
            Reconstructed images (B, in_channels, H, W) approximately

        Why: Residual blocks refine latent features, then progressive upsampling
        reconstructs spatial resolution.
        """
        # Initial projection
        x = self.input_conv(embeddings)

        # Residual blocks at latent resolution
        for block in self.residual_blocks:
            x = block(x)

        # Progressive upsampling through decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Final projection to RGB: (B, C, H, W) -> (B, 3, H, W)
        images: Tensor = self.output_conv(x)

        return images


class VectorQuantizer(nn.Module):
    """Vector quantizer for discrete image tokenization.

    Why: VQ enables discrete representation of continuous embeddings. Unlike RVQ
    (used for audio), single-codebook VQ is simpler and sufficient for images
    where we want direct token-to-embedding mapping for unified vocabulary.

    Embedding-Prediction Context: VQ converts continuous encoder embeddings to
    discrete codes. For the embedding-prediction paradigm, these codes will be
    embedded via the codebook for language modeling. The straight-through estimator
    enables gradient flow through the discrete bottleneck.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        """Initialize vector quantizer.

        Args:
            config: VQVAEConfig with codebook settings

        Why: Single codebook with straight-through estimator provides simple
        and effective discrete bottleneck for images.
        """
        super().__init__()
        self.config = config
        self.codebook_size = config.codebook_size
        self.hidden_size = config.hidden_size
        self.commitment_cost = config.commitment_cost
        self.use_ema_updates = config.use_ema_updates
        self.ema_decay = config.ema_decay

        # Codebook embedding
        self.codebook = nn.Embedding(config.codebook_size, config.hidden_size)

        # Initialize codebook uniformly
        nn.init.uniform_(
            self.codebook.weight,
            -1 / config.codebook_size,
            1 / config.codebook_size,
        )

        # EMA tracking (if enabled)
        if self.use_ema_updates:
            self.register_buffer("ema_cluster_size", torch.zeros(config.codebook_size))
            self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize continuous embeddings to discrete codes.

        Args:
            z: Continuous embeddings (B, C, H, W) or (B, L, C)

        Returns:
            Tuple of:
                - codes: Quantized indices (B, H, W) or (B, L)
                - quantized: Quantized embeddings (same shape as input)
                - vq_loss: VQ loss (codebook + commitment)

        Why: Finds nearest codebook entry for each embedding position.
        Straight-through estimator copies gradients from quantized to input.
        Commitment loss encourages encoder to commit to codebook entries.

        Embedding-Prediction Context: Codes are discrete tokens suitable for
        language modeling. Quantized embeddings are continuous representations
        that can be projected to model embedding space or decoded to images.
        """
        # Handle both 4D (B, C, H, W) and 3D (B, L, C) inputs
        is_4d = z.dim() == 4
        if is_4d:
            B, C, H, W = z.shape
            # Reshape to (B, H*W, C) for distance computation
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        else:
            B, L, C = z.shape
            z_flat = z

        # Compute distances to codebook entries
        # z_flat: (B, N, C), codebook: (K, C)
        # distances: (B, N, K)
        distances = torch.cdist(z_flat, self.codebook.weight)

        # Find nearest codebook entry
        codes = distances.argmin(dim=-1)  # (B, N)

        # Get quantized embeddings
        quantized = self.codebook(codes)  # (B, N, C)

        # Compute losses
        # Codebook loss: move codebook towards encoder outputs
        codebook_loss = F.mse_loss(quantized, z_flat.detach())

        # Commitment loss: move encoder outputs towards codebook
        commitment_loss = F.mse_loss(z_flat, quantized.detach())

        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from quantized to z
        # Why: Enables gradient flow through discrete bottleneck
        quantized = z_flat + (quantized - z_flat).detach()

        # EMA codebook update (if enabled and training)
        if self.use_ema_updates and self.training:
            self._update_ema(z_flat, codes)

        # Reshape back to original format
        if is_4d:
            codes = codes.view(B, H, W)
            quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            # Already in correct shape (B, L, C)
            pass

        return codes, quantized, vq_loss

    def _update_ema(self, z_flat: Tensor, codes: Tensor) -> None:
        """Update codebook using exponential moving average.

        Args:
            z_flat: Flattened encoder outputs (B, N, C)
            codes: Quantized codes (B, N)

        Why: EMA updates can improve codebook utilization by smoothly
        moving codebook entries towards frequently used clusters.
        """
        # Flatten batch and sequence dimensions
        flat_codes = codes.view(-1)  # (B*N,)
        flat_z = z_flat.view(-1, self.hidden_size)  # (B*N, C)

        # Count codes
        one_hot = F.one_hot(flat_codes, self.codebook_size).float()  # (B*N, K)

        # Update cluster sizes
        cluster_size = one_hot.sum(0)  # (K,)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )

        # Update embedding sums
        embed_sum = one_hot.t() @ flat_z  # (K, C)
        self.ema_embed_sum.data.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Update codebook (Laplace smoothing to avoid division by zero)
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        )
        self.codebook.weight.data.copy_(
            self.ema_embed_sum / cluster_size_smoothed.unsqueeze(1)
        )

    def dequantize(self, codes: Tensor) -> Tensor:
        """Convert discrete codes back to continuous embeddings.

        Args:
            codes: Quantized indices (B, H, W) or (B, L)

        Returns:
            Quantized embeddings (B, C, H, W) or (B, L, C)

        Why: Reconstructs continuous representation from discrete codes
        for decoder input or embedding projection.
        """
        is_3d = codes.dim() == 3

        if is_3d:
            B, H, W = codes.shape
            codes_flat = codes.view(B, -1)  # (B, H*W)
        else:
            codes_flat = codes

        # Look up embeddings
        quantized = self.codebook(codes_flat)  # (B, N, C)

        # Reshape if needed
        if is_3d:
            quantized = quantized.view(B, H, W, self.hidden_size)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return quantized

    def get_codebook_usage(self, codes: Tensor) -> float:
        """Analyze codebook usage statistics.

        Args:
            codes: Quantized indices (any shape)

        Returns:
            Fraction of codebook entries used

        Why: Low usage indicates codebook collapse, a training issue.
        """
        flat_codes = codes.view(-1)
        unique_codes = torch.unique(flat_codes).numel()
        return unique_codes / self.codebook_size


class VQVAEImageTokenizer(nn.Module):
    """Complete VQ-VAE image tokenizer combining encoder, VQ, and decoder.

    Why: Main interface for image tokenization. Combines convolutional encoder
    with vector quantization to convert images to discrete tokens. Provides
    an alternative to SigLIP patches for unified vocabulary multimodal models.

    Embedding-Prediction Context: Images are encoded to continuous embeddings,
    then quantized to discrete codes. The codes can be embedded via the codebook
    for unified vocabulary language modeling, or the quantized embeddings can be
    projected directly to model embedding space for early fusion multimodal attention.

    Attributes:
        config: VQVAEConfig for encoder architecture
        encoder: Convolutional encoder stack
        quantizer: Vector quantizer
        decoder: Convolutional decoder for reconstruction
        freeze_encoder: Whether encoder weights are frozen

    Example:
        >>> config = VQVAEConfig()
        >>> tokenizer = VQVAEImageTokenizer(config)
        >>> images = torch.randn(2, 3, 256, 256)  # (B, C, H, W)
        >>> codes = tokenizer.encode(images)  # (2, 32, 32) - 1024 tokens
        >>> reconstructed = tokenizer.decode(codes)  # (2, 3, 256, 256)
    """

    def __init__(
        self,
        config: VQVAEConfig,
        freeze_encoder: bool = False,
    ) -> None:
        """Initialize VQ-VAE image tokenizer.

        Args:
            config: VQVAEConfig with encoder architecture settings
            freeze_encoder: If True, freeze encoder weights

        Why: Unlike SigLIP/EnCodec, VQ-VAE is typically trained from scratch
        jointly with the LM, so freeze_encoder defaults to False.
        """
        super().__init__()
        self.config = config
        self.freeze_encoder = freeze_encoder

        # Image encoder
        self.encoder = VQVAEEncoder(config)

        # Vector quantizer
        self.quantizer = VectorQuantizer(config)

        # Image decoder for reconstruction
        self.decoder = VQVAEDecoder(config)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters.

        Why: Freezing pretrained encoder saves memory (no gradients stored)
        and preserves learned visual representations.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, images: Tensor) -> Tensor:
        """Encode images to discrete codes.

        Args:
            images: Image tensor (B, C, H, W) where C=in_channels, H=W=image_size

        Returns:
            Codes tensor (B, H', W') where H'=W'=latent_size

        Why: Main encoding interface. Returns discrete codes suitable for
        language modeling or storage.
        """
        # Encode to continuous embeddings
        embeddings = self.encoder(images)  # (B, hidden_size, H', W')

        # Quantize to discrete codes
        codes, _, _ = self.quantizer.quantize(embeddings)  # (B, H', W')

        return codes

    def decode(self, codes: Tensor) -> Tensor:
        """Decode discrete codes to images.

        Args:
            codes: Discrete codes (B, H', W')

        Returns:
            Reconstructed images (B, in_channels, H, W) approximately

        Why: Enables image reconstruction from tokens for generation.
        """
        # Dequantize to continuous embeddings
        embeddings = self.quantizer.dequantize(codes)  # (B, hidden_size, H', W')

        # Decode to images
        images: Tensor = self.decoder(embeddings)  # (B, in_channels, ~H, ~W)

        return images

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass with reconstruction.

        Args:
            images: Image tensor (B, C, H, W)

        Returns:
            Tuple of:
                - codes: Discrete codes (B, H', W')
                - quantized: Quantized embeddings (B, hidden_size, H', W')
                - reconstructed: Reconstructed images (B, C, ~H, ~W)
                - vq_loss: VQ loss for training

        Why: Full forward needed for training with reconstruction loss.
        """
        # Encode
        embeddings = self.encoder(images)

        # Quantize
        codes, quantized, vq_loss = self.quantizer.quantize(embeddings)

        # Decode
        reconstructed = self.decoder(quantized)

        return codes, quantized, reconstructed, vq_loss

    def tokenize(self, images: Tensor) -> list[int]:
        """Tokenize images to flat token sequence.

        Args:
            images: Image tensor (1, C, H, W) - single image

        Returns:
            List of token IDs (flattened spatial grid)

        Why: Flattened sequence is compatible with standard tokenizer interface.
        Tokens are ordered row-major: [r0c0, r0c1, ..., r0cW, r1c0, r1c1, ...]

        Embedding-Prediction Context: This flat token sequence can be used with
        standard language model training. Each position maps to a codebook entry.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

        codes = self.encode(images)  # (1, H', W')
        codes = codes[0]  # (H', W')

        # Flatten to list
        tokens = codes.flatten().tolist()
        return [int(t) for t in tokens]

    def detokenize(self, tokens: list[int], height: int | None = None, width: int | None = None) -> Tensor:
        """Convert flat token sequence back to spatial codes.

        Args:
            tokens: List of token IDs
            height: Spatial height (defaults to latent_size)
            width: Spatial width (defaults to latent_size)

        Returns:
            Codes tensor (1, H', W')

        Why: Inverse of tokenize() for reshaping flat tokens to spatial grid.
        """
        if height is None:
            height = self.config.latent_size
        if width is None:
            width = self.config.latent_size

        expected_len = height * width
        if len(tokens) != expected_len:
            raise ValueError(
                f"Token count {len(tokens)} != height * width ({height} * {width} = {expected_len})"
            )

        codes = torch.tensor(tokens, dtype=torch.long).view(1, height, width)
        return codes

    def get_tokens_per_image(self) -> int:
        """Get number of tokens per image.

        Returns:
            Number of discrete tokens per image

        Why: Useful for computing sequence length given number of images.
        """
        return self.config.tokens_per_image

    def get_codebook_embeddings(self) -> Tensor:
        """Get codebook embeddings for unified vocabulary.

        Returns:
            Codebook embeddings (codebook_size, hidden_size)

        Why: Enables direct embedding lookup for unified vocabulary
        where image tokens share embedding space with text tokens.
        """
        return self.quantizer.codebook.weight

    def get_memory_usage_gb(self) -> float:
        """Estimate memory usage of the tokenizer.

        Returns:
            Approximate memory in GB

        Why: Memory budgeting for RTX 5080 16GB constraint.
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Assume FP32 (4 bytes per param)
        return total_params * 4 / 1e9


class ImageProjection(nn.Module):
    """Project VQ-VAE embeddings to model embedding space.

    Why: VQ-VAE hidden_size (256) may differ from model hidden_size (e.g., 2048).
    This trainable projection aligns visual features with text embedding space,
    enabling cross-modal attention.

    Embedding-Prediction Context: Projects quantized VQ-VAE embeddings to the
    shared embedding space where they can attend to text tokens. Similar to
    SigLIP's projection layer.
    """

    def __init__(self, vqvae_hidden_size: int, model_hidden_size: int) -> None:
        """Initialize projection.

        Args:
            vqvae_hidden_size: VQ-VAE embedding dimension (e.g., 256)
            model_hidden_size: Target model dimension (e.g., 2048)

        Why: Linear projection is simple and effective. Can be extended with
        non-linearity or multi-layer projection if needed.
        """
        super().__init__()
        self.proj = nn.Linear(vqvae_hidden_size, model_hidden_size)

    def forward(self, image_embeddings: Tensor) -> Tensor:
        """Project VQ-VAE embeddings.

        Args:
            image_embeddings: Tensor (B, L, vqvae_hidden_size) or (B, C, H, W)

        Returns:
            Projected tensor (B, L, model_hidden_size) or (B, model_hidden_size, H, W)
        """
        is_4d = image_embeddings.dim() == 4

        if is_4d:
            B, C, H, W = image_embeddings.shape
            # Reshape to (B, H*W, C) for projection
            x = image_embeddings.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            x = self.proj(x)
            # Reshape back to (B, C', H, W)
            result: Tensor = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        else:
            result = self.proj(image_embeddings)

        return result


def create_vqvae_tokenizer(
    tritter_config: TritterConfig,
    vqvae_config: VQVAEConfig | None = None,
    freeze_encoder: bool = False,
) -> tuple[VQVAEImageTokenizer, ImageProjection]:
    """Factory function to create VQ-VAE tokenizer compatible with Tritter model.

    Args:
        tritter_config: Tritter model configuration (provides hidden_size)
        vqvae_config: Optional custom VQ-VAE config (defaults to standard settings)
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        Tuple of (VQVAEImageTokenizer, ImageProjection)

    Why: Convenience factory ensures projection dimension matches Tritter model.
    Returns both tokenizer and projection as separate modules for flexibility
    (e.g., tokenizer can be used standalone for image-only tasks).

    Embedding-Prediction Context: The VQ-VAE tokenizer converts images to
    discrete codes, while the projection maps the underlying embeddings to
    model space for multimodal fusion.
    """
    if vqvae_config is None:
        vqvae_config = VQVAEConfig()

    tokenizer = VQVAEImageTokenizer(
        config=vqvae_config,
        freeze_encoder=freeze_encoder,
    )

    projection = ImageProjection(
        vqvae_hidden_size=vqvae_config.hidden_size,
        model_hidden_size=tritter_config.hidden_size,
    )

    return tokenizer, projection


__all__ = [
    "VQVAEConfig",
    "VQVAEEncoder",
    "VQVAEDecoder",
    "VectorQuantizer",
    "VQVAEImageTokenizer",
    "ImageProjection",
    "create_vqvae_tokenizer",
]
