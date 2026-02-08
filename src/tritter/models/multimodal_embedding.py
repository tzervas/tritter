"""Unified multimodal embedding space for text, vision, and audio.

Integrates text embeddings, vision encoders (SigLIP + VQ-VAE), and audio encoder
(EnCodec) into a shared embedding space. Enables early fusion multimodal attention
where tokens from all modalities can attend to each other.

Why: Chameleon-style early fusion allows the model to learn cross-modal relationships
at every layer, rather than just at the final projection. This unified space enables
any-to-any generation (text-to-image, image-to-text, audio-to-text, etc.) by treating
all modalities as sequences in the same embedding space.

Embedding-Prediction Context: All modality encoders produce continuous embeddings
that are projected into the shared model embedding space. Text uses standard token
embeddings, while vision and audio use frozen pretrained encoders with trainable
projections. The discrete tokens from these encoders are temporary scaffolding for
training compatibility - the core computation happens in continuous embedding space.

Reference: docs/project-plan.md (early fusion architecture), SPEC-005 (memory budget)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig

from tritter.audio.encodec import (
    AudioProjection,
    EnCodecAudioTokenizer,
    EnCodecConfig,
)
from tritter.vision.siglip import (
    SigLIPConfig,
    SigLIPVisionEncoder,
)
from tritter.vision.vqvae import (
    ImageProjection,
    VQVAEConfig,
    VQVAEImageTokenizer,
)


class ModalityType(IntEnum):
    """Enum for identifying different modality types in the sequence.

    Why: Integer enum enables efficient masking operations and easy conversion
    to tensor indices. Values are contiguous starting from 0 for direct indexing.

    Embedding-Prediction Context: Modality type tracking enables the model to
    apply modality-specific processing (e.g., different positional encodings)
    while still operating in the unified embedding space.
    """

    TEXT = 0
    IMAGE = 1
    AUDIO = 2


@dataclass
class MultimodalBatch:
    """Container for multimodal input data.

    Why: Encapsulates all possible input modalities in a single structure,
    making it easy to pass multimodal data through the model. Optional fields
    allow flexible combinations (text-only, image-only, text+image, etc.).

    Embedding-Prediction Context: Each modality field contains raw input data
    that will be converted to continuous embeddings. The modality_mask tracks
    which positions in the combined sequence correspond to which modality.

    Attributes:
        input_ids: Text token IDs (B, L_text) or None if no text input.
        pixel_values: Image pixel values (B, C, H, W) or batched images
            (B, N_images, C, H, W) or None if no image input.
        audio_waveforms: Audio waveform (B, 1, T) or (B, T) or None if no audio.
        modality_mask: Tensor (B, L_total) indicating modality type at each position.
            Values correspond to ModalityType enum. Created during forward pass
            if not provided.
        attention_mask: Optional attention mask (B, L_total) for padding. 1 = attend,
            0 = mask out.
    """

    input_ids: Tensor | None = None
    pixel_values: Tensor | None = None
    audio_waveforms: Tensor | None = None
    modality_mask: Tensor | None = None
    attention_mask: Tensor | None = None


@dataclass
class MultimodalEmbeddingConfig:
    """Configuration for the multimodal embedding layer.

    Why: Centralizes all encoder configurations and projection settings.
    Enables easy switching between vision encoder types (SigLIP vs VQ-VAE)
    and allows disabling modalities that aren't needed.

    Embedding-Prediction Context: Configures how each modality's raw input
    is converted to the shared embedding space. The hidden_size must match
    the main model's hidden_size to enable seamless integration.

    Attributes:
        hidden_size: Target embedding dimension (must match model hidden_size).
        vocab_size: Text vocabulary size for embedding table.
        max_seq_len: Maximum combined sequence length across all modalities.
        vision_encoder: Which vision encoder to use ("siglip", "vqvae", "none").
        audio_encoder: Which audio encoder to use ("encodec", "none").
        siglip_config: Configuration for SigLIP encoder (if vision_encoder="siglip").
        vqvae_config: Configuration for VQ-VAE encoder (if vision_encoder="vqvae").
        encodec_config: Configuration for EnCodec encoder (if audio_encoder="encodec").
        freeze_vision_encoder: Whether to freeze vision encoder weights.
        freeze_audio_encoder: Whether to freeze audio encoder weights.
        dropout: Dropout probability for embedding output.
    """

    hidden_size: int
    vocab_size: int
    max_seq_len: int = 131072  # 128K context
    vision_encoder: Literal["siglip", "vqvae", "none"] = "siglip"
    audio_encoder: Literal["encodec", "none"] = "encodec"
    siglip_config: SigLIPConfig | None = None
    vqvae_config: VQVAEConfig | None = None
    encodec_config: EnCodecConfig | None = None
    freeze_vision_encoder: bool = True
    freeze_audio_encoder: bool = True
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration and set defaults.

        Why: Ensures encoder configs are created if the encoder is enabled
        but no config was provided. Also validates that enabled encoders
        have valid configurations.
        """
        # Set default configs for enabled encoders
        if self.vision_encoder == "siglip" and self.siglip_config is None:
            self.siglip_config = SigLIPConfig()
        elif self.vision_encoder == "vqvae" and self.vqvae_config is None:
            self.vqvae_config = VQVAEConfig()

        if self.audio_encoder == "encodec" and self.encodec_config is None:
            self.encodec_config = EnCodecConfig()

        # Validate encoder configs match selection
        if self.vision_encoder == "siglip":
            assert self.siglip_config is not None, (
                "siglip_config required when vision_encoder='siglip'"
            )
        if self.vision_encoder == "vqvae":
            assert self.vqvae_config is not None, (
                "vqvae_config required when vision_encoder='vqvae'"
            )
        if self.audio_encoder == "encodec":
            assert self.encodec_config is not None, (
                "encodec_config required when audio_encoder='encodec'"
            )


@dataclass
class MultimodalOutput:
    """Output from the multimodal embedding layer.

    Why: Bundles the embedded sequence with metadata needed for downstream
    processing (attention masks, modality positions).

    Embedding-Prediction Context: The embeddings tensor contains continuous
    representations in the unified space. The modality_mask enables
    modality-aware processing in subsequent layers if needed.

    Attributes:
        embeddings: Combined embeddings (B, L_total, hidden_size).
        modality_mask: Modality type at each position (B, L_total).
        attention_mask: Attention mask for padding (B, L_total). 1 = attend.
        modality_positions: Dict mapping modality type to (start, end) indices.
    """

    embeddings: Tensor
    modality_mask: Tensor
    attention_mask: Tensor
    modality_positions: dict[str, tuple[int, int]] = field(default_factory=dict)


class MultimodalEmbedding(nn.Module):  # type: ignore[misc]
    """Unified multimodal embedding layer for text, vision, and audio.

    Why: Early fusion multimodal architecture requires all modalities to share
    the same embedding space. This layer handles the conversion of raw inputs
    (tokens, pixels, waveforms) to continuous embeddings that can be concatenated
    and processed by the transformer.

    Embedding-Prediction Context: This layer is the entry point where discrete
    inputs become continuous embeddings. Text tokens are embedded via a standard
    embedding table. Vision and audio inputs pass through frozen pretrained
    encoders (SigLIP/VQ-VAE for vision, EnCodec for audio) followed by trainable
    linear projections that align them to the shared space.

    The output embeddings form a unified sequence:
    [TEXT_TOKENS] [IMAGE_PATCHES/TOKENS] [AUDIO_FRAMES]

    Or for interleaved multimodal (in-context learning):
    [TEXT: "Describe this:"] [IMAGE_PATCHES] [TEXT: "It shows..."]

    Attributes:
        config: MultimodalEmbeddingConfig with architecture settings.
        text_embedding: nn.Embedding for text token embeddings.
        vision_encoder: SigLIPVisionEncoder or VQVAEImageTokenizer or None.
        audio_encoder: EnCodecAudioTokenizer or None.
        vision_projection: Linear projection for vision features (if needed).
        audio_projection: Linear projection for audio features.
        dropout: Dropout layer for regularization.

    Example:
        >>> config = MultimodalEmbeddingConfig(
        ...     hidden_size=2048,
        ...     vocab_size=65536,
        ...     vision_encoder="siglip",
        ...     audio_encoder="encodec",
        ... )
        >>> embedding = MultimodalEmbedding(config)
        >>> batch = MultimodalBatch(
        ...     input_ids=torch.randint(0, 1000, (2, 32)),  # (B, L_text)
        ...     pixel_values=torch.randn(2, 3, 384, 384),   # (B, C, H, W)
        ...     audio_waveforms=torch.randn(2, 1, 24000),   # (B, 1, T)
        ... )
        >>> output = embedding(batch)
        >>> output.embeddings.shape  # (2, 32 + 576 + 75, 2048)
    """

    def __init__(self, config: MultimodalEmbeddingConfig) -> None:
        """Initialize multimodal embedding layer.

        Args:
            config: MultimodalEmbeddingConfig with encoder settings.

        Why: Initializes encoders based on config settings. Vision and audio
        encoders are optionally frozen (no gradient updates) to preserve
        pretrained representations. Only the projection layers train, following
        the efficient LLaVA-style adaptation approach.
        """
        super().__init__()
        self.config = config

        # Text embedding
        # Why: Standard embedding table for text tokens. Size = vocab_size x hidden_size.
        self.text_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        # Vision encoder initialization
        self._init_vision_encoder()

        # Audio encoder initialization
        self._init_audio_encoder()

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def _init_vision_encoder(self) -> None:
        """Initialize vision encoder based on config.

        Why: Supports two vision encoding approaches:
        1. SigLIP: Patch-based encoding producing 576 tokens per image (384x384, patch=16)
        2. VQ-VAE: Discrete tokenization producing 256-1024 tokens per image

        Both approaches project to the shared hidden_size.
        """
        self.vision_encoder: SigLIPVisionEncoder | VQVAEImageTokenizer | None = None
        self.vision_projection: nn.Linear | ImageProjection | None = None

        if self.config.vision_encoder == "siglip":
            assert self.config.siglip_config is not None
            self.vision_encoder = SigLIPVisionEncoder(
                config=self.config.siglip_config,
                model_hidden_size=self.config.hidden_size,
                freeze_encoder=self.config.freeze_vision_encoder,
            )
            # SigLIP already includes projection to hidden_size
            self.vision_projection = None

        elif self.config.vision_encoder == "vqvae":
            assert self.config.vqvae_config is not None
            self.vision_encoder = VQVAEImageTokenizer(
                config=self.config.vqvae_config,
                freeze_encoder=self.config.freeze_vision_encoder,
            )
            # VQ-VAE needs projection from its hidden_size to model hidden_size
            self.vision_projection = ImageProjection(
                vqvae_hidden_size=self.config.vqvae_config.hidden_size,
                model_hidden_size=self.config.hidden_size,
            )

    def _init_audio_encoder(self) -> None:
        """Initialize audio encoder based on config.

        Why: EnCodec provides high-fidelity audio tokenization at 75 tokens/second.
        The encoder is frozen while the projection learns to align audio features
        with the text embedding space.
        """
        self.audio_encoder: EnCodecAudioTokenizer | None = None
        self.audio_projection: AudioProjection | None = None

        if self.config.audio_encoder == "encodec":
            assert self.config.encodec_config is not None
            self.audio_encoder = EnCodecAudioTokenizer(
                config=self.config.encodec_config,
                freeze_encoder=self.config.freeze_audio_encoder,
            )
            self.audio_projection = AudioProjection(
                encodec_hidden_size=self.config.encodec_config.hidden_size,
                model_hidden_size=self.config.hidden_size,
            )

    def embed_text(self, token_ids: Tensor) -> Tensor:
        """Embed text tokens to continuous embeddings.

        Args:
            token_ids: Text token IDs (B, L_text)

        Returns:
            Text embeddings (B, L_text, hidden_size)

        Why: Standard embedding lookup for text tokens. This is the simplest
        modality - direct lookup in the embedding table.

        Embedding-Prediction Context: Text tokens are the primary scaffolding
        for training. In the embedding-prediction paradigm, these embeddings
        will eventually be predicted directly rather than through discrete tokens.
        """
        embeddings: Tensor = self.text_embedding(token_ids)  # (B, L, D)
        return embeddings

    def embed_images(self, pixel_values: Tensor) -> Tensor:
        """Embed images to continuous embeddings.

        Args:
            pixel_values: Image tensor (B, C, H, W) or (B, N_images, C, H, W)

        Returns:
            Image embeddings (B, num_patches, hidden_size) or
            (B, N_images * num_patches, hidden_size) for batched images

        Why: Converts pixel values to embeddings via the configured vision encoder.
        SigLIP produces patch embeddings directly, while VQ-VAE quantizes to codes
        then projects the quantized embeddings.

        Embedding-Prediction Context: Visual features are projected to the same
        space as text embeddings, enabling cross-modal attention. The frozen
        encoder preserves learned visual representations while the projection
        adapts them to this specific model's embedding space.

        Raises:
            ValueError: If no vision encoder is configured.
        """
        if self.vision_encoder is None:
            raise ValueError(
                "No vision encoder configured. Set vision_encoder to 'siglip' or 'vqvae' "
                f"in MultimodalEmbeddingConfig. Current setting: {self.config.vision_encoder}"
            )

        # Handle batched images (B, N, C, H, W)
        batched = pixel_values.dim() == 5
        if batched:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * N, C, H, W)

        if isinstance(self.vision_encoder, SigLIPVisionEncoder):
            # SigLIP outputs (B, num_patches, hidden_size) directly
            embeddings = self.vision_encoder(pixel_values)  # (B*N, num_patches, D)
        else:
            # VQ-VAE: encode to get embeddings, then project
            # Forward gives (codes, quantized, reconstructed, loss)
            _, quantized, _, _ = self.vision_encoder(pixel_values)
            # quantized is (B, hidden_vq, H', W'), reshape for projection
            B_enc, C_enc, H_enc, W_enc = quantized.shape
            quantized = quantized.permute(0, 2, 3, 1).contiguous()  # (B, H', W', C)
            quantized = quantized.view(B_enc, H_enc * W_enc, C_enc)  # (B, L, C)
            embeddings = self.vision_projection(quantized)  # type: ignore[misc] # (B, L, D)

        # Reshape if batched
        if batched:
            num_patches = embeddings.shape[1]
            embeddings = embeddings.reshape(B, N * num_patches, -1)

        result: Tensor = embeddings
        return result

    def embed_audio(self, waveforms: Tensor) -> Tensor:
        """Embed audio waveforms to continuous embeddings.

        Args:
            waveforms: Audio tensor (B, 1, T) or (B, T) where T = num_samples

        Returns:
            Audio embeddings (B, num_frames, hidden_size) where
            num_frames = T // downsample_rate

        Why: Converts raw audio waveform to embeddings via EnCodec encoder.
        The encoder compresses audio at 75 tokens/second, making audio sequences
        manageable for transformer processing.

        Embedding-Prediction Context: Audio features are projected to the same
        space as text and image embeddings. The frozen encoder preserves learned
        audio representations while the projection adapts them to this model.

        Raises:
            ValueError: If no audio encoder is configured.
        """
        if self.audio_encoder is None:
            raise ValueError(
                "No audio encoder configured. Set audio_encoder to 'encodec' "
                f"in MultimodalEmbeddingConfig. Current setting: {self.config.audio_encoder}"
            )

        # Ensure correct shape (B, 1, T)
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)

        # Encode to get quantized embeddings
        # Forward gives (codes, quantized, reconstructed, loss)
        _, quantized, _, _ = self.audio_encoder(waveforms)

        # Project to model hidden size
        embeddings: Tensor = self.audio_projection(quantized)  # type: ignore[misc] # (B, T', D)

        return embeddings

    def forward(
        self,
        batch: MultimodalBatch,
        return_positions: bool = False,
    ) -> MultimodalOutput:
        """Embed all modalities and combine into unified sequence.

        Args:
            batch: MultimodalBatch containing input data for each modality.
            return_positions: If True, include modality position info in output.

        Returns:
            MultimodalOutput with combined embeddings, masks, and positions.

        Why: Main entry point for multimodal embedding. Concatenates embeddings
        from all provided modalities into a single sequence. The order is:
        [TEXT] [IMAGES] [AUDIO]

        This ordering is conventional but the model learns to handle any order
        through attention. The modality_mask enables modality-aware processing.

        Embedding-Prediction Context: This method produces the continuous
        embedding sequence that feeds into the transformer. All discrete inputs
        (tokens, image codes, audio codes) are converted to continuous embeddings
        in the shared space.

        Example sequence organization:
            Position:    0    1    2   ...  31   32  ...  607  608 ...  682
            Modality:  TEXT TEXT TEXT ... TEXT IMAGE ... IMAGE AUDIO ... AUDIO
            Content:   "A"  " " "cat" ...  "." patch ... patch frame ... frame
        """
        embeddings_list: list[Tensor] = []
        modality_types: list[Tensor] = []
        positions: dict[str, tuple[int, int]] = {}
        current_pos = 0

        # Determine batch size from any available input
        batch_size = self._get_batch_size(batch)

        # Process text embeddings
        if batch.input_ids is not None:
            text_embeds = self.embed_text(batch.input_ids)  # (B, L_text, D)
            embeddings_list.append(text_embeds)
            seq_len = text_embeds.shape[1]
            modality_types.append(
                torch.full(
                    (batch_size, seq_len),
                    ModalityType.TEXT,
                    dtype=torch.long,
                    device=text_embeds.device,
                )
            )
            positions["text"] = (current_pos, current_pos + seq_len)
            current_pos += seq_len

        # Process image embeddings
        if batch.pixel_values is not None:
            image_embeds = self.embed_images(batch.pixel_values)  # (B, L_img, D)
            embeddings_list.append(image_embeds)
            seq_len = image_embeds.shape[1]
            modality_types.append(
                torch.full(
                    (batch_size, seq_len),
                    ModalityType.IMAGE,
                    dtype=torch.long,
                    device=image_embeds.device,
                )
            )
            positions["image"] = (current_pos, current_pos + seq_len)
            current_pos += seq_len

        # Process audio embeddings
        if batch.audio_waveforms is not None:
            audio_embeds = self.embed_audio(batch.audio_waveforms)  # (B, L_audio, D)
            embeddings_list.append(audio_embeds)
            seq_len = audio_embeds.shape[1]
            modality_types.append(
                torch.full(
                    (batch_size, seq_len),
                    ModalityType.AUDIO,
                    dtype=torch.long,
                    device=audio_embeds.device,
                )
            )
            positions["audio"] = (current_pos, current_pos + seq_len)
            current_pos += seq_len

        # Concatenate all embeddings
        if not embeddings_list:
            raise ValueError(
                "No inputs provided: at least one of input_ids, pixel_values, "
                "or audio_waveforms must be non-None."
            )

        combined_embeddings = torch.cat(embeddings_list, dim=1)  # (B, L_total, D)
        combined_modality_mask = torch.cat(modality_types, dim=1)  # (B, L_total)

        # Apply dropout
        combined_embeddings = self.dropout(combined_embeddings)

        # Create attention mask if not provided
        if batch.attention_mask is not None:
            attention_mask = batch.attention_mask
        else:
            # All positions attend (no padding)
            attention_mask = torch.ones(
                batch_size,
                combined_embeddings.shape[1],
                dtype=torch.long,
                device=combined_embeddings.device,
            )

        return MultimodalOutput(
            embeddings=combined_embeddings,
            modality_mask=combined_modality_mask,
            attention_mask=attention_mask,
            modality_positions=positions if return_positions else {},
        )

    def _get_batch_size(self, batch: MultimodalBatch) -> int:
        """Determine batch size from available inputs.

        Why: Batch size is needed for creating modality masks even when some
        modalities are missing. We check all possible inputs.

        Raises:
            ValueError: If no inputs are provided (all are None).
        """
        if batch.input_ids is not None:
            return int(batch.input_ids.shape[0])
        if batch.pixel_values is not None:
            return int(batch.pixel_values.shape[0])
        if batch.audio_waveforms is not None:
            return int(batch.audio_waveforms.shape[0])
        raise ValueError(
            "No inputs provided: at least one of input_ids, pixel_values, "
            "or audio_waveforms must be non-None."
        )

    def get_num_patches(self) -> int:
        """Get number of visual tokens per image.

        Returns:
            Number of patches/tokens per image based on vision encoder config.

        Why: Useful for computing expected sequence length given number of images.

        Raises:
            ValueError: If no vision encoder is configured.
        """
        if self.config.vision_encoder == "siglip":
            assert self.config.siglip_config is not None
            return self.config.siglip_config.num_patches
        elif self.config.vision_encoder == "vqvae":
            assert self.config.vqvae_config is not None
            return self.config.vqvae_config.tokens_per_image
        else:
            raise ValueError(f"No vision encoder configured: {self.config.vision_encoder}")

    def get_audio_tokens_per_second(self) -> float:
        """Get number of audio tokens per second of audio.

        Returns:
            Tokens per second for audio encoding.

        Why: Useful for computing expected sequence length given audio duration.

        Raises:
            ValueError: If no audio encoder is configured.
        """
        if self.config.audio_encoder == "encodec":
            assert self.config.encodec_config is not None
            return self.config.encodec_config.tokens_per_second
        else:
            raise ValueError(f"No audio encoder configured: {self.config.audio_encoder}")

    def get_memory_usage_gb(self) -> float:
        """Estimate total memory usage of all encoders.

        Returns:
            Approximate memory in GB for all encoder parameters.

        Why: Memory budgeting for RTX 5080 16GB constraint.
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Assume FP32 (4 bytes per param)
        return total_params * 4 / 1e9  # type: ignore[no-any-return]


def create_multimodal_embedding(
    tritter_config: TritterConfig,
    vision_encoder: Literal["siglip", "vqvae", "none"] = "siglip",
    audio_encoder: Literal["encodec", "none"] = "encodec",
    siglip_config: SigLIPConfig | None = None,
    vqvae_config: VQVAEConfig | None = None,
    encodec_config: EnCodecConfig | None = None,
    freeze_vision_encoder: bool = True,
    freeze_audio_encoder: bool = True,
) -> MultimodalEmbedding:
    """Factory function to create multimodal embedding layer from TritterConfig.

    Args:
        tritter_config: Tritter model configuration (provides hidden_size, vocab_size).
        vision_encoder: Which vision encoder to use ("siglip", "vqvae", "none").
        audio_encoder: Which audio encoder to use ("encodec", "none").
        siglip_config: Optional custom SigLIP config.
        vqvae_config: Optional custom VQ-VAE config.
        encodec_config: Optional custom EnCodec config.
        freeze_vision_encoder: Whether to freeze vision encoder weights.
        freeze_audio_encoder: Whether to freeze audio encoder weights.

    Returns:
        Configured MultimodalEmbedding layer.

    Why: Convenience factory ensures embedding dimensions match the Tritter model.
    Extracts hidden_size and vocab_size from TritterConfig for consistency.

    Embedding-Prediction Context: This factory creates the entry point layer
    that converts all modality inputs to the shared embedding space used by
    the Tritter transformer.

    Example:
        >>> from tritter.core.config import TritterConfig
        >>> config = TritterConfig(model_size="7B")
        >>> embedding = create_multimodal_embedding(
        ...     config,
        ...     vision_encoder="siglip",
        ...     audio_encoder="encodec",
        ... )
    """
    mm_config = MultimodalEmbeddingConfig(
        hidden_size=tritter_config.hidden_size,
        vocab_size=tritter_config.vocab_size,
        max_seq_len=tritter_config.max_position_embeddings,
        vision_encoder=vision_encoder,
        audio_encoder=audio_encoder,
        siglip_config=siglip_config,
        vqvae_config=vqvae_config,
        encodec_config=encodec_config,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_audio_encoder=freeze_audio_encoder,
        dropout=tritter_config.dropout,
    )

    return MultimodalEmbedding(mm_config)


__all__ = [
    "ModalityType",
    "MultimodalBatch",
    "MultimodalEmbeddingConfig",
    "MultimodalEmbedding",
    "MultimodalOutput",
    "create_multimodal_embedding",
]
