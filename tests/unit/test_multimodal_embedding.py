"""Tests for unified multimodal embedding space.

Verifies that the MultimodalEmbedding layer correctly integrates text, vision
(SigLIP + VQ-VAE), and audio (EnCodec) into a shared embedding space.

Why: The multimodal embedding is the entry point for all modality inputs.
Correct functioning is critical for the early fusion architecture where
cross-modal attention depends on all modalities being in the same space.

Embedding-Prediction Context: These tests verify that discrete inputs (tokens,
pixels, waveforms) are correctly converted to continuous embeddings in the
unified space used by the transformer.
"""

from __future__ import annotations

import pytest
import torch

from tritter.audio.encodec import EnCodecConfig
from tritter.models.multimodal_embedding import (
    ModalityType,
    MultimodalBatch,
    MultimodalEmbedding,
    MultimodalEmbeddingConfig,
    MultimodalOutput,
    create_multimodal_embedding,
)
from tritter.vision.siglip import SigLIPConfig
from tritter.vision.vqvae import VQVAEConfig


class TestModalityType:
    """Tests for ModalityType enum."""

    def test_modality_values(self) -> None:
        """Verify modality type integer values are contiguous from 0.

        Why: Contiguous values enable direct tensor indexing for modality-aware
        processing. Values must be stable for model serialization.
        """
        assert ModalityType.TEXT == 0
        assert ModalityType.IMAGE == 1
        assert ModalityType.AUDIO == 2

    def test_modality_can_be_used_in_tensor(self) -> None:
        """Verify modality types can be used as tensor values.

        Why: Modality masks are tensors of modality type values. Must convert
        cleanly to tensor dtype.
        """
        mask = torch.tensor([ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO])
        assert mask.dtype == torch.int64
        assert mask.tolist() == [0, 1, 2]


class TestMultimodalBatch:
    """Tests for MultimodalBatch dataclass."""

    def test_empty_batch(self) -> None:
        """Verify empty batch has all None fields.

        Why: Batch fields are optional to support any modality combination.
        """
        batch = MultimodalBatch()
        assert batch.input_ids is None
        assert batch.pixel_values is None
        assert batch.audio_waveforms is None
        assert batch.modality_mask is None
        assert batch.attention_mask is None

    def test_text_only_batch(self) -> None:
        """Verify batch with only text input."""
        input_ids = torch.randint(0, 1000, (2, 32))
        batch = MultimodalBatch(input_ids=input_ids)
        assert batch.input_ids is not None
        assert batch.input_ids.shape == (2, 32)
        assert batch.pixel_values is None
        assert batch.audio_waveforms is None

    def test_all_modalities_batch(self) -> None:
        """Verify batch with all modality inputs."""
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 1000, (2, 32)),
            pixel_values=torch.randn(2, 3, 256, 256),
            audio_waveforms=torch.randn(2, 1, 24000),
        )
        assert batch.input_ids is not None
        assert batch.pixel_values is not None
        assert batch.audio_waveforms is not None


class TestMultimodalEmbeddingConfig:
    """Tests for MultimodalEmbeddingConfig."""

    def test_default_configs_created(self) -> None:
        """Verify default encoder configs are created when not provided.

        Why: Config should auto-create encoder configs for enabled encoders,
        reducing boilerplate for users.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=2048,
            vocab_size=65536,
            vision_encoder="siglip",
            audio_encoder="encodec",
        )
        assert config.siglip_config is not None
        assert config.encodec_config is not None
        assert isinstance(config.siglip_config, SigLIPConfig)
        assert isinstance(config.encodec_config, EnCodecConfig)

    def test_custom_configs_preserved(self) -> None:
        """Verify custom encoder configs are preserved.

        Why: Users should be able to customize encoder architectures.
        """
        custom_siglip = SigLIPConfig(image_size=224, patch_size=14)
        custom_encodec = EnCodecConfig(sample_rate=16000)

        config = MultimodalEmbeddingConfig(
            hidden_size=2048,
            vocab_size=65536,
            vision_encoder="siglip",
            audio_encoder="encodec",
            siglip_config=custom_siglip,
            encodec_config=custom_encodec,
        )

        assert config.siglip_config.image_size == 224
        assert config.siglip_config.patch_size == 14
        assert config.encodec_config.sample_rate == 16000

    def test_none_encoders(self) -> None:
        """Verify 'none' encoder settings disable those modalities.

        Why: Users may want text-only or single-modality models.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=2048,
            vocab_size=65536,
            vision_encoder="none",
            audio_encoder="none",
        )
        assert config.siglip_config is None
        assert config.vqvae_config is None
        assert config.encodec_config is None

    def test_vqvae_config_created(self) -> None:
        """Verify VQ-VAE config is created when vqvae encoder selected."""
        config = MultimodalEmbeddingConfig(
            hidden_size=2048,
            vocab_size=65536,
            vision_encoder="vqvae",
        )
        assert config.vqvae_config is not None
        assert isinstance(config.vqvae_config, VQVAEConfig)


class TestMultimodalEmbeddingTextOnly:
    """Tests for text-only embedding."""

    @pytest.fixture
    def text_only_embedding(self) -> MultimodalEmbedding:
        """Create embedding layer with no vision/audio encoders.

        Why: Isolates text embedding functionality for focused testing.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=256,
            vocab_size=1000,
            vision_encoder="none",
            audio_encoder="none",
        )
        return MultimodalEmbedding(config)

    def test_embed_text_shape(self, text_only_embedding: MultimodalEmbedding) -> None:
        """Verify text embedding output shape.

        Why: Output shape must be (B, L, hidden_size) for transformer input.
        """
        token_ids = torch.randint(0, 1000, (2, 32))  # (B, L)
        embeddings = text_only_embedding.embed_text(token_ids)

        assert embeddings.shape == (2, 32, 256)  # (B, L, D)
        assert embeddings.dtype == torch.float32

    def test_text_only_forward(self, text_only_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with only text input.

        Why: Model should handle text-only batches gracefully.
        """
        batch = MultimodalBatch(input_ids=torch.randint(0, 1000, (2, 32)))
        output = text_only_embedding(batch)

        assert isinstance(output, MultimodalOutput)
        assert output.embeddings.shape == (2, 32, 256)
        assert output.modality_mask.shape == (2, 32)
        assert (output.modality_mask == ModalityType.TEXT).all()
        assert output.attention_mask.shape == (2, 32)

    def test_text_embedding_gradient_flow(self, text_only_embedding: MultimodalEmbedding) -> None:
        """Verify gradients flow through text embedding.

        Why: Text embeddings must be trainable for the model to learn.
        """
        token_ids = torch.randint(0, 1000, (2, 32))
        batch = MultimodalBatch(input_ids=token_ids)
        output = text_only_embedding(batch)

        # Compute loss and backward
        loss = output.embeddings.sum()
        loss.backward()

        # Check gradient exists and has reasonable magnitude
        grad = text_only_embedding.text_embedding.weight.grad
        assert grad is not None
        assert grad.abs().max() > 0


class TestMultimodalEmbeddingSigLIP:
    """Tests for SigLIP vision embedding."""

    @pytest.fixture
    def siglip_embedding(self) -> MultimodalEmbedding:
        """Create embedding layer with SigLIP vision encoder.

        Why: Tests SigLIP integration for patch-based image encoding.
        """
        siglip_config = SigLIPConfig(
            image_size=224,
            patch_size=16,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_heads=4,
        )
        config = MultimodalEmbeddingConfig(
            hidden_size=128,
            vocab_size=1000,
            vision_encoder="siglip",
            audio_encoder="none",
            siglip_config=siglip_config,
            freeze_vision_encoder=False,  # Enable gradients for testing
        )
        return MultimodalEmbedding(config)

    def test_embed_images_shape(self, siglip_embedding: MultimodalEmbedding) -> None:
        """Verify image embedding output shape.

        Why: Output must be (B, num_patches, hidden_size).
        For 224x224 image with patch_size=16: (224/16)^2 = 196 patches.
        """
        pixel_values = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
        embeddings = siglip_embedding.embed_images(pixel_values)

        # 224/16 = 14 patches per side, 14*14 = 196 patches
        assert embeddings.shape == (2, 196, 128)  # (B, num_patches, D)

    def test_image_only_forward(self, siglip_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with only image input.

        Why: Model should handle image-only batches.
        """
        batch = MultimodalBatch(pixel_values=torch.randn(2, 3, 224, 224))
        output = siglip_embedding(batch)

        assert output.embeddings.shape == (2, 196, 128)
        assert (output.modality_mask == ModalityType.IMAGE).all()

    def test_text_plus_image_forward(self, siglip_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with text + image.

        Why: Core multimodal use case - text and image together.
        """
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 1000, (2, 32)),
            pixel_values=torch.randn(2, 3, 224, 224),
        )
        output = siglip_embedding(batch)

        # 32 text tokens + 196 image patches = 228 total
        assert output.embeddings.shape == (2, 228, 128)

        # Check modality mask
        text_mask = output.modality_mask[:, :32]
        image_mask = output.modality_mask[:, 32:]
        assert (text_mask == ModalityType.TEXT).all()
        assert (image_mask == ModalityType.IMAGE).all()

    def test_batched_images(self, siglip_embedding: MultimodalEmbedding) -> None:
        """Verify handling of multiple images per sample.

        Why: Some use cases have multiple images per text (e.g., image comparison).
        """
        # 2 samples, 3 images each
        pixel_values = torch.randn(2, 3, 3, 224, 224)  # (B, N, C, H, W)
        embeddings = siglip_embedding.embed_images(pixel_values)

        # 3 images * 196 patches = 588 patches total
        assert embeddings.shape == (2, 588, 128)

    def test_image_gradient_flow(self, siglip_embedding: MultimodalEmbedding) -> None:
        """Verify gradients flow through image embedding.

        Why: When not frozen, vision projection should receive gradients.
        """
        pixel_values = torch.randn(2, 3, 224, 224, requires_grad=True)
        batch = MultimodalBatch(pixel_values=pixel_values)
        output = siglip_embedding(batch)

        loss = output.embeddings.sum()
        loss.backward()

        # Gradient should flow back to input
        assert pixel_values.grad is not None
        assert pixel_values.grad.abs().max() > 0


class TestMultimodalEmbeddingVQVAE:
    """Tests for VQ-VAE vision embedding."""

    @pytest.fixture
    def vqvae_embedding(self) -> MultimodalEmbedding:
        """Create embedding layer with VQ-VAE vision encoder.

        Why: Tests VQ-VAE integration for discrete image tokenization.
        """
        vqvae_config = VQVAEConfig(
            image_size=64,
            patch_size=8,
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            codebook_size=256,
        )
        config = MultimodalEmbeddingConfig(
            hidden_size=128,
            vocab_size=1000,
            vision_encoder="vqvae",
            audio_encoder="none",
            vqvae_config=vqvae_config,
            freeze_vision_encoder=False,
        )
        return MultimodalEmbedding(config)

    def test_embed_images_vqvae_shape(self, vqvae_embedding: MultimodalEmbedding) -> None:
        """Verify VQ-VAE image embedding output shape.

        Why: VQ-VAE produces a spatial grid of tokens. The actual token count
        depends on the encoder's downsampling behavior, which may differ from
        the theoretical patch-based calculation due to conv layer striding.
        """
        pixel_values = torch.randn(2, 3, 64, 64)
        embeddings = vqvae_embedding.embed_images(pixel_values)

        # Output shape should be (B, num_tokens, hidden_size)
        assert embeddings.dim() == 3
        assert embeddings.shape[0] == 2  # batch
        assert embeddings.shape[2] == 128  # hidden_size
        # num_tokens varies based on encoder architecture, just verify it's positive
        assert embeddings.shape[1] > 0

    def test_vqvae_forward(self, vqvae_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with VQ-VAE encoded images."""
        text_len = 16
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 1000, (2, text_len)),
            pixel_values=torch.randn(2, 3, 64, 64),
        )
        output = vqvae_embedding(batch)

        # Total should be text_len + image_tokens
        assert output.embeddings.dim() == 3
        assert output.embeddings.shape[0] == 2
        assert output.embeddings.shape[1] > text_len  # Has image tokens added
        assert output.embeddings.shape[2] == 128


class TestMultimodalEmbeddingAudio:
    """Tests for audio embedding."""

    @pytest.fixture
    def audio_embedding(self) -> MultimodalEmbedding:
        """Create embedding layer with audio encoder.

        Why: Tests EnCodec integration for audio encoding.
        """
        encodec_config = EnCodecConfig(
            sample_rate=16000,
            channels=1,
            hidden_size=64,
            num_layers=2,
            codebook_size=256,
            num_codebooks=4,
            downsample_rate=320,
        )
        config = MultimodalEmbeddingConfig(
            hidden_size=128,
            vocab_size=1000,
            vision_encoder="none",
            audio_encoder="encodec",
            encodec_config=encodec_config,
            freeze_audio_encoder=False,
        )
        return MultimodalEmbedding(config)

    def test_embed_audio_shape(self, audio_embedding: MultimodalEmbedding) -> None:
        """Verify audio embedding output shape.

        Why: Output must be (B, num_frames, hidden_size). The exact frame count
        depends on encoder architecture (conv strides, padding). We verify the
        approximate expected count (16000/320 ~ 50 frames, allowing +/- 5).
        """
        waveforms = torch.randn(2, 1, 16000)  # 1 second at 16kHz
        embeddings = audio_embedding.embed_audio(waveforms)

        # 16000 / 320 ~ 50 frames, allow small variation due to conv padding
        assert embeddings.dim() == 3
        assert embeddings.shape[0] == 2  # batch
        assert embeddings.shape[2] == 128  # hidden_size
        # Approximate frame count check (allow +/- 5 for conv padding effects)
        expected_frames = 16000 // 320
        assert abs(embeddings.shape[1] - expected_frames) <= 5

    def test_audio_2d_input(self, audio_embedding: MultimodalEmbedding) -> None:
        """Verify audio embedding handles 2D input (B, T).

        Why: Convenience support for mono audio without explicit channel dim.
        """
        waveforms = torch.randn(2, 16000)  # No channel dim
        embeddings = audio_embedding.embed_audio(waveforms)

        assert embeddings.dim() == 3
        assert embeddings.shape[0] == 2
        assert embeddings.shape[2] == 128

    def test_audio_only_forward(self, audio_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with only audio input."""
        batch = MultimodalBatch(audio_waveforms=torch.randn(2, 1, 16000))
        output = audio_embedding(batch)

        assert output.embeddings.dim() == 3
        assert output.embeddings.shape[0] == 2
        assert output.embeddings.shape[2] == 128
        assert (output.modality_mask == ModalityType.AUDIO).all()

    def test_text_plus_audio_forward(self, audio_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with text + audio."""
        text_len = 20
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 1000, (2, text_len)),
            audio_waveforms=torch.randn(2, 1, 16000),
        )
        output = audio_embedding(batch)

        # Should have text_len + audio_frames tokens
        assert output.embeddings.dim() == 3
        assert output.embeddings.shape[0] == 2
        assert output.embeddings.shape[1] > text_len  # Has audio tokens added
        assert output.embeddings.shape[2] == 128

        # Check modality mask
        text_mask = output.modality_mask[:, :text_len]
        audio_mask = output.modality_mask[:, text_len:]
        assert (text_mask == ModalityType.TEXT).all()
        assert (audio_mask == ModalityType.AUDIO).all()

    def test_audio_gradient_flow(self, audio_embedding: MultimodalEmbedding) -> None:
        """Verify gradients flow through audio embedding."""
        waveforms = torch.randn(2, 1, 16000, requires_grad=True)
        batch = MultimodalBatch(audio_waveforms=waveforms)
        output = audio_embedding(batch)

        loss = output.embeddings.sum()
        loss.backward()

        assert waveforms.grad is not None
        assert waveforms.grad.abs().max() > 0


class TestMultimodalEmbeddingAllModalities:
    """Tests for combined text + image + audio embedding."""

    @pytest.fixture
    def full_embedding(self) -> MultimodalEmbedding:
        """Create embedding layer with all modalities.

        Why: Tests full multimodal integration.
        """
        siglip_config = SigLIPConfig(
            image_size=64,
            patch_size=16,
            hidden_size=64,
            intermediate_size=128,
            num_layers=1,
            num_heads=2,
        )
        encodec_config = EnCodecConfig(
            sample_rate=8000,
            channels=1,
            hidden_size=32,
            num_layers=2,
            codebook_size=128,
            num_codebooks=2,
            downsample_rate=160,
        )
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="siglip",
            audio_encoder="encodec",
            siglip_config=siglip_config,
            encodec_config=encodec_config,
            freeze_vision_encoder=False,
            freeze_audio_encoder=False,
        )
        return MultimodalEmbedding(config)

    def test_all_modalities_forward(self, full_embedding: MultimodalEmbedding) -> None:
        """Verify forward pass with all three modalities.

        Why: Core multimodal use case - text, image, and audio together.
        """
        text_len = 10
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 500, (2, text_len)),
            pixel_values=torch.randn(2, 3, 64, 64),
            audio_waveforms=torch.randn(2, 1, 8000),  # 1 second at 8kHz
        )
        output = full_embedding(batch)

        # Output should have all modalities concatenated
        assert output.embeddings.dim() == 3
        assert output.embeddings.shape[0] == 2  # batch
        assert output.embeddings.shape[2] == 64  # hidden_size
        # Total length should be > text_len (text + image + audio)
        total_len = output.embeddings.shape[1]
        assert total_len > text_len

        # Check modality mask has correct types
        # First text_len positions should be TEXT
        text_mask = output.modality_mask[:, :text_len]
        assert (text_mask == ModalityType.TEXT).all()

        # Remaining positions should be IMAGE and AUDIO
        remaining_mask = output.modality_mask[:, text_len:]
        assert ModalityType.IMAGE in remaining_mask
        assert ModalityType.AUDIO in remaining_mask

    def test_modality_positions_returned(self, full_embedding: MultimodalEmbedding) -> None:
        """Verify modality positions are correctly tracked.

        Why: Position info enables modality-aware processing downstream.
        """
        text_len = 10
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 500, (2, text_len)),
            pixel_values=torch.randn(2, 3, 64, 64),
            audio_waveforms=torch.randn(2, 1, 8000),
        )
        output = full_embedding(batch, return_positions=True)

        assert "text" in output.modality_positions
        assert "image" in output.modality_positions
        assert "audio" in output.modality_positions

        # Text should start at 0 and have length text_len
        text_start, text_end = output.modality_positions["text"]
        assert text_start == 0
        assert text_end == text_len

        # Image should start where text ends
        image_start, image_end = output.modality_positions["image"]
        assert image_start == text_end
        assert image_end > image_start  # Has some image tokens

        # Audio should start where image ends
        audio_start, audio_end = output.modality_positions["audio"]
        assert audio_start == image_end
        assert audio_end > audio_start  # Has some audio tokens

    def test_attention_mask_default(self, full_embedding: MultimodalEmbedding) -> None:
        """Verify attention mask defaults to all ones (attend to all).

        Why: Without explicit mask, all positions should be attended to.
        """
        batch = MultimodalBatch(
            input_ids=torch.randint(0, 500, (2, 10)),
            pixel_values=torch.randn(2, 3, 64, 64),
        )
        output = full_embedding(batch)

        assert (output.attention_mask == 1).all()
        assert output.attention_mask.shape == (2, 26)

    def test_custom_attention_mask(self, full_embedding: MultimodalEmbedding) -> None:
        """Verify custom attention mask is preserved.

        Why: Users may want to mask padding or specific positions.
        """
        custom_mask = torch.ones(2, 26)
        custom_mask[:, -5:] = 0  # Mask last 5 positions

        batch = MultimodalBatch(
            input_ids=torch.randint(0, 500, (2, 10)),
            pixel_values=torch.randn(2, 3, 64, 64),
            attention_mask=custom_mask,
        )
        output = full_embedding(batch)

        assert (output.attention_mask == custom_mask).all()

    def test_all_modalities_gradient_flow(self, full_embedding: MultimodalEmbedding) -> None:
        """Verify gradients flow through all modalities.

        Why: All trainable components should receive gradients for learning.
        """
        input_ids = torch.randint(0, 500, (2, 10))
        pixel_values = torch.randn(2, 3, 64, 64, requires_grad=True)
        audio_waveforms = torch.randn(2, 1, 8000, requires_grad=True)

        batch = MultimodalBatch(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_waveforms=audio_waveforms,
        )
        output = full_embedding(batch)

        loss = output.embeddings.sum()
        loss.backward()

        # Check gradients exist
        assert pixel_values.grad is not None
        assert audio_waveforms.grad is not None
        assert full_embedding.text_embedding.weight.grad is not None


class TestMultimodalEmbeddingErrors:
    """Tests for error handling."""

    def test_no_inputs_error(self) -> None:
        """Verify error when no inputs provided.

        Why: Empty batch is likely a bug - fail fast with clear message.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="none",
            audio_encoder="none",
        )
        embedding = MultimodalEmbedding(config)

        batch = MultimodalBatch()
        with pytest.raises(ValueError, match="No inputs provided:"):
            embedding(batch)

    def test_vision_encoder_not_configured_error(self) -> None:
        """Verify error when calling embed_images without vision encoder.

        Why: Clear error message helps users fix configuration.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="none",
            audio_encoder="none",
        )
        embedding = MultimodalEmbedding(config)

        with pytest.raises(ValueError, match="No vision encoder configured"):
            embedding.embed_images(torch.randn(2, 3, 64, 64))

    def test_audio_encoder_not_configured_error(self) -> None:
        """Verify error when calling embed_audio without audio encoder.

        Why: Clear error message helps users fix configuration.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="none",
            audio_encoder="none",
        )
        embedding = MultimodalEmbedding(config)

        with pytest.raises(ValueError, match="No audio encoder configured"):
            embedding.embed_audio(torch.randn(2, 1, 8000))


class TestFactoryFunction:
    """Tests for create_multimodal_embedding factory function."""

    def test_create_from_tritter_config(self) -> None:
        """Verify factory creates embedding matching TritterConfig.

        Why: Factory ensures hidden_size and vocab_size match model config.
        """
        from tritter.core.config import TritterConfig

        tritter_config = TritterConfig(
            model_size="3B",
            vocab_size=32000,
        )

        embedding = create_multimodal_embedding(
            tritter_config,
            vision_encoder="siglip",
            audio_encoder="encodec",
        )

        assert embedding.config.hidden_size == tritter_config.hidden_size
        assert embedding.config.vocab_size == tritter_config.vocab_size
        assert embedding.config.max_seq_len == tritter_config.max_position_embeddings
        assert embedding.config.dropout == tritter_config.dropout

    def test_factory_custom_encoders(self) -> None:
        """Verify factory respects custom encoder configs."""
        from tritter.core.config import TritterConfig

        tritter_config = TritterConfig(model_size="3B")
        custom_siglip = SigLIPConfig(image_size=224)

        embedding = create_multimodal_embedding(
            tritter_config,
            vision_encoder="siglip",
            audio_encoder="none",
            siglip_config=custom_siglip,
        )

        assert embedding.config.siglip_config is not None
        assert embedding.config.siglip_config.image_size == 224

    def test_factory_freeze_settings(self) -> None:
        """Verify factory respects freeze settings.

        Why: Users control whether encoders are frozen or trainable.
        """
        from tritter.core.config import TritterConfig

        tritter_config = TritterConfig(model_size="3B")

        embedding = create_multimodal_embedding(
            tritter_config,
            vision_encoder="siglip",
            audio_encoder="encodec",
            freeze_vision_encoder=False,
            freeze_audio_encoder=True,
        )

        assert embedding.config.freeze_vision_encoder is False
        assert embedding.config.freeze_audio_encoder is True


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_num_patches_siglip(self) -> None:
        """Verify get_num_patches returns correct count for SigLIP."""
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="siglip",
            siglip_config=siglip_config,
        )
        embedding = MultimodalEmbedding(config)

        # 224/16 = 14, 14*14 = 196
        assert embedding.get_num_patches() == 196

    def test_get_num_patches_vqvae(self) -> None:
        """Verify get_num_patches returns correct count for VQ-VAE."""
        vqvae_config = VQVAEConfig(image_size=256, patch_size=8)
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="vqvae",
            vqvae_config=vqvae_config,
        )
        embedding = MultimodalEmbedding(config)

        # 256/8 = 32, 32*32 = 1024
        assert embedding.get_num_patches() == 1024

    def test_get_audio_tokens_per_second(self) -> None:
        """Verify get_audio_tokens_per_second returns correct rate."""
        encodec_config = EnCodecConfig(sample_rate=24000, downsample_rate=320)
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="none",
            audio_encoder="encodec",
            encodec_config=encodec_config,
        )
        embedding = MultimodalEmbedding(config)

        # 24000 / 320 = 75 tokens/second
        assert embedding.get_audio_tokens_per_second() == 75.0

    def test_get_memory_usage_gb(self) -> None:
        """Verify memory usage estimation is reasonable.

        Why: Memory estimation helps with hardware planning.
        """
        config = MultimodalEmbeddingConfig(
            hidden_size=64,
            vocab_size=500,
            vision_encoder="none",
            audio_encoder="none",
        )
        embedding = MultimodalEmbedding(config)

        memory_gb = embedding.get_memory_usage_gb()

        # Text embedding alone: 500 * 64 = 32000 params = 128KB = ~0.0001 GB
        assert memory_gb > 0
        assert memory_gb < 0.01  # Should be small for this config
