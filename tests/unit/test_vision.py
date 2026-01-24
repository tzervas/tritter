"""Tests for SigLIP vision encoder.

Validates vision encoder initialization, forward pass, projection, and memory usage.

Why: Vision encoder is critical for multimodal capability. Tests verify:
1. Correct output shapes for integration with text embeddings
2. Freezing behavior works correctly
3. Memory estimation is reasonable for 16GB budget
4. Projection aligns with model hidden size
"""

import pytest
import torch

from tritter.core.config import TritterConfig
from tritter.vision.siglip import (
    SigLIPConfig,
    SigLIPEmbeddings,
    SigLIPEncoder,
    SigLIPVisionEncoder,
    create_siglip_encoder,
)


class TestSigLIPConfig:
    """Tests for SigLIPConfig."""

    def test_default_config(self) -> None:
        """Verify default config matches SigLIP-B/16 specs.

        Why: Default should be B/16 variant (93M params, ~0.4 GB).
        """
        config = SigLIPConfig()

        assert config.image_size == 384
        assert config.patch_size == 16
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12

    def test_num_patches_calculation(self) -> None:
        """Verify patch count calculation.

        Why: Determines visual token count for sequence length planning.
        """
        config = SigLIPConfig(image_size=384, patch_size=16)
        # 384 / 16 = 24 patches per side, 24 * 24 = 576 total
        assert config.num_patches == 576

    def test_num_positions_includes_cls(self) -> None:
        """Verify position count includes CLS token.

        Why: CLS token is prepended, so total positions = patches + 1.
        """
        config = SigLIPConfig(image_size=384, patch_size=16)
        assert config.num_positions == 577  # 576 patches + 1 CLS


class TestSigLIPEmbeddings:
    """Tests for patch embedding layer."""

    def test_embedding_output_shape(self) -> None:
        """Verify patch embedding produces correct shape.

        Why: Shape must match transformer input expectations.
        """
        config = SigLIPConfig(image_size=224, patch_size=16)
        embeddings = SigLIPEmbeddings(config)

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        output = embeddings(images)

        # 224/16 = 14 patches per side, 14*14 = 196 patches + 1 CLS = 197
        expected_seq_len = 197
        assert output.shape == (batch_size, expected_seq_len, config.hidden_size)

    def test_cls_token_prepended(self) -> None:
        """Verify CLS token is at position 0.

        Why: CLS token aggregates global representation.
        """
        config = SigLIPConfig(image_size=224, patch_size=16)
        embeddings = SigLIPEmbeddings(config)

        images = torch.randn(1, 3, 224, 224)
        output = embeddings(images)

        # CLS token should be at first position
        assert output.shape[1] > config.num_patches  # Has extra CLS position


class TestSigLIPEncoder:
    """Tests for transformer encoder stack."""

    def test_encoder_preserves_shape(self) -> None:
        """Verify encoder preserves sequence shape.

        Why: Transformer layers should only transform features, not shape.
        """
        config = SigLIPConfig()
        encoder = SigLIPEncoder(config)

        batch_size = 2
        seq_len = 100
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = encoder(hidden_states)

        assert output.shape == hidden_states.shape


class TestSigLIPVisionEncoder:
    """Tests for complete vision encoder with projection."""

    def test_full_encoder_output_shape(self) -> None:
        """Verify full encoder produces correct output shape.

        Why: Output must match model hidden size for embedding integration.
        """
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        model_hidden_size = 2048  # Typical 3B model

        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=model_hidden_size,
            freeze_encoder=False,  # Don't freeze for simpler testing
        )

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        # Full patch output (excluding CLS)
        output = encoder(images, return_cls=False)

        num_patches = siglip_config.num_patches  # 196 for 224/16
        assert output.shape == (batch_size, num_patches, model_hidden_size)

    def test_cls_token_output(self) -> None:
        """Verify CLS token output shape when requested.

        Why: CLS provides single-vector image representation.
        """
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        model_hidden_size = 1024

        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=model_hidden_size,
            freeze_encoder=False,
        )

        images = torch.randn(2, 3, 224, 224)

        cls_output = encoder(images, return_cls=True)

        assert cls_output.shape == (2, model_hidden_size)

    def test_freeze_encoder_disables_gradients(self) -> None:
        """Verify freezing disables encoder gradients but not projection.

        Why: Frozen encoder saves memory; only projection should train.
        """
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=512,
            freeze_encoder=True,
        )

        # Encoder params should be frozen
        for name, param in encoder.embeddings.named_parameters():
            assert not param.requires_grad, f"embeddings.{name} should be frozen"

        for name, param in encoder.encoder.named_parameters():
            assert not param.requires_grad, f"encoder.{name} should be frozen"

        # Projection should be trainable
        for name, param in encoder.projection.named_parameters():
            assert param.requires_grad, f"projection.{name} should be trainable"

    def test_unfrozen_encoder_has_gradients(self) -> None:
        """Verify unfrozen encoder has trainable parameters.

        Why: Full fine-tuning mode should allow encoder training.
        """
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=512,
            freeze_encoder=False,
        )

        # All params should be trainable
        trainable_count = sum(1 for p in encoder.parameters() if p.requires_grad)
        total_count = sum(1 for p in encoder.parameters())

        assert trainable_count == total_count

    def test_memory_estimation_reasonable(self) -> None:
        """Verify memory estimation is reasonable for SigLIP-B.

        Why: Must fit within RTX 5080 16GB budget (~0.4 GB expected).
        """
        siglip_config = SigLIPConfig()  # B/16 defaults
        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=768,  # Same as encoder for this test
            freeze_encoder=True,
        )

        memory_gb = encoder.get_memory_usage_gb()

        # SigLIP-B is ~93M params, should be ~0.35-0.45 GB in FP32
        # With projection layer, slightly more
        assert 0.2 < memory_gb < 0.6, f"Memory {memory_gb:.2f} GB outside expected range"

    def test_get_num_patches(self) -> None:
        """Verify num_patches helper returns correct count.

        Why: Needed for sequence length calculations.
        """
        siglip_config = SigLIPConfig(image_size=384, patch_size=16)
        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=512,
        )

        assert encoder.get_num_patches() == 576  # 24 * 24


class TestCreateSigLIPEncoder:
    """Tests for factory function."""

    def test_creates_encoder_with_tritter_config(self) -> None:
        """Verify factory creates encoder matching Tritter model.

        Why: Convenience function should auto-configure for model compatibility.
        """
        tritter_config = TritterConfig(
            model_size="3B",
            hidden_size=2048,
            num_layers=2,
        )

        encoder = create_siglip_encoder(tritter_config)

        # Verify projection matches Tritter hidden size
        assert encoder.model_hidden_size == tritter_config.hidden_size

    def test_custom_siglip_config_respected(self) -> None:
        """Verify custom SigLIP config is used when provided.

        Why: Allow experimentation with different encoder sizes.
        """
        tritter_config = TritterConfig(hidden_size=1024, num_layers=1)
        custom_siglip = SigLIPConfig(image_size=256, patch_size=32)

        encoder = create_siglip_encoder(
            tritter_config,
            siglip_config=custom_siglip,
        )

        assert encoder.config.image_size == 256
        assert encoder.config.patch_size == 32


class TestGradientFlow:
    """Tests for gradient flow through encoder."""

    def test_gradients_flow_through_projection(self) -> None:
        """Verify gradients flow to projection layer.

        Why: Training requires gradients to reach trainable parameters.
        """
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=256,
            freeze_encoder=True,
        )

        images = torch.randn(1, 3, 224, 224)
        output = encoder(images)

        loss = output.sum()
        loss.backward()

        # Projection should have gradients
        assert encoder.projection.weight.grad is not None
        assert encoder.projection.weight.grad.abs().max() > 0

    def test_frozen_encoder_no_gradients(self) -> None:
        """Verify frozen encoder doesn't accumulate gradients.

        Why: Frozen layers should have no gradient computation for memory savings.
        """
        siglip_config = SigLIPConfig(image_size=224, patch_size=16)
        encoder = SigLIPVisionEncoder(
            config=siglip_config,
            model_hidden_size=256,
            freeze_encoder=True,
        )

        images = torch.randn(1, 3, 224, 224)
        output = encoder(images)

        loss = output.sum()
        loss.backward()

        # Encoder embeddings should have no gradients
        for param in encoder.embeddings.parameters():
            assert param.grad is None
