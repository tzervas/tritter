"""Tests for VQ-VAE image tokenizer.

Validates VQ-VAE encoder initialization, forward pass, vector quantization, and projection.

Why: VQ-VAE image tokenizer is critical for unified vocabulary multimodal capability.
Tests verify:
1. Correct output shapes for integration with text embeddings
2. VQ quantization/dequantization round-trip preserves structure
3. Freezing behavior works correctly
4. Memory estimation is reasonable for 16GB budget
5. Projection aligns with model hidden size
6. Token count calculations are accurate
7. Gradient flow through straight-through estimator
8. Codebook utilization tracking
"""

import pytest
import torch
import torch.nn.functional as F

from tritter.core.config import TritterConfig
from tritter.vision.vqvae import (
    ImageProjection,
    ResidualBlock2D,
    VectorQuantizer,
    VQVAEConfig,
    VQVAEDecoder,
    VQVAEEncoder,
    VQVAEImageTokenizer,
    create_vqvae_tokenizer,
)


class TestVQVAEConfig:
    """Tests for VQVAEConfig."""

    def test_default_config(self) -> None:
        """Verify default config matches target settings.

        Why: Default should provide 256x256 input, 8192 codebook, ~1024 tokens.
        """
        config = VQVAEConfig()

        assert config.image_size == 256
        assert config.patch_size == 8
        assert config.in_channels == 3
        assert config.hidden_size == 256
        assert config.num_layers == 3
        assert config.codebook_size == 8192
        assert config.num_codebooks == 1
        assert config.commitment_cost == 0.25

    def test_num_patches_calculation(self) -> None:
        """Verify patch count calculation.

        Why: Determines spatial positions in latent representation.
        """
        config = VQVAEConfig(image_size=256, patch_size=8)
        # 256 / 8 = 32 patches per side, 32 * 32 = 1024 total
        assert config.num_patches == 1024

    def test_tokens_per_image_calculation(self) -> None:
        """Verify tokens per image calculation.

        Why: Key metric for sequence length planning.
        """
        config = VQVAEConfig(image_size=256, patch_size=8, num_codebooks=1)
        # 1024 patches * 1 codebook = 1024 tokens
        assert config.tokens_per_image == 1024

    def test_latent_size_calculation(self) -> None:
        """Verify latent spatial size calculation.

        Why: Used for reshaping operations.
        """
        config = VQVAEConfig(image_size=256, patch_size=8)
        # 256 / 8 = 32
        assert config.latent_size == 32

    def test_custom_config(self) -> None:
        """Verify custom config values are respected.

        Why: Users should be able to customize architecture.
        """
        config = VQVAEConfig(
            image_size=128,
            patch_size=4,
            hidden_size=512,
            codebook_size=4096,
        )

        assert config.image_size == 128
        assert config.patch_size == 4
        assert config.hidden_size == 512
        assert config.codebook_size == 4096
        assert config.num_patches == 1024  # 128/4 = 32, 32*32 = 1024


class TestResidualBlock2D:
    """Tests for 2D residual block."""

    def test_residual_preserves_shape(self) -> None:
        """Verify residual block preserves input shape.

        Why: Residual connections require same input/output shape.
        """
        block = ResidualBlock2D(channels=64, kernel_size=3)

        x = torch.randn(2, 64, 32, 32)
        output = block(x)

        assert output.shape == x.shape

    def test_residual_different_channels(self) -> None:
        """Verify residual block works with various channel counts.

        Why: Should handle different hidden sizes.
        """
        for channels in [32, 64, 128, 256]:
            block = ResidualBlock2D(channels=channels)
            x = torch.randn(2, channels, 16, 16)
            output = block(x)
            assert output.shape == x.shape


class TestVQVAEEncoder:
    """Tests for convolutional encoder."""

    def test_encoder_output_shape(self) -> None:
        """Verify encoder produces correct output shape.

        Why: Output shape determines latent grid size for quantization.
        """
        config = VQVAEConfig(image_size=256, patch_size=8, hidden_size=256)
        encoder = VQVAEEncoder(config)

        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)

        output = encoder(images)

        # Output should be (B, hidden_size, latent_size, latent_size)
        assert output.shape == (batch_size, config.hidden_size, 32, 32)

    def test_encoder_smaller_image(self) -> None:
        """Verify encoder handles smaller images.

        Why: Different image sizes should produce proportional latents.
        """
        config = VQVAEConfig(image_size=128, patch_size=8, hidden_size=128)
        encoder = VQVAEEncoder(config)

        images = torch.randn(2, 3, 128, 128)
        output = encoder(images)

        # 128 / 8 = 16
        assert output.shape == (2, config.hidden_size, 16, 16)

    def test_encoder_variable_batch_size(self) -> None:
        """Verify encoder handles variable batch sizes.

        Why: Should work with any batch size.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        encoder = VQVAEEncoder(config)

        for batch_size in [1, 2, 4, 8]:
            images = torch.randn(batch_size, 3, 64, 64)
            output = encoder(images)
            assert output.shape[0] == batch_size


class TestVQVAEDecoder:
    """Tests for convolutional decoder."""

    def test_decoder_output_shape(self) -> None:
        """Verify decoder produces approximately correct output shape.

        Why: Decoder should reconstruct original image dimensions.
        """
        config = VQVAEConfig(image_size=256, patch_size=8, hidden_size=256)
        decoder = VQVAEDecoder(config)

        # Latent size is 32x32
        latents = torch.randn(2, config.hidden_size, 32, 32)
        output = decoder(latents)

        assert output.shape[0] == 2
        assert output.shape[1] == config.in_channels
        # Reconstructed size should be close to original
        assert abs(output.shape[2] - config.image_size) <= 4
        assert abs(output.shape[3] - config.image_size) <= 4

    def test_encoder_decoder_roundtrip_shape(self) -> None:
        """Verify encoder-decoder roundtrip produces correct shape.

        Why: End-to-end shape should match input.
        """
        config = VQVAEConfig(image_size=128, patch_size=8, hidden_size=128)
        encoder = VQVAEEncoder(config)
        decoder = VQVAEDecoder(config)

        images = torch.randn(2, 3, 128, 128)
        latents = encoder(images)
        reconstructed = decoder(latents)

        assert reconstructed.shape[0] == 2
        assert reconstructed.shape[1] == 3
        # Allow small tolerance for padding effects
        assert abs(reconstructed.shape[2] - 128) <= 4
        assert abs(reconstructed.shape[3] - 128) <= 4


class TestVectorQuantizer:
    """Tests for vector quantizer."""

    def test_quantize_output_shapes(self) -> None:
        """Verify quantize produces correct shapes.

        Why: Codes shape is critical for tokenization.
        """
        config = VQVAEConfig(codebook_size=8192, hidden_size=256)
        quantizer = VectorQuantizer(config)

        batch_size = 2
        H, W = 32, 32
        z = torch.randn(batch_size, config.hidden_size, H, W)

        codes, quantized, loss = quantizer.quantize(z)

        assert codes.shape == (batch_size, H, W)
        assert quantized.shape == z.shape
        assert loss.dim() == 0  # Scalar loss

    def test_quantize_codes_in_range(self) -> None:
        """Verify quantized codes are within codebook range.

        Why: Codes must be valid indices into codebook.
        """
        config = VQVAEConfig(codebook_size=8192)
        quantizer = VectorQuantizer(config)

        z = torch.randn(2, config.hidden_size, 16, 16)
        codes, _, _ = quantizer.quantize(z)

        assert codes.min() >= 0
        assert codes.max() < config.codebook_size

    def test_dequantize_output_shape(self) -> None:
        """Verify dequantize produces correct shape.

        Why: Dequantized embeddings must match original shape for decoder.
        """
        config = VQVAEConfig()
        quantizer = VectorQuantizer(config)

        codes = torch.randint(0, config.codebook_size, (2, 32, 32))
        quantized = quantizer.dequantize(codes)

        assert quantized.shape == (2, config.hidden_size, 32, 32)

    def test_quantize_dequantize_consistency(self) -> None:
        """Verify quantize then dequantize produces consistent results.

        Why: Round-trip should reconstruct to same quantized embeddings.
        """
        config = VQVAEConfig()
        quantizer = VectorQuantizer(config)

        z = torch.randn(2, config.hidden_size, 16, 16)
        codes, quantized, _ = quantizer.quantize(z)

        # Dequantize should give same result as quantized output
        dequantized = quantizer.dequantize(codes)

        torch.testing.assert_close(quantized, dequantized, atol=1e-5, rtol=1e-5)

    def test_commitment_loss_positive(self) -> None:
        """Verify VQ loss is positive.

        Why: Loss encourages encoder to commit to codebook entries.
        """
        config = VQVAEConfig()
        quantizer = VectorQuantizer(config)

        z = torch.randn(2, config.hidden_size, 16, 16)
        _, _, loss = quantizer.quantize(z)

        assert loss > 0

    def test_get_codebook_usage(self) -> None:
        """Verify codebook usage statistics.

        Why: Low usage indicates codebook collapse.
        """
        config = VQVAEConfig(codebook_size=8192)
        quantizer = VectorQuantizer(config)

        # Random codes should have reasonable usage
        codes = torch.randint(0, config.codebook_size, (10, 32, 32))
        usage = quantizer.get_codebook_usage(codes)

        assert 0 < usage <= 1.0

    def test_quantize_3d_input(self) -> None:
        """Verify quantizer handles 3D input (B, L, C).

        Why: Should support flattened spatial input.
        """
        config = VQVAEConfig()
        quantizer = VectorQuantizer(config)

        z = torch.randn(2, 1024, config.hidden_size)  # (B, L, C)
        codes, quantized, loss = quantizer.quantize(z)

        assert codes.shape == (2, 1024)
        assert quantized.shape == z.shape


class TestVQVAEImageTokenizer:
    """Tests for complete image tokenizer."""

    def test_encode_output_shape(self) -> None:
        """Verify encode produces correct code shape.

        Why: Code shape determines token count for LM.
        """
        config = VQVAEConfig(image_size=128, patch_size=8)
        tokenizer = VQVAEImageTokenizer(config)

        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128)

        codes = tokenizer.encode(images)

        # 128 / 8 = 16, so codes should be (B, 16, 16)
        assert codes.shape == (batch_size, 16, 16)

    def test_decode_output_shape(self) -> None:
        """Verify decode produces approximately correct image shape.

        Why: Decoded images should match original dimensions.
        """
        config = VQVAEConfig(image_size=128, patch_size=8)
        tokenizer = VQVAEImageTokenizer(config)

        # Encode then decode
        images = torch.randn(1, 3, 128, 128)
        codes = tokenizer.encode(images)
        reconstructed = tokenizer.decode(codes)

        assert reconstructed.shape[0] == 1
        assert reconstructed.shape[1] == config.in_channels
        # Reconstructed size should be close to original
        assert abs(reconstructed.shape[2] - 128) <= 4
        assert abs(reconstructed.shape[3] - 128) <= 4

    def test_forward_returns_all_outputs(self) -> None:
        """Verify forward pass returns codes, quantized, reconstructed, loss.

        Why: Training requires all outputs for loss computation.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(2, 3, 64, 64)
        codes, quantized, reconstructed, loss = tokenizer(images)

        assert codes.dim() == 3  # (B, H', W')
        assert quantized.dim() == 4  # (B, C, H', W')
        assert reconstructed.dim() == 4  # (B, C, H, W)
        assert loss.dim() == 0  # Scalar

    def test_tokenize_returns_flat_list(self) -> None:
        """Verify tokenize returns flat token list.

        Why: Flat list is compatible with standard tokenizer interface.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        # 64 / 4 = 16, so 16*16 = 256 tokens
        images = torch.randn(1, 3, 64, 64)
        tokens = tokenizer.tokenize(images)

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == 256  # 16 * 16

    def test_tokenize_tokens_in_range(self) -> None:
        """Verify tokenized values are in valid range.

        Why: Tokens must be valid for vocabulary lookup.
        """
        config = VQVAEConfig(image_size=64, patch_size=4, codebook_size=8192)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(1, 3, 64, 64)
        tokens = tokenizer.tokenize(images)

        assert min(tokens) >= 0
        assert max(tokens) < config.codebook_size

    def test_detokenize_produces_codes(self) -> None:
        """Verify detokenize produces spatial codes.

        Why: Should reshape flat tokens back to spatial grid.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        # Create dummy tokens (16*16 = 256)
        tokens = list(range(256))
        codes = tokenizer.detokenize(tokens, height=16, width=16)

        assert codes.shape == (1, 16, 16)

    def test_tokenize_detokenize_roundtrip(self) -> None:
        """Verify tokenize -> detokenize preserves codes.

        Why: Round-trip should preserve token values.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(1, 3, 64, 64)
        tokens = tokenizer.tokenize(images)
        codes = tokenizer.detokenize(tokens)

        # Re-encode to verify consistency
        original_codes = tokenizer.encode(images)

        torch.testing.assert_close(codes[0], original_codes[0])

    def test_freeze_encoder_disables_gradients(self) -> None:
        """Verify freezing disables encoder gradients.

        Why: Frozen encoder saves memory.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config, freeze_encoder=True)

        # Encoder params should be frozen
        for param in tokenizer.encoder.parameters():
            assert not param.requires_grad

        # Quantizer and decoder should still be trainable
        for param in tokenizer.quantizer.parameters():
            assert param.requires_grad
        for param in tokenizer.decoder.parameters():
            assert param.requires_grad

    def test_unfrozen_encoder_has_gradients(self) -> None:
        """Verify unfrozen encoder has trainable parameters.

        Why: Full training mode should allow encoder updates.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config, freeze_encoder=False)

        # All params should be trainable
        trainable_count = sum(1 for p in tokenizer.parameters() if p.requires_grad)
        total_count = sum(1 for p in tokenizer.parameters())

        assert trainable_count == total_count

    def test_get_tokens_per_image(self) -> None:
        """Verify tokens per image calculation.

        Why: Needed for sequence length planning.
        """
        config = VQVAEConfig(image_size=256, patch_size=8)
        tokenizer = VQVAEImageTokenizer(config)

        # 256/8 = 32, 32*32 = 1024
        assert tokenizer.get_tokens_per_image() == 1024

    def test_get_codebook_embeddings(self) -> None:
        """Verify codebook embeddings retrieval.

        Why: Needed for unified vocabulary integration.
        """
        config = VQVAEConfig(codebook_size=8192, hidden_size=256)
        tokenizer = VQVAEImageTokenizer(config)

        embeddings = tokenizer.get_codebook_embeddings()

        assert embeddings.shape == (config.codebook_size, config.hidden_size)

    def test_memory_estimation_reasonable(self) -> None:
        """Verify memory estimation is reasonable.

        Why: Must fit within RTX 5080 16GB budget.
        """
        config = VQVAEConfig()
        tokenizer = VQVAEImageTokenizer(config)

        memory_gb = tokenizer.get_memory_usage_gb()

        # VQ-VAE should be relatively small (< 1 GB typically)
        assert 0.01 < memory_gb < 2.0, f"Memory {memory_gb:.3f} GB outside expected range"

    def test_handles_3d_image_input(self) -> None:
        """Verify tokenizer handles 3D image (no batch dim).

        Why: Users may pass (C, H, W) instead of (B, C, H, W).
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        image_3d = torch.randn(3, 64, 64)  # (C, H, W)
        tokens = tokenizer.tokenize(image_3d)

        assert len(tokens) == 256  # 16 * 16


class TestImageProjection:
    """Tests for image projection layer."""

    def test_projection_output_shape_3d(self) -> None:
        """Verify projection produces correct output shape for 3D input.

        Why: Output must match model hidden size for integration.
        """
        vqvae_hidden = 256
        model_hidden = 2048

        projection = ImageProjection(vqvae_hidden, model_hidden)

        # (B, L, C) input
        image_embeddings = torch.randn(2, 1024, vqvae_hidden)
        output = projection(image_embeddings)

        assert output.shape == (2, 1024, model_hidden)

    def test_projection_output_shape_4d(self) -> None:
        """Verify projection produces correct output shape for 4D input.

        Why: Should handle spatial format from encoder.
        """
        vqvae_hidden = 256
        model_hidden = 2048

        projection = ImageProjection(vqvae_hidden, model_hidden)

        # (B, C, H, W) input
        image_embeddings = torch.randn(2, vqvae_hidden, 32, 32)
        output = projection(image_embeddings)

        assert output.shape == (2, model_hidden, 32, 32)

    def test_projection_is_trainable(self) -> None:
        """Verify projection layer is trainable.

        Why: Projection should adapt to model embedding space.
        """
        projection = ImageProjection(256, 2048)

        for param in projection.parameters():
            assert param.requires_grad


class TestCreateVQVAETokenizer:
    """Tests for factory function."""

    def test_creates_tokenizer_with_tritter_config(self) -> None:
        """Verify factory creates tokenizer matching Tritter model.

        Why: Convenience function should auto-configure for model compatibility.
        """
        tritter_config = TritterConfig(
            model_size="3B",
            hidden_size=2048,
            num_layers=2,
        )

        tokenizer, projection = create_vqvae_tokenizer(tritter_config)

        # Verify projection matches Tritter hidden size
        test_input = torch.randn(1, 1024, 256)  # VQ-VAE hidden size
        output = projection(test_input)
        assert output.shape[-1] == tritter_config.hidden_size

    def test_custom_vqvae_config_respected(self) -> None:
        """Verify custom VQ-VAE config is used when provided.

        Why: Allow experimentation with different encoder settings.
        """
        tritter_config = TritterConfig(hidden_size=1024, num_layers=1)
        custom_vqvae = VQVAEConfig(codebook_size=4096, hidden_size=128)

        tokenizer, projection = create_vqvae_tokenizer(
            tritter_config,
            vqvae_config=custom_vqvae,
        )

        assert tokenizer.config.codebook_size == 4096
        assert tokenizer.config.hidden_size == 128

    def test_freeze_encoder_option(self) -> None:
        """Verify freeze_encoder option works.

        Why: Should support both frozen and trainable modes.
        """
        tritter_config = TritterConfig(hidden_size=512, num_layers=1)

        tokenizer_frozen, _ = create_vqvae_tokenizer(
            tritter_config,
            freeze_encoder=True,
        )
        tokenizer_unfrozen, _ = create_vqvae_tokenizer(
            tritter_config,
            freeze_encoder=False,
        )

        # Frozen tokenizer encoder should have no trainable params
        frozen_encoder_trainable = sum(
            1 for p in tokenizer_frozen.encoder.parameters() if p.requires_grad
        )
        unfrozen_encoder_trainable = sum(
            1 for p in tokenizer_unfrozen.encoder.parameters() if p.requires_grad
        )

        assert frozen_encoder_trainable == 0
        assert unfrozen_encoder_trainable > 0


class TestGradientFlow:
    """Tests for gradient flow through tokenizer."""

    def test_gradients_flow_through_quantizer(self) -> None:
        """Verify gradients flow through VQ via straight-through.

        Why: Training requires gradients to reach encoder.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config, freeze_encoder=False)

        images = torch.randn(1, 3, 64, 64, requires_grad=True)
        _, quantized, _, _ = tokenizer(images)

        loss = quantized.sum()
        loss.backward()

        # Images should have gradients (via straight-through estimator)
        assert images.grad is not None
        # Gradient magnitude should be reasonable (not zero or exploding)
        grad_max = images.grad.abs().max().item()
        assert 0 < grad_max < 1e6, f"Gradient magnitude {grad_max} outside expected range"

    def test_gradients_flow_through_projection(self) -> None:
        """Verify gradients flow through projection layer.

        Why: Projection is the trainable component for frozen encoders.
        """
        projection = ImageProjection(256, 2048)

        image_embeddings = torch.randn(1, 1024, 256)
        output = projection(image_embeddings)

        loss = output.sum()
        loss.backward()

        assert projection.proj.weight.grad is not None
        assert projection.proj.weight.grad.abs().max() > 0

    def test_vq_loss_is_differentiable(self) -> None:
        """Verify VQ loss enables backprop.

        Why: VQ loss should be backpropable for training.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config, freeze_encoder=False)

        images = torch.randn(1, 3, 64, 64)
        _, _, _, vq_loss = tokenizer(images)

        # VQ loss should be backpropable
        vq_loss.backward()

        # Quantizer codebook should have gradients
        assert tokenizer.quantizer.codebook.weight.grad is not None

    def test_reconstruction_loss_gradient_flow(self) -> None:
        """Verify reconstruction loss enables gradient flow.

        Why: Reconstruction loss is the primary training signal.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config, freeze_encoder=False)

        images = torch.randn(1, 3, 64, 64)
        _, _, reconstructed, _ = tokenizer(images)

        # MSE reconstruction loss
        recon_loss = F.mse_loss(reconstructed, images)
        recon_loss.backward()

        # Encoder should have gradients
        has_encoder_grads = any(
            p.grad is not None and p.grad.abs().max() > 0 for p in tokenizer.encoder.parameters()
        )
        assert has_encoder_grads, "Encoder should have gradients from reconstruction loss"


class TestReconstructionQuality:
    """Tests for reconstruction quality metrics."""

    def test_reconstruction_not_zero(self) -> None:
        """Verify reconstruction is not all zeros.

        Why: Basic sanity check that reconstruction works.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(1, 3, 64, 64)
        _, _, reconstructed, _ = tokenizer(images)

        assert reconstructed.abs().max() > 0

    def test_reconstruction_finite(self) -> None:
        """Verify reconstruction has no NaN/Inf values.

        Why: Numerical stability check.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(1, 3, 64, 64)
        _, _, reconstructed, _ = tokenizer(images)

        assert torch.isfinite(reconstructed).all()

    def test_quantized_embeddings_finite(self) -> None:
        """Verify quantized embeddings are finite.

        Why: Quantized embeddings are used for downstream tasks.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(1, 3, 64, 64)
        _, quantized, _, _ = tokenizer(images)

        assert torch.isfinite(quantized).all()


class TestCodebookUtilization:
    """Tests for codebook utilization tracking."""

    def test_codebook_usage_diverse_inputs(self) -> None:
        """Verify codebook usage with diverse inputs.

        Why: Diverse inputs should use more codebook entries.
        """
        config = VQVAEConfig(image_size=64, patch_size=4, codebook_size=1024)
        tokenizer = VQVAEImageTokenizer(config)

        # Generate diverse images
        images = torch.randn(10, 3, 64, 64)
        codes = tokenizer.encode(images)

        usage = tokenizer.quantizer.get_codebook_usage(codes)

        # With random inputs, should use reasonable fraction of codebook
        assert usage > 0.01, f"Codebook usage {usage:.2%} too low"

    def test_codebook_entries_non_zero(self) -> None:
        """Verify codebook entries are initialized non-zero.

        Why: Zero-initialized codebook would cause all codes to be same.
        """
        config = VQVAEConfig()
        tokenizer = VQVAEImageTokenizer(config)

        embeddings = tokenizer.get_codebook_embeddings()

        # Not all zeros
        assert embeddings.abs().max() > 0

        # Multiple unique rows
        unique_rows = torch.unique(embeddings, dim=0).shape[0]
        assert unique_rows == config.codebook_size, "Codebook should have unique entries"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_detokenize_wrong_length_raises(self) -> None:
        """Verify detokenize raises error for wrong token count.

        Why: Token count must match expected spatial dimensions.
        """
        config = VQVAEConfig(image_size=64, patch_size=4)
        tokenizer = VQVAEImageTokenizer(config)

        # Wrong number of tokens
        tokens = list(range(100))  # Should be 256 for 16x16

        with pytest.raises(ValueError, match="Token count"):
            tokenizer.detokenize(tokens, height=16, width=16)

    def test_small_image_size(self) -> None:
        """Verify tokenizer handles small image sizes.

        Why: Should work with minimal image sizes.
        """
        config = VQVAEConfig(image_size=32, patch_size=4, num_layers=2)
        tokenizer = VQVAEImageTokenizer(config)

        images = torch.randn(1, 3, 32, 32)
        codes = tokenizer.encode(images)

        # 32 / 4 = 8
        assert codes.shape == (1, 8, 8)

    def test_single_codebook(self) -> None:
        """Verify single codebook configuration.

        Why: Unlike audio RVQ, VQ-VAE uses single codebook.
        """
        config = VQVAEConfig(num_codebooks=1)

        assert config.num_codebooks == 1
        assert config.tokens_per_image == config.num_patches * 1
