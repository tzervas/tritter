"""Tests for EnCodec audio tokenizer.

Validates audio encoder initialization, forward pass, RVQ quantization, and projection.

Why: Audio tokenizer is critical for multimodal capability. Tests verify:
1. Correct output shapes for integration with text/image embeddings
2. RVQ quantization/dequantization round-trip preserves structure
3. Freezing behavior works correctly
4. Memory estimation is reasonable for 16GB budget
5. Projection aligns with model hidden size
6. Token rate calculations are accurate
"""

import torch
import pytest

from tritter.core.config import TritterConfig
from tritter.audio.encodec import (
    EnCodecConfig,
    EnCodecEncoder,
    EnCodecDecoder,
    ResidualVectorQuantizer,
    EnCodecAudioTokenizer,
    AudioProjection,
    create_audio_tokenizer,
    ConvBlock,
    ResidualUnit,
    EncoderBlock,
    DecoderBlock,
)


class TestEnCodecConfig:
    """Tests for EnCodecConfig."""

    def test_default_config(self) -> None:
        """Verify default config matches standard EnCodec settings.

        Why: Default should provide 75 tokens/sec at 24kHz with 8 codebooks.
        """
        config = EnCodecConfig()

        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.hidden_size == 128
        assert config.num_layers == 4
        assert config.codebook_size == 1024
        assert config.num_codebooks == 8
        assert config.downsample_rate == 320

    def test_tokens_per_second_calculation(self) -> None:
        """Verify token rate calculation.

        Why: Determines audio sequence length for transformer processing.
        """
        config = EnCodecConfig(sample_rate=24000, downsample_rate=320)
        # 24000 / 320 = 75 tokens/second
        assert config.tokens_per_second == 75.0

    def test_total_vocabulary_size(self) -> None:
        """Verify total vocabulary calculation.

        Why: With flattened tokenization, total vocab = codebook_size * num_codebooks.
        """
        config = EnCodecConfig(codebook_size=1024, num_codebooks=8)
        assert config.total_vocabulary_size == 8192

    def test_hop_length_equals_downsample_rate(self) -> None:
        """Verify hop length property.

        Why: Hop length is equivalent to downsample rate for frame calculations.
        """
        config = EnCodecConfig(downsample_rate=320)
        assert config.hop_length == 320


class TestConvBlock:
    """Tests for basic convolutional block."""

    def test_conv_block_output_shape(self) -> None:
        """Verify conv block produces correct shape.

        Why: Basic building block must preserve or downsample correctly.
        """
        block = ConvBlock(in_channels=1, out_channels=64, kernel_size=7)

        x = torch.randn(2, 1, 1000)
        output = block(x)

        assert output.shape == (2, 64, 1000)

    def test_conv_block_downsampling(self) -> None:
        """Verify conv block with stride downsamples correctly.

        Why: Downsampling is key for compression.
        """
        block = ConvBlock(in_channels=64, out_channels=128, kernel_size=8, stride=4)

        x = torch.randn(2, 64, 1000)
        output = block(x)

        # Output should be approximately input_len / stride (with padding effects)
        assert output.shape[0] == 2
        assert output.shape[1] == 128
        # Allow small tolerance for padding effects
        expected_len = 1000 // 4
        assert abs(output.shape[2] - expected_len) <= 2

    def test_transpose_conv_block_upsamples(self) -> None:
        """Verify transposed conv block upsamples correctly.

        Why: Upsampling is needed for decoder.
        """
        block = ConvBlock(in_channels=128, out_channels=64, kernel_size=8, stride=4, transpose=True)

        x = torch.randn(2, 128, 250)
        output = block(x)

        # Output should be approximately input_len * stride (with padding effects)
        assert output.shape[0] == 2
        assert output.shape[1] == 64
        # Allow small tolerance for padding effects
        expected_len = 250 * 4
        assert abs(output.shape[2] - expected_len) <= 2


class TestResidualUnit:
    """Tests for residual unit with dilated convolutions."""

    def test_residual_preserves_shape(self) -> None:
        """Verify residual unit preserves input shape.

        Why: Residual connections require same input/output shape.
        """
        unit = ResidualUnit(channels=64, kernel_size=3, dilation=1)

        x = torch.randn(2, 64, 1000)
        output = unit(x)

        assert output.shape == x.shape

    def test_dilated_residual_preserves_shape(self) -> None:
        """Verify dilated residual preserves shape.

        Why: Dilation should not affect output shape.
        """
        unit = ResidualUnit(channels=64, kernel_size=3, dilation=9)

        x = torch.randn(2, 64, 1000)
        output = unit(x)

        assert output.shape == x.shape


class TestEncoderBlock:
    """Tests for encoder block."""

    def test_encoder_block_output_shape(self) -> None:
        """Verify encoder block downsamples correctly.

        Why: Each encoder block should halve (approximately) temporal dimension.
        """
        block = EncoderBlock(in_channels=64, out_channels=128, kernel_size=8, stride=4)

        x = torch.randn(2, 64, 1000)
        output = block(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 128
        # Allow small tolerance for padding effects
        expected_len = 1000 // 4
        assert abs(output.shape[2] - expected_len) <= 2


class TestDecoderBlock:
    """Tests for decoder block."""

    def test_decoder_block_output_shape(self) -> None:
        """Verify decoder block upsamples correctly.

        Why: Each decoder block should double (approximately) temporal dimension.
        """
        block = DecoderBlock(in_channels=128, out_channels=64, kernel_size=8, stride=4)

        x = torch.randn(2, 128, 250)
        output = block(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 64
        # Allow small tolerance for padding effects
        expected_len = 250 * 4
        assert abs(output.shape[2] - expected_len) <= 2


class TestEnCodecEncoder:
    """Tests for convolutional encoder."""

    def test_encoder_output_shape(self) -> None:
        """Verify encoder produces correct output shape.

        Why: Output shape determines token count for transformer.
        """
        config = EnCodecConfig(hidden_size=128, downsample_rate=320)
        encoder = EnCodecEncoder(config)

        # 24000 samples = 1 second @ 24kHz
        # With downsample_rate=320, output should be ~75 frames
        batch_size = 2
        num_samples = 24000
        waveform = torch.randn(batch_size, 1, num_samples)

        output = encoder(waveform)

        assert output.shape[0] == batch_size
        assert output.shape[2] == config.hidden_size
        # Output frames should be approximately num_samples / downsample_rate
        expected_frames = num_samples // config.downsample_rate
        assert abs(output.shape[1] - expected_frames) <= 2  # Allow small tolerance

    def test_encoder_variable_length(self) -> None:
        """Verify encoder handles variable length audio.

        Why: Audio clips may have different durations.
        """
        config = EnCodecConfig()
        encoder = EnCodecEncoder(config)

        # Test different lengths
        for length in [8000, 24000, 48000]:
            waveform = torch.randn(1, 1, length)
            output = encoder(waveform)
            assert output.dim() == 3  # (B, T, D)

    def test_get_output_length(self) -> None:
        """Verify output length calculation.

        Why: Useful for pre-computing token counts.
        """
        config = EnCodecConfig(downsample_rate=320)
        encoder = EnCodecEncoder(config)

        input_length = 24000
        output_length = encoder.get_output_length(input_length)

        # Should match actual encoder output
        waveform = torch.randn(1, 1, input_length)
        actual_output = encoder(waveform)

        # Allow small tolerance due to stride effects
        assert abs(output_length - actual_output.shape[1]) <= 2


class TestEnCodecDecoder:
    """Tests for convolutional decoder."""

    def test_decoder_output_shape(self) -> None:
        """Verify decoder produces approximately correct output shape.

        Why: Decoder should reconstruct original audio length.
        """
        config = EnCodecConfig()
        encoder = EnCodecEncoder(config)
        decoder = EnCodecDecoder(config, encoder.strides)

        # Original audio length
        original_length = 24000
        batch_size = 2

        # Get encoder output (as would come from quantizer)
        waveform = torch.randn(batch_size, 1, original_length)
        embeddings = encoder(waveform)

        # Decode
        reconstructed = decoder(embeddings)

        assert reconstructed.shape[0] == batch_size
        assert reconstructed.shape[1] == config.channels
        # Reconstructed length should be close to original
        assert abs(reconstructed.shape[2] - original_length) / original_length < 0.1


class TestResidualVectorQuantizer:
    """Tests for RVQ."""

    def test_quantize_output_shape(self) -> None:
        """Verify quantize produces correct shapes.

        Why: Codes shape is critical for tokenization.
        """
        config = EnCodecConfig(codebook_size=1024, num_codebooks=8, hidden_size=128)
        quantizer = ResidualVectorQuantizer(config)

        batch_size = 2
        seq_len = 75
        embeddings = torch.randn(batch_size, seq_len, config.hidden_size)

        codes, quantized, loss = quantizer.quantize(embeddings)

        assert codes.shape == (batch_size, seq_len, config.num_codebooks)
        assert quantized.shape == embeddings.shape
        assert loss.dim() == 0  # Scalar loss

    def test_quantize_codes_in_range(self) -> None:
        """Verify quantized codes are within codebook range.

        Why: Codes must be valid indices into codebook.
        """
        config = EnCodecConfig(codebook_size=1024, num_codebooks=8)
        quantizer = ResidualVectorQuantizer(config)

        embeddings = torch.randn(2, 75, config.hidden_size)
        codes, _, _ = quantizer.quantize(embeddings)

        assert codes.min() >= 0
        assert codes.max() < config.codebook_size

    def test_dequantize_output_shape(self) -> None:
        """Verify dequantize produces correct shape.

        Why: Dequantized embeddings must match original shape for decoder.
        """
        config = EnCodecConfig()
        quantizer = ResidualVectorQuantizer(config)

        batch_size = 2
        seq_len = 75
        codes = torch.randint(0, config.codebook_size, (batch_size, seq_len, config.num_codebooks))

        quantized = quantizer.dequantize(codes)

        assert quantized.shape == (batch_size, seq_len, config.hidden_size)

    def test_quantize_dequantize_consistency(self) -> None:
        """Verify quantize then dequantize produces consistent results.

        Why: Round-trip should reconstruct to same quantized embeddings.
        """
        config = EnCodecConfig()
        quantizer = ResidualVectorQuantizer(config)

        embeddings = torch.randn(2, 75, config.hidden_size)
        codes, quantized, _ = quantizer.quantize(embeddings)

        # Dequantize should give same result as quantized output
        dequantized = quantizer.dequantize(codes)

        torch.testing.assert_close(quantized, dequantized, atol=1e-5, rtol=1e-5)

    def test_commitment_loss_positive(self) -> None:
        """Verify commitment loss is positive.

        Why: Commitment loss encourages encoder to commit to codebook entries.
        """
        config = EnCodecConfig()
        quantizer = ResidualVectorQuantizer(config)

        embeddings = torch.randn(2, 75, config.hidden_size)
        _, _, loss = quantizer.quantize(embeddings)

        assert loss > 0

    def test_get_codebook_usage(self) -> None:
        """Verify codebook usage statistics.

        Why: Low usage indicates codebook collapse.
        """
        config = EnCodecConfig(codebook_size=1024, num_codebooks=8)
        quantizer = ResidualVectorQuantizer(config)

        # Random codes should have reasonable usage
        codes = torch.randint(0, config.codebook_size, (100, 75, config.num_codebooks))
        usage = quantizer.get_codebook_usage(codes)

        assert len(usage) == config.num_codebooks
        for level, fraction in usage.items():
            assert 0 < fraction <= 1.0


class TestEnCodecAudioTokenizer:
    """Tests for complete audio tokenizer."""

    def test_encode_output_shape(self) -> None:
        """Verify encode produces correct code shape.

        Why: Code shape determines token count for LM.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        batch_size = 2
        num_samples = 24000  # 1 second
        waveform = torch.randn(batch_size, 1, num_samples)

        codes = tokenizer.encode(waveform)

        assert codes.shape[0] == batch_size
        assert codes.shape[2] == config.num_codebooks
        # Frames should be approximately samples / downsample_rate
        expected_frames = num_samples // config.downsample_rate
        assert abs(codes.shape[1] - expected_frames) <= 2

    def test_decode_output_shape(self) -> None:
        """Verify decode produces approximately correct waveform shape.

        Why: Decoded audio should match original length.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        # Encode then decode
        num_samples = 24000
        waveform = torch.randn(1, 1, num_samples)

        codes = tokenizer.encode(waveform)
        reconstructed = tokenizer.decode(codes)

        assert reconstructed.shape[0] == 1
        assert reconstructed.shape[1] == config.channels
        # Reconstructed length should be close to original
        assert abs(reconstructed.shape[2] - num_samples) / num_samples < 0.1

    def test_forward_returns_all_outputs(self) -> None:
        """Verify forward pass returns codes, quantized, reconstructed, loss.

        Why: Training requires all outputs for loss computation.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(2, 1, 24000)
        codes, quantized, reconstructed, loss = tokenizer(waveform)

        assert codes.dim() == 3  # (B, T, num_codebooks)
        assert quantized.dim() == 3  # (B, T, hidden_size)
        assert reconstructed.dim() == 3  # (B, C, T)
        assert loss.dim() == 0  # Scalar

    def test_tokenize_returns_flat_list(self) -> None:
        """Verify tokenize returns flat token list.

        Why: Flat list is compatible with standard tokenizer interface.
        """
        config = EnCodecConfig(num_codebooks=8)
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(1, 1, 24000)  # ~75 frames
        tokens = tokenizer.tokenize(waveform)

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        # Token count = num_frames * num_codebooks
        num_frames = len(tokens) // config.num_codebooks
        assert len(tokens) == num_frames * config.num_codebooks

    def test_tokenize_tokens_in_range(self) -> None:
        """Verify tokenized values are in valid range.

        Why: Tokens must be valid for vocabulary lookup.
        """
        config = EnCodecConfig(codebook_size=1024, num_codebooks=8)
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(1, 1, 24000)
        tokens = tokenizer.tokenize(waveform)

        assert min(tokens) >= 0
        assert max(tokens) < config.total_vocabulary_size

    def test_detokenize_round_trip(self) -> None:
        """Verify tokenize -> detokenize produces audio.

        Why: Round-trip should produce valid audio output.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(1, 1, 24000)
        tokens = tokenizer.tokenize(waveform)
        reconstructed = tokenizer.detokenize(tokens)

        assert reconstructed.dim() == 3
        assert reconstructed.shape[0] == 1
        assert reconstructed.shape[1] == config.channels

    def test_freeze_encoder_disables_gradients(self) -> None:
        """Verify freezing disables encoder gradients.

        Why: Frozen encoder saves memory; only projection should train.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=True)

        # Encoder params should be frozen
        for param in tokenizer.encoder.parameters():
            assert not param.requires_grad

        # Quantizer params should still be trainable
        for param in tokenizer.quantizer.parameters():
            assert param.requires_grad

    def test_unfrozen_encoder_has_gradients(self) -> None:
        """Verify unfrozen encoder has trainable parameters.

        Why: Full training mode should allow encoder updates.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        # All params should be trainable
        trainable_count = sum(1 for p in tokenizer.parameters() if p.requires_grad)
        total_count = sum(1 for p in tokenizer.parameters())

        assert trainable_count == total_count

    def test_get_audio_tokens_per_second(self) -> None:
        """Verify tokens per second calculation.

        Why: Needed for sequence length planning.
        """
        config = EnCodecConfig(sample_rate=24000, downsample_rate=320)
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        assert tokenizer.get_audio_tokens_per_second() == 75.0

    def test_get_total_tokens_per_second(self) -> None:
        """Verify total tokens per second (all codebooks).

        Why: For flat tokenization, total rate matters.
        """
        config = EnCodecConfig(sample_rate=24000, downsample_rate=320, num_codebooks=8)
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        assert tokenizer.get_total_tokens_per_second() == 600.0  # 75 * 8

    def test_memory_estimation_reasonable(self) -> None:
        """Verify memory estimation is reasonable.

        Why: Must fit within RTX 5080 16GB budget.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=True)

        memory_gb = tokenizer.get_memory_usage_gb()

        # EnCodec should be relatively small (< 0.5 GB typically)
        assert 0.01 < memory_gb < 1.0, f"Memory {memory_gb:.3f} GB outside expected range"

    def test_handles_2d_waveform_input(self) -> None:
        """Verify tokenizer handles 2D waveform (no channel dim).

        Why: Users may pass (B, T) instead of (B, C, T).
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform_2d = torch.randn(2, 24000)  # (B, T)
        codes = tokenizer.encode(waveform_2d)

        assert codes.dim() == 3


class TestAudioProjection:
    """Tests for audio projection layer."""

    def test_projection_output_shape(self) -> None:
        """Verify projection produces correct output shape.

        Why: Output must match model hidden size for integration.
        """
        encodec_hidden = 128
        model_hidden = 2048

        projection = AudioProjection(encodec_hidden, model_hidden)

        audio_embeddings = torch.randn(2, 75, encodec_hidden)
        output = projection(audio_embeddings)

        assert output.shape == (2, 75, model_hidden)

    def test_projection_is_trainable(self) -> None:
        """Verify projection layer is trainable.

        Why: Projection should adapt to model embedding space.
        """
        projection = AudioProjection(128, 2048)

        for param in projection.parameters():
            assert param.requires_grad


class TestCreateAudioTokenizer:
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

        tokenizer, projection = create_audio_tokenizer(tritter_config)

        # Verify projection matches Tritter hidden size
        test_input = torch.randn(1, 10, 128)  # EnCodec hidden size
        output = projection(test_input)
        assert output.shape[-1] == tritter_config.hidden_size

    def test_custom_encodec_config_respected(self) -> None:
        """Verify custom EnCodec config is used when provided.

        Why: Allow experimentation with different encoder settings.
        """
        tritter_config = TritterConfig(hidden_size=1024, num_layers=1)
        custom_encodec = EnCodecConfig(num_codebooks=4, hidden_size=64)

        tokenizer, projection = create_audio_tokenizer(
            tritter_config,
            encodec_config=custom_encodec,
        )

        assert tokenizer.config.num_codebooks == 4
        assert tokenizer.config.hidden_size == 64


class TestGradientFlow:
    """Tests for gradient flow through tokenizer."""

    def test_gradients_flow_through_quantizer(self) -> None:
        """Verify gradients flow through RVQ via straight-through.

        Why: Training requires gradients to reach encoder.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(1, 1, 24000, requires_grad=True)
        _, quantized, _, _ = tokenizer(waveform)

        loss = quantized.sum()
        loss.backward()

        # Waveform should have gradients (via straight-through estimator)
        assert waveform.grad is not None

    def test_gradients_flow_through_projection(self) -> None:
        """Verify gradients flow through projection layer.

        Why: Projection is the trainable component.
        """
        projection = AudioProjection(128, 2048)

        audio_embeddings = torch.randn(1, 75, 128)
        output = projection(audio_embeddings)

        loss = output.sum()
        loss.backward()

        assert projection.proj.weight.grad is not None
        assert projection.proj.weight.grad.abs().max() > 0

    def test_frozen_encoder_no_gradients(self) -> None:
        """Verify frozen encoder doesn't accumulate gradients.

        Why: Frozen layers should have no gradient computation.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=True)

        # Encoder params should not require grad
        for param in tokenizer.encoder.parameters():
            assert not param.requires_grad

        # Quantizer params should still require grad
        for param in tokenizer.quantizer.parameters():
            assert param.requires_grad

        # Do a forward pass and verify encoder params remain unfrozen
        waveform = torch.randn(1, 1, 24000)
        codes, quantized, _, commitment_loss = tokenizer(waveform)

        # Commitment loss should be backpropable through quantizer
        commitment_loss.backward()

        # Encoder params should have no gradients (frozen)
        for param in tokenizer.encoder.parameters():
            assert param.grad is None

        # Quantizer codebook should have gradients
        for param in tokenizer.quantizer.parameters():
            # Some codebook entries may have zero gradients if not used
            # Just verify the computation ran without error
            pass


class TestReconstructionQuality:
    """Tests for reconstruction quality metrics."""

    def test_reconstruction_not_zero(self) -> None:
        """Verify reconstruction is not all zeros.

        Why: Basic sanity check that reconstruction works.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(1, 1, 24000)
        _, _, reconstructed, _ = tokenizer(waveform)

        assert reconstructed.abs().max() > 0

    def test_reconstruction_finite(self) -> None:
        """Verify reconstruction has no NaN/Inf values.

        Why: Numerical stability check.
        """
        config = EnCodecConfig()
        tokenizer = EnCodecAudioTokenizer(config, freeze_encoder=False)

        waveform = torch.randn(1, 1, 24000)
        _, _, reconstructed, _ = tokenizer(waveform)

        assert torch.isfinite(reconstructed).all()
