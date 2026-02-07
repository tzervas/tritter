"""EnCodec-style audio encoder for multimodal audio understanding.

Implements EnCodec-style audio tokenization with Residual Vector Quantization (RVQ)
for converting audio waveforms to discrete tokens. The encoder is frozen; only the
projection layer trains.

Why: Audio is a critical modality for multimodal AI. EnCodec/SoundStream architectures
use convolutional encoders with RVQ to compress audio to discrete tokens at high
fidelity. At 24kHz sample rate with downsample_rate=320, we get 75 tokens/second,
making audio sequences manageable for transformer processing. This follows the same
pattern as SigLIP for vision - frozen encoder with trainable projection.

Embedding-Prediction Context: Audio waveforms are encoded to continuous embeddings
via the convolutional encoder, then quantized to discrete codes via RVQ. For the
Tritter model, these codes are projected to the shared embedding space where they
can attend to text and image tokens. The projection layer learns to align audio
semantics with the model's embedding space.

Reference: docs/project-plan.md, SPEC-010-lora-finetuning.md (multimodal support)
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
class EnCodecConfig:
    """Configuration for EnCodec audio encoder.

    Why: Centralized configuration enables easy experimentation with different
    encoder architectures while maintaining consistent interface. The default
    settings (24kHz, 75 tokens/sec) balance audio quality and sequence length
    for transformer processing.

    Embedding-Prediction Context: The codebook_size and num_codebooks determine
    the discrete token vocabulary for audio. With 8 codebooks of 1024 entries each,
    we have fine-grained control over audio reconstruction quality while keeping
    tokens within manageable vocabulary size.

    Attributes:
        sample_rate: Audio sample rate in Hz, default 24000 (24kHz standard for
            speech/music balance between quality and compute)
        channels: Number of audio channels, default 1 (mono for simplicity)
        hidden_size: Encoder hidden dimension, default 128 (compact for efficiency)
        num_layers: Number of encoder convolutional layers, default 4
        codebook_size: Number of entries per codebook, default 1024
        num_codebooks: Number of RVQ codebooks (residual levels), default 8
        downsample_rate: Total downsampling factor, default 320
            (24000/320 = 75 tokens/second)
        kernel_size: Convolution kernel size, default 7
    """

    sample_rate: int = 24000
    channels: int = 1
    hidden_size: int = 128
    num_layers: int = 4
    codebook_size: int = 1024
    num_codebooks: int = 8
    downsample_rate: int = 320
    kernel_size: int = 7

    @property
    def tokens_per_second(self) -> float:
        """Number of audio tokens generated per second of audio.

        Why: Determines sequence length for transformer processing. At 75 tokens/sec,
        a 10-second audio clip becomes 750 tokens, manageable for 128K context.
        """
        return self.sample_rate / self.downsample_rate

    @property
    def hop_length(self) -> int:
        """Samples between adjacent tokens.

        Why: Equivalent to downsample_rate, useful for audio frame calculations.
        """
        return self.downsample_rate

    @property
    def total_vocabulary_size(self) -> int:
        """Total vocabulary size across all codebooks.

        Why: With flattened tokenization, each codebook level uses separate
        token IDs. Total = codebook_size * num_codebooks.
        """
        return self.codebook_size * self.num_codebooks


class ConvBlock(nn.Module):  # type: ignore[misc]
    """Convolutional block with normalization and activation.

    Why: Basic building block for encoder/decoder. Uses ELU activation which
    is common in audio processing (smooth gradient flow, handles negative values).
    Weight normalization improves training stability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        transpose: bool = False,
    ) -> None:
        """Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride (for downsampling/upsampling)
            padding: Padding size (auto-calculated if None)
            transpose: If True, use transposed convolution (for decoder)

        Why: Padding is auto-calculated to maintain temporal structure.
        Transposed convolution enables upsampling in decoder.
        """
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        conv: nn.Module
        if transpose:
            # Transposed conv for upsampling
            # Output padding ensures correct output size after transposed conv
            output_padding = stride - 1 if stride > 1 else 0
            conv = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        else:
            conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )

        # Weight normalization for training stability
        self.conv = nn.utils.weight_norm(conv)
        self.activation = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through conv block.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, C_out, T_out)
        """
        result: Tensor = self.activation(self.conv(x))
        return result


class ResidualUnit(nn.Module):  # type: ignore[misc]
    """Residual unit with dilated convolution for temporal modeling.

    Why: Residual connections enable deep networks without vanishing gradients.
    Dilated convolutions increase receptive field without increasing parameters,
    crucial for capturing long-range audio dependencies.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        """Initialize residual unit.

        Args:
            channels: Number of input/output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for increased receptive field

        Why: Dilation expands receptive field exponentially with depth.
        """
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1)  # 1x1 conv
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.activation = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, C, T)
        """
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class EncoderBlock(nn.Module):  # type: ignore[misc]
    """Encoder block with residual units and downsampling.

    Why: Each encoder block processes audio at a specific resolution, then
    downsamples. Multiple blocks progressively compress the temporal dimension
    while expanding channel dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_residuals: int = 3,
    ) -> None:
        """Initialize encoder block.

        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            kernel_size: Kernel size for downsampling conv
            stride: Downsampling stride
            num_residuals: Number of residual units before downsampling

        Why: Residual units before downsampling allow learning at current resolution.
        """
        super().__init__()

        # Residual units at current resolution
        self.residuals = nn.Sequential(
            *[ResidualUnit(in_channels, kernel_size=3, dilation=3**i) for i in range(num_residuals)]
        )

        # Downsampling convolution
        self.downsample = ConvBlock(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residuals and downsampling.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Downsampled tensor (B, C_out, T // stride)
        """
        x = self.residuals(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):  # type: ignore[misc]
    """Decoder block with upsampling and residual units.

    Why: Mirror of encoder block for reconstruction. Upsamples first, then
    applies residual units to refine at higher resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_residuals: int = 3,
    ) -> None:
        """Initialize decoder block.

        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            kernel_size: Kernel size for upsampling conv
            stride: Upsampling stride
            num_residuals: Number of residual units after upsampling
        """
        super().__init__()

        # Upsampling via transposed convolution
        self.upsample = ConvBlock(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, transpose=True
        )

        # Residual units at higher resolution
        self.residuals = nn.Sequential(
            *[
                ResidualUnit(out_channels, kernel_size=3, dilation=3**i)
                for i in range(num_residuals)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with upsampling and residuals.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Upsampled tensor (B, C_out, T * stride)
        """
        x = self.upsample(x)
        x = self.residuals(x)
        return x


class EnCodecEncoder(nn.Module):  # type: ignore[misc]
    """Convolutional encoder for audio waveforms.

    Why: Converts raw audio waveform to continuous embeddings via progressive
    downsampling. Uses 1D convolutions (not transformers) for efficiency and
    locality bias appropriate for audio signals. The output embeddings are
    ready for RVQ quantization.

    Embedding-Prediction Context: The encoder produces continuous embeddings
    that will be quantized by RVQ. These embeddings capture audio semantics
    in a compressed representation suitable for transformer processing.
    """

    def __init__(self, config: EnCodecConfig) -> None:
        """Initialize encoder.

        Args:
            config: EnCodecConfig with architecture settings

        Why: Progressive downsampling via multiple encoder blocks. Each block
        doubles channels and halves temporal dimension (approximately).
        """
        super().__init__()
        self.config = config

        # Calculate strides for each layer to achieve target downsample_rate
        # downsample_rate = product of all strides
        # For downsample_rate=320 with 4 layers: strides could be [5, 4, 4, 4] = 320
        self.strides = self._calculate_strides(config.downsample_rate, config.num_layers)

        # Initial projection from audio channels to hidden_size
        self.input_conv = ConvBlock(config.channels, config.hidden_size, config.kernel_size)

        # Encoder blocks with progressive downsampling
        channels = config.hidden_size
        self.encoder_blocks = nn.ModuleList()
        for stride in self.strides:
            out_channels = min(channels * 2, config.hidden_size * 8)
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_size=stride * 2,
                    stride=stride,
                )
            )
            channels = out_channels

        # LSTM for temporal modeling at compressed resolution
        # Why: Captures long-range dependencies after compression
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Project LSTM output (bidirectional = 2x channels) back to hidden_size
        self.output_proj = nn.Linear(channels * 2, config.hidden_size)

        self._output_channels = channels

    def _calculate_strides(self, target_rate: int, num_layers: int) -> list[int]:
        """Calculate strides to achieve target downsample rate.

        Why: Distributes downsampling across layers. Aims for balanced strides
        rather than one large stride, which helps preserve information.
        """
        # Factor target_rate into roughly equal strides
        strides = []
        remaining = target_rate

        for i in range(num_layers):
            # Try to find a factor that leaves a reasonable remaining
            stride = int(round(remaining ** (1 / (num_layers - i))))
            stride = max(2, min(stride, remaining))

            # Ensure we can achieve exact target (adjust last stride)
            if i == num_layers - 1:
                stride = remaining
            else:
                # Find closest factor
                while remaining % stride != 0 and stride > 1:
                    stride -= 1
                if stride < 2:
                    stride = 2

            strides.append(stride)
            remaining = remaining // stride if remaining % stride == 0 else remaining

        # Verify product equals target
        product = 1
        for s in strides:
            product *= s

        # If we didn't hit target exactly, adjust last stride
        if product != target_rate and strides:
            # Find the adjustment needed
            ratio = target_rate / product
            strides[-1] = int(strides[-1] * ratio)

        return strides

    def forward(self, waveform: Tensor) -> Tensor:
        """Encode audio waveform to embeddings.

        Args:
            waveform: Audio tensor (B, 1, T) where T = num_samples

        Returns:
            Embeddings tensor (B, T // downsample_rate, hidden_size)

        Why: Progressive downsampling reduces temporal dimension while LSTM
        captures global context. Output shape is suitable for RVQ quantization.
        """
        # Initial projection: (B, 1, T) -> (B, hidden_size, T)
        x = self.input_conv(waveform)

        # Progressive downsampling through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)  # (B, C, T) -> (B, 2C, T//stride)

        # Transpose for LSTM: (B, C, T') -> (B, T', C)
        x = x.transpose(1, 2)

        # LSTM for temporal modeling
        x, _ = self.lstm(x)  # (B, T', 2C) due to bidirectional

        # Project to hidden_size: (B, T', 2C) -> (B, T', hidden_size)
        embeddings: Tensor = self.output_proj(x)

        return embeddings  # (B, T // downsample_rate, hidden_size)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length for given input length.

        Args:
            input_length: Number of audio samples

        Returns:
            Number of output frames (tokens)

        Why: Useful for computing expected token count before encoding.
        """
        length = input_length
        for stride in self.strides:
            length = length // stride
        return length


class EnCodecDecoder(nn.Module):  # type: ignore[misc]
    """Convolutional decoder for waveform reconstruction.

    Why: Mirror of encoder, reconstructs audio waveform from embeddings.
    Used for training the encoder via reconstruction loss and for audio
    generation at inference time.

    Embedding-Prediction Context: Converts discrete codes back to continuous
    embeddings, then upsamples to reconstruct the original waveform. Enables
    round-trip audio → tokens → audio for audio generation.
    """

    def __init__(self, config: EnCodecConfig, encoder_strides: list[int]) -> None:
        """Initialize decoder.

        Args:
            config: EnCodecConfig with architecture settings
            encoder_strides: Strides from encoder (reversed for upsampling)

        Why: Uses reversed encoder strides for symmetric architecture.
        """
        super().__init__()
        self.config = config
        self.strides = list(reversed(encoder_strides))

        # Calculate channel progression (reverse of encoder)
        channels = config.hidden_size
        channel_list = [channels]
        for _ in encoder_strides:
            channels = min(channels * 2, config.hidden_size * 8)
            channel_list.append(channels)

        channel_list = list(reversed(channel_list))

        # Input projection from hidden_size to decoder input channels
        self.input_proj = nn.Linear(config.hidden_size, channel_list[0])

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=channel_list[0],
            hidden_size=channel_list[0] // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Decoder blocks with progressive upsampling
        self.decoder_blocks = nn.ModuleList()
        for i, stride in enumerate(self.strides):
            in_channels = channel_list[i]
            out_channels = channel_list[i + 1]
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stride * 2,
                    stride=stride,
                )
            )

        # Final projection to audio channels
        self.output_conv = nn.Conv1d(
            config.hidden_size,
            config.channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """Decode embeddings to audio waveform.

        Args:
            embeddings: Tensor (B, T', hidden_size)

        Returns:
            Reconstructed waveform (B, channels, T) approximately

        Why: LSTM processes embeddings for global context, then decoder blocks
        progressively upsample to original audio resolution.
        """
        # Input projection: (B, T', hidden_size) -> (B, T', C)
        x = self.input_proj(embeddings)

        # LSTM for temporal modeling
        x, _ = self.lstm(x)

        # Transpose for conv: (B, T', C) -> (B, C, T')
        x = x.transpose(1, 2)

        # Progressive upsampling through decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Final projection to audio: (B, hidden_size, T) -> (B, channels, T)
        waveform: Tensor = self.output_conv(x)

        return waveform


class ResidualVectorQuantizer(nn.Module):  # type: ignore[misc]
    """Residual Vector Quantizer for audio compression.

    Why: RVQ enables high-fidelity discrete representation of continuous embeddings.
    Each codebook level quantizes the residual from previous levels, progressively
    refining the reconstruction. This is more effective than single VQ for audio
    where high quality matters.

    Embedding-Prediction Context: RVQ converts continuous encoder embeddings to
    discrete codes. For the embedding-prediction paradigm, these codes will eventually
    be replaced by direct embedding prediction (KNN/VQ rounding in embedding space).
    Currently, discrete codes are used for training compatibility with standard LM loss.
    """

    def __init__(self, config: EnCodecConfig) -> None:
        """Initialize RVQ.

        Args:
            config: EnCodecConfig with codebook settings

        Why: Multiple codebooks enable progressive refinement. Each codebook
        learns to encode the residual error from previous codebooks.
        """
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.hidden_size = config.hidden_size

        # Codebook embeddings for each quantization level
        # Why: Each codebook is independent, allowing specialization at each level
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size, config.hidden_size)
                for _ in range(config.num_codebooks)
            ]
        )

        # Initialize codebooks uniformly
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1 / config.codebook_size, 1 / config.codebook_size)

    def quantize(self, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize embeddings using residual vector quantization.

        Args:
            embeddings: Continuous embeddings (B, T, hidden_size)

        Returns:
            Tuple of:
                - codes: Quantized indices (B, T, num_codebooks)
                - quantized: Quantized embeddings (B, T, hidden_size)
                - commitment_loss: Loss for training encoder to commit to codes

        Why: Multi-level quantization progressively reduces reconstruction error.
        Each level quantizes the residual from previous levels. Commitment loss
        encourages encoder outputs to be close to codebook entries.

        Embedding-Prediction Context: Codes are discrete tokens suitable for
        language modeling. Quantized embeddings are continuous representations
        that can be projected to model embedding space.
        """
        B, T, D = embeddings.shape

        all_codes = []
        quantized = torch.zeros_like(embeddings)
        residual = embeddings.clone()
        total_commitment_loss = torch.tensor(0.0, device=embeddings.device)

        for codebook in self.codebooks:
            # Find nearest codebook entry for each embedding
            # residual: (B, T, D), codebook.weight: (K, D)
            distances = torch.cdist(
                residual.view(-1, D),  # (B*T, D)
                codebook.weight,  # (K, D)
            )  # (B*T, K)

            # Get nearest codebook indices
            indices = distances.argmin(dim=-1)  # (B*T,)
            all_codes.append(indices.view(B, T))

            # Get quantized vectors from codebook
            quantized_level = codebook(indices).view(B, T, D)  # (B, T, D)

            # Accumulate quantized embeddings
            quantized = quantized + quantized_level

            # Compute commitment loss for this level
            # Why: Encourages encoder to produce embeddings close to codebook entries
            commitment_loss = F.mse_loss(residual.detach(), quantized_level)
            total_commitment_loss = total_commitment_loss + commitment_loss

            # Update residual for next codebook
            residual = residual - quantized_level.detach()

        # Stack codes: (num_codebooks, B, T) -> (B, T, num_codebooks)
        codes = torch.stack(all_codes, dim=-1)

        # Straight-through estimator: use quantized in forward, but pass gradients to embeddings
        # Why: Enables gradient flow through discrete quantization
        quantized = embeddings + (quantized - embeddings).detach()

        return codes, quantized, total_commitment_loss / self.num_codebooks

    def dequantize(self, codes: Tensor) -> Tensor:
        """Convert discrete codes back to continuous embeddings.

        Args:
            codes: Quantized indices (B, T, num_codebooks)

        Returns:
            Quantized embeddings (B, T, hidden_size)

        Why: Reconstructs continuous representation from discrete codes
        for decoder input or embedding projection.
        """
        B, T, num_cb = codes.shape
        assert num_cb == self.num_codebooks, (
            f"Expected {self.num_codebooks} codebook levels, got {num_cb}"
        )

        quantized = torch.zeros(B, T, self.hidden_size, device=codes.device)

        for i, codebook in enumerate(self.codebooks):
            code_level = codes[:, :, i]  # (B, T)
            quantized_level = codebook(code_level)  # (B, T, D)
            quantized = quantized + quantized_level

        return quantized

    def get_codebook_usage(self, codes: Tensor) -> dict[int, float]:
        """Analyze codebook usage statistics.

        Args:
            codes: Quantized indices (B, T, num_codebooks)

        Returns:
            Dictionary mapping codebook level to usage fraction

        Why: Low usage indicates codebook collapse, a training issue.
        """
        usage = {}
        for i in range(self.num_codebooks):
            level_codes = codes[:, :, i].flatten()
            unique_codes = torch.unique(level_codes).numel()
            usage[i] = unique_codes / self.codebook_size
        return usage


class EnCodecAudioTokenizer(nn.Module):  # type: ignore[misc]
    """Complete audio tokenizer combining encoder and RVQ.

    Why: Main interface for audio tokenization. Combines convolutional encoder
    with residual vector quantization to convert audio waveforms to discrete
    tokens. Follows the same pattern as SigLIP for vision - frozen encoder
    with trainable projection.

    Embedding-Prediction Context: Audio is encoded to continuous embeddings,
    then quantized to discrete codes. The codes can be flattened to a token
    sequence for language modeling, or the quantized embeddings can be projected
    to model embedding space for early fusion multimodal attention.

    Attributes:
        config: EnCodecConfig for encoder architecture
        encoder: Convolutional encoder stack
        decoder: Convolutional decoder for reconstruction
        quantizer: Residual vector quantizer
        freeze_encoder: Whether encoder weights are frozen

    Example:
        >>> config = EnCodecConfig()
        >>> tokenizer = EnCodecAudioTokenizer(config)
        >>> waveform = torch.randn(2, 1, 48000)  # 2 sec @ 24kHz
        >>> codes = tokenizer.encode(waveform)  # (2, 150, 8) - 150 tokens
        >>> reconstructed = tokenizer.decode(codes)  # (2, 1, ~48000)
    """

    def __init__(
        self,
        config: EnCodecConfig,
        freeze_encoder: bool = True,
    ) -> None:
        """Initialize audio tokenizer.

        Args:
            config: EnCodecConfig with encoder architecture settings
            freeze_encoder: If True, freeze encoder weights (only train projection)

        Why: freeze_encoder=True is memory efficient and preserves learned audio
        representations. Only the projection layer needs to adapt for the specific
        LM embedding space (similar to SigLIP approach).
        """
        super().__init__()
        self.config = config
        self.freeze_encoder = freeze_encoder

        # Audio encoder
        self.encoder = EnCodecEncoder(config)

        # Residual vector quantizer
        self.quantizer = ResidualVectorQuantizer(config)

        # Audio decoder for reconstruction
        self.decoder = EnCodecDecoder(config, self.encoder.strides)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters.

        Why: Freezing pretrained encoder saves memory (no gradients stored)
        and preserves learned audio representations.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, waveform: Tensor) -> Tensor:
        """Encode audio waveform to discrete codes.

        Args:
            waveform: Audio tensor (B, C, T) where C=channels, T=samples

        Returns:
            Codes tensor (B, T', num_codebooks) where T' = T // downsample_rate

        Why: Main encoding interface. Returns discrete codes suitable for
        language modeling or storage.
        """
        # Ensure correct channel dimension
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # Encode to continuous embeddings
        embeddings = self.encoder(waveform)  # (B, T', hidden_size)

        # Quantize to discrete codes
        codes, _, _ = self.quantizer.quantize(embeddings)  # (B, T', num_codebooks)

        return codes

    def decode(self, codes: Tensor) -> Tensor:
        """Decode discrete codes to audio waveform.

        Args:
            codes: Discrete codes (B, T', num_codebooks)

        Returns:
            Reconstructed waveform (B, channels, T) approximately

        Why: Enables audio reconstruction from tokens for generation.
        """
        # Dequantize to continuous embeddings
        embeddings = self.quantizer.dequantize(codes)  # (B, T', hidden_size)

        # Decode to waveform
        waveform: Tensor = self.decoder(embeddings)  # (B, channels, ~T)

        return waveform

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass with reconstruction.

        Args:
            waveform: Audio tensor (B, C, T)

        Returns:
            Tuple of:
                - codes: Discrete codes (B, T', num_codebooks)
                - quantized: Quantized embeddings (B, T', hidden_size)
                - reconstructed: Reconstructed waveform (B, C, ~T)
                - commitment_loss: Commitment loss for training

        Why: Full forward needed for training with reconstruction loss.
        """
        # Ensure correct channel dimension
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Encode
        embeddings = self.encoder(waveform)

        # Quantize
        codes, quantized, commitment_loss = self.quantizer.quantize(embeddings)

        # Decode
        reconstructed = self.decoder(quantized)

        return codes, quantized, reconstructed, commitment_loss

    def tokenize(self, waveform: Tensor) -> list[int]:
        """Tokenize audio to flat token sequence.

        Args:
            waveform: Audio tensor (1, C, T) - single sample

        Returns:
            List of token IDs (flattened across time and codebooks)

        Why: Flattened sequence is compatible with standard tokenizer interface.
        Tokens are ordered: [t0_cb0, t0_cb1, ..., t0_cbN, t1_cb0, t1_cb1, ...]

        Embedding-Prediction Context: This flat token sequence can be used with
        standard language model training. Each codebook level gets its own
        token offset to distinguish levels: token_id = level * codebook_size + code.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        codes = self.encode(waveform)  # (1, T', num_codebooks)
        codes = codes[0]  # (T', num_codebooks)

        # Flatten with codebook offsets
        tokens = []
        for t in range(codes.size(0)):
            for cb in range(self.config.num_codebooks):
                # Add offset for each codebook level
                token_id = cb * self.config.codebook_size + codes[t, cb].item()
                tokens.append(int(token_id))

        return tokens

    def detokenize(self, tokens: list[int]) -> Tensor:
        """Convert flat token sequence back to audio.

        Args:
            tokens: List of token IDs (length must be multiple of num_codebooks)

        Returns:
            Reconstructed waveform tensor (1, channels, T)

        Why: Inverse of tokenize() for audio generation from LM output.
        """
        num_frames = len(tokens) // self.config.num_codebooks
        assert len(tokens) == num_frames * self.config.num_codebooks, (
            f"Token count {len(tokens)} not divisible by num_codebooks {self.config.num_codebooks}"
        )

        # Reconstruct codes tensor
        codes = torch.zeros(1, num_frames, self.config.num_codebooks, dtype=torch.long)
        idx = 0
        for t in range(num_frames):
            for cb in range(self.config.num_codebooks):
                # Remove codebook offset
                code = tokens[idx] - cb * self.config.codebook_size
                codes[0, t, cb] = code
                idx += 1

        return self.decode(codes)

    def get_audio_tokens_per_second(self) -> float:
        """Get number of audio tokens generated per second of audio.

        Returns:
            Tokens per second (per codebook level)

        Why: Useful for computing sequence length given audio duration.
        With 8 codebooks, total tokens/sec = tokens_per_second * num_codebooks.
        """
        return self.config.tokens_per_second

    def get_total_tokens_per_second(self) -> float:
        """Get total tokens per second across all codebooks.

        Returns:
            Total tokens per second (all codebook levels combined)

        Why: For flat tokenization, this is the actual token rate.
        """
        return self.config.tokens_per_second * self.config.num_codebooks

    def get_memory_usage_gb(self) -> float:
        """Estimate memory usage of the tokenizer.

        Returns:
            Approximate memory in GB

        Why: Memory budgeting for RTX 5080 16GB constraint.
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Assume FP32 (4 bytes per param)
        return total_params * 4 / 1e9  # type: ignore[no-any-return]


class AudioProjection(nn.Module):  # type: ignore[misc]
    """Project audio embeddings to model embedding space.

    Why: EnCodec hidden_size (128) differs from model hidden_size (e.g., 2048).
    This trainable projection aligns audio features with text embedding space,
    enabling cross-modal attention.

    Embedding-Prediction Context: Projects quantized audio embeddings to the
    shared embedding space where they can attend to text and image tokens.
    Similar to SigLIP's projection layer.
    """

    def __init__(self, encodec_hidden_size: int, model_hidden_size: int) -> None:
        """Initialize projection.

        Args:
            encodec_hidden_size: EnCodec embedding dimension (e.g., 128)
            model_hidden_size: Target model dimension (e.g., 2048)

        Why: Linear projection is simple and effective. Can be extended with
        non-linearity or multi-layer projection if needed.
        """
        super().__init__()
        self.proj = nn.Linear(encodec_hidden_size, model_hidden_size)

    def forward(self, audio_embeddings: Tensor) -> Tensor:
        """Project audio embeddings.

        Args:
            audio_embeddings: Tensor (B, T, encodec_hidden_size)

        Returns:
            Projected tensor (B, T, model_hidden_size)
        """
        result: Tensor = self.proj(audio_embeddings)
        return result


def create_audio_tokenizer(
    tritter_config: TritterConfig,
    encodec_config: EnCodecConfig | None = None,
    freeze_encoder: bool = True,
) -> tuple[EnCodecAudioTokenizer, AudioProjection]:
    """Factory function to create audio tokenizer compatible with Tritter model.

    Args:
        tritter_config: Tritter model configuration (provides hidden_size)
        encodec_config: Optional custom EnCodec config (defaults to standard settings)
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        Tuple of (EnCodecAudioTokenizer, AudioProjection)

    Why: Convenience factory ensures projection dimension matches Tritter model.
    Returns both tokenizer and projection as separate modules for flexibility
    (e.g., tokenizer can be used standalone for audio-only tasks).

    Embedding-Prediction Context: The audio tokenizer converts waveforms to
    discrete codes, while the projection maps the underlying embeddings to
    model space for multimodal fusion.
    """
    if encodec_config is None:
        encodec_config = EnCodecConfig()

    tokenizer = EnCodecAudioTokenizer(
        config=encodec_config,
        freeze_encoder=freeze_encoder,
    )

    projection = AudioProjection(
        encodec_hidden_size=encodec_config.hidden_size,
        model_hidden_size=tritter_config.hidden_size,
    )

    return tokenizer, projection


__all__ = [
    "EnCodecConfig",
    "EnCodecEncoder",
    "EnCodecDecoder",
    "ResidualVectorQuantizer",
    "EnCodecAudioTokenizer",
    "AudioProjection",
    "create_audio_tokenizer",
]
