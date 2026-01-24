"""Audio encoder module for multimodal audio understanding.

Provides EnCodec-style audio tokenization for early fusion multimodal architecture.

Why (Embedding-Prediction Paradigm):
    Audio enters the model through the EnCodec encoder, which produces continuous
    embeddings that are quantized via Residual Vector Quantization (RVQ) and then
    projected into the shared embedding space. This enables cross-modal attention
    between text, image, and audio representations. The audio encoder is frozen
    (pretrained) while only the projection layer is trained, following the same
    efficient adaptation pattern as SigLIP for vision.

Reference: EnCodec uses convolutional architecture with RVQ for high-fidelity
audio compression at 75 tokens/second (24kHz, downsample=320). This rate makes
audio sequences manageable for 128K context windows.
See docs/project-plan.md for multimodal architecture details.
"""

from tritter.audio.encodec import (
    AudioProjection,
    EnCodecAudioTokenizer,
    EnCodecConfig,
    EnCodecDecoder,
    EnCodecEncoder,
    ResidualVectorQuantizer,
    create_audio_tokenizer,
)

__all__ = [
    "EnCodecConfig",
    "EnCodecEncoder",
    "EnCodecDecoder",
    "ResidualVectorQuantizer",
    "EnCodecAudioTokenizer",
    "AudioProjection",
    "create_audio_tokenizer",
]
