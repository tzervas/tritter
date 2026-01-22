"""
Multimodal tokenization with unified vocabulary for early fusion.

Supports text, code, image, and audio modalities with a unified embedding space.

Why: Early fusion (Chameleon-style) provides better cross-modal reasoning than late fusion
approaches by allowing the transformer to learn interactions between modalities at every
layer. Unified vocabulary enables any-to-any generation where the model can seamlessly
transition between modalities within a single sequence, essential for tasks like
code-with-comments, documentation-with-diagrams, or audio-transcription-with-code.
"""

from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class ModalityType(Enum):
    """Supported modality types for multimodal processing.

    Why: Explicit enumeration ensures type safety and prevents invalid modality specifications.
    Each modality requires different tokenization strategies (BPE for text, AST-aware for code,
    VQVAE for images, SpeechTokenizer for audio) but must map to the same unified token space
    to enable cross-modal attention and generation.
    """

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"


class MultiModalTokenizer:
    """Unified tokenizer for multiple modalities.

    Implements early fusion by mapping all modalities to a shared token space.
    Each modality gets special prefix tokens to identify its type.

    Why: Prefix tokens allow the model to condition its processing on modality type while
    maintaining a single unified vocabulary. This design follows Chameleon's approach where
    image tokens share vocabulary space with text BPE tokens. The 128K max_length default
    enables repository-level code understanding and long-context multimodal documents.
    The 65K vocab size balances expressiveness (enough for BPE + image codebook + audio codes)
    with memory efficiency on 16GB VRAM (combined with BitNet quantization).
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # Modality prefix tokens
    TEXT_PREFIX = "<text>"
    CODE_PREFIX = "<code>"
    IMAGE_PREFIX = "<image>"
    AUDIO_PREFIX = "<audio>"

    def __init__(
        self,
        vocab_size: int = 65536,
        max_length: int = 131072,
    ) -> None:
        """Initialize multimodal tokenizer.

        Args:
            vocab_size: Size of unified vocabulary (default 65536 = 2^16)
            max_length: Maximum sequence length (default 131072 = 128K tokens)

        Why: vocab_size of 65536 accommodates ~50K text BPE tokens, 8K image VQVAE codes,
        and remaining space for audio tokens and special tokens. max_length of 128K enables
        full repository context and is achievable on RTX 5080 16GB with BitNet quantization
        and INT4 KV-cache compression (per project-plan.md calculations).
        """
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Build special token mappings
        self.special_tokens = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
            self.TEXT_PREFIX: 4,
            self.CODE_PREFIX: 5,
            self.IMAGE_PREFIX: 6,
            self.AUDIO_PREFIX: 7,
        }

        self.modality_prefixes = {
            ModalityType.TEXT: self.TEXT_PREFIX,
            ModalityType.CODE: self.CODE_PREFIX,
            ModalityType.IMAGE: self.IMAGE_PREFIX,
            ModalityType.AUDIO: self.AUDIO_PREFIX,
        }

    def encode(
        self,
        content: Any,
        modality: ModalityType,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode content from any modality to token IDs.

        Args:
            content: Content to encode (varies by modality - str for text/code,
                     tensor/array for image/audio)
            modality: Type of modality being encoded
            add_special_tokens: Whether to add BOS/EOS and modality prefix tokens

        Returns:
            List of token IDs in unified vocabulary space

        Why: Adding modality prefix tokens (e.g., <text>, <code>, <image>) immediately after
        BOS enables the model to apply modality-specific processing in early layers while
        maintaining unified attention. This follows Chameleon's design where knowing the
        modality type helps the model adjust attention patterns and generation strategies.
        Truncation at max_length ensures memory safety during training/inference.
        """
        tokens: list[int] = []

        if add_special_tokens:
            tokens.append(self.special_tokens[self.BOS_TOKEN])
            # Add modality prefix
            prefix = self.modality_prefixes[modality]
            tokens.append(self.special_tokens[prefix])

        # Modality-specific encoding
        if modality == ModalityType.TEXT:
            tokens.extend(self._encode_text(content))
        elif modality == ModalityType.CODE:
            tokens.extend(self._encode_code(content))
        elif modality == ModalityType.IMAGE:
            tokens.extend(self._encode_image(content))
        elif modality == ModalityType.AUDIO:
            tokens.extend(self._encode_audio(content))

        if add_special_tokens:
            tokens.append(self.special_tokens[self.EOS_TOKEN])

        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        return tokens

    def _encode_text(self, text: str) -> list[int]:
        """Encode text using byte-level encoding.

        Args:
            text: Text string to encode

        Returns:
            List of token IDs

        Why: BPE (Byte-Pair Encoding) provides subword tokenization that balances vocabulary
        size with rare word handling. It compresses common words to single tokens while
        decomposing rare words into subwords, achieving better compression than character-level
        (shorter sequences = more content in 128K context) while avoiding the out-of-vocabulary
        issues of word-level tokenization. Essential for multilingual support and code tokens.

        TODO: Replace with proper BPE tokenizer (e.g., tiktoken or HuggingFace tokenizers).
        The current placeholder uses UTF-8 byte-level encoding, which avoids modulo-based
        token collisions while remaining simple and deterministic, but still lacks the
        compression and subword semantics of true BPE tokenization.
        """
        # Encode as UTF-8 bytes to avoid collisions from modulo-based encoding.
        # This requires the unified vocabulary to reserve at least 264 IDs (8 special + 256 bytes).
        if self.vocab_size < 264:
            raise ValueError(
                f"vocab_size={self.vocab_size} is too small for byte-level text encoding; "
                "needs at least 264 (8 special tokens + 256 byte values)."
            )
        # Add offset to avoid collision with special tokens (0-7)
        # Resulting token IDs will be in range [8, 263]
        return [b + 8 for b in text.encode("utf-8")]

    def _encode_code(self, code: str) -> list[int]:
        """Encode source code with AST-aware tokenization.

        Args:
            code: Source code string (Python, Rust, etc.)

        Returns:
            List of token IDs

        Why: AST (Abstract Syntax Tree) aware tokenization respects semantic boundaries
        like function definitions and class declarations rather than arbitrary byte sequences.
        This enables function-level semantic understanding (per project-plan.md's embedding
        prediction goals) and prevents splitting identifiers or keywords mid-token. Critical
        for code completion, refactoring, and understanding control flow. Approaches like
        cAST (EMNLP 2025) and tree-sitter parsing maintain structural integrity.

        TODO: Implement AST-based tokenization using tree-sitter for multi-language support.
        Current implementation treats code as plain text, losing structural information and
        producing suboptimal embeddings for code understanding tasks. Adequate for testing
        the multimodal pipeline but needs replacement for production code generation.
        """
        # For now, treat similar to text
        return self._encode_text(code)

    def _encode_image(self, image: Any) -> list[int]:
        """Encode image using VQVAE tokens.

        Args:
            image: Image tensor or array (expected shape: CxHxW or HxWxC)

        Returns:
            List of token IDs (typically 256-1024 tokens per image)

        Why: VQVAE (Vector Quantized Variational AutoEncoder) compresses images into discrete
        tokens from a learned codebook, enabling integration with transformer vocabularies.
        This approach (used by Chameleon, DALL-E 2) dramatically reduces sequence length vs
        raw pixels (512x512 image → 1024 tokens instead of 262k pixels) while preserving
        semantic content. Critical for fitting images within 128K context budget alongside
        text. The codebook size (typically 8192 tokens) fits within our 65K unified vocab.

        TODO: Implement VQVAE-based image tokenization with pretrained encoder or train custom.
        Current placeholder returns 256 repeated IMAGE_PREFIX tokens, which prevents actual
        image-text interleaving and generation. Adequate for testing unified embedding layer
        but blocks multimodal functionality. Consider SigLIP-B/16 + VQVAE pipeline.
        """
        # Placeholder: return dummy tokens (256 tokens per image as reasonable default)
        return [self.special_tokens[self.IMAGE_PREFIX]] * 256

    def _encode_audio(self, audio: Any) -> list[int]:
        """Encode audio using SpeechTokenizer.

        Args:
            audio: Audio waveform tensor or spectrogram (expected shape: channels x samples)

        Returns:
            List of token IDs (typically 50-200 tokens per second of audio)

        Why: SpeechTokenizer separates semantic content (what was said) from acoustic style
        (how it was said) using Residual Vector Quantization, enabling LLM integration for
        tasks like transcription, audio understanding, and speech generation. This disentangled
        representation is crucial for multimodal AI - the model needs semantic tokens to
        understand content while acoustic tokens preserve speaker identity and prosody.
        Alternative approaches like EnCodec/SoundStream achieve ultra-low bitrate compression
        but don't separate semantic from acoustic as cleanly.

        TODO: Implement SpeechTokenizer-based audio tokenization with pretrained model.
        Current placeholder returns 128 repeated AUDIO_PREFIX tokens, preventing actual
        audio understanding or generation. Blocks audio-to-text and text-to-audio capabilities.
        Consider SpeechTokenizer with 50Hz semantic tokens (2 tokens per 40ms window).
        """
        # Placeholder: return dummy tokens (128 tokens as reasonable default for ~2-3 sec audio)
        return [self.special_tokens[self.AUDIO_PREFIX]] * 128

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs from the unified vocabulary
            skip_special_tokens: Whether to filter out BOS/EOS/PAD/modality prefixes

        Returns:
            Decoded string representation

        Why: Decoding converts model output tokens back to human-readable text. Skipping
        special tokens produces cleaner output for display while preserving them aids
        debugging. Currently only decodes text modality - image/audio tokens would require
        separate decoders (VQVAE decoder, SpeechTokenizer decoder) based on modality prefix.

        Note: This is a simplified placeholder matching the character-level encoding.
        Production implementation needs proper BPE decoding and multi-modal output routing.
        """
        # Simplified decoding for demonstration
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            token_ids = [t for t in token_ids if t not in special_ids]

        # Decode UTF-8 bytes (reverse the +8 offset from encode)
        # Filter out invalid byte values and handle offset
        # Valid range: [8, 263] (8 special tokens + 256 byte values)
        try:
            bytes_list = []
            for t in token_ids:
                if 8 <= t <= 263:  # Valid byte range with offset
                    bytes_list.append(t - 8)
            return bytes(bytes_list).decode("utf-8", errors="ignore")
        except Exception:
            # Fallback for non-byte tokens
            return ""


class UnifiedEmbedding(nn.Module):
    """Unified embedding layer for all modalities.

    Maps tokens from all modalities to a shared embedding space, enabling cross-modal
    attention and any-to-any generation.

    Why: Shared embedding space is fundamental to early fusion multimodal architecture.
    By mapping text tokens, image VQVAE codes, audio tokens, and code AST nodes to the
    same vector space, the transformer can compute attention between any modality pairs
    (e.g., attend from code to documentation images, from audio transcription to code).
    This contrasts with late fusion approaches (like LLaVA) that keep modalities separate
    until final layers, limiting cross-modal reasoning. Chameleon demonstrates this
    approach scales to 34B parameters with strong any-to-any generation capabilities.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ) -> None:
        """Initialize unified embedding.

        Args:
            vocab_size: Size of unified vocabulary (typically 65536)
            embedding_dim: Dimension of embedding vectors (matches model hidden_size)
            padding_idx: Index of padding token (default 0, prevents gradient updates for padding)

        Why: padding_idx=0 ensures padding tokens don't contribute to gradients during
        backpropagation, preventing the model from learning meaningless patterns from padding.
        The embedding dimension must match the transformer's hidden_size to enable direct
        feeding into attention layers without projection.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs from any modality into shared vector space.

        Args:
            token_ids: Token IDs of shape (batch_size, seq_len) containing mixed-modality tokens

        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)

        Why: Single forward pass handles all modalities identically - text BPE tokens,
        image VQVAE codes, audio tokens, and special tokens all get mapped through the same
        embedding matrix. This simplicity enables the transformer to process heterogeneous
        sequences (e.g., [<bos>, <text>, "def", "foo", <image>, 142, 891, <eos>]) without
        modality-specific routing logic. The learned embeddings capture both modality-specific
        semantics and cross-modal relationships during training.
        """
        return self.embedding(token_ids)
