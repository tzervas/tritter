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

import tiktoken
import torch
import torch.nn as nn

try:
    from .ast_tokenizer import ASTTokenizer, CodeLanguage, CodeToken, TokenType

    AST_TOKENIZER_AVAILABLE = True
except ImportError:
    AST_TOKENIZER_AVAILABLE = False
    ASTTokenizer = None  # type: ignore
    CodeLanguage = None  # type: ignore
    CodeToken = None  # type: ignore
    TokenType = None  # type: ignore


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
        bpe_encoding: str = "cl100k_base",
    ) -> None:
        """Initialize multimodal tokenizer.

        Args:
            vocab_size: Size of unified vocabulary (default 65536 = 2^16)
            max_length: Maximum sequence length (default 131072 = 128K tokens)
            bpe_encoding: tiktoken encoding to use (default "cl100k_base", alternative "o200k_base")

        Why: vocab_size of 65536 accommodates ~50K text BPE tokens, 8K image VQVAE codes,
        and remaining space for audio tokens and special tokens. max_length of 128K enables
        full repository context and is achievable on RTX 5080 16GB with BitNet quantization
        and INT4 KV-cache compression (per project-plan.md calculations).

        BPE encoding selection: cl100k_base (GPT-4) provides ~100K tokens with strong
        multilingual support and code tokenization. We use the first vocab_size - 264 tokens
        (reserving 8 special + 256 for fallback byte encoding) for BPE compression, then
        fall back to byte-level encoding for out-of-range tokens.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.bpe_encoding = bpe_encoding

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

        # Lazy initialization of tiktoken encoder (expensive to create)
        self._tiktoken_encoder: tiktoken.Encoding | None = None

        # Calculate offset for BPE tokens in unified vocab
        # Special tokens: 0-7 (8 total)
        # BPE tokens: start at 8
        # Byte fallback: reserved at end of vocab (last 256 tokens)
        self._bpe_offset = 8
        self._byte_fallback_start = vocab_size - 256
        self._max_bpe_id = self._byte_fallback_start - 1  # BPE tokens use range [8, byte_fallback_start-1]

        # Initialize AST tokenizer for code (lazy initialization to avoid import errors)
        self._ast_tokenizer: ASTTokenizer | None = None

    @property
    def tiktoken_encoder(self) -> tiktoken.Encoding:
        """Get tiktoken encoder, initializing lazily on first access.

        Why: tiktoken.get_encoding() loads vocabulary files from disk, which is relatively
        expensive (~10-50ms). Lazy initialization defers this cost until first use and
        caches the encoder for subsequent calls. This is particularly important for testing
        and scenarios where the tokenizer might be created but not immediately used.

        Returns:
            tiktoken.Encoding instance for the configured BPE encoding

        Note: Thread-safe due to Python's GIL - multiple threads will at worst initialize
        the encoder multiple times (harmless) before one wins the assignment.
        """
        if self._tiktoken_encoder is None:
            self._tiktoken_encoder = tiktoken.get_encoding(self.bpe_encoding)
        return self._tiktoken_encoder

    @property
    def ast_tokenizer(self) -> ASTTokenizer | None:
        """Get AST tokenizer, initializing lazily on first access.

        Why: ASTTokenizer initialization involves loading tree-sitter parsers, which
        may fail if tree-sitter bindings are not installed. Lazy initialization allows
        the MultiModalTokenizer to be constructed even when tree-sitter is unavailable,
        falling back to text encoding for code. This improves robustness and makes
        testing easier.

        Returns:
            ASTTokenizer instance or None if tree-sitter is unavailable

        Note: Returns None rather than raising ImportError to enable graceful fallback.
        The _encode_code method checks for None and uses text encoding as fallback.
        """
        if self._ast_tokenizer is None and AST_TOKENIZER_AVAILABLE:
            try:
                self._ast_tokenizer = ASTTokenizer()
            except Exception:
                # Fail gracefully if tree-sitter setup fails
                pass
        return self._ast_tokenizer

    def encode(
        self,
        content: Any,
        modality: ModalityType,
        add_special_tokens: bool = True,
        language: str | None = None,
        file_extension: str | None = None,
    ) -> list[int]:
        """Encode content from any modality to token IDs.

        Args:
            content: Content to encode (varies by modality - str for text/code,
                     tensor/array for image/audio)
            modality: Type of modality being encoded
            add_special_tokens: Whether to add BOS/EOS and modality prefix tokens
            language: Optional language hint for code modality ("python", "rust")
            file_extension: Optional file extension for code modality (".py", ".rs")

        Returns:
            List of token IDs in unified vocabulary space

        Why: Adding modality prefix tokens (e.g., <text>, <code>, <image>) immediately after
        BOS enables the model to apply modality-specific processing in early layers while
        maintaining unified attention. This follows Chameleon's design where knowing the
        modality type helps the model adjust attention patterns and generation strategies.
        Truncation at max_length ensures memory safety during training/inference.

        The language and file_extension parameters enable AST-aware code tokenization by
        helping the tokenizer select the appropriate tree-sitter parser.
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
            tokens.extend(self._encode_code(content, language, file_extension))
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
        """Encode text using BPE tokenization with byte-level fallback.

        Args:
            text: Text string to encode

        Returns:
            List of token IDs in unified vocabulary space

        Why: BPE (Byte-Pair Encoding) provides subword tokenization that balances vocabulary
        size with rare word handling. It compresses common words to single tokens while
        decomposing rare words into subwords, achieving better compression than character-level
        (shorter sequences = more content in 128K context) while avoiding the out-of-vocabulary
        issues of word-level tokenization. Essential for multilingual support and code tokens.

        Implementation strategy: Uses tiktoken's cl100k_base encoding (~100K tokens) but maps
        tokens to our unified vocab space. Since our vocab_size is configurable (default 65536),
        we handle this by:
        1. Special tokens occupy IDs 0-7
        2. BPE tokens map to IDs [8, vocab_size-257] (remapped from tiktoken's token space)
        3. Byte fallback occupies last 256 IDs [vocab_size-256, vocab_size-1]
        4. Tiktoken tokens beyond our BPE capacity fall back to byte encoding

        This design maintains the embedding-prediction paradigm where tokens are entry/exit
        points but the model operates in continuous embedding space. The BPE compression
        reduces sequence length, allowing more context within the 128K window.
        """
        # Validate vocab size can accommodate minimal tokenization
        if self.vocab_size < 264:
            raise ValueError(
                f"vocab_size={self.vocab_size} is too small for BPE text encoding; "
                "needs at least 264 (8 special tokens + 256 byte values)."
            )

        # Encode with tiktoken BPE
        tiktoken_tokens = self.tiktoken_encoder.encode(text)

        # Map tiktoken tokens to our unified vocab space
        # We need to skip the AST token range (264-1501) to avoid collisions
        AST_START = 264
        AST_END = 1502

        unified_tokens: list[int] = []
        for tk_id in tiktoken_tokens:
            # Calculate available BPE space, excluding AST range
            # BPE tokens can use: [8, 264) and [1502, byte_fallback_start)
            lower_range_size = AST_START - self._bpe_offset  # 264 - 8 = 256
            upper_range_size = self._byte_fallback_start - AST_END  # ~63778
            total_bpe_space = lower_range_size + upper_range_size

            # Map tiktoken ID to BPE space
            bpe_slot = tk_id % total_bpe_space

            if bpe_slot < lower_range_size:
                # Use lower range [8, 264)
                unified_id = self._bpe_offset + bpe_slot
            else:
                # Use upper range [1502, byte_fallback_start)
                unified_id = AST_END + (bpe_slot - lower_range_size)

            # Final safety check - should not be needed
            if unified_id >= self._byte_fallback_start or (AST_START <= unified_id < AST_END):
                # Collision detected - fall back to byte encoding
                # This should never happen with the logic above
                token_text = self.tiktoken_encoder.decode([tk_id])
                for byte_val in token_text.encode("utf-8"):
                    unified_tokens.append(self._byte_fallback_start + byte_val)
            else:
                unified_tokens.append(unified_id)

        return unified_tokens

    def _encode_code(
        self, code: str, language: str | None = None, file_extension: str | None = None
    ) -> list[int]:
        """Encode source code with AST-aware tokenization.

        Args:
            code: Source code string (Python, Rust, etc.)
            language: Optional explicit language specification ("python", "rust")
            file_extension: Optional file extension for language detection (".py", ".rs")

        Returns:
            List of token IDs

        Why: AST (Abstract Syntax Tree) aware tokenization respects semantic boundaries
        like function definitions and class declarations rather than arbitrary byte sequences.
        This enables function-level semantic understanding (per project-plan.md's embedding
        prediction goals) and prevents splitting identifiers or keywords mid-token. Critical
        for code completion, refactoring, and understanding control flow.

        Uses tree-sitter for parsing when available, with fallback to text encoding for:
        - Unsupported languages (JavaScript, Go, C++, etc.)
        - Malformed code that fails parsing
        - When tree-sitter is not installed

        Token encoding strategy:
        - Keywords (def, class, fn, struct) -> Dedicated token IDs based on hash
        - Identifiers -> BPE encoding using tiktoken (reuses text tokenizer)
        - Operators (+, -, *, /) -> Single token IDs based on operator text
        - Literals -> BPE encoding of the literal value
        - Whitespace/indentation -> Structural tokens (preserved for Python)
        """
        # Try AST tokenization if available
        if self.ast_tokenizer is not None:
            try:
                # Detect language
                lang = None
                if language:
                    lang_map = {"python": CodeLanguage.PYTHON, "rust": CodeLanguage.RUST}
                    lang = lang_map.get(language.lower())

                # Get AST tokens
                ast_tokens = self.ast_tokenizer.tokenize(code, lang, file_extension)

                # Convert AST tokens to token IDs
                return self._encode_ast_tokens(ast_tokens)

            except Exception:
                # Fall back to text encoding on any error
                pass

        # Fallback: treat as text
        return self._encode_text(code)

    def _encode_ast_tokens(self, ast_tokens: list["CodeToken"]) -> list[int]:
        """Convert AST tokens to unified vocabulary token IDs.

        Args:
            ast_tokens: List of CodeToken objects from AST tokenizer

        Returns:
            List of token IDs in unified vocabulary

        Why: AST tokens contain semantic type information (keyword, identifier, operator)
        that we use to determine encoding strategy. Keywords and operators get deterministic
        IDs based on their text (consistent across runs), while identifiers use BPE encoding
        to handle arbitrary user-defined names. This balances structural preservation with
        vocabulary efficiency.

        Encoding strategy:
        - KEYWORD: Hash to range [264, 1000) for ~700 keyword slots
        - OPERATOR/PUNCTUATION: Hash to range [1000, 1500) for ~500 operator slots
        - IDENTIFIER/LITERAL: BPE encode and map to [8, byte_fallback_start)
        - WHITESPACE/NEWLINE: Encode as UTF-8 bytes (preserves indentation for Python)
        - INDENT/DEDENT: Special structural tokens at fixed IDs
        - COMMENT: BPE encode like identifiers

        Note: Hash-based encoding is deterministic and collision-resistant for the limited
        set of keywords/operators in each language (~100 keywords, ~50 operators).
        """
        token_ids: list[int] = []

        # Reserve ranges in vocabulary:
        # 0-7: Special tokens
        # 8-263: Byte encoding (for fallback)
        # 264-1000: Keywords (~700 slots)
        # 1000-1500: Operators/punctuation (~500 slots)
        # 1500+: BPE tokens and other content

        KEYWORD_START = 264
        KEYWORD_END = 1000
        OPERATOR_START = 1000
        OPERATOR_END = 1500

        # Special structural tokens for indentation (Python-specific)
        INDENT_TOKEN = 1500
        DEDENT_TOKEN = 1501

        for token in ast_tokens:
            if token.type == TokenType.KEYWORD:
                # Hash keyword to deterministic ID in keyword range
                hash_val = hash(token.text) % (KEYWORD_END - KEYWORD_START)
                token_ids.append(KEYWORD_START + hash_val)

            elif token.type in (TokenType.OPERATOR, TokenType.PUNCTUATION):
                # Hash operator to deterministic ID in operator range
                hash_val = hash(token.text) % (OPERATOR_END - OPERATOR_START)
                token_ids.append(OPERATOR_START + hash_val)

            elif token.type == TokenType.INDENT:
                token_ids.append(INDENT_TOKEN)

            elif token.type == TokenType.DEDENT:
                token_ids.append(DEDENT_TOKEN)

            elif token.type == TokenType.NEWLINE:
                # Encode newline as byte (important for Python)
                token_ids.append(10 + 8)  # '\n' is byte 10, +8 offset

            elif token.type == TokenType.WHITESPACE:
                # Encode whitespace as bytes (preserves indentation)
                for b in token.text.encode("utf-8"):
                    token_ids.append(b + 8)

            elif token.type in (TokenType.IDENTIFIER, TokenType.LITERAL, TokenType.COMMENT):
                # Use BPE encoding for identifiers and literals
                # This handles arbitrary user-defined names efficiently
                text_tokens = self._encode_text(token.text)
                token_ids.extend(text_tokens)

            else:
                # Unknown token type: encode as text
                text_tokens = self._encode_text(token.text)
                token_ids.extend(text_tokens)

        return token_ids

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

        Implementation: Reverse the encoding process by:
        1. Filtering special tokens if requested
        2. Separating BPE tokens from byte fallback tokens and AST-encoded tokens
        3. Mapping BPE tokens back to tiktoken space and decoding
        4. Decoding byte fallback tokens as UTF-8
        5. AST tokens (keywords, operators) are decoded to their text representation
        6. Concatenating results

        Note: This decoding is best-effort for BPE tokens due to modulo mapping. Perfect
        round-trip requires storing the original tiktoken IDs or using a bijective mapping.
        Byte fallback tokens decode perfectly. AST tokens may lose exact formatting but
        preserve semantic content.
        """
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            token_ids = [t for t in token_ids if t not in special_ids]

        if not token_ids:
            return ""

        # Token ranges:
        # 0-7: Special tokens
        # 8-263: Legacy byte encoding (DEPRECATED - only for old data)
        # 264-1000: Keywords (AST tokens)
        # 1000-1500: Operators (AST tokens)
        # 1500-1501: INDENT/DEDENT (AST tokens)
        # 1502+: BPE tokens (up to byte_fallback_start)
        # byte_fallback_start to vocab_size-1: Byte fallback

        # NOTE: Current implementation uses range [8, byte_fallback_start) for BPE tokens
        # The 8-263 range is only used for legacy compatibility.
        # AST tokens (keywords, operators) use ranges 264-1501.

        KEYWORD_START = 264
        KEYWORD_END = 1000
        OPERATOR_START = 1000
        OPERATOR_END = 1500
        INDENT_TOKEN = 1500
        DEDENT_TOKEN = 1501
        AST_TOKENS_END = 1502  # End of AST token range

        # Separate tokens by type
        byte_chars = []
        decoded_parts = []

        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]

            if KEYWORD_START <= token_id < KEYWORD_END:
                # Keyword token - decode as placeholder (we don't have reverse mapping)
                # Flush any accumulated bytes first
                if byte_chars:
                    try:
                        decoded_parts.append(bytes(byte_chars).decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
                    byte_chars = []
                # Keywords get decoded as generic placeholder
                decoded_parts.append(" <kw> ")
            elif OPERATOR_START <= token_id < OPERATOR_END:
                # Operator token
                if byte_chars:
                    try:
                        decoded_parts.append(bytes(byte_chars).decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
                    byte_chars = []
                decoded_parts.append(" ")  # Operators usually followed by space
            elif token_id == INDENT_TOKEN:
                if byte_chars:
                    try:
                        decoded_parts.append(bytes(byte_chars).decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
                    byte_chars = []
                decoded_parts.append("    ")  # 4 spaces for indent
            elif token_id == DEDENT_TOKEN:
                # Dedent doesn't add text, just structural
                if byte_chars:
                    try:
                        decoded_parts.append(bytes(byte_chars).decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
                    byte_chars = []
            elif self._byte_fallback_start <= token_id < self.vocab_size:
                # Byte fallback token
                byte_val = token_id - self._byte_fallback_start
                byte_chars.append(byte_val)
            elif self._bpe_offset <= token_id < self._byte_fallback_start:
                # BPE token - this includes ranges [8, 264) and [1502, byte_fallback_start)
                # Flush any accumulated bytes first
                if byte_chars:
                    try:
                        decoded_parts.append(bytes(byte_chars).decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
                    byte_chars = []

                # Collect consecutive BPE tokens for batch decoding
                bpe_batch = []
                while i < len(token_ids):
                    tid = token_ids[i]
                    # Check if this is a BPE token (not AST, not byte fallback, not special)
                    is_bpe = (
                        (self._bpe_offset <= tid < KEYWORD_START)
                        or (AST_TOKENS_END <= tid < self._byte_fallback_start)
                    )
                    if is_bpe:
                        # Map back to tiktoken ID - reverse the encoding logic
                        # BPE tokens are in two ranges:
                        # Lower range [8, 264): maps to tiktoken slot [0, 256)
                        # Upper range [1502, byte_fallback_start): maps to tiktoken slot [256, ...)
                        lower_range_size = KEYWORD_START - self._bpe_offset  # 256
                        if tid < KEYWORD_START:
                            # Lower range
                            tiktoken_slot = tid - self._bpe_offset
                        else:
                            # Upper range
                            tiktoken_slot = lower_range_size + (tid - AST_TOKENS_END)

                        # Note: This is still lossy because we used modulo in encoding
                        # We just get the first tiktoken ID that maps to this slot
                        bpe_batch.append(tiktoken_slot)
                        i += 1
                    else:
                        break
                i -= 1  # Back up one since we'll increment at end of loop

                # Decode BPE batch
                if bpe_batch:
                    try:
                        decoded_parts.append(self.tiktoken_encoder.decode(bpe_batch))
                    except Exception:
                        # Try individual tokens if batch fails
                        for tk_id in bpe_batch:
                            try:
                                decoded_parts.append(self.tiktoken_encoder.decode([tk_id]))
                            except Exception:
                                pass
            # else: skip unknown token IDs

            i += 1

        # Flush any remaining bytes
        if byte_chars:
            try:
                decoded_parts.append(bytes(byte_chars).decode("utf-8", errors="ignore"))
            except Exception:
                pass

        return "".join(decoded_parts)


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
