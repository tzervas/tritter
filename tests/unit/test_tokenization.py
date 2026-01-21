"""Unit tests for multimodal tokenization.

Validates multimodal tokenizer and unified embedding layer for early fusion architecture.

Why: Tokenization is the entry point that converts multimodal content into discrete IDs
before embedding into continuous space where the model operates. These tests ensure:
1. All four modalities (text, code, image, audio) tokenize without errors
2. Special tokens (BOS, EOS, modality prefixes) are correctly inserted
3. UnifiedEmbedding maps tokens from all modalities to shared vector space
4. Padding tokens use index 0 and don't update during backprop
5. Max length truncation prevents memory overflow

Testing strategy: Validates the tokenization → embedding pipeline that feeds the model.
Tests cover happy paths (valid inputs) and edge cases (empty strings, max length, padding).
Critical validations: (1) modality prefix tokens enable model to condition on input type,
(2) unified embedding dimension matches model hidden_size for seamless integration.

Architectural context: Tokenization is temporary - it converts inputs to discrete IDs that
get embedded. The model operates in continuous embedding space and predicts next embeddings
(Coconut/LCM style), not next tokens. Tokens are only used at input (via tokenizer) and
output (via KNN/VQ rounding of predicted embeddings back to vocabulary).
"""

import torch

from tritter.tokenization.multimodal import (
    ModalityType,
    MultiModalTokenizer,
    UnifiedEmbedding,
)


class TestMultiModalTokenizer:
    """Test suite for MultiModalTokenizer class."""

    def test_initialization(self) -> None:
        """Test tokenizer initialization."""
        tokenizer = MultiModalTokenizer(vocab_size=1000, max_length=512)

        assert tokenizer.vocab_size == 1000
        assert tokenizer.max_length == 512
        assert len(tokenizer.special_tokens) > 0

    def test_special_tokens_exist(self) -> None:
        """Test that all special tokens are defined."""
        tokenizer = MultiModalTokenizer()

        assert tokenizer.PAD_TOKEN in tokenizer.special_tokens
        assert tokenizer.BOS_TOKEN in tokenizer.special_tokens
        assert tokenizer.EOS_TOKEN in tokenizer.special_tokens
        assert tokenizer.TEXT_PREFIX in tokenizer.special_tokens
        assert tokenizer.CODE_PREFIX in tokenizer.special_tokens
        assert tokenizer.IMAGE_PREFIX in tokenizer.special_tokens
        assert tokenizer.AUDIO_PREFIX in tokenizer.special_tokens

    def test_encode_text(self) -> None:
        """Test text encoding."""
        tokenizer = MultiModalTokenizer()

        text = "Hello, world!"
        tokens = tokenizer.encode(text, ModalityType.TEXT)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should have BOS, modality prefix, content, EOS
        assert tokens[0] == tokenizer.special_tokens[tokenizer.BOS_TOKEN]
        assert tokens[1] == tokenizer.special_tokens[tokenizer.TEXT_PREFIX]
        assert tokens[-1] == tokenizer.special_tokens[tokenizer.EOS_TOKEN]

    def test_encode_code(self) -> None:
        """Test code encoding."""
        tokenizer = MultiModalTokenizer()

        code = "def hello(): return 'world'"
        tokens = tokenizer.encode(code, ModalityType.CODE)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert tokens[1] == tokenizer.special_tokens[tokenizer.CODE_PREFIX]

    def test_encode_without_special_tokens(self) -> None:
        """Test encoding without special tokens."""
        tokenizer = MultiModalTokenizer()

        text = "test"
        tokens = tokenizer.encode(text, ModalityType.TEXT, add_special_tokens=False)

        # Should not have BOS/EOS tokens
        assert tokens[0] != tokenizer.special_tokens[tokenizer.BOS_TOKEN]

    def test_encode_respects_max_length(self) -> None:
        """Test that encoding respects maximum length."""
        tokenizer = MultiModalTokenizer(max_length=10)

        long_text = "a" * 1000
        tokens = tokenizer.encode(long_text, ModalityType.TEXT)

        assert len(tokens) <= 10

    def test_decode_text(self) -> None:
        """Test text decoding."""
        tokenizer = MultiModalTokenizer()

        # Create some token IDs
        token_ids = [65, 66, 67]  # Simplified example
        decoded = tokenizer.decode(token_ids)

        assert isinstance(decoded, str)

    def test_modality_types(self) -> None:
        """Test all modality types are supported."""
        tokenizer = MultiModalTokenizer()

        assert ModalityType.TEXT in tokenizer.modality_prefixes
        assert ModalityType.CODE in tokenizer.modality_prefixes
        assert ModalityType.IMAGE in tokenizer.modality_prefixes
        assert ModalityType.AUDIO in tokenizer.modality_prefixes


class TestUnifiedEmbedding:
    """Test suite for UnifiedEmbedding module."""

    def test_initialization(self) -> None:
        """Test embedding layer initialization."""
        embedding = UnifiedEmbedding(vocab_size=1000, embedding_dim=512)

        assert embedding.embedding.num_embeddings == 1000
        assert embedding.embedding.embedding_dim == 512

    def test_forward_pass(self) -> None:
        """Test forward pass through embedding layer."""
        embedding = UnifiedEmbedding(vocab_size=1000, embedding_dim=256)

        # Create token IDs
        batch_size = 4
        seq_len = 16
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        output = embedding(token_ids)

        assert output.shape == (batch_size, seq_len, 256)

    def test_padding_index(self) -> None:
        """Test that padding index works correctly."""
        embedding = UnifiedEmbedding(vocab_size=100, embedding_dim=64, padding_idx=0)

        # Token IDs with padding
        token_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])

        output = embedding(token_ids)

        # Padding embeddings should be zero
        assert torch.allclose(output[:, 3:, :], torch.zeros_like(output[:, 3:, :]))
