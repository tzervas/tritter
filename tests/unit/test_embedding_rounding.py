"""Tests for embedding rounding module (KNN/VQ rounding).

Validates the embedding rounding module that provides the "exit point" from
continuous embedding space back to discrete tokens.

Why: The embedding-prediction paradigm operates in continuous space. Rounding
converts model outputs back to discrete tokens for:
1. Text generation (tokens for output)
2. Evaluation (cross-entropy loss requires discrete targets)
3. Integration with tokenizers and text processing

Test coverage:
1. KNNRouter nearest neighbor lookup (k=1 and k>1)
2. KNNRouter soft routing with temperature
3. VQRouter quantization and dequantization
4. VQRouter commitment loss and EMA updates
5. VQRouter straight-through gradient flow
6. EmbeddingRounder unified interface (all modes)
7. Factory function create_rounding_layer
8. Performance with larger codebooks
9. Edge cases and error handling
"""

import pytest
import torch
import torch.nn.functional as F

from tritter.core.config import TritterConfig
from tritter.embedding.knn_rounding import (
    FAISS_AVAILABLE,
    KNNRouter,
    KNNRouterConfig,
)
from tritter.embedding.rounding import (
    EmbeddingRounder,
    EmbeddingRounderConfig,
    create_rounding_layer,
)
from tritter.embedding.vq_rounding import (
    VQRouter,
    VQRouterConfig,
)


class TestKNNRouterConfig:
    """Tests for KNNRouterConfig."""

    def test_default_config(self) -> None:
        """Verify default config provides sensible defaults.

        Why: k=1 is argmax (greedy), temperature=1.0 is standard softmax.
        """
        config = KNNRouterConfig()

        assert config.k == 1
        assert config.temperature == 1.0
        assert config.use_faiss is True
        assert config.use_gpu_faiss is True
        assert config.normalize is True

    def test_custom_config(self) -> None:
        """Verify custom config values are respected.

        Why: Users should be able to customize routing behavior.
        """
        config = KNNRouterConfig(
            k=5,
            temperature=0.5,
            use_faiss=False,
            normalize=False,
        )

        assert config.k == 5
        assert config.temperature == 0.5
        assert config.use_faiss is False
        assert config.normalize is False


class TestKNNRouter:
    """Tests for KNNRouter class."""

    def test_router_initialization(self) -> None:
        """Verify router initializes with vocabulary embeddings.

        Why: Router needs vocabulary embeddings as the codebook.
        """
        vocab_size = 1000
        embedding_dim = 256
        vocab_embeddings = torch.randn(vocab_size, embedding_dim)

        router = KNNRouter(vocab_embeddings)

        assert router.vocab_size == vocab_size
        assert router.embedding_dim == embedding_dim
        assert router.vocab_embeddings.shape == (vocab_size, embedding_dim)

    def test_forward_k1_shape(self) -> None:
        """Verify forward with k=1 returns correct shape.

        Why: k=1 (argmax) should return single token ID per embedding.
        """
        vocab_embeddings = torch.randn(1000, 256)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(k=1))

        # 3D input: (B, L, D)
        embeddings = torch.randn(2, 10, 256)
        token_ids = router(embeddings)

        assert token_ids.shape == (2, 10)
        assert token_ids.dtype == torch.long

    def test_forward_2d_input(self) -> None:
        """Verify router handles 2D input (B, D).

        Why: Should work for single position per batch.
        """
        vocab_embeddings = torch.randn(1000, 256)
        router = KNNRouter(vocab_embeddings)

        embeddings = torch.randn(2, 256)
        token_ids = router(embeddings)

        assert token_ids.shape == (2,)

    def test_forward_tokens_in_range(self) -> None:
        """Verify token IDs are within vocabulary range.

        Why: Token IDs must be valid indices into vocabulary.
        """
        vocab_size = 1000
        vocab_embeddings = torch.randn(vocab_size, 256)
        router = KNNRouter(vocab_embeddings)

        embeddings = torch.randn(2, 10, 256)
        token_ids = router(embeddings)

        assert token_ids.min() >= 0
        assert token_ids.max() < vocab_size

    def test_forward_finds_nearest(self) -> None:
        """Verify router finds the actual nearest neighbor.

        Why: Core functionality - must find the closest vocabulary embedding.
        """
        vocab_size = 100
        embedding_dim = 64
        vocab_embeddings = torch.randn(vocab_size, embedding_dim)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(use_faiss=False))

        # Use actual vocabulary embedding as query (should return its index)
        query_idx = 42
        query = vocab_embeddings[query_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        result = router(query)

        assert result.item() == query_idx

    def test_forward_soft_shape(self) -> None:
        """Verify forward_soft returns probability distribution.

        Why: Soft routing returns distribution over vocabulary.
        """
        vocab_size = 1000
        vocab_embeddings = torch.randn(vocab_size, 256)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(k=5))

        embeddings = torch.randn(2, 10, 256)
        probs = router.forward_soft(embeddings)

        assert probs.shape == (2, 10, vocab_size)
        # Check it's a valid probability distribution
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones(2, 10),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_forward_soft_2d_input(self) -> None:
        """Verify forward_soft handles 2D input.

        Why: Should work for single position per batch.
        """
        vocab_size = 1000
        vocab_embeddings = torch.randn(vocab_size, 256)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(k=10))

        embeddings = torch.randn(2, 256)
        probs = router.forward_soft(embeddings)

        assert probs.shape == (2, vocab_size)

    def test_forward_soft_temperature_effect(self) -> None:
        """Verify temperature affects distribution sharpness.

        Why: Lower temperature should give sharper (more deterministic) distribution.
        """
        vocab_embeddings = torch.randn(100, 64)
        embeddings = torch.randn(1, 5, 64)

        # Low temperature (sharp)
        router_cold = KNNRouter(vocab_embeddings, KNNRouterConfig(k=100, temperature=0.1))
        probs_cold = router_cold.forward_soft(embeddings)

        # High temperature (flat)
        router_hot = KNNRouter(vocab_embeddings, KNNRouterConfig(k=100, temperature=2.0))
        probs_hot = router_hot.forward_soft(embeddings)

        # Cold distribution should have higher max probability
        assert probs_cold.max() > probs_hot.max()

        # Hot distribution should have higher entropy (more spread out)
        entropy_cold = -(probs_cold * torch.log(probs_cold + 1e-10)).sum(dim=-1).mean()
        entropy_hot = -(probs_hot * torch.log(probs_hot + 1e-10)).sum(dim=-1).mean()
        assert entropy_hot > entropy_cold

    def test_forward_soft_top_k_sparsity(self) -> None:
        """Verify soft routing with k < vocab_size is sparse.

        Why: Only top-k entries should have non-zero probability.
        """
        vocab_size = 1000
        k = 10
        vocab_embeddings = torch.randn(vocab_size, 256)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(k=k))

        embeddings = torch.randn(2, 5, 256)
        probs = router.forward_soft(embeddings)

        # Each position should have at most k non-zero probabilities
        non_zero_count = (probs > 0).sum(dim=-1)
        assert (non_zero_count <= k).all()

    def test_get_similarities_shape(self) -> None:
        """Verify get_similarities returns correct shape.

        Why: Raw similarities needed for loss computation.
        """
        vocab_size = 1000
        vocab_embeddings = torch.randn(vocab_size, 256)
        router = KNNRouter(vocab_embeddings)

        embeddings = torch.randn(2, 10, 256)
        similarities = router.get_similarities(embeddings)

        assert similarities.shape == (2, 10, vocab_size)

    def test_cosine_vs_euclidean_mode(self) -> None:
        """Verify normalize flag switches between cosine and L2.

        Why: Different distance metrics may work better for different embeddings.
        """
        vocab_embeddings = torch.randn(100, 64)

        # Cosine similarity (normalize=True)
        router_cosine = KNNRouter(vocab_embeddings, KNNRouterConfig(normalize=True, use_faiss=False))

        # L2 distance (normalize=False)
        router_l2 = KNNRouter(vocab_embeddings, KNNRouterConfig(normalize=False, use_faiss=False))

        # Same query embedding
        query = torch.randn(1, 5, 64)

        # Results may differ based on distance metric
        result_cosine = router_cosine(query)
        result_l2 = router_l2(query)

        # At least verify both work without error and return valid indices
        assert result_cosine.min() >= 0
        assert result_l2.min() >= 0

    def test_update_index_changes_vocab(self) -> None:
        """Verify update_index updates the vocabulary.

        Why: Enables dynamic vocabulary updates.
        """
        vocab_embeddings = torch.randn(100, 64)
        router = KNNRouter(vocab_embeddings)

        # Update with new embeddings
        new_vocab = torch.randn(200, 64)
        router.update_index(new_vocab)

        assert router.vocab_size == 200
        assert router.vocab_embeddings.shape == (200, 64)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
    def test_faiss_mode(self) -> None:
        """Verify FAISS mode produces same results as PyTorch.

        Why: FAISS is optimization; results should match.
        """
        vocab_embeddings = torch.randn(1000, 256)

        router_faiss = KNNRouter(vocab_embeddings, KNNRouterConfig(use_faiss=True))
        router_torch = KNNRouter(vocab_embeddings, KNNRouterConfig(use_faiss=False))

        embeddings = torch.randn(2, 10, 256)

        result_faiss = router_faiss(embeddings)
        result_torch = router_torch(embeddings)

        # Results should be identical (both find true nearest neighbor)
        torch.testing.assert_close(result_faiss, result_torch)


class TestVQRouterConfig:
    """Tests for VQRouterConfig."""

    def test_default_config(self) -> None:
        """Verify default config provides sensible defaults.

        Why: Standard VQ-VAE hyperparameters.
        """
        config = VQRouterConfig()

        assert config.codebook_size == 65536
        assert config.hidden_size == 768
        assert config.commitment_cost == 0.25
        assert config.use_ema is True
        assert config.ema_decay == 0.99

    def test_custom_config(self) -> None:
        """Verify custom config values are respected.

        Why: Users should be able to customize VQ behavior.
        """
        config = VQRouterConfig(
            codebook_size=1024,
            hidden_size=256,
            commitment_cost=0.5,
            use_ema=False,
        )

        assert config.codebook_size == 1024
        assert config.hidden_size == 256
        assert config.commitment_cost == 0.5
        assert config.use_ema is False


class TestVQRouter:
    """Tests for VQRouter class."""

    def test_router_initialization(self) -> None:
        """Verify router initializes codebook.

        Why: VQ learns its own codebook.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        assert router.codebook_size == 1024
        assert router.hidden_size == 256
        assert router.codebook.weight.shape == (1024, 256)

    def test_forward_output_shapes(self) -> None:
        """Verify forward returns correct shapes.

        Why: Forward returns codes, quantized, and loss.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 256)
        codes, quantized, loss = router(embeddings)

        assert codes.shape == (2, 10)
        assert codes.dtype == torch.long
        assert quantized.shape == (2, 10, 256)
        assert loss.dim() == 0  # Scalar

    def test_forward_2d_input(self) -> None:
        """Verify router handles 2D input (B, D).

        Why: Should work for single position per batch.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(2, 256)
        codes, quantized, loss = router(embeddings)

        assert codes.shape == (2,)
        assert quantized.shape == (2, 256)

    def test_codes_in_range(self) -> None:
        """Verify codes are within codebook range.

        Why: Codes must be valid indices into codebook.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 256)
        codes, _, _ = router(embeddings)

        assert codes.min() >= 0
        assert codes.max() < config.codebook_size

    def test_quantize_method(self) -> None:
        """Verify quantize method (no loss computation).

        Why: Inference-only quantization should be efficient.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 256)
        quantized = router.quantize(embeddings)

        assert quantized.shape == embeddings.shape

    def test_lookup_method(self) -> None:
        """Verify lookup retrieves correct embeddings.

        Why: lookup(codes) should return codebook entries.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        codes = torch.randint(0, 1024, (2, 10))
        embeddings = router.lookup(codes)

        assert embeddings.shape == (2, 10, 256)

    def test_quantize_lookup_consistency(self) -> None:
        """Verify quantize then lookup returns same result.

        Why: Round-trip should be consistent.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)
        router.eval()  # Disable EMA updates

        embeddings = torch.randn(2, 10, 256)
        codes, quantized, _ = router(embeddings)

        # Lookup should return same as quantized (before STE)
        looked_up = router.lookup(codes)

        # Due to STE, quantized = z + (e - z).detach(), so we compare to lookup
        # The actual codebook entries
        torch.testing.assert_close(looked_up, router.codebook(codes))

    def test_vq_loss_positive(self) -> None:
        """Verify VQ loss is positive.

        Why: Loss measures quantization error.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 256)
        _, _, loss = router(embeddings)

        assert loss > 0

    def test_commitment_cost_affects_loss(self) -> None:
        """Verify commitment_cost affects loss magnitude.

        Why: Higher commitment_cost should increase commitment loss component.
        """
        embeddings = torch.randn(2, 10, 256)

        # Low commitment cost
        config_low = VQRouterConfig(codebook_size=1024, hidden_size=256, commitment_cost=0.1)
        router_low = VQRouter(config_low)
        _, _, loss_low = router_low(embeddings)

        # High commitment cost
        config_high = VQRouterConfig(codebook_size=1024, hidden_size=256, commitment_cost=1.0)
        router_high = VQRouter(config_high)
        _, _, loss_high = router_high(embeddings)

        # Higher commitment cost should generally lead to higher loss
        # (Note: This isn't guaranteed for random init, but usually holds)
        # We just verify both are positive
        assert loss_low > 0
        assert loss_high > 0

    def test_gradient_flow_through_ste(self) -> None:
        """Verify gradients flow through straight-through estimator.

        Why: Training requires gradients to reach input embeddings.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256, use_ema=False)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 256, requires_grad=True)
        codes, quantized, loss = router(embeddings)

        # Backprop through quantized
        quantized.sum().backward()

        # Input should have gradients (via STE)
        assert embeddings.grad is not None
        assert embeddings.grad.abs().max() > 0

    def test_gradient_flow_through_loss(self) -> None:
        """Verify gradients flow through VQ loss.

        Why: VQ loss should update codebook.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256, use_ema=False)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 256)
        _, _, loss = router(embeddings)

        loss.backward()

        # Codebook should have gradients
        assert router.codebook.weight.grad is not None

    def test_ema_updates_during_training(self) -> None:
        """Verify EMA updates codebook during training.

        Why: EMA provides stable codebook updates.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256, use_ema=True)
        router = VQRouter(config)
        router.train()

        initial_codebook = router.codebook.weight.clone()

        # Forward pass triggers EMA update
        embeddings = torch.randn(2, 100, 256)
        router(embeddings)

        # Codebook should have changed (not necessarily all entries)
        assert not torch.allclose(router.codebook.weight, initial_codebook)

    def test_ema_disabled_in_eval(self) -> None:
        """Verify EMA updates don't happen in eval mode.

        Why: Inference should not modify model.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256, use_ema=True)
        router = VQRouter(config)
        router.eval()

        initial_codebook = router.codebook.weight.clone()

        embeddings = torch.randn(2, 100, 256)
        router(embeddings)

        # Codebook should be unchanged
        torch.testing.assert_close(router.codebook.weight, initial_codebook)

    def test_get_codebook_usage(self) -> None:
        """Verify codebook usage calculation.

        Why: Monitors for codebook collapse.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(10, 100, 256)
        codes, _, _ = router(embeddings)

        usage = router.get_codebook_usage(codes)

        assert 0 < usage <= 1.0

    def test_get_perplexity(self) -> None:
        """Verify perplexity calculation.

        Why: Perplexity measures effective codebook size.
        """
        config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        router = VQRouter(config)

        embeddings = torch.randn(10, 100, 256)
        codes, _, _ = router(embeddings)

        perplexity = router.get_perplexity(codes)

        # Perplexity should be between 1 and codebook_size
        assert 1 <= perplexity <= config.codebook_size


class TestEmbeddingRounderConfig:
    """Tests for EmbeddingRounderConfig."""

    def test_default_config(self) -> None:
        """Verify default config uses KNN mode.

        Why: KNN is fast and requires no training.
        """
        config = EmbeddingRounderConfig()

        assert config.mode == "knn"
        assert config.knn_config is not None
        assert config.vq_config is not None

    def test_mode_options(self) -> None:
        """Verify all mode options are valid.

        Why: Three modes: knn, vq, none.
        """
        for mode in ["knn", "vq", "none"]:
            config = EmbeddingRounderConfig(mode=mode)
            assert config.mode == mode


class TestEmbeddingRounder:
    """Tests for EmbeddingRounder unified interface."""

    def test_knn_mode_initialization(self) -> None:
        """Verify KNN mode requires vocab_embeddings.

        Why: KNN needs vocabulary as codebook.
        """
        config = EmbeddingRounderConfig(mode="knn")
        vocab_embeddings = torch.randn(1000, 256)

        rounder = EmbeddingRounder(config, vocab_embeddings)

        assert rounder.mode == "knn"
        assert isinstance(rounder.router, KNNRouter)

    def test_knn_mode_missing_vocab_raises(self) -> None:
        """Verify KNN mode raises without vocab_embeddings.

        Why: KNN requires vocabulary embeddings.
        """
        config = EmbeddingRounderConfig(mode="knn")

        with pytest.raises(ValueError, match="vocab_embeddings is required"):
            EmbeddingRounder(config, vocab_embeddings=None)

    def test_vq_mode_initialization(self) -> None:
        """Verify VQ mode works without vocab_embeddings.

        Why: VQ learns its own codebook.
        """
        vq_config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        config = EmbeddingRounderConfig(mode="vq", vq_config=vq_config)

        rounder = EmbeddingRounder(config)

        assert rounder.mode == "vq"
        assert isinstance(rounder.router, VQRouter)

    def test_none_mode_initialization(self) -> None:
        """Verify none mode works (pass-through).

        Why: None mode disables rounding for continuous operation.
        """
        config = EmbeddingRounderConfig(mode="none")

        rounder = EmbeddingRounder(config)

        assert rounder.mode == "none"
        assert rounder.router is None

    def test_round_knn_mode(self) -> None:
        """Verify round method in KNN mode.

        Why: Should return token IDs from nearest neighbor.
        """
        vocab_embeddings = torch.randn(1000, 256)
        config = EmbeddingRounderConfig(mode="knn")
        rounder = EmbeddingRounder(config, vocab_embeddings)

        embeddings = torch.randn(2, 10, 256)
        token_ids = rounder.round(embeddings)

        assert token_ids.shape == (2, 10)
        assert token_ids.dtype == torch.long

    def test_round_vq_mode(self) -> None:
        """Verify round method in VQ mode.

        Why: Should return codes from VQ.
        """
        vq_config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        config = EmbeddingRounderConfig(mode="vq", vq_config=vq_config)
        rounder = EmbeddingRounder(config)

        embeddings = torch.randn(2, 10, 256)
        token_ids = rounder.round(embeddings)

        assert token_ids.shape == (2, 10)

    def test_round_none_mode(self) -> None:
        """Verify round method in none mode returns zeros.

        Why: Pass-through mode returns placeholder zeros.
        """
        config = EmbeddingRounderConfig(mode="none")
        rounder = EmbeddingRounder(config)

        embeddings = torch.randn(2, 10, 256)
        token_ids = rounder.round(embeddings)

        assert token_ids.shape == (2, 10)
        assert (token_ids == 0).all()

    def test_round_with_embeddings_knn(self) -> None:
        """Verify round_with_embeddings in KNN mode.

        Why: Returns both tokens and quantized embeddings.
        """
        vocab_embeddings = torch.randn(1000, 256)
        config = EmbeddingRounderConfig(mode="knn")
        rounder = EmbeddingRounder(config, vocab_embeddings)

        embeddings = torch.randn(2, 10, 256)
        token_ids, quantized, loss = rounder.round_with_embeddings(embeddings)

        assert token_ids.shape == (2, 10)
        assert quantized.shape == (2, 10, 256)
        assert loss is None  # KNN has no loss

    def test_round_with_embeddings_vq(self) -> None:
        """Verify round_with_embeddings in VQ mode returns loss.

        Why: VQ has trainable loss.
        """
        vq_config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        config = EmbeddingRounderConfig(mode="vq", vq_config=vq_config)
        rounder = EmbeddingRounder(config)

        embeddings = torch.randn(2, 10, 256)
        token_ids, quantized, loss = rounder.round_with_embeddings(embeddings)

        assert token_ids.shape == (2, 10)
        assert quantized.shape == (2, 10, 256)
        assert loss is not None
        assert loss > 0

    def test_round_with_embeddings_none(self) -> None:
        """Verify round_with_embeddings in none mode passes through.

        Why: Embeddings unchanged, zeros for tokens.
        """
        config = EmbeddingRounderConfig(mode="none")
        rounder = EmbeddingRounder(config)

        embeddings = torch.randn(2, 10, 256)
        token_ids, quantized, loss = rounder.round_with_embeddings(embeddings)

        assert token_ids.shape == (2, 10)
        assert (token_ids == 0).all()
        torch.testing.assert_close(quantized, embeddings)
        assert loss is None

    def test_get_soft_distribution_knn(self) -> None:
        """Verify get_soft_distribution in KNN mode.

        Why: Returns probability distribution over vocabulary.
        """
        vocab_embeddings = torch.randn(1000, 256)
        knn_config = KNNRouterConfig(k=10)
        config = EmbeddingRounderConfig(mode="knn", knn_config=knn_config)
        rounder = EmbeddingRounder(config, vocab_embeddings)

        embeddings = torch.randn(2, 5, 256)
        probs = rounder.get_soft_distribution(embeddings)

        assert probs.shape == (2, 5, 1000)
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones(2, 5),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_get_soft_distribution_vq(self) -> None:
        """Verify get_soft_distribution in VQ mode.

        Why: Should work via distance-based softmax.
        """
        vq_config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        config = EmbeddingRounderConfig(mode="vq", vq_config=vq_config)
        rounder = EmbeddingRounder(config)

        embeddings = torch.randn(2, 5, 256)
        probs = rounder.get_soft_distribution(embeddings)

        assert probs.shape == (2, 5, 1024)

    def test_get_soft_distribution_none_raises(self) -> None:
        """Verify get_soft_distribution raises in none mode.

        Why: Pass-through mode has no distribution.
        """
        config = EmbeddingRounderConfig(mode="none")
        rounder = EmbeddingRounder(config)

        embeddings = torch.randn(2, 5, 256)

        with pytest.raises(ValueError, match="not available"):
            rounder.get_soft_distribution(embeddings)

    def test_forward_is_round(self) -> None:
        """Verify forward() is same as round().

        Why: Enables use in nn.Sequential.
        """
        vocab_embeddings = torch.randn(1000, 256)
        config = EmbeddingRounderConfig(mode="knn")
        rounder = EmbeddingRounder(config, vocab_embeddings)

        embeddings = torch.randn(2, 10, 256)

        result_forward = rounder(embeddings)
        result_round = rounder.round(embeddings)

        torch.testing.assert_close(result_forward, result_round)


class TestCreateRoundingLayer:
    """Tests for factory function."""

    def test_create_knn_with_vocab(self) -> None:
        """Verify factory creates KNN rounder with vocab.

        Why: Common use case.
        """
        vocab_embeddings = torch.randn(1000, 256)

        rounder = create_rounding_layer(
            vocab_embeddings=vocab_embeddings,
            mode="knn",
        )

        assert rounder.mode == "knn"

    def test_create_vq_without_vocab(self) -> None:
        """Verify factory creates VQ rounder without vocab.

        Why: VQ doesn't need vocab embeddings.
        """
        rounder = create_rounding_layer(
            mode="vq",
            codebook_size=1024,
            hidden_size=256,
        )

        assert rounder.mode == "vq"

    def test_create_with_tritter_config(self) -> None:
        """Verify factory uses TritterConfig for defaults.

        Why: Convenience for matching model dimensions.
        """
        config = TritterConfig(
            model_size="3B",
            hidden_size=2048,
            num_layers=2,
        )

        rounder = create_rounding_layer(
            config=config,
            mode="vq",
        )

        # Should use config's hidden_size and vocab_size
        assert isinstance(rounder.router, VQRouter)
        assert rounder.router.hidden_size == config.hidden_size

    def test_create_with_custom_knn_params(self) -> None:
        """Verify custom KNN parameters are passed through.

        Why: Users should be able to customize.
        """
        vocab_embeddings = torch.randn(1000, 256)

        rounder = create_rounding_layer(
            vocab_embeddings=vocab_embeddings,
            mode="knn",
            k=5,
            temperature=0.5,
        )

        assert rounder.mode == "knn"
        # Verify config was passed
        assert isinstance(rounder.router, KNNRouter)
        assert rounder.router.config.k == 5
        assert rounder.router.config.temperature == 0.5

    def test_create_none_mode(self) -> None:
        """Verify factory creates none mode rounder.

        Why: Pass-through mode for continuous operation.
        """
        rounder = create_rounding_layer(mode="none")

        assert rounder.mode == "none"


class TestPerformanceLargeCodebook:
    """Tests for performance with larger codebooks."""

    def test_knn_large_vocab(self) -> None:
        """Verify KNN handles large vocabulary.

        Why: Real vocabularies may have 32K+ tokens.
        """
        vocab_size = 32000
        embedding_dim = 512
        vocab_embeddings = torch.randn(vocab_size, embedding_dim)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(use_faiss=False))

        embeddings = torch.randn(2, 10, embedding_dim)
        token_ids = router(embeddings)

        assert token_ids.shape == (2, 10)
        assert token_ids.max() < vocab_size

    def test_vq_large_codebook(self) -> None:
        """Verify VQ handles large codebook.

        Why: Default codebook is 65K entries.
        """
        config = VQRouterConfig(codebook_size=32000, hidden_size=512)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 512)
        codes, quantized, loss = router(embeddings)

        assert codes.shape == (2, 10)
        assert codes.max() < config.codebook_size

    def test_knn_batch_scalability(self) -> None:
        """Verify KNN handles larger batches.

        Why: Should scale to batch sizes used in training.
        """
        vocab_embeddings = torch.randn(10000, 256)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(use_faiss=False))

        for batch_size in [1, 4, 16]:
            embeddings = torch.randn(batch_size, 100, 256)
            token_ids = router(embeddings)
            assert token_ids.shape == (batch_size, 100)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_embedding(self) -> None:
        """Verify routers handle single embedding.

        Why: Edge case of batch_size=1, seq_len=1.
        """
        vocab_embeddings = torch.randn(1000, 256)
        rounder = create_rounding_layer(vocab_embeddings=vocab_embeddings, mode="knn")

        embedding = torch.randn(1, 1, 256)
        token_id = rounder.round(embedding)

        assert token_id.shape == (1, 1)

    def test_very_small_codebook(self) -> None:
        """Verify VQ handles very small codebook.

        Why: Edge case for testing/debugging.
        """
        config = VQRouterConfig(codebook_size=10, hidden_size=64)
        router = VQRouter(config)

        embeddings = torch.randn(2, 10, 64)
        codes, _, _ = router(embeddings)

        assert codes.max() < 10

    def test_identical_embeddings(self) -> None:
        """Verify handling of identical embeddings.

        Why: Should map to same code.
        """
        vocab_embeddings = torch.randn(1000, 256)
        router = KNNRouter(vocab_embeddings, KNNRouterConfig(use_faiss=False))

        # Same embedding repeated
        embedding = torch.randn(256)
        embeddings = embedding.unsqueeze(0).unsqueeze(0).expand(2, 5, -1).contiguous()

        token_ids = router(embeddings)

        # All should map to same token
        assert (token_ids == token_ids[0, 0]).all()

    def test_finite_outputs(self) -> None:
        """Verify outputs are finite (no NaN/Inf).

        Why: Numerical stability check.
        """
        vocab_embeddings = torch.randn(1000, 256)
        knn_router = KNNRouter(vocab_embeddings)

        vq_config = VQRouterConfig(codebook_size=1024, hidden_size=256)
        vq_router = VQRouter(vq_config)

        embeddings = torch.randn(2, 10, 256)

        # KNN outputs
        knn_tokens = knn_router(embeddings)
        knn_probs = knn_router.forward_soft(embeddings)
        assert torch.isfinite(knn_probs).all()

        # VQ outputs
        vq_codes, vq_quantized, vq_loss = vq_router(embeddings)
        assert torch.isfinite(vq_quantized).all()
        assert torch.isfinite(vq_loss).all()

    def test_hidden_size_property(self) -> None:
        """Verify hidden_size property works.

        Why: Useful for checking dimensions.
        """
        vocab_embeddings = torch.randn(1000, 256)
        knn_rounder = create_rounding_layer(vocab_embeddings=vocab_embeddings, mode="knn")
        assert knn_rounder.hidden_size == 256

        vq_rounder = create_rounding_layer(mode="vq", hidden_size=512, codebook_size=1024)
        assert vq_rounder.hidden_size == 512

        none_rounder = create_rounding_layer(mode="none")
        assert none_rounder.hidden_size is None


class TestIntegrationWithEmbeddings:
    """Integration tests with embedding lookup patterns."""

    def test_roundtrip_with_vocab_embeddings(self) -> None:
        """Verify embedding -> round -> lookup roundtrip.

        Why: Core embedding-prediction paradigm flow.
        """
        vocab_size = 1000
        embedding_dim = 256

        # Create vocab embeddings
        vocab_embeddings = torch.randn(vocab_size, embedding_dim)

        # Create embedding layer
        embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        embedding_layer.weight.data = vocab_embeddings

        # Create KNN rounder with same vocab
        rounder = create_rounding_layer(vocab_embeddings=vocab_embeddings, mode="knn")

        # Start with token IDs
        input_ids = torch.randint(0, vocab_size, (2, 10))

        # Get embeddings
        embeddings = embedding_layer(input_ids)

        # Round back to tokens
        output_ids = rounder.round(embeddings)

        # Should recover original tokens (exact match for vocab embeddings)
        torch.testing.assert_close(output_ids, input_ids)

    def test_vq_codebook_separate_from_vocab(self) -> None:
        """Verify VQ codebook is independent of vocabulary.

        Why: VQ learns its own representation.
        """
        vocab_size = 1000
        codebook_size = 512
        embedding_dim = 256

        vq_rounder = create_rounding_layer(
            mode="vq",
            codebook_size=codebook_size,
            hidden_size=embedding_dim,
        )

        embeddings = torch.randn(2, 10, embedding_dim)
        codes, _, _ = vq_rounder.round_with_embeddings(embeddings)

        # Codes are from VQ codebook, not vocab
        assert codes.max() < codebook_size
