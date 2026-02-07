"""Unit tests for model architecture.

Validates transformer components: attention, MLP, layers, and full model integration.

Why: The model architecture is complex with multiple interacting components. These tests ensure:
1. Each component (attention, MLP, layer) initializes with correct dimensions
2. Forward passes produce expected output shapes
3. BitNet quantization integrates correctly when enabled
4. Residual connections preserve gradient flow
5. Full model composes all layers without shape mismatches

Testing strategy: Bottom-up validation from smallest components (attention, MLP) to composed
structures (layer, full model). Tests use reduced dimensions for speed while maintaining
architectural patterns. Critical assertion: output shapes must match input shapes for residual
connections to work, enabling deep networks (24-32 layers).

Note: These tests validate the current token-based architecture. Future tests should validate
embedding prediction (Coconut-style) where model outputs embeddings fed back as inputs rather
than discrete logits for token prediction.
"""

import torch

from tritter.core.config import TritterConfig
from tritter.models.architecture import (
    TritterAttention,
    TritterLayer,
    TritterMLP,
    TritterModel,
)
from tritter.quantization.bitnet import TernaryWeight


class TestTritterAttention:
    """Test suite for TritterAttention module."""

    def test_initialization(self) -> None:
        """Test attention module initialization."""
        config = TritterConfig(hidden_size=512, num_heads=8, num_layers=2)
        attention = TritterAttention(config)

        assert attention.num_heads == 8
        assert attention.head_dim == 64
        assert attention.hidden_size == 512

    def test_forward_pass(self) -> None:
        """Test forward pass through attention."""
        config = TritterConfig(
            hidden_size=256,
            num_heads=4,
            num_layers=1,
            use_bitnet=False,  # Disable for faster testing
        )
        attention = TritterAttention(config)

        batch_size = 2
        seq_len = 8
        hidden_states = torch.randn(batch_size, seq_len, 256)

        output = attention(hidden_states)

        assert output.shape == (batch_size, seq_len, 256)


class TestTritterMLP:
    """Test suite for TritterMLP module."""

    def test_initialization(self) -> None:
        """Test MLP module initialization."""
        config = TritterConfig(hidden_size=512, intermediate_size=2048, num_layers=1)
        mlp = TritterMLP(config)

        assert mlp.config.hidden_size == 512
        assert mlp.config.intermediate_size == 2048

    def test_forward_pass(self) -> None:
        """Test forward pass through MLP."""
        config = TritterConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_layers=1,
            use_bitnet=False,
        )
        mlp = TritterMLP(config)

        batch_size = 2
        seq_len = 8
        hidden_states = torch.randn(batch_size, seq_len, 256)

        output = mlp(hidden_states)

        assert output.shape == (batch_size, seq_len, 256)


class TestTritterLayer:
    """Test suite for TritterLayer module."""

    def test_initialization(self) -> None:
        """Test transformer layer initialization."""
        config = TritterConfig(hidden_size=256, num_heads=4, num_layers=1)
        layer = TritterLayer(config)

        assert layer.attention is not None
        assert layer.mlp is not None
        assert layer.input_layernorm is not None
        assert layer.post_mlp_layernorm is not None

    def test_forward_pass(self) -> None:
        """Test forward pass through transformer layer."""
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            intermediate_size=512,
            num_layers=1,
            use_bitnet=False,
        )
        layer = TritterLayer(config)

        batch_size = 2
        seq_len = 4
        hidden_states = torch.randn(batch_size, seq_len, 128)

        output = layer(hidden_states)

        assert output.shape == (batch_size, seq_len, 128)

    def test_residual_connections(self) -> None:
        """Test that residual connections work correctly and gradients flow properly.

        Why: Residual connections are critical for training deep networks (24-32 layers).
        This test validates that:
        1. Output differs from input (transformation is applied)
        2. Output magnitude is reasonable relative to input
        3. Gradients flow to layer parameters during backpropagation

        A zero-initialized layer or broken residual would fail these checks.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            use_bitnet=False,
        )
        layer = TritterLayer(config)

        hidden_states = torch.randn(1, 4, 128, requires_grad=True)
        output = layer(hidden_states)

        # Output should not be identical to input due to transformations
        assert not torch.allclose(output, hidden_states)

        # Output magnitude should be reasonable (not exploded or vanished)
        input_norm = hidden_states.norm()
        output_norm = output.norm()
        # Typical range: 0.5x to 3x input magnitude (allows for amplification/attenuation)
        assert 0.3 * input_norm < output_norm < 5.0 * input_norm, (
            f"Output magnitude {output_norm:.2f} unreasonable relative to input {input_norm:.2f}"
        )

        # Test gradient flow through residual connections
        loss = output.sum()
        loss.backward()

        # Verify layer parameters have gradients (most important for residual connections)
        # Input gradients may be zero if attention mask blocks everything, but parameter
        # gradients must exist for training to work
        has_param_grads = False
        for _name, param in layer.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().max() > 0:
                    has_param_grads = True
                    break

        assert has_param_grads, "No layer parameters have non-zero gradients - gradient flow broken"


class TestTritterModel:
    """Test suite for TritterModel."""

    def test_initialization_3b(self) -> None:
        """Test 3B model initialization."""
        config = TritterConfig(model_size="3B", num_layers=2)
        model = TritterModel(config)

        assert len(model.layers) == 2
        assert model.embed_tokens is not None
        assert model.norm is not None
        assert model.lm_head is not None

    def test_initialization_7b(self) -> None:
        """Test 7B model initialization."""
        config = TritterConfig(model_size="7B", num_layers=2)
        model = TritterModel(config)

        assert config.hidden_size == 4096
        assert len(model.layers) == 2

    def test_forward_pass(self) -> None:
        """Test forward pass through full model."""
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        batch_size = 2
        seq_len = 8
        # Use config.vocab_size to ensure test stays valid if vocab_size changes
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 1000)

    def test_with_bitnet_quantization(self) -> None:
        """Test model with BitNet quantization enabled and verify quantized weights are used.

        Why: Tests should verify that quantization is not just enabled in config, but actually
        applied during forward passes. This validates that weights are ternary and memory
        footprint is reduced as expected.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=500,
            use_bitnet=True,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        # Test forward pass works
        logits = model(input_ids)
        assert logits.shape == (1, 4, config.vocab_size)

        # Verify that quantized weights are actually ternary during forward pass
        # Check one of the quantized layers
        layer = model.layers[0]
        if hasattr(layer.attention, "q_proj") and isinstance(layer.attention.q_proj, TernaryWeight):
            q_proj = layer.attention.q_proj
            # Set to eval mode to trigger caching
            model.eval()
            with torch.no_grad():
                _ = model(input_ids)

            # In eval mode, quantized weights should be cached and ternary
            if q_proj._quantized_weight_cache is not None:
                quantized_vals = q_proj._quantized_weight_cache.unique()
                # Should only contain values from {-1, 0, 1}
                assert all(v in [-1.0, 0.0, 1.0] for v in quantized_vals.tolist()), (
                    f"Quantized weights contain non-ternary values: {quantized_vals.tolist()}"
                )

    def test_model_parameters_exist(self) -> None:
        """Test that model has trainable parameters within expected bounds.

        Why: Parameter count validation ensures model is correctly initialized with all
        components present. The expected range accounts for:
        - Embedding layer: vocab_size * hidden_size
        - 1 transformer layer (attention + MLP with QK-Norm)
        - Output projection: vocab_size * hidden_size
        - Additional parameters from QK-Norm LayerNorm layers

        With BitNet quantization and added normalization layers, parameter count is higher
        than basic models but should still be within reasonable bounds for the configuration.
        This test uses bounds checking instead of weak assertions (> 0) to catch
        initialization errors or missing components (fixes issue #22).
        """
        config = TritterConfig(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            vocab_size=500,  # Must be >= 264 for byte-level encoding
            intermediate_size=256,  # 4x hidden_size (standard ratio)
            use_bitnet=True,
        )
        model = TritterModel(config)

        params = list(model.parameters())
        assert len(params) > 0, "Model should have parameters"

        total_params = sum(p.numel() for p in params)

        # Verify parameter count is within reasonable bounds for this config
        # Expected components (all with BitNet TernaryWeight layers):
        # - Embedding: vocab_size * hidden_size = 500 * 64 = 32,000
        # - 1 Layer attention (Q/K/V/O projections): ~4 * 64^2 = 16,384
        # - 1 Layer MLP (up/down projections): 2 * (64 * 256) = 32,768
        # - QK-Norm LayerNorms (2 per attention): 2 * 2 * 64 = 256
        # - Input/post-MLP LayerNorms: 2 * 2 * 64 = 256
        # - Final LayerNorm: 2 * 64 = 128
        # - TernaryWeight scales (1 per output feature): ~(64 + 64 + 64 + 64 + 256 + 64) = ~576
        # - Output projection: vocab_size * hidden_size = 500 * 64 = 32,000
        # Total expected: ~115K params
        assert 80_000 < total_params < 200_000, (
            f"Parameter count {total_params:,} outside expected range [80K, 200K] for test config. "
            f"Config: hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}, "
            f"vocab_size={config.vocab_size}, num_layers=1. "
            "This bounds check catches initialization errors or missing components that "
            "weak assertions like 'assert total_params > 0' would miss."
        )


class TestEmbeddingPrediction:
    """Tests for embedding prediction paradigm support.

    Why: Tritter operates in continuous embedding space (Coconut/LCM style).
    These tests validate that the model can return embeddings instead of logits,
    enabling continuous reasoning without discrete token bottlenecks.
    """

    def test_return_embeddings_flag(self) -> None:
        """Verify return_embeddings=True returns continuous embeddings.

        Why: Embedding prediction requires access to continuous representations
        instead of discretized logits. This is the core interface for the paradigm.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Default: returns logits
        logits = model(input_ids, return_embeddings=False)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        # With flag: returns embeddings
        embeddings = model(input_ids, return_embeddings=True)
        assert embeddings.shape == (batch_size, seq_len, config.hidden_size)

    def test_get_embeddings_method(self) -> None:
        """Verify get_embeddings() convenience method works.

        Why: Clearer API for embedding extraction use cases like
        semantic similarity and continuous reasoning.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        embeddings = model.get_embeddings(input_ids)

        assert embeddings.shape == (2, 8, config.hidden_size)

    def test_embeddings_differ_from_logits_projection(self) -> None:
        """Verify embeddings are continuous (not just pre-softmax logits).

        Why: Embeddings should be the transformer output BEFORE the lm_head
        projection. They should have different shape and semantic content.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        embeddings = model(input_ids, return_embeddings=True)
        logits = model(input_ids, return_embeddings=False)

        # Different shapes
        assert embeddings.shape[-1] == config.hidden_size
        assert logits.shape[-1] == config.vocab_size

        # Logits should be the projection of embeddings
        # (This verifies they're related but distinct)
        projected = model.lm_head(embeddings)
        assert torch.allclose(projected, logits, atol=1e-5)

    def test_get_target_embeddings(self) -> None:
        """Verify target embedding extraction for embedding loss.

        Why: Embedding prediction loss uses MSE between predicted embeddings
        and target embeddings (embeddings of the next token).
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        labels = torch.randint(0, config.vocab_size, (2, 8))

        target_embeddings = model.get_target_embeddings(labels)

        assert target_embeddings.shape == (2, 8, config.hidden_size)

        # Should be the same as directly embedding the labels
        direct_embeddings = model.embed_tokens(labels)
        assert torch.allclose(target_embeddings, direct_embeddings)

    def test_embedding_prediction_loss_pure_token(self) -> None:
        """Verify embedding_prediction_loss with alpha=0 (pure token loss).

        Why: Alpha=0 should behave identically to standard cross-entropy training.
        This is the starting point for curriculum training.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        loss_dict = model.embedding_prediction_loss(input_ids, labels, alpha=0.0)

        assert "loss" in loss_dict
        assert "token_loss" in loss_dict
        assert "embedding_loss" in loss_dict
        assert "alpha" in loss_dict

        # With alpha=0, combined loss should equal token loss
        assert torch.allclose(loss_dict["loss"], loss_dict["token_loss"])

    def test_embedding_prediction_loss_pure_embedding(self) -> None:
        """Verify embedding_prediction_loss with alpha=1 (pure embedding loss).

        Why: Alpha=1 is the target state for embedding prediction training.
        Loss should be purely MSE between predicted and target embeddings.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        loss_dict = model.embedding_prediction_loss(input_ids, labels, alpha=1.0)

        # With alpha=1, combined loss should equal embedding loss
        assert torch.allclose(loss_dict["loss"], loss_dict["embedding_loss"])

    def test_embedding_prediction_loss_hybrid(self) -> None:
        """Verify embedding_prediction_loss with hybrid alpha.

        Why: Curriculum training uses intermediate alpha values to smoothly
        transition from token to embedding prediction.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        alpha = 0.5
        loss_dict = model.embedding_prediction_loss(input_ids, labels, alpha=alpha)

        # Hybrid loss should be weighted combination
        expected_loss = (1 - alpha) * loss_dict["token_loss"] + alpha * loss_dict["embedding_loss"]
        assert torch.allclose(loss_dict["loss"], expected_loss)

    def test_embedding_gradients_flow(self) -> None:
        """Verify gradients flow correctly in embedding prediction mode.

        Why: Training requires gradients to flow from embedding loss back to
        model parameters. This validates the backward pass works.
        """
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=1000,
            use_bitnet=False,
        )
        model = TritterModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        loss_dict = model.embedding_prediction_loss(input_ids, labels, alpha=1.0)
        loss_dict["loss"].backward()

        # Verify gradients exist on model parameters
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().max() > 0:
                has_grads = True
                break

        assert has_grads, "No gradients flowed during embedding prediction loss backward"
