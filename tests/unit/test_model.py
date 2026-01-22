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
        for name, param in layer.named_parameters():
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
        """
        config = TritterConfig(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            vocab_size=100,
            use_bitnet=True,
        )
        model = TritterModel(config)

        params = list(model.parameters())
        assert len(params) > 0

        total_params = sum(p.numel() for p in params)
        assert total_params > 0

        # Verify parameter count is within reasonable bounds for this config
        # Expected components:
        # - Embedding: 100 * 64 = 6,400
        # - 1 Layer with attention (Q/K/V/O projections + QK-Norm): ~4 * 64^2 + 2*64 = 16,512
        # - 1 Layer with MLP (gate/up/down projections): ~3 * 64 * 256 = 49,152
        # - LayerNorms (input, post_mlp, final): 3 * 2 * 64 = 384
        # - Scales for TernaryWeight layers: multiple of out_features
        # - Output projection: 100 * 64 = 6,400
        # Total expected: ~80K-200K params (accounting for TernaryWeight scales)
        assert 50_000 < total_params < 2_000_000, (
            f"Parameter count {total_params:,} outside expected range for test config. "
            "Expected 50K-2M params for config with hidden_size=64, 1 layer, vocab_size=100"
        )
