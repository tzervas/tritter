"""Unit tests for model architecture."""

import torch
import pytest

from tritter.core.config import TritterConfig
from tritter.models.architecture import (
    TritterModel,
    TritterLayer,
    TritterAttention,
    TritterMLP,
)


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
        assert layer.post_attention_layernorm is not None
    
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
        """Test that residual connections work correctly."""
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            use_bitnet=False,
        )
        layer = TritterLayer(config)
        
        hidden_states = torch.randn(1, 4, 128)
        output = layer(hidden_states)
        
        # Output should not be identical to input due to transformations
        assert not torch.allclose(output, hidden_states)


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
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 1000)
    
    def test_with_bitnet_quantization(self) -> None:
        """Test model with BitNet quantization enabled."""
        config = TritterConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            vocab_size=500,
            use_bitnet=True,
        )
        model = TritterModel(config)
        
        input_ids = torch.randint(0, 500, (1, 4))
        logits = model(input_ids)
        
        assert logits.shape == (1, 4, 500)
    
    def test_model_parameters_exist(self) -> None:
        """Test that model has trainable parameters."""
        config = TritterConfig(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            vocab_size=100,
        )
        model = TritterModel(config)
        
        params = list(model.parameters())
        assert len(params) > 0
        
        total_params = sum(p.numel() for p in params)
        assert total_params > 0
