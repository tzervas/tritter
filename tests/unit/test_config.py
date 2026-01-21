"""Unit tests for configuration module."""

import pytest

from tritter.core.config import TritterConfig


class TestTritterConfig:
    """Test suite for TritterConfig class."""
    
    def test_default_config_3b(self) -> None:
        """Test default 3B model configuration."""
        config = TritterConfig()
        
        assert config.model_size == "3B"
        assert config.hidden_size == 2048
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.head_dim == 128
        assert config.max_position_embeddings == 131072  # 128K
    
    def test_config_7b(self) -> None:
        """Test 7B model configuration."""
        config = TritterConfig(model_size="7B")
        
        assert config.model_size == "7B"
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.head_dim == 128
    
    def test_bitnet_enabled_by_default(self) -> None:
        """Test that BitNet quantization is enabled by default."""
        config = TritterConfig()
        
        assert config.use_bitnet is True
        assert config.bitnet_precision == 1.58
    
    def test_multimodal_config(self) -> None:
        """Test multimodal configuration."""
        config = TritterConfig()
        
        assert "text" in config.modalities
        assert "code" in config.modalities
        assert "image" in config.modalities
        assert "audio" in config.modalities
        assert config.use_early_fusion is True
        assert config.unified_embedding is True
    
    def test_hardware_config(self) -> None:
        """Test hardware-specific configuration."""
        config = TritterConfig()
        
        assert config.target_device == "cuda"
        assert config.max_memory_gb == 16  # RTX 5080 GDDR7
    
    def test_attention_optimizations(self) -> None:
        """Test attention optimization flags."""
        config = TritterConfig()
        
        assert config.use_flash_attention is True
        assert config.sliding_window_size == 4096
        assert config.use_streaming_llm is True
        assert config.int4_kv_cache is True
    
    def test_invalid_head_dimension_raises_error(self) -> None:
        """Test that invalid head dimension raises assertion error."""
        with pytest.raises(AssertionError):
            TritterConfig(hidden_size=2049, num_heads=16)
    
    def test_invalid_modality_raises_error(self) -> None:
        """Test that invalid modality raises assertion error."""
        with pytest.raises(AssertionError):
            TritterConfig(modalities=["text", "video", "invalid"])
    
    def test_custom_modalities(self) -> None:
        """Test configuration with custom modality subset."""
        config = TritterConfig(modalities=["text", "code"])
        
        assert len(config.modalities) == 2
        assert "text" in config.modalities
        assert "code" in config.modalities
        assert "image" not in config.modalities
