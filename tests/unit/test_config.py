"""Unit tests for configuration module.

Validates TritterConfig dataclass behavior, default values, and constraint checking.

Why: Configuration is the foundation of model architecture - incorrect config leads to
runtime errors or suboptimal performance. These tests ensure:
1. Default 3B/7B configs match project-plan.md specifications
2. Hardware constraints (16GB VRAM) are respected
3. BitNet quantization defaults are correct
4. Multimodal settings enable proper early fusion
5. Invalid configurations fail fast with clear errors

Testing strategy: Validates both happy paths (correct configs) and error paths (invalid
combinations like hidden_size not divisible by num_heads), ensuring fail-fast behavior.
"""

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
        assert config.sliding_window_size is None  # Not yet implemented
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

    def test_attention_mode_defaults(self) -> None:
        """Test default attention mode configuration.

        Validates that attention_mode defaults to "causal" for standard autoregressive
        pretraining and generation use cases.
        """
        config = TritterConfig()

        assert config.attention_mode == "causal"
        assert config.use_sliding_window is False
        assert config.sliding_window_size is None
        assert config.use_attention_sinks is False
        assert config.num_sink_tokens == 4

    def test_attention_mode_bidirectional(self) -> None:
        """Test bidirectional attention mode configuration.

        Validates that bidirectional mode can be set for semantic embedding extraction
        where all tokens attend to all other tokens.
        """
        config = TritterConfig(attention_mode="bidirectional")

        assert config.attention_mode == "bidirectional"

    def test_attention_mode_prefix_lm(self) -> None:
        """Test prefix-LM attention mode configuration.

        Validates that prefix-LM mode can be configured for instruction tuning where
        the prefix (instructions) uses bidirectional attention and response uses causal.
        """
        config = TritterConfig(attention_mode="prefix_lm")

        assert config.attention_mode == "prefix_lm"

    def test_attention_mode_embedding(self) -> None:
        """Test embedding attention mode configuration.

        Validates that embedding mode (Coconut-style continuous reasoning) can be set
        for latent refinement in continuous embedding space.
        """
        config = TritterConfig(attention_mode="embedding")

        assert config.attention_mode == "embedding"

    def test_invalid_attention_mode_raises_error(self) -> None:
        """Test that invalid attention mode raises assertion error.

        Validates fail-fast behavior for unsupported attention patterns to catch
        configuration errors at init time.
        """
        with pytest.raises(AssertionError):
            TritterConfig(attention_mode="invalid_mode")

    def test_sliding_window_configuration(self) -> None:
        """Test sliding window attention configuration.

        Validates that sliding window parameters can be configured together.
        Sliding window bounds KV-cache to window_size tokens, reducing memory
        from O(N²) to O(N*W).
        """
        config = TritterConfig(
            use_sliding_window=True,
            sliding_window_size=4096,
        )

        assert config.use_sliding_window is True
        assert config.sliding_window_size == 4096

    def test_sliding_window_requires_positive_size(self) -> None:
        """Test that sliding window requires positive size.

        Validates that enabling use_sliding_window=True without a valid window_size
        fails with assertion error. Zero or None window sizes would degenerate attention.
        """
        with pytest.raises(AssertionError):
            TritterConfig(use_sliding_window=True, sliding_window_size=None)

        with pytest.raises(AssertionError):
            TritterConfig(use_sliding_window=True, sliding_window_size=0)

        with pytest.raises(AssertionError):
            TritterConfig(use_sliding_window=True, sliding_window_size=-1)

    def test_attention_sinks_configuration(self) -> None:
        """Test attention sinks (StreamingLLM) configuration.

        Validates that attention sink parameters can be configured for streaming
        generation. Sinks preserve early context (e.g., system prompt) during
        KV-cache eviction.
        """
        config = TritterConfig(
            use_attention_sinks=True,
            num_sink_tokens=8,
        )

        assert config.use_attention_sinks is True
        assert config.num_sink_tokens == 8

    def test_attention_sinks_requires_positive_num_sink_tokens(self) -> None:
        """Test that attention sinks requires positive num_sink_tokens.

        Validates that enabling use_attention_sinks=True without a valid num_sink_tokens
        fails with assertion error. Zero or negative sink tokens would make the
        StreamingLLM attention pattern degenerate or invalid.
        """
        with pytest.raises(AssertionError):
            TritterConfig(use_attention_sinks=True, num_sink_tokens=None)

        with pytest.raises(AssertionError):
            TritterConfig(use_attention_sinks=True, num_sink_tokens=0)

        with pytest.raises(AssertionError):
            TritterConfig(use_attention_sinks=True, num_sink_tokens=-1)

    def test_attention_config_all_modes(self) -> None:
        """Test all valid attention mode combinations.

        Validates that each attention mode can be independently configured
        without affecting other components.
        """
        for mode in ["causal", "bidirectional", "prefix_lm", "embedding"]:
            config = TritterConfig(attention_mode=mode)
            assert config.attention_mode == mode
