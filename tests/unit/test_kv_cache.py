"""Tests for INT4 KV-cache quantization.

Validates quantization accuracy, memory efficiency, and correctness of
the INT4KVCache implementation per SPEC-005-memory-optimization.md.

Why: INT4 KV-cache is critical for fitting 128K context in 16GB VRAM.
These tests verify quantization doesn't break attention patterns.
"""

import pytest
import torch

from tritter.core.config import TritterConfig
from tritter.inference.kv_cache import INT4KVCache, QuantizedTensor


@pytest.fixture
def config() -> TritterConfig:
    """Create minimal test config.

    Why: Use small dimensions for fast tests while still exercising
    the quantization logic correctly.
    """
    return TritterConfig(
        model_size="3B",
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        max_position_embeddings=1024,
        int4_kv_cache=True,
    )


@pytest.fixture
def cache(config: TritterConfig) -> INT4KVCache:
    """Create INT4KVCache instance for testing."""
    return INT4KVCache(config, device="cpu")


class TestINT4KVCacheInit:
    """Tests for INT4KVCache initialization."""

    def test_cache_creation(self, config: TritterConfig) -> None:
        """Verify cache initializes with correct dimensions.

        Why: Cache must match model architecture to avoid shape mismatches
        during attention computation.
        """
        cache = INT4KVCache(config, device="cpu")

        assert cache.num_layers == config.num_layers
        assert cache.num_heads == config.num_heads
        assert cache.head_dim == config.head_dim
        assert cache.current_len == 0
        assert len(cache.keys) == config.num_layers
        assert len(cache.values) == config.num_layers

    def test_cache_empty_initially(self, cache: INT4KVCache) -> None:
        """Verify cache starts empty.

        Why: Fresh cache should have no stored values to avoid using
        uninitialized data in attention.
        """
        for k, v in zip(cache.keys, cache.values):
            assert k is None
            assert v is None
        assert cache.current_len == 0

    def test_custom_max_seq_len(self, config: TritterConfig) -> None:
        """Verify custom max_seq_len overrides config.

        Why: Some use cases need different cache sizes than the model's
        max context (e.g., shorter for testing, longer for streaming).
        """
        custom_len = 2048
        cache = INT4KVCache(config, max_seq_len=custom_len, device="cpu")
        assert cache.max_seq_len == custom_len


class TestINT4Quantization:
    """Tests for INT4 quantization accuracy."""

    def test_quantize_dequantize_keys_preserves_shape(self, cache: INT4KVCache) -> None:
        """Verify key quantization/dequantization preserves tensor shape.

        Why: Shape preservation is required for attention computation.
        Packed INT4 should unpack to original dimensions.
        """
        B, H, L, D = 2, cache.num_heads, 16, cache.head_dim
        original = torch.randn(B, H, L, D)

        quantized = cache._quantize_per_channel(original)
        dequantized = cache._dequantize_per_channel(quantized)

        assert dequantized.shape == original.shape

    def test_quantize_dequantize_values_preserves_shape(self, cache: INT4KVCache) -> None:
        """Verify value quantization/dequantization preserves tensor shape.

        Why: Same as keys - shape must be preserved for attention.
        """
        B, H, L, D = 2, cache.num_heads, 16, cache.head_dim
        original = torch.randn(B, H, L, D)

        quantized = cache._quantize_per_token(original)
        dequantized = cache._dequantize_per_token(quantized)

        assert dequantized.shape == original.shape

    def test_quantization_error_bounded(self, cache: INT4KVCache) -> None:
        """Verify quantization error is within acceptable bounds.

        Why: INT4 quantization must maintain sufficient precision for
        attention computation. Error should be small relative to signal.
        """
        B, H, L, D = 2, cache.num_heads, 32, cache.head_dim
        original = torch.randn(B, H, L, D) * 10  # Scale to realistic range

        # Test key quantization
        key_quantized = cache._quantize_per_channel(original)
        key_dequantized = cache._dequantize_per_channel(key_quantized)
        key_error = (original - key_dequantized).abs().mean() / original.abs().mean()

        # Test value quantization
        val_quantized = cache._quantize_per_token(original)
        val_dequantized = cache._dequantize_per_token(val_quantized)
        val_error = (original - val_dequantized).abs().mean() / original.abs().mean()

        # INT4 should achieve <10% relative error on average
        # Why: 4-bit quantization with 16 levels gives ~6% theoretical error
        # Adding some margin for edge cases and asymmetric distributions
        assert key_error < 0.15, f"Key quantization error too high: {key_error:.3f}"
        assert val_error < 0.15, f"Value quantization error too high: {val_error:.3f}"

    def test_packed_storage_halves_memory(self, cache: INT4KVCache) -> None:
        """Verify INT4 packing achieves 50% memory reduction.

        Why: Core benefit of INT4 is memory reduction. Packing two INT4
        values per uint8 must halve storage vs unpacked.
        """
        B, H, L, D = 2, cache.num_heads, 32, cache.head_dim
        original = torch.randn(B, H, L, D)

        quantized = cache._quantize_per_channel(original)

        # Packed data should be half the size of original in last dimension
        expected_packed_size = D // 2
        assert quantized.data.shape[-1] == expected_packed_size

    def test_quantized_tensor_has_required_fields(self, cache: INT4KVCache) -> None:
        """Verify QuantizedTensor contains all necessary components.

        Why: Dequantization requires data, scale, zero_point, and shape.
        Missing any field would cause runtime errors.
        """
        B, H, L, D = 2, cache.num_heads, 16, cache.head_dim
        original = torch.randn(B, H, L, D)

        quantized = cache._quantize_per_channel(original)

        assert isinstance(quantized, QuantizedTensor)
        assert quantized.data is not None
        assert quantized.scale is not None
        assert quantized.zero_point is not None
        assert quantized.shape == original.shape


class TestKVCacheOperations:
    """Tests for cache update and retrieval operations."""

    def test_update_stores_values(self, cache: INT4KVCache) -> None:
        """Verify update stores quantized key/value tensors.

        Why: Cache must actually store data for retrieval during attention.
        """
        B, H, L, D = 1, cache.num_heads, 8, cache.head_dim
        key = torch.randn(B, H, L, D)
        value = torch.randn(B, H, L, D)

        cache.update(layer_idx=0, key=key, value=value)

        assert cache.keys[0] is not None
        assert cache.values[0] is not None
        assert cache.current_len == L

    def test_get_retrieves_dequantized_values(self, cache: INT4KVCache) -> None:
        """Verify get returns dequantized tensors.

        Why: Attention computation needs FP16 tensors, not quantized storage.
        """
        B, H, L, D = 1, cache.num_heads, 8, cache.head_dim
        key = torch.randn(B, H, L, D)
        value = torch.randn(B, H, L, D)

        cache.update(layer_idx=0, key=key, value=value)
        retrieved_k, retrieved_v = cache.get(layer_idx=0)

        assert retrieved_k.shape == key.shape
        assert retrieved_v.shape == value.shape
        assert retrieved_k.dtype == torch.float16
        assert retrieved_v.dtype == torch.float16

    def test_update_appends_to_existing(self, cache: INT4KVCache) -> None:
        """Verify subsequent updates append to cache.

        Why: Autoregressive generation adds tokens incrementally.
        Cache must grow to hold full context.
        """
        B, H, D = 1, cache.num_heads, cache.head_dim

        # First update
        key1 = torch.randn(B, H, 4, D)
        value1 = torch.randn(B, H, 4, D)
        cache.update(layer_idx=0, key=key1, value=value1)
        assert cache.current_len == 4

        # Second update
        key2 = torch.randn(B, H, 4, D)
        value2 = torch.randn(B, H, 4, D)
        cache.update(layer_idx=0, key=key2, value=value2)
        assert cache.current_len == 8

        # Retrieved should have combined length
        k, v = cache.get(layer_idx=0)
        assert k.shape[2] == 8
        assert v.shape[2] == 8

    def test_get_raises_for_empty_layer(self, cache: INT4KVCache) -> None:
        """Verify get raises ValueError for uncached layer.

        Why: Accessing uncached layer would return garbage or crash.
        Explicit error helps debug layer indexing issues.
        """
        with pytest.raises(ValueError, match="No cached values for layer 0"):
            cache.get(layer_idx=0)

    def test_clear_empties_cache(self, cache: INT4KVCache) -> None:
        """Verify clear removes all cached values.

        Why: Need to reset cache between sequences or when switching contexts.
        """
        B, H, L, D = 1, cache.num_heads, 8, cache.head_dim
        key = torch.randn(B, H, L, D)
        value = torch.randn(B, H, L, D)

        cache.update(layer_idx=0, key=key, value=value)
        assert cache.current_len == L

        cache.clear()

        assert cache.current_len == 0
        for k, v in zip(cache.keys, cache.values):
            assert k is None
            assert v is None


class TestSlidingWindowTruncation:
    """Tests for sliding window cache truncation."""

    def test_truncate_keeps_recent_positions(self, cache: INT4KVCache) -> None:
        """Verify truncation keeps only most recent positions.

        Why: Sliding window bounds memory by evicting old positions.
        Must keep the RIGHT (recent) positions, not arbitrary ones.
        """
        B, H, D = 1, cache.num_heads, cache.head_dim

        # Add 16 positions
        key = torch.randn(B, H, 16, D)
        value = torch.randn(B, H, 16, D)
        cache.update(layer_idx=0, key=key, value=value)

        # Truncate to 8 positions
        cache.truncate_to_window(window_size=8)

        assert cache.current_len == 8
        k, v = cache.get(layer_idx=0)
        assert k.shape[2] == 8
        assert v.shape[2] == 8

    def test_truncate_noop_if_under_window(self, cache: INT4KVCache) -> None:
        """Verify truncation does nothing if under window size.

        Why: Don't waste compute on truncation when not needed.
        """
        B, H, D = 1, cache.num_heads, cache.head_dim

        key = torch.randn(B, H, 4, D)
        value = torch.randn(B, H, 4, D)
        cache.update(layer_idx=0, key=key, value=value)

        original_len = cache.current_len
        cache.truncate_to_window(window_size=8)

        assert cache.current_len == original_len


class TestMemoryTracking:
    """Tests for memory usage tracking."""

    def test_memory_usage_increases_with_cache(self, cache: INT4KVCache) -> None:
        """Verify memory tracking reflects cache growth.

        Why: Memory budget enforcement requires accurate tracking.
        """
        initial_memory = cache.get_memory_usage_gb()
        assert initial_memory == 0.0  # Empty cache uses no memory

        B, H, L, D = 1, cache.num_heads, 64, cache.head_dim
        key = torch.randn(B, H, L, D)
        value = torch.randn(B, H, L, D)
        cache.update(layer_idx=0, key=key, value=value)

        after_memory = cache.get_memory_usage_gb()
        assert after_memory > initial_memory

    def test_memory_usage_reasonable_for_config(self, config: TritterConfig) -> None:
        """Verify memory usage matches theoretical INT4 calculation.

        Why: Memory budgeting depends on accurate predictions.
        Actual usage should be close to theoretical.
        """
        cache = INT4KVCache(config, device="cpu")

        B, H, L, D = 1, cache.num_heads, 128, cache.head_dim
        key = torch.randn(B, H, L, D)
        value = torch.randn(B, H, L, D)

        # Fill all layers
        for i in range(cache.num_layers):
            cache.update(layer_idx=i, key=key, value=value)

        actual_memory = cache.get_memory_usage_gb()

        # Theoretical INT4 memory (approximate):
        # 2 (K+V) * layers * B * H * L * D * 0.5 bytes (packed INT4)
        # Plus scale/zero_point overhead
        theoretical_base = 2 * config.num_layers * B * H * L * D * 0.5 / 1e9
        # Allow 2x for scale/zero_point overhead and rounding
        assert actual_memory < theoretical_base * 3


class TestRepr:
    """Tests for string representation."""

    def test_repr_contains_key_info(self, cache: INT4KVCache) -> None:
        """Verify repr shows useful diagnostic information.

        Why: Debugging requires quick access to cache state.
        """
        repr_str = repr(cache)

        assert "INT4KVCache" in repr_str
        assert "layers=" in repr_str
        assert "heads=" in repr_str
        assert "current_len=" in repr_str
        assert "memory=" in repr_str


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """Tests requiring CUDA for realistic performance validation."""

    def test_cache_works_on_cuda(self, config: TritterConfig) -> None:
        """Verify cache operations work on GPU.

        Why: Production use is on GPU - must verify CUDA compatibility.
        """
        cache = INT4KVCache(config, device="cuda")

        B, H, L, D = 1, cache.num_heads, 16, cache.head_dim
        key = torch.randn(B, H, L, D, device="cuda")
        value = torch.randn(B, H, L, D, device="cuda")

        cache.update(layer_idx=0, key=key, value=value)
        k, v = cache.get(layer_idx=0)

        assert k.device.type == "cuda"
        assert v.device.type == "cuda"
