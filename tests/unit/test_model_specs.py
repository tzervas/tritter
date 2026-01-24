"""Unit tests for model specifications.

Why: Model specs define the architecture for all supported sizes (1B-70B).
These tests ensure:
1. All model sizes have valid, consistent specifications
2. Memory estimates are accurate and reasonable
3. Hardware recommendations are appropriate for each size
4. Parameter counts match expected values
5. GQA configurations are correctly detected

Testing strategy: Validates spec consistency across all sizes, verifies
parameter count calculations, and ensures memory estimates are within
expected ranges for BitNet-quantized models.
"""

import pytest

from tritter.core.model_specs import (
    MODEL_SPECS,
    HardwareRecommendation,
    MemoryEstimate,
    estimate_memory,
    get_model_spec,
    list_models,
    recommend_hardware,
)


class TestModelSpec:
    """Test suite for ModelSpec dataclass."""

    def test_all_sizes_exist(self) -> None:
        """Test that all documented sizes have specs.

        Why: Ensures the registry is complete and no sizes are missing.
        """
        expected_sizes = ["1B", "3B", "7B", "10B", "13B", "30B", "33B", "40B", "65B", "70B"]
        for size in expected_sizes:
            assert size in MODEL_SPECS, f"Missing spec for {size}"

    def test_spec_consistency(self) -> None:
        """Test that all specs have consistent values.

        Why: Validates that hidden_size is divisible by num_heads for all sizes.
        """
        for size, spec in MODEL_SPECS.items():
            assert spec.hidden_size % spec.num_heads == 0, (
                f"{size}: hidden_size ({spec.hidden_size}) must divide by num_heads ({spec.num_heads})"
            )

    def test_effective_head_dim(self) -> None:
        """Test effective_head_dim property.

        Why: Head dimension must be computed correctly from hidden_size / num_heads.
        """
        spec = get_model_spec("7B")
        expected = spec.hidden_size // spec.num_heads
        assert spec.effective_head_dim == expected

    def test_effective_num_kv_heads_mha(self) -> None:
        """Test effective_num_kv_heads for MHA (no GQA).

        Why: When num_kv_heads is None, should equal num_heads.
        """
        spec = get_model_spec("1B")  # Uses MHA
        assert spec.num_kv_heads is None
        assert spec.effective_num_kv_heads == spec.num_heads

    def test_effective_num_kv_heads_gqa(self) -> None:
        """Test effective_num_kv_heads for GQA.

        Why: When GQA is used, should return the configured kv_heads.
        """
        spec = get_model_spec("7B")  # Uses GQA
        assert spec.num_kv_heads is not None
        assert spec.effective_num_kv_heads == spec.num_kv_heads

    def test_uses_gqa_detection(self) -> None:
        """Test GQA detection property.

        Why: uses_gqa should correctly identify when KV heads < query heads.
        """
        # 1B uses MHA
        spec_mha = get_model_spec("1B")
        assert spec_mha.uses_gqa is False

        # 7B uses GQA
        spec_gqa = get_model_spec("7B")
        assert spec_gqa.uses_gqa is True

    def test_total_params_reasonable(self) -> None:
        """Test that total_params returns reasonable values.

        Why: Parameter counts should roughly match the model name.
        """
        # Tolerances are approximate - actual params depend on vocab size and GQA
        tolerances = {
            "1B": (0.8e9, 1.8e9),
            "3B": (2.0e9, 4.0e9),  # Adjusted for actual 3B spec
            "7B": (6.0e9, 8.5e9),
            "10B": (8.0e9, 13.0e9),
            "13B": (11.0e9, 16.0e9),
            "30B": (25.0e9, 38.0e9),
            "33B": (28.0e9, 42.0e9),
            "40B": (35.0e9, 50.0e9),
            "65B": (55.0e9, 75.0e9),
            "70B": (60.0e9, 85.0e9),
        }

        for size, (min_params, max_params) in tolerances.items():
            spec = get_model_spec(size)
            params = spec.total_params()
            assert min_params <= params <= max_params, (
                f"{size}: params {params / 1e9:.2f}B outside expected range "
                f"[{min_params / 1e9:.1f}B, {max_params / 1e9:.1f}B]"
            )

    def test_total_params_billions(self) -> None:
        """Test total_params_billions convenience method."""
        spec = get_model_spec("7B")
        params_b = spec.total_params_billions()
        assert 6.0 <= params_b <= 8.5, f"7B params: {params_b:.2f}B"

    def test_models_scale_correctly(self) -> None:
        """Test that larger models have more parameters.

        Why: Ensures the model progression 1B < 3B < 7B < ... < 70B is maintained.
        """
        sizes = ["1B", "3B", "7B", "10B", "13B", "30B", "33B", "40B", "65B", "70B"]
        prev_params = 0
        for size in sizes:
            spec = get_model_spec(size)
            params = spec.total_params()
            assert params > prev_params, (
                f"{size} ({params / 1e9:.2f}B) should have more params than previous"
            )
            prev_params = params


class TestMemoryEstimate:
    """Test suite for memory estimation."""

    def test_estimate_memory_returns_estimate(self) -> None:
        """Test that estimate_memory returns MemoryEstimate."""
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)
        assert isinstance(mem, MemoryEstimate)

    def test_packed_smaller_than_fp32(self) -> None:
        """Test that packed ternary is smaller than FP32.

        Why: Packed 2-bit storage should be ~16x smaller than FP32.
        """
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)

        ratio = mem.weights_fp32 / mem.weights_packed_ternary
        assert ratio > 10, f"Packed ratio {ratio:.1f}x is too low (expected >10x)"

    def test_kv_cache_scales_with_context(self) -> None:
        """Test that KV-cache scales with context length.

        Why: Longer context requires proportionally more KV-cache memory.
        """
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)

        # 128K should be ~32x larger than 4K
        ratio = mem.kv_cache_128k_fp16 / mem.kv_cache_4k_fp16
        assert 30 <= ratio <= 34, f"KV cache ratio {ratio:.1f}x unexpected"

    def test_int4_kv_smaller_than_fp16(self) -> None:
        """Test that INT4 KV-cache is smaller than FP16.

        Why: INT4 is 4 bits vs FP16's 16 bits = 4x reduction.
        """
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)

        ratio = mem.kv_cache_32k_fp16 / mem.kv_cache_32k_int4
        assert 3.5 <= ratio <= 4.5, f"INT4 ratio {ratio:.1f}x unexpected (expected ~4x)"

    def test_training_memory_larger_than_inference(self) -> None:
        """Test that training requires more memory than inference weights.

        Why: Training needs weights + gradients + optimizer states.
        """
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)

        # FP32 training should be ~4x weight storage (weights + grads + 2x optimizer)
        ratio = mem.training_fp32 / mem.weights_fp32
        assert ratio >= 3.5, f"Training ratio {ratio:.1f}x is too low"

    def test_7b_fits_rtx5080_packed(self) -> None:
        """Test that 7B packed model fits in RTX 5080 16GB.

        Why: Core requirement - 7B inference must work on target hardware.
        """
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)

        # Packed weights + 32K INT4 KV-cache + 2GB overhead should fit
        total_gb = (mem.weights_packed_ternary + mem.kv_cache_32k_int4 + 2 * 1024**3) / (1024**3)

        assert total_gb < 16, f"7B with 32K context needs {total_gb:.1f}GB, exceeds 16GB"

    def test_summary_format(self) -> None:
        """Test that summary() returns formatted string."""
        spec = get_model_spec("7B")
        mem = estimate_memory(spec)
        summary = mem.summary()

        assert "Memory Estimate" in summary
        assert "Weights (FP32)" in summary
        assert "KV-Cache" in summary


class TestHardwareRecommendation:
    """Test suite for hardware recommendations."""

    def test_recommend_hardware_returns_recommendation(self) -> None:
        """Test that recommend_hardware returns HardwareRecommendation."""
        rec = recommend_hardware("7B", target_vram_gb=16.0)
        assert isinstance(rec, HardwareRecommendation)

    def test_small_model_no_streaming(self) -> None:
        """Test that small models don't require streaming on 16GB.

        Why: 3B and smaller should fit entirely in 16GB VRAM.
        """
        rec = recommend_hardware("3B", target_vram_gb=16.0)
        assert rec.use_layer_streaming is False

    def test_large_model_recommends_streaming(self) -> None:
        """Test that large models recommend streaming on limited VRAM.

        Why: 70B model with 32K context exceeds 16GB VRAM.
        """
        # 70B model needs streaming on 16GB
        rec = recommend_hardware("70B", target_vram_gb=16.0)
        assert rec.use_layer_streaming is True

        # Or 30B on 8GB needs streaming
        rec_8gb = recommend_hardware("30B", target_vram_gb=8.0)
        assert rec_8gb.use_layer_streaming is True

    def test_multi_gpu_recommendation(self) -> None:
        """Test multi-GPU recommendations for large models.

        Why: 65B/70B should recommend tensor parallelism on 4+ GPUs.
        """
        rec = recommend_hardware("70B", target_vram_gb=24.0, target_gpus=4)
        assert rec.tensor_parallel_size > 1

    def test_context_recommendation_scales_with_vram(self) -> None:
        """Test that recommended context scales with available VRAM."""
        rec_small = recommend_hardware("7B", target_vram_gb=8.0)
        rec_large = recommend_hardware("7B", target_vram_gb=24.0)

        assert rec_large.recommended_context_length >= rec_small.recommended_context_length

    def test_gqa_detection(self) -> None:
        """Test that GQA is correctly detected from spec."""
        rec_7b = recommend_hardware("7B")
        rec_1b = recommend_hardware("1B")

        assert rec_7b.use_gqa is True  # 7B uses GQA
        assert rec_1b.use_gqa is False  # 1B uses MHA

    def test_training_warning_for_large_models(self) -> None:
        """Test that training gives warnings for insufficient VRAM.

        Why: Large model training requires significant memory for gradients
        and optimizer states.
        """
        rec = recommend_hardware("70B", target_vram_gb=24.0, target_gpus=1, for_training=True)
        # Should have a warning note about insufficient memory
        assert any("WARNING" in note for note in rec.notes)


class TestListModels:
    """Test suite for list_models utility."""

    def test_list_models_returns_all(self) -> None:
        """Test that list_models returns all supported sizes."""
        models = list_models()
        assert len(models) == len(MODEL_SPECS)

    def test_list_models_format(self) -> None:
        """Test list_models returns correct tuple format."""
        models = list_models()
        for size, params_b, desc in models:
            assert isinstance(size, str)
            assert isinstance(params_b, float)
            assert isinstance(desc, str)
            assert params_b > 0


class TestGetModelSpec:
    """Test suite for get_model_spec function."""

    def test_get_valid_size(self) -> None:
        """Test getting spec for valid size."""
        spec = get_model_spec("7B")
        assert spec.name == "7B"

    def test_get_invalid_size_raises(self) -> None:
        """Test that invalid size raises KeyError."""
        with pytest.raises(KeyError, match="Unknown model size"):
            get_model_spec("99B")  # type: ignore[arg-type]


class TestConfigIntegration:
    """Test integration between model_specs and TritterConfig."""

    def test_config_uses_spec_values(self) -> None:
        """Test that TritterConfig correctly applies model specs."""
        from tritter.core.config import TritterConfig

        for size in ["1B", "3B", "7B", "10B", "13B"]:
            spec = get_model_spec(size)  # type: ignore[arg-type]
            config = TritterConfig(model_size=size)  # type: ignore[arg-type]

            assert config.hidden_size == spec.hidden_size, f"{size} hidden_size mismatch"
            assert config.num_layers == spec.num_layers, f"{size} num_layers mismatch"
            assert config.num_heads == spec.num_heads, f"{size} num_heads mismatch"

    def test_config_estimate_memory(self) -> None:
        """Test TritterConfig.estimate_memory method."""
        from tritter.core.config import TritterConfig

        config = TritterConfig(model_size="7B")
        mem = config.estimate_memory()
        assert isinstance(mem, MemoryEstimate)

    def test_config_hardware_recommendation(self) -> None:
        """Test TritterConfig.get_hardware_recommendation method."""
        from tritter.core.config import TritterConfig

        config = TritterConfig(model_size="7B")
        rec = config.get_hardware_recommendation()
        assert isinstance(rec, HardwareRecommendation)

    def test_config_total_params(self) -> None:
        """Test TritterConfig.total_params method."""
        from tritter.core.config import TritterConfig

        config = TritterConfig(model_size="7B")
        params = config.total_params()
        assert 6e9 < params < 8.5e9

    def test_config_gqa_properties(self) -> None:
        """Test TritterConfig GQA properties."""
        from tritter.core.config import TritterConfig

        config_gqa = TritterConfig(model_size="7B")
        assert config_gqa.uses_gqa is True
        assert config_gqa.effective_num_kv_heads == 8

        config_mha = TritterConfig(model_size="1B")
        assert config_mha.uses_gqa is False
        assert config_mha.effective_num_kv_heads == config_mha.num_heads
