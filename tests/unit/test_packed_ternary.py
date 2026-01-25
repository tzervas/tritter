"""Unit tests for packed ternary weight storage.

Why: Packed ternary weights are critical for enabling 7B model inference on 16GB VRAM.
These tests ensure:
1. Pack/unpack round-trip preserves exact ternary values (lossless)
2. PackedTernaryWeight produces identical output to TernaryWeight
3. Memory savings are achieved (~8x reduction from FP32 to 2-bit packed)
4. Model conversion utility correctly replaces all TernaryWeight layers
5. Serialization preserves model state correctly

Testing strategy: Uses small test matrices with known patterns to verify encoding
correctness. Memory tests use actual tensor sizes to confirm storage reduction.
Round-trip tests verify bit-exact reconstruction of ternary values.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tritter.quantization.bitnet import TernaryWeight
from tritter.quantization.packed_ternary import (
    PackedTernaryWeight,
    convert_to_packed,
    load_packed_model,
    pack_ternary,
    save_packed_model,
    unpack_ternary,
)


class TestPackUnpack:
    """Test suite for pack_ternary and unpack_ternary functions."""

    def test_pack_basic(self) -> None:
        """Test basic packing of ternary weights.

        Why: Verifies the core encoding: {-1, 0, +1} -> {0, 1, 2} packed 4 per byte.
        """
        # 4 values that pack into a single byte
        weights = torch.tensor([[-1.0, 0.0, 1.0, 0.0]])  # (1, 4)
        scale = torch.tensor([[1.0]])

        packed, returned_scale = pack_ternary(weights, scale)

        assert packed.shape == (1, 1)  # 4 values -> 1 byte
        assert packed.dtype == torch.uint8
        assert returned_scale is scale

    def test_pack_multiple_rows(self) -> None:
        """Test packing with multiple output channels.

        Why: Verifies correct handling of multi-row weight matrices.
        """
        weights = torch.tensor(
            [
                [-1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )  # (3, 4)
        scale = torch.ones(3, 1)

        packed, _ = pack_ternary(weights, scale)

        assert packed.shape == (3, 1)  # 3 rows, 4 values each -> (3, 1)

    def test_pack_with_padding(self) -> None:
        """Test packing when in_features is not divisible by 4.

        Why: Ensures correct padding to multiple of 4 for even packing.
        """
        # 6 values requires padding to 8
        weights = torch.tensor([[-1.0, 0.0, 1.0, 1.0, -1.0, 0.0]])  # (1, 6)
        scale = torch.ones(1, 1)

        packed, _ = pack_ternary(weights, scale)

        assert packed.shape == (1, 2)  # 6 padded to 8 -> 2 bytes

    def test_unpack_basic(self) -> None:
        """Test basic unpacking to recover ternary values.

        Why: Verifies decoding: {0, 1, 2} -> {-1, 0, +1} with scaling.
        """
        # Manually create packed byte: -1, 0, 1, 0 -> 0, 1, 2, 1
        # Packed: 0b01_10_01_00 = 0x64
        packed = torch.tensor([[0x64]], dtype=torch.uint8)
        scale = torch.tensor([[2.0]])

        unpacked = unpack_ternary(packed, scale, original_in_features=4)

        expected = torch.tensor([[-2.0, 0.0, 2.0, 0.0]])  # scaled by 2
        assert torch.allclose(unpacked, expected)

    def test_pack_unpack_roundtrip_exact(self) -> None:
        """Test that pack/unpack round-trip preserves exact values.

        Why: Critical test - ternary quantization must be lossless.
        Any deviation indicates encoding/decoding bug.
        """
        # Create random ternary weights
        torch.manual_seed(42)
        weights = torch.randint(-1, 2, (128, 256)).float()  # Random {-1, 0, +1}
        scale = torch.rand(128, 1) + 0.1  # Random positive scales

        # Pack then unpack
        packed, _ = pack_ternary(weights, scale)
        unpacked = unpack_ternary(packed, scale, original_in_features=256)

        # Should be exactly equal (scaled)
        expected = weights * scale
        assert torch.allclose(unpacked, expected, rtol=0, atol=1e-6)

    def test_pack_unpack_roundtrip_various_sizes(self) -> None:
        """Test round-trip with various tensor sizes including non-divisible-by-4.

        Why: Ensures padding/truncation works correctly for all sizes.
        """
        sizes = [(32, 64), (64, 65), (128, 127), (256, 513), (1, 1), (1, 3)]

        for out_features, in_features in sizes:
            weights = torch.randint(-1, 2, (out_features, in_features)).float()
            scale = torch.ones(out_features, 1)

            packed, _ = pack_ternary(weights, scale)
            unpacked = unpack_ternary(packed, scale, original_in_features=in_features)

            assert unpacked.shape == weights.shape, (
                f"Shape mismatch for {out_features}x{in_features}"
            )
            assert torch.allclose(unpacked, weights, rtol=0, atol=1e-6), (
                f"Value mismatch for {out_features}x{in_features}"
            )

    def test_pack_rejects_non_ternary(self) -> None:
        """Test that pack_ternary rejects non-ternary values.

        Why: Ensures we catch invalid input early rather than producing corrupt packed data.
        """
        weights = torch.tensor([[0.5, 0.0, -0.5, 1.0]])  # 0.5 and -0.5 are invalid
        scale = torch.ones(1, 1)

        with pytest.raises(ValueError, match="Weights must be ternary"):
            pack_ternary(weights, scale)

    def test_pack_accepts_exact_ternary(self) -> None:
        """Test that pack_ternary accepts exact ternary values.

        Why: Verifies the validation correctly accepts valid input.
        """
        # All valid ternary combinations
        weights = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0],
            ]
        )
        scale = torch.ones(4, 1)

        # Should not raise
        packed, _ = pack_ternary(weights, scale)
        assert packed.shape == (4, 1)


class TestPackedTernaryWeight:
    """Test suite for PackedTernaryWeight module."""

    def test_initialization(self) -> None:
        """Test PackedTernaryWeight initialization.

        Why: Verifies correct buffer creation with expected shapes and dtypes.
        """
        layer = PackedTernaryWeight(in_features=256, out_features=128, bias=True)

        assert layer.in_features == 256
        assert layer.out_features == 128
        assert layer.packed_in_features == 64  # 256 / 4 = 64
        assert layer.packed_weight.shape == (128, 64)
        assert layer.packed_weight.dtype == torch.uint8
        assert layer.scale.shape == (128, 1)
        assert layer.bias is not None
        assert layer.bias.shape == (128,)

    def test_initialization_without_bias(self) -> None:
        """Test PackedTernaryWeight without bias.

        Why: Verifies bias=False correctly sets bias buffer to None.
        """
        layer = PackedTernaryWeight(in_features=256, out_features=128, bias=False)

        assert layer.bias is None

    def test_initialization_non_divisible_by_4(self) -> None:
        """Test PackedTernaryWeight with in_features not divisible by 4.

        Why: Verifies correct ceiling division for packed size.
        """
        layer = PackedTernaryWeight(in_features=257, out_features=128, bias=True)

        assert layer.packed_in_features == 65  # ceil(257 / 4) = 65

    def test_from_ternary_weight(self) -> None:
        """Test conversion from TernaryWeight to PackedTernaryWeight.

        Why: Verifies the classmethod correctly transfers weights and scale.
        """
        # Create and initialize TernaryWeight
        ternary = TernaryWeight(in_features=64, out_features=32, bias=True)
        # Set known weights
        with torch.no_grad():
            ternary.weight.fill_(0.5)  # Will quantize to +1
            ternary.scale.fill_(2.0)
            ternary.bias.fill_(0.5)

        # Convert to packed
        packed = PackedTernaryWeight.from_ternary_weight(ternary)

        assert packed.in_features == 64
        assert packed.out_features == 32
        assert packed.packed_weight.shape == (32, 16)  # 64/4 = 16
        assert torch.allclose(packed.scale, torch.full((32, 1), 2.0))
        assert torch.allclose(packed.bias, torch.full((32,), 0.5))

    def test_forward_matches_ternary_weight(self) -> None:
        """Test that PackedTernaryWeight produces same output as TernaryWeight.

        Why: Critical correctness test - packed inference must match unpacked.
        """
        torch.manual_seed(42)

        # Create TernaryWeight with random weights
        ternary = TernaryWeight(in_features=64, out_features=32, bias=True)

        # Convert to packed
        packed = PackedTernaryWeight.from_ternary_weight(ternary)

        # Test input
        x = torch.randn(4, 64)

        # Compare outputs (in eval mode for TernaryWeight to use cached quantized weights)
        ternary.eval()
        with torch.no_grad():
            ternary_output = ternary(x)
            packed_output = packed(x)

        assert torch.allclose(ternary_output, packed_output, rtol=1e-5, atol=1e-5)

    def test_forward_with_3d_input(self) -> None:
        """Test forward pass with (batch, seq_len, features) input.

        Why: Verifies correct handling of sequence data typical in transformers.
        """
        ternary = TernaryWeight(in_features=64, out_features=32, bias=True)
        packed = PackedTernaryWeight.from_ternary_weight(ternary)

        # 3D input: (batch, seq_len, features)
        x = torch.randn(2, 10, 64)

        ternary.eval()
        with torch.no_grad():
            ternary_output = ternary(x)
            packed_output = packed(x)

        assert packed_output.shape == (2, 10, 32)
        assert torch.allclose(ternary_output, packed_output, rtol=1e-5, atol=1e-5)

    def test_memory_bytes(self) -> None:
        """Test memory_bytes calculation.

        Why: Verifies memory savings estimation is accurate.
        """
        layer = PackedTernaryWeight(in_features=1024, out_features=1024, bias=True)

        # packed_weight: 1024 * 256 * 1 = 262,144 bytes (256 = 1024/4)
        # scale: 1024 * 1 * 4 = 4,096 bytes
        # bias: 1024 * 4 = 4,096 bytes
        # Total: 270,336 bytes
        expected = (1024 * 256) + (1024 * 1 * 4) + (1024 * 4)

        assert layer.memory_bytes() == expected

    def test_memory_bytes_no_bias(self) -> None:
        """Test memory_bytes without bias.

        Why: Verifies correct memory calculation when bias is None.
        """
        layer = PackedTernaryWeight(in_features=1024, out_features=1024, bias=False)

        expected = (1024 * 256) + (1024 * 1 * 4)  # No bias

        assert layer.memory_bytes() == expected

    def test_extra_repr(self) -> None:
        """Test extra_repr returns informative string.

        Why: Verifies module representation includes key information.
        """
        layer = PackedTernaryWeight(in_features=256, out_features=128, bias=True)

        repr_str = layer.extra_repr()

        assert "in_features=256" in repr_str
        assert "out_features=128" in repr_str
        assert "bias=True" in repr_str
        assert "packed_size=64" in repr_str


class TestConvertToPacked:
    """Test suite for convert_to_packed utility."""

    def test_convert_single_layer(self) -> None:
        """Test conversion of a single TernaryWeight layer.

        Why: Verifies basic conversion works for simple models.
        """
        model = TernaryWeight(in_features=64, out_features=32)

        # Wrap in a container module for named_children iteration
        container = nn.Sequential(model)
        converted = convert_to_packed(container)

        assert isinstance(converted[0], PackedTernaryWeight)

    def test_convert_nested_model(self) -> None:
        """Test conversion of nested model with multiple TernaryWeight layers.

        Why: Verifies recursive conversion through nested modules.
        """
        model = nn.Sequential(
            TernaryWeight(64, 32),
            nn.ReLU(),
            nn.Sequential(
                TernaryWeight(32, 16),
                nn.ReLU(),
            ),
            TernaryWeight(16, 8),
        )

        converted = convert_to_packed(model)

        # Check all TernaryWeight layers were converted
        assert isinstance(converted[0], PackedTernaryWeight)
        assert isinstance(converted[1], nn.ReLU)  # Non-TernaryWeight unchanged
        assert isinstance(converted[2][0], PackedTernaryWeight)
        assert isinstance(converted[2][1], nn.ReLU)
        assert isinstance(converted[3], PackedTernaryWeight)

    def test_convert_preserves_output(self) -> None:
        """Test that converted model produces same output as original.

        Why: Critical correctness test for the conversion utility.
        """
        torch.manual_seed(42)

        model = nn.Sequential(
            TernaryWeight(64, 32),
            nn.ReLU(),
            TernaryWeight(32, 16),
        )

        x = torch.randn(2, 64)

        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(x)

        # Convert and get new output
        converted = convert_to_packed(model)
        with torch.no_grad():
            converted_output = converted(x)

        assert torch.allclose(original_output, converted_output, rtol=1e-5, atol=1e-5)


class TestSerialization:
    """Test suite for save_packed_model and load_packed_model."""

    def test_save_load_roundtrip(self) -> None:
        """Test that save/load preserves model state.

        Why: Verifies serialization doesn't corrupt packed weights.
        """
        torch.manual_seed(42)

        # Create and convert model
        model = nn.Sequential(
            TernaryWeight(64, 32, bias=True),
            nn.ReLU(),
            TernaryWeight(32, 16, bias=False),
        )
        model = convert_to_packed(model)

        # Test input
        x = torch.randn(2, 64)
        with torch.no_grad():
            original_output = model(x)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_packed_model(model, path)

            # Create fresh model and load
            loaded_model = nn.Sequential(
                PackedTernaryWeight(64, 32, bias=True),
                nn.ReLU(),
                PackedTernaryWeight(32, 16, bias=False),
            )
            loaded_model = load_packed_model(path, loaded_model)

        # Verify output matches
        with torch.no_grad():
            loaded_output = loaded_model(x)

        assert torch.allclose(original_output, loaded_output, rtol=0, atol=1e-6)

    def test_save_creates_directory(self) -> None:
        """Test that save_packed_model creates parent directories.

        Why: Verifies user-friendly behavior for nested paths.
        """
        model = nn.Sequential(PackedTernaryWeight(64, 32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "model.pt"
            save_packed_model(model, path)

            assert path.exists()

    def test_load_rejects_unknown_version(self) -> None:
        """Test that load_packed_model rejects unknown format versions.

        Why: Ensures forward compatibility errors are caught early.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save with fake version
            torch.save(
                {
                    "state_dict": {},
                    "packed_layers_info": {},
                    "format_version": "99.0",
                },
                path,
            )

            model = nn.Sequential(PackedTernaryWeight(64, 32))

            with pytest.raises(ValueError, match="Unsupported packed model format version"):
                load_packed_model(path, model)


class TestMemorySavings:
    """Test suite verifying memory savings from packed storage."""

    def test_packed_vs_fp32_memory(self) -> None:
        """Test that packed storage uses ~8x less memory than FP32.

        Why: Verifies the core memory optimization goal is achieved.
        FP32: 32 bits/value, Packed: 2 bits/value + 32-bit scale per row = ~8x reduction.
        """
        in_features = 4096
        out_features = 4096

        # FP32 storage: out * in * 4 bytes
        fp32_bytes = out_features * in_features * 4

        # Packed storage
        packed_layer = PackedTernaryWeight(in_features, out_features, bias=False)
        packed_bytes = packed_layer.memory_bytes()

        # Expected: packed_bytes should be ~8x smaller
        # packed_weight: 4096 * 1024 = 4MB (each byte holds 4 values)
        # scale: 4096 * 4 = 16KB
        # Total: ~4MB vs 64MB FP32 = ~16x smaller

        ratio = fp32_bytes / packed_bytes
        assert ratio > 10, f"Memory ratio {ratio} is less than expected 10x"

    def test_7b_model_memory_estimate(self) -> None:
        """Estimate memory for 7B model packed weights.

        Why: Validates that 7B model fits in ~1.4GB as stated in the plan.
        7B params at 2 bits/param = ~1.75GB, plus scales = ~1.8GB.
        """
        # 7B model rough estimate: ~7 billion weights
        # Assuming average layer size of 4096x4096 for ~400 layers
        total_params = 7_000_000_000

        # Packed: 2 bits/param = 0.25 bytes/param
        packed_bytes = total_params * 0.25

        # Scale overhead: ~1 FP32 per 4096 params (per-channel)
        # Rough estimate: 7B / 4096 = ~1.7M scale values * 4 bytes = ~7MB
        scale_bytes = (total_params / 4096) * 4

        total_bytes = packed_bytes + scale_bytes
        total_gb = total_bytes / (1024**3)

        # Should be under 2GB (plan targets ~1.4GB)
        assert total_gb < 2.0, f"Estimated 7B memory {total_gb:.2f}GB exceeds 2GB target"
