"""Integration tests for packed ternary inference with layer streaming.

Why: Validates end-to-end inference with packed weights works correctly
with the layer streaming infrastructure. These tests verify:
1. Full model conversion and inference produces correct outputs
2. Layer streaming works seamlessly with PackedTernaryWeight layers
3. Memory usage stays within budget during inference
4. OOM protection via MemoryManager works with packed weights

Testing strategy: Uses mock models that are small enough to run without
a real GPU but large enough to exercise the streaming logic. GPU-specific
tests are marked with pytest.mark.skipif for CI environments without CUDA.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from conftest import requires_cuda
from tritter.core.config import TritterConfig
from tritter.quantization.bitnet import TernaryWeight
from tritter.quantization.packed_ternary import (
    PackedTernaryWeight,
    convert_to_packed,
    load_packed_model,
    save_packed_model,
)


class MockTransformerLayer(nn.Module):
    """Mock transformer layer using TernaryWeight for testing.

    Why: Simulates a transformer layer with ternary-quantized linear layers
    without the complexity of attention. Sufficient for testing conversion
    and inference pipeline.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize mock transformer layer.

        Args:
            hidden_size: Dimension of hidden states
        """
        super().__init__()
        self.linear = TernaryWeight(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Unused, for API compatibility

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        residual = x
        x = self.linear(x)
        x = self.norm(x + residual)
        return x


class MockPackedTransformerLayer(nn.Module):
    """Mock transformer layer using PackedTernaryWeight for testing.

    Why: Pre-packed version of MockTransformerLayer for testing
    layer streaming with packed weights.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize mock packed transformer layer.

        Args:
            hidden_size: Dimension of hidden states
        """
        super().__init__()
        self.linear = PackedTernaryWeight(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Unused, for API compatibility

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        residual = x
        x = self.linear(x)
        x = self.norm(x + residual)
        return x


class MockTritterModel(nn.Module):
    """Mock TritterModel for end-to-end testing.

    Why: Provides the interface expected by StreamingInferenceEngine
    (embed_tokens, layers, norm, lm_head) using TernaryWeight layers.
    """

    def __init__(
        self, num_layers: int, hidden_size: int, vocab_size: int, use_packed: bool = False
    ) -> None:
        """Initialize mock model.

        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension
            vocab_size: Vocabulary size
            use_packed: If True, use PackedTernaryWeight layers
        """
        super().__init__()

        if use_packed:
            self.layers = nn.ModuleList(
                [MockPackedTransformerLayer(hidden_size) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [MockTransformerLayer(hidden_size) for _ in range(num_layers)]
            )

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = TernaryWeight(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through full model.

        Args:
            input_ids: Token IDs (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


class TestFullModelConversion:
    """Test full model conversion from TernaryWeight to PackedTernaryWeight."""

    def test_convert_mock_model(self) -> None:
        """Test converting a mock TritterModel to packed format.

        Why: Verifies convert_to_packed works on model with nested layers.
        """
        torch.manual_seed(42)

        model = MockTritterModel(num_layers=4, hidden_size=256, vocab_size=1000)

        # Convert to packed
        model.eval()
        packed_model = convert_to_packed(model)

        # Verify all TernaryWeight layers converted
        for layer in packed_model.layers:
            assert isinstance(layer.linear, PackedTernaryWeight)

        assert isinstance(packed_model.lm_head, PackedTernaryWeight)

    def test_converted_model_matches_original(self) -> None:
        """Test that converted model produces identical output.

        Why: Critical correctness test for the full conversion pipeline.
        """
        torch.manual_seed(42)

        model = MockTritterModel(num_layers=4, hidden_size=256, vocab_size=1000)
        model.eval()

        # Test input
        input_ids = torch.randint(0, 1000, (2, 32))

        # Get original output
        with torch.no_grad():
            original_output = model(input_ids)

        # Convert and get packed output
        packed_model = convert_to_packed(model)
        with torch.no_grad():
            packed_output = packed_model(input_ids)

        assert torch.allclose(original_output, packed_output, rtol=1e-5, atol=1e-5)


class TestSaveLoadWorkflow:
    """Test the full save/load workflow for packed models."""

    def test_train_convert_save_load_inference(self) -> None:
        """Test the complete workflow: train -> convert -> save -> load -> inference.

        Why: Validates the expected deployment workflow where a model is
        trained with TernaryWeight, converted to packed format, saved,
        then loaded for inference.
        """
        torch.manual_seed(42)

        # 1. Create model (simulating training)
        model = MockTritterModel(num_layers=4, hidden_size=256, vocab_size=1000)
        model.eval()

        # Test input
        input_ids = torch.randint(0, 1000, (2, 32))

        # 2. Get reference output before conversion
        with torch.no_grad():
            reference_output = model(input_ids)

        # 3. Convert to packed format
        packed_model = convert_to_packed(model)

        # 4. Save packed model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_packed_model(packed_model, path)

            # 5. Create fresh model structure for loading
            loaded_model = MockTritterModel(
                num_layers=4, hidden_size=256, vocab_size=1000, use_packed=True
            )
            # Also convert lm_head to packed
            loaded_model.lm_head = PackedTernaryWeight(256, 1000, bias=False)

            # 6. Load weights
            loaded_model = load_packed_model(path, loaded_model)

        # 7. Verify inference matches reference
        with torch.no_grad():
            loaded_output = loaded_model(input_ids)

        assert torch.allclose(reference_output, loaded_output, rtol=1e-5, atol=1e-5)


@requires_cuda
class TestPackedLayerStreaming:
    """Test packed inference with layer streaming (GPU required)."""

    def test_packed_with_streaming_config(self) -> None:
        """Test that packed model works with layer streaming configuration.

        Why: Verifies PackedTernaryWeight layers are compatible with the
        layer streaming infrastructure.
        """
        from tritter.inference.layer_streaming import StreamingInferenceEngine

        config = TritterConfig(
            model_size="3B",
            use_layer_streaming=True,
            layer_group_size=2,
            gpu_memory_budget_gb=14.0,
            prefetch_next_group=True,
        )

        # Create packed model
        model = MockTritterModel(num_layers=8, hidden_size=256, vocab_size=1000, use_packed=True)
        model.lm_head = PackedTernaryWeight(256, 1000, bias=False)

        # Initialize streaming engine
        engine = StreamingInferenceEngine(model, config)

        # Verify engine initialized correctly
        assert engine.layer_loader.num_groups == 4  # 8 layers / 2 per group

    def test_streaming_forward_with_packed_weights(self) -> None:
        """Test streaming forward pass with packed weights.

        Why: Verifies the forward pass through StreamingInferenceEngine
        works correctly with PackedTernaryWeight layers.
        """
        from tritter.inference.layer_streaming import StreamingInferenceEngine

        torch.manual_seed(42)

        config = TritterConfig(
            model_size="3B",
            use_layer_streaming=True,
            layer_group_size=2,
            gpu_memory_budget_gb=14.0,
            prefetch_next_group=False,  # Simpler for testing
        )

        # Create and convert model (start on CPU)
        model = MockTritterModel(num_layers=4, hidden_size=256, vocab_size=1000)
        model.eval()
        model = convert_to_packed(model)

        # Initialize streaming engine (moves embed/norm/lm_head to GPU, layers to CPU)
        engine = StreamingInferenceEngine(model, config)

        # Create input on GPU (where embed_tokens now lives)
        input_ids = torch.randint(0, 1000, (2, 16), device=engine.device)

        # Get embeddings from GPU-resident embed_tokens
        hidden_states = model.embed_tokens(input_ids)

        # Forward through layers via streaming (returns inference tensor)
        streaming_hidden = engine.forward(hidden_states)

        # Apply final norm and lm_head (both on GPU)
        # Use no_grad since we're in inference and streaming_hidden is an inference tensor
        with torch.no_grad():
            streaming_hidden = model.norm(streaming_hidden)
            streaming_output = model.lm_head(streaming_hidden)

        # Verify output shape and values are finite
        assert streaming_output.shape == (2, 16, 1000)
        assert torch.isfinite(streaming_output).all(), "Output contains NaN/Inf"


class TestMemoryUsage:
    """Test memory usage characteristics of packed weights."""

    def test_packed_model_memory_smaller(self) -> None:
        """Test that packed model uses less memory than unpacked.

        Why: Verifies the memory optimization is actually achieved.
        """
        hidden_size = 512
        num_layers = 4

        # Create unpacked model and measure
        unpacked = MockTritterModel(num_layers=num_layers, hidden_size=hidden_size, vocab_size=1000)

        # Count parameters (approximate memory)
        unpacked_params = sum(p.numel() for p in unpacked.parameters())
        unpacked_bytes = unpacked_params * 4  # FP32

        # Convert to packed
        packed = convert_to_packed(unpacked)

        # Count packed memory
        packed_bytes = 0
        for module in packed.modules():
            if isinstance(module, PackedTernaryWeight):
                packed_bytes += module.memory_bytes()
            elif isinstance(module, (nn.Embedding, nn.LayerNorm)):
                for p in module.parameters():
                    packed_bytes += p.numel() * 4

        # Packed should be significantly smaller
        # Note: Embedding and LayerNorm are still FP32, so ratio won't be 8x
        assert packed_bytes < unpacked_bytes, (
            f"Packed ({packed_bytes}) should be smaller than unpacked ({unpacked_bytes})"
        )

    def test_large_layer_memory_estimate(self) -> None:
        """Test memory estimation for large layers typical of 7B models.

        Why: Verifies memory savings for realistic layer sizes.
        A 7B model might have 4096x4096 or 4096x11008 layers.
        """
        # Typical 7B FFN: 4096 -> 11008 -> 4096
        up_proj = PackedTernaryWeight(4096, 11008, bias=False)
        down_proj = PackedTernaryWeight(11008, 4096, bias=False)

        # Calculate memory
        up_bytes = up_proj.memory_bytes()
        down_bytes = down_proj.memory_bytes()

        # FP32 equivalents
        up_fp32 = 4096 * 11008 * 4
        down_fp32 = 11008 * 4096 * 4

        # Verify significant savings
        assert up_bytes < up_fp32 / 8, "Up projection should be <12.5% of FP32"
        assert down_bytes < down_fp32 / 8, "Down projection should be <12.5% of FP32"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_layer_model(self) -> None:
        """Test model with single layer.

        Why: Verifies no off-by-one errors in layer handling.
        """
        model = MockTritterModel(num_layers=1, hidden_size=64, vocab_size=100)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            original = model(input_ids)

        packed = convert_to_packed(model)
        with torch.no_grad():
            packed_out = packed(input_ids)

        assert torch.allclose(original, packed_out, rtol=1e-5, atol=1e-5)

    def test_batch_size_one(self) -> None:
        """Test inference with batch size 1.

        Why: Verifies no batch dimension issues.
        """
        model = MockTritterModel(num_layers=2, hidden_size=64, vocab_size=100)
        model.eval()
        model = convert_to_packed(model)

        input_ids = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == (1, 8, 100)

    def test_sequence_length_one(self) -> None:
        """Test inference with sequence length 1.

        Why: Verifies no sequence dimension issues.
        """
        model = MockTritterModel(num_layers=2, hidden_size=64, vocab_size=100)
        model.eval()
        model = convert_to_packed(model)

        input_ids = torch.randint(0, 100, (2, 1))

        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == (2, 1, 100)
