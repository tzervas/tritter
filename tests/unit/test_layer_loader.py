"""Tests for LayerLoader progressive layer streaming.

Why: LayerLoader is critical for enabling models larger than VRAM to run
on consumer hardware. These tests verify correct layer loading, eviction,
buffering, and boundary handling without requiring a full TritterModel.
"""

import pytest
import torch
import torch.nn as nn

from tritter.core.config import TritterConfig
from tritter.inference.layer_streaming import LayerGroupBuffer, LayerLoader


class MockLayer(nn.Module):
    """Simple layer for testing LayerLoader without full TritterModel.

    Why: Testing layer streaming doesn't require complex transformer layers.
    A simple linear layer is sufficient to verify device movement and buffering.
    """

    def __init__(self, hidden_size: int):
        """Initialize mock layer.

        Args:
            hidden_size: Dimension of linear layer
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear layer."""
        return self.linear(x)


class MockModel(nn.Module):
    """Mock model for testing LayerLoader.

    Why: LayerLoader needs a model with a .layers attribute (nn.ModuleList).
    This minimal mock avoids the complexity of a full TritterModel while
    providing the interface LayerLoader expects. Must inherit from nn.Module
    to support parameters() and buffers() methods used by pin_model_weights.
    """

    def __init__(self, num_layers: int, hidden_size: int):
        """Initialize mock model.

        Args:
            num_layers: Number of layers to create
            hidden_size: Hidden dimension for each layer
        """
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])


@pytest.fixture
def config() -> TritterConfig:
    """Create config for layer streaming tests.

    Why: LayerLoader needs config settings for layer_group_size,
    gpu_memory_budget_gb, use_pinned_memory, and prefetch_next_group.
    """
    return TritterConfig(
        model_size="3B",
        use_layer_streaming=True,
        layer_group_size=4,
        gpu_memory_budget_gb=14.0,
        use_pinned_memory=True,
        prefetch_next_group=True,
    )


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock model for testing.

    Why: Provides a minimal model with 12 layers for testing layer
    group loading and boundary conditions.
    """
    return MockModel(num_layers=12, hidden_size=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for layer streaming")
class TestLayerLoader:
    """Test suite for LayerLoader."""

    def test_initialization(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test LayerLoader initialization with config.

        Why: Verifies that LayerLoader correctly computes number of groups,
        initializes memory manager and transfer engine, and sets up buffers.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Verify basic properties
        assert loader.num_groups == 3  # 12 layers / 4 per group = 3 groups
        assert loader.layers_per_group == 4
        assert loader.active_buffer is None  # No layers loaded yet

        # Verify components initialized
        assert loader.memory_manager is not None
        assert loader.transfer_engine is not None

    def test_num_groups_non_divisible(self, config: TritterConfig) -> None:
        """Test num_groups calculation when layers don't divide evenly.

        Why: Ensures proper ceiling division when num_layers is not a
        multiple of layer_group_size. For example, 10 layers with group
        size 4 should create 3 groups (not 2).
        """
        model = MockModel(num_layers=10, hidden_size=256)
        loader = LayerLoader(model, config, device=torch.device("cuda:0"))

        # 10 layers / 4 per group = 2.5, rounds up to 3 groups
        assert loader.num_groups == 3

    def test_load_group(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test synchronous layer group loading to GPU.

        Why: Verifies that load_group moves the correct layers to GPU,
        returns them, and updates buffer state correctly.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Load first group (layers 0-3)
        layers = loader.load_group(0)

        assert len(layers) == 4
        assert all(layer.linear.weight.device.type == "cuda" for layer in layers)
        assert loader.active_buffer is not None
        assert loader.active_buffer.group_idx == 0
        assert loader.active_buffer.is_on_gpu is True
        assert len(loader.active_buffer.layers) == 4

    def test_load_group_last_partial(self, config: TritterConfig) -> None:
        """Test loading last group when it has fewer layers.

        Why: Ensures correct handling of remainder layers. With 10 layers
        and group size 4, the last group should have only 2 layers.
        """
        model = MockModel(num_layers=10, hidden_size=256)
        loader = LayerLoader(model, config, device=torch.device("cuda:0"))

        # Load last group (group 2: layers 8-9, only 2 layers)
        layers = loader.load_group(2)

        assert len(layers) == 2  # Last group has only 2 layers
        assert all(layer.linear.weight.device.type == "cuda" for layer in layers)

    def test_load_group_invalid_index(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test that invalid group_idx raises ValueError.

        Why: Ensures proper bounds checking and error messages for invalid
        group indices (negative or beyond num_groups).
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Test negative index
        with pytest.raises(ValueError, match="group_idx must be in"):
            loader.load_group(-1)

        # Test index beyond num_groups
        with pytest.raises(ValueError, match="group_idx must be in"):
            loader.load_group(loader.num_groups)

    def test_evict_group(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test layer group eviction from GPU.

        Why: Verifies that evict_group moves layers back to CPU and
        clears buffer state correctly.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Load then evict
        loader.load_group(0)
        loader.evict_group(0)

        # Verify layers moved back to CPU
        for i in range(4):
            assert mock_model.layers[i].linear.weight.device.type == "cpu"

        # Verify buffer cleared
        assert loader.active_buffer is None

    def test_evict_group_not_loaded(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test evicting a group that was never loaded.

        Why: Ensures evict_group handles the case where the group
        was never loaded without errors (idempotent operation).
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Should not raise error
        loader.evict_group(0)

    def test_get_layer_indices(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test get_layer_indices returns correct range.

        Why: Verifies correct calculation of start/end indices for each
        group, including handling of the last partial group.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # First group
        start, end = loader.get_layer_indices(0)
        assert (start, end) == (0, 4)

        # Middle group
        start, end = loader.get_layer_indices(1)
        assert (start, end) == (4, 8)

        # Last group
        start, end = loader.get_layer_indices(2)
        assert (start, end) == (8, 12)

    def test_get_layer_indices_partial_last_group(self, config: TritterConfig) -> None:
        """Test get_layer_indices with partial last group.

        Why: Ensures correct end index calculation when the last group
        has fewer layers than group_size.
        """
        model = MockModel(num_layers=10, hidden_size=256)
        loader = LayerLoader(model, config, device=torch.device("cuda:0"))

        # Last group should be (8, 10) not (8, 12)
        start, end = loader.get_layer_indices(2)
        assert (start, end) == (8, 10)

    def test_swap_buffers(self, mock_model: MockModel, config: TritterConfig) -> None:
        """Test buffer swapping for double buffering.

        Why: Verifies that swap_buffers correctly exchanges active and
        prefetch buffers, enabling the double buffering pattern.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Load group 0 into buffer A
        loader.load_group(0)
        buffer_a_before = loader.active_buffer

        # Manually set buffer B for testing
        loader._buffer_b = LayerGroupBuffer(
            group_idx=1,
            layers=[],
            is_on_gpu=True,
        )
        buffer_b_before = loader._buffer_b

        # Swap buffers
        loader.swap_buffers()

        # Verify buffers swapped
        assert loader.active_buffer == buffer_b_before
        assert loader._buffer_b == buffer_a_before

    def test_prefetch_async_ignores_invalid_index(
        self, mock_model: MockModel, config: TritterConfig
    ) -> None:
        """Test prefetch_async ignores invalid indices.

        Why: Ensures prefetch_async gracefully handles out-of-range indices
        (e.g., prefetch beyond last group) without errors.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Should not raise error for invalid indices
        loader.prefetch_async(-1)
        loader.prefetch_async(loader.num_groups)

    def test_prefetch_async_ignores_already_loaded(
        self, mock_model: MockModel, config: TritterConfig
    ) -> None:
        """Test prefetch_async skips already loaded groups.

        Why: Ensures prefetch_async is idempotent and doesn't redundantly
        transfer layers that are already on GPU.
        """
        loader = LayerLoader(mock_model, config, device=torch.device("cuda:0"))

        # Load group 0
        loader.load_group(0)

        # Prefetch same group should be no-op
        loader.prefetch_async(0)

        # Buffer B should remain None (no prefetch occurred)
        assert loader._buffer_b is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for layer streaming")
class TestLayerGroupBuffer:
    """Test suite for LayerGroupBuffer dataclass."""

    def test_initialization(self) -> None:
        """Test LayerGroupBuffer initialization.

        Why: Verifies that LayerGroupBuffer correctly stores group metadata
        and defaults is_on_gpu to False.
        """
        layers = [MockLayer(256), MockLayer(256)]
        buffer = LayerGroupBuffer(group_idx=0, layers=layers)

        assert buffer.group_idx == 0
        assert len(buffer.layers) == 2
        assert buffer.is_on_gpu is False

    def test_explicit_is_on_gpu(self) -> None:
        """Test LayerGroupBuffer with explicit is_on_gpu.

        Why: Verifies that is_on_gpu can be explicitly set during initialization.
        """
        layers = [MockLayer(256)]
        buffer = LayerGroupBuffer(group_idx=1, layers=layers, is_on_gpu=True)

        assert buffer.is_on_gpu is True
