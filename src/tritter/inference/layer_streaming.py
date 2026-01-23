"""Progressive layer streaming for unbounded model size inference.

Why: Enables running models larger than GPU VRAM by streaming layer groups
through memory. Uses double buffering and async transfers to minimize
latency impact from layer streaming.

Reference: SPEC-006-progressive-layer-loading.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig
    from tritter.models.architecture import TritterModel

from tritter.inference.memory_manager import MemoryManager
from tritter.inference.transfer_engine import TransferEngine, pin_model_weights


@dataclass
class LayerGroupBuffer:
    """Buffer holding a layer group for double-buffering pattern.

    Why: Double buffering allows loading next group while processing current.
    """

    group_idx: int
    layers: list[nn.Module]
    is_on_gpu: bool = False


class LayerLoader:
    """Manages loading and eviction of layer groups for streaming inference.

    Why: Centralizes layer memory management to enable models larger than
    VRAM. Uses double buffering and async transfers to minimize latency
    impact from layer streaming.

    Embedding-Prediction Context: Layer weights are read-only during inference.
    The model operates in continuous embedding space, transforming embeddings
    through each layer without needing to modify weights.
    """

    def __init__(
        self,
        model: TritterModel,
        config: TritterConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize layer loader.

        Args:
            model: TritterModel with layers on CPU
            config: Configuration with streaming settings
            device: Target GPU device

        Why: Prepares infrastructure for streaming without immediately
        loading layers. Layers remain on CPU until explicitly loaded.
        Optionally pins memory for faster transfers.
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda:0")

        # Core components
        self.memory_manager = MemoryManager(config, self.device)
        self.transfer_engine = TransferEngine(config, self.device)

        # Calculate layer grouping
        self._total_layers = len(model.layers)
        self._group_size = config.layer_group_size
        self._num_groups = (self._total_layers + self._group_size - 1) // self._group_size

        # Double buffering: two slots for current and prefetch
        self._buffer_a: LayerGroupBuffer | None = None
        self._buffer_b: LayerGroupBuffer | None = None

        # Track which groups are loaded
        self._loaded_groups: set[int] = set()

        # Pin model weights for faster transfers if enabled
        if config.use_pinned_memory:
            pin_model_weights(model)

    def load_group(self, group_idx: int) -> list[nn.Module]:
        """Load a layer group to GPU synchronously.

        Args:
            group_idx: Index of layer group to load (0-indexed)

        Returns:
            List of layers now resident on GPU

        Raises:
            ValueError: If group_idx is out of range

        Why: Synchronous loading for simple use cases or initial load.
        For pipelined inference, use prefetch_async() instead.
        """
        if group_idx < 0 or group_idx >= self._num_groups:
            raise ValueError(f"group_idx must be in [0, {self._num_groups}), got {group_idx}")

        # Get layer indices for this group
        start_idx = group_idx * self._group_size
        end_idx = min(start_idx + self._group_size, self._total_layers)

        # Move layers to GPU
        layers = []
        for i in range(start_idx, end_idx):
            layer = self.model.layers[i]
            layer.to(self.device)
            layers.append(layer)

        self._loaded_groups.add(group_idx)

        # Store in buffer A (active buffer)
        self._buffer_a = LayerGroupBuffer(
            group_idx=group_idx,
            layers=layers,
            is_on_gpu=True,
        )

        return layers

    def evict_group(self, group_idx: int) -> None:
        """Remove layer group from GPU memory.

        Args:
            group_idx: Index of layer group to evict

        Why: Frees GPU memory for next layer group. Must be called
        after processing to prevent OOM with large models.
        """
        if group_idx not in self._loaded_groups:
            return  # Already evicted or never loaded

        # Get layer indices for this group
        start_idx = group_idx * self._group_size
        end_idx = min(start_idx + self._group_size, self._total_layers)

        # Move layers back to CPU
        for i in range(start_idx, end_idx):
            self.model.layers[i].to("cpu")

        self._loaded_groups.discard(group_idx)

        # Clear buffers if they held this group
        if self._buffer_a and self._buffer_a.group_idx == group_idx:
            self._buffer_a = None
        if self._buffer_b and self._buffer_b.group_idx == group_idx:
            self._buffer_b = None

    def prefetch_async(self, group_idx: int) -> None:
        """Start async prefetch of layer group.

        Args:
            group_idx: Index of layer group to prefetch

        Why: Enables compute/transfer overlap. Call while processing
        current group to hide transfer latency.
        """
        if group_idx < 0 or group_idx >= self._num_groups:
            return  # Ignore invalid indices (e.g., prefetch beyond last group)

        if group_idx in self._loaded_groups:
            return  # Already loaded

        # Get layer indices for this group
        start_idx = group_idx * self._group_size
        end_idx = min(start_idx + self._group_size, self._total_layers)

        # Async transfer to GPU
        layers = []
        for i in range(start_idx, end_idx):
            layer = self.model.layers[i]
            # Use non-blocking transfer
            for param in layer.parameters():
                if param.device.type == "cpu":
                    self.transfer_engine.transfer_async(param.data)
            for buffer in layer.buffers():
                if buffer.device.type == "cpu":
                    self.transfer_engine.transfer_async(buffer.data)
            layers.append(layer)

        # Store in buffer B (prefetch buffer)
        self._buffer_b = LayerGroupBuffer(
            group_idx=group_idx,
            layers=layers,
            is_on_gpu=False,  # Will be True after sync()
        )

    def sync(self) -> None:
        """Wait for pending async operations to complete.

        Why: Ensures prefetched layers are ready before use.
        Must be called before accessing prefetched layers.
        """
        self.transfer_engine.sync()

        # Mark prefetch buffer as on GPU
        if self._buffer_b:
            # Actually move layers to GPU now that transfers are complete
            for layer in self._buffer_b.layers:
                layer.to(self.device, non_blocking=False)
            self._buffer_b.is_on_gpu = True
            self._loaded_groups.add(self._buffer_b.group_idx)

    def swap_buffers(self) -> None:
        """Swap active and prefetch buffers.

        Why: After processing current group and syncing prefetch,
        swap so prefetched becomes active.
        """
        self._buffer_a, self._buffer_b = self._buffer_b, self._buffer_a

    def get_layer_indices(self, group_idx: int) -> tuple[int, int]:
        """Get start and end layer indices for a group.

        Args:
            group_idx: Group index

        Returns:
            Tuple of (start_idx, end_idx) for slicing model.layers
        """
        start_idx = group_idx * self._group_size
        end_idx = min(start_idx + self._group_size, self._total_layers)
        return start_idx, end_idx

    @property
    def num_groups(self) -> int:
        """Total number of layer groups."""
        return self._num_groups

    @property
    def layers_per_group(self) -> int:
        """Number of layers in each group (except possibly last)."""
        return self._group_size

    @property
    def active_buffer(self) -> LayerGroupBuffer | None:
        """Currently active layer group buffer."""
        return self._buffer_a
