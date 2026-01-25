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


class StreamingInferenceEngine:
    """High-level inference engine with automatic layer streaming.

    Why: Provides simple API for streaming inference without manual
    layer management. Handles buffering, prefetching, and eviction
    automatically based on configuration.

    Usage:
        engine = StreamingInferenceEngine(model, config)
        output = engine.generate(input_ids, max_new_tokens=100)

    Embedding-Prediction Context: This engine operates in continuous embedding
    space for internal computation. The generate() method handles token-to-embedding
    conversion at entry and embedding-to-token at exit.
    """

    def __init__(
        self,
        model: TritterModel,
        config: TritterConfig,
    ) -> None:
        """Initialize streaming inference engine.

        Args:
            model: TritterModel (weights will be moved to CPU if needed)
            config: Configuration with streaming settings

        Why: Prepares model for streaming by moving weights to CPU and
        initializing the LayerLoader for on-demand GPU loading.
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Move model to CPU for streaming (layers will be loaded on-demand)
        if config.use_layer_streaming:
            model.to("cpu")
            # Keep embedding and output layers on GPU (they're small and always needed)
            model.embed_tokens.to(self.device)
            model.norm.to(self.device)
            model.lm_head.to(self.device)

        # Initialize layer loader
        self.layer_loader = LayerLoader(model, config, self.device)

        # KV-cache placeholder (for future KV-cache implementation)
        self._kv_cache = None

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with automatic layer streaming.

        Args:
            hidden_states: Input embeddings (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask for padding/prefix masking

        Returns:
            Output hidden states after all layers

        Why: Automatically manages layer loading and eviction during forward
        pass. Uses double buffering to overlap compute with transfers when
        prefetch is enabled.
        """
        hidden_states = hidden_states.to(self.device)

        if not self.config.use_layer_streaming:
            # Standard forward pass without streaming
            # Ensure layers are on the correct device
            for layer in self.model.layers:
                if layer.linear.weight.device != hidden_states.device:
                    layer.to(hidden_states.device)
                hidden_states = layer(hidden_states, attention_mask)
            return hidden_states

        # Streaming forward pass with layer groups
        for group_idx in range(self.layer_loader.num_groups):
            # Prefetch next group while processing current (if enabled)
            if self.config.prefetch_next_group and group_idx < self.layer_loader.num_groups - 1:
                self.layer_loader.prefetch_async(group_idx + 1)

            # Load current group if not already loaded
            self.layer_loader.load_group(group_idx)

            # Process through all layers in this group
            start_idx, end_idx = self.layer_loader.get_layer_indices(group_idx)
            for i in range(start_idx, end_idx):
                hidden_states = self.model.layers[i](hidden_states, attention_mask)

            # Sync prefetch before evicting current group
            if self.config.prefetch_next_group and group_idx < self.layer_loader.num_groups - 1:
                self.layer_loader.sync()

            # Evict current group to free memory for next
            self.layer_loader.evict_group(group_idx)

        return hidden_states

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate tokens with streaming inference.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top K tokens
            top_p: If set, sample from tokens with cumulative probability <= p

        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)

        Why: Full generation loop with streaming inference. Each forward
        pass through the model uses progressive layer loading.
        """
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        # Generate tokens autoregressively
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get embeddings for current sequence
            hidden_states = self.model.embed_tokens(generated_ids)

            # Forward pass through transformer layers (with streaming)
            hidden_states = self.forward(hidden_states)

            # Apply final layer norm
            hidden_states = self.model.norm(hidden_states)

            # Get logits for last position only
            logits = self.model.lm_head(hidden_states[:, -1, :])  # (batch, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # TODO: Add EOS token detection for early stopping

        return generated_ids

    def __repr__(self) -> str:
        return (
            f"StreamingInferenceEngine("
            f"num_groups={self.layer_loader.num_groups}, "
            f"layers_per_group={self.layer_loader.layers_per_group}, "
            f"prefetch={self.config.prefetch_next_group})"
        )
