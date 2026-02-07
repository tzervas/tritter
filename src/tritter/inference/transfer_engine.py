"""Async transfer engine for progressive layer loading.

Why: Overlapping data transfer with computation is critical for hiding
PCIe latency when streaming layers. CUDA streams enable concurrent
H2D transfers while the GPU processes the current layer group.
"""

import torch
import torch.cuda

from tritter.core.config import TritterConfig


class TransferEngine:
    """Manages asynchronous host-to-device transfers for layer streaming.

    Why: Enables compute/transfer overlap by using dedicated CUDA streams
    for data movement. Combined with double buffering, this can hide most
    of the PCIe transfer latency.

    Embedding-Prediction Context: Layer weights are read-only during inference.
    The model operates in continuous embedding space, transforming embeddings
    through each layer without needing to modify weights.
    """

    def __init__(
        self,
        config: TritterConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize transfer engine.

        Args:
            config: Configuration with use_pinned_memory setting
            device: Target GPU device

        Why: Creates CUDA stream for async transfers and optionally
        allocates pinned memory pool for faster DMA transfers.
        """
        self.config = config
        self.device = device or torch.device("cuda:0")
        self.use_pinned_memory = config.use_pinned_memory

        # Create dedicated stream for H2D transfers
        # Why: Separate stream allows transfers to overlap with compute on default stream
        self._transfer_stream = torch.cuda.Stream(device=self.device)

        # Pending transfers for sync
        self._pending_transfers: list[torch.cuda.Event] = []

    def transfer_async(
        self,
        src: torch.Tensor,
        dst: torch.Tensor | None = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Asynchronously transfer tensor from CPU to GPU.

        Args:
            src: Source tensor on CPU (should be pinned for best performance)
            dst: Optional pre-allocated destination tensor on GPU
            non_blocking: If True, return immediately without waiting

        Returns:
            Tensor on GPU device

        Why: Async transfer allows computation to continue on the default
        stream while data moves on the transfer stream.
        """
        if src.device.type != "cpu":
            raise ValueError(f"Source must be on CPU, got {src.device}")

        if self.use_pinned_memory and not src.is_pinned():
            src = src.pin_memory()
        elif non_blocking and not src.is_pinned():
            # Non-blocking H2D transfers require pinned memory
            non_blocking = False

        with torch.cuda.stream(self._transfer_stream):
            if dst is not None:
                if dst.device != self.device:
                    raise ValueError(
                        f"Destination device mismatch: expected {self.device}, got {dst.device}"
                    )
                dst.copy_(src, non_blocking=non_blocking)
                result = dst
            else:
                result = src.to(self.device, non_blocking=non_blocking)

            # Record event for synchronization
            event = torch.cuda.Event()
            event.record(self._transfer_stream)
            self._pending_transfers.append(event)

        return result

    def sync(self) -> None:
        """Wait for all pending transfers to complete.

        Why: Must be called before using transferred data to ensure
        the transfer has completed. Typically called after prefetch
        and before using the prefetched layers.
        """
        for event in self._pending_transfers:
            event.synchronize()
        self._pending_transfers.clear()

        # Ensure visibility on the default stream
        self._transfer_stream.synchronize()
        torch.cuda.synchronize(self.device)

    def sync_stream(self) -> None:
        """Synchronize the transfer stream with current stream.

        Why: Ensures all transfers on the transfer stream are visible
        to operations on the default stream. Lighter weight than full sync
        when you just need to establish ordering.
        """
        torch.cuda.current_stream(self.device).wait_stream(self._transfer_stream)

    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin a CPU tensor for faster DMA transfers.

        Args:
            tensor: CPU tensor to pin

        Returns:
            Pinned version of the tensor

        Why: Pinned (page-locked) memory enables async DMA transfers
        at full PCIe bandwidth. Without pinning, CUDA must first copy
        to a pinned staging buffer, doubling transfer time.
        """
        if tensor.device.type != "cpu":
            raise ValueError(f"Can only pin CPU tensors, got {tensor.device}")

        if tensor.is_pinned():
            return tensor

        return tensor.pin_memory()

    @property
    def transfer_stream(self) -> torch.cuda.Stream:
        """Get the transfer stream for advanced use cases."""
        return self._transfer_stream

    def has_pending_transfers(self) -> bool:
        """Check if there are pending async transfers."""
        return len(self._pending_transfers) > 0


def pin_model_weights(model: torch.nn.Module) -> None:
    """Pin all model weights for faster CPU-to-GPU transfers.

    Args:
        model: Model with weights on CPU

    Why: Pre-pinning weights enables maximum PCIe bandwidth utilization
    when streaming layers. Should be called once at model load time.
    """
    for param in model.parameters():
        if param.device.type == "cpu" and not param.is_pinned():
            param.data = param.data.pin_memory()

    for buffer in model.buffers():
        if buffer.device.type == "cpu" and not buffer.is_pinned():
            buffer.data = buffer.data.pin_memory()
