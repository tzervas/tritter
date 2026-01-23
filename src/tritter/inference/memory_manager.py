"""GPU memory management for progressive layer loading.

Why: Progressive layer loading requires careful memory tracking to prevent OOM
when streaming layer groups through GPU memory. This manager ensures we never
exceed the configured memory budget and provides allocation/deallocation APIs.
"""

from dataclasses import dataclass

import torch

from tritter.core.config import TritterConfig


@dataclass
class MemoryAllocation:
    """Represents an allocated memory region.

    Why: Tracks individual allocations for debugging and cleanup.
    """

    name: str
    size_bytes: int
    tensor: torch.Tensor | None = None


class MemoryManager:
    """Manages GPU memory allocation for streaming inference.

    Why: Centralizes memory tracking to prevent OOM during layer streaming.
    Enforces budget limits and provides metrics for optimization.

    Embedding-Prediction Context: Memory management is critical for models
    operating in continuous embedding space, where activations and KV-cache
    must persist while layer weights are streamed transiently.
    """

    def __init__(
        self,
        config: TritterConfig,
        device: torch.device | None = None,
    ) -> None:
        """Initialize memory manager.

        Args:
            config: Configuration with gpu_memory_budget_gb
            device: Target GPU device (defaults to cuda:0)

        Why: Establishes memory budget and tracking structures before
        any allocations occur.
        """
        self.config = config
        self.device = device or torch.device("cuda:0")
        self.budget_bytes = int(config.gpu_memory_budget_gb * 1024**3)
        self._allocations: dict[str, MemoryAllocation] = {}
        self._allocated_bytes: int = 0

    def allocate(
        self,
        name: str,
        size_bytes: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Allocate GPU memory with budget enforcement.

        Args:
            name: Unique identifier for this allocation
            size_bytes: Size to allocate in bytes
            dtype: Data type for the allocated tensor

        Returns:
            Allocated tensor on GPU

        Raises:
            RuntimeError: If allocation would exceed budget

        Why: Pre-allocates memory for layer groups, preventing fragmentation
        and enabling budget enforcement before OOM occurs.
        """
        if name in self._allocations:
            raise RuntimeError(f"Allocation '{name}' already exists")

        if self._allocated_bytes + size_bytes > self.budget_bytes:
            raise RuntimeError(
                f"Allocation of {size_bytes / 1024**2:.1f} MB would exceed "
                f"budget ({self.budget_bytes / 1024**3:.1f} GB). "
                f"Currently allocated: {self._allocated_bytes / 1024**2:.1f} MB"
            )

        # Calculate number of elements based on dtype
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size_bytes // element_size

        tensor = torch.empty(num_elements, dtype=dtype, device=self.device)

        self._allocations[name] = MemoryAllocation(
            name=name,
            size_bytes=size_bytes,
            tensor=tensor,
        )
        self._allocated_bytes += size_bytes

        return tensor

    def free(self, name: str) -> None:
        """Free a named allocation.

        Args:
            name: Identifier of allocation to free

        Why: Releases memory for evicted layer groups, making room
        for the next group in the streaming pipeline.
        """
        if name not in self._allocations:
            raise KeyError(f"No allocation named '{name}'")

        alloc = self._allocations.pop(name)
        self._allocated_bytes -= alloc.size_bytes
        # Let garbage collection handle the tensor

    def get_budget(self) -> tuple[int, int, int]:
        """Get memory budget status.

        Returns:
            Tuple of (total_budget, allocated, available) in bytes

        Why: Enables monitoring and adaptive behavior based on
        available memory headroom.
        """
        available = self.budget_bytes - self._allocated_bytes
        return (self.budget_bytes, self._allocated_bytes, available)

    def get_allocation(self, name: str) -> MemoryAllocation | None:
        """Get allocation details by name.

        Args:
            name: Allocation identifier

        Returns:
            MemoryAllocation if found, None otherwise
        """
        return self._allocations.get(name)

    def clear(self) -> None:
        """Free all allocations.

        Why: Clean shutdown path to ensure no memory leaks.
        """
        self._allocations.clear()
        self._allocated_bytes = 0

    @property
    def allocated_bytes(self) -> int:
        """Total bytes currently allocated."""
        return self._allocated_bytes

    @property
    def available_bytes(self) -> int:
        """Bytes available within budget."""
        return self.budget_bytes - self._allocated_bytes
