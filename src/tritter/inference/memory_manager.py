"""GPU memory management for progressive layer loading.

Why: Progressive layer loading requires careful memory tracking to prevent OOM
when streaming layer groups through GPU memory. This manager ensures we never
exceed the configured memory budget and provides allocation/deallocation APIs.

IMPORTANT: Desktop environments (Windows 11, macOS, Linux with compositor)
reserve 0.5-2GB of GPU memory for display rendering. This manager automatically
detects and accounts for OS memory overhead to prevent OOM on consumer hardware.
"""

from dataclasses import dataclass

import torch

from tritter.core.config import TritterConfig
from tritter.utils.memory_utils import (
    calculate_safe_memory_budget,
    get_system_memory_info,
)


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

    IMPORTANT: Automatically detects OS/desktop memory overhead and adjusts
    budget accordingly. On Windows 11 with DWM, macOS with WindowServer, or
    Linux with Wayland/X11, the actual usable VRAM is less than total VRAM.

    Embedding-Prediction Context: Memory management is critical for models
    operating in continuous embedding space, where activations and KV-cache
    must persist while layer weights are streamed transiently.
    """

    def __init__(
        self,
        config: TritterConfig,
        device: torch.device | None = None,
        auto_detect_budget: bool = True,
    ) -> None:
        """Initialize memory manager.

        Args:
            config: Configuration with gpu_memory_budget_gb
            device: Target GPU device (defaults to cuda:0)
            auto_detect_budget: If True, adjust budget based on detected OS overhead.
                              If False, use config.gpu_memory_budget_gb as-is.

        Why: Establishes memory budget and tracking structures before
        any allocations occur. Auto-detection prevents OOM on desktop
        environments where OS reserves VRAM for display compositing.
        """
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._allocations: dict[str, MemoryAllocation] = {}
        self._allocated_bytes: int = 0

        # Determine budget based on configuration and OS detection
        if auto_detect_budget and torch.cuda.is_available():
            # Get safe budget accounting for OS overhead
            safe_budget_gb = calculate_safe_memory_budget()
            config_budget_gb = config.gpu_memory_budget_gb

            # Use the more conservative of config budget and auto-detected budget
            effective_budget_gb = min(config_budget_gb, safe_budget_gb)

            # Store system info for diagnostics
            self._system_info = get_system_memory_info()

            # Log if budget was reduced
            if effective_budget_gb < config_budget_gb:
                self._budget_adjusted = True
                self._original_budget_gb = config_budget_gb
            else:
                self._budget_adjusted = False
                self._original_budget_gb = config_budget_gb

            self.budget_bytes = int(effective_budget_gb * 1024**3)
        else:
            self.budget_bytes = int(config.gpu_memory_budget_gb * 1024**3)
            self._system_info = None
            self._budget_adjusted = False
            self._original_budget_gb = config.gpu_memory_budget_gb

    @property
    def budget_gb(self) -> float:
        """Current budget in GB."""
        return self.budget_bytes / (1024**3)

    def get_budget_info(self) -> dict[str, float | str | bool]:
        """Get detailed budget information.

        Returns:
            Dictionary with budget details including OS overhead info
        """
        info = {
            "budget_gb": self.budget_gb,
            "budget_adjusted": self._budget_adjusted,
            "original_budget_gb": self._original_budget_gb,
        }

        if self._system_info:
            info["os_reserved_gb"] = self._system_info.os_reserved_gb
            info["compositor"] = self._system_info.compositor_name
            info["platform"] = self._system_info.platform
            info["total_gpu_gb"] = self._system_info.total_gpu_memory_gb

        return info

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
