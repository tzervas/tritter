"""Utility functions and helpers for device management and memory optimization.

Why:
    Tritter targets RTX 5080 16GB VRAM. These utilities help optimize for that
    hardware target and track memory usage to ensure the model fits within
    the VRAM budget (7B BitNet ~1.4GB weights + ~8-10GB KV-cache + ~2-3GB overhead).

Components:
    - device_utils: Device detection and RTX 5080 optimizations
    - memory_utils: Memory tracking and budget management (stub)
"""

# Import submodules for backward compatibility (allows `from tritter.utils import device_utils`)
from tritter.utils import device_utils, memory_utils

# Import commonly used functions for convenience
from tritter.utils.device_utils import (
    get_device_memory_info,
    get_optimal_device,
    optimize_for_rtx5080,
)

__all__ = [
    # Submodules (backward compatibility)
    "device_utils",
    "memory_utils",
    # Functions (convenience)
    "get_optimal_device",
    "get_device_memory_info",
    "optimize_for_rtx5080",
]
