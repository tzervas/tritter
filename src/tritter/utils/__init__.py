"""Utility functions and helpers for device management and memory optimization.

Why:
    Tritter targets RTX 5080 16GB VRAM with secondary support for RTX 3090 Ti.
    These utilities help optimize for specific hardware targets and track memory
    usage to ensure the model fits within the VRAM budget.

Components:
    - device_utils: Device detection and RTX 5080 optimizations
    - memory_utils: OS-aware memory tracking and budget management
    - hardware_profiles: Pre-configured settings for different GPUs
"""

# Import submodules for backward compatibility
from tritter.utils import device_utils, hardware_profiles, memory_utils

# Import commonly used functions for convenience
from tritter.utils.device_utils import (
    get_device_memory_info,
    get_optimal_device,
    optimize_for_rtx5080,
)
from tritter.utils.hardware_profiles import (
    HardwareProfile,
    RTX_3090_TI,
    RTX_5080,
    create_config_for_profile,
    detect_gpu_profile,
    get_adjusted_budget,
    list_profiles,
    print_profile_info,
)
from tritter.utils.memory_utils import (
    SystemMemoryInfo,
    calculate_safe_memory_budget,
    check_memory_fit,
    get_memory_status,
    get_system_memory_info,
    print_memory_report,
)

__all__ = [
    # Submodules
    "device_utils",
    "hardware_profiles",
    "memory_utils",
    # Device utilities
    "get_optimal_device",
    "get_device_memory_info",
    "optimize_for_rtx5080",
    # Hardware profiles
    "HardwareProfile",
    "RTX_5080",
    "RTX_3090_TI",
    "detect_gpu_profile",
    "list_profiles",
    "create_config_for_profile",
    "get_adjusted_budget",
    "print_profile_info",
    # Memory utilities
    "SystemMemoryInfo",
    "get_system_memory_info",
    "get_memory_status",
    "calculate_safe_memory_budget",
    "check_memory_fit",
    "print_memory_report",
]
