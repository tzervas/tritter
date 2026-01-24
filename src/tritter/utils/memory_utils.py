"""Memory profiling and optimization utilities.

Why: Critical for managing GPU VRAM budget on consumer hardware (RTX 5080 16GB)
with desktop environments that reserve memory for display compositing.

Handles:
- OS/desktop environment memory detection (Windows 11, macOS, Linux)
- Dynamic memory budget adjustment based on available VRAM
- OOM prevention with graceful degradation
- Memory statistics for monitoring

IMPORTANT: Desktop environments (Windows DWM, macOS WindowServer, Linux compositors)
reserve 0.5-2GB of GPU memory for display rendering. This module detects the actual
available memory and adjusts budgets accordingly.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class SystemMemoryInfo:
    """System and GPU memory information.

    Why: Centralizes memory detection for consistent budget calculations.

    Attributes:
        platform: Operating system (Windows, Darwin, Linux)
        total_gpu_memory_gb: Total GPU VRAM
        available_gpu_memory_gb: Usable GPU memory after OS reservations
        os_reserved_gb: Memory reserved by OS/desktop
        is_desktop_environment: Whether running with GUI compositor
        compositor_name: Name of detected compositor (if any)
    """

    platform: Literal["Windows", "Darwin", "Linux", "Unknown"]
    total_gpu_memory_gb: float
    available_gpu_memory_gb: float
    os_reserved_gb: float
    is_desktop_environment: bool
    compositor_name: str


def detect_os_reserved_memory() -> tuple[float, str]:
    """Detect GPU memory reserved by OS/desktop environment.

    Returns:
        Tuple of (reserved_gb, compositor_name)

    Why: Desktop environments steal GPU memory for display compositing:
    - Windows 11 DWM: 0.5-1.5GB (depends on resolution/monitors)
    - macOS WindowServer: 0.5-1.0GB
    - Linux with Wayland/X11: 0.3-0.8GB

    This detection enables accurate budget calculation to prevent OOM.
    """
    system = platform.system()
    reserved_gb = 0.0
    compositor_name = "none"

    if system == "Windows":
        # Windows Desktop Window Manager (DWM) always runs
        # Reserve more for multi-monitor setups
        try:
            # Check number of monitors via PowerShell (rough estimate)
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorID).Count"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            monitor_count = int(result.stdout.strip()) if result.returncode == 0 else 1
            # ~0.5GB base + 0.3GB per additional monitor
            reserved_gb = 0.5 + (monitor_count - 1) * 0.3
            compositor_name = f"DWM ({monitor_count} monitor{'s' if monitor_count > 1 else ''})"
        except (subprocess.TimeoutExpired, ValueError, OSError):
            reserved_gb = 1.0  # Conservative default for Windows
            compositor_name = "DWM (assumed)"

    elif system == "Darwin":
        # macOS WindowServer always runs
        reserved_gb = 0.8  # macOS typically uses more for window effects
        compositor_name = "WindowServer"

    elif system == "Linux":
        # Check for common compositors
        try:
            # Check Wayland
            wayland_display = subprocess.run(
                ["printenv", "WAYLAND_DISPLAY"],
                capture_output=True,
                text=True,
            )
            if wayland_display.returncode == 0 and wayland_display.stdout.strip():
                reserved_gb = 0.5
                compositor_name = "Wayland"
            else:
                # Check X11
                display = subprocess.run(
                    ["printenv", "DISPLAY"],
                    capture_output=True,
                    text=True,
                )
                if display.returncode == 0 and display.stdout.strip():
                    reserved_gb = 0.4
                    compositor_name = "X11"
                else:
                    # Headless/server environment
                    reserved_gb = 0.1
                    compositor_name = "headless"
        except (subprocess.TimeoutExpired, OSError):
            reserved_gb = 0.5  # Conservative default
            compositor_name = "unknown"
    else:
        reserved_gb = 0.5  # Unknown OS, assume some overhead
        compositor_name = "unknown"

    return reserved_gb, compositor_name


def get_system_memory_info() -> SystemMemoryInfo:
    """Get comprehensive system memory information.

    Returns:
        SystemMemoryInfo with detected memory values

    Why: Single function to gather all memory-related information
    for budget calculations and logging.
    """
    system = platform.system()
    if system not in ("Windows", "Darwin", "Linux"):
        system = "Unknown"

    # Detect OS reserved memory
    os_reserved_gb, compositor_name = detect_os_reserved_memory()
    is_desktop = compositor_name not in ("none", "headless")

    # Get GPU memory info
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        # Account for CUDA runtime overhead (~200-300MB)
        cuda_overhead = 0.3
        available_gb = total_gb - os_reserved_gb - cuda_overhead
    else:
        total_gb = 0.0
        available_gb = 0.0

    return SystemMemoryInfo(
        platform=system,  # type: ignore[arg-type]
        total_gpu_memory_gb=total_gb,
        available_gpu_memory_gb=max(0.0, available_gb),
        os_reserved_gb=os_reserved_gb,
        is_desktop_environment=is_desktop,
        compositor_name=compositor_name,
    )


def calculate_safe_memory_budget(
    target_fraction: float = 0.9,
    min_headroom_gb: float = 1.0,
) -> float:
    """Calculate safe GPU memory budget accounting for all overheads.

    Args:
        target_fraction: Fraction of available memory to use (default 0.9)
        min_headroom_gb: Minimum headroom to reserve (default 1.0 GB)

    Returns:
        Safe memory budget in GB

    Why: Prevents OOM by accounting for:
    - OS/desktop compositor memory
    - CUDA runtime overhead
    - PyTorch caching allocator overhead
    - Memory fragmentation buffer

    Usage:
        >>> budget = calculate_safe_memory_budget()
        >>> config = TritterConfig(gpu_memory_budget_gb=budget)
    """
    info = get_system_memory_info()

    if info.total_gpu_memory_gb == 0:
        return 0.0

    # Calculate usable memory
    usable = info.available_gpu_memory_gb * target_fraction

    # Ensure minimum headroom
    if usable > min_headroom_gb:
        budget = usable - min_headroom_gb
    else:
        budget = usable * 0.5  # Very limited VRAM, be conservative

    return max(0.0, budget)


def get_memory_status() -> dict[str, float]:
    """Get current GPU memory usage status.

    Returns:
        Dictionary with memory statistics in GB:
        - total: Total GPU memory
        - allocated: Currently allocated by PyTorch
        - cached: Reserved by PyTorch allocator
        - available_budget: Safe budget for new allocations
        - os_reserved: Reserved by OS/desktop

    Why: Real-time monitoring for OOM prevention and debugging.
    """
    info = get_system_memory_info()

    if not torch.cuda.is_available():
        return {
            "total": 0.0,
            "allocated": 0.0,
            "cached": 0.0,
            "available_budget": 0.0,
            "os_reserved": info.os_reserved_gb,
        }

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    cached = torch.cuda.memory_reserved(device) / (1024**3)
    budget = calculate_safe_memory_budget()

    return {
        "total": info.total_gpu_memory_gb,
        "allocated": allocated,
        "cached": cached,
        "available_budget": budget,
        "os_reserved": info.os_reserved_gb,
    }


def check_memory_fit(
    required_gb: float,
    warn_threshold: float = 0.9,
) -> tuple[bool, str]:
    """Check if a memory requirement fits within safe budget.

    Args:
        required_gb: Memory required in GB
        warn_threshold: Fraction of budget that triggers warning

    Returns:
        Tuple of (fits, message):
        - fits: True if requirement fits safely
        - message: Status message or warning

    Why: Pre-flight check before large allocations to provide
    actionable feedback before OOM occurs.
    """
    info = get_system_memory_info()
    budget = calculate_safe_memory_budget()

    if budget == 0:
        return False, "No GPU available"

    if required_gb > budget:
        return False, (
            f"Required {required_gb:.2f}GB exceeds safe budget {budget:.2f}GB. "
            f"OS reserves {info.os_reserved_gb:.2f}GB for {info.compositor_name}. "
            f"Consider: smaller model, layer streaming, or closing desktop apps."
        )

    usage_fraction = required_gb / budget
    if usage_fraction > warn_threshold:
        return True, (
            f"Tight fit: {required_gb:.2f}GB uses {usage_fraction:.0%} of "
            f"{budget:.2f}GB budget. OOM possible with long contexts."
        )

    return True, f"OK: {required_gb:.2f}GB fits within {budget:.2f}GB budget"


def suggest_memory_reduction(
    required_gb: float,
    model_size: str,
) -> list[str]:
    """Suggest ways to reduce memory usage.

    Args:
        required_gb: Memory requirement that doesn't fit
        model_size: Model size string (e.g., "7B", "13B")

    Returns:
        List of suggestions for reducing memory usage

    Why: Actionable guidance when models don't fit, rather than
    just failing with OOM.
    """
    budget = calculate_safe_memory_budget()
    gap = required_gb - budget

    suggestions = []

    if gap > 10:
        suggestions.append(
            f"Model {model_size} is significantly larger than available "
            f"budget ({budget:.1f}GB). Consider smaller model size."
        )
    elif gap > 5:
        suggestions.append(
            f"Enable layer streaming: config.use_layer_streaming=True"
        )
        suggestions.append(
            f"Use LoRA for training instead of full fine-tuning"
        )
    elif gap > 1:
        suggestions.append(
            f"Reduce context length (current gap: {gap:.1f}GB)"
        )
        suggestions.append(
            f"Use INT4 KV-cache: reduces cache by 4x"
        )
    else:
        suggestions.append(
            f"Close GPU-intensive desktop applications"
        )
        suggestions.append(
            f"Reduce batch size to 1"
        )

    # Always suggest these
    suggestions.append(
        f"Current safe budget: {budget:.1f}GB "
        f"(after OS reservation)"
    )

    return suggestions


def print_memory_report() -> None:
    """Print detailed memory report to console.

    Why: Quick diagnostic for debugging memory issues.
    """
    info = get_system_memory_info()
    status = get_memory_status()

    print("\n" + "=" * 60)
    print("GPU MEMORY REPORT")
    print("=" * 60)
    print(f"Platform: {info.platform}")
    print(f"Desktop: {info.compositor_name}")
    print()
    print(f"Total GPU Memory:     {info.total_gpu_memory_gb:.2f} GB")
    print(f"OS Reserved:          {info.os_reserved_gb:.2f} GB")
    print(f"Currently Allocated:  {status['allocated']:.2f} GB")
    print(f"PyTorch Cached:       {status['cached']:.2f} GB")
    print(f"Safe Budget:          {status['available_budget']:.2f} GB")
    print("=" * 60)


__all__ = [
    "SystemMemoryInfo",
    "calculate_safe_memory_budget",
    "check_memory_fit",
    "detect_os_reserved_memory",
    "get_memory_status",
    "get_system_memory_info",
    "print_memory_report",
    "suggest_memory_reduction",
]
