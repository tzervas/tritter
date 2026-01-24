"""Hardware profiles for different GPU configurations.

Why: Provides pre-configured memory budgets and optimizations for specific GPUs,
allowing users to easily switch between hardware configurations without manual tuning.

Supported Hardware (Verified):
- RTX 5080 (16GB) - Primary development target
- RTX 3090 Ti (24GB) - Secondary development target

Supported Hardware (Planned):
- RTX 4090, RTX 4080, RTX 3080, etc.
- A100, H100, L40, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from tritter.utils.memory_utils import detect_os_reserved_memory


@dataclass(frozen=True)
class HardwareProfile:
    """Configuration profile for a specific GPU.

    Why: Encapsulates hardware-specific settings for optimal performance
    without requiring users to understand memory budgets and streaming limits.
    """

    name: str
    vram_gb: float
    max_model_inference: str  # Model size string (e.g., "13B")
    max_model_qlora: str  # Model size for QLoRA training
    max_model_full_train: str  # Model size for full training
    default_budget_gb: float  # Default memory budget
    layer_group_size: int  # Optimal layer group size
    prefetch_buffer_mb: int  # Prefetch buffer size
    verified: bool  # Whether this profile has been tested


# Verified hardware profiles (tested on actual hardware)
RTX_5080 = HardwareProfile(
    name="NVIDIA RTX 5080",
    vram_gb=16.0,
    max_model_inference="13B",
    max_model_qlora="40B",
    max_model_full_train="1B",
    default_budget_gb=14.0,
    layer_group_size=4,
    prefetch_buffer_mb=512,
    verified=True,
)

RTX_3090_TI = HardwareProfile(
    name="NVIDIA RTX 3090 Ti",
    vram_gb=24.0,
    max_model_inference="30B",
    max_model_qlora="65B",
    max_model_full_train="3B",
    default_budget_gb=22.0,
    layer_group_size=6,
    prefetch_buffer_mb=768,
    verified=True,
)

# Planned hardware profiles (based on specifications, not yet tested)
RTX_5090 = HardwareProfile(
    name="NVIDIA RTX 5090",
    vram_gb=32.0,
    max_model_inference="40B",
    max_model_qlora="70B",
    max_model_full_train="7B",
    default_budget_gb=30.0,
    layer_group_size=8,
    prefetch_buffer_mb=1024,
    verified=False,
)

RTX_4090 = HardwareProfile(
    name="NVIDIA RTX 4090",
    vram_gb=24.0,
    max_model_inference="30B",
    max_model_qlora="65B",
    max_model_full_train="3B",
    default_budget_gb=22.0,
    layer_group_size=6,
    prefetch_buffer_mb=768,
    verified=False,
)

RTX_4080 = HardwareProfile(
    name="NVIDIA RTX 4080",
    vram_gb=16.0,
    max_model_inference="13B",
    max_model_qlora="40B",
    max_model_full_train="1B",
    default_budget_gb=14.0,
    layer_group_size=4,
    prefetch_buffer_mb=512,
    verified=False,
)

RTX_3090 = HardwareProfile(
    name="NVIDIA RTX 3090",
    vram_gb=24.0,
    max_model_inference="30B",
    max_model_qlora="65B",
    max_model_full_train="3B",
    default_budget_gb=22.0,
    layer_group_size=6,
    prefetch_buffer_mb=768,
    verified=False,
)

RTX_3080 = HardwareProfile(
    name="NVIDIA RTX 3080",
    vram_gb=10.0,
    max_model_inference="7B",
    max_model_qlora="13B",
    max_model_full_train="1B",
    default_budget_gb=8.5,
    layer_group_size=2,
    prefetch_buffer_mb=256,
    verified=False,
)

A100_80G = HardwareProfile(
    name="NVIDIA A100 80GB",
    vram_gb=80.0,
    max_model_inference="70B",
    max_model_qlora="70B",
    max_model_full_train="13B",
    default_budget_gb=76.0,
    layer_group_size=16,
    prefetch_buffer_mb=2048,
    verified=False,
)

A100_40G = HardwareProfile(
    name="NVIDIA A100 40GB",
    vram_gb=40.0,
    max_model_inference="40B",
    max_model_qlora="70B",
    max_model_full_train="7B",
    default_budget_gb=38.0,
    layer_group_size=8,
    prefetch_buffer_mb=1024,
    verified=False,
)

H100 = HardwareProfile(
    name="NVIDIA H100",
    vram_gb=80.0,
    max_model_inference="70B",
    max_model_qlora="70B",
    max_model_full_train="13B",
    default_budget_gb=76.0,
    layer_group_size=16,
    prefetch_buffer_mb=4096,
    verified=False,
)

# Profile registry by GPU name patterns
_PROFILE_REGISTRY: dict[str, HardwareProfile] = {
    "5080": RTX_5080,
    "3090 ti": RTX_3090_TI,
    "3090ti": RTX_3090_TI,
    "5090": RTX_5090,
    "4090": RTX_4090,
    "4080": RTX_4080,
    "3090": RTX_3090,
    "3080": RTX_3080,
    "a100-sxm4-80gb": A100_80G,
    "a100-pcie-80gb": A100_80G,
    "a100 80gb": A100_80G,
    "a100-sxm4-40gb": A100_40G,
    "a100-pcie-40gb": A100_40G,
    "a100 40gb": A100_40G,
    "h100": H100,
}


def detect_gpu_profile() -> HardwareProfile | None:
    """Automatically detect the current GPU and return its profile.

    Returns:
        HardwareProfile for detected GPU, or None if not recognized

    Why: Enables zero-configuration setup on known hardware.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device).lower()

    for pattern, profile in _PROFILE_REGISTRY.items():
        if pattern in gpu_name:
            return profile

    return None


def get_profile(name: str) -> HardwareProfile | None:
    """Get a hardware profile by name.

    Args:
        name: GPU name pattern (e.g., "5080", "3090 Ti", "A100 80GB")

    Returns:
        HardwareProfile if found, None otherwise
    """
    return _PROFILE_REGISTRY.get(name.lower())


def list_profiles() -> list[HardwareProfile]:
    """List all available hardware profiles.

    Returns:
        List of all registered hardware profiles
    """
    # Deduplicate (some aliases point to same profile)
    seen = set()
    profiles = []
    for profile in _PROFILE_REGISTRY.values():
        if profile.name not in seen:
            seen.add(profile.name)
            profiles.append(profile)
    return sorted(profiles, key=lambda p: p.vram_gb, reverse=True)


def get_adjusted_budget(profile: HardwareProfile) -> float:
    """Get memory budget adjusted for OS overhead.

    Args:
        profile: Hardware profile to adjust

    Returns:
        Adjusted memory budget in GB

    Why: Desktop environments steal VRAM; this accounts for that.
    """
    os_reserved, _ = detect_os_reserved_memory()
    cuda_overhead = 0.3  # Typical CUDA runtime overhead

    available = profile.vram_gb - os_reserved - cuda_overhead
    # Use 90% of available with 1GB headroom
    safe_budget = available * 0.9 - 1.0

    return max(0.0, min(safe_budget, profile.default_budget_gb))


def create_config_for_profile(
    profile: HardwareProfile,
    model_size: str,
    use_layer_streaming: bool | None = None,
) -> dict[str, int | float | bool | str]:
    """Create TritterConfig kwargs optimized for a hardware profile.

    Args:
        profile: Target hardware profile
        model_size: Model size string (e.g., "7B")
        use_layer_streaming: Override auto-detection

    Returns:
        Dictionary of config kwargs

    Why: Simplifies configuration by providing sensible defaults for each GPU.

    Usage:
        profile = detect_gpu_profile() or RTX_5080
        config_kwargs = create_config_for_profile(profile, "7B")
        config = TritterConfig(**config_kwargs)
    """
    budget = get_adjusted_budget(profile)

    # Auto-detect if layer streaming is needed
    if use_layer_streaming is None:
        # Parse model size
        size_str = model_size.upper().rstrip("B")
        try:
            size_b = float(size_str)
        except ValueError:
            size_b = 7.0  # Default assumption

        # Need streaming if model is larger than what fits in memory
        max_inference_str = profile.max_model_inference.upper().rstrip("B")
        try:
            max_inference_b = float(max_inference_str)
        except ValueError:
            max_inference_b = 13.0

        use_layer_streaming = size_b > max_inference_b

    return {
        "model_size": model_size,
        "use_layer_streaming": use_layer_streaming,
        "layer_group_size": profile.layer_group_size,
        "gpu_memory_budget_gb": budget,
        "prefetch_next_group": True,
    }


def print_profile_info(profile: HardwareProfile) -> None:
    """Print detailed information about a hardware profile.

    Args:
        profile: Hardware profile to display
    """
    budget = get_adjusted_budget(profile)
    os_reserved, compositor = detect_os_reserved_memory()

    print()
    print("=" * 60)
    print(f"Hardware Profile: {profile.name}")
    print("=" * 60)
    print(f"VRAM:                    {profile.vram_gb:.0f} GB")
    print(f"Verified:                {'Yes' if profile.verified else 'No (planned)'}")
    print()
    print("Capabilities:")
    print(f"  Max Inference:         {profile.max_model_inference}")
    print(f"  Max QLoRA Training:    {profile.max_model_qlora}")
    print(f"  Max Full Training:     {profile.max_model_full_train}")
    print()
    print("Memory Budget:")
    print(f"  Default Budget:        {profile.default_budget_gb:.1f} GB")
    print(f"  OS Reserved ({compositor}): {os_reserved:.1f} GB")
    print(f"  Adjusted Budget:       {budget:.1f} GB")
    print()
    print("Optimizations:")
    print(f"  Layer Group Size:      {profile.layer_group_size}")
    print(f"  Prefetch Buffer:       {profile.prefetch_buffer_mb} MB")
    print("=" * 60)


__all__ = [
    "A100_40G",
    "A100_80G",
    "H100",
    "HardwareProfile",
    "RTX_3080",
    "RTX_3090",
    "RTX_3090_TI",
    "RTX_4080",
    "RTX_4090",
    "RTX_5080",
    "RTX_5090",
    "create_config_for_profile",
    "detect_gpu_profile",
    "get_adjusted_budget",
    "get_profile",
    "list_profiles",
    "print_profile_info",
]
