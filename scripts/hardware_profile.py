#!/usr/bin/env python3
"""Hardware profiling and configuration script.

Usage:
    # Auto-detect current GPU and show info
    python scripts/hardware_profile.py

    # Show info for specific GPU
    python scripts/hardware_profile.py --gpu 5080
    python scripts/hardware_profile.py --gpu "3090 Ti"

    # List all supported GPUs
    python scripts/hardware_profile.py --list

    # Generate config for specific GPU and model
    python scripts/hardware_profile.py --gpu 5080 --model 7B

    # Check if model fits on current hardware
    python scripts/hardware_profile.py --check 7B
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hardware profiling and configuration for Tritter"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        help="GPU name (e.g., '5080', '3090 Ti', 'A100 80GB')",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model size to generate config for (e.g., '7B')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported hardware profiles",
    )
    parser.add_argument(
        "--check",
        type=str,
        help="Check if model size fits on current/specified hardware",
    )

    args = parser.parse_args()

    # Import here to ensure CUDA is initialized after arg parsing
    from tritter.utils.hardware_profiles import (
        create_config_for_profile,
        detect_gpu_profile,
        get_profile,
        list_profiles,
        print_profile_info,
    )
    from tritter.utils.memory_utils import print_memory_report

    if args.list:
        print("\n" + "=" * 60)
        print("Supported Hardware Profiles")
        print("=" * 60)
        print()
        profiles = list_profiles()
        print(f"{'GPU':<25} {'VRAM':<8} {'Inference':<12} {'QLoRA':<10} {'Verified'}")
        print("-" * 60)
        for p in profiles:
            verified = "Yes" if p.verified else "Planned"
            print(
                f"{p.name:<25} {p.vram_gb:>5.0f} GB  "
                f"{p.max_model_inference:<12} {p.max_model_qlora:<10} {verified}"
            )
        print()
        return

    # Get the target profile
    if args.gpu:
        profile = get_profile(args.gpu)
        if profile is None:
            print(f"Error: Unknown GPU '{args.gpu}'")
            print("Use --list to see supported GPUs")
            sys.exit(1)
    else:
        profile = detect_gpu_profile()
        if profile is None:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                print(f"Warning: GPU '{gpu_name}' not in profile registry")
                print("Using memory report instead...")
                print_memory_report()
                return
            else:
                print("No CUDA GPU detected")
                print("Use --gpu to specify a target profile")
                sys.exit(1)

    # Check model fit
    if args.check:
        from tritter.core.model_specs import get_model_spec

        model_size = args.check.upper()
        if not model_size.endswith("B"):
            model_size += "B"

        spec = get_model_spec(model_size)
        if spec is None:
            print(f"Error: Unknown model size '{args.check}'")
            sys.exit(1)

        print(f"\nChecking {model_size} on {profile.name}:")
        print()

        # Parse max model sizes
        max_inf = profile.max_model_inference.upper().rstrip("B")
        max_qlora = profile.max_model_qlora.upper().rstrip("B")
        max_full = profile.max_model_full_train.upper().rstrip("B")
        model_b = float(model_size.rstrip("B"))

        inf_fits = model_b <= float(max_inf)
        qlora_fits = model_b <= float(max_qlora)
        full_fits = model_b <= float(max_full)

        print(f"  Inference:      {'✅ Fits' if inf_fits else '❌ Needs layer streaming'}")
        print(f"  QLoRA Training: {'✅ Fits' if qlora_fits else '❌ Too large'}")
        print(f"  Full Training:  {'✅ Fits' if full_fits else '❌ Too large'}")
        print()

        # Memory details
        print(f"Memory requirements for {model_size}:")
        print(f"  Packed weights: {spec.packed_size_gb:.2f} GB")
        print(f"  KV-cache (128K): ~{spec.packed_size_gb * 8:.1f} GB (INT4)")
        print()
        return

    # Generate config
    if args.model:
        config_kwargs = create_config_for_profile(profile, args.model)
        print(f"\nTritterConfig for {args.model} on {profile.name}:")
        print()
        print("from tritter import TritterConfig")
        print()
        print("config = TritterConfig(")
        for key, value in config_kwargs.items():
            if isinstance(value, str):
                print(f'    {key}="{value}",')
            elif isinstance(value, bool):
                print(f"    {key}={value},")
            elif isinstance(value, float):
                print(f"    {key}={value:.1f},")
            else:
                print(f"    {key}={value},")
        print(")")
        print()
        return

    # Default: show profile info
    print_profile_info(profile)
    print()
    print_memory_report()


if __name__ == "__main__":
    main()
