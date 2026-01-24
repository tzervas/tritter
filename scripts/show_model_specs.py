#!/usr/bin/env python3
"""Display model specifications and memory estimates for all sizes.

USAGE:
    # Show summary of all models
    python scripts/show_model_specs.py

    # Show details for specific model
    python scripts/show_model_specs.py --model 7B

    # Show hardware recommendations
    python scripts/show_model_specs.py --model 7B --recommend --vram 16 --gpus 1

    # Show training requirements
    python scripts/show_model_specs.py --model 7B --training

Why: Provides quick reference for model specifications, memory requirements,
and hardware recommendations without digging through code.
"""

import argparse
import sys

# Add src to path for imports
sys.path.insert(0, str(__file__).replace("/scripts/show_model_specs.py", "/src"))

from tritter.core.model_specs import (
    MODEL_SPECS,
    estimate_memory,
    get_model_spec,
    print_model_summary,
    recommend_hardware,
)


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def print_all_models_summary() -> None:
    """Print summary table of all models."""
    print("\n" + "=" * 100)
    print("TRITTER MODEL SPECIFICATIONS")
    print("=" * 100)
    print()
    print(
        f"{'Size':<6} {'Params':>8} {'Hidden':>7} {'Layers':>7} {'Heads':>6} "
        f"{'KV Heads':>8} {'FFN':>7} {'GQA':>4} {'Packed':>10}"
    )
    print("-" * 100)

    for size, spec in MODEL_SPECS.items():
        mem = estimate_memory(spec)
        gqa = "Yes" if spec.uses_gqa else "No"
        kv_heads = str(spec.effective_num_kv_heads) if spec.uses_gqa else "-"
        packed = format_bytes(mem.weights_packed_ternary)

        print(
            f"{size:<6} {spec.total_params_billions():>7.2f}B {spec.hidden_size:>7} "
            f"{spec.num_layers:>7} {spec.num_heads:>6} {kv_heads:>8} "
            f"{spec.intermediate_size:>7} {gqa:>4} {packed:>10}"
        )

    print("-" * 100)
    print()
    print("Notes:")
    print("  - Packed: 2-bit ternary weight storage (BitNet 1.58)")
    print("  - GQA: Grouped Query Attention (reduces KV-cache memory)")
    print("  - All models support 128K context with sliding window attention")
    print()


def print_model_details(size: str) -> None:
    """Print detailed information for a specific model."""
    try:
        # Validate size exists before printing
        get_model_spec(size)  # type: ignore[arg-type]
    except KeyError:
        print(f"Error: Unknown model size '{size}'")
        print(f"Available sizes: {list(MODEL_SPECS.keys())}")
        return

    print_model_summary(size)  # type: ignore[arg-type]


def print_hardware_recommendation(
    size: str,
    vram_gb: float,
    num_gpus: int,
    for_training: bool,
) -> None:
    """Print hardware recommendations for a model."""
    try:
        rec = recommend_hardware(
            size,  # type: ignore[arg-type]
            target_vram_gb=vram_gb,
            target_gpus=num_gpus,
            for_training=for_training,
        )
    except KeyError:
        print(f"Error: Unknown model size '{size}'")
        return

    mode = "Training" if for_training else "Inference"
    print(f"\n{'=' * 60}")
    print(f"Hardware Recommendations: {size} ({mode})")
    print(f"{'=' * 60}")
    print(f"Target: {vram_gb:.0f}GB VRAM x {num_gpus} GPU(s)")
    print()

    print("Requirements:")
    print(f"  Min VRAM (inference):  {rec.min_vram_inference_gb:.1f} GB")
    print(f"  Min VRAM (training):   {rec.min_vram_training_gb:.1f} GB")
    print(f"  Min GPUs (inference):  {rec.min_gpus_inference}")
    print(f"  Min GPUs (training):   {rec.min_gpus_training}")
    print()

    print("Configuration:")
    print(f"  Layer streaming:       {'Yes' if rec.use_layer_streaming else 'No'}")
    if rec.use_layer_streaming:
        print(f"  Layer group size:      {rec.recommended_layer_group_size}")
    print(f"  Max context length:    {rec.recommended_context_length:,}")
    print(f"  Use GQA:               {'Yes' if rec.use_gqa else 'No'}")
    print(f"  INT4 KV-cache:         {'Yes' if rec.use_int4_kv_cache else 'No'}")
    print()

    if num_gpus > 1:
        print("Multi-GPU:")
        print(f"  Tensor parallel:       {rec.tensor_parallel_size}")
        print(f"  Pipeline parallel:     {rec.pipeline_parallel_size}")
        print()

    if rec.notes:
        print("Notes:")
        for note in rec.notes:
            print(f"  - {note}")
    print()


def print_memory_table() -> None:
    """Print memory requirements table for all models."""
    print("\n" + "=" * 120)
    print("MEMORY REQUIREMENTS BY MODEL SIZE")
    print("=" * 120)
    print()
    print(
        f"{'Size':<6} {'FP32':>10} {'FP16':>10} {'INT8':>10} {'Packed':>10} "
        f"{'Train BF16':>12} {'KV 32K':>10} {'KV 128K':>10}"
    )
    print("-" * 120)

    for size, spec in MODEL_SPECS.items():
        mem = estimate_memory(spec)
        print(
            f"{size:<6} "
            f"{format_bytes(mem.weights_fp32):>10} "
            f"{format_bytes(mem.weights_fp16):>10} "
            f"{format_bytes(mem.weights_int8):>10} "
            f"{format_bytes(mem.weights_packed_ternary):>10} "
            f"{format_bytes(mem.training_bf16_mixed):>12} "
            f"{format_bytes(mem.kv_cache_32k_int4):>10} "
            f"{format_bytes(mem.kv_cache_128k_int4):>10}"
        )

    print("-" * 120)
    print()
    print("Notes:")
    print("  - FP32/FP16/INT8: Standard weight formats")
    print("  - Packed: BitNet 1.58 (2-bit ternary encoding)")
    print("  - Train BF16: Mixed precision training (BF16 weights + FP32 optimizer)")
    print("  - KV: Key-Value cache for attention (INT4 format, batch=1)")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display Tritter model specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Show details for specific model size (e.g., 7B, 13B, 70B)",
    )
    parser.add_argument(
        "--recommend",
        "-r",
        action="store_true",
        help="Show hardware recommendations (requires --model)",
    )
    parser.add_argument(
        "--vram",
        type=float,
        default=16.0,
        help="Target VRAM per GPU in GB (default: 16)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Show training requirements (with --recommend)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Show memory requirements table for all models",
    )

    args = parser.parse_args()

    if args.memory:
        print_memory_table()
    elif args.model and args.recommend:
        print_hardware_recommendation(
            args.model,
            args.vram,
            args.gpus,
            args.training,
        )
    elif args.model:
        print_model_details(args.model)
    else:
        print_all_models_summary()


if __name__ == "__main__":
    main()
