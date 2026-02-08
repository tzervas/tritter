#!/usr/bin/env python3
"""RTX 5080 16GB feasibility analysis for all model sizes.

Determines which models can run on RTX 5080 with various optimization techniques:
- Packed ternary inference (2-bit weights)
- Layer streaming (progressive layer loading)
- INT4 KV-cache
- Grouped Query Attention (GQA)
- Gradient checkpointing (training)

USAGE:
    # Show feasibility matrix for all models
    python scripts/rtx5080_feasibility.py

    # Show detailed analysis for specific model
    python scripts/rtx5080_feasibility.py --model 7B

    # Show training feasibility
    python scripts/rtx5080_feasibility.py --training

    # Generate test plan
    python scripts/rtx5080_feasibility.py --test-plan

Why: Provides actionable guidance on what can actually be tested on RTX 5080,
prioritizing smaller models that fit easily before attempting larger ones
with streaming/other techniques.
"""

import argparse
import sys
from dataclasses import dataclass

sys.path.insert(0, str(__file__).replace("/scripts/rtx5080_feasibility.py", "/src"))

from tritter.core.model_specs import (
    MODEL_SPECS,
    estimate_memory,
    get_model_spec,
)
from tritter.training.lora import LoRAConfig, estimate_lora_memory

# RTX 5080 specifications
RTX5080_VRAM_GB = 16.0
RTX5080_VRAM_BYTES = int(RTX5080_VRAM_GB * 1024**3)
CUDA_OVERHEAD_GB = 0.5  # Reserved for CUDA runtime
USABLE_VRAM_GB = RTX5080_VRAM_GB - CUDA_OVERHEAD_GB


@dataclass
class FeasibilityResult:
    """Result of feasibility analysis for a model configuration."""

    model_size: str
    fits: bool
    technique: str
    context_length: int
    memory_used_gb: float
    headroom_gb: float
    notes: list[str]


def format_gb(value: float) -> str:
    """Format GB value."""
    return f"{value:.2f} GB"


def analyze_inference_feasibility(
    model_size: str,
    use_streaming: bool = False,
    use_int4_kv: bool = True,
) -> list[FeasibilityResult]:
    """Analyze inference feasibility for a model on RTX 5080.

    Returns results for different context lengths.
    """
    spec = get_model_spec(model_size)  # type: ignore[arg-type]
    mem = estimate_memory(spec)

    results = []
    context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

    for ctx_len in context_lengths:
        notes = []

        # Calculate memory components
        if use_streaming:
            # With streaming: only 1-2 layer groups in memory at once
            # Estimate ~10-15% of weights in memory at any time
            weights_gb = mem.weights_packed_ternary / (1024**3) * 0.15
            notes.append("Layer streaming: ~15% weights resident")
        else:
            weights_gb = mem.weights_packed_ternary / (1024**3)

        # KV-cache scales with context length
        if use_int4_kv:
            # INT4: 0.5 bytes per value
            kv_scale = ctx_len / 32768  # Relative to 32K baseline
            kv_gb = mem.kv_cache_32k_int4 / (1024**3) * kv_scale
            notes.append("INT4 KV-cache")
        else:
            kv_scale = ctx_len / 32768
            kv_gb = mem.kv_cache_32k_fp16 / (1024**3) * kv_scale

        # Activation memory (scales with batch and context)
        activation_gb = 0.5 + (ctx_len / 32768) * 0.5  # Rough estimate

        total_gb = weights_gb + kv_gb + activation_gb + CUDA_OVERHEAD_GB
        headroom = USABLE_VRAM_GB - total_gb
        fits = headroom >= 0

        technique = "packed"
        if use_streaming:
            technique += "+streaming"
        if use_int4_kv:
            technique += "+int4kv"

        if spec.uses_gqa:
            notes.append(f"GQA {spec.num_heads}:{spec.effective_num_kv_heads}")

        results.append(
            FeasibilityResult(
                model_size=model_size,
                fits=fits,
                technique=technique,
                context_length=ctx_len,
                memory_used_gb=total_gb,
                headroom_gb=headroom,
                notes=notes,
            )
        )

    return results


def analyze_training_feasibility(model_size: str) -> FeasibilityResult:
    """Analyze training feasibility for a model on RTX 5080.

    Training requires:
    - Full FP32/BF16 weights (for STE gradient flow)
    - Gradients
    - Optimizer states (2x weights for AdamW)
    - Activations (can reduce with gradient checkpointing)
    """
    spec = get_model_spec(model_size)  # type: ignore[arg-type]
    mem = estimate_memory(spec)

    notes = []

    # Training memory with gradient checkpointing
    # BF16 mixed precision: BF16 weights + FP32 optimizer states
    weights_bf16_gb = mem.weights_fp16 / (1024**3)
    optimizer_gb = mem.weights_fp32 / (1024**3) * 2  # AdamW m and v

    # With gradient checkpointing, activation memory is greatly reduced
    # Roughly: sqrt(num_layers) * activation_per_layer
    # Estimate ~1-2GB for small models, ~3-4GB for larger
    if spec.num_layers <= 26:
        activation_gb = 1.5
    elif spec.num_layers <= 40:
        activation_gb = 2.5
    else:
        activation_gb = 4.0

    notes.append("BF16 mixed precision")
    notes.append("Gradient checkpointing")

    # Short context for training (e.g., 2K-4K)
    context_length = 2048
    kv_scale = context_length / 32768
    kv_gb = mem.kv_cache_32k_int4 / (1024**3) * kv_scale

    total_gb = weights_bf16_gb + optimizer_gb + activation_gb + kv_gb + CUDA_OVERHEAD_GB
    headroom = USABLE_VRAM_GB - total_gb
    fits = headroom >= 0

    # If doesn't fit, try even shorter context
    if not fits:
        context_length = 512
        kv_scale = context_length / 32768
        kv_gb = mem.kv_cache_32k_int4 / (1024**3) * kv_scale
        total_gb = weights_bf16_gb + optimizer_gb + activation_gb + kv_gb + CUDA_OVERHEAD_GB
        headroom = USABLE_VRAM_GB - total_gb
        fits = headroom >= 0
        if fits:
            notes.append(f"Reduced context: {context_length}")

    return FeasibilityResult(
        model_size=model_size,
        fits=fits,
        technique="bf16+checkpointing",
        context_length=context_length,
        memory_used_gb=total_gb,
        headroom_gb=headroom,
        notes=notes,
    )


def analyze_lora_training_feasibility(
    model_size: str,
    lora_rank: int = 16,
    target_attention: bool = True,
    target_mlp: bool = False,
) -> FeasibilityResult:
    """Analyze LoRA/QLoRA training feasibility for a model on RTX 5080.

    LoRA training requires:
    - Packed ternary base weights (frozen, no gradients)
    - LoRA adapters (A and B matrices) in FP16
    - LoRA gradients
    - LoRA optimizer states (2x LoRA params for AdamW)
    - Activations (with gradient checkpointing)
    - KV-cache for forward pass

    This is MUCH more memory efficient than full training because:
    - No gradients for 99%+ of parameters
    - No optimizer states for base weights
    """
    from tritter.core.config import TritterConfig

    spec = get_model_spec(model_size)  # type: ignore[arg-type]
    mem = estimate_memory(spec)

    notes = []

    # Create config for memory estimation
    model_config = TritterConfig(model_size=model_size)

    # Build target modules list
    target_modules = []
    if target_attention:
        target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if target_mlp:
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])

    lora_config = LoRAConfig(rank=lora_rank, target_modules=target_modules)
    lora_estimates = estimate_lora_memory(model_config, lora_config)

    # Memory components for QLoRA training:
    # 1. Base weights in packed ternary (frozen)
    weights_gb = mem.weights_packed_ternary / (1024**3)

    # 2. LoRA parameters, gradients, and optimizer
    lora_total_gb = lora_estimates["total_lora_gb"]

    # 3. Activations with gradient checkpointing (reduced)
    if spec.num_layers <= 26:
        activation_gb = 1.0
    elif spec.num_layers <= 40:
        activation_gb = 1.5
    else:
        activation_gb = 2.5

    # 4. KV-cache for short training context (2K-4K)
    context_length = 2048
    kv_scale = context_length / 32768
    kv_gb = mem.kv_cache_32k_int4 / (1024**3) * kv_scale

    notes.append(f"LoRA r={lora_rank}")
    if target_attention and target_mlp:
        notes.append("Attn+MLP")
    elif target_attention:
        notes.append("Attn only")
    notes.append("Grad ckpt")

    total_gb = weights_gb + lora_total_gb + activation_gb + kv_gb + CUDA_OVERHEAD_GB
    headroom = USABLE_VRAM_GB - total_gb
    fits = headroom >= 0

    # If doesn't fit with 2K, try with smaller context
    if not fits:
        context_length = 512
        kv_scale = context_length / 32768
        kv_gb = mem.kv_cache_32k_int4 / (1024**3) * kv_scale
        total_gb = weights_gb + lora_total_gb + activation_gb + kv_gb + CUDA_OVERHEAD_GB
        headroom = USABLE_VRAM_GB - total_gb
        fits = headroom >= 0
        if fits:
            notes.append(f"ctx={context_length}")

    return FeasibilityResult(
        model_size=model_size,
        fits=fits,
        technique=f"qlora-r{lora_rank}",
        context_length=context_length,
        memory_used_gb=total_gb,
        headroom_gb=headroom,
        notes=notes,
    )


def print_feasibility_matrix() -> None:
    """Print feasibility matrix for all models."""
    print("\n" + "=" * 100)
    print("RTX 5080 16GB FEASIBILITY MATRIX")
    print("=" * 100)
    print()

    # Inference feasibility
    print("INFERENCE (Packed Ternary + INT4 KV-cache)")
    print("-" * 100)
    print(
        f"{'Model':<6} {'4K':>8} {'8K':>8} {'16K':>8} {'32K':>8} {'64K':>8} {'128K':>8} {'Streaming?':<12}"
    )
    print("-" * 100)

    for size in MODEL_SPECS.keys():
        results = analyze_inference_feasibility(size, use_streaming=False)
        streaming_results = analyze_inference_feasibility(size, use_streaming=True)

        row = f"{size:<6}"
        needs_streaming = False

        for r in results:
            if r.fits:
                row += f" {'✓':>7}"
            else:
                # Check if streaming helps
                streaming_r = next(
                    sr for sr in streaming_results if sr.context_length == r.context_length
                )
                if streaming_r.fits:
                    row += f" {'~':>7}"  # Fits with streaming
                    needs_streaming = True
                else:
                    row += f" {'✗':>7}"

        streaming_note = (
            "Required" if needs_streaming else "Optional" if size in ["7B", "10B", "13B"] else "No"
        )
        row += f" {streaming_note:<12}"
        print(row)

    print("-" * 100)
    print("✓ = Fits    ~ = Needs streaming    ✗ = Doesn't fit")
    print()

    # Training feasibility
    print("\nTRAINING (BF16 Mixed Precision + Gradient Checkpointing)")
    print("-" * 70)
    print(f"{'Model':<6} {'Fits':>6} {'Memory':>10} {'Context':>8} {'Notes':<30}")
    print("-" * 70)

    for size in MODEL_SPECS.keys():
        r = analyze_training_feasibility(size)
        fits_str = "✓" if r.fits else "✗"
        ctx_str = f"{r.context_length}"
        notes_str = ", ".join(r.notes[:2]) if r.notes else ""
        print(
            f"{size:<6} {fits_str:>6} {format_gb(r.memory_used_gb):>10} {ctx_str:>8} {notes_str:<30}"
        )

    print("-" * 70)
    print()

    # LoRA/QLoRA training feasibility
    print("\nLoRA/QLoRA TRAINING (Packed Base + LoRA Adapters)")
    print("-" * 90)
    print(
        f"{'Model':<6} {'r=8':>8} {'r=16':>8} {'r=32':>8} {'r=64':>8} {'Memory (r=16)':>14} {'Notes':<20}"
    )
    print("-" * 90)

    for size in MODEL_SPECS.keys():
        row = f"{size:<6}"

        # Test different ranks
        for rank in [8, 16, 32, 64]:
            r = analyze_lora_training_feasibility(size, lora_rank=rank)
            if r.fits:
                row += f" {'✓':>7}"
            else:
                row += f" {'✗':>7}"

        # Show memory for rank=16 (most common)
        r16 = analyze_lora_training_feasibility(size, lora_rank=16)
        row += f" {format_gb(r16.memory_used_gb):>14}"

        # Notes
        notes_str = ", ".join(r16.notes[:2]) if r16.notes else ""
        row += f" {notes_str:<20}"
        print(row)

    print("-" * 90)
    print("✓ = Fits on RTX 5080 16GB    ✗ = Doesn't fit")
    print("r = LoRA rank (higher = more capacity, more memory)")
    print()


def print_model_analysis(model_size: str) -> None:
    """Print detailed analysis for a specific model."""
    spec = get_model_spec(model_size)  # type: ignore[arg-type]
    mem = estimate_memory(spec)

    print(f"\n{'=' * 70}")
    print(f"DETAILED ANALYSIS: {model_size} on RTX 5080 16GB")
    print(f"{'=' * 70}")

    print(f"\nModel: {spec.name} ({spec.total_params_billions():.2f}B parameters)")
    print(f"Architecture: {spec.hidden_size}h × {spec.num_layers}L × {spec.num_heads}heads")
    if spec.uses_gqa:
        print(f"GQA: {spec.num_heads}:{spec.effective_num_kv_heads} ratio")

    print("\nWeight Storage:")
    print(f"  FP32:    {format_gb(mem.weights_fp32 / 1024**3)}")
    print(f"  FP16:    {format_gb(mem.weights_fp16 / 1024**3)}")
    print(f"  Packed:  {format_gb(mem.weights_packed_ternary / 1024**3)}")

    print("\n--- INFERENCE ANALYSIS ---")
    results = analyze_inference_feasibility(model_size, use_streaming=False)
    streaming_results = analyze_inference_feasibility(model_size, use_streaming=True)

    print("\nWithout Layer Streaming:")
    for r in results:
        status = "✓ FITS" if r.fits else "✗ NO"
        print(
            f"  {r.context_length // 1024}K context: {status} ({format_gb(r.memory_used_gb)} used)"
        )

    print("\nWith Layer Streaming:")
    for r in streaming_results:
        status = "✓ FITS" if r.fits else "✗ NO"
        print(
            f"  {r.context_length // 1024}K context: {status} ({format_gb(r.memory_used_gb)} used)"
        )

    print("\n--- TRAINING ANALYSIS ---")
    train_r = analyze_training_feasibility(model_size)
    status = "✓ FEASIBLE" if train_r.fits else "✗ NOT FEASIBLE"
    print(f"Status: {status}")
    print(f"Memory: {format_gb(train_r.memory_used_gb)}")
    print(f"Max context: {train_r.context_length}")
    print(f"Techniques: {', '.join(train_r.notes)}")

    if not train_r.fits:
        print("\nFull training alternatives:")
        print("  - Use gradient accumulation with micro-batches")
        print("  - Use LoRA/QLoRA for fine-tuning (see below)")
        print("  - Use multi-GPU setup")

    # LoRA training analysis
    print("\n--- LoRA/QLoRA TRAINING ANALYSIS ---")
    for rank in [8, 16, 32, 64]:
        lora_r = analyze_lora_training_feasibility(model_size, lora_rank=rank)
        status = "✓ FITS" if lora_r.fits else "✗ NO"
        print(
            f"  Rank {rank:>2}: {status} ({format_gb(lora_r.memory_used_gb)} used, ctx {lora_r.context_length})"
        )

    # Recommendation
    print("\n--- RECOMMENDATION ---")
    lora_r16 = analyze_lora_training_feasibility(model_size, lora_rank=16)
    if lora_r16.fits:
        print(f"  QLoRA with rank=16 is recommended for {model_size}")
        print(f"  Trainable params: ~{lora_r16.memory_used_gb:.2f}GB total memory")
        print(f"  Max context: {lora_r16.context_length}")
    else:
        # Try to find a rank that works
        for rank in [8, 4]:
            lora_r = analyze_lora_training_feasibility(model_size, lora_rank=rank)
            if lora_r.fits:
                print(f"  QLoRA with rank={rank} is recommended for {model_size}")
                print("  (Lower rank due to memory constraints)")
                break
        else:
            print(f"  {model_size} requires multi-GPU for training even with QLoRA")
    print()


def print_test_plan() -> None:
    """Print recommended testing order for RTX 5080."""
    print("\n" + "=" * 80)
    print("RTX 5080 TESTING PLAN")
    print("=" * 80)
    print()
    print("Test models in this order, from easiest to most challenging:")
    print()

    # Categorize models
    easy_inference = []
    medium_inference = []
    streaming_required = []
    training_possible = []

    for size in MODEL_SPECS.keys():
        inf_results = analyze_inference_feasibility(size, use_streaming=False)
        train_result = analyze_training_feasibility(size)

        # Check 32K context without streaming
        r_32k = next(r for r in inf_results if r.context_length == 32768)

        if r_32k.fits and r_32k.headroom_gb > 4:
            easy_inference.append((size, r_32k))
        elif r_32k.fits:
            medium_inference.append((size, r_32k))
        else:
            streaming_required.append(size)

        if train_result.fits:
            training_possible.append((size, train_result))

    print("PHASE 1: Easy Inference (fits with 32K context, good headroom)")
    print("-" * 60)
    for size, r in easy_inference:
        print(f"  {size}: {format_gb(r.memory_used_gb)} used, {format_gb(r.headroom_gb)} headroom")

    print()
    print("PHASE 2: Medium Inference (fits with 32K, limited headroom)")
    print("-" * 60)
    for size, r in medium_inference:
        print(f"  {size}: {format_gb(r.memory_used_gb)} used, {format_gb(r.headroom_gb)} headroom")

    print()
    print("PHASE 3: Streaming Required (test with layer streaming)")
    print("-" * 60)
    for size in streaming_required:
        spec = get_model_spec(size)  # type: ignore[arg-type]
        print(f"  {size}: {spec.total_params_billions():.1f}B params")

    print()
    print("FULL TRAINING CANDIDATES (BF16 + checkpointing):")
    print("-" * 60)
    if training_possible:
        for size, r in training_possible:
            print(f"  {size}: {format_gb(r.memory_used_gb)} used, context {r.context_length}")
    else:
        print("  None fit for full training")

    print()
    print("LoRA/QLoRA TRAINING CANDIDATES (rank=16):")
    print("-" * 60)
    lora_training_possible = []
    for size in MODEL_SPECS.keys():
        lora_r = analyze_lora_training_feasibility(size, lora_rank=16)
        if lora_r.fits:
            lora_training_possible.append((size, lora_r))

    for size, r in lora_training_possible:
        print(f"  {size}: {format_gb(r.memory_used_gb)} used, context {r.context_length}")

    print()
    print("=" * 80)
    print("RECOMMENDED TESTING COMMANDS:")
    print("=" * 80)
    print()

    if easy_inference:
        first_model = easy_inference[0][0]
        print(f"# 1. Quick validation with {first_model}")
        print('python -c "')
        print("from tritter.core import TritterConfig")
        print(f"config = TritterConfig(model_size='{first_model}')")
        print("print(f'{config.model_size}: {config.hidden_size}h x {config.num_layers}L')")
        print('"')
        print()

    print("# 2. Run inference benchmark")
    print("python scripts/benchmark_packed.py --memory")
    print()

    print("# 3. Test layer streaming (if needed)")
    print("python scripts/benchmark_packed.py --simulate-7b")
    print()

    print("# 4. Run model-specific tests")
    print("pytest tests/unit/test_model_specs.py -v")
    print("pytest tests/unit/test_packed_ternary.py -v")
    print("pytest tests/integration/test_packed_inference.py -v")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RTX 5080 16GB feasibility analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Analyze specific model size (e.g., 7B)",
    )
    parser.add_argument(
        "--training",
        "-t",
        action="store_true",
        help="Focus on training feasibility",
    )
    parser.add_argument(
        "--test-plan",
        action="store_true",
        help="Generate recommended test plan",
    )

    args = parser.parse_args()

    print(f"RTX 5080: {RTX5080_VRAM_GB:.0f}GB VRAM ({USABLE_VRAM_GB:.1f}GB usable)")

    if args.test_plan:
        print_test_plan()
    elif args.model:
        print_model_analysis(args.model)
    else:
        print_feasibility_matrix()


if __name__ == "__main__":
    main()
