#!/usr/bin/env python3
"""Benchmark script for packed ternary inference.

This script measures memory usage, transfer speeds, and inference throughput
for packed vs unpacked ternary weights.

USAGE:
    # Memory comparison (CPU-only, fast)
    python scripts/benchmark_packed.py --memory

    # Full benchmark with transfer timing (requires CUDA)
    python scripts/benchmark_packed.py --full

    # Verify 7B model memory estimation
    python scripts/benchmark_packed.py --verify-7b

    # Simulate 7B model layer-by-layer (requires ~2GB VRAM)
    python scripts/benchmark_packed.py --simulate-7b

    # Run all benchmarks
    python scripts/benchmark_packed.py --all

OUTPUTS:
    - Memory usage comparison (packed vs unpacked)
    - CPU->GPU transfer time (if CUDA available)
    - Inference throughput (tokens/sec)
    - Layer-by-layer memory profile

Why: Validates that packed ternary weights achieve the expected
memory reduction (~16x from FP32, ~8x from INT8) and don't
significantly impact inference speed.
"""

import argparse
import gc
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

# Add src to path for imports
sys.path.insert(0, str(__file__).replace("/scripts/benchmark_packed.py", "/src"))

from tritter.quantization.bitnet import TernaryWeight
from tritter.quantization.packed_ternary import (
    PackedTernaryWeight,
    pack_ternary,
    unpack_ternary,
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    value: float
    unit: str
    description: str


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def time_function(fn: Callable, warmup: int = 3, iterations: int = 10) -> float:
    """Time a function with warmup and averaging."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Sync if CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations


def benchmark_memory() -> list[BenchmarkResult]:
    """Benchmark memory usage: FP32 vs INT8 vs Packed.

    Returns list of benchmark results.
    """
    results = []

    # Test sizes typical of transformer layers
    sizes = [
        (4096, 4096, "Attention projection"),
        (4096, 11008, "FFN up projection (7B)"),
        (11008, 4096, "FFN down projection (7B)"),
        (4096, 32000, "LM head (7B)"),
    ]

    print("\n=== Memory Usage Comparison ===")
    print("-" * 80)
    print(f"{'Layer':<25} {'FP32':<12} {'INT8':<12} {'Packed':<12} {'Ratio':<10}")
    print("-" * 80)

    total_fp32 = 0
    total_int8 = 0
    total_packed = 0

    for out_features, in_features, name in sizes:
        # FP32 (standard PyTorch)
        fp32_bytes = out_features * in_features * 4

        # INT8 (naive quantization)
        int8_bytes = out_features * in_features * 1

        # Packed (2 bits per weight + scales)
        packed_layer = PackedTernaryWeight(in_features, out_features, bias=False)
        packed_bytes = packed_layer.memory_bytes()

        ratio = fp32_bytes / packed_bytes

        print(
            f"{name:<25} "
            f"{format_bytes(fp32_bytes):<12} "
            f"{format_bytes(int8_bytes):<12} "
            f"{format_bytes(packed_bytes):<12} "
            f"{ratio:.1f}x"
        )

        total_fp32 += fp32_bytes
        total_int8 += int8_bytes
        total_packed += packed_bytes

    print("-" * 80)
    print(
        f"{'TOTAL':<25} "
        f"{format_bytes(total_fp32):<12} "
        f"{format_bytes(total_int8):<12} "
        f"{format_bytes(total_packed):<12} "
        f"{total_fp32 / total_packed:.1f}x"
    )

    results.append(
        BenchmarkResult(
            name="memory_reduction_vs_fp32",
            value=total_fp32 / total_packed,
            unit="x",
            description="Memory reduction factor compared to FP32",
        )
    )

    results.append(
        BenchmarkResult(
            name="memory_reduction_vs_int8",
            value=total_int8 / total_packed,
            unit="x",
            description="Memory reduction factor compared to INT8",
        )
    )

    return results


def benchmark_transfer_speed() -> list[BenchmarkResult]:
    """Benchmark CPU->GPU transfer speed (requires CUDA).

    Returns list of benchmark results.
    """
    if not torch.cuda.is_available():
        print("\n=== Transfer Speed ===")
        print("CUDA not available, skipping transfer benchmark")
        return []

    results = []
    device = torch.device("cuda:0")

    # Simulate a large FFN layer
    out_features, in_features = 4096, 11008
    iterations = 20

    # Create packed and unpacked weights on CPU
    ternary = TernaryWeight(in_features, out_features, bias=False)
    packed = PackedTernaryWeight.from_ternary_weight(ternary)

    # Ensure on CPU
    ternary.to("cpu")

    # Get the actual tensors we'll transfer
    fp32_weight = ternary.weight.detach().clone()
    packed_weight = packed.packed_weight.detach().clone()

    print("\n=== CPU->GPU Transfer Speed ===")
    print(f"Layer size: {out_features} x {in_features}")
    print("-" * 60)

    # Time FP32 transfer
    def transfer_fp32():
        return fp32_weight.to(device, non_blocking=False)

    fp32_time = time_function(transfer_fp32, warmup=3, iterations=iterations)
    fp32_bytes = fp32_weight.numel() * 4
    fp32_gbps = (fp32_bytes / fp32_time) / 1e9

    print(
        f"FP32 transfer: {fp32_time * 1000:.2f}ms ({format_bytes(fp32_bytes)}) = {fp32_gbps:.2f} GB/s"
    )

    # Time packed transfer
    def transfer_packed():
        return packed_weight.to(device, non_blocking=False)

    packed_time = time_function(transfer_packed, warmup=3, iterations=iterations)
    packed_bytes = packed_weight.numel() * 1
    packed_gbps = (packed_bytes / packed_time) / 1e9

    print(
        f"Packed transfer: {packed_time * 1000:.2f}ms ({format_bytes(packed_bytes)}) = {packed_gbps:.2f} GB/s"
    )

    speedup = fp32_time / packed_time
    print(f"Transfer speedup: {speedup:.2f}x")

    results.append(
        BenchmarkResult(
            name="transfer_speedup",
            value=speedup,
            unit="x",
            description="CPU->GPU transfer speedup from packing",
        )
    )

    return results


def benchmark_inference_speed() -> list[BenchmarkResult]:
    """Benchmark inference speed: packed vs unpacked.

    Returns list of benchmark results.
    """
    results = []

    # Use CPU for fair comparison (GPU would be memory-bound anyway)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create layers
    in_features, out_features = 4096, 4096
    batch_size, seq_len = 1, 128

    ternary = TernaryWeight(in_features, out_features, bias=True).to(device)
    packed = PackedTernaryWeight.from_ternary_weight(ternary).to(device)

    x = torch.randn(batch_size, seq_len, in_features, device=device)

    print(f"\n=== Inference Speed (on {device}) ===")
    print(f"Layer size: {out_features} x {in_features}")
    print(f"Input shape: ({batch_size}, {seq_len}, {in_features})")
    print("-" * 60)

    # Time TernaryWeight forward
    ternary.eval()

    def forward_ternary():
        with torch.no_grad():
            return ternary(x)

    ternary_time = time_function(forward_ternary, warmup=5, iterations=50)
    print(f"TernaryWeight forward: {ternary_time * 1000:.3f}ms")

    # Time PackedTernaryWeight forward (includes unpacking)
    def forward_packed():
        with torch.no_grad():
            return packed(x)

    packed_time = time_function(forward_packed, warmup=5, iterations=50)
    print(f"PackedTernaryWeight forward: {packed_time * 1000:.3f}ms")

    overhead = (packed_time - ternary_time) / ternary_time * 100
    print(f"Unpacking overhead: {overhead:.1f}%")

    results.append(
        BenchmarkResult(
            name="inference_overhead",
            value=overhead,
            unit="%",
            description="Inference overhead from on-the-fly unpacking",
        )
    )

    return results


def benchmark_pack_unpack_speed() -> list[BenchmarkResult]:
    """Benchmark pack/unpack operations.

    Returns list of benchmark results.
    """
    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create large tensor
    out_features, in_features = 4096, 4096
    weights = torch.randint(-1, 2, (out_features, in_features), device=device).float()
    scale = torch.ones(out_features, 1, device=device)

    print(f"\n=== Pack/Unpack Speed (on {device}) ===")
    print(f"Tensor size: {out_features} x {in_features}")
    print("-" * 60)

    # Time packing
    def do_pack():
        return pack_ternary(weights.cpu(), scale.cpu())

    pack_time = time_function(do_pack, warmup=2, iterations=10)
    print(f"Pack time: {pack_time * 1000:.2f}ms")

    # Get packed tensor for unpack timing
    packed, _ = pack_ternary(weights.cpu(), scale.cpu())
    packed = packed.to(device)
    scale_gpu = scale.to(device)

    # Time unpacking (this is the hot path during inference)
    def do_unpack():
        return unpack_ternary(packed, scale_gpu, in_features)

    unpack_time = time_function(do_unpack, warmup=5, iterations=50)
    print(f"Unpack time: {unpack_time * 1000:.3f}ms")

    results.append(
        BenchmarkResult(
            name="unpack_time_ms",
            value=unpack_time * 1000,
            unit="ms",
            description="Time to unpack a 4096x4096 layer",
        )
    )

    return results


def verify_7b_memory() -> list[BenchmarkResult]:
    """Verify memory estimation for 7B model.

    Returns list of benchmark results.
    """
    results = []

    print("\n=== 7B Model Memory Estimation ===")
    print("-" * 60)

    # 7B model architecture (approximate)
    # Based on LLaMA-7B: 32 layers, hidden=4096, intermediate=11008, vocab=32000
    num_layers = 32
    hidden_size = 4096
    intermediate_size = 11008
    vocab_size = 32000
    num_heads = 32
    head_dim = hidden_size // num_heads

    # Per-layer params (ternary quantized)
    # - Q, K, V projections: 3 * hidden * hidden = 3 * 4096 * 4096
    # - O projection: hidden * hidden = 4096 * 4096
    # - Up projection: hidden * intermediate = 4096 * 11008
    # - Gate projection: hidden * intermediate = 4096 * 11008
    # - Down projection: intermediate * hidden = 11008 * 4096

    layer_params = (
        3 * hidden_size * hidden_size  # Q, K, V
        + hidden_size * hidden_size  # O
        + 2 * hidden_size * intermediate_size  # Up, Gate
        + intermediate_size * hidden_size  # Down
    )

    total_layer_params = num_layers * layer_params

    # Embedding and LM head
    embedding_params = vocab_size * hidden_size  # Usually not quantized
    lm_head_params = hidden_size * vocab_size  # Can be quantized

    total_params = total_layer_params + embedding_params + lm_head_params

    print(f"Estimated total parameters: {total_params / 1e9:.2f}B")

    # Memory calculations
    # FP32: 4 bytes/param
    fp32_bytes = total_params * 4
    print(f"FP32 storage: {format_bytes(fp32_bytes)}")

    # Packed ternary: 2 bits/param + scales
    # Packed weights: total_params * 0.25 bytes
    # Scales: ~total_params / 4096 * 4 bytes (per-channel)
    packed_weight_bytes = total_params * 0.25
    scale_bytes = (total_params / 4096) * 4
    packed_total = packed_weight_bytes + scale_bytes

    print(f"Packed storage: {format_bytes(int(packed_total))}")
    print(f"Reduction: {fp32_bytes / packed_total:.1f}x")

    # Memory budget breakdown for RTX 5080 16GB
    print("\n--- RTX 5080 16GB Memory Budget ---")
    print(f"Model weights (packed): {format_bytes(int(packed_total))}")

    # KV-cache estimate for 128K context
    # Per layer: 2 * batch * seq_len * num_heads * head_dim * dtype_size
    # With INT4 KV-cache: 0.5 bytes per value
    batch_size = 1
    context_length = 128 * 1024  # 128K
    kv_per_layer = 2 * batch_size * context_length * num_heads * head_dim * 0.5  # INT4
    kv_total = num_layers * kv_per_layer

    print(f"KV-cache (128K, INT4): {format_bytes(int(kv_total))}")

    # Activations (rough estimate)
    # Peak activation: batch * seq_len * hidden * 4 * num_concurrent_layers
    activation_estimate = batch_size * 4096 * hidden_size * 4 * 2  # 2 layers worth

    print(f"Activations (estimate): {format_bytes(int(activation_estimate))}")

    total_gpu_usage = packed_total + kv_total + activation_estimate
    print(f"\nTotal estimated GPU usage: {format_bytes(int(total_gpu_usage))}")
    print(f"RTX 5080 16GB headroom: {format_bytes(int(16 * 1024**3 - total_gpu_usage))}")

    results.append(
        BenchmarkResult(
            name="7b_packed_size_gb",
            value=packed_total / 1024**3,
            unit="GB",
            description="Estimated 7B model packed size",
        )
    )

    return results


def simulate_7b_inference() -> list[BenchmarkResult]:
    """Simulate 7B model inference layer-by-layer.

    Requires ~2GB VRAM to run.

    Returns list of benchmark results.
    """
    if not torch.cuda.is_available():
        print("\n=== 7B Simulation ===")
        print("CUDA not available, skipping simulation")
        return []

    results = []
    device = torch.device("cuda:0")

    print("\n=== 7B Model Layer Simulation ===")
    print("Simulating layer-by-layer forward pass...")
    print("-" * 60)

    # Simulate a single 7B-like layer
    hidden_size = 4096
    intermediate_size = 11008
    batch_size = 1
    seq_len = 1024  # Start with shorter sequence

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    initial_memory = torch.cuda.memory_allocated(device)
    print(f"Initial GPU memory: {format_bytes(initial_memory)}")

    # Create packed layers for one transformer block
    q_proj = PackedTernaryWeight(hidden_size, hidden_size, bias=False).to(device)
    k_proj = PackedTernaryWeight(hidden_size, hidden_size, bias=False).to(device)
    v_proj = PackedTernaryWeight(hidden_size, hidden_size, bias=False).to(device)
    o_proj = PackedTernaryWeight(hidden_size, hidden_size, bias=False).to(device)
    up_proj = PackedTernaryWeight(hidden_size, intermediate_size, bias=False).to(device)
    gate_proj = PackedTernaryWeight(hidden_size, intermediate_size, bias=False).to(device)
    down_proj = PackedTernaryWeight(intermediate_size, hidden_size, bias=False).to(device)

    layer_memory = torch.cuda.memory_allocated(device) - initial_memory
    print(f"One layer packed weights: {format_bytes(layer_memory)}")

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Forward pass simulation
    with torch.no_grad():
        # Attention projections
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        # Simplified attention (no actual attention computation)
        attn_out = q + k + v  # Placeholder
        proj_out = o_proj(attn_out)

        # FFN
        up = up_proj(proj_out)
        gate = gate_proj(proj_out)
        hidden = up * torch.sigmoid(gate)  # SwiGLU-like
        _ = down_proj(hidden)  # Final output (unused, just for memory profiling)

    peak_memory = torch.cuda.max_memory_allocated(device)
    print(f"Peak GPU memory: {format_bytes(peak_memory)}")

    # Estimate for 32 layers
    estimated_32_layers = layer_memory * 32
    print(f"\nEstimated 32 layers (no streaming): {format_bytes(estimated_32_layers)}")
    print("With layer streaming: Only 1-2 layer groups in memory at once")

    results.append(
        BenchmarkResult(
            name="single_layer_memory",
            value=layer_memory / 1024**2,
            unit="MB",
            description="Memory for one packed 7B layer",
        )
    )

    return results


def run_all_benchmarks() -> None:
    """Run all benchmarks and summarize results."""
    all_results = []

    all_results.extend(benchmark_memory())
    all_results.extend(benchmark_transfer_speed())
    all_results.extend(benchmark_inference_speed())
    all_results.extend(benchmark_pack_unpack_speed())
    all_results.extend(verify_7b_memory())
    all_results.extend(simulate_7b_inference())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in all_results:
        print(f"{result.name}: {result.value:.2f} {result.unit}")
        print(f"  {result.description}")


def main() -> None:
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark packed ternary inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--memory", action="store_true", help="Run memory comparison benchmark")
    parser.add_argument(
        "--transfer", action="store_true", help="Run CPU->GPU transfer benchmark (requires CUDA)"
    )
    parser.add_argument("--inference", action="store_true", help="Run inference speed benchmark")
    parser.add_argument(
        "--pack-unpack", action="store_true", help="Run pack/unpack speed benchmark"
    )
    parser.add_argument("--verify-7b", action="store_true", help="Verify 7B memory estimation")
    parser.add_argument(
        "--simulate-7b", action="store_true", help="Simulate 7B inference (requires CUDA)"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite (memory + transfer + inference)",
    )

    args = parser.parse_args()

    # Default to --all if no flags specified
    if not any(vars(args).values()):
        args.all = True

    print("=" * 60)
    print("Packed Ternary Inference Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")

    if args.all:
        run_all_benchmarks()
    else:
        if args.memory or args.full:
            benchmark_memory()
        if args.transfer or args.full:
            benchmark_transfer_speed()
        if args.inference or args.full:
            benchmark_inference_speed()
        if args.pack_unpack:
            benchmark_pack_unpack_speed()
        if args.verify_7b:
            verify_7b_memory()
        if args.simulate_7b:
            simulate_7b_inference()


if __name__ == "__main__":
    main()
