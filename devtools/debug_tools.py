#!/usr/bin/env python3
"""Comprehensive debugging tools for Tritter development.

Why: Debugging deep learning models requires structured, comprehensive output.
This module provides tools for model introspection, memory profiling, and
gradient analysis with well-formatted output.

Usage:
    from devtools.debug_tools import (
        print_model_summary,
        print_memory_breakdown,
        print_gradient_flow,
        validate_checkpoint,
    )

    # Or via CLI:
    python -m devtools.debug_tools summary --model 7B
    python -m devtools.debug_tools memory
    python -m devtools.debug_tools checkpoint path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LayerInfo:
    """Information about a single layer."""

    name: str
    module_type: str
    param_count: int
    trainable_count: int
    param_size_mb: float
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None


@dataclass
class ModelSummary:
    """Comprehensive model summary."""

    model_name: str
    total_params: int
    trainable_params: int
    frozen_params: int
    total_size_mb: float
    layers: list[LayerInfo]
    layer_type_counts: dict[str, int]


def get_model_summary(model: nn.Module, name: str = "Model") -> ModelSummary:
    """Get comprehensive model summary.

    Args:
        model: PyTorch model to summarize
        name: Name to use in output

    Returns:
        ModelSummary with detailed layer information
    """
    layers = []
    type_counts = defaultdict(int)

    total_params = 0
    trainable_params = 0

    for layer_name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # Skip container modules

        module_type = module.__class__.__name__
        type_counts[module_type] += 1

        param_count = 0
        trainable_count = 0

        for param in module.parameters(recurse=False):
            param_count += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()

        total_params += param_count
        trainable_params += trainable_count

        if param_count > 0:
            param_size_mb = param_count * 4 / (1024**2)  # Assume FP32
            layers.append(
                LayerInfo(
                    name=layer_name,
                    module_type=module_type,
                    param_count=param_count,
                    trainable_count=trainable_count,
                    param_size_mb=param_size_mb,
                )
            )

    return ModelSummary(
        model_name=name,
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=total_params - trainable_params,
        total_size_mb=total_params * 4 / (1024**2),
        layers=layers,
        layer_type_counts=dict(type_counts),
    )


def print_model_summary(
    model: nn.Module,
    name: str = "Model",
    show_layers: bool = True,
    max_layers: int = 50,
) -> None:
    """Print formatted model summary.

    Args:
        model: PyTorch model
        name: Model name for display
        show_layers: Whether to show individual layers
        max_layers: Max layers to show (to avoid overwhelming output)
    """
    summary = get_model_summary(model, name)

    print()
    print("=" * 80)
    print(f"MODEL SUMMARY: {summary.model_name}")
    print("=" * 80)
    print()
    print(f"Total Parameters:     {summary.total_params:>15,}")
    print(f"Trainable Parameters: {summary.trainable_params:>15,}")
    print(f"Frozen Parameters:    {summary.frozen_params:>15,}")
    print(f"Total Size (FP32):    {summary.total_size_mb:>15.2f} MB")
    print()

    # Layer type breakdown
    print("Layer Type Distribution:")
    print("-" * 40)
    for layer_type, count in sorted(
        summary.layer_type_counts.items(), key=lambda x: -x[1]
    ):
        print(f"  {layer_type:<25} {count:>5}")
    print()

    if show_layers and summary.layers:
        print("Top Layers by Parameter Count:")
        print("-" * 80)
        print(f"{'Layer Name':<50} {'Type':<15} {'Params':>12}")
        print("-" * 80)

        sorted_layers = sorted(summary.layers, key=lambda x: -x.param_count)
        for layer in sorted_layers[:max_layers]:
            name_display = (
                layer.name[:47] + "..."
                if len(layer.name) > 50
                else layer.name
            )
            params_str = f"{layer.param_count:,}"
            print(f"{name_display:<50} {layer.module_type:<15} {params_str:>12}")

        if len(summary.layers) > max_layers:
            print(f"... and {len(summary.layers) - max_layers} more layers")

    print("=" * 80)


def get_memory_breakdown(device: torch.device | str = "cuda:0") -> dict[str, Any]:
    """Get detailed GPU memory breakdown.

    Args:
        device: Target device

    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device(device)
    device_idx = device.index if device.index is not None else 0

    props = torch.cuda.get_device_properties(device_idx)
    allocated = torch.cuda.memory_allocated(device_idx)
    reserved = torch.cuda.memory_reserved(device_idx)
    max_allocated = torch.cuda.max_memory_allocated(device_idx)
    max_reserved = torch.cuda.max_memory_reserved(device_idx)

    # Get memory snapshot if available
    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        snapshot = []

    return {
        "device": str(device),
        "device_name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "allocated_gb": allocated / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "free_gb": (props.total_memory - reserved) / (1024**3),
        "max_allocated_gb": max_allocated / (1024**3),
        "max_reserved_gb": max_reserved / (1024**3),
        "fragmentation": (reserved - allocated) / reserved if reserved > 0 else 0,
        "allocation_count": len(snapshot),
    }


def print_memory_breakdown(device: torch.device | str = "cuda:0") -> None:
    """Print formatted memory breakdown."""
    info = get_memory_breakdown(device)

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    print()
    print("=" * 60)
    print(f"GPU MEMORY BREAKDOWN: {info['device_name']}")
    print("=" * 60)
    print()
    print(f"Device:           {info['device']}")
    print(f"Total Memory:     {info['total_memory_gb']:.2f} GB")
    print()
    print("Current Usage:")
    print(f"  Allocated:      {info['allocated_gb']:.2f} GB")
    print(f"  Reserved:       {info['reserved_gb']:.2f} GB")
    print(f"  Free:           {info['free_gb']:.2f} GB")
    print()
    print("Peak Usage:")
    print(f"  Max Allocated:  {info['max_allocated_gb']:.2f} GB")
    print(f"  Max Reserved:   {info['max_reserved_gb']:.2f} GB")
    print()
    print(f"Fragmentation:    {info['fragmentation']:.1%}")
    print(f"Active Allocations: {info['allocation_count']}")
    print("=" * 60)


def get_gradient_stats(model: nn.Module) -> dict[str, dict[str, float]]:
    """Get gradient statistics for all parameters.

    Args:
        model: Model after backward pass

    Returns:
        Dictionary mapping parameter names to gradient stats
    """
    stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
                "numel": grad.numel(),
                "has_nan": torch.isnan(grad).any().item(),
                "has_inf": torch.isinf(grad).any().item(),
            }
        else:
            stats[name] = {"grad": None}

    return stats


def print_gradient_flow(
    model: nn.Module,
    threshold: float = 1e-6,
    show_all: bool = False,
) -> None:
    """Print gradient flow analysis.

    Args:
        model: Model after backward pass
        threshold: Threshold for "vanishing" gradient warning
        show_all: Show all gradients, not just problematic ones
    """
    stats = get_gradient_stats(model)

    print()
    print("=" * 80)
    print("GRADIENT FLOW ANALYSIS")
    print("=" * 80)
    print()

    issues = []
    healthy = []

    for name, grad_stats in stats.items():
        if grad_stats.get("grad") is None:
            issues.append((name, "No gradient (not in computation graph)"))
            continue

        if grad_stats["has_nan"]:
            issues.append((name, "NaN in gradient!"))
        elif grad_stats["has_inf"]:
            issues.append((name, "Inf in gradient!"))
        elif grad_stats["norm"] < threshold:
            issues.append((name, f"Vanishing gradient (norm={grad_stats['norm']:.2e})"))
        elif grad_stats["norm"] > 1000:
            issues.append((name, f"Exploding gradient (norm={grad_stats['norm']:.2e})"))
        else:
            healthy.append(name)

    if issues:
        print("⚠️  Gradient Issues Found:")
        print("-" * 80)
        for name, issue in issues:
            short_name = name[:60] + "..." if len(name) > 60 else name
            print(f"  {short_name:<63} {issue}")
        print()

    print(f"✅ {len(healthy)} parameters with healthy gradients")

    if show_all and healthy:
        print()
        print("Gradient Statistics (all parameters):")
        print("-" * 80)
        print(f"{'Parameter':<50} {'Norm':>12} {'Mean':>12}")
        print("-" * 80)

        for name in healthy[:30]:
            s = stats[name]
            short_name = name[:47] + "..." if len(name) > 50 else name
            print(f"{short_name:<50} {s['norm']:>12.4e} {s['mean']:>12.4e}")

    print("=" * 80)


def validate_checkpoint(path: str | Path) -> dict[str, Any]:
    """Validate checkpoint integrity and structure.

    Args:
        path: Path to checkpoint file or directory

    Returns:
        Validation results dictionary
    """
    path = Path(path)
    results = {
        "valid": True,
        "path": str(path),
        "errors": [],
        "warnings": [],
        "info": {},
    }

    if not path.exists():
        results["valid"] = False
        results["errors"].append(f"Path does not exist: {path}")
        return results

    # Check format
    if path.is_dir():
        # Progressive checkpoint or multi-file
        if (path / "weights.safetensors").exists():
            results["info"]["format"] = "progressive"
            results["info"]["has_weights"] = True

            if (path / "progressive.json").exists():
                results["info"]["has_metadata"] = True
                try:
                    with open(path / "progressive.json") as f:
                        meta = json.load(f)
                    results["info"]["metadata"] = meta
                except Exception as e:
                    results["warnings"].append(f"Failed to parse metadata: {e}")
            else:
                results["warnings"].append("Missing progressive.json metadata")

        else:
            results["errors"].append("Directory missing expected checkpoint files")
            results["valid"] = False

    elif path.suffix == ".safetensors":
        results["info"]["format"] = "safetensors"
        try:
            import safetensors.torch as st
            tensors = st.load_file(str(path))
            results["info"]["tensor_count"] = len(tensors)
            results["info"]["total_params"] = sum(t.numel() for t in tensors.values())
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to load safetensors: {e}")

    elif path.suffix in (".pt", ".pth", ".bin"):
        results["info"]["format"] = "pytorch"
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    results["info"]["tensor_count"] = len(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    results["info"]["tensor_count"] = len(checkpoint["state_dict"])
                else:
                    results["info"]["tensor_count"] = len(checkpoint)
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to load PyTorch checkpoint: {e}")

    elif path.suffix == ".gguf":
        results["info"]["format"] = "gguf"
        results["warnings"].append("GGUF validation not fully implemented")

    else:
        results["valid"] = False
        results["errors"].append(f"Unknown checkpoint format: {path.suffix}")

    return results


def print_checkpoint_validation(path: str | Path) -> None:
    """Print formatted checkpoint validation results."""
    results = validate_checkpoint(path)

    print()
    print("=" * 60)
    print("CHECKPOINT VALIDATION")
    print("=" * 60)
    print()
    print(f"Path:   {results['path']}")
    print(f"Valid:  {'✅ Yes' if results['valid'] else '❌ No'}")
    print()

    if results["info"]:
        print("Information:")
        for key, value in results["info"].items():
            if key == "metadata":
                print(f"  {key}: (detailed below)")
            else:
                print(f"  {key}: {value}")
        print()

    if results.get("info", {}).get("metadata"):
        meta = results["info"]["metadata"]
        print("Metadata:")
        print(f"  Format Version: {meta.get('format_version', 'unknown')}")
        if "current_size" in meta:
            size = meta["current_size"]
            print(f"  Model Size: {size.get('model_size', 'unknown')}")
            print(f"  Hidden Dim: {size.get('hidden_size', 'unknown')}")
            print(f"  Layers: {size.get('num_layers', 'unknown')}")
        if "training_progress" in meta:
            prog = meta["training_progress"]
            print(f"  Tokens Seen: {prog.get('tokens_seen', 0):,}")
            print(f"  Training Steps: {prog.get('steps', 0):,}")
        print()

    if results["errors"]:
        print("❌ Errors:")
        for error in results["errors"]:
            print(f"  - {error}")
        print()

    if results["warnings"]:
        print("⚠️  Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
        print()

    print("=" * 60)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Tritter debugging tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Print model summary")
    summary_parser.add_argument("--model", default="1B", help="Model size (1B, 3B, 7B)")

    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Print memory breakdown")
    memory_parser.add_argument("--device", default="cuda:0", help="Device to check")

    # Checkpoint command
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Validate checkpoint")
    checkpoint_parser.add_argument("path", help="Path to checkpoint")

    args = parser.parse_args()

    if args.command == "summary":
        # Need to import tritter
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from tritter import TritterConfig, TritterModel
            config = TritterConfig(model_size=args.model)
            model = TritterModel(config)
            print_model_summary(model, f"Tritter-{args.model}")
        except ImportError as e:
            print(f"Error importing tritter: {e}")
            return 1

    elif args.command == "memory":
        print_memory_breakdown(args.device)

    elif args.command == "checkpoint":
        print_checkpoint_validation(args.path)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
