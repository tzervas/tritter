#!/usr/bin/env python3
"""Benchmark Hybrid Predictive Training vs Full Training.

Compares:
1. Local checkpoints (100M, 500M models)
2. HuggingFace models (tzervas/tritter-100m-bitnet, etc.)

Generates comparison report with:
- Loss trajectories
- Speedup metrics
- Backward reduction percentages
- Memory usage

Usage:
    # Compare local checkpoints
    python scripts/benchmark_hybrid_vs_full.py --local

    # Download and compare HuggingFace models (requires HF token)
    python scripts/benchmark_hybrid_vs_full.py --hf

    # Full comparison
    python scripts/benchmark_hybrid_vs_full.py --all
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_local_metrics(checkpoint_dir: Path) -> dict[str, Any]:
    """Load metrics from a local checkpoint."""
    metrics_path = checkpoint_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def load_training_summary(checkpoint_dir: Path) -> dict[str, Any]:
    """Load training summary from checkpoint directory."""
    summary_path = checkpoint_dir / "training_metrics.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def compare_local_checkpoints(base_dir: Path) -> dict[str, Any]:
    """Compare all local checkpoints."""
    results = {}

    for model_size in ["100M", "500M", "1B"]:
        model_dir = base_dir / model_size
        if not model_dir.exists():
            continue

        results[model_size] = {}

        # Full training baseline
        full_dir = model_dir / "final"
        if full_dir.exists():
            metrics = load_local_metrics(full_dir)
            if not metrics:
                metrics = load_training_summary(model_dir)
            results[model_size]["full_training"] = metrics

        # Hybrid training
        hybrid_dir = model_dir / "hybrid" / "final"
        if hybrid_dir.exists():
            results[model_size]["hybrid_embedding"] = load_local_metrics(hybrid_dir)

        # Phase hybrid training
        phase_hybrid_dir = model_dir / "phase_hybrid" / "final"
        if phase_hybrid_dir.exists():
            results[model_size]["phase_hybrid"] = load_local_metrics(phase_hybrid_dir)

        # Hybrid RS compatible
        hybrid_rs_dir = model_dir / "hybrid_rs" / "final"
        if hybrid_rs_dir.exists():
            results[model_size]["hybrid_rs"] = load_local_metrics(hybrid_rs_dir)

    return results


def list_hf_models(username: str = "tzervas") -> list[dict]:
    """List models from HuggingFace."""
    if not HF_AVAILABLE:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return []

    api = HfApi()
    models = api.list_models(author=username)
    return [
        {"id": m.id, "downloads": getattr(m, 'downloads', 0), "updated": str(getattr(m, 'lastModified', ''))}
        for m in models
        if "tritter" in m.id.lower()
    ]


def download_hf_metrics(model_id: str) -> dict[str, Any]:
    """Download metrics file from HuggingFace model."""
    if not HF_AVAILABLE:
        return {}

    try:
        metrics_path = hf_hub_download(repo_id=model_id, filename="training_metrics.json")
        with open(metrics_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not download metrics for {model_id}: {e}")
        return {}


def format_comparison_table(results: dict[str, dict]) -> str:
    """Format results as a comparison table."""
    lines = []
    lines.append("=" * 80)
    lines.append("TRAINING METHODOLOGY COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    for model_size, variants in results.items():
        lines.append(f"\n## {model_size} Model\n")
        lines.append(f"{'Variant':<25} {'Final Loss':>12} {'Min Loss':>12} {'Steps':>10} {'Tok/s':>12}")
        lines.append("-" * 75)

        for variant_name, metrics in variants.items():
            final_loss = metrics.get('final_loss', metrics.get('loss', 'N/A'))
            min_loss = metrics.get('min_loss', 'N/A')
            steps = metrics.get('total_steps', metrics.get('step', 'N/A'))
            tok_s = metrics.get('tokens_per_second', 'N/A')

            # Format values
            fl_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else str(final_loss)
            ml_str = f"{min_loss:.4f}" if isinstance(min_loss, (int, float)) else str(min_loss)
            steps_str = f"{steps:,}" if isinstance(steps, int) else str(steps)
            tok_str = f"{tok_s:,.0f}" if isinstance(tok_s, (int, float)) else str(tok_s)

            lines.append(f"{variant_name:<25} {fl_str:>12} {ml_str:>12} {steps_str:>10} {tok_str:>12}")

        # Add backward reduction info for hybrid variants
        for variant_name, metrics in variants.items():
            if 'backward_reduction_percent' in metrics:
                br = metrics['backward_reduction_percent']
                div_events = metrics.get('divergence_events', 0)
                if isinstance(div_events, list):
                    div_events = len(div_events)
                lines.append(f"\n  {variant_name}:")
                lines.append(f"    Backward reduction: {br:.1f}%")
                lines.append(f"    Divergence events: {div_events}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark hybrid vs full training")
    parser.add_argument("--local", action="store_true", help="Compare local checkpoints")
    parser.add_argument("--hf", action="store_true", help="Compare HuggingFace models")
    parser.add_argument("--all", action="store_true", help="Full comparison")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("/home/kang/Documents/projects/github/python-ai/tritter/checkpoints"),
                        help="Local checkpoint directory")
    parser.add_argument("--output", type=Path, default=None, help="Output file for report")

    args = parser.parse_args()

    if not any([args.local, args.hf, args.all]):
        args.local = True  # Default to local

    results = {}

    if args.local or args.all:
        print("Comparing local checkpoints...")
        local_results = compare_local_checkpoints(args.checkpoint_dir)
        for model, variants in local_results.items():
            if model not in results:
                results[model] = {}
            results[model].update(variants)

    if args.hf or args.all:
        print("Listing HuggingFace models...")
        hf_models = list_hf_models("tzervas")
        print(f"Found {len(hf_models)} tritter models on HuggingFace:")
        for m in hf_models:
            print(f"  - {m['id']} ({m['downloads']} downloads)")

        # Note: Full HF comparison would require downloading model metrics
        # For now, just list them
        results["huggingface"] = {"models": hf_models}

    # Format and print report
    report = format_comparison_table(results)
    print(report)

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
            f.write("\n\n## Raw Data\n\n```json\n")
            f.write(json.dumps(results, indent=2, default=str))
            f.write("\n```\n")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
