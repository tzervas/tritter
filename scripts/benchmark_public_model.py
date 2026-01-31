#!/usr/bin/env python3
"""Benchmark Hybrid Predictive Training on Public GPT-2 Model.

This script provides a standardized 1-hour benchmark using:
- Model: GPT-2 small (124M params) from HuggingFace
- Dataset: WikiText-103 (public, no auth required)
- Evaluation: Perplexity + optional lm-eval-harness benchmarks

Benchmark Protocol:
1. Download GPT-2 small and WikiText-103
2. Train for 1 hour with hybrid methodology
3. Train baseline (full training) for 1 hour
4. Compare: loss curves, perplexity, throughput

Usage:
    # Full benchmark (2 hours total)
    python scripts/benchmark_public_model.py --full

    # Quick test (10 minutes each)
    python scripts/benchmark_public_model.py --quick

    # Hybrid only (1 hour)
    python scripts/benchmark_public_model.py --hybrid-only

    # Evaluation only (with existing checkpoints)
    python scripts/benchmark_public_model.py --eval-only --checkpoint path/to/model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Try to import transformers for GPT-2
try:
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. Install: pip install transformers")

# Try to import datasets for WikiText
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available. Install: pip install datasets")


# =============================================================================
# Dataset Wrapper
# =============================================================================

class WikiTextDataset(Dataset):
    """WikiText-103 dataset wrapper."""

    def __init__(self, tokenizer, split: str = "train", max_length: int = 1024):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load WikiText-103
        print(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Tokenize and concatenate all text
        all_tokens = []
        for item in dataset:
            text = item["text"]
            if text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)

        # Create chunks of max_length
        self.chunks = []
        for i in range(0, len(all_tokens) - max_length, max_length):
            self.chunks.append(all_tokens[i:i + max_length])

        print(f"Created {len(self.chunks)} chunks of {max_length} tokens")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
        }


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Model
    model_name: str = "gpt2"  # HuggingFace model name

    # Training
    batch_size: int = 4
    max_seq_length: int = 1024
    learning_rate: float = 3e-5  # Lower for fine-tuning
    weight_decay: float = 0.01
    max_time_seconds: int = 3600  # 1 hour

    # Hybrid config
    warmup_steps: int = 100
    full_steps: int = 20
    predict_steps: int = 15
    correct_every: int = 8

    # Output
    output_dir: Path = Path("benchmarks")
    save_every: int = 500
    log_every: int = 10


# =============================================================================
# Simple Phase Controller (Standalone)
# =============================================================================

class SimplePhaseController:
    """Simplified phase controller for benchmark."""

    def __init__(self, warmup_steps: int, full_steps: int, predict_steps: int, correct_every: int):
        self.warmup_steps = warmup_steps
        self.full_steps = full_steps
        self.predict_steps = predict_steps
        self.correct_every = correct_every

        self.step = 0
        self.phase_step = 0
        self.phase = "WARMUP"
        self.predict_horizon = 0
        self.correct_counter = 0

        self.grad_history = []
        self.loss_history = []

    def should_use_full_gradient(self) -> bool:
        return self.phase in ("WARMUP", "FULL", "CORRECT")

    def update(self, loss: float, grad_norm: float, gradients=None) -> str:
        self.step += 1
        self.phase_step += 1
        self.loss_history.append(loss)

        # Store gradient stats for prediction
        if gradients is not None and len(gradients) > 0:
            self.grad_history.append({k: v.clone() for k, v in list(gradients.items())[:5]})
            if len(self.grad_history) > 10:
                self.grad_history.pop(0)

        # Phase transitions
        if self.phase == "WARMUP":
            if self.phase_step >= self.warmup_steps:
                self.phase = "FULL"
                self.phase_step = 0

        elif self.phase == "FULL":
            if self.phase_step >= self.full_steps and len(self.grad_history) >= 5:
                self.phase = "PREDICT"
                self.phase_step = 0
                self.predict_horizon = 0

        elif self.phase == "PREDICT":
            self.predict_horizon += 1
            if self.predict_horizon >= self.predict_steps:
                self.phase = "CORRECT"
                self.phase_step = 0
                self.correct_counter = 0

        elif self.phase == "CORRECT":
            self.correct_counter += 1
            if self.correct_counter >= self.correct_every:
                self.phase = "FULL"
                self.phase_step = 0

        return self.phase

    def predict_gradients(self):
        """Simple gradient prediction using EMA."""
        if len(self.grad_history) < 3:
            return None

        result = {}
        for key in self.grad_history[-1].keys():
            grads = [h[key] for h in self.grad_history[-5:] if key in h]
            if grads:
                # Exponential moving average
                weights = [0.5 ** i for i in range(len(grads) - 1, -1, -1)]
                total_weight = sum(weights)
                avg = sum(g * w for g, w in zip(grads, weights)) / total_weight
                result[key] = avg

        return result


# =============================================================================
# Training Functions
# =============================================================================

def train_hybrid(model, dataloader, config: BenchmarkConfig, device) -> dict:
    """Train using hybrid methodology."""
    print("\n" + "=" * 60)
    print("HYBRID PREDICTIVE TRAINING")
    print("=" * 60)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=True) if device.type == "cuda" else None

    controller = SimplePhaseController(
        config.warmup_steps, config.full_steps, config.predict_steps, config.correct_every
    )

    start_time = time.time()
    step = 0
    total_tokens = 0
    loss_history = []
    phase_history = []
    backward_passes = 0
    forward_passes = 0

    data_iter = iter(dataloader)

    while time.time() - start_time < config.max_time_seconds:
        step += 1
        forward_passes += 1

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()

        # Forward pass
        with autocast("cuda", enabled=scaler is not None):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            continue

        # Phase-specific logic
        gradients = None
        if controller.should_use_full_gradient():
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            if math.isfinite(grad_norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                backward_passes += 1
        else:
            # Use predicted gradients
            optimizer.zero_grad()
            predicted = controller.predict_gradients()
            if predicted:
                for name, param in model.named_parameters():
                    if name in predicted:
                        param.grad = predicted[name]
                optimizer.step()
            grad_norm = 0.0

        phase = controller.update(loss_value, grad_norm, gradients)
        total_tokens += input_ids.numel()
        loss_history.append((step, loss_value))
        phase_history.append(phase)

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            remaining = config.max_time_seconds - elapsed
            tokens_per_sec = total_tokens / elapsed
            skip_pct = (1 - backward_passes / forward_passes) * 100

            print(f"Step {step:>5} | Phase: {phase:<8} | Loss: {loss_value:.4f} | "
                  f"Tok/s: {tokens_per_sec:,.0f} | Skip: {skip_pct:.1f}% | "
                  f"Remaining: {remaining/60:.1f}m")

    total_time = time.time() - start_time
    return {
        "methodology": "hybrid",
        "total_steps": step,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "final_loss": loss_history[-1][1] if loss_history else 0,
        "min_loss": min(l[1] for l in loss_history) if loss_history else 0,
        "backward_passes": backward_passes,
        "forward_passes": forward_passes,
        "backward_reduction": (1 - backward_passes / forward_passes) * 100,
        "tokens_per_second": total_tokens / total_time,
        "loss_history": loss_history[-100:],  # Keep last 100
    }


def train_full(model, dataloader, config: BenchmarkConfig, device) -> dict:
    """Train using full (traditional) methodology."""
    print("\n" + "=" * 60)
    print("FULL (BASELINE) TRAINING")
    print("=" * 60)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=True) if device.type == "cuda" else None

    start_time = time.time()
    step = 0
    total_tokens = 0
    loss_history = []

    data_iter = iter(dataloader)

    while time.time() - start_time < config.max_time_seconds:
        step += 1

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()

        # Forward + backward
        optimizer.zero_grad()

        with autocast("cuda", enabled=scaler is not None):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_tokens += input_ids.numel()
        loss_history.append((step, loss_value))

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            remaining = config.max_time_seconds - elapsed
            tokens_per_sec = total_tokens / elapsed

            print(f"Step {step:>5} | Loss: {loss_value:.4f} | "
                  f"Tok/s: {tokens_per_sec:,.0f} | Remaining: {remaining/60:.1f}m")

    total_time = time.time() - start_time
    return {
        "methodology": "full",
        "total_steps": step,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "final_loss": loss_history[-1][1] if loss_history else 0,
        "min_loss": min(l[1] for l in loss_history) if loss_history else 0,
        "backward_passes": step,
        "forward_passes": step,
        "backward_reduction": 0.0,
        "tokens_per_second": total_tokens / total_time,
        "loss_history": loss_history[-100:],
    }


def evaluate_perplexity(model, dataloader, device) -> float:
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    model.train()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(config: BenchmarkConfig):
    """Run full benchmark comparison."""
    if not HF_AVAILABLE or not DATASETS_AVAILABLE:
        print("Error: transformers and datasets libraries required")
        print("Install: pip install transformers datasets")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model and tokenizer
    print(f"\nLoading {config.model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("Loading WikiText-103...")
    train_dataset = WikiTextDataset(tokenizer, split="train", max_length=config.max_seq_length)
    valid_dataset = WikiTextDataset(tokenizer, split="validation", max_length=config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    # Output directory
    output_dir = config.output_dir / f"gpt2_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # =========================================================================
    # Hybrid Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Hybrid Predictive Training")
    print("=" * 70)

    model_hybrid = GPT2LMHeadModel.from_pretrained(config.model_name).to(device)
    hybrid_results = train_hybrid(model_hybrid, train_loader, config, device)

    # Evaluate
    hybrid_ppl = evaluate_perplexity(model_hybrid, valid_loader, device)
    hybrid_results["perplexity"] = hybrid_ppl

    # Save
    torch.save(model_hybrid.state_dict(), output_dir / "hybrid_model.pt")
    results["hybrid"] = hybrid_results

    del model_hybrid
    torch.cuda.empty_cache()

    # =========================================================================
    # Full Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Full (Baseline) Training")
    print("=" * 70)

    model_full = GPT2LMHeadModel.from_pretrained(config.model_name).to(device)
    full_results = train_full(model_full, train_loader, config, device)

    # Evaluate
    full_ppl = evaluate_perplexity(model_full, valid_loader, device)
    full_results["perplexity"] = full_ppl

    # Save
    torch.save(model_full.state_dict(), output_dir / "full_model.pt")
    results["full"] = full_results

    # =========================================================================
    # Comparison Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Hybrid':<15} {'Full':<15} {'Improvement':<15}")
    print("-" * 70)

    metrics = [
        ("Total Steps", "total_steps", "d"),
        ("Total Tokens", "total_tokens", ",d"),
        ("Final Loss", "final_loss", ".4f"),
        ("Min Loss", "min_loss", ".4f"),
        ("Perplexity", "perplexity", ".2f"),
        ("Tokens/Second", "tokens_per_second", ",.0f"),
        ("Backward Reduction %", "backward_reduction", ".1f"),
    ]

    for name, key, fmt in metrics:
        h_val = hybrid_results.get(key, 0)
        f_val = full_results.get(key, 0)

        if key in ("final_loss", "min_loss", "perplexity"):
            improvement = ((f_val - h_val) / f_val * 100) if f_val > 0 else 0
            imp_str = f"{improvement:+.1f}%"
        elif key == "tokens_per_second":
            improvement = (h_val / f_val) if f_val > 0 else 1
            imp_str = f"{improvement:.2f}x"
        elif key == "backward_reduction":
            imp_str = f"{h_val:.1f}%"
        else:
            imp_str = "-"

        h_str = f"{h_val:{fmt}}"
        f_str = f"{f_val:{fmt}}"
        print(f"{name:<25} {h_str:<15} {f_str:<15} {imp_str:<15}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hybrid vs Full Training")

    parser.add_argument("--full", action="store_true", help="Full benchmark (2 hours)")
    parser.add_argument("--quick", action="store_true", help="Quick test (20 minutes)")
    parser.add_argument("--hybrid-only", action="store_true", help="Hybrid training only")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-time", type=int, default=3600, help="Max time per method (seconds)")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_time_seconds=args.max_time if not args.quick else 600,  # 10 min for quick
    )

    if args.quick:
        print("Running QUICK benchmark (10 minutes per method)")
    elif args.full:
        print("Running FULL benchmark (1 hour per method)")
    else:
        print("Running DEFAULT benchmark (1 hour per method)")

    run_benchmark(config)


if __name__ == "__main__":
    main()
