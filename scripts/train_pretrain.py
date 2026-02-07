#!/usr/bin/env python3
"""Pretraining script for Tritter models.

This script handles the full pretraining pipeline for generating pretrained
weights that can be distributed via HuggingFace Hub.

Usage:
    # Train 1B model (baseline, ~4 hours on RTX 5080)
    python scripts/train_pretrain.py --model 1B --data-dir data/pretrain

    # Train 3B model (primary, ~12 hours on RTX 5080)
    python scripts/train_pretrain.py --model 3B --data-dir data/pretrain

    # Train 7B model (flagship, ~36 hours on RTX 5080)
    python scripts/train_pretrain.py --model 7B --data-dir data/pretrain

    # Resume from checkpoint
    python scripts/train_pretrain.py --model 3B --resume checkpoints/3b-step-50000/

    # Progressive training (3B → 7B)
    python scripts/train_pretrain.py --model 7B --init-from checkpoints/3b-final/

Data Preparation:
    See scripts/prepare_pretrain_data.py for data curation.

Output:
    Checkpoints saved to: checkpoints/{model}-step-{step}/
    Final model saved to: checkpoints/{model}-final/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tritter import TritterConfig, TritterModel
from tritter.tokenization import ModalityType
from tritter.training import Trainer, TrainingConfig
from tritter.utils import check_memory_fit, detect_gpu_profile


class PretrainDataset(IterableDataset):
    """Streaming dataset for pretraining.

    Why: Large pretraining corpora don't fit in memory. Stream from disk.

    Expected data format (JSONL):
        {"text": "...", "source": "python", "quality_score": 0.95}

    Preprocessing:
        - Tokenize on-the-fly
        - Pack sequences to max_length
        - Skip low-quality samples
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        min_quality: float = 0.7,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_quality = min_quality

        # Find all data shards
        self.shards = sorted(self.data_dir.glob("*.jsonl"))
        if not self.shards:
            raise ValueError(f"No .jsonl files found in {data_dir}")

    def __iter__(self):
        """Yield packed sequences."""
        buffer = []
        buffer_len = 0

        for shard in self.shards:
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Quality filter
                    quality = item.get("quality_score", 1.0)
                    if quality < self.min_quality:
                        continue

                    # Tokenize
                    text = item.get("text", "")
                    if not text:
                        continue

                    # Determine modality from language field
                    language = item.get("language", "python")
                    modality = (
                        ModalityType.CODE
                        if language
                        in ("python", "rust", "javascript", "typescript", "go", "java", "c", "cpp")
                        else ModalityType.TEXT
                    )

                    tokens = self.tokenizer.encode(text, modality=modality, language=language)

                    # Add to buffer
                    buffer.extend(tokens)
                    buffer_len += len(tokens)

                    # Yield packed sequences
                    while buffer_len >= self.max_length:
                        seq = buffer[: self.max_length]
                        buffer = buffer[self.max_length :]
                        buffer_len -= self.max_length

                        yield {
                            "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                            "labels": torch.tensor(seq[1:], dtype=torch.long),
                        }


def create_training_config(
    model_size: str,
    profile,
    total_tokens: int = 100_000_000_000,  # 100B tokens
    warmup_tokens: int = 1_000_000_000,  # 1B tokens
) -> TrainingConfig:
    """Create training config optimized for the hardware profile.

    Why: Different GPUs have different batch size / accumulation tradeoffs.
    """
    # Base config - reduced for memory efficiency
    # 1B model with full precision requires ~12GB+ for weights+gradients+optimizer
    batch_size = 1  # Reduced for memory
    seq_length = 512  # Reduced from 2048 for memory

    # Increase gradient accumulation to compensate for smaller batch
    if profile.vram_gb >= 24:
        grad_accum = 32
    elif profile.vram_gb >= 16:
        grad_accum = 64  # More accumulation to reach effective batch
    else:
        grad_accum = 128

    # Effective batch size in tokens
    effective_batch_tokens = batch_size * seq_length * grad_accum

    # Calculate steps
    total_steps = max(1, total_tokens // effective_batch_tokens)
    warmup_steps = min(
        warmup_tokens // effective_batch_tokens, total_steps // 10
    )  # Cap warmup at 10% of total

    # Learning rate based on model size
    # Handle various size formats: "1B", "125M", "test"
    size_upper = model_size.upper()
    if size_upper == "TEST":
        model_b = 0.01  # ~10M params
    elif size_upper.endswith("M"):
        model_b = float(size_upper.rstrip("M")) / 1000  # Convert M to B
    else:
        model_b = float(size_upper.rstrip("B"))

    if model_b <= 0.5:
        lr = 5e-4  # Higher LR for tiny models
    elif model_b <= 1:
        lr = 3e-4
    elif model_b <= 3:
        lr = 2e-4
    elif model_b <= 7:
        lr = 1e-4
    else:
        lr = 5e-5

    return TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        max_steps=total_steps,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        weight_decay=0.1,
        save_steps=5000,
        eval_steps=1000,
        log_steps=100,
        use_amp=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Pretrain Tritter models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model size (1B, 3B, 7B, etc.)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data (.jsonl files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        help="Initialize from a smaller model (for progressive training)",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=100_000_000_000,
        help="Total tokens to train on (default: 100B)",
    )
    parser.add_argument(
        "--no-bitnet",
        action="store_true",
        help="Disable BitNet quantization (use standard FP16/32 training)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without training",
    )

    args = parser.parse_args()

    # Detect hardware
    profile = detect_gpu_profile()
    if profile is None:
        print("Warning: Unknown GPU, using conservative settings")
        from tritter.utils.hardware_profiles import RTX_5080

        profile = RTX_5080

    print(f"\n{'=' * 60}")
    print("Tritter Pretraining")
    print(f"{'=' * 60}")
    print(f"Model size:    {args.model}")
    print(f"Hardware:      {profile.name} ({profile.vram_gb:.0f}GB)")
    print(f"Data dir:      {args.data_dir}")
    print(f"Total tokens:  {args.total_tokens:,}")
    print(f"{'=' * 60}\n")

    # Check memory fit
    from tritter.core.model_specs import get_model_spec

    spec = get_model_spec(args.model)
    if spec is None:
        print(f"Error: Unknown model size '{args.model}'")
        sys.exit(1)

    # Estimate training memory (weights + gradients + optimizer + activations)
    # BitNet 1.58-bit = 2 bits per weight, but training uses full precision
    params = spec.total_params()
    packed_size_gb = (params * 2) / 8 / (1024**3)  # 2-bit packed weights
    training_memory = packed_size_gb * 20  # Rough estimate for full training
    fits, message = check_memory_fit(training_memory)

    if not fits:
        print(f"Warning: {message}")
        print("Consider using QLoRA fine-tuning instead (scripts/train_lora.py)")
        if not args.dry_run:
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                sys.exit(1)

    # Create model config
    use_bitnet = not args.no_bitnet
    model_config = TritterConfig(model_size=args.model, use_bitnet=use_bitnet)
    print(f"BitNet:        {'Enabled' if use_bitnet else 'Disabled'}")

    # Create training config
    train_config = create_training_config(args.model, profile, total_tokens=args.total_tokens)

    # BitNet-specific training adjustments
    # Why: BitNet's STE (straight-through estimator) requires:
    # 1. Disable AMP - FP16 scaling interferes with ternary quantization gradients
    # 2. Slightly lower learning rate for stability
    # 3. Gradient clipping already in config (max_grad_norm=1.0)
    if use_bitnet:
        train_config.use_amp = False
        train_config.learning_rate = train_config.learning_rate * 0.5  # 2x lower for QAT
        print("BitNet QAT mode: AMP disabled, LR adjusted for stability")

    print("Model config:")
    print(f"  Hidden dim:  {model_config.hidden_size}")
    print(f"  Layers:      {model_config.num_layers}")
    print(f"  Heads:       {model_config.num_heads}")
    print()
    print("Training config:")
    print(f"  Batch size:  {train_config.batch_size}")
    print(f"  Grad accum:  {train_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Total steps: {train_config.max_steps:,}")
    print(f"  Warmup:      {train_config.warmup_steps:,}")
    print()

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    # Create model
    print("Creating model...")
    model = TritterModel(model_config)

    # Enable gradient checkpointing for memory efficiency
    # Why: Reduces memory by ~60% by recomputing activations during backward
    print("Enabling gradient checkpointing for memory efficiency...")
    model.gradient_checkpointing_enable()

    # Load from smaller model for progressive training
    if args.init_from:
        print(f"Loading weights from {args.init_from}...")
        # TODO: Implement progressive initialization (DUS)
        print("Warning: Progressive training not yet implemented")
        print("Starting from scratch...")

    # Create tokenizer
    from tritter.tokenization import MultiModalTokenizer

    tokenizer = MultiModalTokenizer()

    # Create dataset
    print(f"Loading data from {args.data_dir}...")
    dataset = PretrainDataset(
        args.data_dir,
        tokenizer,
        max_length=model_config.max_position_embeddings,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=train_config,
        train_dataloader=dataloader,
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    # Create output directory
    output_dir = Path(args.output_dir)
    model_dir = output_dir / f"{args.model.lower()}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {model_dir}")
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        save_path = model_dir / f"interrupt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        trainer.save_checkpoint(str(save_path))
        print(f"Checkpoint saved to: {save_path}")
        sys.exit(1)

    # Save final model
    final_path = model_dir / "final"
    trainer.save_checkpoint(str(final_path))
    print(f"\nFinal model saved to: {final_path}")

    # Save metadata
    metadata = {
        "model_size": args.model,
        "total_tokens": args.total_tokens,
        "training_config": train_config.__dict__,
        "hardware": profile.name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(final_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
