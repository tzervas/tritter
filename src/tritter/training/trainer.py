"""Training loop for Tritter with BitNet quantization-aware training.

Why: BitNet 1.58-bit quantization requires special handling during training:
1. Full-precision shadow weights maintained for gradient updates
2. Straight-through estimator (STE) for gradients through quantization
3. Quantized weights used in forward pass only

This trainer starts with standard token prediction (cross-entropy loss).
Embedding prediction training (Coconut-style) will be added later via
curriculum scheduling.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tritter.core.config import TritterConfig
from tritter.models.architecture import TritterModel
from tritter.utils.memory_utils import log_memory_snapshot
from tritter.utils.profile_naming import resolve_profile_name


@dataclass
class TrainingMetrics:
    """Metrics tracked during training.

    Why: Centralized metrics tracking enables consistent logging and reporting.
    """

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    lr: float = 0.0
    tokens_per_sec: float = 0.0
    gpu_memory_gb: float = 0.0
    grad_norm: float = 0.0


class TrainingProgress:
    """Visual progress tracking for training loop.

    Why: Provides clear visual feedback on training status, progress, timing,
    and resource usage. Essential for monitoring long-running training jobs
    to distinguish between stuck loops and normal training.

    Attributes:
        total_steps: Maximum training steps
        log_steps: Steps between log updates
        start_time: Training start timestamp
        current_status: Current training phase description

    Example:
        >>> progress = TrainingProgress(total_steps=10000, log_steps=10)
        >>> progress.start()
        >>> for step in range(10000):
        >>>     progress.update(step, loss=0.5, lr=1e-4)
        >>> progress.finish()
    """

    # Status indicators
    STATUS_WARMUP = "🔥 Warming up"
    STATUS_TRAINING = "🚀 Training"
    STATUS_EVALUATING = "📊 Evaluating"
    STATUS_SAVING = "💾 Saving"
    STATUS_COMPLETE = "✅ Complete"
    STATUS_ERROR = "❌ Error"

    def __init__(
        self,
        total_steps: int,
        log_steps: int = 10,
        warmup_steps: int = 0,
        output: TextIO | None = None,
    ) -> None:
        """Initialize progress tracker.

        Args:
            total_steps: Maximum training steps
            log_steps: Steps between progress updates
            warmup_steps: Number of warmup steps
            output: Output stream (defaults to stdout)
        """
        self.total_steps = total_steps
        self.log_steps = log_steps
        self.warmup_steps = warmup_steps
        self.output = output or sys.stdout

        self.start_time: datetime | None = None
        self.step_times: list[float] = []
        self.current_status = self.STATUS_TRAINING
        self.last_metrics: TrainingMetrics | None = None
        self._last_update_time: float | None = None

    def _write(self, text: str, end: str = "\n") -> None:
        """Write to output stream."""
        self.output.write(text + end)
        self.output.flush()

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def _format_eta(self, remaining_steps: int) -> str:
        """Estimate time remaining based on recent step times."""
        if len(self.step_times) < 2:
            return "calculating..."

        # Use recent steps for ETA (last 100 or all if less)
        recent = self.step_times[-100:]
        avg_step_time = sum(recent) / len(recent)
        eta_seconds = avg_step_time * remaining_steps

        return self._format_duration(eta_seconds)

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9  # type: ignore[no-any-return]
        return 0.0

    def _progress_bar(self, progress: float, width: int = 30) -> str:
        """Create ASCII progress bar."""
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def start(self) -> None:
        """Record training start and print header."""
        self.start_time = datetime.now()
        self._last_update_time = time.time()

        self._write("\n" + "═" * 70)
        self._write("  TRITTER TRAINING")
        self._write("═" * 70)
        self._write(f"  Start time:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"  Total steps:   {self.total_steps:,}")
        self._write(f"  Warmup steps:  {self.warmup_steps:,}")
        self._write(f"  Log interval:  every {self.log_steps} steps")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self._write(f"  GPU:           {gpu_name} ({gpu_memory:.1f} GB)")
        self._write("═" * 70 + "\n")

    def update(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_per_sec: float = 0.0,
        grad_norm: float = 0.0,
        epoch: int = 0,
    ) -> None:
        """Update progress with current metrics.

        Args:
            step: Current training step
            loss: Current loss value
            lr: Current learning rate
            tokens_per_sec: Training throughput
            grad_norm: Gradient norm (for monitoring)
            epoch: Current epoch
        """
        now = time.time()

        # Track step timing (need at least one previous timestamp to compute delta)
        if self._last_update_time is not None:
            step_time = now - self._last_update_time
            self.step_times.append(step_time)
        self._last_update_time = now

        # Store metrics
        self.last_metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            lr=lr,
            tokens_per_sec=tokens_per_sec,
            gpu_memory_gb=self._get_gpu_memory(),
            grad_norm=grad_norm,
        )

        # Determine status
        if step < self.warmup_steps:
            self.current_status = self.STATUS_WARMUP
        else:
            self.current_status = self.STATUS_TRAINING

        # Only log at intervals
        if step % self.log_steps != 0:
            return

        # Calculate progress
        progress = step / self.total_steps
        remaining_steps = self.total_steps - step
        eta = self._format_eta(remaining_steps)

        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        elapsed_str = self._format_duration(elapsed.total_seconds())

        # Format progress line
        bar = self._progress_bar(progress)
        pct = progress * 100

        # Build status line
        status_line = (
            f"{self.current_status} │ "
            f"Step {step:>7,}/{self.total_steps:,} │ "
            f"{bar} {pct:>5.1f}% │ "
            f"Loss: {loss:.4f} │ "
            f"LR: {lr:.2e} │ "
            f"Elapsed: {elapsed_str} │ "
            f"ETA: {eta}"
        )

        self._write(status_line)

        # Additional metrics line (GPU memory, throughput)
        if torch.cuda.is_available() or tokens_per_sec > 0:
            gpu_mem = self._get_gpu_memory()
            extras = []
            if gpu_mem > 0:
                extras.append(f"GPU: {gpu_mem:.2f} GB")
            if tokens_per_sec > 0:
                extras.append(f"Throughput: {tokens_per_sec:.0f} tok/s")
            if grad_norm > 0:
                extras.append(f"Grad norm: {grad_norm:.3f}")
            if extras:
                self._write(f"           │ {' │ '.join(extras)}")

    def set_status(self, status: str) -> None:
        """Update current status indicator.

        Args:
            status: New status string (use STATUS_* constants)
        """
        self.current_status = status
        self._write(f"\n{status}")

    def log_eval(self, metrics: dict[str, float]) -> None:
        """Log evaluation results.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        self._write("\n" + "─" * 50)
        self._write(f"  {self.STATUS_EVALUATING} Results:")
        for name, value in metrics.items():
            self._write(f"    {name}: {value:.4f}")
        self._write("─" * 50 + "\n")

    def log_checkpoint(self, path: str) -> None:
        """Log checkpoint save.

        Args:
            path: Checkpoint path
        """
        self._write(f"\n{self.STATUS_SAVING} Checkpoint saved to: {path}\n")

    def finish(self, error: str | None = None) -> None:
        """Print training completion summary.

        Args:
            error: Error message if training failed
        """
        end_time = datetime.now()
        total_duration = end_time - self.start_time if self.start_time else timedelta(0)

        self._write("\n" + "═" * 70)

        if error:
            self._write(f"  {self.STATUS_ERROR}")
            self._write(f"  Error: {error}")
        else:
            self._write(f"  {self.STATUS_COMPLETE}")

        self._write("═" * 70)
        self._write(f"  End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"  Total duration: {self._format_duration(total_duration.total_seconds())}")

        if self.last_metrics:
            self._write(f"  Final step:    {self.last_metrics.step:,}")
            self._write(f"  Final loss:    {self.last_metrics.loss:.4f}")

        if len(self.step_times) > 0:
            avg_step = sum(self.step_times) / len(self.step_times)
            self._write(f"  Avg step time: {avg_step * 1000:.1f} ms")

        self._write("═" * 70 + "\n")


@dataclass
class TrainingConfig:
    """Configuration for training.

    Why: Separates training hyperparameters from model architecture config.
    This enables independent tuning of training dynamics (learning rate, batch size)
    without modifying model architecture. Also allows sharing model configs across
    different training regimes (pretraining vs fine-tuning).
    """

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Training mode
    training_mode: str = "pretrain"
    """Training mode tag for profiling.

    Why: Differentiates full pretraining from fine-tuning and adapter-based
    training in naming and reporting. "inference" is supported for profile
    naming helpers but should not be used in the training loop.
    """

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "checkpoints"

    # Logging
    log_steps: int = 10

    # Profile naming overrides
    profile_name_override: str | None = None
    profile_tag_overrides: dict[str, str | float] = field(default_factory=dict)

    # Mixed precision
    use_amp: bool = True

    def __post_init__(self) -> None:
        """Validate training configuration.

        Why: Early validation catches configuration errors before expensive training starts.
        Prevents issues like negative learning rates or warmup longer than total training.
        """
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert self.warmup_steps >= 0, f"warmup_steps must be non-negative, got {self.warmup_steps}"
        assert self.max_steps > self.warmup_steps, (
            f"max_steps ({self.max_steps}) must be greater than warmup_steps ({self.warmup_steps})"
        )
        assert self.gradient_accumulation_steps > 0, (
            f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
        )

        valid_training_modes = {"pretrain", "finetune", "lora", "qlora", "inference"}
        if self.training_mode not in valid_training_modes:
            raise ValueError(
                "training_mode must be one of "
                f"{sorted(valid_training_modes)}, got {self.training_mode}"
            )


class Trainer:
    """Training loop with BitNet QAT support.

    Why: Handles the training loop including:
    - Forward/backward with STE for quantized weights (BitNet handles this internally)
    - Gradient accumulation for larger effective batch size on limited VRAM
    - Mixed precision (FP16/BF16) for memory efficiency and speed
    - Checkpointing for fault tolerance and experiment resumption
    - Logging for monitoring training progress

    This trainer implements standard autoregressive language modeling with cross-entropy loss.
    Embedding-prediction training (Coconut/LCM-style) will be added later through curriculum
    learning that gradually transitions from token prediction to embedding prediction.
    """

    def __init__(
        self,
        model: TritterModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize trainer.

        Args:
            model: TritterModel to train
            model_config: Model configuration
            training_config: Training hyperparameters
            train_dataloader: Training data
            eval_dataloader: Optional evaluation data
            device: Training device (defaults to CUDA if available)
            verbose: Enable visual progress feedback

        Why: Device selection prioritizes CUDA for speed. Model is moved to device
        before optimizer creation to ensure optimizer states are on correct device.
        Loss function ignores padding (index 0) to avoid backprop through padding tokens.
        """
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Move model to device
        self.model.to(self.device)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Profile naming and memory logging
        raw_vram_gb = 0.0
        loaded_vram_gb = 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            device_index = self.device.index or 0
            raw_vram_gb = torch.cuda.get_device_properties(device_index).total_memory / 1e9
            loaded_vram_gb = torch.cuda.memory_allocated(device_index) / 1e9

        tag_overrides = {
            **model_config.profile_tag_overrides,
            **self.config.profile_tag_overrides,
            "optimizer": "adamw",
        }
        profile_name, profile_metadata = resolve_profile_name(
            config=model_config,
            training_mode=self.config.training_mode,
            use_amp=self.config.use_amp,
            vram_raw_gb=raw_vram_gb if raw_vram_gb > 0 else None,
            vram_loaded_gb=loaded_vram_gb if loaded_vram_gb > 0 else None,
            name_override=self.config.profile_name_override or model_config.profile_name_override,
            optimizer_name="adamw",
            tag_overrides=tag_overrides,
        )
        self.profile_name = profile_name
        self.profile_metadata = profile_metadata

        profile_payload = profile_metadata.to_dict()
        profile_payload["name"] = profile_name
        profile_path = Path(self.config.output_dir) / "profile.json"
        profile_path.write_text(json.dumps(profile_payload, indent=2))

        log_memory_snapshot(
            Path(self.config.output_dir) / "memory_logs.jsonl",
            tag="model_loaded",
            extra={
                "raw_vram_gb": raw_vram_gb,
                "loaded_vram_gb": loaded_vram_gb,
                "profile_name": profile_name,
            },
        )

        # Optimizer (created after model.to(device))
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (CUDA only)
        # Why: GradScaler prevents underflow in FP16 training by scaling gradients.
        # Only needed for CUDA; CPU training uses FP32.
        self.scaler = GradScaler() if self.config.use_amp and self.device.type == "cuda" else None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Loss function
        # Why: CrossEntropyLoss with ignore_index=0 skips padding tokens. Padding doesn't
        # contribute to loss or gradients, preventing the model from learning to predict padding.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # Progress tracker
        # Why: Visual feedback helps monitor training progress and distinguish between
        # stuck loops and normal training. Essential for long-running jobs.
        self.progress = (
            TrainingProgress(
                total_steps=training_config.max_steps,
                log_steps=training_config.log_steps,
                warmup_steps=training_config.warmup_steps,
            )
            if verbose
            else None
        )

        # Create output directory
        # (already created before profile logging)

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay.

        Returns:
            Configured AdamW optimizer

        Why: AdamW (Adam with decoupled weight decay) separates L2 regularization from
        gradient updates, which works better than standard Adam with L2 penalty.
        We exclude certain parameters from weight decay:
        - Biases: Small number of parameters, minimal impact on overfitting
        - LayerNorm weights: Normalization layers shouldn't be regularized
        This follows standard practice from BERT, GPT, and other transformers.
        """
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude biases and normalization layers from weight decay
            if "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(param_groups, lr=self.config.learning_rate)

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with linear warmup and cosine decay.

        Returns:
            Learning rate scheduler

        Why: Warmup prevents early training instability when gradients are large and
        model is far from convergence. Linear warmup over first N steps gradually
        increases LR from 0 to target. Cosine decay provides smooth reduction from
        target to near-zero, avoiding sharp drops that can destabilize training.
        This schedule is standard for transformer pretraining (GPT, BERT, etc).
        """

        def lr_lambda(step: int) -> float:
            """Compute LR multiplier for given step.

            Args:
                step: Current training step

            Returns:
                LR multiplier (0.0 to 1.0)
            """
            if step < self.config.warmup_steps:
                # Linear warmup: 0 → 1 over warmup_steps
                return step / max(1, self.config.warmup_steps)
            # Cosine decay: 1 → ~0 over remaining steps
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            # Cosine annealing: 0.5 * (1 + cos(π * progress))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())  # type: ignore[no-any-return]

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute single training step.

        Args:
            batch: Dictionary with 'input_ids' and optionally 'attention_mask'

        Returns:
            Loss value for this step

        Why: Autoregressive language modeling shifts input by one position to create
        targets. If input is [A, B, C, D], model sees [A, B, C] and predicts [B, C, D].
        This teacher-forcing approach is standard for transformer training. Mixed precision
        (autocast) reduces memory usage and speeds training on modern GPUs with Tensor Cores.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Targets are input shifted by one position
        # input: [A, B, C, D] → model_input: [A, B, C], targets: [B, C, D]
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1].contiguous()

        # Forward pass with optional mixed precision
        # Why: autocast automatically uses FP16 for certain ops (matmul, conv) while
        # keeping others in FP32 (softmax, loss) for numerical stability.
        if self.scaler is not None:
            with autocast():
                logits = self.model(input_ids, attention_mask)  # (B, L, V)
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),  # (B*L, V)
                    targets.view(-1),  # (B*L,)
                )
        else:
            logits = self.model(input_ids, attention_mask)
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        # Scale loss for gradient accumulation
        # Why: When accumulating gradients over N steps, each step's gradients should
        # be 1/N to maintain the same effective learning rate as a single large batch.
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        # Why: GradScaler scales loss before backward to prevent FP16 underflow in
        # gradients. Gradients are unscaled before optimizer step.
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return float(loss.item() * self.config.gradient_accumulation_steps)

    def train(self) -> None:
        """Run training loop.

        Why: Main training loop handling:
        - Gradient accumulation: Accumulate gradients over multiple steps to simulate
          larger batch size without increasing memory usage
        - Gradient clipping: Prevent exploding gradients in early training
        - Optimizer step: Update weights only after accumulation completes
        - Logging: Track loss and learning rate
        - Checkpointing: Save model state for recovery and deployment
        - Evaluation: Monitor validation performance

        Visual feedback: Progress bar, ETA, throughput, and GPU memory are displayed
        at each log interval to monitor training status.
        """
        # Start progress tracking
        if self.progress:
            self.progress.start()

        try:
            self._train_loop()
        except KeyboardInterrupt:
            if self.progress:
                self.progress.finish(error="Training interrupted by user")
            raise
        except Exception as e:
            if self.progress:
                self.progress.finish(error=str(e))
            raise

        # Training complete
        if self.progress:
            self.progress.finish()

    def _train_loop(self) -> None:
        """Internal training loop implementation.

        Why: Separated from train() to enable clean exception handling and
        progress tracking in the outer method.
        """
        self.model.train()
        accumulated_loss = 0.0
        step_start_time = time.time()
        tokens_processed = 0

        # Create iterator (will cycle through dataset)
        data_iter = iter(self.train_dataloader)

        while self.global_step < self.config.max_steps:
            # Get batch (restart iterator if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Track tokens for throughput
            batch_tokens = batch["input_ids"].numel()
            tokens_processed += batch_tokens

            # Training step
            loss = self.train_step(batch)
            accumulated_loss += loss

            # Gradient accumulation check
            # Why: Only update weights after accumulating N steps to get effective
            # batch size of (batch_size * gradient_accumulation_steps)
            grad_norm = 0.0
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                # Why: Unscale gradients before clipping (if using mixed precision).
                # Clip by global norm to prevent exploding gradients while preserving
                # gradient direction. Max norm of 1.0 is standard for transformers.
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                ).item()

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            # Logging with visual progress
            if self.global_step % self.config.log_steps == 0:
                avg_loss = accumulated_loss / self.config.log_steps
                lr = self.scheduler.get_last_lr()[0]

                # Calculate throughput
                elapsed = time.time() - step_start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                if self.progress:
                    self.progress.update(
                        step=self.global_step,
                        loss=avg_loss,
                        lr=lr,
                        tokens_per_sec=tokens_per_sec,
                        grad_norm=grad_norm,
                        epoch=self.epoch,
                    )
                else:
                    print(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")

                # Reset accumulators
                accumulated_loss = 0.0
                step_start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                if self.progress:
                    self.progress.set_status(TrainingProgress.STATUS_EVALUATING)
                eval_metrics = self.evaluate()
                if self.progress:
                    self.progress.log_eval(eval_metrics)
                self.model.train()  # Return to training mode

            # Checkpointing
            if self.global_step % self.config.save_steps == 0:
                if self.progress:
                    self.progress.set_status(TrainingProgress.STATUS_SAVING)
                checkpoint_path = self.save_checkpoint()
                if self.progress:
                    self.progress.log_checkpoint(checkpoint_path)

        # Final save
        if self.progress:
            self.progress.set_status(TrainingProgress.STATUS_SAVING)
        final_path = self.save_checkpoint()
        if self.progress:
            self.progress.log_checkpoint(final_path)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def evaluate(self) -> dict[str, float]:
        """Run evaluation.

        Returns:
            Dictionary of evaluation metrics (eval_loss, perplexity)

        Why: Evaluation uses inference_mode (not no_grad) for better performance.
        Perplexity = exp(loss) measures model uncertainty - lower is better.
        Perplexity of 1 means perfect prediction, higher values indicate more confusion.
        Standard metric for language models (reported in all LM papers).
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Shift for autoregressive prediction
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1].contiguous()

            logits = self.model(input_ids, attention_mask)
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            # Accumulate loss weighted by number of tokens
            # Why: Different batches may have different numbers of non-padding tokens.
            # Weighting by token count gives correct average loss across full dataset.
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            num_batches += 1

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Only print if not using progress tracker (it will handle display)
        if not self.progress:
            print(f"Eval: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "eval_batches": num_batches,
            "eval_tokens": total_tokens,
        }

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save training checkpoint.

        Args:
            path: Optional custom path, defaults to output_dir/checkpoint-{step}

        Returns:
            Path where checkpoint was saved

        Why: Checkpoints enable:
        - Fault tolerance: Resume training after crashes or preemption
        - Model deployment: Use saved weights for inference
        - Experiment tracking: Compare models from different training stages

        We save full training state (model, optimizer, scheduler, scaler, step count)
        to enable exact resumption. For deployment, only model_state_dict is needed.
        """
        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f"checkpoint-{self.global_step}",
            )

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
            "model_config": self.model_config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, os.path.join(path, "trainer_state.pt"))

        # Only print if not using progress tracker
        if not self.progress:
            print(f"Saved checkpoint to {path}")

        return path

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint directory

        Why: Loading checkpoint restores full training state to resume from exact
        point where training stopped. map_location ensures checkpoint loads on
        correct device (e.g., can load CUDA checkpoint on CPU for debugging).
        weights_only=False is required to load training state (optimizer, scheduler).
        """
        checkpoint_file = os.path.join(path, "trainer_state.pt")
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {path} at step {self.global_step}")
