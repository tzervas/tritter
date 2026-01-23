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

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tritter.core.config import TritterConfig
from tritter.models.architecture import TritterModel


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
    ) -> None:
        """Initialize trainer.

        Args:
            model: TritterModel to train
            model_config: Model configuration
            training_config: Training hyperparameters
            train_dataloader: Training data
            eval_dataloader: Optional evaluation data
            device: Training device (defaults to CUDA if available)

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

        # Move model to device
        self.model.to(self.device)

        # Optimizer (created after model.to(device))
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (CUDA only)
        # Why: GradScaler prevents underflow in FP16 training by scaling gradients.
        # Only needed for CUDA; CPU training uses FP32.
        self.scaler = (
            GradScaler()
            if self.config.use_amp and self.device.type == "cuda"
            else None
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Loss function
        # Why: CrossEntropyLoss with ignore_index=0 skips padding tokens. Padding doesn't
        # contribute to loss or gradients, preventing the model from learning to predict padding.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

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
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

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
        """
        self.model.train()
        accumulated_loss = 0.0

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

            # Training step
            loss = self.train_step(batch)
            accumulated_loss += loss

            # Gradient accumulation check
            # Why: Only update weights after accumulating N steps to get effective
            # batch size of (batch_size * gradient_accumulation_steps)
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                # Why: Unscale gradients before clipping (if using mixed precision).
                # Clip by global norm to prevent exploding gradients while preserving
                # gradient direction. Max norm of 1.0 is standard for transformers.
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_steps == 0:
                avg_loss = accumulated_loss / self.config.log_steps
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                accumulated_loss = 0.0

            # Evaluation
            if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                self.evaluate()
                self.model.train()  # Return to training mode

            # Checkpointing
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

        # Final save
        self.save_checkpoint()

    @torch.inference_mode()
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

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"Eval: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

        return {"eval_loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, path: str | None = None) -> None:
        """Save training checkpoint.

        Args:
            path: Optional custom path, defaults to output_dir/checkpoint-{step}

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
        print(f"Saved checkpoint to {path}")

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
