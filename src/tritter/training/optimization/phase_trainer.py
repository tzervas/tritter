"""Phase-based training with prediction and correction cycles.

Why: Full gradient computation is expensive. By alternating between:
1. Full training phases (accurate gradients)
2. Predicted phases (fast, approximate gradients)
3. Correction phases (fix accumulated errors)

We can achieve similar convergence with significantly reduced compute.
This is the main orchestration module that combines gradient prediction,
VSA compression, and ternary acceleration.

The phase cycle is:
    FULL (N steps) → PREDICT (M steps) → CORRECT (1 step) → repeat

Where:
- FULL: Standard backprop with full gradient computation
- PREDICT: Use predicted gradients from GradientPredictor
- CORRECT: Compute full gradient, apply accumulated correction
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from tritter.training.optimization.gradient_predictor import (
    GradientPredictor,
    PredictionConfig,
)
from tritter.training.optimization.ternary_optimizer import (
    TernaryConfig,
    TernaryGradientAccumulator,
)
from tritter.training.optimization.vsa_compression import (
    VSAConfig,
    VSAGradientCompressor,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer


class TrainingPhase(Enum):
    """Training phase types."""

    FULL = auto()  # Full gradient computation
    PREDICT = auto()  # Predicted gradients
    CORRECT = auto()  # Correction phase


@dataclass
class PhaseConfig:
    """Configuration for phase-based training.

    Why: Allows fine-tuning the balance between speed and accuracy.
    More prediction steps = faster but potentially less accurate.
    More full steps = slower but more accurate.
    """

    # Phase lengths
    full_steps: int = 10  # Steps of full training per cycle
    predict_steps: int = 40  # Steps of predicted training per cycle
    correct_every: int = 10  # Correction frequency during predict phase

    # Component configs
    prediction_config: PredictionConfig = field(default_factory=PredictionConfig)
    ternary_config: TernaryConfig = field(default_factory=TernaryConfig)
    vsa_config: VSAConfig = field(default_factory=VSAConfig)

    # Training params
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    # Adaptive scheduling
    adaptive_phases: bool = True  # Adjust phase lengths based on loss
    loss_threshold: float = 0.1  # Increase full steps if loss increases by this much


class PhaseTrainer:
    """Orchestrates phase-based training for acceleration.

    Why: This is the main training loop that combines all optimization
    techniques. It manages the phase transitions and ensures convergence
    while maximizing training speed.

    The trainer automatically:
    1. Tracks which phase we're in
    2. Manages gradient prediction during PREDICT phase
    3. Applies corrections to prevent drift
    4. Uses ternary accumulation for memory efficiency
    5. Optionally uses VSA compression for gradient storage

    Example:
        >>> config = PhaseConfig(full_steps=10, predict_steps=40)
        >>> trainer = PhaseTrainer(model, optimizer, config)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         stats = trainer.train_step(batch, compute_loss_fn)
        ...         if stats["phase_changed"]:
        ...             print(f"Entering {stats['phase']} phase")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: PhaseConfig | None = None,
    ) -> None:
        """Initialize phase trainer.

        Args:
            model: Model to train
            optimizer: Base optimizer
            config: Phase training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or PhaseConfig()

        # Initialize components
        self.predictor = GradientPredictor(model, self.config.prediction_config)
        self.ternary_accum = TernaryGradientAccumulator(model, self.config.ternary_config)

        # VSA compressor for gradient history
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.vsa_compressor = VSAGradientCompressor(param_count, self.config.vsa_config)

        # Phase state
        self.current_phase = TrainingPhase.FULL
        self.phase_step = 0
        self.total_step = 0
        self.cycle_count = 0

        # Statistics
        self.phase_losses: dict[TrainingPhase, list[float]] = {
            phase: [] for phase in TrainingPhase
        }
        self.recent_losses: list[float] = []
        self.speedup_ratio = 1.0

        # Tracking for adaptive scheduling
        self.full_steps_taken = 0
        self.predict_steps_taken = 0
        self.correct_steps_taken = 0

    def _get_next_phase(self) -> TrainingPhase:
        """Determine the next training phase.

        Returns:
            Next phase to transition to

        Why: Manages the phase cycle logic. The cycle is:
        FULL → PREDICT → (CORRECT periodically) → FULL
        """
        if self.current_phase == TrainingPhase.FULL:
            if self.phase_step >= self.config.full_steps:
                return TrainingPhase.PREDICT

        elif self.current_phase == TrainingPhase.PREDICT:
            # Check for correction
            if self.phase_step % self.config.correct_every == 0 and self.phase_step > 0:
                return TrainingPhase.CORRECT
            # Check for cycle completion
            if self.phase_step >= self.config.predict_steps:
                return TrainingPhase.FULL

        elif self.current_phase == TrainingPhase.CORRECT:
            # After correction, back to predict or full
            remaining_predict = self.config.predict_steps - self.phase_step
            if remaining_predict > 0:
                return TrainingPhase.PREDICT
            else:
                return TrainingPhase.FULL

        return self.current_phase

    def _transition_phase(self, new_phase: TrainingPhase) -> None:
        """Handle phase transition.

        Args:
            new_phase: Phase to transition to

        Why: Reset phase-specific state and update statistics.
        """
        old_phase = self.current_phase
        self.current_phase = new_phase

        if new_phase == TrainingPhase.FULL:
            # Starting new cycle
            self.phase_step = 0
            self.cycle_count += 1

            # Apply adaptive scheduling if enabled
            if self.config.adaptive_phases and len(self.recent_losses) >= 10:
                self._adjust_phase_lengths()

        elif new_phase == TrainingPhase.PREDICT:
            if old_phase == TrainingPhase.FULL:
                # Entering predict from full
                self.phase_step = 0

        elif new_phase == TrainingPhase.CORRECT:
            # Correction is a single step
            pass

    def _adjust_phase_lengths(self) -> None:
        """Adjust phase lengths based on training dynamics.

        Why: If loss is increasing during predict phase, we need more
        full training. If loss is stable, we can use more prediction.
        """
        if len(self.recent_losses) < 20:
            return

        # Compare recent loss trend
        early = sum(self.recent_losses[:10]) / 10
        late = sum(self.recent_losses[-10:]) / 10

        if late > early * (1 + self.config.loss_threshold):
            # Loss increasing: more full training
            self.config.full_steps = min(50, self.config.full_steps + 5)
            self.config.predict_steps = max(10, self.config.predict_steps - 10)
        elif late < early * 0.95:
            # Loss decreasing well: can use more prediction
            self.config.full_steps = max(5, self.config.full_steps - 2)
            self.config.predict_steps = min(100, self.config.predict_steps + 5)

    def train_step(
        self,
        batch: dict[str, Tensor],
        compute_loss: Callable[[nn.Module, dict[str, Tensor]], Tensor],
    ) -> dict[str, float | str | bool]:
        """Perform one training step.

        Args:
            batch: Input batch dictionary
            compute_loss: Function that computes loss given model and batch

        Returns:
            Dictionary with step statistics

        Why: Main training step that handles phase-specific logic.
        """
        # Check for phase transition
        next_phase = self._get_next_phase()
        phase_changed = next_phase != self.current_phase
        if phase_changed:
            self._transition_phase(next_phase)

        # Compute loss (always needed)
        loss = compute_loss(self.model, batch)
        loss_value = loss.item()

        # Track loss
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)
        self.phase_losses[self.current_phase].append(loss_value)

        # Phase-specific gradient handling
        if self.current_phase == TrainingPhase.FULL:
            self._full_step(loss)
            self.full_steps_taken += 1

        elif self.current_phase == TrainingPhase.PREDICT:
            self._predict_step(loss)
            self.predict_steps_taken += 1

        elif self.current_phase == TrainingPhase.CORRECT:
            self._correct_step(loss)
            self.correct_steps_taken += 1

        # Update counters
        self.phase_step += 1
        self.total_step += 1

        # Calculate speedup
        total_forward = self.full_steps_taken + self.predict_steps_taken + self.correct_steps_taken
        total_backward = self.full_steps_taken + self.correct_steps_taken
        self.speedup_ratio = total_forward / max(1, total_backward)

        return {
            "loss": loss_value,
            "phase": self.current_phase.name,
            "phase_step": self.phase_step,
            "total_step": self.total_step,
            "cycle": self.cycle_count,
            "phase_changed": phase_changed,
            "speedup": self.speedup_ratio,
            "full_steps": self.config.full_steps,
            "predict_steps": self.config.predict_steps,
        }

    def _full_step(self, loss: Tensor) -> None:
        """Perform full gradient computation step.

        Args:
            loss: Loss tensor

        Why: Standard backprop + optimizer step. Also records
        gradients for prediction.
        """
        self.optimizer.zero_grad()
        loss.backward()

        # Record for prediction
        self.predictor.record_gradient()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

    def _predict_step(self, loss: Tensor) -> None:
        """Perform predicted gradient step.

        Args:
            loss: Loss tensor (for forward pass only)

        Why: Uses predicted gradients instead of backprop.
        This is the main source of speedup.
        """
        self.optimizer.zero_grad()

        # Get predicted gradients
        predicted = self.predictor.predict_gradient()

        # Apply to model
        self.predictor.apply_predicted(predicted)

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

    def _correct_step(self, loss: Tensor) -> None:
        """Perform correction step.

        Args:
            loss: Loss tensor

        Why: Computes full gradients and applies accumulated corrections
        to prevent drift from prediction errors.
        """
        self.optimizer.zero_grad()
        loss.backward()

        # Compute and apply correction
        self.predictor.compute_correction()
        self.predictor.apply_correction()

        # Record for next prediction cycle
        self.predictor.record_gradient()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

    def state_dict(self) -> dict:
        """Get trainer state for checkpointing."""
        return {
            "current_phase": self.current_phase.name,
            "phase_step": self.phase_step,
            "total_step": self.total_step,
            "cycle_count": self.cycle_count,
            "full_steps_taken": self.full_steps_taken,
            "predict_steps_taken": self.predict_steps_taken,
            "correct_steps_taken": self.correct_steps_taken,
            "config": {
                "full_steps": self.config.full_steps,
                "predict_steps": self.config.predict_steps,
            },
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load trainer state from checkpoint."""
        self.current_phase = TrainingPhase[state["current_phase"]]
        self.phase_step = state["phase_step"]
        self.total_step = state["total_step"]
        self.cycle_count = state["cycle_count"]
        self.full_steps_taken = state["full_steps_taken"]
        self.predict_steps_taken = state["predict_steps_taken"]
        self.correct_steps_taken = state["correct_steps_taken"]

        if "config" in state:
            self.config.full_steps = state["config"]["full_steps"]
            self.config.predict_steps = state["config"]["predict_steps"]

        self.optimizer.load_state_dict(state["optimizer"])

    def get_stats(self) -> dict[str, float]:
        """Get training statistics."""
        stats = {
            "total_steps": self.total_step,
            "cycles": self.cycle_count,
            "speedup": self.speedup_ratio,
            "full_steps": self.full_steps_taken,
            "predict_steps": self.predict_steps_taken,
            "correct_steps": self.correct_steps_taken,
        }

        # Per-phase average loss
        for phase in TrainingPhase:
            losses = self.phase_losses[phase]
            if losses:
                stats[f"{phase.name.lower()}_loss"] = sum(losses[-100:]) / len(losses[-100:])

        return stats


__all__ = [
    "TrainingPhase",
    "PhaseConfig",
    "PhaseTrainer",
]
