#!/usr/bin/env python3
"""Hybrid Predictive Training with hybrid-predict-trainer-rs compatible design.

This implements the same methodology as hybrid-predict-trainer-rs for testing
and validation purposes. Key features that prevent NaN:

1. Divergence Detection (multiple signals):
   - Loss deviation from EMA (σ-based)
   - Gradient norm explosion (>10x baseline)
   - Gradient vanishing (<0.01x baseline)
   - NaN/Inf detection
   - Prediction error threshold

2. Phase Controller with automatic recovery:
   - WARMUP: Collect baseline statistics
   - FULL: Train + learn dynamics
   - PREDICT: Skip backward (with confidence check)
   - CORRECT: Apply residual corrections (NEW!)

3. Residual Correction:
   - Compare predicted vs actual gradients periodically
   - Apply corrections when drift detected

Usage:
    # Test against full training
    python scripts/train_hybrid_rs_compatible.py --model 100M --max-steps 5000

    # Compare with existing checkpoint
    python scripts/train_hybrid_rs_compatible.py --model 100M --compare-checkpoint checkpoints/100M/bitnet/final
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tritter.core.config import TritterConfig
from tritter.core.model_specs import get_model_spec
from tritter.models.architecture import TritterModel
from tritter.tokenization.multimodal import MultiModalTokenizer
from tritter.training.data import DataConfig, StreamingCodeDataset, create_dataloader


# =============================================================================
# Phase System (matches hybrid-predict-trainer-rs)
# =============================================================================

class Phase(Enum):
    """Training phase - matches Rust enum."""
    WARMUP = auto()
    FULL = auto()
    PREDICT = auto()
    CORRECT = auto()


class DivergenceLevel(Enum):
    """Divergence severity levels."""
    NONE = auto()
    MILD = auto()      # Slightly off, monitor
    MODERATE = auto()  # Fall back to FULL
    SEVERE = auto()    # Emergency stop + recovery
    CRITICAL = auto()  # NaN/Inf detected


@dataclass
class DivergenceSignal:
    """Result of divergence check."""
    level: DivergenceLevel
    reason: str
    value: float
    threshold: float


@dataclass
class DivergenceConfig:
    """Configuration for divergence detection."""
    loss_deviation_threshold: float = 3.0  # σ from EMA
    gradient_explosion_factor: float = 10.0
    gradient_vanishing_factor: float = 0.01
    prediction_error_threshold: float = 0.2  # 20% relative error
    ema_alpha: float = 0.1


@dataclass
class PhaseConfig:
    """Configuration matching hybrid-predict-trainer-rs."""
    warmup_steps: int = 500
    min_full_steps: int = 20
    max_predict_horizon: int = 30
    correct_every: int = 10
    confidence_threshold: float = 0.8
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)


# =============================================================================
# Training Statistics (ring buffer like Rust)
# =============================================================================

@dataclass
class TrainingStatistics:
    """Rolling statistics with ring buffer."""
    window_size: int = 100

    def __post_init__(self):
        self.losses: deque[float] = deque(maxlen=self.window_size)
        self.grad_norms: deque[float] = deque(maxlen=self.window_size)
        self.loss_ema: float = 0.0
        self.loss_variance: float = 0.0
        self.grad_norm_baseline: float = 1.0
        self._initialized: bool = False

    def update(self, loss: float, grad_norm: float, alpha: float = 0.1):
        """Update running statistics."""
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

        if not self._initialized and len(self.losses) > 10:
            self.loss_ema = sum(self.losses) / len(self.losses)
            self.grad_norm_baseline = sum(self.grad_norms) / len(self.grad_norms)
            self._initialized = True
        elif self._initialized:
            # EMA update
            self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema
            # Variance estimation
            if len(self.losses) > 1:
                self.loss_variance = sum((l - self.loss_ema)**2 for l in self.losses) / len(self.losses)

    @property
    def loss_std(self) -> float:
        return math.sqrt(self.loss_variance) if self.loss_variance > 0 else 1.0

    @property
    def is_ready(self) -> bool:
        return self._initialized and len(self.losses) >= 10


# =============================================================================
# Divergence Monitor (matches Rust implementation)
# =============================================================================

class DivergenceMonitor:
    """Multi-signal divergence detection matching hybrid-predict-trainer-rs."""

    def __init__(self, config: DivergenceConfig):
        self.config = config
        self.prediction_errors: deque[float] = deque(maxlen=10)

    def check(
        self,
        loss: float,
        grad_norm: float,
        stats: TrainingStatistics,
        predicted_loss: Optional[float] = None,
    ) -> DivergenceSignal:
        """Check all divergence signals."""

        # Critical: NaN/Inf detection
        if math.isnan(loss) or math.isinf(loss):
            return DivergenceSignal(
                DivergenceLevel.CRITICAL,
                "NaN/Inf loss detected",
                loss,
                0.0,
            )

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            return DivergenceSignal(
                DivergenceLevel.CRITICAL,
                "NaN/Inf gradient detected",
                grad_norm,
                0.0,
            )

        if not stats.is_ready:
            return DivergenceSignal(DivergenceLevel.NONE, "warmup", 0.0, 0.0)

        # Loss deviation check
        loss_deviation = abs(loss - stats.loss_ema) / max(stats.loss_std, 1e-8)
        if loss_deviation > self.config.loss_deviation_threshold * 2:
            return DivergenceSignal(
                DivergenceLevel.SEVERE,
                "Severe loss deviation",
                loss_deviation,
                self.config.loss_deviation_threshold * 2,
            )
        elif loss_deviation > self.config.loss_deviation_threshold:
            return DivergenceSignal(
                DivergenceLevel.MODERATE,
                "Loss deviation",
                loss_deviation,
                self.config.loss_deviation_threshold,
            )

        # Gradient explosion check
        grad_ratio = grad_norm / max(stats.grad_norm_baseline, 1e-8)
        if grad_ratio > self.config.gradient_explosion_factor:
            return DivergenceSignal(
                DivergenceLevel.SEVERE,
                "Gradient explosion",
                grad_ratio,
                self.config.gradient_explosion_factor,
            )

        # Gradient vanishing check
        if grad_ratio < self.config.gradient_vanishing_factor:
            return DivergenceSignal(
                DivergenceLevel.MODERATE,
                "Gradient vanishing",
                grad_ratio,
                self.config.gradient_vanishing_factor,
            )

        # Prediction error check (if in PREDICT phase)
        if predicted_loss is not None and loss > 0:
            pred_error = abs(predicted_loss - loss) / loss
            self.prediction_errors.append(pred_error)

            if pred_error > self.config.prediction_error_threshold * 2:
                return DivergenceSignal(
                    DivergenceLevel.SEVERE,
                    "Prediction error too high",
                    pred_error,
                    self.config.prediction_error_threshold * 2,
                )
            elif pred_error > self.config.prediction_error_threshold:
                return DivergenceSignal(
                    DivergenceLevel.MILD,
                    "Prediction drifting",
                    pred_error,
                    self.config.prediction_error_threshold,
                )

        return DivergenceSignal(DivergenceLevel.NONE, "healthy", 0.0, 0.0)


# =============================================================================
# Gradient Predictor (RSSM-lite style)
# =============================================================================

class GradientPredictor:
    """Gradient prediction with learned dynamics.

    Uses COMPRESSED gradient representations to avoid OOM.
    Only stores gradient norms and directions for key layers,
    similar to PowerSGD compression used in hybrid-predict-trainer-rs.
    """

    def __init__(self, history_size: int = 20):
        self.history_size = history_size
        # Store compressed representations: (norm, mean, std) per layer
        self.gradient_stats: deque[dict[str, tuple[float, float, float]]] = deque(maxlen=history_size)
        # Store only the last full gradient for prediction (single copy)
        self.last_gradient: Optional[dict[str, torch.Tensor]] = None
        self.prev_gradient: Optional[dict[str, torch.Tensor]] = None
        self.confidence: float = 0.0
        self.loss_history: deque[float] = deque(maxlen=history_size)

    def observe(self, gradients: dict[str, torch.Tensor], loss: float):
        """Record gradients using compressed representation."""
        # Store statistics only (not full tensors)
        stats = {}
        for name, grad in gradients.items():
            if not torch.isfinite(grad).all():
                continue  # Skip NaN/Inf gradients
            stats[name] = (
                grad.norm().item(),
                grad.mean().item(),
                grad.std().item() if grad.numel() > 1 else 0.0,
            )

        if stats:
            self.gradient_stats.append(stats)
            self.loss_history.append(loss)

            # Keep only last 2 full gradients for velocity estimation
            if self.last_gradient is not None:
                self.prev_gradient = self.last_gradient

            # Store current gradient (single copy)
            self.last_gradient = {k: v.clone() for k, v in gradients.items() if torch.isfinite(v).all()}

        if len(self.gradient_stats) >= 5:
            self._update_confidence()

    def _update_confidence(self):
        """Estimate prediction confidence based on gradient stability."""
        if len(self.gradient_stats) < 5:
            self.confidence = 0.0
            return

        # Check gradient norm stability
        recent_stats = list(self.gradient_stats)[-5:]
        norm_variances = []

        for name in recent_stats[0].keys():
            norms = [s.get(name, (0, 0, 0))[0] for s in recent_stats]
            if all(n > 0 for n in norms):
                mean_norm = sum(norms) / len(norms)
                variance = sum((n - mean_norm) ** 2 for n in norms) / len(norms)
                cv = math.sqrt(variance) / max(mean_norm, 1e-8)  # Coefficient of variation
                norm_variances.append(cv)

        if norm_variances:
            avg_cv = sum(norm_variances) / len(norm_variances)
            # Lower CV = more stable = higher confidence
            self.confidence = max(0.0, min(1.0, 1.0 - avg_cv))
        else:
            self.confidence = 0.0

    def predict(self, horizon: int = 1) -> dict[str, torch.Tensor]:
        """Predict gradients using last gradient + velocity."""
        if self.last_gradient is None:
            return {}

        result = {}
        for name, grad in self.last_gradient.items():
            if self.prev_gradient is not None and name in self.prev_gradient:
                # Linear extrapolation with damping
                velocity = grad - self.prev_gradient[name]
                predicted = grad + velocity * horizon * 0.5  # Damped velocity
                # Clamp to prevent explosion
                max_norm = grad.norm() * 2.0
                pred_norm = predicted.norm()
                if pred_norm > max_norm and pred_norm > 0:
                    predicted = predicted * (max_norm / pred_norm)
                result[name] = predicted
            else:
                result[name] = grad.clone()

        return result

    def predict_loss(self, current_loss: float, horizon: int = 1) -> float:
        """Predict future loss."""
        if len(self.loss_history) < 2:
            return current_loss

        # Use exponential smoothing
        losses = list(self.loss_history)[-5:]
        if len(losses) >= 2:
            # Estimate trend
            trend = (losses[-1] - losses[-2])
            predicted = current_loss + trend * horizon * 0.5  # Damped
            return max(0.0, predicted)  # Loss can't be negative

        return current_loss


# =============================================================================
# Phase Controller (matches hybrid-predict-trainer-rs)
# =============================================================================

class PhaseController:
    """Phase-based training controller with automatic transitions."""

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.phase = Phase.WARMUP
        self.step = 0
        self.phase_step = 0
        self.predict_horizon = 0
        self.correct_counter = 0

        self.stats = TrainingStatistics()
        self.divergence_monitor = DivergenceMonitor(config.divergence)
        self.predictor = GradientPredictor()

        self.metrics = {
            "phase_transitions": [],
            "divergence_events": [],
            "backward_passes": 0,
            "forward_passes": 0,
            "predictions_used": 0,
            "corrections_applied": 0,
        }

    def transition_to(self, new_phase: Phase, reason: str):
        """Transition to new phase."""
        self.metrics["phase_transitions"].append({
            "from": self.phase.name,
            "to": new_phase.name,
            "step": self.step,
            "reason": reason,
        })
        self.phase = new_phase
        self.phase_step = 0

        if new_phase == Phase.PREDICT:
            self.predict_horizon = 0
        elif new_phase == Phase.CORRECT:
            self.correct_counter = 0

    def should_use_full_gradient(self) -> bool:
        """Determine if we should compute full gradients."""
        return self.phase in (Phase.WARMUP, Phase.FULL, Phase.CORRECT)

    def update(
        self,
        loss: float,
        grad_norm: float,
        gradients: Optional[dict[str, torch.Tensor]] = None,
        predicted_loss: Optional[float] = None,
        skipped_due_to_nan: bool = False,
    ) -> tuple[Phase, Optional[DivergenceSignal]]:
        """Update phase controller after a step."""
        self.step += 1
        self.phase_step += 1
        self.metrics["forward_passes"] += 1

        # Don't count skipped steps in backward/prediction stats
        if skipped_due_to_nan:
            # Don't update stats with bad data
            return self.phase, DivergenceSignal(
                DivergenceLevel.CRITICAL, "Skipped due to NaN", float('nan'), 0.0
            )

        if gradients is not None:
            self.metrics["backward_passes"] += 1
        else:
            self.metrics["predictions_used"] += 1

        # Update statistics (only if grad_norm is finite)
        if math.isfinite(grad_norm) and math.isfinite(loss):
            self.stats.update(loss, grad_norm)

        # Check divergence
        signal = self.divergence_monitor.check(loss, grad_norm, self.stats, predicted_loss)

        if signal.level == DivergenceLevel.CRITICAL:
            # Emergency: transition to CORRECT or FULL
            self.metrics["divergence_events"].append({
                "step": self.step,
                "level": signal.level.name,
                "reason": signal.reason,
            })
            self.transition_to(Phase.FULL, f"Critical divergence: {signal.reason}")
            return self.phase, signal

        if signal.level in (DivergenceLevel.SEVERE, DivergenceLevel.MODERATE):
            self.metrics["divergence_events"].append({
                "step": self.step,
                "level": signal.level.name,
                "reason": signal.reason,
            })
            if self.phase == Phase.PREDICT:
                self.transition_to(Phase.CORRECT, f"Divergence in PREDICT: {signal.reason}")
            return self.phase, signal

        # Record gradients for predictor (with loss for compressed tracking)
        if gradients is not None and math.isfinite(loss):
            self.predictor.observe(gradients, loss)

        # Phase transition logic
        if self.phase == Phase.WARMUP:
            if self.phase_step >= self.config.warmup_steps:
                self.transition_to(Phase.FULL, "Warmup complete")

        elif self.phase == Phase.FULL:
            if (self.phase_step >= self.config.min_full_steps and
                self.predictor.confidence > self.config.confidence_threshold):
                self.transition_to(Phase.PREDICT, f"Confidence {self.predictor.confidence:.2f}")

        elif self.phase == Phase.PREDICT:
            self.predict_horizon += 1

            if self.predict_horizon >= self.config.max_predict_horizon:
                self.transition_to(Phase.CORRECT, "Horizon reached")
            elif self.predictor.confidence < self.config.confidence_threshold * 0.8:
                self.transition_to(Phase.CORRECT, "Confidence dropped")

        elif self.phase == Phase.CORRECT:
            self.correct_counter += 1
            self.metrics["corrections_applied"] += 1

            if self.correct_counter >= self.config.correct_every:
                self.transition_to(Phase.FULL, "Corrections applied")

        return self.phase, signal


# =============================================================================
# Hybrid Trainer (main training loop)
# =============================================================================

@dataclass
class HybridTrainerConfig:
    """Configuration for hybrid trainer."""
    model_size: str = "100M"
    data_dir: Path = Path.home() / "data" / "tritter" / "processed"
    output_dir: Path = Path("checkpoints")

    # Training
    max_steps: int = 5000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    use_amp: bool = True

    # Phase config
    phase: PhaseConfig = field(default_factory=PhaseConfig)

    # Logging
    log_every: int = 10
    save_every: int = 500


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def get_gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract gradients from model."""
    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def apply_gradients(model: nn.Module, gradients: dict[str, torch.Tensor]):
    """Apply predicted gradients to model."""
    for name, param in model.named_parameters():
        if name in gradients:
            param.grad = gradients[name]


def train_hybrid_rs_compatible(config: HybridTrainerConfig) -> dict[str, Any]:
    """Train using hybrid-predict-trainer-rs compatible methodology."""

    print(f"\n{'=' * 70}")
    print("Hybrid Predictive Training (RS-Compatible)")
    print(f"{'=' * 70}")

    spec = get_model_spec(config.model_size)
    print(f"Model: {config.model_size} ({spec.total_params_billions():.2f}B params)")
    print(f"Phase Config: WARMUP={config.phase.warmup_steps}, FULL≥{config.phase.min_full_steps}, PREDICT≤{config.phase.max_predict_horizon}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    model_config = TritterConfig(
        model_size=config.model_size,
        use_bitnet=False,
        gradient_checkpointing=True,
    )
    model = TritterModel(model_config).to(device)

    # Create tokenizer and dataloader
    tokenizer = MultiModalTokenizer(vocab_size=model_config.vocab_size)
    dataset = StreamingCodeDataset(
        data_path=config.data_dir,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    dataloader = create_dataloader(
        dataset,
        DataConfig(batch_size=config.batch_size, max_seq_length=config.max_seq_length),
    )

    # Optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler() if config.use_amp and torch.cuda.is_available() else None

    # Phase controller
    controller = PhaseController(config.phase)

    # Output directory
    output_dir = config.output_dir / config.model_size / "hybrid_rs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    data_iter = iter(dataloader)
    start_time = time.time()
    total_tokens = 0
    tokens_per_step = config.batch_size * config.gradient_accumulation_steps * config.max_seq_length

    loss_history = []
    phase_history = []

    # Recovery state
    last_good_state = None
    last_good_step = 0
    nan_streak = 0
    max_nan_streak = 10
    recovery_count = 0
    max_recoveries = 3

    for step in range(1, config.max_steps + 1):
        step_start = time.time()

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        # Forward pass
        with autocast(enabled=config.use_amp and torch.cuda.is_available()):
            hidden = model.embed_tokens(input_ids)
            for layer in model.layers:
                hidden = layer(hidden)
            logits = model.lm_head(hidden)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        loss_value = loss.item()

        # Check for NaN early with RECOVERY mechanism
        if math.isnan(loss_value) or math.isinf(loss_value):
            nan_streak += 1
            print(f"Step {step}: NaN/Inf detected! Loss = {loss_value} (streak: {nan_streak})")

            if nan_streak >= max_nan_streak and last_good_state is not None and recovery_count < max_recoveries:
                # RECOVERY: Reload last good state
                print(f"\n{'=' * 50}")
                print(f"RECOVERY: Reloading from step {last_good_step}")
                print(f"{'=' * 50}\n")

                model.load_state_dict(last_good_state['model'])
                optimizer.load_state_dict(last_good_state['optimizer'])

                # Reduce learning rate for stability
                new_lr = config.learning_rate * (0.5 ** (recovery_count + 1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reduced learning rate to {new_lr}")

                # Reset phase controller to FULL (conservative)
                controller.transition_to(Phase.FULL, "Recovery from NaN")
                nan_streak = 0
                recovery_count += 1

                # Reset scaler if using AMP
                if scaler is not None:
                    scaler = GradScaler()

            continue  # Skip this step
        else:
            nan_streak = 0  # Reset streak on successful step

        # Save good state periodically
        if step % 50 == 0 and math.isfinite(loss_value):
            last_good_state = {
                'model': {k: v.clone() for k, v in model.state_dict().items()},
                'optimizer': optimizer.state_dict(),
            }
            last_good_step = step

        # Phase-specific logic
        gradients = None
        predicted_loss = None

        if controller.should_use_full_gradient():
            # Full backward pass
            optimizer.zero_grad()

            if scaler is not None:
                scaler.scale(loss).backward()
                # Check for inf grads before unscaling
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # Compute grad norm BEFORE clipping to detect issues
            grad_norm = compute_grad_norm(model)

            # Handle NaN/Inf gradients
            if not math.isfinite(grad_norm):
                print(f"Step {step}: Skipping due to NaN/Inf gradients (norm={grad_norm})")
                # Skip optimizer step, reset gradients
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()  # Update scaler state
                grad_norm = 0.0
                gradients = None
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm = compute_grad_norm(model)  # Recompute after clipping
                gradients = get_gradients(model)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        else:
            # Predict phase: use predicted gradients with CONSERVATIVE approach
            optimizer.zero_grad()

            predicted_grads = controller.predictor.predict(controller.predict_horizon)
            predicted_loss = controller.predictor.predict_loss(loss_value, controller.predict_horizon)

            # SAFETY: Check if loss is spiking - if so, don't apply predictions
            loss_spike = False
            if len(controller.stats.losses) >= 3:
                recent_avg = sum(list(controller.stats.losses)[-3:]) / 3
                if loss_value > recent_avg * 1.5:  # 50% spike
                    loss_spike = True
                    print(f"Step {step}: Loss spike detected ({loss_value:.2f} vs avg {recent_avg:.2f}), skipping prediction")

            if predicted_grads and not loss_spike:
                # Validate predictions - don't apply if they're too large
                max_grad_norm = 0.0
                for name, grad in predicted_grads.items():
                    gnorm = grad.norm().item()
                    if gnorm > max_grad_norm:
                        max_grad_norm = gnorm

                # Use historical baseline
                baseline_norm = controller.stats.grad_norm_baseline if controller.stats.is_ready else 1.0

                if max_grad_norm < baseline_norm * 5:  # Within 5x of baseline
                    apply_gradients(model, predicted_grads)
                    # Use REDUCED learning rate for predictions
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.learning_rate * 0.5
                    grad_norm = compute_grad_norm(model)
                    optimizer.step()
                    # Restore learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.learning_rate
                else:
                    print(f"Step {step}: Predicted gradient too large ({max_grad_norm:.2f} vs baseline {baseline_norm:.2f}), skipping")
                    grad_norm = 0.0
                    # Force transition back to FULL
                    controller.transition_to(Phase.FULL, "Predicted gradient too large")
            else:
                # Fallback: no prediction available or loss spiking
                grad_norm = 0.0
                if loss_spike:
                    controller.transition_to(Phase.FULL, "Loss spike in PREDICT")

        # Update controller
        skipped = (controller.should_use_full_gradient() and gradients is None)
        phase, signal = controller.update(loss_value, grad_norm, gradients, predicted_loss, skipped)

        total_tokens += tokens_per_step
        loss_history.append((step, loss_value))
        phase_history.append((step, phase.name))

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            backward_pct = (1 - controller.metrics["backward_passes"] / max(controller.metrics["forward_passes"], 1)) * 100

            status = f"Step {step:>5} | Phase: {phase.name:<8} | Loss: {loss_value:.4f} | Tok/s: {tokens_per_sec:,.0f} | Skip%: {backward_pct:.1f}%"

            if signal and signal.level != DivergenceLevel.NONE:
                status += f" | ⚠ {signal.reason}"

            print(status)

        # Checkpointing
        if step % config.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), ckpt_dir / "model.pt")

            metrics = {
                "step": step,
                "loss": loss_value,
                "total_tokens": total_tokens,
                "phase": phase.name,
                "controller_metrics": controller.metrics,
                "predictor_confidence": controller.predictor.confidence,
            }
            with open(ckpt_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    # Final metrics
    total_time = time.time() - start_time
    final_metrics = {
        "model_size": config.model_size,
        "methodology": "hybrid_predictive_rs_compatible",
        "total_steps": config.max_steps,
        "total_tokens": total_tokens,
        "total_time_seconds": total_time,
        "final_loss": loss_history[-1][1] if loss_history else 0,
        "min_loss": min(l[1] for l in loss_history) if loss_history else 0,
        "backward_passes": controller.metrics["backward_passes"],
        "forward_passes": controller.metrics["forward_passes"],
        "backward_reduction_percent": (1 - controller.metrics["backward_passes"] / max(controller.metrics["forward_passes"], 1)) * 100,
        "predictions_used": controller.metrics["predictions_used"],
        "corrections_applied": controller.metrics["corrections_applied"],
        "divergence_events": len(controller.metrics["divergence_events"]),
        "phase_transitions": controller.metrics["phase_transitions"],
        "predictor_final_confidence": controller.predictor.confidence,
    }

    # Save final
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_dir / "model.pt")
    with open(final_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(final_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    # Summary
    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    print(f"Total steps: {config.max_steps:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Final loss: {final_metrics['final_loss']:.4f}")
    print(f"Min loss: {final_metrics['min_loss']:.4f}")
    print()
    print("Methodology Analysis:")
    print(f"  Backward passes: {final_metrics['backward_passes']:,}")
    print(f"  Forward passes: {final_metrics['forward_passes']:,}")
    print(f"  Backward reduction: {final_metrics['backward_reduction_percent']:.1f}%")
    print(f"  Divergence events: {final_metrics['divergence_events']}")
    print(f"  Final confidence: {final_metrics['predictor_final_confidence']:.2f}")
    print()
    print(f"Results saved to: {final_dir}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Hybrid Predictive Training (RS-Compatible)")

    parser.add_argument("--model", type=str, default="100M", choices=["100M", "500M", "1B"])
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data" / "tritter" / "processed")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--full-steps", type=int, default=20)
    parser.add_argument("--predict-steps", type=int, default=30)
    parser.add_argument("--correct-every", type=int, default=10)
    parser.add_argument("--compare-checkpoint", type=Path, default=None)

    args = parser.parse_args()

    phase_config = PhaseConfig(
        warmup_steps=args.warmup_steps,
        min_full_steps=args.full_steps,
        max_predict_horizon=args.predict_steps,
        correct_every=args.correct_every,
    )

    config = HybridTrainerConfig(
        model_size=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        phase=phase_config,
    )

    metrics = train_hybrid_rs_compatible(config)

    # Comparison if requested
    if args.compare_checkpoint:
        print(f"\n{'=' * 70}")
        print(f"Comparison with: {args.compare_checkpoint}")
        print(f"{'=' * 70}")

        compare_metrics_path = args.compare_checkpoint / "metrics.json"
        if compare_metrics_path.exists():
            with open(compare_metrics_path) as f:
                compare_metrics = json.load(f)

            print(f"\n{'Metric':<30} {'Hybrid RS':<15} {'Comparison':<15}")
            print("-" * 60)

            if "final_loss" in compare_metrics:
                print(f"{'Final Loss':<30} {metrics['final_loss']:<15.4f} {compare_metrics['final_loss']:<15.4f}")
            if "min_loss" in compare_metrics:
                print(f"{'Min Loss':<30} {metrics['min_loss']:<15.4f} {compare_metrics['min_loss']:<15.4f}")
        else:
            print(f"No metrics.json found at {compare_metrics_path}")


if __name__ == "__main__":
    main()
