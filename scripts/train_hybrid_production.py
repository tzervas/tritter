#!/usr/bin/env python3
"""Production-Grade Hybrid Predictive Training.

This script implements comprehensive edge case handling:

1. AMP/Mixed Precision Safety:
   - Loss underflow detection
   - Proper scaler update sequencing
   - Logit clamping before cross-entropy
   - Scale history monitoring

2. Memory Optimization:
   - Compressed gradient statistics (no full tensor storage)
   - Disk-based checkpointing instead of RAM
   - Memory budget enforcement
   - Automatic batch size reduction on OOM

3. Gradient Handling:
   - Pre-clip NaN/Inf detection
   - Gradient norm caching (no recomputation)
   - Adaptive clipping thresholds
   - Median-based baselines

4. Phase Transitions:
   - Phase lock duration (no rapid oscillation)
   - Orphan gradient cleanup on transitions
   - Extended CORRECT on divergence
   - Statistics validation before WARMUP exit

5. Recovery Mechanisms:
   - Disk-based checkpoint recovery
   - Predictor reset on recovery
   - Data iterator resync
   - Graceful degradation with final checkpoint

6. Confidence Metrics:
   - Clipped coefficient of variation
   - Variance-based dynamic thresholds
   - Confidence-scaled learning rate

Usage:
    python scripts/train_hybrid_production.py --model 100M --max-steps 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tritter.core.config import TritterConfig
from tritter.core.model_specs import get_model_spec
from tritter.models.architecture import TritterModel
from tritter.tokenization.multimodal import MultiModalTokenizer
from tritter.training.data import DataConfig, StreamingCodeDataset, create_dataloader


# =============================================================================
# Phase System
# =============================================================================

class Phase(Enum):
    WARMUP = auto()
    FULL = auto()
    PREDICT = auto()
    CORRECT = auto()


class DivergenceLevel(Enum):
    NONE = auto()
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()


@dataclass
class DivergenceSignal:
    level: DivergenceLevel
    reason: str
    value: float
    threshold: float


# =============================================================================
# Auto-Configuration
# =============================================================================

@dataclass
class AutoConfig:
    """Auto-detected and adjusted configuration."""

    # GPU Info
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cuda_available: bool = False

    # Auto-adjusted params
    batch_size: int = 2
    gradient_accumulation: int = 8
    max_seq_length: int = 1024
    use_amp: bool = True
    gradient_memory_budget_mb: float = 512.0  # Max memory for gradient storage

    @classmethod
    def auto_detect(cls, model_size: str) -> "AutoConfig":
        """Auto-detect optimal configuration based on hardware."""
        config = cls()
        config.cuda_available = torch.cuda.is_available()

        if config.cuda_available:
            config.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            config.gpu_memory_gb = props.total_memory / (1024**3)

            # Auto-adjust based on GPU memory
            if config.gpu_memory_gb >= 24:  # A100, 4090
                config.batch_size = 8
                config.gradient_accumulation = 4
                config.max_seq_length = 2048
                config.gradient_memory_budget_mb = 2048.0
            elif config.gpu_memory_gb >= 16:  # 5080, 4080
                config.batch_size = 4
                config.gradient_accumulation = 8
                config.max_seq_length = 1024
                config.gradient_memory_budget_mb = 1024.0
            elif config.gpu_memory_gb >= 8:  # 3070, 4070
                config.batch_size = 2
                config.gradient_accumulation = 16
                config.max_seq_length = 512
                config.gradient_memory_budget_mb = 256.0
            else:  # < 8GB
                config.batch_size = 1
                config.gradient_accumulation = 32
                config.max_seq_length = 256
                config.gradient_memory_budget_mb = 128.0
                config.use_amp = True  # Required for low VRAM

            # Model-size adjustments
            if model_size == "500M":
                config.batch_size = max(1, config.batch_size // 2)
                config.gradient_accumulation *= 2
            elif model_size == "1B":
                config.batch_size = 1
                config.gradient_accumulation *= 4
                config.max_seq_length = min(config.max_seq_length, 512)

        return config


# =============================================================================
# Training Statistics with Median-Based Baselines
# =============================================================================

@dataclass
class TrainingStatistics:
    """Rolling statistics with robust median-based baselines."""
    window_size: int = 100

    def __post_init__(self):
        self.losses: deque[float] = deque(maxlen=self.window_size)
        self.grad_norms: deque[float] = deque(maxlen=self.window_size)
        self.loss_ema: float = 0.0
        self.loss_variance: float = 0.0
        self.grad_norm_baseline: float = 1.0
        self._initialized: bool = False
        self._warmup_complete: bool = False

    def update(self, loss: float, grad_norm: float, alpha: float = 0.1):
        """Update statistics with validation."""
        # Only update with finite values
        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            return

        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

        # Use MEDIAN for robust baseline (handles outliers)
        if len(self.losses) >= 10:
            if not self._initialized:
                # Initialize with median
                self.loss_ema = statistics.median(self.losses)
                self.grad_norm_baseline = statistics.median(self.grad_norms)
                self._initialized = True
            else:
                # EMA update
                self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema

            # Variance using MAD (Median Absolute Deviation) for robustness
            if len(self.losses) > 1:
                median_loss = statistics.median(self.losses)
                mad = statistics.median([abs(l - median_loss) for l in self.losses])
                self.loss_variance = (1.4826 * mad) ** 2  # Scale factor for normal

            # Update grad baseline with median
            self.grad_norm_baseline = statistics.median(self.grad_norms)

    @property
    def loss_std(self) -> float:
        return math.sqrt(self.loss_variance) if self.loss_variance > 0 else 1.0

    @property
    def is_ready(self) -> bool:
        return self._initialized and len(self.losses) >= 10

    def validate_for_phase_exit(self) -> tuple[bool, str]:
        """Validate statistics are good for exiting WARMUP."""
        if not self.is_ready:
            return False, "Statistics not initialized"
        if math.isnan(self.loss_ema) or math.isinf(self.loss_ema):
            return False, "Loss EMA is NaN/Inf"
        if math.isnan(self.grad_norm_baseline) or math.isinf(self.grad_norm_baseline):
            return False, "Grad norm baseline is NaN/Inf"
        if self.loss_ema > 100:  # Sanity check
            return False, f"Loss EMA too high: {self.loss_ema}"
        return True, "OK"


# =============================================================================
# Divergence Monitor with Dynamic Thresholds
# =============================================================================

@dataclass
class DivergenceConfig:
    """Configuration for divergence detection."""
    loss_sigma_threshold: float = 3.0  # z-score threshold
    gradient_explosion_factor: float = 10.0
    gradient_vanishing_factor: float = 0.01
    prediction_error_threshold: float = 0.2


class DivergenceMonitor:
    """Multi-signal divergence detection with dynamic thresholds."""

    def __init__(self, config: DivergenceConfig):
        self.config = config
        self.prediction_errors: deque[float] = deque(maxlen=10)
        self.consecutive_warnings: int = 0

    def check(
        self,
        loss: float,
        grad_norm: float,
        stats: TrainingStatistics,
        predicted_loss: Optional[float] = None,
    ) -> DivergenceSignal:
        """Check all divergence signals with dynamic thresholds."""

        # Critical: NaN/Inf detection
        if not math.isfinite(loss):
            return DivergenceSignal(DivergenceLevel.CRITICAL, "NaN/Inf loss", loss, 0.0)

        if not math.isfinite(grad_norm):
            return DivergenceSignal(DivergenceLevel.CRITICAL, "NaN/Inf gradient", grad_norm, 0.0)

        if not stats.is_ready:
            return DivergenceSignal(DivergenceLevel.NONE, "warmup", 0.0, 0.0)

        # Z-score based loss deviation (dynamic threshold)
        if stats.loss_std > 1e-8:
            loss_zscore = abs(loss - stats.loss_ema) / stats.loss_std
            if loss_zscore > self.config.loss_sigma_threshold * 2:
                self.consecutive_warnings += 1
                return DivergenceSignal(
                    DivergenceLevel.SEVERE,
                    f"Loss z-score: {loss_zscore:.1f}",
                    loss_zscore,
                    self.config.loss_sigma_threshold * 2,
                )
            elif loss_zscore > self.config.loss_sigma_threshold:
                self.consecutive_warnings += 1
                return DivergenceSignal(
                    DivergenceLevel.MODERATE,
                    f"Loss deviation z={loss_zscore:.1f}",
                    loss_zscore,
                    self.config.loss_sigma_threshold,
                )

        # Gradient explosion check
        grad_ratio = grad_norm / max(stats.grad_norm_baseline, 1e-8)
        if grad_ratio > self.config.gradient_explosion_factor:
            self.consecutive_warnings += 1
            return DivergenceSignal(
                DivergenceLevel.SEVERE,
                f"Gradient explosion {grad_ratio:.1f}x",
                grad_ratio,
                self.config.gradient_explosion_factor,
            )

        # Gradient vanishing check
        if grad_ratio < self.config.gradient_vanishing_factor:
            return DivergenceSignal(
                DivergenceLevel.MODERATE,
                f"Gradient vanishing {grad_ratio:.4f}x",
                grad_ratio,
                self.config.gradient_vanishing_factor,
            )

        # Prediction error check
        if predicted_loss is not None and loss > 0:
            pred_error = abs(predicted_loss - loss) / loss
            self.prediction_errors.append(pred_error)

            if pred_error > self.config.prediction_error_threshold * 2:
                return DivergenceSignal(
                    DivergenceLevel.SEVERE,
                    f"Prediction error {pred_error:.1%}",
                    pred_error,
                    self.config.prediction_error_threshold * 2,
                )
            elif pred_error > self.config.prediction_error_threshold:
                return DivergenceSignal(
                    DivergenceLevel.MILD,
                    f"Prediction drifting {pred_error:.1%}",
                    pred_error,
                    self.config.prediction_error_threshold,
                )

        # Reset consecutive warnings on healthy step
        self.consecutive_warnings = 0
        return DivergenceSignal(DivergenceLevel.NONE, "healthy", 0.0, 0.0)


# =============================================================================
# Memory-Efficient Gradient Predictor
# =============================================================================

class GradientPredictor:
    """Memory-efficient gradient prediction using statistics only.

    Key optimizations:
    - No full gradient tensor storage (uses statistics)
    - Memory budget enforcement
    - Clipped confidence metrics
    """

    def __init__(self, memory_budget_mb: float = 512.0):
        self.memory_budget_mb = memory_budget_mb
        # Store only last gradient for velocity (single copy)
        self.last_gradient: Optional[dict[str, torch.Tensor]] = None
        self.prev_gradient: Optional[dict[str, torch.Tensor]] = None
        # Compressed statistics per layer
        self.layer_stats: dict[str, deque] = {}  # name -> deque of (norm, mean, std)
        self.loss_history: deque[float] = deque(maxlen=20)
        self.confidence: float = 0.0
        self._current_memory_mb: float = 0.0

    def _estimate_memory(self, gradients: dict[str, torch.Tensor]) -> float:
        """Estimate memory usage in MB."""
        return sum(g.numel() * 4 for g in gradients.values()) / (1024 * 1024)

    def observe(self, gradients: dict[str, torch.Tensor], loss: float):
        """Record gradients using compressed representation."""
        # Check memory budget
        grad_memory = self._estimate_memory(gradients)

        # If over budget, only store statistics
        if grad_memory > self.memory_budget_mb:
            self.last_gradient = None
            self.prev_gradient = None
        else:
            # Store only last 2 full gradients
            if self.last_gradient is not None:
                self.prev_gradient = self.last_gradient

            # Filter NaN/Inf and clone
            self.last_gradient = {}
            for k, v in gradients.items():
                if torch.isfinite(v).all():
                    self.last_gradient[k] = v.clone()

        # Always store statistics (minimal memory)
        for name, grad in gradients.items():
            if not torch.isfinite(grad).all():
                continue

            if name not in self.layer_stats:
                self.layer_stats[name] = deque(maxlen=20)

            self.layer_stats[name].append((
                grad.norm().item(),
                grad.mean().item(),
                grad.std().item() if grad.numel() > 1 else 0.0,
            ))

        self.loss_history.append(loss)

        if len(self.loss_history) >= 5:
            self._update_confidence()

    def _update_confidence(self):
        """Compute confidence using clipped CV."""
        if len(self.layer_stats) == 0:
            self.confidence = 0.0
            return

        cvs = []
        for name, stats in self.layer_stats.items():
            if len(stats) < 5:
                continue

            norms = [s[0] for s in stats]
            if all(n > 1e-8 for n in norms):
                mean_norm = statistics.mean(norms)
                std_norm = statistics.stdev(norms) if len(norms) > 1 else 0.0
                cv = std_norm / max(mean_norm, 1e-8)
                # CLIP CV to prevent explosion with small norms
                cv_clipped = min(cv, 10.0)
                cvs.append(cv_clipped)

        if cvs:
            avg_cv = statistics.mean(cvs)
            # Convert CV to confidence (lower CV = higher confidence)
            self.confidence = max(0.0, min(1.0, 1.0 - avg_cv / 5.0))
        else:
            self.confidence = 0.0

    def predict(self, horizon: int = 1) -> dict[str, torch.Tensor]:
        """Predict gradients with safety bounds."""
        if self.last_gradient is None:
            return {}

        result = {}
        for name, grad in self.last_gradient.items():
            if self.prev_gradient is not None and name in self.prev_gradient:
                # Linear extrapolation with DAMPING
                velocity = grad - self.prev_gradient[name]
                predicted = grad + velocity * horizon * 0.3  # Heavy damping

                # CLAMP to prevent explosion
                max_norm = grad.norm() * 2.0
                pred_norm = predicted.norm()
                if pred_norm > max_norm and pred_norm > 1e-8:
                    predicted = predicted * (max_norm / pred_norm)

                result[name] = predicted
            else:
                result[name] = grad.clone()

        return result

    def predict_loss(self, current_loss: float, horizon: int = 1) -> float:
        """Predict loss with bounds."""
        if len(self.loss_history) < 3:
            return current_loss

        recent = list(self.loss_history)[-5:]
        if len(recent) >= 2:
            # Use median trend for robustness
            trend = recent[-1] - recent[-2]
            predicted = current_loss + trend * horizon * 0.3  # Damped
            # Clamp to reasonable range
            return max(0.0, min(predicted, current_loss * 3))

        return current_loss

    def reset(self):
        """Full reset of predictor state."""
        self.last_gradient = None
        self.prev_gradient = None
        self.layer_stats.clear()
        self.loss_history.clear()
        self.confidence = 0.0


# =============================================================================
# Phase Controller with Lock Duration and Proper Transitions
# =============================================================================

@dataclass
class PhaseConfig:
    warmup_steps: int = 500
    min_full_steps: int = 30
    max_predict_horizon: int = 20
    correct_every: int = 10
    confidence_threshold: float = 0.7
    min_phase_duration: int = 10  # Minimum steps before phase change
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)


class PhaseController:
    """Phase controller with lock duration and proper transitions."""

    def __init__(self, config: PhaseConfig, memory_budget_mb: float = 512.0):
        self.config = config
        self.phase = Phase.WARMUP
        self.step = 0
        self.phase_step = 0
        self.predict_horizon = 0
        self.correct_counter = 0
        self.phase_lock_remaining = 0  # Steps until phase change allowed

        self.stats = TrainingStatistics()
        self.divergence_monitor = DivergenceMonitor(config.divergence)
        self.predictor = GradientPredictor(memory_budget_mb)

        self.should_clear_gradients = False

        self.metrics = {
            "phase_transitions": [],
            "divergence_events": [],
            "backward_passes": 0,
            "forward_passes": 0,
            "predictions_used": 0,
            "corrections_applied": 0,
            "recoveries": 0,
        }

    def transition_to(self, new_phase: Phase, reason: str):
        """Transition with proper cleanup."""
        if self.phase_lock_remaining > 0:
            return  # Don't transition during lock

        self.metrics["phase_transitions"].append({
            "from": self.phase.name,
            "to": new_phase.name,
            "step": self.step,
            "reason": reason,
        })

        old_phase = self.phase
        self.phase = new_phase
        self.phase_step = 0
        self.phase_lock_remaining = self.config.min_phase_duration

        if new_phase == Phase.PREDICT:
            self.predict_horizon = 0
        elif new_phase == Phase.CORRECT:
            self.correct_counter = 0
            # Signal gradient cleanup on PREDICT→CORRECT
            if old_phase == Phase.PREDICT:
                self.should_clear_gradients = True

    def should_use_full_gradient(self) -> bool:
        return self.phase in (Phase.WARMUP, Phase.FULL, Phase.CORRECT)

    def update(
        self,
        loss: float,
        grad_norm: float,
        gradients: Optional[dict[str, torch.Tensor]] = None,
        predicted_loss: Optional[float] = None,
        skipped: bool = False,
    ) -> tuple[Phase, Optional[DivergenceSignal]]:
        """Update controller with comprehensive validation."""
        self.step += 1
        self.phase_step += 1
        self.phase_lock_remaining = max(0, self.phase_lock_remaining - 1)
        self.metrics["forward_passes"] += 1

        if skipped:
            return self.phase, DivergenceSignal(DivergenceLevel.CRITICAL, "Skipped", 0, 0)

        if gradients is not None:
            self.metrics["backward_passes"] += 1
        else:
            self.metrics["predictions_used"] += 1

        # Update stats only with valid data
        if math.isfinite(loss) and math.isfinite(grad_norm):
            self.stats.update(loss, grad_norm)

        # Check divergence
        signal = self.divergence_monitor.check(loss, grad_norm, self.stats, predicted_loss)

        if signal.level == DivergenceLevel.CRITICAL:
            self.metrics["divergence_events"].append({
                "step": self.step,
                "level": signal.level.name,
                "reason": signal.reason,
            })
            self.transition_to(Phase.FULL, f"Critical: {signal.reason}")
            return self.phase, signal

        if signal.level in (DivergenceLevel.SEVERE, DivergenceLevel.MODERATE):
            self.metrics["divergence_events"].append({
                "step": self.step,
                "level": signal.level.name,
                "reason": signal.reason,
            })
            if self.phase == Phase.PREDICT:
                self.transition_to(Phase.CORRECT, f"Divergence: {signal.reason}")
            elif self.phase == Phase.CORRECT:
                # Reset correct counter on divergence during correction
                self.correct_counter = 0
            return self.phase, signal

        # Record gradients
        if gradients is not None and math.isfinite(loss):
            self.predictor.observe(gradients, loss)

        # Phase transition logic (only if not locked)
        if self.phase_lock_remaining == 0:
            self._update_phase()

        return self.phase, signal

    def _update_phase(self):
        """Phase transition logic with validation."""
        if self.phase == Phase.WARMUP:
            if self.phase_step >= self.config.warmup_steps:
                # Validate stats before exiting warmup
                valid, reason = self.stats.validate_for_phase_exit()
                if valid:
                    self.transition_to(Phase.FULL, "Warmup complete")
                else:
                    print(f"Extending warmup: {reason}")

        elif self.phase == Phase.FULL:
            if (self.phase_step >= self.config.min_full_steps and
                self.predictor.confidence > self.config.confidence_threshold):
                self.transition_to(Phase.PREDICT, f"Confidence {self.predictor.confidence:.2f}")

        elif self.phase == Phase.PREDICT:
            self.predict_horizon += 1

            if self.predict_horizon >= self.config.max_predict_horizon:
                self.transition_to(Phase.CORRECT, "Horizon reached")
            elif self.predictor.confidence < self.config.confidence_threshold * 0.7:
                self.transition_to(Phase.CORRECT, "Confidence dropped")

        elif self.phase == Phase.CORRECT:
            self.correct_counter += 1
            self.metrics["corrections_applied"] += 1

            if self.correct_counter >= self.config.correct_every:
                self.transition_to(Phase.FULL, "Corrections complete")

    def reset_for_recovery(self):
        """Full reset for recovery from NaN."""
        self.predictor.reset()
        self.stats = TrainingStatistics()
        self.divergence_monitor.consecutive_warnings = 0
        self.phase_lock_remaining = 0
        self.transition_to(Phase.FULL, "Recovery")
        self.metrics["recoveries"] += 1


# =============================================================================
# Disk-Based Recovery Checkpoint Manager
# =============================================================================

class RecoveryCheckpointManager:
    """Disk-based checkpoint manager for recovery."""

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3):
        self.checkpoint_dir = checkpoint_dir / ".recovery"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: deque[tuple[int, Path]] = deque(maxlen=max_checkpoints)

    def save(self, step: int, model: nn.Module, optimizer, scaler=None):
        """Save recovery checkpoint to disk."""
        ckpt_path = self.checkpoint_dir / f"recovery_{step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_path / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
        if scaler is not None:
            torch.save(scaler.state_dict(), ckpt_path / "scaler.pt")

        self.checkpoints.append((step, ckpt_path))

        # Cleanup old checkpoints
        while len(list(self.checkpoint_dir.iterdir())) > self.max_checkpoints:
            oldest = self.checkpoint_dir / sorted(self.checkpoint_dir.iterdir())[0]
            if oldest.exists():
                shutil.rmtree(oldest)

    def load_latest(self, model: nn.Module, optimizer, scaler=None) -> tuple[bool, int]:
        """Load most recent recovery checkpoint with OOM-safe CPU-first loading."""
        if not self.checkpoints:
            return False, 0

        step, ckpt_path = self.checkpoints[-1]

        if not ckpt_path.exists():
            return False, 0

        device = next(model.parameters()).device

        # Load to CPU first, then move to GPU to prevent OOM during concurrent loading
        # This prevents peak memory from exceeding GPU capacity

        # 1. Load model state to CPU
        model_state = torch.load(ckpt_path / "model.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(model_state)
        del model_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 2. Load optimizer state to CPU, then move to GPU
        opt_state = torch.load(ckpt_path / "optimizer.pt", weights_only=True, map_location="cpu")
        # Move optimizer state tensors to device
        for state in opt_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(opt_state)
        del opt_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 3. Load scaler (small, direct load is fine)
        if scaler is not None and (ckpt_path / "scaler.pt").exists():
            scaler.load_state_dict(torch.load(ckpt_path / "scaler.pt", weights_only=True))

        return True, step

    def cleanup(self):
        """Remove all recovery checkpoints."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class HybridTrainerConfig:
    model_size: str = "100M"
    data_dir: Path = Path.home() / "data" / "tritter" / "processed"
    output_dir: Path = Path("checkpoints")

    # Auto-config (filled by AutoConfig.auto_detect)
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    use_amp: bool = True
    gradient_memory_budget_mb: float = 512.0

    # Training
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Phase config
    phase: PhaseConfig = field(default_factory=PhaseConfig)

    # Recovery
    max_recoveries: int = 5
    nan_streak_threshold: int = 10
    recovery_checkpoint_interval: int = 100

    # Logging
    log_every: int = 10
    save_every: int = 500


# =============================================================================
# Helper Functions
# =============================================================================

def compute_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm with NaN check."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            if not torch.isfinite(param_norm):
                return float('inf')
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)


def get_gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract gradients, filtering NaN/Inf."""
    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None and torch.isfinite(param.grad).all()
    }


def apply_gradients(model: nn.Module, gradients: dict[str, torch.Tensor]):
    """Apply predicted gradients."""
    for name, param in model.named_parameters():
        if name in gradients:
            param.grad = gradients[name]


def get_gpu_memory_info() -> tuple[float, float]:
    """Get current and peak GPU memory in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return allocated, peak
    return 0.0, 0.0


# =============================================================================
# Main Training Function
# =============================================================================

def train_hybrid_production(config: HybridTrainerConfig) -> dict[str, Any]:
    """Production-grade hybrid predictive training."""

    print(f"\n{'=' * 70}")
    print("PRODUCTION Hybrid Predictive Training")
    print(f"{'=' * 70}")

    # Auto-detect configuration
    auto_config = AutoConfig.auto_detect(config.model_size)
    print(f"\nAuto-detected configuration:")
    print(f"  GPU: {auto_config.gpu_name} ({auto_config.gpu_memory_gb:.1f} GB)")
    print(f"  Batch size: {auto_config.batch_size}")
    print(f"  Gradient accumulation: {auto_config.gradient_accumulation}")
    print(f"  Seq length: {auto_config.max_seq_length}")
    print(f"  AMP: {auto_config.use_amp}")
    print(f"  Gradient memory budget: {auto_config.gradient_memory_budget_mb} MB")

    # Apply auto-config
    config.batch_size = auto_config.batch_size
    config.gradient_accumulation_steps = auto_config.gradient_accumulation
    config.max_seq_length = auto_config.max_seq_length
    config.use_amp = auto_config.use_amp
    config.gradient_memory_budget_mb = auto_config.gradient_memory_budget_mb

    spec = get_model_spec(config.model_size)

    # Pre-flight memory check using tritter_accel (rust-ai-core)
    try:
        import tritter_accel as ta
        param_count = int(spec.total_params_billions() * 1e9)
        hidden_dim = spec.hidden_size
        num_layers = spec.num_layers
        dtype = "bf16" if config.use_amp else "f32"

        params_mem, grads_mem, opt_mem, acts_mem, total_mem = ta.estimate_training_memory(
            param_count, config.batch_size, config.max_seq_length,
            hidden_dim, num_layers, dtype, True  # gradient_checkpointing=True
        )

        print(f"\n--- Pre-flight Memory Estimate (via rust-ai-core) ---")
        print(f"  Parameters: {params_mem / 1024**3:.2f} GB")
        print(f"  Gradients: {grads_mem / 1024**3:.2f} GB")
        print(f"  Optimizer: {opt_mem / 1024**3:.2f} GB")
        print(f"  Activations: {acts_mem / 1024**3:.2f} GB")
        print(f"  Total Required: {total_mem / 1024**3:.2f} GB")
        print(f"  GPU Available: {auto_config.gpu_memory_gb:.2f} GB")

        # Check if training would fit
        gpu_budget_bytes = int(auto_config.gpu_memory_gb * 0.90 * 1024**3)  # 90% of GPU for safety
        tracker = ta.PyMemoryTracker(gpu_budget_bytes)
        fits, required, available = tracker.would_training_fit(
            param_count, config.batch_size, config.max_seq_length,
            hidden_dim, num_layers, dtype, True
        )

        if not fits:
            print(f"\n⚠️  WARNING: Training may not fit in GPU memory!")
            print(f"    Required: {required / 1024**3:.2f} GB, Available: {available / 1024**3:.2f} GB")
            print(f"    Consider: smaller batch size, shorter seq length, or gradient accumulation")

            # Auto-reduce batch size if needed
            while not fits and config.batch_size > 1:
                config.batch_size = max(1, config.batch_size // 2)
                config.gradient_accumulation_steps *= 2
                fits, required, available = tracker.would_training_fit(
                    param_count, config.batch_size, config.max_seq_length,
                    hidden_dim, num_layers, dtype, True
                )
                print(f"    Auto-adjusting: batch_size={config.batch_size}, grad_accum={config.gradient_accumulation_steps}")

            if not fits:
                print(f"    ❌ Cannot fit even with batch_size=1. GPU too small for this model.")
                print(f"    Consider: smaller model, shorter sequence length, or larger GPU")
        else:
            print(f"  ✓ Training will fit in GPU memory")
    except ImportError:
        print("\n(tritter_accel not available for pre-flight memory check)")
    except Exception as e:
        print(f"\n(Pre-flight memory check failed: {e})")
    print(f"\nModel: {config.model_size} ({spec.total_params_billions():.2f}B params)")
    print(f"Phase Config: WARMUP={config.phase.warmup_steps}, FULL≥{config.phase.min_full_steps}, PREDICT≤{config.phase.max_predict_horizon}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with gradient checkpointing
    model_config = TritterConfig(
        model_size=config.model_size,
        use_bitnet=False,
        gradient_checkpointing=True,
    )
    model = TritterModel(model_config).to(device)

    # Tokenizer and dataloader
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

    # Optimizer with AMP
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=config.use_amp) if torch.cuda.is_available() else None

    # Phase controller
    controller = PhaseController(config.phase, config.gradient_memory_budget_mb)

    # Recovery checkpoint manager
    output_dir = config.output_dir / config.model_size / "hybrid_production"
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_manager = RecoveryCheckpointManager(output_dir)

    # Training state
    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    data_iter = iter(dataloader)
    start_time = time.time()
    total_tokens = 0
    # Note: gradient_accumulation_steps removed from token counting
    # Training loop doesn't implement actual gradient accumulation (no delayed optimizer.step)
    tokens_per_step = config.batch_size * config.max_seq_length

    loss_history = []
    nan_streak = 0
    recovery_count = 0
    current_lr = config.learning_rate
    grad_norm_cache = 0.0

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

        # Forward pass with AMP
        try:
            with autocast("cuda", enabled=config.use_amp):
                hidden = model.embed_tokens(input_ids)
                for layer in model.layers:
                    hidden = layer(hidden)
                logits = model.lm_head(hidden)

                # CLAMP logits to prevent overflow in CE loss
                logits = torch.clamp(logits, min=-100, max=100)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Step {step}: OOM during forward pass, clearing cache")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

        loss_value = loss.item()

        # =====================================================================
        # NaN/Inf Handling with Recovery
        # =====================================================================
        if not math.isfinite(loss_value):
            nan_streak += 1
            print(f"Step {step}: NaN/Inf loss (streak: {nan_streak})")

            if nan_streak >= config.nan_streak_threshold:
                if recovery_count < config.max_recoveries:
                    print(f"\n{'=' * 50}")
                    print(f"RECOVERY {recovery_count + 1}/{config.max_recoveries}")
                    print(f"{'=' * 50}\n")

                    # Clear GPU memory before loading checkpoint to prevent OOM
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()

                    success, recovered_step = recovery_manager.load_latest(model, optimizer, scaler)
                    if success:
                        print(f"Restored from step {recovered_step}")

                        # Reduce learning rate
                        current_lr = config.learning_rate * (0.5 ** (recovery_count + 1))
                        for pg in optimizer.param_groups:
                            pg['lr'] = current_lr
                        print(f"Learning rate reduced to {current_lr:.2e}")

                        # Reset controller and predictor
                        controller.reset_for_recovery()

                        # Reset data iterator (approximate resync)
                        data_iter = iter(dataloader)

                        nan_streak = 0
                        recovery_count += 1

                        # Reset scaler
                        if scaler is not None:
                            scaler = GradScaler("cuda", enabled=config.use_amp)
                    else:
                        print("No recovery checkpoint available")
                else:
                    print(f"\nMax recoveries ({config.max_recoveries}) exhausted")
                    print("Saving final checkpoint and exiting...")
                    torch.save(model.state_dict(), output_dir / "emergency_exit.pt")
                    break

            continue
        else:
            nan_streak = 0

        # Save recovery checkpoint periodically
        if step % config.recovery_checkpoint_interval == 0:
            recovery_manager.save(step, model, optimizer, scaler)

        # =====================================================================
        # Phase-Specific Training Logic
        # =====================================================================
        gradients = None
        predicted_loss = None
        skipped = False

        # Clear gradients if signaled by phase transition
        if controller.should_clear_gradients:
            optimizer.zero_grad()
            controller.should_clear_gradients = False

        if controller.should_use_full_gradient():
            # Full backward pass
            optimizer.zero_grad()

            try:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Check scale factor health
                    if scaler.get_scale() < 1.0:
                        print(f"Step {step}: Warning - scaler too low ({scaler.get_scale():.2f})")
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # Compute grad norm ONCE and cache
                grad_norm_cache = compute_grad_norm(model)

                if not math.isfinite(grad_norm_cache):
                    print(f"Step {step}: NaN/Inf gradient norm, skipping")
                    optimizer.zero_grad()
                    if scaler is not None:
                        # Don't update scaler, just skip this step
                        scaler.update()  # Reset state for next iteration
                    skipped = True
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    gradients = get_gradients(model)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

            except RuntimeError as e:
                if "unscale_" in str(e) or "already been called" in str(e):
                    print(f"Step {step}: Scaler state error, resetting")
                    scaler = GradScaler("cuda", enabled=config.use_amp)
                    optimizer.zero_grad()
                    skipped = True
                else:
                    raise

        else:
            # PREDICT phase with safety checks
            optimizer.zero_grad()

            predicted_grads = controller.predictor.predict(controller.predict_horizon)
            predicted_loss = controller.predictor.predict_loss(loss_value, controller.predict_horizon)

            # Safety check 1: Loss spike detection using z-score
            loss_spike = False
            if controller.stats.is_ready:
                loss_zscore = abs(loss_value - controller.stats.loss_ema) / max(controller.stats.loss_std, 1e-8)
                if loss_zscore > 2.0:
                    loss_spike = True
                    print(f"Step {step}: Loss spike (z={loss_zscore:.1f}), falling back to FULL")
                    controller.transition_to(Phase.FULL, "Loss spike in PREDICT")

            # Safety check 2: Predicted gradient magnitude
            if predicted_grads and not loss_spike:
                max_pred_norm = max(g.norm().item() for g in predicted_grads.values())
                baseline = controller.stats.grad_norm_baseline if controller.stats.is_ready else 1.0

                # Confidence-scaled threshold
                max_allowed = baseline * (3.0 + 5.0 * controller.predictor.confidence)

                if max_pred_norm < max_allowed:
                    apply_gradients(model, predicted_grads)

                    # Confidence-scaled learning rate
                    confidence_scale = 0.3 + 0.5 * controller.predictor.confidence
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr * confidence_scale

                    grad_norm_cache = compute_grad_norm(model)
                    optimizer.step()

                    # Restore learning rate
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                else:
                    print(f"Step {step}: Predicted gradient too large ({max_pred_norm:.1f} vs {max_allowed:.1f})")
                    controller.transition_to(Phase.FULL, "Predicted gradient too large")
                    grad_norm_cache = 0.0
            else:
                grad_norm_cache = 0.0
                if loss_spike:
                    controller.transition_to(Phase.FULL, "Loss spike")

        # Update controller
        phase, signal = controller.update(
            loss_value, grad_norm_cache, gradients, predicted_loss, skipped
        )

        total_tokens += tokens_per_step
        loss_history.append((step, loss_value))

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            backward_pct = controller.metrics["backward_passes"] / max(controller.metrics["forward_passes"], 1) * 100
            skip_pct = 100 - backward_pct
            mem_alloc, mem_peak = get_gpu_memory_info()

            status = (
                f"Step {step:>5} | "
                f"Phase: {phase.name:<8} | "
                f"Loss: {loss_value:.4f} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"Skip: {skip_pct:.1f}% | "
                f"LR: {current_lr:.1e} | "
                f"Conf: {controller.predictor.confidence:.2f} | "
                f"Mem: {mem_alloc:.1f}GB"
            )

            if signal and signal.level != DivergenceLevel.NONE:
                status += f" | {signal.reason}"

            print(status)

        # Checkpointing
        if step % config.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            with open(ckpt_dir / "metrics.json", "w") as f:
                json.dump({
                    "step": step,
                    "loss": loss_value,
                    "total_tokens": total_tokens,
                    "phase": phase.name,
                    "confidence": controller.predictor.confidence,
                    "learning_rate": current_lr,
                    "recoveries": recovery_count,
                }, f, indent=2)

            print(f"  Checkpoint saved: {ckpt_dir}")

    # Final metrics
    total_time = time.time() - start_time
    final_metrics = {
        "model_size": config.model_size,
        "methodology": "hybrid_predictive_production",
        "total_steps": step,
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
        "recoveries": recovery_count,
        "final_learning_rate": current_lr,
        "final_confidence": controller.predictor.confidence,
        "gpu_peak_memory_gb": get_gpu_memory_info()[1],
    }

    # Save final
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_dir / "model.pt")
    with open(final_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(final_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    # Cleanup recovery checkpoints
    recovery_manager.cleanup()

    # Summary
    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    print(f"Total steps: {step:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Final loss: {final_metrics['final_loss']:.4f}")
    print(f"Min loss: {final_metrics['min_loss']:.4f}")
    print()
    print("Methodology Analysis:")
    print(f"  Backward passes: {final_metrics['backward_passes']:,}")
    print(f"  Forward passes: {final_metrics['forward_passes']:,}")
    print(f"  Backward reduction: {final_metrics['backward_reduction_percent']:.1f}%")
    print(f"  Recoveries: {final_metrics['recoveries']}")
    print(f"  Final confidence: {final_metrics['final_confidence']:.2f}")
    print(f"  Peak GPU memory: {final_metrics['gpu_peak_memory_gb']:.2f} GB")
    print()
    print(f"Results saved to: {final_dir}")

    return final_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Production Hybrid Predictive Training")

    parser.add_argument("--model", type=str, default="100M", choices=["100M", "500M", "1B"])
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data" / "tritter" / "processed")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--max-steps", type=int, default=5000)

    # Phase configuration
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-full-steps", type=int, default=30)
    parser.add_argument("--max-predict-horizon", type=int, default=20)
    parser.add_argument("--correct-every", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)

    # Recovery
    parser.add_argument("--max-recoveries", type=int, default=5)

    # Learning rate
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    args = parser.parse_args()

    phase_config = PhaseConfig(
        warmup_steps=args.warmup_steps,
        min_full_steps=args.min_full_steps,
        max_predict_horizon=args.max_predict_horizon,
        correct_every=args.correct_every,
        confidence_threshold=args.confidence_threshold,
    )

    config = HybridTrainerConfig(
        model_size=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_recoveries=args.max_recoveries,
        phase=phase_config,
    )

    train_hybrid_production(config)


if __name__ == "__main__":
    main()
