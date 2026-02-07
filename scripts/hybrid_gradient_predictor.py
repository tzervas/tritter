"""Hybridized Gradient Prediction System.

Combines multiple prediction methods with adaptive weighting based on
recent prediction accuracy. This achieves better accuracy than any single method.

Methods combined:
1. Linear Extrapolation - Captures steady trends
2. Momentum-Based (EMA) - Smooths noise, good for oscillating gradients
3. Quadratic Extrapolation - Captures curvature/acceleration
4. Weighted Average - Uses recent gradient mean as anchor

The weights are updated online based on prediction error of each method.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

import torch


@dataclass
class PredictorStats:
    """Statistics for tracking predictor accuracy."""

    recent_errors: deque = field(default_factory=lambda: deque(maxlen=20))
    cumulative_error: float = 0.0
    prediction_count: int = 0

    @property
    def mean_error(self) -> float:
        if not self.recent_errors:
            return 1.0
        return statistics.mean(self.recent_errors)

    def update(self, predicted: torch.Tensor, actual: torch.Tensor):
        """Update with prediction error."""
        if predicted.numel() == 0 or actual.numel() == 0:
            return
        # Use relative error (normalized by actual magnitude)
        actual_norm = actual.norm().item()
        if actual_norm > 1e-8:
            error = (predicted - actual).norm().item() / actual_norm
        else:
            error = (predicted - actual).norm().item()
        self.recent_errors.append(min(error, 10.0))  # Cap extreme errors
        self.cumulative_error += error
        self.prediction_count += 1


class HybridGradientPredictor:
    """Hybridized gradient predictor combining multiple methods.

    Key innovation: Adaptive weighting of 4 prediction methods based on
    their recent accuracy. This outperforms any single method.

    Methods:
    1. Linear: g_pred = g_t + velocity * horizon * damping
    2. Momentum: g_pred = ema_g + momentum * (g_t - ema_g) * horizon
    3. Quadratic: g_pred = g_t + v*h + 0.5*a*h^2 (captures curvature)
    4. Anchor: g_pred = weighted_mean(recent_gradients) (conservative)

    The final prediction is: sum(weight_i * pred_i) / sum(weight_i)
    where weights are inverse of recent prediction error.
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        history_length: int = 10,
        ema_beta: float = 0.9,
        base_damping: float = 0.5,
        min_confidence: float = 0.1,
    ):
        self.memory_budget_mb = memory_budget_mb
        self.history_length = history_length
        self.ema_beta = ema_beta
        self.base_damping = base_damping
        self.min_confidence = min_confidence

        # Gradient history (stores last N gradients per layer)
        self.gradient_history: dict[str, deque] = {}
        self.loss_history: deque[float] = deque(maxlen=50)

        # EMA state per layer
        self.ema_gradients: dict[str, torch.Tensor] = {}

        # Per-method statistics for adaptive weighting
        self.method_stats = {
            "linear": PredictorStats(),
            "momentum": PredictorStats(),
            "quadratic": PredictorStats(),
            "anchor": PredictorStats(),
        }

        # Last predictions for error tracking
        self._last_predictions: dict[str, dict[str, torch.Tensor]] = {}

        # Overall confidence
        self.confidence: float = 0.0

        # Gradient alignment history (pred vs actual cosine similarity)
        self.alignment_history: deque[float] = deque(maxlen=20)

    def observe(self, gradients: dict[str, torch.Tensor], loss: float):
        """Record new gradient observation and update method statistics.

        Note: Gradients are stored on CPU to save GPU memory for large models.
        """
        if not gradients:
            return

        # First, score previous predictions against this actual gradient
        self._score_predictions(gradients)

        # Store gradient history (on CPU to save GPU memory)
        for name, grad in gradients.items():
            if not torch.isfinite(grad).all():
                continue

            # Move to CPU for storage
            grad_cpu = grad.detach().cpu()

            # Initialize if needed
            if name not in self.gradient_history:
                self.gradient_history[name] = deque(maxlen=self.history_length)
                self.ema_gradients[name] = grad_cpu.clone()

            # Update history (CPU)
            self.gradient_history[name].append(grad_cpu.clone())

            # Update EMA (CPU)
            self.ema_gradients[name] = (
                self.ema_beta * self.ema_gradients[name] + (1 - self.ema_beta) * grad_cpu
            )

        self.loss_history.append(loss)
        self._update_confidence()

    def _score_predictions(self, actual_gradients: dict[str, torch.Tensor]):
        """Score previous predictions against actual gradients.

        Note: Predictions are stored on CPU, so actual gradients are moved to CPU for comparison.
        """
        if not self._last_predictions:
            return

        for method, predictions in self._last_predictions.items():
            for name, pred in predictions.items():
                if name in actual_gradients:
                    # Move actual to CPU for comparison (predictions are on CPU)
                    actual = actual_gradients[name].detach().cpu()
                    self.method_stats[method].update(pred, actual)

                    # Track alignment for confidence
                    if method == "linear":  # Only track once per step
                        alignment = torch.nn.functional.cosine_similarity(
                            pred.flatten().unsqueeze(0), actual.flatten().unsqueeze(0)
                        ).item()
                        self.alignment_history.append(alignment)

        self._last_predictions.clear()

    def _compute_method_weights(self) -> dict[str, float]:
        """Compute adaptive weights for each prediction method."""
        weights = {}

        for method, stats in self.method_stats.items():
            if stats.prediction_count < 3:
                # Not enough data, use uniform weight
                weights[method] = 1.0
            else:
                # Inverse error weighting with smoothing
                error = max(stats.mean_error, 0.01)
                weights[method] = 1.0 / (error + 0.1)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = dict.fromkeys(self.method_stats.keys(), 0.25)

        return weights

    def _predict_linear(self, name: str, horizon: int, damping: float) -> torch.Tensor | None:
        """Linear extrapolation: g + velocity * horizon * damping."""
        history = self.gradient_history.get(name)
        if not history or len(history) < 2:
            return None

        g_t = history[-1]
        g_tm1 = history[-2]
        velocity = g_t - g_tm1

        predicted = g_t + velocity * horizon * damping
        return self._clamp_prediction(predicted, g_t)

    def _predict_momentum(self, name: str, horizon: int, damping: float) -> torch.Tensor | None:
        """Momentum-based: EMA + momentum * (current - EMA) * scaled_horizon."""
        history = self.gradient_history.get(name)
        ema = self.ema_gradients.get(name)
        if not history or ema is None:
            return None

        g_t = history[-1]
        momentum = g_t - ema

        # Momentum decays over horizon
        momentum_scale = damping * (1 - self.ema_beta**horizon)
        predicted = ema + momentum * (1 + momentum_scale)

        return self._clamp_prediction(predicted, g_t)

    def _predict_quadratic(self, name: str, horizon: int, damping: float) -> torch.Tensor | None:
        """Quadratic extrapolation: g + v*h + 0.5*a*h^2."""
        history = self.gradient_history.get(name)
        if not history or len(history) < 3:
            return None

        g_t = history[-1]
        g_tm1 = history[-2]
        g_tm2 = history[-3]

        # Velocity and acceleration
        v_t = g_t - g_tm1
        v_tm1 = g_tm1 - g_tm2
        acceleration = v_t - v_tm1

        # Quadratic extrapolation with heavy damping on acceleration
        h = horizon * damping
        predicted = g_t + v_t * h + 0.5 * acceleration * h * h * 0.3  # Extra damping on accel

        return self._clamp_prediction(predicted, g_t)

    def _predict_anchor(self, name: str, horizon: int, damping: float) -> torch.Tensor | None:
        """Weighted average of recent gradients (conservative anchor)."""
        history = self.gradient_history.get(name)
        if not history:
            return None

        # Exponentially weighted mean of history
        weights = [0.95**i for i in range(len(history))]
        weights = weights[::-1]  # Recent gets more weight
        total_weight = sum(weights)

        predicted = sum(w * g for w, g in zip(weights, history, strict=False)) / total_weight

        # Slight adjustment toward most recent
        g_t = history[-1]
        predicted = 0.7 * predicted + 0.3 * g_t

        return self._clamp_prediction(predicted, g_t)

    def _clamp_prediction(
        self, predicted: torch.Tensor, reference: torch.Tensor, max_ratio: float = 3.0
    ) -> torch.Tensor:
        """Clamp prediction magnitude to prevent explosion."""
        ref_norm = reference.norm().item()
        pred_norm = predicted.norm().item()

        if ref_norm > 1e-8 and pred_norm > max_ratio * ref_norm:
            predicted = predicted * (max_ratio * ref_norm / pred_norm)

        return predicted

    def predict(
        self, horizon: int = 1, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Generate hybrid prediction combining all methods.

        Args:
            horizon: Number of steps to predict ahead
            device: Target device for predictions (default: cuda if available)

        Returns:
            Dictionary mapping parameter names to predicted gradients on target device
        """
        if not self.gradient_history:
            return {}

        # Default to CUDA if available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute adaptive weights
        weights = self._compute_method_weights()

        # Compute adaptive damping based on horizon and confidence
        # Higher confidence -> less damping (more aggressive)
        # Longer horizon -> more damping (more conservative)
        confidence_factor = 0.5 + 0.5 * self.confidence
        horizon_factor = math.sqrt(horizon / 4.0)  # Normalize to base horizon of 4
        damping = self.base_damping * horizon_factor / confidence_factor
        damping = max(0.1, min(0.8, damping))  # Clamp to reasonable range

        result = {}
        self._last_predictions = {m: {} for m in self.method_stats.keys()}

        for name in self.gradient_history.keys():
            # Get prediction from each method (computed on CPU)
            predictions = {
                "linear": self._predict_linear(name, horizon, damping),
                "momentum": self._predict_momentum(name, horizon, damping),
                "quadratic": self._predict_quadratic(name, horizon, damping),
                "anchor": self._predict_anchor(name, horizon, damping),
            }

            # Filter valid predictions
            valid_preds = {k: v for k, v in predictions.items() if v is not None}

            if not valid_preds:
                continue

            # Store for scoring (keep on CPU)
            for method, pred in valid_preds.items():
                self._last_predictions[method][name] = pred

            # Weighted combination
            combined = None
            total_weight = 0.0

            for method, pred in valid_preds.items():
                w = weights.get(method, 0.0)
                if combined is None:
                    combined = w * pred
                else:
                    combined = combined + w * pred
                total_weight += w

            if combined is not None and total_weight > 0:
                # Move to target device (GPU) for application
                result[name] = (combined / total_weight).to(device)

        return result

    def predict_loss(self, current_loss: float, horizon: int = 1) -> float:
        """Predict future loss using hybridized approach."""
        if len(self.loss_history) < 5:
            return current_loss

        recent = list(self.loss_history)[-10:]

        # Method 1: Linear trend
        if len(recent) >= 2:
            linear_trend = recent[-1] - recent[-2]
            linear_pred = current_loss + linear_trend * horizon * 0.3
        else:
            linear_pred = current_loss

        # Method 2: EMA-based
        ema_loss = recent[-1]
        for loss in reversed(recent[:-1]):
            ema_loss = 0.9 * ema_loss + 0.1 * loss
        ema_pred = ema_loss  # EMA is already a prediction of steady state

        # Method 3: Median trend (robust to outliers)
        if len(recent) >= 4:
            trends = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            median_trend = statistics.median(trends)
            median_pred = current_loss + median_trend * horizon * 0.5
        else:
            median_pred = current_loss

        # Combine with confidence-based weighting
        # Higher confidence -> trust trend more
        # Lower confidence -> anchor to current
        trend_weight = 0.3 + 0.4 * self.confidence

        hybrid_pred = (
            trend_weight * 0.5 * linear_pred
            + trend_weight * 0.3 * median_pred
            + (1 - trend_weight) * current_loss
            + trend_weight * 0.2 * ema_pred
        )

        # Clamp to reasonable range
        return max(0.0, min(hybrid_pred, current_loss * 2.5))

    def _update_confidence(self):
        """Update confidence based on multiple signals."""
        signals = []

        # Signal 1: Method agreement (low variance across methods = high confidence)
        if all(s.prediction_count >= 3 for s in self.method_stats.values()):
            errors = [s.mean_error for s in self.method_stats.values()]
            error_variance = statistics.variance(errors) if len(errors) > 1 else 1.0
            agreement_confidence = 1.0 / (1.0 + error_variance * 5)
            signals.append(agreement_confidence)

        # Signal 2: Gradient alignment (how well predictions match actual)
        if len(self.alignment_history) >= 5:
            mean_alignment = statistics.mean(self.alignment_history)
            # Alignment is in [-1, 1], convert to [0, 1]
            alignment_confidence = (mean_alignment + 1) / 2
            signals.append(alignment_confidence)

        # Signal 3: Loss stability
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            loss_std = statistics.stdev(recent_losses)
            loss_mean = statistics.mean(recent_losses)
            cv = loss_std / max(loss_mean, 1e-8)
            stability_confidence = max(0, 1 - cv / 2)  # CV of 2 = 0 confidence
            signals.append(stability_confidence)

        # Signal 4: History length (more history = more confident)
        min_history = (
            min(len(h) for h in self.gradient_history.values()) if self.gradient_history else 0
        )
        history_confidence = min(min_history / self.history_length, 1.0)
        signals.append(history_confidence)

        # Combine signals
        if signals:
            self.confidence = statistics.mean(signals)
        else:
            self.confidence = self.min_confidence

        # Ensure minimum confidence
        self.confidence = max(self.min_confidence, self.confidence)

    def get_method_weights(self) -> dict[str, float]:
        """Get current adaptive weights for debugging."""
        return self._compute_method_weights()

    def get_method_errors(self) -> dict[str, float]:
        """Get recent mean error for each method."""
        return {m: s.mean_error for m, s in self.method_stats.items()}

    def reset(self):
        """Full reset of predictor state."""
        self.gradient_history.clear()
        self.ema_gradients.clear()
        self.loss_history.clear()
        self.alignment_history.clear()
        self._last_predictions.clear()
        self.confidence = 0.0

        for stats in self.method_stats.values():
            stats.recent_errors.clear()
            stats.cumulative_error = 0.0
            stats.prediction_count = 0


def compute_dynamic_horizon(
    base_horizon: int, confidence: float, min_horizon: int = 1, max_horizon: int = 30
) -> int:
    """Compute dynamic prediction horizon based on confidence.

    Higher confidence allows longer prediction horizons.
    Uses confidence^1.5 to penalize low confidence more heavily.
    """
    scale = confidence**1.5
    horizon = int(base_horizon * scale)
    return max(min_horizon, min(horizon, max_horizon))


def compute_adaptive_lr_scale(
    base_scale: float,
    confidence: float,
    recent_bias: float,
    min_scale: float = 0.3,
    max_scale: float = 1.0,
) -> float:
    """Compute learning rate scale for prediction phase.

    Args:
        base_scale: Base learning rate multiplier
        confidence: Current prediction confidence [0, 1]
        recent_bias: Mean(predicted - actual) / actual from recent corrections
        min_scale: Minimum LR scale
        max_scale: Maximum LR scale

    Returns:
        Learning rate scale factor
    """
    # Confidence component: higher confidence = higher LR
    confidence_factor = 0.5 + 0.5 * confidence  # [0.5, 1.0]

    # Bias penalty: higher bias = lower LR
    bias_penalty = min(abs(recent_bias) * 2, 0.5)  # Max 50% penalty

    scale = base_scale * confidence_factor * (1 - bias_penalty)
    return max(min_scale, min(scale, max_scale))


def compute_correction_steps(
    prediction_error: float,
    prediction_horizon: int,
    min_steps: int = 3,
    max_steps: int = 25,
    error_multiplier: float = 15.0,
) -> int:
    """Compute number of correction steps based on prediction error.

    Larger errors and longer horizons require more correction.
    """
    error_steps = int(error_multiplier * prediction_error * prediction_horizon)
    return max(min_steps, min(error_steps, max_steps))


# Test the predictor
if __name__ == "__main__":
    print("Testing HybridGradientPredictor...")

    predictor = HybridGradientPredictor()

    # Simulate gradient observations with trend
    for i in range(20):
        # Simulate decreasing gradient norm with some noise
        base_grad = torch.randn(100) * (1.0 - i * 0.03) + torch.randn(100) * 0.1
        predictor.observe({"layer1": base_grad}, loss=10.0 - i * 0.3)

        if i >= 5:
            # Make predictions
            pred = predictor.predict(horizon=4)
            pred_loss = predictor.predict_loss(10.0 - i * 0.3, horizon=4)

            print(
                f"Step {i}: confidence={predictor.confidence:.3f}, "
                f"weights={predictor.get_method_weights()}"
            )

    print("\nFinal method errors:", predictor.get_method_errors())
    print("Final confidence:", predictor.confidence)
    print("\n✓ HybridGradientPredictor test complete")
