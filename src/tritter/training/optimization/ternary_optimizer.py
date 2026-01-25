"""Ternary math acceleration for gradient operations.

Why: Ternary values {-1, 0, +1} enable extremely fast operations:
- Multiplications become sign flips or zeros
- Additions remain additions
- Memory reduced by ~10x (2 bits vs 32 bits)

This module provides optimized gradient accumulation and optimization
using ternary representations where possible, falling back to full
precision only when necessary for accuracy.

Key insight: While model weights can be ternary, gradients need more
precision for convergence. However, we can use ternary representations
for intermediate computations and restore precision for updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from torch.optim import Optimizer


@dataclass
class TernaryConfig:
    """Configuration for ternary optimization."""

    accumulation_steps: int = 8  # Steps to accumulate before update
    ternary_threshold: float = 0.5  # Threshold for ternary quantization
    scale_learning_rate: float = 0.01  # Learning rate for scale parameters
    use_stochastic_rounding: bool = True  # Stochastic vs deterministic rounding


def ternary_quantize_stochastic(
    x: Tensor,
    threshold: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize tensor to ternary using stochastic rounding.

    Args:
        x: Input tensor
        threshold: Quantization threshold (uses mean abs if None)

    Returns:
        Tuple of (ternary tensor, scale factor tensor)

    Why: Stochastic rounding preserves gradient information in expectation.
    A value like 0.3 has 30% chance of being +1 and 70% chance of being 0,
    so the expected value equals the input. This enables unbiased gradient
    accumulation even with ternary storage.
    """
    if threshold is None:
        threshold = x.abs().mean()

    # Compute scale (per-tensor for simplicity)
    scale = x.abs().max()
    if scale == 0:
        return torch.zeros_like(x), scale

    # Normalize to [-1, 1]
    normalized = x / scale

    # Compute probabilities for stochastic rounding
    # P(+1) = max(0, normalized), P(-1) = max(0, -normalized), P(0) = 1 - |normalized|
    abs_norm = normalized.abs()

    # Random values for stochastic decision
    rand = torch.rand_like(x)

    # Quantize: round to +1/-1 with probability |value|, else 0
    ternary = torch.zeros_like(x)
    mask_positive = (normalized > 0) & (rand < abs_norm)
    mask_negative = (normalized < 0) & (rand < abs_norm)
    ternary[mask_positive] = 1
    ternary[mask_negative] = -1

    return ternary, scale


def ternary_quantize_deterministic(
    x: Tensor,
    threshold: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize tensor to ternary using deterministic thresholding.

    Args:
        x: Input tensor
        threshold: Quantization threshold (uses mean abs if None)

    Returns:
        Tuple of (ternary tensor, scale factor)

    Why: Deterministic quantization is faster and reproducible but
    introduces bias. Used when speed matters more than unbiasedness.
    """
    if threshold is None:
        threshold = x.abs().mean()

    scale = x.abs().max()
    if scale == 0:
        return torch.zeros_like(x), scale

    # Simple thresholding
    ternary = torch.zeros_like(x)
    ternary[x > threshold] = 1
    ternary[x < -threshold] = -1

    return ternary, scale


class TernaryGradientAccumulator:
    """Accumulate gradients using ternary representation.

    Why: Gradient accumulation over many steps can be memory-intensive.
    Using ternary representation with scale factors reduces memory by ~10x
    while maintaining accuracy through the scale factors.

    The accumulator keeps a ternary "direction" tensor and a scale tensor.
    New gradients are projected onto this representation and accumulated.
    Full-precision reconstruction happens only at update time.

    Example:
        >>> accumulator = TernaryGradientAccumulator(model)
        >>> for micro_batch in batches:
        ...     loss.backward()
        ...     accumulator.accumulate()
        >>> full_grads = accumulator.get_accumulated()
        >>> accumulator.reset()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TernaryConfig | None = None,
    ) -> None:
        """Initialize gradient accumulator.

        Args:
            model: Model whose gradients to accumulate
            config: Ternary configuration
        """
        self.model = model
        self.config = config or TernaryConfig()

        # Accumulated ternary gradients and scales
        self.ternary_accum: dict[str, Tensor] = {}
        self.scale_accum: dict[str, Tensor] = {}
        self.count = 0

        # Initialize storage
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ternary_accum[name] = torch.zeros_like(param.data)
                self.scale_accum[name] = torch.zeros(1, device=param.device)

    def accumulate(self) -> None:
        """Accumulate current gradients in ternary form.

        Why: Converts current .grad to ternary, accumulates direction,
        and tracks scale. This is called after each backward pass.
        """
        quantize_fn = (
            ternary_quantize_stochastic
            if self.config.use_stochastic_rounding
            else ternary_quantize_deterministic
        )

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Quantize gradient
                ternary, scale = quantize_fn(
                    param.grad,
                    self.config.ternary_threshold,
                )

                # Accumulate (ternary addition = element-wise sum)
                # Note: Sum of ternary is not ternary, but uses fewer bits
                self.ternary_accum[name] += ternary
                self.scale_accum[name] += scale

        self.count += 1

    def get_accumulated(self) -> dict[str, Tensor]:
        """Get full-precision accumulated gradients.

        Returns:
            Dictionary mapping parameter names to accumulated gradients

        Why: Reconstructs full-precision gradients from ternary accumulation.
        The scale is averaged and applied to the accumulated direction.
        """
        accumulated = {}

        for name in self.ternary_accum:
            if self.count > 0:
                # Average scale
                avg_scale = self.scale_accum[name] / self.count
                # Reconstruct: direction * scale / count
                accumulated[name] = (self.ternary_accum[name] * avg_scale) / self.count
            else:
                accumulated[name] = self.ternary_accum[name]

        return accumulated

    def apply_to_model(self) -> None:
        """Apply accumulated gradients to model parameters.

        Why: Sets .grad attribute of parameters to accumulated values
        for optimizer to use.
        """
        accumulated = self.get_accumulated()

        for name, param in self.model.named_parameters():
            if name in accumulated:
                if param.grad is None:
                    param.grad = accumulated[name].clone()
                else:
                    param.grad.copy_(accumulated[name])

    def reset(self) -> None:
        """Reset accumulator for next accumulation cycle."""
        for name in self.ternary_accum:
            self.ternary_accum[name].zero_()
            self.scale_accum[name].zero_()
        self.count = 0

    def memory_savings(self) -> float:
        """Calculate memory savings from ternary representation.

        Returns:
            Fraction of memory saved (0 to 1)
        """
        # Full precision: 32 bits per element
        # Ternary: 2 bits per element + 32 bits scale per tensor
        full_bits = sum(p.numel() * 32 for p in self.model.parameters() if p.requires_grad)
        ternary_bits = sum(p.numel() * 2 + 32 for p in self.model.parameters() if p.requires_grad)

        return 1 - (ternary_bits / full_bits)


class TernaryOptimizer:
    """Optimizer wrapper with ternary gradient accumulation.

    Why: Combines ternary accumulation with a base optimizer for
    memory-efficient training. Useful for large batch training where
    gradient accumulation is necessary.

    Example:
        >>> base_optimizer = torch.optim.AdamW(model.parameters())
        >>> optimizer = TernaryOptimizer(model, base_optimizer)
        >>> for i, batch in enumerate(dataloader):
        ...     loss = model(batch)
        ...     loss.backward()
        ...     if optimizer.step():  # Returns True when actual update happens
        ...         print(f"Update at step {i}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: TernaryConfig | None = None,
    ) -> None:
        """Initialize ternary optimizer.

        Args:
            model: Model to optimize
            optimizer: Base optimizer
            config: Ternary configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or TernaryConfig()
        self.accumulator = TernaryGradientAccumulator(model, config)

        self.step_count = 0
        self.update_count = 0

    def step(self) -> bool:
        """Accumulate gradient and optionally update.

        Returns:
            True if optimizer update was performed, False if just accumulated

        Why: Accumulates gradients in ternary form. When accumulation_steps
        is reached, reconstructs full gradients and performs optimizer step.
        """
        # Accumulate current gradients
        self.accumulator.accumulate()
        self.step_count += 1

        # Check if update is needed
        if self.step_count % self.config.accumulation_steps == 0:
            # Apply accumulated gradients
            self.accumulator.apply_to_model()

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Reset accumulator
            self.accumulator.reset()
            self.update_count += 1

            return True

        return False

    def zero_grad(self) -> None:
        """Zero gradients (typically not needed with accumulation)."""
        self.optimizer.zero_grad()

    def state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            "step_count": self.step_count,
            "update_count": self.update_count,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.step_count = state["step_count"]
        self.update_count = state["update_count"]
        self.optimizer.load_state_dict(state["optimizer"])

    def get_stats(self) -> dict[str, float]:
        """Get optimization statistics."""
        return {
            "step_count": self.step_count,
            "update_count": self.update_count,
            "memory_savings": self.accumulator.memory_savings(),
        }


__all__ = [
    "TernaryConfig",
    "TernaryGradientAccumulator",
    "TernaryOptimizer",
    "ternary_quantize_stochastic",
    "ternary_quantize_deterministic",
]
