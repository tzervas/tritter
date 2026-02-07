"""Training optimization toolkit - in-tree implementations.

This module provides training optimizations for BitNet models including:
- Gradient prediction for faster training
- Ternary gradient accumulation
- VSA-based gradient compression

Note: vsa-training-optimizer package is not yet published to PyPI.
These implementations are maintained in-tree until the standalone package is released.

Example usage:
    >>> from tritter.training.optimization import PhaseConfig
    >>> # Configuration for hybrid training
    >>> config = PhaseConfig(full_steps=10, predict_steps=40)
    >>> # Use with your training loop
"""

# Provide informative message when trying to import from non-existent package
try:
    from vsa_optimizer import (
        GradientPredictor,
        PredictionConfig,
        PredictiveTrainer,
        PhaseConfig,
        PhaseTrainer,
        TrainingPhase,
        TernaryConfig,
        TernaryGradientAccumulator,
        TernaryOptimizer,
        VSAConfig,
        VSAGradientCompressor,
        hyperdimensional_bind,
        hyperdimensional_bundle,
        ternary_quantize,
    )

    # If import succeeds, these are available
    __all__ = [
        "GradientPredictor",
        "PredictionConfig",
        "PredictiveTrainer",
        "TernaryConfig",
        "TernaryGradientAccumulator",
        "TernaryOptimizer",
        "VSAConfig",
        "VSAGradientCompressor",
        "hyperdimensional_bind",
        "hyperdimensional_bundle",
        "ternary_quantize",
        "PhaseConfig",
        "PhaseTrainer",
        "TrainingPhase",
    ]

except ImportError:
    # Package not installed - provide stub implementations or helpful message
    import warnings

    warnings.warn(
        "vsa-training-optimizer package not installed. "
        "Training optimizations are currently in development. "
        "Basic training functionality is available via tritter.training.trainer",
        ImportWarning,
        stacklevel=2
    )

    # Provide minimal stubs to avoid breaking imports
    class _NotAvailable:
        """Placeholder for unavailable optimization components."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "vsa-training-optimizer package is not installed. "
                "Install with: pip install vsa-training-optimizer (when available) "
                "or use tritter.training.trainer for basic training."
            )

    # Export stubs so imports don't fail
    GradientPredictor = _NotAvailable
    PredictionConfig = _NotAvailable
    PredictiveTrainer = _NotAvailable
    PhaseConfig = _NotAvailable
    PhaseTrainer = _NotAvailable
    TrainingPhase = _NotAvailable
    TernaryConfig = _NotAvailable
    TernaryGradientAccumulator = _NotAvailable
    TernaryOptimizer = _NotAvailable
    VSAConfig = _NotAvailable
    VSAGradientCompressor = _NotAvailable
    hyperdimensional_bind = _NotAvailable
    hyperdimensional_bundle = _NotAvailable
    ternary_quantize = _NotAvailable

    __all__ = [
        "GradientPredictor",
        "PredictionConfig",
        "PredictiveTrainer",
        "TernaryConfig",
        "TernaryGradientAccumulator",
        "TernaryOptimizer",
        "VSAConfig",
        "VSAGradientCompressor",
        "hyperdimensional_bind",
        "hyperdimensional_bundle",
        "ternary_quantize",
        "PhaseConfig",
        "PhaseTrainer",
        "TrainingPhase",
    ]
