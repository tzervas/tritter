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

from tritter.training.optimization.vsa_utils import VSAKeyedBundler, make_random_keys

_VSA_AVAILABLE: bool = False

try:
    from vsa_optimizer import (
        # Gradient prediction
        GradientPredictor,
        # Phase-based training
        PhaseConfig,
        PhaseTrainer,
        PredictionConfig,
        PredictiveTrainer,
        # Ternary optimization
        TernaryConfig,
        TernaryGradientAccumulator,
        TernaryOptimizer,
        TrainingPhase,
        # VSA compression
        VSAConfig,
        VSAGradientCompressor,
        hyperdimensional_bind,
        hyperdimensional_bundle,
        ternary_quantize,
    )

    _VSA_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # Always available (local implementations)
    "VSAKeyedBundler",
    "make_random_keys",
    "_VSA_AVAILABLE",
]

if _VSA_AVAILABLE:
    __all__ += [
        # Gradient prediction
        "GradientPredictor",
        "PredictionConfig",
        "PredictiveTrainer",
        # Ternary optimization
        "TernaryConfig",
        "TernaryGradientAccumulator",
        "TernaryOptimizer",
        # VSA compression
        "VSAConfig",
        "VSAGradientCompressor",
        "hyperdimensional_bind",
        "hyperdimensional_bundle",
        "ternary_quantize",
        # Phase-based training
        "PhaseConfig",
        "PhaseTrainer",
        "TrainingPhase",
    ]
