"""Training optimization toolkit - re-exported from vsa-training-optimizer.

This module re-exports the VSA Training Optimizer components for backwards
compatibility. The optimization toolkit has been extracted to a standalone
package for broader use.

Install the standalone package:
    pip install vsa-training-optimizer

Or install tritter with training extras:
    pip install tritter[training]

For direct usage, import from vsa_optimizer:
    from vsa_optimizer import PhaseTrainer, PhaseConfig

Example usage:
    >>> from tritter.training.optimization import PhaseTrainer, PhaseConfig
    >>> config = PhaseConfig(full_steps=10, predict_steps=40)
    >>> trainer = PhaseTrainer(model, optimizer, config)
    >>> for batch in dataloader:
    ...     stats = trainer.train_step(batch, compute_loss_fn)
    ...     print(f"Step {stats['total_step']}: loss={stats['loss']:.4f}, speedup={stats['speedup']:.2f}x")
"""

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
    _VSA_AVAILABLE = False

from tritter.training.optimization.vsa_utils import VSAKeyedBundler, make_random_keys

__all__ = [
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
    "VSAKeyedBundler",
    "make_random_keys",
    # Phase-based training
    "PhaseConfig",
    "PhaseTrainer",
    "TrainingPhase",
]
