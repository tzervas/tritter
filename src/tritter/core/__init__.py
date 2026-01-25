"""Core configuration and foundational components."""

from tritter.core.config import TritterConfig
from tritter.core.model_specs import (
    MODEL_SPECS,
    HardwareRecommendation,
    MemoryEstimate,
    ModelSize,
    ModelSpec,
    estimate_memory,
    get_model_spec,
    list_models,
    print_model_summary,
    recommend_hardware,
)

__all__ = [
    "HardwareRecommendation",
    "MemoryEstimate",
    "MODEL_SPECS",
    "ModelSize",
    "ModelSpec",
    "TritterConfig",
    "estimate_memory",
    "get_model_spec",
    "list_models",
    "print_model_summary",
    "recommend_hardware",
]
