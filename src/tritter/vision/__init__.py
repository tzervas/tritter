"""Vision encoder module for multimodal image understanding.

Provides SigLIP-based vision encoding for early fusion multimodal architecture.

Why (Embedding-Prediction Paradigm):
    Images enter the model through the vision encoder, which produces continuous
    embeddings that are projected into the shared embedding space. This enables
    cross-modal attention between text and image representations. The vision
    encoder is frozen (pretrained) while only the projection layer is trained,
    following LLaVA-style efficient multimodal adaptation.

Reference: SmolVLM findings show SigLIP-B/16 (93M) outperforms larger encoders
for compact LMs. See docs/project-plan.md for memory budget calculations.
"""

from tritter.vision.siglip import (
    SigLIPConfig,
    SigLIPVisionEncoder,
    create_siglip_encoder,
)

__all__ = [
    "SigLIPConfig",
    "SigLIPVisionEncoder",
    "create_siglip_encoder",
]
