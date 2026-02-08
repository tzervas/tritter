"""Vision encoder module for multimodal image understanding.

Provides SigLIP-based vision encoding and VQ-VAE image tokenization for early
fusion multimodal architecture.

Why (Embedding-Prediction Paradigm):
    Images enter the model through vision encoders, which produce continuous
    embeddings that are projected into the shared embedding space. This enables
    cross-modal attention between text and image representations.

    Two approaches are supported:
    1. SigLIP: Patch-based encoding with frozen pretrained encoder (LLaVA-style)
    2. VQ-VAE: Discrete tokenization for unified vocabulary (256-512 tokens/image)

Reference: SmolVLM findings show SigLIP-B/16 (93M) outperforms larger encoders
for compact LMs. VQ-VAE provides aggressive compression for unified vocabulary.
See docs/project-plan.md for memory budget calculations.
"""

from tritter.vision.siglip import (
    SigLIPConfig,
    SigLIPVisionEncoder,
    create_siglip_encoder,
)
from tritter.vision.vqvae import (
    ImageProjection,
    VectorQuantizer,
    VQVAEConfig,
    VQVAEDecoder,
    VQVAEEncoder,
    VQVAEImageTokenizer,
    create_vqvae_tokenizer,
)

__all__ = [
    # SigLIP (patch-based)
    "SigLIPConfig",
    "SigLIPVisionEncoder",
    "create_siglip_encoder",
    # VQ-VAE (discrete tokens)
    "VQVAEConfig",
    "VQVAEEncoder",
    "VQVAEDecoder",
    "VectorQuantizer",
    "VQVAEImageTokenizer",
    "ImageProjection",
    "create_vqvae_tokenizer",
]
