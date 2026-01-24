"""Model architecture components.

Provides core transformer architecture and multimodal embedding layers for the
Tritter multimodal model. Includes attention mechanisms, transformer blocks,
and unified embedding space for text, vision, and audio modalities.

Why (Embedding-Prediction Paradigm):
    The model operates in continuous embedding space rather than discrete token
    space. All modalities (text, image, audio) are converted to embeddings in
    a shared space, enabling cross-modal attention. The discrete tokenization
    is temporary scaffolding for training compatibility.

Reference: CLAUDE.md architecture section, docs/project-plan.md
"""

from tritter.models.architecture import TritterModel
from tritter.models.flex_attention import (
    HAS_FLEX_ATTENTION,
    FlexAttentionLayer,
    create_attention_mask,
)
from tritter.models.multimodal_embedding import (
    ModalityType,
    MultimodalBatch,
    MultimodalEmbedding,
    MultimodalEmbeddingConfig,
    MultimodalOutput,
    create_multimodal_embedding,
)

__all__ = [
    # Core architecture
    "TritterModel",
    # Attention
    "FlexAttentionLayer",
    "create_attention_mask",
    "HAS_FLEX_ATTENTION",
    # Multimodal embedding
    "ModalityType",
    "MultimodalBatch",
    "MultimodalEmbedding",
    "MultimodalEmbeddingConfig",
    "MultimodalOutput",
    "create_multimodal_embedding",
]
