"""Checkpoint management for Tritter models.

Why: Supports multiple output formats for different deployment targets:
- safetensors: HuggingFace Hub distribution
- GGUF: llama.cpp, Ollama, local inference
- Progressive: Enables model size expansion

Components:
- save_checkpoint: Save model with metadata
- load_checkpoint: Load model from any supported format
- export_safetensors: Export to HuggingFace-compatible format
- export_gguf: Export to llama.cpp-compatible format
- offload_checkpoint: Upload to remote server
"""

from tritter.checkpoints.formats import (
    CheckpointFormat,
    export_gguf,
    export_safetensors,
    load_checkpoint,
    save_checkpoint,
)
from tritter.checkpoints.offload import (
    list_remote_checkpoints,
    offload_checkpoint,
    retrieve_checkpoint,
)
from tritter.checkpoints.progressive import (
    ProgressiveMetadata,
    expand_model,
    load_progressive,
    save_progressive,
)

__all__ = [
    # Formats
    "CheckpointFormat",
    "save_checkpoint",
    "load_checkpoint",
    "export_safetensors",
    "export_gguf",
    # Progressive
    "ProgressiveMetadata",
    "save_progressive",
    "load_progressive",
    "expand_model",
    # Offload
    "offload_checkpoint",
    "retrieve_checkpoint",
    "list_remote_checkpoints",
]
