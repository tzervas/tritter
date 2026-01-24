"""Checkpoint format handling for safetensors and GGUF.

Why: Different deployment targets need different formats:
- safetensors: HuggingFace Hub, PyTorch inference
- GGUF: llama.cpp, Ollama, quantized CPU/GPU inference

Both formats are directly writable without intermediate conversion.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import torch
import torch.nn as nn

try:
    import safetensors.torch as st

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


class CheckpointFormat(Enum):
    """Supported checkpoint formats."""

    PYTORCH = "pytorch"  # .pt/.pth files
    SAFETENSORS = "safetensors"  # .safetensors files
    GGUF = "gguf"  # .gguf files
    PROGRESSIVE = "progressive"  # .tprog directory


@dataclass
class CheckpointMetadata:
    """Metadata stored with checkpoints.

    Why: Enables loading without knowing model architecture beforehand.
    """

    model_size: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_position_embeddings: int
    use_bitnet: bool
    training_step: int | None = None
    tokens_seen: int | None = None


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    format: CheckpointFormat = CheckpointFormat.SAFETENSORS,
    metadata: CheckpointMetadata | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save model checkpoint in the specified format.

    Args:
        model: Model to save
        path: Output path (directory for progressive, file otherwise)
        format: Output format
        metadata: Optional metadata to include
        optimizer: Optional optimizer state to save

    Returns:
        Path to saved checkpoint

    Why: Unified API for all checkpoint formats.
    """
    path = Path(path)

    if format == CheckpointFormat.SAFETENSORS:
        return _save_safetensors(model, path, metadata)
    elif format == CheckpointFormat.PYTORCH:
        return _save_pytorch(model, path, metadata, optimizer)
    elif format == CheckpointFormat.GGUF:
        return _save_gguf(model, path, metadata)
    elif format == CheckpointFormat.PROGRESSIVE:
        # Progressive format handled in progressive.py
        from tritter.checkpoints.progressive import save_progressive

        return save_progressive(model, path, metadata)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], CheckpointMetadata | None]:
    """Load checkpoint from any supported format.

    Args:
        path: Path to checkpoint
        device: Target device for tensors

    Returns:
        Tuple of (state_dict, metadata)

    Why: Auto-detects format for seamless loading.
    """
    path = Path(path)

    # Detect format
    if path.is_dir():
        if (path / "progressive.json").exists():
            from tritter.checkpoints.progressive import load_progressive

            return load_progressive(path, device)
        elif (path / "weights.safetensors").exists():
            return _load_safetensors(path / "weights.safetensors", device)

    suffix = path.suffix.lower()
    if suffix in (".safetensors",):
        return _load_safetensors(path, device)
    elif suffix in (".pt", ".pth", ".bin"):
        return _load_pytorch(path, device)
    elif suffix == ".gguf":
        return _load_gguf(path, device)
    else:
        raise ValueError(f"Unknown checkpoint format: {path}")


def export_safetensors(
    model: nn.Module,
    output_path: str | Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Export model to safetensors format.

    Args:
        model: Model to export
        output_path: Output .safetensors file path
        metadata: Optional string metadata (safetensors requires str values)

    Returns:
        Path to saved file

    Why: Direct export to HuggingFace-compatible format.
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not installed: pip install safetensors")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    # Convert metadata values to strings (safetensors requirement)
    str_metadata = {}
    if metadata:
        for k, v in metadata.items():
            str_metadata[k] = str(v)

    st.save_file(state_dict, str(output_path), metadata=str_metadata)
    return output_path


def export_gguf(
    model: nn.Module,
    output_path: str | Path,
    metadata: CheckpointMetadata | None = None,
    quantization: str = "f16",  # f32, f16, q8_0, q4_0, q4_1
) -> Path:
    """Export model to GGUF format.

    Args:
        model: Model to export
        output_path: Output .gguf file path
        metadata: Model metadata
        quantization: Quantization type for weights

    Returns:
        Path to saved file

    Why: Direct export to llama.cpp/Ollama-compatible format.

    Note: This is a simplified GGUF writer. For production use,
    consider using llama.cpp's convert tools for full compatibility.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    with open(output_path, "wb") as f:
        _write_gguf(f, state_dict, metadata, quantization)

    return output_path


# ============================================================================
# Internal format handlers
# ============================================================================


def _save_safetensors(
    model: nn.Module,
    path: Path,
    metadata: CheckpointMetadata | None,
) -> Path:
    """Save as safetensors."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not installed: pip install safetensors")

    # Ensure .safetensors extension
    if not path.suffix:
        path = path.with_suffix(".safetensors")

    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    # Convert metadata to strings
    str_metadata = {}
    if metadata:
        str_metadata["model_size"] = metadata.model_size
        str_metadata["hidden_size"] = str(metadata.hidden_size)
        str_metadata["num_layers"] = str(metadata.num_layers)
        str_metadata["num_heads"] = str(metadata.num_heads)
        str_metadata["num_kv_heads"] = str(metadata.num_kv_heads)
        str_metadata["vocab_size"] = str(metadata.vocab_size)
        str_metadata["use_bitnet"] = str(metadata.use_bitnet)

    st.save_file(state_dict, str(path), metadata=str_metadata)
    return path


def _load_safetensors(
    path: Path,
    device: str,
) -> tuple[dict[str, torch.Tensor], CheckpointMetadata | None]:
    """Load from safetensors."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not installed: pip install safetensors")

    state_dict = st.load_file(str(path), device=device)

    # Try to extract metadata
    metadata = None
    try:
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
            if "__metadata__" in header:
                m = header["__metadata__"]
                metadata = CheckpointMetadata(
                    model_size=m.get("model_size", "unknown"),
                    hidden_size=int(m.get("hidden_size", 0)),
                    num_layers=int(m.get("num_layers", 0)),
                    num_heads=int(m.get("num_heads", 0)),
                    num_kv_heads=int(m.get("num_kv_heads", 0)),
                    vocab_size=int(m.get("vocab_size", 0)),
                    max_position_embeddings=int(m.get("max_position_embeddings", 0)),
                    use_bitnet=m.get("use_bitnet", "False") == "True",
                )
    except Exception:
        pass

    return state_dict, metadata


def _save_pytorch(
    model: nn.Module,
    path: Path,
    metadata: CheckpointMetadata | None,
    optimizer: torch.optim.Optimizer | None,
) -> Path:
    """Save as PyTorch checkpoint."""
    if not path.suffix:
        path = path.with_suffix(".pt")

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if metadata:
        checkpoint["metadata"] = {
            "model_size": metadata.model_size,
            "hidden_size": metadata.hidden_size,
            "num_layers": metadata.num_layers,
            "num_heads": metadata.num_heads,
            "num_kv_heads": metadata.num_kv_heads,
            "vocab_size": metadata.vocab_size,
            "max_position_embeddings": metadata.max_position_embeddings,
            "use_bitnet": metadata.use_bitnet,
        }

    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    return path


def _load_pytorch(
    path: Path,
    device: str,
) -> tuple[dict[str, torch.Tensor], CheckpointMetadata | None]:
    """Load from PyTorch checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = None
        if "metadata" in checkpoint:
            m = checkpoint["metadata"]
            metadata = CheckpointMetadata(**m)
    else:
        # Raw state dict
        state_dict = checkpoint
        metadata = None

    return state_dict, metadata


# ============================================================================
# GGUF Format Support
# ============================================================================

# GGUF magic number and version
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF data types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types (quantization)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8


def _write_gguf(
    f: BinaryIO,
    state_dict: dict[str, torch.Tensor],
    metadata: CheckpointMetadata | None,
    quantization: str,
) -> None:
    """Write GGUF file.

    GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    """
    # Determine tensor type
    if quantization == "f32":
        tensor_type = GGML_TYPE_F32
    elif quantization == "f16":
        tensor_type = GGML_TYPE_F16
    elif quantization == "q8_0":
        tensor_type = GGML_TYPE_Q8_0
    elif quantization == "q4_0":
        tensor_type = GGML_TYPE_Q4_0
    elif quantization == "q4_1":
        tensor_type = GGML_TYPE_Q4_1
    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    # Build metadata key-value pairs
    kv_pairs = []
    if metadata:
        kv_pairs.append(("general.architecture", "tritter"))
        kv_pairs.append(("general.name", f"tritter-{metadata.model_size}"))
        kv_pairs.append(("tritter.context_length", metadata.max_position_embeddings))
        kv_pairs.append(("tritter.embedding_length", metadata.hidden_size))
        kv_pairs.append(("tritter.block_count", metadata.num_layers))
        kv_pairs.append(("tritter.attention.head_count", metadata.num_heads))
        kv_pairs.append(("tritter.attention.head_count_kv", metadata.num_kv_heads))
        kv_pairs.append(("tritter.vocab_size", metadata.vocab_size))

    # Write header
    f.write(struct.pack("<I", GGUF_MAGIC))
    f.write(struct.pack("<I", GGUF_VERSION))
    f.write(struct.pack("<Q", len(state_dict)))  # tensor count
    f.write(struct.pack("<Q", len(kv_pairs)))  # metadata kv count

    # Write metadata
    for key, value in kv_pairs:
        _write_gguf_string(f, key)
        if isinstance(value, str):
            f.write(struct.pack("<I", GGUF_TYPE_STRING))
            _write_gguf_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack("<I", GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", value))
        elif isinstance(value, float):
            f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
            f.write(struct.pack("<f", value))

    # Write tensor info
    tensor_data_start = f.tell()
    alignment = 32  # GGUF alignment

    tensor_infos = []
    offset = 0

    for name, tensor in state_dict.items():
        # Convert name to GGUF format
        gguf_name = _pytorch_to_gguf_name(name)

        # Get tensor info
        shape = tensor.shape
        n_dims = len(shape)

        # Calculate size
        if tensor_type == GGML_TYPE_F32:
            element_size = 4
        elif tensor_type == GGML_TYPE_F16:
            element_size = 2
        else:
            # Quantized types have different sizes
            element_size = 1  # Simplified

        size = tensor.numel() * element_size
        aligned_offset = (offset + alignment - 1) // alignment * alignment

        tensor_infos.append((gguf_name, n_dims, shape, tensor_type, aligned_offset))
        offset = aligned_offset + size

    # Write tensor info headers
    for gguf_name, n_dims, shape, ttype, toffset in tensor_infos:
        _write_gguf_string(f, gguf_name)
        f.write(struct.pack("<I", n_dims))
        for dim in reversed(shape):  # GGUF uses reversed order
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", ttype))
        f.write(struct.pack("<Q", toffset))

    # Align to start of tensor data
    current_pos = f.tell()
    padding = (alignment - (current_pos % alignment)) % alignment
    f.write(b"\x00" * padding)

    # Write tensor data
    for name, tensor in state_dict.items():
        # Align
        current_pos = f.tell()
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b"\x00" * padding)

        # Convert tensor
        if tensor_type == GGML_TYPE_F16:
            data = tensor.half().cpu().numpy().tobytes()
        elif tensor_type == GGML_TYPE_F32:
            data = tensor.float().cpu().numpy().tobytes()
        else:
            # For quantized types, use float16 as fallback
            # Full quantization requires more complex code
            data = tensor.half().cpu().numpy().tobytes()

        f.write(data)


def _write_gguf_string(f: BinaryIO, s: str) -> None:
    """Write GGUF string (length-prefixed)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _pytorch_to_gguf_name(name: str) -> str:
    """Convert PyTorch parameter name to GGUF format."""
    # Simple mapping - expand as needed
    name = name.replace(".", "_")
    name = name.replace("layers_", "blk.")
    name = name.replace("_weight", ".weight")
    name = name.replace("_bias", ".bias")
    return name


def _save_gguf(
    model: nn.Module,
    path: Path,
    metadata: CheckpointMetadata | None,
) -> Path:
    """Save as GGUF with default F16 quantization."""
    if not path.suffix:
        path = path.with_suffix(".gguf")

    return export_gguf(model, path, metadata, quantization="f16")


def _load_gguf(
    path: Path,
    device: str,
) -> tuple[dict[str, torch.Tensor], CheckpointMetadata | None]:
    """Load from GGUF file.

    Note: GGUF loading is complex. For now, raise NotImplementedError.
    Use llama.cpp tools to convert GGUF to safetensors if needed.
    """
    raise NotImplementedError(
        "GGUF loading not yet implemented. "
        "Use llama.cpp's convert tool to convert to safetensors."
    )
