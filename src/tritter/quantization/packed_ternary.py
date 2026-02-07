"""Packed ternary weight storage for inference-optimized BitNet models.

Why: Full-precision TernaryWeight stores FP32 shadow weights (7B = 28GB), making
7B inference impossible on 16GB VRAM. Ternary values {-1, 0, +1} only need 1.58 bits
but we store 32 bits. This module packs 4 ternary values per byte (2-bit encoding),
reducing 7B weights from ~28GB to ~1.4GB.

Encoding scheme:
    {-1, 0, +1} -> {0, 1, 2} (2 bits each)
    4 values per byte: (v0) | (v1 << 2) | (v2 << 4) | (v3 << 6)

Embedding-Prediction Context: Packed weights are used for inference only.
Training still uses full-precision shadow weights via TernaryWeight for STE gradient flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tritter.quantization.bitnet import TernaryWeight


def pack_ternary(weights: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ternary weights {-1,0,+1} into uint8 (4 values/byte).

    Args:
        weights: Ternary tensor of shape (out_features, in_features) with values in {-1, 0, +1}
        scale: Per-channel FP32 scale of shape (out_features, 1)

    Returns:
        packed: uint8 tensor of shape (out_features, ceil(in_features/4))
        scale: FP32 per-channel scale (unchanged, returned for convenience)

    Why: Reduces memory from 32 bits to 2 bits per weight. A 7B model drops from
    28GB to ~1.4GB, enabling 7B inference on RTX 5080 16GB.

    Encoding:
        -1 -> 0b00 (0)
         0 -> 0b01 (1)
        +1 -> 0b10 (2)

    Raises:
        ValueError: If weights contain values other than {-1, 0, +1}
    """
    # Validate input contains only ternary values
    valid_mask = (weights == -1) | (weights == 0) | (weights == 1)
    if not valid_mask.all():
        invalid_values = torch.unique(weights[~valid_mask])
        raise ValueError(
            f"Weights must be ternary {{-1, 0, +1}}, but found values: {invalid_values.tolist()}"
        )

    out_features, in_features = weights.shape

    # Encode: {-1, 0, +1} -> {0, 1, 2}
    # Using +1 offset: -1+1=0, 0+1=1, +1+1=2
    encoded = (weights + 1).to(torch.uint8)  # (out_features, in_features)

    # Pad in_features to multiple of 4 for even packing
    padding_needed = (4 - in_features % 4) % 4
    if padding_needed > 0:
        # Pad with encoded 0 (which is 1 in our encoding)
        encoded = F.pad(encoded, (0, padding_needed), value=1)

    padded_in_features = encoded.shape[1]

    # Reshape to groups of 4
    encoded = encoded.view(out_features, padded_in_features // 4, 4)  # (O, I//4, 4)

    # Pack 4 values per byte: v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
    packed = (
        encoded[:, :, 0]
        | (encoded[:, :, 1] << 2)
        | (encoded[:, :, 2] << 4)
        | (encoded[:, :, 3] << 6)
    )  # (out_features, ceil(in_features/4))

    return packed, scale


def unpack_ternary(
    packed: torch.Tensor,
    scale: torch.Tensor,
    original_in_features: int,
) -> torch.Tensor:
    """Unpack uint8 to scaled ternary weights.

    Args:
        packed: uint8 tensor of shape (out_features, ceil(in_features/4)) from pack_ternary
        scale: Per-channel FP32 scale of shape (out_features, 1)
        original_in_features: Original in_features dimension (before padding)

    Returns:
        weights: FP32 tensor of shape (out_features, in_features) with scaled ternary values

    Why: Unpacks on-the-fly during inference. GPU unpacking is cheap (~1-2 ops per value)
    while PCIe transfer of packed data is ~8x faster than FP32.
    """
    out_features = packed.shape[0]

    # Extract each of the 4 values per byte using bitwise operations
    v0 = packed & 0x03  # bits 0-1
    v1 = (packed >> 2) & 0x03  # bits 2-3
    v2 = (packed >> 4) & 0x03  # bits 4-5
    v3 = (packed >> 6) & 0x03  # bits 6-7

    # Interleave back to original order
    # Stack along new dim then reshape to flatten
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1)  # (O, I//4, 4)
    unpacked = unpacked.view(out_features, -1)  # (O, padded_in_features)

    # Truncate to original size (remove padding)
    unpacked = unpacked[:, :original_in_features]

    # Decode: {0, 1, 2} -> {-1, 0, +1}
    decoded = unpacked.to(scale.dtype) - 1.0  # (out_features, in_features)

    # Apply per-channel scaling
    scaled = decoded * scale  # (out_features, in_features)

    return scaled


class PackedTernaryWeight(nn.Module):  # type: ignore[misc]
    """Inference-only ternary layer with packed weight storage.

    Why: For inference, we don't need FP32 shadow weights (STE is training-only).
    Storing packed uint8 reduces 7B model from 28GB to ~1.4GB, enabling
    inference on RTX 5080 16GB with room for KV-cache and activations.

    Embedding-Prediction Context: This layer transforms embeddings in continuous
    space. Token-level prediction at the output is temporary scaffolding;
    production will use KNN/VQ rounding in embedding space.
    """

    # Type annotations for registered buffers
    packed_weight: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize packed ternary weight layer (empty, use from_ternary_weight to populate).

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: If True, the layer will have a learnable bias

        Why: Constructor creates empty buffers. Use from_ternary_weight() classmethod
        to create a properly initialized layer from a trained TernaryWeight.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Calculate packed size (ceil(in_features / 4))
        self.packed_in_features = (in_features + 3) // 4

        # Packed weights stored as non-trainable buffer (uint8)
        self.register_buffer(
            "packed_weight",
            torch.zeros(out_features, self.packed_in_features, dtype=torch.uint8),
        )

        # Per-channel scale (FP32, required for accurate reconstruction)
        self.register_buffer(
            "scale",
            torch.ones(out_features, 1),
        )

        # Bias (FP32, optional)
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_ternary_weight(cls, ternary: TernaryWeight) -> PackedTernaryWeight:
        """Convert trained TernaryWeight to packed inference format.

        Args:
            ternary: Trained TernaryWeight layer with FP32 shadow weights

        Returns:
            PackedTernaryWeight layer with packed uint8 storage

        Why: One-time conversion after training. Reduces weight memory by ~16x
        (FP32 -> 2-bit packed) while preserving exact ternary values.
        """
        # Create packed layer with same configuration
        packed_layer = cls(
            in_features=ternary.in_features,
            out_features=ternary.out_features,
            bias=ternary.bias is not None,
        )

        # Quantize weights to ternary values
        with torch.no_grad():
            quantized = ternary.quantize_weights(ternary.weight)  # (O, I) in {-1, 0, +1}

            # Pack into uint8
            packed_weights, _ = pack_ternary(quantized, ternary.scale)

            # Copy to packed layer buffers
            packed_layer.packed_weight.copy_(packed_weights)
            packed_layer.scale.copy_(ternary.scale.detach())

            if ternary.bias is not None and packed_layer.bias is not None:
                packed_layer.bias.copy_(ternary.bias.detach())

        return packed_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly unpacking.

        Args:
            x: Input tensor of shape (batch_size, in_features) or
               (batch_size, seq_len, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features) or
            (batch_size, seq_len, out_features)

        Why: Unpacks weights from uint8 to FP32 on GPU. Unpacking is cheap
        (~1-2 ops per value) and occurs only once per forward pass.
        The benefit is ~8x reduction in CPU->GPU transfer time.
        """
        # Unpack weights on-the-fly
        weight = unpack_ternary(
            self.packed_weight,
            self.scale,
            self.in_features,
        )  # (out_features, in_features)

        # Linear transformation
        output = F.linear(x, weight, self.bias)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for module."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed_size={self.packed_in_features}"
        )

    def memory_bytes(self) -> int:
        """Return approximate memory usage in bytes.

        Why: Useful for verifying memory savings vs TernaryWeight.
        """
        packed_bytes = self.packed_weight.numel() * 1  # uint8 = 1 byte
        scale_bytes = self.scale.numel() * 4  # FP32 = 4 bytes
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0
        return packed_bytes + scale_bytes + bias_bytes  # type: ignore[no-any-return]


def convert_to_packed(model: nn.Module) -> nn.Module:
    """Recursively convert TernaryWeight layers to PackedTernaryWeight.

    Args:
        model: PyTorch model containing TernaryWeight layers

    Returns:
        Same model with TernaryWeight layers replaced by PackedTernaryWeight

    Why: One-time conversion after training, before inference deployment.
    Modifies the model in-place and returns it for convenience.

    Note: This function modifies the model in-place. The model should be
    in eval mode and gradients should not be needed after conversion.
    """
    for name, module in model.named_children():
        if isinstance(module, TernaryWeight):
            # Convert TernaryWeight to PackedTernaryWeight
            packed = PackedTernaryWeight.from_ternary_weight(module)
            setattr(model, name, packed)
        else:
            # Recursively process children
            convert_to_packed(module)

    return model


def save_packed_model(model: nn.Module, path: str | Path) -> None:
    """Save model with packed ternary weights.

    Args:
        model: Model (with PackedTernaryWeight layers) to save
        path: File path to save the model

    Why: Standard torch.save with metadata for model reconstruction.
    Includes original architecture info needed for loading.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect metadata about packed layers
    packed_layers_info: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedTernaryWeight):
            packed_layers_info[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            }

    save_dict = {
        "state_dict": model.state_dict(),
        "packed_layers_info": packed_layers_info,
        "format_version": "1.0",
    }

    torch.save(save_dict, path)


def load_packed_model(
    path: str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load packed model weights into an existing model.

    Args:
        path: Path to saved packed model
        model: Model instance to load weights into (should have PackedTernaryWeight layers)
        device: Device to load weights onto

    Returns:
        Model with loaded weights

    Why: Loads packed weights into a model that has already been converted
    to PackedTernaryWeight layers (or was built with them).

    Raises:
        ValueError: If format version is incompatible
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Check format version
    format_version = checkpoint.get("format_version", "unknown")
    if format_version not in ["1.0"]:
        raise ValueError(
            f"Unsupported packed model format version: {format_version}. Expected one of: ['1.0']"
        )

    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])

    return model
