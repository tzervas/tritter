"""INT4 quantized KV-cache for efficient long-context inference.

Why: FP16 KV-cache for 128K context with 7B model requires ~67 GB VRAM
(2 * 32 layers * 32 heads * 128 head_dim * 131072 seq_len * 2 bytes = 67 GB).
INT4 quantization reduces this to ~8.4 GB, fitting within the RTX 5080 16GB budget.

Quantization strategy per KIVI paper (arXiv:2402.02750):
- Keys: Per-channel quantization (dim=-1) preserves attention pattern structure
- Values: Per-token quantization (dim=1) improves reconstruction quality

Embedding-Prediction Context: The KV-cache stores intermediate representations in
continuous embedding space. Quantization affects precision of cached states, but
INT4 maintains sufficient fidelity for autoregressive generation.

Reference: SPEC-005-memory-optimization.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig


@dataclass
class QuantizedTensor:
    """Container for INT4 quantized tensor with scale and zero-point.

    Why: INT4 quantization requires storing the quantized values along with
    per-group scale and zero-point for dequantization. This dataclass bundles
    them together for clean API.

    Attributes:
        data: Packed INT4 values as uint8 (2 values per byte)
        scale: FP16 scale factors for dequantization
        zero_point: FP16 zero points for asymmetric quantization
        shape: Original tensor shape before quantization
    """

    data: Tensor  # Packed INT4 as uint8, shape varies based on packing
    scale: Tensor  # FP16 scale factors, shape depends on quantization granularity
    zero_point: Tensor  # FP16 zero points, same shape as scale
    shape: tuple[int, ...]  # Original shape for unpacking


class INT4KVCache:
    """INT4 quantized KV-cache for memory-efficient 128K context.

    Why: Enables 128K context window within 16GB VRAM budget by reducing
    KV-cache memory from 67GB (FP16) to 8.4GB (INT4). Uses asymmetric
    quantization with different granularity for keys vs values based on
    KIVI paper findings.

    Embedding-Prediction Context: KV-cache stores attention key/value states
    in continuous embedding space. INT4 quantization introduces some precision
    loss but maintains sufficient fidelity for autoregressive generation. The
    dequantized values are used in attention computation before being discarded.

    Design decisions:
    - Asymmetric quantization: Uses zero_point for better dynamic range
    - Per-channel keys: Preserves attention pattern structure across head_dim
    - Per-token values: Better reconstruction for varied token content
    - Packed storage: 2 INT4 values per uint8 byte halves storage

    Attributes:
        config: TritterConfig instance
        max_seq_len: Maximum sequence length for cache
        keys: List of quantized key tensors per layer
        values: List of quantized value tensors per layer
        current_len: Current number of cached positions

    Example:
        >>> config = TritterConfig(model_size="7B", int4_kv_cache=True)
        >>> cache = INT4KVCache(config, max_seq_len=131072)
        >>> # During attention
        >>> cache.update(layer_idx=0, key=k, value=v)
        >>> k_cached, v_cached = cache.get(layer_idx=0)
    """

    # INT4 range: [-8, 7] for signed, [0, 15] for unsigned
    # We use signed for better centering around zero
    QMIN: int = -8
    QMAX: int = 7
    NUM_BITS: int = 4

    def __init__(
        self,
        config: TritterConfig,
        max_seq_len: int | None = None,
        batch_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize INT4 KV-cache.

        Args:
            config: TritterConfig with model architecture settings
            max_seq_len: Maximum sequence length (defaults to config.max_position_embeddings)
            batch_size: Batch size for cache
            device: Target device for cache tensors

        Why: Pre-allocate cache structure to avoid dynamic allocation during
        inference. Cache is initially empty (current_len=0) and grows as
        tokens are generated.
        """
        self.config = config
        self.max_seq_len = max_seq_len or config.max_position_embeddings
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Model dimensions from config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Initialize empty cache lists (populated on first update)
        # Why: Lazy initialization avoids pre-allocating massive tensors
        # when sliding window or early stopping might not need full cache.
        self.keys: list[QuantizedTensor | None] = [None] * self.num_layers
        self.values: list[QuantizedTensor | None] = [None] * self.num_layers

        # Track current cache length
        self.current_len = 0

    def _quantize_per_channel(self, tensor: Tensor) -> QuantizedTensor:
        """Quantize tensor to INT4 with per-channel (last dim) granularity.

        Args:
            tensor: FP16/FP32 tensor, shape (B, H, L, D) for keys

        Returns:
            QuantizedTensor with per-channel scales, shape (B, H, L, D/2) packed

        Why: Per-channel quantization (along head_dim) preserves relative
        magnitudes within each head dimension. This maintains attention pattern
        quality because queries attend to keys based on dot-product similarity,
        and channel-wise scaling preserves these relationships.
        """
        # tensor shape: (batch, heads, seq_len, head_dim)
        original_shape = tensor.shape

        # Compute min/max along last dimension (head_dim) for per-channel quant
        # Why: Each head dimension channel gets its own scale to preserve
        # the feature-wise distribution that attention relies on.
        min_val = tensor.amin(dim=-1, keepdim=True)  # (B, H, L, 1)
        max_val = tensor.amax(dim=-1, keepdim=True)  # (B, H, L, 1)

        # Compute scale and zero_point for asymmetric quantization
        # Why: Asymmetric quantization uses full INT4 range [-8, 7] by
        # computing separate scale and zero_point. This gives better
        # precision than symmetric quantization for non-centered data.
        scale = (max_val - min_val) / (self.QMAX - self.QMIN)
        # Prevent division by zero for constant channels
        scale = torch.clamp(scale, min=1e-8)
        zero_point = self.QMIN - min_val / scale

        # Quantize: q = round(x / scale + zero_point)
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, self.QMIN, self.QMAX)

        # Convert to unsigned for packing (shift from [-8,7] to [0,15])
        quantized_unsigned = (quantized - self.QMIN).to(torch.uint8)

        # Pack two INT4 values into one uint8
        # Why: Halves storage by packing adjacent values. head_dim must be even.
        packed = self._pack_int4(quantized_unsigned)

        return QuantizedTensor(
            data=packed,
            scale=scale.to(torch.float16).squeeze(-1),  # (B, H, L)
            zero_point=zero_point.to(torch.float16).squeeze(-1),  # (B, H, L)
            shape=original_shape,
        )

    def _quantize_per_token(self, tensor: Tensor) -> QuantizedTensor:
        """Quantize tensor to INT4 with per-token (seq dim) granularity.

        Args:
            tensor: FP16/FP32 tensor, shape (B, H, L, D) for values

        Returns:
            QuantizedTensor with per-token scales, shape (B, H, L, D/2) packed

        Why: Per-token quantization (along seq_len) gives each position its
        own scale. Values are weighted by attention scores, so per-token
        scaling ensures each position's contribution is accurately preserved
        regardless of content variation across the sequence.
        """
        # tensor shape: (batch, heads, seq_len, head_dim)
        original_shape = tensor.shape

        # Compute min/max along head_dim for per-token quant
        # Note: We flatten (H, D) and compute per-(B, L) for true per-token
        B, H, L, D = tensor.shape
        reshaped = tensor.permute(0, 2, 1, 3).reshape(B, L, H * D)  # (B, L, H*D)

        min_val = reshaped.amin(dim=-1, keepdim=True)  # (B, L, 1)
        max_val = reshaped.amax(dim=-1, keepdim=True)  # (B, L, 1)

        # Compute scale and zero_point
        scale = (max_val - min_val) / (self.QMAX - self.QMIN)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = self.QMIN - min_val / scale

        # Quantize
        quantized = torch.round(reshaped / scale + zero_point)
        quantized = torch.clamp(quantized, self.QMIN, self.QMAX)

        # Reshape back to (B, H, L, D)
        quantized = quantized.reshape(B, L, H, D).permute(0, 2, 1, 3)

        # Convert to unsigned and pack
        quantized_unsigned = (quantized - self.QMIN).to(torch.uint8)
        packed = self._pack_int4(quantized_unsigned)

        return QuantizedTensor(
            data=packed,
            scale=scale.to(torch.float16).squeeze(-1),  # (B, L)
            zero_point=zero_point.to(torch.float16).squeeze(-1),  # (B, L)
            shape=original_shape,
        )

    def _pack_int4(self, tensor: Tensor) -> Tensor:
        """Pack two INT4 values into one uint8.

        Args:
            tensor: uint8 tensor with values in [0, 15], shape (..., D)

        Returns:
            Packed tensor, shape (..., D//2)

        Why: INT4 uses 4 bits but torch.uint8 is minimum storage unit.
        Packing pairs reduces memory by 50%. Adjacent values along last
        dim are combined: high nibble + low nibble.
        """
        # Ensure last dim is even for packing
        assert tensor.shape[-1] % 2 == 0, (
            f"head_dim must be even for INT4 packing, got {tensor.shape[-1]}"
        )

        # Pack: even indices go to high nibble, odd to low nibble
        # tensor[..., ::2] << 4 | tensor[..., 1::2]
        high = tensor[..., ::2] << 4  # Even positions to high nibble
        low = tensor[..., 1::2]  # Odd positions to low nibble
        packed = (high | low).to(torch.uint8)

        return packed

    def _unpack_int4(self, packed: Tensor, last_dim: int) -> Tensor:
        """Unpack uint8 to two INT4 values.

        Args:
            packed: Packed uint8 tensor, shape (..., D//2)
            last_dim: Original last dimension size (for unpacking)

        Returns:
            Unpacked tensor, shape (..., D)

        Why: Reverses packing to restore original tensor structure for
        dequantization and attention computation.
        """
        # Extract high and low nibbles
        high = (packed >> 4) & 0x0F  # High nibble
        low = packed & 0x0F  # Low nibble

        # Interleave back to original order
        unpacked = torch.zeros(
            *packed.shape[:-1], last_dim, dtype=torch.uint8, device=packed.device
        )
        unpacked[..., ::2] = high
        unpacked[..., 1::2] = low

        return unpacked

    def _dequantize_per_channel(self, qt: QuantizedTensor) -> Tensor:
        """Dequantize per-channel quantized tensor back to FP16.

        Args:
            qt: QuantizedTensor from _quantize_per_channel

        Returns:
            Dequantized FP16 tensor in original shape

        Why: Attention computation requires FP16/FP32 tensors. Dequantization
        restores approximate original values using stored scale and zero_point.
        """
        # Unpack INT4 to uint8
        unpacked = self._unpack_int4(qt.data, qt.shape[-1])

        # Convert back to signed (shift [0,15] to [-8,7])
        signed = unpacked.to(torch.float16) + self.QMIN

        # Dequantize: x = (q - zero_point) * scale
        # scale and zero_point are (B, H, L), need to unsqueeze for broadcast
        scale = qt.scale.unsqueeze(-1)  # (B, H, L, 1)
        zero_point = qt.zero_point.unsqueeze(-1)  # (B, H, L, 1)

        dequantized = (signed - zero_point) * scale

        return dequantized

    def _dequantize_per_token(self, qt: QuantizedTensor) -> Tensor:
        """Dequantize per-token quantized tensor back to FP16.

        Args:
            qt: QuantizedTensor from _quantize_per_token

        Returns:
            Dequantized FP16 tensor in original shape

        Why: Same as per-channel but handles different scale granularity.
        Per-token scales are (B, L) and must broadcast to (B, H, L, D).
        """
        B, H, L, D = qt.shape

        # Unpack INT4 to uint8
        unpacked = self._unpack_int4(qt.data, D)

        # Convert back to signed
        signed = unpacked.to(torch.float16) + self.QMIN

        # Dequantize with per-token scales
        # scale is (B, L), need to reshape for broadcast to (B, H, L, D)
        scale = qt.scale.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
        zero_point = qt.zero_point.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)

        dequantized = (signed - zero_point) * scale

        return dequantized

    def update(
        self,
        layer_idx: int,
        key: Tensor,
        value: Tensor,
    ) -> None:
        """Update cache with new key/value tensors for a layer.

        Args:
            layer_idx: Index of transformer layer (0-indexed)
            key: New key tensor, shape (batch, heads, new_len, head_dim)
            value: New value tensor, shape (batch, heads, new_len, head_dim)

        Why: During autoregressive generation, each step produces new K/V
        that must be cached. This method handles quantization and appending
        to existing cache.

        Design: For efficiency, we quantize and store the entire new sequence
        in one operation rather than per-token to amortize quantization overhead.
        """
        # Quantize new tensors
        key_quantized = self._quantize_per_channel(key)
        value_quantized = self._quantize_per_token(value)

        if self.keys[layer_idx] is None:
            # First update - store directly
            self.keys[layer_idx] = key_quantized
            self.values[layer_idx] = value_quantized
            self.current_len = key.shape[2]
        else:
            # Append to existing cache
            # Why: For incremental updates, concatenate along seq dimension.
            # This requires unpacking, concatenating, and repacking, which
            # adds overhead. For production, consider ring buffer approach.
            existing_k = self._dequantize_per_channel(self.keys[layer_idx])  # type: ignore[arg-type]
            existing_v = self._dequantize_per_token(self.values[layer_idx])  # type: ignore[arg-type]

            combined_k = torch.cat([existing_k, key], dim=2)
            combined_v = torch.cat([existing_v, value], dim=2)

            self.keys[layer_idx] = self._quantize_per_channel(combined_k)
            self.values[layer_idx] = self._quantize_per_token(combined_v)
            self.current_len = combined_k.shape[2]

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Retrieve dequantized key/value tensors for a layer.

        Args:
            layer_idx: Index of transformer layer (0-indexed)

        Returns:
            Tuple of (key, value) as FP16 tensors, shape (B, H, L, D)

        Raises:
            ValueError: If layer has no cached values

        Why: Attention computation needs FP16 tensors. Dequantization is
        performed on-demand to minimize memory footprint (only one layer's
        dequantized K/V in memory at a time).
        """
        if self.keys[layer_idx] is None or self.values[layer_idx] is None:
            raise ValueError(f"No cached values for layer {layer_idx}")

        key = self._dequantize_per_channel(self.keys[layer_idx])  # type: ignore[arg-type]
        value = self._dequantize_per_token(self.values[layer_idx])  # type: ignore[arg-type]

        return key, value

    def clear(self) -> None:
        """Clear all cached values.

        Why: Reset cache between sequences or when context is full.
        Releases memory held by quantized tensors.
        """
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self.current_len = 0

    def get_memory_usage_gb(self) -> float:
        """Calculate current memory usage of cache in GB.

        Returns:
            Memory usage in gigabytes

        Why: Memory monitoring for budget enforcement. Useful for deciding
        when to trigger cache eviction or sliding window truncation.
        """
        total_bytes = 0

        for k, v in zip(self.keys, self.values, strict=False):
            if k is not None:
                # Packed data + scales + zero_points
                total_bytes += k.data.numel()  # uint8, 1 byte each
                total_bytes += k.scale.numel() * 2  # FP16, 2 bytes each
                total_bytes += k.zero_point.numel() * 2
            if v is not None:
                total_bytes += v.data.numel()
                total_bytes += v.scale.numel() * 2
                total_bytes += v.zero_point.numel() * 2

        return total_bytes / 1e9

    def truncate_to_window(self, window_size: int) -> None:
        """Truncate cache to keep only recent window_size positions.

        Args:
            window_size: Number of recent positions to keep

        Why: For sliding window attention, older positions are evicted to
        bound memory. This maintains recent context while allowing unbounded
        generation length.
        """
        if self.current_len <= window_size:
            return  # Nothing to truncate

        for i in range(self.num_layers):
            if self.keys[i] is not None:
                # Dequantize, slice, re-quantize
                k = self._dequantize_per_channel(self.keys[i])  # type: ignore[arg-type]
                v = self._dequantize_per_token(self.values[i])  # type: ignore[arg-type]

                # Keep only last window_size positions
                k = k[:, :, -window_size:, :]
                v = v[:, :, -window_size:, :]

                self.keys[i] = self._quantize_per_channel(k)
                self.values[i] = self._quantize_per_token(v)

        self.current_len = window_size

    def __repr__(self) -> str:
        return (
            f"INT4KVCache("
            f"layers={self.num_layers}, "
            f"heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"current_len={self.current_len}, "
            f"max_len={self.max_seq_len}, "
            f"memory={self.get_memory_usage_gb():.3f}GB)"
        )


__all__ = [
    "INT4KVCache",
    "QuantizedTensor",
]
