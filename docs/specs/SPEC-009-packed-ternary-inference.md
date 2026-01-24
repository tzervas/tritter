# SPEC-009: Packed Ternary Inference

## Status

**Implemented** - 2026-01-23

## Summary

Pack ternary weights {-1, 0, +1} from 32 bits to 2 bits per value, reducing 7B model weight storage from ~28GB to ~1.4GB. This enables 7B model inference on RTX 5080 16GB VRAM.

## Problem Statement

Current `TernaryWeight` stores FP32 shadow weights for STE gradient flow during training:
- 7B model = ~7 billion FP32 weights = ~28GB
- RTX 5080 has 16GB VRAM
- Need ~8GB for KV-cache (128K context, INT4)
- Need ~3GB for activations and overhead

**Result**: Cannot run 7B inference on RTX 5080 with current implementation.

## Solution

### Encoding Scheme

Ternary values only need log2(3) ≈ 1.58 bits. We use 2-bit encoding for efficiency:

```
Value   Encoded
-1  ->  0b00 (0)
 0  ->  0b01 (1)
+1  ->  0b10 (2)
```

Four values pack into one byte:
```
byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
```

Unpacking uses bitwise operations:
```python
v0 = byte & 0x03
v1 = (byte >> 2) & 0x03
v2 = (byte >> 4) & 0x03
v3 = (byte >> 6) & 0x03
# Then decode: value = encoded - 1
```

### Memory Analysis

| Component | Size |
|-----------|------|
| 7B weights (packed) | ~1.75GB (7B × 0.25 bytes) |
| Per-channel scales | ~7MB (~1.7M channels × 4 bytes) |
| **Total packed** | **~1.8GB** |

Compared to:
- FP32: 28GB (15.5x larger)
- INT8: 7GB (3.9x larger)

### API Design

#### Core Functions

```python
def pack_ternary(weights: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ternary weights into uint8.

    Args:
        weights: (out_features, in_features) tensor with values in {-1, 0, +1}
        scale: (out_features, 1) per-channel scale

    Returns:
        packed: (out_features, ceil(in_features/4)) uint8 tensor
        scale: unchanged scale tensor
    """

def unpack_ternary(packed: torch.Tensor, scale: torch.Tensor,
                   original_in_features: int) -> torch.Tensor:
    """Unpack uint8 to scaled ternary weights.

    Returns:
        (out_features, in_features) FP32 tensor
    """
```

#### PackedTernaryWeight Module

```python
class PackedTernaryWeight(nn.Module):
    """Inference-only ternary layer with packed storage.

    Stores weights as uint8 buffers (not parameters).
    Unpacks on-the-fly during forward pass.
    """

    @classmethod
    def from_ternary_weight(cls, ternary: TernaryWeight) -> "PackedTernaryWeight":
        """Convert trained TernaryWeight to packed format."""
```

#### Model Conversion

```python
def convert_to_packed(model: nn.Module) -> nn.Module:
    """Recursively replace TernaryWeight with PackedTernaryWeight."""
```

### Integration with Layer Streaming

`PackedTernaryWeight` integrates seamlessly with the existing layer streaming infrastructure:

1. **LayerLoader**: No changes needed - treats packed weights as regular buffers
2. **TransferEngine**: Benefits from ~8x smaller CPU→GPU transfers
3. **MemoryManager**: Same API, just smaller weight footprint

### Workflow

Training to deployment:

```
1. Train with TernaryWeight (FP32 shadow weights, STE gradients)
   ↓
2. Save trained model (FP32 checkpoint)
   ↓
3. Load and convert: model = convert_to_packed(model)
   ↓
4. Save packed model: save_packed_model(model, "7b_packed.pt")
   ↓
5. Deploy: load_packed_model("7b_packed.pt", model)
```

## Files

| File | Purpose |
|------|---------|
| `src/tritter/quantization/packed_ternary.py` | Core implementation |
| `tests/unit/test_packed_ternary.py` | Unit tests |
| `tests/integration/test_packed_inference.py` | Integration tests |
| `scripts/benchmark_packed.py` | Benchmarks and memory verification |

## Testing

### Unit Tests

1. Pack/unpack round-trip preserves exact values
2. PackedTernaryWeight output matches TernaryWeight
3. Memory savings verified (~16x vs FP32)
4. Padding/truncation for non-divisible-by-4 sizes

### Integration Tests

1. Full model conversion pipeline
2. Streaming inference with packed weights
3. Save/load workflow
4. Edge cases (single layer, batch=1, seq=1)

### Manual Verification

```bash
# Run benchmarks (CPU-only, fast)
python scripts/benchmark_packed.py --memory

# Verify 7B memory estimate
python scripts/benchmark_packed.py --verify-7b

# Full GPU benchmark (requires CUDA)
python scripts/benchmark_packed.py --full
```

## Performance Characteristics

### Memory

- Weight storage: ~16x reduction vs FP32
- Transfer time: ~8x faster (smaller data over PCIe)
- Forward pass: Unpacking adds ~1-5% overhead

### Computation

Unpacking on GPU is cheap:
- 2 bitwise operations per value
- Fully vectorized
- Dominated by matmul time anyway

## Training vs Inference Memory

### Training (with STE)

Training requires FP32 shadow weights for straight-through estimator gradient flow:

| Component | Memory |
|-----------|--------|
| FP32 shadow weights | ~28GB |
| Optimizer states (AdamW) | ~56GB (2x weights) |
| Gradients | ~28GB |
| **Minimum for training** | **~112GB+** |

**Implication**: Training 7B requires multi-GPU setup or model parallelism.
For single-GPU training, use smaller models (3B) or reduce context window significantly.

### Inference (with packing)

Packed weights enable single-GPU inference:

| Component | Memory |
|-----------|--------|
| Packed weights | ~1.6GB |
| KV-cache (context-dependent) | Varies |
| Activations | ~0.5GB |

KV-cache sizes (INT4, batch=1):
- 4K context: ~0.5GB
- 32K context: ~4GB
- 128K context: ~16GB

**Recommendation**: For RTX 5080 16GB, use ~32K context or implement sliding window attention for longer contexts.

## Limitations

1. **Inference only**: No gradient support (use TernaryWeight for training)
2. **Fixed precision**: Exactly 2 bits, no adaptive precision
3. **Pure PyTorch**: No custom CUDA kernels (could optimize further with Triton)
4. **KV-cache dominates**: For long contexts, KV-cache is the memory bottleneck, not weights

## Future Work

1. **Triton kernel**: Fused unpack + matmul for ~10% speedup
2. **INT4 variant**: For even more compression on non-ternary models
3. **Streaming packer**: Pack weights during transfer for memory-constrained hosts

## References

- BitNet b1.58: "The Era of 1-bit LLMs"
- SPEC-006: Progressive Layer Loading
- ADR-002: Progressive Layer Loading Decision
