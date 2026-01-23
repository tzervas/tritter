# Memory Optimization Specification

**Spec ID**: SPEC-005
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-22
**Target Hardware**: NVIDIA RTX 5080 16GB GDDR7

## 1. Overview

### 1.1 Purpose

This specification defines the memory optimization strategy for Tritter to enable a 7B parameter model with 128K context window on consumer GPU hardware (RTX 5080 16GB).

### 1.2 Memory Budget

| Component | Budget | Technique |
|-----------|--------|-----------|
| Model weights (7B) | 1.4 GB | BitNet 1.58-bit quantization |
| KV-cache (128K) | 8-10 GB | INT4 quantization + sliding window |
| Activations | 2-3 GB | Gradient checkpointing |
| Vision encoder | 0.4 GB | SigLIP-B frozen |
| Overhead | 1-2 GB | CUDA runtime, fragmentation |
| **Total** | **~14 GB** | Within 16 GB budget |

### 1.3 Key Constraints

- **Hard limit**: 16 GB VRAM (RTX 5080)
- **Context target**: 128K tokens
- **Batch size**: At least 1 for inference, flexible for training
- **No CPU offload**: Latency-sensitive inference

---

## 2. Weight Quantization

### 2.1 BitNet b1.58 Strategy

**Compression Ratio**: ~10x (14 GB FP16 → 1.4 GB ternary)

**Implementation**: `src/tritter/quantization/bitnet.py`

```
Standard FP16:  16 bits/parameter × 7B = 14 GB
BitNet 1.58:    1.58 bits/parameter × 7B = 1.4 GB
```

**Ternary Weight Format**:
- Values: {-1, 0, +1}
- Storage: 2 bits per weight (packed)
- Scale factor: FP16 per tensor

**Memory Layout**:
```
┌─────────────────────────────────────────────┐
│ Packed Weights: 2 bits × num_parameters     │
│ Scale Factors: FP16 × num_tensors           │
│ Shadow Weights (training only): FP32        │
└─────────────────────────────────────────────┘
```

### 2.2 Inference vs Training Memory

| Mode | Weight Memory | Shadow Weights | Total |
|------|--------------|----------------|-------|
| Inference | 1.4 GB | 0 | 1.4 GB |
| Training | 1.4 GB | 14 GB (FP32) | 15.4 GB |

**Implication**: Training 7B requires memory optimization or smaller batches.

---

## 3. KV-Cache Optimization

### 3.1 KV-Cache Memory Formula

```
KV_bytes = 2 × layers × heads × head_dim × seq_len × batch × bytes_per_element
```

**7B Model Configuration**:
- layers = 32
- heads = 32
- head_dim = 128
- seq_len = 131072 (128K)

**Memory by Precision**:

| Precision | bytes_per_element | Memory (128K, batch=1) |
|-----------|-------------------|------------------------|
| FP16 | 2 | 67 GB (impossible) |
| FP8 | 1 | 34 GB (impossible) |
| INT4 | 0.5 | 8.4 GB (fits!) |
| INT2 | 0.25 | 4.2 GB (aggressive) |

### 3.2 INT4 KV-Cache Strategy

**Quantization Approach** (per KIVI paper):

| Tensor | Quantization | Why |
|--------|-------------|-----|
| Keys | Per-channel | Preserves attention patterns |
| Values | Per-token | Better reconstruction |

**Implementation Plan**:

```python
class INT4KVCache:
    """INT4 quantized KV-cache for 128K context.

    Why: FP16 KV-cache for 128K would require 67 GB.
    INT4 reduces this to 8.4 GB, fitting in 16 GB budget.

    Quantization strategy:
    - Keys: Per-channel (dim=-1) quantization
    - Values: Per-token (dim=1) quantization
    - Scale/zero-point stored as FP16
    """

    def quantize_key(self, key: torch.Tensor) -> QuantizedTensor:
        """Quantize key tensor to INT4.

        Args:
            key: (batch, heads, seq_len, head_dim) FP16 tensor

        Returns:
            Quantized tensor with per-channel scales

        Why per-channel: Keys define attention patterns. Per-channel
        quantization preserves relative magnitudes within each head
        dimension, maintaining attention distribution quality.
        """
        pass

    def quantize_value(self, value: torch.Tensor) -> QuantizedTensor:
        """Quantize value tensor to INT4.

        Args:
            value: (batch, heads, seq_len, head_dim) FP16 tensor

        Returns:
            Quantized tensor with per-token scales

        Why per-token: Values are weighted by attention. Per-token
        quantization ensures each position's contribution is preserved
        with its own scale, improving reconstruction for varied content.
        """
        pass
```

### 3.3 Sliding Window KV-Cache

**With 4K sliding window**:

```
KV_bytes = 2 × 32 × 32 × 128 × 4096 × 1 × 0.5 = 0.5 GB
```

**Comparison**:

| Strategy | Context | KV Memory | Notes |
|----------|---------|-----------|-------|
| Full INT4 | 128K | 8.4 GB | Maximum context |
| Sliding 4K INT4 | Effective 128K* | 0.5 GB | Bounded memory |
| Sliding 8K INT4 | Effective 256K* | 1.0 GB | More context |

*Effective context via sliding window + attention sinks

---

## 4. Activation Memory

### 4.1 Activation Memory Analysis

For forward pass (no gradient):
```
Activations ≈ batch × seq_len × hidden_size × num_layers × bytes
           = 1 × 131072 × 4096 × 32 × 2
           = 34 GB (impossible without optimization)
```

### 4.2 Gradient Checkpointing

**Strategy**: Checkpoint every N layers, recompute during backward.

| Checkpoint Frequency | Memory Reduction | Compute Overhead |
|---------------------|------------------|------------------|
| Every layer | ~90% | 33% |
| Every 4 layers | ~75% | 8% |
| Every 8 layers | ~60% | 4% |

**Recommended**: Checkpoint every 4 layers (8 checkpoints for 32 layers)

```python
class TritterModel(nn.Module):
    def forward(self, input_ids):
        hidden = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            if self.training and i % 4 == 0:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, use_reentrant=False
                )
            else:
                hidden = layer(hidden)

        return self.output_projection(hidden)
```

### 4.3 Inference Activation Memory

Without gradients, activation memory is much smaller:

```
Peak activations ≈ batch × seq_len × hidden_size × 2 (current + next)
                = 1 × 131072 × 4096 × 2 × 2
                = 2.1 GB
```

---

## 5. Memory Profiling Interface

### 5.1 Memory Tracker Specification

**Location**: `src/tritter/utils/memory_utils.py`

```python
@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state.

    Attributes:
        allocated: Currently allocated memory (GB)
        reserved: Reserved by allocator (GB)
        peak: Peak allocated since reset (GB)
        timestamp: When snapshot was taken
    """
    allocated: float
    reserved: float
    peak: float
    timestamp: float


class MemoryTracker:
    """Track GPU memory usage during model operations.

    Why: Memory profiling is essential for:
    1. Verifying model fits in 16 GB budget
    2. Identifying memory leaks
    3. Optimizing batch sizes
    4. Debugging OOM errors

    Usage:
        tracker = MemoryTracker()
        tracker.reset()
        model(input_ids)
        snapshot = tracker.snapshot()
        print(f"Peak: {snapshot.peak:.2f} GB")
    """

    def reset(self) -> None:
        """Reset peak memory tracking."""
        torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> MemorySnapshot:
        """Take current memory snapshot."""
        return MemorySnapshot(
            allocated=torch.cuda.memory_allocated() / 1e9,
            reserved=torch.cuda.memory_reserved() / 1e9,
            peak=torch.cuda.max_memory_allocated() / 1e9,
            timestamp=time.time(),
        )

    def check_budget(self, budget_gb: float = 15.0) -> bool:
        """Verify current usage within budget.

        Args:
            budget_gb: Maximum allowed memory (default 15 GB for headroom)

        Returns:
            True if within budget

        Raises:
            MemoryBudgetExceeded: If peak exceeds budget
        """
        snapshot = self.snapshot()
        if snapshot.peak > budget_gb:
            raise MemoryBudgetExceeded(
                f"Peak memory {snapshot.peak:.2f} GB exceeds budget {budget_gb} GB"
            )
        return True
```

### 5.2 Memory Budget Calculator

```python
def calculate_memory_budget(config: TritterConfig) -> dict[str, float]:
    """Calculate expected memory usage for configuration.

    Args:
        config: Model configuration

    Returns:
        Dictionary with memory breakdown in GB

    Why: Pre-flight check before training/inference to verify
    configuration will fit in available VRAM.
    """
    # Weight memory
    params = estimate_parameters(config)
    if config.use_bitnet:
        weight_gb = params * 1.58 / 8 / 1e9  # 1.58 bits per param
    else:
        weight_gb = params * 2 / 1e9  # FP16

    # KV-cache memory
    kv_elements = (
        2  # K and V
        * config.num_layers
        * config.num_heads
        * config.head_dim
        * config.max_position_embeddings
    )
    if config.int4_kv_cache:
        kv_gb = kv_elements * 0.5 / 1e9  # INT4
    else:
        kv_gb = kv_elements * 2 / 1e9  # FP16

    # Apply sliding window if enabled
    if config.use_sliding_window:
        window_ratio = config.sliding_window_size / config.max_position_embeddings
        kv_gb *= window_ratio

    # Activation memory (inference estimate)
    activation_gb = (
        config.max_position_embeddings
        * config.hidden_size
        * 2  # FP16
        * 2  # Current + workspace
        / 1e9
    )

    return {
        "weights": weight_gb,
        "kv_cache": kv_gb,
        "activations": activation_gb,
        "overhead": 1.5,  # CUDA runtime, fragmentation
        "total": weight_gb + kv_gb + activation_gb + 1.5,
    }
```

---

## 6. Configuration Recommendations

### 6.1 Inference Configurations

| Configuration | Context | Batch | Memory | Use Case |
|--------------|---------|-------|--------|----------|
| Maximum context | 128K | 1 | ~14 GB | Long document analysis |
| Balanced | 32K | 4 | ~12 GB | General use |
| Throughput | 8K | 16 | ~10 GB | High throughput serving |
| Streaming | 4K window | 1 | ~4 GB | Streaming inference |

### 6.2 Training Configurations

| Configuration | Context | Batch | Grad Ckpt | Memory | Use Case |
|--------------|---------|-------|-----------|--------|----------|
| Full context | 128K | 1 | Every 4 | ~15 GB | Pretraining |
| Efficient | 32K | 4 | Every 4 | ~14 GB | Fine-tuning |
| Development | 8K | 8 | None | ~12 GB | Rapid iteration |

---

## 7. OOM Prevention

### 7.1 Automatic Memory Management

```python
class MemoryManager:
    """Automatic memory management for OOM prevention.

    Why: GPU OOM kills training without recovery. Proactive
    management prevents wasted compute and enables graceful
    handling of memory pressure.
    """

    def __init__(self, budget_gb: float = 15.0, headroom_gb: float = 1.0):
        self.budget = budget_gb
        self.headroom = headroom_gb

    def can_allocate(self, size_gb: float) -> bool:
        """Check if allocation is safe."""
        current = torch.cuda.memory_allocated() / 1e9
        return current + size_gb < self.budget - self.headroom

    def clear_cache(self) -> None:
        """Clear CUDA cache to free fragmented memory."""
        torch.cuda.empty_cache()
        gc.collect()

    @contextmanager
    def managed_forward(self, model, input_ids):
        """Context manager for memory-safe forward pass.

        Usage:
            with memory_manager.managed_forward(model, input_ids) as output:
                loss = compute_loss(output)
        """
        try:
            yield model(input_ids)
        except torch.cuda.OutOfMemoryError:
            self.clear_cache()
            # Retry with gradient checkpointing
            with torch.utils.checkpoint.checkpoint_sequential(model.layers, 8):
                yield model(input_ids)
```

### 7.2 Graceful Degradation

When memory pressure is detected:

1. **Clear cache**: `torch.cuda.empty_cache()`
2. **Enable checkpointing**: Increase checkpoint frequency
3. **Reduce batch size**: Halve batch, retry
4. **Reduce context**: Truncate to sliding window only
5. **Fail gracefully**: Log error with memory stats

---

## 8. Verification Tests

### 8.1 Memory Budget Tests

```python
@pytest.mark.gpu
class TestMemoryBudget:
    """Verify memory stays within RTX 5080 budget."""

    def test_7b_inference_fits(self):
        """7B inference must fit in 15 GB."""
        config = TritterConfig(model_size="7B", use_bitnet=True)
        model = TritterModel(config).cuda()

        tracker = MemoryTracker()
        tracker.reset()

        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (1, 8192), device="cuda")
            _ = model(input_ids)

        tracker.check_budget(15.0)

    def test_kv_cache_scaling(self):
        """KV-cache must scale correctly with sequence length."""
        config = TritterConfig(model_size="7B", int4_kv_cache=True)
        budget = calculate_memory_budget(config)

        assert budget["kv_cache"] < 10.0, "KV-cache exceeds 10 GB budget"
```

---

## Appendix A: RTX 5080 Specifications

| Specification | Value |
|--------------|-------|
| VRAM | 16 GB GDDR7 |
| Bandwidth | 960 GB/s |
| CUDA Cores | 10,752 |
| Tensor Cores | 336 |
| Compute Capability | 9.0 (Blackwell) |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-22 | Claude | Initial draft |
