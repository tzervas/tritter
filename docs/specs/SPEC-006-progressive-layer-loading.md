# Progressive Layer Loading Specification

**Spec ID**: SPEC-006
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-23
**ADR Reference**: [ADR-002](../adr/002-progressive-layer-loading.md)
**Target Module**: `src/tritter/inference/layer_streaming.py`

## 1. Overview

### 1.1 Purpose

This specification defines the progressive layer loading mechanism for Tritter, enabling inference of models larger than available GPU VRAM by streaming layer groups through GPU memory.

### 1.2 Goals

- Enable 70B+ model inference on 16GB VRAM
- Minimize latency impact through pipelining
- Maintain compatibility with existing TritterModel API
- Support both synchronous and asynchronous loading modes

### 1.3 Dependencies

- PyTorch >= 2.0 (CUDA streams, pinned memory)
- CUDA >= 12.0 (async memory operations)
- Existing `TritterConfig` and `TritterModel` architecture
- KV-cache implementation (SPEC-005)

### 1.4 References

| Reference | Description |
|-----------|-------------|
| [ADR-002](../adr/002-progressive-layer-loading.md) | Architecture decision for progressive loading |
| [SPEC-005](./SPEC-005-memory-optimization.md) | Memory optimization and KV-cache |
| [FlexGen Paper](https://arxiv.org/abs/2303.06865) | Offloading strategies |
| [vLLM Paper](https://arxiv.org/abs/2309.06180) | PagedAttention patterns |

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-001 | Layer group loading | Must Have | Load N layers from CPU to GPU as atomic unit |
| FR-002 | Layer group eviction | Must Have | Free GPU memory after layer group processing |
| FR-003 | Double buffering | Should Have | Load next group while processing current |
| FR-004 | Configurable group size | Must Have | `layer_group_size` in TritterConfig |
| FR-005 | Memory budget enforcement | Must Have | Never exceed configured GPU memory limit |
| FR-006 | KV-cache persistence | Must Have | KV-cache remains in VRAM across layer groups |
| FR-007 | Sync/async modes | Should Have | Support both blocking and pipelined loading |
| FR-008 | Progress callbacks | Could Have | Notify caller of loading progress |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target | Verification |
|----|-------------|--------|--------------|
| NFR-001 | Transfer efficiency | >90% PCIe bandwidth utilization | Benchmark |
| NFR-002 | Memory overhead | <5% of GPU memory for management | Profile |
| NFR-003 | Latency hiding | >80% compute/transfer overlap | Timeline trace |
| NFR-004 | First token latency | <2x vs full-resident model | Benchmark |

### 2.3 Constraints

- **C-001**: Must not modify layer weights (read-only streaming)
- **C-002**: Must maintain numerical equivalence with full-resident inference
- **C-003**: Must work with BitNet quantized weights
- **C-004**: Must support gradient-free inference only (training uses full-resident)

---

## 3. Architecture

### 3.1 Module Structure

```
src/tritter/inference/
├── __init__.py
├── layer_streaming.py      # NEW: Core streaming logic
├── memory_manager.py       # NEW: GPU memory allocation
├── transfer_engine.py      # NEW: Async H2D transfers
└── kv_cache.py            # Existing: KV-cache management
```

### 3.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    StreamingInferenceEngine                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  LayerLoader    │    │ MemoryManager   │                     │
│  │                 │    │                 │                     │
│  │ - load_group()  │    │ - allocate()    │                     │
│  │ - evict_group() │    │ - free()        │                     │
│  │ - prefetch()    │    │ - get_budget()  │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌─────────────────────────────────────────┐                    │
│  │           TransferEngine                │                    │
│  │                                         │                    │
│  │  - H2D stream (compute)                 │                    │
│  │  - D2H stream (optional)                │                    │
│  │  - Pinned memory pool                   │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  ┌─────────────────────────────────────────┐                    │
│  │           LayerGroupBuffer              │                    │
│  │                                         │                    │
│  │  Buffer A (active)  │  Buffer B (next)  │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow

```
Forward Pass with Layer Streaming:

Input: hidden_states (B, L, D), layer_weights on CPU

for group_idx in range(num_groups):
    ┌──────────────────────────────────────────────────────────┐
    │ 1. PREFETCH (async): Load group[i+1] to Buffer B         │
    │    transfer_engine.prefetch_async(group_idx + 1)         │
    ├──────────────────────────────────────────────────────────┤
    │ 2. COMPUTE: Process through group[i] layers in Buffer A  │
    │    for layer in buffer_a.layers:                         │
    │        hidden_states = layer(hidden_states, kv_cache)    │
    ├──────────────────────────────────────────────────────────┤
    │ 3. SYNC: Wait for prefetch completion                    │
    │    transfer_engine.sync()                                │
    ├──────────────────────────────────────────────────────────┤
    │ 4. SWAP: Exchange buffer roles (A↔B)                     │
    │    buffer_a, buffer_b = buffer_b, buffer_a               │
    └──────────────────────────────────────────────────────────┘

Output: hidden_states after all layers
```

---

## 4. Interface Specification

### 4.1 TritterConfig Extensions

```python
@dataclass
class TritterConfig:
    # ... existing fields ...

    # Progressive Layer Loading
    use_layer_streaming: bool = False
    """Enable progressive layer loading for large model inference.

    Why: Allows running models larger than GPU VRAM by streaming layer
    groups through memory. Set to True for models that don't fit in VRAM.
    """

    layer_group_size: int = 4
    """Number of layers to load per group.

    Why: Larger groups reduce transfer overhead but require more VRAM.
    Optimal value depends on model size and available memory.

    Constraints:
        - Must be > 0
        - Should divide num_layers evenly (or remainder handled)
        - Typical range: 2-8 layers
    """

    gpu_memory_budget_gb: float = 14.0
    """Maximum GPU memory to use for inference (in GB).

    Why: Reserves headroom for CUDA overhead and unexpected allocations.
    Typically set to 85-90% of total VRAM.

    Constraints:
        - Must be > 0
        - Should be less than physical VRAM
    """

    use_pinned_memory: bool = True
    """Use CUDA pinned memory for faster host-to-device transfers.

    Why: Pinned memory enables async DMA transfers at full PCIe bandwidth.
    Disable if system RAM is limited.
    """

    prefetch_next_group: bool = True
    """Prefetch next layer group while processing current group.

    Why: Overlaps transfer with computation to hide latency.
    Requires double the layer group memory but significantly improves throughput.
    """
```

### 4.2 LayerLoader Class

```python
class LayerLoader:
    """Manages loading and eviction of layer groups for streaming inference.

    Why: Centralizes layer memory management to enable models larger than
    VRAM. Uses double buffering and async transfers to minimize latency
    impact from layer streaming.

    Embedding-Prediction Context: Layer weights are read-only during inference.
    The model operates in continuous embedding space, transforming embeddings
    through each layer without needing to modify weights.
    """

    def __init__(
        self,
        model: TritterModel,
        config: TritterConfig,
        device: torch.device,
    ) -> None:
        """Initialize layer loader.

        Args:
            model: TritterModel with layers on CPU
            config: Configuration with streaming settings
            device: Target GPU device

        Why: Prepares infrastructure for streaming without immediately
        loading layers. Layers remain on CPU until explicitly loaded.
        """

    def load_group(self, group_idx: int) -> list[TritterLayer]:
        """Load a layer group to GPU synchronously.

        Args:
            group_idx: Index of layer group to load (0-indexed)

        Returns:
            List of layers now resident on GPU

        Why: Synchronous loading for simple use cases or initial load.
        For pipelined inference, use prefetch_async() instead.
        """

    def evict_group(self, group_idx: int) -> None:
        """Remove layer group from GPU memory.

        Args:
            group_idx: Index of layer group to evict

        Why: Frees GPU memory for next layer group. Must be called
        after processing to prevent OOM with large models.
        """

    def prefetch_async(self, group_idx: int) -> None:
        """Start async prefetch of layer group.

        Args:
            group_idx: Index of layer group to prefetch

        Why: Enables compute/transfer overlap. Call while processing
        current group to hide transfer latency.
        """

    def sync(self) -> None:
        """Wait for pending async operations to complete.

        Why: Ensures prefetched layers are ready before use.
        Must be called before accessing prefetched layers.
        """

    @property
    def num_groups(self) -> int:
        """Total number of layer groups."""

    @property
    def layers_per_group(self) -> int:
        """Number of layers in each group."""
```

### 4.3 StreamingInferenceEngine Class

```python
class StreamingInferenceEngine:
    """High-level inference engine with automatic layer streaming.

    Why: Provides simple API for streaming inference without manual
    layer management. Handles buffering, prefetching, and eviction
    automatically based on configuration.

    Usage:
        engine = StreamingInferenceEngine(model, config)
        output = engine.generate(input_ids, max_new_tokens=100)
    """

    def __init__(
        self,
        model: TritterModel,
        config: TritterConfig,
    ) -> None:
        """Initialize streaming inference engine.

        Args:
            model: TritterModel (weights will be moved to CPU if needed)
            config: Configuration with streaming settings
        """

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Forward pass with automatic layer streaming.

        Args:
            hidden_states: Input embeddings (batch, seq_len, hidden_size)
            kv_cache: Optional KV-cache for incremental decoding

        Returns:
            Output hidden states after all layers
        """

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate tokens with streaming inference.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
```

---

## 5. Memory Budget Analysis

### 5.1 Memory Components

| Component | Size | Persistence |
|-----------|------|-------------|
| KV-cache (128K, INT4) | ~8-10 GB | Persistent |
| Active layer group (4 layers, 7B) | ~0.2 GB | Transient |
| Prefetch buffer | ~0.2 GB | Transient |
| Activations | ~1 GB | Per-forward |
| CUDA overhead | ~1 GB | Persistent |
| **Total** | **~11-13 GB** | |

### 5.2 Scaling Analysis

| Model Size | Weights (BitNet) | Groups (4 layers) | Transfer Time |
|------------|------------------|-------------------|---------------|
| 7B | 1.4 GB | 6-8 | ~44ms total |
| 13B | 2.6 GB | 10-12 | ~81ms total |
| 30B | 6.0 GB | 20-24 | ~188ms total |
| 70B | 14.0 GB | 40-48 | ~438ms total |

*Transfer times assume PCIe 5.0 at 32 GB/s with 100% efficiency*

### 5.3 Throughput Impact

With pipelining (compute/transfer overlap):
- **7B**: <5% throughput reduction (transfer hidden by compute)
- **13B**: ~10% reduction
- **30B**: ~25% reduction
- **70B**: ~50% reduction (transfer-bound)

---

## 6. Test Specification

### 6.1 Unit Tests

| Test ID | Description | Verification |
|---------|-------------|--------------|
| T-001 | Layer group loading | Group loads to correct device |
| T-002 | Layer group eviction | Memory freed after eviction |
| T-003 | Double buffering | Both buffers functional |
| T-004 | Numerical equivalence | Output matches full-resident |
| T-005 | Memory budget | Never exceeds configured limit |
| T-006 | Config validation | Invalid settings rejected |

### 6.2 Integration Tests

| Test ID | Description | Verification |
|---------|-------------|--------------|
| IT-001 | Full model forward | Correct output shape and values |
| IT-002 | Incremental generation | KV-cache works across groups |
| IT-003 | Batch inference | Multiple sequences processed |
| IT-004 | Memory profiling | Peak memory within budget |

### 6.3 Performance Benchmarks

| Benchmark | Baseline | Target |
|-----------|----------|--------|
| Transfer efficiency | N/A | >90% PCIe utilization |
| Compute overlap | 0% | >80% overlap |
| Memory overhead | N/A | <5% of budget |

---

## 7. Implementation Plan

### 7.1 Phase 1: Core Infrastructure

**Deliverables:**
- `LayerLoader` class with sync loading
- `MemoryManager` for GPU memory tracking
- Config extensions for streaming settings

**Acceptance:**
- Layer groups load and evict correctly
- Memory tracking accurate

### 7.2 Phase 2: Async Pipelining

**Deliverables:**
- `TransferEngine` with CUDA streams
- Double buffering implementation
- Prefetch scheduling

**Acceptance:**
- Compute/transfer overlap verified
- No race conditions

### 7.3 Phase 3: Integration

**Deliverables:**
- `StreamingInferenceEngine` class
- Integration with TritterModel
- Generation API

**Acceptance:**
- End-to-end generation works
- Performance meets targets

---

## 8. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PCIe bandwidth saturation | High | Medium | Larger layer groups, compression |
| CUDA stream synchronization bugs | High | Low | Comprehensive testing, timeline traces |
| Memory fragmentation | Medium | Medium | Pre-allocated buffers, defragmentation |
| Incompatibility with quantization | High | Low | Test with BitNet weights |

---

## Appendix A: Example Usage

```python
from tritter import TritterConfig, TritterModel
from tritter.inference import StreamingInferenceEngine

# Configure for streaming inference
config = TritterConfig(
    model_size="70B",
    use_layer_streaming=True,
    layer_group_size=4,
    gpu_memory_budget_gb=14.0,
    prefetch_next_group=True,
)

# Load model (weights go to CPU)
model = TritterModel.from_pretrained("tritter-70b", config=config)

# Create streaming engine
engine = StreamingInferenceEngine(model, config)

# Generate with automatic layer streaming
output_ids = engine.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
)
```

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-23 | Claude | Initial draft |
