# FlexAttention Integration Specification

**Spec ID**: SPEC-001
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-22
**Target Module**: `src/tritter/models/flex_attention.py`

## 1. Overview

### 1.1 Purpose

This specification defines the integration of PyTorch FlexAttention into Tritter's attention architecture to support advanced masking patterns required for:

- Document-packed sequence training
- Sliding window attention for bounded KV-cache
- StreamingLLM attention sinks
- Prefix-LM instruction tuning

### 1.2 Background

Tritter's 128K context window with document-packed training requires efficient attention patterns beyond simple causal masking. FlexAttention (PyTorch 2.5+) provides composable mask functions that compile to fused Triton kernels without materializing O(n²) mask tensors.

### 1.3 Dependencies

- PyTorch >= 2.5.0 (FlexAttention API)
- CUDA >= 12.0 (Triton kernel compilation)
- Existing `TritterConfig` attention mode configuration

### 1.4 References

| Reference | Description |
|-----------|-------------|
| [PyTorch FlexAttention Blog](https://pytorch.org/blog/flexattention/) | Official PyTorch FlexAttention documentation |
| [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) | Memory-efficient attention algorithm |
| [StreamingLLM Paper](https://arxiv.org/abs/2309.17453) | Attention sinks for streaming inference |
| `docs/tritter-comprehensive-implementation-plan.md` | Parent implementation plan |
| `docs/clean-datasets.md` | Training data strategy requiring document masking |

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-001 | Causal mask primitive | Must Have | `causal_mask(b, h, q_idx, kv_idx)` returns `q_idx >= kv_idx` |
| FR-002 | Sliding window mask | Must Have | Attention limited to `window_size` tokens; configurable via `TritterConfig.sliding_window_size` |
| FR-003 | Document boundary mask | Must Have | No cross-document attention in packed sequences |
| FR-004 | StreamingLLM sinks | Should Have | First `num_sink_tokens` always attended regardless of window |
| FR-005 | Prefix-LM mask | Should Have | Bidirectional on prefix, causal after `prefix_length` |
| FR-006 | Composite mask creation | Must Have | `and_masks()` combines multiple mask functions |
| FR-007 | BlockMask caching | Should Have | Cache and reuse BlockMask for same sequence structure |
| FR-008 | SDPA fallback | Must Have | Fall back to `is_causal=True` SDPA when FlexAttention unavailable |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target | Verification |
|----|-------------|--------|--------------|
| NFR-001 | Memory efficiency | No O(n²) mask materialization | Profile with 128K sequence |
| NFR-002 | Kernel compilation | <30s first-run compilation | Benchmark compilation time |
| NFR-003 | Runtime performance | Within 10% of manual SDPA baseline | A/B benchmark |
| NFR-004 | RTX 5080 compatibility | Works on CUDA compute 9.0+ | Test on target hardware |

### 2.3 Constraints

- **C-001**: Must not increase memory footprint beyond current SDPA implementation
- **C-002**: Must maintain backward compatibility with existing `TritterAttention` API
- **C-003**: Must support both training and inference modes
- **C-004**: BlockMask creation must be deterministic for reproducibility

---

## 3. Architecture

### 3.1 Module Structure

```
src/tritter/models/
├── architecture.py          # Existing TritterAttention (unchanged API)
├── flex_attention.py        # NEW: FlexAttention primitives and wrapper
└── attention_patterns.py    # NEW: Mask pattern definitions
```

### 3.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        TritterAttention                         │
│  (Existing public API - unchanged)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ FlexAttention   │    │ SDPA Fallback   │                    │
│  │ Backend         │    │ (is_causal=True)│                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────┐                   │
│  │         AttentionPatterns               │                   │
│  │  • causal_mask()                        │                   │
│  │  • sliding_window_mask()                │                   │
│  │  • document_mask()                      │                   │
│  │  • streamingllm_mask()                  │                   │
│  │  • prefix_lm_mask()                     │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow

```
Input: hidden_states (B, L, D)
       config.attention_mode
       optional: doc_boundaries tensor

                    ┌──────────────────┐
                    │ Check PyTorch    │
                    │ version >= 2.5   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │ FlexAttention   │          │ SDPA Fallback   │
    │ available       │          │ (is_causal only)│
    └────────┬────────┘          └────────┬────────┘
             │                            │
             ▼                            │
    ┌─────────────────┐                   │
    │ Build BlockMask │                   │
    │ (cached if same │                   │
    │  structure)     │                   │
    └────────┬────────┘                   │
             │                            │
             ▼                            │
    ┌─────────────────┐                   │
    │ flex_attention()│                   │
    └────────┬────────┘                   │
             │                            │
             └──────────────┬─────────────┘
                            │
                            ▼
               Output: attn_output (B, L, D)
```

---

## 4. Interface Specification

### 4.1 Mask Primitive Functions

```python
def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Standard autoregressive causal mask.

    Args:
        b: Batch index (unused, for signature compatibility)
        h: Head index (unused, for signature compatibility)
        q_idx: Query position index
        kv_idx: Key/Value position index

    Returns:
        True if q_idx should attend to kv_idx

    Behavior:
        Returns q_idx >= kv_idx (attend to self and past)
    """

def sliding_window_mask(window_size: int) -> Callable[[int, int, int, int], bool]:
    """Create sliding window mask function.

    Args:
        window_size: Maximum distance between query and key positions

    Returns:
        Mask function with signature (b, h, q_idx, kv_idx) -> bool

    Behavior:
        Returns (q_idx >= kv_idx) AND (q_idx - kv_idx <= window_size)
    """

def document_mask(doc_ids: torch.Tensor) -> Callable[[int, int, int, int], bool]:
    """Create document boundary mask.

    Args:
        doc_ids: Tensor of shape (batch, seq_len) mapping positions to document IDs

    Returns:
        Mask function that returns True only for same-document positions

    Behavior:
        Returns doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    """

def streamingllm_mask(
    sink_tokens: int,
    window_size: int
) -> Callable[[int, int, int, int], bool]:
    """StreamingLLM attention sink mask.

    Args:
        sink_tokens: Number of initial tokens to always attend to
        window_size: Sliding window size for recent tokens

    Returns:
        Mask function combining sinks and sliding window

    Behavior:
        Returns (kv_idx < sink_tokens) OR sliding_window_condition
    """

def prefix_lm_mask(prefix_length: int) -> Callable[[int, int, int, int], bool]:
    """Prefix-LM mask for instruction tuning.

    Args:
        prefix_length: Length of bidirectional prefix region

    Returns:
        Mask function with bidirectional prefix and causal suffix

    Behavior:
        If q_idx < prefix_length AND kv_idx < prefix_length: True (bidirectional)
        If q_idx >= prefix_length: kv_idx <= q_idx (causal)
    """
```

### 4.2 BlockMask Factory

```python
def create_attention_mask(
    config: TritterConfig,
    seq_len: int,
    doc_ids: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> BlockMask:
    """Create BlockMask based on configuration.

    Args:
        config: TritterConfig with attention settings
        seq_len: Sequence length for mask
        doc_ids: Optional document ID tensor for packed sequences
        device: Target device for mask

    Returns:
        BlockMask suitable for flex_attention()

    Raises:
        ValueError: If attention_mode is invalid
        RuntimeError: If FlexAttention unavailable and complex mask needed

    Configuration Mapping:
        attention_mode="causal" + no sliding window -> Simple causal
        attention_mode="causal" + sliding window -> Causal + window
        attention_mode="prefix_lm" -> Prefix-LM composite
        attention_mode="bidirectional" -> Full attention (no mask)
        doc_ids provided -> Add document boundary mask
        use_attention_sinks=True -> Add sink tokens
    """
```

### 4.3 FlexAttention Layer

```python
class FlexAttentionLayer(nn.Module):
    """FlexAttention wrapper with automatic fallback.

    Attributes:
        config: TritterConfig instance
        _mask_cache: Dict mapping cache keys to BlockMask

    Methods:
        forward(query, key, value, block_mask=None, score_mod=None) -> Tensor
        clear_cache() -> None
    """
```

---

## 5. Configuration Integration

### 5.1 TritterConfig Fields

The following fields in `TritterConfig` control FlexAttention behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `attention_mode` | str | "causal" | One of: "causal", "bidirectional", "prefix_lm", "embedding" |
| `use_sliding_window` | bool | False | Enable sliding window attention |
| `sliding_window_size` | int | 4096 | Window size when sliding window enabled |
| `use_attention_sinks` | bool | False | Enable StreamingLLM attention sinks |
| `num_sink_tokens` | int | 4 | Number of sink tokens when enabled |

### 5.2 Validation Rules

```python
# In TritterConfig.__post_init__:

# Sliding window requires positive size
if self.use_sliding_window:
    assert self.sliding_window_size > 0

# Attention sinks require sliding window
if self.use_attention_sinks:
    assert self.use_sliding_window, "Attention sinks require sliding window"
    assert self.num_sink_tokens > 0

# prefix_lm mode requires prefix_length (if added)
if self.attention_mode == "prefix_lm":
    assert hasattr(self, 'prefix_length') and self.prefix_length > 0
```

---

## 6. Memory Analysis

### 6.1 Current Implementation (SDPA with manual mask)

For 128K sequence:
- Mask tensor: `(1, 1, 131072, 131072) * 4 bytes = 64 GB` (impossible)
- With `is_causal=True`: `0 bytes` mask allocation

### 6.2 FlexAttention Implementation

- BlockMask metadata: `~50 MB` (negligible)
- No mask tensor materialization
- Compiled Triton kernel handles sparsity

### 6.3 Memory Budget Impact

| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| Mask storage | 0 (using is_causal) | ~50 MB | +50 MB |
| KV-cache | ~8.4 GB | ~8.4 GB | 0 |
| Total impact | - | - | <1% increase |

---

## 7. Test Specification

### 7.1 Unit Tests

| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| T-001 | Causal mask correctness | q_idx=5, kv_idx=3 | True |
| T-002 | Causal mask correctness | q_idx=3, kv_idx=5 | False |
| T-003 | Sliding window boundary | q_idx=100, kv_idx=50, window=40 | False |
| T-004 | Sliding window in range | q_idx=100, kv_idx=80, window=40 | True |
| T-005 | Document mask same doc | doc_ids=[0,0,1,1], q=0, kv=1 | True |
| T-006 | Document mask diff doc | doc_ids=[0,0,1,1], q=0, kv=2 | False |
| T-007 | StreamingLLM sink | kv_idx=2, sinks=4 | True (always) |
| T-008 | Prefix-LM bidirectional | q=5, kv=8, prefix=10 | True |
| T-009 | Prefix-LM causal | q=15, kv=12, prefix=10 | True |
| T-010 | Prefix-LM causal blocked | q=12, kv=15, prefix=10 | False |

### 7.2 Integration Tests

| Test ID | Description | Verification |
|---------|-------------|--------------|
| IT-001 | Forward pass with FlexAttention | Output shape matches input |
| IT-002 | Gradient flow through FlexAttention | Gradients non-zero |
| IT-003 | SDPA fallback when FlexAttention unavailable | No errors on PyTorch < 2.5 |
| IT-004 | BlockMask caching | Same mask reused for same structure |
| IT-005 | Memory usage with 128K sequence | Peak < 15 GB on RTX 5080 |

### 7.3 Performance Benchmarks

| Benchmark | Baseline | Target | Measurement |
|-----------|----------|--------|-------------|
| Compilation time | N/A | < 30s | First-run timing |
| Throughput (128K) | SDPA is_causal | Within 10% | tokens/sec |
| Memory (128K) | SDPA is_causal | Within 5% | peak VRAM |

---

## 8. Implementation Plan

### 8.1 Phase 1: Core Primitives

**Deliverables:**
- `attention_patterns.py` with mask primitive functions
- Unit tests T-001 through T-010
- Documentation strings with "Why" sections

**Acceptance:**
- All unit tests pass
- Lint and type check pass

### 8.2 Phase 2: BlockMask Factory

**Deliverables:**
- `create_attention_mask()` function
- Configuration validation in `TritterConfig`
- Integration with existing attention mode settings

**Acceptance:**
- BlockMask created correctly for all config combinations
- Validation catches invalid configurations

### 8.3 Phase 3: FlexAttention Layer

**Deliverables:**
- `FlexAttentionLayer` class
- SDPA fallback logic
- BlockMask caching

**Acceptance:**
- Integration tests IT-001 through IT-005 pass
- Performance benchmarks meet targets

### 8.4 Phase 4: TritterAttention Integration

**Deliverables:**
- Update `TritterAttention.forward()` to use FlexAttention
- Backward compatibility with existing API
- Updated documentation

**Acceptance:**
- Existing tests continue to pass
- No API breaking changes

---

## 9. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PyTorch 2.5 not available | High | Low | SDPA fallback path |
| Triton compilation failures | Medium | Medium | Pre-compiled kernels, error handling |
| Performance regression | Medium | Low | Comprehensive benchmarking |
| BlockMask caching memory | Low | Low | LRU cache with size limit |

---

## 10. Open Questions

1. **Q**: Should we support custom `score_mod` functions (e.g., ALiBi)?
   **Status**: Deferred to future spec

2. **Q**: How to handle multi-GPU BlockMask synchronization?
   **Status**: Not in scope for single-GPU RTX 5080 target

3. **Q**: Should BlockMask be serializable for checkpoint saving?
   **Status**: To be determined during implementation

---

## Appendix A: Example Usage

```python
from tritter import TritterConfig, TritterModel
from tritter.models.flex_attention import create_attention_mask

# Configure for document-packed training with sliding window
config = TritterConfig(
    model_size="7B",
    attention_mode="causal",
    use_sliding_window=True,
    sliding_window_size=4096,
)

model = TritterModel(config)

# Training with document packing
batch_size, seq_len = 2, 8192
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
doc_ids = torch.tensor([[0]*4096 + [1]*4096] * batch_size)  # Two docs per sequence

# Model handles mask creation internally based on config
logits = model(input_ids, doc_ids=doc_ids)
```

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-22 | Claude | Initial draft |
