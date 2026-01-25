# ADR-002: Progressive Layer Loading for Unbounded Model Size

**Status**: Proposed
**Date**: 2026-01-23
**Decision Makers**: Project maintainers
**Tags**: memory, inference, scalability, architecture

## Context and Problem Statement

Tritter targets RTX 5080 with 16GB VRAM, which limits the maximum model size that can run entirely in GPU memory. While BitNet 1.58-bit quantization reduces the 7B model to ~1.4GB weights, larger models (13B, 30B, 70B+) exceed available VRAM even with aggressive quantization.

**The Problem**: How do we enable running arbitrarily large models on fixed-size GPU memory?

**Constraints**:
- RTX 5080: 16GB GDDR7 at 960 GB/s internal bandwidth
- PCIe 5.0: ~32 GB/s host-to-device bandwidth
- KV-cache must persist across layers (cannot be offloaded mid-generation)
- Target: Interactive latency for code completion use cases

## Decision

Implement **Progressive Layer Loading** - a layer streaming mechanism that:

1. **Loads layer groups on-demand** from CPU RAM (or disk) to GPU
2. **Processes tokens through loaded layers** before moving to next group
3. **Frees GPU memory** after layer group processing completes
4. **Pipelines loading with computation** to hide transfer latency

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU VRAM (16GB)                         │
├─────────────────────────────────────────────────────────────────┤
│  KV-Cache (persistent)     │  Active Layer Group  │  Workspace  │
│       ~8-10 GB             │      ~2-4 GB         │   ~2 GB     │
└─────────────────────────────────────────────────────────────────┘
                                    ↑
                                    │ PCIe 5.0 (~32 GB/s)
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CPU RAM (Staging)                          │
│  Layer Group 0 │ Layer Group 1 │ Layer Group 2 │ ... │ Group N  │
└─────────────────────────────────────────────────────────────────┘
                                    ↑
                                    │ NVMe (~7 GB/s)
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Disk (Cold Storage)                      │
│                    Full Model Weights (~1.4GB for 7B)            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Layer Grouping**: Process 4-8 layers per group (tunable based on VRAM)
2. **Double Buffering**: Load next group while processing current group
3. **Pinned Memory**: Use CUDA pinned memory for faster transfers
4. **Async Streams**: Overlap H2D transfers with computation
5. **KV-Cache Priority**: Never evict KV-cache; layer weights are transient

### Rationale

This approach is chosen because:

1. **Proven Pattern**: Used successfully in llama.cpp, vLLM, and other inference engines
2. **No Quality Loss**: Unlike pruning or further quantization, layer streaming preserves full model capability
3. **Flexible Scaling**: Same code works for 7B on consumer GPU or 70B on workstation
4. **PCIe 5.0 Advantage**: RTX 5080's PCIe 5.0 provides 2x bandwidth over PCIe 4.0

## Consequences

### Positive

- **Unbounded Model Size**: Run 70B+ models on 16GB VRAM
- **Graceful Degradation**: Throughput scales with available memory (more VRAM = more layers resident)
- **Cloud Compatibility**: Same mechanism works for multi-GPU setups with layer sharding
- **Future Proof**: Architecture supports future optimizations (layer caching, predictive loading)

### Negative

- **Reduced Throughput**: PCIe bandwidth becomes bottleneck for large models
  - Estimate: 7B BitNet (~1.4GB) transfers in ~44ms per full pass
  - 70B would be ~440ms per pass (without pipelining)
- **Increased Latency Variance**: First token may be slower due to initial layer loading
- **Memory Pressure**: CPU RAM must hold full model for optimal streaming
- **Complexity**: Adds significant architectural complexity to inference path

### Neutral

- **Batch Size Trade-off**: Larger batches amortize transfer cost but increase KV-cache size
- **Layer Group Size**: Tunable parameter trading memory for transfer efficiency

## Alternatives Considered

### Alternative 1: Full CPU Offloading (llama.cpp style)
**Rationale**: Keep all weights on CPU, transfer per-layer during forward pass
**Rejected because**: Higher latency due to no pipelining, less GPU utilization

### Alternative 2: Model Parallelism (Multi-GPU)
**Rationale**: Shard layers across multiple GPUs
**Rejected because**: Targets single RTX 5080; multi-GPU is future enhancement

### Alternative 3: Aggressive Quantization (2-bit, 1-bit)
**Rationale**: Reduce weight size further to fit in VRAM
**Rejected because**: Quality degradation; BitNet 1.58-bit is already near minimum for quality preservation

### Alternative 4: Mixture of Experts (MoE) with Expert Offloading
**Rationale**: Only load active experts per token
**Rejected because**: Requires MoE architecture; Tritter is dense transformer

### Alternative 5: No Large Model Support
**Rationale**: Limit to models that fit in VRAM
**Rejected because**: Limits future scaling; progressive loading is standard practice

## Implementation Considerations

### Phase 1: Basic Layer Streaming
- Implement LayerLoader class with sync loading
- Add layer group configuration to TritterConfig
- Modify TritterModel.forward() for progressive execution

### Phase 2: Async Pipelining
- Add CUDA stream management for H2D overlap
- Implement double-buffered layer groups
- Add prefetching based on layer execution

### Phase 3: Optimization
- Pinned memory allocation
- Layer caching for frequently-used layers
- Adaptive group sizing based on available VRAM

## References

1. [llama.cpp GPU offloading](https://github.com/ggerganov/llama.cpp) - Reference implementation
2. [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) - Memory management patterns
3. [FlexGen](https://arxiv.org/abs/2303.06865) - Offloading strategies for LLM inference
4. [DeepSpeed Inference](https://www.deepspeed.ai/inference/) - Layer parallelism patterns
5. [PCIe 5.0 Specification](https://pcisig.com/) - Bandwidth characteristics

## Open Questions

1. **Disk Streaming**: Should we support direct disk-to-GPU streaming for systems with limited RAM?
2. **Checkpointing**: How to handle KV-cache overflow when context exceeds VRAM budget?
3. **Multi-Request**: How to share layer loading across concurrent inference requests?
