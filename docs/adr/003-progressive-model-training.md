# ADR-003: Progressive Model Training Strategy

**Status**: Accepted
**Date**: 2026-01-24
**Deciders**: Tyler Zervas, Claude (Agentic Development)

## Context

Users want to progressively train models from smaller sizes (3B) to larger sizes (10B, 100B, 1T+) without starting from scratch each time. This enables:
- Iterative development with incremental compute investment
- Skill accumulation across training phases
- Accessible training for users with limited resources
- Knowledge preservation when scaling up

Current state-of-the-art approaches for model expansion include:
- **Net2Net / bert2BERT**: Function-preserving width and depth transformations
- **SOLAR DUS**: Depth Up-Scaling by duplicating and concatenating layers
- **LiGO**: Learned linear growth operators for efficient expansion
- **Progressive MoE**: Adding experts incrementally for new knowledge
- **LEMON**: Lossless model expansion preserving exact functional mapping

## Decision

We will implement a **multi-strategy progressive training system** that supports:

1. **Depth Up-Scaling (DUS)** - Primary method for vertical scaling
2. **Width Up-Scaling** - For horizontal capacity expansion
3. **Mixture of Experts (MoE) progression** - For specialized knowledge domains
4. **Checkpoint compatibility** - Enabling seamless size transitions

### Rationale

**Why DUS as primary?**
- Simple implementation with no architectural changes needed
- Immediately compatible with standard training frameworks
- SOLAR 10.7B demonstrated strong results expanding Mistral 7B to 10.7B
- No complex routing or gating mechanisms required

**Why Width Up-Scaling?**
- Complementary to depth scaling
- Increases model capacity without adding inference latency
- bert2BERT proved effective for transformer architectures

**Why Progressive MoE?**
- Enables domain-specific expansion (e.g., add code expert, add vision expert)
- Preserves existing knowledge in frozen experts
- Allows adding capacity without full retraining

**Why not pure Knowledge Distillation?**
- Requires additional compute for teacher inference
- Model expansion directly uses weights with minimal overhead
- LiGO showed 50% compute savings vs. training from scratch

## Consequences

### Positive
- Users can start small and grow incrementally
- Knowledge from previous training phases is preserved
- Compute costs scale with actual needs
- Enables community collaboration on base models

### Negative
- Multiple expansion strategies increase complexity
- Continued pretraining required after expansion
- May not perfectly preserve all learned capabilities
- Requires careful management of checkpoint compatibility

### Risks
- Catastrophic forgetting during continued pretraining
- Suboptimal layer initialization choices
- Incompatibility between expansion strategies

### Mitigations
- Implement regularization-based forgetting prevention (EWC, SI)
- Use function-preserving transformations where possible
- Validate expansion with benchmark suite before/after
- Document clear upgrade paths between model sizes

## Implementation Phases

### Phase 1: Depth Up-Scaling (DUS)
- Implement layer duplication and concatenation
- Add continued pretraining support
- Target: 3B → 7B, 7B → 10B paths

### Phase 2: Width Up-Scaling
- Implement neuron splitting (Net2Net style)
- Add attention head expansion
- Target: Increase hidden_size, intermediate_size

### Phase 3: Progressive MoE
- Add expert addition infrastructure
- Implement router retraining
- Target: Add specialized domain experts

### Phase 4: Hybrid Expansion
- Combine depth + width in single expansion
- Support arbitrary size targets
- Auto-compute expansion parameters

## References

- [SOLAR 10.7B](https://arxiv.org/abs/2312.15166) - Depth Up-Scaling methodology
- [LiGO](https://vita-group.github.io/LiGO/) - Learned growth operators (ICLR 2023)
- [bert2BERT](https://aclanthology.org/2022.acl-long.151/) - Width/depth expansion for transformers
- [Net2Net](https://arxiv.org/abs/1511.05641) - Original function-preserving transformations
- [LEMON](https://openreview.net/forum?id=0e705ac30e57) - Lossless model expansion (ICLR 2024)
- [LLM Continual Learning Survey](https://dl.acm.org/doi/10.1145/3735633) - ACM Computing Surveys 2025
- [Progressive MoE](https://arxiv.org/html/2503.07137v1) - Mixture of Experts survey
- [Densing Law](https://www.nature.com/articles/s42256-025-01137-0) - Capability per parameter trends

## Decision Record

| Date | Action |
|------|--------|
| 2026-01-24 | Initial proposal and acceptance |
