# Tritter Documentation

## Directory Structure

```
docs/
├── getting-started/     # Onboarding and setup
├── development/         # Developer guides and standards
├── architecture/        # Design and roadmap
├── training/            # Training data and strategy
├── specs/               # Technical specifications
├── adr/                 # Architecture Decision Records
└── legacy/              # Deprecated documents
```

## Quick Links

### Getting Started

- [Hardware Requirements](getting-started/HARDWARE_REQUIREMENTS.md)
- [CUDA Setup](getting-started/CUDA_SETUP.md)

### Development

- [Contributing Guide](development/CONTRIBUTING.md)
- [Development Standards](development/DEVELOPMENT_STANDARDS.md)
- [API Conventions](development/API_CONVENTIONS.md)
- [FlexAttention Implementation Guide](development/guides/GUIDE-001-flexattention-implementation.md)

### Architecture

- [Project Plan](architecture/project-plan.md) - Full technical blueprint
- [Roadmap](architecture/ROADMAP.md) - Current status and planned work
- [Research Considerations](architecture/considerations.md) - Transformer alternatives
- [Embedding-Prediction Analysis](architecture/embedding-prediction-position-analysis.md)

### Training

- [Training Strategy](training/TRAINING_STRATEGY.md) - Data mix, persona, alignment
- [Dataset Cleaning](training/clean-datasets.md) - Quality gates
- [Triton Kernel Curation](training/triton/) - GPU kernel dataset

### Specifications

| Spec | Title | Status |
|------|-------|--------|
| [SPEC-001](specs/SPEC-001-flexattention.md) | FlexAttention | ✅ Implemented |
| [SPEC-002](specs/SPEC-002-attention-modes.md) | Attention Modes | ⏳ Partial |
| [SPEC-003](specs/SPEC-003-embedding-prediction.md) | Embedding Prediction | 📋 Planned |
| [SPEC-004](specs/SPEC-004-test-strategy.md) | Test Strategy | ✅ Implemented |
| [SPEC-005](specs/SPEC-005-memory-optimization.md) | Memory Optimization | ✅ Implemented |
| [SPEC-006](specs/SPEC-006-progressive-layer-loading.md) | Progressive Layer Loading | ✅ Implemented |
| [SPEC-007](specs/SPEC-007-dataset-quality-gates.md) | Dataset Quality Gates | ✅ Implemented |
| [SPEC-008](specs/SPEC-008-progressive-model-training.md) | Progressive Model Training | 📋 Planned |
| [SPEC-009](specs/SPEC-009-packed-ternary-inference.md) | Packed Ternary Inference | ✅ Implemented |
| [SPEC-010](specs/SPEC-010-lora-finetuning.md) | LoRA/QLoRA Fine-tuning | ✅ Implemented |
| [SPEC-011](specs/SPEC-011-progressive-checkpoint-format.md) | Progressive Checkpoints | 🔬 Research |

### Architecture Decision Records

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](adr/001-sequence-position-vs-token-semantics.md) | Sequence Position vs Token Semantics | Accepted |
| [ADR-002](adr/002-progressive-layer-loading.md) | Progressive Layer Loading | Accepted |
| [ADR-003](adr/003-progressive-model-training.md) | Progressive Model Training | Accepted |
| [ADR-004](adr/004-blackwell-gpu-support.md) | Blackwell GPU Support | Accepted |

## Status Legend

- ✅ **Implemented** - Feature complete and tested
- ⏳ **Partial** - Partially implemented
- 📋 **Planned** - Documented, not yet implemented
- 🔬 **Research** - Requires additional research

## See Also

- [CLAUDE.md](../CLAUDE.md) - Quick reference for Claude Code
- [README.md](../README.md) - Project overview
