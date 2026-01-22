# Architecture Overview

Display the current architecture and implementation status.

## Instructions

Provide a concise overview of Tritter's architecture:

### Core Architecture
- **Model**: 3B/7B decoder-only transformer with BitNet 1.58-bit quantization
- **Context**: 128K tokens via sliding window attention + INT4 KV-cache
- **Paradigm**: Embedding-prediction (Coconut/LCM style) - operates in continuous space

### Key Components
1. **TritterConfig** - Central configuration with auto-scaling for 7B
2. **TernaryWeight** - BitNet quantization with STE gradient flow
3. **TritterAttention** - Multi-head attention with QK-Norm
4. **TritterMLP** - FFN with Squared ReLU (BitNet requirement)
5. **TritterLayer** - Post-FFN LayerNorm (Chameleon-style)

### BitNet Requirements
- Squared ReLU activation (not SiLU/GELU)
- QK-Norm for training stability
- Shadow weights in full precision for STE training

### Memory Budget (RTX 5080 16GB)
- 7B BitNet weights: 1.4 GB
- INT4 KV-cache (128K): ~8-10 GB
- Activations: ~2-3 GB
- **Total: ~12-15 GB**

### Implementation Roadmap
1. Current: Basic SDPA attention
2. Next: FlexAttention for dynamic masking
3. Future: Hybrid Mamba-2/Transformer architecture

Read CLAUDE.md and docs/tritter-comprehensive-implementation-plan.md for full details.
