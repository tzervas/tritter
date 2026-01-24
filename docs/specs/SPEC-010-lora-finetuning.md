# SPEC-010: LoRA/QLoRA Memory-Efficient Fine-Tuning

## Status

**Implemented** - Core LoRA and QLoRA functionality complete.

## Problem Statement

Full fine-tuning of large models (7B+) is infeasible on consumer GPUs:

| Model | Full Training Memory | RTX 5080 VRAM | Fits? |
|-------|---------------------|---------------|-------|
| 1B    | ~12 GB              | 16 GB         | ✓     |
| 3B    | ~24 GB              | 16 GB         | ✗     |
| 7B    | ~61 GB              | 16 GB         | ✗     |
| 13B   | ~112 GB             | 16 GB         | ✗     |
| 40B   | ~398 GB             | 16 GB         | ✗     |

Memory breakdown for full fine-tuning:
- BF16 weights: 2 bytes × params
- Gradients: 2 bytes × params
- Optimizer (AdamW m + v): 2 × 4 bytes × params
- Activations: ~1-4 GB (with gradient checkpointing)

**Total**: ~20 bytes per parameter + activations

## Solution: LoRA/QLoRA

Low-Rank Adaptation (LoRA) freezes base weights and adds small trainable adapters:

```
output = base_layer(x) + (x @ A @ B) * scaling
```

Where:
- `base_layer`: Frozen pretrained weights (no gradients, no optimizer states)
- `A`: (in_features, rank) trainable matrix
- `B`: (rank, out_features) trainable matrix
- `scaling`: alpha / rank (controls adaptation magnitude)

### Memory Savings

| Model | Full Training | QLoRA (r=16) | Reduction |
|-------|--------------|--------------|-----------|
| 1B    | 12.2 GB      | 1.9 GB       | 6.4x      |
| 3B    | 24.4 GB      | 2.3 GB       | 10.6x     |
| 7B    | 60.8 GB      | 3.7 GB       | 16.4x     |
| 13B   | 111.8 GB     | 5.1 GB       | 21.9x     |
| 40B   | 397.8 GB     | 13.7 GB      | 29.0x     |

QLoRA memory breakdown:
- Packed ternary base weights: ~0.25 bytes × params (frozen)
- LoRA params: ~0.05% of base (FP16)
- LoRA gradients: same as LoRA params
- Optimizer: 2 × FP32 × LoRA params
- Activations: 1-2.5 GB

## Implementation

### Core Classes

```python
from tritter.training.lora import (
    LoRAConfig,      # Configuration dataclass
    LoRALinear,      # LoRA-wrapped layer
    apply_lora,      # Apply LoRA to model
    save_lora_adapters,   # Save adapters only
    load_lora_adapters,   # Load adapters
    merge_lora_weights,   # Merge for deployment
)
```

### Usage

```python
from tritter.core.config import TritterConfig
from tritter.models.architecture import TritterModel
from tritter.training.lora import LoRAConfig, apply_lora, LoRATrainer

# Create base model
config = TritterConfig(model_size="7B", use_bitnet=True)
model = TritterModel(config)

# Apply QLoRA (LoRA on ternary weights)
lora_config = LoRAConfig(
    rank=16,                    # Low-rank dimension
    alpha=16.0,                 # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    dropout=0.05,               # Regularization
)
model = apply_lora(model, lora_config)

# Create trainer (only LoRA params in optimizer)
trainer = LoRATrainer(model, lora_config, learning_rate=1e-4)

# Training loop...
# trainer.optimizer only contains ~17M params instead of 7B

# Save adapters (tiny: ~34MB for 7B model)
trainer.save_checkpoint("checkpoints/7b_lora/")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 16 | Low-rank dimension (4-64 typical) |
| `alpha` | 16.0 | Scaling factor (usually = rank) |
| `dropout` | 0.0 | Dropout for LoRA output |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Modules to adapt |
| `bias` | `"none"` | `"none"`, `"lora_only"`, or `"all"` |
| `use_rslora` | False | Rank-stabilized scaling |
| `init_lora_weights` | `"gaussian"` | `"gaussian"` or `"pissa"` |

### Target Module Patterns

| Pattern | Trainable Params (7B) | Use Case |
|---------|----------------------|----------|
| Attention only (q,k,v,o) | ~17M (0.2%) | Default, most efficient |
| + MLP (gate,up,down) | ~50M (0.7%) | More capacity |
| Full model | ~7B (100%) | Full fine-tuning |

## RTX 5080 16GB Feasibility

```
Model     r=8    r=16   r=32   r=64   Memory (r=16)
------------------------------------------------------
1B         ✓       ✓       ✓       ✓        1.86 GB
3B         ✓       ✓       ✓       ✓        2.28 GB
7B         ✓       ✓       ✓       ✓        3.70 GB
10B        ✓       ✓       ✓       ✓        4.48 GB
13B        ✓       ✓       ✓       ✓        5.10 GB
30B        ✓       ✓       ✓       ✓       10.34 GB
33B        ✓       ✓       ✓       ✓       10.74 GB
40B        ✓       ✓       ✓       ✗       13.69 GB
65B        ✗       ✗       ✗       ✗       17.17 GB
70B        ✗       ✗       ✗       ✗       20.23 GB
```

**Key insight**: QLoRA enables training models up to 40B on RTX 5080 16GB.

## Deployment

### Option 1: Keep Adapters Separate
- Base model + adapters loaded separately
- Swap adapters for different tasks
- Slightly slower inference

### Option 2: Merge Weights
```python
from tritter.training.lora import merge_lora_weights

# Merge LoRA into base weights
merge_lora_weights(model)

# Zero out LoRA to avoid double-counting
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.lora_B.zero_()
```

### Option 3: Convert to Packed
```python
from tritter.quantization.packed_ternary import convert_to_packed

# After merge, convert for efficient inference
model = convert_to_packed(model)
```

## Risk Analysis

### Memory Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OOM during training | Medium | High | Use lower rank, smaller batch, gradient accumulation |
| Activation spikes | Low | Medium | Gradient checkpointing already enabled |
| KV-cache overflow | Medium | High | Limit context length during training (2K-4K) |

### Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Underfitting (rank too low) | Medium | Medium | Start with r=16, increase if needed |
| Overfitting (small dataset) | High | Medium | Use dropout, early stopping |
| Catastrophic forgetting | Low | High | Use lower learning rate, shorter training |

### Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Merge precision loss (BitNet) | High | Low | Keep adapters separate for production |
| Incompatible with layer streaming | Low | Medium | Tested and working |
| Save/load version mismatch | Low | Medium | Version field in checkpoint |

## Testing

```bash
# Unit tests
pytest tests/unit/test_lora.py -v

# Integration tests
pytest tests/integration/test_lora_training.py -v

# Memory verification (on GPU)
python scripts/rtx5080_feasibility.py --model 7B
```

## References

1. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
2. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
3. Liu et al. "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
