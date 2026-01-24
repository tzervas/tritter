# SPEC-011: Progressive Checkpoint Format

## Status

**Research Required** - Design phase

## Problem Statement

Standard checkpoint formats (safetensors, PyTorch state_dict) are designed for fixed architectures. Progressive training requires:

1. **Depth Up-Scaling (DUS)**: 3B (26 layers) → 7B (32 layers)
2. **Width Up-Scaling (Net2Net)**: Expand hidden dimensions without losing learned features
3. **Knowledge preservation**: EWC-style regularization to prevent forgetting
4. **Checkpoint compatibility**: Load 3B checkpoint into 7B architecture

Current safetensors limitations:
- Fixed tensor shapes per key
- No metadata about expansion strategy
- No hooks for interpolation/duplication

## Research Questions

### 1. Layer Expansion Strategy

**DUS (Depth Up-Scaling)** from LLaMA-Pro:
- Duplicate middle layers and insert
- Works: 26 layers → 32 layers by duplicating layers 13-18
- Question: Which layers contain most transferable knowledge?

**Open research**:
- Should we duplicate adjacent layers or spread duplicates?
- How much fine-tuning is needed after expansion?
- Can we identify "critical" vs "expandable" layers?

### 2. Width Expansion Strategy

**Net2Net** approach:
- Split neurons: `W_new[:, :n] = W_old`, `W_new[:, n:] = W_old + noise`
- Preserves function: Output unchanged immediately after expansion
- Question: How to expand attention heads? (4096 → 5120 hidden, 32 → 40 heads)

**Open research**:
- Does random noise or structured initialization work better?
- Should we expand all layers uniformly or selectively?
- How does width expansion interact with BitNet quantization?

### 3. Checkpoint Metadata Requirements

Minimum metadata for progressive checkpoints:

```json
{
    "format_version": "1.0",
    "model_family": "tritter",
    "architecture": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "intermediate_size": 11008
    },
    "training_state": {
        "total_tokens_seen": 100000000000,
        "current_step": 50000,
        "loss_history": [...]
    },
    "expansion_config": {
        "supports_depth_expansion": true,
        "supports_width_expansion": true,
        "layer_importance_scores": [...],
        "recommended_next_size": "13B"
    },
    "ewc_fisher": {
        "stored": true,
        "path": "fisher_diag.safetensors"
    }
}
```

### 4. Compatibility with Safetensors

Proposed approach: **Extend safetensors with metadata file**

```
checkpoint/
├── model.safetensors      # Standard safetensors weights
├── optimizer.safetensors  # Optimizer state (optional)
├── progressive.json       # Expansion metadata
├── fisher_diag.safetensors  # EWC Fisher diagonal (optional)
└── expansion_hooks.py     # Custom expansion logic (optional)
```

This maintains full safetensors compatibility while adding progressive features.

### 5. Expansion Operations

**Layer Duplication**:
```python
def duplicate_layer(state_dict: dict, layer_idx: int) -> dict:
    """Duplicate a transformer layer for DUS expansion.

    Creates a copy with small noise to break symmetry.
    """
    new_state = {}
    prefix = f"layers.{layer_idx}."
    new_prefix = f"layers.{layer_idx + 1}."

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key.replace(prefix, new_prefix)
            # Add small noise to break symmetry
            noise = torch.randn_like(value) * 0.01
            new_state[new_key] = value + noise

    return new_state
```

**Width Expansion**:
```python
def expand_linear(weight: Tensor, bias: Tensor | None,
                  new_in: int, new_out: int) -> tuple[Tensor, Tensor | None]:
    """Expand linear layer dimensions using Net2Net.

    Preserves function by splitting neurons.
    """
    old_out, old_in = weight.shape

    # Output expansion: duplicate rows
    if new_out > old_out:
        extra = new_out - old_out
        indices = torch.randint(0, old_out, (extra,))
        weight = torch.cat([weight, weight[indices] + noise], dim=0)
        if bias is not None:
            bias = torch.cat([bias, bias[indices] + noise], dim=0)

    # Input expansion: duplicate columns (scale to preserve magnitude)
    if new_in > old_in:
        extra = new_in - old_in
        indices = torch.randint(0, old_in, (extra,))
        new_cols = weight[:, indices] / 2  # Split magnitude
        weight[:, indices] /= 2  # Original columns also halved
        weight = torch.cat([weight, new_cols + noise], dim=1)

    return weight, bias
```

### 6. EWC (Elastic Weight Consolidation)

To prevent catastrophic forgetting during expansion:

```python
def compute_fisher_diagonal(model: Module, dataloader: DataLoader) -> dict:
    """Compute Fisher Information diagonal for EWC regularization.

    High Fisher values = important parameters for current task.
    """
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)

    model.eval()
    for batch in dataloader:
        loss = model(**batch).loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)

    # Normalize
    for name in fisher:
        fisher[name] /= len(dataloader)

    return fisher


def ewc_loss(model: Module, fisher: dict, old_params: dict,
             lambda_ewc: float = 1000.0) -> Tensor:
    """Compute EWC regularization loss.

    Penalizes changes to important parameters.
    """
    loss = 0.0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - old_params[name]).pow(2)).sum()
    return lambda_ewc * loss
```

## Proposed Checkpoint Format

### Format Name: `tritter-progressive`

### File Structure

```
{model}-{size}-{step}.tprog/
├── weights.safetensors      # Model weights (safetensors format)
├── config.json              # TritterConfig
├── progressive.json         # Progressive training metadata
├── tokenizer/               # Tokenizer files
│   ├── tokenizer.json
│   └── special_tokens.json
├── training/                # Training state (optional)
│   ├── optimizer.safetensors
│   ├── scheduler.json
│   └── rng_state.pt
└── ewc/                     # EWC state (optional)
    ├── fisher.safetensors
    └── old_params.safetensors
```

### progressive.json Schema

```json
{
    "format_version": "1.0.0",
    "model_family": "tritter",

    "current_size": {
        "model_size": "3B",
        "hidden_size": 2560,
        "num_layers": 26,
        "num_heads": 20,
        "num_kv_heads": 4
    },

    "training_progress": {
        "tokens_seen": 200000000000,
        "steps": 100000,
        "epochs": 2.0,
        "best_loss": 1.85
    },

    "expansion_history": [
        {
            "from_size": "1B",
            "to_size": "3B",
            "method": "depth_upscaling",
            "tokens_after": 50000000000
        }
    ],

    "recommended_expansion": {
        "next_size": "7B",
        "target_hidden_size": 4096,
        "target_num_layers": 32,
        "method": "depth_upscaling",
        "layer_duplication_indices": [13, 14, 15, 16, 17, 18]
    },

    "layer_analysis": {
        "importance_scores": [0.8, 0.7, 0.9, ...],
        "expendable_layers": [10, 11, 12],
        "critical_layers": [0, 1, 25]
    },

    "ewc_config": {
        "enabled": true,
        "lambda": 1000.0,
        "fisher_samples": 10000
    }
}
```

## Implementation Phases

### Phase 1: Basic Progressive Checkpoints (Week 1-2)

- [ ] Define `progressive.json` schema
- [ ] Implement save/load with metadata
- [ ] Add layer importance scoring
- [ ] Unit tests for format

### Phase 2: Depth Up-Scaling (Week 3-4)

- [ ] Implement layer duplication
- [ ] Noise injection for symmetry breaking
- [ ] Integration tests with actual models
- [ ] Benchmarks on 1B → 3B expansion

### Phase 3: Width Up-Scaling (Week 5-6)

- [ ] Implement Net2Net expansion
- [ ] Handle attention head expansion
- [ ] Handle MLP expansion
- [ ] Integration tests

### Phase 4: EWC Integration (Week 7-8)

- [ ] Fisher diagonal computation
- [ ] EWC regularization loss
- [ ] Combined expansion + EWC workflow
- [ ] Validation on forgetting metrics

### Phase 5: HuggingFace Integration (Week 9-10)

- [ ] Hub upload/download support
- [ ] Model cards with expansion info
- [ ] CLI tools for expansion
- [ ] Documentation

## Open Questions (Need Research)

1. **Optimal layer duplication strategy**: Middle layers vs uniform distribution?
2. **Width expansion for GQA**: How to expand with grouped query attention?
3. **BitNet compatibility**: Does quantization affect expansion quality?
4. **Minimum fine-tuning tokens**: How many tokens needed after expansion?
5. **Combined expansion**: Can we do depth + width simultaneously?
6. **Automatic size selection**: Can we learn when to expand?

## References

1. LLaMA-Pro: Progressive LLaMA with Block Expansion
2. Net2Net: Accelerating Learning via Knowledge Transfer
3. EWC: Overcoming catastrophic forgetting in neural networks
4. Progressive Neural Networks (Google DeepMind)
5. LoRA: Low-Rank Adaptation of Large Language Models

## Next Steps

1. Implement Phase 1 (basic format) to unblock other work
2. Run experiments on 1B → 3B expansion to validate DUS approach
3. Evaluate Fisher diagonal computation cost
4. Design CLI for `tritter expand --from 3B --to 7B`
