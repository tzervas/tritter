# Progressive Model Training Specification

**Spec ID**: SPEC-008
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-24
**Related ADR**: [ADR-003](../adr/003-progressive-model-training.md)

## 1. Overview

### 1.1 Purpose

This specification defines the progressive model training system for Tritter, enabling users to:
- Start training with smaller models (e.g., 3B parameters)
- Expand to larger models (7B, 10B, 100B+) preserving learned knowledge
- Add specialized capabilities through expert expansion
- Incrementally improve models over time without full retraining

### 1.2 Goals

| Goal | Description | Success Criteria |
|------|-------------|------------------|
| Knowledge preservation | Expanded model retains base capabilities | <5% performance drop on base benchmarks |
| Compute efficiency | Expansion faster than training from scratch | ≥40% compute savings |
| Flexibility | Support multiple expansion strategies | DUS, width scaling, MoE all functional |
| Accessibility | Enable incremental training for individuals | Works on single GPU for small expansions |

### 1.3 Non-Goals

- Full architectural search (NAS) - out of scope
- Cross-architecture transfer (e.g., Mamba to Transformer)
- Compression/pruning (inverse of expansion)

---

## 2. Architecture

### 2.1 Expansion Strategies

```
                    ┌─────────────────────────────────────┐
                    │     Progressive Training System     │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
    │ Depth       │           │ Width       │           │ MoE         │
    │ Up-Scaling  │           │ Up-Scaling  │           │ Progression │
    │ (DUS)       │           │ (Net2Net)   │           │             │
    └─────────────┘           └─────────────┘           └─────────────┘
           │                          │                          │
           ▼                          ▼                          ▼
    ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
    │ Add layers  │           │ Expand dims │           │ Add experts │
    │ by duplication          │ by splitting│           │ for domains │
    └─────────────┘           └─────────────┘           └─────────────┘
```

### 2.2 Supported Expansion Paths

| Source | Target | Method | Compute Savings |
|--------|--------|--------|-----------------|
| 3B | 7B | DUS (depth) | ~45% |
| 7B | 10B | DUS (depth) | ~40% |
| 3B | 10B | DUS + width | ~35% |
| 7B | 14B | Width scaling | ~40% |
| Any | +Expert | MoE addition | Variable |

---

## 3. Depth Up-Scaling (DUS)

### 3.1 Algorithm

Based on [SOLAR 10.7B](https://arxiv.org/abs/2312.15166):

```python
def depth_upscale(
    model: TritterModel,
    source_layers: int,
    target_layers: int,
    removal_layers: int = 8,
) -> TritterModel:
    """Expand model depth via layer duplication and concatenation.

    Args:
        model: Source model with source_layers transformer blocks
        source_layers: Number of layers in source model
        target_layers: Desired number of layers in target model
        removal_layers: Layers to remove from each copy before concat

    Returns:
        Expanded model with target_layers

    Why: DUS preserves learned representations by keeping layer weights intact.
    Removing middle layers and concatenating prevents excessive depth while
    maintaining early (feature extraction) and late (task-specific) layers.

    Example (SOLAR-style 32 → 48):
        - Source: 32 layers
        - Duplicate: 2 copies of 32 layers each
        - Remove last 8 from copy 1: layers 0-23
        - Remove first 8 from copy 2: layers 8-31
        - Concatenate: 24 + 24 = 48 layers
    """
    # Step 1: Duplicate model weights
    weights_copy1 = deepcopy(model.layers[:source_layers - removal_layers])
    weights_copy2 = deepcopy(model.layers[removal_layers:])

    # Step 2: Create new model with target architecture
    new_config = model.config.copy()
    new_config.num_layers = target_layers
    new_model = TritterModel(new_config)

    # Step 3: Initialize with concatenated weights
    for i, layer_weights in enumerate(weights_copy1 + weights_copy2):
        new_model.layers[i].load_state_dict(layer_weights.state_dict())

    # Step 4: Copy embedding and output projection
    new_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
    new_model.lm_head.load_state_dict(model.lm_head.state_dict())

    return new_model
```

### 3.2 Layer Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| SOLAR | Remove middle layers from each copy | General purpose |
| LLaMA Pro | Add new layers at specific positions | Domain adaptation |
| Stack | Duplicate entire model and stack | Maximum depth |
| Interpolate | Insert copies between existing layers | Gradual expansion |

### 3.3 Continued Pretraining

After DUS expansion, continued pretraining is **required** to:
- Adapt copied layers to new positions
- Learn cross-layer interactions
- Recover performance drop from concatenation

```python
@dataclass
class ContinuedPretrainingConfig:
    """Configuration for post-expansion pretraining.

    Why: Newly expanded model needs training to adapt duplicated layers.
    Learning rate should be lower than original pretraining to preserve
    existing knowledge while allowing adaptation.
    """
    learning_rate: float = 1e-5  # Lower than pretraining (1e-4)
    warmup_ratio: float = 0.05
    max_steps: int = 10000  # ~5-10% of original pretraining

    # Forgetting prevention
    use_ewc: bool = True  # Elastic Weight Consolidation
    ewc_lambda: float = 1000.0
```

---

## 4. Width Up-Scaling

### 4.1 Net2WiderNet Algorithm

```python
def width_upscale(
    model: TritterModel,
    new_hidden_size: int,
    new_intermediate_size: int | None = None,
    new_num_heads: int | None = None,
) -> TritterModel:
    """Expand model width via function-preserving neuron splitting.

    Args:
        model: Source model
        new_hidden_size: Target hidden dimension
        new_intermediate_size: Target FFN intermediate size (default: 4x hidden)
        new_num_heads: Target attention heads (default: scale proportionally)

    Returns:
        Wider model with identical function at initialization

    Why: Width expansion increases capacity without adding layers/latency.
    Function-preserving initialization means expanded model produces identical
    outputs to source, enabling stable continued training.
    """
    old_hidden = model.config.hidden_size
    scale_factor = new_hidden_size / old_hidden

    # Expand embedding layer
    new_embed = expand_embedding(model.embed_tokens, new_hidden_size)

    # Expand each transformer layer
    new_layers = []
    for layer in model.layers:
        new_layer = expand_transformer_layer(
            layer,
            new_hidden_size,
            new_intermediate_size or int(new_hidden_size * 4),
            new_num_heads or int(model.config.num_heads * scale_factor),
        )
        new_layers.append(new_layer)

    # Expand output projection
    new_lm_head = expand_linear_out(model.lm_head, new_hidden_size)

    return assemble_model(new_embed, new_layers, new_lm_head)


def expand_linear(
    linear: nn.Linear,
    new_in_features: int,
    new_out_features: int,
) -> nn.Linear:
    """Expand linear layer with neuron splitting.

    Why: Neuron splitting preserves function by duplicating neurons and
    halving their output weights. Net2Net proved this maintains identical
    forward pass at initialization.

    Example: Expand 768 → 1024
        - Select neurons to split (256 = 1024 - 768)
        - Duplicate selected neuron rows in weight matrix
        - Halve corresponding output connection weights
    """
    old_weight = linear.weight.data  # (out, in)
    old_bias = linear.bias.data if linear.bias is not None else None

    # Create expanded weight matrix
    new_weight = torch.zeros(new_out_features, new_in_features)

    # Copy original weights to upper-left
    new_weight[:old_weight.shape[0], :old_weight.shape[1]] = old_weight

    # Split neurons for expansion
    neurons_to_add = new_out_features - old_weight.shape[0]
    if neurons_to_add > 0:
        # Randomly select neurons to duplicate
        indices = torch.randint(0, old_weight.shape[0], (neurons_to_add,))
        new_weight[old_weight.shape[0]:, :old_weight.shape[1]] = old_weight[indices]

        # Halve weights for duplicated neurons (function preservation)
        for i, idx in enumerate(indices):
            new_weight[idx] *= 0.5
            new_weight[old_weight.shape[0] + i] *= 0.5

    new_linear = nn.Linear(new_in_features, new_out_features, bias=linear.bias is not None)
    new_linear.weight.data = new_weight
    if old_bias is not None:
        new_bias = torch.zeros(new_out_features)
        new_bias[:old_bias.shape[0]] = old_bias
        new_linear.bias.data = new_bias

    return new_linear
```

### 4.2 Head Expansion

For multi-head attention, heads can be:
- **Split**: Each head becomes two heads with halved dimension
- **Added**: New heads initialized from existing head copies
- **Grouped**: Use Grouped Query Attention to add query heads without KV expansion

```python
def expand_attention_heads(
    attention: TritterAttention,
    new_num_heads: int,
    strategy: str = "split",
) -> TritterAttention:
    """Expand number of attention heads.

    Strategies:
        split: Split existing heads (requires even divisibility)
        copy: Copy and add new heads
        gqa: Add query heads only (Grouped Query Attention)
    """
    ...
```

---

## 5. Mixture of Experts (MoE) Progression

### 5.1 Progressive Expert Addition

```python
@dataclass
class MoEExpansionConfig:
    """Configuration for adding experts to existing model.

    Why: MoE enables domain-specific expansion without affecting existing
    knowledge. New experts handle new domains while frozen experts preserve
    existing capabilities.
    """
    base_experts: int = 8  # Experts in base model
    new_experts: int = 4   # Experts to add
    freeze_base: bool = True  # Freeze original experts
    router_update: str = "fine_tune"  # How to update router

    # Domain targeting
    domain: str | None = None  # e.g., "code", "vision", "math"
    domain_data_path: str | None = None  # Training data for new domain


class ProgressiveMoE:
    """Progressive Mixture of Experts for incremental scaling.

    Why: Traditional dense models require full retraining when expanding.
    MoE allows adding experts incrementally:
    1. Freeze existing experts (preserve knowledge)
    2. Add new experts (expand capacity)
    3. Retrain router (learn to dispatch)
    4. Fine-tune new experts on domain data

    This matches human learning: we don't forget English when learning French.
    """

    def add_experts(
        self,
        model: TritterMoEModel,
        config: MoEExpansionConfig,
    ) -> TritterMoEModel:
        """Add new experts to existing MoE model."""

        # Freeze base experts
        if config.freeze_base:
            for expert in model.experts[:config.base_experts]:
                for param in expert.parameters():
                    param.requires_grad = False

        # Initialize new experts
        # Option 1: Clone from best-performing expert
        # Option 2: Random initialization
        # Option 3: Knowledge distillation from dense model
        new_experts = self._create_experts(
            config.new_experts,
            init_from=model.experts[0],  # Clone from expert 0
        )

        # Expand router
        new_router = self._expand_router(
            model.router,
            config.base_experts + config.new_experts,
        )

        # Assemble expanded model
        model.experts.extend(new_experts)
        model.router = new_router
        model.num_experts = config.base_experts + config.new_experts

        return model
```

### 5.2 Router Adaptation

When adding experts, the router must learn to utilize them:

```python
def adapt_router(
    model: TritterMoEModel,
    new_expert_indices: list[int],
    domain_dataloader: DataLoader,
    epochs: int = 3,
) -> None:
    """Fine-tune router to dispatch to new experts.

    Why: New experts won't be used if router doesn't know to send tokens
    to them. Targeted fine-tuning on domain data teaches router to
    dispatch domain-specific tokens to new experts.
    """
    # Only train router, keep experts frozen initially
    for param in model.parameters():
        param.requires_grad = False
    for param in model.router.parameters():
        param.requires_grad = True

    # Train router to use new experts for domain data
    for epoch in range(epochs):
        for batch in domain_dataloader:
            # Forward pass
            outputs = model(batch["input_ids"])

            # Encourage routing to new experts for this domain
            router_logits = model.get_router_logits()
            new_expert_probs = router_logits[:, :, new_expert_indices].mean()

            # Loss: task loss + routing encouragement
            loss = outputs.loss - 0.1 * new_expert_probs
            loss.backward()
```

---

## 6. Checkpoint Compatibility

### 6.1 Checkpoint Format

```python
@dataclass
class ProgressiveCheckpoint:
    """Checkpoint format supporting progressive expansion.

    Why: Checkpoints must track expansion history for reproducibility
    and to enable further expansion from any point.
    """
    # Core state
    model_state_dict: dict
    optimizer_state_dict: dict | None
    scheduler_state_dict: dict | None

    # Expansion history
    expansion_history: list[ExpansionRecord]
    base_model_hash: str  # SHA256 of original checkpoint

    # Configuration
    model_config: TritterConfig
    training_config: TrainingConfig | None

    # Metrics
    step: int
    epoch: int
    best_loss: float
    benchmark_scores: dict[str, float]


@dataclass
class ExpansionRecord:
    """Record of a single expansion operation."""
    timestamp: str
    expansion_type: str  # "depth", "width", "moe"
    source_config: dict
    target_config: dict
    method: str  # "dus", "net2net", "ligo", etc.
    continued_training_steps: int
    performance_delta: dict[str, float]  # Benchmark changes
```

### 6.2 Expansion Path Planning

```python
def plan_expansion_path(
    source_params: int,
    target_params: int,
    available_compute: float,  # GPU-hours
    strategy: str = "auto",
) -> list[ExpansionStep]:
    """Plan optimal expansion path from source to target size.

    Args:
        source_params: Current model parameters (e.g., 3B)
        target_params: Desired model parameters (e.g., 10B)
        available_compute: Budget in GPU-hours
        strategy: Expansion strategy ("depth_first", "width_first", "balanced", "auto")

    Returns:
        List of expansion steps with estimated compute requirements

    Example: 3B → 10B with "balanced" strategy
        Step 1: 3B → 5B via DUS (depth 24 → 40)
        Step 2: 5B → 7B via width (hidden 2048 → 2560)
        Step 3: 7B → 10B via DUS (depth 40 → 56)

    Why: Breaking into multiple steps enables:
    - Validation at each checkpoint
    - Compute budget spreading
    - Early stopping if quality degrades
    """
    steps = []
    current = source_params

    while current < target_params:
        # Determine next expansion
        if strategy == "depth_first":
            step = _plan_depth_step(current, target_params)
        elif strategy == "width_first":
            step = _plan_width_step(current, target_params)
        else:
            step = _plan_balanced_step(current, target_params)

        steps.append(step)
        current = step.target_params

    return steps
```

---

## 7. Knowledge Preservation

### 7.1 Forgetting Prevention Techniques

| Technique | Description | Compute Cost | Effectiveness |
|-----------|-------------|--------------|---------------|
| EWC | Elastic Weight Consolidation - penalize changes to important weights | Low | Medium |
| SI | Synaptic Intelligence - online importance estimation | Low | Medium |
| Replay | Mix old data with new during continued training | Medium | High |
| Distillation | Use original model as teacher | High | High |
| Progressive Freezing | Freeze early layers, train later layers | Low | Medium |

### 7.2 Implementation

```python
class ElasticWeightConsolidation:
    """EWC regularization for catastrophic forgetting prevention.

    Why: During expansion and continued training, important weights for
    previous tasks may be overwritten. EWC adds regularization term that
    penalizes changes to weights proportional to their importance for
    previous tasks.

    Loss = L_new + λ * Σ F_i * (θ_i - θ*_i)²

    Where:
    - L_new: Loss on new task
    - F_i: Fisher information (importance) of parameter i
    - θ_i: Current parameter value
    - θ*_i: Parameter value after previous training
    - λ: Regularization strength
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lambda_ewc: float = 1000.0,
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc

        # Compute Fisher information
        self.fisher = self._compute_fisher(dataloader)

        # Store optimal parameters
        self.optimal_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }

    def _compute_fisher(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Compute Fisher information matrix diagonal."""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(batch["input_ids"])
            loss = output.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)

        # Normalize
        for n in fisher:
            fisher[n] /= len(dataloader)

        return fisher

    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.lambda_ewc * loss
```

---

## 8. Training Pipeline

### 8.1 Progressive Training Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│                    Progressive Training Pipeline                    │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ 1. Load Base Checkpoint │
                    │    (e.g., 3B model)     │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ 2. Run Baseline Evals   │
                    │    Store benchmark      │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ 3. Apply Expansion      │
                    │    (DUS/Width/MoE)      │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ 4. Continued Pretrain   │
                    │    + EWC regularization │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ 5. Post-Expansion Evals │
                    │    Compare to baseline  │
                    └─────────────────────────┘
                                  │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐  ┌─────────────────┐
          │ Performance OK? │  │ Performance Bad │
          │ Save checkpoint │  │ Rollback/retry  │
          └─────────────────┘  └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ Target reached? │───No──→ Go to step 3
          └─────────────────┘
                    │Yes
                    ▼
          ┌─────────────────┐
          │ Final Model     │
          │ Ready for use   │
          └─────────────────┘
```

### 8.2 CLI Interface

```bash
# Expand 3B model to 7B via DUS
tritter expand \
    --checkpoint ./checkpoints/tritter-3b \
    --target-params 7B \
    --method dus \
    --output ./checkpoints/tritter-7b-expanded

# Continue pretraining after expansion
tritter train \
    --checkpoint ./checkpoints/tritter-7b-expanded \
    --mode continue \
    --steps 10000 \
    --ewc-lambda 1000 \
    --data ./data/pretraining

# Add code expert to MoE model
tritter expand \
    --checkpoint ./checkpoints/tritter-moe-7b \
    --method moe-add \
    --num-experts 4 \
    --domain code \
    --domain-data ./data/code
```

---

## 9. Validation

### 9.1 Expansion Validation Suite

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| Forward equivalence | Expanded model produces same output (for width) | MSE < 1e-6 |
| Gradient flow | All new parameters receive gradients | grad.abs().sum() > 0 |
| Memory scaling | Memory scales as expected | Within 10% of theoretical |
| Benchmark retention | Base capabilities preserved | < 5% degradation |
| New capability | Model can learn new skills | Improvement on target task |

### 9.2 Benchmark Suite

```python
PROGRESSIVE_BENCHMARKS = {
    "base_capabilities": [
        "hellaswag",
        "arc_challenge",
        "mmlu",
        "truthfulqa",
    ],
    "code": [
        "humaneval",
        "mbpp",
    ],
    "math": [
        "gsm8k",
        "math",
    ],
    "reasoning": [
        "bbh",
        "arc_challenge",
    ],
}
```

---

## 10. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] `ModelExpander` base class
- [ ] Checkpoint versioning system
- [ ] Expansion history tracking
- [ ] Basic DUS implementation

### Phase 2: Depth Up-Scaling
- [ ] Layer duplication logic
- [ ] SOLAR-style layer removal
- [ ] Continued pretraining integration
- [ ] EWC regularization

### Phase 3: Width Up-Scaling
- [ ] Neuron splitting (Net2Net)
- [ ] Attention head expansion
- [ ] Function-preserving verification
- [ ] Combined depth+width expansion

### Phase 4: MoE Progression
- [ ] Expert addition infrastructure
- [ ] Router expansion
- [ ] Domain-targeted training
- [ ] Expert freezing/unfreezing

### Phase 5: Tooling
- [ ] CLI commands
- [ ] Expansion planning
- [ ] Benchmark automation
- [ ] Checkpoint conversion utilities

---

## Appendix A: Expansion Size Guide

| Source | Target | Method | Layers | Hidden | Approx. Params |
|--------|--------|--------|--------|--------|----------------|
| 3B | 7B | DUS | 24→48 | 2048 | 6.7B |
| 3B | 7B | Width | 24 | 2048→3072 | 6.9B |
| 7B | 10B | DUS | 32→48 | 4096 | 10.7B |
| 7B | 14B | Width | 32 | 4096→5120 | 13.8B |
| 7B | 70B | MoE (8x) | 32 | 4096 | 8x7B experts |

---

## Appendix B: References

1. [SOLAR 10.7B](https://arxiv.org/abs/2312.15166) - Depth Up-Scaling (NAACL 2024)
2. [Net2Net](https://arxiv.org/abs/1511.05641) - Function-preserving transformations
3. [bert2BERT](https://aclanthology.org/2022.acl-long.151/) - Transformer expansion
4. [LiGO](https://openreview.net/forum?id=cDYRS5iZ16f) - Learned growth operators (ICLR 2023)
5. [LEMON](https://openreview.net/forum?id=0e705ac30e57) - Lossless expansion (ICLR 2024)
6. [LLM Continual Learning Survey](https://dl.acm.org/doi/10.1145/3735633) - ACM Computing Surveys 2025
7. [MoE Survey](https://arxiv.org/html/2503.07137v1) - Comprehensive MoE review

---

## Appendix C: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-24 | Claude | Initial draft |
