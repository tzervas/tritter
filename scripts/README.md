# Tritter Scripts

Utility scripts for model validation, training, and evaluation.

## Validation Scripts

### validate_bitnet_weights.py

Validates that Tritter architecture is compatible with Microsoft's BitNet b1.58-2B-4T weights.

**Why**: Before investing compute in training, we need to verify our architecture matches the reference BitNet implementation. This catches architecture mismatches early and enables us to use Microsoft's pretrained weights as a starting point for continued pretraining.

**Usage**:

```bash
# Basic validation (offline, uses reference config)
python scripts/validate_bitnet_weights.py

# With HuggingFace config check (requires internet)
python scripts/validate_bitnet_weights.py --check-hf

# Check different model
python scripts/validate_bitnet_weights.py --model-id microsoft/bitnet-b1.58-2B-4T-bf16
```

**What it checks**:

1. Architecture compatibility:
   - Hidden size: 2048
   - Num layers: 24
   - Num heads: 32
   - Head dimension: 64
   - Intermediate size: 5632
   - Vocab size: 152064

2. Model instantiation:
   - Creates TritterConfig matching BitNet-2B
   - Instantiates TritterModel with config
   - Counts parameters (should be ~2B)
   - Runs test forward pass

3. Output validation:
   - Verifies output shape is correct
   - Confirms gradient flow

**Expected output**:

```
============================================================
BitNet b1.58-2B-4T Architecture Validation
============================================================

BitNet Reference Configuration:
  hidden_size: 2048
  num_layers: 24
  num_heads: 32
  head_dim: 64
  intermediate_size: 5632
  vocab_size: 152064
  max_position_embeddings: 4096
  rope_theta: 10000.0

Creating compatible TritterConfig...
  hidden_size: 2048
  num_layers: 24
  num_heads: 32
  head_dim: 64
  intermediate_size: 5632
  vocab_size: 152064

Architecture is COMPATIBLE with BitNet-2B

Instantiating TritterModel...
  Total parameters: 2,048,576,512
  Expected ~2B: compatible

Testing forward pass...
  Output shape correct: torch.Size([2, 128, 152064])

============================================================
VALIDATION PASSED - Ready for continued pretraining
============================================================
```

**Exit codes**:

- 0: Validation passed
- 1: Validation failed (architecture mismatch or runtime error)

**Dependencies**:

```bash
# Core dependencies (required)
pip install torch

# Optional (for --check-hf)
pip install transformers huggingface_hub safetensors
```

**Related documents**:

- [TRAINING_STRATEGY.md](../docs/TRAINING_STRATEGY.md) - Bootstrap strategy using BitNet-2B
- [project-plan.md](../docs/project-plan.md) - Full technical blueprint
- [CLAUDE.md](../CLAUDE.md) - Architecture decisions

**Next steps after validation**:

1. Load actual BitNet-2B weights (requires weight conversion script)
2. Run continued pretraining on code datasets
3. Validate training loop and checkpointing
4. Scale to full 7B model
