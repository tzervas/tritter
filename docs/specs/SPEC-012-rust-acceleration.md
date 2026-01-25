# SPEC-012: Rust Acceleration via rust-ai Workspace

## Overview

This specification outlines the integration of Rust crates from the `rust-ai` workspace to accelerate training and inference in Tritter.

## Rust AI Workspace Crates

Location: `/home/kang/Documents/projects/rust-ai/`

### Available Crates

| Crate | Version | Purpose | Python Bindings |
|-------|---------|---------|-----------------|
| **bitnet-rs** | 0.1.0 | BitNet b1.58 ternary quantization | Planned |
| **ternary-rs** (trit-vsa) | 0.1.0 | Packed ternary arithmetic & VSA ops | Priority |
| **vsa-optim-rs** | 0.1.0 | Gradient compression & optimization | PyO3 Ready |
| **peft-rs** | 1.0.0 | Parameter-efficient fine-tuning adapters | Planned |
| **qlora-rs** | 1.0.0 | 4-bit quantized LoRA | Planned |
| **unsloth-rs** | 0.1.0-alpha | GPU-optimized transformer kernels | Planned |
| **axolotl-rs** | 1.0.1 | High-level training orchestration | CLI only |

## Integration Priority

### Phase 1: Core Ternary Operations (Immediate)

**Goal**: Replace Python ternary operations with Rust for 10-100x speedup.

**Crates**: `ternary-rs`, `bitnet-rs`

**Integration Points**:
1. `tritter.quantization.bitnet.TernaryWeight` → `bitnet_rs::BitLinear`
2. `tritter.quantization.packed_ternary` → `trit_vsa::PackedTritVec`
3. Weight packing/unpacking for inference

**Python Bindings** (PyO3):
```rust
#[pyfunction]
fn pack_ternary_weights(weights: PyReadonlyArray2<f32>) -> PyResult<PyObject> {
    // Convert to PackedTritVec
}

#[pyfunction]
fn bitlinear_forward(
    input: PyReadonlyArray2<f32>,
    packed_weights: &PackedTritVec,
    scale: f32,
) -> PyResult<PyObject> {
    // Efficient ternary matmul
}
```

### Phase 2: VSA Gradient Optimization

**Goal**: Accelerate training with Rust gradient compression.

**Crates**: `vsa-optim-rs`

**Integration Points**:
1. `tritter.training.optimization` → `vsa_optim_rs`
2. Gradient compression during distributed training
3. Phase-based training acceleration

**Note**: vsa-optim-rs already has PyO3 support via the `python` feature flag.

### Phase 3: LoRA/QLoRA Acceleration

**Goal**: Fast adapter operations in Rust.

**Crates**: `peft-rs`, `qlora-rs`

**Integration Points**:
1. `tritter.training.lora.LoRALinear` → `peft_rs::LoraLayer`
2. 4-bit quantization → `qlora_rs::QuantizedLinear`
3. Adapter merging and switching

### Phase 4: GPU Kernels

**Goal**: Custom CUDA kernels for BitNet/ternary operations.

**Crates**: `unsloth-rs`

**Integration Points**:
1. Fused attention with ternary weights
2. Optimized RoPE for BitNet
3. Memory-efficient backward pass

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python (tritter)                           │
├─────────────────────────────────────────────────────────────────┤
│  tritter.quantization  │  tritter.training  │  tritter.inference│
│         ↓              │         ↓          │         ↓         │
├─────────────────────────────────────────────────────────────────┤
│                    PyO3 Bindings Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                      Rust (rust-ai)                             │
├──────────────────┬──────────────────┬──────────────────────────┤
│    bitnet-rs     │   vsa-optim-rs   │      peft-rs/qlora-rs    │
│        ↓         │         ↓        │            ↓              │
├──────────────────┴──────────────────┴──────────────────────────┤
│                      trit-vsa (ternary-rs)                      │
├─────────────────────────────────────────────────────────────────┤
│                   candle-core (tensor ops)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Step 1: Create Python Extension Module

Create `tritter-accel` crate in rust-ai workspace:

```toml
# Cargo.toml
[package]
name = "tritter-accel"
version = "0.1.0"

[dependencies]
bitnet-rs = { path = "../bitnet-rs" }
trit-vsa = { path = "../ternary-rs" }
vsa-optim-rs = { path = "../vsa-optim-rs", features = ["python"] }
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"

[lib]
crate-type = ["cdylib"]
name = "tritter_accel"
```

### Step 2: Build with Maturin

```bash
cd rust-ai
maturin develop --release -m tritter-accel/Cargo.toml
```

### Step 3: Use in Tritter

```python
# In tritter/quantization/bitnet.py
try:
    from tritter_accel import pack_ternary, bitlinear_forward
    USE_RUST_ACCEL = True
except ImportError:
    USE_RUST_ACCEL = False

class TernaryWeight(nn.Module):
    def forward(self, x):
        if USE_RUST_ACCEL:
            return bitlinear_forward(x, self._packed_weights, self.scale)
        else:
            # Python fallback
            ...
```

## Expected Performance Gains

| Operation | Python (torch) | Rust (trit-vsa) | Speedup |
|-----------|---------------|-----------------|---------|
| Weight packing | 100ms | 5ms | 20x |
| Ternary matmul | 50ms | 2ms | 25x |
| Gradient compression | 200ms | 10ms | 20x |
| LoRA forward | 10ms | 1ms | 10x |

## Memory Savings

| Component | Python | Rust | Reduction |
|-----------|--------|------|-----------|
| 7B weights (packed) | 1.8 GB | 1.45 GB | 20% |
| Gradient compression | 400 MB | 40 MB | 90% |
| LoRA adapters | 50 MB | 35 MB | 30% |

## Testing Strategy

1. **Unit tests**: Compare Python and Rust outputs for numerical equivalence
2. **Benchmark tests**: Measure speedup for each operation
3. **Integration tests**: Full training loop with Rust acceleration
4. **Stress tests**: Memory pressure under large batch sizes

## Rollout Plan

1. **v0.3.0**: Optional Rust acceleration (feature flag)
2. **v0.4.0**: Rust as default with Python fallback
3. **v1.0.0**: Full Rust acceleration required for optimal performance

## Dependencies

- Rust 1.75+ (for stable async traits)
- PyO3 0.22+
- Maturin 1.4+
- NumPy 1.24+

## References

- [rust-ai workspace](https://github.com/tzervas/rust-ai)
- [PyO3 documentation](https://pyo3.rs/)
- [Maturin documentation](https://maturin.rs/)
- [BitNet paper](https://arxiv.org/abs/2402.17764)
