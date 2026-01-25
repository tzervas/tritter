---
license: apache-2.0
language:
- en
- code
tags:
- tritter
- bitnet
- ternary-quantization
- code-generation
- multimodal
- transformer
library_name: transformers
pipeline_tag: text-generation
model-index:
- name: Tritter-{SIZE}
  results: []
---

# Tritter-{SIZE}

A {SIZE} parameter multimodal transformer with BitNet 1.58-bit ternary quantization, optimized for efficient inference on consumer GPUs.

## Model Description

Tritter implements the **embedding-prediction paradigm** where the model operates in continuous embedding space rather than discrete token space. This architectural choice enables more expressive reasoning while maintaining compatibility with standard training pipelines.

### Key Innovations

| Feature | Description |
|---------|-------------|
| **BitNet 1.58-bit** | Ternary weights {-1, 0, +1} achieve 16x compression vs FP32 |
| **Multimodal** | Text, code (AST-aware), images (SigLIP), audio (EnCodec) |
| **128K Context** | Extended RoPE with sliding window attention |
| **Memory Efficient** | 7B model inference fits 16GB VRAM |

### Architecture Details

```
Hidden Size:        {HIDDEN_SIZE}
Layers:             {NUM_LAYERS}
Attention Heads:    {NUM_HEADS}
KV Heads:           {NUM_KV_HEADS}
FFN Intermediate:   {INTERMEDIATE_SIZE}
Vocabulary:         {VOCAB_SIZE}
Context Length:     {MAX_POSITION_EMBEDDINGS}
```

**Design Decisions:**
- **Squared ReLU** activation (required for BitNet stability)
- **QK-Norm** prevents attention score explosion
- **Post-FFN LayerNorm** (Chameleon-style) for training stability
- **Grouped Query Attention** reduces KV-cache memory by {GQA_RATIO}x

## Intended Use

### Primary Use Cases

- Code generation and completion (Python, Rust, Triton)
- Code explanation and documentation
- Technical question answering
- Repository-level code understanding (128K context)

### Out of Scope

- General conversation (not optimized for chat)
- Non-English languages (English-focused training)
- Safety-critical applications without human review

## Training

| Aspect | Details |
|--------|---------|
| **Data** | {TRAINING_TOKENS}B tokens of curated code (Stack-v2, high-quality repos) |
| **Method** | BitNet QAT (Quantization-Aware Training) with STE gradients |
| **Hardware** | {TRAINING_HARDWARE} |
| **Precision** | BF16 activations, ternary weights |

### Data Composition

- 35% Stack v2 Python (permissive licenses, >100 stars)
- 20% Stack v2 Rust (same criteria)
- 10% Triton GPU kernels (from PyTorch, JAX, ML repos)
- 20% High-quality repositories (curated, well-documented)
- 10% Technical documentation (arXiv CS, official docs)
- 5% Synthetic persona conversations (Constitutional AI)

## How to Use

### Installation

```bash
pip install tritter
# or for development
pip install git+https://github.com/tzervas/tritter.git
```

### Basic Usage

```python
from tritter import TritterConfig, TritterModel
from tritter.tokenization import MultiModalTokenizer

# Load model
config = TritterConfig(model_size="{SIZE}")
model = TritterModel.from_pretrained("tritter-ai/tritter-{size_lower}")
tokenizer = MultiModalTokenizer()

# Generate code
prompt = "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\""
inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Advanced: Layer Streaming for Large Models

```python
from tritter import TritterConfig
from tritter.inference import StreamingInferenceEngine

# Run models larger than VRAM by streaming layers
config = TritterConfig(
    model_size="{SIZE}",
    use_layer_streaming=True,
    layer_group_size=4,
    gpu_memory_budget_gb=14.0,
)

engine = StreamingInferenceEngine(model, config)
output = engine.generate(input_ids, max_new_tokens=256)
```

### QLoRA Fine-Tuning

```python
from tritter.training.lora import LoRAConfig, apply_lora, LoRATrainer

# Memory-efficient fine-tuning
lora_config = LoRAConfig(
    rank=16,
    alpha=16.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = apply_lora(model, lora_config)
trainer = LoRATrainer(model, lora_config, learning_rate=1e-4)
```

## Evaluation

| Benchmark | Score | Notes |
|-----------|-------|-------|
| HumanEval | {HUMANEVAL_SCORE} | Python code generation |
| MBPP | {MBPP_SCORE} | Python programming problems |
| MultiPL-E (Rust) | {MULTIPLE_RUST_SCORE} | Rust code generation |

*Evaluation details and reproduction scripts available in the repository.*

## Limitations

### Known Limitations

1. **Code-focused**: Primarily trained on code; less capable on general text tasks
2. **English-centric**: Limited multilingual capability
3. **Safety**: May generate insecure code patterns - always review generated code
4. **Hallucinations**: Can confidently generate incorrect code or documentation

### Ethical Considerations

- Generated code should be reviewed before use in production
- Model may perpetuate biases present in training data
- Not suitable for generating code for security-critical applications without expert review

## Technical Specifications

### Memory Requirements

| Context Length | Precision | KV-Cache | Total (Inference) |
|----------------|-----------|----------|-------------------|
| 4K | INT4 | {KV_4K} GB | {TOTAL_4K} GB |
| 32K | INT4 | {KV_32K} GB | {TOTAL_32K} GB |
| 128K | INT4 | {KV_128K} GB | {TOTAL_128K} GB |

### Hardware Compatibility

| GPU | Inference | QLoRA Training |
|-----|-----------|----------------|
| RTX 5080 (16GB) | 128K context | Yes |
| RTX 4090 (24GB) | 128K context | Yes |
| RTX 3080 (10GB) | 32K context | rank=8 |
| A100 (40GB) | 128K context | Yes |

## Model Files

| File | Description |
|------|-------------|
| `model.safetensors` | Model weights in safetensors format |
| `config.json` | Model configuration |
| `generation_config.json` | Generation defaults |
| `tokenizer.json` | Tokenizer vocabulary |

## Citation

```bibtex
@misc{tritter2026,
  title={Tritter: A BitNet Multimodal Transformer for Code},
  author={Tritter Contributors},
  year={2026},
  url={https://github.com/tzervas/tritter},
  note={BitNet 1.58-bit ternary quantization for efficient inference}
}
```

## License

This model is released under the **Apache 2.0** license.

## Acknowledgments

- BitNet paper: [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- Architecture inspiration: Llama, Mistral, Chameleon
- Training infrastructure: PyTorch, HuggingFace

---

*Model card generated by [Tritter](https://github.com/tzervas/tritter)*
