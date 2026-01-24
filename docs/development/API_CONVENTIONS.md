# Tritter API Conventions & Schema

**Purpose:** Reference document for consistent API design across the codebase.

**Audience:** Contributors implementing new components or modifying existing APIs.

**Companion Document:** See `DEVELOPMENT_STANDARDS.md` for code quality requirements.

---

## Table of Contents

1. [Configuration Schema](#configuration-schema)
2. [Model Architecture Schema](#model-architecture-schema)
3. [Tokenization Schema](#tokenization-schema)
4. [Quantization Schema](#quantization-schema)
5. [Training Schema](#training-schema)
6. [Inference Schema](#inference-schema)
7. [Utility Functions Schema](#utility-functions-schema)

---

## Configuration Schema

### TritterConfig Dataclass

**Purpose:** Single source of truth for all model hyperparameters and system constraints.

**Required Attributes:**

```python
@dataclass
class TritterConfig:
    # Model Architecture (REQUIRED)
    model_size: Literal["3B", "7B"] = "3B"
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 131072  # 128K
    vocab_size: int = 65536

    # Quantization (REQUIRED)
    use_bitnet: bool = True
    bitnet_precision: float = 1.58

    # Attention Optimizations (REQUIRED)
    use_flash_attention: bool = True
    sliding_window_size: int = 4096
    use_streaming_llm: bool = True

    # Memory Optimizations (REQUIRED)
    int4_kv_cache: bool = True
    gradient_checkpointing: bool = True

    # Multimodal Configuration (REQUIRED)
    modalities: list[str] = field(default_factory=lambda: ["text", "code", "image", "audio"])
    use_early_fusion: bool = True
    unified_embedding: bool = True

    # Hardware Targeting (REQUIRED)
    target_device: str = "cuda"  # RTX 5080 optimized
    max_memory_gb: int = 16  # GDDR7

    # Training Hyperparameters (REQUIRED)
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Advanced Features (OPTIONAL)
    use_vsa: bool = False
    use_hrr: bool = False

    def __post_init__(self) -> None:
        """REQUIRED: Validate configuration and auto-configure variants."""

    @property
    def head_dim(self) -> int:
        """REQUIRED: Computed property for attention head dimension."""
```

**Validation Rules:**

1. `hidden_size % num_heads == 0` (MUST assert in `__post_init__`)
2. `modalities` ⊆ `{"text", "code", "image", "audio"}` (MUST validate)
3. `model_size == "7B"` → auto-configure `hidden_size=4096, num_layers=32, num_heads=32`
4. All memory-critical parameters must have "Why" comments explaining VRAM impact

**Documentation Requirements:**

- Class docstring: Explain overall purpose + memory budget breakdown
- Each attribute: Inline comment with "Why" + default value rationale
- `__post_init__`: Docstring explaining validation logic
- `head_dim` property: Docstring explaining computation + typical values

---

## Model Architecture Schema

### Base Model Interface

**All models MUST implement:**

```python
class TritterModel(nn.Module):
    """Transformer model with BitNet quantization and multimodal support.

    Why: [Architecture rationale, embedding-prediction paradigm]

    Attributes:
        config: TritterConfig instance
        embed: UnifiedEmbedding layer (vocab → hidden_size)
        layers: ModuleList of TritterLayer instances
        norm: Final layer normalization
        output_projection: Linear(hidden_size → vocab_size)
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize model from configuration.

        Args:
            config: Model configuration with validated hyperparameters

        Why: [Explain initialization choices, BitNet setup, memory allocation]
        """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional mask (batch_size, seq_len, seq_len)
            position_ids: Optional positions (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)

        Why: [Explain embedding-prediction paradigm, why logits are temporary]
        """
```

### Attention Layer Interface

```python
class TritterAttention(nn.Module):
    """Multi-head attention with optional BitNet quantization.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head (hidden_size // num_heads)
        hidden_size: Total hidden dimension
        q_proj: Query projection (may be TernaryWeight if BitNet enabled)
        k_proj: Key projection
        v_proj: Value projection
        o_proj: Output projection
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize attention layer.

        Args:
            config: Model configuration

        Why: [Explain head splitting, BitNet integration, RoPE usage]
        """

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        attention_mask: torch.Tensor | None = None,  # (B, L, L)
        position_ids: torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:  # (B, L, D)
        """Compute multi-head attention.

        Args:
            hidden_states: Input of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask with 0=attend, -inf=mask
            position_ids: Optional positions for RoPE encoding

        Returns:
            Attention output, same shape as input for residual connection

        Why: [Explain attention computation, why same shape matters]
        """
```

### MLP Layer Interface

```python
class TritterMLP(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Attributes:
        gate_proj: Linear(hidden_size → intermediate_size)
        up_proj: Linear(hidden_size → intermediate_size)
        down_proj: Linear(intermediate_size → hidden_size)
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize MLP layer.

        Args:
            config: Model configuration

        Why: [Explain SwiGLU choice, intermediate_size=4*hidden_size rationale]
        """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            hidden_states: Input of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output of shape (batch_size, seq_len, hidden_size)

        Why: [Explain SwiGLU formula, why it outperforms ReLU/GELU]
        """
```

---

## Tokenization Schema

### Tokenizer Interface

```python
class MultiModalTokenizer:
    """Unified tokenizer for all modalities.

    Attributes:
        vocab_size: Total vocabulary size (default 65536)
        max_length: Maximum sequence length (default 131072 = 128K)
        special_tokens: Dict mapping token strings to IDs
        modality_prefixes: Dict mapping ModalityType to prefix tokens
    """

    # Special Tokens (REQUIRED, IDs 0-7 reserved)
    PAD_TOKEN: str = "<pad>"      # ID: 0
    BOS_TOKEN: str = "<bos>"      # ID: 1
    EOS_TOKEN: str = "<eos>"      # ID: 2
    UNK_TOKEN: str = "<unk>"      # ID: 3
    TEXT_PREFIX: str = "<text>"   # ID: 4
    CODE_PREFIX: str = "<code>"   # ID: 5
    IMAGE_PREFIX: str = "<image>" # ID: 6
    AUDIO_PREFIX: str = "<audio>" # ID: 7

    def __init__(self, vocab_size: int = 65536, max_length: int = 131072) -> None:
        """Initialize tokenizer.

        Args:
            vocab_size: Size of unified vocabulary
            max_length: Maximum sequence length

        Why: [Explain vocab allocation, max_length based on memory budget]
        """

    def encode(
        self,
        content: Any,
        modality: ModalityType,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode content to token IDs.

        Args:
            content: Content to encode (type varies by modality)
            modality: Modality type (TEXT, CODE, IMAGE, AUDIO)
            add_special_tokens: Whether to add BOS/EOS/prefix tokens

        Returns:
            List of token IDs in range [0, vocab_size)

        Why: [Explain modality-specific encoding, prefix token usage]
        """

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip BOS/EOS/PAD/prefixes

        Returns:
            Decoded string

        Why: [Explain decode strategy, MUST be symmetric with encode]

        Note: CRITICAL - Must handle full range of encode's output.
        If encode produces IDs in [0, vocab_size), decode must handle same range.
        """

    # Modality-specific encoders (PRIVATE)
    def _encode_text(self, text: str) -> list[int]: ...
    def _encode_code(self, code: str) -> list[int]: ...
    def _encode_image(self, image: Any) -> list[int]: ...
    def _encode_audio(self, audio: Any) -> list[int]: ...
```

**Symmetry Requirement:**

```python
# MUST satisfy for all valid inputs:
tokenizer = MultiModalTokenizer()
tokens = tokenizer.encode(content, modality, add_special_tokens=False)
decoded = tokenizer.decode(tokens, skip_special_tokens=True)
assert decoded == content or similar(decoded, content)  # For lossy encodings
```

### Embedding Layer Interface

```python
class UnifiedEmbedding(nn.Module):
    """Unified embedding layer for all modalities.

    Attributes:
        embedding: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ) -> None:
        """Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension (MUST == config.hidden_size)
            padding_idx: Padding token ID (default 0, no gradient updates)

        Why: [Explain shared space, early fusion benefits, padding_idx=0 rationale]
        """

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens to continuous space.

        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)

        Why: [Explain entry point to embedding space, cross-modal attention]
        """
```

---

## Quantization Schema

### TernaryWeight Interface

```python
class TernaryWeight(nn.Module):
    """BitNet 1.58-bit ternary quantization layer.

    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        weight: Full-precision weights (trainable)
        bias: Optional bias term (full precision)
        scale: Learnable scaling factor
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize ternary weight layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias term

        Why: [Explain BitNet b1.58, memory savings, STE training]
        """

    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to {-1, 0, +1}.

        Args:
            weights: Full-precision weights

        Returns:
            Quantized weights in {-1, 0, +1}

        Why: [Explain AbsMean thresholding, why ternary vs binary]
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor

        Returns:
            Output using ternary weights

        Why: [Explain STE gradient flow, why maintain full-precision shadow]
        """
```

### BitNetQuantizer Utility

```python
class BitNetQuantizer:
    """Utility for converting models to BitNet.

    Methods:
        quantize_linear_layer: Convert nn.Linear → TernaryWeight
        quantize_model: Convert all Linear layers in a model
    """

    @staticmethod
    def quantize_linear_layer(
        layer: nn.Linear,
        keep_bias: bool = True,
    ) -> TernaryWeight:
        """Convert Linear layer to TernaryWeight.

        Args:
            layer: Standard PyTorch Linear layer
            keep_bias: Whether to preserve bias term

        Returns:
            TernaryWeight layer with copied parameters

        Why: [Explain in-place conversion, when to use]
        """
```

---

## Training Schema

### Trainer Interface (Planned)

```python
class Trainer:
    """Main training loop for embedding-prediction model.

    Attributes:
        model: TritterModel instance
        config: TrainingConfig with hyperparameters
        optimizer: Optimizer handling BitNet STE gradients
        scheduler: Learning rate scheduler
        dataloader: MultimodalDataLoader with balanced sampling
    """

    def __init__(
        self,
        model: TritterModel,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            train_dataset: Training data
            eval_dataset: Optional validation data

        Why: [Explain embedding-prediction training, curriculum learning]
        """

    def train(self) -> dict[str, Any]:
        """Run training loop.

        Returns:
            Training metrics and final model state

        Why: [Explain training loop structure, checkpointing, evaluation]
        """

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Epoch metrics (loss, gradient norms, etc.)

        Why: [Explain epoch structure, gradient accumulation, mixed precision]
        """
```

**Key Training Components:**

1. **EmbeddingPredictionLoss**: MSE in embedding space, not cross-entropy
2. **CurriculumScheduler**: Token prediction → embedding prediction transition
3. **MultimodalDataLoader**: Balanced sampling across modalities
4. **BitNetOptimizer**: STE-aware gradient updates
5. **MemoryOptimizer**: Gradient checkpointing for 16GB VRAM

---

## Inference Schema

### InferenceEngine Interface (Planned)

```python
class InferenceEngine:
    """Optimized inference for embedding-prediction models.

    Attributes:
        model: TritterModel instance
        config: InferenceConfig
        kv_cache: KVCacheManager with INT4 quantization
        embedding_rounder: Fast embedding→token mapping
    """

    def __init__(
        self,
        model: TritterModel,
        config: InferenceConfig,
    ) -> None:
        """Initialize inference engine.

        Args:
            model: Trained model
            config: Inference configuration (batch size, cache settings, etc.)

        Why: [Explain inference optimizations, KV-cache setup, device placement]
        """

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs (batch_size, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs (batch_size, prompt_len + max_new_tokens)

        Why: [Explain generation loop, embedding→token rounding, when to stop]
        """
```

**Key Inference Components:**

1. **KVCacheManager**: INT4-quantized cache with paging (vLLM integration)
2. **EmbeddingRounder**: KNN or VQ mapping from continuous to discrete
3. **AttentionOptimizer**: FlashAttention3 kernels
4. **BatchScheduler**: Dynamic batching for throughput
5. **MemoryTracker**: VRAM monitoring and OOM prevention

---

## Utility Functions Schema

### Device Utilities

```python
def get_optimal_device() -> torch.device:
    """Get optimal device (CUDA if available, else CPU).

    Returns:
        torch.device instance

    Why: Abstracts device selection for portability.
    """

def get_device_memory_info(device: torch.device) -> dict[str, float]:
    """Get memory statistics for device.

    Args:
        device: Device to query

    Returns:
        Dict with 'total_gb', 'allocated_gb', 'free_gb'

    Why: Monitor memory usage during training/inference.
    """

def optimize_for_rtx5080() -> None:
    """Enable optimizations for RTX 5080 Blackwell architecture.

    Why: TF32 and cuDNN benchmarking provide ~20% speedup on Blackwell.
    Must be called before model creation to affect kernel selection.
    """
```

**Usage Pattern:**

```python
from tritter.utils.device_utils import get_optimal_device, optimize_for_rtx5080

# MUST call before model creation
optimize_for_rtx5080()
device = get_optimal_device()

# Then create model
model = TritterModel(config).to(device)
```

---

## Type Annotations

### Required Type Hints

All public APIs MUST use type hints:

```python
# ✅ CORRECT
def forward(
    self,
    hidden_states: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...

# ❌ INCORRECT (missing types)
def forward(self, hidden_states, mask=None):
    ...
```

### Standard Type Aliases

```python
# Use throughout codebase for consistency
from typing import Literal

ModelSize = Literal["3B", "7B"]
DeviceType = Literal["cuda", "cpu"]
ModalityList = list[Literal["text", "code", "image", "audio"]]
```

---

## Error Handling

### Validation Errors

```python
# Use assertions for config validation (fail fast)
assert self.hidden_size % self.num_heads == 0, (
    f"hidden_size ({self.hidden_size}) must be divisible "
    f"by num_heads ({self.num_heads})"
)

# Use ValueError for runtime validation
if not 0 <= temperature <= 2.0:
    raise ValueError(
        f"temperature must be in [0, 2.0], got {temperature}"
    )

# Use TypeError for type mismatches
if not isinstance(input_ids, torch.Tensor):
    raise TypeError(
        f"input_ids must be torch.Tensor, got {type(input_ids)}"
    )
```

### Error Messages

- MUST include actual values and expected values
- MUST suggest fix if obvious
- SHOULD reference documentation if complex

```python
# ✅ GOOD error message
raise ValueError(
    f"vocab_size {vocab_size} exceeds maximum 65536. "
    f"Reduce vocab_size or increase embedding layer capacity. "
    f"See CONFIG.md for vocab allocation guidelines."
)

# ❌ BAD error message
raise ValueError("Invalid vocab_size")
```

---

## Deprecation Process

When changing public APIs:

1. Add deprecation warning in current version
2. Document replacement in docstring
3. Remove in next major version

```python
def old_function(x):
    """Old function (DEPRECATED).

    .. deprecated:: 0.2.0
        Use :func:`new_function` instead.

    Why: new_function provides better error handling and supports batching.
    """
    import warnings
    warnings.warn(
        "old_function is deprecated and will be removed in v0.3.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function(x)
```

---

## Version Compatibility

### Semantic Versioning

- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, no API changes

### Compatibility Policy

- Config format must remain loadable for all minor versions within major version
- Checkpoint format must be forward-compatible for all minor versions
- Python version support: 3.12+ (for modern type hints)

---

## Summary Checklist

When implementing new components:

- [ ] Follows naming conventions from this document
- [ ] Implements required methods for interface type
- [ ] Has proper type hints for all public methods
- [ ] Validates inputs with clear error messages
- [ ] Documents parameters with shapes and constraints
- [ ] Includes "Why" explanations in docstrings
- [ ] Handles edge cases (empty inputs, None values, etc.)
- [ ] Maintains symmetry (encode/decode, quantize/dequantize, etc.)
- [ ] Respects memory budget (RTX 5080 16GB)
- [ ] Acknowledges embedding-prediction paradigm where relevant

---

## Change Log

**v1.0 (2026-01-21)**
- Initial API conventions based on existing implementation
- Documented all major interfaces and schemas
- Established type annotation requirements
- Defined error handling patterns
