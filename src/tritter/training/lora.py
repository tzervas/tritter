"""LoRA and QLoRA implementation for memory-efficient fine-tuning.

Why: Full fine-tuning of large models (7B+) requires massive GPU memory for:
1. Full-precision weights (for gradient computation)
2. Gradients (same size as weights)
3. Optimizer states (2x weights for AdamW: momentum and variance)

For 7B model: ~28GB weights + ~28GB gradients + ~56GB optimizer = ~112GB minimum.
This exceeds RTX 5080's 16GB by ~7x.

LoRA (Low-Rank Adaptation) solution:
- Freeze pretrained weights (no gradients, no optimizer states)
- Add small trainable low-rank matrices A and B
- Output = base_output + (x @ A @ B) * scaling
- Typical rank=16: reduces trainable params from 7B to ~17M (0.2%)
- Memory: ~34MB weights + ~34MB gradients + ~134MB optimizer = ~200MB

QLoRA extension:
- Base model uses quantized weights (ternary in our case)
- LoRA adapters remain in FP16/BF16 for training precision
- Enables 7B fine-tuning on 16GB: ~1.4GB base + ~200MB LoRA + ~8GB KV-cache

Reference: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
           Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from tritter.quantization.bitnet import TernaryWeight
from tritter.quantization.packed_ternary import PackedTernaryWeight

if TYPE_CHECKING:
    from tritter.core.config import TritterConfig


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    Why: Centralizes LoRA hyperparameters for easy experimentation.
    Default values follow the original LoRA paper recommendations.

    Attributes:
        rank: Rank of the low-rank decomposition. Higher = more capacity but more memory.
              Typical values: 4, 8, 16, 32, 64. Start with 16 for most tasks.
        alpha: Scaling factor. The actual scaling applied is alpha/rank.
               Typical: alpha=rank or alpha=2*rank. Higher alpha = stronger adaptation.
        dropout: Dropout applied to LoRA output for regularization.
        target_modules: List of module name patterns to apply LoRA to.
                       Supports regex patterns. Default targets attention projections.
        modules_to_save: Module names to keep trainable (not just LoRA-adapted).
                        Useful for task-specific heads.
        bias: How to handle bias terms. Options:
              - "none": Don't train any biases
              - "lora_only": Only train biases in LoRA layers
              - "all": Train all biases in target modules
        use_rslora: Use Rank-Stabilized LoRA scaling (alpha/sqrt(rank) instead of alpha/rank).
                   Better for very low or high ranks.
        init_lora_weights: Initialization strategy for LoRA weights.
                          - "gaussian": Initialize A with Kaiming, B with zeros
                          - "pissa": Principal singular value adaptation (advanced)
    """

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention projections
        ]
    )
    modules_to_save: list[str] = field(default_factory=list)
    bias: str = "none"  # "none", "lora_only", "all"
    use_rslora: bool = False
    init_lora_weights: str = "gaussian"  # "gaussian" or "pissa"

    def __post_init__(self) -> None:
        """Validate configuration.

        Why: Early validation prevents cryptic errors during training.
        """
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.bias not in ("none", "lora_only", "all"):
            raise ValueError(f"bias must be 'none', 'lora_only', or 'all', got {self.bias}")
        if self.init_lora_weights not in ("gaussian", "pissa"):
            raise ValueError(
                f"init_lora_weights must be 'gaussian' or 'pissa', got {self.init_lora_weights}"
            )

    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor.

        Why: Scaling controls the magnitude of LoRA's contribution to the output.
        Standard scaling is alpha/rank, which keeps the initial LoRA contribution
        roughly constant regardless of rank choice.

        RS-LoRA uses alpha/sqrt(rank) for better stability with extreme ranks.
        """
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


class LoRALinear(nn.Module):  # type: ignore[misc]
    """Linear layer with LoRA adaptation.

    Why: Implements the core LoRA computation:
        output = base_layer(x) + (x @ A @ B) * scaling

    The base layer is frozen and only A, B matrices are trained.
    A is initialized with Kaiming/He for proper gradient flow.
    B is initialized to zeros so LoRA starts as identity (no change to base).

    Supports wrapping:
    - nn.Linear: Standard PyTorch linear layer
    - TernaryWeight: BitNet quantized layer (QLoRA)
    - PackedTernaryWeight: Packed inference layer (QLoRA)
    """

    base_layer: nn.Linear | TernaryWeight | PackedTernaryWeight

    def __init__(
        self,
        base_layer: nn.Linear | TernaryWeight | PackedTernaryWeight,
        config: LoRAConfig,
    ) -> None:
        """Initialize LoRA adapter around base layer.

        Args:
            base_layer: The frozen base layer to adapt
            config: LoRA configuration

        Why: We wrap the base layer rather than replacing it to:
        1. Preserve base layer weights exactly (no conversion loss)
        2. Enable easy merging/unmerging of LoRA weights
        3. Support different base layer types (Linear, TernaryWeight, etc.)
        """
        super().__init__()
        self.config = config

        # Store base layer (will be frozen)
        self.base_layer = base_layer

        # Get dimensions from base layer
        if isinstance(base_layer, (nn.Linear, TernaryWeight)):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif isinstance(base_layer, PackedTernaryWeight):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise TypeError(
                f"base_layer must be Linear, TernaryWeight, or PackedTernaryWeight, "
                f"got {type(base_layer)}"
            )

        # LoRA matrices
        # A: (in_features, rank) - projects input to low-rank space
        # B: (rank, out_features) - projects back to output space
        # Why separate A and B instead of single low-rank matrix:
        #   - Allows different initialization strategies
        #   - B=0 init means LoRA starts as identity
        #   - Standard practice from original paper
        # Why use same device/dtype as base layer:
        #   - apply_lora may be called after model.to(device)
        #   - Ensures LoRA params are on same device as base for forward pass
        # Note: PackedTernaryWeight uses buffers (uint8) not parameters,
        #       so we get device from buffers but use float32 for LoRA params
        try:
            base_tensor = next(base_layer.parameters())
            base_device = base_tensor.device
            base_dtype = base_tensor.dtype
        except StopIteration:
            # PackedTernaryWeight stores packed data as uint8 buffers
            # Use device from buffers but default to float32 for trainable LoRA params
            base_tensor = next(base_layer.buffers())
            base_device = base_tensor.device
            base_dtype = torch.float32  # LoRA needs float dtype for gradients
        self.lora_A = nn.Parameter(
            torch.zeros(self.in_features, config.rank, device=base_device, dtype=base_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(config.rank, self.out_features, device=base_device, dtype=base_dtype)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()

        # Scaling factor
        self.scaling = config.scaling

        # Initialize weights
        self._init_weights()

        # Freeze base layer
        self._freeze_base()

    def _init_weights(self) -> None:
        """Initialize LoRA weights.

        Why: Proper initialization is critical for training stability.
        - A uses Kaiming/He init for ReLU-like activation (good general choice)
        - B is zeros so LoRA(x) = x @ A @ B * scaling = 0 initially
        This means the model starts identical to the pretrained base.
        """
        if self.config.init_lora_weights == "gaussian":
            # Kaiming uniform for A (fan_in mode, like nn.Linear)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # B initialized to zeros so LoRA starts as identity
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "pissa":
            # PiSSA: Initialize from principal singular vectors of base weights
            # More advanced but requires SVD computation
            self._init_pissa()

    def _init_pissa(self) -> None:
        """Initialize using Principal Singular value adaptation (PiSSA).

        Why: PiSSA initializes LoRA from the principal components of base weights,
        providing a better starting point than random initialization for tasks
        that need to preserve base model behavior while adapting.
        """
        with torch.no_grad():
            # Get base weights
            if isinstance(self.base_layer, (nn.Linear, TernaryWeight)):
                W = self.base_layer.weight.data  # (out, in)
            elif isinstance(self.base_layer, PackedTernaryWeight):
                # Unpack for SVD
                from tritter.quantization.packed_ternary import unpack_ternary

                W = unpack_ternary(
                    self.base_layer.packed_weight,
                    self.base_layer.scale,
                    self.base_layer.in_features,
                )  # (out, in)
            else:
                # Fallback to Gaussian
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                return

            # SVD: W = U @ S @ V^T
            # Take top-r singular vectors
            # A = V[:, :r] * sqrt(S[:r])  -> (in, r)
            # B = sqrt(S[:r]) * U[:, :r]^T -> (r, out)
            try:
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                r = self.config.rank

                # Scale by sqrt of singular values
                sqrt_S = torch.sqrt(S[:r] + 1e-8)

                # A: V^T[:r, :].T = V[:, :r], scaled
                self.lora_A.data = (Vh[:r, :].T * sqrt_S).to(self.lora_A.dtype)

                # B: U[:, :r] @ diag(sqrt_S) -> then transpose for (r, out)
                self.lora_B.data = (U[:, :r] * sqrt_S).T.to(self.lora_B.dtype)

            except RuntimeError:
                # SVD failed (e.g., on MPS), fall back to Gaussian
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def _freeze_base(self) -> None:
        """Freeze base layer parameters.

        Why: Base layer should not be updated during LoRA training.
        Freezing prevents gradient computation and optimizer state allocation
        for base parameters, saving massive amounts of memory.
        """
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)

        Why: Computes output = base_layer(x) + dropout(x @ A) @ B * scaling
        The base layer output captures pretrained knowledge, while LoRA
        provides task-specific adaptation. Dropout regularizes the adaptation.
        """
        # Base layer forward (frozen weights)
        base_output = self.base_layer(x)  # (..., out_features)

        # LoRA forward: x @ A @ B * scaling
        # Apply dropout to intermediate representation for regularization
        lora_output = self.dropout(x @ self.lora_A)  # (..., rank)
        lora_output = lora_output @ self.lora_B  # (..., out_features)
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> None:
        """Merge LoRA weights into base layer for inference.

        Why: After training, merging LoRA into base layer eliminates the
        LoRA computation overhead during inference. The merged model is
        mathematically equivalent but faster.

        Note: Only works with nn.Linear and TernaryWeight base layers.
        PackedTernaryWeight would need unpacking, merging, and repacking.
        """
        if isinstance(self.base_layer, nn.Linear):
            with torch.no_grad():
                # W_merged = W + A @ B * scaling
                delta = (self.lora_A @ self.lora_B).T * self.scaling  # (out, in)
                self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)
        elif isinstance(self.base_layer, TernaryWeight):
            with torch.no_grad():
                # Add to shadow weights (will be re-quantized in forward)
                delta = (self.lora_A @ self.lora_B).T * self.scaling
                self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)
                self.base_layer._cache_valid = False  # Invalidate quantized cache
        else:
            raise NotImplementedError(
                f"merge_weights not supported for {type(self.base_layer)}. "
                "Use unmerged model for inference."
            )

    def unmerge_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor) -> None:
        """Unmerge previously merged LoRA weights.

        Args:
            lora_A: Original A matrix before merge
            lora_B: Original B matrix before merge

        Why: Allows reverting a merged model back to base + LoRA form,
        useful for continuing training or swapping adapters.
        """
        if isinstance(self.base_layer, nn.Linear):
            with torch.no_grad():
                delta = (lora_A @ lora_B).T * self.scaling
                self.base_layer.weight.data -= delta.to(self.base_layer.weight.dtype)
        elif isinstance(self.base_layer, TernaryWeight):
            with torch.no_grad():
                delta = (lora_A @ lora_B).T * self.scaling
                self.base_layer.weight.data -= delta.to(self.base_layer.weight.dtype)
                self.base_layer._cache_valid = False
        else:
            raise NotImplementedError(f"unmerge_weights not supported for {type(self.base_layer)}")


def _module_name_matches(name: str, patterns: list[str]) -> bool:
    """Check if module name matches any pattern.

    Args:
        name: Module name (e.g., "layers.0.attention.q_proj")
        patterns: List of patterns to match (e.g., ["q_proj", "v_proj"])

    Returns:
        True if name ends with or matches any pattern

    Why: Flexible matching allows targeting specific layer types across
    the entire model without hardcoding full paths.
    """
    for pattern in patterns:
        # Check if pattern is a regex
        if "*" in pattern or "?" in pattern or "[" in pattern:
            # Convert glob-style to regex
            regex = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
            if re.search(regex, name):
                return True
        # Simple suffix match (most common case)
        elif name.endswith(pattern) or name == pattern:
            return True
    return False


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Apply LoRA adapters to a model.

    Args:
        model: Model to adapt (will be modified in-place)
        config: LoRA configuration

    Returns:
        The modified model with LoRA adapters

    Why: Recursively traverses the model and replaces matching layers with
    LoRA-wrapped versions. The base layer is frozen internally by LoRALinear.

    Example:
        >>> model = TritterModel(config)
        >>> lora_config = LoRAConfig(rank=16, target_modules=["q_proj", "v_proj"])
        >>> model = apply_lora(model, lora_config)
        >>> # Only LoRA parameters are trainable now
    """
    # Track which modules to replace (can't modify during iteration)
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []

    for name, module in model.named_modules():
        # Check if this module should get LoRA
        if _module_name_matches(name, config.target_modules):
            # Check if it's a supported layer type
            if isinstance(module, (nn.Linear, TernaryWeight, PackedTernaryWeight)):
                # Find parent module and attribute name
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]

                # Create LoRA wrapper
                lora_layer = LoRALinear(module, config)
                replacements.append((parent, attr_name, lora_layer))

    # Apply replacements
    for parent, attr_name, lora_layer in replacements:
        setattr(parent, attr_name, lora_layer)

    # Handle modules_to_save (keep them trainable)
    for name, param in model.named_parameters():
        # Check if this parameter should stay trainable
        should_save = any(
            _module_name_matches(name, [pattern]) for pattern in config.modules_to_save
        )
        if not should_save:
            # Check if it's a LoRA parameter (those should be trainable)
            is_lora = "lora_A" in name or "lora_B" in name
            if not is_lora:
                param.requires_grad = False

    # Handle bias training based on config
    if config.bias == "none":
        # Freeze all biases
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = False
    elif config.bias == "lora_only":
        # Only unfreeze biases in LoRA layers
        for _name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                if hasattr(module.base_layer, "bias") and module.base_layer.bias is not None:
                    module.base_layer.bias.requires_grad = True
    elif config.bias == "all":
        # Unfreeze all biases in target modules
        for name, module in model.named_modules():
            if _module_name_matches(name, config.target_modules):
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = True

    return model


def get_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Get only LoRA parameters from a model.

    Args:
        model: Model with LoRA adapters

    Yields:
        LoRA parameters (A and B matrices)

    Why: For optimizer creation, we only want to optimize LoRA parameters.
    This generator filters out frozen base parameters.

    Example:
        >>> optimizer = AdamW(get_lora_parameters(model), lr=1e-4)
    """
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            yield param


def get_trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Get all trainable parameters from a model.

    Args:
        model: Model (with or without LoRA)

    Yields:
        All parameters with requires_grad=True

    Why: More general than get_lora_parameters - includes biases and
    modules_to_save if configured.
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and LoRA parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts:
        - total: All parameters
        - trainable: Parameters with requires_grad=True
        - lora: LoRA A and B parameters
        - frozen: Parameters with requires_grad=False

    Why: Useful for verifying LoRA setup and memory estimation.
    """
    total = 0
    trainable = 0
    lora = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            if "lora_A" in name or "lora_B" in name:
                lora += count

    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "frozen": total - trainable,
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base layers.

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with merged weights (LoRA layers become base layers)

    Why: For deployment, merging eliminates LoRA overhead during inference.
    The merged model produces identical outputs but runs faster.

    Warning: This modifies base layer weights. Keep a backup or use
    save_lora_adapters before merging if you need to unmerge later.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    return model


def save_lora_adapters(model: nn.Module, path: str) -> None:
    """Save only LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        path: Path to save adapters

    Why: LoRA adapters are tiny compared to full model. Saving only adapters
    enables efficient storage and sharing of fine-tuned models.

    File format:
        {
            "lora_config": LoRAConfig dict,
            "adapters": {
                "layer_name.lora_A": tensor,
                "layer_name.lora_B": tensor,
                ...
            }
        }
    """
    adapters = {}
    config_dict = None

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[f"{name}.lora_A"] = module.lora_A.data.cpu()
            adapters[f"{name}.lora_B"] = module.lora_B.data.cpu()
            if config_dict is None:
                config_dict = {
                    "rank": module.config.rank,
                    "alpha": module.config.alpha,
                    "dropout": module.config.dropout,
                    "target_modules": module.config.target_modules,
                    "bias": module.config.bias,
                    "use_rslora": module.config.use_rslora,
                    "init_lora_weights": module.config.init_lora_weights,
                }

    if not adapters:
        raise ValueError("No LoRA adapters found in model")

    torch.save(
        {
            "lora_config": config_dict,
            "adapters": adapters,
        },
        path,
    )


def load_lora_adapters(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    """Load LoRA adapters into a model.

    Args:
        model: Base model (should already have LoRA applied with same config)
        path: Path to saved adapters
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Model with loaded adapters

    Why: Enables loading fine-tuned adapters without reloading the full base model.
    Useful for:
    - Swapping between different fine-tuned versions
    - Deploying different task-specific adapters

    Example:
        >>> model = TritterModel(config)
        >>> model = apply_lora(model, lora_config)
        >>> model = load_lora_adapters(model, "checkpoints/lora_adapters.pt")
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    adapters = checkpoint["adapters"]

    loaded = set()
    expected = set()

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            expected.add(a_key)
            expected.add(b_key)

            if a_key in adapters and b_key in adapters:
                module.lora_A.data = adapters[a_key].to(module.lora_A.device)
                module.lora_B.data = adapters[b_key].to(module.lora_B.device)
                loaded.add(a_key)
                loaded.add(b_key)

    if strict:
        missing = expected - loaded
        unexpected = set(adapters.keys()) - loaded
        if missing:
            raise ValueError(f"Missing adapter keys: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected adapter keys: {unexpected}")

    return model


def estimate_lora_memory(
    model_config: TritterConfig,  # noqa: F821 - forward reference
    lora_config: LoRAConfig,
    dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    """Estimate memory usage for LoRA fine-tuning.

    Args:
        model_config: Model configuration (for dimensions)
        lora_config: LoRA configuration
        dtype: Data type for LoRA parameters

    Returns:
        Dictionary with memory estimates in GB:
        - lora_params: LoRA parameter storage
        - lora_gradients: Gradient storage for LoRA params
        - lora_optimizer: AdamW optimizer state (2x params for m and v)
        - total_lora: Total LoRA memory overhead

    Why: Pre-compute memory requirements before training to verify
    the configuration will fit in available VRAM.
    """
    # Count LoRA parameters
    # Each target module: in_features * rank + rank * out_features
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 2)

    # Estimate number of target modules
    # Attention: q, k, v, o projections
    num_attention_targets = sum(
        1 for t in lora_config.target_modules if t in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    # Per-layer LoRA params
    h = model_config.hidden_size
    ffn = model_config.intermediate_size
    r = lora_config.rank
    num_layers = model_config.num_layers

    # Attention LoRA params per layer: 4 projections × (h×r + r×h)
    attention_params = num_attention_targets * (h * r + r * h)

    # MLP LoRA params per layer
    # gate/up: h×r + r×ffn, down: ffn×r + r×h
    mlp_params = 0
    for t in lora_config.target_modules:
        if t == "gate_proj" or t == "up_proj":
            mlp_params += h * r + r * ffn
        elif t == "down_proj":
            mlp_params += ffn * r + r * h

    # Total LoRA params
    total_params = num_layers * (attention_params + mlp_params)

    # Memory calculations (in GB)
    param_bytes = total_params * bytes_per_element
    param_gb = param_bytes / (1024**3)

    # Gradients: same size as params
    grad_gb = param_gb

    # AdamW optimizer: 2 states (m and v) in FP32
    optimizer_gb = total_params * 4 * 2 / (1024**3)

    return {
        "lora_params_gb": param_gb,
        "lora_gradients_gb": grad_gb,
        "lora_optimizer_gb": optimizer_gb,
        "total_lora_gb": param_gb + grad_gb + optimizer_gb,
        "total_params": total_params,
    }


class LoRATrainer:
    """Trainer specialized for LoRA fine-tuning.

    Why: Extends base trainer functionality with LoRA-specific features:
    - Automatic LoRA parameter filtering for optimizer
    - Memory-efficient gradient accumulation
    - LoRA-specific checkpointing (save adapters separately)
    - Adapter merging for deployment

    This class is meant to work with the existing Trainer infrastructure
    while providing LoRA-specific conveniences.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize LoRA trainer.

        Args:
            model: Model with LoRA already applied
            lora_config: LoRA configuration
            learning_rate: Learning rate for LoRA parameters
            weight_decay: Weight decay (typically 0 for LoRA)

        Why: LoRA typically uses lower weight decay than full fine-tuning
        because the adapter weights are small and regularization can
        prevent effective adaptation.
        """
        self.model = model
        self.lora_config = lora_config

        # Create optimizer with only trainable (LoRA) parameters
        trainable_params = list(get_trainable_parameters(model))
        if not trainable_params:
            raise ValueError("No trainable parameters found. Did you apply LoRA?")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Log parameter counts
        counts = count_parameters(model)
        self._param_counts = counts

    def get_param_counts(self) -> dict[str, int]:
        """Get parameter count summary.

        Returns:
            Dictionary with total, trainable, lora, and frozen counts
        """
        return self._param_counts

    def save_checkpoint(self, path: str, save_base: bool = False) -> None:
        """Save LoRA checkpoint.

        Args:
            path: Checkpoint directory path
            save_base: If True, also save base model (default: False)

        Why: By default, only saves LoRA adapters (tiny).
        Full model can be reconstructed from base + adapters.
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Always save LoRA adapters
        save_lora_adapters(self.model, os.path.join(path, "lora_adapters.pt"))

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

        # Optionally save full model
        if save_base:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def load_checkpoint(self, path: str) -> None:
        """Load LoRA checkpoint.

        Args:
            path: Checkpoint directory path
        """
        import os

        # Load LoRA adapters
        load_lora_adapters(self.model, os.path.join(path, "lora_adapters.pt"))

        # Load optimizer state
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=True)
            )
