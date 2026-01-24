"""Progressive checkpoint format for model expansion.

Why: Standard checkpoints don't support loading smaller models into larger
architectures. This format tracks expansion history and enables:
- Depth Up-Scaling (DUS): 3B → 7B by layer duplication
- Width Up-Scaling: Expand hidden dimensions via Net2Net
- Knowledge preservation: EWC regularization to prevent forgetting

See SPEC-011 for full specification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import safetensors.torch as st

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


@dataclass
class ExpansionStep:
    """Record of a single expansion operation."""

    from_size: str
    to_size: str
    method: str  # "depth_upscaling" or "width_upscaling"
    tokens_after: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressiveMetadata:
    """Metadata for progressive checkpoints.

    Why: Tracks model evolution for reproducibility and further expansion.
    """

    format_version: str = "1.0.0"
    model_family: str = "tritter"

    # Current architecture
    model_size: str = "1B"
    hidden_size: int = 2048
    num_layers: int = 16
    num_heads: int = 16
    num_kv_heads: int = 4
    vocab_size: int = 32000
    max_position_embeddings: int = 131072

    # Training progress
    tokens_seen: int = 0
    training_steps: int = 0
    best_loss: float = float("inf")

    # Expansion history
    expansion_history: list[ExpansionStep] = field(default_factory=list)

    # Recommended next expansion
    recommended_next_size: str | None = None
    recommended_method: str | None = None
    layer_duplication_indices: list[int] | None = None

    # EWC config
    ewc_enabled: bool = False
    ewc_lambda: float = 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "format_version": self.format_version,
            "model_family": self.model_family,
            "current_size": {
                "model_size": self.model_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
            },
            "training_progress": {
                "tokens_seen": self.tokens_seen,
                "steps": self.training_steps,
                "best_loss": self.best_loss,
            },
            "expansion_history": [
                {
                    "from_size": e.from_size,
                    "to_size": e.to_size,
                    "method": e.method,
                    "tokens_after": e.tokens_after,
                    "details": e.details,
                }
                for e in self.expansion_history
            ],
            "recommended_expansion": {
                "next_size": self.recommended_next_size,
                "method": self.recommended_method,
                "layer_duplication_indices": self.layer_duplication_indices,
            },
            "ewc_config": {
                "enabled": self.ewc_enabled,
                "lambda": self.ewc_lambda,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressiveMetadata":
        """Create from dictionary."""
        current = data.get("current_size", {})
        progress = data.get("training_progress", {})
        expansion = data.get("recommended_expansion", {})
        ewc = data.get("ewc_config", {})

        history = []
        for h in data.get("expansion_history", []):
            history.append(
                ExpansionStep(
                    from_size=h["from_size"],
                    to_size=h["to_size"],
                    method=h["method"],
                    tokens_after=h["tokens_after"],
                    details=h.get("details", {}),
                )
            )

        return cls(
            format_version=data.get("format_version", "1.0.0"),
            model_family=data.get("model_family", "tritter"),
            model_size=current.get("model_size", "unknown"),
            hidden_size=current.get("hidden_size", 0),
            num_layers=current.get("num_layers", 0),
            num_heads=current.get("num_heads", 0),
            num_kv_heads=current.get("num_kv_heads", 0),
            vocab_size=current.get("vocab_size", 0),
            max_position_embeddings=current.get("max_position_embeddings", 0),
            tokens_seen=progress.get("tokens_seen", 0),
            training_steps=progress.get("steps", 0),
            best_loss=progress.get("best_loss", float("inf")),
            expansion_history=history,
            recommended_next_size=expansion.get("next_size"),
            recommended_method=expansion.get("method"),
            layer_duplication_indices=expansion.get("layer_duplication_indices"),
            ewc_enabled=ewc.get("enabled", False),
            ewc_lambda=ewc.get("lambda", 1000.0),
        )


def save_progressive(
    model: nn.Module,
    path: str | Path,
    metadata: ProgressiveMetadata | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    fisher_diag: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Save progressive checkpoint.

    Args:
        model: Model to save
        path: Output directory
        metadata: Progressive metadata
        optimizer: Optional optimizer state
        fisher_diag: Optional EWC Fisher diagonal

    Returns:
        Path to saved checkpoint directory

    Why: Saves all information needed for model expansion.
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not installed: pip install safetensors")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save weights
    state_dict = model.state_dict()
    st.save_file(state_dict, str(path / "weights.safetensors"))

    # Save metadata
    if metadata:
        with open(path / "progressive.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    # Save optimizer
    if optimizer:
        opt_state = optimizer.state_dict()
        # Convert optimizer state to saveable format
        opt_tensors = {}
        for k, v in opt_state.get("state", {}).items():
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    opt_tensors[f"state.{k}.{sk}"] = sv

        if opt_tensors:
            st.save_file(opt_tensors, str(path / "optimizer.safetensors"))

        # Save non-tensor optimizer state
        opt_meta = {k: v for k, v in opt_state.items() if k != "state"}
        opt_meta["param_groups"] = opt_state.get("param_groups", [])
        with open(path / "optimizer.json", "w") as f:
            json.dump(opt_meta, f, indent=2, default=str)

    # Save Fisher diagonal for EWC
    if fisher_diag:
        st.save_file(fisher_diag, str(path / "fisher_diag.safetensors"))

    return path


def load_progressive(
    path: str | Path,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], ProgressiveMetadata | None]:
    """Load progressive checkpoint.

    Args:
        path: Checkpoint directory
        device: Target device

    Returns:
        Tuple of (state_dict, metadata)
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not installed: pip install safetensors")

    path = Path(path)

    # Load weights
    weights_path = path / "weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights.safetensors in {path}")

    state_dict = st.load_file(str(weights_path), device=device)

    # Load metadata
    metadata = None
    meta_path = path / "progressive.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = ProgressiveMetadata.from_dict(json.load(f))

    return state_dict, metadata


def expand_model(
    source_checkpoint: str | Path,
    target_config,  # TritterConfig
    method: str = "depth_upscaling",
    output_path: str | Path | None = None,
) -> tuple[dict[str, torch.Tensor], ProgressiveMetadata]:
    """Expand a checkpoint to a larger architecture.

    Args:
        source_checkpoint: Path to source progressive checkpoint
        target_config: TritterConfig for target architecture
        method: Expansion method ("depth_upscaling" or "width_upscaling")
        output_path: Optional path to save expanded checkpoint

    Returns:
        Tuple of (expanded_state_dict, updated_metadata)

    Why: Enables progressive training by loading smaller models into
    larger architectures with appropriate initialization.
    """
    # Load source
    source_state, source_meta = load_progressive(source_checkpoint)

    if source_meta is None:
        raise ValueError("Source checkpoint missing progressive metadata")

    if method == "depth_upscaling":
        expanded_state = _depth_upscale(source_state, source_meta, target_config)
    elif method == "width_upscaling":
        expanded_state = _width_upscale(source_state, source_meta, target_config)
    else:
        raise ValueError(f"Unknown expansion method: {method}")

    # Update metadata
    new_meta = ProgressiveMetadata(
        model_size=target_config.model_size,
        hidden_size=target_config.hidden_size,
        num_layers=target_config.num_hidden_layers,
        num_heads=target_config.num_attention_heads,
        num_kv_heads=target_config.num_key_value_heads,
        vocab_size=target_config.vocab_size,
        max_position_embeddings=target_config.max_position_embeddings,
        tokens_seen=source_meta.tokens_seen,
        training_steps=source_meta.training_steps,
        best_loss=source_meta.best_loss,
        expansion_history=source_meta.expansion_history
        + [
            ExpansionStep(
                from_size=source_meta.model_size,
                to_size=target_config.model_size,
                method=method,
                tokens_after=source_meta.tokens_seen,
            )
        ],
    )

    # Save if requested
    if output_path:
        # Create a temporary module to save
        class StateHolder(nn.Module):
            def __init__(self, state: dict[str, torch.Tensor]):
                super().__init__()
                for name, param in state.items():
                    self.register_buffer(name.replace(".", "_"), param)

            def state_dict(self, *args, **kwargs):
                return {
                    name.replace("_", "."): buf
                    for name, buf in self.named_buffers()
                }

        holder = StateHolder(expanded_state)
        save_progressive(holder, output_path, new_meta)

    return expanded_state, new_meta


def _depth_upscale(
    state: dict[str, torch.Tensor],
    source_meta: ProgressiveMetadata,
    target_config,
) -> dict[str, torch.Tensor]:
    """Expand model by duplicating layers.

    Uses the LLaMA-Pro DUS approach: duplicate middle layers.
    """
    source_layers = source_meta.num_layers
    target_layers = target_config.num_hidden_layers

    if target_layers <= source_layers:
        raise ValueError(
            f"Target layers ({target_layers}) must be greater than "
            f"source layers ({source_layers})"
        )

    # Determine which layers to duplicate
    # Default: duplicate middle layers
    layers_to_add = target_layers - source_layers
    mid = source_layers // 2
    start = mid - layers_to_add // 2
    duplicate_indices = list(range(start, start + layers_to_add))

    print(f"Depth upscaling: {source_layers} → {target_layers} layers")
    print(f"Duplicating layers: {duplicate_indices}")

    # Build new state dict
    new_state = {}
    new_layer_idx = 0
    duplicated = 0

    for old_idx in range(source_layers):
        # Copy original layer
        for key, value in state.items():
            if f".layers.{old_idx}." in key or f"layers.{old_idx}." in key:
                new_key = key.replace(
                    f"layers.{old_idx}.", f"layers.{new_layer_idx}."
                )
                new_state[new_key] = value.clone()

        new_layer_idx += 1

        # Duplicate if this layer is in the list
        if old_idx in duplicate_indices and duplicated < layers_to_add:
            for key, value in state.items():
                if f".layers.{old_idx}." in key or f"layers.{old_idx}." in key:
                    new_key = key.replace(
                        f"layers.{old_idx}.", f"layers.{new_layer_idx}."
                    )
                    # Add small noise to break symmetry
                    noise = torch.randn_like(value) * 0.01
                    new_state[new_key] = value.clone() + noise

            new_layer_idx += 1
            duplicated += 1

    # Copy non-layer parameters
    for key, value in state.items():
        if ".layers." not in key and "layers." not in key:
            new_state[key] = value.clone()

    return new_state


def _width_upscale(
    state: dict[str, torch.Tensor],
    source_meta: ProgressiveMetadata,
    target_config,
) -> dict[str, torch.Tensor]:
    """Expand model by increasing hidden dimension.

    Uses Net2Net-style expansion: split neurons to preserve function.
    """
    source_hidden = source_meta.hidden_size
    target_hidden = target_config.hidden_size

    if target_hidden <= source_hidden:
        raise ValueError(
            f"Target hidden ({target_hidden}) must be greater than "
            f"source hidden ({source_hidden})"
        )

    print(f"Width upscaling: {source_hidden} → {target_hidden} hidden dim")

    new_state = {}

    for key, value in state.items():
        if value.dim() < 2:
            # Scalar or 1D tensor - just copy
            new_state[key] = value.clone()
            continue

        # Check if this tensor needs expansion
        shape = value.shape
        new_shape = list(shape)
        expanded = False

        # Expand output dimension
        if shape[0] == source_hidden:
            new_shape[0] = target_hidden
            expanded = True

        # Expand input dimension
        if len(shape) > 1 and shape[1] == source_hidden:
            new_shape[1] = target_hidden
            expanded = True

        if not expanded:
            new_state[key] = value.clone()
            continue

        # Create expanded tensor
        new_value = torch.zeros(new_shape, dtype=value.dtype, device=value.device)

        # Copy original values
        slices = tuple(slice(0, s) for s in shape)
        new_value[slices] = value

        # Fill new dimensions with split neurons (Net2Net)
        if shape[0] < new_shape[0]:
            extra = new_shape[0] - shape[0]
            indices = torch.randint(0, shape[0], (extra,))
            noise = torch.randn(extra, *shape[1:], dtype=value.dtype) * 0.01
            new_value[shape[0] :] = value[indices] / 2 + noise
            new_value[indices] /= 2  # Halve original to preserve magnitude

        if len(shape) > 1 and shape[1] < new_shape[1]:
            extra = new_shape[1] - shape[1]
            indices = torch.randint(0, shape[1], (extra,))
            noise = torch.randn(new_shape[0], extra, dtype=value.dtype) * 0.01
            new_value[:, shape[1] :] = new_value[:, indices] / 2 + noise
            new_value[:, indices] /= 2

        new_state[key] = new_value

    return new_state


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader,
    num_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """Compute Fisher Information diagonal for EWC.

    Args:
        model: Trained model
        dataloader: DataLoader for representative data
        num_samples: Number of samples to use

    Returns:
        Dictionary mapping parameter names to Fisher diagonal

    Why: Fisher diagonal approximates parameter importance for EWC.
    """
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    model.eval()
    samples = 0

    for batch in dataloader:
        if samples >= num_samples:
            break

        model.zero_grad()
        output = model(**batch)
        loss = output.loss if hasattr(output, "loss") else output
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2).detach()

        samples += 1

    # Normalize
    for name in fisher:
        fisher[name] /= samples

    return fisher


def ewc_loss(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
    old_params: dict[str, torch.Tensor],
    lambda_ewc: float = 1000.0,
) -> torch.Tensor:
    """Compute EWC regularization loss.

    Args:
        model: Current model
        fisher: Fisher diagonal from compute_fisher_diagonal
        old_params: Parameters from before expansion
        lambda_ewc: Regularization strength

    Returns:
        EWC loss term to add to training loss

    Why: Prevents forgetting of previously learned knowledge.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in fisher and name in old_params:
            loss += (fisher[name] * (param - old_params[name]).pow(2)).sum()

    return lambda_ewc * loss


__all__ = [
    "ExpansionStep",
    "ProgressiveMetadata",
    "compute_fisher_diagonal",
    "ewc_loss",
    "expand_model",
    "load_progressive",
    "save_progressive",
]
