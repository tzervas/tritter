"""Keyed VSA bundling helpers for ternary optimization.

Why: VSA training benefits from bundling tensors with per-tensor keys to
preserve separability. A keyed bundler ensures each tensor can be unbound
using its corresponding key, avoiding collisions and tensor splitting errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

try:
    from vsa_optimizer import hyperdimensional_bind, hyperdimensional_bundle
except ImportError:

    def hyperdimensional_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fallback element-wise multiply binding when vsa_optimizer is not installed."""
        return a * b

    def hyperdimensional_bundle(tensors: list[torch.Tensor]) -> torch.Tensor:
        """Fallback majority-vote bundling when vsa_optimizer is not installed."""
        stacked = torch.stack(tensors)
        return torch.sign(stacked.sum(dim=0))


@dataclass
class VSAKeyedBundler:
    """Bundle and unbundle tensors with per-key bindings.

    Why: Proper VSA training requires binding each tensor with a unique key
    before bundling, then unbinding with the same key to recover individual
    tensors. This avoids destructive interference in ternary spaces.
    """

    key_vectors: Mapping[str, torch.Tensor]

    def bundle(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Bind tensors with keys and bundle them.

        Args:
            tensors: Mapping of tensor name to tensor value

        Returns:
            Bundled tensor

        Why: Keyed binding protects individual tensors from interference during
        bundling, enabling later unbinding with the same key.
        """
        bound = []
        for name, tensor in tensors.items():
            if name not in self.key_vectors:
                raise KeyError(f"Missing VSA key for tensor '{name}'")
            key = self.key_vectors[name]
            bound.append(hyperdimensional_bind(tensor, key))

        return hyperdimensional_bundle(bound)

    def unbind(self, bundle: torch.Tensor, name: str) -> torch.Tensor:
        """Unbind a tensor from a bundle using its key.

        Args:
            bundle: Bundled tensor
            name: Name of tensor to recover

        Returns:
            Recovered tensor estimate

        Why: Binding is assumed to be self-inverse in ternary VSA (element-wise
        multiply). Unbinding recovers the component for training or updates.
        """
        if name not in self.key_vectors:
            raise KeyError(f"Missing VSA key for tensor '{name}'")
        key = self.key_vectors[name]
        return hyperdimensional_bind(bundle, key)


def make_random_keys(
    names: list[str],
    dimension: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Create random ternary VSA keys for a list of tensor names.

    Args:
        names: Tensor names to create keys for
        dimension: VSA vector dimension
        device: Optional torch device
        dtype: Optional torch dtype

    Returns:
        Dictionary mapping names to ternary key vectors

    Why: Deterministic keyed bundling requires explicit key vectors. This
    helper creates ternary keys aligned with BitNet/VSA assumptions.
    """
    rng = torch.Generator(device=device)
    keys: dict[str, torch.Tensor] = {}
    for name in names:
        key = torch.randint(
            low=-1,
            high=2,
            size=(dimension,),
            generator=rng,
            device=device,
            dtype=dtype or torch.float32,
        )
        keys[name] = key
    return keys
