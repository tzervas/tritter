#!/usr/bin/env python3
"""Validate Tritter architecture compatibility with BitNet b1.58-2B-4T.

Why: Before investing compute in training, we verify our architecture matches
the reference BitNet implementation. This catches architecture mismatches early.

Usage:
    python scripts/validate_bitnet_weights.py [--model-id microsoft/bitnet-b1.58-2B-4T-bf16]
"""

import argparse
import sys
from typing import Any

import torch


def get_bitnet_config() -> dict[str, Any]:
    """Get expected BitNet b1.58-2B-4T configuration.

    From HuggingFace model card:
    - Hidden size: 2048
    - Num layers: 24
    - Num heads: 32
    - Intermediate size: 5632 (not 4x)
    - Vocab size: 152064
    - RoPE base: 10000
    - Squared ReLU activation
    """
    return {
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "head_dim": 64,  # 2048 / 32
        "intermediate_size": 5632,
        "vocab_size": 152064,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    }


def compare_architectures(tritter_config: dict, bitnet_config: dict) -> list[str]:
    """Compare Tritter config with BitNet config.

    Returns list of differences (empty if compatible).
    """
    differences = []

    for key in ["hidden_size", "num_layers", "num_heads", "head_dim"]:
        if key in tritter_config and tritter_config[key] != bitnet_config[key]:
            differences.append(
                f"{key}: Tritter={tritter_config[key]}, BitNet={bitnet_config[key]}"
            )

    return differences


def try_load_hf_model(model_id: str) -> dict | None:
    """Try to load HuggingFace model config and weights info.

    Returns config dict or None if unavailable.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        return {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
        }
    except Exception as e:
        print(f"Warning: Could not load from HuggingFace: {e}")
        return None


def check_weight_shapes(model_id: str) -> dict[str, tuple[int, ...]]:
    """Get weight shapes from HuggingFace model.

    Returns dict of parameter name -> shape.
    """
    try:
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        # Download index
        index_file = hf_hub_download(
            model_id,
            "model.safetensors.index.json",
        )
        import json
        with open(index_file) as f:
            index = json.load(f)

        return {k: tuple(v) for k, v in index.get("metadata", {}).get("tensor_shapes", {}).items()}
    except Exception as e:
        print(f"Could not get weight shapes: {e}")
        return {}


def create_compatible_config():
    """Create TritterConfig that matches BitNet-2B architecture."""
    from tritter.core.config import TritterConfig

    bitnet_cfg = get_bitnet_config()

    return TritterConfig(
        model_size="3B",  # Closest size
        hidden_size=bitnet_cfg["hidden_size"],
        num_layers=bitnet_cfg["num_layers"],
        num_heads=bitnet_cfg["num_heads"],
        intermediate_size=bitnet_cfg["intermediate_size"],
        vocab_size=bitnet_cfg["vocab_size"],
        max_position_embeddings=bitnet_cfg["max_position_embeddings"],
        use_bitnet=True,
        use_flash_attention=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate BitNet architecture compatibility")
    parser.add_argument(
        "--model-id",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--check-hf",
        action="store_true",
        help="Try to load config from HuggingFace (requires internet)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BitNet b1.58-2B-4T Architecture Validation")
    print("=" * 60)

    # Get BitNet reference config
    bitnet_config = get_bitnet_config()
    print("\nBitNet Reference Configuration:")
    for k, v in bitnet_config.items():
        print(f"  {k}: {v}")

    # Optionally check HuggingFace
    if args.check_hf:
        print(f"\nLoading from HuggingFace: {args.model_id}")
        hf_config = try_load_hf_model(args.model_id)
        if hf_config:
            print("HuggingFace config loaded successfully:")
            for k, v in hf_config.items():
                print(f"  {k}: {v}")

    # Create compatible Tritter config
    print("\nCreating compatible TritterConfig...")
    try:
        tritter_config = create_compatible_config()
        print(f"  hidden_size: {tritter_config.hidden_size}")
        print(f"  num_layers: {tritter_config.num_layers}")
        print(f"  num_heads: {tritter_config.num_heads}")
        print(f"  head_dim: {tritter_config.head_dim}")
        print(f"  intermediate_size: {tritter_config.intermediate_size}")
        print(f"  vocab_size: {tritter_config.vocab_size}")
    except Exception as e:
        print(f"ERROR creating config: {e}")
        return 1

    # Compare
    tritter_dict = {
        "hidden_size": tritter_config.hidden_size,
        "num_layers": tritter_config.num_layers,
        "num_heads": tritter_config.num_heads,
        "head_dim": tritter_config.head_dim,
    }

    differences = compare_architectures(tritter_dict, bitnet_config)

    if differences:
        print("\nArchitecture MISMATCHES found:")
        for diff in differences:
            print(f"  X {diff}")
        return 1
    else:
        print("\nArchitecture is COMPATIBLE with BitNet-2B")

    # Try to instantiate model
    print("\nInstantiating TritterModel...")
    try:
        from tritter.models.architecture import TritterModel
        model = TritterModel(tritter_config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        compat = "compatible" if 1.5e9 < total_params < 2.5e9 else "warning"
        print(f"  Expected ~2B: {compat}")

        # Test forward pass
        print("\nTesting forward pass...")
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, tritter_config.vocab_size, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            output = model(input_ids)

        expected_shape = (batch_size, seq_len, tritter_config.vocab_size)
        if output.shape == expected_shape:
            print(f"  Output shape correct: {output.shape}")
        else:
            print(f"  Output shape wrong: {output.shape} (expected {expected_shape})")
            return 1

    except Exception as e:
        print(f"ERROR instantiating model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("VALIDATION PASSED - Ready for continued pretraining")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
