#!/usr/bin/env python3
"""Upload Tritter model to HuggingFace Hub.

Why: Provides a clean, reliable pipeline for distributing Tritter models
via HuggingFace Hub. Handles format conversion, metadata generation,
validation, and upload with proper progress reporting.

Usage:
    # Upload a trained checkpoint
    python scripts/upload_to_huggingface.py \\
        --checkpoint-dir checkpoints/7b-final \\
        --repo-id tritter-ai/tritter-7b \\
        --model-size 7B

    # Upload as private model
    python scripts/upload_to_huggingface.py \\
        --checkpoint-dir checkpoints/7b-final \\
        --repo-id myorg/tritter-7b-private \\
        --model-size 7B \\
        --private

    # Dry run (validate without uploading)
    python scripts/upload_to_huggingface.py \\
        --checkpoint-dir checkpoints/7b-final \\
        --repo-id tritter-ai/tritter-7b \\
        --model-size 7B \\
        --dry-run

Requirements:
    pip install huggingface_hub safetensors
    huggingface-cli login

Example workflow:
    1. Train model: python scripts/train_pretrain.py --model-size 7B
    2. Validate: python scripts/upload_to_huggingface.py --dry-run ...
    3. Upload: python scripts/upload_to_huggingface.py ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Defer heavy imports until after arg parsing
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload Tritter model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to checkpoint directory or .safetensors file",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., tritter-ai/tritter-7b)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        required=True,
        choices=["1B", "3B", "7B", "10B", "13B", "30B", "33B", "40B", "65B", "70B"],
        help="Model size variant",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and prepare files without uploading",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload Tritter model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (defaults to cached login)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch to upload to (default: main)",
    )

    return parser.parse_args()


def validate_checkpoint(checkpoint_dir: Path) -> tuple[bool, list[str]]:
    """Validate checkpoint directory structure.

    Why: Fail fast with clear error messages rather than during upload.

    Args:
        checkpoint_dir: Path to checkpoint

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    if not checkpoint_dir.exists():
        errors.append(f"Checkpoint path does not exist: {checkpoint_dir}")
        return False, errors

    # Check for supported formats
    if checkpoint_dir.is_file():
        suffix = checkpoint_dir.suffix.lower()
        if suffix not in (".safetensors", ".pt", ".pth", ".bin"):
            errors.append(f"Unsupported checkpoint format: {suffix}")
    elif checkpoint_dir.is_dir():
        # Look for weight files
        weight_files = (
            list(checkpoint_dir.glob("*.safetensors"))
            + list(checkpoint_dir.glob("*.pt"))
            + list(checkpoint_dir.glob("*.pth"))
            + list(checkpoint_dir.glob("*.bin"))
        )
        if not weight_files:
            errors.append(f"No weight files found in {checkpoint_dir}")
            errors.append("Expected: .safetensors, .pt, .pth, or .bin files")
    else:
        errors.append(f"Checkpoint path is neither file nor directory: {checkpoint_dir}")

    return len(errors) == 0, errors


def generate_config_json(model_size: str, repo_id: str) -> dict[str, Any]:
    """Generate HuggingFace-compatible config.json.

    Why: HuggingFace requires config.json for model loading. We generate
    this from TritterConfig to ensure consistency.

    Args:
        model_size: Model size variant
        repo_id: HuggingFace repository ID

    Returns:
        Configuration dictionary
    """
    from tritter.core.model_specs import get_model_spec

    spec = get_model_spec(model_size)

    config = {
        "_name_or_path": repo_id,
        "architectures": ["TritterModel"],
        "model_type": "tritter",
        "auto_map": {
            "AutoConfig": "tritter.core.config.TritterConfig",
            "AutoModelForCausalLM": "tritter.models.architecture.TritterModel",
        },
        "hidden_size": spec.hidden_size,
        "num_hidden_layers": spec.num_layers,
        "num_attention_heads": spec.num_heads,
        "num_key_value_heads": spec.effective_num_kv_heads,
        "intermediate_size": spec.intermediate_size,
        "vocab_size": spec.vocab_size,
        "max_position_embeddings": spec.max_position_embeddings,
        "rope_theta": spec.rope_theta,
        "use_bitnet": True,
        "bitnet_precision": 1.58,
        "use_flash_attention": True,
        "use_qk_norm": True,
        "activation_function": "squared_relu",
        "layer_norm_eps": 1e-5,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    }

    return config


def generate_model_card(model_size: str, repo_id: str) -> str:
    """Generate model card README.md with filled placeholders.

    Why: Model cards provide essential documentation for users.
    We fill in architecture-specific values automatically.

    Args:
        model_size: Model size variant
        repo_id: HuggingFace repository ID

    Returns:
        Formatted model card markdown
    """
    from tritter.core.model_specs import estimate_memory, get_model_spec

    spec = get_model_spec(model_size)
    mem = estimate_memory(spec)

    # Calculate GQA ratio
    gqa_ratio = spec.num_heads // spec.effective_num_kv_heads if spec.uses_gqa else 1

    # Memory estimates in GB
    packed_gb = mem.weights_packed_ternary / (1024**3)
    kv_4k = mem.kv_cache_4k_int4 / (1024**3)
    kv_32k = mem.kv_cache_32k_int4 / (1024**3)
    kv_128k = mem.kv_cache_128k_int4 / (1024**3)

    # Total inference memory (packed weights + KV + overhead)
    overhead = 2.0  # GB for activations and CUDA overhead
    total_4k = packed_gb + kv_4k + overhead
    total_32k = packed_gb + kv_32k + overhead
    total_128k = packed_gb + kv_128k + overhead

    # Read template
    template_path = PROJECT_ROOT / "huggingface" / "README.md"
    with open(template_path) as f:
        template = f.read()

    # Fill placeholders
    replacements = {
        "{SIZE}": model_size,
        "{size_lower}": model_size.lower(),
        "{HIDDEN_SIZE}": str(spec.hidden_size),
        "{NUM_LAYERS}": str(spec.num_layers),
        "{NUM_HEADS}": str(spec.num_heads),
        "{NUM_KV_HEADS}": str(spec.effective_num_kv_heads),
        "{INTERMEDIATE_SIZE}": str(spec.intermediate_size),
        "{VOCAB_SIZE}": str(spec.vocab_size),
        "{MAX_POSITION_EMBEDDINGS}": str(spec.max_position_embeddings),
        "{GQA_RATIO}": str(gqa_ratio),
        "{TRAINING_TOKENS}": "100",  # Placeholder - update after training
        "{TRAINING_HARDWARE}": "8x A100 80GB",  # Placeholder
        "{HUMANEVAL_SCORE}": "TBD",  # Placeholder - fill after evaluation
        "{MBPP_SCORE}": "TBD",
        "{MULTIPLE_RUST_SCORE}": "TBD",
        "{KV_4K}": f"{kv_4k:.2f}",
        "{KV_32K}": f"{kv_32k:.2f}",
        "{KV_128K}": f"{kv_128k:.2f}",
        "{TOTAL_4K}": f"{total_4k:.1f}",
        "{TOTAL_32K}": f"{total_32k:.1f}",
        "{TOTAL_128K}": f"{total_128k:.1f}",
    }

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    return template


def prepare_upload_directory(
    checkpoint_dir: Path,
    model_size: str,
    repo_id: str,
    output_dir: Path,
) -> list[Path]:
    """Prepare files for upload in a staging directory.

    Why: Organize all required files before upload to ensure completeness
    and enable dry-run validation.

    Args:
        checkpoint_dir: Source checkpoint
        model_size: Model size variant
        repo_id: HuggingFace repository ID
        output_dir: Staging directory

    Returns:
        List of files prepared for upload
    """
    import torch

    try:
        import safetensors.torch as st

        SAFETENSORS_AVAILABLE = True
    except ImportError:
        SAFETENSORS_AVAILABLE = False

    from tritter.checkpoints.formats import load_checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    files_to_upload: list[Path] = []

    print(f"Preparing files in {output_dir}")

    # Load and convert weights to safetensors
    print("  Loading checkpoint...")
    state_dict, metadata = load_checkpoint(checkpoint_dir)

    weights_path = output_dir / "model.safetensors"
    if SAFETENSORS_AVAILABLE:
        print(f"  Saving weights to {weights_path.name}...")
        # Add metadata to safetensors
        str_metadata = {
            "model_size": model_size,
            "format": "tritter",
            "version": "1.0",
        }
        st.save_file(state_dict, str(weights_path), metadata=str_metadata)
    else:
        # Fallback to PyTorch format
        weights_path = output_dir / "pytorch_model.bin"
        print(f"  Saving weights to {weights_path.name} (safetensors not available)...")
        torch.save(state_dict, weights_path)
    files_to_upload.append(weights_path)

    # Generate config.json
    print("  Generating config.json...")
    config = generate_config_json(model_size, repo_id)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    files_to_upload.append(config_path)

    # Generate generation_config.json
    print("  Generating generation_config.json...")
    gen_config_src = PROJECT_ROOT / "huggingface" / "generation_config.json"
    gen_config_dst = output_dir / "generation_config.json"
    shutil.copy(gen_config_src, gen_config_dst)
    files_to_upload.append(gen_config_dst)

    # Generate model card
    print("  Generating README.md (model card)...")
    model_card = generate_model_card(model_size, repo_id)
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    files_to_upload.append(readme_path)

    return files_to_upload


def upload_to_hub(
    staging_dir: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
    token: str | None,
    revision: str,
) -> str:
    """Upload prepared files to HuggingFace Hub.

    Why: Centralized upload logic with proper error handling and
    progress reporting.

    Args:
        staging_dir: Directory containing files to upload
        repo_id: HuggingFace repository ID
        private: Whether to make repository private
        commit_message: Commit message
        token: HuggingFace token
        revision: Branch to upload to

    Returns:
        URL of the uploaded model
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create or get repository
    print(f"\nCreating/accessing repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            token=token,
            exist_ok=True,
        )
    except Exception as e:
        print(f"  Note: {e}")

    # Upload files
    print(f"\nUploading files to {repo_id}...")
    for file_path in staging_dir.iterdir():
        if file_path.is_file():
            print(f"  Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"{commit_message}: {file_path.name}",
                revision=revision,
            )

    return f"https://huggingface.co/{repo_id}"


def print_summary(files: list[Path], repo_id: str, dry_run: bool) -> None:
    """Print upload summary.

    Args:
        files: List of files prepared for upload
        repo_id: HuggingFace repository ID
        dry_run: Whether this is a dry run
    """
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No files were uploaded")
    else:
        print("UPLOAD COMPLETE")
    print("=" * 60)

    print(f"\nRepository: {repo_id}")
    print(f"Files prepared: {len(files)}")
    print()

    total_size = 0
    for f in files:
        size = f.stat().st_size
        total_size += size
        size_str = format_size(size)
        print(f"  {f.name:<30} {size_str:>10}")

    print(f"  {'─' * 40}")
    print(f"  {'Total':<30} {format_size(total_size):>10}")

    if not dry_run:
        print(f"\nModel URL: https://huggingface.co/{repo_id}")


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()

    # Validate checkpoint
    print("Validating checkpoint...")
    is_valid, errors = validate_checkpoint(args.checkpoint_dir)
    if not is_valid:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Model size: {args.model_size}")
    print(f"  Repository: {args.repo_id}")

    # Check for required packages
    try:
        import huggingface_hub  # noqa: F401

        del huggingface_hub  # Only checking availability
    except ImportError:
        print("\nError: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return 1

    # Prepare files
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = Path(staging_dir)

        try:
            files = prepare_upload_directory(
                checkpoint_dir=args.checkpoint_dir,
                model_size=args.model_size,
                repo_id=args.repo_id,
                output_dir=staging_path,
            )
        except Exception as e:
            print(f"\nError preparing files: {e}")
            return 1

        # Upload or show dry-run summary
        if args.dry_run:
            print_summary(files, args.repo_id, dry_run=True)
        else:
            try:
                upload_to_hub(
                    staging_dir=staging_path,
                    repo_id=args.repo_id,
                    private=args.private,
                    commit_message=args.commit_message,
                    token=args.token,
                    revision=args.revision,
                )
                print_summary(files, args.repo_id, dry_run=False)
            except Exception as e:
                print(f"\nError uploading: {e}")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
