"""Example usage of Tritter multimodal model."""

import torch

from tritter import TritterConfig, TritterModel
from tritter.tokenization.multimodal import ModalityType, MultiModalTokenizer
from tritter.utils.device_utils import get_optimal_device, optimize_for_rtx5080


def main() -> None:
    """Demonstrate basic usage of Tritter model."""
    print("=" * 60)
    print("Tritter Multimodal AI - Example Usage")
    print("=" * 60)

    # Configure for RTX 5080
    optimize_for_rtx5080()
    device = get_optimal_device()
    print(f"\nUsing device: {device}")

    # Create configuration
    print("\n1. Creating 3B model configuration...")
    config = TritterConfig(
        model_size="3B",
        hidden_size=512,  # Reduced for example
        num_heads=8,
        num_layers=4,
        vocab_size=10000,
        max_position_embeddings=1024,  # Reduced for example
        use_bitnet=True,
        use_flash_attention=True,
    )
    print(f"   Model: {config.model_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Context window: {config.max_position_embeddings}")
    print(f"   BitNet quantization: {config.use_bitnet}")

    # Initialize model
    print("\n2. Initializing model...")
    model = TritterModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create tokenizer
    print("\n3. Creating multimodal tokenizer...")
    tokenizer = MultiModalTokenizer(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings,
    )

    # Example 1: Text processing
    print("\n4. Processing text input...")
    text = "Hello from Tritter multimodal AI!"
    text_tokens = tokenizer.encode(text, ModalityType.TEXT)
    print(f"   Input: {text}")
    print(f"   Tokens: {len(text_tokens)}")

    # Convert to tensor and run inference
    input_ids = torch.tensor([text_tokens]).to(device)
    print(f"   Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)
    print(f"   Output shape: {logits.shape}")
    print(f"   Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Example 2: Code processing
    print("\n5. Processing code input...")
    code = "def hello(): return 'world'"
    code_tokens = tokenizer.encode(code, ModalityType.CODE)
    print(f"   Input: {code}")
    print(f"   Tokens: {len(code_tokens)}")

    # Example 3: Multiple modalities
    print("\n6. Supported modalities:")
    for modality in config.modalities:
        print(f"   - {modality}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
