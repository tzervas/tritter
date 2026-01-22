"""Example usage of Tritter multimodal model.

Demonstrates basic setup, configuration, and inference with the Tritter model,
showcasing multimodal tokenization, BitNet quantization, and device optimization.

Why: This example serves as both documentation and validation that the core architecture
works end-to-end. It demonstrates the minimal viable path to get started with Tritter
while highlighting key features (multimodal support, BitNet efficiency, RTX optimization).
The reduced parameters (512 hidden_size vs 2048 production) enable fast experimentation
without requiring 16GB VRAM, making the example accessible for development and testing.

Usage:
    python examples/basic_usage.py

Expected output:
    - Model initialization with parameter count
    - Successful tokenization of text and code
    - Forward pass producing logits of expected shape
    - No errors or warnings (validates architecture integrity)
"""

import torch

from tritter import TritterConfig, TritterModel
from tritter.tokenization.multimodal import ModalityType, MultiModalTokenizer
from tritter.utils.device_utils import get_optimal_device, optimize_for_rtx5080


def main() -> None:
    """Demonstrate basic usage of Tritter model.

    Walks through the complete workflow:
    1. Device optimization for RTX 5080
    2. Model configuration with BitNet quantization
    3. Model initialization and parameter counting
    4. Multimodal tokenization (text and code)
    5. Forward pass inference

    Why: This end-to-end workflow validates that all components integrate correctly.
    Each step builds on the previous, exposing potential integration issues early.
    The progression from config → model → tokenizer → inference mirrors the typical
    usage pattern, making this example template for actual applications.

    Returns:
        None (prints output to console)

    Raises:
        RuntimeError: If model initialization or forward pass fails
        ValueError: If configuration validation fails
    """
    print("=" * 60)
    print("Tritter Multimodal AI - Example Usage")
    print("=" * 60)

    # Configure for RTX 5080
    # Why: Must enable TF32 and cuDNN benchmarking before model creation to ensure
    # PyTorch uses optimal kernels. These optimizations provide ~20% speedup on Blackwell.
    optimize_for_rtx5080()
    device = get_optimal_device()
    print(f"\nUsing device: {device}")

    # Create configuration
    print("\n1. Creating 3B model configuration...")
    # Why: Reduced parameters for fast example execution while demonstrating architecture.
    # Production 3B uses hidden_size=2048, num_layers=24, max_position_embeddings=131072.
    # This smaller config (~10M params vs 3B) runs on any GPU/CPU for testing but maintains
    # the same architectural patterns (BitNet quantization, multi-head attention, etc.).
    config = TritterConfig(
        model_size="3B",
        hidden_size=512,  # Reduced from 2048 - Why: Faster initialization for demo
        num_heads=8,  # Must divide hidden_size evenly (512 / 8 = 64 per head)
        num_layers=4,  # Reduced from 24 - Why: Faster forward pass for demo
        vocab_size=10000,  # Reduced from 65536 - Why: Match reduced tokenizer vocab
        max_position_embeddings=1024,  # Reduced from 128K - Why: Less memory for demo
        use_bitnet=True,  # Why: Demonstrate ternary quantization even in small model
        use_flash_attention=True,  # Why: Show attention optimization (placeholder currently)
    )
    print(f"   Model: {config.model_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Context window: {config.max_position_embeddings}")
    print(f"   BitNet quantization: {config.use_bitnet}")

    # Initialize model
    print("\n2. Initializing model...")
    # Why: .to(device) must happen after initialization to move all parameters to GPU.
    # BitNet layers contain both quantized weights (ternary) and full-precision shadow
    # weights for training - both get moved to device.
    model = TritterModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create tokenizer
    print("\n3. Creating multimodal tokenizer...")
    # Why: Tokenizer vocab_size must match model config to prevent index out of bounds.
    # max_length should match max_position_embeddings to utilize full context window.
    tokenizer = MultiModalTokenizer(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings,
    )

    # Example 1: Text processing
    print("\n4. Processing text input...")
    text = "Hello from Tritter multimodal AI!"
    # Why: Tokenization is just the entry point - converts text to discrete IDs that get
    # embedded into continuous space. The model operates on embeddings, not tokens.
    text_tokens = tokenizer.encode(text, ModalityType.TEXT)
    print(f"   Input: {text}")
    print(f"   Tokens: {len(text_tokens)}")

    # Convert to tensor and run inference
    input_ids = torch.tensor([text_tokens]).to(device)
    print(f"   Input shape: {input_ids.shape}")

    # Why: Forward pass operates in continuous embedding space. The model predicts next
    # embeddings (Coconut/LCM style) rather than next tokens. Logits are only for
    # compatibility - production model will output embeddings that get rounded to tokens
    # via KNN lookup or vector quantization at generation time.
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
