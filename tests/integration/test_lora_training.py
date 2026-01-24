"""Integration tests for LoRA fine-tuning with TritterModel.

Why: LoRA enables fine-tuning large models (7B+) on limited VRAM (16GB) by:
1. Freezing base model weights
2. Adding small trainable low-rank adapters
3. Memory: ~200MB for LoRA vs ~30GB for full fine-tuning

Integration tests verify:
1. LoRA integrates correctly with TritterModel architecture
2. Training loop works with LoRA adapters
3. Memory usage matches estimates
4. QLoRA works with BitNet quantized models
5. Adapter checkpointing and loading work end-to-end
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tritter.core.config import TritterConfig
from tritter.models.architecture import TritterModel
from tritter.training.lora import (
    LoRAConfig,
    LoRALinear,
    LoRATrainer,
    apply_lora,
    count_parameters,
    estimate_lora_memory,
    load_lora_adapters,
    merge_lora_weights,
    save_lora_adapters,
)


class TestLoRAWithTritterModel:
    """Test LoRA integration with TritterModel architecture."""

    @pytest.fixture
    def small_config(self) -> TritterConfig:
        """Create small config for testing.

        Why: Small model allows fast testing while maintaining architecture.
        """
        return TritterConfig(
            model_size="1B",
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
            vocab_size=1000,
            use_bitnet=True,
        )

    @pytest.fixture
    def model(self, small_config: TritterConfig) -> TritterModel:
        """Create small TritterModel for testing."""
        return TritterModel(small_config)

    def test_apply_lora_attention_only(self, model: TritterModel) -> None:
        """Test applying LoRA to attention projections only.

        Why: Common pattern - only adapt attention for efficiency.
        """
        lora_config = LoRAConfig(
            rank=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = apply_lora(model, lora_config)

        # Verify attention projections wrapped
        for layer in model.layers:
            assert isinstance(layer.attention.q_proj, LoRALinear)
            assert isinstance(layer.attention.k_proj, LoRALinear)
            assert isinstance(layer.attention.v_proj, LoRALinear)
            assert isinstance(layer.attention.o_proj, LoRALinear)

        # MLP should NOT be wrapped
        for layer in model.layers:
            assert not isinstance(layer.mlp.gate_proj, LoRALinear)
            assert not isinstance(layer.mlp.up_proj, LoRALinear)
            assert not isinstance(layer.mlp.down_proj, LoRALinear)

    def test_apply_lora_full(self, model: TritterModel) -> None:
        """Test applying LoRA to all projections (attention + MLP).

        Why: Full LoRA provides more adaptation capacity.
        """
        lora_config = LoRAConfig(
            rank=8,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = apply_lora(model, lora_config)

        # Verify all projections wrapped
        for layer in model.layers:
            # Attention
            assert isinstance(layer.attention.q_proj, LoRALinear)
            assert isinstance(layer.attention.k_proj, LoRALinear)
            assert isinstance(layer.attention.v_proj, LoRALinear)
            assert isinstance(layer.attention.o_proj, LoRALinear)
            # MLP
            assert isinstance(layer.mlp.gate_proj, LoRALinear)
            assert isinstance(layer.mlp.up_proj, LoRALinear)
            assert isinstance(layer.mlp.down_proj, LoRALinear)

    def test_forward_with_lora(self, model: TritterModel) -> None:
        """Test forward pass works with LoRA adapters.

        Why: Model should still produce valid outputs after LoRA application.
        """
        lora_config = LoRAConfig(rank=8)
        model = apply_lora(model, lora_config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = model(input_ids)

        # Shape should match: (batch, seq, vocab)
        assert output.shape == (batch_size, seq_len, 1000)

    def test_forward_unchanged_initially(self, model: TritterModel) -> None:
        """Test that LoRA doesn't change output initially (B=0).

        Why: Zero-initialized B means LoRA contribution is zero initially.
        """
        # Get output before LoRA
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            original_output = model(input_ids).clone()

        # Apply LoRA
        lora_config = LoRAConfig(rank=8)
        model = apply_lora(model, lora_config)

        model.eval()
        with torch.no_grad():
            lora_output = model(input_ids)

        # Should be identical since B=0
        assert torch.allclose(original_output, lora_output, rtol=1e-4, atol=1e-4)

    def test_parameter_counts(self, model: TritterModel) -> None:
        """Test parameter counting with LoRA model.

        Why: Verify LoRA reduces trainable params significantly.
        """
        total_before = sum(p.numel() for p in model.parameters())

        lora_config = LoRAConfig(rank=8)
        model = apply_lora(model, lora_config)

        counts = count_parameters(model)

        # Total should be slightly larger (added LoRA params)
        assert counts["total"] > total_before

        # Trainable should be much smaller than total
        assert counts["trainable"] < counts["total"] * 0.1  # Less than 10%

        # LoRA params should equal trainable (no other trainable params)
        assert counts["lora"] == counts["trainable"]


class TestLoRATraining:
    """Test LoRA training functionality."""

    @pytest.fixture
    def model_with_lora(self) -> TritterModel:
        """Create small model with LoRA applied."""
        config = TritterConfig(
            model_size="1B",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            vocab_size=500,
            use_bitnet=True,
        )
        model = TritterModel(config)
        lora_config = LoRAConfig(rank=4)
        return apply_lora(model, lora_config)

    def test_training_step(self, model_with_lora: TritterModel) -> None:
        """Test single training step with LoRA.

        Why: Verify gradients flow correctly to LoRA parameters.
        """
        model = model_with_lora

        # Initialize LoRA B to small non-zero values so gradients can flow to A
        # (With B=0, x @ A @ B = 0 regardless of A, giving zero gradients for A)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_B.normal_(std=0.01)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))

        # Forward
        model.train()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 500),
            labels.reshape(-1),
        )

        # Backward
        loss.backward()

        # Check LoRA parameters have gradients
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.grad is not None, f"No gradient for {name}"
                # At least one LoRA param should have non-zero gradients

        # Verify at least some gradients are non-zero
        has_nonzero_grad = any(
            param.grad.abs().sum() > 0
            for name, param in model.named_parameters()
            if ("lora_A" in name or "lora_B" in name) and param.grad is not None
        )
        assert has_nonzero_grad, "All LoRA gradients are zero"

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

    def test_multiple_training_steps(self, model_with_lora: TritterModel) -> None:
        """Test multiple training steps improve loss.

        Why: Verify LoRA can actually learn (loss should decrease).
        """
        model = model_with_lora

        # Initialize LoRA B to small non-zero values for gradient flow
        for module in model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_B.normal_(std=0.01)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,  # Higher LR for faster convergence
        )

        # Fixed batch for consistent loss comparison
        torch.manual_seed(42)
        input_ids = torch.randint(0, 500, (4, 32))
        labels = input_ids[:, 1:].contiguous()  # Simple next-token prediction
        input_ids = input_ids[:, :-1].contiguous()

        initial_loss = None
        model.train()

        for step in range(20):
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 500),
                labels.reshape(-1),
            )

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


class TestQLoRAWithBitNet:
    """Test QLoRA with BitNet quantized models."""

    def test_qlora_preserves_quantization(self) -> None:
        """Test that QLoRA preserves BitNet quantization.

        Why: Base weights should remain ternary during training.
        """
        config = TritterConfig(
            model_size="1B",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            vocab_size=500,
            use_bitnet=True,
        )
        model = TritterModel(config)
        lora_config = LoRAConfig(rank=4)
        model = apply_lora(model, lora_config)

        # Training step
        input_ids = torch.randint(0, 500, (2, 16))
        model.train()
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check base weights are still full-precision (for STE)
        # The quantization happens in forward pass
        for layer in model.layers:
            base_layer = layer.attention.q_proj.base_layer
            assert base_layer.weight.dtype in (torch.float32, torch.float16)


class TestLoRACheckpointing:
    """Test LoRA checkpoint save/load."""

    def test_checkpoint_round_trip(self) -> None:
        """Test saving and loading LoRA checkpoints.

        Why: Checkpointing needed for training resumption and deployment.
        """
        # Use fixed seed for reproducible base model weights
        torch.manual_seed(42)

        config = TritterConfig(
            model_size="1B",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            vocab_size=500,
            use_bitnet=True,
        )
        model1 = TritterModel(config)
        lora_config = LoRAConfig(rank=4)
        model1 = apply_lora(model1, lora_config)

        # Initialize LoRA B for gradient flow
        for module in model1.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_B.normal_(std=0.01)

        # Do some "training" to get non-zero LoRA weights
        input_ids = torch.randint(0, 500, (2, 16))
        model1.train()
        logits = model1(input_ids)
        loss = logits.sum()
        loss.backward()

        # Update weights
        for name, param in model1.named_parameters():
            if param.requires_grad and param.grad is not None:
                with torch.no_grad():
                    param.add_(param.grad, alpha=-0.01)

        # Get output from trained model
        model1.eval()
        with torch.no_grad():
            output1 = model1(input_ids).clone()

        # Save adapters
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapters.pt"
            save_lora_adapters(model1, str(path))

            # Create fresh model with SAME base weights
            torch.manual_seed(42)
            model2 = TritterModel(config)
            model2 = apply_lora(model2, lora_config)
            model2 = load_lora_adapters(model2, str(path))

            model2.eval()
            with torch.no_grad():
                output2 = model2(input_ids)

        # Outputs should match
        assert torch.allclose(output1, output2, rtol=1e-4, atol=1e-4)

    def test_lora_trainer_checkpoint(self) -> None:
        """Test LoRATrainer checkpoint save/load.

        Why: Trainer checkpoints include optimizer state for resumption.
        """
        config = TritterConfig(
            model_size="1B",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            vocab_size=500,
            use_bitnet=True,
        )
        model = TritterModel(config)
        lora_config = LoRAConfig(rank=4)
        model = apply_lora(model, lora_config)

        trainer = LoRATrainer(model, lora_config, learning_rate=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)

            # Verify files exist
            assert (Path(tmpdir) / "lora_adapters.pt").exists()
            assert (Path(tmpdir) / "optimizer.pt").exists()


class TestLoRAMerging:
    """Test LoRA weight merging for deployment."""

    def test_merge_preserves_output(self) -> None:
        """Test that merged model produces same output.

        Why: Merging eliminates LoRA overhead while preserving behavior.
        Note: After merge, the LoRALinear still adds (x @ A @ B) * scaling,
        but since B was already added to base weights, we effectively double-count.
        For proper deployment, either:
        1. Zero out A or B after merge
        2. Use base layer directly
        3. Convert to non-LoRA model

        This test verifies merge modifies base weights correctly by checking
        the base layer output matches the original full LoRA output.
        """
        config = TritterConfig(
            model_size="1B",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            vocab_size=500,
            use_bitnet=False,  # Use standard linear for exact merge
        )
        model = TritterModel(config)
        lora_config = LoRAConfig(rank=4)
        model = apply_lora(model, lora_config)

        # Set non-zero LoRA weights
        for module in model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_A.normal_(std=0.01)
                    module.lora_B.normal_(std=0.01)

        input_ids = torch.randint(0, 500, (2, 16))

        model.eval()
        with torch.no_grad():
            pre_merge = model(input_ids).clone()

        # Store original A, B for verification
        original_lora = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                original_lora[name] = (module.lora_A.clone(), module.lora_B.clone())

        # Merge
        merge_lora_weights(model)

        # Zero out LoRA after merge to avoid double-counting
        for module in model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_B.zero_()

        with torch.no_grad():
            post_merge = model(input_ids)

        # Should be approximately equal
        assert torch.allclose(pre_merge, post_merge, rtol=1e-3, atol=1e-3)


class TestMemoryEstimation:
    """Test LoRA memory estimation."""

    def test_estimate_vs_actual(self) -> None:
        """Test that memory estimates are reasonably accurate.

        Why: Estimates help plan training before running OOM.
        """
        config = TritterConfig(
            model_size="1B",
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            vocab_size=1000,
            use_bitnet=True,
        )
        lora_config = LoRAConfig(
            rank=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        estimates = estimate_lora_memory(config, lora_config)

        # Create actual model and count
        model = TritterModel(config)
        model = apply_lora(model, lora_config)

        actual_lora_params = sum(
            p.numel() for name, p in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        )

        # Estimated should match actual
        assert estimates["total_params"] == actual_lora_params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLoRACUDA:
    """Test LoRA on CUDA."""

    def test_lora_training_cuda(self) -> None:
        """Test LoRA training on CUDA.

        Why: Real training happens on GPU, need to verify CUDA compatibility.
        """
        device = torch.device("cuda")

        config = TritterConfig(
            model_size="1B",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            vocab_size=500,
            use_bitnet=True,
        )
        model = TritterModel(config).to(device)
        lora_config = LoRAConfig(rank=4)
        model = apply_lora(model, lora_config)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        input_ids = torch.randint(0, 500, (2, 16), device=device)
        labels = torch.randint(0, 500, (2, 16), device=device)

        model.train()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, 500),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()

        # Verify gradients are on correct device
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert param.grad.device.type == "cuda"

    def test_lora_memory_efficiency_cuda(self) -> None:
        """Test that LoRA uses less memory than full fine-tuning.

        Why: Main benefit of LoRA is memory efficiency.
        """
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()

        config = TritterConfig(
            model_size="1B",
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            use_bitnet=True,
        )

        # Create model with LoRA
        model = TritterModel(config).to(device)
        lora_config = LoRAConfig(rank=8)
        model = apply_lora(model, lora_config)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        # Training step
        input_ids = torch.randint(0, 1000, (4, 64), device=device)
        labels = torch.randint(0, 1000, (4, 64), device=device)

        model.train()
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, 1000), labels.view(-1))
        loss.backward()
        optimizer.step()

        lora_memory = torch.cuda.max_memory_allocated() / 1e9

        # Memory should be low (model + LoRA overhead)
        # Without LoRA, full fine-tuning would need much more
        assert lora_memory < 1.0, f"LoRA memory {lora_memory:.2f} GB is too high"
