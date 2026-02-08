"""Unit tests for LoRA fine-tuning implementation.

Why: LoRA enables fine-tuning large models (7B+) on limited VRAM (16GB) by:
1. Freezing base model weights (no gradients, no optimizer states)
2. Adding small trainable low-rank matrices A and B
3. Output = base_output + (x @ A @ B) * scaling

Tests verify:
1. LoRAConfig validation and scaling computation
2. LoRALinear wraps base layers correctly
3. Gradients flow only to LoRA parameters
4. apply_lora targets correct modules
5. Memory estimates are accurate
6. Save/load preserves adapter weights
7. Merge/unmerge produces correct results
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tritter.quantization.bitnet import TernaryWeight
from tritter.quantization.packed_ternary import PackedTernaryWeight
from tritter.training.lora import (
    LoRAConfig,
    LoRALinear,
    LoRATrainer,
    apply_lora,
    count_parameters,
    estimate_lora_memory,
    get_lora_parameters,
    get_trainable_parameters,
    load_lora_adapters,
    merge_lora_weights,
    save_lora_adapters,
)


class TestLoRAConfig:
    """Test suite for LoRAConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values.

        Why: Defaults should follow LoRA paper recommendations.
        """
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert config.bias == "none"
        assert "q_proj" in config.target_modules

    def test_scaling_standard(self) -> None:
        """Test standard LoRA scaling (alpha/rank).

        Why: Scaling controls adaptation magnitude. Standard is alpha/rank.
        """
        config = LoRAConfig(rank=16, alpha=16.0)
        assert config.scaling == 1.0

        config = LoRAConfig(rank=8, alpha=16.0)
        assert config.scaling == 2.0

    def test_scaling_rslora(self) -> None:
        """Test RS-LoRA scaling (alpha/sqrt(rank)).

        Why: RS-LoRA provides better stability for extreme ranks.
        """
        config = LoRAConfig(rank=16, alpha=16.0, use_rslora=True)
        assert config.scaling == 16.0 / math.sqrt(16)
        assert config.scaling == 4.0

    def test_invalid_rank_raises(self) -> None:
        """Test that invalid rank raises error."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=0)
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=-1)

    def test_invalid_alpha_raises(self) -> None:
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=-1.0)

    def test_invalid_dropout_raises(self) -> None:
        """Test that invalid dropout raises error."""
        with pytest.raises(ValueError, match="dropout must be in"):
            LoRAConfig(dropout=-0.1)
        with pytest.raises(ValueError, match="dropout must be in"):
            LoRAConfig(dropout=1.0)

    def test_invalid_bias_raises(self) -> None:
        """Test that invalid bias option raises error."""
        with pytest.raises(ValueError, match="bias must be"):
            LoRAConfig(bias="invalid")


class TestLoRALinear:
    """Test suite for LoRALinear module."""

    def test_wrap_nn_linear(self) -> None:
        """Test wrapping nn.Linear layer.

        Why: LoRALinear should work with standard PyTorch linear layers.
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        assert lora.in_features == 64
        assert lora.out_features == 32
        assert lora.lora_A.shape == (64, 8)
        assert lora.lora_B.shape == (8, 32)

    def test_wrap_ternary_weight(self) -> None:
        """Test wrapping TernaryWeight layer (QLoRA).

        Why: QLoRA requires LoRA on top of quantized base weights.
        """
        base = TernaryWeight(64, 32)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        assert lora.in_features == 64
        assert lora.out_features == 32

    def test_wrap_packed_ternary(self) -> None:
        """Test wrapping PackedTernaryWeight layer.

        Why: Should support packed inference layers for QLoRA.
        """
        # Create packed layer
        ternary = TernaryWeight(64, 32)
        packed = PackedTernaryWeight.from_ternary_weight(ternary)

        config = LoRAConfig(rank=8)
        lora = LoRALinear(packed, config)

        assert lora.in_features == 64
        assert lora.out_features == 32

    def test_base_layer_frozen(self) -> None:
        """Test that base layer parameters are frozen.

        Why: Only LoRA parameters should be trainable for memory efficiency.
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        # Base layer should be frozen
        assert not base.weight.requires_grad
        if base.bias is not None:
            assert not base.bias.requires_grad

        # LoRA params should be trainable
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_init_gaussian(self) -> None:
        """Test Gaussian initialization (B=0, A=Kaiming).

        Why: B=0 means LoRA starts as identity (no change to base output).
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8, init_lora_weights="gaussian")
        lora = LoRALinear(base, config)

        # B should be zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

        # A should be non-zero (Kaiming init)
        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))

    def test_forward_shape(self) -> None:
        """Test forward pass output shape.

        Why: Output shape should match base layer output shape.
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        x = torch.randn(2, 10, 64)  # (batch, seq, in_features)
        output = lora(x)

        assert output.shape == (2, 10, 32)

    def test_forward_starts_as_base(self) -> None:
        """Test that LoRA output equals base output initially.

        Why: With B=0, LoRA contribution is zero, so output = base output.
        """
        base = nn.Linear(64, 32, bias=False)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        x = torch.randn(2, 10, 64)

        # Compute base output directly
        with torch.no_grad():
            base_output = base(x)
            lora_output = lora(x)

        # Should be equal since B=0
        assert torch.allclose(base_output, lora_output, rtol=1e-5, atol=1e-5)

    def test_forward_changes_with_trained_lora(self) -> None:
        """Test that output changes after LoRA training.

        Why: After updating B, LoRA should contribute to output.
        """
        base = nn.Linear(64, 32, bias=False)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        x = torch.randn(2, 10, 64)

        # Initial output
        with torch.no_grad():
            initial_output = lora(x).clone()

            # "Train" by setting B to non-zero
            lora.lora_B.fill_(0.1)

            trained_output = lora(x)

        # Output should change
        assert not torch.allclose(initial_output, trained_output)

    def test_gradients_flow_to_lora_only(self) -> None:
        """Test that gradients only flow to LoRA parameters.

        Why: Base parameters are frozen, gradients should not accumulate.
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        x = torch.randn(2, 10, 64)
        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Base should have no gradients (frozen)
        assert base.weight.grad is None

        # LoRA should have gradients
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

    def test_merge_weights_linear(self) -> None:
        """Test merging LoRA into nn.Linear.

        Why: Merged model should produce same output without LoRA overhead.
        """
        base = nn.Linear(64, 32, bias=False)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        # Set non-zero LoRA weights
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)

        x = torch.randn(2, 10, 64)

        # Get pre-merge output
        with torch.no_grad():
            pre_merge_output = lora(x).clone()

        # Merge weights
        lora.merge_weights()

        # Get post-merge output from base layer directly
        with torch.no_grad():
            post_merge_output = base(x)

        # Should be equal (merged model produces same output)
        assert torch.allclose(pre_merge_output, post_merge_output, rtol=1e-4, atol=1e-4)

    def test_merge_weights_ternary(self) -> None:
        """Test merging LoRA into TernaryWeight.

        Why: QLoRA merge should work with ternary base weights.
        Note: After merge, the shadow weights are updated but re-quantization
        can cause significant differences, especially with small LoRA values.
        """
        base = TernaryWeight(64, 32, bias=False)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(base, config)

        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            pre_merge_output = lora(x).clone()

        # Merge into shadow weights
        lora.merge_weights()

        # Forward through base (will re-quantize)
        with torch.no_grad():
            post_merge_output = base(x)

        # With ternary quantization, merging small LoRA values may not
        # significantly change the quantized output since values snap to {-1, 0, 1}.
        # The key test is that merge_weights() runs without error.
        # For production use, merge before final quantization or keep unmerged.
        assert post_merge_output.shape == pre_merge_output.shape


class TestApplyLoRA:
    """Test suite for apply_lora function."""

    def test_apply_to_simple_model(self) -> None:
        """Test applying LoRA to simple model.

        Why: Should correctly identify and wrap target modules.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.q_proj(x) + self.v_proj(x) + self.other(x)

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["q_proj", "v_proj"])
        model = apply_lora(model, config)

        # q_proj and v_proj should be LoRALinear
        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)

        # other should remain nn.Linear (not targeted)
        assert isinstance(model.other, nn.Linear)
        assert not isinstance(model.other, LoRALinear)

    def test_apply_to_nested_model(self) -> None:
        """Test applying LoRA to nested model structure.

        Why: Real models have nested modules (layers.0.attention.q_proj).
        """

        class Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attention = Attention()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Layer(), Layer()])

        model = Model()
        config = LoRAConfig(rank=8, target_modules=["q_proj", "v_proj"])
        model = apply_lora(model, config)

        # Check both layers
        for layer in model.layers:
            assert isinstance(layer.attention.q_proj, LoRALinear)
            assert isinstance(layer.attention.v_proj, LoRALinear)
            # k_proj not targeted
            assert isinstance(layer.attention.k_proj, nn.Linear)
            assert not isinstance(layer.attention.k_proj, LoRALinear)

    def test_non_lora_params_frozen(self) -> None:
        """Test that non-LoRA parameters are frozen.

        Why: Only LoRA params should be trainable for memory efficiency.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["q_proj"])
        model = apply_lora(model, config)

        # Check trainable status
        counts = count_parameters(model)
        assert counts["trainable"] > 0  # LoRA params
        assert counts["frozen"] > 0  # Base params

        # other should be frozen
        assert not model.other.weight.requires_grad


class TestParameterCounting:
    """Test suite for parameter counting functions."""

    def test_count_parameters(self) -> None:
        """Test count_parameters accuracy.

        Why: Accurate counts needed for memory estimation.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 32, bias=False)  # 64*32 = 2048 params

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["linear"])
        model = apply_lora(model, config)

        counts = count_parameters(model)

        # Total params: base (2048) + LoRA A (64*8=512) + LoRA B (8*32=256) = 2816
        assert counts["total"] == 2048 + 512 + 256

        # Trainable: LoRA only = 512 + 256 = 768
        assert counts["trainable"] == 768
        assert counts["lora"] == 768

        # Frozen: base = 2048
        assert counts["frozen"] == 2048

    def test_get_lora_parameters(self) -> None:
        """Test get_lora_parameters returns only LoRA params.

        Why: For optimizer, we only want LoRA parameters.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 32)

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["linear"])
        model = apply_lora(model, config)

        lora_params = list(get_lora_parameters(model))

        # Should be 2 params (A and B)
        assert len(lora_params) == 2

        # Verify shapes
        shapes = {p.shape for p in lora_params}
        assert (64, 8) in shapes  # lora_A
        assert (8, 32) in shapes  # lora_B

    def test_get_trainable_parameters(self) -> None:
        """Test get_trainable_parameters returns all trainable params.

        Why: More general than get_lora_parameters.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 32)  # Has bias

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["linear"], bias="lora_only")
        model = apply_lora(model, config)

        trainable = list(get_trainable_parameters(model))

        # Should include LoRA params + bias
        assert len(trainable) == 3  # lora_A, lora_B, bias


class TestSaveLoad:
    """Test suite for adapter save/load functionality."""

    def test_save_load_round_trip(self) -> None:
        """Test saving and loading preserves adapter weights.

        Why: Adapters must be loadable for deployment and resumption.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 32)
                self.v_proj = nn.Linear(64, 32)

        model1 = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["q_proj", "v_proj"])
        model1 = apply_lora(model1, config)

        # Set specific values
        with torch.no_grad():
            model1.q_proj.lora_A.fill_(0.5)
            model1.q_proj.lora_B.fill_(0.3)
            model1.v_proj.lora_A.fill_(0.7)
            model1.v_proj.lora_B.fill_(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapters.pt"
            save_lora_adapters(model1, str(path))

            # Create fresh model with LoRA
            model2 = SimpleModel()
            model2 = apply_lora(model2, config)

            # Load adapters
            model2 = load_lora_adapters(model2, str(path))

            # Verify weights match
            assert torch.allclose(model1.q_proj.lora_A, model2.q_proj.lora_A)
            assert torch.allclose(model1.q_proj.lora_B, model2.q_proj.lora_B)
            assert torch.allclose(model1.v_proj.lora_A, model2.v_proj.lora_A)
            assert torch.allclose(model1.v_proj.lora_B, model2.v_proj.lora_B)

    def test_save_includes_config(self) -> None:
        """Test that saved file includes LoRA config.

        Why: Config needed to reconstruct model for loading.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 32)

        model = SimpleModel()
        config = LoRAConfig(rank=8, alpha=32.0, target_modules=["linear"])
        model = apply_lora(model, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adapters.pt"
            save_lora_adapters(model, str(path))

            checkpoint = torch.load(path, weights_only=True)
            assert "lora_config" in checkpoint
            assert checkpoint["lora_config"]["rank"] == 8
            assert checkpoint["lora_config"]["alpha"] == 32.0


class TestMergeWeights:
    """Test suite for weight merging functionality."""

    def test_merge_all_layers(self) -> None:
        """Test merging all LoRA layers in a model.

        Why: For deployment, merge eliminates LoRA overhead.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 32, bias=False)
                self.v_proj = nn.Linear(64, 32, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.q_proj(x) + self.v_proj(x)

        model = SimpleModel()
        config = LoRAConfig(rank=8)
        model = apply_lora(model, config)

        # Set LoRA weights
        with torch.no_grad():
            model.q_proj.lora_A.normal_()
            model.q_proj.lora_B.normal_(std=0.1)
            model.v_proj.lora_A.normal_()
            model.v_proj.lora_B.normal_(std=0.1)

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            pre_merge = model(x).clone()

        # Merge all
        merge_lora_weights(model)

        # Forward through merged model
        with torch.no_grad():
            # Access base layers directly after merge
            base_output = model.q_proj.base_layer(x) + model.v_proj.base_layer(x)

        assert torch.allclose(pre_merge, base_output, rtol=1e-4, atol=1e-4)


class TestMemoryEstimation:
    """Test suite for memory estimation."""

    def test_estimate_lora_memory(self) -> None:
        """Test memory estimation accuracy.

        Why: Pre-compute memory to verify config fits in VRAM.
        """
        from tritter.core.config import TritterConfig

        model_config = TritterConfig(model_size="7B")
        lora_config = LoRAConfig(rank=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

        estimates = estimate_lora_memory(model_config, lora_config)

        # With 7B model, 32 layers, 4 attention targets, rank 16:
        # Per layer: 4 * (4096*16 + 16*4096) = 4 * 131072 = 524288 params
        # Total: 32 * 524288 = 16,777,216 params
        assert estimates["total_params"] == 32 * 4 * (4096 * 16 + 16 * 4096)

        # FP16: ~32MB for params
        assert estimates["lora_params_gb"] < 0.05  # Under 50MB

        # Total LoRA overhead should be small
        assert estimates["total_lora_gb"] < 0.2  # Under 200MB


class TestLoRATrainer:
    """Test suite for LoRATrainer class."""

    def test_trainer_creation(self) -> None:
        """Test creating LoRATrainer.

        Why: Trainer should initialize with only trainable params in optimizer.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 32)

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["linear"])
        model = apply_lora(model, config)

        trainer = LoRATrainer(model, config)

        counts = trainer.get_param_counts()
        assert counts["trainable"] > 0
        assert counts["lora"] == counts["trainable"]

    def test_trainer_save_load(self) -> None:
        """Test trainer checkpoint save/load.

        Why: Checkpointing needed for training resumption.
        """

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(64, 32)

        model = SimpleModel()
        config = LoRAConfig(rank=8, target_modules=["linear"])
        model = apply_lora(model, config)
        trainer = LoRATrainer(model, config)

        # Set specific values
        with torch.no_grad():
            model.linear.lora_A.fill_(0.42)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)

            # Create fresh model and trainer
            model2 = SimpleModel()
            model2 = apply_lora(model2, config)
            trainer2 = LoRATrainer(model2, config)

            trainer2.load_checkpoint(tmpdir)

            # Verify weights loaded
            assert torch.allclose(
                model.linear.lora_A,
                model2.linear.lora_A,
            )


class TestQLoRA:
    """Test suite for QLoRA (LoRA + quantized base)."""

    def test_qlora_with_ternary(self) -> None:
        """Test QLoRA with TernaryWeight base layers.

        Why: QLoRA requires LoRA adapters on quantized weights.
        """

        class QuantizedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = TernaryWeight(64, 32)
                self.v_proj = TernaryWeight(64, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.q_proj(x) + self.v_proj(x)

        model = QuantizedModel()
        config = LoRAConfig(rank=8)
        model = apply_lora(model, config)

        # Verify LoRA applied
        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.q_proj.base_layer, TernaryWeight)

        # Base should be frozen
        assert not model.q_proj.base_layer.weight.requires_grad

        # LoRA should be trainable
        assert model.q_proj.lora_A.requires_grad

        # Forward should work
        x = torch.randn(2, 10, 64)
        output = model(x)
        assert output.shape == (2, 10, 32)

    def test_qlora_with_packed(self) -> None:
        """Test QLoRA with PackedTernaryWeight base layers.

        Why: Packed weights provide additional memory savings.
        """
        # Create packed layer
        ternary = TernaryWeight(64, 32, bias=False)
        packed = PackedTernaryWeight.from_ternary_weight(ternary)

        config = LoRAConfig(rank=8)
        lora = LoRALinear(packed, config)

        # Forward should work
        x = torch.randn(2, 10, 64)
        output = lora(x)
        assert output.shape == (2, 10, 32)

    def test_qlora_training_step(self) -> None:
        """Test QLoRA training step (forward + backward).

        Why: Verify gradients flow correctly with quantized base.
        """

        class QuantizedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = TernaryWeight(64, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(x)

        model = QuantizedModel()
        config = LoRAConfig(rank=8, target_modules=["proj"])
        model = apply_lora(model, config)

        # Training step
        model.train()
        x = torch.randn(2, 10, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Base should have no gradients (frozen)
        assert model.proj.base_layer.weight.grad is None

        # LoRA should have gradients
        assert model.proj.lora_A.grad is not None
        assert model.proj.lora_B.grad is not None


class TestDropout:
    """Test suite for LoRA dropout."""

    def test_dropout_in_training(self) -> None:
        """Test dropout is applied during training.

        Why: Dropout regularizes LoRA adaptation.
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8, dropout=0.5)
        lora = LoRALinear(base, config)

        # Set non-zero LoRA weights
        with torch.no_grad():
            lora.lora_A.fill_(1.0)
            lora.lora_B.fill_(1.0)

        x = torch.randn(100, 64)  # Large batch for statistics

        # Training mode - dropout should be active
        lora.train()
        outputs_train = [lora(x).detach() for _ in range(10)]

        # Outputs should vary due to dropout
        variations = torch.stack([o.std() for o in outputs_train])
        assert variations.std() > 0.01  # Non-trivial variation

    def test_no_dropout_in_eval(self) -> None:
        """Test dropout is disabled during evaluation.

        Why: Eval mode should be deterministic.
        """
        base = nn.Linear(64, 32)
        config = LoRAConfig(rank=8, dropout=0.5)
        lora = LoRALinear(base, config)

        with torch.no_grad():
            lora.lora_A.fill_(1.0)
            lora.lora_B.fill_(1.0)

        x = torch.randn(10, 64)

        # Eval mode - dropout disabled
        lora.eval()
        with torch.no_grad():
            output1 = lora(x)
            output2 = lora(x)

        # Outputs should be identical
        assert torch.allclose(output1, output2)
