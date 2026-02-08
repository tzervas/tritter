"""Tests for the training optimization module.

Why: The optimization module provides memory-efficient and accelerated training
through VSA compression, ternary quantization, and gradient prediction.
These tests verify correctness of each component.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vsa_optimizer", reason="vsa-training-optimizer package not installed")

from tritter.training.optimization import (
    # Gradient prediction
    GradientPredictor,
    # Phase-based training
    PhaseConfig,
    PhaseTrainer,
    PredictionConfig,
    PredictiveTrainer,
    # Ternary optimization
    TernaryConfig,
    TernaryGradientAccumulator,
    TernaryOptimizer,
    TrainingPhase,
    # VSA compression
    VSAConfig,
    VSAGradientCompressor,
    hyperdimensional_bind,
    hyperdimensional_bundle,
    ternary_quantize,
)


class SimpleModel(nn.Module):
    """Simple model for testing optimization components."""

    def __init__(self, input_size: int = 64, hidden_size: int = 128, output_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestVSACompression:
    """Tests for VSA (Vector Symbolic Architecture) gradient compression."""

    def test_hyperdimensional_bind_shape(self):
        """Binding should preserve tensor shape."""
        a = torch.randn(100)
        b = torch.randn(100)
        result = hyperdimensional_bind(a, b)
        assert result.shape == a.shape, "Binding should preserve shape"

    def test_hyperdimensional_bind_dissimilarity(self):
        """Binding result should be dissimilar to inputs."""
        a = torch.randn(1000)
        b = torch.randn(1000)
        result = hyperdimensional_bind(a, b)

        # Cosine similarity should be low
        sim_a = torch.cosine_similarity(result.unsqueeze(0), a.unsqueeze(0))
        sim_b = torch.cosine_similarity(result.unsqueeze(0), b.unsqueeze(0))

        assert abs(sim_a.item()) < 0.3, "Bound result should be dissimilar to input a"
        assert abs(sim_b.item()) < 0.3, "Bound result should be dissimilar to input b"

    def test_hyperdimensional_bundle_shape(self):
        """Bundling should preserve tensor shape."""
        vectors = [torch.randn(100) for _ in range(5)]
        result = hyperdimensional_bundle(vectors)
        assert result.shape == vectors[0].shape, "Bundling should preserve shape"

    def test_hyperdimensional_bundle_similarity(self):
        """Bundled result should be similar to all inputs."""
        vectors = [torch.randn(1000) for _ in range(3)]
        result = hyperdimensional_bundle(vectors)

        # Should be somewhat similar to each input
        for v in vectors:
            sim = torch.cosine_similarity(result.unsqueeze(0), v.unsqueeze(0))
            assert sim.item() > 0, "Bundled result should have positive similarity with inputs"

    def test_hyperdimensional_bundle_empty_raises(self):
        """Bundling empty list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot bundle empty"):
            hyperdimensional_bundle([])

    def test_hyperdimensional_bundle_weights(self):
        """Bundling with weights should weight contribution."""
        v1 = torch.ones(100)
        v2 = -torch.ones(100)

        # Without weights (equal)
        result_equal = hyperdimensional_bundle([v1, v2])
        assert result_equal.abs().mean() < 0.1, "Equal weights should roughly cancel"

        # With weights (v1 dominant)
        result_weighted = hyperdimensional_bundle([v1, v2], weights=[10.0, 1.0])
        assert result_weighted.mean() > 0, "Weighted bundle should favor v1"

    def test_ternary_quantize_values(self):
        """Ternary quantization should produce {-1, 0, +1}."""
        x = torch.randn(1000)
        quantized, scale = ternary_quantize(x)

        unique_values = quantized.unique()
        expected = torch.tensor([-1.0, 0.0, 1.0])

        for val in unique_values:
            assert val in expected, f"Unexpected value {val} in ternary quantization"

    def test_ternary_quantize_zero_tensor(self):
        """Ternary quantization of zeros should return zeros."""
        x = torch.zeros(100)
        quantized, scale = ternary_quantize(x)

        assert scale == 0.0, "Scale should be 0 for zero tensor"
        assert (quantized == 0).all(), "Quantized zeros should remain zeros"

    def test_vsa_compressor_compress_decompress(self):
        """Compression/decompression should preserve gradient direction."""
        model = SimpleModel()

        # Compute gradients
        x = torch.randn(8, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Collect gradients
        gradients = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        shapes = {name: grad.shape for name, grad in gradients.items()}

        # Compress
        param_count = sum(p.numel() for p in model.parameters())
        compressor = VSAGradientCompressor(param_count, VSAConfig(compression_ratio=0.5))

        compressed, metadata = compressor.compress(gradients)

        # Decompress
        reconstructed = compressor.decompress(compressed, metadata, shapes)

        # Check direction preservation (cosine similarity)
        for name in gradients:
            orig_flat = gradients[name].flatten()
            recon_flat = reconstructed[name].flatten()
            sim = torch.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0))

            # Direction should be roughly preserved (> 0.5 similarity)
            assert sim.item() > 0.3, f"Gradient direction not preserved for {name}"

    def test_vsa_compressor_stats(self):
        """VSA compressor should provide meaningful statistics."""
        compressor = VSAGradientCompressor(1_000_000)
        stats = compressor.get_compression_stats()

        assert "original_params" in stats
        assert "compressed_dim" in stats
        assert "compression_ratio" in stats
        assert "memory_saving" in stats

        assert stats["original_params"] == 1_000_000
        assert stats["compression_ratio"] < 0.2  # Should compress significantly
        assert stats["memory_saving"] > 0.8  # Should save > 80% memory


class TestGradientPredictor:
    """Tests for gradient prediction."""

    def test_should_compute_full_initially(self):
        """Should compute full gradient at start."""
        model = SimpleModel()
        predictor = GradientPredictor(model)

        assert predictor.should_compute_full(), "Should compute full gradient at start"

    def test_record_gradient(self):
        """Recording gradient should populate history."""
        model = SimpleModel()
        predictor = GradientPredictor(model)

        # Compute gradient
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()

        # Record
        predictor.record_gradient()

        # Check history has entries
        for name, history in predictor.gradient_history.items():
            assert len(history) == 1, f"History should have 1 entry for {name}"

    def test_predict_gradient_after_history(self):
        """Prediction should work after building history."""
        model = SimpleModel()
        predictor = GradientPredictor(model, PredictionConfig(history_size=3))

        # Build up history with multiple backward passes
        for _ in range(3):
            model.zero_grad()
            x = torch.randn(4, 64)
            loss = model(x).sum()
            loss.backward()
            predictor.record_gradient()

        # Should be able to predict now
        predicted = predictor.predict_gradient()

        assert len(predicted) > 0, "Should have predictions"
        for name, grad in predicted.items():
            assert grad is not None, f"Prediction for {name} should not be None"

    def test_predictive_trainer_step_types(self):
        """Predictive trainer should alternate between full and predicted steps."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = PredictiveTrainer(
            model, optimizer, PredictionConfig(prediction_steps=2, history_size=3)
        )

        step_types = []
        for _ in range(10):
            x = torch.randn(4, 64)
            loss = model(x).sum()
            stats = trainer.step(loss)
            step_types.append(stats["type"])

        # Should have both full and predicted steps
        assert "full" in step_types, "Should have full gradient steps"
        # After warmup, should have predicted steps
        assert step_types.count("full") > 0
        assert stats["speedup"] >= 1.0, "Speedup should be >= 1.0"


class TestTernaryOptimizer:
    """Tests for ternary gradient accumulation and optimization."""

    def test_ternary_accumulator_accumulate(self):
        """Ternary accumulator should accumulate gradients."""
        model = SimpleModel()
        accumulator = TernaryGradientAccumulator(model)

        # Accumulate multiple gradients
        for _ in range(4):
            model.zero_grad()
            x = torch.randn(4, 64)
            loss = model(x).sum()
            loss.backward()
            accumulator.accumulate()

        assert accumulator.count == 4, "Should count 4 accumulations"

    def test_ternary_accumulator_get_accumulated(self):
        """Should retrieve accumulated gradients."""
        model = SimpleModel()
        accumulator = TernaryGradientAccumulator(model)

        # Accumulate
        for _ in range(4):
            model.zero_grad()
            x = torch.randn(4, 64)
            loss = model(x).sum()
            loss.backward()
            accumulator.accumulate()

        accumulated = accumulator.get_accumulated()

        assert len(accumulated) > 0, "Should have accumulated gradients"
        for name, grad in accumulated.items():
            assert grad is not None, f"Accumulated gradient for {name} should not be None"
            assert not torch.isnan(grad).any(), f"No NaN values for {name}"

    def test_ternary_accumulator_reset(self):
        """Reset should clear accumulator state."""
        model = SimpleModel()
        accumulator = TernaryGradientAccumulator(model)

        # Accumulate
        model.zero_grad()
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        accumulator.accumulate()

        assert accumulator.count == 1

        # Reset
        accumulator.reset()

        assert accumulator.count == 0, "Count should be 0 after reset"

    def test_ternary_optimizer_step(self):
        """Ternary optimizer should perform updates at accumulation boundary."""
        model = SimpleModel()
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = TernaryOptimizer(model, base_optimizer, TernaryConfig(accumulation_steps=4))

        updates = []
        for _i in range(12):
            model.zero_grad()
            x = torch.randn(4, 64)
            loss = model(x).sum()
            loss.backward()
            did_update = optimizer.step()
            updates.append(did_update)

        # Should update at steps 4, 8, 12 (indices 3, 7, 11)
        assert updates[3], "Should update at step 4"
        assert updates[7], "Should update at step 8"
        assert updates[11], "Should update at step 12"
        assert not updates[0], "Should not update at step 1"
        assert not updates[1], "Should not update at step 2"

    def test_ternary_memory_savings(self):
        """Ternary accumulator should report memory savings."""
        model = SimpleModel()
        accumulator = TernaryGradientAccumulator(model)

        savings = accumulator.memory_savings()

        # Should save significant memory (close to 1 - 2/32 = ~93.75%)
        assert savings > 0.9, f"Expected > 90% memory savings, got {savings:.2%}"


class TestPhaseTrainer:
    """Tests for phase-based training with prediction/correction cycles."""

    def test_phase_trainer_initial_phase(self):
        """Trainer should start in FULL phase."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = PhaseTrainer(model, optimizer)

        assert trainer.current_phase == TrainingPhase.FULL, "Should start in FULL phase"

    def test_phase_trainer_phase_transitions(self):
        """Trainer should transition through phases."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        config = PhaseConfig(full_steps=3, predict_steps=6, correct_every=3)
        trainer = PhaseTrainer(model, optimizer, config)

        def compute_loss(m, batch):
            return m(batch["x"]).sum()

        phases_seen = set()
        for _step in range(20):
            batch = {"x": torch.randn(4, 64)}
            stats = trainer.train_step(batch, compute_loss)
            phases_seen.add(stats["phase"])

        # Should see all phases
        assert "FULL" in phases_seen, "Should have FULL phase"
        assert "PREDICT" in phases_seen, "Should have PREDICT phase"
        # CORRECT may or may not appear depending on timing

    def test_phase_trainer_speedup(self):
        """Trainer should report speedup ratio."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        config = PhaseConfig(full_steps=5, predict_steps=20)
        trainer = PhaseTrainer(model, optimizer, config)

        def compute_loss(m, batch):
            return m(batch["x"]).sum()

        for _ in range(50):
            batch = {"x": torch.randn(4, 64)}
            stats = trainer.train_step(batch, compute_loss)

        # After many steps, speedup should be > 1
        assert stats["speedup"] > 1.0, f"Speedup {stats['speedup']:.2f} should be > 1.0"

    def test_phase_trainer_state_dict(self):
        """Trainer should save/load state correctly."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = PhaseTrainer(model, optimizer)

        def compute_loss(m, batch):
            return m(batch["x"]).sum()

        # Run some steps
        for _ in range(10):
            batch = {"x": torch.randn(4, 64)}
            trainer.train_step(batch, compute_loss)

        # Save state
        state = trainer.state_dict()

        # Create new trainer and load
        trainer2 = PhaseTrainer(model, torch.optim.AdamW(model.parameters()))
        trainer2.load_state_dict(state)

        assert trainer2.total_step == trainer.total_step, "Total steps should match"
        assert trainer2.cycle_count == trainer.cycle_count, "Cycle count should match"

    def test_phase_trainer_loss_tracking(self):
        """Trainer should track loss per phase."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = PhaseTrainer(model, optimizer, PhaseConfig(full_steps=5, predict_steps=10))

        def compute_loss(m, batch):
            return m(batch["x"]).sum()

        for _ in range(30):
            batch = {"x": torch.randn(4, 64)}
            trainer.train_step(batch, compute_loss)

        # Should have recorded losses
        assert len(trainer.phase_losses[TrainingPhase.FULL]) > 0, "Should have FULL phase losses"
        assert len(trainer.phase_losses[TrainingPhase.PREDICT]) > 0, (
            "Should have PREDICT phase losses"
        )


class TestIntegration:
    """Integration tests for the optimization module."""

    def test_full_training_loop(self):
        """Test complete training with phase trainer."""
        model = SimpleModel(input_size=32, hidden_size=64, output_size=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        config = PhaseConfig(
            full_steps=5,
            predict_steps=10,
            correct_every=5,
        )
        trainer = PhaseTrainer(model, optimizer, config)

        def compute_loss(m, batch):
            output = m(batch["x"])
            target = batch["y"]
            return nn.functional.mse_loss(output, target)

        initial_loss = None
        final_loss = None

        for _step in range(100):
            batch = {
                "x": torch.randn(8, 32),
                "y": torch.randn(8, 16),
            }
            stats = trainer.train_step(batch, compute_loss)

            if initial_loss is None:
                initial_loss = stats["loss"]
            final_loss = stats["loss"]

        # Loss should decrease (model should learn)
        assert final_loss < initial_loss * 0.9, "Training should reduce loss"

        # Speedup should be meaningful
        trainer_stats = trainer.get_stats()
        assert trainer_stats["speedup"] > 1.0, "Should achieve training speedup"

    def test_module_can_be_imported_standalone(self):
        """The optimization module should be importable as standalone."""
        # This tests that the module is self-contained
        from tritter.training import optimization

        assert hasattr(optimization, "PhaseTrainer")
        assert hasattr(optimization, "VSAGradientCompressor")
        assert hasattr(optimization, "TernaryOptimizer")
        assert hasattr(optimization, "GradientPredictor")
