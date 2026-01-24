"""Tests for Trainer and TrainingConfig.

Why: These tests validate that the training loop functions correctly with:
1. Configuration validation catches invalid settings
2. Optimizer properly separates weight decay groups
3. Learning rate scheduler implements warmup and cosine decay
4. Training step returns loss and updates gradients
5. Checkpoint save/load preserves training state
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tritter.core.config import TritterConfig
from tritter.training.trainer import Trainer, TrainingConfig


class SimpleModel(nn.Module):
    """Mock model for testing trainer.

    Why: Use a simple model to test trainer logic without the complexity
    of the full TritterModel. This isolates trainer tests from model bugs.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        """Initialize simple model.

        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Optional attention mask

        Returns:
            Logits (B, L, V)
        """
        hidden = self.embed(input_ids)  # (B, L, hidden_size)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)  # (B, L, vocab_size)
        return logits


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for testing.

    Why: Tests are designed to run on CPU (use_amp=False). When CUDA is
    available but the GPU architecture is incompatible (e.g., RTX 5080/sm_120),
    torch.cuda.is_available() returns True but operations fail. Explicitly
    using CPU avoids this issue.
    """
    return torch.device("cpu")


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create training config for tests.

    Returns:
        TrainingConfig with small values for fast tests

    Why: Use small values (few steps, small batch) to make tests fast
    while still validating the logic.
    """
    return TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.1,
        max_grad_norm=1.0,
        warmup_steps=10,
        max_steps=100,
        batch_size=2,
        gradient_accumulation_steps=2,
        save_steps=50,
        eval_steps=25,
        log_steps=5,
        output_dir="test_checkpoints",
        use_amp=False,  # Disable AMP for CPU tests
    )


@pytest.fixture
def model_config() -> TritterConfig:
    """Create model config for tests.

    Returns:
        TritterConfig with small model size

    Why: Use minimal config to reduce memory and speed up tests.
    """
    return TritterConfig(
        model_size="3B",
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=256,
        vocab_size=1000,
        use_bitnet=False,  # Standard weights for simplicity
        use_flash_attention=False,
    )


@pytest.fixture
def simple_model(model_config: TritterConfig) -> SimpleModel:
    """Create simple model for tests.

    Args:
        model_config: Model configuration

    Returns:
        SimpleModel instance

    Why: Use SimpleModel instead of full TritterModel to isolate
    trainer tests from model implementation details.
    """
    return SimpleModel(model_config.vocab_size, model_config.hidden_size)


@pytest.fixture
def train_dataloader(model_config: TritterConfig) -> DataLoader:
    """Create training dataloader.

    Args:
        model_config: Model configuration

    Returns:
        DataLoader with synthetic data

    Why: Generate random sequences for testing. Real data not needed
    since we're testing trainer mechanics, not model quality.
    """
    # Create synthetic data: random sequences of tokens
    num_samples = 20
    seq_len = 32
    vocab_size = model_config.vocab_size

    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(input_ids)

    # Collate function to create batch dict
    def collate_fn(batch: list[tuple[torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        return {"input_ids": input_ids}

    return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


@pytest.fixture
def eval_dataloader(model_config: TritterConfig) -> DataLoader:
    """Create evaluation dataloader.

    Args:
        model_config: Model configuration

    Returns:
        DataLoader with synthetic eval data

    Why: Separate eval set to test evaluation logic.
    """
    # Create synthetic eval data
    num_samples = 10
    seq_len = 32
    vocab_size = model_config.vocab_size

    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(input_ids)

    def collate_fn(batch: list[tuple[torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        return {"input_ids": input_ids}

    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)


class TestTrainingConfig:
    """Tests for TrainingConfig validation.

    Why: Configuration validation should catch invalid settings before
    expensive training starts.
    """

    def test_valid_config(self, training_config: TrainingConfig) -> None:
        """Test that valid config passes validation.

        Args:
            training_config: Valid training config

        Why: Sanity check that valid configs are accepted.
        """
        assert training_config.learning_rate > 0
        assert training_config.warmup_steps >= 0
        assert training_config.max_steps > training_config.warmup_steps

    def test_negative_learning_rate(self) -> None:
        """Test that negative learning rate is rejected.

        Why: Negative learning rate would cause divergence.
        """
        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.001)

    def test_negative_warmup_steps(self) -> None:
        """Test that negative warmup steps are rejected.

        Why: Negative warmup steps are nonsensical.
        """
        with pytest.raises(AssertionError, match="warmup_steps must be non-negative"):
            TrainingConfig(warmup_steps=-10)

    def test_max_steps_less_than_warmup(self) -> None:
        """Test that max_steps must exceed warmup_steps.

        Why: Training would end before warmup completes.
        """
        with pytest.raises(
            AssertionError,
            match="max_steps .* must be greater than warmup_steps",
        ):
            TrainingConfig(warmup_steps=1000, max_steps=500)

    def test_zero_gradient_accumulation(self) -> None:
        """Test that zero gradient accumulation is rejected.

        Why: Zero gradient accumulation would cause division by zero.
        """
        with pytest.raises(
            AssertionError,
            match="gradient_accumulation_steps must be positive",
        ):
            TrainingConfig(gradient_accumulation_steps=0)


class TestTrainer:
    """Tests for Trainer class.

    Why: Validate that trainer correctly implements training loop,
    optimizer setup, LR scheduling, and checkpointing.
    """

    def test_trainer_initialization(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that trainer initializes correctly.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Verify trainer sets up all components without errors.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            device=cpu_device,
        )

        assert trainer.model == simple_model
        assert trainer.config == training_config
        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_optimizer_parameter_groups(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that optimizer separates decay and no-decay parameters.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Weight decay should only apply to weights, not biases or norms.
        This follows best practices from BERT/GPT and prevents over-regularization.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            device=cpu_device,
        )

        # Optimizer should have two parameter groups
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) == 2

        # First group: decay parameters (weight_decay > 0)
        decay_group = param_groups[0]
        assert decay_group["weight_decay"] == training_config.weight_decay

        # Second group: no-decay parameters (weight_decay == 0)
        no_decay_group = param_groups[1]
        assert no_decay_group["weight_decay"] == 0.0

        # Check that biases and norms are in no-decay group
        # Why: SimpleModel has LayerNorm which should not be regularized
        assert len(no_decay_group["params"]) > 0

    def test_lr_scheduler_warmup(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that learning rate scheduler implements warmup correctly.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Warmup should linearly increase LR from 0 to target over
        warmup_steps. This prevents early training instability.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            device=cpu_device,
        )

        # Initial LR should be near 0 (step 0 of warmup)
        initial_lr = trainer.scheduler.get_last_lr()[0]
        assert initial_lr < training_config.learning_rate * 0.1

        # After half warmup, LR should be ~50% of target
        for _ in range(training_config.warmup_steps // 2):
            trainer.scheduler.step()
        halfway_lr = trainer.scheduler.get_last_lr()[0]
        assert (
            0.4 * training_config.learning_rate < halfway_lr < 0.6 * training_config.learning_rate
        )

        # After full warmup, LR should be at target
        for _ in range(training_config.warmup_steps // 2):
            trainer.scheduler.step()
        peak_lr = trainer.scheduler.get_last_lr()[0]
        assert abs(peak_lr - training_config.learning_rate) < 1e-6

    def test_train_step_returns_loss(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that train_step returns a valid loss value.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Training step should compute forward pass, loss, and
        backward pass, returning the loss value.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            device=cpu_device,
        )

        # Get a batch
        batch = next(iter(train_dataloader))

        # Run training step
        loss = trainer.train_step(batch)

        # Loss should be a positive finite number
        assert isinstance(loss, float)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))

    def test_train_step_updates_gradients(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that train_step computes gradients.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Backward pass should populate .grad for all parameters.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            device=cpu_device,
        )

        # Get a batch
        batch = next(iter(train_dataloader))

        # Zero gradients
        trainer.optimizer.zero_grad()

        # Run training step
        trainer.train_step(batch)

        # Check that at least some parameters have gradients
        has_grad = False
        for param in simple_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed during train_step"

    def test_checkpoint_save_load(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that checkpoint save/load preserves training state.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Checkpointing enables fault tolerance. Save/load should
        restore exact training state including step count and optimizer state.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer and run a few steps
            trainer = Trainer(
                simple_model,
                model_config,
                training_config,
                train_dataloader,
                device=cpu_device,
            )

            # Run a few training steps
            for _ in range(5):
                batch = next(iter(train_dataloader))
                trainer.train_step(batch)
                if (trainer.global_step + 1) % training_config.gradient_accumulation_steps == 0:
                    trainer.optimizer.step()
                    trainer.scheduler.step()
                    trainer.optimizer.zero_grad()
                trainer.global_step += 1

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint"
            trainer.save_checkpoint(str(checkpoint_path))

            # Save model state for comparison
            original_step = trainer.global_step
            original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

            # Create new trainer with same model
            new_model = SimpleModel(model_config.vocab_size, model_config.hidden_size)
            new_trainer = Trainer(
                new_model,
                model_config,
                training_config,
                train_dataloader,
                device=cpu_device,
            )

            # Load checkpoint
            new_trainer.load_checkpoint(str(checkpoint_path))

            # Verify state is restored
            assert new_trainer.global_step == original_step

            # Check that model weights match
            for key in original_state.keys():
                original_param = original_state[key]
                loaded_param = new_model.state_dict()[key]
                assert torch.allclose(
                    original_param,
                    loaded_param,
                    rtol=1e-5,
                    atol=1e-5,
                ), f"Parameter {key} not restored correctly"

    def test_evaluate(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that evaluation computes metrics correctly.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            eval_dataloader: Evaluation data
            cpu_device: CPU device for testing

        Why: Evaluation should compute loss and perplexity without
        updating model parameters.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            eval_dataloader,
            device=cpu_device,
        )

        # Save initial model state
        initial_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        # Run evaluation
        metrics = trainer.evaluate()

        # Check metrics exist
        assert "eval_loss" in metrics
        assert "perplexity" in metrics

        # Loss should be positive finite
        assert metrics["eval_loss"] > 0
        assert not torch.isnan(torch.tensor(metrics["eval_loss"]))

        # Perplexity should be >= 1
        # Why: perplexity = exp(loss), and loss >= 0, so perplexity >= 1
        assert metrics["perplexity"] >= 1.0

        # Model weights should not change during evaluation
        for key in initial_state.keys():
            initial_param = initial_state[key]
            current_param = simple_model.state_dict()[key]
            assert torch.allclose(
                initial_param,
                current_param,
                rtol=1e-5,
                atol=1e-5,
            ), f"Parameter {key} changed during evaluation"

    def test_gradient_accumulation(
        self,
        simple_model: SimpleModel,
        model_config: TritterConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        cpu_device: torch.device,
    ) -> None:
        """Test that gradients accumulate correctly over multiple steps.

        Args:
            simple_model: Mock model
            model_config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data
            cpu_device: CPU device for testing

        Why: Gradient accumulation simulates larger batch sizes without
        increasing memory usage. Gradients should accumulate over N steps
        before optimizer updates.
        """
        trainer = Trainer(
            simple_model,
            model_config,
            training_config,
            train_dataloader,
            device=cpu_device,
        )

        # Get initial parameter value
        initial_param = next(simple_model.parameters()).clone()

        # Run one accumulation cycle
        accumulation_steps = training_config.gradient_accumulation_steps
        for step in range(accumulation_steps):
            batch = next(iter(train_dataloader))
            trainer.train_step(batch)

            # Parameters should not change until accumulation completes
            current_param = next(simple_model.parameters())
            if step < accumulation_steps - 1:
                assert torch.allclose(
                    initial_param,
                    current_param,
                    rtol=1e-5,
                    atol=1e-5,
                ), f"Parameters changed before accumulation completed at step {step}"
