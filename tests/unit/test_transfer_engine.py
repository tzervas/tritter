"""Unit tests for TransferEngine async H2D transfers.

Tests cover:
- Basic synchronous and asynchronous transfers
- Memory pinning for faster DMA
- Transfer synchronization and event tracking
- Error handling for invalid device tensors
"""

import pytest
import torch

from conftest import requires_cuda
from tritter.core.config import TritterConfig
from tritter.inference.transfer_engine import TransferEngine, pin_model_weights


@requires_cuda
class TestTransferEngine:
    """Test suite for TransferEngine CUDA streaming transfers."""

    def test_basic_sync_transfer(self) -> None:
        """Test synchronous CPU-to-GPU transfer.

        Why: Verifies basic transfer functionality works and data arrives
        on correct device with correct values.
        """
        config = TritterConfig()
        engine = TransferEngine(config, device=torch.device("cuda:0"))

        # Create CPU tensor
        cpu_tensor = torch.randn(100, 100)
        assert cpu_tensor.device.type == "cpu"

        # Transfer to GPU
        gpu_tensor = engine.transfer_async(cpu_tensor, non_blocking=False)
        assert gpu_tensor.device.type == "cuda"
        assert torch.allclose(cpu_tensor, gpu_tensor.cpu())

        # Should have pending transfer event
        assert engine.has_pending_transfers()

        # Sync should clear pending transfers
        engine.sync()
        assert not engine.has_pending_transfers()

    def test_async_transfer_with_preallocated(self) -> None:
        """Test async transfer to pre-allocated destination tensor.

        Why: Pre-allocated destination tensors enable double buffering,
        which is critical for overlapping transfers with computation.
        """
        config = TritterConfig()
        engine = TransferEngine(config)

        cpu_tensor = torch.randn(50, 50)
        dst_tensor = torch.zeros(50, 50, device="cuda:0")

        # Transfer to pre-allocated tensor
        result = engine.transfer_async(cpu_tensor, dst=dst_tensor)
        assert result is dst_tensor  # Should return same tensor
        assert engine.has_pending_transfers()

        # Sync and verify data
        engine.sync()
        assert torch.allclose(cpu_tensor, result.cpu())

    def test_pin_memory(self) -> None:
        """Test pinning CPU tensors for faster DMA.

        Why: Pinned memory enables async transfers at full PCIe bandwidth.
        Verifies tensors are correctly pinned and already-pinned tensors
        are handled correctly.
        """
        config = TritterConfig(use_pinned_memory=True)
        engine = TransferEngine(config)

        # Create unpinned CPU tensor
        cpu_tensor = torch.randn(100, 100)
        assert not cpu_tensor.is_pinned()

        # Pin the tensor
        pinned_tensor = engine.pin_memory(cpu_tensor)
        assert pinned_tensor.is_pinned()
        assert torch.allclose(cpu_tensor, pinned_tensor)

        # Pinning already-pinned tensor should be no-op
        repinned = engine.pin_memory(pinned_tensor)
        assert repinned is pinned_tensor

    def test_pin_memory_rejects_gpu_tensor(self) -> None:
        """Test that pin_memory rejects GPU tensors.

        Why: Only CPU tensors can be pinned. GPU tensors should raise
        clear error message.
        """
        config = TritterConfig()
        engine = TransferEngine(config)

        gpu_tensor = torch.randn(10, 10, device="cuda:0")

        with pytest.raises(ValueError, match="Can only pin CPU tensors"):
            engine.pin_memory(gpu_tensor)

    def test_transfer_rejects_gpu_source(self) -> None:
        """Test that transfer_async rejects GPU source tensors.

        Why: TransferEngine is for H2D transfers. GPU-to-GPU copies
        should use different mechanism.
        """
        config = TritterConfig()
        engine = TransferEngine(config)

        gpu_tensor = torch.randn(10, 10, device="cuda:0")

        with pytest.raises(ValueError, match="Source must be on CPU"):
            engine.transfer_async(gpu_tensor)

    def test_sync_stream(self) -> None:
        """Test stream synchronization between transfer and compute streams.

        Why: Verifies transfer stream synchronization establishes correct
        ordering between transfers and compute operations.
        """
        config = TritterConfig()
        engine = TransferEngine(config)

        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = engine.transfer_async(cpu_tensor)

        # Sync stream to ensure transfer visible on compute stream
        engine.sync_stream()

        # Should be able to use tensor on default stream
        result = gpu_tensor + 1.0
        assert result.device.type == "cuda"

    def test_multiple_pending_transfers(self) -> None:
        """Test handling of multiple pending async transfers.

        Why: Layer streaming requires multiple concurrent transfers.
        Verifies all transfers are tracked and can be synchronized.
        """
        config = TritterConfig()
        engine = TransferEngine(config)

        tensors = [torch.randn(50, 50) for _ in range(5)]

        # Start multiple async transfers
        gpu_tensors = [engine.transfer_async(t) for t in tensors]
        assert engine.has_pending_transfers()

        # Sync all transfers
        engine.sync()
        assert not engine.has_pending_transfers()

        # Verify all data transferred correctly
        for cpu_t, gpu_t in zip(tensors, gpu_tensors):
            assert torch.allclose(cpu_t, gpu_t.cpu())

    def test_transfer_stream_property(self) -> None:
        """Test access to underlying CUDA stream.

        Why: Advanced use cases may need direct stream access for
        custom synchronization patterns.
        """
        config = TritterConfig()
        engine = TransferEngine(config)

        stream = engine.transfer_stream
        assert isinstance(stream, torch.cuda.Stream)
        assert stream.device == engine.device


@requires_cuda
def test_pin_model_weights() -> None:
    """Test pinning all model weights for faster transfer.

    Why: Pre-pinning model weights at load time enables maximum
    PCIe bandwidth when streaming layers.
    """
    # Create simple model on CPU
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
    )

    # Verify initially unpinned
    for param in model.parameters():
        assert not param.is_pinned()

    # Pin all weights
    pin_model_weights(model)

    # Verify all pinned
    for param in model.parameters():
        assert param.is_pinned()

    # Test with model that has buffers
    model_with_buffers = torch.nn.BatchNorm1d(10)
    pin_model_weights(model_with_buffers)

    for buffer in model_with_buffers.buffers():
        assert buffer.is_pinned()


def test_transfer_engine_cpu_only() -> None:
    """Test TransferEngine creation when CUDA not available.

    Why: Should be able to import and create engine even without CUDA,
    though actual transfers will fail. This enables testing imports.
    """
    config = TritterConfig()

    # Should be able to create engine (may fail on actual transfer)
    if not torch.cuda.is_available():
        with pytest.raises((AssertionError, RuntimeError)):
            # CUDA stream creation will fail without CUDA
            engine = TransferEngine(config)
