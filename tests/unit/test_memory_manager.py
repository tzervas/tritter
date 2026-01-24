"""Unit tests for memory manager module.

Validates MemoryManager class behavior for progressive layer loading.

Why: Memory management is critical for streaming inference - incorrect tracking
or budget enforcement can lead to OOM crashes. These tests ensure:
1. Allocations are tracked correctly
2. Budget limits are enforced before OOM occurs
3. Free operations clean up properly
4. Multiple allocations can coexist
5. Clear() removes all allocations

Testing strategy: Validates both happy paths (correct allocation/free) and error
paths (budget exceeded, duplicate names), ensuring fail-fast behavior.
"""

import pytest
import torch

from tritter.core.config import TritterConfig
from tritter.inference.memory_manager import MemoryManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for MemoryManager tests")
class TestMemoryManager:
    """Test suite for MemoryManager class."""

    def test_initialization(self) -> None:
        """Test memory manager initialization.

        Validates that MemoryManager correctly initializes with config
        and sets up budget tracking.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        assert manager.config == config
        assert manager.device.type == "cuda"
        # Budget may be adjusted for OS overhead (auto_detect_budget=True by default)
        # So budget_bytes should be <= configured budget
        config_budget_bytes = int(config.gpu_memory_budget_gb * 1024**3)
        assert manager.budget_bytes <= config_budget_bytes
        assert manager.budget_bytes > 0
        assert manager.allocated_bytes == 0
        assert manager.available_bytes == manager.budget_bytes

    def test_initialization_with_custom_device(self) -> None:
        """Test memory manager initialization with custom device.

        Validates that custom device parameter is respected.
        """
        config = TritterConfig()
        device = torch.device("cuda:1")
        manager = MemoryManager(config, device=device)

        assert manager.device == device

    def test_basic_allocation(self) -> None:
        """Test basic memory allocation.

        Validates that allocate() creates a tensor on GPU and updates
        tracking state correctly.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size_mb = 100
        size_bytes = size_mb * 1024 * 1024

        tensor = manager.allocate("test_alloc", size_bytes, dtype=torch.float32)

        # Verify tensor properties
        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.float32
        assert tensor.numel() == size_bytes // 4  # float32 = 4 bytes

        # Verify tracking
        assert manager.allocated_bytes == size_bytes
        assert manager.available_bytes == manager.budget_bytes - size_bytes

    def test_multiple_allocations(self) -> None:
        """Test multiple allocations tracking.

        Validates that multiple allocations are tracked independently
        and total allocated bytes is correct.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size1 = 100 * 1024 * 1024  # 100 MB
        size2 = 200 * 1024 * 1024  # 200 MB

        manager.allocate("alloc1", size1)
        manager.allocate("alloc2", size2)

        assert manager.allocated_bytes == size1 + size2
        assert manager.available_bytes == manager.budget_bytes - (size1 + size2)

        # Verify both allocations exist
        alloc1 = manager.get_allocation("alloc1")
        alloc2 = manager.get_allocation("alloc2")

        assert alloc1 is not None
        assert alloc2 is not None
        assert alloc1.size_bytes == size1
        assert alloc2.size_bytes == size2

    def test_free_allocation(self) -> None:
        """Test freeing an allocation.

        Validates that free() removes allocation and updates tracking.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size_bytes = 100 * 1024 * 1024
        manager.allocate("test_alloc", size_bytes)

        assert manager.allocated_bytes == size_bytes

        manager.free("test_alloc")

        assert manager.allocated_bytes == 0
        assert manager.available_bytes == manager.budget_bytes
        assert manager.get_allocation("test_alloc") is None

    def test_budget_enforcement(self) -> None:
        """Test that allocations exceeding budget raise RuntimeError.

        Validates that MemoryManager prevents OOM by enforcing budget
        before attempting allocation.
        """
        config = TritterConfig(gpu_memory_budget_gb=1.0)  # 1 GB budget
        manager = MemoryManager(config)

        # Try to allocate more than budget
        excessive_size = int(1.5 * 1024**3)  # 1.5 GB

        with pytest.raises(RuntimeError) as exc_info:
            manager.allocate("too_large", excessive_size)

        # Verify error message contains helpful info
        error_msg = str(exc_info.value)
        assert "exceed budget" in error_msg
        assert "1.0 GB" in error_msg

    def test_budget_enforcement_cumulative(self) -> None:
        """Test that cumulative allocations respect budget.

        Validates that budget is enforced across multiple allocations,
        not just individual ones.
        """
        config = TritterConfig(gpu_memory_budget_gb=1.0)
        manager = MemoryManager(config)

        size = 600 * 1024 * 1024  # 600 MB

        # First allocation should succeed (600 MB < 1 GB)
        manager.allocate("alloc1", size)

        # Second allocation should fail (600 + 600 = 1200 MB > 1024 MB)
        with pytest.raises(RuntimeError) as exc_info:
            manager.allocate("alloc2", size)

        assert "exceed budget" in str(exc_info.value)

    def test_duplicate_allocation_name_raises_error(self) -> None:
        """Test that duplicate allocation names raise RuntimeError.

        Validates that allocation names must be unique to prevent
        accidental overwrites or tracking errors.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size = 100 * 1024 * 1024

        manager.allocate("duplicate", size)

        with pytest.raises(RuntimeError) as exc_info:
            manager.allocate("duplicate", size)

        assert "already exists" in str(exc_info.value)

    def test_free_nonexistent_allocation_raises_error(self) -> None:
        """Test that freeing non-existent allocation raises KeyError.

        Validates that free() fails fast when given invalid allocation name.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        with pytest.raises(KeyError) as exc_info:
            manager.free("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_clear_removes_all_allocations(self) -> None:
        """Test that clear() removes all allocations.

        Validates that clear() is a safe shutdown path that resets
        all tracking state.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        # Create multiple allocations
        size = 100 * 1024 * 1024
        manager.allocate("alloc1", size)
        manager.allocate("alloc2", size)
        manager.allocate("alloc3", size)

        assert manager.allocated_bytes == 3 * size

        # Clear all
        manager.clear()

        assert manager.allocated_bytes == 0
        assert manager.available_bytes == manager.budget_bytes
        assert manager.get_allocation("alloc1") is None
        assert manager.get_allocation("alloc2") is None
        assert manager.get_allocation("alloc3") is None

    def test_get_budget(self) -> None:
        """Test get_budget() returns correct values.

        Validates that get_budget() returns (total, allocated, available)
        tuple with correct values.
        """
        config = TritterConfig(gpu_memory_budget_gb=2.0)
        manager = MemoryManager(config)

        size = 500 * 1024 * 1024  # 500 MB
        manager.allocate("alloc1", size)

        total, allocated, available = manager.get_budget()

        assert total == int(2.0 * 1024**3)
        assert allocated == size
        assert available == total - size

    def test_allocation_with_different_dtypes(self) -> None:
        """Test allocations with different data types.

        Validates that allocate() correctly handles different dtypes
        with varying element sizes.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size_bytes = 1024  # 1 KB

        # float32 (4 bytes per element)
        tensor_f32 = manager.allocate("f32", size_bytes, dtype=torch.float32)
        assert tensor_f32.dtype == torch.float32
        assert tensor_f32.numel() == size_bytes // 4

        # float16 (2 bytes per element)
        tensor_f16 = manager.allocate("f16", size_bytes, dtype=torch.float16)
        assert tensor_f16.dtype == torch.float16
        assert tensor_f16.numel() == size_bytes // 2

        # int8 (1 byte per element)
        tensor_i8 = manager.allocate("i8", size_bytes, dtype=torch.int8)
        assert tensor_i8.dtype == torch.int8
        assert tensor_i8.numel() == size_bytes

    def test_get_allocation_returns_none_for_missing(self) -> None:
        """Test that get_allocation() returns None for missing allocations.

        Validates that get_allocation() is safe to call without raising
        errors for non-existent allocations.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        result = manager.get_allocation("nonexistent")

        assert result is None

    def test_get_allocation_returns_correct_details(self) -> None:
        """Test that get_allocation() returns correct allocation details.

        Validates that MemoryAllocation object contains accurate information.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size_bytes = 100 * 1024 * 1024
        tensor = manager.allocate("test_alloc", size_bytes, dtype=torch.float16)

        alloc = manager.get_allocation("test_alloc")

        assert alloc is not None
        assert alloc.name == "test_alloc"
        assert alloc.size_bytes == size_bytes
        assert alloc.tensor is tensor

    def test_allocation_after_free(self) -> None:
        """Test that freed memory can be reallocated.

        Validates that free() correctly releases memory for reuse.
        """
        config = TritterConfig(gpu_memory_budget_gb=1.0)
        manager = MemoryManager(config)

        size = 600 * 1024 * 1024  # 600 MB

        # Allocate and free
        manager.allocate("temp", size)
        manager.free("temp")

        # Should be able to allocate again with same name
        manager.allocate("temp", size)

        assert manager.allocated_bytes == size

    def test_available_bytes_property(self) -> None:
        """Test available_bytes property.

        Validates that available_bytes property returns correct value.
        """
        config = TritterConfig(gpu_memory_budget_gb=1.0)
        manager = MemoryManager(config)

        size = 256 * 1024 * 1024  # 256 MB
        manager.allocate("alloc", size)

        expected_available = int(1.0 * 1024**3) - size
        assert manager.available_bytes == expected_available

    def test_allocated_bytes_property(self) -> None:
        """Test allocated_bytes property.

        Validates that allocated_bytes property returns correct value.
        """
        config = TritterConfig()
        manager = MemoryManager(config)

        size1 = 100 * 1024 * 1024
        size2 = 200 * 1024 * 1024

        manager.allocate("alloc1", size1)
        manager.allocate("alloc2", size2)

        assert manager.allocated_bytes == size1 + size2
