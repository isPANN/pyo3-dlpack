"""
Integration tests for pyo3-dlpack DLPack protocol implementation.

Tests both CPU and GPU paths for:
1. Import: Python framework tensor -> DLPack capsule -> Rust PyTensor
2. Capsule ownership protocol (no double-free, no use-after-free)

These tests cover both import (Python -> Rust) and export (Rust -> Python) paths,
including zero-copy pointer checks where possible.

Run with: pytest tests/test_dlpack_integration.py -v
"""

import gc
import pytest
import numpy as np

# Import the test module (built with maturin)
import dlpack_test_module as dtm

# Try to import torch, mark tests as skipped if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Check if CUDA is available
HAS_CUDA = HAS_TORCH and torch.cuda.is_available()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_counters():
    """Reset drop/deleter counters before each test."""
    dtm.reset_counters()
    gc.collect()
    yield
    gc.collect()


# ============================================================================
# CPU Path Tests - Import (Python -> Rust)
# ============================================================================

class TestCpuImport:
    """Test importing CPU tensors from Python to Rust."""

    def test_import_numpy_tensor(self):
        """Import a numpy array to Rust via DLPack."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        # Import to Rust
        info = dtm.import_tensor(arr)

        assert info["shape"] == [2, 3]
        assert info["ndim"] == 2
        assert info["numel"] == 6
        assert info["itemsize"] == 4  # f32
        assert info["nbytes"] == 24
        assert info["is_contiguous"] == True
        assert info["is_cpu"] == True
        assert info["is_cuda"] == False

    def test_import_numpy_zero_copy_ptr(self):
        """Verify numpy -> Rust import shares the same data pointer (zero-copy)."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        np_ptr = arr.__array_interface__["data"][0]
        rust_ptr = dtm.get_data_ptr(arr)
        assert rust_ptr == np_ptr

    def test_import_numpy_1d_tensor(self):
        """Import a 1D numpy array."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        info = dtm.import_tensor(arr)

        assert info["shape"] == [5]
        assert info["ndim"] == 1
        assert info["numel"] == 5

    def test_import_numpy_scalar(self):
        """Import a 0D numpy array (scalar)."""
        arr = np.array(42.0, dtype=np.float32)
        info = dtm.import_tensor(arr)

        assert info["shape"] == []
        assert info["ndim"] == 0
        assert info["numel"] == 1

    def test_import_numpy_f64_tensor(self):
        """Import a float64 numpy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        info = dtm.import_tensor(arr)

        assert info["itemsize"] == 8  # f64

    def test_import_numpy_int32_tensor(self):
        """Import an int32 numpy array."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        info = dtm.import_tensor(arr)

        assert info["itemsize"] == 4  # i32

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_import_torch_cpu_tensor(self):
        """Import a PyTorch CPU tensor to Rust via DLPack."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

        # Import to Rust
        info = dtm.import_tensor(tensor)

        assert info["shape"] == [2, 3]
        assert info["is_cpu"] == True
        assert info["is_cuda"] == False

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_import_torch_transposed_tensor(self):
        """Import a non-contiguous (transposed) tensor."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).T

        info = dtm.import_tensor(tensor)

        assert info["shape"] == [3, 2]
        # Transposed tensor is not contiguous in row-major order
        assert info["is_contiguous"] == False

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_import_torch_sliced_tensor(self):
        """Import a sliced (non-contiguous) tensor."""
        tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        sliced = tensor[:, ::2]  # Take every other column

        info = dtm.import_tensor(sliced)

        assert info["shape"] == [2, 2]
        assert info["is_contiguous"] == False

    def test_capsule_marked_as_used_after_import(self):
        """Verify capsule is renamed to 'used_dltensor' after import."""
        arr = np.array([1, 2, 3], dtype=np.float32)

        # Get the capsule directly
        capsule = arr.__dlpack__()

        # Check initial name
        assert dtm.get_capsule_name(capsule) == "dltensor"

        # Import it
        info = dtm.import_from_capsule(capsule)

        # Check name changed
        assert dtm.get_capsule_name(capsule) == "used_dltensor"
        assert dtm.is_capsule_consumed(capsule) == True

    def test_second_import_of_same_capsule_fails(self):
        """Verify that importing the same capsule twice fails."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        capsule = arr.__dlpack__()

        # First import should succeed
        info = dtm.import_from_capsule(capsule)
        assert info["shape"] == [3]

        # Second import should fail (capsule renamed to used_dltensor)
        with pytest.raises(Exception):
            dtm.import_from_capsule(capsule)


# ============================================================================
# CPU Path Tests - Export (Rust -> Python)
# ============================================================================

class TestCpuExport:
    """Test exporting CPU tensors from Rust to Python."""

    def test_export_to_numpy_zero_copy_ptr(self):
        """Verify Rust export -> numpy from_dlpack shares the same data pointer."""
        capsule = dtm.export_cpu_tensor()
        capsule_ptr = dtm.capsule_data_ptr(capsule)

        class _CapsuleWrapper:
            def __init__(self, cap):
                self._cap = cap

            def __dlpack__(self):
                return self._cap

        arr = np.from_dlpack(_CapsuleWrapper(capsule))
        np_ptr = arr.__array_interface__["data"][0]

        assert np_ptr == capsule_ptr


# ============================================================================
# GPU Path Tests - Import (Python -> Rust)
# ============================================================================

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestGpuImport:
    """Test importing GPU tensors from Python to Rust."""

    def test_import_torch_cuda_tensor(self):
        """Import a PyTorch CUDA tensor to Rust via DLPack."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda:0")

        info = dtm.import_tensor(tensor)

        assert info["shape"] == [2, 3]
        assert info["is_cpu"] == False
        assert info["is_cuda"] == True
        assert info["device_id"] == 0

    def test_import_torch_cuda_1d(self):
        """Import a 1D CUDA tensor."""
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda:0")

        info = dtm.import_tensor(tensor)

        assert info["shape"] == [5]
        assert info["is_cuda"] == True

    def test_import_torch_cuda_f64(self):
        """Import a float64 CUDA tensor."""
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda:0")

        info = dtm.import_tensor(tensor)

        assert info["itemsize"] == 8
        assert info["is_cuda"] == True

    def test_import_torch_cuda_non_contiguous(self):
        """Import a non-contiguous CUDA tensor."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda:0").T

        info = dtm.import_tensor(tensor)

        assert info["shape"] == [3, 2]
        assert info["is_contiguous"] == False
        assert info["is_cuda"] == True

    def test_cuda_capsule_marked_as_used(self):
        """Verify CUDA capsule is renamed after import."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32, device="cuda:0")
        capsule = tensor.__dlpack__()

        assert dtm.get_capsule_name(capsule) == "dltensor"

        info = dtm.import_from_capsule(capsule)

        assert dtm.get_capsule_name(capsule) == "used_dltensor"
        assert info["is_cuda"] == True


# ============================================================================
# Memory Safety Tests
# ============================================================================

class TestMemorySafety:
    """Tests for memory safety (no double-free, no use-after-free)."""

    def test_no_use_after_free_prevented_by_rename(self):
        """Ensure capsule rename prevents use-after-free on double import."""
        arr = np.array([1, 2, 3], dtype=np.float32)

        # Get capsule
        capsule = arr.__dlpack__()

        # First import (should succeed and rename)
        info1 = dtm.import_from_capsule(capsule)
        assert info1["shape"] == [3]

        # Second import should fail (capsule renamed to used_dltensor)
        with pytest.raises(Exception):
            dtm.import_from_capsule(capsule)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_capsule_protection(self):
        """Test that torch capsule is protected after consumption."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        capsule = tensor.__dlpack__()

        # Import to Rust (consumes capsule)
        info = dtm.import_from_capsule(capsule)

        # Capsule should be marked as used
        assert dtm.is_capsule_consumed(capsule) == True

        # Second import should fail
        with pytest.raises(Exception):
            dtm.import_from_capsule(capsule)


# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for memory and performance."""

    def test_many_numpy_imports(self):
        """Test many numpy array imports."""
        for i in range(100):
            arr = np.random.randn(100, 100).astype(np.float32)
            info = dtm.import_tensor(arr)
            assert info["shape"] == [100, 100]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_many_torch_imports(self):
        """Test many torch tensor imports."""
        for i in range(100):
            tensor = torch.randn(100, 100)
            info = dtm.import_tensor(tensor)
            assert info["shape"] == [100, 100]

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_many_cuda_imports(self):
        """Test many CUDA tensor imports."""
        for i in range(100):
            tensor = torch.randn(100, 100, device="cuda:0")
            info = dtm.import_tensor(tensor)
            assert info["is_cuda"] == True

        gc.collect()
        torch.cuda.empty_cache()

    def test_large_tensor_numpy(self):
        """Test with large numpy arrays."""
        arr = np.random.randn(1000, 1000).astype(np.float32)
        info = dtm.import_tensor(arr)

        assert info["shape"] == [1000, 1000]
        assert info["numel"] == 1000000

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_large_tensor_cuda(self):
        """Test with large CUDA tensors."""
        tensor = torch.randn(1000, 1000, device="cuda:0")
        info = dtm.import_tensor(tensor)

        assert info["shape"] == [1000, 1000]
        assert info["is_cuda"] == True


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
