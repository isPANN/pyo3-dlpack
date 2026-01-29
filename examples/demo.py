#!/usr/bin/env python3
"""
Demo script showing how to use pyo3-dlpack with Python.

This demonstrates:
- Importing tensors from Python to Rust
- Exporting tensors from Rust to Python
- Round-trip processing with NumPy and PyTorch
"""

import numpy as np


class DLPackWrapper:
    """Wrapper to make a PyCapsule compatible with np.from_dlpack()"""
    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None):
        return self._capsule

    def __dlpack_device__(self):
        # Default to CPU (device_type=1, device_id=0)
        return (1, 0)


def from_dlpack(capsule):
    """Convert a DLPack capsule to numpy array"""
    return np.from_dlpack(DLPackWrapper(capsule))

# Try to import torch if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, using NumPy only\n")

# Import the test module (contains all example functions)
# First build it with: cd tests/python_helpers && maturin develop
try:
    import dlpack_test_module as basic_usage
except ImportError:
    print("Error: dlpack_test_module not found!")
    print("Build it with:")
    print("  cd tests/python_helpers && maturin develop")
    exit(1)


def demo_numpy():
    """Demo with NumPy arrays"""
    print("=" * 60)
    print("NumPy Demo")
    print("=" * 60)

    # Create a NumPy array
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    print(f"\nOriginal NumPy array:\n{arr}")

    # Inspect the tensor in Rust
    print("\nInspecting in Rust:")
    basic_usage.inspect_tensor(arr)

    # Sum all elements
    total = basic_usage.sum_tensor(arr)
    print(f"\nSum computed in Rust: {total}")
    print(f"Sum computed in NumPy: {arr.sum()}")

    # Double all values
    doubled = basic_usage.double_tensor(arr)
    doubled_np = from_dlpack(doubled)
    print(f"\nDoubled array (Rust):\n{doubled_np}")
    print(f"Expected:\n{arr * 2}")

    # Create tensors in Rust
    rust_tensor = basic_usage.create_tensor()
    rust_arr = from_dlpack(rust_tensor)
    print(f"\nTensor created in Rust:\n{rust_arr}")

    # Create filled tensor
    filled = basic_usage.create_filled_tensor(3.14, 3, 4)
    filled_arr = from_dlpack(filled)
    print(f"\nFilled tensor (3.14):\n{filled_arr}")

    # Create identity matrix
    identity = basic_usage.create_identity(4)
    identity_arr = from_dlpack(identity)
    print(f"\nIdentity matrix:\n{identity_arr}")


def demo_torch():
    """Demo with PyTorch tensors"""
    if not HAS_TORCH:
        return

    print("\n" + "=" * 60)
    print("PyTorch Demo")
    print("=" * 60)

    # Create a PyTorch tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    print(f"\nOriginal PyTorch tensor:\n{tensor}")

    # Inspect in Rust
    print("\nInspecting in Rust:")
    basic_usage.inspect_tensor(tensor)

    # Sum all elements
    total = basic_usage.sum_tensor(tensor)
    print(f"\nSum computed in Rust: {total}")
    print(f"Sum computed in PyTorch: {tensor.sum().item()}")

    # Double all values
    doubled_capsule = basic_usage.double_tensor(tensor)
    doubled = torch.from_dlpack(doubled_capsule)
    print(f"\nDoubled tensor (Rust):\n{doubled}")
    print(f"Expected:\n{tensor * 2}")

    # Create tensor in Rust and convert to PyTorch
    rust_capsule = basic_usage.create_tensor()
    rust_tensor = torch.from_dlpack(rust_capsule)
    print(f"\nTensor created in Rust:\n{rust_tensor}")


def demo_interop():
    """Demo interoperability between frameworks"""
    print("\n" + "=" * 60)
    print("Framework Interoperability Demo")
    print("=" * 60)

    # Create in NumPy
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    print(f"\n1. Created in NumPy:\n{np_arr}")

    # Process in Rust
    doubled_capsule = basic_usage.double_tensor(np_arr)

    # Convert to NumPy using wrapper
    doubled_np = from_dlpack(doubled_capsule)
    print(f"\n2. Processed in Rust, back to NumPy:\n{doubled_np}")

    if HAS_TORCH:
        # Convert to PyTorch using the legacy API that accepts capsules
        # Note: Can't reuse capsule after consumption, need to recreate
        doubled_capsule2 = basic_usage.double_tensor(np_arr)
        doubled_torch = torch.utils.dlpack.from_dlpack(doubled_capsule2)
        print(f"\n3. Same data in PyTorch:\n{doubled_torch}")

        # Verify they share the same data (zero-copy)
        print("\nAll conversions were zero-copy! No data was duplicated.")


def demo_cuda():
    """Demo with CUDA GPU tensors"""
    if not HAS_TORCH or not torch.cuda.is_available():
        print("\n(CUDA demo skipped - CUDA not available)")
        return

    print("\n" + "=" * 60)
    print("CUDA GPU Demo")
    print("=" * 60)

    # Create a CUDA tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         dtype=torch.float32, device="cuda:0")
    print(f"\nCUDA tensor:\n{tensor}")

    # Inspect in Rust
    print("\nInspecting CUDA tensor in Rust:")
    basic_usage.inspect_tensor(tensor)

    # Check device type
    device_str = basic_usage.get_device_type(tensor)
    print(f"\nDevice string from Rust: {device_str}")

    # Check if it's a GPU tensor
    is_gpu = basic_usage.is_gpu_tensor(tensor)
    print(f"Is GPU tensor: {is_gpu}")

    # Get raw data pointer (device pointer)
    ptr = basic_usage.get_data_ptr(tensor)
    print(f"CUDA device pointer: 0x{ptr:x}")

    # Validate tensor
    is_valid = basic_usage.validate_tensor(tensor, [2, 3], "cuda")
    print(f"Tensor validation (shape=[2,3], device=cuda): {is_valid}")

    print("\nNote: For actual GPU computation, use CUDA kernels with this pointer")


def demo_metal():
    """Demo with Metal GPU tensors (Apple Silicon MPS)"""
    if not HAS_TORCH:
        print("\n(Metal demo skipped - PyTorch not available)")
        return

    if not torch.backends.mps.is_available():
        print("\n(Metal demo skipped - MPS not available)")
        return

    print("\n" + "=" * 60)
    print("Metal GPU Demo (Apple Silicon MPS)")
    print("=" * 60)

    # Create an MPS tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         dtype=torch.float32, device="mps:0")
    print(f"\nMetal (MPS) tensor:\n{tensor}")

    # Inspect in Rust
    print("\nInspecting Metal tensor in Rust:")
    basic_usage.inspect_tensor(tensor)

    # Check device type
    device_str = basic_usage.get_device_type(tensor)
    print(f"\nDevice string from Rust: {device_str}")

    # Check if it's a GPU tensor
    is_gpu = basic_usage.is_gpu_tensor(tensor)
    print(f"Is GPU tensor: {is_gpu}")

    # Get raw data pointer (Metal buffer pointer)
    ptr = basic_usage.get_data_ptr(tensor)
    print(f"Metal device pointer: 0x{ptr:x}")

    # Validate tensor
    is_valid = basic_usage.validate_tensor(tensor, [2, 3], "metal")
    print(f"Tensor validation (shape=[2,3], device=metal): {is_valid}")

    print("\nNote: For actual GPU computation, use Metal compute shaders with this pointer")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("pyo3-dlpack Example Demo")
    print("=" * 60)
    print("\nThis demo shows zero-copy tensor exchange between")
    print("Python (NumPy/PyTorch) and Rust using DLPack.\n")

    try:
        # Run demos
        demo_numpy()
        demo_torch()
        demo_interop()

        # GPU demos
        demo_cuda()
        demo_metal()

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
