# pyo3-dlpack Examples

This directory contains practical examples demonstrating how to use pyo3-dlpack.

## Features Demonstrated

- **Importing** tensors from Python (NumPy, PyTorch) to Rust
- **Inspecting** tensor metadata (shape, dtype, device)
- **Processing** tensor data in Rust
- **Exporting** tensors from Rust back to Python
- **Round-trip** processing (Python → Rust → Python)
- **CUDA tensors** - handling NVIDIA GPU tensors
- **Metal tensors** - handling Apple Silicon GPU tensors (MPS)

## Running the Examples

### Quick Start

```bash
cd /path/to/pyo3-dlpack

# Build the test module (includes all example functions)
cd tests/python_helpers && maturin develop && cd ../..

# Run the Python demo
python examples/demo.py
```

Or use interactively:

```python
import numpy as np
import dlpack_test_module as m

# Create a NumPy array
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

# Inspect it in Rust
m.inspect_tensor(arr)

# Sum all elements
total = m.sum_tensor(arr)
print(f"Sum: {total}")

# Double all values
doubled = m.double_tensor(arr)
result = np.from_dlpack(doubled)
print(result)

# Create tensors in Rust
rust_tensor = m.create_tensor()
arr = np.from_dlpack(rust_tensor)
print(arr)
```

### With PyTorch

If you have PyTorch installed:

```python
import torch
import dlpack_test_module as m

# Works with PyTorch tensors too!
tensor = torch.randn(3, 4)
m.inspect_tensor(tensor)

total = m.sum_tensor(tensor)

# Create tensor in Rust, convert to PyTorch
rust_capsule = m.create_identity(5)
identity = torch.from_dlpack(rust_capsule)
print(identity)
```

### With CUDA (NVIDIA GPU)

If you have PyTorch with CUDA support:

```python
import torch
import dlpack_test_module as m

# Create a CUDA tensor
cuda_tensor = torch.randn(3, 4, device="cuda:0")

# Inspect in Rust - shows CUDA device info
m.inspect_tensor(cuda_tensor)

# Check device type
device = m.get_device_type(cuda_tensor)
print(f"Device: {device}")  # Output: cuda:0

# Get the raw CUDA device pointer (for kernel interop)
ptr = m.get_data_ptr(cuda_tensor)
print(f"CUDA pointer: 0x{ptr:x}")

# Validate tensor properties
is_valid = m.validate_tensor(cuda_tensor, [3, 4], "cuda")
```

### With Metal (Apple Silicon GPU)

If you have PyTorch with MPS (Metal Performance Shaders) support on macOS:

```python
import torch
import dlpack_test_module as m

# Create an MPS tensor (Apple Silicon GPU)
mps_tensor = torch.randn(3, 4, device="mps:0")

# Inspect in Rust - shows Metal device info
m.inspect_tensor(mps_tensor)

# Check device type
device = m.get_device_type(mps_tensor)
print(f"Device: {device}")  # Output: metal:0

# Get the raw Metal buffer pointer
ptr = m.get_data_ptr(mps_tensor)
print(f"Metal pointer: 0x{ptr:x}")

# Check if it's a GPU tensor
is_gpu = m.is_gpu_tensor(mps_tensor)
print(f"Is GPU: {is_gpu}")  # Output: True
```

## What the Examples Demonstrate

### Import Path (Python → Rust)

```rust
#[pyfunction]
fn process_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<f32> {
    // Import any Python tensor (NumPy, PyTorch, etc.)
    let tensor = PyTensor::from_pyany(py, obj)?;

    // Access metadata
    println!("Shape: {:?}", tensor.shape());
    println!("Device: {:?}", tensor.device());

    // Access data (if CPU)
    let ptr = tensor.data_ptr() as *const f32;
    // ... process data

    Ok(result)
}
```

### Export Path (Rust → Python)

```rust
struct MyTensor {
    data: Vec<f32>,
    shape: Vec<i64>,
}

impl IntoDLPack for MyTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            self.shape.clone(),
        )
    }
}

#[pyfunction]
fn create_tensor(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = MyTensor {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![2, 2],
    };

    // Export as DLPack capsule
    tensor.into_dlpack(py)
}
```

Then in Python:

```python
import numpy as np
import torch

# Get tensor from Rust
capsule = my_module.create_tensor()

# Use with NumPy
arr = np.from_dlpack(capsule)

# Or use with PyTorch
# tensor = torch.from_dlpack(capsule)  # Note: capsule consumed, need new one
```

### GPU Tensor Handling (CUDA and Metal)

```rust
#[pyfunction]
fn process_gpu_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let tensor = PyTensor::from_pyany(py, obj)?;
    let device = tensor.device();

    // Check device type
    if device.is_cuda() {
        println!("CUDA tensor on GPU {}", device.device_id);
        // Get device pointer for CUDA kernel
        let cuda_ptr = tensor.data_ptr();
        // Pass cuda_ptr to your CUDA kernels...
    } else if device.is_metal() {
        println!("Metal tensor on GPU {}", device.device_id);
        // Get Metal buffer pointer
        let metal_ptr = tensor.data_ptr();
        // Pass metal_ptr to Metal compute shaders...
    } else if device.is_cpu() {
        println!("CPU tensor");
        // Safe to access as regular memory
    }

    Ok(())
}
```

## Key Features Demonstrated

✅ **Zero-copy** data sharing
✅ **CPU** and **GPU** tensors
✅ **CUDA support** (NVIDIA GPUs)
✅ **Metal support** (Apple Silicon GPUs via MPS)
✅ **ROCm support** (AMD GPUs)
✅ **Multiple frameworks** (NumPy, PyTorch)
✅ **Contiguous** and **non-contiguous** tensors
✅ **Different dtypes** (f32, f64, i32, etc.)
✅ **Round-trip** processing
✅ **Memory safety** (no double-free, proper ownership)

## Requirements

- Rust toolchain
- Python 3.9+
- maturin (`pip install maturin`)
- numpy (`pip install numpy`)
- torch (optional, `pip install torch`)

## Building for Production

To build an optimized release version:

```bash
maturin build --release
```

This creates a wheel in `target/wheels/` that can be installed with pip.
