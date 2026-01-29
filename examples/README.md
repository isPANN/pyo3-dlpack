# pyo3-dlpack Examples

This directory contains practical examples demonstrating how to use pyo3-dlpack.

## Examples

### `basic_usage.rs`

A complete example showing:
- **Importing** tensors from Python (NumPy, PyTorch) to Rust
- **Inspecting** tensor metadata (shape, dtype, device)
- **Processing** tensor data in Rust
- **Exporting** tensors from Rust back to Python
- **Round-trip** processing (Python → Rust → Python)

## Running the Examples

### Option 1: As a Rust Binary

Run the demo directly from Rust:

```bash
cargo run --example basic_usage
```

This will run the built-in demo function that creates NumPy arrays and processes them.

### Option 2: As a Python Extension

Build as a Python module and use from Python:

```bash
# Build the extension
cd examples
maturin develop --manifest-path Cargo.toml -E basic_usage

# Run the Python demo
python3 demo.py
```

Or use interactively:

```python
import numpy as np
import basic_usage

# Create a NumPy array
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

# Inspect it in Rust
basic_usage.inspect_tensor(arr)

# Sum all elements
total = basic_usage.sum_tensor(arr)
print(f"Sum: {total}")

# Double all values
doubled = basic_usage.double_tensor(arr)
result = np.from_dlpack(doubled)
print(result)

# Create tensors in Rust
rust_tensor = basic_usage.create_tensor()
arr = np.from_dlpack(rust_tensor)
print(arr)
```

### Option 3: With PyTorch

If you have PyTorch installed:

```python
import torch
import basic_usage

# Works with PyTorch tensors too!
tensor = torch.randn(3, 4)
basic_usage.inspect_tensor(tensor)

total = basic_usage.sum_tensor(tensor)

# Create tensor in Rust, convert to PyTorch
rust_capsule = basic_usage.create_identity(5)
identity = torch.from_dlpack(rust_capsule)
print(identity)
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

## Key Features Demonstrated

✅ **Zero-copy** data sharing
✅ **CPU** and **GPU** tensors
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
