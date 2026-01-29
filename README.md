# pyo3-dlpack

Zero-copy DLPack tensor interop for PyO3.

This crate provides a safe and ergonomic way to exchange tensor data between
Rust and Python ML frameworks (PyTorch, JAX, TensorFlow, CuPy, etc.) using
the [DLPack](https://github.com/dmlc/dlpack) protocol.

## Features

- **Zero-copy**: Tensors are shared directly without copying data
- **PyO3 0.27+**: Uses the modern API (no deprecation warnings)
- **Bidirectional**: Import tensors from Python and export tensors to Python
- **Device-agnostic**: Works with CPU, CUDA, ROCm, and other devices

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pyo3-dlpack = "0.1"
pyo3 = "0.27"
```

## Usage

### Importing a tensor from Python

```rust
use pyo3::prelude::*;
use pyo3_dlpack::PyTensor;

#[pyfunction]
fn process_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let tensor = PyTensor::from_pyany(py, obj)?;

    println!("Shape: {:?}", tensor.shape());
    println!("Device: {:?}", tensor.device());
    println!("Dtype: {:?}", tensor.dtype());

    if tensor.device().is_cpu() {
        // Safe to access data on CPU
        let ptr = tensor.data_ptr() as *const f32;
        // ... process the data
    }

    Ok(())
}
```

### Exporting a tensor to Python

```rust
use pyo3::prelude::*;
use pyo3_dlpack::{IntoDLPack, TensorInfo, cuda_device, dtype_f32};
use std::ffi::c_void;

struct MyGpuTensor {
    device_ptr: u64,
    shape: Vec<i64>,
    device_id: i32,
}

impl IntoDLPack for MyGpuTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.device_ptr as *mut c_void,
            cuda_device(self.device_id),
            dtype_f32(),
            self.shape.clone(),
        )
    }
}

#[pyfunction]
fn create_tensor(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = MyGpuTensor {
        device_ptr: 0x12345678, // your actual device pointer
        shape: vec![2, 3],
        device_id: 0,
    };
    tensor.into_dlpack(py)
}
```

Python side:

```python
import torch

# Call your Rust function that returns a DLPack capsule
capsule = create_tensor()

# Convert to PyTorch tensor (zero-copy)
tensor = torch.from_dlpack(capsule)
```

## Supported Data Types

- Float: f16, f32, f64, bf16
- Integer: i8, i16, i32, i64
- Unsigned: u8, u16, u32, u64
- Boolean

## Supported Devices

- CPU
- CUDA
- CUDA Host (pinned memory)
- ROCm
- Metal
- Vulkan
- And more (see `DLDeviceType`)

## Performance

DLPack enables true zero-copy tensor sharing. Benchmark results on Apple M3 Max:

| Operation | Time | vs Copy |
|-----------|------|---------|
| DLPack capsule export (1M f32) | **8.3 µs** | 7.3x faster |
| DLPack capsule import (1M f32) | **7.9 µs** | 7.7x faster |
| Vec clone baseline (1M f32) | 60.9 µs | - |

The DLPack overhead is constant regardless of tensor size - only metadata is processed,
not the actual data. This makes it ideal for large tensors where copying would be expensive.

```
# Rust criterion benchmarks (cargo bench)
export_capsule_1k       time:   [155.44 ns 159.74 ns 166.84 ns]
export_capsule_1m       time:   [7.71 µs 8.26 µs 8.89 µs]
import_capsule_1m       time:   [7.44 µs 7.89 µs 8.41 µs]
vec_clone_1m            time:   [60.45 µs 60.90 µs 61.38 µs]
```

Run benchmarks yourself:
- `make bench-rust` - Rust criterion benchmarks
- `make bench-python` - Python benchmarks
- `make bench` - All benchmarks

## Testing

Validate correctness and zero-copy behavior:
- `make test` - Unit + integration tests (105 tests)
- Tests verify data pointers are preserved across transfers

### Python environment
The test module is built with `maturin` using the same interpreter as tests.
Override it with `PYTHON=/path/to/python` if needed (e.g., a venv).

## License

Licensed under the MIT license. See [LICENSE](LICENSE) for details.
