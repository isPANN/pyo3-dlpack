# pyo3-dlpack

Zero-copy DLPack tensor interop for PyO3.

This crate provides a safe and ergonomic way to exchange tensor data between
Rust and Python ML frameworks (PyTorch, JAX, TensorFlow, CuPy, etc.) using
the [DLPack](https://github.com/dmlc/dlpack) protocol.

## Features

- **Zero-copy**: Tensors are shared directly without copying data
- **PyO3 0.28+**: Uses the modern API (no deprecation warnings)
- **Bidirectional**: Import tensors from Python and export tensors to Python
- **Device-agnostic**: Works with CPU, CUDA, ROCm, and other devices
- **DLPack 1.0**: Versioned protocol with read-only tensors — auto-negotiated on import, fully backward-compatible with legacy producers
- **Benchmarked**: zero-copy stays O(1)/flat-memory where copy-based interop is O(n)/2×-memory (≈54,000× faster at 100M elements), and at parity with `dlpark` — see [BENCHMARKS.md](BENCHMARKS.md)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pyo3-dlpack = "0.2"
pyo3 = "0.28"
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

    // Respect the producer's read-only flag (DLPack 1.0); legacy producers
    // always report `false`.
    if tensor.is_read_only() {
        // Treat the data as immutable.
    }

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

### Read-only and versioned DLPack

`pyo3-dlpack` speaks both the legacy and the versioned (DLPack 1.0) protocol,
and negotiation is automatic — you do not have to choose.

- **Import** (`PyTensor::from_pyany`) advertises versioned support to the
  producer and transparently accepts either a legacy `dltensor` capsule or a
  versioned `dltensor_versioned` one. Call `tensor.is_read_only()` to check the
  read-only flag (always `false` for legacy producers, which cannot express it).
- **Export** keeps `into_dlpack` unchanged (a writable legacy capsule, for
  maximum consumer compatibility). To export a read-only tensor, use
  `into_dlpack_readonly`, which emits a versioned capsule with the read-only
  flag set:

```rust
#[pyfunction]
fn create_readonly_tensor(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = MyTensor { /* ... */ };
    tensor.into_dlpack_readonly(py)
}
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

DLPack enables true zero-copy tensor sharing: only metadata is processed, never the
data, so cost is constant regardless of tensor size. Copy-based interop is O(n) in
both time and peak memory. Representative results on Apple M3 (see
[BENCHMARKS.md](BENCHMARKS.md) for the full methodology, the `dlpark` head-to-head,
and reproduce commands):

| Operation (1M f32) | pyo3-dlpack (zero-copy) | Copy baseline |
|--------------------|-------------------------|---------------|
| Export Rust → Python | **~3.2 µs** | ~99 µs (`Vec::clone` / `rust-numpy`) |
| Import Python → Rust | **~2.5 µs** | — |

The gap widens with size. Importing zero-copy from Python stays **flat at ~0.5 µs**
from 1M to 100M elements, while `numpy.copy()` grows to **~28 ms** at 100M — roughly
**54,000× faster** — and a zero-copy import adds **0 MiB** of resident memory where a
copy adds the full buffer (≈191 MiB for a 191 MiB array).

Against [`dlpark`](https://github.com/SunDoge/dlpark) (the mature Rust DLPack crate),
raw throughput is at **parity** — both are zero-copy capsule wrappers. See
[BENCHMARKS.md](BENCHMARKS.md) for the per-size head-to-head.

Run the benchmarks yourself (`cargo bench` needs Rust ≥ 1.85 for the `dlpark`
dev-dependency):
- `make bench-rust` — Rust criterion head-to-head (`cargo bench --bench dlpack`)
- `make bench-python` — Python benchmarks; also `python benchmarks/bench_dlpack.py --compare` and `--memory`, and `python benchmarks/interop_probe.py`
- `make bench` — all benchmarks

## Testing

Validate correctness and zero-copy behavior:
- `make test` - Rust unit tests + Python integration tests
- Tests verify data pointers are preserved across transfers, capsule
  ownership (no double-free), and the versioned/read-only round-trip

### Python environment
The test module is built with `maturin` using the same interpreter as tests.
Override it with `PYTHON=/path/to/python` if needed (e.g., a venv).
Default tests include PyTorch (`pip install -e ".[test]"`). For CI or lightweight runs, use `pip install -e ".[test-lite]"`.

## License

Licensed under the MIT license. See [LICENSE](LICENSE) for details.
