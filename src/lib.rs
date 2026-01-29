//! # pyo3-dlpack
//!
//! Zero-copy DLPack tensor interop for PyO3.
//!
//! This crate provides a safe and ergonomic way to exchange tensor data between
//! Rust and Python ML frameworks (PyTorch, JAX, TensorFlow, CuPy, etc.) using
//! the [DLPack](https://github.com/dmlc/dlpack) protocol.
//!
//! ## Features
//!
//! - **Zero-copy**: Tensors are shared directly without copying data
//! - **PyO3 0.23+**: Uses the modern `IntoPyObject` trait (no deprecation warnings)
//! - **Bidirectional**: Import tensors from Python and export tensors to Python
//! - **Device-agnostic**: Works with CPU, CUDA, ROCm, and other devices
//!
//! ## Example: Importing a PyTorch tensor
//!
//! ```ignore
//! use pyo3::prelude::*;
//! use pyo3_dlpack::PyTensor;
//!
//! #[pyfunction]
//! fn process_tensor(tensor: PyTensor) -> PyResult<()> {
//!     // Access tensor metadata
//!     println!("Shape: {:?}", tensor.shape());
//!     println!("Device: {:?}", tensor.device());
//!
//!     // Get the raw data pointer (for GPU tensors, this is a device pointer)
//!     let ptr = tensor.data_ptr();
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Example: Exporting a tensor to Python
//!
//! ```ignore
//! use pyo3::prelude::*;
//! use pyo3_dlpack::{ExportConfig, IntoDLPack};
//!
//! struct MyGpuTensor {
//!     ptr: *mut f32,
//!     shape: Vec<i64>,
//!     device_id: i32,
//! }
//!
//! impl IntoDLPack for MyGpuTensor {
//!     // ... implement the trait
//! }
//!
//! #[pyfunction]
//! fn create_tensor(py: Python<'_>) -> PyResult<PyObject> {
//!     let tensor = MyGpuTensor { /* ... */ };
//!     tensor.into_dlpack(py)
//! }
//! ```

mod export;
mod ffi;
mod managed;

// Re-export public API
pub use export::{IntoDLPack, TensorInfo};
pub use ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};
pub use managed::PyTensor;

// Convenience constructors
pub use ffi::{
    cpu_device, cuda_device, dtype_bf16, dtype_bool, dtype_f16, dtype_f32, dtype_f64, dtype_i16,
    dtype_i32, dtype_i64, dtype_i8, dtype_u16, dtype_u32, dtype_u64, dtype_u8, metal_device,
};

/// The DLPack capsule name for tensor exchange
pub const DLPACK_CAPSULE_NAME: &std::ffi::CStr = c"dltensor";

/// The DLPack capsule name after consumption (to prevent double-free)
pub const DLPACK_CAPSULE_NAME_USED: &std::ffi::CStr = c"used_dltensor";
