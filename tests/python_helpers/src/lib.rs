//! Test module for DLPack integration tests.
//!
//! This module provides Python bindings for testing the DLPack capsule
//! ownership protocol on both CPU and GPU paths.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_dlpack::{cpu_device, cuda_device, dtype_f32, IntoDLPack, PyTensor, TensorInfo};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Drop tracking for memory safety tests
// ============================================================================

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
static DELETER_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Reset all counters
#[pyfunction]
fn reset_counters() {
    DROP_COUNT.store(0, Ordering::SeqCst);
    DELETER_CALL_COUNT.store(0, Ordering::SeqCst);
}

/// Get the current drop count
#[pyfunction]
fn get_drop_count() -> usize {
    DROP_COUNT.load(Ordering::SeqCst)
}

/// Get the current deleter call count
#[pyfunction]
fn get_deleter_count() -> usize {
    DELETER_CALL_COUNT.load(Ordering::SeqCst)
}

// ============================================================================
// Test tensor types for export (Rust -> Python)
// ============================================================================

/// A CPU tensor that tracks drops
struct TrackedCpuTensor {
    data: Vec<f32>,
    shape: Vec<i64>,
}

impl Drop for TrackedCpuTensor {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        DELETER_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

impl IntoDLPack for TrackedCpuTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            self.shape.clone(),
        )
    }
}

/// A GPU tensor (simulated with a fake device pointer) that tracks drops
struct TrackedGpuTensor {
    /// Simulated device pointer (in real usage this would be a CUDA pointer)
    device_ptr: u64,
    shape: Vec<i64>,
    device_id: i32,
    /// Keep data alive for testing
    #[allow(dead_code)]
    backing_data: Vec<f32>,
}

impl Drop for TrackedGpuTensor {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        DELETER_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

impl IntoDLPack for TrackedGpuTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.device_ptr as *mut c_void,
            cuda_device(self.device_id),
            dtype_f32(),
            self.shape.clone(),
        )
    }
}

// ============================================================================
// Export functions (Rust -> Python DLPack capsule)
// ============================================================================

/// Create a CPU tensor and export it as a DLPack capsule.
/// The tensor data is [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] with shape [2, 3].
#[pyfunction]
fn export_cpu_tensor(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = TrackedCpuTensor {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape: vec![2, 3],
    };
    tensor.into_dlpack(py)
}

/// Create a GPU tensor (simulated) and export it as a DLPack capsule.
/// Uses the specified device_id (default 0).
#[pyfunction]
#[pyo3(signature = (device_id=0))]
fn export_gpu_tensor(py: Python<'_>, device_id: i32) -> PyResult<Py<PyAny>> {
    let backing_data = vec![1.0f32; 256];
    let tensor = TrackedGpuTensor {
        device_ptr: backing_data.as_ptr() as u64,
        shape: vec![16, 16],
        device_id,
        backing_data,
    };
    tensor.into_dlpack(py)
}

/// Create a larger CPU tensor for stress testing.
#[pyfunction]
#[pyo3(signature = (size=1000000))]
fn export_large_cpu_tensor(py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
    let tensor = TrackedCpuTensor {
        data: vec![1.0; size],
        shape: vec![size as i64],
    };
    tensor.into_dlpack(py)
}

// ============================================================================
// Import functions (Python DLPack capsule -> Rust)
// ============================================================================

/// Import a tensor from a Python object via DLPack protocol.
/// Returns a dict with tensor metadata.
#[pyfunction]
fn import_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let tensor = PyTensor::from_pyany(py, obj)?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("shape", tensor.shape().to_vec())?;
    dict.set_item("ndim", tensor.ndim())?;
    dict.set_item("numel", tensor.numel())?;
    dict.set_item("itemsize", tensor.itemsize())?;
    dict.set_item("nbytes", tensor.nbytes())?;
    dict.set_item("is_contiguous", tensor.is_contiguous())?;
    dict.set_item("is_cpu", tensor.device().is_cpu())?;
    dict.set_item("is_cuda", tensor.device().is_cuda())?;
    dict.set_item("device_id", tensor.device().device_id)?;
    dict.set_item("byte_offset", tensor.byte_offset())?;

    // Drop the tensor, which should call the deleter
    drop(tensor);

    Ok(dict.into())
}

/// Import a tensor from a DLPack capsule directly.
#[pyfunction]
fn import_from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Py<PyAny>> {
    let py = capsule.py();
    let tensor = PyTensor::from_capsule(capsule)?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("shape", tensor.shape().to_vec())?;
    dict.set_item("ndim", tensor.ndim())?;
    dict.set_item("is_cpu", tensor.device().is_cpu())?;
    dict.set_item("is_cuda", tensor.device().is_cuda())?;

    Ok(dict.into())
}

/// Try to import a tensor twice from the same object.
/// This should fail on the second import due to the "used_dltensor" rename.
#[pyfunction]
fn try_double_import(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(bool, bool)> {
    // First import should succeed
    let first_result = PyTensor::from_pyany(py, obj);
    let first_ok = first_result.is_ok();

    // Keep the first tensor alive while trying second import
    let _first_tensor = first_result.ok();

    // Second import should fail (capsule already consumed)
    let second_result = PyTensor::from_pyany(py, obj);
    let second_ok = second_result.is_ok();

    Ok((first_ok, second_ok))
}

// ============================================================================
// Capsule state inspection (for debugging)
// ============================================================================

/// Check if a capsule has been consumed (name changed to "used_dltensor").
#[pyfunction]
fn is_capsule_consumed(capsule: &Bound<'_, PyCapsule>) -> PyResult<bool> {
    let name = capsule.name()?;
    match name {
        Some(n) => {
            // Safety: We use the CStr immediately and don't store it
            let cstr = unsafe { n.as_cstr() };
            Ok(cstr.to_bytes() == b"used_dltensor")
        }
        None => Ok(false),
    }
}

/// Get the capsule name for inspection.
#[pyfunction]
fn get_capsule_name(capsule: &Bound<'_, PyCapsule>) -> PyResult<Option<String>> {
    let name = capsule.name()?;
    match name {
        Some(n) => {
            // Safety: We use the CStr immediately and don't store it
            let cstr = unsafe { n.as_cstr() };
            Ok(Some(String::from_utf8_lossy(cstr.to_bytes()).to_string()))
        }
        None => Ok(None),
    }
}

// ============================================================================
// Python module definition
// ============================================================================

#[pymodule]
fn dlpack_test_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Counter functions
    m.add_function(wrap_pyfunction!(reset_counters, m)?)?;
    m.add_function(wrap_pyfunction!(get_drop_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_deleter_count, m)?)?;

    // Export functions (Rust -> Python)
    m.add_function(wrap_pyfunction!(export_cpu_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(export_gpu_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(export_large_cpu_tensor, m)?)?;

    // Import functions (Python -> Rust)
    m.add_function(wrap_pyfunction!(import_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(import_from_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(try_double_import, m)?)?;

    // Capsule inspection
    m.add_function(wrap_pyfunction!(is_capsule_consumed, m)?)?;
    m.add_function(wrap_pyfunction!(get_capsule_name, m)?)?;

    Ok(())
}
