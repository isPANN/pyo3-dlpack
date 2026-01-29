//! Test module for DLPack integration tests.
//!
//! This module provides Python bindings for testing the DLPack capsule
//! ownership protocol on both CPU and GPU paths.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_dlpack::{cpu_device, cuda_device, dtype_f32, DLDeviceType, IntoDLPack, PyTensor, TensorInfo};
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
// Example/Demo functions
// ============================================================================

/// A simple CPU tensor that owns its data (for demo purposes)
struct SimpleTensor {
    data: Vec<f32>,
    shape: Vec<i64>,
}

impl SimpleTensor {
    fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        Self { data, shape }
    }
}

impl IntoDLPack for SimpleTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            self.shape.clone(),
        )
    }
}

/// Import a tensor from Python and print its metadata
#[pyfunction]
fn inspect_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let tensor = PyTensor::from_pyany(py, obj)?;

    println!("Tensor imported successfully!");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Dimensions: {}", tensor.ndim());
    println!("  Total elements: {}", tensor.numel());
    println!("  Data type size: {} bytes", tensor.itemsize());
    println!("  Total size: {} bytes", tensor.nbytes());
    println!("  Is contiguous: {}", tensor.is_contiguous());

    let device = tensor.device();
    let device_name = match device.device_type_enum() {
        Some(DLDeviceType::Cpu) => "CPU",
        Some(DLDeviceType::Cuda) => "CUDA (NVIDIA GPU)",
        Some(DLDeviceType::CudaHost) => "CUDA Host (Pinned Memory)",
        Some(DLDeviceType::CudaManaged) => "CUDA Managed (Unified Memory)",
        Some(DLDeviceType::Metal) => "Metal (Apple GPU)",
        Some(DLDeviceType::Rocm) => "ROCm (AMD GPU)",
        Some(DLDeviceType::Vulkan) => "Vulkan",
        Some(DLDeviceType::OpenCL) => "OpenCL",
        Some(other) => {
            println!("  Device: {:?} (id: {})", other, device.device_id);
            return Ok(());
        }
        None => "Unknown",
    };
    println!("  Device: {} (id: {})", device_name, device.device_id);
    println!("  Is CPU: {}", device.is_cpu());
    println!("  Is CUDA: {}", device.is_cuda());
    println!("  Is Metal: {}", device.is_metal());
    println!("  Is ROCm: {}", device.is_rocm());

    Ok(())
}

/// Sum all elements in a CPU tensor
#[pyfunction]
fn sum_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<f32> {
    let tensor = PyTensor::from_pyany(py, obj)?;

    if !tensor.device().is_cpu() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only CPU tensors are supported",
        ));
    }

    if !tensor.dtype().is_f32() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only float32 tensors are supported",
        ));
    }

    let ptr = tensor.data_ptr() as *const f32;
    let numel = tensor.numel();

    let sum = unsafe {
        let slice = std::slice::from_raw_parts(ptr, numel);
        slice.iter().sum()
    };

    Ok(sum)
}

/// Create a simple 2x3 tensor and export it to Python
#[pyfunction]
fn create_tensor(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = SimpleTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );
    tensor.into_dlpack(py)
}

/// Create a tensor filled with a specific value
#[pyfunction]
fn create_filled_tensor(py: Python<'_>, value: f32, rows: usize, cols: usize) -> PyResult<Py<PyAny>> {
    let data = vec![value; rows * cols];
    let tensor = SimpleTensor::new(data, vec![rows as i64, cols as i64]);
    tensor.into_dlpack(py)
}

/// Create an identity matrix
#[pyfunction]
fn create_identity(py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
    let mut data = vec![0.0; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    let tensor = SimpleTensor::new(data, vec![size as i64, size as i64]);
    tensor.into_dlpack(py)
}

/// Multiply all elements in a tensor by 2 and return a new tensor
#[pyfunction]
fn double_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let input = PyTensor::from_pyany(py, obj)?;

    if !input.device().is_cpu() || !input.dtype().is_f32() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only CPU float32 tensors are supported",
        ));
    }

    let ptr = input.data_ptr() as *const f32;
    let numel = input.numel();
    let input_data = unsafe { std::slice::from_raw_parts(ptr, numel) };

    let output_data: Vec<f32> = input_data.iter().map(|&x| x * 2.0).collect();
    let output = SimpleTensor::new(output_data, input.shape().to_vec());

    output.into_dlpack(py)
}

/// Check if a tensor is on a GPU device (CUDA, Metal, or ROCm)
#[pyfunction]
fn is_gpu_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let tensor = PyTensor::from_pyany(py, obj)?;
    let device = tensor.device();
    Ok(device.is_cuda() || device.is_metal() || device.is_rocm())
}

/// Get the device type as a string
#[pyfunction]
fn get_device_type(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let tensor = PyTensor::from_pyany(py, obj)?;
    let device = tensor.device();

    let name = match device.device_type_enum() {
        Some(DLDeviceType::Cpu) => "cpu",
        Some(DLDeviceType::Cuda) => "cuda",
        Some(DLDeviceType::CudaHost) => "cuda_host",
        Some(DLDeviceType::CudaManaged) => "cuda_managed",
        Some(DLDeviceType::Metal) => "metal",
        Some(DLDeviceType::Rocm) => "rocm",
        Some(DLDeviceType::Vulkan) => "vulkan",
        Some(DLDeviceType::OpenCL) => "opencl",
        _ => "unknown",
    };

    Ok(format!("{}:{}", name, device.device_id))
}

/// Get the raw data pointer as an integer (useful for GPU interop)
#[pyfunction]
fn get_data_ptr(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    let tensor = PyTensor::from_pyany(py, obj)?;
    Ok(tensor.data_ptr() as usize)
}

/// Verify tensor metadata matches expected values
#[pyfunction]
fn validate_tensor(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    expected_shape: Vec<i64>,
    expected_device_type: Option<&str>,
) -> PyResult<bool> {
    let tensor = PyTensor::from_pyany(py, obj)?;

    if tensor.shape() != expected_shape.as_slice() {
        return Ok(false);
    }

    if let Some(expected) = expected_device_type {
        let device = tensor.device();
        let actual = match device.device_type_enum() {
            Some(DLDeviceType::Cpu) => "cpu",
            Some(DLDeviceType::Cuda) => "cuda",
            Some(DLDeviceType::Metal) => "metal",
            Some(DLDeviceType::Rocm) => "rocm",
            _ => "unknown",
        };
        if actual != expected {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Run a demo showing all features
#[pyfunction]
fn demo(py: Python<'_>) -> PyResult<()> {
    println!("=== pyo3-dlpack Demo ===\n");

    let numpy = py.import("numpy")?;

    println!("Demo 1: Inspecting a NumPy array");
    let np_array = numpy
        .getattr("array")?
        .call1(([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],))?;
    inspect_tensor(py, &np_array)?;

    println!("\nDemo 2: Sum all elements");
    let sum = sum_tensor(py, &np_array)?;
    println!("  Sum: {}", sum);

    println!("\nDemo 3: Creating tensor in Rust");
    let rust_tensor = create_tensor(py)?;
    println!("  Created tensor: {:?}", rust_tensor);

    println!("\nDemo 4: Double all values");
    let doubled = double_tensor(py, &np_array)?;
    println!("  Doubled tensor: {:?}", doubled);

    println!("\nDemo 5: GPU tensor support");
    demo_gpu_tensors(py)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_gpu_tensors(py: Python<'_>) -> PyResult<()> {
    let torch = match py.import("torch") {
        Ok(t) => t,
        Err(_) => {
            println!("  PyTorch not available, skipping GPU demo");
            return Ok(());
        }
    };

    let cuda_available: bool = torch
        .getattr("cuda")?
        .getattr("is_available")?
        .call0()?
        .extract()?;

    if cuda_available {
        println!("  CUDA is available!");
        let cuda_tensor = torch
            .getattr("randn")?
            .call1((vec![2i64, 3],))?
            .call_method1("to", ("cuda:0",))?;
        println!("  Created CUDA tensor:");
        inspect_tensor(py, &cuda_tensor)?;

        let device_str = get_device_type(py, &cuda_tensor)?;
        println!("  Device string: {}", device_str);

        let ptr = get_data_ptr(py, &cuda_tensor)?;
        println!("  CUDA device pointer: 0x{:x}", ptr);
    } else {
        println!("  CUDA not available");
    }

    let mps_available: bool = torch
        .getattr("backends")?
        .getattr("mps")?
        .getattr("is_available")?
        .call0()?
        .extract()?;

    if mps_available {
        println!("\n  Metal (MPS) is available!");
        let mps_tensor = torch
            .getattr("randn")?
            .call1((vec![2i64, 3],))?
            .call_method1("to", ("mps:0",))?;
        println!("  Created MPS tensor:");
        inspect_tensor(py, &mps_tensor)?;

        let device_str = get_device_type(py, &mps_tensor)?;
        println!("  Device string: {}", device_str);

        let ptr = get_data_ptr(py, &mps_tensor)?;
        println!("  Metal device pointer: 0x{:x}", ptr);
    } else {
        println!("  Metal (MPS) not available");
    }

    Ok(())
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

    // Example/Demo functions
    m.add_function(wrap_pyfunction!(inspect_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(sum_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(create_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(create_filled_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(create_identity, m)?)?;
    m.add_function(wrap_pyfunction!(double_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_type, m)?)?;
    m.add_function(wrap_pyfunction!(get_data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(demo, m)?)?;

    Ok(())
}
