//! Basic usage example for pyo3-dlpack
//!
//! This example demonstrates:
//! 1. Importing tensors from Python (NumPy/PyTorch) to Rust
//! 2. Exporting tensors from Rust to Python
//! 3. Processing tensor data in Rust
//!
//! Build and test:
//!   cargo run --example basic_usage
//!
//! Or use with Python:
//!   cargo build --example basic_usage --release
//!   python3 -c "import basic_usage; basic_usage.demo()"

use pyo3::prelude::*;
use pyo3_dlpack::{cpu_device, dtype_f32, IntoDLPack, PyTensor, TensorInfo};
use std::ffi::c_void;

// ============================================================================
// Example 1: Import tensor from Python and process it
// ============================================================================

/// Import a tensor from Python and print its metadata
#[pyfunction]
fn inspect_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    // Import the tensor via DLPack (works with NumPy, PyTorch, etc.)
    let tensor = PyTensor::from_pyany(py, obj)?;

    println!("Tensor imported successfully!");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Dimensions: {}", tensor.ndim());
    println!("  Total elements: {}", tensor.numel());
    println!("  Data type size: {} bytes", tensor.itemsize());
    println!("  Total size: {} bytes", tensor.nbytes());
    println!("  Is contiguous: {}", tensor.is_contiguous());
    println!("  Device: {:?}", tensor.device());
    println!("  Is CPU: {}", tensor.device().is_cpu());
    println!("  Is CUDA: {}", tensor.device().is_cuda());

    Ok(())
}

/// Sum all elements in a CPU tensor
#[pyfunction]
fn sum_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<f32> {
    let tensor = PyTensor::from_pyany(py, obj)?;

    // Verify it's a CPU tensor with f32 dtype
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

    // Access the raw data pointer
    let ptr = tensor.data_ptr() as *const f32;
    let numel = tensor.numel();

    // Sum all elements
    let sum = unsafe {
        let slice = std::slice::from_raw_parts(ptr, numel);
        slice.iter().sum()
    };

    Ok(sum)
}

// ============================================================================
// Example 2: Export tensor from Rust to Python
// ============================================================================

/// A simple CPU tensor that owns its data
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

/// Create a simple 2x3 tensor and export it to Python
#[pyfunction]
fn create_tensor(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = SimpleTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    );

    // Export as DLPack capsule
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

// ============================================================================
// Example 3: Round-trip processing
// ============================================================================

/// Multiply all elements in a tensor by 2 and return a new tensor
#[pyfunction]
fn double_tensor(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import the tensor
    let input = PyTensor::from_pyany(py, obj)?;

    if !input.device().is_cpu() || !input.dtype().is_f32() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only CPU float32 tensors are supported",
        ));
    }

    // Read input data
    let ptr = input.data_ptr() as *const f32;
    let numel = input.numel();
    let input_data = unsafe { std::slice::from_raw_parts(ptr, numel) };

    // Create output with doubled values
    let output_data: Vec<f32> = input_data.iter().map(|&x| x * 2.0).collect();
    let output = SimpleTensor::new(output_data, input.shape().to_vec());

    // Export back to Python
    output.into_dlpack(py)
}

// ============================================================================
// Demo function
// ============================================================================

/// Run a demo showing all features
#[pyfunction]
fn demo(py: Python<'_>) -> PyResult<()> {
    println!("=== pyo3-dlpack Demo ===\n");

    // Import numpy
    let numpy = py.import("numpy")?;

    // Demo 1: Inspect a NumPy array
    println!("Demo 1: Inspecting a NumPy array");
    let np_array = numpy
        .getattr("array")?
        .call1(([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],))?;
    inspect_tensor(py, &np_array)?;

    // Demo 2: Sum elements
    println!("\nDemo 2: Sum all elements");
    let sum = sum_tensor(py, &np_array)?;
    println!("  Sum: {}", sum);

    // Demo 3: Create and export tensor
    println!("\nDemo 3: Creating tensor in Rust");
    let rust_tensor = create_tensor(py)?;
    println!("  Created tensor: {:?}", rust_tensor);

    // Demo 4: Round-trip processing
    println!("\nDemo 4: Double all values");
    let doubled = double_tensor(py, &np_array)?;
    println!("  Doubled tensor: {:?}", doubled);

    println!("\n=== Demo Complete ===");
    Ok(())
}

// ============================================================================
// Python module
// ============================================================================

#[pymodule]
fn basic_usage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inspect_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(sum_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(create_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(create_filled_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(create_identity, m)?)?;
    m.add_function(wrap_pyfunction!(double_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(demo, m)?)?;
    Ok(())
}

// ============================================================================
// Standalone example (for cargo run)
// ============================================================================

#[cfg(not(feature = "extension-module"))]
fn main() {
    let _ = Python::initialize();

    Python::attach(|py| {
        if let Err(e) = demo(py) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    });
}

#[cfg(feature = "extension-module")]
fn main() {
    // Extension module - main not used
    println!("This example is built as a Python extension module.");
    println!("Import it in Python to use:");
    println!("  import basic_usage");
    println!("  basic_usage.demo()");
}
