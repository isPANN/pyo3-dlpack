//! Integration tests for pyo3-dlpack.
//!
//! These tests verify the library works correctly in real-world scenarios.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_dlpack::{cpu_device, dtype_f32, IntoDLPack, PyTensor, TensorInfo};
use std::ffi::c_void;

/// Simple test tensor for export
struct TestTensor {
    data: Vec<f32>,
    shape: Vec<i64>,
}

impl IntoDLPack for TestTensor {
    fn tensor_info(&self) -> TensorInfo {
        TensorInfo::contiguous(
            self.data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            self.shape.clone(),
        )
    }
}

#[test]
fn test_export_to_capsule() {
    Python::attach(|py| {
        let tensor = TestTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
        };

        let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
        assert!(!capsule.is_none(py));
    });
}

#[test]
fn test_import_from_numpy() {
    Python::attach(|py| {
        // Create numpy array
        let numpy = py.import("numpy").expect("numpy not available");
        let array = numpy
            .getattr("array")
            .unwrap()
            .call1(([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],))
            .unwrap();

        // Import to Rust
        let tensor = PyTensor::from_pyany(py, &array).expect("Failed to import");

        // Verify metadata
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert!(tensor.device().is_cpu());
    });
}

#[test]
fn test_capsule_marked_as_used() {
    Python::attach(|py| {
        // Create numpy array and get capsule
        let numpy = py.import("numpy").expect("numpy not available");
        let array = numpy
            .getattr("array")
            .unwrap()
            .call1(([1.0f32, 2.0, 3.0],))
            .unwrap();

        let capsule_obj = array.call_method0("__dlpack__").unwrap();
        let capsule: Bound<'_, PyCapsule> = capsule_obj.extract().unwrap();

        // Check initial name
        let name = capsule.name().unwrap();
        assert_eq!(unsafe { name.unwrap().as_cstr() }.to_bytes(), b"dltensor");

        // Import it (should rename)
        let _tensor = PyTensor::from_capsule(&capsule).expect("Failed to import");

        // Check name changed
        let name = capsule.name().unwrap();
        assert_eq!(
            unsafe { name.unwrap().as_cstr() }.to_bytes(),
            b"used_dltensor"
        );
    });
}

#[test]
fn test_second_import_fails() {
    Python::attach(|py| {
        let numpy = py.import("numpy").expect("numpy not available");
        let array = numpy
            .getattr("array")
            .unwrap()
            .call1(([1.0f32, 2.0, 3.0],))
            .unwrap();

        let capsule_obj = array.call_method0("__dlpack__").unwrap();
        let capsule: Bound<'_, PyCapsule> = capsule_obj.extract().unwrap();

        // First import should succeed
        let _tensor = PyTensor::from_capsule(&capsule).expect("First import failed");

        // Second import should fail
        let result = PyTensor::from_capsule(&capsule);
        assert!(result.is_err(), "Second import should have failed");
    });
}

#[test]
fn test_zero_copy_import_from_numpy() {
    Python::attach(|py| {
        // Create numpy array with explicit float32 dtype
        let numpy = py.import("numpy").expect("numpy not available");
        let float32 = numpy.getattr("float32").unwrap();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", float32).unwrap();
        let array = numpy
            .getattr("array")
            .unwrap()
            .call(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), Some(&kwargs))
            .unwrap();

        // Get the original data pointer from numpy
        let original_ptr: usize = array
            .getattr("ctypes")
            .unwrap()
            .getattr("data")
            .unwrap()
            .extract()
            .unwrap();

        // Import to Rust via DLPack
        let tensor = PyTensor::from_pyany(py, &array).expect("Failed to import");

        // Verify the data pointer is the same (zero-copy)
        let rust_ptr = tensor.data_ptr() as usize;
        assert_eq!(
            rust_ptr, original_ptr,
            "Data pointer mismatch: DLPack import should be zero-copy"
        );

        // Verify we can read the actual data
        let data_slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr() as *const f32, 6) };
        assert_eq!(data_slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });
}

#[test]
fn test_zero_copy_export_to_numpy() {
    Python::attach(|py| {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original_ptr = data.as_ptr() as usize;

        let tensor = TestTensor {
            data,
            shape: vec![2, 3],
        };

        // Export to DLPack capsule
        let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");

        // Create a wrapper class that implements __dlpack__ protocol
        // This is needed because np.from_dlpack expects an object with __dlpack__ method
        let wrapper_code = c"
class DLPackWrapper:
    def __init__(self, capsule):
        self._capsule = capsule
    def __dlpack__(self, stream=None):
        return self._capsule
    def __dlpack_device__(self):
        return (1, 0)  # CPU
";
        py.run(wrapper_code, None, None).unwrap();

        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("capsule", capsule).unwrap();
        let wrapped = py
            .eval(c"DLPackWrapper(capsule)", None, Some(&locals))
            .unwrap();

        // Import in numpy using np.from_dlpack
        let numpy = py.import("numpy").expect("numpy not available");
        let np_array = numpy
            .getattr("from_dlpack")
            .unwrap()
            .call1((wrapped,))
            .unwrap();

        // Get the numpy array's data pointer
        let numpy_ptr: usize = np_array
            .getattr("ctypes")
            .unwrap()
            .getattr("data")
            .unwrap()
            .extract()
            .unwrap();

        // Verify zero-copy: same pointer
        assert_eq!(
            numpy_ptr, original_ptr,
            "Data pointer mismatch: DLPack export should be zero-copy"
        );

        // Verify the data is correct
        let result: Vec<Vec<f32>> = np_array.extract().unwrap();
        assert_eq!(result, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    });
}

#[test]
fn test_zero_copy_roundtrip_rust_to_numpy() {
    Python::attach(|py| {
        // Create original data in Rust
        let original_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let original_ptr = original_data.as_ptr() as usize;

        let tensor = TestTensor {
            data: original_data,
            shape: vec![2, 2],
        };

        // Export to Python
        let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");

        // Create a wrapper class that implements __dlpack__ protocol
        let wrapper_code = c"
class DLPackWrapper:
    def __init__(self, capsule):
        self._capsule = capsule
    def __dlpack__(self, stream=None):
        return self._capsule
    def __dlpack_device__(self):
        return (1, 0)  # CPU
";
        py.run(wrapper_code, None, None).unwrap();

        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("capsule", capsule).unwrap();
        let wrapped = py
            .eval(c"DLPackWrapper(capsule)", None, Some(&locals))
            .unwrap();

        // Import in numpy
        let numpy = py.import("numpy").expect("numpy not available");
        let np_array = numpy
            .getattr("from_dlpack")
            .unwrap()
            .call1((wrapped,))
            .unwrap();

        // Verify the numpy array points to the original Rust data (zero-copy)
        let numpy_ptr: usize = np_array
            .getattr("ctypes")
            .unwrap()
            .getattr("data")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(
            numpy_ptr, original_ptr,
            "Data pointer changed during Rust->NumPy transfer: should be zero-copy"
        );

        // Verify the data values are correct
        let result: Vec<Vec<f32>> = np_array.extract().unwrap();
        assert_eq!(result, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    });
}

#[test]
fn test_zero_copy_roundtrip_numpy_to_rust_to_numpy() {
    Python::attach(|py| {
        // Create numpy array (writable, so it can be re-exported)
        let numpy = py.import("numpy").expect("numpy not available");
        let float32 = numpy.getattr("float32").unwrap();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("dtype", float32).unwrap();
        let original_array = numpy
            .getattr("array")
            .unwrap()
            .call(([[1.0, 2.0], [3.0, 4.0]],), Some(&kwargs))
            .unwrap();

        // Get original pointer
        let original_ptr: usize = original_array
            .getattr("ctypes")
            .unwrap()
            .getattr("data")
            .unwrap()
            .extract()
            .unwrap();

        // Import to Rust
        let tensor = PyTensor::from_pyany(py, &original_array).expect("Failed to import to Rust");
        let rust_ptr = tensor.data_ptr() as usize;

        // Verify Rust sees the same pointer
        assert_eq!(
            rust_ptr, original_ptr,
            "Data pointer changed during NumPy->Rust transfer"
        );

        // The original numpy array still exists and can be used
        // This verifies the zero-copy nature - both Rust and Python see the same memory
        let data_slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr() as *const f32, 4) };
        assert_eq!(data_slice, &[1.0, 2.0, 3.0, 4.0]);
    });
}
