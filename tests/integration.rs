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
        assert_eq!(
            unsafe { name.unwrap().as_cstr() }.to_bytes(),
            b"dltensor"
        );

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
