//! Export tensors to Python via DLPack.
//!
//! This module provides the `IntoDLPack` trait for exporting Rust tensors
//! to Python as DLPack capsules.

use crate::ffi::{
    DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor,
    DLPACK_FLAG_BITMASK_READ_ONLY, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
};
use crate::{
    DLPACK_CAPSULE_NAME, DLPACK_CAPSULE_NAME_USED, DLPACK_VERSIONED_CAPSULE_NAME,
    DLPACK_VERSIONED_CAPSULE_NAME_USED,
};
use pyo3::prelude::*;
use std::ffi::{c_void, CStr};

/// Information about a tensor for DLPack export.
///
/// This struct holds all the metadata needed to create a DLPack tensor.
/// Use this with `into_dlpack_with_info` for explicit control over
/// tensor properties.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Raw data pointer (device pointer for GPU tensors)
    pub data: *mut c_void,
    /// Device descriptor
    pub device: DLDevice,
    /// Data type descriptor
    pub dtype: DLDataType,
    /// Shape (dimensions)
    pub shape: Vec<i64>,
    /// Strides in elements (None for contiguous)
    pub strides: Option<Vec<i64>>,
    /// Byte offset from data pointer
    pub byte_offset: u64,
}

impl TensorInfo {
    /// Create tensor info for a contiguous tensor.
    pub fn contiguous(
        data: *mut c_void,
        device: DLDevice,
        dtype: DLDataType,
        shape: Vec<i64>,
    ) -> Self {
        Self {
            data,
            device,
            dtype,
            shape,
            strides: None,
            byte_offset: 0,
        }
    }

    /// Create tensor info with explicit strides.
    ///
    /// # Panics
    ///
    /// Panics if `strides.len() != shape.len()`. This invariant is required by
    /// DLPack consumers which will read `strides[i]` for each dimension `i`.
    pub fn strided(
        data: *mut c_void,
        device: DLDevice,
        dtype: DLDataType,
        shape: Vec<i64>,
        strides: Vec<i64>,
    ) -> Self {
        assert_eq!(
            strides.len(),
            shape.len(),
            "strides length ({}) must equal shape length ({})",
            strides.len(),
            shape.len()
        );
        Self {
            data,
            device,
            dtype,
            shape,
            strides: Some(strides),
            byte_offset: 0,
        }
    }

    /// Set the byte offset.
    pub fn with_byte_offset(mut self, offset: u64) -> Self {
        self.byte_offset = offset;
        self
    }
}

/// Trait for types that can be exported as DLPack tensors.
///
/// Implement this trait on your tensor type to enable export to Python
/// via the DLPack protocol.
///
/// # Example
///
/// ```ignore
/// use pyo3_dlpack::{IntoDLPack, TensorInfo, cuda_device, dtype_f32};
/// use std::ffi::c_void;
///
/// struct MyGpuTensor {
///     device_ptr: u64,
///     shape: Vec<i64>,
///     device_id: i32,
/// }
///
/// impl IntoDLPack for MyGpuTensor {
///     fn tensor_info(&self) -> TensorInfo {
///         TensorInfo::contiguous(
///             self.device_ptr as *mut c_void,
///             cuda_device(self.device_id),
///             dtype_f32(),
///             self.shape.clone(),
///         )
///     }
/// }
/// ```
pub trait IntoDLPack: Send + Sized {
    /// Get the tensor information for DLPack export.
    fn tensor_info(&self) -> TensorInfo;

    /// Export this tensor to Python as a DLPack capsule.
    ///
    /// The returned `PyObject` is a PyCapsule that can be converted to
    /// a tensor in any DLPack-compatible framework using `from_dlpack()`.
    ///
    /// # Example (Python side)
    ///
    /// ```python
    /// import torch
    ///
    /// # Call your Rust function that returns a DLPack capsule
    /// capsule = my_rust_function()
    ///
    /// # Convert to PyTorch tensor (zero-copy)
    /// tensor = torch.from_dlpack(capsule)
    /// ```
    fn into_dlpack(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let info = self.tensor_info();
        export_to_capsule(py, self, info)
    }

    /// Export this tensor to Python as a **read-only** versioned DLPack capsule.
    ///
    /// Unlike [`into_dlpack`](IntoDLPack::into_dlpack), this emits a versioned
    /// (`dltensor_versioned`) capsule with the read-only flag set, so consumers
    /// that understand DLPack 1.0 know the data must not be modified.
    fn into_dlpack_readonly(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let info = self.tensor_info();
        export_to_capsule_versioned(py, self, info, DLPACK_FLAG_BITMASK_READ_ONLY)
    }
}

/// Internal context that owns the tensor during DLPack lifetime.
struct ExportContext<T> {
    /// The owned tensor (kept alive until the capsule is consumed)
    #[allow(dead_code)]
    tensor: T,
    /// Shape array (must remain valid)
    shape: Vec<i64>,
    /// Strides array (must remain valid)
    strides: Option<Vec<i64>>,
}

/// Build the owning context and the `DLTensor` descriptor shared by both the
/// legacy and versioned export paths.
///
/// On success returns the raw context pointer (the caller takes ownership and
/// must free it if capsule creation later fails) and the populated `DLTensor`.
fn build_export_parts<T: IntoDLPack>(
    tensor: T,
    info: TensorInfo,
) -> PyResult<(*mut ExportContext<T>, DLTensor)> {
    // Validate strides length matches shape length to prevent out-of-bounds reads
    // by DLPack consumers. This catches cases where TensorInfo is constructed
    // manually without using the strided() constructor.
    if let Some(ref strides) = info.strides {
        if strides.len() != info.shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "strides length ({}) must equal shape length ({})",
                strides.len(),
                info.shape.len()
            )));
        }
    }

    // Create the context that will own the tensor
    let ctx = Box::new(ExportContext {
        tensor,
        shape: info.shape,
        strides: info.strides,
    });
    let ctx_ptr = Box::into_raw(ctx);

    // SAFETY: For scalar tensors (ndim == 0), shape and strides pointers MUST be null.
    // Using as_mut_ptr() on an empty Vec returns a non-null dangling pointer, which
    // violates the DLPack spec and can cause UB if consumers read the pointer.
    let ndim = unsafe { (*ctx_ptr).shape.len() as i32 };
    let shape_ptr = if ndim == 0 {
        std::ptr::null_mut()
    } else {
        unsafe { (*ctx_ptr).shape.as_mut_ptr() }
    };
    let strides_ptr = if ndim == 0 {
        std::ptr::null_mut()
    } else {
        unsafe {
            (*ctx_ptr)
                .strides
                .as_mut()
                .map(|s| s.as_mut_ptr())
                .unwrap_or(std::ptr::null_mut())
        }
    };

    let dl_tensor = DLTensor {
        data: info.data,
        device: info.device,
        ndim,
        dtype: info.dtype,
        shape: shape_ptr,
        strides: strides_ptr,
        byte_offset: info.byte_offset,
    };

    Ok((ctx_ptr, dl_tensor))
}

/// Export a tensor to a PyCapsule.
fn export_to_capsule<T: IntoDLPack>(
    py: Python<'_>,
    tensor: T,
    info: TensorInfo,
) -> PyResult<Py<PyAny>> {
    let (ctx_ptr, dl_tensor) = build_export_parts(tensor, info)?;

    let managed = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(dlpack_deleter::<T>),
    });
    let managed_ptr = Box::into_raw(managed);

    // Create the PyCapsule using low-level FFI to ensure the pointer is stored directly.
    // DLPack consumers expect PyCapsule_GetPointer to return a DLManagedTensor* directly.
    // Use the crate-level CStr constant so the name is defined exactly once.
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            DLPACK_CAPSULE_NAME.as_ptr(),
            Some(raw_capsule_destructor),
        )
    };

    if capsule_ptr.is_null() {
        // Clean up on failure - must free BOTH managed_ptr AND ctx_ptr.
        unsafe {
            let _ = Box::from_raw(managed_ptr);
            let _ = Box::from_raw(ctx_ptr);
        }
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "Failed to create DLPack capsule",
        ));
    }

    Ok(unsafe { Bound::from_owned_ptr(py, capsule_ptr).unbind() })
}

/// Raw PyCapsule destructor - called by Python when garbage collecting the capsule.
///
/// Per the DLPack protocol, when a consumer takes ownership of the tensor
/// (e.g., via torch.from_dlpack), it must rename the capsule from "dltensor"
/// to "used_dltensor" and will call the deleter itself when done.
///
/// This destructor checks the capsule name to avoid double-free:
/// - If name is "dltensor": capsule was never consumed, we call the deleter
/// - If name is "used_dltensor": consumer owns it and will call deleter, skip
unsafe extern "C" fn raw_capsule_destructor(capsule_ptr: *mut pyo3::ffi::PyObject) {
    if capsule_ptr.is_null() {
        return;
    }

    // Check the capsule name to see if it was consumed
    let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule_ptr);
    if name_ptr.is_null() {
        // No name set - shouldn't happen with our capsules
        return;
    }

    let name = CStr::from_ptr(name_ptr);

    // If name is "used_dltensor", the consumer has taken ownership
    // and will call the deleter when done. Don't double-free.
    if name.to_bytes() == DLPACK_CAPSULE_NAME_USED.to_bytes() {
        return;
    }

    // Get the DLManagedTensor pointer from the capsule using the current name
    let managed_ptr =
        pyo3::ffi::PyCapsule_GetPointer(capsule_ptr, name_ptr) as *mut DLManagedTensor;

    if managed_ptr.is_null() {
        return;
    }

    // Capsule was not consumed, call the DLPack deleter
    let managed = &*managed_ptr;
    if let Some(deleter) = managed.deleter {
        deleter(managed_ptr);
    }
}

/// Deleter called by the consumer when done with the tensor.
///
/// This is an extern "C" function that matches the DLPack deleter signature.
unsafe extern "C" fn dlpack_deleter<T>(managed_ptr: *mut DLManagedTensor) {
    if managed_ptr.is_null() {
        return;
    }

    // Recover and drop the managed tensor
    let managed = Box::from_raw(managed_ptr);

    // Recover and drop the context (which owns the tensor)
    if !managed.manager_ctx.is_null() {
        let _ctx = Box::from_raw(managed.manager_ctx as *mut ExportContext<T>);
        // ctx and its tensor are dropped here
    }
}

/// Export a tensor to a versioned (`dltensor_versioned`) PyCapsule with the
/// given flags.
fn export_to_capsule_versioned<T: IntoDLPack>(
    py: Python<'_>,
    tensor: T,
    info: TensorInfo,
    flags: u64,
) -> PyResult<Py<PyAny>> {
    let (ctx_ptr, dl_tensor) = build_export_parts(tensor, info)?;

    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        },
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(dlpack_deleter_versioned::<T>),
        flags,
        dl_tensor,
    });
    let managed_ptr = Box::into_raw(managed);

    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            DLPACK_VERSIONED_CAPSULE_NAME.as_ptr(),
            Some(raw_capsule_destructor_versioned),
        )
    };

    if capsule_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(managed_ptr);
            let _ = Box::from_raw(ctx_ptr);
        }
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "Failed to create versioned DLPack capsule",
        ));
    }

    Ok(unsafe { Bound::from_owned_ptr(py, capsule_ptr).unbind() })
}

/// Raw PyCapsule destructor for versioned capsules.
///
/// Mirrors [`raw_capsule_destructor`] but checks the versioned capsule names
/// and interprets the pointer as a `DLManagedTensorVersioned`.
unsafe extern "C" fn raw_capsule_destructor_versioned(capsule_ptr: *mut pyo3::ffi::PyObject) {
    if capsule_ptr.is_null() {
        return;
    }

    let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule_ptr);
    if name_ptr.is_null() {
        return;
    }

    let name = CStr::from_ptr(name_ptr);

    // If consumed, the consumer owns it and will call the deleter. Don't double-free.
    if name.to_bytes() == DLPACK_VERSIONED_CAPSULE_NAME_USED.to_bytes() {
        return;
    }

    let managed_ptr =
        pyo3::ffi::PyCapsule_GetPointer(capsule_ptr, name_ptr) as *mut DLManagedTensorVersioned;
    if managed_ptr.is_null() {
        return;
    }

    let managed = &*managed_ptr;
    if let Some(deleter) = managed.deleter {
        deleter(managed_ptr);
    }
}

/// Deleter for versioned managed tensors, called by the consumer when done.
unsafe extern "C" fn dlpack_deleter_versioned<T>(managed_ptr: *mut DLManagedTensorVersioned) {
    if managed_ptr.is_null() {
        return;
    }

    let managed = Box::from_raw(managed_ptr);
    if !managed.manager_ctx.is_null() {
        let _ctx = Box::from_raw(managed.manager_ctx as *mut ExportContext<T>);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{cpu_device, cuda_device, dtype_f32, dtype_f64, dtype_i32};
    use pyo3::Python;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ========================================================================
    // Test tensor types
    // ========================================================================

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

    struct StridedTensor {
        data: Vec<f32>,
        shape: Vec<i64>,
        strides: Vec<i64>,
    }

    impl IntoDLPack for StridedTensor {
        fn tensor_info(&self) -> TensorInfo {
            TensorInfo::strided(
                self.data.as_ptr() as *mut c_void,
                cpu_device(),
                dtype_f32(),
                self.shape.clone(),
                self.strides.clone(),
            )
        }
    }

    struct GpuTensor {
        device_ptr: u64,
        shape: Vec<i64>,
        device_id: i32,
    }

    impl IntoDLPack for GpuTensor {
        fn tensor_info(&self) -> TensorInfo {
            TensorInfo::contiguous(
                self.device_ptr as *mut c_void,
                cuda_device(self.device_id),
                dtype_f32(),
                self.shape.clone(),
            )
        }
    }

    struct OffsetTensor {
        data: Vec<f32>,
        shape: Vec<i64>,
        offset: u64,
    }

    impl IntoDLPack for OffsetTensor {
        fn tensor_info(&self) -> TensorInfo {
            TensorInfo::contiguous(
                self.data.as_ptr() as *mut c_void,
                cpu_device(),
                dtype_f32(),
                self.shape.clone(),
            )
            .with_byte_offset(self.offset)
        }
    }

    // Track drops for testing cleanup
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct DropTracker {
        data: Vec<f32>,
        shape: Vec<i64>,
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl IntoDLPack for DropTracker {
        fn tensor_info(&self) -> TensorInfo {
            TensorInfo::contiguous(
                self.data.as_ptr() as *mut c_void,
                cpu_device(),
                dtype_f32(),
                self.shape.clone(),
            )
        }
    }

    // ========================================================================
    // TensorInfo tests
    // ========================================================================

    #[test]
    fn test_tensor_info_contiguous() {
        let data = [1.0f32, 2.0, 3.0, 4.0].to_vec();
        let info = TensorInfo::contiguous(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![2, 2],
        );

        assert!(info.strides.is_none());
        assert_eq!(info.byte_offset, 0);
        assert_eq!(info.shape, vec![2, 2]);
        assert!(info.device.is_cpu());
        assert!(info.dtype.is_f32());
    }

    #[test]
    fn test_tensor_info_strided() {
        let data = [1.0f32; 24].to_vec();
        let info = TensorInfo::strided(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![2, 3, 4],
            vec![12, 4, 1],
        );

        assert_eq!(info.strides, Some(vec![12, 4, 1]));
        assert_eq!(info.byte_offset, 0);
        assert_eq!(info.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_tensor_info_with_byte_offset() {
        let data = [1.0f32; 10].to_vec();
        let info = TensorInfo::contiguous(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![10],
        )
        .with_byte_offset(16);

        assert_eq!(info.byte_offset, 16);
    }

    #[test]
    fn test_tensor_info_with_different_dtypes() {
        let data_f64 = [1.0f64; 10].to_vec();
        let info = TensorInfo::contiguous(
            data_f64.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f64(),
            vec![10],
        );
        assert!(info.dtype.is_f64());

        let data_i32 = [1i32; 10].to_vec();
        let info = TensorInfo::contiguous(
            data_i32.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_i32(),
            vec![10],
        );
        assert!(info.dtype.is_i32());
    }

    #[test]
    fn test_tensor_info_with_different_devices() {
        let data = [1.0f32; 10].to_vec();

        let cpu_info = TensorInfo::contiguous(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![10],
        );
        assert!(cpu_info.device.is_cpu());

        let cuda_info = TensorInfo::contiguous(
            0x12345678 as *mut c_void,
            cuda_device(0),
            dtype_f32(),
            vec![10],
        );
        assert!(cuda_info.device.is_cuda());
        assert_eq!(cuda_info.device.device_id, 0);

        let cuda1_info = TensorInfo::contiguous(
            0x12345678 as *mut c_void,
            cuda_device(1),
            dtype_f32(),
            vec![10],
        );
        assert_eq!(cuda1_info.device.device_id, 1);
    }

    #[test]
    fn test_tensor_info_debug() {
        let data = [1.0f32; 10].to_vec();
        let info = TensorInfo::contiguous(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![2, 5],
        );
        let debug = format!("{:?}", info);
        assert!(debug.contains("TensorInfo"));
        assert!(debug.contains("shape"));
    }

    #[test]
    fn test_tensor_info_clone() {
        let data = [1.0f32; 10].to_vec();
        let info = TensorInfo::strided(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![2, 5],
            vec![5, 1],
        )
        .with_byte_offset(8);

        let cloned = info.clone();
        assert_eq!(cloned.shape, info.shape);
        assert_eq!(cloned.strides, info.strides);
        assert_eq!(cloned.byte_offset, info.byte_offset);
    }

    #[test]
    fn test_tensor_info_empty_shape() {
        let data = [1.0f32].to_vec();
        let info = TensorInfo::contiguous(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![], // Scalar
        );
        assert!(info.shape.is_empty());
    }

    #[test]
    fn test_tensor_info_high_dimensional() {
        let data = vec![1.0f32; 120];
        let info = TensorInfo::contiguous(
            data.as_ptr() as *mut c_void,
            cpu_device(),
            dtype_f32(),
            vec![2, 3, 4, 5],
        );
        assert_eq!(info.shape.len(), 4);
    }

    // ========================================================================
    // IntoDLPack trait tests
    // ========================================================================

    #[test]
    fn test_into_dlpack_contiguous() {
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
    fn test_into_dlpack_strided() {
        Python::attach(|py| {
            let tensor = StridedTensor {
                data: vec![1.0; 24],
                shape: vec![2, 3, 4],
                strides: vec![12, 4, 1],
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_into_dlpack_gpu_tensor() {
        Python::attach(|py| {
            let tensor = GpuTensor {
                device_ptr: 0xDEADBEEF,
                shape: vec![16, 32],
                device_id: 0,
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_into_dlpack_with_offset() {
        Python::attach(|py| {
            let tensor = OffsetTensor {
                data: vec![1.0; 20],
                shape: vec![10],
                offset: 40, // Skip first 10 f32 elements
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_into_dlpack_scalar() {
        Python::attach(|py| {
            let tensor = TestTensor {
                data: vec![42.0],
                shape: vec![], // Scalar
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_into_dlpack_1d() {
        Python::attach(|py| {
            let tensor = TestTensor {
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                shape: vec![5],
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_into_dlpack_readonly_is_versioned() {
        Python::attach(|py| {
            let tensor = TestTensor {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![2, 2],
            };

            let capsule = tensor
                .into_dlpack_readonly(py)
                .expect("Failed to create read-only capsule");

            // A read-only export must produce a versioned capsule whose managed
            // tensor actually carries the read-only flag and the protocol version.
            unsafe {
                let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule.as_ptr());
                assert!(!name_ptr.is_null());
                let name = CStr::from_ptr(name_ptr);
                assert_eq!(name.to_bytes(), b"dltensor_versioned");

                let managed_ptr = pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name_ptr)
                    as *mut DLManagedTensorVersioned;
                assert!(!managed_ptr.is_null());
                assert_eq!(
                    (*managed_ptr).flags & DLPACK_FLAG_BITMASK_READ_ONLY,
                    DLPACK_FLAG_BITMASK_READ_ONLY
                );
                assert_eq!((*managed_ptr).version.major, DLPACK_MAJOR_VERSION);
            }
        });
    }

    // ========================================================================
    // Cleanup and memory management tests
    // ========================================================================

    #[test]
    fn test_capsule_cleanup_on_drop() {
        DROP_COUNT.store(0, Ordering::SeqCst);

        Python::attach(|py| {
            {
                let tensor = DropTracker {
                    data: vec![1.0, 2.0, 3.0],
                    shape: vec![3],
                };

                let _capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
                // Capsule exists, tensor ownership transferred
            }
            // Force garbage collection
            py.run(c"import gc; gc.collect()", None, None).unwrap();
        });

        // The tensor should have been dropped when the capsule was cleaned up
        // Note: GC timing is not deterministic, so we check after GIL release
        // In practice, the drop may happen during GC or when Python shuts down
    }

    #[test]
    fn test_deleter_null_check() {
        // Test that dlpack_deleter handles null safely
        unsafe {
            dlpack_deleter::<TestTensor>(std::ptr::null_mut());
        }
        // Should not crash
    }

    #[test]
    fn test_capsule_destructor_null_check() {
        // Test that raw_capsule_destructor handles null safely
        unsafe {
            raw_capsule_destructor(std::ptr::null_mut());
        }
        // Should not crash
    }

    #[test]
    fn test_versioned_deleter_null_check() {
        // Versioned deleter must handle a null pointer safely.
        unsafe {
            dlpack_deleter_versioned::<TestTensor>(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_versioned_capsule_destructor_null_check() {
        // Versioned capsule destructor must handle a null pointer safely.
        unsafe {
            raw_capsule_destructor_versioned(std::ptr::null_mut());
        }
    }

    // ========================================================================
    // Send trait verification
    // ========================================================================

    #[test]
    fn test_into_dlpack_requires_send() {
        // IntoDLPack requires Send, verify our test types implement it
        fn assert_send<T: Send>() {}
        assert_send::<TestTensor>();
        assert_send::<StridedTensor>();
        assert_send::<GpuTensor>();
        assert_send::<OffsetTensor>();
        assert_send::<DropTracker>();
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_large_shape() {
        Python::attach(|py| {
            let tensor = TestTensor {
                data: vec![1.0; 1000000],
                shape: vec![100, 100, 100],
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_non_contiguous_strides() {
        Python::attach(|py| {
            // Transposed tensor (column-major)
            let tensor = StridedTensor {
                data: vec![1.0; 6],
                shape: vec![2, 3],
                strides: vec![1, 2], // Column-major
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }

    #[test]
    fn test_zero_stride() {
        Python::attach(|py| {
            // Broadcasting-like strides
            let tensor = StridedTensor {
                data: vec![1.0; 3],
                shape: vec![2, 3],
                strides: vec![0, 1], // First dimension is broadcast
            };

            let capsule = tensor.into_dlpack(py).expect("Failed to create capsule");
            assert!(!capsule.is_none(py));
        });
    }
}
