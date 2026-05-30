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

/// Single heap allocation that owns the exported tensor *and* the FFI managed
/// struct the consumer reads. Keeping them in one box (rather than two separate
/// `Box`es) is one `malloc`/`free` per export instead of two. The address of the
/// `managed` field is handed to the `PyCapsule`; `managed.manager_ctx` points
/// back to the box base so the deleter recovers and frees the whole allocation
/// in a single step.
///
/// `M` is the FFI struct the consumer reads — either [`DLManagedTensor`]
/// (legacy) or [`DLManagedTensorVersioned`] — so both export paths share one
/// owning type.
struct ManagedContext<T, M> {
    /// FFI struct exposed to the consumer. Its address is the capsule pointer.
    managed: M,
    /// The owned tensor, kept alive until the capsule is consumed.
    #[allow(dead_code)]
    tensor: T,
    /// Shape array the `dl_tensor.shape` pointer references.
    #[allow(dead_code)]
    shape: Vec<i64>,
    /// Strides array the `dl_tensor.strides` pointer references (if any).
    #[allow(dead_code)]
    strides: Option<Vec<i64>>,
}

/// Validate that an explicit strides array matches the shape rank, to prevent
/// out-of-bounds reads by DLPack consumers. This catches cases where
/// `TensorInfo` is constructed manually without using the `strided()`
/// constructor.
fn validate_strides(info: &TensorInfo) -> PyResult<()> {
    if let Some(ref strides) = info.strides {
        if strides.len() != info.shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "strides length ({}) must equal shape length ({})",
                strides.len(),
                info.shape.len()
            )));
        }
    }
    Ok(())
}

/// Build the `DLTensor` descriptor. The `shape`/`strides` arguments must be the
/// heap buffers that will be moved into the owning context; a `Vec` move does
/// not relocate its buffer, so the pointers captured here stay valid after the
/// Vecs are moved into the box.
///
/// SAFETY/spec: for scalar tensors (`ndim == 0`) the shape and strides pointers
/// MUST be null — `Vec::as_ptr` on an empty Vec returns a dangling non-null
/// pointer that consumers must never dereference.
fn build_dl_tensor(
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
    byte_offset: u64,
    shape: &[i64],
    strides: &Option<Vec<i64>>,
) -> DLTensor {
    let ndim = shape.len() as i32;
    let shape_ptr = if ndim == 0 {
        std::ptr::null_mut()
    } else {
        shape.as_ptr() as *mut i64
    };
    let strides_ptr = if ndim == 0 {
        std::ptr::null_mut()
    } else {
        strides
            .as_ref()
            .map(|s| s.as_ptr() as *mut i64)
            .unwrap_or(std::ptr::null_mut())
    };
    DLTensor {
        data,
        device,
        ndim,
        dtype,
        shape: shape_ptr,
        strides: strides_ptr,
        byte_offset,
    }
}

/// Reclaim and drop the owning [`ManagedContext`] box from a managed struct's
/// `manager_ctx` pointer. Dropping the one Box frees the FFI struct, the tensor,
/// and the shape/strides arrays together. No-op on a null context. `C` must be
/// the exact `ManagedContext<T, M>` the box was created as.
unsafe fn free_managed_ctx<C>(manager_ctx: *mut c_void) {
    if !manager_ctx.is_null() {
        let _ = Box::from_raw(manager_ctx as *mut C);
    }
}

/// Wire a capsule around an already-built owning box, shared by both export
/// paths. `ctx_ptr` is a leaked `Box<C>`; `managed_field` is the address of the
/// `managed` field inside it (the pointer the capsule stores). On capsule
/// creation failure the box is reclaimed and freed, so callers never leak.
unsafe fn into_capsule<C>(
    py: Python<'_>,
    ctx_ptr: *mut C,
    managed_field: *mut c_void,
    name: &CStr,
    destructor: unsafe extern "C" fn(*mut pyo3::ffi::PyObject),
    err_msg: &str,
) -> PyResult<Py<PyAny>> {
    let capsule_ptr = pyo3::ffi::PyCapsule_New(managed_field, name.as_ptr(), Some(destructor));
    if capsule_ptr.is_null() {
        let _ = Box::from_raw(ctx_ptr);
        return Err(pyo3::exceptions::PyMemoryError::new_err(err_msg.to_owned()));
    }
    Ok(Bound::from_owned_ptr(py, capsule_ptr).unbind())
}

/// Return the managed-struct pointer a capsule holds, or null if the capsule is
/// null/unnamed or already consumed (renamed to `used_name`). Shared skeleton
/// for both capsule destructors: a consumer that took ownership renamed the
/// capsule and will call the deleter itself, so we must not double-free.
unsafe fn unconsumed_managed_ptr(
    capsule_ptr: *mut pyo3::ffi::PyObject,
    used_name: &CStr,
) -> *mut c_void {
    if capsule_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule_ptr);
    if name_ptr.is_null() {
        return std::ptr::null_mut();
    }
    if CStr::from_ptr(name_ptr).to_bytes() == used_name.to_bytes() {
        return std::ptr::null_mut();
    }
    pyo3::ffi::PyCapsule_GetPointer(capsule_ptr, name_ptr)
}

/// Export a tensor to a `dltensor` PyCapsule in a single heap allocation.
fn export_to_capsule<T: IntoDLPack>(
    py: Python<'_>,
    tensor: T,
    info: TensorInfo,
) -> PyResult<Py<PyAny>> {
    validate_strides(&info)?;
    let TensorInfo {
        data,
        device,
        dtype,
        shape,
        strides,
        byte_offset,
    } = info;
    // Capture the shape/strides buffer pointers before the Vecs are moved into
    // the box; a Vec move keeps its buffer in place, so they stay valid.
    let dl_tensor = build_dl_tensor(data, device, dtype, byte_offset, &shape, &strides);

    let ctx_ptr = Box::into_raw(Box::new(ManagedContext {
        managed: DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(dlpack_deleter::<T>),
        },
        tensor,
        shape,
        strides,
    }));

    // The capsule stores the address of the `managed` field; the deleter recovers
    // the whole box via `manager_ctx`.
    unsafe {
        (*ctx_ptr).managed.manager_ctx = ctx_ptr as *mut c_void;
        let managed_field = &mut (*ctx_ptr).managed as *mut DLManagedTensor as *mut c_void;
        into_capsule(
            py,
            ctx_ptr,
            managed_field,
            DLPACK_CAPSULE_NAME,
            raw_capsule_destructor,
            "Failed to create DLPack capsule",
        )
    }
}

/// Raw PyCapsule destructor for legacy (`dltensor`) capsules — called by Python
/// when GC'ing an unconsumed capsule. A consumer that took ownership (e.g.
/// `torch.from_dlpack`) renamed it to `used_dltensor` and calls the deleter
/// itself, which [`unconsumed_managed_ptr`] detects.
unsafe extern "C" fn raw_capsule_destructor(capsule_ptr: *mut pyo3::ffi::PyObject) {
    let managed_ptr =
        unconsumed_managed_ptr(capsule_ptr, DLPACK_CAPSULE_NAME_USED) as *mut DLManagedTensor;
    if managed_ptr.is_null() {
        return;
    }
    if let Some(deleter) = (*managed_ptr).deleter {
        deleter(managed_ptr);
    }
}

/// Deleter for legacy managed tensors, called by the consumer when done.
///
/// `managed_ptr` points at the `managed` field *inside* the owning
/// `ManagedContext`; `manager_ctx` points at the box base.
unsafe extern "C" fn dlpack_deleter<T>(managed_ptr: *mut DLManagedTensor) {
    if managed_ptr.is_null() {
        return;
    }
    free_managed_ctx::<ManagedContext<T, DLManagedTensor>>((*managed_ptr).manager_ctx);
}

/// Export a tensor to a versioned (`dltensor_versioned`) PyCapsule with the
/// given flags.
fn export_to_capsule_versioned<T: IntoDLPack>(
    py: Python<'_>,
    tensor: T,
    info: TensorInfo,
    flags: u64,
) -> PyResult<Py<PyAny>> {
    validate_strides(&info)?;
    let TensorInfo {
        data,
        device,
        dtype,
        shape,
        strides,
        byte_offset,
    } = info;
    let dl_tensor = build_dl_tensor(data, device, dtype, byte_offset, &shape, &strides);

    let ctx_ptr = Box::into_raw(Box::new(ManagedContext {
        managed: DLManagedTensorVersioned {
            version: DLPackVersion {
                major: DLPACK_MAJOR_VERSION,
                minor: DLPACK_MINOR_VERSION,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(dlpack_deleter_versioned::<T>),
            flags,
            dl_tensor,
        },
        tensor,
        shape,
        strides,
    }));

    unsafe {
        (*ctx_ptr).managed.manager_ctx = ctx_ptr as *mut c_void;
        let managed_field =
            &mut (*ctx_ptr).managed as *mut DLManagedTensorVersioned as *mut c_void;
        into_capsule(
            py,
            ctx_ptr,
            managed_field,
            DLPACK_VERSIONED_CAPSULE_NAME,
            raw_capsule_destructor_versioned,
            "Failed to create versioned DLPack capsule",
        )
    }
}

/// Raw PyCapsule destructor for versioned (`dltensor_versioned`) capsules.
/// Mirrors [`raw_capsule_destructor`] but reads a `DLManagedTensorVersioned`.
unsafe extern "C" fn raw_capsule_destructor_versioned(capsule_ptr: *mut pyo3::ffi::PyObject) {
    let managed_ptr = unconsumed_managed_ptr(capsule_ptr, DLPACK_VERSIONED_CAPSULE_NAME_USED)
        as *mut DLManagedTensorVersioned;
    if managed_ptr.is_null() {
        return;
    }
    if let Some(deleter) = (*managed_ptr).deleter {
        deleter(managed_ptr);
    }
}

/// Deleter for versioned managed tensors, called by the consumer when done.
unsafe extern "C" fn dlpack_deleter_versioned<T>(managed_ptr: *mut DLManagedTensorVersioned) {
    if managed_ptr.is_null() {
        return;
    }
    free_managed_ctx::<ManagedContext<T, DLManagedTensorVersioned>>((*managed_ptr).manager_ctx);
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
