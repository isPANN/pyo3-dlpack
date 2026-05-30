//! Managed tensor for importing from Python via DLPack.
//!
//! This module provides `PyTensor`, a wrapper around a DLPack tensor
//! received from Python that provides safe access to tensor metadata.

use crate::ffi::{
    DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor,
    DLPACK_FLAG_BITMASK_READ_ONLY, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
};
use crate::{
    DLPACK_CAPSULE_NAME, DLPACK_CAPSULE_NAME_USED, DLPACK_VERSIONED_CAPSULE_NAME,
    DLPACK_VERSIONED_CAPSULE_NAME_USED,
};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;

/// Which managed-tensor layout backs a [`PyTensor`].
///
/// The embedded `DLTensor` lives at a different offset in the unversioned vs.
/// versioned struct, and each has its own deleter signature, so we keep the
/// typed owning pointer and branch where layout matters.
#[derive(Clone, Copy)]
enum ManagedPtr {
    Unversioned(NonNull<DLManagedTensor>),
    Versioned(NonNull<DLManagedTensorVersioned>),
}

impl ManagedPtr {
    /// Borrow the embedded `DLTensor`, which lives at a different offset in the
    /// unversioned vs. versioned managed struct.
    ///
    /// # Safety
    /// The pointer must still address a live managed tensor of its layout.
    unsafe fn dl_tensor(&self) -> &DLTensor {
        match *self {
            ManagedPtr::Unversioned(p) => &p.as_ref().dl_tensor,
            ManagedPtr::Versioned(p) => &p.as_ref().dl_tensor,
        }
    }

    /// Invoke the producer's deleter (if present) at the correct struct offset.
    ///
    /// # Safety
    /// Must be called at most once, when relinquishing ownership of the tensor.
    unsafe fn run_deleter(&self) {
        match *self {
            ManagedPtr::Unversioned(p) => {
                if let Some(deleter) = p.as_ref().deleter {
                    deleter(p.as_ptr());
                }
            }
            ManagedPtr::Versioned(p) => {
                if let Some(deleter) = p.as_ref().deleter {
                    deleter(p.as_ptr());
                }
            }
        }
    }
}

/// A tensor imported from Python via the DLPack protocol.
///
/// This type wraps a `DLManagedTensor` received from a Python object
/// (typically a PyTorch, JAX, or NumPy tensor) and provides safe access
/// to the tensor's metadata and data pointer.
///
/// # Lifetime
///
/// The tensor data is valid as long as this `PyTensor` is alive.
/// When dropped, the tensor's deleter is called to notify the producer.
///
/// # Example
///
/// ```ignore
/// use pyo3::prelude::*;
/// use pyo3_dlpack::PyTensor;
///
/// #[pyfunction]
/// fn process(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
///     let tensor = PyTensor::from_pyany(py, obj)?;
///
///     println!("Shape: {:?}", tensor.shape());
///     println!("Device: {:?}", tensor.device());
///     println!("Dtype: {:?}", tensor.dtype());
///
///     if tensor.device().is_cpu() {
///         // Safe to access data on CPU
///         let ptr = tensor.data_ptr() as *const f32;
///         // ...
///     }
///
///     Ok(())
/// }
/// ```
pub struct PyTensor {
    managed: ManagedPtr,
    /// We store the capsule to prevent it from being garbage collected
    /// while we hold a reference to the managed tensor.
    #[allow(dead_code)]
    capsule: Py<PyCapsule>,
}

// Safety: The underlying DLManagedTensor is thread-safe to send
// (the producer guarantees this by implementing DLPack)
unsafe impl Send for PyTensor {}

/// Reject a managed tensor whose `ndim` is negative. `ndim` is an `i32`, and a
/// negative value would cast to a near-`usize::MAX` length in `shape()` /
/// `strides()`, producing a slice that reads far out of bounds. Refuse it at the
/// import boundary before any accessor can trust it.
fn validate_ndim(ndim: i32) -> PyResult<()> {
    if ndim < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DLPack tensor has negative ndim: {ndim}"
        )));
    }
    Ok(())
}

impl PyTensor {
    /// Borrow the embedded `DLTensor`, which lives at a different offset in the
    /// unversioned vs. versioned managed struct.
    fn dl_tensor(&self) -> &DLTensor {
        unsafe { self.managed.dl_tensor() }
    }

    /// Create a PyTensor from a Python object that supports the DLPack protocol.
    ///
    /// This calls `__dlpack__()` on the object to get a DLPack capsule,
    /// then extracts the tensor information.
    ///
    /// # Arguments
    ///
    /// * `py` - Python GIL token
    /// * `obj` - A Python object that implements `__dlpack__()` (e.g., PyTorch tensor)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The object doesn't have a `__dlpack__` method
    /// - The returned capsule is invalid
    /// - The capsule doesn't contain a valid DLManagedTensor
    pub fn from_pyany(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();

        // Advertise versioned support via max_version. Producers whose
        // __dlpack__ predates the kwarg raise TypeError; fall back to a no-arg
        // call for them. The actual capsule kind is decided later by name.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item(
            pyo3::intern!(py, "max_version"),
            (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        )?;

        let capsule_obj = match obj.call_method("__dlpack__", (), Some(&kwargs)) {
            Ok(c) => c,
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {
                obj.call_method0("__dlpack__")?
            }
            Err(e) => return Err(e),
        };

        let capsule: Bound<'_, PyCapsule> = capsule_obj.cast_into().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "__dlpack__ did not return a PyCapsule: {:?}",
                e.into_inner()
            ))
        })?;
        Self::from_capsule(&capsule)
    }

    /// Create a PyTensor directly from a DLPack PyCapsule.
    ///
    /// # Arguments
    ///
    /// * `capsule` - A PyCapsule containing a DLManagedTensor
    ///
    /// # Errors
    ///
    /// Returns an error if the capsule is invalid or has the wrong name.
    pub fn from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        // Decide which DLPack layout this capsule carries by reading its name.
        // A producer may return a legacy capsule even when versioned was
        // requested, so we dispatch on the actual name, never on assumptions.
        let name_ptr = unsafe { pyo3::ffi::PyCapsule_GetName(capsule.as_ptr()) };
        if name_ptr.is_null() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "DLPack capsule has no name",
            ));
        }
        let name = unsafe { CStr::from_ptr(name_ptr) };
        let name_bytes = name.to_bytes();

        if name_bytes == DLPACK_CAPSULE_NAME.to_bytes() {
            Self::from_unversioned_capsule(capsule)
        } else if name_bytes == DLPACK_VERSIONED_CAPSULE_NAME.to_bytes() {
            Self::from_versioned_capsule(capsule)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unexpected DLPack capsule name: {:?}",
                name
            )))
        }
    }

    /// Consume an unversioned (`dltensor`) capsule.
    fn from_unversioned_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME))?;
        let managed = NonNull::new(ptr.as_ptr() as *mut DLManagedTensor).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("DLPack capsule contains null pointer")
        })?;

        // Reject a malformed negative ndim before renaming, so a rejected
        // capsule is left unconsumed for the producer's destructor to free.
        validate_ndim(unsafe { managed.as_ref().dl_tensor.ndim })?;

        // Per DLPack protocol, rename to "used_dltensor" to take ownership and
        // prevent double-consume / double-free.
        //
        // SAFETY: reading the pointer above and renaming the capsule here are two
        // steps, not one atomic operation. They are sound only because the GIL
        // serializes consumers; a future free-threaded (no-GIL) build would need
        // an external lock to prevent a double-consume race.
        let set_name_result = unsafe {
            pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), DLPACK_CAPSULE_NAME_USED.as_ptr())
        };
        if set_name_result != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to mark DLPack capsule as consumed: PyCapsule_SetName failed",
            ));
        }

        Ok(Self {
            managed: ManagedPtr::Unversioned(managed),
            capsule: capsule.clone().unbind(),
        })
    }

    /// Consume a versioned (`dltensor_versioned`) capsule.
    fn from_versioned_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let ptr = capsule.pointer_checked(Some(DLPACK_VERSIONED_CAPSULE_NAME))?;
        let managed =
            NonNull::new(ptr.as_ptr() as *mut DLManagedTensorVersioned).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("DLPack capsule contains null pointer")
            })?;

        // Reject ANY major-version mismatch, per the DLPack spec: a different
        // major version may lay out the struct body (flags, dl_tensor)
        // differently, so we must not read past the version field. Minor
        // versions are ABI-compatible, so they are accepted. We return before
        // renaming the capsule, leaving the producer's destructor to call the
        // deleter (which lives at a stable offset across major versions).
        let version = unsafe { managed.as_ref().version };
        if version.major != DLPACK_MAJOR_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported DLPack major version {}.{} (this build supports major version {})",
                version.major, version.minor, DLPACK_MAJOR_VERSION
            )));
        }

        // Reject a malformed negative ndim before renaming (see
        // `from_unversioned_capsule`); the version check above guarantees the
        // struct layout is ours, so reading `dl_tensor.ndim` is sound.
        validate_ndim(unsafe { managed.as_ref().dl_tensor.ndim })?;

        // SAFETY: as in `from_unversioned_capsule`, the read-then-rename consume
        // is sound only under the GIL's serialization of consumers.
        let set_name_result = unsafe {
            pyo3::ffi::PyCapsule_SetName(
                capsule.as_ptr(),
                DLPACK_VERSIONED_CAPSULE_NAME_USED.as_ptr(),
            )
        };
        if set_name_result != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to mark DLPack capsule as consumed: PyCapsule_SetName failed",
            ));
        }

        Ok(Self {
            managed: ManagedPtr::Versioned(managed),
            capsule: capsule.clone().unbind(),
        })
    }

    /// Get the device where the tensor data resides.
    pub fn device(&self) -> DLDevice {
        self.dl_tensor().device
    }

    /// Get the data type of the tensor elements.
    pub fn dtype(&self) -> DLDataType {
        self.dl_tensor().dtype
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dl_tensor().ndim as usize
    }

    /// Get the shape as a slice.
    ///
    /// The length of the slice equals `ndim()`.
    pub fn shape(&self) -> &[i64] {
        let tensor = self.dl_tensor();
        if tensor.shape.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) }
        }
    }

    /// Get the strides as a slice, or `None` for contiguous tensors.
    ///
    /// Strides are in number of elements (not bytes).
    /// If `None`, the tensor is assumed to be in compact row-major order.
    pub fn strides(&self) -> Option<&[i64]> {
        let tensor = self.dl_tensor();
        if tensor.strides.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(tensor.strides, tensor.ndim as usize) })
        }
    }

    /// Check if the tensor is contiguous in row-major (C) order.
    pub fn is_contiguous(&self) -> bool {
        match self.strides() {
            None => true,
            Some(strides) => {
                let shape = self.shape();
                if shape.is_empty() {
                    return true;
                }

                let mut expected_stride = 1i64;
                for i in (0..shape.len()).rev() {
                    if strides[i] != expected_stride {
                        return false;
                    }
                    expected_stride *= shape[i];
                }
                true
            }
        }
    }

    /// Get the raw data pointer.
    ///
    /// For GPU tensors, this is a device pointer that cannot be directly
    /// dereferenced on the CPU.
    ///
    /// The pointer is adjusted by `byte_offset()`.
    pub fn data_ptr(&self) -> *mut c_void {
        let tensor = self.dl_tensor();
        // `wrapping_add` (not `add`): the base may be null (0-element tensor) or a
        // non-host device pointer, where `add`'s in-bounds/provenance requirement
        // would be undefined behavior. The numeric result is identical.
        (tensor.data as *mut u8).wrapping_add(tensor.byte_offset as usize) as *mut c_void
    }

    /// Get the raw data pointer without byte offset adjustment.
    pub fn data_ptr_raw(&self) -> *mut c_void {
        self.dl_tensor().data
    }

    /// Get the byte offset from the raw data pointer.
    pub fn byte_offset(&self) -> u64 {
        self.dl_tensor().byte_offset
    }

    /// Get the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    /// Get the size of one element in bytes.
    pub fn itemsize(&self) -> usize {
        self.dtype().itemsize()
    }

    /// Get the total size of the tensor data in bytes.
    pub fn nbytes(&self) -> usize {
        self.numel() * self.itemsize()
    }

    /// Whether the tensor is marked read-only.
    ///
    /// Only versioned (DLPack 1.0) tensors can carry this flag; legacy tensors
    /// always report `false`.
    pub fn is_read_only(&self) -> bool {
        match self.managed {
            ManagedPtr::Unversioned(_) => false,
            ManagedPtr::Versioned(p) => unsafe {
                p.as_ref().flags & DLPACK_FLAG_BITMASK_READ_ONLY != 0
            },
        }
    }
}

impl Drop for PyTensor {
    fn drop(&mut self) {
        // Call the producer's deleter at the correct struct offset for each layout.
        unsafe { self.managed.run_deleter() }
    }
}

impl std::fmt::Debug for PyTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyTensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("byte_offset", &self.byte_offset())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{cpu_device, cuda_device, dtype_f32, dtype_f64, DLTensor};
    use pyo3::Python;
    use std::ffi::CString;

    /// Wrapper to make pointer Send for testing
    #[repr(transparent)]
    struct SendableTestPtr(*mut DLManagedTensor);
    unsafe impl Send for SendableTestPtr {}

    /// Helper to create a test DLManagedTensor with given parameters
    struct TestManagedTensor {
        managed: Box<DLManagedTensor>,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
        #[allow(dead_code)]
        data: Vec<u8>,
    }

    impl TestManagedTensor {
        fn new(
            shape: Vec<i64>,
            strides: Option<Vec<i64>>,
            dtype: DLDataType,
            device: DLDevice,
        ) -> Self {
            let numel: usize = shape.iter().map(|&d| d as usize).product();
            let data = vec![0u8; numel.max(1) * dtype.itemsize()];

            let mut result = Self {
                managed: Box::new(DLManagedTensor {
                    dl_tensor: DLTensor {
                        data: std::ptr::null_mut(),
                        device,
                        ndim: shape.len() as i32,
                        dtype,
                        shape: std::ptr::null_mut(),
                        strides: std::ptr::null_mut(),
                        byte_offset: 0,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: None,
                }),
                shape,
                strides,
                data,
            };

            // Set up pointers
            result.managed.dl_tensor.data = result.data.as_ptr() as *mut c_void;
            result.managed.dl_tensor.shape = result.shape.as_mut_ptr();
            if let Some(ref mut s) = result.strides {
                result.managed.dl_tensor.strides = s.as_mut_ptr();
            }

            result
        }

        fn with_byte_offset(mut self, offset: u64) -> Self {
            self.managed.dl_tensor.byte_offset = offset;
            self
        }

        fn as_ptr(&self) -> *mut DLManagedTensor {
            &*self.managed as *const _ as *mut _
        }
    }

    // ========================================================================
    // is_contiguous tests
    // ========================================================================

    #[test]
    fn test_is_contiguous_no_strides() {
        // No strides means contiguous by default
        let tensor = TestManagedTensor::new(vec![2, 3, 4], None, dtype_f32(), cpu_device());

        // Create a mock PyTensor-like check using the raw managed tensor
        let managed = unsafe { &*tensor.as_ptr() };
        let strides_ptr = managed.dl_tensor.strides;

        // No strides pointer = contiguous
        assert!(strides_ptr.is_null());
    }

    #[test]
    fn test_is_contiguous_with_contiguous_strides() {
        // Row-major contiguous strides for shape [2, 3, 4]
        // strides should be [12, 4, 1]
        let tensor = TestManagedTensor::new(
            vec![2, 3, 4],
            Some(vec![12, 4, 1]),
            dtype_f32(),
            cpu_device(),
        );

        let shape = &tensor.shape;
        let strides = tensor.strides.as_ref().unwrap();

        // Verify contiguity check logic
        let mut expected_stride = 1i64;
        let mut is_contiguous = true;
        for i in (0..shape.len()).rev() {
            if strides[i] != expected_stride {
                is_contiguous = false;
                break;
            }
            expected_stride *= shape[i];
        }
        assert!(is_contiguous);
    }

    #[test]
    fn test_is_contiguous_with_non_contiguous_strides() {
        // Non-contiguous strides (transposed)
        let tensor = TestManagedTensor::new(
            vec![2, 3, 4],
            Some(vec![1, 2, 6]), // Column-major like strides
            dtype_f32(),
            cpu_device(),
        );

        let shape = &tensor.shape;
        let strides = tensor.strides.as_ref().unwrap();

        let mut expected_stride = 1i64;
        let mut is_contiguous = true;
        for i in (0..shape.len()).rev() {
            if strides[i] != expected_stride {
                is_contiguous = false;
                break;
            }
            expected_stride *= shape[i];
        }
        assert!(!is_contiguous);
    }

    #[test]
    fn test_is_contiguous_empty_tensor() {
        let tensor = TestManagedTensor::new(vec![], None, dtype_f32(), cpu_device());
        // Empty shape is contiguous
        assert!(tensor.shape.is_empty());
    }

    #[test]
    fn test_is_contiguous_1d() {
        let tensor = TestManagedTensor::new(vec![10], Some(vec![1]), dtype_f32(), cpu_device());
        let strides = tensor.strides.as_ref().unwrap();
        assert_eq!(strides[0], 1);
    }

    // ========================================================================
    // numel and nbytes tests
    // ========================================================================

    #[test]
    fn test_numel_calculation() {
        let shapes_and_expected: Vec<(Vec<i64>, usize)> = vec![
            (vec![], 1), // Scalar (product of empty = 1)
            (vec![5], 5),
            (vec![2, 3], 6),
            (vec![2, 3, 4], 24),
            (vec![1, 1, 1, 1], 1),
            (vec![10, 20, 30], 6000),
        ];

        for (shape, expected) in shapes_and_expected {
            let numel: usize = if shape.is_empty() {
                1 // Scalar case
            } else {
                shape.iter().map(|&d| d as usize).product()
            };
            assert_eq!(numel, expected, "Failed for shape {:?}", shape);
        }
    }

    #[test]
    fn test_nbytes_calculation() {
        // f32 tensor [2, 3, 4] = 24 elements * 4 bytes = 96 bytes
        let tensor = TestManagedTensor::new(vec![2, 3, 4], None, dtype_f32(), cpu_device());
        let numel: usize = tensor.shape.iter().map(|&d| d as usize).product();
        let itemsize = dtype_f32().itemsize();
        assert_eq!(numel * itemsize, 96);

        // f64 tensor [2, 3] = 6 elements * 8 bytes = 48 bytes
        let tensor2 = TestManagedTensor::new(vec![2, 3], None, dtype_f64(), cpu_device());
        let numel2: usize = tensor2.shape.iter().map(|&d| d as usize).product();
        let itemsize2 = dtype_f64().itemsize();
        assert_eq!(numel2 * itemsize2, 48);
    }

    // ========================================================================
    // data_ptr tests
    // ========================================================================

    #[test]
    fn test_data_ptr_with_offset() {
        let tensor =
            TestManagedTensor::new(vec![10], None, dtype_f32(), cpu_device()).with_byte_offset(16);

        let managed = unsafe { &*tensor.as_ptr() };
        let base_ptr = managed.dl_tensor.data as usize;
        let offset = managed.dl_tensor.byte_offset as usize;
        let adjusted_ptr = base_ptr + offset;

        assert_eq!(offset, 16);
        assert_eq!(adjusted_ptr, base_ptr + 16);
    }

    #[test]
    fn test_data_ptr_no_offset() {
        let tensor = TestManagedTensor::new(vec![10], None, dtype_f32(), cpu_device());

        let managed = unsafe { &*tensor.as_ptr() };
        assert_eq!(managed.dl_tensor.byte_offset, 0);
    }

    // ========================================================================
    // Device and dtype accessor tests
    // ========================================================================

    #[test]
    fn test_device_accessor() {
        let cpu_tensor = TestManagedTensor::new(vec![2, 3], None, dtype_f32(), cpu_device());
        let managed = unsafe { &*cpu_tensor.as_ptr() };
        assert!(managed.dl_tensor.device.is_cpu());

        let cuda_tensor = TestManagedTensor::new(vec![2, 3], None, dtype_f32(), cuda_device(1));
        let managed = unsafe { &*cuda_tensor.as_ptr() };
        assert!(managed.dl_tensor.device.is_cuda());
        assert_eq!(managed.dl_tensor.device.device_id, 1);
    }

    #[test]
    fn test_dtype_accessor() {
        let f32_tensor = TestManagedTensor::new(vec![2, 3], None, dtype_f32(), cpu_device());
        let managed = unsafe { &*f32_tensor.as_ptr() };
        assert!(managed.dl_tensor.dtype.is_f32());

        let f64_tensor = TestManagedTensor::new(vec![2, 3], None, dtype_f64(), cpu_device());
        let managed = unsafe { &*f64_tensor.as_ptr() };
        assert!(managed.dl_tensor.dtype.is_f64());
    }

    // ========================================================================
    // ndim and shape tests
    // ========================================================================

    #[test]
    fn test_ndim() {
        let shapes: Vec<Vec<i64>> = vec![
            vec![],
            vec![5],
            vec![2, 3],
            vec![2, 3, 4],
            vec![1, 2, 3, 4, 5],
        ];

        for shape in shapes {
            let expected_ndim = shape.len();
            let tensor = TestManagedTensor::new(shape.clone(), None, dtype_f32(), cpu_device());
            let managed = unsafe { &*tensor.as_ptr() };
            assert_eq!(managed.dl_tensor.ndim as usize, expected_ndim);
        }
    }

    #[test]
    fn test_shape_accessor() {
        let shape = vec![2i64, 3, 4];
        let tensor = TestManagedTensor::new(shape.clone(), None, dtype_f32(), cpu_device());
        let managed = unsafe { &*tensor.as_ptr() };

        let shape_slice = unsafe {
            std::slice::from_raw_parts(managed.dl_tensor.shape, managed.dl_tensor.ndim as usize)
        };
        assert_eq!(shape_slice, &[2, 3, 4]);
    }

    // ========================================================================
    // PyCapsule integration tests (require Python)
    // ========================================================================

    #[test]
    fn test_capsule_creation_and_extraction() {
        Python::attach(|py| {
            // Create a test managed tensor
            let mut shape = vec![2i64, 3];
            let data = [0u8; 24].to_vec(); // 6 f32 elements

            let managed = Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data.as_ptr() as *mut c_void,
                    device: cpu_device(),
                    ndim: 2,
                    dtype: dtype_f32(),
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            });

            let managed_ptr = Box::into_raw(managed);
            let sendable = SendableTestPtr(managed_ptr);
            let name = CString::new("dltensor").unwrap();

            // Create a PyCapsule with Send wrapper
            let capsule =
                PyCapsule::new(py, sendable, Some(name)).expect("Failed to create capsule");

            // Verify capsule name exists
            let capsule_name = capsule.name().expect("Failed to get name");
            assert!(capsule_name.is_some());

            // Extract the pointer back - pointer_checked returns NonNull on success
            let _extracted = capsule
                .pointer_checked(Some(DLPACK_CAPSULE_NAME))
                .expect("Failed to extract pointer");

            // Clean up
            unsafe {
                let _ = Box::from_raw(managed_ptr);
            }
        });
    }

    #[test]
    fn test_capsule_wrong_name() {
        /// Wrapper for test data
        #[allow(dead_code)]
        struct TestData(i32);
        unsafe impl Send for TestData {}

        Python::attach(|py| {
            let data = TestData(42);
            let name = CString::new("wrong_name").unwrap();

            let capsule = PyCapsule::new(py, data, Some(name)).expect("Failed to create capsule");

            // Should fail when extracting with wrong expected name
            let result = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME));
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_pytensor_send() {
        // Verify PyTensor implements Send
        fn assert_send<T: Send>() {}
        assert_send::<PyTensor>();
    }

    // ========================================================================
    // PyTensor comprehensive tests using direct DLManagedTensor capsules
    // ========================================================================

    use std::sync::atomic::{AtomicUsize, Ordering};

    static DELETER_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Helper struct to hold all the data for a test tensor capsule
    struct TestTensorContext {
        data: Vec<f32>,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
    }

    /// Create a DLPack-compatible capsule for testing PyTensor
    fn create_test_capsule(
        py: Python<'_>,
        ctx: Box<TestTensorContext>,
        device: DLDevice,
        dtype: DLDataType,
        byte_offset: u64,
        with_deleter: bool,
    ) -> PyResult<Bound<'_, PyCapsule>> {
        let ctx_ptr = Box::into_raw(ctx);

        unsafe {
            let ctx_ref = &mut *ctx_ptr;

            let managed = Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: ctx_ref.data.as_ptr() as *mut c_void,
                    device,
                    ndim: ctx_ref.shape.len() as i32,
                    dtype,
                    shape: ctx_ref.shape.as_mut_ptr(),
                    strides: ctx_ref
                        .strides
                        .as_mut()
                        .map(|s| s.as_mut_ptr())
                        .unwrap_or(std::ptr::null_mut()),
                    byte_offset,
                },
                manager_ctx: ctx_ptr as *mut c_void,
                deleter: if with_deleter {
                    Some(test_deleter)
                } else {
                    None
                },
            });

            let managed_ptr = Box::into_raw(managed);
            let wrapper = SendableTestPtr(managed_ptr);
            let name = CString::new("dltensor").unwrap();

            PyCapsule::new(py, wrapper, Some(name))
        }
    }

    /// Test deleter that increments a counter
    unsafe extern "C" fn test_deleter(managed_ptr: *mut DLManagedTensor) {
        if !managed_ptr.is_null() {
            DELETER_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
            let managed = Box::from_raw(managed_ptr);
            if !managed.manager_ctx.is_null() {
                let _ = Box::from_raw(managed.manager_ctx as *mut TestTensorContext);
            }
        }
    }

    #[test]
    fn test_pytensor_all_accessors() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: vec![2, 3],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            // Create PyTensor - need to read the pointer from the capsule correctly
            let ptr = capsule
                .pointer_checked(Some(DLPACK_CAPSULE_NAME))
                .expect("Failed to get pointer");
            // The capsule stores SendableTestPtr, so we need to dereference to get the actual pointer
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).expect("Null pointer");

            // Manually construct PyTensor for testing
            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            // Test all accessor methods
            assert!(pytensor.device().is_cpu());
            assert!(pytensor.dtype().is_f32());
            assert_eq!(pytensor.ndim(), 2);
            assert_eq!(pytensor.shape(), &[2, 3]);
            assert!(pytensor.strides().is_none());
            assert!(pytensor.is_contiguous());
            assert!(!pytensor.data_ptr().is_null());
            assert!(!pytensor.data_ptr_raw().is_null());
            assert_eq!(pytensor.byte_offset(), 0);
            assert_eq!(pytensor.numel(), 6);
            assert_eq!(pytensor.itemsize(), 4);
            assert_eq!(pytensor.nbytes(), 24);

            // Test Debug
            let debug = format!("{:?}", pytensor);
            assert!(debug.contains("PyTensor"));
            assert!(debug.contains("shape"));
            assert!(debug.contains("dtype"));
            assert!(debug.contains("device"));

            // Prevent double-free by not running the deleter
            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_with_strides_contiguous() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0; 24],
                shape: vec![2, 3, 4],
                strides: Some(vec![12, 4, 1]), // Row-major contiguous
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert_eq!(pytensor.ndim(), 3);
            assert_eq!(pytensor.shape(), &[2, 3, 4]);
            assert_eq!(pytensor.strides(), Some(&[12i64, 4, 1][..]));
            assert!(pytensor.is_contiguous());
            assert_eq!(pytensor.numel(), 24);

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_non_contiguous() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0; 6],
                shape: vec![2, 3],
                strides: Some(vec![1, 2]), // Column-major (non-contiguous for row-major check)
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert!(!pytensor.is_contiguous());
            assert_eq!(pytensor.strides(), Some(&[1i64, 2][..]));

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_scalar() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![42.0],
                shape: vec![],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert_eq!(pytensor.ndim(), 0);
            assert!(pytensor.shape().is_empty());
            assert!(pytensor.is_contiguous());
            assert_eq!(pytensor.numel(), 1);

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_with_byte_offset() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0; 20],
                shape: vec![10],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 16, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert_eq!(pytensor.byte_offset(), 16);
            let raw = pytensor.data_ptr_raw() as usize;
            let adjusted = pytensor.data_ptr() as usize;
            assert_eq!(adjusted, raw + 16);

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_cuda_device() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0; 512],
                shape: vec![16, 32],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cuda_device(1), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert!(pytensor.device().is_cuda());
            assert_eq!(pytensor.device().device_id, 1);

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_f64_dtype() {
        Python::attach(|py| {
            // Use f32 data but declare f64 dtype for testing
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0; 6], // 6 f32 = 24 bytes = 3 f64
                shape: vec![3],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f64(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert!(pytensor.dtype().is_f64());
            assert_eq!(pytensor.itemsize(), 8);
            assert_eq!(pytensor.nbytes(), 24);

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_empty_strides_scalar() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0],
                shape: vec![],
                strides: Some(vec![]), // Empty strides for scalar
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            assert!(pytensor.is_contiguous());
            assert!(pytensor.strides().is_some());
            assert!(pytensor.strides().unwrap().is_empty());

            std::mem::forget(pytensor);
        });
    }

    #[test]
    fn test_pytensor_drop_calls_deleter() {
        DELETER_CALL_COUNT.store(0, Ordering::SeqCst);

        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0, 2.0, 3.0],
                shape: vec![3],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, true)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            {
                let pytensor = PyTensor {
                    managed: ManagedPtr::Unversioned(managed),
                    capsule: capsule.clone().unbind(),
                };

                // PyTensor exists, deleter not called yet
                assert_eq!(DELETER_CALL_COUNT.load(Ordering::SeqCst), 0);

                // Drop the PyTensor
                drop(pytensor);
            }

            // Deleter should have been called
            assert_eq!(DELETER_CALL_COUNT.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn test_pytensor_drop_no_deleter() {
        Python::attach(|py| {
            let ctx = Box::new(TestTensorContext {
                data: vec![1.0],
                shape: vec![1],
                strides: None,
            });

            let capsule = create_test_capsule(py, ctx, cpu_device(), dtype_f32(), 0, false)
                .expect("Failed to create capsule");

            let ptr = capsule.pointer_checked(Some(DLPACK_CAPSULE_NAME)).unwrap();
            let managed_ptr = unsafe { *(ptr.as_ptr() as *const *mut DLManagedTensor) };
            let managed = NonNull::new(managed_ptr).unwrap();

            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };

            // Drop without deleter should not crash
            drop(pytensor);

            // Clean up manually since no deleter
            unsafe {
                let managed = Box::from_raw(managed_ptr);
                if !managed.manager_ctx.is_null() {
                    let _ = Box::from_raw(managed.manager_ctx as *mut TestTensorContext);
                }
            }
        });
    }

    // ========================================================================
    // Versioned / read-only round-trip tests
    // ========================================================================

    struct RoundTripTensor {
        data: Vec<f32>,
        shape: Vec<i64>,
    }

    impl crate::IntoDLPack for RoundTripTensor {
        fn tensor_info(&self) -> crate::TensorInfo {
            crate::TensorInfo::contiguous(
                self.data.as_ptr() as *mut c_void,
                cpu_device(),
                dtype_f32(),
                self.shape.clone(),
            )
        }
    }

    #[test]
    fn test_roundtrip_versioned_readonly() {
        use crate::IntoDLPack;
        Python::attach(|py| {
            let t = RoundTripTensor {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![2, 2],
            };
            let capsule_obj = t.into_dlpack_readonly(py).unwrap();
            let bound = capsule_obj.into_bound(py);
            let capsule: Bound<'_, PyCapsule> = bound.cast_into().unwrap();

            let tensor = PyTensor::from_capsule(&capsule).unwrap();
            assert!(tensor.is_read_only());
            assert_eq!(tensor.shape(), &[2, 2]);
            assert!(tensor.device().is_cpu());
            assert!(tensor.dtype().is_f32());
            // Dropping `tensor` runs the versioned deleter and frees the context.
        });
    }

    #[test]
    fn test_roundtrip_unversioned_not_readonly() {
        use crate::IntoDLPack;
        Python::attach(|py| {
            let t = RoundTripTensor {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![2, 2],
            };
            let capsule_obj = t.into_dlpack(py).unwrap();
            let bound = capsule_obj.into_bound(py);
            let capsule: Bound<'_, PyCapsule> = bound.cast_into().unwrap();

            let tensor = PyTensor::from_capsule(&capsule).unwrap();
            assert!(!tensor.is_read_only());
            assert_eq!(tensor.shape(), &[2, 2]);
        });
    }

    #[test]
    fn test_from_capsule_rejects_unknown_name() {
        Python::attach(|py| {
            // A capsule whose name is neither "dltensor" nor "dltensor_versioned"
            // must be rejected by the name dispatcher.
            let dummy = Box::new(0u8);
            let dummy_ptr = Box::into_raw(dummy);
            let capsule_ptr = unsafe {
                pyo3::ffi::PyCapsule_New(
                    dummy_ptr as *mut c_void,
                    c"not_a_dlpack_capsule".as_ptr(),
                    None,
                )
            };
            assert!(!capsule_ptr.is_null());
            let capsule: Bound<'_, PyCapsule> = unsafe { Bound::from_owned_ptr(py, capsule_ptr) }
                .cast_into()
                .unwrap();

            let result = PyTensor::from_capsule(&capsule);
            assert!(result.is_err());

            // from_capsule rejected before consuming; reclaim the dummy box.
            unsafe {
                let _ = Box::from_raw(dummy_ptr);
            }
        });
    }

    #[test]
    fn test_versioned_rejects_too_new_major() {
        Python::attach(|py| {
            // A versioned capsule claiming a major version newer than we support
            // must be rejected (we may misinterpret a future struct layout).
            let mut shape = vec![1i64];
            let data = vec![0.0f32];
            let managed = Box::new(DLManagedTensorVersioned {
                version: crate::ffi::DLPackVersion {
                    major: DLPACK_MAJOR_VERSION + 1,
                    minor: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
                flags: 0,
                dl_tensor: DLTensor {
                    data: data.as_ptr() as *mut c_void,
                    device: cpu_device(),
                    ndim: 1,
                    dtype: dtype_f32(),
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            });
            let managed_ptr = Box::into_raw(managed);
            let capsule_ptr = unsafe {
                pyo3::ffi::PyCapsule_New(
                    managed_ptr as *mut c_void,
                    c"dltensor_versioned".as_ptr(),
                    None,
                )
            };
            assert!(!capsule_ptr.is_null());
            let capsule: Bound<'_, PyCapsule> = unsafe { Bound::from_owned_ptr(py, capsule_ptr) }
                .cast_into()
                .unwrap();

            let result = PyTensor::from_capsule(&capsule);
            assert!(result.is_err());

            // from_capsule rejected before consuming, so reclaim the box ourselves.
            unsafe {
                let _ = Box::from_raw(managed_ptr);
            }
            // Keep the backing arrays alive until after the pointers are done.
            drop(shape);
            drop(data);
        });
    }

    #[test]
    fn test_versioned_rejects_mismatched_lower_major() {
        Python::attach(|py| {
            // A versioned capsule claiming a major version LOWER than ours
            // (e.g. 0) is malformed/ABI-incompatible and must be rejected too —
            // we must not read flags/dl_tensor at our assumed offsets.
            let mut shape = vec![1i64];
            let data = vec![0.0f32];
            let managed = Box::new(DLManagedTensorVersioned {
                version: crate::ffi::DLPackVersion {
                    major: DLPACK_MAJOR_VERSION - 1,
                    minor: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
                flags: 0,
                dl_tensor: DLTensor {
                    data: data.as_ptr() as *mut c_void,
                    device: cpu_device(),
                    ndim: 1,
                    dtype: dtype_f32(),
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            });
            let managed_ptr = Box::into_raw(managed);
            let capsule_ptr = unsafe {
                pyo3::ffi::PyCapsule_New(
                    managed_ptr as *mut c_void,
                    c"dltensor_versioned".as_ptr(),
                    None,
                )
            };
            assert!(!capsule_ptr.is_null());
            let capsule: Bound<'_, PyCapsule> = unsafe { Bound::from_owned_ptr(py, capsule_ptr) }
                .cast_into()
                .unwrap();

            let result = PyTensor::from_capsule(&capsule);
            assert!(result.is_err());

            // from_capsule rejected before consuming, so reclaim the box ourselves.
            unsafe {
                let _ = Box::from_raw(managed_ptr);
            }
            drop(shape);
            drop(data);
        });
    }
}
