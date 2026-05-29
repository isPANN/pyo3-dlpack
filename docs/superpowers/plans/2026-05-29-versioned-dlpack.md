# Versioned DLPack (DLPack 1.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add versioned DLPack 1.0 support (read-only tensors) on both export and import sides, keeping the legacy path and the simple/stable public API unchanged.

**Architecture:** Add the `DLManagedTensorVersioned` ABI type alongside the existing legacy `DLManagedTensor`. Export gains one new self-explanatory method (`into_dlpack_readonly`) that emits a versioned `dltensor_versioned` capsule with the `READ_ONLY` flag; `into_dlpack` is unchanged and keeps emitting legacy `dltensor`. Import (`from_pyany` / `from_capsule`) transparently accepts both capsule kinds, negotiates `max_version`, and exposes one new reader (`is_read_only`). `PyTensor` holds an internal enum recording which layout it wraps so `Drop` calls the deleter at the correct struct offset.

**Tech Stack:** Rust, PyO3 0.28 (raw `pyo3::ffi` capsule API), DLPack C ABI, maturin + pytest for Python integration.

**Spec:** `docs/superpowers/specs/2026-05-29-versioned-dlpack-design.md`

**Design invariant (do not violate):** No `ExportConfig`, no user-facing flag bitmask, no boolean parameters on existing methods. New public surface is exactly: `into_dlpack_readonly`, `is_read_only`, and the re-exported FFI types/constants. Existing signatures and behavior are unchanged.

---

## File Structure

- `src/ffi.rs` — add `DLPackVersion`, `DLManagedTensorVersioned`, its deleter typedef, flag-bitmask constants, version constants, and layout tests.
- `src/lib.rs` — re-export the new FFI types/constants; add versioned capsule-name `&CStr` constants.
- `src/export.rs` — extract a shared `build_export_parts` helper (DRY), add the versioned export path (`into_dlpack_readonly`, versioned deleter, versioned destructor, `export_to_capsule_versioned`).
- `src/managed.rs` — replace `PyTensor`'s `managed` field with a `ManagedPtr` enum, route accessors through a `dl_tensor()` helper, dispatch `from_capsule` on capsule name, add `is_read_only`, add `max_version` negotiation in `from_pyany`.
- `tests/python_helpers/src/lib.rs` — add `export_cpu_tensor_readonly` and `import_is_readonly` helper `#[pyfunction]`s and register them.
- `tests/test_dlpack_integration.py` — add versioned/read-only integration tests.
- `README.md`, `CHANGELOG.md` — document the feature.

---

## Task 1: FFI types, constants, and layout tests

**Files:**
- Modify: `src/ffi.rs` (add types/constants after the `DLManagedTensor` definition at `src/ffi.rs:311`; add tests in the existing `#[cfg(test)]` module near `src/ffi.rs:860`)
- Modify: `src/lib.rs:64`, `src/lib.rs:68-71`, `src/lib.rs:74-77`

- [ ] **Step 1: Write the failing layout tests**

In `src/ffi.rs`, inside `mod tests`, after `test_dl_managed_tensor_size` (around `src/ffi.rs:866`), add:

```rust
    #[test]
    fn test_dl_pack_version_layout() {
        // DLPackVersion is two u32 fields, no padding.
        assert_eq!(std::mem::size_of::<DLPackVersion>(), 8);
        assert_eq!(std::mem::offset_of!(DLPackVersion, major), 0);
        assert_eq!(std::mem::offset_of!(DLPackVersion, minor), 4);
    }

    #[test]
    fn test_dl_managed_tensor_versioned_layout() {
        // DLPack 1.0 field order: version, manager_ctx, deleter, flags, dl_tensor.
        // version must be first; dl_tensor must be last (after flags).
        assert_eq!(std::mem::offset_of!(DLManagedTensorVersioned, version), 0);
        assert!(
            std::mem::offset_of!(DLManagedTensorVersioned, manager_ctx)
                >= std::mem::size_of::<DLPackVersion>()
        );
        assert!(
            std::mem::offset_of!(DLManagedTensorVersioned, dl_tensor)
                > std::mem::offset_of!(DLManagedTensorVersioned, flags)
        );
        // The versioned struct embeds a full DLTensor plus header fields.
        assert!(
            std::mem::size_of::<DLManagedTensorVersioned>() > std::mem::size_of::<DLTensor>()
        );
    }

    #[test]
    fn test_read_only_flag_value() {
        assert_eq!(DLPACK_FLAG_BITMASK_READ_ONLY, 1);
        assert_eq!(DLPACK_FLAG_BITMASK_IS_COPIED, 2);
        assert_eq!(DLPACK_MAJOR_VERSION, 1);
    }
```

- [ ] **Step 2: Run tests to verify they fail (do not compile)**

Run: `cargo test --lib ffi::tests::test_dl_managed_tensor_versioned_layout`
Expected: FAIL — compile error `cannot find type DLManagedTensorVersioned` / `cannot find value DLPACK_FLAG_BITMASK_READ_ONLY`.

- [ ] **Step 3: Add the FFI types and constants**

In `src/ffi.rs`, immediately after the `DLManagedTensor` struct (after `src/ffi.rs:311`, before the `// Convenience constructors` banner), add:

```rust
/// DLPack protocol version, as carried by `DLManagedTensorVersioned`.
///
/// Corresponds to `DLPackVersion` in the DLPack specification.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLPackVersion {
    /// Major version. Incremented on ABI-breaking changes.
    pub major: u32,
    /// Minor version. Incremented on backward-compatible additions.
    pub minor: u32,
}

/// The DLPack major version this crate produces and accepts.
pub const DLPACK_MAJOR_VERSION: u32 = 1;
/// The DLPack minor version this crate produces.
pub const DLPACK_MINOR_VERSION: u32 = 1;

/// Flag bitmask: the tensor data is read-only.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
/// Flag bitmask: the tensor data was copied by the producer.
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;
/// Flag bitmask: a sub-byte-typed tensor is padded to a byte boundary.
pub const DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED: u64 = 1 << 2;

/// Deleter function signature for `DLManagedTensorVersioned`.
///
/// Note this takes a pointer to the *versioned* struct, which has a different
/// layout from `DLManagedTensor`, so it is a distinct type from
/// [`DLManagedTensorDeleter`].
pub type DLManagedTensorVersionedDeleter =
    unsafe extern "C" fn(*mut DLManagedTensorVersioned);

/// A versioned managed tensor (DLPack 1.0).
///
/// Corresponds to `DLManagedTensorVersioned` in the DLPack specification.
/// The field order differs from [`DLManagedTensor`]: `version` is first and
/// `dl_tensor` is last. This is an ABI contract — do not reorder.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// Protocol version of this struct.
    pub version: DLPackVersion,
    /// Opaque manager context for the producer's use.
    pub manager_ctx: *mut c_void,
    /// Deleter function called when the consumer is done. Can be null.
    pub deleter: Option<DLManagedTensorVersionedDeleter>,
    /// Bitmask of `DLPACK_FLAG_BITMASK_*` flags.
    pub flags: u64,
    /// The underlying tensor descriptor.
    pub dl_tensor: DLTensor,
}
```

- [ ] **Step 4: Re-export from lib.rs**

In `src/lib.rs`, change the FFI re-export at `src/lib.rs:64` from:

```rust
pub use ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};
```

to:

```rust
pub use ffi::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned,
    DLPackVersion, DLTensor, DLPACK_FLAG_BITMASK_IS_COPIED,
    DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED, DLPACK_FLAG_BITMASK_READ_ONLY,
    DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
};
```

Then, after the existing capsule-name constants at `src/lib.rs:74-77`, add:

```rust
/// The DLPack capsule name for versioned (DLPack 1.0) tensor exchange.
pub const DLPACK_VERSIONED_CAPSULE_NAME: &std::ffi::CStr = c"dltensor_versioned";

/// The versioned DLPack capsule name after consumption (to prevent double-free).
pub const DLPACK_VERSIONED_CAPSULE_NAME_USED: &std::ffi::CStr = c"used_dltensor_versioned";
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib ffi::tests::`
Expected: PASS (all ffi tests, including the three new ones).

- [ ] **Step 6: Commit**

```bash
git add src/ffi.rs src/lib.rs
git commit -m "feat(ffi): add DLManagedTensorVersioned types, flags, and version constants"
```

---

## Task 2: Export — extract shared `build_export_parts` helper (DRY refactor)

This is a pure refactor with no behavior change. The existing export tests are the safety net.

**Files:**
- Modify: `src/export.rs:158-256` (`export_to_capsule`)

- [ ] **Step 1: Add the shared helper**

In `src/export.rs`, immediately before `fn export_to_capsule` (before `src/export.rs:159`), add:

```rust
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
```

- [ ] **Step 2: Rewrite `export_to_capsule` to use the helper**

Replace the entire body of `fn export_to_capsule` (`src/export.rs:159-256`) with:

```rust
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
    // Use static name so it remains valid for the capsule's lifetime.
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            DLPACK_CAPSULE_NAME.as_ptr() as *const i8,
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

    // Store a reference to ctx_ptr in the capsule context so the destructor
    // can check if the capsule was consumed and clean up properly.
    unsafe {
        pyo3::ffi::PyCapsule_SetContext(capsule_ptr, ctx_ptr as *mut c_void);
    }

    Ok(unsafe { Bound::from_owned_ptr(py, capsule_ptr).unbind() })
}
```

- [ ] **Step 3: Run the existing export tests to verify no regression**

Run: `cargo test --lib export::`
Expected: PASS — all existing export tests still pass (no behavior change).

- [ ] **Step 4: Commit**

```bash
git add src/export.rs
git commit -m "refactor(export): extract shared build_export_parts helper"
```

---

## Task 3: Export — versioned read-only path

**Files:**
- Modify: `src/export.rs` (add trait method near `src/export.rs:138`; add statics, functions, and a test)

- [ ] **Step 1: Write the failing test**

In `src/export.rs`, inside `mod tests`, after `test_into_dlpack_1d` (around `src/export.rs:666`), add:

```rust
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

            // A read-only export must produce a versioned capsule.
            let name = unsafe {
                let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule.as_ptr());
                assert!(!name_ptr.is_null());
                CStr::from_ptr(name_ptr).to_owned()
            };
            assert_eq!(name.to_bytes(), b"dltensor_versioned");
        });
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib export::tests::test_into_dlpack_readonly_is_versioned`
Expected: FAIL — `no method named into_dlpack_readonly`.

- [ ] **Step 3: Update imports and add the versioned export machinery**

In `src/export.rs`, change the FFI import at `src/export.rs:6` from:

```rust
use crate::ffi::{DLDataType, DLDevice, DLManagedTensor, DLTensor};
```

to:

```rust
use crate::ffi::{
    DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor,
    DLPACK_FLAG_BITMASK_READ_ONLY, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
};
```

Add the versioned capsule-name statics after the existing ones at `src/export.rs:156`:

```rust
/// The versioned DLPack capsule name (null-terminated for C compatibility)
static DLPACK_VERSIONED_CAPSULE_NAME: &[u8] = b"dltensor_versioned\0";

/// The name for consumed versioned DLPack capsules (per DLPack protocol)
static USED_DLTENSOR_VERSIONED_NAME: &[u8] = b"used_dltensor_versioned\0";
```

Add the new trait default method inside `trait IntoDLPack`, immediately after `into_dlpack` (after `src/export.rs:137`, before the closing `}` of the trait at `src/export.rs:138`):

```rust
    /// Export this tensor to Python as a **read-only** versioned DLPack capsule.
    ///
    /// Unlike [`into_dlpack`](IntoDLPack::into_dlpack), this emits a versioned
    /// (`dltensor_versioned`) capsule with the read-only flag set, so consumers
    /// that understand DLPack 1.0 know the data must not be modified.
    fn into_dlpack_readonly(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let info = self.tensor_info();
        export_to_capsule_versioned(py, self, info, DLPACK_FLAG_BITMASK_READ_ONLY)
    }
```

Add the versioned export function after `export_to_capsule` (after `src/export.rs:256`):

```rust
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
            DLPACK_VERSIONED_CAPSULE_NAME.as_ptr() as *const i8,
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

    unsafe {
        pyo3::ffi::PyCapsule_SetContext(capsule_ptr, ctx_ptr as *mut c_void);
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
    if name.to_bytes()
        == USED_DLTENSOR_VERSIONED_NAME[..USED_DLTENSOR_VERSIONED_NAME.len() - 1].as_ref()
    {
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib export::tests::test_into_dlpack_readonly_is_versioned`
Expected: PASS.

- [ ] **Step 5: Run the full export suite + clippy**

Run: `cargo test --lib export:: && cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS, no clippy warnings.

- [ ] **Step 6: Commit**

```bash
git add src/export.rs
git commit -m "feat(export): add into_dlpack_readonly versioned capsule path"
```

---

## Task 4: Import — `ManagedPtr` enum representation (refactor)

Pure refactor: `PyTensor` must track which layout it wraps. Existing tests are the safety net. No new behavior.

**Files:**
- Modify: `src/managed.rs` (struct at `src/managed.rs:53-59`, accessors `src/managed.rs:144-249`, `Drop` at `src/managed.rs:252-262`, and the 10 in-test `PyTensor { ... }` literals)

- [ ] **Step 1: Update imports**

In `src/managed.rs`, change the FFI import at `src/managed.rs:6` from:

```rust
use crate::ffi::{DLDataType, DLDevice, DLManagedTensor};
```

to:

```rust
use crate::ffi::{
    DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor,
    DLPACK_FLAG_BITMASK_READ_ONLY, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
};
```

- [ ] **Step 2: Replace the `managed` field with the enum**

In `src/managed.rs`, before the `PyTensor` struct doc comment (before `src/managed.rs:19`), add the enum:

```rust
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
```

Change the `PyTensor` struct field at `src/managed.rs:54` from:

```rust
    managed: NonNull<DLManagedTensor>,
```

to:

```rust
    managed: ManagedPtr,
```

- [ ] **Step 3: Add the `dl_tensor()` helper and route accessors through it**

In `src/managed.rs`, at the top of `impl PyTensor` (immediately after `impl PyTensor {` at `src/managed.rs:65`), add:

```rust
    /// Borrow the embedded `DLTensor`, which lives at a different offset in the
    /// unversioned vs. versioned managed struct.
    fn dl_tensor(&self) -> &DLTensor {
        unsafe {
            match self.managed {
                ManagedPtr::Unversioned(p) => &p.as_ref().dl_tensor,
                ManagedPtr::Versioned(p) => &p.as_ref().dl_tensor,
            }
        }
    }
```

Then replace each accessor body that reads `self.managed.as_ref().dl_tensor`:

- `device` (`src/managed.rs:145-147`):
```rust
    pub fn device(&self) -> DLDevice {
        self.dl_tensor().device
    }
```
- `dtype` (`src/managed.rs:150-152`):
```rust
    pub fn dtype(&self) -> DLDataType {
        self.dl_tensor().dtype
    }
```
- `ndim` (`src/managed.rs:155-157`):
```rust
    pub fn ndim(&self) -> usize {
        self.dl_tensor().ndim as usize
    }
```
- `shape` (`src/managed.rs:162-171`):
```rust
    pub fn shape(&self) -> &[i64] {
        let tensor = self.dl_tensor();
        if tensor.shape.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) }
        }
    }
```
- `strides` (`src/managed.rs:177-189`):
```rust
    pub fn strides(&self) -> Option<&[i64]> {
        let tensor = self.dl_tensor();
        if tensor.strides.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(tensor.strides, tensor.ndim as usize) })
        }
    }
```
- `data_ptr` (`src/managed.rs:219-224`):
```rust
    pub fn data_ptr(&self) -> *mut c_void {
        let tensor = self.dl_tensor();
        unsafe { (tensor.data as *mut u8).add(tensor.byte_offset as usize) as *mut c_void }
    }
```
- `data_ptr_raw` (`src/managed.rs:227-229`):
```rust
    pub fn data_ptr_raw(&self) -> *mut c_void {
        self.dl_tensor().data
    }
```
- `byte_offset` (`src/managed.rs:232-234`):
```rust
    pub fn byte_offset(&self) -> u64 {
        self.dl_tensor().byte_offset
    }
```

- [ ] **Step 4: Update `Drop` to branch on the enum**

Replace the `Drop` impl (`src/managed.rs:252-262`) with:

```rust
impl Drop for PyTensor {
    fn drop(&mut self) {
        // Call the deleter if present, at the correct struct offset for each layout.
        unsafe {
            match self.managed {
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
}
```

- [ ] **Step 5: Update the `from_capsule` construction site**

In `from_capsule` (`src/managed.rs:138-141`), change the returned struct literal from:

```rust
        Ok(Self {
            managed,
            capsule: capsule.clone().unbind(),
        })
```

to:

```rust
        Ok(Self {
            managed: ManagedPtr::Unversioned(managed),
            capsule: capsule.clone().unbind(),
        })
```

- [ ] **Step 6: Update the 10 in-test `PyTensor` literals**

Each test constructs `PyTensor { managed, capsule: capsule.clone().unbind() }` where `managed` is a `NonNull<DLManagedTensor>`. At each of these 10 sites (`src/managed.rs:722`, `:769`, `:800`, `:828`, `:858`, `:888`, `:917`, `:946`, `:978`, `:1011`), change the field `managed,` to `managed: ManagedPtr::Unversioned(managed),`.

The 9 top-level sites share this exact text:

```rust
            let pytensor = PyTensor {
                managed,
                capsule: capsule.clone().unbind(),
            };
```

Change each to:

```rust
            let pytensor = PyTensor {
                managed: ManagedPtr::Unversioned(managed),
                capsule: capsule.clone().unbind(),
            };
```

The 1 nested site (`src/managed.rs:978`) is indented one extra level:

```rust
                let pytensor = PyTensor {
                    managed,
                    capsule: capsule.clone().unbind(),
                };
```

Change it to:

```rust
                let pytensor = PyTensor {
                    managed: ManagedPtr::Unversioned(managed),
                    capsule: capsule.clone().unbind(),
                };
```

- [ ] **Step 7: Run the full managed test suite to verify no regression**

Run: `cargo test --lib managed:: && cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS — all existing tests pass, no clippy warnings. (`DLPACK_FLAG_BITMASK_READ_ONLY`, `DLPACK_MAJOR_VERSION`, `DLPACK_MINOR_VERSION`, and `DLManagedTensorVersioned` are imported but not yet used; clippy may warn about unused imports. If so, that is expected and resolved in Task 5 which uses them. To keep this commit clippy-clean, proceed directly to Task 5 before running clippy, OR temporarily verify with `cargo test --lib managed::` only here.)

- [ ] **Step 8: Commit**

```bash
git add src/managed.rs
git commit -m "refactor(import): represent PyTensor backing as ManagedPtr enum"
```

---

## Task 5: Import — accept versioned capsules, `is_read_only`, version check

**Files:**
- Modify: `src/managed.rs` (`from_capsule` at `src/managed.rs:103-142`, add `is_read_only`, add versioned name static, add round-trip tests)

- [ ] **Step 1: Write the failing round-trip tests**

In `src/managed.rs`, inside `mod tests`, after `test_nbytes_calculation` (around `src/managed.rs:467`), add:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib managed::tests::test_roundtrip_versioned_readonly`
Expected: FAIL — `no method named is_read_only` and `from_capsule` rejects the `dltensor_versioned` name (wrong-name error).

- [ ] **Step 3: Add the versioned used-name static and the crate-root import**

In `src/managed.rs`, change the import at `src/managed.rs:7` from:

```rust
use crate::DLPACK_CAPSULE_NAME;
```

to:

```rust
use crate::{DLPACK_CAPSULE_NAME, DLPACK_VERSIONED_CAPSULE_NAME};
```

Add `CStr` to the std import at `src/managed.rs:10`, changing:

```rust
use std::ffi::{c_char, c_void};
```

to:

```rust
use std::ffi::{c_char, c_void, CStr};
```

After the existing `USED_DLTENSOR_NAME` static (`src/managed.rs:17`), add:

```rust
/// The name for consumed versioned DLPack capsules (per DLPack protocol).
static USED_DLTENSOR_VERSIONED_NAME: &[u8] = b"used_dltensor_versioned\0";
```

- [ ] **Step 4: Rewrite `from_capsule` to dispatch on capsule name**

Replace the entire `from_capsule` method (`src/managed.rs:103-142`) with a dispatcher plus two private helpers:

```rust
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

        // Per DLPack protocol, rename to "used_dltensor" to take ownership and
        // prevent double-consume / double-free.
        let set_name_result = unsafe {
            pyo3::ffi::PyCapsule_SetName(
                capsule.as_ptr(),
                USED_DLTENSOR_NAME.as_ptr() as *const c_char,
            )
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

        // Reject protocol versions newer than we understand, per the DLPack spec:
        // a higher major version may reinterpret the struct layout.
        let version = unsafe { managed.as_ref().version };
        if version.major > DLPACK_MAJOR_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported DLPack version {}.{} (this build supports up to {}.{})",
                version.major, version.minor, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION
            )));
        }

        let set_name_result = unsafe {
            pyo3::ffi::PyCapsule_SetName(
                capsule.as_ptr(),
                USED_DLTENSOR_VERSIONED_NAME.as_ptr() as *const c_char,
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
```

- [ ] **Step 5: Add the `is_read_only` accessor**

In `src/managed.rs`, after the `nbytes` method (after `src/managed.rs:249`, inside `impl PyTensor`), add:

```rust
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
```

- [ ] **Step 6: Run the round-trip tests to verify they pass**

Run: `cargo test --lib managed::tests::test_roundtrip`
Expected: PASS — both `test_roundtrip_versioned_readonly` and `test_roundtrip_unversioned_not_readonly`.

- [ ] **Step 7: Run the full lib suite + clippy**

Run: `cargo test --lib && cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 8: Commit**

```bash
git add src/managed.rs
git commit -m "feat(import): accept versioned capsules and add is_read_only()"
```

---

## Task 6: Import — `max_version` negotiation in `from_pyany`

**Files:**
- Modify: `src/managed.rs` (`from_pyany` at `src/managed.rs:82-92`)

- [ ] **Step 1: Rewrite `from_pyany` to negotiate `max_version` with fallback**

Replace the `from_pyany` body (`src/managed.rs:82-92`) with:

```rust
    pub fn from_pyany(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();

        // Advertise versioned support via max_version. Producers whose
        // __dlpack__ predates the kwarg raise TypeError; fall back to a no-arg
        // call for them. The actual capsule kind is decided later by name.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("max_version", (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))?;

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
```

- [ ] **Step 2: Run the full lib suite to verify no regression**

Run: `cargo test --lib && cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS, no warnings. (Negotiation behavior is exercised by the Python integration tests in Task 7; the Rust unit tests confirm the rewrite compiles and existing `from_capsule` round-trips still pass.)

- [ ] **Step 3: Commit**

```bash
git add src/managed.rs
git commit -m "feat(import): negotiate max_version in from_pyany with legacy fallback"
```

---

## Task 7: Python integration tests + helper functions

**Files:**
- Modify: `tests/python_helpers/src/lib.rs` (add two `#[pyfunction]`s and register them)
- Modify: `tests/test_dlpack_integration.py` (add a versioned/read-only test class)

- [ ] **Step 1: Add the helper `#[pyfunction]`s**

In `tests/python_helpers/src/lib.rs`, after `export_large_cpu_tensor` (after `src/lib.rs:134`), add:

```rust
/// Create a CPU tensor and export it as a **read-only** versioned DLPack capsule.
/// The tensor data is [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] with shape [2, 3].
#[pyfunction]
fn export_cpu_tensor_readonly(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let tensor = TrackedCpuTensor {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape: vec![2, 3],
    };
    tensor.into_dlpack_readonly(py)
}
```

After `import_tensor` and its helpers (a safe spot is right before the `#[pymodule]` block, after `src/lib.rs:563`), add:

```rust
/// Import a tensor via DLPack and report whether it is read-only.
#[pyfunction]
fn import_is_readonly(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let tensor = PyTensor::from_pyany(py, obj)?;
    Ok(tensor.is_read_only())
}
```

- [ ] **Step 2: Register the new functions in the module**

In `tests/python_helpers/src/lib.rs`, in the `#[pymodule]` block, after `export_large_cpu_tensor` registration (`tests/python_helpers/src/lib.rs:575`), add:

```rust
    m.add_function(wrap_pyfunction!(export_cpu_tensor_readonly, m)?)?;
```

After `import_from_capsule` registration (`tests/python_helpers/src/lib.rs:579`), add:

```rust
    m.add_function(wrap_pyfunction!(import_is_readonly, m)?)?;
```

- [ ] **Step 3: Add the Python integration tests**

In `tests/test_dlpack_integration.py`, append a new test class at the end of the file:

```python
# ============================================================================
# Versioned DLPack (DLPack 1.0) / read-only tests
# ============================================================================

class TestVersionedCpu:
    """Test versioned (DLPack 1.0) export/import on the CPU path."""

    def test_readonly_export_is_versioned_capsule(self):
        """A read-only export produces a 'dltensor_versioned' capsule."""
        capsule = dtm.export_cpu_tensor_readonly()
        assert dtm.get_capsule_name(capsule) == "dltensor_versioned"

    @pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
    def test_readonly_capsule_consumed_by_torch(self):
        """torch.from_dlpack consumes our versioned read-only capsule, zero-copy."""
        capsule = dtm.export_cpu_tensor_readonly()
        t = torch.from_dlpack(capsule)
        assert list(t.shape) == [2, 3]
        assert t.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_import_numpy_is_not_readonly(self):
        """A normal numpy array imports as writable (not read-only)."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        assert dtm.import_is_readonly(arr) is False

    @pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
    def test_import_torch_via_versioned_negotiation(self):
        """A torch tensor still imports after max_version negotiation."""
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        info = dtm.import_tensor(t)
        assert info["shape"] == [2, 3]
        assert info["is_cpu"] is True
        assert dtm.import_is_readonly(t) is False
```

- [ ] **Step 4: Build the test module and run the integration tests**

Run:
```bash
maturin build --manifest-path tests/python_helpers/Cargo.toml -o dist
pip install --no-deps --force-reinstall dist/*.whl
pytest tests/test_dlpack_integration.py::TestVersionedCpu -v
```
Expected: PASS (torch tests skipped if torch is not installed).

- [ ] **Step 5: Run the full Python suite to confirm no regression**

Run: `pytest tests/test_dlpack_integration.py -v`
Expected: PASS (or skips for torch/GPU as before).

- [ ] **Step 6: Commit**

```bash
git add tests/python_helpers/src/lib.rs tests/test_dlpack_integration.py
git commit -m "test: versioned/read-only DLPack integration tests"
```

---

## Task 8: Documentation (README + CHANGELOG)

**Files:**
- Modify: `README.md` (after the export example, around the "Supported Data Types" section)
- Modify: `CHANGELOG.md` (the `Unreleased` section)

- [ ] **Step 1: Add a README section on versioned / read-only DLPack**

In `README.md`, after the "Exporting a tensor to Python" example block (before `## Supported Data Types`), add:

```markdown
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
```

- [ ] **Step 2: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add an `### Added` entry (create the heading if the section was merged away; otherwise append to the existing `### Added` list):

```markdown
- Versioned DLPack 1.0 support. Import (`PyTensor::from_pyany` / `from_capsule`)
  now negotiates `max_version` and transparently accepts both `dltensor` and
  `dltensor_versioned` capsules, with a new `PyTensor::is_read_only()` reader.
  Export gains `IntoDLPack::into_dlpack_readonly` for emitting a read-only
  versioned capsule; `into_dlpack` is unchanged (legacy, writable). New FFI
  types `DLManagedTensorVersioned` / `DLPackVersion`, flag-bitmask constants,
  and versioned capsule-name constants are re-exported.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document versioned/read-only DLPack support"
```

---

## Task 9: Final verification

- [ ] **Step 1: Format, lint, and full Rust test run**

Run:
```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib
```
Expected: clean formatting, no clippy warnings, all tests pass.

- [ ] **Step 2: Rebuild the Python module and run the full integration suite**

Run:
```bash
maturin build --manifest-path tests/python_helpers/Cargo.toml -o dist
pip install --no-deps --force-reinstall dist/*.whl
pytest tests/test_dlpack_integration.py -v
```
Expected: all pass (torch/GPU skips as applicable).

- [ ] **Step 3: Confirm semver classification (non-breaking)**

This change is purely additive (new methods, new re-exports; no changed signatures). Confirm with:

Run: `cargo semver-checks check-release` (requires `cargo-semver-checks`; install with `cargo install cargo-semver-checks --locked` if missing)
Expected: reported as a **minor** (non-breaking) change versus the latest crates.io release.

- [ ] **Step 4: Open the PR**

The branch `feat/versioned-dlpack` is ready. Open a PR referencing issue #7 and the spec. Do not merge — wait for CI green and explicit approval.

---

## Self-Review Notes

- **Spec coverage:** FFI types/constants (Task 1) ✓; capsule-name constants (Task 1) ✓; versioned export path + `into_dlpack_readonly` (Task 3) ✓; `into_dlpack` unchanged/legacy (Task 2 keeps it, Task 3 does not touch it) ✓; import dispatch on actual capsule name (Task 5) ✓; `max_version` negotiation with TypeError fallback (Task 6) ✓; `is_read_only` (Task 5) ✓; version-major rejection (Task 5) ✓; `PyTensor` internal kind enum + correct `Drop` offset (Task 4) ✓; ABI layout tests (Task 1) ✓; round-trip + double-free coverage (Tasks 3, 5 via destructor/deleter on drop) ✓; Python integration incl. torch consumption (Task 7) ✓; README + CHANGELOG (Task 8) ✓; backward-compatibility / semver (Task 9) ✓. Out-of-scope items (ExportConfig, IS_COPIED setting, device/copy negotiation, versioned read-write export) are intentionally not implemented.
- **Type/name consistency:** `into_dlpack_readonly`, `is_read_only`, `ManagedPtr::{Unversioned,Versioned}`, `dl_tensor()`, `build_export_parts`, `export_to_capsule_versioned`, `dlpack_deleter_versioned`, `raw_capsule_destructor_versioned`, `DLManagedTensorVersioned`, `DLPackVersion`, `DLPACK_FLAG_BITMASK_READ_ONLY`, `DLPACK_MAJOR_VERSION`, `DLPACK_MINOR_VERSION`, `DLPACK_VERSIONED_CAPSULE_NAME` are used consistently across tasks.
- **Clippy ordering note:** Task 4 imports constants used only in Task 5. If executing strictly task-by-task and running clippy after Task 4, expect unused-import warnings until Task 5 lands; Step 7 of Task 4 documents this. All other tasks are clippy-clean on their own.
```
