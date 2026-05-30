//! Smoke tests that lock the exact dlpark + rust-numpy API surface we benchmark.
//! If a competitor releases a breaking change, THIS fails first — not the bench.
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[test]
fn dlpark_legacy_roundtrips_a_vec() {
    use dlpark::prelude::*;
    Python::initialize();
    Python::attach(|py| {
        let v = vec![1.0f32, 2.0, 3.0];
        let src_ptr = v.as_ptr() as usize;
        let capsule = SafeManagedTensor::new(v).unwrap().into_pyobject(py).unwrap();
        let back = SafeManagedTensor::extract((&capsule).into()).unwrap();
        let slice: &[f32] = back.as_slice_contiguous().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
        assert_eq!(back.data_ptr() as usize, src_ptr);
    });
}

#[test]
fn dlpark_versioned_roundtrips_a_vec() {
    use dlpark::prelude::*;
    Python::initialize();
    Python::attach(|py| {
        let v = vec![4.0f32, 5.0, 6.0];
        let src_ptr = v.as_ptr() as usize;
        let capsule = SafeManagedTensorVersioned::new(v).unwrap().into_pyobject(py).unwrap();
        let cap = capsule.cast::<PyCapsule>().unwrap();
        // SAFETY: dlpark's capsule names are statically allocated string literals.
        assert_eq!(unsafe { cap.name().unwrap().unwrap().as_cstr() }.to_bytes(), b"dltensor_versioned");
        // cast to PyAny: extract() requires Bound<PyAny>
        let back = SafeManagedTensorVersioned::extract((capsule.as_any()).into()).unwrap();
        let slice: &[f32] = back.as_slice_contiguous().unwrap();
        assert_eq!(slice, &[4.0, 5.0, 6.0]);
        assert_eq!(back.data_ptr() as usize, src_ptr);
    });
}

#[test]
fn rust_numpy_from_slice_copies() {
    use numpy::PyArray1;
    use numpy::PyArrayMethods;
    Python::initialize();
    Python::attach(|py| {
        let data = vec![7.0f32, 8.0, 9.0];
        let arr = PyArray1::from_slice(py, &data);
        let np_ptr = arr.data() as usize;
        assert_ne!(np_ptr, data.as_ptr() as usize);
        assert_eq!(arr.readonly().as_slice().unwrap(), &[7.0, 8.0, 9.0]);
    });
}
