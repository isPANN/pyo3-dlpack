use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_dlpack::{cpu_device, dtype_f32, IntoDLPack, PyTensor, TensorInfo};
use std::ffi::c_void;
use std::hint::black_box;
use std::sync::Once;

/// A simple owned CPU tensor used as the pyo3-dlpack export source.
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

fn init_python() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        Python::initialize();
    });
}

/// Element counts swept by every benchmark group: 1K (small), 1M, 10M (large).
const SIZES: &[usize] = &[1_000, 1_000_000, 10_000_000];

fn make_vec(size: usize) -> Vec<f32> {
    vec![0.0f32; size]
}

// ----------------------------------------------------------------------------
// Export: hand a Vec<f32> to Python. Zero-copy crates (pyo3-dlpack, dlpark) are
// O(1); copy-based ones (rust-numpy, Vec::clone) are O(n).
// ----------------------------------------------------------------------------

fn bench_export_pyo3_dlpack(c: &mut Criterion) {
    init_python();
    let mut group = c.benchmark_group("export/pyo3-dlpack-legacy");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || TestTensor {
                    data: make_vec(size),
                    shape: vec![size as i64],
                },
                |tensor| {
                    Python::attach(|py| {
                        drop(black_box(tensor.into_dlpack(py).unwrap()));
                    });
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_export_dlpark_legacy(c: &mut Criterion) {
    use dlpark::prelude::*;
    init_python();
    let mut group = c.benchmark_group("export/dlpark-legacy");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || make_vec(size),
                |v| {
                    Python::attach(|py| {
                        let cap = SafeManagedTensor::new(v)
                            .unwrap()
                            .into_pyobject(py)
                            .unwrap();
                        drop(black_box(cap));
                    });
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_export_dlpark_versioned(c: &mut Criterion) {
    use dlpark::prelude::*;
    init_python();
    let mut group = c.benchmark_group("export/dlpark-versioned");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || make_vec(size),
                |v| {
                    Python::attach(|py| {
                        let cap = SafeManagedTensorVersioned::new(v)
                            .unwrap()
                            .into_pyobject(py)
                            .unwrap();
                        drop(black_box(cap));
                    });
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_export_rust_numpy_copy(c: &mut Criterion) {
    use numpy::PyArray1;
    init_python();
    let mut group = c.benchmark_group("export/rust-numpy-copy");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || make_vec(size),
                |v| {
                    Python::attach(|py| {
                        let arr = PyArray1::from_slice(py, &v);
                        drop(black_box(arr));
                    });
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_export_copy_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("export/copy-baseline");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || make_vec(size),
                |v| {
                    black_box(v.clone());
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ----------------------------------------------------------------------------
// Import: consume a DLPack capsule into Rust. Both crates are zero-copy → parity.
// ----------------------------------------------------------------------------

fn bench_import_pyo3_dlpack(c: &mut Criterion) {
    init_python();
    let mut group = c.benchmark_group("import/pyo3-dlpack");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    Python::attach(|py| {
                        let t = TestTensor {
                            data: make_vec(size),
                            shape: vec![size as i64],
                        };
                        t.into_dlpack(py).unwrap()
                    })
                },
                |capsule_obj| {
                    Python::attach(|py| {
                        let capsule = capsule_obj.bind(py).cast::<PyCapsule>().unwrap();
                        drop(black_box(PyTensor::from_capsule(capsule).unwrap()));
                    });
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_import_dlpark(c: &mut Criterion) {
    use dlpark::prelude::*;
    init_python();
    let mut group = c.benchmark_group("import/dlpark");
    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    Python::attach(|py| {
                        SafeManagedTensor::new(make_vec(size))
                            .unwrap()
                            .into_pyobject(py)
                            .unwrap()
                            .unbind()
                    })
                },
                |capsule_obj| {
                    Python::attach(|py| {
                        let capsule = capsule_obj.bind(py);
                        let t = SafeManagedTensor::extract(capsule.into()).unwrap();
                        let s: &[f32] = t.as_slice_contiguous().unwrap();
                        black_box(s);
                    });
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_export_pyo3_dlpack,
    bench_export_dlpark_legacy,
    bench_export_dlpark_versioned,
    bench_export_rust_numpy_copy,
    bench_export_copy_baseline,
    bench_import_pyo3_dlpack,
    bench_import_dlpark,
);
criterion_main!(benches);
