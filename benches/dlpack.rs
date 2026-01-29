use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_dlpack::{cpu_device, dtype_f32, IntoDLPack, PyTensor, TensorInfo};
use std::ffi::c_void;
use std::hint::black_box;
use std::sync::Once;

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

fn bench_export_small(c: &mut Criterion) {
    init_python();
    let size = 1_000usize;

    c.bench_function("export_capsule_1k", |b| {
        b.iter_batched(
            || TestTensor {
                data: vec![0.0; size],
                shape: vec![size as i64],
            },
            |tensor| {
                Python::attach(|py| {
                    drop(tensor.into_dlpack(py).unwrap());
                });
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_export_large(c: &mut Criterion) {
    init_python();
    let size = 1_000_000usize;

    c.bench_function("export_capsule_1m", |b| {
        b.iter_batched(
            || TestTensor {
                data: vec![0.0; size],
                shape: vec![size as i64],
            },
            |tensor| {
                Python::attach(|py| {
                    drop(tensor.into_dlpack(py).unwrap());
                });
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_import_large(c: &mut Criterion) {
    init_python();
    let size = 1_000_000usize;

    c.bench_function("import_capsule_1m", |b| {
        b.iter_batched(
            || {
                Python::attach(|py| {
                    let tensor = TestTensor {
                        data: vec![0.0; size],
                        shape: vec![size as i64],
                    };
                    tensor.into_dlpack(py).unwrap()
                })
            },
            |capsule_obj| {
                Python::attach(|py| {
                    let capsule = capsule_obj.bind(py).cast::<PyCapsule>().unwrap();
                    drop(PyTensor::from_capsule(capsule).unwrap());
                });
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_copy_baseline(c: &mut Criterion) {
    let size = 1_000_000usize;
    let data = vec![0.0f32; size];

    c.bench_function("vec_clone_1m", |b| {
        b.iter(|| {
            let cloned = data.clone();
            black_box(cloned);
        })
    });
}

criterion_group!(
    benches,
    bench_export_small,
    bench_export_large,
    bench_import_large,
    bench_copy_baseline
);
criterion_main!(benches);
