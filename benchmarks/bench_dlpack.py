#!/usr/bin/env python3
"""Simple DLPack benchmarks for pyo3-dlpack.

Usage:
  python benchmarks/bench_dlpack.py --size 10000000 --iters 200
"""

import argparse
import time
import numpy as np

import dlpack_test_module as dtm


def _bench(label, fn, iters):
    # Warmup
    for _ in range(5):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()

    total = end - start
    ns_per = (total / iters) * 1e9
    print(f"{label}: {ns_per:,.0f} ns/op")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000, help="Number of f32 elements")
    parser.add_argument("--iters", type=int, default=200, help="Iterations per benchmark")
    args = parser.parse_args()

    size = args.size
    iters = args.iters

    arr = np.arange(size, dtype=np.float32)
    bytes_total = arr.nbytes
    mb = bytes_total / (1024 * 1024)

    print(f"Array size: {size} elements ({mb:.2f} MiB)")

    def bench_import():
        dtm.import_tensor(arr)

    def bench_export():
        cap = dtm.export_large_cpu_tensor(size)
        class _CapsuleWrapper:
            def __init__(self, capsule):
                self._capsule = capsule

            def __dlpack__(self):
                return self._capsule

        np.from_dlpack(_CapsuleWrapper(cap))

    def bench_copy():
        _ = arr.copy()

    _bench("import_numpy_to_rust", bench_import, iters)
    _bench("export_rust_to_numpy", bench_export, iters)
    _bench("numpy_copy_baseline", bench_copy, iters)


if __name__ == "__main__":
    main()
