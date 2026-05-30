#!/usr/bin/env python3
"""Simple DLPack benchmarks for pyo3-dlpack.

Usage:
  python benchmarks/bench_dlpack.py --size 10000000 --iters 200
"""

import argparse
import resource
import sys
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


def _time_ns(fn, iters):
    for _ in range(5):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return ((time.perf_counter() - start) / iters) * 1e9


def run_compare(sizes, iters):
    """Latency vs size: zero-copy (pyo3-dlpack) vs numpy copy vs torch copy."""
    try:
        import torch
        have_torch = True
    except ImportError:
        have_torch = False

    print(f"{'elements':>12} {'pyo3-dlpack (zc)':>18} {'numpy.copy':>14} {'torch.clone':>14}")
    for size in sizes:
        arr = np.arange(size, dtype=np.float32)

        def zerocopy():
            dtm.import_tensor(arr)

        def npcopy():
            _ = arr.copy()

        zc = _time_ns(zerocopy, iters)
        nc = _time_ns(npcopy, iters)
        if have_torch:
            t = torch.from_numpy(arr)
            tc = _time_ns(lambda: t.clone(), iters)
            tc_s = f"{tc:>12,.0f}ns"
        else:
            tc_s = f"{'n/a':>14}"
        print(f"{size:>12,} {zc:>15,.0f}ns {nc:>11,.0f}ns {tc_s}")


def _peak_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def run_memory(size):
    """Peak RSS: zero-copy import (no duplicate) vs an explicit numpy copy."""
    arr = np.arange(size, dtype=np.float32)
    base = _peak_rss_mb()
    dtm.import_tensor(arr)            # zero-copy: must NOT duplicate the buffer
    after_zc = _peak_rss_mb()
    dup = arr.copy()                  # copy path: duplicates the whole buffer
    after_copy = _peak_rss_mb()
    mb = arr.nbytes / (1024 * 1024)
    print(f"buffer: {mb:.1f} MiB")
    print(f"  baseline peak RSS:        {base:8.1f} MiB")
    print(f"  after zero-copy import:   {after_zc:8.1f} MiB  (+{after_zc-base:.1f})")
    print(f"  after numpy.copy():       {after_copy:8.1f} MiB  (+{after_copy-after_zc:.1f})")
    del dup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000, help="Number of f32 elements")
    parser.add_argument("--iters", type=int, default=200, help="Iterations per benchmark")
    parser.add_argument("--compare", action="store_true", help="latency vs size sweep")
    parser.add_argument("--memory", action="store_true", help="peak-RSS comparison")
    args = parser.parse_args()

    if args.compare:
        run_compare([1_000_000, 10_000_000, 100_000_000], args.iters)
        return
    if args.memory:
        run_memory(50_000_000)
        return

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
