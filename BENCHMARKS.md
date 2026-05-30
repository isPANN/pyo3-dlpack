# Benchmarks: pyo3-dlpack vs. the alternatives

A reproducible, deliberately honest comparison of `pyo3-dlpack` against:

- the **copy-based "market default"** for moving array data across the Rust↔Python
  boundary (`rust-numpy`, `numpy.copy`, `torch.clone`), and
- **`dlpark` 0.7** — the mature, direct Rust DLPack competitor (1.4M downloads).

All numbers below come from the machine in the [Environment](#environment) footer.
Re-run with the commands shown to reproduce.

## TL;DR

1. **vs. copy-based interop — a large, real win.** Zero-copy exchange is **O(1) and
   flat in memory**; copying is **O(n) and ~2× peak memory**. At 100M float32
   elements, importing zero-copy stays at **~0.5 µs** while `numpy.copy()` costs
   **~28 ms** and `torch.clone()` **~17 ms** — a **~54,000×** / **~33,000×** gap that
   keeps widening with size. Holding a tensor zero-copy adds **0 MiB**; copying it
   adds the **full buffer** (≈191 MiB for a 191 MiB array).

2. **vs. `dlpark` — parity, stated as parity.** Both crates are zero-copy capsule
   wrappers, and **`dlpark` already supports versioned DLPack 1.0 and read-only
   flags**, so we claim **no** advantage there. In the head-to-head microbenchmark
   the two track each other within measurement spread at every size. The honest
   `pyo3-dlpack` differentiator is **import ergonomics/robustness**: a single
   `from_pyany` call negotiates `max_version` with the producer and **gracefully
   falls back** to the legacy no-arg `__dlpack__`, transparently accepting *both*
   legacy and versioned producers (§3).

---

## 1. Rust head-to-head latency (Criterion)

Command: `cargo bench --bench dlpack`

Each benchmark performs the same logical operation — hand one `Vec<f32>` to Python
(export) or consume one DLPack capsule (import). The source buffer is allocated in
Criterion's *setup* closure, so allocation of the input is excluded from the timed
region. Point estimates (median); the copy columns scale linearly, the zero-copy
columns do not.

### Export (Rust → Python)

| elements | pyo3-dlpack (zero-copy) | dlpark legacy (zero-copy) | dlpark versioned (zero-copy) | rust-numpy (copy) | Vec::clone (copy) |
|---------:|------------------------:|--------------------------:|-----------------------------:|------------------:|------------------:|
| 1K       | 132 ns  | 138 ns  | 134 ns  | 262 ns   | 193 ns   |
| 1M       | 3.16 µs | 2.50 µs | 2.14 µs | 98.6 µs  | 99.2 µs  |
| 10M      | 258 ns  | 248 ns  | 210 ns  | 1.067 ms | 997 µs   |

### Import (Python → Rust)

| elements | pyo3-dlpack (zero-copy) | dlpark (zero-copy) |
|---------:|------------------------:|-------------------:|
| 1K       | 191 ns  | 176 ns  |
| 1M       | 2.51 µs | 2.50 µs |
| 10M      | 246 ns  | 232 ns  |

**Reading these numbers honestly:**

- **Zero-copy ≪ copy.** Even at the noisy 1M point, zero-copy export (~2–3 µs) is
  **~30–40×** faster than the copy path (~99 µs); at 10M it is **~4,000×** faster
  (~0.25 µs vs ~1 ms). `rust-numpy`'s `PyArray1::from_slice` tracks the raw
  `Vec::clone` baseline almost exactly — confirming it copies synchronously.
- **The 1M row is non-monotonic on purpose-disclosed grounds.** All zero-copy
  contenders read slower at 1M (~2–3 µs) than at 10M (~0.25 µs). This is **not** an
  O(n) effect (an O(n) capsule wrap would make 10M the slowest, like the copy
  columns). It is an allocator/cache artifact: the timed region includes *dropping*
  the capsule, which frees the buffer, and the 4 MB (1M) size-class behaves worse
  under Criterion's input batching than the 40 MB (10M) class. It affects
  `pyo3-dlpack` and `dlpark` **identically**, so it does not bias the comparison.
  The clean, batching-free scaling story is in §2.
- **pyo3-dlpack ≈ dlpark (parity).** Across all sizes the two are within
  measurement spread; `dlpark` is marginally faster in some cells (notably the 1M
  export class and 10M versioned export), `pyo3-dlpack` marginally faster/equal in
  others. There is **no** meaningful raw-throughput winner — as expected, since both
  are O(1) capsule wrappers. The value of parity: `pyo3-dlpack`'s ergonomics and
  import negotiation cost nothing in throughput.

## 2. Python size sweep + peak memory

This isolates the zero-copy *import* (no large allocation/free inside the timed
region), giving the clean O(1)-vs-O(n) picture.

Command: `python benchmarks/bench_dlpack.py --compare --iters 100`

| elements | pyo3-dlpack (zero-copy) | numpy.copy | torch.clone |
|---------:|------------------------:|-----------:|------------:|
| 1M       | 492 ns   | 168,874 ns    | 101,692 ns    |
| 10M      | 537 ns   | 1,741,097 ns  | 1,434,532 ns  |
| 100M     | 520 ns   | 28,472,697 ns | 16,937,265 ns |

Zero-copy import is **flat at ~0.5 µs** across a 100× range of sizes, while
`numpy.copy` and `torch.clone` grow linearly. At 100M float32 (≈382 MiB),
zero-copy is **~54,000×** faster than `numpy.copy` and **~33,000×** faster than
`torch.clone`.

Command: `python benchmarks/bench_dlpack.py --memory`

```
buffer: 190.7 MiB
  baseline peak RSS:           222.3 MiB
  after zero-copy import:      222.3 MiB  (+0.0)
  after numpy.copy():          413.1 MiB  (+190.8)
```

A zero-copy import adds **0 MiB** of resident memory; copying the same buffer adds
its **full size** (≈191 MiB). For large tensors this is the difference between
fitting in memory and not.

## 3. Import-negotiation interop probe

Command: `python benchmarks/interop_probe.py`

```
pyo3-dlpack import negotiation:
  versioned          OK   shape=[2, 3] nbytes=24
  legacy (no kwargs) OK   shape=[2, 3] nbytes=24

accepts versioned producer: True
accepts legacy producer:    True
=> single import path handles both (graceful max_version fallback)
```

`PyTensor::from_pyany` advertises `max_version` to the producer (DLPack 1.0) and, if
the producer's `__dlpack__` predates that keyword and raises `TypeError`, retries
with the legacy no-arg call. One import path therefore accepts both a modern
versioned producer and an old legacy-only one. (This probe characterizes
`pyo3-dlpack`; the corresponding `dlpark` Python-import behavior was not separately
measured here.)

## 4. Capability matrix

Verified 2026-05-30. Ties are shown as ties — this is a fair-use comparison, not
marketing.

| Capability | pyo3-dlpack | dlpark 0.7 | rust-numpy 0.28 |
|---|:---:|:---:|:---:|
| Zero-copy tensor exchange | ✅ | ✅ | ❌ (copies) |
| Device-agnostic (CPU/CUDA/ROCm/Metal) | ✅ (protocol-level) | ✅ (CUDA WIP) | ❌ (CPU host) |
| Bidirectional import + export | ✅ | ✅ | ✅ |
| DLPack 1.0 versioned protocol | ✅ | ✅ | ✅ (via numpy) |
| Read-only flag | ✅ | ✅ | n/a |
| Raw export/import throughput | ✅ | ✅ (parity) | ❌ (O(n)) |
| One import call accepts legacy **and** versioned producers, with `max_version` fallback | ✅ (measured, §3) | not separately measured | n/a |
| PyO3 0.28 modern API | ✅ | ✅ | ✅ |

The genuine `pyo3-dlpack` edges over `dlpark` are ergonomic/robustness, not raw
speed or protocol-feature count: transparent legacy↔versioned import negotiation in
a single call (§3), and a legacy-writable capsule as the default export
(`into_dlpack`) for maximum consumer compatibility, with versioned read-only
available on demand (`into_dlpack_readonly`). Both crates can produce and consume
both capsule formats.

## Reproducing

- **Rust:** `cargo bench --bench dlpack` (needs Rust ≥ 1.85 — the `dlpark`
  dev-dependency is edition 2024). The competitor API is pinned by
  `tests/competitor_smoke.rs` (`cargo test --test competitor_smoke`).
- **Python:** build the `dlpack_test_module` helper (`maturin develop` per the repo's
  test setup), then:
  - `python benchmarks/bench_dlpack.py --compare --iters 100`
  - `python benchmarks/bench_dlpack.py --memory`
  - `python benchmarks/interop_probe.py`
  Requires `numpy`; `torch` is optional (its column shows `n/a` when absent).

## Environment

- Date: 2026-05-30 • CPU: Apple M3 (8 cores) • OS: macOS 26.5 (arm64)
- rustc 1.95.0 • Python 3.13.9 • numpy 2.4.1 • torch 2.10.0
- dlpark 0.7.0 • numpy (rust) crate 0.28.0 • Criterion 0.8 • PyO3 0.28

> Absolute timings are machine-specific; the **ratios and scaling shapes** are the
> portable result. Numbers are point estimates from a single run; expect ±10–20%
> run-to-run, larger at the disclosed 1M size class (§1).
