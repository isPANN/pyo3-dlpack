# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on `pyo3`:** `pyo3` is a *public dependency* — its types (`Python`,
> `Bound<'_, PyAny>`, `PyResult`, `PyCapsule`, …) appear in this crate's public
> API. Your project must use the **same `pyo3` minor version** as `pyo3-dlpack`,
> and every breaking `pyo3` release forces a breaking release here.
>
> | pyo3-dlpack | pyo3   |
> | ----------- | ------ |
> | 0.2.x       | 0.28   |
> | 0.1.x       | 0.27   |

## [Unreleased]

### Security
- Raised the `pytest` floor in the `test`/`test-lite` extras from `>=7.0` to
  `>=9.0.3` to exclude versions affected by CVE-2025-71176 (tmpdir handling)
  and pull in a patched `Pygments` (CVE-2026-4539). Test-only dependencies —
  not shipped with the published crate.

### Added
- Versioned DLPack 1.0 support. Import (`PyTensor::from_pyany` / `from_capsule`)
  now negotiates `max_version` and transparently accepts both `dltensor` and
  `dltensor_versioned` capsules, with a new `PyTensor::is_read_only()` reader.
  Export gains `IntoDLPack::into_dlpack_readonly` for emitting a read-only
  versioned capsule; `into_dlpack` is unchanged (legacy, writable). New FFI
  types `DLManagedTensorVersioned` / `DLPackVersion`, flag-bitmask constants,
  and versioned capsule-name constants are re-exported.
- `CHANGELOG.md` following the Keep a Changelog format.
- CI: MSRV job (Rust 1.83) verifying the crate builds on the declared
  `rust-version`.
- CI: `cargo-semver-checks` job that compares the API surface against the
  latest crates.io release and fails on undeclared breaking changes.

## [0.2.0] - 2026-05-29

### Changed
- **Breaking:** upgraded `pyo3` from `0.27` to `0.28` and adapted to its
  breaking changes. Because `pyo3` is a public dependency, consumers must
  upgrade their own `pyo3` to `0.28` in lockstep.
- **Breaking:** raised the minimum supported Rust version (MSRV) from `1.70`
  to `1.83`.

### Fixed
- Use `div_ceil` for itemsize calculation (clippy lint).

### Internal
- Added Dependabot configuration for `cargo` and `github-actions`.

## [0.1.0] - 2026-01-30

### Added
- Initial release: zero-copy DLPack tensor interop for PyO3 (`pyo3` 0.27).
- Import tensors from Python via `PyTensor::from_pyany` / `from_capsule`,
  with capsule lifetime management and double-free protection (`dltensor` →
  `used_dltensor` rename on consume).
- Export Rust tensors to Python via the `IntoDLPack` trait and `TensorInfo`
  (contiguous and strided layouts, byte-offset support).
- `#[repr(C)]` FFI types mirroring the DLPack ABI (`DLTensor`,
  `DLManagedTensor`, `DLDevice`, `DLDataType`) with struct-layout tests.
- Convenience constructors for dtypes (f16/f32/f64/bf16, i8–i64, u8–u64, bool)
  and devices (CPU, CUDA, CUDA host, ROCm, Metal, Vulkan, …).
- Rust (criterion) and Python benchmarks; unit + integration test suite.

[Unreleased]: https://github.com/isPANN/pyo3-dlpack/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/isPANN/pyo3-dlpack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/isPANN/pyo3-dlpack/releases/tag/v0.1.0
