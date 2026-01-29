# pyo3-dlpack Tests

This directory contains all tests for the pyo3-dlpack library.

## Structure

```
tests/
├── python_helpers/                 # Python extension for integration testing
│   ├── Cargo.toml                  # Rust crate config
│   └── src/lib.rs                  # Test module implementation
├── conftest.py                     # pytest configuration
├── test_dlpack_integration.py      # Python integration tests
└── README.md                       # This file
```

## Test Types

### 1. Unit Tests (in `src/`)
Rust unit tests are embedded in the source files using `#[cfg(test)]`:
- `src/export.rs` - Export path tests
- `src/managed.rs` - Import path tests
- `src/ffi.rs` - FFI type tests

Run with:
```bash
cargo test
```

### 2. Integration Tests (in `tests/`)
Python integration tests that verify the library works correctly with real Python frameworks (NumPy, PyTorch):

**Test Coverage:**
- ✅ Import path: Python tensor → DLPack capsule → Rust `PyTensor`
- ✅ CPU tensors (contiguous, non-contiguous, various dtypes)
- ✅ GPU tensors (CUDA, if available)
- ✅ Capsule ownership protocol (rename to "used_dltensor")
- ✅ Memory safety (no double-free, no use-after-free)
- ✅ Stress tests (many imports, large tensors)

Run with:
```bash
make test-integration
# or
cd tests && ./test.sh cpu
cd tests && ./test.sh gpu    # requires CUDA
```

## Test Module

`python_helpers/` is a separate Rust crate that compiles to a Python extension module (`dlpack_test_module`). It provides:

- **Export functions**: Create test tensors and export as DLPack capsules
- **Import functions**: Import capsules and return metadata for verification
- **Counter functions**: Track drops/deletes for memory safety tests
- **Inspection functions**: Check capsule state (consumed or not)

This module is only used for testing and is not part of the main library distribution.

## Running Tests

### All tests
```bash
make test
```

### Unit tests only
```bash
make test-unit
```

### Integration tests only
```bash
make test-integration
```

### CPU tests only
```bash
make test-cpu
```

### GPU tests only (requires CUDA)
```bash
make test-gpu
```

### Memory safety tests
```bash
make test-memory
```

### Stress tests
```bash
make test-stress
```

## Benchmarks

### Rust benchmarks (Criterion)
```bash
make bench-rust
```

### Python benchmarks
```bash
make bench-python
```

### All benchmarks
```bash
make bench
```

## Requirements

- Python 3.9+
- maturin
- pytest
- numpy
- torch (optional, for PyTorch tests)
- CUDA (optional, for GPU tests)
