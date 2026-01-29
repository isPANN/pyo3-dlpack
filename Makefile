# Makefile for pyo3-dlpack
#
# Usage:
#   make test          # Run all tests (unit + integration)
#   make test-unit     # Run Rust unit tests only
#   make test-cpu      # Run CPU integration tests
#   make test-gpu      # Run GPU integration tests
#   make build         # Build the test module
#   make clean         # Clean all artifacts

.PHONY: all build test test-unit test-cpu test-gpu test-integration clean help

# Default target
all: test

# Detect Python command (prefer venv if available)
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

# Build the Python test module from tests/test_module
build:
	@echo "Building test module..."
	maturin develop

# Run all tests
test: test-unit test-integration

# Python library directory (for linking during tests)
PYTHON_LIBDIR := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')")

# Run Rust unit tests
# Set library paths for all platforms: macOS (DYLD), Linux (LD), Windows (PATH)
test-unit:
	@echo "Running Rust unit tests..."
	DYLD_LIBRARY_PATH="$(PYTHON_LIBDIR)" \
	LD_LIBRARY_PATH="$(PYTHON_LIBDIR)" \
	PATH="$(PYTHON_LIBDIR):$(PATH)" \
	cargo test

# Run all integration tests
test-integration: build
	@echo "Running integration tests..."
	$(PYTHON) -m pytest tests/test_dlpack_integration.py -v

# Run CPU integration tests only
test-cpu: build
	@echo "Running CPU integration tests..."
	$(PYTHON) -m pytest tests/test_dlpack_integration.py -v -k "Cpu or not Gpu"

# Run GPU integration tests only
test-gpu: build
	@echo "Running GPU integration tests..."
	$(PYTHON) -m pytest tests/test_dlpack_integration.py -v -k "Gpu"

# Run memory safety tests
test-memory: build
	@echo "Running memory safety tests..."
	$(PYTHON) -m pytest tests/test_dlpack_integration.py -v -k "MemorySafety"

# Run stress tests
test-stress: build
	@echo "Running stress tests..."
	$(PYTHON) -m pytest tests/test_dlpack_integration.py -v -k "Stress"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf target/
	rm -rf tests/test_module/target/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.so" -delete

# Check code
check:
	cargo check
	cargo clippy

# Format code
fmt:
	cargo fmt

# Show help
help:
	@echo "pyo3-dlpack Test Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  test            Run all tests (unit + integration)"
	@echo "  test-unit       Run Rust unit tests only"
	@echo "  test-integration Run Python integration tests"
	@echo "  test-cpu        Run CPU integration tests only"
	@echo "  test-gpu        Run GPU integration tests only"
	@echo "  test-memory     Run memory safety tests"
	@echo "  test-stress     Run stress tests"
	@echo "  build           Build the Python test module"
	@echo "  clean           Clean all build artifacts"
	@echo "  check           Run cargo check and clippy"
	@echo "  fmt             Format Rust code"
	@echo "  help            Show this help"
