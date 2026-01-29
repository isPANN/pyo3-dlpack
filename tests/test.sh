#!/usr/bin/env bash
#
# Test runner for pyo3-dlpack integration tests.
#
# Usage:
#   ./test.sh          # Run all tests (CPU + GPU if available)
#   ./test.sh cpu      # Run only CPU tests
#   ./test.sh gpu      # Run only GPU tests
#   ./test.sh build    # Only build the test module (no tests)
#   ./test.sh clean    # Clean build artifacts
#
# Requirements:
#   - Rust toolchain
#   - Python 3.9+
#   - maturin (pip install maturin)
#   - pytest (pip install pytest)
#   - numpy (pip install numpy)
#   - torch (optional, for PyTorch tests)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Change to project root (parent of tests/)
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking dependencies"

    # Check Rust
    if ! command -v cargo &> /dev/null; then
        print_error "Rust/Cargo not found. Install from https://rustup.rs/"
        exit 1
    fi
    print_success "Rust: $(cargo --version)"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python: $(python3 --version)"

    # Check maturin
    if ! python3 -c "import maturin" &> /dev/null; then
        print_warning "maturin not found, installing..."
        pip install maturin
    fi
    print_success "maturin: installed"

    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        print_warning "pytest not found, installing..."
        pip install pytest
    fi
    print_success "pytest: installed"

    # Check numpy
    if ! python3 -c "import numpy" &> /dev/null; then
        print_warning "numpy not found, installing..."
        pip install numpy
    fi
    print_success "numpy: installed"

    # Check torch (optional)
    if python3 -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        print_success "torch: $TORCH_VERSION"

        # Check CUDA
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &> /dev/null; then
            CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
            DEVICE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
            print_success "CUDA: $CUDA_VERSION ($DEVICE_COUNT device(s))"
            HAS_CUDA=1
        else
            print_warning "CUDA not available (GPU tests will be skipped)"
            HAS_CUDA=0
        fi
    else
        print_warning "torch not found (PyTorch tests will be skipped)"
        print_warning "Install with: pip install torch"
        HAS_CUDA=0
    fi
}

# Build the test module
build_module() {
    print_header "Building test module"

    # Build with maturin in development mode
    maturin develop

    print_success "Test module built successfully"
}

# Run tests
run_tests() {
    local test_type="$1"

    print_header "Running tests"

    local pytest_args="-v"

    case "$test_type" in
        cpu)
            pytest_args="$pytest_args -k 'Cpu or not Gpu'"
            echo "Running CPU tests only..."
            ;;
        gpu)
            if [ "$HAS_CUDA" != "1" ]; then
                print_error "CUDA not available, cannot run GPU tests"
                exit 1
            fi
            pytest_args="$pytest_args -k 'Gpu'"
            echo "Running GPU tests only..."
            ;;
        *)
            echo "Running all tests..."
            ;;
    esac

    python3 -m pytest tests/test_dlpack_integration.py $pytest_args
}

# Clean build artifacts
clean() {
    print_header "Cleaning build artifacts"

    rm -rf target/
    rm -rf tests/python_helpers/target/
    rm -rf *.egg-info/
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    rm -rf tests/__pycache__/
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    find . -name "*.so" -delete

    print_success "Cleaned build artifacts"
}

# Main
main() {
    local command="${1:-all}"

    case "$command" in
        build)
            check_dependencies
            build_module
            ;;
        cpu)
            check_dependencies
            build_module
            run_tests cpu
            ;;
        gpu)
            check_dependencies
            build_module
            run_tests gpu
            ;;
        all)
            check_dependencies
            build_module
            run_tests all
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  all     Run all tests (default)"
            echo "  cpu     Run only CPU tests"
            echo "  gpu     Run only GPU tests"
            echo "  build   Only build the test module"
            echo "  clean   Clean build artifacts"
            echo "  help    Show this help"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
