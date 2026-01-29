"""
Pytest configuration for pyo3-dlpack integration tests.
"""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "cpu: mark test as CPU-only")
    config.addinivalue_line("markers", "gpu: mark test as requiring CUDA GPU")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """Add markers based on test class names."""
    for item in items:
        # Add gpu marker to GPU test classes
        if "Gpu" in item.nodeid or "gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        # Add cpu marker to CPU test classes
        elif "Cpu" in item.nodeid or "cpu" in item.nodeid:
            item.add_marker(pytest.mark.cpu)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests (requires CUDA)",
    )
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Run only CPU tests",
    )
