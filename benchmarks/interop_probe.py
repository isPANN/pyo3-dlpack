#!/usr/bin/env python3
"""Empirically probe pyo3-dlpack's import negotiation against producers that
expose different DLPack capabilities. Demonstrates graceful legacy fallback."""
import numpy as np
import dlpack_test_module as dtm


class VersionedProducer:
    """A modern producer: __dlpack__ accepts max_version (DLPack 1.0 aware)."""
    def __init__(self, arr):
        self._arr = arr

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        # numpy's own array implements the versioned protocol; delegate to it.
        return self._arr.__dlpack__(max_version=max_version)

    def __dlpack_device__(self):
        return self._arr.__dlpack_device__()


class LegacyProducer:
    """An old producer: __dlpack__ takes NO kwargs (pre-1.0). Passing
    max_version raises TypeError; a correct consumer must fall back."""
    def __init__(self, arr):
        self._arr = arr

    def __dlpack__(self):
        return self._arr.__dlpack__()

    def __dlpack_device__(self):
        return self._arr.__dlpack_device__()


def probe(label, producer):
    try:
        meta = dtm.import_tensor(producer)
        print(f"  {label:18} OK   shape={meta['shape']} nbytes={meta['nbytes']}")
        return True
    except Exception as e:  # noqa: BLE001 - probe reports failures verbatim
        print(f"  {label:18} FAIL {type(e).__name__}: {e}")
        return False


def main():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    print("pyo3-dlpack import negotiation:")
    ok_versioned = probe("versioned", VersionedProducer(arr))
    ok_legacy = probe("legacy (no kwargs)", LegacyProducer(arr))
    print()
    print(f"accepts versioned producer: {ok_versioned}")
    print(f"accepts legacy producer:    {ok_legacy}")
    print("=> single import path handles both (graceful max_version fallback)"
          if ok_versioned and ok_legacy else "=> mixed result, see above")


if __name__ == "__main__":
    main()
