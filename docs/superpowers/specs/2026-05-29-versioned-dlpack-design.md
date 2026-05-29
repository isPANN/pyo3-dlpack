# Design: Versioned DLPack (DLPack 1.0) support

**Date:** 2026-05-29
**Status:** Approved (design), pending implementation plan
**Tracking issue:** #7

## Goal

Add support for the **versioned** DLPack 1.0 protocol
(`DLManagedTensorVersioned`, capsule name `dltensor_versioned`) on both the
import and export sides, while keeping the legacy unversioned path working
unchanged. The one user-visible new capability is **read-only tensors**: the
ability to mark an exported tensor as read-only, and to detect read-only on
import.

## Guiding principle (non-negotiable)

`pyo3-dlpack`'s differentiator versus `dlpark` is that it is **simple, direct,
and API-stable**. This feature must not erode that:

- **Minimal new surface.** Users learn at most a couple of new, self-explanatory
  method names — no configuration objects, no flag bitmasks exposed at the API
  level.
- **No churn.** Existing methods keep their names and behavior. Future protocol
  additions, if ever needed, arrive as new self-explanatory methods, never as
  changes to existing signatures.
- **Automatic where possible.** Protocol-version negotiation is invisible to the
  user; the library handles old and new capsules transparently.

When a design choice trades "more flexible" against "simpler/stabler," choose
simpler.

## Background (plain terms)

A DLPack exchange passes a *claim ticket* (metadata: data pointer, shape, dtype,
device) instead of copying the tensor data — that is the zero-copy mechanism.
The ticket travels inside a Python `PyCapsule`. Producers expose it via
`__dlpack__()`; consumers take ownership and rename the capsule `dltensor` →
`used_dltensor` to prevent double-free.

DLPack 1.0 ("versioned") adds two things to the ticket:

1. A **version stamp** (`major.minor`) so producer and consumer can detect
   format mismatches instead of silently misreading memory. (Pure safety.)
2. A **flags** bitmask. The one that matters here is `READ_ONLY` ("look but
   don't modify"), which the legacy protocol cannot express.

The versioned struct is **not** the legacy struct plus fields — the field order
differs (`version` first, `dl_tensor` last) and the deleter takes a pointer to
the versioned struct. This must be handled as a distinct ABI type, not a
superset.

Today nothing is broken: PyTorch/NumPy/JAX still default `__dlpack__()` (no
args) to a legacy capsule. This work closes the forward-compatibility gap and
adds read-only.

## Public API (the complete set of new user-facing things)

| Scenario | Call |
| --- | --- |
| Export (writable — unchanged) | `tensor.into_dlpack(py)` |
| Export (read-only — new) | `tensor.into_dlpack_readonly(py)` |
| Import (automatic — unchanged) | `PyTensor::from_pyany(py, obj)` |
| Query read-only after import (new) | `tensor.is_read_only()` |

That is the entire new surface at the API level. No `ExportConfig`, no exposed
flag enum, no boolean parameters. Existing code compiles and behaves exactly as
before.

### Export

- `into_dlpack(self, py)` — **unchanged**. Continues to emit a **legacy**
  `dltensor` capsule (read-write). Keeping it legacy maximizes compatibility
  with consumers that only understand the old format, and guarantees zero
  behavior change for existing users.
- `into_dlpack_readonly(self, py)` — **new** default method on the `IntoDLPack`
  trait. Emits a **versioned** `dltensor_versioned` capsule with
  `version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}` and
  `flags = READ_ONLY`. Read-only is a 1.0 feature, so
  this path is inherently versioned. Same `TensorInfo` source as `into_dlpack`;
  the only difference is the capsule kind and the flag.

Because `into_dlpack_readonly` is a **default trait method**, adding it does not
break any existing `impl IntoDLPack for ...`.

### Import

- `from_pyany(py, obj)` — **unchanged signature and usage**. Internally:
  1. Call `__dlpack__(max_version=(1, MINOR))` to advertise versioned support.
  2. If the producer's `__dlpack__` does not accept the kwarg (older
     producers raise `TypeError`), fall back to a no-arg `__dlpack__()` call.
  3. Dispatch on the **actual returned capsule name** (`dltensor` vs
     `dltensor_versioned`) — never assume, because a producer may return a
     legacy capsule even when versioned was requested.
- `from_capsule(capsule)` — accepts **both** capsule kinds. On consume, renames
  to the matching `used_*` name (`used_dltensor` / `used_dltensor_versioned`),
  preserving the existing double-free protection for each kind.
- `is_read_only(&self) -> bool` — **new** reader. Returns the `READ_ONLY` flag
  for versioned tensors; returns `false` for legacy tensors (which cannot
  express it).

`PyTensor` gains an internal record of which capsule kind it holds (an enum, not
public) so that `Drop` calls the deleter at the correct struct offset. This is
an internal change; the public `PyTensor` accessors (`shape`, `dtype`, `device`,
`data_ptr`, etc.) are unaffected because they read the embedded `DLTensor`,
which is identical in both layouts.

## FFI additions (`src/ffi.rs`)

Internal/ABI types, re-exported from `lib.rs` for completeness but not part of
the everyday API:

- `DLPackVersion { major: u32, minor: u32 }`.
- `DLManagedTensorVersioned` with the correct 1.0 field order:
  `version`, `manager_ctx`, `deleter`, `flags`, `dl_tensor`.
- Deleter typedef taking `*mut DLManagedTensorVersioned`.
- Flag constants: at minimum `DLPACK_FLAG_BITMASK_READ_ONLY` (1 << 0). Define
  the others (`IS_COPIED` = 1 << 1, `IS_SUBBYTE_TYPE_PADDED` = 1 << 2) as
  constants for completeness, but the API does not act on them yet (YAGNI).
- Version constants: `DLPACK_MAJOR_VERSION`, `DLPACK_MINOR_VERSION`.
- Capsule-name constants: `dltensor_versioned` / `used_dltensor_versioned`,
  alongside the existing `dltensor` / `used_dltensor`.

## Export internals (`src/export.rs`)

The versioned path needs its own monomorphized deleter and capsule destructor
because the struct layout and deleter signature differ from legacy:

- A versioned deleter that interprets `manager_ctx`/`deleter` at the versioned
  offsets and frees the boxed export context.
- A versioned capsule destructor mirroring `raw_capsule_destructor` but checking
  `dltensor_versioned` / `used_dltensor_versioned` for the consumed/unconsumed
  decision.

The two paths share the `TensorInfo` → `DLTensor` construction; only the wrapper
struct, deleter, destructor, and capsule name differ.

## Backward compatibility

- `into_dlpack`, `from_pyany`, `from_capsule` keep their signatures and observed
  behavior. Existing capsules and existing user code are unaffected.
- New trait method is a default method (no breakage for implementers).
- This is an **additive, non-breaking** change under semver. Under 0.x it can
  ship in a minor release; `cargo-semver-checks` (PR #8) should classify it as
  non-breaking.

## Testing

Mirror the existing unversioned test suite for the versioned path:

- **ABI layout tests** for `DLManagedTensorVersioned` (size + field order),
  matching the style of the existing `DLManagedTensor` layout tests in
  `ffi.rs`. This is the #1 regression guard for an FFI change.
- **Round-trip tests**: export read-only → import → `is_read_only()` is `true`;
  export normal → `is_read_only()` is `false`.
- **Capsule lifetime / double-free**: versioned capsule renamed to
  `used_dltensor_versioned` on consume; deleter not called twice.
- **Import negotiation**: a producer that accepts `max_version` returns
  versioned and is handled; a producer that rejects the kwarg falls back to the
  no-arg call and the legacy path; a producer that returns legacy despite
  `max_version` is dispatched correctly by capsule name.
- **Python integration**: verify a real framework (`torch.from_dlpack`) consumes
  the versioned read-only capsule, and that a versioned capsule produced by a
  framework imports correctly.

## Out of scope (YAGNI)

- No `ExportConfig` / builder / flag-bitmask user API.
- No user-facing setting of `IS_COPIED` or other flags on export (constants
  defined, not acted upon).
- No `dl_device` / `copy` negotiation kwargs on import (device-transfer
  requests). Can be added later as a separate, self-explanatory method if a real
  need appears.
- No versioned *read-write* export path. If needed later, add a third
  self-explanatory method rather than parameterizing existing ones.

## Documentation

- README: a short note explaining versioned vs unversioned, that negotiation is
  automatic, and the read-only export/import methods.
- CHANGELOG: an `Added` entry under `Unreleased`.
