# Errors

The embedded core does not use exceptions.

Compile-time errors are used for shape, topology, unsupported layer kinds, and insufficient statically sized external arenas.

Runtime user errors return `edge::Status`:

- `Ok`: operation succeeded
- `NullPointer`: runtime pointer or view was null
- `InsufficientArena`: runtime arena byte count is too small
- `UnalignedArena`: arena address does not satisfy `Model::alignment`
- `InvalidBufferLength`: import/export buffer is too small
- `InvalidArgument`: invalid runtime configuration such as batch size zero
- `UnsupportedBackend`: reserved for strict backend dispatch failures
- `NotInitialized`: object has not been bound to valid storage

Internal invariants use `EDGE_ASSERT`, which defaults to `assert`. There is no separate safe/fast mode in v0.1.

The main external arena API is compile-time checked. If a future `void* + size_t` API is added, it must return `Status` and be documented as runtime-checked only.

