# Memory Model

EdgeLearning++ uses one contiguous arena in v0.1. A default `Model` owns an internal static arena sized by `Model::required_memory`.

The compile-time memory breakdown is:

- `parameter_bytes`: weights and biases
- `gradient_bytes`: accumulated parameter gradients
- `optimizer_bytes`: reserved optimizer state, large enough for Adam moments
- `activation_bytes`: input, layer outputs, and requested pre-activations
- `workspace_bytes`: temporary backward buffers
- `total_bytes`: bytes through the last section, including inter-section alignment padding
- `required_memory`: aligned arena size

Metadata such as shapes, layer count, offsets, and policies is represented in types and constants, not stored in the arena.

The planner also accounts for the model precision policy. For example, if `ParameterT` is `double`, `parameter_bytes` is `parameter_count * sizeof(double)`. If a policy keeps activations as `float` but accumulates in a wider type, `activation_bytes` and `workspace_bytes` are computed independently and aligned independently.

External arenas are accepted only through APIs where the byte count is part of the type:

```cpp
std::array<std::byte, N>&
std::span<std::byte, N>
std::byte (&)[N]
```

The constructor statically rejects `N < Model::required_memory`. Alignment is checked at runtime because variable alignment is not encoded in those types. Pass `alignas(Model::alignment)` storage.

This avoids the common embedded failure mode where a model appears to construct correctly and later crashes because a runtime heap allocation fails or a shared RTOS heap is fragmented. The training path binds a statically sized arena once, checks status, and then runs without heap allocation. A `void* + size_t` API would not have the same compile-time property; if added later, it must return `Status` and be documented as runtime-checked only.

A future multi-region planner can split parameters, activations, workspace, and optimizer state across memory regions. In v0.1, users who need DTCM or another region should place the entire arena there.
