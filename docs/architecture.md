# Architecture

Models are defined as compile-time type lists:

```cpp
using Model = edge::Model<
    edge::Input<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;
```

The model infers each Dense input dimension from the previous layer. v0.1 supports Dense layers only. Future layers should satisfy the layer concepts in `include/edge/layers/layer_concepts.hpp` and provide compile-time shape and memory traits.

Backend selection is a model-level policy:

```cpp
edge::Model<edge::Backend::Generic, edge::Input<8>, edge::Dense<1>>;
```

Activation policies are semantic policies, not hardware policies. A user writes `Dense<32, ReLU>`; a backend may later specialize that operation for a target. The generic backend uses scalar C++ and supports custom activations.

The v0.1 M55 backend is a clean tag and fallback point. It intentionally does not include vendor headers or post-baseline optimized code.

## Project Layout

The directory layout follows a C++ library structure rather than the original C runtime levels:

- `include/edge/`: public template API and header-only runtime core
- `include/edge/layers/`: layer specifications and layer concepts
- `include/edge/optimizers/`: optimizer policies
- `src/`: optional future non-template backend helpers
- `tests/`: host CTest executables
- `examples/`: small user-facing examples
- `benchmarks/`: host timing and regression methodology
- `docs/`: architecture, memory, error, extension, and provenance notes

The reason is practical: the model topology is encoded in C++ types, so most core code must stay visible to the compiler for shape inference, inlining, and compile-time memory planning. The original C layering was useful for a procedural runtime with runtime descriptors; the C++ version keeps those responsibilities, but expresses them as headers, policies, concepts, and tests.

This can still be improved. A future cleanup could split the public API into clearer subfolders such as `core/`, `backends/`, `training/`, and `serialization/` while preserving the single umbrella header `edge/edge.hpp`.

## Layer Support

| Layer or extension | Supported in v0.1 | Notes |
|---|---:|---|
| Dense | Yes | Fully implemented |
| Conv2D | No | Planned extension |
| Dropout | No | Planned extension |
| LayerNorm | No | Planned extension |
| Custom layer | Not yet | Concepts exist, but model execution is Dense-only |
| Custom activation | Yes | User policies supported |
| Custom loss | Yes | User policies supported |
