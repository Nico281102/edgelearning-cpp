# Architecture

Models are defined as compile-time type lists:

```cpp
using Model = edge::Model<
    edge::Input<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;
```

The model infers each layer input dimension from the previous layer. v0.1 supports Dense layers, direct Conv2D layers, and vector-shaped custom layers. Custom layers satisfy the layer concepts in `include/edge/layers/layer_concepts.hpp` and provide compile-time shape and memory traits.

Backend selection is a model-level policy:

```cpp
edge::Model<edge::Backend::Generic, edge::Input<8>, edge::Dense<1>>;
```

Activation policies are semantic policies, not hardware policies. A user writes `Dense<32, ReLU>`; a backend may later specialize that operation for a target. The generic backend uses scalar C++ and supports custom activations.

Precision is a model-level policy. `edge::precision::FP32` is the default, and user policies can provide `ParameterT`, `ActivationT`, `GradientT`, `AccumulatorT`, `OptimizerStateT`, and `LossT`. These aliases feed both the layer signatures and the compile-time memory planner.

The M55 backend is a clean policy and fallback point. Host builds use the generic path. Cortex-M55/MVE float builds can use original EdgeLearning++ FP32 Dense hooks, while unsupported operations fall back to generic kernels.

## Why C++

The C++ version is not a rejection of the C baseline. The C runtime established the important embedded constraints: deterministic memory ownership, explicit arenas, flat parameter layout, and a small training path. C++20 is used here because those same constraints can be expressed more strongly with types.

Genericity is the main architectural gain. Layer specs, activations, precision policies, backends, optimizers, and custom layers are compile-time policies rather than runtime objects. This lets `Model` derive output sizes, parameter counts, cache counts, workspace counts, and arena byte requirements without storing runtime descriptors.

Runtime overhead stays low because the design uses static polymorphism. There are no virtual layer objects, no heap allocation in the training path, and no dynamic graph walk. The model type itself carries the structure, so the compiler can inline layer calls, fold offsets, and specialize activation/backend paths.

The API also becomes more precise. `TensorView<const T, N>` expresses both constness and extent, external arenas use types that carry their byte count, and custom layers fail at compile time if they do not expose the required shape and memory constants. This gives a cleaner user-facing API while keeping embedded failure modes explicit.

Performance should still be measured case by case. The expectation is not that C++ is magically faster than C, but that static polymorphism can compile down to comparable code and sometimes improve on runtime-dispatched C paths by exposing more constants to the optimizer.

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
| Conv2D | Yes | Direct CHW convolution with stride and padding |
| Dropout | No | Planned extension |
| LayerNorm | No | Planned extension |
| Custom layer | Yes | Vector-shaped custom layers |
| Custom activation | Yes | User policies supported |
| Custom loss | Yes | User policies supported |
| Custom precision policy | Yes | Model-level policy supported |
