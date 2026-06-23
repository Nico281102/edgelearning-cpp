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

