# Current Limitations and Future Work

EdgeLearning++ v0.1 keeps the core training path narrow: static model topology,
explicit arena ownership, no heap allocation in the training path, and
compile-time memory planning. That scope is useful for embedded experiments,
but it limits some higher-level layer interfaces.

## Shape-Aware Sequential Layers

Layer inputs and outputs carry compile-time tensor metadata:

```cpp
TensorSpec<Shape<1, 28, 28>, Layout::CHW>
TensorSpec<Shape<32, 64>, Layout::TimeFeature>
TensorSpec<Shape<8, 16, 32>, Layout::HeadTokenFeature>
```

A custom layer receives that information through `Instance<InputSpec>` and returns its own `output_spec`:

```cpp
template<typename InputSpec>
struct Instance {
    using input_spec = InputSpec;
    using output_spec = edge::Vector<InputSpec::elements>;

    static constexpr std::size_t in_features = input_spec::elements;
    static constexpr std::size_t out_features = output_spec::elements;
    static constexpr std::size_t parameter_count = 0;
    static constexpr std::size_t cache_count = 0;
    static constexpr std::size_t workspace_count = 0;
};
```

The storage is still contiguous. `in_features` and `out_features` remain the flattened extents used by `TensorView`, while `input_spec` and `output_spec` preserve semantic shape and layout for compile-time validation.

With that information, the model computes:

- output shape
- parameter count
- activation cache count
- workspace count
- required arena bytes
- backend dispatch compatibility

all at compile time.

The main v0.1 limitation is topology, not shape vocabulary: models are
sequential single-input/single-output chains. `Dense` requires a flat
`Vector<N>` input, `Conv2D` requires `CHW<C,H,W>`, and `Flatten` explicitly
converts a shaped tensor into a vector before Dense.

## Sequential Model Boundary

The current `edge::Model` is a sequential model:

```cpp
Input -> Layer0 -> Layer1 -> ... -> LayerN -> Output
```

Its planner propagates one tensor spec at a time:

```cpp
CurrentSpec -> Layer::Instance<CurrentSpec> -> output_spec -> next layer
```

Custom layers therefore receive one `InputSpec` and return one `output_spec`.
The limitation is in the model topology contract, not in
`TensorSpec`, `TensorView`, precision policies, or backend policies.

Many graph-like operations can still be implemented today as composite
sequential layers. For example, a residual block can be represented as one layer
that internally computes `Block(input) + input`, as long as the whole block has
one input tensor and one output tensor from the model planner's point of view.

A general graph model would be a separate planner rather than a small extension
of the current sequential recursion. It would need named or indexed nodes,
multiple input edges per node, multiple consumers of the same activation,
gradient accumulation at merge points, topological forward order, reverse
backward order, and activation lifetime analysis. A future `GraphModel` could
reuse the existing layer kernels, tensor specs, precision policies, backend
policies, and static arena rules.

Keeping this boundary in the API lets `Model` stay small and predictable while
graph support can use a DAG planner.

## Input-Gradient Policy

The layer backward contract is parameterized by whether the layer must propagate a gradient to its input:

```cpp
template<bool PropagateInputGradient, typename Types>
static void backward(
    TensorView<const typename Types::ActivationT, in_features> input,
    TensorView<const typename Types::ActivationT, out_features> output,
    TensorView<const typename Types::AccumulatorT, out_features> upstream,
    TensorView<typename Types::AccumulatorT,
               PropagateInputGradient ? in_features : 0U> downstream,
    TensorView<const typename Types::ParameterT, parameter_count> params,
    TensorView<typename Types::GradientT, parameter_count> gradients,
    TensorView<const typename Types::ActivationT, cache_count> cache,
    TensorView<typename Types::AccumulatorT, workspace_count> workspace) noexcept;
```

The model instantiates `backward<false>` for the first layer during ordinary training, because the external input sample is not a trainable quantity. It instantiates `backward<true>` for internal layers that must feed gradients farther backward. Layer implementations should still accumulate parameter gradients in both cases, and should guard downstream writes with `if constexpr (PropagateInputGradient)`.

Future work may add an explicit model-level option to request gradients all the way to the model input for saliency, adversarial-example generation, or using a model as a differentiable sub-block in a larger graph.

## Examples

Conv2D says:

```cpp
using input_spec = TensorSpec<Shape<1, 28, 28>, Layout::CHW>;
using output_spec = TensorSpec<Shape<4, 26, 26>, Layout::CHW>;
```

LayerNorm should be able to say:

```cpp
using input_spec = TensorSpec<Shape<64>, Layout::Flat>;
using output_spec = input_spec;
static constexpr std::size_t parameter_count = 2 * 64; // gamma and beta
static constexpr std::size_t cache_count = 2;           // mean and inverse stddev
```

GRU should be able to expose sequence and hidden-state requirements:

```cpp
using input_spec = TensorSpec<Shape<16, 32>, Layout::TimeFeature>; // 16 steps, 32 features
using output_spec = TensorSpec<Shape<16, 64>, Layout::TimeFeature>;
static constexpr std::size_t state_count = 64;
```

Attention should be able to expose token and head dimensions:

```cpp
using input_spec = TensorSpec<Shape<32, 128>, Layout::TokenFeature>; // 32 tokens, 128 dims
using output_spec = TensorSpec<Shape<32, 128>, Layout::TokenFeature>;
static constexpr std::size_t workspace_count = 32 * 32; // example attention-score buffer
```

These examples are static. If a future layer needs runtime sequence length, the
embedded version should usually expose a compile-time maximum and a runtime
active length:

```cpp
MaxTokens = 64
active_tokens <= MaxTokens
```

The arena is sized for the maximum, while the runtime loop can process the active length.

## State and Multi-Input Layers

Some future layers need more than one input or need state:

- RNN and GRU need hidden state.
- LSTM needs hidden state and cell state.
- Attention may need query, key, value, and mask.
- Residual blocks need a skip input.
- Gating layers may combine multiple tensors.

v0.1 models are primarily sequential single-input/single-output chains. A future graph or multi-input contract should remain static:

```cpp
using inputs = TypeList<QuerySpec, KeySpec, ValueSpec>;
using outputs = TypeList<OutputSpec>;
```

That is future work. For v0.1, keep custom layers single-input/single-output and
use explicit `Flatten` boundaries when moving from shaped tensors to Dense
layers.

## Backend Specialization

Backend specialization should remain a compile-time decision. A shape-rich custom layer should still be able to branch on the backend policy:

```cpp
if constexpr (std::is_same_v<typename Types::BackendT, edge::Backend::M55> &&
              input_spec::layout == Layout::CHW &&
              std::is_same_v<typename Types::ActivationT, float>) {
    // target-specific kernel
}
```

The backend should not discover shape from runtime strings or descriptors. Shape, layout, precision, and memory needs should stay in types and `static constexpr` values.

## Summary

Current v0.1 boundaries:

- Dense and Conv2D are supported as built-in layers.
- Flatten is the explicit boundary from shaped tensors to vector Dense layers.
- Custom layers receive `InputSpec` and expose `output_spec`, but models remain sequential.
- Precision policy, memory planning, shape propagation, and backend dispatch are compile-time.
- Future work is state and multi-input support without losing compile-time arena sizing.
