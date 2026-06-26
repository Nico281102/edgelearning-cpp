# Current Limitations and Future Work

EdgeLearning++ v0.1 keeps the core training path intentionally small: static model topology, explicit arena ownership, no heap allocation in the training path, and compile-time memory planning. That makes the current implementation useful for embedded experiments, but it also means some higher-level layer interfaces are deliberately limited.

## Custom Layers Are Vector-Shaped

The current custom-layer contract is based on a flattened feature count:

```cpp
template<std::size_t InFeatures>
struct Instance {
    static constexpr std::size_t in_features = InFeatures;
    static constexpr std::size_t out_features = InFeatures;
    static constexpr std::size_t parameter_count = 0;
    static constexpr std::size_t cache_count = 0;
    static constexpr std::size_t workspace_count = 0;
};
```

This works well for layers that naturally operate on vectors:

```cpp
Input<8> -> Dense<16> -> Dense<1>
```

It also works for simple custom layers such as scaling, clipping, elementwise transforms, residual-like vector operations, or any layer where the author is happy to treat the tensor as a flat array.

The limitation is that a flat count does not carry tensor meaning. For example, `784` elements could mean:

```text
784
1 x 28 x 28
28 x 28 x 1
49 tokens x 16 features
```

Those shapes have the same element count, but they do not have the same semantics. A Conv2D layer needs channels, height, width, kernel geometry, stride, padding, and layout. A normalization layer needs to know which axes are normalized. An attention layer needs token count, feature dimension, number of heads, and sometimes a mask layout. A recurrent layer needs hidden-state shape and sequence semantics.

## What v0.1 Already Does

Built-in layers can still encode richer facts internally. `edge::Conv2D` already exposes compile-time geometry such as input channels, input height, input width, output channels, kernel size, stride, padding, output height, output width, parameter count, cache count, and workspace count.

The issue is not that shape-rich layers are impossible. The issue is that the generic custom-layer interface does not yet provide a standard tensor-shape vocabulary for user-defined layers. A user can flatten the data and implement the math, but the framework does not yet help the custom layer express that the flat data is `CHW`, `HWC`, `Time x Feature`, `Token x Feature`, or `Head x Token x Feature`.

## Why This Matters

Shape metadata should remain compile-time. That is the key embedded constraint. The goal is not to copy a dynamic framework where tensor shapes are runtime descriptors. The goal is to let the compiler know more:

```cpp
TensorSpec<Shape<1, 28, 28>, Layout::CHW>
TensorSpec<Shape<32, 64>, Layout::TimeFeature>
TensorSpec<Shape<8, 16, 32>, Layout::HeadTokenFeature>
```

With that information, the model can still compute:

- output shape
- parameter count
- activation cache count
- workspace count
- required arena bytes
- backend dispatch compatibility

all at compile time.

## Proposed Future Direction

A future custom-layer API should move from `Instance<InFeatures>` to an input tensor specification:

```cpp
template<std::size_t... Dims>
struct Shape {};

enum class Layout {
    Flat,
    CHW,
    HWC,
    TimeFeature,
    TokenFeature,
    HeadTokenFeature
};

template<typename ShapeT, Layout LayoutV>
struct TensorSpec {
    using shape = ShapeT;
    static constexpr Layout layout = LayoutV;
    static constexpr std::size_t elements = /* product of dims */;
};

template<typename InputSpec>
struct Instance {
    using input = InputSpec;
    using output = TensorSpec<Shape<4, 26, 26>, Layout::CHW>;

    static constexpr std::size_t parameter_count = /* compile-time */;
    static constexpr std::size_t cache_count = /* compile-time */;
    static constexpr std::size_t workspace_count = /* compile-time */;
};
```

The low-level storage can still be contiguous. The view can still be passed by value. The difference is that the layer receives shape meaning in addition to element count.

## Input-Gradient Policy

The current layer contract can expose two backward entry points:

```cpp
backward(..., downstream, ...);
backward_inputless(...);
```

`backward` computes parameter gradients and writes `dLoss/dInput`. `backward_inputless` computes only the parameter gradients. The model can use the inputless form for the first trainable layer because ordinary supervised training does not need to propagate gradients into the external input sample. This keeps the current API compatible and lets layers opt in to the more memory-efficient path.

A future cleanup could make this contract more explicit with one templated backward function:

```cpp
template<bool NeedInputGradient, typename Types>
static void backward(
    TensorView<const typename Types::ActivationT, in_features> input,
    TensorView<const typename Types::ActivationT, out_features> output,
    TensorView<const typename Types::AccumulatorT, out_features> upstream,
    ConditionalTensorView<NeedInputGradient,
                          typename Types::AccumulatorT,
                          in_features> downstream,
    TensorView<const typename Types::ParameterT, parameter_count> params,
    TensorView<typename Types::GradientT, parameter_count> gradients,
    TensorView<const typename Types::ActivationT, cache_count> cache,
    TensorView<typename Types::AccumulatorT, workspace_count> workspace) noexcept;
```

The model would instantiate `backward<false>` for the first trainable layer and `backward<true>` for earlier layers that must feed gradients farther backward. That is cleaner than maintaining two similarly shaped functions, and it lets the compiler remove the downstream path with `if constexpr`. It is future work because it changes the custom-layer API and needs a migration path for existing layers.

## Examples

Conv2D should be able to say:

```cpp
using input = TensorSpec<Shape<1, 28, 28>, Layout::CHW>;
using output = TensorSpec<Shape<4, 26, 26>, Layout::CHW>;
```

LayerNorm should be able to say:

```cpp
using input = TensorSpec<Shape<64>, Layout::Flat>;
using output = input;
static constexpr std::size_t parameter_count = 2 * 64; // gamma and beta
static constexpr std::size_t cache_count = 2;           // mean and inverse stddev
```

GRU should be able to expose sequence and hidden-state requirements:

```cpp
using input = TensorSpec<Shape<16, 32>, Layout::TimeFeature>; // 16 steps, 32 features
using output = TensorSpec<Shape<16, 64>, Layout::TimeFeature>;
static constexpr std::size_t state_count = 64;
```

Attention should be able to expose token and head dimensions:

```cpp
using input = TensorSpec<Shape<32, 128>, Layout::TokenFeature>; // 32 tokens, 128 dims
using output = TensorSpec<Shape<32, 128>, Layout::TokenFeature>;
static constexpr std::size_t workspace_count = 32 * 32; // example attention-score buffer
```

These examples are intentionally static. If a future layer needs runtime sequence length, the embedded-friendly version should usually expose a compile-time maximum and a runtime active length:

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

That is future work. The current recommendation is to keep v0.1 custom layers vector-shaped and use built-in layers for shape-rich operations that are already supported, such as `Conv2D`.

## Backend Specialization

Backend specialization should remain a compile-time decision. A shape-rich custom layer should still be able to branch on the backend policy:

```cpp
if constexpr (std::is_same_v<typename Types::BackendT, edge::Backend::M55> &&
              input::layout == Layout::CHW &&
              std::is_same_v<typename Types::ActivationT, float>) {
    // target-specific kernel
}
```

The backend should not discover shape from runtime strings or descriptors. Shape, layout, precision, and memory needs should stay in types and `static constexpr` values.

## Summary

The current design is intentionally conservative:

- Dense and Conv2D are supported as built-in layers.
- Custom layers are supported, but the public custom-layer interface is vector-shaped.
- Precision policy, memory planning, and backend dispatch are already compile-time.
- Future work is to add a shape-rich tensor specification, then state and multi-input support, without losing compile-time arena sizing.
