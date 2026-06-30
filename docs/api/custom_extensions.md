# Custom Extensions

## Activation

A custom activation declares its storage requirement and operations:

```cpp
struct Swish {
    static constexpr edge::ActivationStorage storage =
        edge::ActivationStorage::OutputAndPreActivation;

    template<typename T>
    static T forward(T z);

    template<typename T>
    static T derivative(T z, T a);
};
```

The generic backend supports custom activations. If the derivative can be computed from output only, provide `derivative_from_output(a)` and use `OutputOnly`.

## Loss

A loss can provide a value-only path and a value-plus-gradient path:

```cpp
struct MyLoss {
    template<typename Prediction, typename Target>
    static float value(const Prediction& y, const Target& t);

    template<typename Prediction, typename Target, typename Gradient>
    static float evaluate(const Prediction& y, const Target& t, Gradient& g);
};
```

The trainer uses `evaluate` to avoid a second pass.

The return type may be `float`, `double`, or another arithmetic type compatible with the model loss type. Built-in `MSE` computes from the prediction and target value types instead of forcing `float`.

## Initializer

An initializer provides:

```cpp
template<typename T, std::size_t In, std::size_t Out>
static void fill(T* weights, edge::DeterministicRng& rng, const edge::InitConfig& config);
```

Bias values are filled by the model from `InitConfig::bias`.

## Precision Policy

A precision policy is a model-level bundle of arithmetic types:

```cpp
struct DoublePrecision {
    using ParameterT = double;
    using ActivationT = double;
    using GradientT = double;
    using AccumulatorT = double;
    using OptimizerStateT = double;
    using LossT = double;
};

using Model = edge::Model<DoublePrecision, edge::InputVector<2>, edge::Dense<1>>;
```

The policy drives both APIs and memory planning. `Model::parameter_data()` returns `ParameterT*`, `Model::forward()` accepts `ActivationT`, and `Model::required_memory` includes the exact section sizes and alignments implied by the policy.

## Custom Layer

Custom layers provide a nested instance specialized by the inferred input tensor spec:

```cpp
struct MyLayer {
    template<typename InputSpec>
    struct Instance {
        using input_spec = InputSpec;
        using output_spec = edge::Vector<InputSpec::elements>;

        static constexpr std::size_t in_features = input_spec::elements;
        static constexpr std::size_t out_features = output_spec::elements;
        static constexpr std::size_t parameter_count = 0;
        static constexpr std::size_t cache_count = 0;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void initialize(edge::TensorView<typename Types::ParameterT, parameter_count>,
                               edge::DeterministicRng&,
                               const edge::InitConfig&) noexcept;

        template<typename Types>
        static void forward(edge::TensorView<const typename Types::ActivationT, in_features>,
                            edge::TensorView<typename Types::ActivationT, out_features>,
                            edge::TensorView<const typename Types::ParameterT, parameter_count>,
                            edge::TensorView<typename Types::ActivationT, cache_count>,
                            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept;

        template<bool PropagateInputGradient, typename Types>
        static void backward(edge::TensorView<const typename Types::ActivationT, in_features>,
                             edge::TensorView<const typename Types::ActivationT, out_features>,
                             edge::TensorView<const typename Types::AccumulatorT, out_features>,
                             edge::TensorView<typename Types::AccumulatorT,
                                              PropagateInputGradient ? in_features : 0U>,
                             edge::TensorView<const typename Types::ParameterT, parameter_count>,
                             edge::TensorView<typename Types::GradientT, parameter_count>,
                             edge::TensorView<const typename Types::ActivationT, cache_count>,
                             edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept;
    };
};
```

For a vector-only layer, add a compile-time layout check:

```cpp
static_assert(InputSpec::layout == edge::Layout::Flat);
```

Forward writes the layer output. Backward receives `dLoss/dOutput` and always accumulates parameter gradients. It writes `dLoss/dInput` only when `PropagateInputGradient` is `true`:

```cpp
if constexpr (PropagateInputGradient) {
    downstream[i] = /* dLoss/dInput[i] */;
}
```

The model sets `PropagateInputGradient` to `false` for the first layer during ordinary training because there is normally no need to propagate a gradient into the external input sample. Internal layers receive `true` so their downstream gradient can feed the previous layer. Use `TensorView<const T, N>` for read-only inputs and `TensorView<T, N>` for outputs. The view is passed by value because it is only a typed pointer plus compile-time extent.

## Backend-Specific Layer Paths

Backend specialization is selected through the model type, not through runtime virtual classes:

```cpp
using Model = edge::Model<
    edge::Backend::M55,
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;
```

The layer receives that policy through the `Types` template parameter:

```cpp
template<typename Types>
static void forward(...);
```

`Types::BackendT` is the selected backend. `Types::ParameterT`, `Types::ActivationT`, `Types::GradientT`, and `Types::AccumulatorT` are the selected precision types. A backend-specific fast path should check all of the assumptions that the optimized kernel requires:

```cpp
template<typename Types>
static void forward(edge::TensorView<const typename Types::ActivationT, in_features> input,
                    edge::TensorView<typename Types::ActivationT, out_features> output,
                    edge::TensorView<const typename Types::ParameterT, parameter_count> params,
                    edge::TensorView<typename Types::ActivationT, cache_count> cache,
                    edge::TensorView<typename Types::AccumulatorT, workspace_count> workspace) noexcept {
    using ActivationT = typename Types::ActivationT;
    using AccumulatorT = typename Types::AccumulatorT;

    if constexpr (std::is_same_v<typename Types::BackendT, edge::Backend::M55> &&
                  std::is_same_v<typename Types::ParameterT, float> &&
                  std::is_same_v<ActivationT, float> &&
                  std::is_same_v<AccumulatorT, float>) {
        if (m55_my_layer_forward<in_features, out_features>(
                input, output, params, cache)) {
            return;
        }
    }

    generic_my_layer_forward(input, output, params, cache, workspace);
}
```

`if constexpr` is a compile-time branch. When the model backend is not `M55`,
the M55 branch is discarded during compilation, so the layer does not need a
runtime backend switch.

The optimized hook can live in a backend header, for example `include/edge/backends/m55.hpp`. The current Dense/M55 path follows this shape: the hook is compiled only when the backend, precision, and target feature checks match. On host builds `Backend::M55::falls_back_to_generic` lets the same model type use the generic implementation without requiring Cortex-M55/MVE headers or instructions.

`falls_back_to_generic` is a compile-time policy. If it is `true`, a layer may use its generic implementation when no target path is available. If it is `false`, the layer should `static_assert` when the selected backend has no compatible implementation:

```cpp
if constexpr (!std::is_same_v<typename Types::BackendT, edge::Backend::Generic> &&
              !edge::backend_falls_back_to_generic_v<typename Types::BackendT>) {
    static_assert(edge::detail::always_false_v<typename Types::BackendT>,
                  "Selected backend does not provide this layer and generic fallback is disabled");
}
```

Here is a trimmed example for a fused Dense+ReLU forward path. If the public
semantics are still "Dense followed by ReLU", specialize
`edge::Dense<Out, edge::ReLU>` internally. Expose a `FusedReLUDense` type only
when the fusion is part of the user-facing layer model.

The backend hook is target-specific and can live in `include/edge/backends/m55.hpp`:

```cpp
template<std::size_t InFeatures, std::size_t OutFeatures>
bool m55_fused_relu_dense_forward(
    edge::TensorView<const float, InFeatures> input,
    edge::TensorView<float, OutFeatures> output,
    edge::TensorView<const float, InFeatures * OutFeatures + OutFeatures> params) noexcept {
#if EDGE_DETAIL_HAS_M55_MVE
    // Target implementation: MVE/CMSIS-NN/etc.
    // Computes output = relu(weights * input + bias).
    return true;
#else
    (void)input;
    (void)output;
    (void)params;
    return false;
#endif
}
```

The layer definition keeps the same typed contract and only selects the fused path when the backend and precision are compatible:

```cpp
template<std::size_t OutFeatures>
struct FusedReLUDense {
    template<typename InputSpec>
    struct Instance {
        static_assert(InputSpec::layout == edge::Layout::Flat);

        using input_spec = InputSpec;
        using output_spec = edge::Vector<OutFeatures>;

        static constexpr std::size_t in_features = input_spec::elements;
        static constexpr std::size_t out_features = OutFeatures;
        static constexpr std::size_t weight_count = in_features * out_features;
        static constexpr std::size_t bias_count = out_features;
        static constexpr std::size_t parameter_count = weight_count + bias_count;
        static constexpr std::size_t cache_count = 0;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void forward(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<typename Types::ActivationT, out_features> output,
            edge::TensorView<const typename Types::ParameterT, parameter_count> params,
            edge::TensorView<typename Types::ActivationT, cache_count>,
            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            using ActivationT = typename Types::ActivationT;
            using AccumulatorT = typename Types::AccumulatorT;

            if constexpr (std::is_same_v<typename Types::BackendT, edge::Backend::M55> &&
                          std::is_same_v<typename Types::ParameterT, float> &&
                          std::is_same_v<ActivationT, float> &&
                          std::is_same_v<AccumulatorT, float>) {
                if (m55_fused_relu_dense_forward<in_features, out_features>(
                        input, output, params)) {
                    return;
                }
            }

            const auto* weights = params.data();
            const auto* bias = params.data() + weight_count;
            for (std::size_t out = 0; out < out_features; ++out) {
                AccumulatorT z = static_cast<AccumulatorT>(bias[out]);
                for (std::size_t in = 0; in < in_features; ++in) {
                    z += static_cast<AccumulatorT>(weights[out * in_features + in]) *
                         static_cast<AccumulatorT>(input[in]);
                }
                output[out] = static_cast<ActivationT>(z > AccumulatorT{0} ? z : AccumulatorT{0});
            }
        }

        // initialize() and backward<PropagateInputGradient>() are still required by the custom layer
        // contract. Backward can use output[out] > 0 to apply the ReLU
        // derivative, then accumulate gradients and optionally downstream as usual.
    };
};
```

Used from a model:

```cpp
using Model = edge::Model<
    edge::Backend::M55,
    edge::InputVector<8>,
    FusedReLUDense<16>,
    edge::Dense<1>>;
```

Backward uses the same pattern:

```cpp
if constexpr (std::is_same_v<typename Types::BackendT, edge::Backend::M55> &&
              std::is_same_v<typename Types::ParameterT, float> &&
              std::is_same_v<typename Types::ActivationT, float> &&
              std::is_same_v<typename Types::GradientT, float> &&
              std::is_same_v<typename Types::AccumulatorT, float>) {
    if (m55_my_layer_backward<PropagateInputGradient>(
            input, output, upstream, downstream, params, gradients, cache)) {
        return;
    }
}

generic_my_layer_backward<PropagateInputGradient>(
    input, output, upstream, downstream, params, gradients, cache, workspace);
```

Keep the public layer semantic and specialize the implementation inside
`forward` and `backward`. For example, prefer `Dense` with an M55 fast path over
a user-facing `M55Dense` layer. The model then stays portable:

```cpp
using GenericModel = edge::Model<edge::Backend::Generic, edge::InputVector<8>, edge::Dense<16>>;
using M55Model = edge::Model<edge::Backend::M55, edge::InputVector<8>, edge::Dense<16>>;
```

Both types describe the same network. Only the backend policy changes.

Adding a new backend policy does not require runtime registration. Define a tag
type with `is_backend_policy = true`; add `falls_back_to_generic` only when the
default generic fallback behavior is not what you want:

```cpp
struct MyTargetBackend {
    static constexpr bool is_backend_policy = true;
    static constexpr const char* name = "MyTarget";
    static constexpr bool falls_back_to_generic = true;
};

using MyTargetModel =
    edge::Model<MyTargetBackend, edge::InputVector<8>, edge::Dense<16>>;
```

Then layer implementations can branch on `Types::BackendT`. The explicit
`is_backend_policy` marker keeps backend selection compile-time and prevents
accidental runtime backend strings or unvalidated backend objects from entering
the model type.

## Generic Conv2D Layer

`edge::Conv2D` is implemented with the same typed layer contract and can be used as the generic Conv2D example. The public shape is explicit:

```cpp
using Model = edge::Model<
    edge::Input<edge::CHW<1, 28, 28>>,
    edge::Conv2D<4,
                 edge::Kernel<3, 3>,
                 edge::ReLU,
                 edge::DefaultInitializer,
                 edge::Stride<1, 1>,
                 edge::Padding<1, 1>>,
    edge::Flatten,
    edge::Dense<10>>;
```

Input and output tensors are flattened in `CHW` order. The implementation is direct convolution, not im2col, so it is simple and portable on any host or embedded CPU. Backend-specific code can later optimize the same semantic operation.

## Future Layers

v0.1 custom layers are shape-aware but sequential: they expose `input_spec`, `output_spec`, `in_features`, `out_features`, parameter count, cache count, and workspace count. See `docs/design/limitations.md` for the current limits around state, multi-input layers, RNN, GRU, gates, and attention.
