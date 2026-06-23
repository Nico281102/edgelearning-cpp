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

using Model = edge::Model<DoublePrecision, edge::Input<2>, edge::Dense<1>>;
```

The policy drives both APIs and memory planning. `Model::parameter_data()` returns `ParameterT*`, `Model::forward()` accepts `ActivationT`, and `Model::required_memory` includes the exact section sizes and alignments implied by the policy.

## Custom Layer

Custom vector layers provide a nested instance specialized by the inferred input width:

```cpp
struct MyLayer {
    template<std::size_t InFeatures>
    struct Instance {
        static constexpr std::size_t in_features = InFeatures;
        static constexpr std::size_t out_features = InFeatures;
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

        template<typename Types>
        static void backward(edge::TensorView<const typename Types::ActivationT, in_features>,
                             edge::TensorView<const typename Types::ActivationT, out_features>,
                             edge::TensorView<const typename Types::AccumulatorT, out_features>,
                             edge::TensorView<typename Types::AccumulatorT, in_features>,
                             edge::TensorView<const typename Types::ParameterT, parameter_count>,
                             edge::TensorView<typename Types::GradientT, parameter_count>,
                             edge::TensorView<const typename Types::ActivationT, cache_count>,
                             edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept;
    };
};
```

Forward writes the layer output. Backward receives `dLoss/dOutput`, accumulates parameter gradients, and writes `dLoss/dInput`. Use `TensorView<const T, N>` for read-only inputs and `TensorView<T, N>` for outputs. The view is passed by value because it is only a typed pointer plus compile-time extent.

## Future Layers

Future shape-rich layers such as Conv2D should expose output shape, parameter count, activation storage needs, and backend dispatch hooks at compile time. v0.1 custom layers are vector-shaped.
