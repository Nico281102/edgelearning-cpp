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

## Initializer

An initializer provides:

```cpp
template<std::size_t In, std::size_t Out>
static void fill(float* weights, edge::DeterministicRng& rng, const edge::InitConfig& config);
```

Bias values are filled by the model from `InitConfig::bias`.

## Future Layers

Future layer specs should expose output shape, parameter count, activation storage needs, and backend dispatch hooks at compile time. v0.1 intentionally keeps only Dense implemented.

