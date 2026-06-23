# EdgeLearning++

EdgeLearning++ (`edgelearning-cpp`) started from an embedded systems project where the first runtime was written in C for deterministic, static-memory neural-network training on constrained targets. The C implementation made the core ideas concrete: sample-wise execution, explicit memory ownership, flat parameter layout, and a small training path that could run without a host framework.

This C++20 version is a new implementation inspired by that C design, built to make the framework more generic, more type-safe, and easier to extend. Modern C++ templates let the model topology, layer shapes, activation policies, optimizer state, and memory requirements become compile-time facts instead of runtime descriptors. The redesign was also an opportunity to learn and iterate faster with modern AI coding tools, while keeping the embedded constraints explicit.

The framework is sample-wise by design: it processes one sample at a time, accumulates gradients, and applies an optimizer step according to the configured batch policy. This keeps activation memory bounded by one sample rather than by the whole batch, following the same broad motivation as memory-efficient large-batch and edge-training work such as Piao et al. (2023) and Re-Forward-style memory-efficient backpropagation for edge reinforcement learning.

The v0.1 implementation fully supports Dense layers, vector-shaped custom layers, custom activations, custom losses, custom initializers, and model-level precision policies on the generic scalar backend. It does not include PPO, reinforcement learning applications, CarRacing, Pendulum, STM32N6 application code, STAI integration, host-MCU protocols, private datasets, generated models, or post-baseline optimized kernels. The old C baseline is referenced for methodology and regression measurements only; its source is not vendored in this repository.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The CMake target `edge::edgelearning` is header-only. For GNU/Clang builds, CMake applies `-fno-exceptions` and `-fno-rtti`. On AppleClang installations where libc++ headers live only in the macOS SDK, CMake adds the SDK `usr/include/c++/v1` path when detected.

## Easy Utilization

```cpp
#include <array>

#include <edge/edge.hpp>

using Model = edge::Model<
    edge::Input<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;

static_assert(Model::parameter_bytes > 0);
static_assert(Model::required_memory > 0);

int main() {
    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::AdamConfig{.learning_rate = 1.0e-3F});

    trainer.model().initialize(edge::InitConfig{.seed = 42U});

    std::array<float, 8> input{0.0F, 1.0F, 0.5F, -0.5F, 0.25F, 0.75F, -1.0F, 0.1F};
    std::array<float, 1> target{0.25F};

    return edge::is_ok(trainer.train_step(input, target)) ? 0 : 1;
}
```

The input dimension of each Dense layer is inferred from the previous layer. Backend selection is model-level:

```cpp
using M = edge::Model<
    edge::Backend::Generic,
    edge::Input<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;
```

`Backend::M55` is a tag placeholder in v0.1 and falls back to the generic implementation. Vendor STM32 headers are not included by public generic headers.

Precision is also a model-level policy. The default is `edge::precision::FP32`, but a project can define its own policy:

```cpp
struct DoublePrecision {
    using ParameterT = double;
    using ActivationT = double;
    using GradientT = double;
    using AccumulatorT = double;
    using OptimizerStateT = double;
    using LossT = double;
};

using DoubleModel = edge::Model<
    DoublePrecision,
    edge::Input<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;
```

The same syntax can be combined with an explicit backend:

```cpp
using M = edge::Model<
    edge::Backend::Generic,
    DoublePrecision,
    edge::Input<8>,
    edge::Dense<1>>;
```

See `examples/custom_precision.cpp` for a compilable example.

## Current Support

| Component | v0.1 status | Notes |
|---|---:|---|
| Dense layer | Yes | Forward, backward, gradient accumulation, initialization, serialization |
| Conv2D layer | No | Future layer type |
| Dropout layer | No | Future layer type |
| LayerNorm layer | No | Future layer type |
| Custom layer | Yes | Vector-shaped layers with compile-time feature, parameter, cache, and workspace counts |
| Custom activation | Yes | Generic backend supports user activation policies |
| Custom loss | Yes | Loss computes `value` and `dLoss/dOutput` through the training loss API |
| Custom initializer | Yes | User initializers can fill Dense weights with the model parameter type |
| Custom precision policy | Yes | Model-level type bundle for parameters, activations, gradients, accumulators, optimizer state, and loss |

## Memory Planning

The default model owns one static arena:

```cpp
Model model;
```

The model exposes:

```cpp
Model::parameter_bytes;
Model::gradient_bytes;
Model::optimizer_bytes;
Model::activation_bytes;
Model::workspace_bytes;
Model::total_bytes;
Model::required_memory;
Model::alignment;
```

External arenas are checked at compile time because the size is part of the type:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory> arena;

Model model{edge::external_arena(arena)};
```

`std::span<std::byte, N>` and `std::byte (&)[N]` are also supported. A future `void* + size_t` API, if added, must return `Status` and be documented as runtime-checked only.

The planner is compile-time because the model topology and precision policy are types. For a given `Model`, the compiler knows every layer input/output size, parameter count, activation cache requirement, optimizer state count, and the byte size/alignment of each precision type. The arena is then split into typed sections:

- parameters: `Model::parameter_type`
- gradients: `Model::gradient_type`
- optimizer state: `Model::optimizer_state_type`
- activations and layer caches: `Model::activation_type`
- temporary backward workspace: `Model::accumulator_type`

This is strong for embedded and RTOS targets because the framework does not allocate from the heap at runtime. If the external arena is too small, typed arena APIs fail at compile time; if the storage is not correctly aligned, construction reports `Status::UnalignedArena`. The shared RTOS heap is not touched by the training path.

## Training

```cpp
edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
    edge::AdamConfig{.learning_rate = 1.0e-3F});
trainer.model().initialize(edge::InitConfig{.seed = 42U});

std::array<float, 8> input{};
std::array<float, 1> target{};
edge::Status status = trainer.train_step(input, target);
```

Training is sample-wise. Gradients accumulate until `TrainerConfig::batch_size` samples are seen, then the optimizer step runs. `GradientReduction::Mean` is the default and divides once before the optimizer step. Call `flush()` to update on an incomplete final batch.

The low-level API also supports:

```cpp
model.forward(input);
model.backward(output_gradient);
```

## Custom Loss

Custom losses provide `value` and `evaluate`:

```cpp
struct MyLoss {
    template<typename Prediction, typename Target>
    static auto value(const Prediction& y, const Target& t);

    template<typename Prediction, typename Target, typename Gradient>
    static auto evaluate(const Prediction& y, const Target& t, Gradient& g);
};
```

Returning `float` is valid, but a loss can also return the policy loss type or another arithmetic type compatible with it.

## Custom Activation

```cpp
struct MyActivation {
    static constexpr edge::ActivationStorage storage =
        edge::ActivationStorage::OutputAndPreActivation;

    template<typename T>
    static T forward(T z);

    template<typename T>
    static T derivative(T z, T a);
};
```

Built-ins are `Linear`, `ReLU`, `Tanh`, and `Sigmoid`.

## Custom Layer

A custom vector layer provides a nested `Instance<InFeatures>` with compile-time shape and memory counts, plus typed `initialize`, `forward`, and `backward` functions:

```cpp
struct TrainableScale {
    template<std::size_t InFeatures>
    struct Instance {
        static constexpr std::size_t in_features = InFeatures;
        static constexpr std::size_t out_features = InFeatures;
        static constexpr std::size_t parameter_count = InFeatures;
        static constexpr std::size_t cache_count = 0;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void initialize(edge::TensorView<typename Types::ParameterT, parameter_count> p,
                               edge::DeterministicRng& rng,
                               const edge::InitConfig& config) noexcept;

        template<typename Types>
        static void forward(edge::TensorView<const typename Types::ActivationT, in_features> x,
                            edge::TensorView<typename Types::ActivationT, out_features> y,
                            edge::TensorView<const typename Types::ParameterT, parameter_count> p,
                            edge::TensorView<typename Types::ActivationT, cache_count> cache,
                            edge::TensorView<typename Types::AccumulatorT, workspace_count> work) noexcept;

        template<typename Types>
        static void backward(edge::TensorView<const typename Types::ActivationT, in_features> x,
                             edge::TensorView<const typename Types::ActivationT, out_features> y,
                             edge::TensorView<const typename Types::AccumulatorT, out_features> dy,
                             edge::TensorView<typename Types::AccumulatorT, in_features> dx,
                             edge::TensorView<const typename Types::ParameterT, parameter_count> p,
                             edge::TensorView<typename Types::GradientT, parameter_count> dp,
                             edge::TensorView<const typename Types::ActivationT, cache_count> cache,
                             edge::TensorView<typename Types::AccumulatorT, workspace_count> work) noexcept;
    };
};

using M = edge::Model<edge::Input<4>, TrainableScale, edge::Dense<1>>;
```

`TensorView` is passed by value because it is a small typed view, similar to `std::span<T, N>`. Data that the layer must not modify is exposed through `TensorView<const T, N>`. Backend code can still use `view.data()` when a contiguous pointer is needed for an optimized kernel. See `examples/custom_layer.cpp` for a full implementation.

## Benchmarks

```sh
cmake -S . -B build -DEDGE_BUILD_BENCHMARKS=ON
cmake --build build --parallel
./build/benchmarks/benchmark_edgelearning_cpp
./build/benchmarks/benchmark_regression_vs_c_baseline
```

The regression benchmark generates `benchmarks/results/host_regression_report.md`. It records methodology for comparing against the old C baseline at commit `0085814908ca1b57ece4fe367361d084fd74aa3e` without vendoring or republishing that C source.

## STM32 Toolchain Notes

The core is C++20 and avoids exceptions, RTTI, heap allocation, virtual functions, and `std::function`. For STM32CubeIDE or `arm-none-eabi-g++`, use flags equivalent to:

```sh
arm-none-eabi-g++ -std=c++20 -ffreestanding -fno-exceptions -fno-rtti
```

Include `edgelearning-cpp/include` in the CubeIDE project include paths. Firmware execution tests are outside v0.1.

## References

```bibtex
@article{Piao_2023,
   title={Enabling Large Batch Size Training for DNN Models Beyond the Memory Limit While Maintaining Performance},
   volume={11},
   ISSN={2169-3536},
   url={http://dx.doi.org/10.1109/ACCESS.2023.3312572},
   DOI={10.1109/access.2023.3312572},
   journal={IEEE Access},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Piao, Xinyu and Synn, Doangjoo and Park, Jooyoung and Kim, Jong-Kook},
   year={2023},
   pages={102981--102990}
}

@inproceedings{pispisa_reforward_2026,
  author    = {Pispisa, Gaetano and De Vita, Fabrizio and Bruneo, Dario and Giacalone, Davide and Merlino, Giovanni and Longo, Francesco},
  title     = {Re-Forward: Memory-efficient Backpropagation for Reinforcement Learning at the Edge},
  booktitle = {Workshop paper},
  year      = {2026},
  note      = {Bibliographic entry to be replaced with the official proceedings version when available}
}
```
